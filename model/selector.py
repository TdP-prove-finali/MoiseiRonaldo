from __future__ import annotations

from typing import Dict, Sequence, Mapping, Any, List, Tuple

import numpy as np
import pandas as pd

from model.stock import Stock


class PortfolioSelector:
    """
    Classe per eseguire la selezione combinatoria.

    Responsabilità:
    - definire la politica di selezione:
        * pre-filtro quantitativo dei candidati (build_selector_universe)
        * ordinamento dei candidati (sort_candidates)
    - applicare i vincoli hard (rating_min, max_unrated_share, rho_pair_max, settore)
    - eseguire la ricorsione combinatoria per trovare il sottoinsieme ottimo.
    """

    def __init__(
        self,
        rho: pd.DataFrame,
        rating_scores: Mapping[str, float | None],
        sectors: Mapping[str, str | None],
        has_rating: Mapping[str, bool],
        mu: Mapping[str, float],
        params: Mapping[str, Any],
    ):
        # ---------- 1. Conversione a rappresentazione NumPy ----------

        # Tickers nell'ordine delle colonne di rho
        self.tickers: List[str] = list(rho.columns)
        self.n_assets: int = len(self.tickers)
        self.t2i: Dict[str, int] = {t: i for i, t in enumerate(self.tickers)}

        # Matrice di correlazione come NumPy array
        rho_sub = rho.loc[self.tickers, self.tickers]
        self.rho_np: np.ndarray = rho_sub.values.astype(float)

        # Rating
        self.rating_np: np.ndarray = np.full(self.n_assets, np.nan, dtype=float)
        self.has_rating_np: np.ndarray = np.zeros(self.n_assets, dtype=bool)
        for t, i in self.t2i.items():
            r = rating_scores.get(t)
            if r is not None and has_rating.get(t, False):
                self.rating_np[i] = float(r)
                self.has_rating_np[i] = True

        # Settori: mappiamo ogni settore a un ID intero, -1 = nessun settore
        sectors_values = [s for s in sectors.values() if s is not None]
        unique_sectors = sorted(set(sectors_values))
        self.sector_to_id: Dict[str, int] = {s: i for i, s in enumerate(unique_sectors)}
        self.sector_id_np: np.ndarray = np.full(self.n_assets, -1, dtype=np.int32)
        for t, i in self.t2i.items():
            sec = sectors.get(t)
            if sec is not None:
                self.sector_id_np[i] = self.sector_to_id[sec]

        # Rendimenti attesi mu
        self.mu_np: np.ndarray = np.zeros(self.n_assets, dtype=float)
        for t, i in self.t2i.items():
            v = mu.get(t)
            self.mu_np[i] = float(v) if v is not None else 0.0

        # ---------- 2. Parametri / Vincoli (pre-calcolati) ----------

        self.params = params

        self.K: int = int(params.get("K", 0))
        self.rating_min: float = float(params.get("rating_min", 0.0))
        self.max_unrated_share: float = float(params.get("max_unrated_share", 1.0))
        self.max_share_per_sector: float = float(params.get("max_share_per_sector", 1.0))
        self.max_count_per_sector_param = params.get("max_count_per_sector", None)
        self.rho_pair_max = params.get("rho_pair_max", None)

        # Pesi dello score
        self.alpha: float = float(params.get("alpha", 1.0))
        self.beta: float = float(params.get("beta", 0.0))
        self.gamma: float = float(params.get("gamma", 0.0))
        self.delta: float = float(params.get("delta", 0.0))

        # Effettivo max_share usato nella penalità settoriale
        if self.max_share_per_sector <= 0.0:
            self._max_share_eff: float = 0.0
        else:
            self._max_share_eff = self.max_share_per_sector

        # Vincolo K per settore (come prima logica)
        self.max_count_per_sector: int | None = None
        if self.max_count_per_sector_param is not None:
            self.max_count_per_sector = int(self.max_count_per_sector_param)
        elif self.max_share_per_sector < 1.0 and self.K > 0:
            self.max_count_per_sector = max(
                1, int(np.floor(self.max_share_per_sector * self.K + 1e-9))
            )

        # Max rating globale (per bound ottimistico)
        mask_valid_r = self.has_rating_np & ~np.isnan(self.rating_np)
        if np.any(mask_valid_r):
            self.max_rating_global: float = float(np.nanmax(self.rating_np[mask_valid_r]))
        else:
            self.max_rating_global = 0.0

        # ---------- 3. Stato dei risultati ----------

        self.best_subset: List[str] | None = None
        self.best_score: float = float("-inf")
        self.best_subset_indices: List[int] | None = None

        # Questi saranno inizializzati in select(...)
        self._cand_indices: np.ndarray | None = None
        self._mu_prefix: np.ndarray | None = None
        self._n_cand: int = 0

    # ---------- PARAMETRI DI DEFAULT ----------

    @staticmethod
    def build_default_params() -> Dict[str, Any]:
        """
        Costruisce un dizionario di parametri di default per il selettore.
        """
        params = {
            "K": 20,
            "rating_min": 13.0,
            "max_share_per_sector": 0.4,
            # Tetto massimo di correlazione per coppia (in valore assoluto)
            # Nessuna coppia con |rho_ij| > 0.8 è ammessa nel portafoglio.
            "rho_pair_max": 0.8,
            "max_unrated_share": 0.2,
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 0.5,
            "delta": 0.2,
        }
        return params

    # ---------- STATIC METHODS: POLITICA DI SELEZIONE / PREFILTRO ----------

    @staticmethod
    def build_selector_universe(
        base_universe: List[str],
        K: int,
        mu: pd.Series | Mapping[str, float] | None,
        Sigma_sh: pd.DataFrame | None,
        stocks: Mapping[str, Stock],
        a1: float = 1.0,
        a2: float = 0.5,
        a3: float = 0.3,
        min_size: int = 40,
        factor: int = 4,
    ) -> List[str]:
        """
        Pre-filtro quantitativo sui candidati per la combinatoria.

        Per ogni ticker in base_universe calcola:
            - mu_i (dai rendimenti attesi)
            - sigma_i (sqrt(diagonale di Sigma_sh))
            - rating_score_norm (rating normalizzato in [0,1] sui soli rated)

        Definisce un punteggio semplice:
            asset_score_i = a1 * mu_i - a2 * sigma_i + a3 * rating_score_norm

        Ordina per asset_score decrescente e taglia a:
            selector_universe = primi N,
        dove N = min( max(factor * K, min_size), len(base_universe) ).

        Se mu o Sigma_sh sono None → ritorna base_universe senza modifiche.
        """
        if not base_universe:
            return []

        if mu is None or Sigma_sh is None:
            # nessuna informazione → usa l'universo così com'è
            return list(base_universe)

        # normalizzo a tipi noti
        if isinstance(mu, pd.Series):
            mu_series = mu
        else:
            mu_series = pd.Series(mu, dtype=float)

        Sigma_df: pd.DataFrame = Sigma_sh

        asset_data: List[tuple[str, float, float, float | None]] = []
        rating_values: List[float] = []

        # Prima passata: raccogli mu, sigma, rating
        for t in base_universe:
            # mu_i
            if t in mu_series.index and pd.notna(mu_series.loc[t]):
                mu_i = float(mu_series.loc[t])
            else:
                mu_i = 0.0

            # sigma_i dalla diagonale della covarianza shrinkata
            if (
                t in Sigma_df.index
                and t in Sigma_df.columns
                and pd.notna(Sigma_df.loc[t, t])
            ):
                var_i = float(Sigma_df.loc[t, t])
                sigma_i = np.sqrt(var_i) if var_i > 0 else 0.0
            else:
                sigma_i = 0.0

            stock = stocks.get(t)
            r = stock.rating_score if stock is not None else None
            if r is not None:
                rating_values.append(r)

            asset_data.append((t, mu_i, sigma_i, r))

        # Normalizzazione rating in [0,1] sui soli titoli con rating
        if rating_values:
            r_min = min(rating_values)
            r_max = max(rating_values)
            denom_r = (r_max - r_min) if (r_max > r_min) else 1.0
        else:
            r_min = 0.0
            denom_r = 1.0

        asset_scores: Dict[str, float] = {}
        for t, mu_i, sigma_i, r in asset_data:
            if r is not None:
                rating_norm = (r - r_min) / denom_r
            else:
                rating_norm = 0.0

            score = a1 * mu_i - a2 * sigma_i + a3 * rating_norm
            asset_scores[t] = float(score)

        # Ordina per asset_score decrescente
        sorted_tickers = sorted(
            base_universe,
            key=lambda x: asset_scores.get(x, float("-inf")),
            reverse=True,
        )

        # Dimensione massima per la combinatoria:
        #   - almeno min_size titoli
        #   - oppure factor*K se più grande di min_size
        if K <= 0:
            max_dim = len(sorted_tickers)
        else:
            max_dim = max(factor * K, min_size)

        final_dim = min(max_dim, len(sorted_tickers))
        selector_universe = sorted_tickers[:final_dim]

        return selector_universe

    @staticmethod
    def sort_candidates(
        tickers: Sequence[str],
        rating_scores: Mapping[str, float | None],
        has_rating: Mapping[str, bool],
        mu: Mapping[str, float],
    ) -> List[str]:
        """
        Euristica di ordinamento dei candidati:

        - prima i titoli con rating (has_rating=True),
        - poi in ordine decrescente di rating_score,
        - a parità, in ordine decrescente di mu atteso.
        """

        def sort_key(t: str):
            hr = has_rating.get(t, False)
            rs = rating_scores.get(t)
            rs_val = rs if rs is not None else -1e9
            mu_val = mu.get(t, 0.0)
            # rated prima (0), poi unrated (1); dentro ciascun gruppo ordina per rating/mu
            return (0 if hr else 1, -rs_val, -mu_val)

        return sorted(tickers, key=sort_key)

    # ---------- METODI PRIVATI: SCORE E VINCOLI (versione NumPy) ----------

    def _score_indices(self, indices: Sequence[int]) -> float:
        """
        Calcola lo score combinatorio di un sottoinsieme rappresentato da indici interi.
        Replica esattamente la logica di _getScore su ticker.
        """
        n = len(indices)
        if n == 0:
            return float("-inf")

        idx_arr = np.asarray(indices, dtype=np.int32)

        # 1) Correlazione media (modulo)
        if n < 2:
            mean_corr = 0.0
        else:
            sub = self.rho_np[np.ix_(idx_arr, idx_arr)]
            iu = np.triu_indices(n, k=1)
            vals = sub[iu]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                mean_corr = 0.0
            else:
                mean_corr = float(np.mean(np.abs(vals)))

        # 2) Rating medio sui soli titoli con rating
        r_vals = self.rating_np[idx_arr]
        mask_rated = self.has_rating_np[idx_arr] & ~np.isnan(r_vals)
        if np.any(mask_rated):
            mean_rating = float(np.mean(r_vals[mask_rated]))
        else:
            mean_rating = 0.0

        # 3) Penalità settoriale
        if n == 0:
            pen_sector = 0.0
        else:
            sec_ids = self.sector_id_np[idx_arr]
            sec_ids = sec_ids[sec_ids != -1]
            if sec_ids.size == 0:
                pen_sector = 0.0
            else:
                counts = np.bincount(sec_ids)
                shares = counts / float(n)
                excess = shares - self._max_share_eff
                excess[excess < 0.0] = 0.0
                pen_sector = float(np.sum(excess))

        # 4) Rendimento medio atteso
        mu_vals = self.mu_np[idx_arr]
        if mu_vals.size > 0:
            mean_return = float(np.mean(mu_vals))
        else:
            mean_return = 0.0

        score = (
            self.alpha * (-mean_corr)
            + self.beta * mean_rating
            - self.gamma * pen_sector
            + self.delta * mean_return
        )
        return float(score)

    def _violates_constraints_indices(
        self,
        partial_indices: Sequence[int],
        new_idx: int,
    ) -> bool:
        """
        Verifica i vincoli hard sui soli indici interi.
        Replica _violates_constraints con la stessa logica.
        """
        n_total = len(partial_indices) + 1

        # 1) Rating minimo sul nuovo titolo
        if self.has_rating_np[new_idx]:
            rs = self.rating_np[new_idx]
            if not np.isnan(rs) and rs < self.rating_min:
                return True

        # 2) Limiti per settore (usa self.max_count_per_sector)
        if self.max_count_per_sector is not None:
            sec_new = self.sector_id_np[new_idx]
            if sec_new != -1:
                count = 1  # includo il nuovo
                for idx in partial_indices:
                    if self.sector_id_np[idx] == sec_new:
                        count += 1
                # Nota: nel codice originale si usava ">" (non ">=")
                if count > self.max_count_per_sector:
                    return True

        # 3) Quota massima unrated
        if self.max_unrated_share < 1.0 and n_total > 0:
            n_unrated = 0
            for idx in partial_indices:
                if not self.has_rating_np[idx]:
                    n_unrated += 1
            if not self.has_rating_np[new_idx]:
                n_unrated += 1

            share_unrated = n_unrated / float(n_total)
            if share_unrated > self.max_unrated_share:
                return True

        # 4) Max correlazione per coppia
        if self.rho_pair_max is not None and partial_indices:
            rho_max = float(self.rho_pair_max)
            arr_partial = np.asarray(partial_indices, dtype=np.int32)
            corrs = self.rho_np[arr_partial, new_idx]
            mask_valid = ~np.isnan(corrs)
            if np.any(np.abs(corrs[mask_valid]) > rho_max):
                return True

        return False

    # ---------- RICORSIONE + BRANCH & BOUND ----------

    def _search(
        self,
        partial_indices: List[int],
        start_pos: int,
        sum_mu_partial: float,
    ) -> None:
        """
        Ricorsione combinatoria con pruning:
        - pruning di lunghezza,
        - bound ottimistico su rating+mu (corr/settore ignorati nel bound → bound sicuro).
        """
        k_curr = len(partial_indices)

        # Base case: portafoglio completo
        if k_curr == self.K:
            s = self._score_indices(partial_indices)
            if s > self.best_score:
                self.best_score = s
                self.best_subset_indices = list(partial_indices)
            return

        remaining_candidates = self._n_cand - start_pos
        if k_curr + remaining_candidates < self.K:
            # Nemmeno prendendo tutti i rimanenti arrivo a K
            return

        remaining_to_pick = self.K - k_curr

        # Bound ottimistico (solo se abbiamo già una soluzione best)
        if self.best_score != float("-inf"):
            # Somma massima di mu che posso ancora aggiungere (ignorando vincoli)
            # I candidati sono ordinati con un certo criterio, ma mu_prefix è
            # la cumsum delle mu in quell'ordine.
            max_future_mu_sum = (
                self._mu_prefix[start_pos + remaining_to_pick]
                - self._mu_prefix[start_pos]
            )
            mu_sum_upper = sum_mu_partial + max_future_mu_sum

            # rating_term <= beta * max_rating_global
            # mu_term <= delta * (mu_sum_upper / K)
            optimistic_bound = (
                self.beta * self.max_rating_global
                + self.delta * (mu_sum_upper / float(self.K))
            )

            # Se anche nel caso più ottimistico non posso battere best_score → prune
            if optimistic_bound <= self.best_score:
                return

        # Loop ricorsivo
        for pos in range(start_pos, self._n_cand):
            idx = int(self._cand_indices[pos])

            # Vincoli hard
            if self._violates_constraints_indices(partial_indices, idx):
                continue

            partial_indices.append(idx)
            self._search(
                partial_indices,
                pos + 1,
                sum_mu_partial + self.mu_np[idx],
            )
            partial_indices.pop()

    # ---------- METODO PUBBLICO ----------

    def select(self, candidati: Sequence[str]) -> Tuple[List[str] | None, float]:
        """
        Funzione di ingresso per il selettore combinatorio.
        Avvia la ricerca e restituisce i risultati.
        """
        # K non valido
        if self.K <= 0:
            self.best_subset = None
            self.best_score = float("-inf")
            self.best_subset_indices = None
            return None, float("-inf")

        # 1. Converti candidati (stringhe) in indici interi
        valid_indices: List[int] = []
        for t in candidati:
            idx = self.t2i.get(t)
            if idx is not None:
                valid_indices.append(idx)

        if not valid_indices:
            self.best_subset = None
            self.best_score = float("-inf")
            self.best_subset_indices = None
            return None, float("-inf")

        # 2. Ordinamento candidati (euristica):
        # usiamo la stessa logica di sort_candidates ma sugli indici
        def sort_key_idx(i: int):
            hr = self.has_rating_np[i]
            rs = self.rating_np[i]
            rs_val = rs if not np.isnan(rs) else -1e9
            mu_val = self.mu_np[i]
            return (0 if hr else 1, -rs_val, -mu_val)

        sorted_indices = sorted(valid_indices, key=sort_key_idx)

        self._cand_indices = np.asarray(sorted_indices, dtype=np.int32)
        self._n_cand = len(self._cand_indices)

        # 3. Prefix-sum delle mu nell'ordine scelto (per il bound)
        mu_ordered = self.mu_np[self._cand_indices]
        self._mu_prefix = np.zeros(self._n_cand + 1, dtype=float)
        np.cumsum(mu_ordered, out=self._mu_prefix[1:])

        # 4. Reset dei risultati
        self.best_subset_indices = None
        self.best_score = float("-inf")

        # 5. Seed greedy iniziale: costruisce rapidamente un portafoglio valido
        greedy: List[int] = []
        sum_mu_greedy = 0.0
        for idx in self._cand_indices:
            if len(greedy) == self.K:
                break
            if not self._violates_constraints_indices(greedy, idx):
                greedy.append(idx)
                sum_mu_greedy += self.mu_np[idx]

        if len(greedy) == self.K:
            greedy_score = self._score_indices(greedy)
            self.best_subset_indices = list(greedy)
            self.best_score = greedy_score

        # 6. Ricerca ricorsiva con pruning
        self._search(
            partial_indices=[],
            start_pos=0,
            sum_mu_partial=0.0,
        )

        # 7. Conversione risultato in ticker
        if self.best_subset_indices is None:
            self.best_subset = None
            return None, float("-inf")

        self.best_subset = [self.tickers[i] for i in self.best_subset_indices]
        return self.best_subset, self.best_score


if __name__ == "__main__":
    import traceback

    print("=== TEST SELECTOR (NumPy + B&B) ===")

    try:
        tickers = ["A", "B", "C", "D", "E"]
        data = {
            "A": [1.0, 0.6, 0.1, -0.2, 0.3],
            "B": [0.6, 1.0, 0.5, 0.0, 0.2],
            "C": [0.1, 0.5, 1.0, 0.3, 0.4],
            "D": [-0.2, 0.0, 0.3, 1.0, 0.1],
            "E": [0.3, 0.2, 0.4, 0.1, 1.0],
        }
        rho_test = pd.DataFrame(data, index=tickers, columns=tickers)
        rating_scores = {"A": 18.0, "B": 16.0, "C": 13.0, "D": None, "E": 10.0}
        has_rating = {t: (rating_scores[t] is not None) for t in tickers}
        sectors = {
            "A": "Tech",
            "B": "Tech",
            "C": "Health",
            "D": "Energy",
            "E": "Finance",
        }
        mu = {"A": 0.08, "B": 0.07, "C": 0.06, "D": 0.12, "E": 0.09}

        params = PortfolioSelector.build_default_params()
        params["K"] = 3
        params["max_unrated_share"] = 0.34
        params["rating_min"] = 13.0
        params["max_share_per_sector"] = 0.67

        selector = PortfolioSelector(
            rho=rho_test,
            rating_scores=rating_scores,
            sectors=sectors,
            has_rating=has_rating,
            mu=mu,
            params=params,
        )

        cand_pref = PortfolioSelector.sort_candidates(
            tickers=tickers,
            rating_scores=rating_scores,
            has_rating=has_rating,
            mu=mu,
        )
        print("Candidati ordinati:", cand_pref)

        best_subset, best_score = selector.select(candidati=cand_pref)

        print("Best subset:", best_subset)
        print("Best score:", best_score)

    except Exception:
        print("TEST SELECTOR FALLITO:")
        traceback.print_exc()
    else:
        print("TEST SELECTOR OK.")
