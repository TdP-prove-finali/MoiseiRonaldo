from __future__ import annotations

from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from database.DAO import DAO
from model.stock import Stock
from model.risk_estimator import RiskEstimator
from model.graph_builder import GraphBuilder
from model.selector import PortfolioSelector
from model.portfolio_weights import PortfolioWeights


class Model:
    """
    Classe principale del modello.

    Responsabilità:
        - carica dati con il DAO
        - costruisce prices_df, ratings_df, stocks, returns_df
        - stima rho, Sigma_sh, mu sull'intera storia disponibile
        - costruisce il grafo di correlazione (GraphBuilder)
        - applica filtri (soglia, k-NN) e costruisce un universo ridotto U'
        - delega la selezione combinatoria del portafoglio a PortfolioSelector
        - calcola i pesi (PortfolioWeights)
        - costruisce un portafoglio min-correlazione via Dijkstra
        - esegue simulazioni Monte Carlo sugli ultimi portafogli trovati
    """

    def __init__(self, dao: Optional[DAO] = None) -> None:
        self._dao = dao if dao is not None else DAO()

        self.prices_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None
        self.stocks: Dict[str, Stock] = {}

        self.returns_df: pd.DataFrame | None = None

        # Oggetti di rischio correnti
        self.current_rho: pd.DataFrame | None = None
        self.current_Sigma_sh: pd.DataFrame | None = None
        self.current_mu: pd.Series | None = None
        self.current_universe: List[str] = []

        # Info su rating
        self.tickers_with_rating: List[str] = []
        self.tickers_without_rating: List[str] = []
        self.map_has_rating: Dict[str, bool] = {}

        # Grafo e universo ridotto
        self.current_graph: nx.Graph | None = None
        self.reduced_universe: List[str] = []

        # Universo usato dal selettore (dopo pre-filtro quantitativo)
        self.selector_universe: List[str] = []

        # Risultati ultima ottimizzazione B&B
        self.last_portfolio_tickers: List[str] = []
        self.last_portfolio_weights: Dict[str, float] = {}
        self.last_portfolio_score: float | None = None

        # Risultati ultimo portafoglio Dijkstra
        self.last_dij_tickers: List[str] = []
        self.last_dij_weights: Dict[str, float] = {}

        # Profilo di rischio corrente (parametri "globali" per grafi e selettore)
        self._risk_profile: Dict[str, Any] | None = None

        # INIZIALIZZAZIONE AUTOMATICA: CARICAMENTO DATI E STIMA RISCHIO
        print("[Model] Inizio caricamento dati dal DAO...")
        self.load_data_from_dao()
        print("[Model] Dati caricati. Stima del rischio...")
        self.estimate_risk(
            shrink_lambda=0.1,
            min_non_na_ratio=0.8,
            winsor_lower=0.01,
            winsor_upper=0.99,
        )
        print("[Model] Stima di rho, Sigma_sh e mu completata.")

    # PROFILO DI RISCHIO

    def set_risk_profile(self, profile: Dict[str, Any]) -> None:
        """
        Imposta (o sovrascrive) il profilo di rischio corrente.

        Esempio di dizionario:
            {
                "tau": 0.30,
                "k": 10,
                "max_size": 60,
                "max_unrated_share": 0.25,
                "min_rating_score": 13.0,
                "target_rated_share": 0.7,
                "rating_min": 13.0,
                "rho_pair_max": 0.8,
                "weights_mode": "mv",
                "mv_risk_aversion": 1.0,
            }
        """
        self._risk_profile = dict(profile)

    # CARICAMENTO DATI

    def load_data_from_dao(self) -> None:
        """
        Carica i dati di base (prezzi, rating, oggetti Stock) dal DAO
        e costruisce returns_df, più alcune liste di supporto sui rating.
        """
        prices, ratings, stock_dict = self._dao.load_universe()

        self.prices_df = prices
        self.ratings_df = ratings
        self.stocks = stock_dict

        # DataFrame dei rendimenti da tutti gli Stock
        returns_dict = {
            ticker: stock.returns
            for ticker, stock in self.stocks.items()
        }
        self.returns_df = pd.DataFrame(returns_dict).sort_index()

        # liste con/senza rating e dizionario has_rating
        self.tickers_with_rating = [
            t for t, s in self.stocks.items() if s.rating_score is not None
        ]
        self.tickers_without_rating = [
            t for t, s in self.stocks.items() if s.rating_score is None
        ]

        # mappatura per controlli rapidi
        self.map_has_rating = {
            t: (s.rating_score is not None)
            for t, s in self.stocks.items()
        }

    # STIMA RISCHIO SULL'INTERA STORIA

    def estimate_risk(
        self,
        shrink_lambda: float = 0.1,
        min_non_na_ratio: float = 0.8,
        winsor_lower: float | None = 0.01,
        winsor_upper: float | None = 0.99,
    ) -> None:
        """
        Stima rho, Sigma_sh, mu.

        Usa RiskEstimator.estimate_from_returns come "libreria".
        """
        if self.returns_df is None:
            raise RuntimeError(
                "returns_df non è stato caricato. Chiama load_data_from_dao() prima."
            )

        rho, Sigma_sh, mu = RiskEstimator.estimate_from_returns(
            self.returns_df,
            shrink_lambda=shrink_lambda,
            min_non_na_ratio=min_non_na_ratio,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            shrink_target="diagonal",
        )

        self.current_rho = rho
        self.current_Sigma_sh = Sigma_sh
        self.current_mu = mu
        self.current_universe = list(rho.columns)

        # reset grafo/universo ridotto / selector_universe (verranno ricostruiti)
        self.current_graph = None
        self.reduced_universe = []
        self.selector_universe = []

    # GRAFO E UNIVERSO RIDOTTO U'

    def build_reduced_universe(
        self,
        tau: float | None = None,
        k: int | None = None,
        max_size: int | None = None,
        max_unrated_share: float | None = None,
        min_rating_score: float | None = None,
        target_rated_share: float | None = None,
    ) -> List[str]:
        """
        Costruisce:
        - il grafo di correlazione corrente (self.current_graph),
        - un universo ridotto U' (lista di ticker) salvato in self.reduced_universe.

        I parametri possono essere passati direttamente (per uso avanzato) oppure
        lasciati a None: in quel caso, se presente, verrà utilizzato il profilo
        di rischio corrente (_risk_profile), altrimenti i default interni.
        """
        if self.current_rho is None:
            raise RuntimeError(
                "current_rho non disponibile. Chiama estimate_risk() prima di build_reduced_universe()."
            )

        # Parametri effettivi: override diretto > profilo di rischio > default
        prof = self._risk_profile or {}

        if tau is None:
            tau = prof.get("tau", None)
        if k is None:
            k = prof.get("k", None)
        if max_size is None:
            max_size = int(prof.get("max_size", 60))
        if max_unrated_share is None:
            max_unrated_share = float(prof.get("max_unrated_share", 1.0))
        if min_rating_score is None:
            min_rating_score = float(prof.get("min_rating_score", 13.0))
        if target_rated_share is None:
            target_rated_share = float(prof.get("target_rated_share", 0.7))

        rho_full = self.current_rho.copy()
        tickers_all = list(rho_full.columns)

        # 1) Restrizione iniziale in base a max_unrated_share
        base_tickers, rho = self._build_universe_base(
            tickers_all=tickers_all,
            rho_full=rho_full,
            max_unrated_share=max_unrated_share,
        )

        # 2) Costruzione grafo + adj + dist_knn tramite GraphBuilder
        G, adj, dist_knn = GraphBuilder.build_filtered_graph(
            rho=rho,
            tau=tau,
            k=k,
            signed=False,
        )
        self.current_graph = G

        # 3) Selezione effettiva di U'
        reduced = self._select_reduced_universe(
            adj=adj,
            min_rating_score=min_rating_score,
            max_unrated_share=max_unrated_share,
            max_size=max_size,
            target_rated_share=target_rated_share,
        )

        self.reduced_universe = reduced
        # azzera l'eventuale selector_universe perché dipende da K
        self.selector_universe = []

        return reduced

    def _build_universe_base(
        self,
        tickers_all: List[str],
        rho_full: pd.DataFrame,
        max_unrated_share: float,
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Applica la logica di base su rated/unrated per decidere
        l'universo iniziale su cui costruire il grafo.
        """
        if max_unrated_share == 0.0:
            # Universo iniziale: solo titoli con rating
            base_tickers = [
                t for t in tickers_all if self.map_has_rating.get(t, False)
            ]
            if not base_tickers:
                raise ValueError(
                    "Nessun titolo con rating disponibile nell'universo corrente."
                )
        else:
            base_tickers = tickers_all

        rho = rho_full.loc[base_tickers, base_tickers]
        return base_tickers, rho

    def _select_reduced_universe(
        self,
        adj: pd.DataFrame,
        min_rating_score: float,
        max_unrated_share: float,
        max_size: int,
        target_rated_share: float,
    ) -> List[str]:
        """
        Contiene la logica di bilanciamento rated/unrated e strength.
        """
        strength = adj.sum(axis=1)  # somma dei pesi |rho_ij| per ogni nodo
        all_candidates = list(strength.index)

        pool_rated: List[str] = []
        pool_unrated: List[str] = []

        for t in all_candidates:
            stock = self.stocks.get(t)
            rs = stock.rating_score if stock is not None else None
            if rs is not None and rs >= min_rating_score:
                pool_rated.append(t)
            else:
                pool_unrated.append(t)

        # limito max_size al numero totale di candidati
        if max_size is None or max_size > len(all_candidates):
            max_size = len(all_candidates)

        # Quote per rated / unrated
        if max_unrated_share == 0.0:
            target_rated_share_effective = 1.0
        else:
            target_unrated_share = 1.0 - target_rated_share
            if target_unrated_share > max_unrated_share:
                target_unrated_share = max_unrated_share
                target_rated_share_effective = 1.0 - target_unrated_share
            else:
                target_rated_share_effective = target_rated_share

        N_rated_target = int(round(max_size * target_rated_share_effective))
        N_unrated_target = max_size - N_rated_target

        if max_unrated_share == 0.0:
            N_unrated_target = 0
            N_rated_target = max_size

        # clamp ai pool disponibili
        N_rated = min(N_rated_target, len(pool_rated))
        N_unrated = min(N_unrated_target, len(pool_unrated))

        # Ordinamento per strength crescente (scegliamo i titoli meno correlati)
        rated_strength = (
            strength.loc[pool_rated].sort_values(ascending=True)
            if pool_rated
            else pd.Series(dtype=float)
        )
        unrated_strength = (
            strength.loc[pool_unrated].sort_values(ascending=True)
            if pool_unrated
            else pd.Series(dtype=float)
        )

        selected_rated = list(rated_strength.index[:N_rated])
        selected_unrated = list(unrated_strength.index[:N_unrated])

        selected = set(selected_rated + selected_unrated)

        # se mancano ancora titoli per arrivare a max_size,
        # riempio con i restanti (sempre per strength bassa)
        total_selected = len(selected)
        if total_selected < max_size:
            remaining_slots = max_size - total_selected

            remaining_candidates = [
                t for t in all_candidates if t not in selected
            ]
            if remaining_candidates:
                remaining_strength = strength.loc[
                    remaining_candidates
                ].sort_values(ascending=True)
                extra = list(remaining_strength.index[:remaining_slots])
                selected.update(extra)

        return list(selected)

    # OTTIMIZZAZIONE PORTAFOGLIO (Branch & Bound)

    def optimize_portfolio(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_reduced_universe: bool = True,
    ) -> Tuple[List[str] | None, Dict[str, float], float | None]:
        """
        Esegue la selezione combinatoria del portafoglio.

        - Se 'params' è None, parte dai default del PortfolioSelector e
          integra, se presente, il profilo di rischio.
        - Se 'params' è fornito, i valori espliciti in params hanno priorità
          rispetto al profilo di rischio.
        """
        if self.current_rho is None or self.current_mu is None:
            raise RuntimeError(
                "current_rho / current_mu non disponibili. "
                "Chiama prima estimate_risk(), poi (eventualmente) build_reduced_universe()."
            )

        # 1) Costruzione parametri completi
        full_params = self._build_selector_params(params)
        K = int(full_params.get("K", 0))
        if K <= 0:
            raise ValueError("Parametro K mancante o non valido nei params.")

        # 2) Scegli l'universo base: U' se disponibile, altrimenti tutto
        base_universe = self._get_base_universe_for_selection(use_reduced_universe)
        if not base_universe:
            raise ValueError(
                "Nessun titolo disponibile per l'ottimizzazione (universo vuoto)."
            )

        # 3) Pre-filtro quantitativo (nel Selector)
        universe = PortfolioSelector.build_selector_universe(
            base_universe=base_universe,
            K=K,
            mu=self.current_mu,
            Sigma_sh=self.current_Sigma_sh,
            stocks=self.stocks,
        )
        self.selector_universe = list(universe)

        if not universe:
            raise ValueError(
                "selector_universe vuoto dopo il pre-filtro quantitativo."
            )

        # 4) Costruzione dizionari per il Selector
        rho_sub, rating_scores, sectors, has_rating, mu_dict = \
            self._build_selector_inputs(universe)

        # 5) Ordinamento candidati (nel Selector)
        candidati = PortfolioSelector.sort_candidates(
            tickers=universe,
            rating_scores=rating_scores,
            has_rating=has_rating,
            mu=mu_dict,
        )

        # 6) Selezione combinatoria
        selector = PortfolioSelector(
            rho=rho_sub,
            rating_scores=rating_scores,
            sectors=sectors,
            has_rating=has_rating,
            mu=mu_dict,
            params=full_params,
        )
        best_subset, best_score = selector.select(candidati=candidati)

        if best_subset is None or len(best_subset) == 0:
            # Nessuna soluzione che rispetti i vincoli
            self.last_portfolio_tickers = []
            self.last_portfolio_weights = {}
            self.last_portfolio_score = None
            return None, {}, None

        # 7) Calcolo pesi (solo long)
        mode = full_params.get("weights_mode", "mv")          # "mv" o "eq"
        risk_aversion = float(full_params.get("mv_risk_aversion", 1.0))

        weights = PortfolioWeights.compute(
            tickers=list(best_subset),
            mode=mode,
            mu=self.current_mu,
            Sigma_sh=self.current_Sigma_sh,
            risk_aversion=risk_aversion,
        )

        self.last_portfolio_tickers = list(best_subset)
        self.last_portfolio_weights = weights
        self.last_portfolio_score = float(best_score)

        return list(best_subset), weights, float(best_score)

    def _build_selector_params(
        self,
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge tra default del PortfolioSelector, profilo di rischio (se presente)
        e params espliciti passati dal chiamante (che hanno priorità).
        """
        base = PortfolioSelector.build_default_params()

        # Integra profilo di rischio (solo se presente)
        if self._risk_profile is not None:
            prof = self._risk_profile
            for key in [
                "rating_min",
                "max_unrated_share",
                "rho_pair_max",
                "weights_mode",
                "mv_risk_aversion",
            ]:
                if key in prof:
                    base[key] = prof[key]

        # Sovrascrive con i params espliciti (priorità al chiamante)
        if params:
            for k, v in params.items():
                base[k] = v

        return base

    def _get_base_universe_for_selection(
        self,
        use_reduced_universe: bool,
    ) -> List[str]:
        """
        Sceglie l'universo di partenza per la selezione combinatoria:
        U' se use_reduced_universe=True e non vuoto, altrimenti l'universo pieno.
        """
        rho = self.current_rho
        mu_series = self.current_mu
        assert rho is not None and mu_series is not None

        if use_reduced_universe and self.reduced_universe:
            base_universe = [
                t
                for t in self.reduced_universe
                if t in rho.columns and t in mu_series.index
            ]
        else:
            base_universe = [
                t for t in rho.columns if t in mu_series.index
            ]

        return base_universe

    def _build_selector_inputs(
        self,
        universe: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, float | None], Dict[str, str | None],
               Dict[str, bool], Dict[str, float]]:
        """
        Costruisce i dizionari (rating_scores, sectors, has_rating, mu_dict)
        e la rho ridotta all'universo scelto, da passare al PortfolioSelector.
        """
        rho = self.current_rho
        mu_series = self.current_mu
        assert rho is not None and mu_series is not None

        rating_scores: Dict[str, float | None] = {}
        sectors: Dict[str, str | None] = {}
        has_rating: Dict[str, bool] = {}
        mu_dict: Dict[str, float] = {}

        for t in universe:
            stock = self.stocks.get(t)
            if stock is not None:
                rating_scores[t] = stock.rating_score
                sectors[t] = stock.sector
                has_rating[t] = (stock.rating_score is not None)
            else:
                rating_scores[t] = None
                sectors[t] = None
                has_rating[t] = False

            # mu corrente per il titolo (se presente)
            if t in mu_series.index and pd.notna(mu_series.loc[t]):
                mu_dict[t] = float(mu_series.loc[t])
            else:
                mu_dict[t] = 0.0

        rho_sub = rho.loc[universe, universe]

        return rho_sub, rating_scores, sectors, has_rating, mu_dict

    # GRAFO PER DIJKSTRA

    def build_graph_for_dijkstra(
        self,
        use_reduced_universe: bool = True,
        signed: bool = False,
        tau: float | None = None,
        k: int | None = None,
    ) -> nx.Graph:
        """
        Costruisce e salva in self.current_graph il grafo da usare con Dijkstra.

        Passi:
        - sceglie l'universo di nodi: U' (self.reduced_universe) oppure l'intero
          universo di current_rho;
        - restringe current_rho all'universo scelto;
        - chiama GraphBuilder.build_filtered_graph con i parametri specificati
          (signed, tau, k);
        - aggiorna self.current_graph.
        """
        if self.current_rho is None:
            raise RuntimeError(
                "current_rho non disponibile. "
                "Chiama estimate_risk() prima di build_graph_for_dijkstra()."
            )

        prof = self._risk_profile or {}
        # usa graph_tau/graph_k se specifici, altrimenti tau/k generali
        if tau is None:
            tau = prof.get("graph_tau", prof.get("tau", None))
        if k is None:
            k = prof.get("graph_k", prof.get("k", None))

        rho_full = self.current_rho

        # 1) Scegli l'universo di nodi
        if use_reduced_universe and self.reduced_universe:
            base_universe = [
                t for t in self.reduced_universe
                if t in rho_full.columns
            ]
        else:
            base_universe = list(rho_full.columns)

        if not base_universe:
            raise ValueError(
                "Universo per il grafo di Dijkstra vuoto. "
                "Controlla current_rho e/o reduced_universe."
            )

        # 2) Sottocampiona la matrice di correlazione all'universo scelto
        rho_sub = rho_full.loc[base_universe, base_universe]

        # 3) Costruisci grafo + adj + dist_knn con GraphBuilder
        G, adj, dist_knn = GraphBuilder.build_filtered_graph(
            rho=rho_sub,
            tau=tau,
            k=k,
            signed=signed,
        )

        # 4) Salva il grafo nel Model
        self.current_graph = G

        return G

    # PORTAFOGLIO VIA DIJKSTRA (MIN CORRELAZIONE DA SOURCE)

    def build_portfolio_dijkstra(
        self,
        source: str,
        K: int,
        use_reduced_universe: bool = True,
        require_rating: bool = False,
        min_rating_score: float = 13.0,
        rho_pair_max: float | None = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Costruisce un portafoglio di dimensione K usando Dijkstra sul grafo di correlazione.
        """
        if self.current_graph is None:
            raise RuntimeError(
                "current_graph non disponibile. "
                "Chiama prima build_reduced_universe() o build_graph_for_dijkstra()."
            )

        if source not in self.current_graph:
            raise ValueError(f"Ticker source '{source}' non presente nel grafo corrente.")

        if K <= 0:
            raise ValueError("K deve essere > 0.")

        # Universo base: U' se richiesto e non vuoto, altrimenti tutti i nodi del grafo
        universe = self._build_dijkstra_universe(
            source=source,
            K=K,
            use_reduced_universe=use_reduced_universe,
            require_rating=require_rating,
            min_rating_score=min_rating_score,
        )

        # Distanze minime dal source usando l'attributo 'distance'
        lengths = self._compute_dijkstra_distances_from_source(source)

        # Consideriamo solo nodi dell'universo e raggiungibili, escluso il source
        candidates_dist = [
            (t, d) for t, d in lengths.items()
            if t in universe and t != source
        ]

        if not candidates_dist:
            # Non ci sono nodi raggiungibili diversi dal source
            return [source], {source: 1.0}

        # ordina per distanza decrescente (più lontani = più decorrelati)
        candidates_dist.sort(key=lambda x: x[1], reverse=True)

        # Costruzione portafoglio
        portfolio: List[str] = [source]

        # Aggiunge titoli finché si arriva a K, rispettando rho_pair_max
        for t, d in candidates_dist:
            if len(portfolio) >= K:
                break
            if not self._ok_rho_pairwise(portfolio, t, rho_pair_max):
                continue
            portfolio.append(t)

        # Se non abbiamo raggiunto K (troppi vincoli), riempiamo ignorando rho_pair_max
        if len(portfolio) < K:
            for t, d in candidates_dist:
                if len(portfolio) >= K:
                    break
                if t in portfolio:
                    continue
                portfolio.append(t)

        # Calcolo pesi (ottimizzati in MV con avversione al rischio 1.0)
        weights = PortfolioWeights.compute(
            tickers=portfolio,
            mode="mv",
            mu=self.current_mu,
            Sigma_sh=self.current_Sigma_sh,
            risk_aversion=1.0,
        )

        return portfolio, weights

    def _build_dijkstra_universe(
        self,
        source: str,
        K: int,
        use_reduced_universe: bool,
        require_rating: bool,
        min_rating_score: float,
    ) -> set[str]:
        """
        Applica tutti i filtri sull'universo per il portafoglio Dijkstra
        (U' vs universo pieno, rating minimo, ecc.).
        """
        if use_reduced_universe and self.reduced_universe:
            universe = set(self.reduced_universe)
        else:
            universe = set(self.current_graph.nodes)

        # Il source deve sempre essere incluso nell'universo
        if source not in universe:
            universe.add(source)

        # Eventuale vincolo di rating
        if require_rating:
            filtered_universe = set()
            for t in universe:
                stock = self.stocks.get(t)
                if stock is None:
                    continue
                rs = stock.rating_score
                if rs is not None and rs >= min_rating_score:
                    filtered_universe.add(t)
            # assicuriamoci di non perdere il source
            stock_src = self.stocks.get(source)
            if (
                stock_src is None
                or stock_src.rating_score is None
                or stock_src.rating_score < min_rating_score
            ):
                # forza l'inclusione del source anche se non rispetta il rating
                filtered_universe.add(source)

            universe = filtered_universe

        if len(universe) < K:
            raise ValueError(
                f"Universo disponibile ({len(universe)}) troppo piccolo per K={K}."
            )

        return universe

    def _compute_dijkstra_distances_from_source(self, source: str) -> Dict[str, float]:
        """
        Calcola le distanze minime da 'source' usando l'attributo 'distance'
        (se presente) oppure 'weight' come fallback.
        """
        assert self.current_graph is not None

        use_attr = "distance"
        any_edge = next(iter(self.current_graph.edges(data=True)), None)
        if any_edge is not None and "distance" not in any_edge[2]:
            use_attr = "weight"

        lengths: Dict[str, float] = nx.single_source_dijkstra_path_length(
            self.current_graph,
            source,
            weight=use_attr,
        )
        return lengths

    def _ok_rho_pairwise(
        self,
        portfolio: List[str],
        new_t: str,
        rho_pair_max: float | None,
    ) -> bool:
        """
        Controlla il vincolo di correlazione massima in valore assoluto
        tra il nuovo titolo e quelli già in portafoglio.
        """
        if rho_pair_max is None or self.current_rho is None:
            return True

        rho_max_val = float(rho_pair_max)
        for t in portfolio:
            if (
                t not in self.current_rho.index
                or new_t not in self.current_rho.columns
            ):
                continue
            cij = self.current_rho.loc[t, new_t]
            if pd.notna(cij) and abs(cij) > rho_max_val:
                return False
        return True

    # WRAPPER ALTO LIVELLO: COSTRUISCI GRAFO + PORTAFOGLIO (DIJKSTRA)

    def build_dijkstra_portfolio(
        self,
        source: str,
        K: int,
        use_reduced_universe: bool = True,
        require_rating: bool = False,
        min_rating_score: float = 13.0,
        rho_pair_max: float | None = None,
        graph_signed: bool = False,
        graph_tau: float | None = None,
        graph_k: int | None = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Metodo di alto livello che:
        1) costruisce il grafo da usare con Dijkstra (build_graph_for_dijkstra),
        2) esegue build_portfolio_dijkstra per ottenere il portafoglio.
        """
        # 1) costruzione/aggiornamento del grafo per Dijkstra
        self.build_graph_for_dijkstra(
            use_reduced_universe=use_reduced_universe,
            signed=graph_signed,
            tau=graph_tau,
            k=graph_k,
        )

        # 2) costruzione del portafoglio tramite Dijkstra
        portfolio, weights = self.build_portfolio_dijkstra(
            source=source,
            K=K,
            use_reduced_universe=use_reduced_universe,
            require_rating=require_rating,
            min_rating_score=min_rating_score,
            rho_pair_max=rho_pair_max,
        )

        # salva ultimo portafoglio Dijkstra (per Monte Carlo)
        self.last_dij_tickers = list(portfolio)
        self.last_dij_weights = dict(weights)

        return portfolio, weights

    # SIMULAZIONI MONTE CARLO

    def simulate_portfolio_paths(
        self,
        tickers: list[str],
        weights: dict[str, float],
        n_paths: int = 100,
        n_days: int = 252,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simula traiettorie di valore di portafoglio in un modello gaussiano
        multivariato, usando mu e Sigma_sh stimati sull'intera storia.

        Restituisce:
        - paths: array shape (n_paths, n_days+1) con i valori simulati
                 (partenza 1.0 al tempo 0).
        - mean_path: array shape (n_days+1,) con la traiettoria media.
        """
        if self.current_mu is None or self.current_Sigma_sh is None:
            raise RuntimeError(
                "mu / Sigma_sh non disponibili. "
                "Assicurati di aver chiamato estimate_risk() prima delle simulazioni."
            )

        if not tickers or not weights:
            raise ValueError("Portafoglio vuoto: impossibile simulare.")

        # vettore pesi nell'ordine dei tickers
        w_vec = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
        if w_vec.sum() <= 0:
            raise ValueError("Somma pesi nulla nel portafoglio da simulare.")
        w_vec = w_vec / w_vec.sum()

        # allineo mu e Sigma_sh all'universo dei tickers
        missing_mu = [t for t in tickers if t not in self.current_mu.index]
        missing_S = [
            t for t in tickers
            if t not in self.current_Sigma_sh.index
            or t not in self.current_Sigma_sh.columns
        ]
        if missing_mu or missing_S:
            raise ValueError(
                f"Titoli mancanti in mu o Sigma_sh. "
                f"missing_mu={missing_mu}, missing_S={missing_S}"
            )

        mu_vec = self.current_mu.loc[tickers].astype(float).values  # (K,)
        Sigma_sub = self.current_Sigma_sh.loc[tickers, tickers].astype(float).values

        # rendimento atteso e varianza del portafoglio
        mu_p = float(w_vec @ mu_vec)
        var_p = float(w_vec @ Sigma_sub @ w_vec)
        sigma_p = np.sqrt(var_p) if var_p > 0 else 0.0

        # processo lognormale su rendimenti log
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # shape (n_paths, n_days)
        shocks = rng.normal(loc=mu_p, scale=sigma_p, size=(n_paths, n_days))
        # valori cumulati (partenza 0 in log → valore 1 in livello)
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(shocks, axis=1)],
            axis=1,
        )
        paths = np.exp(log_paths)  # parte da 1.0

        # traiettoria media
        mean_path = paths.mean(axis=0)

        return paths, mean_path

    def simulate_mc_for_last_portfolios(
        self,
        n_paths: int = 100,
        n_days: int = 252,
        seed_bb: int = 42,
        seed_dij: int = 99,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Esegue la simulazione Monte Carlo per:
        - l'ultimo portafoglio ottimo (B&B) salvato in last_portfolio_tickers/weights,
        - l'ultimo portafoglio Dijkstra (calcolato via build_dijkstra_portfolio).

        Restituisce un dizionario:
            {
                "bb": {"paths": ..., "mean": ...},
                "dij": {"paths": ..., "mean": ...}
            }
        """
        # portafoglio B&B
        if not self.last_portfolio_tickers or not self.last_portfolio_weights:
            raise RuntimeError(
                "Portafoglio B&B non disponibile. "
                "Esegui prima optimize_portfolio dal Controller."
            )

        tickers_bb = list(self.last_portfolio_tickers)
        weights_bb = dict(self.last_portfolio_weights)

        paths_bb, mean_bb = self.simulate_portfolio_paths(
            tickers=tickers_bb,
            weights=weights_bb,
            n_paths=n_paths,
            n_days=n_days,
            seed=seed_bb,
        )

        # portafoglio Dijkstra
        if not self.last_dij_tickers or not self.last_dij_weights:
            raise RuntimeError(
                "Portafoglio Dijkstra non disponibile. "
                "Esegui prima build_dijkstra_portfolio dal Controller."
            )

        tickers_dij = list(self.last_dij_tickers)
        weights_dij = dict(self.last_dij_weights)

        paths_dij, mean_dij = self.simulate_portfolio_paths(
            tickers=tickers_dij,
            weights=weights_dij,
            n_paths=n_paths,
            n_days=n_days,
            seed=seed_dij,
        )

        return {
            "bb": {"paths": paths_bb, "mean": mean_bb},
            "dij": {"paths": paths_dij, "mean": mean_dij},
        }


if __name__ == "__main__":
    # Test rapido complessivo
    import traceback
    import time

    model = Model()

    try:
        # STEP 1: caricamento dati
        model.load_data_from_dao()

        print("=== TEST MODEL (Step 1: load_data_from_dao) ===")
        print(f"Shape prices_df: {model.prices_df.shape if model.prices_df is not None else None}")
        print(f"Shape ratings_df: {model.ratings_df.shape if model.ratings_df is not None else None}")
        print(f"Num stocks: {len(model.stocks)}")
        print(f"Shape returns_df: {model.returns_df.shape if model.returns_df is not None else None}")
        print(f"Ticker con rating: {len(model.tickers_with_rating)}")
        print(f"Ticker senza rating: {len(model.tickers_without_rating)}")

        if model.prices_df is None or model.ratings_df is None or model.returns_df is None:
            raise AssertionError("Dati non caricati correttamente in Model.")
        if len(model.stocks) == 0:
            raise AssertionError("model.stocks è vuoto.")

        # STEP 2: stima rischio su tutta la storia
        print("\n=== TEST MODEL (Step 2: estimate_risk) ===")
        model.estimate_risk(shrink_lambda=0.1)

        rho = model.current_rho
        Sigma_sh = model.current_Sigma_sh
        mu = model.current_mu

        print(f"Shape current_rho: {rho.shape if rho is not None else None}")
        print(f"Shape current_Sigma_sh: {Sigma_sh.shape if Sigma_sh is not None else None}")
        print(f"Shape current_mu: {mu.shape if mu is not None else None}")

        if rho is None or Sigma_sh is None or mu is None:
            raise AssertionError("Le matrici di rischio correnti non sono state stimate.")

        # STEP 3: costruzione universo ridotto U'
        print("\n=== TEST MODEL (Step 3: build_reduced_universe) ===")
        reduced = model.build_reduced_universe(
            tau=0.3,
            k=10,
            max_size=60,
            max_unrated_share=0.0
        )

        print(f"Dimensione universo ridotto U': {len(reduced)}")
        print("Primi 10 ticker in U':", reduced[:10])
        print(
            "Numero nodi nel grafo corrente:",
            model.current_graph.number_of_nodes() if model.current_graph is not None else 0
        )
        print(
            "Numero archi nel grafo corrente:",
            model.current_graph.number_of_edges() if model.current_graph is not None else 0
        )

    except AssertionError as e:
        print("\nTEST MODEL STEP 1–3 FALLITO:")
        print(" -", e)

    except Exception as e:
        print("\nERRORE IMPREVISTO DURANTE IL TEST MODEL (Step 1–3):")
        traceback.print_exc()

    else:
        print("\nTEST MODEL STEP 1–3 OK.")

        # STEP 5–6: optimize_portfolio + pesi
        print("\n=== TEST MODEL (Step 5–6: optimize_portfolio + weights) ===")

        if model.reduced_universe:
            print(f"Ticker in U' (reduced_universe): {len(model.reduced_universe)}")
        else:
            print("Attenzione: reduced_universe è vuoto, userò l'universo completo (current_rho.columns).")

        params = PortfolioSelector.build_default_params()
        params["K"] = 4
        params["max_unrated_share"] = 0.2
        params["rating_min"] = 13.0
        params["max_share_per_sector"] = 0.5

        params["weights_mode"] = "mv"
        params["mv_risk_aversion"] = 1.0

        print("Parametri ottimizzazione:")
        print(f"  K = {params['K']}")
        print(f"  max_unrated_share = {params['max_unrated_share']}")
        print(f"  rating_min = {params['rating_min']}")
        print(f"  max_share_per_sector = {params['max_share_per_sector']}")
        print(f"  weights_mode = {params['weights_mode']}")
        print(f"  mv_risk_aversion = {params['mv_risk_aversion']}")
        print("Avvio optimize_portfolio(...)\n", flush=True)

        t0 = time.time()
        try:
            best_subset, weights, best_score = model.optimize_portfolio(
                params=params,
                use_reduced_universe=True
            )
        except Exception:
            t1 = time.time()
            print(f"ERRORE durante optimize_portfolio dopo {t1 - t0:.2f} secondi:")
            traceback.print_exc()
        else:
            t1 = time.time()
            print(f"optimize_portfolio terminato in {t1 - t0:.2f} secondi.\n")

            print(f"Dim selector_universe: {len(model.selector_universe)}")

            if best_subset is None:
                print("Nessuna soluzione trovata con i vincoli correnti.")
            else:
                print(f"Portafoglio ottimo (len={len(best_subset)}): {best_subset}")
                print(f"Score combinatorio: {best_score}")
                print("Pesi assegnati:")
                print(weights)
                print("Somma pesi:", sum(weights.values()))

                # TEST DIJKSTRA (min-corr da un ticker del portafoglio ottimo)
                print("\n=== TEST MODEL (Dijkstra min-corr da source nel portafoglio ottimo) ===")

                # scegliamo come source il titolo con peso più alto nel portafoglio ottimo
                source = max(weights, key=weights.get)
                print(f"Ticker source scelto dal portafoglio ottimo: {source}")

                t0_d = time.time()
                try:
                    port_dij, w_dij = model.build_dijkstra_portfolio(
                        source=source,
                        K=len(best_subset),      # stesso K del portafoglio ottimo
                        use_reduced_universe=True,
                        require_rating=False,
                        rho_pair_max=0.8,
                    )
                except Exception:
                    t1_d = time.time()
                    print(f"ERRORE durante build_dijkstra_portfolio dopo {t1_d - t0_d:.2f} secondi:")
                    traceback.print_exc()
                else:
                    t1_d = time.time()
                    print(f"build_dijkstra_portfolio terminato in {t1_d - t0_d:.2f} secondi.\n")
                    print("Portafoglio Dijkstra:", port_dij)
                    print("Pesi Dijkstra:", w_dij)
                    print("Somma pesi:", sum(w_dij.values()))
