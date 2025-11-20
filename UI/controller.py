from __future__ import annotations

from typing import Any, Dict, Tuple

import flet as ft
import numpy as np

from model.selector import PortfolioSelector


class Controller:

    def __init__(self, view, model) -> None:
        self._view = view
        self._model = model

    # HANDLER: COSTRUZIONE UNIVERSO RIDOTTO U' (log su pannello ottimizzazione)

    def handle_build_universe(self, e: ft.ControlEvent) -> None:
        try:
            lv = self._view.txt_result_opt
            if lv is None:
                return

            lv.controls.clear()
            lv.controls.append(ft.Text("Costruzione universo ridotto U'..."))

            # lettura dei controlli base per il profilo di rischio
            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )

            # override con eventuali input avanzati U'
            tau = self._parse_optional_float(
                getattr(self._view, "txt_tau", None),
                prof["tau"],
                "soglia correlazione τ (U')",
            )
            k_knn = self._parse_optional_int(
                getattr(self._view, "txt_k_knn", None),
                prof["k"],
                "k per grafo k-NN (U')",
            )
            max_size = self._parse_optional_int(
                getattr(self._view, "txt_max_size", None),
                prof["max_size"],
                "dimensione massima universo U'",
            )

            prof["tau"] = tau
            prof["k"] = k_knn
            prof["max_size"] = max_size

            # opzionale: memorizza il profilo di rischio nel Model
            if hasattr(self._model, "set_risk_profile"):
                self._model.set_risk_profile(prof)

            # CHIAMATA AL MODELLO PER COSTRUIRE U'
            reduced = self._model.build_reduced_universe(
                tau=tau,
                k=k_knn,
                max_size=max_size,
                max_unrated_share=prof["max_unrated_share"],
                min_rating_score=prof["min_rating_score"],
                target_rated_share=prof["target_rated_share"],
            )

            # log dei risultati
            if self._model.current_rho is not None:
                n_total = len(self._model.current_rho.columns)
            else:
                n_total = 0
            n_red = len(reduced)

            n_rated = 0
            n_unrated = 0
            for t in reduced:
                stock = self._model.stocks.get(t)
                if stock is not None and stock.rating_score is not None:
                    if stock.rating_score >= prof["min_rating_score"]:
                        n_rated += 1
                    else:
                        n_unrated += 1
                else:
                    n_unrated += 1

            lv.controls.append(ft.Text(f"Universo completo: {n_total} titoli."))
            lv.controls.append(
                ft.Text(
                    f"Universo ridotto U': {n_red} titoli "
                    f"(rated>=soglia: {n_rated}, speculativi/unrated: {n_unrated})."
                )
            )
            if n_red > 0:
                preview = ", ".join(reduced[:10])
                lv.controls.append(ft.Text(f"Primi 10 ticker in U': {preview}"))

            # ABILITA LE FASI SUCCESSIVE
            self._view.enable_optimize_portfolio(True)
            self._view.enable_dijkstra(True)
            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore costruzione universo ridotto U': {ex}", target="opt")

    def handle_advanced_universe_params(self, e: ft.ControlEvent) -> None:
        # mostra/nasconde il pannello avanzato U'
        if hasattr(self._view, "toggle_universe_advanced_panel"):
            self._view.toggle_universe_advanced_panel()

        try:
            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )
        except Exception as ex:
            self._show_error(f"Errore lettura parametri avanzati U': {ex}", target="opt")
            return

        # precompila i campi avanzati U' con i default, se vuoti
        txt_tau = getattr(self._view, "txt_tau", None)
        if txt_tau is not None and not (txt_tau.value or "").strip():
            txt_tau.value = f"{prof['tau']:.2f}"

        txt_k_knn = getattr(self._view, "txt_k_knn", None)
        if txt_k_knn is not None and not (txt_k_knn.value or "").strip():
            txt_k_knn.value = f"{prof['k']}"

        txt_max_size = getattr(self._view, "txt_max_size", None)
        if txt_max_size is not None and not (txt_max_size.value or "").strip():
            txt_max_size.value = f"{prof['max_size']}"

        self._view.update_page()

    # HANDLER: OTTIMIZZA PORTAFOGLIO (Branch & Bound)

    def handle_optimize_portfolio(self, e: ft.ControlEvent) -> None:
        try:
            lv = self._view.txt_result_opt
            if lv is None:
                return

            lv.controls.clear()

            # LETTURA INPUT PRINCIPALI
            K = self._parse_k()
            capital = self._parse_capital()

            if K <= 0:
                raise ValueError("K deve essere un intero positivo.")

            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )

            params = PortfolioSelector.build_default_params()

            # valori base dai profili di rischio + default PortfolioSelector
            rating_min = prof["rating_min"]
            max_unrated = prof["max_unrated_share"]
            rho_pair_max = prof["rho_pair_max"]
            max_share_per_sector = params.get("max_share_per_sector", 0.5)
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 0.5)
            gamma = params.get("gamma", 0.5)
            delta = params.get("delta", 0.2)
            mv_risk_aversion = params.get("mv_risk_aversion", 1.0)

            # override con eventuali input avanzati B&B
            rating_min = self._parse_optional_float(
                getattr(self._view, "txt_opt_rating_min", None),
                rating_min,
                "rating_min (B&B)",
            )
            max_unrated = self._parse_optional_float(
                getattr(self._view, "txt_opt_max_unrated", None),
                max_unrated,
                "max_unrated_share (B&B)",
            )
            rho_pair_max = self._parse_optional_float(
                getattr(self._view, "txt_opt_rho_pair_max", None),
                rho_pair_max,
                "rho_pair_max (B&B)",
            )
            max_share_per_sector = self._parse_optional_float(
                getattr(self._view, "txt_opt_max_sector", None),
                max_share_per_sector,
                "max_share_per_sector (B&B)",
            )
            alpha = self._parse_optional_float(
                getattr(self._view, "txt_opt_alpha", None),
                alpha,
                "α (B&B)",
            )
            beta = self._parse_optional_float(
                getattr(self._view, "txt_opt_beta", None),
                beta,
                "β (B&B)",
            )
            gamma = self._parse_optional_float(
                getattr(self._view, "txt_opt_gamma", None),
                gamma,
                "γ (B&B)",
            )
            delta = self._parse_optional_float(
                getattr(self._view, "txt_opt_delta", None),
                delta,
                "δ (B&B)",
            )
            mv_risk_aversion = self._parse_optional_float(
                getattr(self._view, "txt_opt_mv_risk_aversion", None),
                mv_risk_aversion,
                "risk_aversion (MV B&B)",
            )

            # aggiorna profilo di rischio con gli override B&B principali
            prof["rating_min"] = rating_min
            prof["max_unrated_share"] = max_unrated
            prof["rho_pair_max"] = rho_pair_max

            if hasattr(self._model, "set_risk_profile"):
                self._model.set_risk_profile(prof)

            # parametri finali per PortfolioSelector
            params["K"] = K
            params["rating_min"] = rating_min
            params["max_unrated_share"] = max_unrated
            params["rho_pair_max"] = rho_pair_max
            params["max_share_per_sector"] = max_share_per_sector
            params["alpha"] = alpha
            params["beta"] = beta
            params["gamma"] = gamma
            params["delta"] = delta
            params["weights_mode"] = "mv"
            params["mv_risk_aversion"] = mv_risk_aversion

            # log dei parametri usati
            lv.controls.append(
                ft.Text(
                    "Ottimizzazione B&B con: "
                    f"K={K}, rating_min={rating_min:.1f}, "
                    f"max_unrated_share={max_unrated:.2f}, "
                    f"rho_pair_max={rho_pair_max:.2f}, "
                    f"max_share_per_sector={max_share_per_sector:.2f}"
                )
            )
            lv.controls.append(
                ft.Text(
                    "Score combinatorio: "
                    f"α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, δ={delta:.2f}; "
                    f"MV risk_aversion={mv_risk_aversion:.2f}"
                )
            )

            # CHIAMATA AL MODELLO PER OTTIMIZZARE
            best_subset, weights, best_score = self._model.optimize_portfolio(
                params=params,
                use_reduced_universe=True,
            )

            if not best_subset:
                lv.controls.append(
                    ft.Text("Nessuna soluzione trovata con i vincoli correnti.")
                )
                self._view.update_page()
                return

            # log del portafoglio risultato
            self._log_portfolio(
                tickers=best_subset,
                weights=weights,
                score=best_score,
                capital=capital,
                title="Portafoglio ottimo (Branch & Bound)",
                lv=lv,
            )

            # suggerisci come source per Dijkstra il titolo più pesato
            if weights and self._view.txt_source is not None:
                source = max(weights, key=weights.get)
                self._view.txt_source.value = source

            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore ottimizzazione portafoglio: {ex}", target="opt")

    def handle_advanced_optimize_params(self, e: ft.ControlEvent) -> None:
        # mostra/nasconde il pannello avanzato B&B
        if hasattr(self._view, "toggle_optimize_advanced_panel"):
            self._view.toggle_optimize_advanced_panel()

        try:
            K = self._parse_k()
            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )
        except Exception as ex:
            self._show_error(
                f"Errore lettura parametri avanzati ottimizzazione: {ex}",
                target="opt",
            )
            return

        params = PortfolioSelector.build_default_params()
        params["K"] = K

        # valori base per precompilazione
        rating_min = prof["rating_min"]
        max_unrated = prof["max_unrated_share"]
        rho_pair_max = prof["rho_pair_max"]
        max_share_per_sector = params.get("max_share_per_sector", 0.5)
        alpha = params.get("alpha", 1.0)
        beta = params.get("beta", 0.5)
        gamma = params.get("gamma", 0.5)
        delta = params.get("delta", 0.2)
        mv_risk_aversion = params.get("mv_risk_aversion", 1.0)

        # precompila i campi avanzati con i default se vuoti
        if self._view.txt_opt_rating_min is not None and not (self._view.txt_opt_rating_min.value or "").strip():
            self._view.txt_opt_rating_min.value = f"{rating_min:.1f}"

        if self._view.txt_opt_max_unrated is not None and not (self._view.txt_opt_max_unrated.value or "").strip():
            self._view.txt_opt_max_unrated.value = f"{max_unrated:.2f}"

        if self._view.txt_opt_rho_pair_max is not None and not (self._view.txt_opt_rho_pair_max.value or "").strip():
            self._view.txt_opt_rho_pair_max.value = f"{rho_pair_max:.2f}"

        if self._view.txt_opt_max_sector is not None and not (self._view.txt_opt_max_sector.value or "").strip():
            self._view.txt_opt_max_sector.value = f"{max_share_per_sector:.2f}"

        if self._view.txt_opt_alpha is not None and not (self._view.txt_opt_alpha.value or "").strip():
            self._view.txt_opt_alpha.value = f"{alpha:.2f}"

        if self._view.txt_opt_beta is not None and not (self._view.txt_opt_beta.value or "").strip():
            self._view.txt_opt_beta.value = f"{beta:.2f}"

        if self._view.txt_opt_gamma is not None and not (self._view.txt_opt_gamma.value or "").strip():
            self._view.txt_opt_gamma.value = f"{gamma:.2f}"

        if self._view.txt_opt_delta is not None and not (self._view.txt_opt_delta.value or "").strip():
            self._view.txt_opt_delta.value = f"{delta:.2f}"

        if self._view.txt_opt_mv_risk_aversion is not None and not (
                self._view.txt_opt_mv_risk_aversion.value or "").strip():
            self._view.txt_opt_mv_risk_aversion.value = f"{mv_risk_aversion:.2f}"

        self._view.update_page()

    # HANDLER: DIJKSTRA – costruzione portafoglio min-correlazione

    def handle_build_dijkstra(self, e: ft.ControlEvent) -> None:
        try:
            lv = self._view.txt_result_dij
            if lv is None:
                return

            lv.controls.clear()

            # LETTURA INPUT PRINCIPALI
            K = self._parse_k()
            if K <= 0:
                raise ValueError("K deve essere un intero positivo.")

            source = ""
            if self._view.txt_source is not None and self._view.txt_source.value:
                source = self._view.txt_source.value.strip().upper()
            if not source:
                raise ValueError("Devi specificare un ticker sorgente per Dijkstra.")

            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )

            use_reduced_universe = self._get_dijkstra_use_reduced()

            # default da profilo
            graph_tau = prof["tau"]
            graph_k = prof["k"]
            rho_pair_max = prof["rho_pair_max"]
            rating_min = prof["rating_min"]
            require_rating_default = prof["max_unrated_share"] <= 0.0
            graph_signed = False

            # override con input avanzati Dijkstra
            graph_tau = self._parse_optional_float(
                getattr(self._view, "txt_dij_tau", None),
                graph_tau,
                "τ Dijkstra",
            )
            graph_k = self._parse_optional_int(
                getattr(self._view, "txt_dij_k", None),
                graph_k,
                "k Dijkstra",
            )
            rho_pair_max = self._parse_optional_float(
                getattr(self._view, "txt_dij_rho_pair_max", None),
                rho_pair_max,
                "rho_pair_max Dijkstra",
            )
            rating_min = self._parse_optional_float(
                getattr(self._view, "txt_dij_rating_min", None),
                rating_min,
                "rating_min Dijkstra",
            )

            require_rating = require_rating_default
            chk_req = getattr(self._view, "chk_dij_require_rating", None)
            if chk_req is not None:
                require_rating = bool(chk_req.value)

            chk_signed = getattr(self._view, "chk_dij_signed", None)
            if chk_signed is not None:
                graph_signed = bool(chk_signed.value)

            # aggiorna profilo di rischio con i valori rilevanti
            prof["rho_pair_max"] = rho_pair_max
            prof["rating_min"] = rating_min
            prof["tau"] = graph_tau
            prof["k"] = graph_k

            if hasattr(self._model, "set_risk_profile"):
                self._model.set_risk_profile(prof)

            # log dei parametri usati
            lv.controls.append(
                ft.Text(
                    f"Costruzione portafoglio min-correlazione da source={source}, "
                    f"K={K}, "
                    f"{'uso universo ridotto U\'' if use_reduced_universe else 'uso universo completo'}; "
                    f"τ={graph_tau:.2f}, k={graph_k}, "
                    f"rho_pair_max={rho_pair_max:.2f}, "
                    f"rating_min={rating_min:.1f}, "
                    f"{'distanza signed (1-ρ)/2' if graph_signed else 'distanza 1-|ρ|'}; "
                    f"rating obbligatorio titoli={'sì' if require_rating else 'no'}"
                )
            )

            # CHIAMATA AL MODELLO (Wrapper alto livello: costruisce grafo + portafoglio)
            port_dij, w_dij = self._model.build_dijkstra_portfolio(
                source=source,
                K=K,
                use_reduced_universe=use_reduced_universe,
                require_rating=require_rating,
                min_rating_score=rating_min,
                rho_pair_max=rho_pair_max,
                graph_signed=graph_signed,
                graph_tau=graph_tau,
                graph_k=graph_k,
            )

            if not port_dij:
                lv.controls.append(ft.Text("Portafoglio Dijkstra vuoto."))
                self._view.update_page()
                return

            capital = self._parse_capital()
            self._log_dijkstra(
                source=source,
                tickers=port_dij,
                weights=w_dij,
                capital=capital,
                lv=lv,
            )

            # abilita Monte Carlo dopo un portafoglio Dijkstra valido
            self._view.enable_montecarlo(True)

            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore portafoglio Dijkstra: {ex}", target="dij")

    def handle_advanced_dijkstra_params(self, e: ft.ControlEvent) -> None:
        # mostra/nasconde il pannello avanzato Dijkstra
        if hasattr(self._view, "toggle_dijkstra_advanced_panel"):
            self._view.toggle_dijkstra_advanced_panel()

        try:
            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )
        except Exception as ex:
            self._show_error(f"Errore lettura parametri avanzati Dijkstra: {ex}", target="dij")
            return

        # valori base per precompilazione
        tau = prof["tau"]
        k = prof["k"]
        rho_pair_max = prof["rho_pair_max"]
        rating_min = prof["rating_min"]
        require_rating_default = prof["max_unrated_share"] <= 0.0
        signed_default = False

        # precompila campi avanzati Dijkstra con i default se vuoti
        if self._view.txt_dij_tau is not None and not (self._view.txt_dij_tau.value or "").strip():
            self._view.txt_dij_tau.value = f"{tau:.2f}"

        if self._view.txt_dij_k is not None and not (self._view.txt_dij_k.value or "").strip():
            self._view.txt_dij_k.value = f"{k}"

        if self._view.txt_dij_rho_pair_max is not None and not (self._view.txt_dij_rho_pair_max.value or "").strip():
            self._view.txt_dij_rho_pair_max.value = f"{rho_pair_max:.2f}"

        if self._view.txt_dij_rating_min is not None and not (self._view.txt_dij_rating_min.value or "").strip():
            self._view.txt_dij_rating_min.value = f"{rating_min:.1f}"

        if self._view.chk_dij_require_rating is not None:
            self._view.chk_dij_require_rating.value = require_rating_default

        if self._view.chk_dij_signed is not None:
            self._view.chk_dij_signed.value = signed_default

        self._view.update_page()

    # HANDLER: MONTECARLO

    def handle_montecarlo(self, e: ft.ControlEvent) -> None:
        try:
            lv = self._view.txt_result_mc
            if lv is None:
                return

            lv.controls.clear()

            # orizzonte: uso gli stessi controlli di rischio (slider anni)
            years, risk_level, max_unrated_share = self._read_risk_controls()
            n_days = years * 252  # approssimazione giorni di borsa

            # numero di traiettorie: default 100, overridabile da campo view
            n_paths_default = 100
            txt_n_paths = getattr(self._view, "txt_mc_n_paths", None)
            n_paths = self._parse_optional_int(
                txt_n_paths,
                n_paths_default,
                "numero traiettorie Monte Carlo",
            )

            # log iniziale
            lv.controls.append(
                ft.Text(
                    f"Simulazione Monte Carlo con {n_paths} traiettorie, "
                    f"orizzonte {years} anni (~{n_days} giorni di borsa)."
                )
            )

            # chiamata al Model: usa gli ultimi portafogli B&B e Dijkstra
            results = self._model.simulate_mc_for_last_portfolios(
                n_paths=n_paths,
                n_days=n_days,
            )

            # per ogni portafoglio (B&B e Dijkstra) calcolo statistiche sui valori finali
            for key, label in [("bb", "B&B"), ("dij", "Dijkstra")]:
                data = results.get(key)
                if not data:
                    continue

                paths = data["paths"]      # shape (n_paths, n_days+1)
                mean_path = data["mean"]   # shape (n_days+1,)

                if paths is None or paths.size == 0:
                    lv.controls.append(
                        ft.Text(f"Portafoglio {label}: nessun percorso simulato.")
                    )
                    continue

                final_vals = paths[:, -1]  # valore a fine orizzonte (partenza 1.0)
                mean_final = float(final_vals.mean())
                std_final = float(final_vals.std(ddof=0))
                p5, p50, p95 = np.percentile(final_vals, [5, 50, 95])
                prob_loss = float((final_vals < 1.0).mean())

                lv.controls.append(ft.Text(f"--- Portafoglio {label} ---"))
                lv.controls.append(
                    ft.Text(
                        f"Valore atteso finale: {mean_final:.3f} (partenza 1.000); "
                        f"deviazione std: {std_final:.3f}"
                    )
                )
                lv.controls.append(
                    ft.Text(
                        f"Quantili finali (valore portafoglio): "
                        f"5%={p5:.3f}, mediana={p50:.3f}, 95%={p95:.3f}"
                    )
                )
                lv.controls.append(
                    ft.Text(
                        f"Probabilità di perdita (valore finale < 1.0): "
                        f"{prob_loss:.1%}"
                    )
                )

            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore simulazione Monte Carlo: {ex}", target="mc")

    # HELPER LETTURA INPUT

    def _parse_capital(self) -> float:
        if self._view.txt_capital is None:
            raise ValueError("Inserire capitale.")

        raw = (self._view.txt_capital.value or "").strip()
        if raw == "":
            raise ValueError("Inserire capitale.")

        # normalizzazione dell'input per gestire "," e "." come separatori decimali
        raw_normalized = raw.replace(" ", "")
        if raw_normalized.count(",") == 1 and raw_normalized.count(".") > 1:
            raw_normalized = raw_normalized.replace(".", "").replace(",", ".")
        else:
            raw_normalized = raw_normalized.replace(",", ".")

        try:
            value = float(raw_normalized)
        except ValueError:
            raise ValueError("Capitale non valido. Inserire un numero (es. 100000).")

        if value <= 0:
            raise ValueError("Capitale deve essere > 0.")
        return value

    def _parse_k(self) -> int:
        if self._view.txt_k is None:
            raise ValueError("Campo K non disponibile nella View.")

        raw = (self._view.txt_k.value or "").strip()
        if raw == "":
            raise ValueError("Inserire numero titoli K.")

        if not raw.isdigit():
            raise ValueError("K deve essere un intero positivo.")

        k = int(raw)
        if k <= 0:
            raise ValueError("K deve essere un intero positivo.")
        return k

    def _read_risk_controls(self) -> Tuple[int, int, float]:
        # lettura degli slider per Anni, Livello di Rischio e Max Unrated Share
        years = 10
        if self._view.sld_years is not None:
            idx = int(round(self._view.sld_years.value))
            idx = max(0, min(3, idx))
            years_map = [5, 10, 20, 30]
            years = years_map[idx]

        risk_level = 2
        if self._view.sld_risk is not None:
            val = int(round(self._view.sld_risk.value))
            risk_level = max(1, min(4, val))

        max_unrated_share = 0.25
        if self._view.sld_max_unrated is not None:
            idx = int(round(self._view.sld_max_unrated.value))
            idx = max(0, min(3, idx))
            perc_map = [0.0, 0.25, 0.5, 1.0]
            max_unrated_share = perc_map[idx]

        return years, risk_level, max_unrated_share

    def _get_risk_profile_params(
            self,
            risk_level: int,
            max_unrated_share: float,
            years: int,
    ) -> Dict[str, Any]:
        # imposta i parametri di default per U'
        tau = 0.30
        k_for_knn = 10
        max_size = 60

        # imposta i parametri specifici in base al livello di rischio
        if risk_level == 1:  # Basso
            min_rating_score = 15.0
            target_rated_share = 0.85
            rho_pair_max = 0.70
        elif risk_level == 2:  # Medio
            min_rating_score = 13.0
            target_rated_share = 0.70
            rho_pair_max = 0.80
        elif risk_level == 3:  # Alto
            min_rating_score = 10.0
            target_rated_share = 0.60
            rho_pair_max = 0.90
        else:  # Personalizzato / Molto Alto
            min_rating_score = 10.0
            target_rated_share = 0.50
            rho_pair_max = 0.95

        max_unrated = max_unrated_share

        return {
            "tau": tau,
            "k": k_for_knn,
            "max_size": max_size,
            "max_unrated_share": max_unrated,
            "min_rating_score": min_rating_score,
            "target_rated_share": target_rated_share,
            "rating_min": min_rating_score,
            "rho_pair_max": rho_pair_max,
        }

    def _get_dijkstra_use_reduced(self) -> bool:
        """
        Legge (se presente) il controllo della View per usare/non usare U'
        in Dijkstra. Di default True (usa U').
        """
        use_reduced = True
        ctrl = getattr(self._view, "chk_dij_use_reduced", None)
        if ctrl is not None:
            try:
                use_reduced = bool(ctrl.value)
            except Exception:
                use_reduced = True
        return use_reduced

    def _parse_optional_float(
            self,
            field: ft.TextField | None,
            default: float,
            field_name: str,
    ) -> float:
        """
        Se il campo è vuoto o None → default.
        Se è valorizzato ma non parsabile come float → solleva ValueError.
        """
        if field is None:
            return default

        raw = (field.value or "").strip()
        if raw == "":
            return default

        # normalizza il separatore decimale
        raw_norm = raw.replace(" ", "").replace(",", ".")
        try:
            return float(raw_norm)
        except ValueError:
            raise ValueError(f"Valore non valido per {field_name}: '{raw}'.")

    def _parse_optional_int(
            self,
            field: ft.TextField | None,
            default: int,
            field_name: str,
    ) -> int:
        """
        Se il campo è vuoto o None → default.
        Se è valorizzato ma non parsabile come int → solleva ValueError.
        """
        if field is None:
            return default

        raw = (field.value or "").strip()
        if raw == "":
            return default

        # verifica che sia un intero
        if not (raw.lstrip("+-").isdigit()):
            raise ValueError(f"Valore non valido per {field_name}: '{raw}' (atteso intero).")

        return int(raw)

    # LOG

    def _log_portfolio(
            self,
            tickers: list[str],
            weights: Dict[str, float],
            score: float | None,
            capital: float | None,
            title: str,
            lv: ft.ListView,
    ) -> None:
        if lv is None:
            return

        if title:
            lv.controls.append(ft.Text(title))
        if score is not None:
            lv.controls.append(ft.Text(f"Score combinatorio: {score:.4f}"))

        total_w = sum(weights.values()) if weights else 0.0
        lv.controls.append(ft.Text(f"Numero titoli: {len(tickers)}"))
        lv.controls.append(ft.Text(f"Somma pesi: {total_w:.4f}"))

        # dettagli per ogni titolo
        for t in tickers:
            w = weights.get(t, 0.0)
            stock = self._model.stocks.get(t)

            line = f"{t}: w = {w:.2%}"

            if capital is not None:
                alloc = capital * w
                line += f"  → {alloc:,.2f} EUR"

            # aggiungi rating e settore
            if stock is not None:
                if stock.rating_score is not None:
                    line += f"  | rating = {stock.rating_score:.1f}"
                else:
                    line += "  | rating = unknown"

                if stock.sector is not None:
                    line += f"  | settore = {stock.sector}"
                else:
                    line += "  | settore = unknown"
            else:
                line += "  | rating = unknown  | settore = unknown"

            lv.controls.append(ft.Text(line))

    def _log_dijkstra(
            self,
            source: str,
            tickers: list[str],
            weights: Dict[str, float],
            capital: float | None,
            lv: ft.ListView,
    ) -> None:
        """
        Logger specifico per il portafoglio min-correlazione via Dijkstra.
        """
        if lv is None:
            return

        # header
        lv.controls.append(
            ft.Text(f"Portafoglio min-correlazione (Dijkstra) da source={source}")
        )

        # riuso la logica di _log_portfolio
        self._log_portfolio(
            tickers=tickers,
            weights=weights,
            score=None,
            capital=capital,
            title="",  # titolo già inserito sopra
            lv=lv,
        )

    # GESTIONE ERRORI

    def _show_error(self, message: str, target: str = "opt") -> None:
        if target == "dij":
            lv = self._view.txt_result_dij
        elif target == "mc":
            lv = self._view.txt_result_mc
        else:
            lv = self._view.txt_result_opt

        if lv is None:
            return

        lv.controls.clear()
        lv.controls.append(ft.Text(message, color="red"))
        self._view.update_page()
