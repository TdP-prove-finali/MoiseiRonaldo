from __future__ import annotations

from typing import Any, Dict, Tuple

import flet as ft

from model.selector import PortfolioSelector


class Controller:


    def __init__(self, view, model) -> None:
        self._view = view
        self._model = model

    def handle_build_universe(self, e: ft.ControlEvent) -> None:
        try:
            # pulizia area risultati
            self._view.txt_result.controls.clear()
            self._view.txt_result.controls.append(ft.Text("Costruzione universo ridotto U'..."))

            # Parametri di rischio dalla View
            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )

            # Costruzione universo ridotto
            reduced = self._model.build_reduced_universe(
                tau=prof["tau"],
                k=prof["k"],
                max_size=prof["max_size"],
                max_unrated_share=prof["max_unrated_share"],
                min_rating_score=prof["min_rating_score"],
                target_rated_share=prof["target_rated_share"],
            )

            # Info su universo completo
            if self._model.current_rho is not None:
                n_total = len(self._model.current_rho.columns)
            else:
                n_total = 0
            n_red = len(reduced)

            # Conteggio rated vs unrated
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

            self._view.txt_result.controls.append(ft.Text(f"Universo completo: {n_total} titoli."))
            self._view.txt_result.controls.append(
                ft.Text(
                    f"Universo ridotto U': {n_red} titoli "
                    f"(rated>=soglia: {n_rated}, speculativi/unrated: {n_unrated})."
                )
            )
            if n_red > 0:
                preview = ", ".join(reduced[:10])
                self._view.txt_result.controls.append(ft.Text(f"Primi 10 ticker in U': {preview}"))

            # Dopo aver costruito U' abilito ottimizzazione e Dijkstra
            self._view.enable_optimize_portfolio(True)
            self._view.enable_dijkstra(True)

            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore costruzione universo ridotto U': {ex}")

    def handle_advanced_universe_params(self, e: ft.ControlEvent) -> None:
        """
        Mostra/nasconde il pannello dei parametri avanzati U', se la View lo offre.
        """
        if hasattr(self._view, "toggle_universe_advanced_panel"):
            self._view.toggle_universe_advanced_panel()
        else:
            self._view.txt_result.controls.append(
                ft.Text("Parametri avanzati per U' non ancora implementati.")
            )
        self._view.update_page()

    def handle_optimize_portfolio(self, e: ft.ControlEvent) -> None:
        try:
            # reset area risultati
            self._view.txt_result.controls.clear()

            # Capitale e K obbligatori
            K = self._parse_k()
            capital = self._parse_capital()

            if K <= 0:
                raise ValueError("K deve essere un intero positivo.")

            # Parametri di rischio
            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )

            # Parametri per PortfolioSelector
            params = PortfolioSelector.build_default_params()
            params["K"] = K
            params["rating_min"] = prof["rating_min"]
            params["max_unrated_share"] = prof["max_unrated_share"]
            params["rho_pair_max"] = prof["rho_pair_max"]
            params["max_share_per_sector"] = 0.5
            params["weights_mode"] = "mv"
            params["mv_risk_aversion"] = 1.0

            self._view.txt_result.controls.append(
                ft.Text(
                    f"Ottimizzazione B&B con K={K}, rating_min={params['rating_min']}, "
                    f"max_unrated_share={params['max_unrated_share']:.2f}, "
                    f"rho_pair_max={params['rho_pair_max']:.2f}..."
                )
            )

            best_subset, weights, best_score = self._model.optimize_portfolio(
                params=params,
                use_reduced_universe=True,
            )

            if not best_subset:
                self._view.txt_result.controls.append(
                    ft.Text("Nessuna soluzione trovata con i vincoli correnti.")
                )
                self._view.update_page()
                return

            self._log_portfolio(
                tickers=best_subset,
                weights=weights,
                score=best_score,
                capital=capital,
                title="Portafoglio ottimo (Branch & Bound)",
            )

            # Precompila ticker sorgente per Dijkstra con quello a peso massimo
            if weights and self._view.txt_source is not None:
                source = max(weights, key=weights.get)
                self._view.txt_source.value = source

            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore ottimizzazione portafoglio: {ex}")

    def handle_advanced_optimize_params(self, e: ft.ControlEvent) -> None:
        self._view.txt_result.controls.append(
            ft.Text("Parametri avanzati di ottimizzazione non ancora implementati.")
        )
        self._view.update_page()


    def handle_build_dijkstra(self, e: ft.ControlEvent) -> None:
        try:
            # reset area risultati
            self._view.txt_result.controls.clear()

            # K obbligatorio
            K = self._parse_k()
            if K <= 0:
                raise ValueError("K deve essere un intero positivo.")

            # Ticker sorgente
            source = ""
            if self._view.txt_source is not None and self._view.txt_source.value:
                source = self._view.txt_source.value.strip().upper()
            if not source:
                raise ValueError("Devi specificare un ticker sorgente per Dijkstra.")

            # Parametri di rischio
            years, risk_level, max_unrated_share = self._read_risk_controls()
            prof = self._get_risk_profile_params(
                risk_level=risk_level,
                max_unrated_share=max_unrated_share,
                years=years,
            )

            require_rating = prof["max_unrated_share"] <= 0.0

            self._view.txt_result.controls.append(
                ft.Text(
                    f"Costruzione portafoglio min-correlazione da source={source}, K={K}..."
                )
            )

            port_dij, w_dij = self._model.build_portfolio_dijkstra(
                source=source,
                K=K,
                use_reduced_universe=True,
                require_rating=require_rating,
                min_rating_score=prof["rating_min"],
                rho_pair_max=prof["rho_pair_max"],
            )

            if not port_dij:
                self._view.txt_result.controls.append(ft.Text("Portafoglio Dijkstra vuoto."))
                self._view.update_page()
                return

            # Capitale obbligatorio se vuoi anche le allocazioni in EUR
            capital = self._parse_capital()
            self._log_portfolio(
                tickers=port_dij,
                weights=w_dij,
                score=None,
                capital=capital,
                title="Portafoglio min-correlazione (Dijkstra)",
            )

            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore portafoglio Dijkstra: {ex}")

    def handle_advanced_dijkstra_params(self, e: ft.ControlEvent) -> None:
        self._view.txt_result.controls.append(
            ft.Text("Parametri avanzati Dijkstra non ancora implementati.")
        )
        self._view.update_page()


    def _parse_capital(self) -> float:
        """
        Legge il capitale dalla TextField.
        Se vuoto/non valido → ValueError.
        """
        if self._view.txt_capital is None:
            raise ValueError("Inserire capitale.")

        raw = (self._view.txt_capital.value or "").strip()
        if raw == "":
            raise ValueError("Inserire capitale.")

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
        """
        Legge K dalla TextField.
        Se vuoto/non valido → ValueError.
        """
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
        """
        Legge:
            - anni di investimento (5,10,20,30)
            - livello di rischio (1..4)
            - max_unrated_share (0,0.25,0.5,1.0)
        a partire dagli slider della View.
        """
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
        """
        Mappa (risk_level, max_unrated_share, years) in un set di parametri
        usati da:
            - build_reduced_universe
            - optimize_portfolio
            - build_portfolio_dijkstra
        """
        tau = 0.30
        k_for_knn = 10
        max_size = 60

        if risk_level == 1:
            min_rating_score = 15.0
            target_rated_share = 0.85
            rho_pair_max = 0.70
        elif risk_level == 2:
            min_rating_score = 13.0
            target_rated_share = 0.70
            rho_pair_max = 0.80
        elif risk_level == 3:
            min_rating_score = 10.0
            target_rated_share = 0.60
            rho_pair_max = 0.90
        else:
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

    def _log_portfolio(
        self,
        tickers: list[str],
        weights: Dict[str, float],
        score: float | None,
        capital: float | None,
        title: str,
    ) -> None:
        self._view.txt_result.controls.append(ft.Text(title))
        if score is not None:
            self._view.txt_result.controls.append(
                ft.Text(f"Score combinatorio: {score:.4f}")
            )

        total_w = sum(weights.values()) if weights else 0.0
        self._view.txt_result.controls.append(
            ft.Text(f"Numero titoli: {len(tickers)}")
        )
        self._view.txt_result.controls.append(
            ft.Text(f"Somma pesi: {total_w:.4f}")
        )

        for t in tickers:
            w = weights.get(t, 0.0)
            stock = self._model.stocks.get(t)

            line = f"{t}: w = {w:.2%}"

            if capital is not None:
                alloc = capital * w
                line += f"  → {alloc:,.2f} EUR"

            if stock is not None:
                if stock.rating_score is not None:
                    line += f"  | rating = {stock.rating_score:.1f}"
                if stock.sector is not None:
                    line += f"  | settore = {stock.sector}"

            self._view.txt_result.controls.append(ft.Text(line))

    def _show_error(self, message: str) -> None:
        """
        Mostra un messaggio di errore in txt_result.
        """
        self._view.txt_result.controls.clear()
        self._view.txt_result.controls.append(ft.Text(message, color="red"))
        self._view.update_page()
