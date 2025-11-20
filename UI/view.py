import flet as ft
from datetime import date, timedelta


class View(ft.UserControl):

    def __init__(self, page: ft.Page):
        super().__init__()
        self._page = page
        self._page.title = "Portfolio Optimizer – Thesis Prototype"
        self._page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self._page.theme_mode = ft.ThemeMode.DARK
        self._page.window_height = 800
        self._page.window_width = 1300
        self._page.window_center()
        self._page.scroll = ft.ScrollMode.AUTO

        self._controller = None

        self._theme_switch: ft.Switch | None = None

        # campi di input per dati generali
        self._txt_capital: ft.TextField | None = None
        self._txt_k: ft.TextField | None = None

        self._sld_years: ft.Slider | None = None
        self._lbl_years: ft.Text | None = None

        self._sld_risk: ft.Slider | None = None
        self._lbl_risk: ft.Text | None = None

        self._sld_max_unrated: ft.Slider | None = None
        self._lbl_max_unrated: ft.Text | None = None

        # pulsanti per costruzione U'
        self._btn_build_u: ft.ElevatedButton | None = None
        self._btn_build_u_adv: ft.TextButton | None = None

        # Pannello parametri avanzati U'
        self._universe_adv_panel: ft.Container | None = None
        self._txt_tau: ft.TextField | None = None
        self._txt_k_knn: ft.TextField | None = None
        self._txt_max_size: ft.TextField | None = None

        # Sezione ottimizzazione (B&B)
        self._btn_optimize: ft.ElevatedButton | None = None
        self._btn_optimize_adv: ft.TextButton | None = None
        self._opt_adv_panel: ft.Container | None = None

        # Campi avanzati B&B
        self._txt_opt_rating_min: ft.TextField | None = None
        self._txt_opt_max_unrated: ft.TextField | None = None
        self._txt_opt_rho_pair_max: ft.TextField | None = None
        self._txt_opt_max_sector: ft.TextField | None = None
        self._txt_opt_alpha: ft.TextField | None = None
        self._txt_opt_beta: ft.TextField | None = None
        self._txt_opt_gamma: ft.TextField | None = None
        self._txt_opt_delta: ft.TextField | None = None
        self._txt_opt_mv_risk_aversion: ft.TextField | None = None

        # Sezione Dijkstra
        self._txt_source: ft.TextField | None = None
        self._chk_dij_use_reduced: ft.Checkbox | None = None
        self._btn_dijkstra: ft.ElevatedButton | None = None
        self._btn_dijkstra_adv: ft.TextButton | None = None
        self._dij_adv_panel: ft.Container | None = None

        # Campi avanzati Dijkstra
        self._txt_dij_tau: ft.TextField | None = None
        self._txt_dij_k: ft.TextField | None = None
        self._txt_dij_rho_pair_max: ft.TextField | None = None
        self._txt_dij_rating_min: ft.TextField | None = None
        self._chk_dij_require_rating: ft.Checkbox | None = None
        self._chk_dij_signed: ft.Checkbox | None = None

        # Monte Carlo
        self._btn_mc_sim: ft.ElevatedButton | None = None
        self._txt_mc_n_paths: ft.TextField | None = None
        self._txt_mc_start: ft.TextField | None = None
        self._txt_mc_end: ft.TextField | None = None
        self._mc_base_date = None  # data "oggi" fissata alla creazione UI

        # ListView per i risultati
        self.txt_result_opt: ft.ListView | None = None      # risultati universo + B&B
        self.txt_result_dij: ft.ListView | None = None      # risultati Dijkstra
        self.txt_result_mc: ft.ListView | None = None       # risultati Monte Carlo

    def load_interface(self) -> None:
        # HEADER: tema + titolo
        self._theme_switch = ft.Switch(
            label="Dark theme",
            value=True,
            on_change=self._on_theme_change,
        )

        title_txt = ft.Text(
            "Portfolio Optimizer",
            weight=ft.FontWeight.BOLD,
            size=24,
        )

        header_row = ft.Row(
            controls=[
                self._theme_switch,
                ft.Container(expand=True),
                title_txt,
                ft.Container(expand=True),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # SEZIONE INPUT DATI
        self._txt_capital = ft.TextField(
            label="Capitale (EUR)",
            width=200,
            text_align=ft.TextAlign.RIGHT,
        )

        self._txt_k = ft.TextField(
            label="Numero titoli portafoglio K",
            width=220,
            text_align=ft.TextAlign.RIGHT,
        )

        row_inputs = ft.Row(
            controls=[
                self._txt_capital,
                self._txt_k,
            ],
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
        )

        self._lbl_years = ft.Text("Anni invest.: 10")
        self._sld_years = ft.Slider(
            min=0,
            max=3,
            divisions=3,
            value=1,
            width=300,
            on_change=self._on_years_change,
        )

        col_years = ft.Column(
            controls=[self._lbl_years, self._sld_years],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._lbl_risk = ft.Text("Rischio: 2 (medio)")
        self._sld_risk = ft.Slider(
            min=1,
            max=4,
            divisions=3,
            value=2,
            width=300,
            on_change=self._on_risk_change,
        )

        col_risk = ft.Column(
            controls=[self._lbl_risk, self._sld_risk],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._lbl_max_unrated = ft.Text("Max unrated: 25%")
        self._sld_max_unrated = ft.Slider(
            min=0,
            max=3,
            divisions=3,
            value=1,
            width=300,
            on_change=self._on_unrated_change,
        )

        col_unrated = ft.Column(
            controls=[self._lbl_max_unrated, self._sld_max_unrated],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        row_sliders = ft.Row(
            controls=[col_years, col_risk, col_unrated],
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
        )

        data_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Dati di input", weight=ft.FontWeight.BOLD),
                    row_inputs,
                    row_sliders,
                ],
                spacing=15,
            ),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.05, ft.colors.ON_SURFACE),
        )

        # SEZIONE: COSTRUISCI U'
        self._btn_build_u = ft.ElevatedButton(
            text="1. Costruisci universo ridotto U'",
            disabled=False,
            on_click=lambda e: (
                self._controller.handle_build_universe(e)
                if self._controller is not None
                else None
            ),
        )
        self._btn_build_u_adv = ft.TextButton(
            text="Parametri avanzati",
            disabled=False,
            on_click=lambda e: (
                self._controller.handle_advanced_universe_params(e)
                if self._controller is not None
                else None
            ),
        )

        row_build_u = ft.Row(
            controls=[self._btn_build_u, self._btn_build_u_adv],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self._txt_tau = ft.TextField(
            label="Soglia correlazione τ (0–1)",
            width=180,
            hint_text="es. 0.30",
        )
        self._txt_k_knn = ft.TextField(
            label="k per grafo k-NN",
            width=150,
            hint_text="es. 10",
        )
        self._txt_max_size = ft.TextField(
            label="Dimensione max universo U'",
            width=200,
            hint_text="es. 60",
        )

        adv_row_1 = ft.Row(
            controls=[self._txt_tau, self._txt_k_knn, self._txt_max_size],
            alignment=ft.MainAxisAlignment.START,
        )

        self._universe_adv_panel = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Parametri avanzati U'", weight=ft.FontWeight.BOLD),
                    adv_row_1,
                ],
                spacing=10,
            ),
            padding=10,
            border_radius=8,
            bgcolor=ft.colors.with_opacity(0.03, ft.colors.ON_SURFACE),
            visible=False,
        )

        # SEZIONE: OTTIMIZZA PORTAFOGLIO (Branch & Bound)
        self._btn_optimize = ft.ElevatedButton(
            text="2. Ottimizza portafoglio (Branch & Bound)",
            disabled=True,
            on_click=lambda e: (
                self._controller.handle_optimize_portfolio(e)
                if self._controller is not None
                else None
            ),
        )
        self._btn_optimize_adv = ft.TextButton(
            text="Parametri avanzati",
            disabled=True,
            on_click=lambda e: (
                self._controller.handle_advanced_optimize_params(e)
                if self._controller is not None
                else None
            ),
        )

        row_optimize = ft.Row(
            controls=[self._btn_optimize, self._btn_optimize_adv],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self._txt_opt_rating_min = ft.TextField(
            label="rating_min",
            width=130,
            hint_text="es. 13.0",
        )
        self._txt_opt_max_unrated = ft.TextField(
            label="max_unrated_share (0–1)",
            width=160,
            hint_text="es. 0.25",
        )
        self._txt_opt_rho_pair_max = ft.TextField(
            label="rho_pair_max (|ρ_ij| max)",
            width=170,
            hint_text="es. 0.80",
        )
        self._txt_opt_max_sector = ft.TextField(
            label="max_share_per_sector (0–1)",
            width=180,
            hint_text="es. 0.50",
        )

        self._txt_opt_alpha = ft.TextField(
            label="α (-corr)",
            width=110,
            hint_text="1.0",
        )
        self._txt_opt_beta = ft.TextField(
            label="β (rating)",
            width=110,
            hint_text="0.5",
        )
        self._txt_opt_gamma = ft.TextField(
            label="γ (settore)",
            width=110,
            hint_text="0.5",
        )
        self._txt_opt_delta = ft.TextField(
            label="δ (μ attesa)",
            width=110,
            hint_text="0.2",
        )
        self._txt_opt_mv_risk_aversion = ft.TextField(
            label="risk_aversion (MV)",
            width=160,
            hint_text="1.0",
        )

        self._opt_adv_panel = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Parametri avanzati ottimizzazione (Branch & Bound)",
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Row(
                        controls=[
                            self._txt_opt_rating_min,
                            self._txt_opt_max_unrated,
                            self._txt_opt_rho_pair_max,
                            self._txt_opt_max_sector,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Row(
                        controls=[
                            self._txt_opt_alpha,
                            self._txt_opt_beta,
                            self._txt_opt_gamma,
                            self._txt_opt_delta,
                            self._txt_opt_mv_risk_aversion,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                ],
                spacing=10,
            ),
            padding=10,
            border_radius=8,
            bgcolor=ft.colors.with_opacity(0.03, ft.colors.ON_SURFACE),
            visible=False,
        )

        bb_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Selezione combinatoria con vincoli (Branch & Bound)",
                        weight=ft.FontWeight.BOLD,
                    ),
                    row_build_u,
                    self._universe_adv_panel,
                    row_optimize,
                    self._opt_adv_panel,
                ],
                spacing=15,
            ),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.05, ft.colors.ON_SURFACE),
        )

        # risultati ottimizzazione
        self.txt_result_opt = ft.ListView(
            expand=False,
            spacing=5,
            padding=10,
            auto_scroll=True,
        )
        opt_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Risultati – Ottimizzazione portafoglio", weight=ft.FontWeight.BOLD),
                    self.txt_result_opt,
                ],
                spacing=10,
            ),
            padding=10,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE),
        )

        # SEZIONE: PORTAFOGLIO DIJKSTRA
        self._txt_source = ft.TextField(
            label="Ticker sorgente",
            width=220,
            disabled=True,
        )
        self._chk_dij_use_reduced = ft.Checkbox(
            label="Usa universo ridotto U'",
            value=True,
            disabled=True,
            tooltip="Se spuntato, Dijkstra lavora solo su U'; altrimenti sull'universo completo.",
        )
        self._btn_dijkstra = ft.ElevatedButton(
            text="3. Costruisci portafoglio min-correlazione (Dijkstra)",
            disabled=True,
            on_click=lambda e: (
                self._controller.handle_build_dijkstra(e)
                if self._controller is not None
                else None
            ),
        )
        self._btn_dijkstra_adv = ft.TextButton(
            text="Parametri avanzati",
            disabled=True,
            on_click=lambda e: (
                self._controller.handle_advanced_dijkstra_params(e)
                if self._controller is not None
                else None
            ),
        )

        row_dij = ft.Row(
            controls=[
                self._btn_dijkstra,
                self._txt_source,
                self._chk_dij_use_reduced,
                self._btn_dijkstra_adv,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        # Campi pannello avanzato Dijkstra
        self._txt_dij_tau = ft.TextField(
            label="τ soglia |ρ_ij|",
            width=140,
            hint_text="es. 0.30",
        )
        self._txt_dij_k = ft.TextField(
            label="k (k-NN)",
            width=110,
            hint_text="es. 10",
        )
        self._txt_dij_rho_pair_max = ft.TextField(
            label="rho_pair_max",
            width=140,
            hint_text="es. 0.80",
        )
        self._txt_dij_rating_min = ft.TextField(
            label="rating_min",
            width=130,
            hint_text="es. 13.0",
        )
        self._chk_dij_require_rating = ft.Checkbox(
            label="Richiedi rating (escluso source)",
            value=False,
        )
        self._chk_dij_signed = ft.Checkbox(
            label="Usa distanza signed (1-ρ)/2",
            value=False,
        )

        self._dij_adv_panel = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Parametri avanzati Dijkstra",
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Row(
                        controls=[
                            self._txt_dij_tau,
                            self._txt_dij_k,
                            self._txt_dij_rho_pair_max,
                            self._txt_dij_rating_min,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Row(
                        controls=[
                            self._chk_dij_require_rating,
                            self._chk_dij_signed,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                ],
                spacing=10,
            ),
            padding=10,
            border_radius=8,
            bgcolor=ft.colors.with_opacity(0.03, ft.colors.ON_SURFACE),
            visible=False,
        )

        dij_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Portafoglio di minima correlazione via Dijkstra",
                        weight=ft.FontWeight.BOLD,
                    ),
                    row_dij,
                    self._dij_adv_panel,
                ],
                spacing=15,
            ),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.05, ft.colors.ON_SURFACE),
        )

        # Risultati Dijkstra
        self.txt_result_dij = ft.ListView(
            expand=False,
            spacing=5,
            padding=10,
            auto_scroll=True,
        )
        dij_res_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Risultati – Dijkstra", weight=ft.FontWeight.BOLD),
                    self.txt_result_dij,
                ],
                spacing=10,
            ),
            padding=10,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE),
        )

        # SEZIONE MONTE CARLO
        self._mc_base_date = date.today()

        self._btn_mc_sim = ft.ElevatedButton(
            text="4. MonteCarlo simulation",
            disabled=True,  # da abilitare dal controller
            on_click=lambda e: (
                self._controller.handle_montecarlo(e)
                if self._controller is not None
                else None
            ),
        )
        self._txt_mc_n_paths = ft.TextField(
            label="Numero traiettorie MC",
            width=180,
            hint_text="es. 100",
            text_align=ft.TextAlign.RIGHT,
        )
        self._txt_mc_start = ft.TextField(
            label="Data inizio (YYYY-MM-DD)",
            width=180,
            read_only=True,
        )
        self._txt_mc_end = ft.TextField(
            label="Data fine (YYYY-MM-DD)",
            width=180,
            read_only=True,
        )

        row_mc = ft.Row(
            controls=[
                self._btn_mc_sim,
                self._txt_mc_n_paths,
                self._txt_mc_start,
                self._txt_mc_end,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.txt_result_mc = ft.ListView(
            expand=False,
            spacing=5,
            padding=10,
            auto_scroll=True,
        )

        mc_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("MonteCarlo simulation", weight=ft.FontWeight.BOLD),
                    row_mc,
                    ft.Text("Risultati – Monte Carlo", weight=ft.FontWeight.BOLD),
                    self.txt_result_mc,
                ],
                spacing=10,
            ),
            padding=10,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE),
        )

        # LAYOUT COMPLESSIVO
        self._page.controls.clear()
        self._page.add(
            header_row,
            ft.Divider(),
            data_card,
            ft.Divider(),
            bb_card,
            opt_card,
            ft.Divider(),
            dij_card,
            dij_res_card,
            ft.Divider(),
            mc_card,
        )

        # inizializza etichette slider e date MonteCarlo
        self._on_years_change()
        self._on_risk_change()
        self._on_unrated_change()

        self._page.update()

    # GESTORI EVENTI
    def _on_theme_change(self, e: ft.ControlEvent) -> None:
        if self._page.theme_mode == ft.ThemeMode.DARK:
            self._page.theme_mode = ft.ThemeMode.LIGHT
            if self._theme_switch is not None:
                self._theme_switch.label = "Light theme"
        else:
            self._page.theme_mode = ft.ThemeMode.DARK
            if self._theme_switch is not None:
                self._theme_switch.label = "Dark theme"
        self._page.update()

    def _on_years_change(self, e=None) -> None:
        if self._sld_years is None or self._lbl_years is None:
            return
        idx = int(round(self._sld_years.value))
        idx = max(0, min(3, idx))
        self._sld_years.value = idx

        years_map = [5, 10, 20, 30]
        years = years_map[idx]
        self._sld_years.label = str(years)
        self._lbl_years.value = f"Anni invest.: {years}"

        # aggiorna date MonteCarlo: start = oggi fissato, end = oggi + anni
        if self._mc_base_date is None:
            self._mc_base_date = date.today()
        start = self._mc_base_date
        end = start + timedelta(days=365 * years)

        if self._txt_mc_start is not None:
            self._txt_mc_start.value = start.isoformat()
        if self._txt_mc_end is not None:
            self._txt_mc_end.value = end.isoformat()

        self._page.update()

    def _on_risk_change(self, e=None) -> None:
        if self._sld_risk is None or self._lbl_risk is None:
            return
        val = int(round(self._sld_risk.value))
        val = max(1, min(4, val))
        self._sld_risk.value = val
        labels = {
            1: "1 (basso)",
            2: "2 (medio)",
            3: "3 (alto)",
            4: "4 (personalizzato)",
        }
        txt = labels.get(val, f"{val}")
        self._sld_risk.label = str(val)
        self._lbl_risk.value = f"Rischio: {txt}"
        self._page.update()

    def _on_unrated_change(self, e=None) -> None:
        if self._sld_max_unrated is None or self._lbl_max_unrated is None:
            return
        idx = int(round(self._sld_max_unrated.value))
        idx = max(0, min(3, idx))
        self._sld_max_unrated.value = idx
        perc_map = [0, 25, 50, 100]
        p = perc_map[idx]
        self._sld_max_unrated.label = f"{p}%"
        self._lbl_max_unrated.value = f"Max unrated: {p}%"
        self._page.update()

    # ACCESSORI PER IL CONTROLLER
    @property
    def controller(self):
        return self._controller

    def set_controller(self, controller):
        self._controller = controller

    def enable_build_universe(self, enabled: bool) -> None:
        if self._btn_build_u is not None:
            self._btn_build_u.disabled = not enabled
        if self._btn_build_u_adv is not None:
            self._btn_build_u_adv.disabled = not enabled
        self._page.update()

    def enable_optimize_portfolio(self, enabled: bool) -> None:
        if self._btn_optimize is not None:
            self._btn_optimize.disabled = not enabled
        if self._btn_optimize_adv is not None:
            self._btn_optimize_adv.disabled = not enabled
        self._page.update()

    def enable_dijkstra(self, enabled: bool) -> None:
        if self._btn_dijkstra is not None:
            self._btn_dijkstra.disabled = not enabled
        if self._btn_dijkstra_adv is not None:
            self._btn_dijkstra_adv.disabled = not enabled
        if self._txt_source is not None:
            self._txt_source.disabled = not enabled
        if self._chk_dij_use_reduced is not None:
            self._chk_dij_use_reduced.disabled = not enabled
        self._page.update()

    def enable_montecarlo(self, enabled: bool) -> None:
        if self._btn_mc_sim is not None:
            self._btn_mc_sim.disabled = not enabled
        self._page.update()

    def toggle_universe_advanced_panel(self) -> None:
        if self._universe_adv_panel is None:
            return
        self._universe_adv_panel.visible = not self._universe_adv_panel.visible
        self._page.update()

    def toggle_optimize_advanced_panel(self) -> None:
        if self._opt_adv_panel is None:
            return
        self._opt_adv_panel.visible = not self._opt_adv_panel.visible
        self._page.update()

    def toggle_dijkstra_advanced_panel(self) -> None:
        if self._dij_adv_panel is None:
            return
        self._dij_adv_panel.visible = not self._dij_adv_panel.visible
        self._page.update()

    # getter rapidi per il controller
    @property
    def txt_capital(self) -> ft.TextField:
        return self._txt_capital

    @property
    def txt_k(self) -> ft.TextField:
        return self._txt_k

    @property
    def sld_years(self) -> ft.Slider:
        return self._sld_years

    @property
    def sld_risk(self) -> ft.Slider:
        return self._sld_risk

    @property
    def sld_max_unrated(self) -> ft.Slider:
        return self._sld_max_unrated

    @property
    def txt_source(self) -> ft.TextField:
        return self._txt_source

    @property
    def chk_dij_use_reduced(self) -> ft.Checkbox:
        return self._chk_dij_use_reduced

    @property
    def txt_tau(self) -> ft.TextField:
        return self._txt_tau

    @property
    def txt_k_knn(self) -> ft.TextField:
        return self._txt_k_knn

    @property
    def txt_max_size(self) -> ft.TextField:
        return self._txt_max_size

    @property
    def txt_mc_start(self) -> ft.TextField:
        return self._txt_mc_start

    @property
    def txt_mc_end(self) -> ft.TextField:
        return self._txt_mc_end

    @property
    def txt_mc_n_paths(self) -> ft.TextField:
        return self._txt_mc_n_paths

    # getter per parametri avanzati B&B
    @property
    def txt_opt_rating_min(self) -> ft.TextField:
        return self._txt_opt_rating_min

    @property
    def txt_opt_max_unrated(self) -> ft.TextField:
        return self._txt_opt_max_unrated

    @property
    def txt_opt_rho_pair_max(self) -> ft.TextField:
        return self._txt_opt_rho_pair_max

    @property
    def txt_opt_max_sector(self) -> ft.TextField:
        return self._txt_opt_max_sector

    @property
    def txt_opt_alpha(self) -> ft.TextField:
        return self._txt_opt_alpha

    @property
    def txt_opt_beta(self) -> ft.TextField:
        return self._txt_opt_beta

    @property
    def txt_opt_gamma(self) -> ft.TextField:
        return self._txt_opt_gamma

    @property
    def txt_opt_delta(self) -> ft.TextField:
        return self._txt_opt_delta

    @property
    def txt_opt_mv_risk_aversion(self) -> ft.TextField:
        return self._txt_opt_mv_risk_aversion

    # getter per parametri avanzati Dijkstra
    @property
    def txt_dij_tau(self) -> ft.TextField:
        return self._txt_dij_tau

    @property
    def txt_dij_k(self) -> ft.TextField:
        return self._txt_dij_k

    @property
    def txt_dij_rho_pair_max(self) -> ft.TextField:
        return self._txt_dij_rho_pair_max

    @property
    def txt_dij_rating_min(self) -> ft.TextField:
        return self._txt_dij_rating_min

    @property
    def chk_dij_require_rating(self) -> ft.Checkbox:
        return self._chk_dij_require_rating

    @property
    def chk_dij_signed(self) -> ft.Checkbox:
        return self._chk_dij_signed

    def update_page(self) -> None:
        self._page.update()
