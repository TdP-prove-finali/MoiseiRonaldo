import flet as ft


class View(ft.UserControl):

    def __init__(self, page: ft.Page):
        super().__init__()
        # Page
        self._page = page
        self._page.title = "Portfolio Optimizer – Thesis Prototype"
        self._page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self._page.theme_mode = ft.ThemeMode.DARK
        self._page.window_height = 800
        self._page.window_width = 1300
        self._page.window_center()
        self._page.scroll = ft.ScrollMode.AUTO

        self._controller = None

        # Controls principali (inizializzati in load_interface)
        self._theme_switch: ft.Switch | None = None

        self._txt_capital: ft.TextField | None = None
        self._txt_k: ft.TextField | None = None

        self._sld_years: ft.Slider | None = None
        self._lbl_years: ft.Text | None = None

        self._sld_risk: ft.Slider | None = None
        self._lbl_risk: ft.Text | None = None

        self._sld_max_unrated: ft.Slider | None = None
        self._lbl_max_unrated: ft.Text | None = None

        self._btn_build_u: ft.ElevatedButton | None = None
        self._btn_build_u_adv: ft.TextButton | None = None

        # Pannello parametri avanzati U'
        self._universe_adv_panel: ft.Container | None = None
        self._txt_tau: ft.TextField | None = None
        self._txt_k_knn: ft.TextField | None = None
        self._txt_max_size: ft.TextField | None = None

        self._btn_optimize: ft.ElevatedButton | None = None
        self._btn_optimize_adv: ft.TextButton | None = None

        self._txt_source: ft.TextField | None = None
        self._btn_dijkstra: ft.ElevatedButton | None = None
        self._btn_dijkstra_adv: ft.TextButton | None = None

        # ListView usata dal Controller per stampare risultati
        self.txt_result: ft.ListView | None = None

    def load_interface(self) -> None:
        # ----------------- HEADER: tema + titolo -----------------
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

        # ----------------- SEZIONE INPUT DATI -----------------
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

        # inizializza etichette slider
        self._on_years_change()
        self._on_risk_change()
        self._on_unrated_change()

        # ----------------- SEZIONE: COSTRUISCI U' -----------------
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

        # ----------------- SEZIONE: OTTIMIZZA PORTAFOGLIO (B&B) -----------------
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
                ],
                spacing=15,
            ),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.05, ft.colors.ON_SURFACE),
        )

        # ----------------- SEZIONE: PORTAFOGLIO MIN-CORR (DIJKSTRA) -----------------
        self._txt_source = ft.TextField(
            label="Ticker sorgente",
            width=220,
            disabled=True,
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
            controls=[self._btn_dijkstra, self._txt_source, self._btn_dijkstra_adv],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        dij_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Portafoglio di minima correlazione via Dijkstra",
                        weight=ft.FontWeight.BOLD,
                    ),
                    row_dij,
                ],
                spacing=15,
            ),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.05, ft.colors.ON_SURFACE),
        )

        # ----------------- SEZIONE  RISULTATI -----------------
        self.txt_result = ft.ListView(
            expand=False,      # niente expand, così il contenitore prende l’altezza naturale
            spacing=5,
            padding=10,
            auto_scroll=True,
        )

        log_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Risultati", weight=ft.FontWeight.BOLD),
                    self.txt_result,
                ],
                spacing=10,
            ),
            padding=10,
            border_radius=10,
            bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE),
        )


        self._page.controls.clear()
        self._page.add(
            header_row,
            ft.Divider(),
            data_card,
            ft.Divider(),
            bb_card,
            ft.Divider(),
            dij_card,
            ft.Divider(),
            log_card,
        )
        self._page.update()


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
        self._page.update()

    def toggle_universe_advanced_panel(self) -> None:
        if self._universe_adv_panel is None:
            return
        self._universe_adv_panel.visible = not self._universe_adv_panel.visible
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
    def txt_tau(self) -> ft.TextField:
        return self._txt_tau

    @property
    def txt_k_knn(self) -> ft.TextField:
        return self._txt_k_knn

    @property
    def txt_max_size(self) -> ft.TextField:
        return self._txt_max_size

    def update_page(self) -> None:
        self._page.update()
