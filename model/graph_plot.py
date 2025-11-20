from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from model.modello import Model
from model.selector import PortfolioSelector


def main() -> None:
    # cartelle di output
    output_dir_mc = Path("mc_plots")
    output_dir_mc.mkdir(parents=True, exist_ok=True)

    output_dir_conv = Path("convergence_plots_K3_K6")
    output_dir_conv.mkdir(parents=True, exist_ok=True)

    # inizializzazione model
    print("[INIT] Creo il Model e stimo il rischio iniziale...")
    model = Model()
    print("[INIT] Model creato.\n")

    # profili di rischio
    risk_profiles: Dict[int, Dict[str, Any]] = {
        1: {
            "name": "prudente",
            "risk_profile": {
                "tau": 0.35,
                "k": 8,
                "max_size": 50,
                "max_unrated_share": 0.0,
                "min_rating_score": 13.5,
                "target_rated_share": 0.9,
                "rating_min": 13.5,
                "rho_pair_max": 0.7,
                "weights_mode": "mv",
                "mv_risk_aversion": 3.0,
            },
            "selector_params": {
                "K": 2,  # sovrascritto nei test
                "max_unrated_share": 0.0,
                "rating_min": 13.5,
                "max_share_per_sector": 0.4,
            },
        },
        2: {
            "name": "bilanciato",
            "risk_profile": {
                "tau": 0.30,
                "k": 10,
                "max_size": 60,
                "max_unrated_share": 0.25,
                "min_rating_score": 13.0,
                "target_rated_share": 0.7,
                "rating_min": 13.0,
                "rho_pair_max": 0.8,
                "weights_mode": "mv",
                "mv_risk_aversion": 1.5,
            },
            "selector_params": {
                "K": 2,
                "max_unrated_share": 0.25,
                "rating_min": 13.0,
                "max_share_per_sector": 0.5,
            },
        },
        3: {
            "name": "dinamico",
            "risk_profile": {
                "tau": 0.25,
                "k": 12,
                "max_size": 70,
                "max_unrated_share": 0.40,
                "min_rating_score": 12.0,
                "target_rated_share": 0.6,
                "rating_min": 12.0,
                "rho_pair_max": 0.9,
                "weights_mode": "mv",
                "mv_risk_aversion": 0.8,
            },
            "selector_params": {
                "K": 2,
                "max_unrated_share": 0.40,
                "rating_min": 12.0,
                "max_share_per_sector": 0.6,
            },
        },
    }

    # parametri globali monte carlo
    years_list = [5, 10, 20, 30]
    n_paths = 500
    n_show_paths = 20
    K_list_mc = [3, 4, 5, 6]

    # sezione monte carlo
    print("=== SEZIONE MONTE CARLO (K=3,4,5,6) ===")

    for level, cfg in risk_profiles.items():
        name = cfg["name"]
        rp = cfg["risk_profile"]
        sel_cfg = cfg["selector_params"]

        print(f"\n[MC][R{level}] Profilo '{name}'")
        print(f"[MC][R{level}] set_risk_profile: {rp}")
        model.set_risk_profile(rp)

        # costruzione universo ridotto U'
        try:
            print(f"[MC][R{level}] build_reduced_universe() in corso...")
            reduced = model.build_reduced_universe(
                tau=rp["tau"],
                k=rp["k"],
                max_size=rp["max_size"],
                max_unrated_share=rp["max_unrated_share"],
                min_rating_score=rp["min_rating_score"],
                target_rated_share=rp["target_rated_share"],
            )
        except Exception as e:
            print(f"[MC][R{level}] Errore build_reduced_universe: {e}")
            continue

        print(f"[MC][R{level}] Dimensione U' = {len(reduced)}")

        # loop su K per monte carlo
        for K in K_list_mc:
            print(f"\n[MC][R{level}, K={K}] Avvio ottimizzazione B&B")

            if len(reduced) < K:
                print(f"[MC][R{level}, K={K}] U' troppo piccolo per K={K}, salto.")
                continue

            params = PortfolioSelector.build_default_params()
            params.update(sel_cfg)

            params["K"] = K
            params["rating_min"] = rp["rating_min"]
            params["max_unrated_share"] = rp["max_unrated_share"]
            params["rho_pair_max"] = rp["rho_pair_max"]
            params["weights_mode"] = rp.get("weights_mode", params.get("weights_mode", "mv"))
            params["mv_risk_aversion"] = rp.get(
                "mv_risk_aversion", params.get("mv_risk_aversion", 1.0)
            )

            print(
                f"[MC][R{level}, K={K}] Parametri B&B: "
                f"K={params['K']}, rating_min={params['rating_min']}, "
                f"max_unrated_share={params['max_unrated_share']}, "
                f"rho_pair_max={params['rho_pair_max']}"
            )

            # B&B
            try:
                print(f"[MC][R{level}, K={K}] optimize_portfolio() in corso...")
                t0 = time.perf_counter()
                best_subset, weights, best_score = model.optimize_portfolio(
                    params=params,
                    use_reduced_universe=True,
                )
                t1 = time.perf_counter()
            except Exception as e:
                print(f"[MC][R{level}, K={K}] Errore optimize_portfolio: {e}")
                continue

            dt_bb = t1 - t0
            print(f"[MC][R{level}, K={K}] B&B terminato in {dt_bb:.3f} s")

            if not best_subset:
                print(f"[MC][R{level}, K={K}] Nessun portafoglio B&B trovato, salto MC.")
                continue

            print(
                f"[MC][R{level}, K={K}] Portafoglio B&B (len={len(best_subset)}): "
                f"{best_subset}"
            )
            print(f"[MC][R{level}, K={K}] Score combinatorio: {best_score:.4f}")

            # Dijkstra (solo per costruire il portafoglio min-correlazione)
            source = max(weights, key=weights.get)
            print(f"[MC][R{level}, K={K}] Ticker source per Dijkstra: {source}")

            try:
                print(f"[MC][R{level}, K={K}] build_dijkstra_portfolio() in corso...")
                t0 = time.perf_counter()
                port_dij, w_dij = model.build_dijkstra_portfolio(
                    source=source,
                    K=K,
                    use_reduced_universe=True,
                    require_rating=False,
                    min_rating_score=rp["rating_min"],
                    rho_pair_max=rp["rho_pair_max"],
                    graph_signed=False,
                    graph_tau=rp["tau"],
                    graph_k=rp["k"],
                )
                t1 = time.perf_counter()
            except Exception as e:
                print(f"[MC][R{level}, K={K}] Errore build_dijkstra_portfolio: {e}")
                continue

            dt_dij = t1 - t0
            print(f"[MC][R{level}, K={K}] Dijkstra terminato in {dt_dij:.3f} s")

            if not port_dij:
                print(f"[MC][R{level}, K={K}] Portafoglio Dijkstra vuoto, salto MC.")
                continue

            print(
                f"[MC][R{level}, K={K}] Portafoglio Dijkstra (len={len(port_dij)}): "
                f"{port_dij}"
            )

            # monte carlo per ogni orizzonte
            for years in years_list:
                n_days = years * 252
                seed_base = level * 10_000 + K * 100 + years * 10
                seed_bb = seed_base
                seed_dij = seed_base + 1

                print(
                    f"[MC][R{level}, K={K}, T={years} anni] "
                    f"Monte Carlo con {n_paths} traiettorie (~{n_days} giorni)"
                )

                try:
                    results = model.simulate_mc_for_last_portfolios(
                        n_paths=n_paths,
                        n_days=n_days,
                        seed_bb=seed_bb,
                        seed_dij=seed_dij,
                    )
                except Exception as e:
                    print(
                        f"[MC][R{level}, K={K}, T={years}] "
                        f"Errore simulate_mc_for_last_portfolios: {e}"
                    )
                    continue

                data_bb = results.get("bb")
                data_dij = results.get("dij")
                if not data_bb or not data_dij:
                    print(
                        f"[MC][R{level}, K={K}, T={years}] "
                        f"Risultati MC incompleti, salto."
                    )
                    continue

                paths_bb = data_bb["paths"]
                mean_bb = data_bb["mean"]
                paths_dij = data_dij["paths"]
                mean_dij = data_dij["mean"]

                if (
                    paths_bb is None
                    or paths_bb.size == 0
                    or paths_dij is None
                    or paths_dij.size == 0
                ):
                    print(
                        f"[MC][R{level}, K={K}, T={years}] "
                        f"paths vuote, salto."
                    )
                    continue

                T_len = min(paths_bb.shape[1], paths_dij.shape[1])
                x = np.arange(T_len) / 252.0

                fig, ax = plt.subplots(figsize=(8, 4))

                n_show = min(n_show_paths, paths_bb.shape[0], paths_dij.shape[0])

                for i in range(n_show):
                    ax.plot(
                        x,
                        paths_bb[i, :T_len],
                        color="green",
                        alpha=0.15,
                        linewidth=0.7,
                    )
                    ax.plot(
                        x,
                        paths_dij[i, :T_len],
                        color="blue",
                        alpha=0.15,
                        linewidth=0.7,
                    )

                ax.plot(
                    x,
                    mean_bb[:T_len],
                    color="green",
                    linewidth=2.0,
                    label="B&B – media",
                )
                ax.plot(
                    x,
                    mean_dij[:T_len],
                    color="blue",
                    linewidth=2.0,
                    label="Dijkstra – media",
                )

                ax.axhline(
                    1.0,
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                )

                ax.set_title(
                    f"Profilo R{level} ({name}) – K={K}, orizzonte {years} anni\n"
                    f"{n_paths} traiettorie"
                )
                ax.set_xlabel("Tempo (anni)")
                ax.set_ylabel("Valore portafoglio (base 1.0)")
                ax.legend()
                ax.grid(True, alpha=0.3)

                fig.tight_layout()

                filename = f"mc_R{level}_{name}_K{K}_{years}y.png"
                outfile = output_dir_mc / filename
                fig.savefig(outfile, dpi=130, bbox_inches="tight")
                plt.close(fig)

                print(
                    f"[MC][R{level}, K={K}, T={years}] "
                    f"Salvata immagine: {outfile}"
                )

    # sezione benchmark convergenza
    print("\n=== SEZIONE BENCHMARK CONVERGENZA (K=3,4,5,6) ===")

    risk_profiles_conv: Dict[int, Dict[str, Any]] = risk_profiles
    K_list_conv = [3, 4, 5, 6]

    for level, cfg in risk_profiles_conv.items():
        name = cfg["name"]
        rp = cfg["risk_profile"]
        sel_cfg = cfg["selector_params"]

        print(f"\n[CONV][R{level}] Profilo '{name}'")

        times_bb: list[float] = []
        Ks_ok: list[int] = []

        for K in K_list_conv:
            print(f"[CONV][R{level}] Avvio test K={K}")

            print(f"[CONV][R{level}, K={K}] set_risk_profile: {rp}")
            model.set_risk_profile(rp)

            try:
                print(f"[CONV][R{level}, K={K}] build_reduced_universe() in corso...")
                reduced = model.build_reduced_universe(
                    tau=rp["tau"],
                    k=rp["k"],
                    max_size=rp["max_size"],
                    max_unrated_share=rp["max_unrated_share"],
                    min_rating_score=rp["min_rating_score"],
                    target_rated_share=rp["target_rated_share"],
                )
            except Exception as e:
                print(f"[CONV][R{level}, K={K}] Errore build_reduced_universe: {e}")
                continue

            print(f"[CONV][R{level}, K={K}] |U'| = {len(reduced)}")
            if len(reduced) < K:
                print(f"[CONV][R{level}, K={K}] U' troppo piccolo, salto questo K.")
                continue

            params = PortfolioSelector.build_default_params()
            params.update(sel_cfg)
            params["K"] = K
            params["rating_min"] = rp["rating_min"]
            params["max_unrated_share"] = rp["max_unrated_share"]
            params["rho_pair_max"] = rp["rho_pair_max"]
            params["weights_mode"] = rp.get("weights_mode", params.get("weights_mode", "mv"))
            params["mv_risk_aversion"] = rp.get(
                "mv_risk_aversion", params.get("mv_risk_aversion", 1.0)
            )

            print(
                f"[CONV][R{level}, K={K}] Parametri B&B: "
                f"rating_min={params['rating_min']}, "
                f"max_unrated_share={params['max_unrated_share']}, "
                f"rho_pair_max={params['rho_pair_max']}"
            )

            t0 = time.perf_counter()
            try:
                best_subset, weights, best_score = model.optimize_portfolio(
                    params=params,
                    use_reduced_universe=True,
                )
            except Exception as e:
                t1 = time.perf_counter()
                print(
                    f"[CONV][R{level}, K={K}] Errore optimize_portfolio "
                    f"dopo {t1 - t0:.3f} s: {e}"
                )
                continue
            t1 = time.perf_counter()
            dt_bb = t1 - t0
            print(
                f"[CONV][R{level}, K={K}] B&B terminato in {dt_bb:.3f} s "
                f"(score={best_score if best_score is not None else 'NA'})"
            )

            if not best_subset:
                print(f"[CONV][R{level}, K={K}] Nessun portafoglio B&B, salto Dijkstra.")
                continue

            source = max(weights, key=weights.get)
            print(f"[CONV][R{level}, K={K}] source Dijkstra = {source}")

            t0 = time.perf_counter()
            try:
                port_dij, w_dij = model.build_dijkstra_portfolio(
                    source=source,
                    K=K,
                    use_reduced_universe=True,
                    require_rating=False,
                    min_rating_score=rp["rating_min"],
                    rho_pair_max=rp["rho_pair_max"],
                    graph_signed=False,
                    graph_tau=rp["tau"],
                    graph_k=rp["k"],
                )
            except Exception as e:
                t1 = time.perf_counter()
                print(
                    f"[CONV][R{level}, K={K}] Errore build_dijkstra_portfolio "
                    f"dopo {t1 - t0:.3f} s: {e}"
                )
                continue
            t1 = time.perf_counter()
            dt_dij = t1 - t0
            print(
                f"[CONV][R{level}, K={K}] Dijkstra terminato in {dt_dij:.3f} s "
                f"(len_port={len(port_dij) if port_dij else 0})"
            )

            if not port_dij:
                print(
                    f"[CONV][R{level}, K={K}] Portafoglio Dijkstra vuoto, non salvo tempi."
                )
                continue

            Ks_ok.append(K)
            times_bb.append(dt_bb)

        if Ks_ok:
            xs = np.array(Ks_ok, dtype=float)
            times_bb_arr = np.array(times_bb, dtype=float)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, times_bb_arr, marker="o", label="B&B")
            ax.set_xlabel("K (numero titoli)")
            ax.set_ylabel("Tempo (s)")
            ax.set_title(f"Tempo convergenza B&B – R{level} ({name})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            outfile_bb = output_dir_conv / f"convergence_BB_R{level}_{name}_K3_K6.png"
            fig.savefig(outfile_bb, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"[CONV][R{level}] Salvata immagine B&B: {outfile_bb}")
        else:
            print(f"[CONV][R{level}] Nessun K valido, nessun plot generato.")

    print("\n[TASK] Script completato.")


if __name__ == "__main__":
    main()
