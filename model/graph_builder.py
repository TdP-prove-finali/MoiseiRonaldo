from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


class GraphBuilder:
    """
    Funzioni per:
    - Costruire matrici di distanza da una matrice di correlazione,
    - Applicare filtri (soglia, k-NN),
    - Costruire il grafo NetworkX e la matrice di adiacenza.
    """

    @staticmethod
    def build_distance_matrix(rho: pd.DataFrame, signed: bool = False) -> pd.DataFrame:
        """
        COSTRUISCE UNA MATRICE DELLE DISTANZE (d_ij) a partire dalla matrice di correlazione (rho).

        - se signed=False: d_ij = 1 - |rho_ij|
        - se signed=True:  d_ij = (1 - rho_ij) / 2
        """
        if rho is None or rho.empty:
            raise ValueError("rho è vuota.")

        if signed:
            # distanza signed: (1 - rho) / 2
            d_values = (1.0 - rho.values) / 2.0
        else:
            # distanza non signed: 1 - |rho|
            d_values = 1.0 - np.abs(rho.values)

        d = pd.DataFrame(d_values, index=rho.index, columns=rho.columns)
        np.fill_diagonal(d.values, 0.0)
        return d

    @staticmethod
    def threshold_filter(rho: pd.DataFrame, tau: float) -> pd.DataFrame:
        """
        Applica un filtro a soglia alla matrice di correlazione:
        - mantiene i valori con |rho_ij| >= tau
        - pone a 0 gli altri
        """
        if rho is None or rho.empty:
            raise ValueError("rho è vuota.")
        if not (0.0 <= tau <= 1.0):
            raise ValueError("tau deve essere in [0,1].")

        rho_f = rho.copy()
        mask = rho_f.abs() < tau
        rho_f[mask] = 0.0
        np.fill_diagonal(rho_f.values, 0.0)
        return rho_f

    @staticmethod
    def knn_filter(
            distance: pd.DataFrame,
            k: int,
            symmetric: bool = True
    ) -> pd.DataFrame:
        """
        APPLICA UN FILTRO K-NN sulla matrice delle distanze.

        Restituisce una matrice D_knn dove:
        - per ciascuna riga, solo le k distanze più piccole restano finite (le altre sono +inf).
        - la matrice è resa simmetrica se symmetric=True.
        """
        if distance is None or distance.empty:
            raise ValueError("distance è vuota.")
        if k <= 0:
            raise ValueError("k deve essere > 0.")

        n = distance.shape[0]
        if k >= n:
            # se k è troppo grande, restituisci l'originale con diagonale a 0
            d_knn = distance.copy()
            np.fill_diagonal(d_knn.values, 0.0)
            return d_knn

        inf = np.inf
        # inizializza la matrice risultante a infinito
        d_knn = pd.DataFrame(
            inf,
            index=distance.index,
            columns=distance.columns
        )

        for i, row_label in enumerate(distance.index):
            row = distance.loc[row_label].copy()
            # escludi la diagonale prima di trovare i k minimi
            row_no_diag = row.drop(labels=row_label)
            k_smallest = row_no_diag.nsmallest(k)

            # imposta le k distanze più piccole
            for col_label, val in k_smallest.items():
                d_knn.at[row_label, col_label] = val

        if symmetric:
            # rende la matrice simmetrica usando il minimo tra d_ij e d_ji
            vals = d_knn.values
            for i in range(n):
                for j in range(i + 1, n):
                    vij = vals[i, j]
                    vji = vals[j, i]
                    v = min(vij, vji)
                    vals[i, j] = v
                    vals[j, i] = v

            d_knn = pd.DataFrame(vals, index=distance.index, columns=distance.columns)

        np.fill_diagonal(d_knn.values, 0.0)
        return d_knn

    @staticmethod
    def build_filtered_graph(
            rho: pd.DataFrame,
            tau: float | None = None,
            k: int | None = None,
            signed: bool = False,
    ) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
        """
        COSTRUISCE IL GRAFO FILTRATO E LE MATRICI DI ADIACENZA E DISTANZA.

        Il grafo NetworkX ha attributi:
            weight = |rho_ij| (intensità di correlazione)
            corr = rho_ij (correlazione firmata)
            distance = distanza usata per Dijkstra (1-|rho| o signed)

        Restituisce: (G, adj, dist_knn).
        """
        if rho is None or rho.empty:
            raise ValueError("rho è vuota.")

        # 1) matrice distanze base
        dist = GraphBuilder.build_distance_matrix(rho, signed=signed)

        # 2) modulo della correlazione, con diagonale a 0
        abs_rho = rho.abs().copy()
        np.fill_diagonal(abs_rho.values, 0.0)

        # 3) filtro a soglia sulla distanza (dove |rho| < tau → distanza = +inf)
        if tau is not None:
            if not (0.0 <= tau <= 1.0):
                raise ValueError("tau deve essere in [0,1].")
            mask_below = abs_rho.values < tau
            dist.values[mask_below] = np.inf

        # 4) filtro k-NN (se richiesto)
        if k is not None:
            dist_knn = GraphBuilder.knn_filter(dist, k=k, symmetric=True)
        else:
            dist_knn = dist

        # 5) matrice di adiacenza: |rho_ij| se dist_ij finita, 0 altrove
        adj = abs_rho.copy()
        # gli archi sono solo dove la distanza è finita (non infinita per i filtri)
        mask_no_edge = ~np.isfinite(dist_knn.values)
        adj.values[mask_no_edge] = 0.0
        np.fill_diagonal(adj.values, 0.0)

        # 6) costruzione grafo NetworkX
        G = nx.Graph()
        for t in adj.index:
            G.add_node(t)

        cols = list(adj.columns)
        for i, ti in enumerate(adj.index):
            row_vals = adj.iloc[i].values
            for j in range(i + 1, len(cols)):
                w = row_vals[j]
                if w <= 0.0:
                    continue
                tj = cols[j]
                corr_val = rho.loc[ti, tj]
                dist_val = float(dist_knn.loc[ti, tj])

                # l'adiacenza positiva implica che la distanza è finita
                G.add_edge(
                    ti,
                    tj,
                    weight=float(w),  # intensità di correlazione |rho|
                    corr=float(corr_val),  # correlazione firmata
                    distance=dist_val,  # distanza usata per Dijkstra
                )

        return G, adj, dist_knn


if __name__ == "__main__":
    # Mini test indipendente con una matrice fittizia
    import traceback

    data = {
        "A": [1.0, 0.5, -0.2],
        "B": [0.5, 1.0, 0.3],
        "C": [-0.2, 0.3, 1.0],
    }
    rho_test = pd.DataFrame(data, index=["A", "B", "C"])

    print("=== TEST GRAPH_BUILDER (OOP) ===")
    try:
        # Costruisce il grafo con tau=0.25 e k=1
        G, adj, dist_knn = GraphBuilder.build_filtered_graph(
            rho_test,
            tau=0.25,
            k=1,
            signed=False,
        )
        print("Adj:\n", adj)
        print("Distanza (k-NN):\n", dist_knn)
        print("Nodi grafo:", G.nodes())
        print("Archi grafo (con attributi):")
        for u, v, attr in G.edges(data=True):
            print(f"{u}-{v}: {attr}")

    except Exception:
        print("TEST GRAPH_BUILDER FALLITO:")
        traceback.print_exc()
    else:
        print("TEST GRAPH_BUILDER OK.")