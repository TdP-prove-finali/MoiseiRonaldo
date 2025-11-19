from __future__ import annotations

from typing import Dict, Tuple
import os

import pandas as pd
import numpy as np

from model.stock import Stock


# DEFINIZIONE PATH
BASE_DIR = os.path.dirname(__file__)
DEFAULT_PRICES_PATH = os.path.join(BASE_DIR, "all_stocks_5yr.csv")
DEFAULT_RATINGS_PATH = os.path.join(BASE_DIR, "corporateCreditRatingWithFinancialRatios.csv")


class DAO:
    """
    Data Access Object: gestisce il caricamento dei dati da file CSV
    e la costruzione del dizionario di oggetti Stock.
    """

    def __init__(self,
                 prices_path: str = DEFAULT_PRICES_PATH,
                 ratings_path: str = DEFAULT_RATINGS_PATH) -> None:
        self._prices_path = prices_path
        self._ratings_path = ratings_path

        self._prices_df: pd.DataFrame | None = None
        self._ratings_df: pd.DataFrame | None = None

    # METODI PRIVATI DI CARICAMENTO

    def _load_prices(self) -> pd.DataFrame:
        """
        Legge all_stocks_5yr.csv e costruisce un DataFrame pivotato:
        index = date, columns = ticker (Name), values = close.
        """
        if self._prices_df is not None:
            return self._prices_df

        df = pd.read_csv(self._prices_path)

        # normalizza la data e pivotta il DataFrame
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        prices = df.pivot(index="date", columns="Name", values="close")
        prices = prices.sort_index()

        self._prices_df = prices
        return self._prices_df

    def _load_ratings(self) -> pd.DataFrame:
        """
        Legge corporateCreditRatingWithFinancialRatios.csv
        e aggiunge la colonna numerica rating_score.
        """
        if self._ratings_df is not None:
            return self._ratings_df

        df = pd.read_csv(self._ratings_path)

        # MAPPING RATING ALFABETICO → NUMERICO
        rating_map = {
            "AAA": 22, "AA+": 21, "AA": 20, "AA-": 19,
            "A+": 18, "A": 17, "A-": 16,
            "BBB+": 15, "BBB": 14, "BBB-": 13,
            "BB+": 12, "BB": 11, "BB-": 10,
            "B+": 9, "B": 8, "B-": 7,
            "CCC+": 6, "CCC": 5, "CCC-": 4,
            "CC": 3, "C": 2, "D": 1
        }

        df["rating_score"] = df["Rating"].map(rating_map)
        df["Rating Date"] = pd.to_datetime(df["Rating Date"])

        self._ratings_df = df
        return self._ratings_df

    # METODI PUBBLICI DI ACCESSO

    def get_prices_df(self) -> pd.DataFrame:
        """
        Restituisce il DataFrame dei prezzi (date × ticker).
        """
        return self._load_prices()

    def get_ratings_df(self) -> pd.DataFrame:
        """
        Restituisce il DataFrame dei rating con rating_score.
        """
        return self._load_ratings()

    def build_stock_dict(self) -> Dict[str, Stock]:
        """
        Restituisce un dizionario ticker → Stock per TUTTI i ticker dei prezzi.
        I rating vengono aggiunti se disponibili (ultimo rating valido).
        """
        prices = self._load_prices()
        ratings = self._load_ratings()

        tickers_prices = set(prices.columns)
        tickers_ratings = set(ratings["Ticker"].unique())

        # estrai l'ULTIMO rating per ticker (per i soli ticker che hanno un rating)
        last_ratings = (
            ratings.sort_values("Rating Date")
                   .groupby("Ticker")
                   .tail(1)
                   .set_index("Ticker")
        )

        stock_dict: Dict[str, Stock] = {}

        for ticker in sorted(tickers_prices):
            if ticker in tickers_ratings:
                # Ticker con rating disponibile
                row = last_ratings.loc[ticker]

                rating_score = float(row["rating_score"]) if pd.notna(row["rating_score"]) else None
                # Il settore non è sempre disponibile in tutte le righe del rating DF
                sector = row["Sector"] if "Sector" in row.index else None
                rating_date = row["Rating Date"]
            else:
                # Ticker senza rating: imposta a None
                rating_score = None
                sector = None
                rating_date = None

            s = Stock(
                ticker=ticker,
                sector=sector,
                rating_score=rating_score,
                rating_date=rating_date
            )

            # assegna la serie prezzi 'close' allo Stock
            price_series = prices[ticker]
            s.set_prices(price_series)

            stock_dict[ticker] = s

        return stock_dict

    def load_universe(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Stock]]:
        """
        Metodo comodo: carica tutto e restituisce
        (prices_df, ratings_df, stock_dict).
        """
        prices = self.get_prices_df()
        ratings = self.get_ratings_df()
        stock_dict = self.build_stock_dict()
        return prices, ratings, stock_dict


if __name__ == "__main__":
    # TEST RAPIDO DEL DAO
    dao = DAO()

    prices = dao.get_prices_df()
    ratings = dao.get_ratings_df()
    stocks = dao.build_stock_dict()

    print("=== TEST DAO ===")
    print(f"Shape prices_df (date x ticker): {prices.shape}")
    print(f"Shape ratings_df: {ratings.shape}")
    print(f"Numero di ticker in prices_df: {len(prices.columns)}")
    print(f"Numero di ticker unici in ratings_df: {ratings['Ticker'].nunique()}")
    print(f"Numero di Stock nel dizionario: {len(stocks)}")

    sample_tickers = list(stocks.keys())[:5]
    print(f"Primi 5 ticker: {sample_tickers}")

    try:
        if prices.shape[0] == 0:
            raise AssertionError("prices_df è vuoto!")

        if ratings.shape[0] == 0:
            raise AssertionError("ratings_df è vuoto!")

        if len(stocks) == 0:
            raise AssertionError("stock_dict è vuoto!")

        # I ticker dello stock_dict devono essere colonne di prices
        if not set(stocks.keys()).issubset(set(prices.columns)):
            raise AssertionError(
                "Ci sono ticker nello stock_dict che non compaiono in prices_df!"
            )

    except AssertionError as e:
        print("TEST DAO FALLITO:")
        print(" -", e)

    except Exception as e:
        print("ERRORE IMPREVISTO DURANTE IL TEST DAO:")
        print(" -", repr(e))

    else:
        print("TEST DAO OK.")