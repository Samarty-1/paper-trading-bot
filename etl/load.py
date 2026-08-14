"""
Load stage: idempotent upsert of transformed bars into a local DuckDB
warehouse, plus read_from_warehouse() so consumers (paper_trade.py) have a
single source of truth instead of each re-fetching from yfinance.
"""

import os

import duckdb
import pandas as pd

from etl.config import BARS_TABLE, DUCKDB_PATH

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {BARS_TABLE} (
    ticker VARCHAR,
    date VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    fast_ema DOUBLE,
    slow_ema DOUBLE,
    rsi DOUBLE,
    atr DOUBLE,
    PRIMARY KEY (ticker, date)
)
"""


def _connect():
    os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
    con = duckdb.connect(DUCKDB_PATH)
    con.execute(CREATE_TABLE_SQL)
    return con


def load(df):
    if df.empty:
        return 0
    # duckdb 1.1.3 doesn't recognize pandas 3.x's default StringDtype ("str")
    # columns via the zero-copy registration path; cast to legacy object dtype.
    df = df.astype({c: object for c in df.select_dtypes(include="string").columns})
    con = _connect()
    try:
        ticker = str(df["ticker"].iloc[0])
        min_date, max_date = str(df["date"].min()), str(df["date"].max())
        con.execute(
            f"DELETE FROM {BARS_TABLE} WHERE ticker = ? AND date BETWEEN ? AND ?",
            [ticker, min_date, max_date],
        )
        con.register("staging", df)
        con.execute(f"INSERT INTO {BARS_TABLE} SELECT * FROM staging")
        return len(df)
    finally:
        con.close()


def read_from_warehouse(ticker):
    con = _connect()
    try:
        df = con.execute(
            f"SELECT * FROM {BARS_TABLE} WHERE ticker = ? ORDER BY date", [ticker.upper()]
        ).df()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError(f"No warehouse data for {ticker}. Run etl/pipeline.py first.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })
    return df
