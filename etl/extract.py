"""
Extract stage: pulls daily OHLCV bars from yfinance and writes an immutable
raw snapshot to disk before any cleaning happens, so raw pulls stay
auditable/reproducible independent of whatever the transform stage does.
"""

import os
from datetime import date

import pandas as pd
import yfinance as yf

from etl.config import FETCH_INTERVAL, FETCH_PERIOD, RAW_DIR


def fetch(ticker):
    df = yf.download(ticker, period=FETCH_PERIOD, interval=FETCH_INTERVAL, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def snapshot_path(ticker, as_of=None):
    as_of = as_of or date.today().isoformat()
    return os.path.join(RAW_DIR, f"{ticker.upper()}_{as_of}.csv")


def extract(ticker):
    df = fetch(ticker)
    os.makedirs(RAW_DIR, exist_ok=True)
    path = snapshot_path(ticker)
    df.to_csv(path)
    return df, path
