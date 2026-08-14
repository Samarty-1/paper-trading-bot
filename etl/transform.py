"""
Transform stage: validates a raw OHLCV frame and computes the indicator set
the strategy needs (reusing indicators.py, the same functions paper_trade.py
and backtest.py already rely on), returning one tidy DataFrame ready to load.
"""

import pandas as pd

from indicators import atr, ema, rsi

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

FAST_LEN = 20
SLOW_LEN = 50
RSI_LEN = 14
ATR_LEN = 14


def validate(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("NaN values found in OHLC data")
    if (df["Volume"] < 0).any():
        raise ValueError("Negative volume values found")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not in increasing order")


def transform(ticker, df):
    validate(df)
    out = df.copy()
    out["fast_ema"] = ema(out["Close"], FAST_LEN)
    out["slow_ema"] = ema(out["Close"], SLOW_LEN)
    out["rsi"] = rsi(out["Close"], RSI_LEN)
    out["atr"] = atr(out["High"], out["Low"], out["Close"], ATR_LEN)

    out = out.reset_index()
    date_col = out.columns[0]
    out = out.rename(columns={
        date_col: "date",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out.insert(0, "ticker", ticker.upper())
    return out
