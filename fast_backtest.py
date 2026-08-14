"""
Validates the intraday momentum breakout strategy on 5-minute bars.
CAVEAT printed clearly: yfinance only gives ~60 days of free 5-minute
history, vs. the 8 years used for the daily strategy -- this is a much
thinner, more preliminary test. Treat "good" results here with real
skepticism until it's been paper traded for a while too.
"""

import sys

import pandas as pd
import yfinance as yf
from backtesting import Backtest

from fast_strategy import MomentumBreakout

TICKERS = ["IWM", "AMD", "PLTR", "COIN"]
CASH = 10_000
COMMISSION = 0.0005  # proxy for spread + slippage on liquid large-caps


def add_vwap(df):
    df = df.copy()
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.date
    cum_vol = (df["Volume"] + 1e-9).groupby(day).cumsum()
    cum_pv = (typical * df["Volume"]).groupby(day).cumsum()
    df["VWAP"] = cum_pv / cum_vol
    return df


def load_data(ticker):
    df = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return add_vwap(df)


def run_one(ticker, **strat_kwargs):
    data = load_data(ticker)
    bt = Backtest(data, MomentumBreakout, cash=CASH, commission=COMMISSION,
                   exclusive_orders=True, margin=0.5)
    stats = bt.run(**strat_kwargs)
    n = int(stats["# Trades"])
    days = (data.index[-1] - data.index[0]).days
    return {
        "Ticker": ticker,
        "Return %": round(stats["Return [%]"], 2),
        "Buy&Hold %": round(stats["Buy & Hold Return [%]"], 2),
        "Max DD %": round(stats["Max. Drawdown [%]"], 2),
        "Sharpe": round(stats["Sharpe Ratio"], 2),
        "Win Rate %": round(stats["Win Rate [%]"], 1) if n else 0.0,
        "Profit Factor": round(stats["Profit Factor"], 2) if n else float("nan"),
        "Trades": n,
        "Days": days,
    }


def main(**strat_kwargs):
    rows = []
    for t in TICKERS:
        print(f"Backtesting {t} (5-min bars)...", file=sys.stderr)
        rows.append(run_one(t, **strat_kwargs))
    df = pd.DataFrame(rows).set_index("Ticker")
    print(f"\n=== Intraday Momentum Breakout, 5-min bars, ~{df['Days'].iloc[0]} days of free data ===")
    print("CAVEAT: this is a MUCH thinner test than the 8-year daily backtest -- treat as preliminary.")
    print(df.drop(columns="Days").to_string())
    print(
        f"\nAvg Return: {df['Return %'].mean():.2f}%  |  Avg Max DD: {df['Max DD %'].mean():.2f}%  |  "
        f"Avg Sharpe: {df['Sharpe'].mean():.2f}  |  Avg Trades: {df['Trades'].mean():.1f}"
    )
    return df


if __name__ == "__main__":
    main()
