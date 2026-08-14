"""
Validates the strategy before trusting it with paper trading.
Runs it across several liquid tickers and prints the key reliability metrics:
return, max drawdown, win rate, Sharpe ratio, and trade count.
A strategy that only works on one ticker is probably overfit, not good.
"""

import sys

import pandas as pd
import yfinance as yf
from backtesting import Backtest

from strategy import EmaTrendRsiAtr

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT"]
PERIOD = "8y"
CASH = 10_000
COMMISSION = 0.0005  # 5 bps, roughly matches typical commission-free broker slippage


def load_data(ticker, period=None, start=None, end=None):
    if start:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, period=period or PERIOD, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df[["Open", "High", "Low", "Close", "Volume"]]


def run_one(ticker, start=None, end=None):
    data = load_data(ticker, start=start, end=end)
    bt = Backtest(data, EmaTrendRsiAtr, cash=CASH, commission=COMMISSION, exclusive_orders=True, margin=0.5)
    stats = bt.run()
    n_trades = int(stats["# Trades"])
    return {
        "Ticker": ticker,
        "CAGR %": round(stats["CAGR [%]"], 2),
        "Buy&Hold %": round(stats["Buy & Hold Return [%]"], 1),
        "Max DD %": round(stats["Max. Drawdown [%]"], 2),
        "Sharpe": round(stats["Sharpe Ratio"], 2),
        "Sortino": round(stats["Sortino Ratio"], 2),
        "Calmar": round(stats["Calmar Ratio"], 2),
        "Win Rate %": round(stats["Win Rate [%]"], 1) if n_trades else 0.0,
        "Profit Factor": round(stats["Profit Factor"], 2) if n_trades else float("nan"),
        "SQN": round(stats["SQN"], 2) if n_trades else float("nan"),
        "# Trades": n_trades,
    }, bt, stats


def run_period(label, start=None, end=None, period=None):
    results = []
    for t in TICKERS:
        print(f"Backtesting {t} ({label})...", file=sys.stderr)
        row, bt, stats = run_one(t, start=start, end=end) if start else run_one(t)
        results.append(row)
    df = pd.DataFrame(results).set_index("Ticker")
    print(f"\n=== {label} ===")
    print(df.to_string())
    print(
        f"Average CAGR: {df['CAGR %'].mean():.2f}%  |  Avg Max DD: {df['Max DD %'].mean():.2f}%  |  "
        f"Avg Sharpe: {df['Sharpe'].mean():.2f}  |  Avg Win Rate: {df['Win Rate %'].mean():.1f}%"
    )
    return df


def main():
    bull_df = run_period(f"Bull market, last {PERIOD}")
    gfc_df = run_period("2007 Financial Crisis (2007-06-01 to 2009-06-01)", start="2007-06-01", end="2009-06-01")
    bear2022_df = run_period("2022 Rate-Hike Bear Market (2022-01-01 to 2022-12-31)", start="2022-01-01", end="2022-12-31")

    print("\n=== Summary (CAGR % | Max DD % | Sharpe | Win Rate %) ===")
    for label, d in [("Bull market", bull_df), ("2008 GFC", gfc_df), ("2022 bear", bear2022_df)]:
        print(
            f"{label:12s} strategy: {d['CAGR %'].mean():6.2f}% | {d['Max DD %'].mean():6.2f}% | "
            f"{d['Sharpe'].mean():5.2f} | {d['Win Rate %'].mean():5.1f}%   "
            f"(buy&hold {d['Buy&Hold %'].mean():.1f}% total return)"
        )

    bull_df.to_csv("backtest_bull.csv")
    gfc_df.to_csv("backtest_gfc.csv")
    bear2022_df.to_csv("backtest_2022bear.csv")
    print("\nSaved backtest_bull.csv, backtest_gfc.csv, backtest_2022bear.csv")


if __name__ == "__main__":
    main()
