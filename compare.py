"""
A/B comparison: baseline (no ADX filter) vs ADX-filtered vs ADX-filtered + leverage.
Not part of the permanent toolkit -- a one-off experiment script to decide
whether the regime filter (and leverage) actually help before keeping them.
"""

import sys

import pandas as pd
import yfinance as yf
from backtesting import Backtest

from strategy import EmaTrendRsiAtr

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT"]
CASH = 10_000
COMMISSION = 0.0005

PERIODS = [
    ("Bull market (8y)", dict(period="8y")),
    ("2008 GFC", dict(start="2007-06-01", end="2009-06-01")),
    ("2022 bear", dict(start="2022-01-01", end="2022-12-31")),
]

VARIANTS = [
    ("Baseline (no ADX)", dict(adx_min=0), dict(margin=1.0)),
    ("ADX-filtered", dict(adx_min=20), dict(margin=1.0)),
    ("ADX-filtered + 1.5x leverage", dict(adx_min=20), dict(margin=1 / 1.5)),
    ("Vol-targeted sizing (2x max)", dict(adx_min=0, use_vol_sizing=True, max_leverage=2.0), dict(margin=0.5)),
    ("Vol-targeted sizing, higher risk (2.5%, 2x max)",
     dict(adx_min=0, use_vol_sizing=True, risk_per_trade=0.025, max_leverage=2.0), dict(margin=0.5)),
]


def load_data(ticker, period=None, start=None, end=None):
    if start:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    rows = []
    for period_label, period_kwargs in PERIODS:
        for ticker in TICKERS:
            data = load_data(ticker, **period_kwargs)
            for variant_label, strat_kwargs, bt_kwargs in VARIANTS:
                bt = Backtest(data, EmaTrendRsiAtr, cash=CASH, commission=COMMISSION,
                               exclusive_orders=True, **bt_kwargs)
                stats = bt.run(**strat_kwargs)
                n = int(stats["# Trades"])
                rows.append({
                    "Period": period_label,
                    "Variant": variant_label,
                    "Ticker": ticker,
                    "CAGR %": round(stats["CAGR [%]"], 2),
                    "MaxDD %": round(stats["Max. Drawdown [%]"], 2),
                    "Sharpe": round(stats["Sharpe Ratio"], 2),
                    "WinRate %": round(stats["Win Rate [%]"], 1) if n else 0.0,
                    "ProfitFactor": round(stats["Profit Factor"], 2) if n else float("nan"),
                    "Trades": n,
                })
            print(f"done: {period_label} / {ticker}", file=sys.stderr)

    df = pd.DataFrame(rows)
    summary = df.groupby(["Period", "Variant"], sort=False)[
        ["CAGR %", "MaxDD %", "Sharpe", "WinRate %", "ProfitFactor", "Trades"]
    ].mean(numeric_only=True).round(2)
    print("\n=== Averaged across SPY/QQQ/AAPL/MSFT ===")
    print(summary.to_string())
    df.to_csv("compare_results.csv", index=False)
    print("\nFull per-ticker results saved to compare_results.csv")


if __name__ == "__main__":
    main()
