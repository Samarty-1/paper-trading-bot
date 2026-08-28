"""
Portfolio-level backtest: the exact same, unmodified EmaTrendRsiAtr strategy
run independently across a diversified instrument set, then combined into one
equal-weighted portfolio equity curve.

Why this exists (see README "Why diversification, not parameter tuning"):
single-instrument Sharpe here is weak (0.0-0.35) mostly because Exposure Time
is only ~10-17% -- the strategy sits in cash most of the time, and idle days
mechanically dilute a calendar-time Sharpe/CAGR even when the per-trade edge
is genuinely positive (Profit Factor > 1 on most instruments). No strategy
parameter is touched here; the fix is portfolio construction, not curve
fitting -- run the same trend-following rule across instruments whose trades
don't fire on the same days (different asset classes trend at different
times), so the portfolio is "in a trade" far more often than any single
instrument, even though each instrument's own trade frequency is unchanged.
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest

from strategy import EmaTrendRsiAtr

# SPY/QQQ/AAPL/MSFT (equities) + GLD (gold), TLT (long bonds), DBC (commodities),
# EFA (international developed equities) -- spans asset classes that don't all
# trend at the same time, which is the entire point.
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GLD", "TLT", "DBC", "EFA"]
CASH = 10_000
COMMISSION = 0.0005

PERIODS = [
    ("Bull market (8y)", dict(period="8y")),
    ("2008 GFC", dict(start="2007-06-01", end="2009-06-01")),
    ("2022 bear", dict(start="2022-01-01", end="2022-12-31")),
]


def load_data(ticker, period=None, start=None, end=None):
    if start:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def run_portfolio(label, **period_kwargs):
    curves = {}
    single_rows = []
    for ticker in TICKERS:
        data = load_data(ticker, **period_kwargs)
        if len(data) < 60:
            print(f"  skipping {ticker}: insufficient data for this window", file=sys.stderr)
            continue
        bt = Backtest(data, EmaTrendRsiAtr, cash=CASH, commission=COMMISSION, exclusive_orders=True, margin=0.5)
        stats = bt.run()
        curves[ticker] = stats["_equity_curve"]["Equity"]
        n = int(stats["# Trades"])
        single_rows.append({
            "Ticker": ticker,
            "Exposure %": round(stats["Exposure Time [%]"], 1),
            "Sharpe": round(stats["Sharpe Ratio"], 2),
            "CAGR %": round(stats["CAGR [%]"], 2),
            "Profit Factor": round(stats["Profit Factor"], 2) if n else float("nan"),
            "Trades": n,
        })

    # Equal-weighted: each instrument trades its own 1/N slice of capital
    # independently; portfolio daily return is the mean of each instrument's
    # daily equity return (0 on days that instrument is flat/in cash).
    rets = pd.DataFrame({t: curves[t].pct_change() for t in curves}).fillna(0)
    port_ret = rets.mean(axis=1)
    port_equity = (1 + port_ret).cumprod()
    port_sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else float("nan")
    running_max = port_equity.cummax()
    max_dd = (port_equity / running_max - 1).min() * 100
    total_return = (port_equity.iloc[-1] - 1) * 100
    avg_single_sharpe = np.mean([r["Sharpe"] for r in single_rows])

    single_df = pd.DataFrame(single_rows).set_index("Ticker")
    print(f"\n=== {label} ===")
    print(single_df.to_string())
    print(
        f"\nAvg single-instrument Sharpe: {avg_single_sharpe:.2f}  |  "
        f"Portfolio Sharpe: {port_sharpe:.2f}  |  "
        f"Portfolio total return: {total_return:.2f}%  |  "
        f"Portfolio Max DD: {max_dd:.2f}%"
    )
    return {
        "label": label, "avg_single_sharpe": avg_single_sharpe, "portfolio_sharpe": port_sharpe,
        "portfolio_total_return_pct": total_return, "portfolio_max_dd_pct": max_dd,
    }


def main():
    summary = [run_portfolio(label, **kwargs) for label, kwargs in PERIODS]
    print("\n=== Summary: does diversification actually help? ===")
    for row in summary:
        delta = row["portfolio_sharpe"] - row["avg_single_sharpe"]
        verdict = "HELPS" if delta > 0.1 else ("NO BENEFIT" if delta > -0.1 else "HURTS")
        print(
            f"{row['label']:20s} avg single Sharpe {row['avg_single_sharpe']:5.2f} -> "
            f"portfolio Sharpe {row['portfolio_sharpe']:5.2f}  ({verdict})"
        )
    pd.DataFrame(summary).to_csv("portfolio_backtest_results.csv", index=False)
    print("\nSaved portfolio_backtest_results.csv")


if __name__ == "__main__":
    main()
