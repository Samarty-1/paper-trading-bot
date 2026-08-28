"""
Paper trading engine. Simulates the EmaTrendRsiAtr strategy forward in time
using real market data, with a virtual portfolio (starting cash, no real
orders ever placed). Run once per day, after market close, so the day's bar
is final.

Usage:
    python paper_trade.py                  # run/update for default ticker (SPY)
    python paper_trade.py --ticker QQQ
    python paper_trade.py --reset           # wipe state and start over
"""

import argparse
import json
import math
import os

from etl.load import read_from_warehouse

FAST_LEN = 20
SLOW_LEN = 50
RSI_LONG_MAX = 70
ATR_STOP_MULT = 2.0
RISK_PER_TRADE = 0.015  # % of equity risked per trade, sized against the ATR stop distance
MAX_LEVERAGE = 2.0      # position value capped at this multiple of equity even in low-vol regimes
INITIAL_CASH = 10_000


def state_path(ticker):
    return f"state_{ticker.upper()}.json"


def load_state(ticker, reset=False):
    path = state_path(ticker)
    if reset and os.path.exists(path):
        os.remove(path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "ticker": ticker.upper(),
        "cash": INITIAL_CASH,
        "shares": 0,
        "entry_price": None,
        "stop": None,
        "last_processed_date": None,
        "trade_log": [],
        "equity_curve": [],
        "bh_shares": None,  # buy-and-hold benchmark, bought once on day 1
    }


def save_state(state, ticker):
    with open(state_path(ticker), "w") as f:
        json.dump(state, f, indent=2, default=str)


def run(ticker, reset=False):
    state = load_state(ticker, reset=reset)
    df = read_from_warehouse(ticker)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_date = str(df.index[-1].date())

    if state["last_processed_date"] == last_date:
        print(f"[{ticker}] Already processed {last_date}. Nothing new. Run again after the next close.")
        print_status(state, last["Close"], last_date)
        return

    close = float(last["Close"])
    low = float(last["Low"])

    if state["bh_shares"] is None:
        state["bh_shares"] = state["cash"] / close  # benchmark: all-in on day 1, fractional ok for tracking only

    long_cross = last["fast_ema"] > last["slow_ema"] and prev["fast_ema"] <= prev["slow_ema"]
    exit_cross = last["slow_ema"] > last["fast_ema"] and prev["slow_ema"] <= prev["fast_ema"]

    if state["shares"] == 0:
        if long_cross and last["rsi"] < RSI_LONG_MAX:
            stop_distance = last["atr"] * ATR_STOP_MULT
            equity = state["cash"]  # flat position, so equity == cash here
            dollar_risk = equity * RISK_PER_TRADE
            shares_to_buy = math.floor(dollar_risk / stop_distance) if stop_distance > 0 else 0
            max_shares = math.floor(equity * MAX_LEVERAGE / close)
            shares_to_buy = min(shares_to_buy, max_shares)
            if shares_to_buy > 0:
                stop = close - stop_distance
                state["cash"] -= shares_to_buy * close  # may go negative: simulated margin debit, tracked in equity
                state["shares"] = shares_to_buy
                state["entry_price"] = close
                state["stop"] = stop
                log_trade(state, last_date, "BUY", shares_to_buy, close, f"EMA{FAST_LEN}/{SLOW_LEN} crossover, RSI {last['rsi']:.1f}, risk-sized")
                print(f"[{ticker}] BUY {shares_to_buy} @ {close:.2f} (stop {stop:.2f}, {shares_to_buy*close/equity:.1f}x equity)")
    else:
        new_stop = close - last["atr"] * ATR_STOP_MULT
        if new_stop > state["stop"]:
            state["stop"] = new_stop

        if low <= state["stop"]:
            fill = state["stop"]
            proceeds = state["shares"] * fill
            pnl = proceeds - state["shares"] * state["entry_price"]
            log_trade(state, last_date, "SELL (stop)", state["shares"], fill, f"P&L {pnl:+.2f}")
            print(f"[{ticker}] STOP HIT: SELL {state['shares']} @ {fill:.2f}  P&L {pnl:+.2f}")
            state["cash"] += proceeds
            state["shares"] = 0
            state["entry_price"] = None
            state["stop"] = None
        elif exit_cross:
            proceeds = state["shares"] * close
            pnl = proceeds - state["shares"] * state["entry_price"]
            log_trade(state, last_date, "SELL (trend reversal)", state["shares"], close, f"P&L {pnl:+.2f}")
            print(f"[{ticker}] TREND REVERSAL: SELL {state['shares']} @ {close:.2f}  P&L {pnl:+.2f}")
            state["cash"] += proceeds
            state["shares"] = 0
            state["entry_price"] = None
            state["stop"] = None

    equity = state["cash"] + state["shares"] * close
    state["equity_curve"].append({"date": last_date, "equity": round(equity, 2)})
    state["last_processed_date"] = last_date
    save_state(state, ticker)
    print_status(state, close, last_date)


def log_trade(state, date, action, shares, price, note):
    state["trade_log"].append({"date": date, "action": action, "shares": shares, "price": round(price, 2), "note": note})


def print_status(state, last_close, last_date):
    equity = state["cash"] + state["shares"] * last_close
    bh_equity = state["bh_shares"] * last_close if state["bh_shares"] else INITIAL_CASH
    strat_return = (equity / INITIAL_CASH - 1) * 100
    bh_return = (bh_equity / INITIAL_CASH - 1) * 100

    print(f"\n--- {state['ticker']} paper trading status as of {last_date} ---")
    if state["shares"] > 0:
        print(f"Position: LONG {state['shares']} shares @ entry {state['entry_price']:.2f}, trailing stop {state['stop']:.2f}")
    else:
        print("Position: FLAT (waiting for entry signal)")
    print(f"Cash: ${state['cash']:.2f}  |  Equity: ${equity:.2f}  ({strat_return:+.1f}%)")
    print(f"Buy & hold benchmark equity: ${bh_equity:.2f}  ({bh_return:+.1f}%)")
    print(f"Total trades so far: {len(state['trade_log'])}")
    print_sentiment_advisory(state["ticker"])


def print_sentiment_advisory(ticker):
    """Advisory-only headline sentiment read (see sentiment.py for why this
    doesn't feed the entry/exit logic above). Skipped quietly if the optional
    sentiment dependencies (scikit-learn/datasets/joblib) aren't installed."""
    try:
        import sentiment
    except ImportError:
        return
    try:
        result = sentiment.score_ticker_sentiment(ticker)
    except Exception as exc:
        print(f"(sentiment advisory unavailable: {exc})")
        return
    if result["label"] == "no_data":
        return
    print(
        f"Advisory only, not a trade signal - headline sentiment: {result['label']} "
        f"(score {result['sentiment_score']:+.2f}, {result['n_headlines']} headlines)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run(args.ticker, reset=args.reset)
