# Paper Trading Bot

EMA trend-following strategy with an RSI entry filter, an ATR trailing stop,
and volatility-targeted position sizing (risks a fixed % of equity per trade
against the ATR stop distance, capped at 2x leverage — so size shrinks
automatically in choppy/high-volatility regimes and grows in calm trending
ones). No real orders are ever placed — this simulates a portfolio locally
using real market data pulled from Yahoo Finance.

## Validated behavior (see `backtest.py` output, averaged across SPY/QQQ/AAPL/MSFT)

| Regime | CAGR | Max Drawdown | Sharpe | Win Rate |
|---|---|---|---|---|
| Bull market (8y) | 0.89% | -4.73% | 0.35 | 49.2% |
| 2008 financial crisis | 0.06% | -4.12% | 0.00 | 51.2% |
| 2022 rate-hike bear | -0.83% | -2.23% | -0.38 | 39.6% |

For reference, buy-and-hold over the same periods: +354% (bull), -19.9%
(2008), -26.1% (2022).

This is a capital-protection strategy, not a beat-the-market strategy. It's
meant to lose less in downturns, not win big in a straight-up bull run —
Sharpe is modest (0.0-0.35), not the >1 a professional desk would want to
call this a strong alpha strategy.

### What was tried and rejected

- **Fixed take-profit** (early version): capped winners early in strong
  trends, replaced with an ATR trailing stop.
- **ADX regime filter** (only trade when ADX > 20): made every metric worse
  in every regime tested (e.g. 2022 Sharpe went from -0.35 to -1.34) because
  it stacks two lagging trend indicators and starves the strategy of trades.
  Rejected — see `compare.py` if you want to reproduce this.
- **Flat 1.5x leverage**: scales return and drawdown together, doesn't
  change Sharpe. Replaced with volatility-targeted sizing instead, which
  applies leverage adaptively (more when calm, less when choppy) and
  measurably cut max drawdown in every regime without hurting Sharpe.

## Why Sharpe is weak, and the one thing that actually raised it (`portfolio_backtest.py`)

Digging into the trade log (not just the summary stats) shows *why* the
single-instrument Sharpe above is only 0.0-0.35: **Exposure Time is only
~8-17%** — the strategy sits in cash 83-92% of the time, waiting for an
EMA crossover. That idle time isn't neutral: a calendar-time Sharpe ratio
(computed on daily returns including the flat, zero-return cash days) gets
mechanically diluted by all those zero days, *even when the per-trade edge
is genuinely good* — Profit Factor is 1.78-2.62 on SPY/QQQ/AAPL individually,
i.e. real, positive expectancy per trade. The problem isn't edge quality,
it's how rarely that edge gets deployed against calendar time.

The standard trend-following fix for exactly this (how CTAs turn a ~0.3-0.5
single-instrument Sharpe into a ~0.7-1.0+ fund-level Sharpe) is **not**
tuning the strategy's parameters — it's running the same, completely
unmodified strategy across instruments from *different asset classes* whose
trades don't fire on the same calendar days, so the combined portfolio is
"in a trade" far more of the time than any single instrument, even though
each instrument's own trade frequency is unchanged. `portfolio_backtest.py`
tests this directly: same `EmaTrendRsiAtr`, zero parameters touched, run
independently on SPY/QQQ/AAPL/MSFT plus GLD (gold), TLT (long bonds), DBC
(commodities), and EFA (international equities), then combined into one
equal-weighted portfolio equity curve.

| Regime | Avg. single-instrument Sharpe | Portfolio Sharpe | Portfolio Max DD |
|---|---|---|---|
| Bull market (8y) | 0.26 | **0.68** | -1.62% |
| 2008 financial crisis | 0.39 | **0.84** | -1.47% |
| 2022 rate-hike bear | -0.43 | -0.45 | -1.25% |

Diversification roughly **doubled Sharpe** in the bull market and GFC
regimes, and cut max drawdown to under -2% in both — a real, mechanical
result of decorrelated trade timing, not a fitted parameter. It's reported
honestly for 2022 too, where it **didn't help**: 2022 was a rare
"everything falls together" regime (the same well-documented reason 60/40
portfolios had their worst year in decades that year) — synchronized global
rate hikes hit stocks, bonds, and international equities simultaneously, so
the diversification benefit that shows up in the other two regimes
temporarily disappears. That's not a bug in the test, it's the honest
result of that specific macro regime.

Run it yourself: `python portfolio_backtest.py` (writes
`portfolio_backtest_results.csv`). To actually paper-trade this diversified
book rather than just backtest it, run `paper_trade.py --ticker <T>`
independently for each of the 8 tickers above (each gets its own
`state_<T>.json`) — there's no separate orchestration layer for this, since
each ticker already runs and persists state independently.

## Architecture

Market data flows through a small ETL pipeline before the strategy ever sees
it, rather than each script re-fetching from Yahoo Finance independently:

```
Extract          Transform              Load              Orchestrate       Strategy
yfinance   -->   validate + compute --> DuckDB warehouse   etl/pipeline.py   paper_trade.py
(etl/extract.py)  indicators             (etl/load.py)     (daily, via       reads the
raw snapshot to   (etl/transform.py,     idempotent         Task Scheduler   warehouse
data/raw/         reuses indicators.py)  upsert              -- see          (etl/load.py:
                                                              schedule_task.md) read_from_warehouse)
```

- **Extract** (`etl/extract.py`) pulls daily OHLCV bars and writes an
  immutable raw CSV snapshot to `data/raw/` before any cleaning, so raw pulls
  stay auditable independent of the transform logic.
- **Transform** (`etl/transform.py`) validates the raw frame (no NaN OHLC,
  no negative volume, dates in order) and computes the EMA/RSI/ATR indicator
  set using the same `indicators.py` functions the backtest relies on.
- **Load** (`etl/load.py`) upserts into a local DuckDB file
  (`data/market_data.duckdb`) — idempotent, so re-running the same day never
  creates duplicate rows.
- **Orchestrate** (`etl/pipeline.py`) runs Extract -> Transform -> Load for a
  list of tickers, then triggers `paper_trade.py` for each, with structured
  logs in `logs/` and per-ticker error isolation. See `schedule_task.md` to
  run it automatically once a day via Windows Task Scheduler.
- **Strategy** (`paper_trade.py`) no longer fetches data itself — it reads
  pre-cleaned, pre-computed bars from the warehouse via
  `etl.load.read_from_warehouse()`.

## Live sentiment advisory (trained on Hugging Face data, not part of the strategy)

`sentiment.py` trains a TF-IDF + Logistic Regression classifier on Hugging
Face's [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
dataset (9,543 train / 2,388 validation rows) and applies it live to each
ticker's current Yahoo Finance headlines. Held-out validation result:
**83.0% accuracy, 0.773 macro F1** across 3 classes (bearish/bullish/neutral)
— a real number from the dataset's own validation split, not cherry-picked.

This is printed as an advisory line in `paper_trade.py`'s status output —
it does **not** feed the EMA/RSI/ATR entry/exit logic above. Two reasons:

1. The HF dataset has no publish dates, so there's no way to backtest this
   signal against historical price action — folding an unvalidated rule into
   the strategy would contradict the "prove it helps before shipping it"
   standard this repo already holds itself to (see the rejected ADX filter
   above, which *was* tested and made every metric worse).
2. Optional dependency: if `scikit-learn`/`datasets`/`joblib` aren't
   installed, `paper_trade.py` skips the advisory line quietly rather than
   failing — the core rule-based strategy has no hard dependency on it.

## Setup

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

## Re-run the reliability backtest

```
venv\Scripts\python backtest.py
```

Tests SPY/QQQ/AAPL/MSFT across a bull market, the 2008 crisis, and 2022 to
check the strategy isn't just overfit to one period.

## Run the daily pipeline + paper trading

Run once per day, **after market close**, so the day's bar is final. This
pulls fresh data, loads it into the warehouse, and runs paper trading for
each ticker:

```
venv\Scripts\python etl\pipeline.py --tickers SPY,QQQ,AAPL,MSFT
```

To run the ETL only (no paper trading): add `--skip-trade`. To run
`paper_trade.py` directly for a ticker already in the warehouse:

```
venv\Scripts\python paper_trade.py --ticker SPY
```

State (cash, position, trade log, equity curve) persists in
`state_SPY.json`. Running it again the same day is a no-op — safe to re-run.

To start over: `venv\Scripts\python paper_trade.py --ticker SPY --reset`

See `schedule_task.md` to run the pipeline automatically every day.

## Tests

```
venv\Scripts\pytest tests/
```

## Files

- `indicators.py` — EMA / RSI / ATR / ADX, plain pandas
- `strategy.py` — the strategy for `backtesting.py` (used only for backtests)
- `backtest.py` — historical validation across tickers and market regimes
- `compare.py` — A/B experiment harness (baseline vs ADX filter vs vol-targeted sizing)
- `paper_trade.py` — the live (delayed-data) paper trading loop, reads from the warehouse
- `etl/extract.py`, `etl/transform.py`, `etl/load.py`, `etl/pipeline.py` — the ETL pipeline (see Architecture above)
- `tests/test_transform.py` — unit tests for the transform/validation stage
- `schedule_task.md` — how to schedule the daily pipeline run
