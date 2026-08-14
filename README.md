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
