"""
Orchestrates the full workflow: Extract -> Transform -> Load -> Strategy.

Usage:
    python etl/pipeline.py                          # default ticker set
    python etl/pipeline.py --tickers SPY,QQQ         # specific tickers
    python etl/pipeline.py --tickers SPY --skip-trade  # ETL only, no paper trading
"""

import argparse
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.config import DEFAULT_TICKERS, LOG_DIR
from etl.extract import extract
from etl.load import load
from etl.transform import transform


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"pipeline_{date.today().isoformat()}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger("pipeline")


def run_ticker(log, ticker, skip_trade):
    log.info(f"[{ticker}] extract...")
    raw_df, raw_path = extract(ticker)
    log.info(f"[{ticker}] extracted {len(raw_df)} rows -> {raw_path}")

    log.info(f"[{ticker}] transform...")
    tidy_df = transform(ticker, raw_df)

    log.info(f"[{ticker}] load...")
    n = load(tidy_df)
    log.info(f"[{ticker}] loaded {n} rows into warehouse")

    if not skip_trade:
        import paper_trade
        log.info(f"[{ticker}] running paper trade strategy...")
        paper_trade.run(ticker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--skip-trade", action="store_true", help="run ETL only, skip paper_trade.run()")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    log = setup_logging()
    failures = []
    for ticker in tickers:
        try:
            run_ticker(log, ticker, args.skip_trade)
        except Exception as e:
            log.error(f"[{ticker}] pipeline failed: {e}")
            failures.append(ticker)

    if failures:
        log.warning(f"Completed with failures: {failures}")
        sys.exit(1)
    log.info("Pipeline completed successfully for all tickers.")


if __name__ == "__main__":
    main()
