import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT"]

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DUCKDB_PATH = os.path.join(ROOT_DIR, "data", "market_data.duckdb")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

FETCH_PERIOD = "1y"
FETCH_INTERVAL = "1d"

BARS_TABLE = "bars"
BARS_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume",
    "fast_ema", "slow_ema", "rsi", "atr",
]
