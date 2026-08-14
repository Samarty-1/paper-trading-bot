import numpy as np
import pandas as pd
import pytest

from etl.transform import transform, validate

N = 60


def make_ohlcv():
    dates = pd.date_range("2024-01-01", periods=N, freq="D")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, N))
    high = close + rng.uniform(0, 1, N)
    low = close - rng.uniform(0, 1, N)
    open_ = close + rng.normal(0, 0.5, N)
    volume = rng.integers(1_000, 10_000, N)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates
    )


def test_validate_passes_on_clean_data():
    validate(make_ohlcv())  # should not raise


def test_validate_raises_on_missing_column():
    df = make_ohlcv().drop(columns=["Volume"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(df)


def test_validate_raises_on_nan_ohlc():
    df = make_ohlcv()
    df.loc[df.index[5], "Close"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate(df)


def test_validate_raises_on_negative_volume():
    df = make_ohlcv()
    df.loc[df.index[0], "Volume"] = -1
    with pytest.raises(ValueError, match="Negative volume"):
        validate(df)


def test_validate_raises_on_non_monotonic_dates():
    df = make_ohlcv().iloc[::-1]
    with pytest.raises(ValueError, match="increasing order"):
        validate(df)


def test_transform_output_shape_and_columns():
    out = transform("spy", make_ohlcv())
    expected_cols = {
        "ticker", "date", "open", "high", "low", "close", "volume",
        "fast_ema", "slow_ema", "rsi", "atr",
    }
    assert expected_cols.issubset(out.columns)
    assert len(out) == N
    assert (out["ticker"] == "SPY").all()


def test_transform_indicators_populated_after_warmup():
    out = transform("spy", make_ohlcv())
    tail = out.iloc[-1]
    assert not pd.isna(tail["fast_ema"])
    assert not pd.isna(tail["slow_ema"])
    assert not pd.isna(tail["rsi"])
    assert not pd.isna(tail["atr"])
