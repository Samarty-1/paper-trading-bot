import numpy as np
import pandas as pd


def ema(series, length):
    return pd.Series(series).ewm(span=length, adjust=False).mean().to_numpy()


def rsi(series, length=14):
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).to_numpy()


def atr(high, low, close, length=14):
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean().to_numpy()


def rolling_max(series, length):
    return pd.Series(series).rolling(length).max().shift(1).to_numpy()


def rolling_mean(series, length):
    return pd.Series(series).rolling(length).mean().to_numpy()


def adx(high, low, close, length=14):
    """Wilder's ADX. >~20-25 = trending regime, <20 = choppy/sideways."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    smooth_tr = tr.ewm(alpha=1 / length, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1 / length, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1 / length, adjust=False).mean()

    plus_di = 100 * smooth_plus_dm / smooth_tr
    minus_di = 100 * smooth_minus_dm / smooth_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / length, adjust=False).mean().to_numpy()
