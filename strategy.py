import math

from backtesting import Strategy
from backtesting.lib import crossover

from indicators import adx, atr, ema, rsi


class EmaTrendRsiAtr(Strategy):
    """
    Long-only trend-following strategy.
    Entry: fast EMA crosses above slow EMA, RSI is not already overbought,
           optionally AND ADX confirms a trending regime (adx_min=0 disables).
    Exit:  ATR trailing stop (ratchets up, never down) or a trend-reversal
           crossunder.
    Sizing: if use_vol_sizing, risk a fixed % of equity against the ATR stop
            distance instead of always going all-in -- position size shrinks
            automatically in choppy/high-ATR regimes and grows in calm
            trending ones, capped at max_leverage. Requires the Backtest
            object's margin to permit max_leverage (margin=1/max_leverage).
    """

    fast_len = 20
    slow_len = 50
    rsi_len = 14
    rsi_long_max = 70
    atr_len = 14
    atr_stop_mult = 2.0
    adx_len = 14
    adx_min = 0  # tested an ADX regime filter (adx_min=20): hurt Sharpe/CAGR in every regime, dropped
    use_vol_sizing = True  # risk a fixed % of equity per trade instead of always going all-in
    risk_per_trade = 0.015
    max_leverage = 2.0  # requires Backtest(..., margin=1/max_leverage) to actually allow this

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        self.fast_ema = self.I(ema, close, self.fast_len)
        self.slow_ema = self.I(ema, close, self.slow_len)
        self.rsi_val = self.I(rsi, close, self.rsi_len)
        self.atr_val = self.I(atr, high, low, close, self.atr_len)
        self.adx_val = self.I(adx, high, low, close, self.adx_len)

    def next(self):
        price = self.data.Close[-1]

        if not self.position:
            trending = self.adx_val[-1] > self.adx_min
            if crossover(self.fast_ema, self.slow_ema) and self.rsi_val[-1] < self.rsi_long_max and trending:
                stop_distance = self.atr_val[-1] * self.atr_stop_mult
                sl = price - stop_distance
                if sl >= price:
                    return
                if self.use_vol_sizing:
                    dollar_risk = self.equity * self.risk_per_trade
                    shares = math.floor(dollar_risk / stop_distance)
                    max_shares = math.floor(self.equity * self.max_leverage / price)
                    shares = min(shares, max_shares)
                    if shares > 0:
                        self.buy(size=shares, sl=sl)
                else:
                    self.buy(sl=sl)
            return

        trade = self.trades[-1]
        new_sl = price - self.atr_val[-1] * self.atr_stop_mult
        if trade.sl is None or new_sl > trade.sl:
            trade.sl = new_sl

        if crossover(self.slow_ema, self.fast_ema):
            self.position.close()
