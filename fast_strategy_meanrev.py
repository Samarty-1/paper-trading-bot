import math

from backtesting import Strategy

from indicators import atr, rsi


class MeanReversion(Strategy):
    """
    Intraday mean-reversion, long-only.
    Entry: RSI drops into oversold territory (a sharp short-term dip).
    Exit:  RSI recovers back to neutral, ATR trailing stop, or forced flat
           before the trading day ends.
    Tests the opposite hypothesis to MomentumBreakout, after that strategy's
    trade log showed breakouts were getting faded rather than continuing.
    """

    rsi_len = 7
    rsi_oversold = 30
    rsi_exit = 50
    atr_len = 14
    atr_stop_mult = 1.5
    risk_per_trade = 0.015
    max_leverage = 2.0
    entry_start_hour = 10
    entry_start_minute = 0
    entry_cutoff_hour = 15
    entry_cutoff_minute = 30
    flat_cutoff_hour = 15
    flat_cutoff_minute = 55

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        self.rsi_val = self.I(rsi, close, self.rsi_len)
        self.atr_val = self.I(atr, high, low, close, self.atr_len)

    def _past_cutoff(self, t, hour, minute):
        return t.hour > hour or (t.hour == hour and t.minute >= minute)

    def next(self):
        price = self.data.Close[-1]
        t = self.data.index[-1].time()

        if self.position:
            new_stop = price - self.atr_val[-1] * self.atr_stop_mult
            trade = self.trades[-1]
            if trade.sl is None or new_stop > trade.sl:
                trade.sl = new_stop
            if self.rsi_val[-1] > self.rsi_exit or self._past_cutoff(t, self.flat_cutoff_hour, self.flat_cutoff_minute):
                self.position.close()
            return

        if self._past_cutoff(t, self.entry_cutoff_hour, self.entry_cutoff_minute):
            return
        if not self._past_cutoff(t, self.entry_start_hour, self.entry_start_minute):
            return

        if self.rsi_val[-1] < self.rsi_oversold:
            stop_distance = self.atr_val[-1] * self.atr_stop_mult
            if stop_distance <= 0:
                return
            dollar_risk = self.equity * self.risk_per_trade
            shares = math.floor(dollar_risk / stop_distance)
            max_shares = math.floor(self.equity * self.max_leverage / price)
            shares = min(shares, max_shares)
            if shares > 0:
                self.buy(size=shares, sl=price - stop_distance)
