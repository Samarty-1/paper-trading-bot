import math

from backtesting import Strategy

from indicators import atr, rolling_max, rolling_mean


class MomentumBreakout(Strategy):
    """
    Intraday momentum breakout, long-only.
    Entry: price breaks above the highest high of the last `breakout_len`
           bars, confirmed by volume well above its recent average
           ("it's taking off, with real buying behind it" -- not a fakeout).
    Exit:  ATR trailing stop, or forced flat before the trading day ends
           (no overnight holds -- avoids gap risk from after-hours news).
    Sizing: same risk-per-trade-against-ATR-stop approach as the daily
            strategy, capped at max_leverage.
    """

    breakout_len = 20
    vol_len = 20
    vol_mult = 1.5
    atr_len = 14
    atr_stop_mult = 1.5
    risk_per_trade = 0.015
    max_leverage = 2.0
    entry_start_hour = 10  # skip the noisy first 30 min after the 9:30 open
    entry_start_minute = 0
    entry_cutoff_hour = 15
    entry_cutoff_minute = 30
    flat_cutoff_hour = 15
    flat_cutoff_minute = 55

    def init(self):
        high = self.data.High
        volume = self.data.Volume
        close = self.data.Close
        low = self.data.Low
        self.hh = self.I(rolling_max, high, self.breakout_len)
        self.vol_avg = self.I(rolling_mean, volume, self.vol_len)
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
            if self._past_cutoff(t, self.flat_cutoff_hour, self.flat_cutoff_minute):
                self.position.close()
            return

        if self._past_cutoff(t, self.entry_cutoff_hour, self.entry_cutoff_minute):
            return
        if not self._past_cutoff(t, self.entry_start_hour, self.entry_start_minute):
            return
        if self.data.Volume[-1] == 0:
            return

        above_vwap = price > self.data.VWAP[-1]  # only trade with the day's dominant flow; else hold/wait
        breakout = price > self.hh[-1] and self.data.Volume[-1] > self.vol_avg[-1] * self.vol_mult
        if not (breakout and above_vwap):
            return

        stop_distance = self.atr_val[-1] * self.atr_stop_mult
        if stop_distance <= 0:
            return
        dollar_risk = self.equity * self.risk_per_trade
        shares = math.floor(dollar_risk / stop_distance)
        max_shares = math.floor(self.equity * self.max_leverage / price)
        shares = min(shares, max_shares)
        if shares > 0:
            self.buy(size=shares, sl=price - stop_distance)
