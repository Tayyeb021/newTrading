"""Time-series momentum as published: monthly decisions, held between them.

Moskowitz, Ooi and Pedersen (2012): the sign of the trailing return decides
the side, positions are held for a month, and the rule is re-evaluated then.
Research entry 010.

What entry 007 measured was not this. That rule re-decided every day, exited
on any close through a moving average and re-entered the next morning, and
re-entered straight after a stop. On 33 markets it traded 10 to 40 times a
year per market and paid three dollars of friction for every dollar of gross,
with gross positive in every run. The published rule decides once a month.

Mechanics:

- The decision is taken on the first bar of each calendar month (a calendar
  rule, so backtest and live agree). Side is the sign of the price change
  over `lookback` bars; zero is flat.
- Between decisions the position is held. If there is none - the month's
  trade was stopped out, or there was no signal - nothing is opened until the
  next decision. That single rule is most of the difference from 007.
- The stop is a 4-ATR disaster stop. The risk engine will not hold an
  unstopped position, and at four ATR the monthly decision, not the stop, is
  what normally closes a trade.
- `continuous=True` maps the trend's t-statistic to confidence on decision
  days and re-proposes it between them; off by default, and off in 010.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent, Strategy, forecast_to_confidence, is_month_start
from core.types import Position, Side
from features.indicators import atr, price_vol


class TSMOM(Strategy):
    name = "tsmom"

    def __init__(
        self,
        lookback: int = 250,
        atr_period: int = 14,
        atr_stop_multiple: float = 4.0,
        continuous: bool = False,
        forecast_cap: float = 2.0,
    ) -> None:
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.continuous = continuous
        self.forecast_cap = forecast_cap
        self.rebalances = continuous
        self.warmup = max(lookback, atr_period) + 5
        self._decided: tuple[Side | None, float] = (None, 1.0)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df, self.atr_period)
        df["mom"] = df["close"] - df["close"].shift(self.lookback)
        horizon_vol = price_vol(df["close"], self.lookback) * np.sqrt(self.lookback)
        df["forecast"] = df["mom"] / horizon_vol.replace(0.0, np.nan)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        row = df.iloc[i]
        atr_now, mom = row["atr"], row["mom"]
        if pd.isna(atr_now) or atr_now <= 0:
            return FLAT
        stop = float(atr_now) * self.atr_stop_multiple

        if not is_month_start(df, i):
            if position is None:
                return FLAT  # wait for the next decision; no re-entry after a stop
            side, confidence = self._decided
            if side is not position.side:
                side, confidence = position.side, 1.0  # adopted or restored position: keep it
            return Intent(side=side, stop_distance=stop, confidence=confidence, reason="hold")

        if pd.isna(mom) or mom == 0:
            self._decided = (None, 1.0)
            return FLAT
        side = Side.BUY if mom > 0 else Side.SELL
        confidence = 1.0
        if self.continuous:
            confidence = forecast_to_confidence(float(row["forecast"]), self.forecast_cap)
            if confidence <= 0.0:
                self._decided = (None, 1.0)
                return FLAT
        self._decided = (side, confidence)
        return Intent(side=side, stop_distance=stop, confidence=confidence,
                      reason=f"monthly mom={mom:+.5g} f={row['forecast']:+.2f}")
