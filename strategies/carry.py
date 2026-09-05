"""Carry — the second signal, and the first that is not a price pattern.

A futures curve tells you what the market pays you to hold a position. When
the next contract trades below the front (backwardation), a long earns the
difference as the contract rolls toward spot; when it trades above (contango),
a short earns it. Koijen, Moskowitz, Pedersen and Vrugt (2018) showed this
"carry" predicts returns in the time series of every asset class they tested
— equities, bonds, currencies, commodities — and that a carry book is weakly
correlated with a trend book, which is the whole reason to run both.

The signal is read straight off the curve by `data/continuous.stitch`, which
puts an annualised `carry` column on every continuous bar. This strategy does
three things with it, all with fixed constants declared in research entry 009:

- smooth it over 20 days, because the day-to-day print of a spread is noisy;
- divide by its own 252-day standard deviation, so a 2% bond carry and a 40%
  natural-gas carry are judged on the same scale — comparable forecasts are
  what let one risk budget serve the whole universe;
- take the sign as the side and the magnitude, capped, as the confidence.

Below a quarter of a standard deviation there is no view and the book is flat
in that market. The stop is the same 2.5 ATR as the trend rule, because carry
says nothing about where a position should be wrong.

Continuous by default: the position is resized as the curve moves, through
the same `RiskEngine.resize` path as the continuous trend rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent, Strategy, forecast_to_confidence
from core.types import Position, Side
from features.indicators import atr, ema


class Carry(Strategy):
    name = "carry"

    def __init__(
        self,
        smooth: int = 20,
        norm_window: int = 252,
        atr_period: int = 14,
        atr_stop_multiple: float = 2.5,
        entry_threshold: float = 0.25,
        continuous: bool = True,
        forecast_cap: float = 2.0,
    ) -> None:
        self.smooth = smooth
        self.norm_window = norm_window
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.entry_threshold = entry_threshold
        self.continuous = continuous
        self.forecast_cap = forecast_cap
        self.rebalances = continuous
        self.warmup = max(norm_window // 2 + smooth, atr_period) + 5

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df, self.atr_period)
        if "carry" not in df.columns:
            # No curve, no carry: a CFD or a single expiry. Every bar reads flat.
            df["carry"] = np.nan
        raw = df["carry"].astype(float).ffill()
        df["carry_s"] = ema(raw, self.smooth)
        df["carry_sd"] = raw.rolling(self.norm_window, min_periods=self.norm_window // 2).std()
        df["forecast"] = df["carry_s"] / df["carry_sd"].replace(0.0, np.nan)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        row = df.iloc[i]
        forecast, atr_now = row["forecast"], row["atr"]
        if pd.isna(forecast) or pd.isna(atr_now) or atr_now <= 0:
            return FLAT
        if abs(forecast) < self.entry_threshold:
            return FLAT
        side = Side.BUY if forecast > 0 else Side.SELL
        confidence = forecast_to_confidence(float(forecast), self.forecast_cap) if self.continuous else 1.0
        if confidence <= 0.0:
            return FLAT
        return Intent(
            side=side,
            stop_distance=float(atr_now) * self.atr_stop_multiple,
            confidence=confidence,
            reason=f"carry={row['carry']:+.3f}/yr f={forecast:+.2f}",
        )
