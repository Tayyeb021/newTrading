"""Carry — the second signal, and the first that is not a price pattern.

A futures curve tells you what the market pays you to hold a position. When
the next contract trades below the front (backwardation), a long earns the
difference as the contract rolls toward spot; when it trades above (contango),
a short earns it. Koijen, Moskowitz, Pedersen and Vrugt (2018) showed this
"carry" predicts returns in the time series of every asset class they tested
— equities, bonds, currencies, commodities — and that a carry book is weakly
correlated with a trend book, which is the whole reason to run both.

The signal is read straight off the curve by `data/continuous.stitch`, which
puts an annualised `carry` column on every continuous bar, and the raw front
close beside it. Two ways to turn that into a forecast:

- `normalise="own_sd"` (entry 009): the 20-day-smoothed carry divided by its
  own 252-day standard deviation. Dead in 009's form. Its flaw was visible in
  the verdict: in a market whose carry sits at zero, a quarter of a tiny
  standard deviation is not a signal, but the rule traded it daily.
- `normalise="price_vol"` (entry 010): the smoothed carry divided by the
  market's annualised price volatility - carry in risk units, as Carver runs
  it. Ten percent of a market's annual volatility is a signal; the threshold
  and cap are declared in the log.

`decide_monthly=True` takes the decision on the first bar of each month and
holds between decisions, with no re-entry after a stop. The 009 form decided
daily and resized continuously; the diagnostic in the log shows what that
cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent, Strategy, forecast_to_confidence, is_month_start
from core.types import Position, Side
from features.indicators import atr, ema, price_vol


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
        normalise: str = "own_sd",
        vol_window: int = 63,
        decide_monthly: bool = False,
    ) -> None:
        if normalise not in ("own_sd", "price_vol"):
            raise ValueError(f"normalise must be 'own_sd' or 'price_vol', got {normalise!r}")
        self.smooth = smooth
        self.norm_window = norm_window
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.entry_threshold = entry_threshold
        self.continuous = continuous
        self.forecast_cap = forecast_cap
        self.normalise = normalise
        self.vol_window = vol_window
        self.decide_monthly = decide_monthly
        self.rebalances = continuous
        window = norm_window // 2 if normalise == "own_sd" else vol_window
        self.warmup = max(window + smooth, atr_period) + 5
        self._decided: tuple[Side | None, float] = (None, 1.0)

    @classmethod
    def published(cls) -> "Carry":
        """The entry-010 form: carry in risk units, monthly, discrete, wide stop."""
        return cls(normalise="price_vol", decide_monthly=True, continuous=False,
                   entry_threshold=0.10, forecast_cap=0.50, atr_stop_multiple=4.0)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df, self.atr_period)
        if "carry" not in df.columns:
            # No curve, no carry: a CFD or a single expiry. Every bar reads flat.
            df["carry"] = np.nan
        raw = df["carry"].astype(float).ffill()
        df["carry_s"] = ema(raw, self.smooth)
        if self.normalise == "own_sd":
            df["carry_sd"] = raw.rolling(self.norm_window, min_periods=self.norm_window // 2).std()
            df["forecast"] = df["carry_s"] / df["carry_sd"].replace(0.0, np.nan)
        else:
            if "raw_close" not in df.columns:
                df["forecast"] = np.nan  # a level is needed and there is none
            else:
                level = df["raw_close"].astype(float).replace(0.0, np.nan).abs()
                vol_ann = price_vol(df["close"], self.vol_window) * np.sqrt(252.0) / level
                df["forecast"] = df["carry_s"] / vol_ann.replace(0.0, np.nan)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        row = df.iloc[i]
        forecast, atr_now = row["forecast"], row["atr"]
        if pd.isna(atr_now) or atr_now <= 0:
            return FLAT
        stop = float(atr_now) * self.atr_stop_multiple

        if self.decide_monthly and not is_month_start(df, i):
            if position is None:
                return FLAT
            side, confidence = self._decided
            if side is not position.side:
                side, confidence = position.side, 1.0
            return Intent(side=side, stop_distance=stop, confidence=confidence, reason="hold")

        if pd.isna(forecast) or abs(forecast) < self.entry_threshold:
            self._decided = (None, 1.0)
            return FLAT
        side = Side.BUY if forecast > 0 else Side.SELL
        confidence = forecast_to_confidence(float(forecast), self.forecast_cap) if self.continuous else 1.0
        if confidence <= 0.0:
            self._decided = (None, 1.0)
            return FLAT
        self._decided = (side, confidence)
        return Intent(
            side=side, stop_distance=stop, confidence=confidence,
            reason=f"carry={row['carry']:+.3f}/yr f={forecast:+.2f}",
        )
