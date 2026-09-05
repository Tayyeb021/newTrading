"""S1 — time-series momentum. The baseline.

From the strategy research, F1: the best-documented systematic effect in liquid
markets. Moskowitz, Ooi and Pedersen found significant time-series momentum across
58 futures and forwards, persisting one to twelve months.

The rules are deliberately crude, and the parameters are fixed from reasoning
rather than optimisation:

- **60-day lookback** — inside the documented 1-12 month band, and roughly a
  quarter, which is how position-taking institutions actually think.
- **2.5x ATR(14) stop** — wide enough that ordinary daily noise does not touch it.
- **Trade on the daily close, enter next open** — no intraday timing claim.

This exists to be *beaten*. It is the honest benchmark that the phase 4 machine
learning work has to clear out-of-sample. If a regime model and a meta-label
filter cannot improve on sixty lines of arithmetic, that is a real result and you
ship this instead.

Expect a low win rate near 35-40%, a long right tail, and multi-month flat
stretches. A trend system that wins often is usually a trend system with a hidden
look-ahead bug.

**Continuous mode** (`continuous=True`, research entry 008) keeps the direction
rule and changes only how much: the momentum is expressed in units of the
volatility expected over the lookback — a t-statistic for the trend — and
mapped to `confidence`, so a weak trend gets a small position and a strong one
full size. While in a position the strategy re-proposes that confidence and a
fresh ATR stop distance every bar; the engines resize toward the target through
the risk engine, which also ratchets the stop tighter. The discrete baseline is
untouched, which is what makes the two comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent, Strategy, forecast_to_confidence
from core.types import Position, Side
from features.indicators import atr, ema, realized_vol, rolling_percentile, rolling_return


class TrendFollowing(Strategy):
    name = "S1_trend"

    def __init__(
        self,
        lookback: int = 60,
        ema_period: int = 60,
        atr_period: int = 14,
        atr_stop_multiple: float = 2.5,
        vol_filter_percentile: float | None = 0.95,
        continuous: bool = False,
        forecast_cap: float = 2.0,
    ) -> None:
        self.lookback = lookback
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.vol_filter_percentile = vol_filter_percentile
        self.continuous = continuous
        self.forecast_cap = forecast_cap
        self.rebalances = continuous
        self.warmup = max(lookback, ema_period, atr_period) + 5

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df, self.atr_period)
        df["ema"] = ema(df["close"], self.ema_period)
        df["mom"] = rolling_return(df["close"], self.lookback)
        if self.vol_filter_percentile is not None:
            df["atr_pct"] = rolling_percentile(df["atr"], 252)
        # Trend strength: the lookback return over the volatility expected across
        # that horizon. Comparable across a bond and a gas contract, which is what
        # lets one cap serve the whole universe.
        horizon_vol = realized_vol(df["close"], self.lookback) / np.sqrt(252.0) * np.sqrt(self.lookback)
        df["forecast"] = df["mom"] / horizon_vol.replace(0.0, np.nan)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        row = df.iloc[i]
        atr_now, mom, ema_now = row["atr"], row["mom"], row["ema"]
        close = row["close"]

        if pd.isna(atr_now) or pd.isna(mom) or pd.isna(ema_now) or atr_now <= 0:
            return FLAT

        stop_distance = float(atr_now) * self.atr_stop_multiple

        # Direction needs both to agree: momentum sign and price versus its own
        # trend. Either alone flips far too often in a range.
        if mom > 0 and close > ema_now:
            side = Side.BUY
        elif mom < 0 and close < ema_now:
            side = Side.SELL
        else:
            # No trend. Exit if held; the stop handles the rest.
            return FLAT

        # Volatility filter. Opening a fresh trend position right after an
        # extreme move is how trend systems buy the exact top of a spike.
        if self.vol_filter_percentile is not None and position is None:
            pct = row.get("atr_pct")
            if pd.notna(pct) and pct > self.vol_filter_percentile:
                return FLAT

        confidence = 1.0
        if self.continuous:
            confidence = forecast_to_confidence(float(row["forecast"]), self.forecast_cap)
            if confidence <= 0.0:
                return FLAT

        return Intent(
            side=side,
            stop_distance=stop_distance,
            confidence=confidence,
            reason=f"mom={mom:+.3f} close_vs_ema={close - ema_now:+.5f}"
                   + (f" f={row['forecast']:+.2f}" if self.continuous else ""),
        )


class BuyAndHold(Strategy):
    """Control group: always long, wide stop. If S1 cannot beat this, S1 is noise.

    The stop multiple has to stay within reach of risk-based sizing. At 100x ATR
    the required position falls below the broker's minimum lot and the control
    never trades at all - which looks like a data problem and is really a
    misspecified benchmark.
    """

    name = "buy_and_hold"

    def __init__(self, atr_period: int = 14, atr_stop_multiple: float = 10.0) -> None:
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.warmup = atr_period + 2

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df, self.atr_period)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        atr_now = df.iloc[i]["atr"]
        if pd.isna(atr_now) or atr_now <= 0:
            return FLAT
        return Intent(side=Side.BUY, stop_distance=float(atr_now) * self.atr_stop_multiple)
