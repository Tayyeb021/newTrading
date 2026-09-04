"""Top-down: higher timeframe decides direction, M5/M15 decides when.

Three conditions, in this order, and all three must hold:

1. **BIAS** -- from H1/H4/D1. Which way, and only that way. Read from bars that
   have already closed (see `features.mtf`); using the forming higher-timeframe
   bar is the classic look-ahead that makes this whole family of strategies look
   far better than it is.

2. **LOCATION** -- price must have pulled back into value, measured as distance
   from the execution-timeframe EMA in ATR units. Entering an extended move is
   how a trend follower buys the exact top: the stop has to sit far away, the
   reward-to-risk collapses, and the first pullback stops you out.

3. **TRIGGER** -- momentum resumes in the direction of bias. Without it, "pulled
   back into value" is indistinguishable from "falling", and you catch the move
   that keeps going.

Two intraday rules the broker's own numbers force, not preferences:

- **Session filter.** Outside London and New York you pay a wider spread for less
  movement.
- **Flat before rollover.** XAUUSDc charges -512.30 cents per lot per night on
  longs. The M5 ATR on gold is around 400 cents. One night of financing costs
  more than the average five-minute range moves, so an intraday system that holds
  overnight gives back more than it can make.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent, Strategy
from core.types import Position, Side
from features.indicators import atr, ema
from features.mtf import align_higher_timeframe, bars_until, session_mask


def structure_frame(htf: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Higher-timeframe ATR, for sizing the trade to the STRUCTURE.

    The correction that makes top-down trading work. If an "M5 trade" is stopped
    at 1.5x the M5 ATR, it is targeting a five-minute-sized move -- about 31
    points on EURUSD against 16 points of friction, which is a coin flip paying a
    50% toll. Nobody wins that, however good the entry.

    Entering on M5 is about *precision*, not about holding for five minutes. The
    move being traded belongs to H1 or H4, and so does the stop. Sizing from H1
    takes friction from 51% of the stop down to 11%.
    """
    out = pd.DataFrame({"ts": pd.to_datetime(htf["ts"], utc=True)})
    out["atr"] = atr(htf, atr_period)
    return out


def bias_frame(htf: pd.DataFrame, ema_period: int = 50, slope_bars: int = 5) -> pd.DataFrame:
    """Reduce a higher-timeframe series to a single direction column.

    Deliberately crude: a bias model with many conditions is a bias model fitted
    to the sample. Trend of the EMA plus price on the right side of it.
    """
    out = pd.DataFrame({"ts": pd.to_datetime(htf["ts"], utc=True)})
    e = ema(htf["close"], ema_period)
    slope = e.diff(slope_bars)

    bullish = (htf["close"] > e) & (slope > 0)
    bearish = (htf["close"] < e) & (slope < 0)
    out["bias"] = np.select([bullish, bearish], [1, -1], default=0)
    out["ema"] = e
    return out


class MTFPullback(Strategy):
    name = "MTF_pullback"

    def __init__(
        self,
        execution_timeframe: str = "M15",
        bias_frames: dict[str, pd.DataFrame] | None = None,
        bias_timeframes: tuple[str, ...] = ("H4", "H1"),
        # location
        ema_period: int = 20,
        max_extension_atr: float = 1.0,   # entry must be within this many ATR of the EMA
        min_pullback_atr: float = 0.25,   # ... but must have actually pulled back
        # risk
        atr_period: int = 14,
        atr_stop_multiple: float = 1.5,
        #: Timeframe the STOP is sized from. None means the execution frame,
        #: which turns a precision entry into a five-minute trade -- see
        #: `structure_frame`. Set this to H1 or H4 for genuine top-down trading.
        stop_timeframe: str | None = None,
        reward_multiple: float = 2.0,
        # session, UTC
        session_start: int = 7,
        session_end: int = 20,
        flat_by_hour: int = 20,
        min_bars_before_flat: int = 4,
        require_all_bias_agree: bool = True,
    ) -> None:
        self.execution_timeframe = execution_timeframe
        self.bias_frames = bias_frames or {}
        self.bias_timeframes = bias_timeframes
        self.ema_period = ema_period
        self.max_extension_atr = max_extension_atr
        self.min_pullback_atr = min_pullback_atr
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.stop_timeframe = stop_timeframe
        self.reward_multiple = reward_multiple
        self.session_start = session_start
        self.session_end = session_end
        self.flat_by_hour = flat_by_hour
        self.min_bars_before_flat = min_bars_before_flat
        self.require_all_bias_agree = require_all_bias_agree
        self.warmup = max(ema_period, atr_period) * 3

    # ------------------------------------------------------------------ prepare

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        df["atr"] = atr(df, self.atr_period)
        df["ema"] = ema(df["close"], self.ema_period)
        df["ext_atr"] = (df["close"] - df["ema"]) / df["atr"]

        # Did price actually come back to the EMA recently, or has it simply sat
        # near it? A pullback needs a prior excursion.
        df["ext_max_5"] = df["ext_atr"].rolling(5, min_periods=1).max()
        df["ext_min_5"] = df["ext_atr"].rolling(5, min_periods=1).min()

        df["in_session"] = session_mask(df["ts"], self.session_start, self.session_end)
        df["bars_to_flat"] = bars_until(df["ts"], self.flat_by_hour, self.execution_timeframe)

        # Stop sized from the structure timeframe, not the execution one.
        if self.stop_timeframe:
            htf = self.bias_frames.get(self.stop_timeframe)
            if htf is not None and not htf.empty:
                df = align_higher_timeframe(
                    df, structure_frame(htf, self.atr_period),
                    self.stop_timeframe, columns=["atr"], prefix="stop",
                )
                df["stop_atr"] = df["stop_atr"].ffill()
            else:
                df["stop_atr"] = df["atr"]
        else:
            df["stop_atr"] = df["atr"]

        # Higher-timeframe bias, joined on CLOSE time so nothing forming leaks.
        cols = []
        for tf in self.bias_timeframes:
            htf = self.bias_frames.get(tf)
            if htf is None or htf.empty:
                continue
            df = align_higher_timeframe(df, bias_frame(htf), tf, columns=["bias"], prefix=tf)
            cols.append(f"{tf.lower()}_bias")

        if cols:
            biases = df[cols].fillna(0)
            if self.require_all_bias_agree:
                agree = (biases.eq(1).all(axis=1)).astype(int) - (biases.eq(-1).all(axis=1)).astype(int)
            else:
                agree = np.sign(biases.sum(axis=1)).astype(int)
            df["bias"] = agree
        else:
            df["bias"] = 0
        return df

    # ----------------------------------------------------------------- evaluate

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        row = df.iloc[i]

        atr_now = row["stop_atr"]
        if not np.isfinite(atr_now) or atr_now <= 0:
            return FLAT
        # Location and trigger still read the EXECUTION frame's ATR; only the
        # stop is sized from the structure.
        exec_atr = row["atr"]
        if not np.isfinite(exec_atr) or exec_atr <= 0:
            return FLAT

        # --- exits first: they must fire even when entry conditions do not ----
        if position is not None:
            # Close before rollover. On gold this rule is worth more than the
            # entry logic - see the module docstring.
            if row["bars_to_flat"] <= self.min_bars_before_flat:
                return FLAT
            if row["bias"] != position.side.sign:
                return FLAT  # higher timeframe turned against the trade
            return Intent(
                side=position.side,
                stop_distance=float(atr_now) * self.atr_stop_multiple,
                reason="hold",
            )

        # --- 1. bias ----------------------------------------------------------
        bias = int(row["bias"])
        if bias == 0:
            return FLAT
        side = Side.BUY if bias > 0 else Side.SELL

        # --- session and time-of-day -----------------------------------------
        if not bool(row["in_session"]):
            return FLAT
        if row["bars_to_flat"] <= self.min_bars_before_flat * 2:
            return FLAT  # not enough runway left to be worth opening

        # --- 2. location: pulled back into value, not extended ----------------
        ext = row["ext_atr"]
        if not np.isfinite(ext):
            return FLAT
        if abs(ext) > self.max_extension_atr:
            return FLAT  # chasing

        if side is Side.BUY:
            pulled_back = row["ext_min_5"] <= -self.min_pullback_atr or ext < 0
        else:
            pulled_back = row["ext_max_5"] >= self.min_pullback_atr or ext > 0
        if not pulled_back:
            return FLAT

        # --- 3. trigger: momentum resuming in the bias direction --------------
        prev = df.iloc[i - 1]
        if side is Side.BUY:
            resuming = row["close"] > row["open"] and row["close"] > prev["high"]
        else:
            resuming = row["close"] < row["open"] and row["close"] < prev["low"]
        if not resuming:
            return FLAT

        return Intent(
            side=side,
            stop_distance=float(atr_now) * self.atr_stop_multiple,
            reason=f"bias={bias} ext={ext:+.2f}ATR",
        )


def load_bias_frames(
    store,
    symbol: str,
    timeframes: tuple[str, ...] = ("H4", "H1"),
) -> dict[str, pd.DataFrame]:
    """Read the higher-timeframe series a strategy needs from the bar store."""
    out: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        df = store.read(symbol, tf)
        if not df.empty:
            out[tf] = df
    return out
