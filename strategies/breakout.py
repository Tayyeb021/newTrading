"""Donchian channel breakout, as the Turtles traded it. Research entry 014.

Enter long when the close exceeds the highest high of the previous N days,
short below the lowest low. Leave on the opposite M-day channel, M = N/2, or on
a 2-ATR stop, whichever comes first.

What is deliberately absent: a moving-average filter, a volatility filter, a
confirmation signal, a profit target. Every one of those would turn a rule with
published evidence into one without any, and would be a parameter chosen by me
rather than by the literature.

**Why this does not repeat the turnover failure of entries 007-009.** Those
rules re-decided every day and paid three dollars of friction for every dollar
of gross. A breakout decides twice per trade: once when the channel gives way,
once when the opposite channel does. Between those it holds, whatever the price
does in the middle. Low turnover is a property of the rule here, not a calendar
imposed on top of it.

The channel excludes the current bar - see `features.indicators.donchian` and
the test that asserts it. Including today's high in "the highest high" makes a
breakout trivially true at the moment it is tested, which is the most common
look-ahead bug in breakout backtests and the reason this rule so often looks
wonderful in a spreadsheet.
"""

from __future__ import annotations

import pandas as pd

from core.strategy import FLAT, Intent, Strategy
from core.types import Position, Side
from features.indicators import atr, donchian


class Breakout(Strategy):
    name = "breakout"

    def __init__(
        self,
        entry: int = 55,
        exit: int | None = None,
        atr_period: int = 14,
        atr_stop_multiple: float = 2.0,
    ) -> None:
        self.entry = entry
        #: The Turtles used half the entry length. Kept as a ratio rather than a
        #: free parameter, so the three speeds are one choice and not six.
        self.exit = exit if exit is not None else max(2, entry // 2)
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.warmup = max(entry, self.exit, atr_period) + 5

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df, self.atr_period)
        df["entry_hi"], df["entry_lo"] = donchian(df, self.entry)
        df["exit_hi"], df["exit_lo"] = donchian(df, self.exit)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        row = df.iloc[i]
        atr_now, close = row["atr"], row["close"]
        if pd.isna(atr_now) or atr_now <= 0:
            return FLAT
        stop = float(atr_now) * self.atr_stop_multiple

        if position is not None:
            # Hold until the OPPOSITE channel gives way. The exit channel is
            # shorter than the entry one, so a trade gives back less than it
            # took to get in - which is where a trend rule's asymmetry lives.
            if position.side is Side.BUY:
                out = pd.notna(row["exit_lo"]) and close < float(row["exit_lo"])
            else:
                out = pd.notna(row["exit_hi"]) and close > float(row["exit_hi"])
            if out:
                return FLAT
            return Intent(side=position.side, stop_distance=stop, reason="hold")

        hi, lo = row["entry_hi"], row["entry_lo"]
        if pd.notna(hi) and close > float(hi):
            return Intent(Side.BUY, stop_distance=stop,
                          reason=f"break {self.entry}d high {float(hi):.5g}")
        if pd.notna(lo) and close < float(lo):
            return Intent(Side.SELL, stop_distance=stop,
                          reason=f"break {self.entry}d low {float(lo):.5g}")
        return FLAT
