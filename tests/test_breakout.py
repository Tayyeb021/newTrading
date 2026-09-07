"""014: the Donchian breakout rule."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Position, Side  # noqa: E402
from strategies.breakout import Breakout  # noqa: E402

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _frame(closes, highs=None, lows=None):
    n = len(closes)
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    return pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC"),
        "open": closes, "high": highs, "low": lows, "close": closes, "volume": 1.0,
    })


def _held(side=Side.BUY):
    return Position("X", side, 1.0, 100.0, NOW, stop_loss=90.0)


def test_the_exit_channel_is_half_the_entry_channel():
    assert Breakout(entry=20).exit == 10
    assert Breakout(entry=55).exit == 27
    assert Breakout(entry=100).exit == 50
    assert Breakout(entry=20, exit=5).exit == 5, "an explicit exit still wins"


def test_a_break_above_the_channel_goes_long():
    closes = [100.0] * 25 + [110.0]
    b = Breakout(entry=20, atr_period=5)
    df = b.prepare(_frame(closes))
    it = b.evaluate(df, len(df) - 1, None)
    assert it.side is Side.BUY and "break 20d high" in it.reason
    assert it.stop_distance == pytest.approx(df["atr"].iloc[-1] * 2.0)


def test_a_break_below_the_channel_goes_short():
    closes = [100.0] * 25 + [90.0]
    b = Breakout(entry=20, atr_period=5)
    df = b.prepare(_frame(closes))
    assert b.evaluate(df, len(df) - 1, None).side is Side.SELL


def test_inside_the_channel_is_flat():
    rng = np.random.default_rng(3)
    closes = list(100 + rng.normal(0, 0.2, 60))
    b = Breakout(entry=20, atr_period=5)
    df = b.prepare(_frame(closes))
    flats = sum(b.evaluate(df, i, None).flat for i in range(b.warmup, len(df)))
    assert flats > 0.7 * (len(df) - b.warmup), "noise inside the range must not trade"


def test_a_position_is_held_until_the_opposite_channel_gives_way():
    # rise, then drift back down through the 10-day low
    closes = [100.0] * 25 + [110.0] * 12 + [104.0]
    b = Breakout(entry=20, atr_period=5)
    df = b.prepare(_frame(closes))

    mid = b.evaluate(df, 30, _held())
    assert mid.side is Side.BUY and mid.reason == "hold", "no re-deciding between the two channels"
    assert b.evaluate(df, len(df) - 1, _held()).flat, "closing under the exit channel leaves"


def test_the_current_bar_cannot_break_its_own_channel():
    """The classic look-ahead: if today's high counts toward 'the highest high',
    every bar breaks out of itself."""
    closes = list(range(100, 140))  # a clean uptrend, every bar a new high
    b = Breakout(entry=20, atr_period=5)
    df = b.prepare(_frame(closes))
    i = len(df) - 1
    assert df["entry_hi"].iloc[i] < df["high"].iloc[i], "the channel must exclude today"


def test_turnover_is_low_by_construction():
    """The failure of entries 007-009 was friction. A breakout must decide twice
    per trade, not twice per day."""
    rng = np.random.default_rng(7)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 800))))
    b = Breakout(entry=55, atr_period=14)
    df = b.prepare(_frame(closes))

    position, changes = None, 0
    for i in range(b.warmup, len(df)):
        it = b.evaluate(df, i, position)
        want = None if it.flat else it.side
        now = position.side if position else None
        if want != now:
            changes += 1
            position = _held(want) if want else None
    years = len(df) / 252
    assert changes / years < 12, f"{changes / years:.1f} side changes a year is not a breakout rule"
