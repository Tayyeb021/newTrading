"""009: carry read off the futures curve, and the strategy that trades it."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from core.types import Side  # noqa: E402
from data.continuous import stitch  # noqa: E402
from strategies.carry import Carry  # noqa: E402


def _expiry(start, end, level):
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"ts": pd.to_datetime(days, utc=True), "open": level, "high": level * 1.001,
                         "low": level * 0.999, "close": level, "volume": 1000.0})


def test_stitch_reads_carry_off_the_curve():
    mes = MICRO_UNIVERSE["MES"]
    # Dec at 6000, Mar at 6060, Jun at 6120: contango, a long pays 1% a quarter
    exp = {
        (2025, 12): _expiry("2025-06-02", "2025-12-19", 6000.0),
        (2026, 3): _expiry("2025-09-01", "2026-03-20", 6060.0),
        (2026, 6): _expiry("2025-12-01", "2026-06-19", 6120.0),
    }
    cont, rolls = stitch(mes, exp, start=date(2025, 9, 15), end=date(2026, 4, 1))
    assert "carry" in cont.columns

    dec = cont[cont["contract"] == "MESZ5"]
    days = (mes.last_trade(2026, 3) - mes.last_trade(2025, 12)).days
    expected = (6000.0 - 6060.0) / 6000.0 * 365.0 / days
    assert expected < 0, "contango is negative carry for a long"
    assert dec["carry"].notna().all()
    assert dec["carry"].iloc[-1] == pytest.approx(expected, rel=1e-9)

    mar = cont[cont["contract"] == "MESH6"]
    assert mar["carry"].notna().all()
    assert mar["carry"].iloc[0] == pytest.approx((6060.0 - 6120.0) / 6060.0 * 365.0
                                                 / (mes.last_trade(2026, 6) - mes.last_trade(2026, 3)).days)
    jun = cont[cont["contract"] == "MESM6"]
    assert jun["carry"].isna().all(), "the last contract has no next to measure against"
    # the back-adjustment moved prices, not carry: the roll gap is still logged
    assert [round(r.gap) for r in rolls] == [60, 60]


def _frame(n=400, carry_level=0.05, noise=0.01, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    px = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({"ts": ts, "open": px, "high": px + 0.5, "low": px - 0.5, "close": px,
                         "volume": 1.0, "carry": carry_level + rng.normal(0, noise, n)})


def test_carry_goes_long_backwardation_short_contango_and_flat_without_a_view():
    long_side = Carry()
    df = long_side.prepare(_frame(carry_level=+0.05))
    it = long_side.evaluate(df, len(df) - 1, None)
    assert it.side is Side.BUY and it.confidence == 1.0  # five sigma of carry: full size
    assert it.stop_distance > 0 and "carry=" in it.reason

    df = Carry().prepare(_frame(carry_level=-0.05))
    assert Carry().evaluate(df, len(df) - 1, None).side is Side.SELL

    # Noise around zero: mostly flat, and never conviction. A 20-day smoothing
    # of white noise keeps about a quarter of a sigma, so a 0.25 threshold is
    # crossed a minority of the time and only ever at the floor-sized end.
    df = Carry().prepare(_frame(carry_level=0.0))
    intents = [Carry().evaluate(df, i, None) for i in range(Carry().warmup, len(df))]
    assert sum(it.flat for it in intents) > 0.5 * len(intents)
    assert max((it.confidence for it in intents if not it.flat), default=0.0) < 0.5
    assert {it.side for it in intents if not it.flat} == {Side.BUY, Side.SELL}

    df = Carry().prepare(_frame().drop(columns=["carry"]))  # no curve at all
    assert all(Carry().evaluate(df, i, None).flat for i in range(len(df)))


def test_carry_is_continuous_by_default():
    assert Carry().rebalances is True
    assert Carry(continuous=False).rebalances is False
    df = Carry(continuous=False).prepare(_frame(carry_level=0.006, noise=0.01))  # ~0.6 sigma
    it = Carry(continuous=False).evaluate(df, len(df) - 1, None)
    assert it.flat or it.confidence == 1.0
    df = Carry().prepare(_frame(carry_level=0.006, noise=0.01))
    it = Carry().evaluate(df, len(df) - 1, None)
    assert it.flat or 0.25 <= it.confidence < 1.0
