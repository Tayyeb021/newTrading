"""010: the published forms - monthly decisions, held between them."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from core.strategy import is_month_start  # noqa: E402
from core.types import Position, Side  # noqa: E402
from data.continuous import stitch  # noqa: E402
from strategies.carry import Carry  # noqa: E402
from strategies.tsmom import TSMOM  # noqa: E402

NOW = pd.Timestamp("2025-06-02", tz="UTC")


def _frame(n=600, drift=0.0008, seed=3, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="B", tz="UTC")
    px = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame({"ts": ts, "open": px, "high": px * 1.005, "low": px * 0.995, "close": px, "volume": 1.0})


def test_month_start_is_a_calendar_rule():
    df = _frame(60)
    starts = [i for i in range(len(df)) if is_month_start(df, i)]
    assert starts[0] == 0
    for i in starts[1:]:
        assert df["ts"].iloc[i].month != df["ts"].iloc[i - 1].month


def test_tsmom_decides_only_on_month_starts_and_never_reenters_between():
    s = TSMOM(lookback=60)
    df = s.prepare(_frame(drift=0.002))  # a strong uptrend
    decisions, holds, flats_with_no_position = 0, 0, 0
    for i in range(s.warmup, len(df)):
        if is_month_start(df, i):
            it = s.evaluate(df, i, None)
            assert it.side is Side.BUY and it.stop_distance == pytest.approx(df["atr"].iloc[i] * 4.0)
            decisions += 1
        else:
            assert s.evaluate(df, i, None).flat  # stopped out mid-month: wait
            flats_with_no_position += 1
            pos = Position("X", Side.BUY, 1.0, 100.0, NOW.to_pydatetime(), stop_loss=90.0)
            hold = s.evaluate(df, i, pos)
            assert hold.side is Side.BUY and hold.reason == "hold"
            holds += 1
    assert decisions >= 20 and holds > decisions * 10 and flats_with_no_position == holds


def test_tsmom_flips_with_the_sign_and_is_flat_on_zero():
    s = TSMOM(lookback=60)
    df = s.prepare(_frame(drift=-0.002))
    i = next(i for i in range(s.warmup, len(df)) if is_month_start(df, i))
    assert s.evaluate(df, i, None).side is Side.SELL
    df.loc[df.index[i], "mom"] = 0.0
    assert s.evaluate(df, i, None).flat
    assert TSMOM().rebalances is False and TSMOM(continuous=True).rebalances is True


def _expiry(start, end, level):
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"ts": pd.to_datetime(days, utc=True), "open": level, "high": level * 1.001,
                         "low": level * 0.999, "close": level, "volume": 1000.0})


def test_stitch_keeps_the_raw_front_close_beside_the_adjusted_one():
    mes = MICRO_UNIVERSE["MES"]
    exp = {(2025, 12): _expiry("2025-06-02", "2025-12-19", 6000.0),
           (2026, 3): _expiry("2025-09-01", "2026-03-20", 6060.0),
           (2026, 6): _expiry("2025-12-01", "2026-06-19", 6120.0)}
    cont, rolls = stitch(mes, exp, start=date(2025, 9, 15), end=date(2026, 4, 1))
    dec = cont[cont["contract"] == "MESZ5"]
    assert (dec["raw_close"] == 6000.0).all()
    assert np.allclose(dec["close"], 6000.0 + 60.0 + 60.0)  # shifted by the two roll gaps ahead


def test_carry_in_risk_units_needs_a_level_and_reads_the_right_sign():
    df = _frame(400)
    df["carry"] = 0.06          # six percent a year of backwardation
    df["raw_close"] = df["close"]
    s = Carry.published()
    assert s.rebalances is False and s.decide_monthly and s.normalise == "price_vol"
    p = s.prepare(df.copy())
    i = max(j for j in range(s.warmup, len(p)) if is_month_start(p, j))
    it = s.evaluate(p, i, None)
    assert it.side is Side.BUY and it.confidence == 1.0
    assert 0.1 < p["forecast"].iloc[i] < 1.0  # 6% carry over roughly 16% annual vol
    assert it.stop_distance == pytest.approx(p["atr"].iloc[i] * 4.0)

    weak = Carry.published().prepare(df.assign(carry=0.005).copy())  # half a percent: below 10% of vol
    assert Carry.published().evaluate(weak, i, None).flat

    no_level = Carry.published().prepare(df.drop(columns=["raw_close"]).copy())
    assert Carry.published().evaluate(no_level, i, None).flat

    # between decisions: hold what is held, open nothing
    j = next(j for j in range(i + 1, len(p)) if not is_month_start(p, j))
    assert s.evaluate(p, j, None).flat
    pos = Position("X", Side.BUY, 1.0, 100.0, NOW.to_pydatetime(), stop_loss=90.0)
    assert s.evaluate(p, j, pos).side is Side.BUY


def test_carry_rejects_unknown_normalisation():
    with pytest.raises(ValueError):
        Carry(normalise="magic")
