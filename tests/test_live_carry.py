"""Carry, live: the front month against the next, fed to the rule at runtime.

In a backtest the curve arrives as a `carry` column written by the stitcher. A
live bar feed supplies only the front contract, so the carry rule saw nothing
and read flat - it was in the codebase, tested, and doing nothing. This wires
the same number from two live histories, so the rule sees an identical quantity
live and in research.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("ib_async")

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from execution.ib_adapter import IBAdapter  # noqa: E402
from execution.ib_fake import FakeIB, _HistBar  # noqa: E402
from execution.shadow import ShadowAdapter  # noqa: E402
from strategies.carry import Carry  # noqa: E402

MES = MICRO_UNIVERSE["MES"]
SEP, DEC = (2026, 9), (2026, 12)


def _adapter(front_px=7722.25, next_px=7785.25, days=40, today=date(2026, 8, 1), wobble=0.0):
    """`wobble` gives the series real price variation, which the risk-unit carry
    form needs: it divides by 63 days of price volatility, and a flat series has
    none. The basis between the two months stays constant regardless."""
    import numpy as np
    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(0, wobble, days)) if wobble else np.zeros(days)
    fake = FakeIB(MICRO_UNIVERSE, prices={"MES": front_px}, equity=1_000_000.0)
    start = datetime(2026, 3, 2, tzinfo=timezone.utc)
    for month, px in ((SEP, front_px), (DEC, next_px)):
        code = f"{MES.ib_month(*month)}"
        fake.history[code] = [
            _HistBar(start + timedelta(days=i), px + walk[i], px + walk[i] + 1,
                     px + walk[i] - 1, px + walk[i], 1000.0)
            for i in range(days)
        ]
    ad = IBAdapter(ib=fake, roots=MICRO_UNIVERSE, today=today)
    ad.connect()
    return ad, fake


def _patch_history_by_month(fake):
    """FakeIB keys history by symbol; key it by delivery month so the two
    contracts can carry different prices, as real ones do."""
    def req(c, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate=2):
        return list(fake.history.get(str(c.lastTradeDateOrContractMonth), []))
    fake.reqHistoricalData = req


def test_carry_is_computed_from_two_delivery_months():
    ad, fake = _adapter()
    _patch_history_by_month(fake)
    series = ad.carry_series("MES", 40)
    assert series, "the curve should produce a carry value per overlapping day"

    days = (MES.last_trade(*DEC) - MES.last_trade(*SEP)).days
    expected = (7722.25 - 7785.25) / 7722.25 * (365.0 / days)
    assert all(v == pytest.approx(expected, rel=1e-9) for v in series.values())
    assert expected < 0, "the next month priced higher is contango, negative for a long"
    assert abs(expected) < 1.0, "an annualised roll yield in the tens of percent would be a unit error"


def test_backwardation_gives_positive_carry():
    ad, fake = _adapter(front_px=7800.0, next_px=7750.0)
    _patch_history_by_month(fake)
    assert all(v > 0 for v in ad.carry_series("MES", 40).values())


def test_bar_extras_align_with_the_bars_the_rule_will_see():
    ad, fake = _adapter()
    _patch_history_by_month(fake)
    bars = ad.bars("MES", "D1", 30)
    extras = ad.bar_extras("MES", "D1", 30)
    assert set(extras) == {"carry", "raw_close"}
    assert len(extras["carry"]) == len(bars) and len(extras["raw_close"]) == len(bars)
    assert extras["raw_close"][-1] == bars[-1].close


def test_no_curve_means_no_extras_rather_than_a_fabricated_number():
    ad, fake = _adapter()
    fake.history.clear()  # the deferred month has no history
    fake.reqHistoricalData = lambda *a, **k: []
    assert ad.carry_series("MES", 30) == {}
    assert ad.bar_extras("MES", "D1", 30) == {}


def test_intraday_timeframes_get_no_extras():
    ad, fake = _adapter()
    _patch_history_by_month(fake)
    assert ad.bar_extras("MES", "M15", 30) == {}


def test_the_shadow_adapter_relays_the_curve():
    """A rule must behave identically whether fills are shadowed or real."""
    ad, fake = _adapter()
    _patch_history_by_month(fake)
    specs = {"MES": ad.spec("MES")}
    shadow = ShadowAdapter(ad, specs)
    assert shadow.bar_extras("MES", "D1", 30) == ad.bar_extras("MES", "D1", 30)


def test_the_rule_reads_the_relayed_carry_and_takes_a_side():
    import pandas as pd

    # 200 bars with real variation: the risk-unit form divides by 63 days of
    # price volatility, so a short flat series can only produce NaN.
    ad, fake = _adapter(front_px=7800.0, next_px=7750.0, days=200, wobble=12.0)
    _patch_history_by_month(fake)
    bars = ad.bars("MES", "D1", 200)
    extras = ad.bar_extras("MES", "D1", 200)
    df = pd.DataFrame({
        "ts": pd.to_datetime([b.ts for b in bars], utc=True),
        "open": [b.open for b in bars], "high": [b.high for b in bars],
        "low": [b.low for b in bars], "close": [b.close for b in bars],
        "volume": [b.volume for b in bars], **extras,
    })
    rule = Carry.published()
    prepared = rule.prepare(df)
    assert prepared["carry"].notna().any(), "the rule must see a curve, not NaN"
    assert prepared["forecast"].notna().any(), "and turn it into a forecast"
