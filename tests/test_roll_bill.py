"""The roll bill the backtests never charged, and the arithmetic that prices it.

A continuous futures series has no expiries in it, so a position held across
one costs nothing to carry. In the market it is closed and reopened, crossing a
spread and paying commission both ways. `roll_cost_cash` has priced that since
phase 4 and no backtest has ever called it.

The first version of the measuring script reached for a `Roll.date` attribute
that does not exist, got an empty calendar, and reported a $0 bill - a bug that
reads as "there is no problem". These tests exist because the failure mode of a
cost measurement is silently measuring zero.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.contracts import FULL_UNIVERSE  # noqa: E402
from data.continuous import Roll, roll_cost_cash  # noqa: E402

_spec = importlib.util.spec_from_file_location("roll_bill", ROOT / "research" / "roll_bill.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules["roll_bill"] = rb
_spec.loader.exec_module(rb)


def test_roll_has_an_on_field_and_not_a_date_field():
    """The bug that produced a $0 bill. If Roll ever gains a `date` attribute,
    or `on` is renamed, the calendar builder must be revisited rather than
    silently returning nothing."""
    assert "on" in Roll.__dataclass_fields__
    assert "date" not in Roll.__dataclass_fields__
    assert "ts" not in Roll.__dataclass_fields__


def test_an_empty_calendar_prices_nothing_and_that_is_the_dangerous_case():
    """Documented rather than defended: with no roll dates the bill is zero and
    looks like good news. Any change here must keep the calendar non-empty for
    a real market - see the integration test below."""
    root = FULL_UNIVERSE["ZT"]
    assert roll_cost_cash(root, contracts=0) == 0.0


def test_roll_cost_charges_both_legs_of_both_kinds_of_friction():
    root = FULL_UNIVERSE["ES"]
    one = roll_cost_cash(root, contracts=1, spread_ticks=1.0)
    expected = 2 * root.commission_per_side + 2 * 1.0 * root.tick_value
    assert one == pytest.approx(expected)
    assert roll_cost_cash(root, contracts=10) == pytest.approx(10 * one), "linear in size"
    assert roll_cost_cash(root, 1, spread_ticks=2.0) > one, "a wider spread costs more"


def test_only_rolls_strictly_inside_a_holding_period_are_charged():
    """A roll on the entry day is paid by the entry, not by a carry. A roll on
    the exit day is crossed. Off-by-one here moves the bill by whole percent."""
    cal = [pd.Timestamp(d, tz="UTC") for d in
           ("2021-03-10", "2021-06-10", "2021-09-10", "2021-12-10")]
    entry = pd.Timestamp("2021-03-10", tz="UTC")
    exit_ = pd.Timestamp("2021-09-10", tz="UTC")
    crossed = sum(1 for d in cal if entry < d <= exit_)
    assert crossed == 2, "June and September; March is the entry day itself"


def test_a_position_held_for_years_crosses_many_rolls():
    """The case that matters: ZT's two big trades ran for two and three years
    on a quarterly contract and were charged nothing to carry."""
    cal = [pd.Timestamp(f"{y}-{m:02d}-10", tz="UTC")
           for y in (2021, 2022, 2023, 2024) for m in (3, 6, 9, 12)]
    entry = pd.Timestamp("2021-09-02", tz="UTC")
    exit_ = pd.Timestamp("2024-08-02", tz="UTC")
    crossed = sum(1 for d in cal if entry < d <= exit_)
    # Sep and Dec 2021, four in 2022, four in 2023, Mar and Jun 2024.
    assert crossed == 12
    bill = crossed * roll_cost_cash(FULL_UNIVERSE["ZT"], contracts=256, spread_ticks=1.0) * 2.0
    assert bill > 100_000, "twelve rolls on 256 lots is not a rounding error"


def test_the_calendar_builder_returns_real_dates_for_a_real_market():
    """The integration guard. If this returns an empty list the bill is zero and
    the tool reports no problem where there is one."""
    if not (ROOT / "data" / "futures" / "ZT").exists():
        pytest.skip("no ZT price data on this machine")
    cal = rb.roll_dates("ZT", ROOT / "data" / "futures", since=2011)
    assert len(cal) > 40, f"15 years of a quarterly contract should be ~60 rolls, got {len(cal)}"
    assert all(isinstance(d, pd.Timestamp) and d.tzinfo is not None for d in cal)
    assert cal == sorted(cal)
