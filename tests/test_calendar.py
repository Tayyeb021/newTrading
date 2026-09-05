"""The contract calendar, checked against CME Group's published rules.

Dates below were read off the exchange specs by hand. If one of these fails
after an edit, the edit is wrong, not the exchange.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (  # noqa: E402
    ALL_ROOTS, FULL_UNIVERSE, MICRO_OF, MICRO_UNIVERSE, PARENT_OF, SECTORS,
    data_root, first_notice_date, last_trade_date, tradeable,
)


# ------------------------------------------------------------- expiry rules

def test_grains_expire_the_business_day_before_the_15th():
    assert last_trade_date("grains", 2025, 12) == date(2025, 12, 12)   # 15th is a Monday -> Friday 12th
    assert last_trade_date("grains", 2026, 3) == date(2026, 3, 13)     # 15th is a Sunday -> Friday 13th
    assert last_trade_date("grains", 2026, 5) == date(2026, 5, 14)     # 15th is a Friday -> Thursday 14th


def test_natgas_expires_third_to_last_business_day_of_prior_month():
    assert last_trade_date("natgas", 2026, 3) == date(2026, 2, 25)     # Feb 2026 ends Fri 27: 27, 26, 25
    assert last_trade_date("natgas", 2026, 4) == date(2026, 3, 27)     # Mar 2026 ends Tue 31: 31, 30, 27


def test_refined_products_expire_last_business_day_of_prior_month():
    assert last_trade_date("refined", 2026, 3) == date(2026, 2, 27)
    assert last_trade_date("refined", 2026, 1) == date(2025, 12, 31)


def test_short_treasuries_and_cattle_expire_last_business_day():
    assert last_trade_date("last_busday", 2026, 3) == date(2026, 3, 31)
    assert last_trade_date("last_busday", 2026, 5) == date(2026, 5, 29)  # 31st is a Sunday


def test_lean_hogs_expire_tenth_business_day():
    assert last_trade_date("lean_hogs", 2026, 2) == date(2026, 2, 13)
    assert last_trade_date("lean_hogs", 2026, 4) == date(2026, 4, 14)


def test_feeder_cattle_expire_last_thursday():
    assert last_trade_date("feeder", 2026, 1) == date(2026, 1, 29)
    assert last_trade_date("feeder", 2026, 3) == date(2026, 3, 26)


def test_metals_rule_is_the_gold_rule():
    assert last_trade_date("metals", 2026, 2) == last_trade_date("gold", 2026, 2) == date(2026, 2, 25)


def test_unknown_rules_raise():
    with pytest.raises(ValueError):
        last_trade_date("lottery", 2026, 1)
    with pytest.raises(ValueError):
        first_notice_date("whenever", 2026, 1)


# ------------------------------------------------------------ first notice

def test_first_notice_is_the_prior_month_end():
    assert first_notice_date("prior_month_end", 2026, 3) == date(2026, 2, 27)
    assert first_notice_date("prior_month_end", 2026, 1) == date(2025, 12, 31)


def test_physically_delivered_contracts_roll_before_first_notice():
    """The bug this guards: ZN used to roll off last trade, which is deep inside
    the delivery month -- after first position day. A long would already have
    been at risk of delivery."""
    zn = FULL_UNIVERSE["ZN"]
    assert zn.first_notice(2026, 3) == date(2026, 2, 27)
    assert zn.roll_date(2026, 3) < zn.first_notice(2026, 3) < zn.last_trade(2026, 3)
    assert zn.roll_date(2026, 3) == date(2026, 2, 24)  # three business days before Feb 27

    zc = FULL_UNIVERSE["ZC"]
    assert zc.roll_date(2026, 3) == date(2026, 2, 20)  # five business days before Feb 27
    assert zc.roll_date(2026, 3) < zc.first_notice(2026, 3)

    gc = FULL_UNIVERSE["GC"]
    assert gc.roll_date(2026, 2) < gc.first_notice(2026, 2) == date(2026, 1, 30)


def test_cash_settled_contracts_roll_off_last_trade():
    es = FULL_UNIVERSE["ES"]
    assert es.first_notice(2026, 3) is None
    assert es.roll_date(2026, 3) == date(2026, 3, 13)  # five business days before Fri Mar 20
    he = FULL_UNIVERSE["HE"]
    assert he.first_notice(2026, 2) is None and not he.physically_delivered


def test_energy_last_trade_precedes_delivery_so_no_first_notice():
    for r in ("CL", "NG", "RB", "HO"):
        root = FULL_UNIVERSE[r]
        assert root.first_notice_rule is None
        # every last trade falls before the contract month starts
        for y, m in root.listed(date(2024, 1, 1), date(2026, 12, 31)):
            assert root.last_trade(y, m) < date(y, m, 1), (r, y, m)


# ---------------------------------------------------------------- universe

def test_universe_has_seven_sectors_and_thirty_three_markets():
    assert len(FULL_UNIVERSE) == 33
    assert sorted(sum(SECTORS.values(), ())) == sorted(FULL_UNIVERSE)
    for sector, roots in SECTORS.items():
        for r in roots:
            assert FULL_UNIVERSE[r].bucket == sector


@pytest.mark.parametrize("name", sorted(ALL_ROOTS))
def test_every_root_has_a_sane_contiguous_schedule(name):
    root = ALL_ROOTS[name]
    assert root.tick_value > 0 and root.multiplier > 0 and root.tick_size > 0
    assert root.digits >= 0
    windows = root.schedule(date(2018, 1, 1), date(2026, 12, 31))
    assert len(windows) >= 8 * len(root.months) * 0.9, f"{name}: too few windows"
    for i, w in enumerate(windows):
        assert w.month in root.months
        # the first window opens on the requested start, which may itself be a roll day
        assert (w.active_from <= w.roll_on) if i == 0 else (w.active_from < w.roll_on), (name, w)
        assert w.roll_on < w.last_trade, (name, w)
        if w.first_notice is not None:
            assert w.roll_on < w.first_notice, (name, w)
    for a, b in zip(windows, windows[1:]):
        assert b.active_from == a.roll_on, (name, a, b)   # no gaps, no overlaps
        assert (b.year, b.month) > (a.year, a.month)


@pytest.mark.parametrize("name", sorted(ALL_ROOTS))
def test_front_is_never_past_its_roll(name):
    root = ALL_ROOTS[name]
    for day in (date(2025, 1, 2), date(2025, 6, 30), date(2025, 12, 31), date(2026, 2, 27)):
        y, m = root.front(day)
        assert root.roll_date(y, m) > day


def test_micro_map_is_consistent():
    for full, micro in MICRO_OF.items():
        assert full in FULL_UNIVERSE and micro in MICRO_UNIVERSE
        f, m = FULL_UNIVERSE[full], MICRO_UNIVERSE[micro]
        assert m.multiplier < f.multiplier
        assert m.months == f.months and m.expiry_rule == f.expiry_rule, (full, micro)
        assert m.physically_delivered == f.physically_delivered, (full, micro)
        assert PARENT_OF[micro] == full
    assert tradeable("ES").root == "MES" and tradeable("ZN").root == "ZN" and tradeable("MES").root == "MES"
    assert data_root("MES").root == "ES" and data_root("ES").root == "ES" and data_root("ZN").root == "ZN"


def test_tick_values_match_the_exchange():
    tv = {n: round(r.tick_value, 4) for n, r in ALL_ROOTS.items()}
    assert tv["ES"] == 12.5 and tv["NQ"] == 5.0 and tv["YM"] == 5.0 and tv["RTY"] == 5.0
    assert tv["ZT"] == 7.8125 and tv["ZF"] == 7.8125 and tv["ZN"] == 15.625 and tv["ZB"] == 31.25
    assert tv["6E"] == 6.25 and tv["6J"] == 6.25 and tv["6B"] == 6.25 and tv["6S"] == 12.5
    assert tv["GC"] == 10.0 and tv["SI"] == 25.0 and tv["HG"] == 12.5 and tv["PL"] == 5.0
    assert tv["CL"] == 10.0 and tv["NG"] == 10.0 and tv["RB"] == 4.2 and tv["HO"] == 4.2
    assert tv["ZC"] == 12.5 and tv["ZS"] == 12.5 and tv["ZM"] == 10.0 and tv["ZL"] == 6.0
    assert tv["LE"] == 10.0 and tv["HE"] == 10.0 and tv["GF"] == 12.5
    assert tv["MES"] == 1.25 and tv["MYM"] == 0.5 and tv["M2K"] == 0.5 and tv["SIL"] == 5.0
    assert tv["MHG"] == 1.25 and tv["M6A"] == 1.0 and tv["M6B"] == 0.625
