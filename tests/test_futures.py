"""Tests for the futures side: calendar, stitching, the IB adapter, COT, order flow.

Every expiry date below is checked against the CME/COMEX/NYMEX rule, on dates
where the weekend-only calendar and the real one agree. Rolls, the thing that
silently corrupts futures backtests, get the most attention.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import MICRO_UNIVERSE, FuturesRoot, last_trade_date  # noqa: E402
from core.types import OrderRequest, Side  # noqa: E402
from data.continuous import Roll, roll_cost_cash, stitch  # noqa: E402
from data.cot import features, join_to_bars, parse_legacy, select_market  # noqa: E402
from execution.ib_adapter import IBAdapter  # noqa: E402
from execution.ib_fake import FakeIB  # noqa: E402
from execution.oms import OrderManager, client_id  # noqa: E402
from features.orderflow import delta_bars, imbalance, infer_side, open_imbalance  # noqa: E402

MES = MICRO_UNIVERSE["MES"]
MGC = MICRO_UNIVERSE["MGC"]
M6E = MICRO_UNIVERSE["M6E"]
MCL = MICRO_UNIVERSE["MCL"]


# ======================================================================
# Calendar
# ======================================================================

def test_index_last_trade_is_third_friday():
    assert last_trade_date("third_friday", 2025, 12) == date(2025, 12, 19)
    assert last_trade_date("third_friday", 2026, 3) == date(2026, 3, 20)
    assert last_trade_date("third_friday", 2026, 6) == date(2026, 6, 19)


def test_gold_last_trade_is_third_to_last_business_day():
    # Dec 2025: last business day Wed 31st -> 29th.
    assert last_trade_date("gold", 2025, 12) == date(2025, 12, 29)
    # Feb 2026: last business day Fri 27th -> Wed 25th.
    assert last_trade_date("gold", 2026, 2) == date(2026, 2, 25)


def test_fx_last_trade_is_two_business_days_before_third_wednesday():
    # Dec 2025: third Wednesday is the 17th -> Monday 15th.
    assert last_trade_date("fx", 2025, 12) == date(2025, 12, 15)
    # Mar 2026: third Wednesday is the 18th -> Monday 16th.
    assert last_trade_date("fx", 2026, 3) == date(2026, 3, 16)


def test_crude_last_trade_counts_back_from_the_25th_of_prior_month():
    # Feb 2026 contract: 25 Jan 2026 is a Sunday -> Fri 23rd -> three business days back = Tue 20th.
    assert last_trade_date("crude", 2026, 2) == date(2026, 1, 20)


def test_front_month_flips_on_the_roll_date():
    roll = MES.roll_date(2025, 12)  # five business days before Dec 19 -> Dec 12
    assert roll == date(2025, 12, 12)
    assert MES.front(roll - timedelta(days=1)) == (2025, 12)
    assert MES.front(roll) == (2026, 3), "on the roll date you are already in the next contract"


def test_gold_front_skips_illiquid_months():
    assert MGC.months == (2, 4, 6, 8, 10, 12)
    assert MGC.front(date(2026, 1, 15)) == (2026, 2)


def test_ticker_codes():
    assert MES.code(2025, 12) == "MESZ5"
    assert MGC.code(2026, 2) == "MGCG6"
    assert MES.ib_month(2026, 3) == "202603"


def test_spec_from_root_has_no_swap_and_whole_contracts():
    s = MES.to_spec()
    assert s.tick_value == pytest.approx(1.25)
    assert s.value_per_price_unit == pytest.approx(5.0)
    assert s.volume_min == 1.0 and s.volume_step == 1.0
    assert s.swap_long == 0.0 and s.swap_mode == 0
    assert M6E.to_spec().value_per_price_unit == pytest.approx(12_500.0)
    assert MGC.to_spec().tick_value == pytest.approx(1.0)


def test_schedule_covers_a_span_contiguously():
    windows = MES.schedule(date(2025, 1, 1), date(2025, 12, 31))
    assert [(w.year, w.month) for w in windows][:4] == [(2025, 3), (2025, 6), (2025, 9), (2025, 12)]
    for a, b in zip(windows, windows[1:]):
        assert b.active_from == a.roll_on, "windows must abut at the roll date"


# ======================================================================
# Continuous contracts
# ======================================================================

def _expiry(root: FuturesRoot, year: int, month: int, level: float, start: date, end: date, drift=0.0):
    days = pd.bdate_range(start, end)
    px = level + drift * np.arange(len(days))
    return pd.DataFrame({
        "ts": pd.to_datetime(days, utc=True), "open": px, "high": px + 1, "low": px - 1,
        "close": px, "volume": 100.0,
    })


def test_stitch_removes_the_roll_gap():
    """Two contracts, next priced 10 above front (contango). After stitching
    the series must not jump at the roll, and the roll must be logged."""
    roll = MES.roll_date(2025, 12)
    front = _expiry(MES, 2025, 12, 6000.0, date(2025, 9, 15), date(2025, 12, 19))
    nxt = _expiry(MES, 2026, 3, 6010.0, date(2025, 11, 1), date(2026, 3, 20))

    cont, rolls = stitch(MES, {(2025, 12): front, (2026, 3): nxt},
                         start=date(2025, 10, 1), end=date(2026, 2, 1))
    assert len(rolls) == 1
    assert rolls[0].on == roll
    assert rolls[0].gap == pytest.approx(10.0)
    assert rolls[0].from_contract == "MESZ5" and rolls[0].to_contract == "MESH6"

    cont["day"] = cont["ts"].dt.date
    before = cont[cont["day"] < roll]["close"].iloc[-1]
    after = cont[cont["day"] >= roll]["close"].iloc[0]
    assert abs(after - before) < 1e-9, f"jump at roll: {before} -> {after}"
    # Earlier prices were shifted UP by the gap so returns are preserved.
    assert cont[cont["day"] < roll]["close"].iloc[0] == pytest.approx(6010.0)
    assert (cont[cont["day"] >= roll]["contract"] == "MESH6").all()


def test_stitch_preserves_returns_not_levels():
    front = _expiry(MES, 2025, 12, 6000.0, date(2025, 9, 15), date(2025, 12, 19), drift=1.0)
    nxt = _expiry(MES, 2026, 3, 6100.0, date(2025, 11, 1), date(2026, 3, 20), drift=1.0)
    cont, _ = stitch(MES, {(2025, 12): front, (2026, 3): nxt}, start=date(2025, 10, 1), end=date(2026, 2, 1))
    diffs = cont["close"].diff().dropna()
    assert diffs.abs().max() < 1.5, "a day-to-day change larger than the drift means a roll leaked"


def test_roll_cost_is_two_round_trips_of_friction():
    assert roll_cost_cash(MES, 2, spread_ticks=1.0) == pytest.approx(2 * 0.85 * 2 + 2 * 1.25 * 2)


# ======================================================================
# IB adapter through the fake
# ======================================================================

TODAY = date(2025, 12, 1)


def make_adapter(**fake_kw) -> tuple[IBAdapter, FakeIB]:
    # The fake's clock is real "now" so connect()'s drift check passes; which
    # contract month is "front" is driven by the adapter's injected `today`.
    fake = FakeIB(MICRO_UNIVERSE, prices={"MES": 6000.0, "MGC": 2650.0, "M6E": 1.0800},
                  now=datetime.now(timezone.utc), **fake_kw)
    ad = IBAdapter(ib=fake, today=TODAY, fill_timeout=1.0)
    ad.connect()
    return ad, fake


def test_connect_verifies_the_clock():
    ad, fake = make_adapter()
    assert "drift" in ad.clock_message
    fake.now = datetime.now(timezone.utc) + timedelta(hours=3)
    bad = IBAdapter(ib=fake, today=TODAY)
    with pytest.raises(Exception, match="server time"):
        bad.connect()


def test_spec_comes_from_the_exchange():
    ad, _ = make_adapter()
    s = ad.spec("MES")
    assert s.tick_size == 0.25 and s.tick_value == pytest.approx(1.25)
    assert s.value_per_price_unit == pytest.approx(5.0)
    assert s.swap_long == 0.0


def test_root_resolves_to_the_front_contract():
    ad, _ = make_adapter()
    c = ad.contract("MES")
    assert c.lastTradeDateOrContractMonth == "202512"
    ad2 = IBAdapter(ib=make_adapter()[1], today=date(2025, 12, 15))
    ad2.connect()
    assert ad2.contract("MES").lastTradeDateOrContractMonth == "202603"


def test_submit_attaches_a_stop_and_positions_show_it():
    ad, fake = make_adapter()
    res = ad.submit(OrderRequest("MES", Side.BUY, 2, stop_loss=5980.0, comment="trend#abc"))
    assert res.ok and res.filled_volume == 2
    assert res.fill_price > 6000.0, "a market buy must fill at the ask"

    pos = ad.positions("MES")
    assert len(pos) == 1
    assert pos[0].side is Side.BUY and pos[0].volume == 2
    assert pos[0].stop_loss == pytest.approx(5980.0)
    assert pos[0].comment == "trend#abc"


def test_modify_moves_the_stop():
    ad, _ = make_adapter()
    res = ad.submit(OrderRequest("MES", Side.BUY, 1, stop_loss=5980.0, comment="t#1"))
    ad.modify(res.ticket, stop_loss=5990.0)
    assert ad.positions("MES")[0].stop_loss == pytest.approx(5990.0)


def test_close_flattens_and_cancels_the_stop():
    ad, fake = make_adapter()
    res = ad.submit(OrderRequest("MES", Side.SELL, 3, stop_loss=6020.0, comment="t#2"))
    out = ad.close(res.ticket)
    assert out.ok and out.filled_volume == 3
    assert ad.positions("MES") == []
    assert not fake.openTrades(), "the child stop must be cancelled on close"


def test_roll_moves_the_position_to_the_next_contract():
    ad, fake = make_adapter()
    res = ad.submit(OrderRequest("MES", Side.BUY, 2, stop_loss=5980.0, comment="trend#r"))
    assert not ad.roll_due("MES")

    ad._today = MES.roll_date(2025, 12)  # the roll date arrives
    assert ad.roll_due("MES")
    results = ad.roll("MES")
    assert all(r.ok for r in results) and len(results) == 2

    keys = list(fake._positions)
    assert keys == ["MES:202603"], f"position should now sit in the March contract, got {keys}"
    pos = ad.positions("MES")[0]
    assert pos.side is Side.BUY and pos.volume == 2 and pos.stop_loss == pytest.approx(5980.0)


def test_oms_idempotency_holds_on_ib():
    """The lost-reply case, on the futures adapter: one position, not two."""
    ad, fake = make_adapter()
    fake.fail_next_fill = True
    oms = OrderManager(ad, sleep=lambda _s: None)
    cid = client_id("trend", "MES", "BUY", datetime(2025, 12, 1, tzinfo=timezone.utc))
    res = oms.submit(OrderRequest("MES", Side.BUY, 1, stop_loss=5980.0), cid)
    assert res.ok
    assert sum(abs(p.position) for p in fake.positions()) == 1


def test_bars_are_utc_aware():
    from execution.ib_fake import _HistBar
    ad, fake = make_adapter()
    fake.history["MES"] = [_HistBar(datetime(2025, 11, 28, tzinfo=timezone.utc), 5990, 6010, 5980, 6000, 1000)]
    bars = ad.bars("MES", "D1", count=5)
    assert bars and bars[0].ts.tzinfo is not None


# ======================================================================
# COT
# ======================================================================

COT_CSV = """Market and Exchange Names,As of Date in Form YYYY-MM-DD,Open Interest (All),Noncommercial Positions-Long (All),Noncommercial Positions-Short (All),Noncommercial Positions-Spreading (All),Commercial Positions-Long (All),Commercial Positions-Short (All),Nonreportable Positions-Long (All),Nonreportable Positions-Short (All)
GOLD - COMMODITY EXCHANGE INC.,2026-08-04,500000,300000,80000,50000,100000,350000,50000,20000
GOLD - COMMODITY EXCHANGE INC.,2026-08-11,510000,310000,75000,50000,95000,360000,55000,25000
E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE,2026-08-04,2000000,600000,700000,100000,900000,800000,400000,400000
GOLD (SPREAD) - COMMODITY EXCHANGE INC.,2026-08-04,1,0,0,0,0,0,0,0
"""


def test_cot_parse_and_market_selection():
    df = parse_legacy(COT_CSV)
    gold, name = select_market(df, "GC")
    assert name == "GOLD - COMMODITY EXCHANGE INC.", "must pick the base contract, not the spread row"
    assert len(gold) == 2
    es, _ = select_market(df, "ES")
    assert len(es) == 1


def test_cot_features_are_causal_and_stamped_after_release():
    gold, _ = select_market(parse_legacy(COT_CSV), "GC")
    f = features(gold, lookback_weeks=2)
    assert f["spec_net"].tolist() == [220000, 235000]
    # As-of Tuesday 2026-08-04; published Friday 8/7; available Friday 21:00 UTC.
    assert f["available_at"].iloc[0] == pd.Timestamp("2026-08-07 21:00", tz="UTC")


def test_cot_join_has_no_lookahead():
    gold, _ = select_market(parse_legacy(COT_CSV), "GC")
    cot = features(gold, lookback_weeks=2)
    bars = pd.DataFrame({
        "ts": pd.to_datetime(["2026-08-05 00:00", "2026-08-07 20:00", "2026-08-10 00:00", "2026-08-13 00:00"], utc=True),
        "close": [2650.0, 2655.0, 2660.0, 2665.0],
    })
    joined = join_to_bars(bars, cot)
    # Wed 8/5 and Fri 8/7 20:00: the 8/4 report is NOT yet public -> no row.
    assert pd.isna(joined["cot_as_of"].iloc[0]) and pd.isna(joined["cot_as_of"].iloc[1])
    # Mon 8/10: the 8/4 report is public; the 8/11 report does not exist yet.
    assert joined["cot_as_of"].iloc[2] == pd.Timestamp("2026-08-04")
    # Thu 8/13: still only the 8/4 report (8/11's publishes Fri 8/14).
    assert joined["cot_as_of"].iloc[3] == pd.Timestamp("2026-08-04")


# ======================================================================
# Order flow
# ======================================================================

def _trades():
    ts = pd.date_range("2026-03-02 13:30", periods=60, freq="30s", tz="UTC")
    price = 6000 + np.repeat([0, 0.25, 0.5, 0.25, 0.0, -0.25], 10)
    side = np.repeat([1, 1, 1, -1, -1, -1], 10)
    return pd.DataFrame({"ts": ts, "price": price, "size": 2.0, "side": side})


def test_delta_bars_sum_buys_and_sells():
    b = delta_bars(_trades(), "5min")
    assert b["volume"].sum() == pytest.approx(120.0)
    assert b["delta"].iloc[0] == pytest.approx(20.0)  # first 5 min: 10 buys x 2
    assert b["delta"].iloc[-1] == pytest.approx(-20.0)
    assert b["cum_delta"].iloc[-1] == pytest.approx(0.0)


def test_tick_rule_infers_side():
    t = pd.DataFrame({"price": [1.0, 1.1, 1.1, 1.0, 1.0, 1.2]})
    assert infer_side(t).tolist() == [0, 1, 1, -1, -1, 1]


def test_imbalance_and_open_window():
    b = delta_bars(_trades(), "5min")
    imb = imbalance(b, window=2)
    assert imb.iloc[1] == pytest.approx(1.0)  # two all-buy bars
    ow = open_imbalance(b, 13, 30, minutes=15)
    assert len(ow) == 1
    assert ow["open_delta"].iloc[0] == pytest.approx(60.0)
    assert ow["available_at"].iloc[0] > b["ts"].iloc[2]
