"""Time-in-force and session routing on IB orders.

Found on 2026-09-07 by the second real order attempt, at 04:35 New York:

    Error 10349: Order TIF was set to DAY based on order preset.  -> Cancelled

`ib_async` leaves `tif` empty and `outsideRth` False, and IB's preset fills in
DAY. Two separate faults follow, and the second is the dangerous one:

1. Outside 09:30-16:15 New York every order is refused, in a contract that
   trades nearly around the clock. Visible immediately - the order is cancelled.
2. A protective stop with DAY time-in-force is **cancelled at the session
   close**. Invisible. Every strategy here holds overnight, so the position
   would wake up unprotected while the risk engine still believed a stop was
   attached and every aggregate risk figure derived from stops was wrong.
   `UnstoppedPosition` would not catch it: the position did have a stop when it
   was last looked at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("ib_async")

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from core.types import OrderRequest, Side  # noqa: E402
from execution.ib_adapter import IBAdapter  # noqa: E402
from execution.ib_fake import FakeIB  # noqa: E402


def _adapter():
    fake = FakeIB(MICRO_UNIVERSE, prices={"MES": 6000.0}, equity=1_000_000.0)
    ad = IBAdapter(ib=fake, roots=MICRO_UNIVERSE)
    ad.connect()
    return ad, fake


def test_a_protective_stop_must_survive_the_session():
    """The whole point. DAY here means an unprotected position overnight."""
    stop = IBAdapter.stop_order("SELL", 1, 5990.25)
    assert stop.tif == "GTC", "a DAY stop is cancelled at the close and the position is naked"
    assert stop.outsideRth is True, "a stop that cannot trigger outside RTH is not a stop"


def test_market_orders_work_when_the_market_is_actually_open():
    order = IBAdapter.market_order("BUY", 1)
    assert order.outsideRth is True, "CME index futures trade nearly 24h; RTH-only refuses most of it"
    assert order.tif == "DAY", "a market order fills now or not at all; DAY is correct once RTH is lifted"


def test_neither_order_leaves_the_time_in_force_for_ib_to_guess():
    """An empty tif is what let the preset set DAY on the stop."""
    for order in (IBAdapter.market_order("BUY", 1), IBAdapter.stop_order("SELL", 1, 100.0)):
        assert order.tif, "empty tif hands the decision to IB's order preset"


def test_the_terms_survive_the_whole_submit_path():
    """Not just the factory: what actually reaches the client."""
    ad, fake = _adapter()
    res = ad.submit(OrderRequest("MES", Side.BUY, 1, stop_loss=5990.0, comment="terms"))
    assert res.ok

    placed = {t.order.orderType: t.order for t in fake.trades()}
    assert set(placed) == {"MKT", "STP"}
    assert placed["MKT"].outsideRth is True and placed["MKT"].tif == "DAY"
    assert placed["STP"].outsideRth is True and placed["STP"].tif == "GTC"
    assert placed["STP"].parentId == placed["MKT"].orderId, "the stop must be a child of the entry"


def test_a_modified_stop_keeps_its_terms():
    """Moving a stop must not silently reset it to a DAY order."""
    ad, fake = _adapter()
    ad.submit(OrderRequest("MES", Side.BUY, 1, stop_loss=5990.0, comment="terms"))
    ticket = next(iter(ad._orders))
    ad.modify(ticket, stop_loss=5995.0)

    stop = next(t.order for t in fake.trades() if t.order.orderType == "STP")
    assert stop.auxPrice == 5995.0
    assert stop.tif == "GTC" and stop.outsideRth is True
