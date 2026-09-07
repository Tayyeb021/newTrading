"""Rolling a position between delivery months.

Two faults found on 2026-09-07 by a real roll on IB paper, both silent:

1. **The stop was carried across at its face value.** Two delivery months of
   the same future are different prices. The September S&P micro traded at
   7722.25 with its stop at 7714.75, seven points away. After rolling into
   December at 7785 the stop was still 7714.75 - seventy points away, nine
   times the intended risk. Nothing errored. A short would have been worse:
   the stop would have landed on the far side of the market and liquidated
   the position the moment it was placed.

2. **The close leg was judged after a quarter of a second.** A fill that
   arrived a moment later was reported REJECTED, marking a roll as failed
   after it had actually worked - and, worse, leaving the code free to reopen
   in the front month while the expiring contract was still held.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("ib_async")

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from core.types import OrderRequest, Side  # noqa: E402
from execution.ib_adapter import IBAdapter  # noqa: E402
from execution.ib_fake import FakeIB  # noqa: E402

MES = MICRO_UNIVERSE["MES"]
SEP, DEC = (2026, 9), (2026, 12)
# The real prices from 2026-09-07: sixty-three points of carry between months.
SEP_PX, DEC_PX = 7722.25, 7785.25


def _adapter(sep=SEP_PX, dec=DEC_PX, today=None):
    fake = FakeIB(
        MICRO_UNIVERSE,
        prices={"MES": sep, "MES:202609": sep, "MES:202612": dec},
        equity=1_000_000.0,
    )
    ad = IBAdapter(ib=fake, roots=MICRO_UNIVERSE, today=today or date(2026, 8, 1))
    ad.connect()
    return ad, fake


def _open_long(ad, stop_offset=7.5):
    tick = ad.tick("MES")
    res = ad.submit(OrderRequest("MES", Side.BUY, 1, stop_loss=tick.ask - stop_offset, comment="rt"))
    assert res.ok
    return res


def test_the_basis_between_months_is_measured_not_assumed():
    ad, _ = _adapter(today=MES.roll_date(*SEP))  # on the roll day the front is December
    assert ad.front_month("MES") == DEC
    basis = ad.roll_basis("MES", SEP)
    assert basis == pytest.approx(DEC_PX - SEP_PX, abs=0.01), "the carry between months is the basis"


def test_the_basis_is_zero_when_there_is_nothing_to_roll_into():
    """Before the roll date the front month IS the held month; measuring a basis
    against yourself must give zero, not a spurious shift."""
    ad, _ = _adapter(today=date(2026, 8, 1))
    assert ad.front_month("MES") == SEP
    assert ad.roll_basis("MES", SEP) == pytest.approx(0.0, abs=1e-9)


def test_rolling_preserves_the_stop_DISTANCE_not_its_price():
    """The bug: a stop seven points away became seventy after the roll."""
    ad, _ = _adapter(today=date(2026, 8, 1))
    entry = _open_long(ad, stop_offset=7.5)
    before = ad.positions("MES")[0]
    distance_before = abs(before.entry_price - before.stop_loss)
    assert distance_before == pytest.approx(7.5, abs=0.3)

    ad._today = MES.roll_date(*SEP)
    results = ad.roll("MES")
    assert all(r.ok for r in results), [r.reason for r in results]

    after = ad.positions("MES")[0]
    distance_after = abs(after.entry_price - after.stop_loss)
    assert after.entry_price == pytest.approx(DEC_PX, abs=1.0), "we are in December now"
    assert distance_after == pytest.approx(distance_before, abs=0.3), (
        f"risk changed across the roll: {distance_before:.2f} -> {distance_after:.2f}"
    )
    assert after.stop_loss > SEP_PX, "the stop must move up with the contract, not stay behind"


def test_a_short_stop_stays_on_the_correct_side_of_the_market():
    """The catastrophic case. Carried across unadjusted, a short's stop ends up
    BELOW a higher-priced contract and liquidates the position on arrival."""
    ad, _ = _adapter(today=date(2026, 8, 1))
    tick = ad.tick("MES")
    assert ad.submit(OrderRequest("MES", Side.SELL, 1, stop_loss=tick.bid + 7.5, comment="rt")).ok

    ad._today = MES.roll_date(*SEP)
    assert all(r.ok for r in ad.roll("MES"))

    after = ad.positions("MES")[0]
    assert after.side is Side.SELL
    assert after.stop_loss > after.entry_price, "a short's stop sits ABOVE the market or it fires at once"
    assert abs(after.stop_loss - after.entry_price) == pytest.approx(7.5, abs=0.3)


def test_the_roll_holds_when_the_close_leg_does_not_fill():
    """Reopening while the expiring contract is still held doubles the position
    in the month you are trying to leave."""
    ad, fake = _adapter(today=date(2026, 8, 1))
    _open_long(ad)
    ad._today = MES.roll_date(*SEP)

    fake.fail_next_fill = True  # close leg fills but never confirms
    results = ad.roll("MES")
    assert len(results) == 1 and not results[0].ok, "must stop after a close that did not confirm"
    assert "roll-close" in results[0].request.comment


def test_backwardation_moves_the_stop_the_other_way():
    """Not a special case for contango: the sign follows the curve."""
    ad, _ = _adapter(sep=7722.25, dec=7690.25, today=date(2026, 8, 1))
    _open_long(ad, stop_offset=7.5)
    ad._today = MES.roll_date(*SEP)
    assert all(r.ok for r in ad.roll("MES"))

    after = ad.positions("MES")[0]
    assert after.stop_loss < SEP_PX - 7.0, "a cheaper next month drags the stop down with it"
    assert abs(after.entry_price - after.stop_loss) == pytest.approx(7.5, abs=0.3)
