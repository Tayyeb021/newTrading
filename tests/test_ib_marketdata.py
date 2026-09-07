"""Market-data entitlement on the IB adapter.

Found on 2026-09-07, the first time the system reached a real IB paper account:
a free-trial account has no live data subscription, so `reqMktData` answers NaN
and IB reports error 354 asynchronously. Nothing raises. The quote step failed
with `bid nan ask nan`, and every downstream number was NaN with it.

The fix is to walk the entitlement ladder by VALUE - live, then delayed, then
delayed-frozen - and to record which one answered, because a delayed quote is
fine for verifying a code path and worthless for measuring a fill.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from execution.base import ExecutionError  # noqa: E402
from execution.ib_adapter import IBAdapter  # noqa: E402
from execution.ib_fake import FakeIB  # noqa: E402

PRICES = {"MES": 6000.0, "MGC": 2650.0}


def _adapter(entitled=None):
    fake = FakeIB(MICRO_UNIVERSE, prices=PRICES, now=datetime.now(timezone.utc))
    fake.entitled_types = entitled
    ad = IBAdapter(ib=fake, roots=MICRO_UNIVERSE, today=None)
    ad.connect()
    return ad, fake


def test_live_data_is_preferred_when_entitled():
    ad, fake = _adapter(entitled=None)
    tick = ad.tick("MES")
    assert tick.bid > 0 and tick.ask > 0
    assert ad.market_data_type == IBAdapter.LIVE
    assert fake.market_data_type == 1


def test_falls_back_to_delayed_on_a_free_trial_account():
    """No live entitlement: IB returns NaN, not an error. Fall back by value."""
    ad, fake = _adapter(entitled={3, 4})
    tick = ad.tick("MES")
    assert tick.bid > 0 and tick.ask > 0, "a delayed quote is still a quote"
    assert ad.market_data_type == IBAdapter.DELAYED
    assert fake.market_data_type == 3


def test_falls_back_again_to_frozen_when_the_market_is_shut():
    ad, _ = _adapter(entitled={4})
    assert ad.tick("MES").bid > 0
    assert ad.market_data_type == IBAdapter.DELAYED_FROZEN


def test_no_entitlement_at_all_raises_rather_than_returning_nan():
    ad, _ = _adapter(entitled=set())
    with pytest.raises(ExecutionError, match="no quote"):
        ad.tick("MES")


def test_the_established_type_is_remembered_and_not_retried():
    """Once delayed is known to be the ceiling, stop asking for live every tick:
    each failed attempt costs a two-second wait per symbol per bar."""
    ad, fake = _adapter(entitled={3, 4})
    ad.tick("MES")
    assert ad.market_data_type == IBAdapter.DELAYED
    fake.market_data_type = 0  # prove the next call sets it deliberately
    ad.tick("MGC")
    assert fake.market_data_type == 3, "should have gone straight to delayed"
    assert ad.market_data_type == IBAdapter.DELAYED


def test_delayed_type_is_visible_so_callers_can_refuse_to_measure_with_it():
    ad, _ = _adapter(entitled={3, 4})
    ad.tick("MES")
    assert ad.market_data_type != IBAdapter.LIVE
