"""The test double must not invent API that the real client does not have.

The bug this exists to prevent, found on 2026-09-07 by the first real order:

    AttributeError: 'IB' object has no attribute 'make_market_order'

`make_market_order`, `make_stop_order` and `make_future` existed only on
`FakeIB`. The adapter called them, the double answered, and twelve green checks
said the futures path worked. It did not. A double written to match the code
under test, rather than the thing it stands in for, validates a fiction.

So: every attribute the adapter reaches for on its client must exist on the real
`ib_async.IB`, and the double must implement that same surface. This is a static
check by design - it needs no TWS, no account and no network, which is exactly
why it can run on every commit.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from execution.ib_fake import FakeIB  # noqa: E402

ib_async = pytest.importorskip("ib_async", reason="the real client is what we conform to")
ADAPTER = ROOT / "execution" / "ib_adapter.py"


def client_attributes_used() -> set[str]:
    """Every `self.ib.X` / `self._ib.X` the adapter touches, read from the AST."""
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Attribute):
            continue
        inner = node.value
        if isinstance(inner.value, ast.Name) and inner.value.id == "self" and inner.attr in ("ib", "_ib"):
            used.add(node.attr)
    return used


def test_the_adapter_only_calls_methods_the_real_client_has():
    used = client_attributes_used()
    assert used, "parsed nothing - the AST walk is broken, not the adapter"
    missing = sorted(a for a in used if not hasattr(ib_async.IB, a))
    assert not missing, (
        f"the adapter calls {missing} on its client, but ib_async.IB has no such attribute. "
        f"This is the make_market_order bug: it will pass every test against the double "
        f"and raise AttributeError on the first real order."
    )


def test_the_double_implements_everything_the_adapter_uses():
    used = client_attributes_used()
    missing = sorted(a for a in used if not hasattr(FakeIB, a))
    assert not missing, f"FakeIB is missing {missing}, so tests exercise a different path than production"


def test_the_double_invents_nothing_the_real_client_lacks():
    """A public method on the double with no counterpart on IB is a trap: code
    can start using it and no test will notice until a live order."""
    invented = sorted(
        name for name, _ in inspect.getmembers(FakeIB, callable)
        if not name.startswith("_") and not hasattr(ib_async.IB, name)
    )
    assert not invented, (
        f"FakeIB defines {invented}, which ib_async.IB does not have. "
        f"Either the real client grew them, or the double is inventing API again."
    )


def test_orders_are_built_from_the_library_not_the_client():
    """The fix: order objects come from ib_async, so the double receives exactly
    what IB receives."""
    from execution.ib_adapter import IBAdapter

    market = IBAdapter.market_order("BUY", 2)
    stop = IBAdapter.stop_order("SELL", 2, 5990.25)
    assert isinstance(market, ib_async.MarketOrder) and isinstance(stop, ib_async.StopOrder)
    assert market.action == "BUY" and market.totalQuantity == 2 and market.orderType == "MKT"
    assert stop.orderType == "STP" and stop.auxPrice == 5990.25
    assert market.orderId == 0, "IB assigns the id on placement; the double must do the same"
