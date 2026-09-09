"""A shadow adapter must reconnect the half that can actually break.

Found 2026-09-09. `Runner._heal_feed` prefers `adapter.reconnect()` and falls
back to `disconnect()`/`connect()`. Neither shadow adapter defined `reconnect`,
so the fallback ran `PaperAdapter.connect` - the paper book. The paper book
cannot fail. The broker link, which had failed, was never touched.

The healer therefore bounced the healthy half and wrote `ok=True,
"reconnected"` into the journal each time: ten of those on the forward record
overnight while it was halted for 91% of the window, and four on the shadow
week over 92 minutes of continuous silence with every symbol returning
`IPC send failed`.

A reconnect that reports success while fixing nothing is worse than none at
all, because afterwards the journal says it healed. These tests assert the
live side is what gets rebuilt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from execution.paper import FIXTURE_SPECS, make_tick  # noqa: E402
from execution.shadow import ShadowAdapter as FuturesShadow  # noqa: E402


class FakeLive:
    """A broker whose link drops and can be rebuilt, and which counts both."""

    def __init__(self):
        self.connected = True
        self.connects = 0
        self.disconnects = 0

    def account(self):
        from core.types import AccountState
        return AccountState(equity=100_000.0, balance=100_000.0, margin_used=0.0,
                            margin_free=100_000.0, currency="USD")

    def connect(self):
        self.connects += 1
        self.connected = True

    def disconnect(self):
        self.disconnects += 1
        self.connected = False

    def tick(self, symbol):
        if not self.connected:
            from execution.base import ExecutionError
            raise ExecutionError(f"no tick for {symbol!r}: (-10001, 'IPC send failed')")
        from datetime import datetime, timezone
        return make_tick(symbol, 1.0800, 0.0001, datetime.now(timezone.utc))

    def spec(self, symbol):
        return FIXTURE_SPECS[symbol]

    def bars(self, symbol, timeframe, count, end=None):
        return []


def _cfd_shadow(live):
    spec = importlib.util.spec_from_file_location("shadow_script", ROOT / "scripts" / "shadow.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shadow_script"] = mod
    spec.loader.exec_module(mod)
    return mod.ShadowAdapter(live, {"EURUSD": FIXTURE_SPECS["EURUSD"]})


def _futures_shadow(live):
    return FuturesShadow(live, {"EURUSD": FIXTURE_SPECS["EURUSD"]})


@pytest.mark.parametrize("build", [_cfd_shadow, _futures_shadow],
                         ids=["scripts/shadow.py", "execution/shadow.py"])
def test_reconnect_exists_at_all(build):
    """_heal_feed prefers `reconnect` and silently falls back to `connect` -
    the paper book - when it is absent. Absence is the bug."""
    a = build(FakeLive())
    assert callable(getattr(a, "reconnect", None)), (
        "without this, healing bounces the paper book and reports success")


@pytest.mark.parametrize("build", [_cfd_shadow, _futures_shadow],
                         ids=["scripts/shadow.py", "execution/shadow.py"])
def test_reconnect_rebuilds_the_live_link_not_the_paper_book(build):
    live = FakeLive()
    a = build(live)
    before = live.connects

    live.connected = False          # the broker link drops
    a.reconnect()

    assert live.disconnects >= 1, "the dead link must be torn down"
    assert live.connects == before + 1, "and rebuilt"
    assert live.connected is True


@pytest.mark.parametrize("build", [_cfd_shadow, _futures_shadow],
                         ids=["scripts/shadow.py", "execution/shadow.py"])
def test_ticks_flow_again_after_a_reconnect(build):
    """The property that actually matters: the runner un-halts because ticks
    arrive, not because a journal line said ok."""
    live = FakeLive()
    a = build(live)

    live.connected = False
    with pytest.raises(Exception):
        a.tick("EURUSD")

    a.reconnect()
    assert a.tick("EURUSD") is not None


@pytest.mark.parametrize("build", [_cfd_shadow, _futures_shadow],
                         ids=["scripts/shadow.py", "execution/shadow.py"])
def test_a_live_link_that_refuses_to_close_still_gets_reopened(build):
    """The link is already broken when this runs, so a failing disconnect is
    expected and must not stop the reopen."""
    live = FakeLive()

    def angry_disconnect():
        live.disconnects += 1
        raise RuntimeError("the pipe is already gone")

    live.disconnect = angry_disconnect
    a = build(live)
    live.connected = False

    a.reconnect()
    assert live.connected is True, "a noisy close must not prevent the reopen"


def test_the_runner_calls_reconnect_rather_than_connect_when_one_exists():
    """Pins the contract this whole file depends on: _heal_feed must prefer a
    reconnect method over the connect/disconnect fallback."""
    src = (ROOT / "live" / "runner.py").read_text(encoding="utf-8")
    heal = src.split("def _heal_feed")[1].split("def ")[0]
    assert 'getattr(self.adapter, "reconnect", None)' in heal
    assert heal.index('getattr(self.adapter, "reconnect", None)') < heal.index('"connect"'), (
        "reconnect must be tried before the connect fallback")
