"""A dead feed must halt trading AND then heal itself.

Found on 2026-09-08. Both runners sat halted for seven hours with an infinite
feed age while their venues were healthy: IB Gateway had done its nightly
restart, and the MetaTrader terminal was ticking one second old while the
connection *inside* the process was dead.

Halting was correct. Never recovering was not, and the reason is subtle.
`_ticks` swallows per-symbol failures so one dead symbol cannot stop the rest,
which means a wholly dead feed raises nothing at all - so the loop's own error
path, the one that reconnects, was never reached. The feed simply went quiet
and the system waited forever.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RiskProfile  # noqa: E402
from core.strategy import FLAT, Strategy  # noqa: E402
from execution.base import ExecutionError  # noqa: E402
from execution.paper import FIXTURE_SPECS, PaperAdapter, make_tick  # noqa: E402
from live.runner import Runner  # noqa: E402
from live.state import StateStore  # noqa: E402
from ops.journal import Journal  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.killswitch import KillFile  # noqa: E402

SPEC = FIXTURE_SPECS["EURUSD"]


class Quiet(Strategy):
    name = "quiet"
    warmup = 1

    def evaluate(self, df, i, position):
        return FLAT


class DeadFeed(PaperAdapter):
    """A venue whose quotes stop while everything else keeps answering - the
    exact shape of a client connection that died under a healthy terminal."""

    def __init__(self, specs):
        super().__init__(specs)
        self.alive = True
        self.reconnects = 0

    def tick(self, symbol):
        if not self.alive:
            raise ExecutionError("no quote: connection is dead")
        return super().tick(symbol)

    def reconnect(self):
        self.reconnects += 1
        self.alive = True


def _runner(tmp_path, adapter, threshold=3):
    engine = build_engine(RiskProfile.load("challenge"), 100_000.0, {"EURUSD": SPEC})
    r = Runner(adapter=adapter, risk=engine, strategies={"EURUSD": Quiet()},
               specs={"EURUSD": SPEC}, state=StateStore(tmp_path / "s.json"),
               journal=Journal(tmp_path / "j.jsonl"), kill=KillFile(tmp_path / "KILL"),
               poll_seconds=0.0)
    r.stale_polls_before_reconnect = threshold
    return r


def test_a_dead_feed_halts_and_then_reconnects(tmp_path):
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.0800, 0.0001, datetime.now(timezone.utc)))
    r = _runner(tmp_path, a, threshold=3)
    r.worker.start()
    try:
        r.tick()
        assert not r.halted, "a live feed must not halt"

        a.alive = False
        for _ in range(2):
            r.tick()
        assert r.halted, "a dead feed must stop trading"
        assert a.reconnects == 0, "and must not thrash the connection on the first miss"

        r.tick()  # the third consecutive empty poll
        assert a.reconnects == 1, "after the threshold it must rebuild the connection"
        events = [e["event"] for e in r.journal.read()]
        assert "feed_reconnect" in events, "and say so in the journal"
    finally:
        r.worker.stop()


def test_recovery_resets_the_counter_so_it_does_not_reconnect_forever(tmp_path):
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.0800, 0.0001, datetime.now(timezone.utc)))
    r = _runner(tmp_path, a, threshold=2)
    r.worker.start()
    try:
        a.alive = False
        r.tick(); r.tick()
        assert a.reconnects == 1

        a.feed_tick(make_tick("EURUSD", 1.0801, 0.0001, datetime.now(timezone.utc)))
        r.tick()
        assert r._stale_polls == 0, "a live tick clears the count"

        a.alive = False
        r.tick()
        assert a.reconnects == 1, "one miss after recovery is not a reason to reconnect"
    finally:
        r.worker.stop()


def test_a_failing_reconnect_is_recorded_not_raised(tmp_path):
    """The loop must survive a venue that is simply down."""
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.0800, 0.0001, datetime.now(timezone.utc)))

    def boom():
        raise ExecutionError("gateway is not running")
    a.reconnect = boom

    r = _runner(tmp_path, a, threshold=1)
    r.worker.start()
    try:
        a.alive = False
        r.tick()  # must not raise
        rec = [e for e in r.journal.read() if e["event"] == "feed_reconnect"]
        assert rec and rec[-1]["ok"] is False
        assert "gateway is not running" in rec[-1]["detail"]
        assert r.halted, "still halted: a failed reconnect does not license trading"
    finally:
        r.worker.stop()


def test_per_symbol_errors_are_captured_for_the_journal(tmp_path):
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.0800, 0.0001, datetime.now(timezone.utc)))
    r = _runner(tmp_path, a, threshold=1)
    r.worker.start()
    try:
        a.alive = False
        r.tick()
        rec = [e for e in r.journal.read() if e["event"] == "feed_reconnect"][-1]
        assert "EURUSD" in rec["last_errors"]
        assert "connection is dead" in rec["last_errors"]["EURUSD"]
    finally:
        r.worker.stop()
