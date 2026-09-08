"""A dead feed must halt trading AND then heal itself, on the same clock
whatever the poll rate.

Found on 2026-09-08. Both runners sat halted for seven hours with an infinite
feed age while their venues were healthy: IB Gateway had done its nightly
restart, and the MetaTrader terminal was ticking one second old while the
connection *inside* the process was dead.

Halting was correct. Never recovering was not, and the reason is subtle.
`_ticks` swallows per-symbol failures so one dead symbol cannot stop the rest,
which means a wholly dead feed raises nothing at all - so the loop's own error
path, the one that reconnects, was never reached. The feed simply went quiet
and the system waited forever.

The first fix counted polls, and that unit was wrong. Three polls is
thirty-six seconds on the twelve-second shadow loop and three hours on the
hourly forward loop, so the runner that most needed to heal waited longest -
the forward record lost seventeen hours on its first night. The threshold is
now wall-clock silence measured from the last tick that arrived, which is why
several of these tests rewind that timestamp rather than sleeping.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def _runner(tmp_path, adapter, stale_seconds=120.0, cooldown=900.0):
    engine = build_engine(RiskProfile.load("challenge"), 100_000.0, {"EURUSD": SPEC})
    r = Runner(adapter=adapter, risk=engine, strategies={"EURUSD": Quiet()},
               specs={"EURUSD": SPEC}, state=StateStore(tmp_path / "s.json"),
               journal=Journal(tmp_path / "j.jsonl"), kill=KillFile(tmp_path / "KILL"),
               poll_seconds=0.0)
    r.stale_seconds_before_reconnect = stale_seconds
    r.reconnect_cooldown_seconds = cooldown
    return r


def _live(adapter):
    adapter.feed_tick(make_tick("EURUSD", 1.0800, 0.0001, datetime.now(timezone.utc)))


def _rewind(runner, seconds: float) -> None:
    """Age the last-tick clock, so the runner sees a long silence without one."""
    runner._last_fed_at -= timedelta(seconds=seconds)


def test_a_dead_feed_halts_and_then_reconnects(tmp_path):
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    _live(a)
    r = _runner(tmp_path, a, stale_seconds=120.0)
    r.worker.start()
    try:
        r.tick()
        assert not r.halted, "a live feed must not halt"

        a.alive = False
        r.tick()
        assert r.halted, "a dead feed must stop trading"
        assert a.reconnects == 0, "and must not thrash the connection on the first miss"

        _rewind(r, 121)
        r.tick()
        assert a.reconnects == 1, "after two minutes of silence it must rebuild the connection"
        rec = [e for e in r.journal.read() if e["event"] == "feed_reconnect"]
        assert rec and rec[-1]["ok"] is True, "and say so in the journal"
        assert rec[-1]["silent_seconds"] >= 120
    finally:
        r.worker.stop()


def test_an_hourly_runner_heals_on_its_first_dead_poll(tmp_path):
    """The regression. Under the poll-counting version this runner needed three
    polls - three hours - and the forward record sat halted from 13:51 to 06:36
    because of it. An hour of silence is already past the threshold, so the very
    first empty poll must act."""
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    _live(a)
    r = _runner(tmp_path, a, stale_seconds=120.0)
    r.worker.start()
    try:
        r.tick()
        a.alive = False
        _rewind(r, 3600)          # an hour between polls, as run_forward.py has

        r.tick()
        assert a.reconnects == 1, "an hourly poll must not wait three more hours"
    finally:
        r.worker.stop()


def test_a_fast_runner_waits_the_same_two_minutes(tmp_path):
    """The other side of the same property: twelve-second polls must not
    reconnect ten times in the time the hourly one reconnects once."""
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    _live(a)
    r = _runner(tmp_path, a, stale_seconds=120.0)
    r.worker.start()
    try:
        r.tick()
        a.alive = False
        for _ in range(10):       # 10 polls x 12s = two minutes of a fast loop
            _rewind(r, 12)
            r.tick()
        assert a.reconnects == 1, "one rebuild for two minutes of silence, not ten"
    finally:
        r.worker.stop()


def test_a_closed_market_does_not_thrash_the_connection(tmp_path):
    """7,405 of the shadow week's 7,595 stale-feed breaches were a weekend. An
    empty feed is not proof of a broken one, so retries are rate-limited."""
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    _live(a)
    r = _runner(tmp_path, a, stale_seconds=120.0, cooldown=900.0)
    r.worker.start()
    try:
        r.tick()
        a.alive = False
        a.reconnect = lambda: setattr(a, "reconnects", a.reconnects + 1)  # stays dead

        _rewind(r, 121)
        r.tick()
        assert a.reconnects == 1

        for _ in range(50):
            _rewind(r, 12)
            r.tick()
        assert a.reconnects == 1, "a market that is merely closed must not be hammered"

        r._last_reconnect_at -= timedelta(seconds=901)
        r.tick()
        assert a.reconnects == 2, "but after the cooldown it tries again"
    finally:
        r.worker.stop()


def test_recovery_resets_the_clock_so_it_does_not_reconnect_forever(tmp_path):
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    _live(a)
    r = _runner(tmp_path, a, stale_seconds=120.0)
    r.worker.start()
    try:
        r.tick()
        a.alive = False
        _rewind(r, 121)
        r.tick()
        assert a.reconnects == 1          # DeadFeed.reconnect revives the venue

        _live(a)
        r.tick()
        assert not r.halted, "a live tick clears the halt"

        a.alive = False
        r.tick()
        assert a.reconnects == 1, "one miss after recovery is not a reason to reconnect"
    finally:
        r.worker.stop()


def test_a_cold_start_with_a_dead_feed_does_not_reconnect_immediately(tmp_path):
    """Nothing has ever arrived, so there is no silence to measure yet - and a
    connection that has not had a chance to work must not be torn down."""
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    a.alive = False
    r = _runner(tmp_path, a, stale_seconds=120.0)
    r.worker.start()
    try:
        r.tick()
        assert a.reconnects == 0
        assert r._last_fed_at is not None, "the silence is timed from the first poll"

        _rewind(r, 121)
        r.tick()
        assert a.reconnects == 1
    finally:
        r.worker.stop()


def test_a_failing_reconnect_is_recorded_not_raised(tmp_path):
    """The loop must survive a venue that is simply down."""
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    _live(a)

    def boom():
        raise ExecutionError("gateway is not running")
    a.reconnect = boom

    r = _runner(tmp_path, a, stale_seconds=0.0)
    r.worker.start()
    try:
        r.tick()          # one live poll, so the silence has something to start from
        a.alive = False
        r.tick()          # must not raise
        rec = [e for e in r.journal.read() if e["event"] == "feed_reconnect"]
        assert rec and rec[-1]["ok"] is False
        assert "gateway is not running" in rec[-1]["detail"]
        assert r.halted, "still halted: a failed reconnect does not license trading"
    finally:
        r.worker.stop()


def test_per_symbol_errors_are_captured_for_the_journal(tmp_path):
    a = DeadFeed(FIXTURE_SPECS)
    a.connect()
    _live(a)
    r = _runner(tmp_path, a, stale_seconds=0.0)
    r.worker.start()
    try:
        r.tick()          # one live poll, so the silence has something to start from
        a.alive = False
        r.tick()
        rec = [e for e in r.journal.read() if e["event"] == "feed_reconnect"][-1]
        assert "EURUSD" in rec["last_errors"]
        assert "connection is dead" in rec["last_errors"]["EURUSD"]
    finally:
        r.worker.stop()
