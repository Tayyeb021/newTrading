"""Tests for the live layer: state persistence, OMS, and the runner loop.

The chaos scenarios in `scripts/chaos_test.py` cover recovery end to end. These
are the unit-level assertions underneath them, including the regressions worth
locking down permanently.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RiskProfile  # noqa: E402
from core.strategy import FLAT, Intent, Strategy  # noqa: E402
from core.types import OrderRequest, OrderResult, OrderStatus, Side  # noqa: E402
from execution.oms import OrderManager, Retryable, classify, client_id  # noqa: E402
from execution.paper import FIXTURE_SPECS, PaperAdapter, make_tick  # noqa: E402
from live.runner import ExecutionWorker, Runner  # noqa: E402
from live.state import StateStore, restore_book  # noqa: E402
from ops.journal import Journal  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.killswitch import KillFile  # noqa: E402

NOW = datetime(2026, 9, 4, 14, 0, tzinfo=timezone.utc)
TODAY = NOW.date()


@pytest.fixture()
def adapter() -> PaperAdapter:
    a = PaperAdapter(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.0800, 0.00010, NOW))
    return a


def book(equity: float = 100_000.0):
    b = build_engine(RiskProfile.load("challenge"), equity, dict(FIXTURE_SPECS)).book
    b.trading_day = TODAY
    return b


# ======================================================================= state

def test_day_start_equity_is_not_reset_by_a_restart(tmp_path: Path):
    """The account killer. A restart must not re-arm the daily loss limit."""
    store = StateStore(tmp_path / "s.json")
    b = book()
    b.observe_equity(96_500.0, today=TODAY)  # -3.5% morning
    store.save(b)

    restored, notes = restore_book(store, current_equity=96_500.0, today=TODAY)
    assert restored.day_start_equity == 100_000.0, "daily limit was silently re-armed"
    assert any("CARRIED FORWARD" in n for n in notes)


def test_day_roll_resets_daily_but_keeps_drawdown(tmp_path: Path):
    store = StateStore(tmp_path / "s.json")
    b = book()
    b.observe_equity(105_000.0, today=TODAY)
    b.observe_equity(97_000.0, today=TODAY)
    store.save(b)

    restored, _ = restore_book(store, 97_000.0, today=TODAY + timedelta(days=1))
    assert restored.day_start_equity == 97_000.0  # new day
    assert restored.high_water_equity == 105_000.0  # survives
    assert restored.starting_equity == 100_000.0  # survives


def test_first_run_opens_a_fresh_book(tmp_path: Path):
    restored, notes = restore_book(StateStore(tmp_path / "none.json"), 50_000.0, TODAY)
    assert restored.starting_equity == restored.day_start_equity == 50_000.0
    assert any("fresh session" in n for n in notes)


def test_kill_state_survives_a_restart(tmp_path: Path):
    store = StateStore(tmp_path / "s.json")
    b = book()
    b.kill("daily loss breach")
    store.save(b)

    restored, notes = restore_book(store, 96_000.0, TODAY)
    assert restored.killed
    assert any("KILL SWITCH IS STILL ENGAGED" in n for n in notes)


def test_consecutive_losses_survive_a_restart(tmp_path: Path):
    store = StateStore(tmp_path / "s.json")
    b = book()
    b.record_close("trend", -100.0, NOW)
    b.record_close("trend", -100.0, NOW)
    store.save(b)

    restored, _ = restore_book(store, 99_800.0, TODAY)
    assert restored.consecutive_losses["trend"] == 2
    assert restored.last_loss_ts["trend"] == NOW


def test_state_write_is_atomic(tmp_path: Path):
    store = StateStore(tmp_path / "s.json")
    store.save(book())
    assert store.load() is not None
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"


def test_unknown_schema_is_refused_not_guessed(tmp_path: Path):
    path = tmp_path / "s.json"
    path.write_text('{"schema": 99, "session_date": "2026-01-01"}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        StateStore(path).load()


# ======================================================================== kill

def test_kill_file_round_trip(tmp_path: Path):
    kill = KillFile(tmp_path / "KILL")
    assert not kill.engaged()

    kill.engage("daily loss", by="risk_engine")
    assert kill.engaged()
    record = kill.read()
    assert record.reason == "daily loss" and record.by == "risk_engine"

    kill.clear()
    assert not kill.engaged()


def test_unreadable_kill_file_still_means_kill(tmp_path: Path):
    """Fail closed. The only safe reading of a corrupt stop signal is 'stop'."""
    path = tmp_path / "KILL"
    path.write_text("{ not json", encoding="utf-8")
    record = KillFile(path).read()
    assert record.engaged


# ========================================================================= oms

def test_failure_classification():
    assert classify("Requote") is Retryable.YES
    assert classify("connection timeout") is Retryable.AMBIGUOUS
    assert classify("Invalid stops") is Retryable.NO
    assert classify("not enough money") is Retryable.NO
    assert classify("something nobody has seen") is Retryable.NO  # investigate, don't retry


def test_client_id_is_deterministic_and_fits_the_comment_field():
    a = client_id("S1_trend", "EURUSD", "BUY", NOW)
    b = client_id("S1_trend", "EURUSD", "BUY", NOW)
    c = client_id("S1_trend", "EURUSD", "BUY", NOW + timedelta(days=1))
    assert a == b
    assert a != c
    assert len(a) <= 31, "MT5 comment field is 31 characters"


def test_terminal_failures_are_not_retried(adapter: PaperAdapter):
    oms = OrderManager(adapter, sleep=lambda _s: None)
    cid = client_id("t", "XAUUSD", "BUY", NOW)
    adapter.feed_tick(make_tick("XAUUSD", 3300.0, 0.30, NOW))

    result = oms.submit(OrderRequest("XAUUSD", Side.BUY, 0.001, stop_loss=3200.0), cid)
    assert not result.ok
    assert len(oms.attempts) == 1, "a terminal rejection must not be retried"


class LosesTheReply(PaperAdapter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lie_once = True

    def submit(self, request):
        result = super().submit(request)
        if self.lie_once and result.ok:
            self.lie_once = False
            return OrderResult(OrderStatus.REJECTED, request, reason="connection timeout")
        return result


def test_ambiguous_failure_never_doubles_the_position():
    a = LosesTheReply(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.0800, 0.00010, NOW))

    oms = OrderManager(a, sleep=lambda _s: None)
    cid = client_id("trend", "EURUSD", "BUY", NOW)
    result = oms.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.0700), cid)

    assert len(a.positions("EURUSD")) == 1, "duplicated the position"
    assert result.ok and "recovered" in result.reason


def test_resubmitting_the_same_intent_is_idempotent(adapter: PaperAdapter):
    oms = OrderManager(adapter, sleep=lambda _s: None)
    cid = client_id("trend", "EURUSD", "BUY", NOW)
    req = OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.0700)

    first = oms.submit(req, cid)
    second = oms.submit(req, cid)
    assert first.ok and second.ok
    assert second.reason == "already_open"
    assert len(adapter.positions("EURUSD")) == 1


def test_orphans_are_positions_the_journal_never_saw(adapter: PaperAdapter, tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    oms = OrderManager(adapter, sleep=lambda _s: None)

    cid = client_id("trend", "EURUSD", "BUY", NOW)
    known = oms.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.07), cid)
    journal.fill(known, cid)

    stray = adapter.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.07))
    orphans = oms.orphans(journal.known_tickets())

    assert [p.ticket for p in orphans] == [stray.ticket]


def test_ensure_stops_repairs_a_naked_position(adapter: PaperAdapter):
    adapter.submit(OrderRequest("EURUSD", Side.BUY, 0.10))
    assert adapter.positions("EURUSD")[0].stop_loss is None

    OrderManager(adapter, sleep=lambda _s: None).ensure_stops({"EURUSD": 0.0100})
    repaired = adapter.positions("EURUSD")[0]
    assert repaired.stop_loss is not None
    assert repaired.stop_loss < repaired.entry_price


def test_close_all_never_raises(adapter: PaperAdapter):
    class Broken(PaperAdapter):
        def positions(self, symbol=None):
            from execution.base import ExecutionError
            raise ExecutionError("broker is down")

    broken = Broken(FIXTURE_SPECS)
    broken.connect()
    assert OrderManager(broken, sleep=lambda _s: None).close_all() == []


# ===================================================================== journal

def test_journal_records_and_replays(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.write("startup", equity=100_000.0)
    journal.heartbeat(equity=99_000.0, positions=1)

    assert len(journal.read()) == 2
    assert len(journal.read("heartbeat")) == 1


def test_journal_survives_a_truncated_final_line(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    journal.write("startup", equity=100.0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": "2026-09-04", "eve')  # crash mid-write

    records = journal.read()
    assert len(records) == 1, "a truncated line must not lose the whole journal"


def test_journal_encodes_dataclasses_and_enums(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    result = OrderResult(
        OrderStatus.FILLED, OrderRequest("EURUSD", Side.BUY, 0.1),
        ticket=7, fill_price=1.08, requested_price=1.0799,
    )
    journal.fill(result, "cid#1")
    record = journal.read("fill")[0]
    assert record["status"] == "filled" and record["ticket"] == 7
    assert record["slippage"] == pytest.approx(0.0001, abs=1e-6)


# ====================================================================== runner

class OneShotLong(Strategy):
    name = "test_long"
    warmup = 3

    def __init__(self):
        self.fired = False

    def evaluate(self, df, i, position):
        if position is not None or self.fired:
            return FLAT
        self.fired = True
        return Intent(side=Side.BUY, stop_distance=0.0100)


def make_runner(adapter: PaperAdapter, tmp_path: Path, strategy=None) -> Runner:
    engine = build_engine(RiskProfile.load("challenge"), 100_000.0, {"EURUSD": FIXTURE_SPECS["EURUSD"]})
    return Runner(
        adapter=adapter,
        risk=engine,
        strategies={"EURUSD": strategy or OneShotLong()},
        specs={"EURUSD": FIXTURE_SPECS["EURUSD"]},
        state=StateStore(tmp_path / "s.json"),
        journal=Journal(tmp_path / "j.jsonl"),
        kill=KillFile(tmp_path / "KILL"),
        poll_seconds=0.0,
    )


def _feed_bars(adapter: PaperAdapter, n: int = 12) -> None:
    from core.types import Bar

    bars = [
        Bar("EURUSD", NOW - timedelta(days=n - i), 1.08, 1.085, 1.075, 1.08 + i * 0.001, 100.0)
        for i in range(n)
    ]
    adapter.feed_bars("EURUSD", "D1", bars)


def test_runner_start_reports_and_persists(adapter: PaperAdapter, tmp_path: Path):
    _feed_bars(adapter)
    runner = make_runner(adapter, tmp_path)
    notes = runner.start()
    try:
        assert any("connected" in n for n in notes)
        assert any("no orphan positions" in n for n in notes)
        assert runner.journal.read("startup")
    finally:
        runner.shutdown()


def test_runner_places_a_trade_through_the_risk_engine(adapter: PaperAdapter, tmp_path: Path):
    _feed_bars(adapter)
    # The fixture tick is stamped at the fixed NOW constant. Feed age is now
    # checked against the real clock, so the happy path needs a fresh quote or
    # FeedHeartbeat correctly halts it - which is a different test.
    adapter.feed_tick(make_tick("EURUSD", 1.0800, 0.00010, datetime.now(timezone.utc)))
    runner = make_runner(adapter, tmp_path)
    runner.start()
    try:
        runner.tick()
        runner.worker.stop()
        assert adapter.positions("EURUSD"), "no position opened"
        decisions = runner.journal.read("decision")
        assert decisions and decisions[0]["approved"]
        assert decisions[0]["risk_fraction"] <= 0.0075
    finally:
        runner.shutdown()


def test_runner_refuses_to_trade_when_killed(adapter: PaperAdapter, tmp_path: Path):
    _feed_bars(adapter)
    runner = make_runner(adapter, tmp_path)
    runner.start()
    try:
        runner.kill.engage("manual stop", by="test")
        runner.tick()
        runner.worker.stop()
        assert not adapter.positions("EURUSD")
        assert runner.journal.read("kill_engaged")
    finally:
        runner.shutdown()


def test_runner_state_survives_across_instances(adapter: PaperAdapter, tmp_path: Path):
    _feed_bars(adapter)
    first = make_runner(adapter, tmp_path)
    first.start()
    first.risk.book.observe_equity(97_000.0, today=TODAY)
    first.tick()
    first.shutdown()

    second = make_runner(adapter, tmp_path)
    second.start()
    try:
        assert second.risk.book.starting_equity == 100_000.0
        assert second.risk.book.day_start_equity == 100_000.0
    finally:
        second.shutdown()


def test_worker_survives_an_exception(tmp_path: Path):
    """A dead worker means a system that cannot close positions."""
    class Exploding(PaperAdapter):
        def submit(self, request):
            raise RuntimeError("boom")

    a = Exploding(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.08, 0.0001, NOW))
    journal = Journal(tmp_path / "j.jsonl")
    worker = ExecutionWorker(OrderManager(a, sleep=lambda _s: None), journal)
    worker.start()
    try:
        worker.submit(OrderRequest("EURUSD", Side.BUY, 0.1, stop_loss=1.07), "cid")
        worker._thread.join(timeout=2.0)
        assert worker._thread.is_alive(), "worker died on an exception"
        assert journal.read("worker_error")
    finally:
        worker.stop()


def test_stale_feed_halts_the_runner(tmp_path: Path):
    """Regression: the live loop never passed feed age to the risk snapshot, so
    FeedHeartbeat could never fire. Shadow mode on a closed market showed the
    runner evaluating 10-hour-old prices with halted=False."""
    a = PaperAdapter(FIXTURE_SPECS)
    a.connect()
    a.feed_tick(make_tick("EURUSD", 1.08, 0.0001, NOW - timedelta(hours=10)))
    _feed_bars(a)
    runner = make_runner(a, tmp_path)
    runner.start()
    try:
        runner.tick()
        assert runner.halted, "a 10-hour-stale feed did not halt the runner"
        assert any(b["limit"] == "feed_heartbeat" for b in runner.journal.read("breach"))
        beats = runner.journal.read("heartbeat")
        assert beats and beats[-1]["feed_age_s"] > 30_000
    finally:
        runner.shutdown()
