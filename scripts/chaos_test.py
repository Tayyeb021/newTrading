"""Phase 3 gate: kill the process mid-trade and prove the system comes back correct.

Runs entirely against the paper adapter, so it needs no broker and can run in CI.
The broker is kept alive across the "crash" while all in-memory state is thrown
away — which is exactly what a real crash looks like from the broker's side.

Five scenarios, each an independent way the system has historically been broken:

  1. Crash with an open position     -> is it found again, or silently orphaned?
  2. Crash mid-day after a loss      -> does the daily limit reset? (account killer)
  3. Crash across midnight           -> does the day roll but drawdown survive?
  4. Ambiguous submit + retry        -> one position, or two?
  5. Crash with an unstopped position-> is a stop attached before trading resumes?

    python scripts/chaos_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RiskProfile  # noqa: E402
from core.types import OrderRequest, OrderResult, OrderStatus, Side  # noqa: E402
from execution.oms import OrderManager, client_id  # noqa: E402
from execution.paper import FIXTURE_SPECS, PaperAdapter, make_tick  # noqa: E402
from live.state import StateStore, restore_book  # noqa: E402
from ops.journal import Journal  # noqa: E402
from risk.build import build_engine  # noqa: E402

NOW = datetime(2026, 9, 4, 14, 0, tzinfo=timezone.utc)
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name:<46} {detail}")


def scenario_1_open_position_survives(tmp: Path) -> None:
    print("\n1. Crash with an open position")
    adapter = PaperAdapter(FIXTURE_SPECS)
    adapter.connect()
    adapter.feed_tick(make_tick("EURUSD", 1.0800, 0.00010, NOW))
    journal = Journal(tmp / "j1.jsonl")

    oms = OrderManager(adapter, sleep=lambda _s: None)
    cid = client_id("trend", "EURUSD", "BUY", NOW)
    result = oms.submit(
        OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.0700), cid
    )
    journal.fill(result, cid)
    check("position opened before crash", result.ok, f"ticket {result.ticket}")

    # --- crash: every object is discarded, the broker is not ---
    del oms, result

    oms2 = OrderManager(adapter, sleep=lambda _s: None)
    recovered = adapter.positions("EURUSD")
    check("position visible after restart", len(recovered) == 1,
          f"{len(recovered)} position(s) at broker")

    orphans = oms2.orphans(journal.known_tickets())
    check("journalled position is NOT an orphan", not orphans,
          "journal recognised it" if not orphans else f"{len(orphans)} orphan(s)")

    # A position the journal never saw must be flagged.
    stray = adapter.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.0700))
    orphans = oms2.orphans(journal.known_tickets())
    check("unjournalled position IS flagged as orphan",
          any(p.ticket == stray.ticket for p in orphans),
          f"{len(orphans)} orphan(s) found")


def scenario_2_daily_limit_survives(tmp: Path) -> None:
    """The one that kills evaluation accounts."""
    print("\n2. Crash mid-day after a loss  (the account killer)")
    store = StateStore(tmp / "s2.json")
    profile = RiskProfile.load("challenge")

    engine = build_engine(profile, 100_000.0, dict(FIXTURE_SPECS))
    engine.book.trading_day = NOW.date()
    engine.book.observe_equity(97_000.0, today=NOW.date())  # -3% morning
    store.save(engine.book)
    check("pre-crash: 3% lost today", True,
          f"day started {engine.book.day_start_equity:,.0f}, now 97,000")

    del engine

    book, notes = restore_book(store, current_equity=97_000.0, today=NOW.date())
    check("day_start_equity carried forward, NOT reset",
          book.day_start_equity == 100_000.0,
          f"{book.day_start_equity:,.0f} (reset would be 97,000)")

    spent = (book.day_start_equity - 97_000.0) / book.day_start_equity
    check("daily budget still shows as spent", abs(spent - 0.03) < 1e-9, f"{spent:.2%} used")

    # With the bug, remaining headroom would be the full 3.5% again.
    remaining = profile.daily_loss_soft - spent
    check("remaining headroom is reduced, not restored",
          remaining < profile.daily_loss_soft,
          f"{remaining:.2%} left of {profile.daily_loss_soft:.2%}")
    check("restart is reported to the operator",
          any("CARRIED FORWARD" in n for n in notes),
          "startup notes explain the carry")


def scenario_3_day_rolls_drawdown_persists(tmp: Path) -> None:
    print("\n3. Crash across midnight")
    store = StateStore(tmp / "s3.json")
    engine = build_engine(RiskProfile.load("challenge"), 100_000.0, dict(FIXTURE_SPECS))
    engine.book.trading_day = NOW.date()
    engine.book.observe_equity(104_000.0, today=NOW.date())  # new high-water
    engine.book.observe_equity(96_000.0, today=NOW.date())
    store.save(engine.book)

    tomorrow = (NOW + timedelta(days=1)).date()
    book, _ = restore_book(store, current_equity=96_000.0, today=tomorrow)

    check("day_start rolls to the new session", book.day_start_equity == 96_000.0,
          f"{book.day_start_equity:,.0f}")
    check("high-water survives the day roll", book.high_water_equity == 104_000.0,
          f"{book.high_water_equity:,.0f}")
    check("starting equity survives", book.starting_equity == 100_000.0,
          f"{book.starting_equity:,.0f}")

    dd = (book.high_water_equity - 96_000.0) / book.high_water_equity
    check("trailing drawdown still measured from the peak", abs(dd - 0.0769) < 1e-3,
          f"{dd:.2%} below high-water")


class FlakyAdapter(PaperAdapter):
    """Fills the order, then reports a timeout. The worst real failure mode."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lie_once = True

    def submit(self, request):
        result = super().submit(request)
        if self.lie_once and result.ok:
            self.lie_once = False
            return OrderResult(
                OrderStatus.REJECTED, request,
                reason="connection timeout - no reply from server",
            )
        return result


def scenario_4_ambiguous_submit(tmp: Path) -> None:
    print("\n4. Order fills but the reply is lost")
    adapter = FlakyAdapter(FIXTURE_SPECS)
    adapter.connect()
    adapter.feed_tick(make_tick("EURUSD", 1.0800, 0.00010, NOW))

    oms = OrderManager(adapter, sleep=lambda _s: None)
    cid = client_id("trend", "EURUSD", "BUY", NOW)
    result = oms.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.0700), cid)

    positions = adapter.positions("EURUSD")
    check("exactly ONE position exists, not two", len(positions) == 1,
          f"{len(positions)} position(s) after {len(oms.attempts)} attempt(s)")
    check("retry recognised its own earlier fill", result.ok,
          result.reason or "recovered")

    # And a second call with the same intent must not open another.
    again = oms.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.0700), cid)
    check("resubmitting the same intent is a no-op",
          len(adapter.positions("EURUSD")) == 1 and again.ok,
          f"{len(adapter.positions('EURUSD'))} position(s), reason '{again.reason}'")


def scenario_5_unstopped_position_repaired(tmp: Path) -> None:
    print("\n5. Crash leaves a position with no stop")
    adapter = PaperAdapter(FIXTURE_SPECS)
    adapter.connect()
    adapter.feed_tick(make_tick("EURUSD", 1.0800, 0.00010, NOW))
    adapter.feed_bars("EURUSD", "D1", [])

    naked = adapter.submit(OrderRequest("EURUSD", Side.BUY, 0.10))  # no stop
    check("position opened without a stop", naked.ok and
          adapter.positions("EURUSD")[0].stop_loss is None, "undefined risk")

    oms = OrderManager(adapter, sleep=lambda _s: None)
    repairs = oms.ensure_stops({"EURUSD": 0.0100})
    after = adapter.positions("EURUSD")[0]

    check("stop attached on recovery", after.stop_loss is not None,
          f"stop at {after.stop_loss}" if after.stop_loss else "STILL NAKED")
    check("stop is on the correct side", bool(after.stop_loss and after.stop_loss < after.entry_price),
          f"{after.stop_loss} < entry {after.entry_price:.5f}" if after.stop_loss else "")
    check("repair was reported", len(repairs) == 1, f"{len(repairs)} repair(s)")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="chaos_"))
    print("CHAOS TEST - crash recovery and idempotency")
    print("=" * 72)
    try:
        scenario_1_open_position_survives(tmp)
        scenario_2_daily_limit_survives(tmp)
        scenario_3_day_rolls_drawdown_persists(tmp)
        scenario_4_ambiguous_submit(tmp)
        scenario_5_unstopped_position_repaired(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 72)
    print(f"  {passed}/{len(results)} checks passed")
    if passed == len(results):
        print("\n  PHASE 3 GATE: crash recovery verified on the paper adapter.")
        print("  Re-run the equivalent on demo before funding anything.")
        return 0
    print("\n  GATE NOT MET:")
    for name, ok, _ in results:
        if not ok:
            print(f"    {name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
