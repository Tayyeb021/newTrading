"""The live trading loop.

Startup order is not arbitrary; each step depends on the one before:

1. connect, read equity
2. **restore the session book** — carrying today's daily loss forward if the
   process is restarting mid-session (see `live.state`)
3. **reconcile against the broker** — adopt or report orphan positions
4. **repair missing stops** — an unstopped position makes every aggregate limit
   wrong, so nothing may trade until it is fixed
5. only then, start trading

`order_send` blocks, so it runs on a worker thread behind a queue. The loop never
waits on the broker: a slow fill on gold during a news spike must not stall the
ingest of everything else. The MT5 adapter's docstring promised this wiring; here
it is.

Shutdown is graceful on SIGINT/SIGTERM: stop accepting signals, let the worker
drain, save state, disconnect. Positions are *not* closed on shutdown — stopping
the software is not the same as wanting to be flat, and silently liquidating on
Ctrl+C would be a nasty surprise. Use the kill switch or `flatten_all.py` for that.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.sleeve import Sleeve, sleeve_of
from core.strategy import Strategy
from core.types import OrderRequest, OrderResult, Position, Side, Signal, SymbolSpec
from execution.base import ExecutionAdapter
from execution.oms import OrderManager, client_id
from live.state import StateStore, restore_book
from ops.journal import Journal
from risk.engine import RiskEngine
from risk.killswitch import KillFile
from risk.limits import Severity
from risk.voltarget import Allocation, LegIntent, VolTarget

log = logging.getLogger(__name__)


@dataclass
class Decided:
    """One leg's view on its newly closed bar, before the book is sized."""

    sleeve: str
    symbol: str
    strategy: Strategy
    intent: object
    held_all: list[Position]
    last_ts: datetime


@dataclass
class Job:
    kind: str  # "submit" | "close_all" | "close_ticket" | "modify"
    request: OrderRequest | None = None
    cid: str = ""
    ticket: int | None = None
    volume: float | None = None  # close_ticket: partial close of this much; None = all
    stop: float | None = None  # modify: the new stop, only ever tighter


@dataclass
class ExecutionWorker:
    """Runs blocking broker calls off the event loop."""

    oms: OrderManager
    journal: Journal
    inbox: "queue.Queue[Job | None]" = field(default_factory=queue.Queue)
    results: "queue.Queue[OrderResult]" = field(default_factory=queue.Queue)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="execution", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self.inbox.put(None)
        if self._thread is not None:
            self._thread.join(timeout)

    def submit(self, request: OrderRequest, cid: str) -> None:
        self.inbox.put(Job("submit", request, cid))

    def close_all(self) -> None:
        self.inbox.put(Job("close_all"))

    def close_ticket(self, ticket: int, volume: float | None = None) -> None:
        """Close one position, or part of it. A strategy exiting its own trade
        must not flatten every other strategy's positions too."""
        self.inbox.put(Job("close_ticket", ticket=ticket, volume=volume))

    def modify_stop(self, ticket: int, stop: float) -> None:
        """Ratchet a stop. Callers only ever pass a tighter level."""
        self.inbox.put(Job("modify", ticket=ticket, stop=stop))

    def drain(self) -> list[OrderResult]:
        out: list[OrderResult] = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        while True:
            job = self.inbox.get()
            if job is None:
                return
            try:
                if job.kind == "submit" and job.request is not None:
                    result = self.oms.submit(job.request, job.cid)
                    self.journal.fill(result, job.cid)
                    self.results.put(result)
                elif job.kind == "close_all":
                    for result in self.oms.close_all():
                        self.journal.fill(result, "flatten")
                        self.results.put(result)
                elif job.kind == "close_ticket" and job.ticket is not None:
                    result = self.oms.adapter.close(job.ticket, job.volume)
                    self.journal.fill(result, f"{'trim' if job.volume else 'exit'}#{job.ticket}")
                    self.results.put(result)
                elif job.kind == "modify" and job.ticket is not None:
                    result = self.oms.adapter.modify(job.ticket, stop_loss=job.stop)
                    self.journal.write("stop_ratchet", ticket=job.ticket, stop=job.stop,
                                       status=result.status.value, reason=result.reason)
            except Exception as exc:  # noqa: BLE001 - the worker must never die
                log.exception("execution worker error: %s", exc)
                self.journal.write("worker_error", error=str(exc), job=job.kind)


class Runner:
    def __init__(
        self,
        adapter: ExecutionAdapter,
        risk: RiskEngine,
        strategies: dict[str, Strategy] | None = None,
        specs: dict[str, SymbolSpec] | None = None,
        timeframe: str = "D1",
        sleeves: list[Sleeve] | None = None,
        poll_seconds: float = 5.0,
        state: StateStore | None = None,
        journal: Journal | None = None,
        kill: KillFile | None = None,
        allocator: VolTarget | None = None,
    ) -> None:
        self.adapter = adapter
        self.risk = risk
        self.specs = specs or {}
        self.timeframe = timeframe
        #: Entry 011: with an allocator, every leg decides first and the book is
        #: sized once, to its volatility target, before anything is sent. The
        #: allocator's window is in daily bars, so it only serves daily legs.
        self.allocator = allocator
        self._fed: dict[str, datetime] = {}
        # Legs: one (sleeve, symbol, strategy) each. A legacy symbol->strategy
        # dict becomes a single sleeve named "default".
        self.legs: list[tuple[str, str, Strategy, str]] = []
        if sleeves:
            for sl in sleeves:
                for sym in sl.symbols:
                    self.legs.append((sl.name, sym, sl.build(sym), sl.timeframe))
        elif strategies:
            for sym, strat in strategies.items():
                strat.name = "default"
                self.legs.append(("default", sym, strat, timeframe))
        self.strategies = {sym: strat for _, sym, strat, _ in self.legs}
        if self.allocator is not None:
            bad = [(s, sym, tf) for s, sym, _, tf in self.legs if tf != "D1"]
            if bad:
                raise ValueError(f"volatility targeting needs daily legs; these are not: {bad}")
        self.poll_seconds = poll_seconds
        self.state = state or StateStore()
        self.journal = journal or Journal()
        self.kill = kill or KillFile()

        self.oms = OrderManager(adapter)
        self.worker = ExecutionWorker(self.oms, self.journal)
        self._stop = threading.Event()
        self._last_bar: dict[tuple[str, str], datetime] = {}
        self.halted = False

    # ------------------------------------------------------------------ startup

    def start(self) -> list[str]:
        """Connect, restore, reconcile, repair. Returns the startup notes."""
        notes: list[str] = []
        self.adapter.connect()
        account = self.adapter.account()
        notes.append(f"connected: equity {account.equity:,.2f} {account.currency}")

        book, state_notes = restore_book(self.state, account.equity)
        self.risk.book = book
        notes.extend(state_notes)

        known = self.journal.known_tickets()
        orphans = self.oms.orphans(known)
        if orphans:
            for pos in orphans:
                notes.append(
                    f"ORPHAN: {pos.symbol} {pos.side.name} {pos.volume:g} "
                    f"ticket {pos.ticket} - opened by a process that did not journal it"
                )
                self.journal.write(
                    "orphan_adopted", symbol=pos.symbol, ticket=pos.ticket,
                    side=pos.side.name, volume=pos.volume, stop_loss=pos.stop_loss,
                )
        else:
            notes.append("no orphan positions")

        unstopped = [p for p in self.adapter.positions() if p.stop_loss is None]
        if unstopped:
            notes.append(f"{len(unstopped)} position(s) missing a stop - repairing before trading")
            defaults = self._default_stop_distances()
            for result in self.oms.ensure_stops(defaults):
                notes.append(f"  stop repair: {result.status.value} {result.reason}")

        if self.kill.engaged():
            record = self.kill.read()
            self.risk.book.kill(record.reason)
            notes.append(str(record))

        if self.allocator is not None:
            notes.append(self._seed_allocator())

        self.worker.start()
        self.journal.write("startup", notes=notes, equity=account.equity)
        return notes

    def _seed_allocator(self) -> str:
        """Fill the allocator's window from bar history, so a restart is not blind."""
        seeded = 0
        for symbol, spec in self.specs.items():
            try:
                bars = self.adapter.bars(symbol, "D1", count=self.allocator.window + 2)
            except Exception:  # noqa: BLE001
                continue
            closed = bars[:-1]
            for prev, cur in zip(closed, closed[1:]):
                self.allocator.observe(cur.ts.date(), {symbol: (cur.close - prev.close) * spec.value_per_price_unit})
            if closed:
                self._fed[symbol] = closed[-1].ts
            seeded += 1
        ready = sum(1 for s in self.specs if self.allocator.ready(s))
        return f"volatility target {self.allocator.target:.0%}: history seeded for {seeded} symbols, {ready} ready"

    def _default_stop_distances(self) -> dict[str, float]:
        """Fallback stops for repair: 2x the last daily range. Crude on purpose.

        This is emergency risk containment, not a strategy decision. A crude stop
        that exists beats an elegant one that does not.
        """
        out: dict[str, float] = {}
        for symbol in self.specs:
            try:
                bars = self.adapter.bars(symbol, "D1", count=2)
                out[symbol] = 2.0 * (bars[-1].high - bars[-1].low)
            except Exception:  # noqa: BLE001
                continue
        return out

    # --------------------------------------------------------------------- loop

    def run(self, max_iterations: int | None = None) -> None:
        self._install_signal_handlers()
        iteration = 0
        try:
            while not self._stop.is_set():
                if max_iterations is not None and iteration >= max_iterations:
                    break
                iteration += 1
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - a loop that dies stops managing risk
                    log.exception("loop error: %s", exc)
                    self.journal.write("loop_error", error=str(exc))
                self._stop.wait(self.poll_seconds)
        finally:
            self.shutdown()

    def tick(self) -> None:
        """One iteration. Safe to call directly from a test."""
        now = datetime.now(timezone.utc)

        # 1. kill switch, before anything else
        if self.kill.engaged() and not self.risk.book.killed:
            record = self.kill.read()
            self.risk.book.kill(record.reason)
            self.journal.write("kill_engaged", reason=record.reason, by=record.by)
            self.worker.close_all()

        # 2. account state and the account-level limits
        account = self.adapter.account()
        self.risk.book.observe_equity(account.equity, today=now.date())
        positions = self.adapter.positions()

        # Feed age from the freshest tick across every symbol. Without this the
        # FeedHeartbeat limit sits in the register and can never fire, and the
        # runner will happily evaluate Friday's prices all weekend. Found by
        # running shadow mode on a closed market and watching halted stay False.
        ticks = self._ticks()
        feed_age = (
            min((now - t.ts).total_seconds() for t in ticks.values())
            if ticks else float("inf")
        )
        state = self.risk.snapshot(
            equity=account.equity, balance=account.balance,
            margin_level=account.margin_level, positions=positions, now=now,
            current_price={s: t.mid for s, t in ticks.items()},
            current_spread={s: t.spread for s, t in ticks.items()},
            feed_age_seconds=feed_age,
        )
        breaches = self.risk.check_account(state)
        for breach in breaches:
            self.journal.breach(breach)

        if any(b.severity is Severity.FLATTEN for b in breaches):
            self.halted = True
            self.worker.close_all()
            self.kill.engage(f"auto: {breaches[0].message}", by="risk_engine")
            self.risk.book.kill(breaches[0].message)
            self._persist()
            return

        self.halted = any(b.severity in (Severity.HALT, Severity.FLATTEN) for b in breaches)

        # 2b. futures: roll any position sitting in a contract past its roll date.
        #     Done before evaluation so the strategy sees the new contract's bars.
        if hasattr(self.adapter, "roll_due"):
            for symbol in self.specs:
                try:
                    if self.adapter.roll_due(symbol):
                        results = self.adapter.roll(symbol)
                        self.journal.write("roll", symbol=symbol,
                                           legs=[{"status": r.status.value, "fill": r.fill_price,
                                                  "volume": r.filled_volume, "comment": r.request.comment}
                                                 for r in results])
                except Exception as exc:  # noqa: BLE001
                    log.exception("roll failed for %s: %s", symbol, exc)
                    self.journal.write("roll_error", symbol=symbol, error=str(exc))

        # 3. strategy evaluation, only on a newly closed bar. Two passes: every
        #    leg decides, the book is sized once, then each leg acts.
        if not self.halted:
            decided: list[Decided] = []
            for sleeve_name, symbol, strategy, tf in self.legs:
                try:
                    item = self._intent(sleeve_name, symbol, strategy, tf, state)
                except Exception as exc:  # noqa: BLE001
                    log.exception("%s/%s: %s", sleeve_name, symbol, exc)
                    self.journal.write("strategy_error", sleeve=sleeve_name, symbol=symbol, error=str(exc))
                    continue
                if item is not None:
                    decided.append(item)
            plan = self._allocate(decided, state) if (self.allocator is not None and decided) else {}
            for item in decided:
                try:
                    self._act(item, state, now, plan.get((item.sleeve, item.symbol)))
                except Exception as exc:  # noqa: BLE001
                    log.exception("%s/%s: %s", item.sleeve, item.symbol, exc)
                    self.journal.write("strategy_error", sleeve=item.sleeve, symbol=item.symbol, error=str(exc))

        # 4. collect fills, record, persist
        for result in self.worker.drain():
            if result.ok:
                self.journal.write("fill_confirmed", ticket=result.ticket,
                                   symbol=result.request.symbol)

        self.journal.heartbeat(
            equity=account.equity, positions=len(positions),
            halted=self.halted, daily=state.daily_pnl_fraction,
            drawdown=state.drawdown_fraction(trailing=False),
            feed_age_s=round(feed_age, 1) if feed_age != float("inf") else None,
        )
        self._persist()

    def _intent(self, sleeve_name: str, symbol: str, strategy: Strategy, timeframe: str,
                state) -> "Decided | None":
        """What this leg wants on its newly closed bar, or None if there is none."""
        import pandas as pd

        bars = self.adapter.bars(symbol, timeframe, count=strategy.warmup + 60)
        if len(bars) < strategy.warmup + 2:
            return None

        # Only act on a CLOSED bar. The most recent bar is still forming, so it
        # is dropped - acting on a partial bar is live-trading's look-ahead bug.
        closed = bars[:-1]
        last_ts = closed[-1].ts
        if self._last_bar.get((sleeve_name, symbol)) == last_ts:
            return None
        self._last_bar[(sleeve_name, symbol)] = last_ts

        # the allocator learns each completed bar once, whichever leg saw it first
        if self.allocator is not None and self._fed.get(symbol) != last_ts and len(closed) >= 2:
            spec = self.specs.get(symbol)
            if spec is not None:
                self.allocator.observe(last_ts.date(), {
                    symbol: (closed[-1].close - closed[-2].close) * spec.value_per_price_unit})
            self._fed[symbol] = last_ts

        df = pd.DataFrame([{
            "ts": b.ts, "open": b.open, "high": b.high,
            "low": b.low, "close": b.close, "volume": b.volume,
        } for b in closed])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = strategy.prepare(df)

        # This sleeve's position on this symbol - another sleeve may hold the
        # same symbol, possibly the other way, and that is not ours to touch.
        held_all = [p for p in state.positions if p.symbol == symbol and sleeve_of(p) == sleeve_name]
        held = held_all[0] if held_all else None
        intent = strategy.evaluate(df, len(df) - 1, held)
        return Decided(sleeve_name, symbol, strategy, intent, held_all, last_ts)

    def _allocate(self, decided: list["Decided"], state) -> dict:
        """Size the whole book: legs deciding now are free, everything else held."""
        intents: list[LegIntent] = []
        deciding = {(d.sleeve, d.symbol): d for d in decided}
        for d in decided:
            spec = self.specs.get(d.symbol)
            if spec is None or d.intent.flat:
                continue
            held = d.held_all[0] if d.held_all else None
            sizing_now = held is None or d.intent.side is not held.side or (d.strategy.rebalances and d.intent.resize)
            if sizing_now:
                intents.append(LegIntent((d.sleeve, d.symbol), d.symbol, d.intent.side, d.intent.stop_distance,
                                         spec.value_per_price_unit))
        held_by_key: dict[tuple[str, str], list[Position]] = {}
        for p in state.positions:
            held_by_key.setdefault((sleeve_of(p), p.symbol), []).append(p)
        for key, ps in held_by_key.items():
            d = deciding.get(key)
            if d is not None and (d.intent.flat or d.intent.side is not ps[0].side
                                  or (d.strategy.rebalances and d.intent.resize)):
                continue  # closing, flipping, or being resized above
            spec = self.specs.get(key[1])
            if spec is None:
                continue
            volume = sum(p.volume for p in ps)
            stops = [abs(p.entry_price - p.stop_loss) for p in ps if p.stop_loss is not None]
            intents.append(LegIntent(key, key[1], ps[0].side, max(stops) if stops else 0.0,
                                     spec.value_per_price_unit, held_contracts=volume))
        plan = self.allocator.allocate(state.equity, intents)
        for key, alloc in plan.items():
            self.journal.write("allocation", sleeve=key[0], symbol=key[1], contracts=alloc.contracts,
                               risk_fraction=alloc.risk_fraction, note=alloc.note)
        return plan

    @staticmethod
    def _sizing(alloc: Allocation | None) -> tuple[float | None, float | None]:
        if alloc is None or alloc.risk_fraction is None:
            return None, None  # the engine's own risk per trade
        return alloc.risk_fraction, alloc.contracts

    def _act(self, item: "Decided", state, now: datetime, alloc: Allocation | None = None) -> None:
        sleeve_name, symbol, strategy, intent, held_all, last_ts = (
            item.sleeve, item.symbol, item.strategy, item.intent, item.held_all, item.last_ts)
        held = held_all[0] if held_all else None

        if intent.flat:
            for p in held_all:
                self.journal.write("exit_signal", symbol=symbol, ticket=p.ticket)
                self.worker.close_ticket(p.ticket)
            return
        if held is not None and held.side is intent.side:
            if strategy.rebalances and intent.resize:
                self._rebalance(sleeve_name, symbol, strategy, intent, held_all, state, now, last_ts, alloc)
            return

        spec = self.specs.get(symbol)
        if alloc is not None and alloc.risk_fraction is not None and spec is not None \
                and alloc.contracts < spec.volume_min:
            self.journal.write("vol_target_skip", symbol=symbol, sleeve=sleeve_name, note=alloc.note)
            return

        signal = Signal(
            symbol=symbol, side=intent.side, stop_distance=intent.stop_distance,
            confidence=intent.confidence, strategy=strategy.name, ts=now,
        )
        risk_fraction, max_volume = self._sizing(alloc)
        decision = self.risk.evaluate(signal, state, risk_fraction=risk_fraction, max_volume=max_volume)
        self.journal.decision(signal, decision, {
            "equity": state.equity,
            "daily": state.daily_pnl_fraction,
            "drawdown": state.drawdown_fraction(trailing=False),
            "positions": len(state.positions),
        })
        if not decision.approved:
            return

        # Re-check the kill switch immediately before submitting. The gap between
        # deciding and trading is exactly where a stop instruction arrives.
        if self.kill.engaged():
            self.journal.write("submit_aborted", symbol=symbol, reason="kill switch")
            return

        cid = client_id(strategy.name, symbol, intent.side.name, last_ts)
        self.worker.submit(decision.order, cid)

    def _rebalance(self, sleeve_name: str, symbol: str, strategy: Strategy, intent,
                   held: list[Position], state, now: datetime, last_ts,
                   alloc: Allocation | None = None) -> None:
        """A continuous strategy re-proposed its view on a position it holds.

        On a hedging venue several tickets can make up one leg; they are judged
        as one position and acted on newest-first. The decision itself is the
        risk engine's, exactly as in the backtester.
        """
        volume = sum(p.volume for p in held)
        average = sum(p.entry_price * p.volume for p in held) / volume
        side = held[0].side
        stops = [p.stop_loss for p in held if p.stop_loss is not None]
        # the loosest stop is the one that defines the risk on the table
        loosest = (min(stops) if side is Side.BUY else max(stops)) if stops else None
        whole = Position(
            symbol=symbol, side=side, volume=volume, entry_price=average, opened_at=held[0].opened_at,
            stop_loss=loosest, ticket=held[0].ticket, comment=held[0].comment,
        )
        signal = Signal(
            symbol=symbol, side=intent.side, stop_distance=intent.stop_distance,
            confidence=intent.confidence, strategy=strategy.name, ts=now,
        )
        risk_fraction, max_volume = self._sizing(alloc)
        decision = self.risk.resize(signal, state, whole, inertia=strategy.inertia,
                                    risk_fraction=risk_fraction, max_volume=max_volume)
        self.journal.write(
            "resize", symbol=symbol, sleeve=sleeve_name, action=decision.action, held=volume,
            target=decision.target_volume, delta=decision.delta, stop=decision.stop_loss,
            note=decision.note, breaches=[b.limit for b in decision.breaches],
        )

        if decision.stop_loss is not None:
            for p in held:
                looser = p.stop_loss is None or (
                    decision.stop_loss > p.stop_loss if side is Side.BUY else decision.stop_loss < p.stop_loss
                )
                if looser:
                    self.worker.modify_stop(p.ticket, decision.stop_loss)

        if decision.action == "reduce":
            remaining = -decision.delta
            for p in sorted(held, key=lambda p: p.opened_at, reverse=True):
                if remaining <= 1e-9:
                    break
                take = min(p.volume, remaining)
                self.worker.close_ticket(p.ticket, None if take >= p.volume - 1e-9 else take)
                remaining -= take
        elif decision.action == "increase":
            if self.kill.engaged():
                self.journal.write("submit_aborted", symbol=symbol, reason="kill switch")
                return
            # Its own client id: the OMS adopts by exact id, and the entry's id
            # would make it adopt the position it is meant to add to.
            cid = client_id(strategy.name, symbol, f"{intent.side.name}_ADD", last_ts)
            self.worker.submit(decision.order, cid)

    # ------------------------------------------------------------------ helpers

    def _ticks(self) -> dict:
        """One tick per symbol, fetched once per iteration and shared."""
        out = {}
        for symbol in self.specs:
            try:
                out[symbol] = self.adapter.tick(symbol)
            except Exception:  # noqa: BLE001
                continue
        return out

    def _persist(self) -> None:
        self.state.save(self.risk.book)

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.info("signal %s received; shutting down", signum)
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not on the main thread, e.g. under a test runner

    def shutdown(self) -> None:
        """Stop cleanly. Deliberately does NOT close positions."""
        self.worker.stop()
        self._persist()
        self.journal.write("shutdown", halted=self.halted)
        try:
            self.adapter.disconnect()
        except Exception:  # noqa: BLE001
            pass
