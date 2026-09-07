"""Phase 1 gate: open, modify and close real orders end to end.

This is the last thing standing between the foundation and a working system. It
places an actual order at the broker's minimum lot, moves its stop, closes it, and
verifies the account is flat afterwards. Along the way it measures the numbers you
will otherwise be guessing at for months: real fill latency, real slippage, and
whether stop modification actually takes.

    python scripts/verify_roundtrip.py                             # every active instrument
    python scripts/verify_roundtrip.py --symbols EURUSD XAUUSD
    python scripts/verify_roundtrip.py --all                       # every configured instrument
    python scripts/verify_roundtrip.py --dry-run                   # no order sent

Each measured round trip is appended to `config/measured_fills.json`, which
`CostModel.calibrate()` reads. That file is the whole point of the exercise: until
it exists every backtest in this repository runs on an ASSUMED half-spread of
slippage, and `calibrated=False` is printed on every report to say so.

SAFETY. The script refuses to run on anything other than a demo account. That is
not a formality — it opens a position, and a bug in the close path leaves it open.
Each symbol is traded one at a time and closed before the next begins, so at most
one position exists at any moment. Run it on demo, read the report, and only then
decide anything about real money. The `--allow-live` flag exists because you will
eventually want to measure live execution, and when you use it you should have
read this file first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig, RiskProfile  # noqa: E402
from core.types import OrderRequest, Side  # noqa: E402
from execution.base import reconcile  # noqa: E402
from risk.sizing import size_position  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
MEASURED = Path(__file__).resolve().parent.parent / "config" / "measured_fills.json"


def record_fill(symbol: str, entry_spread: float, entry_slip: float, exit_slip: float,
                path: Path = MEASURED) -> None:
    """Append one measured round trip, in the shape `CostModel.calibrate` reads.

    Two observations per trip: the entry and the exit each crossed the spread and
    each slipped. Both are kept - an exit under stress is the one that hurts, and
    a median over one direction only would flatter the model.
    """
    store = {}
    if path.exists():
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            store = {}
    fills = store.setdefault(symbol, [])
    stamp = datetime.now(timezone.utc).isoformat()
    fills.append({"ts": stamp, "spread": entry_spread, "slippage": entry_slip, "leg": "entry"})
    fills.append({"ts": stamp, "spread": entry_spread, "slippage": exit_slip, "leg": "exit"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class Step:
    name: str
    ok: bool
    detail: str = ""
    millis: float = 0.0

    def __str__(self) -> str:
        mark = PASS if self.ok else FAIL
        timing = f"{self.millis:>8.0f}ms" if self.millis else " " * 10
        return f"  [{mark}] {self.name:<34}{timing}  {self.detail}"


@dataclass
class Report:
    steps: list[Step] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", millis: float = 0.0) -> Step:
        step = Step(name, ok, detail, millis)
        self.steps.append(step)
        print(step, flush=True)
        return step

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="instruments to verify; default is the active list in instruments.yaml")
    ap.add_argument("--symbol", default=None, help="one instrument (kept for older invocations)")
    ap.add_argument("--all", action="store_true", help="every instrument in instruments.yaml, not just the active ones")
    ap.add_argument("--profile", default="challenge")
    ap.add_argument("--dry-run", action="store_true", help="check everything, send nothing")
    ap.add_argument("--allow-live", action="store_true", help="permit a non-demo account")
    ap.add_argument("--stop-atr-multiple", type=float, default=None)
    args = ap.parse_args()

    from execution.mt5_adapter import MT5Adapter

    instruments = InstrumentConfig.load()
    profile = RiskProfile.load(args.profile)

    if args.symbol and args.symbols:
        print("give --symbol or --symbols, not both")
        return 1
    symbols = args.symbols or ([args.symbol] if args.symbol else None)
    if symbols is None:
        symbols = list(instruments.symbols if args.all else instruments.active)
    unknown = [s for s in symbols if s not in instruments.symbols]
    if unknown:
        print(f"unknown instruments {unknown}; configured: {', '.join(instruments.symbols)}")
        return 1

    print(f"\nROUND TRIP VERIFICATION - {len(symbols)} instrument(s): {', '.join(symbols)}")
    print(f"profile {profile.name}, {'DRY RUN - nothing is sent' if args.dry_run else 'LIVE ORDERS on a demo account'}")
    print("one position at a time; each is closed before the next opens")

    adapter = MT5Adapter(aliases=instruments.aliases)
    connect = Report()
    print()
    try:
        _, ms = timed(adapter.connect)
        connect.add("connect to terminal", True, adapter.name, ms)
    except Exception as exc:  # noqa: BLE001
        connect.add("connect to terminal", False, str(exc))
        return 1

    results: dict[str, tuple[bool, Report]] = {}
    try:
        for symbol in symbols:
            report = Report()
            print(f"\n{'=' * 78}\n{symbol}\n{'=' * 78}")
            try:
                code = _run(adapter, symbol, args, profile, report)
            except Exception as exc:  # noqa: BLE001 - one bad instrument must not abort the rest
                report.add("unhandled error", False, f"{type(exc).__name__}: {exc}")
                code = 1
            results[symbol] = (code == 0, report)
    finally:
        adapter.disconnect()

    return _overall(results, args.dry_run)


def _overall(results: dict, dry_run: bool) -> int:
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for symbol, (ok, report) in results.items():
        failed = [s.name for s in report.steps if not s.ok]
        passed = sum(1 for s in report.steps if s.ok)
        detail = "" if ok else "  failed: " + ", ".join(failed)
        print(f"  [{PASS if ok else FAIL}] {symbol:<10}{passed}/{len(report.steps)} checks{detail}")

    every = all(ok for ok, _ in results.values())
    print()
    if every and not dry_run:
        print(f"  PHASE 1 GATE MET on {len(results)} instrument(s).")
        print(f"  Measurements appended to {MEASURED.relative_to(MEASURED.parent.parent)}.")
        print("  Next: python scripts/calibrate_costs.py   (turns them into the cost model)")
    elif every:
        print("  DRY RUN clean. Re-run without --dry-run to measure real fills.")
    else:
        print("  GATE NOT MET. See the failures above before trusting anything downstream.")
    print()
    return 0 if every else 1


def _run(adapter, symbol: str, args, profile, report: Report) -> int:
    mt5 = adapter.mt5

    # ------------------------------------------------------------ safety gate
    info = mt5.account_info()
    if info is None:
        report.add("read account", False, "account_info returned None")
        return 1

    is_demo = int(info.trade_mode) == int(mt5.ACCOUNT_TRADE_MODE_DEMO)
    # MT5: ACCOUNT_TRADE_MODE_DEMO=0, CONTEST=1, REAL=2. The first version had this
    # map backwards; the guard below always used the constant and was correct.
    mode = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(int(info.trade_mode), "UNKNOWN")
    report.add(
        "account is demo",
        is_demo or args.allow_live,
        f"{mode} #{info.login} @ {info.server}, equity {info.equity:,.2f} {info.currency}",
    )
    if not is_demo and not args.allow_live:
        print(
            f"\n  Refusing to place an order on a {mode} account.\n"
            f"  Run against demo, or pass --allow-live if you have read this script.\n"
        )
        return 1

    terminal = mt5.terminal_info()
    trade_allowed = bool(terminal.trade_allowed) if terminal else False
    report.add(
        "algo trading enabled",
        trade_allowed,
        "on" if trade_allowed else "enable 'Algo Trading' in the terminal toolbar",
    )
    if not trade_allowed:
        return 1

    # ------------------------------------------------------------------ specs
    try:
        spec, ms = timed(lambda: adapter.spec(symbol))
        report.add(
            "load contract spec", True,
            f"min {spec.volume_min:g} step {spec.volume_step:g} "
            f"${spec.value_per_price_unit:,.2f}/1.0 move, stops_level {spec.stops_level_points}",
            ms,
        )
    except Exception as exc:  # noqa: BLE001
        report.add("load contract spec", False, str(exc))
        return 1

    try:
        tick, ms = timed(lambda: adapter.tick(symbol))
        age = (datetime.now(timezone.utc) - tick.ts).total_seconds()
        report.add(
            "read tick", True,
            f"bid {tick.bid:.{spec.digits}f} ask {tick.ask:.{spec.digits}f} "
            f"spread {tick.spread / spec.point:.1f} pts, {age:.0f}s old",
            ms,
        )
    except Exception as exc:  # noqa: BLE001
        report.add("read tick", False, str(exc))
        return 1

    # ------------------------------------------------------------------ sizing
    multiple = args.stop_atr_multiple or profile.atr_stop_multiple
    atr = _recent_atr(adapter, symbol, profile.atr_period)
    stop_distance = max(atr * multiple, spec.min_stop_distance * 2 or tick.spread * 20)

    account = adapter.account()
    size = size_position(spec, account.equity, profile.risk_per_trade, stop_distance)
    report.add(
        "size position",
        size.tradeable,
        f"{size.volume:g} lots, stop {stop_distance:.{spec.digits}f} "
        f"({stop_distance / spec.point:.0f} pts), risking {size.risk_fraction:.3%}"
        if size.tradeable
        else size.reason,
    )
    if not size.tradeable:
        return 1

    # Deliberately trade the smallest legal size, not the sized one. The point of
    # this script is to exercise the path, not to take a position.
    volume = spec.volume_min
    entry_ref = tick.ask
    stop = spec.normalize_price(entry_ref - stop_distance)
    target = spec.normalize_price(entry_ref + stop_distance * 2)

    if args.dry_run:
        report.add(
            "DRY RUN - no order sent", True,
            f"would buy {volume:g} {symbol} @ ~{entry_ref:.{spec.digits}f}, "
            f"sl {stop:.{spec.digits}f} tp {target:.{spec.digits}f}",
        )
        return 0 if report.ok else 1

    # ------------------------------------------------------------------- open
    request = OrderRequest(
        symbol=symbol, side=Side.BUY, volume=volume,
        stop_loss=stop, take_profit=target, comment="roundtrip",
    )
    result, ms = timed(lambda: adapter.submit(request))
    slip = result.slippage()
    report.add(
        "open position",
        result.ok,
        f"ticket {result.ticket} @ {result.fill_price:.{spec.digits}f}, "
        f"slippage {slip / spec.point:+.1f} pts"
        if result.ok
        else result.reason,
        ms,
    )
    if not result.ok:
        return 1
    ticket = result.ticket

    try:
        positions = adapter.positions(symbol)
        found = [p for p in positions if p.ticket == ticket]
        report.add(
            "position visible at broker",
            bool(found),
            f"{found[0].volume:g} lots, sl {found[0].stop_loss}" if found else "not found",
        )

        stop_attached = bool(found and found[0].stop_loss)
        report.add(
            "stop loss attached",
            stop_attached,
            f"{found[0].stop_loss:.{spec.digits}f}" if stop_attached
            else "NO STOP - broker rejected it; the position has undefined risk",
        )

        # ---------------------------------------------------------- modify
        new_stop = spec.normalize_price(stop + stop_distance * 0.25)
        mod, ms = timed(lambda: adapter.modify(ticket, stop_loss=new_stop))
        report.add("modify stop loss", mod.ok, mod.reason or f"-> {new_stop:.{spec.digits}f}", ms)

        after = [p for p in adapter.positions(symbol) if p.ticket == ticket]
        moved = bool(after and after[0].stop_loss and abs(after[0].stop_loss - new_stop) < spec.point * 2)
        report.add(
            "modification took effect",
            moved,
            f"{after[0].stop_loss:.{spec.digits}f}" if after and after[0].stop_loss else "unchanged",
        )

        # --------------------------------------------------- reconciliation
        drift = reconcile(adapter, {symbol: volume})
        report.add(
            "reconcile against expectation",
            not drift,
            "in sync" if not drift else f"DRIFT {drift}",
        )
    finally:
        # ----------------------------------------------------------- close
        closed, ms = timed(lambda: adapter.close(ticket))
        cslip = closed.slippage()
        report.add(
            "close position",
            closed.ok,
            f"@ {closed.fill_price:.{spec.digits}f}, slippage {cslip / spec.point:+.1f} pts"
            if closed.ok
            else closed.reason,
            ms,
        )

    remaining = [p for p in adapter.positions(symbol) if p.ticket == ticket]
    report.add("account is flat", not remaining, "no open position" if not remaining else "STILL OPEN")

    # reconcile() unions expected with actual, so this checks the WHOLE account is
    # flat, not just this symbol. That is what we want between instruments.
    drift = reconcile(adapter, {})
    report.add("account reconciles flat", not drift, "clean" if not drift else f"DRIFT {drift}")

    # The measurement this whole script exists to take.
    if result.ok and closed.ok:
        record_fill(symbol, entry_spread=tick.spread, entry_slip=slip or 0.0, exit_slip=cslip or 0.0)
        report.add(
            "record measurement", True,
            f"spread {tick.spread / spec.point:.1f} pts, entry slip {(slip or 0.0) / spec.point:+.1f}, "
            f"exit slip {(cslip or 0.0) / spec.point:+.1f} -> {MEASURED.name}",
        )
    _summary(report, spec, volume, stop_distance)
    return 0 if report.ok else 1


def _recent_atr(adapter, symbol: str, period: int) -> float:
    """True range average over recent daily bars. Enough for a sane stop here."""
    bars = adapter.bars(symbol, "D1", count=period + 1)
    if len(bars) < 2:
        raise RuntimeError(f"not enough daily bars for {symbol} to compute ATR")
    trs = []
    for prev, cur in zip(bars, bars[1:]):
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    return sum(trs) / len(trs)


def _summary(report: Report, spec=None, volume: float = 0.0, stop_distance: float = 0.0) -> None:
    passed = sum(1 for s in report.steps if s.ok)
    line = f"  {passed}/{len(report.steps)} checks passed"
    if report.ok and spec is not None and volume:
        line += f", {spec.risk_for(volume, stop_distance):,.2f} was at risk on {volume:g} lots"
    if not report.ok:
        line += " - FAILED: " + ", ".join(s.name for s in report.steps if not s.ok)
    print(line)


if __name__ == "__main__":
    raise SystemExit(main())
