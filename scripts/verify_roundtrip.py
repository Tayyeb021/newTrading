"""Phase 1 gate: open, modify and close one real order end to end.

This is the last thing standing between the foundation and a working system. It
places an actual order at the broker's minimum lot, moves its stop, closes it, and
verifies the account is flat afterwards. Along the way it measures the numbers you
will otherwise be guessing at for months: real fill latency, real slippage, and
whether stop modification actually takes.

    python scripts/verify_roundtrip.py --symbol EURUSD
    python scripts/verify_roundtrip.py --symbol EURUSD --dry-run   # no order sent

SAFETY. The script refuses to run on anything other than a demo account. That is
not a formality — it opens a position, and a bug in the close path leaves it open.
Run it on demo, read the report, and only then decide anything about real money.
The `--allow-live` flag exists because you will eventually want to measure live
execution, and when you use it you should have read this file first.
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--profile", default="challenge")
    ap.add_argument("--dry-run", action="store_true", help="check everything, send nothing")
    ap.add_argument("--allow-live", action="store_true", help="permit a non-demo account")
    ap.add_argument("--stop-atr-multiple", type=float, default=None)
    args = ap.parse_args()

    from execution.mt5_adapter import MT5Adapter

    instruments = InstrumentConfig.load()
    profile = RiskProfile.load(args.profile)
    report = Report()

    print(f"\nROUND TRIP VERIFICATION - {args.symbol}")
    print(f"profile {profile.name}, {'DRY RUN' if args.dry_run else 'LIVE ORDER'}\n")

    adapter = MT5Adapter(aliases=instruments.aliases)

    # ---------------------------------------------------------------- connect
    try:
        _, ms = timed(adapter.connect)
        report.add("connect to terminal", True, adapter.name, ms)
    except Exception as exc:  # noqa: BLE001
        report.add("connect to terminal", False, str(exc))
        return 1

    try:
        return _run(adapter, args, profile, report)
    finally:
        adapter.disconnect()


def _run(adapter, args, profile, report: Report) -> int:
    symbol = args.symbol
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
        _summary(report)
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

    drift = reconcile(adapter, {})
    report.add("final reconciliation", not drift, "clean" if not drift else f"DRIFT {drift}")

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
    print()
    passed = sum(1 for s in report.steps if s.ok)
    print(f"  {passed}/{len(report.steps)} checks passed")

    if report.ok:
        print("\n  PHASE 1 GATE: round trip verified.")
        if spec is not None and volume:
            print(
                f"  Real cost of this trade: "
                f"{spec.risk_for(volume, stop_distance):,.2f} at risk on {volume:g} lots."
            )
        print("  Record the slippage figures above - they calibrate the backtest cost model.")
    else:
        failed = [s.name for s in report.steps if not s.ok]
        print(f"\n  GATE NOT MET. Failed: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
