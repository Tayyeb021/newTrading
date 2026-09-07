"""The forward paper record: the monthly book, live, on Interactive Brokers.

    python scripts/run_forward.py --dry-run              # startup checks, no loop
    python scripts/run_forward.py                        # shadow: IB reads, paper fills
    python scripts/run_forward.py --live-paper           # orders to the IB paper account
    python scripts/run_forward.py --report               # what the record says so far

**Why this exists.** Thirteen research entries, 196 counted trials, and nothing
passed its declared thresholds. The best form found - monthly time-series
momentum across three speeds, net Sharpe 0.31 with carry - failed the 0.40 bar,
and the one variant that looks better in hindsight, the 250-day speed alone, was
chosen AFTER seeing the results and so is not licensed by that data. There is
exactly one honest way to test it: run it forward on prices it has never seen
and never revise a decision after the fact.

That is what this does. Every decision is journaled at the moment it is taken,
with the state behind it. No parameter may change while it runs; changing one
starts a new record, and the old one stands.

**Two execution modes, and why the default is shadow.**

- *shadow* (default): IB supplies prices, contracts, specs and the account;
  fills are simulated in-process by `PaperAdapter`, which crosses the spread and
  charges slippage against you. No order reaches IB.
- *live-paper* (`--live-paper`): orders go to the IB paper account.

Neither measures real slippage. IB's paper server fills at the quote - measured
on 2026-09-07, zero slippage on eight consecutive legs across instruments
quoting 0 to 120 points. So on that axis the in-process paper book is the more
pessimistic of the two, and the record's value lies in the decisions and the
costs, not in whose simulator filled them.

**Cadence.** The rules decide on the first trading day of each calendar month
and hold between decisions. Started mid-month it will do nothing until the next
one, which is deliberate: entering on an arbitrary start date makes the record
depend on when it happened to be switched on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import RiskProfile  # noqa: E402
from core.contracts import MICRO_UNIVERSE  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from execution.shadow import ShadowAdapter  # noqa: E402
from live.runner import Runner  # noqa: E402
from live.state import StateStore  # noqa: E402
from ops.journal import Journal  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.killswitch import KillFile  # noqa: E402
from risk.voltarget import VolTarget  # noqa: E402
from strategies.carry import Carry  # noqa: E402
from strategies.tsmom import TSMOM  # noqa: E402

STATE = ROOT / "state"
JOURNAL = STATE / "forward_journal.jsonl"
SESSION = STATE / "forward_session.json"
KILL = STATE / "FORWARD_KILL"
MANIFEST = STATE / "forward_manifest.json"

#: Declared once, at the start of the record, and never edited while it runs.
LOOKBACKS = (60, 120, 250)
TARGET_VOL = 0.12
#: Every micro in the tradeable universe. ZN has no micro; it is the full
#: contract and is included because a rates leg matters more to a trend book
#: than the extra size costs.
UNIVERSE = tuple(MICRO_UNIVERSE)


def build_sleeves(symbols: tuple[str, ...], with_carry: bool) -> list[Sleeve]:
    """Trend at three speeds plus carry - the combination entry 010 measured at
    0.32, the best result the research produced, and better than either alone at
    a correlation of about 0.2 between them.

    Carry reads the front contract against the next delivery month, which the
    adapter now supplies live through `bar_extras`. Before that was wired it sat
    in the book reading flat, which is worse than not running it at all.
    """
    sleeves = [
        Sleeve(f"tsmom{lb}", (lambda s, lb=lb: TSMOM(lookback=lb)), symbols, timeframe="D1")
        for lb in LOOKBACKS
    ]
    if with_carry:
        sleeves.append(Sleeve("carry", lambda s: Carry.published(), symbols, timeframe="D1"))
    return sleeves


def manifest(args, symbols, sleeves, mode: str) -> dict:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "universe": list(symbols),
        "lookbacks": list(LOOKBACKS),
        "sleeves": [s.name for s in sleeves],
        "profile": args.profile,
        "target_vol": TARGET_VOL if args.vol_target else None,
        "risk_per_trade": RiskProfile.load(args.profile).risk_per_trade,
        "note": ("Declared at the start of the record. Changing any of these ends this "
                 "record and starts a new one; the old one stands as it is."),
    }


def report() -> int:
    if not JOURNAL.exists():
        print(f"no record yet at {JOURNAL.relative_to(ROOT)}")
        return 2
    j = Journal(JOURNAL)
    records = j.read()
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        print(f"\nFORWARD RECORD - started {m['started_at'][:16]} UTC, mode {m['mode']}")
        print(f"  {len(m['universe'])} markets, sleeves {', '.join(m['sleeves'])}, "
              f"risk {m['risk_per_trade']:.2%}/trade"
              + (f", vol target {m['target_vol']:.0%}" if m.get("target_vol") else ""))
    beats = [r for r in records if r["event"] == "heartbeat"]
    decisions = [r for r in records if r["event"] == "decision"]
    fills = [r for r in records if r["event"] == "fill" and r.get("status") == "filled"]
    print(f"\n  heartbeats {len(beats):,}   decisions {len(decisions)}   fills {len(fills)}")
    if beats:
        eq = [b["equity"] for b in beats if b.get("equity") is not None]
        span_days = (datetime.fromisoformat(beats[-1]["ts"]) - datetime.fromisoformat(beats[0]["ts"])).days
        print(f"  running {span_days} days; equity {eq[0]:,.2f} -> {eq[-1]:,.2f} "
              f"({(eq[-1] / eq[0] - 1) * 100:+.2f}%)" if eq else "")
        print(f"  halted on {sum(1 for b in beats if b.get('halted'))} of {len(beats)} heartbeats")
    approved = [d for d in decisions if d.get("approved")]
    print(f"  decisions approved {len(approved)} of {len(decisions)}")
    for d in approved[-10:]:
        print(f"    {d['ts'][:16]} {d.get('strategy'):<10} {d.get('symbol'):<5} {d.get('side'):<4} "
              f"{d.get('volume')} lots")
    rolls = [r for r in records if r["event"] in ("roll", "shadow_roll")]
    if rolls:
        print(f"  contract rolls: {len(rolls)}")
    print("\n  Decisions are journaled when taken and are never revised.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--port", type=int, default=4002, help="4002 IB Gateway paper, 7497 TWS paper")
    ap.add_argument("--client-id", type=int, default=21)
    ap.add_argument("--profile", default="research")
    ap.add_argument("--poll", type=float, default=3600.0,
                    help="seconds. Hourly by default: the rules decide monthly, and IB allows only "
                         "60 historical requests per 10 minutes - 39 legs in one burst then an hour "
                         "of quiet sits well inside that, while polling every 5 minutes would not")
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long; default runs until killed")
    ap.add_argument("--live-paper", action="store_true", help="send orders to the IB paper account")
    ap.add_argument("--no-carry", action="store_true",
                    help="momentum only. Carry is part of the declared design; dropping it "
                         "changes the terms of the record")
    ap.add_argument("--vol-target", action="store_true", help="size the book to 12% annualised (entry 011)")
    ap.add_argument("--dry-run", action="store_true", help="startup checks, then exit")
    ap.add_argument("--report", action="store_true", help="summarise the record so far")
    args = ap.parse_args()

    if args.report:
        return report()

    from execution.ib_adapter import IBAdapter

    symbols = tuple(args.symbols or UNIVERSE)
    unknown = [s for s in symbols if s not in MICRO_UNIVERSE]
    if unknown:
        print(f"unknown roots {unknown}; tradeable: {', '.join(MICRO_UNIVERSE)}")
        return 1

    live = IBAdapter(port=args.port, client_id=args.client_id, roots=MICRO_UNIVERSE)
    live.connect()
    account = live.account()

    specs: dict = {}
    unavailable: list[str] = []
    for s in symbols:
        try:
            specs[s] = live.spec(s)
        except Exception as exc:  # noqa: BLE001
            unavailable.append(f"{s} ({type(exc).__name__})")
    symbols = tuple(s for s in symbols if s in specs)
    if not symbols:
        print("no tradeable contracts resolved; is the Gateway logged in?")
        return 1

    mode = "live-paper" if args.live_paper else "shadow"
    adapter = live if args.live_paper else ShadowAdapter(live, specs)
    sleeves = build_sleeves(symbols, with_carry=not args.no_carry)
    profile = RiskProfile.load(args.profile)
    engine = build_engine(profile, account.equity, specs, sleeves)
    allocator = VolTarget(target_annual_vol=TARGET_VOL,
                          max_risk_fraction=profile.max_risk_per_trade) if args.vol_target else None

    print(f"\nFORWARD PAPER RECORD - {mode}")
    print(f"  account   : {account.equity:,.2f} {account.currency} on IB port {args.port}")
    print(f"  markets   : {len(symbols)} - {', '.join(symbols)}")
    if unavailable:
        print(f"  UNAVAILABLE: {', '.join(unavailable)}")
    print(f"  sleeves   : {', '.join(s.name for s in sleeves)}  ({len(sleeves) * len(symbols)} legs)")
    print(f"  risk      : {profile.risk_per_trade:.2%}/trade, profile {profile.name}"
          + (f", book volatility target {TARGET_VOL:.0%}" if allocator else ""))
    print(f"  data      : {'DELAYED' if live.market_data_type not in (None, 1) else 'live'} "
          f"(type {live.market_data_type})")
    print(f"  cadence   : decisions on the first trading day of each month, held between")
    print(f"  journal   : {JOURNAL.relative_to(ROOT)}")
    print(f"  stop with : create {KILL.relative_to(ROOT)}\n")

    runner = Runner(
        adapter=adapter, risk=engine, sleeves=sleeves, specs=specs,
        timeframe="D1", poll_seconds=args.poll,
        state=StateStore(SESSION), journal=Journal(JOURNAL), kill=KillFile(KILL),
        allocator=allocator,
    )
    for note in runner.start():
        print(f"  {note}")

    if args.dry_run:
        print("\n  DRY RUN - startup verified, not entering the loop\n")
        runner.worker.stop()
        live.disconnect()
        return 0

    STATE.mkdir(exist_ok=True)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest(args, symbols, sleeves, mode), indent=2), encoding="utf-8")
        print(f"\n  manifest written to {MANIFEST.relative_to(ROOT)} - the terms of this record")
    runner.journal.write("forward_start", mode=mode, symbols=list(symbols),
                         sleeves=[s.name for s in sleeves], equity=account.equity,
                         market_data_type=live.market_data_type)

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    iterations = errors = 0
    try:
        while deadline is None or time.time() < deadline:
            iterations += 1
            try:
                runner.tick()
            except Exception as exc:  # noqa: BLE001 - a record that dies is no record
                errors += 1
                runner.journal.write("loop_error", error=f"{type(exc).__name__}: {exc}")
                print(f"  [{datetime.now(timezone.utc):%m-%d %H:%M}] error: {type(exc).__name__}: {exc}",
                      flush=True)
                if errors % 3 == 0:
                    try:
                        live.disconnect(); live.connect()
                        runner.journal.write("reconnect", ok=True)
                    except Exception as exc2:  # noqa: BLE001
                        runner.journal.write("reconnect", ok=False, error=str(exc2))
                time.sleep(min(120.0, args.poll))
                continue
            if iterations % 12 == 1:
                acct = adapter.account()
                print(f"  [{datetime.now(timezone.utc):%m-%d %H:%M}] equity {acct.equity:>12,.2f}  "
                      f"positions {len(adapter.positions())}  halted={runner.halted}  errors={errors}",
                      flush=True)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        runner.shutdown()
        live.disconnect()

    print(f"\n  {iterations} iterations, {errors} errors")
    print(f"  {runner.journal.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
