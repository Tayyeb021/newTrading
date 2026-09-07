"""Sample live spreads through the session and profile them by hour. Read-only.

    python scripts/spread_profile.py --until 2026-09-07T21:00Z
    python scripts/spread_profile.py --report            # what the samples say

Cost has two halves. Slippage - the gap between the price you asked for and the
one you got - can only be measured by sending a real order, and that is the
operator's job (`verify_roundtrip.py`). The spread is quoted continuously and
can simply be watched, which is what this does: one tick per symbol per
interval, appended to `state/spread_samples.jsonl`, and never an order.

Why by hour. `snapshot_broker.py` samples twelve ticks over five seconds, which
answers "what is the spread right now". The number the cost model needs is the
spread you actually trade at, and on an index CFD that is three times wider
before the New York open than after it. An hourly profile shows which hours are
cheap, sets an honest median for the model, and tells the SpreadGuard limit what
"twice normal" really means.

Nothing here can trade. It calls `tick()` and writes a file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "state" / "spread_samples.jsonl"


def report(path: Path, symbols: list[str]) -> int:
    if not path.exists():
        print(f"no samples at {path.relative_to(ROOT)}; run without --report first")
        return 2
    by_hour: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_symbol: dict[str, list[float]] = defaultdict(list)
    points: dict[str, float] = {}
    total = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = r["symbol"]
            pts = r["spread_points"]
            hour = datetime.fromisoformat(r["ts"]).astimezone(timezone.utc).hour
            by_hour[(sym, hour)].append(pts)
            by_symbol[sym].append(pts)
            points[sym] = r.get("point", points.get(sym, 0.0))
            total += 1

    present = [s for s in symbols if s in by_symbol] or sorted(by_symbol)
    print(f"\nSPREAD PROFILE - {total:,} samples, {len(present)} instruments, spreads in POINTS\n")
    head = f"  {'hour UTC':<10}" + "".join(f"{s:>12}" for s in present)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for hour in range(24):
        cells = ""
        any_here = False
        for sym in present:
            obs = by_hour.get((sym, hour))
            if obs:
                any_here = True
                cells += f"{statistics.median(obs):>12.1f}"
            else:
                cells += f"{'-':>12}"
        if any_here:
            print(f"  {hour:02d}:00{'':<5}{cells}")

    print()
    head2 = f"  {'':<10}" + "".join(f"{s:>12}" for s in present)
    print(head2)
    for label, fn in (("median", statistics.median),
                      ("cheapest hr", None), ("worst hr", None), ("samples", None)):
        cells = ""
        for sym in present:
            hours = {h: statistics.median(v) for (s, h), v in by_hour.items() if s == sym}
            if label == "median":
                cells += f"{fn(by_symbol[sym]):>12.1f}"
            elif label == "cheapest hr":
                h = min(hours, key=hours.get)
                cells += f"{f'{h:02d}:00 ({hours[h]:.0f})':>12}"
            elif label == "worst hr":
                h = max(hours, key=hours.get)
                cells += f"{f'{h:02d}:00 ({hours[h]:.0f})':>12}"
            else:
                cells += f"{len(by_symbol[sym]):>12,}"
        print(f"  {label:<10}{cells}")

    print("\n  The cheapest hour is when to take the round-trip measurement.")
    print("  The ratio worst/median is what SpreadGuard's multiple has to tolerate.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--until", default=None, metavar="ISO8601", help="absolute UTC end, overrides --minutes")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--out", default=str(SAMPLES))
    ap.add_argument("--report", action="store_true", help="summarise existing samples and exit")
    args = ap.parse_args()

    inst = InstrumentConfig.load()
    symbols = args.symbols or list(inst.symbols)
    out = Path(args.out)

    if args.report:
        return report(out, symbols)

    if args.until:
        end = datetime.fromisoformat(args.until.replace("Z", "+00:00"))
        end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end.astimezone(timezone.utc)
        args.minutes = (end - datetime.now(timezone.utc)).total_seconds() / 60
        if args.minutes <= 0:
            print("that deadline has passed")
            return 0

    from execution.mt5_adapter import MT5Adapter

    adapter = MT5Adapter(aliases=inst.aliases)
    adapter.connect()
    out.parent.mkdir(parents=True, exist_ok=True)
    specs = {}
    for s in symbols:
        try:
            specs[s] = adapter.spec(s)
        except Exception as exc:  # noqa: BLE001
            print(f"  {s}: no spec ({exc}); skipping")
    symbols = [s for s in symbols if s in specs]

    print(f"sampling {len(symbols)} instruments every {args.interval:g}s for {args.minutes:.0f} min "
          f"-> {out.relative_to(ROOT)}", flush=True)

    deadline = time.time() + args.minutes * 60
    written = errors = 0
    try:
        with out.open("a", encoding="utf-8") as fh:
            while time.time() < deadline:
                now = datetime.now(timezone.utc)
                for sym in symbols:
                    try:
                        tick = adapter.tick(sym)
                    except Exception:  # noqa: BLE001 - a closed market is not an error worth stopping for
                        errors += 1
                        continue
                    age = (now - tick.ts).total_seconds()
                    if age > 120:
                        continue  # stale quote: the market is shut, not cheap
                    point = specs[sym].point
                    fh.write(json.dumps({
                        "ts": now.isoformat(), "symbol": sym, "bid": tick.bid, "ask": tick.ask,
                        "spread": tick.spread, "spread_points": tick.spread / point if point else 0.0,
                        "point": point, "age_s": round(age, 1),
                    }) + "\n")
                    written += 1
                fh.flush()
                if written and written % (len(symbols) * 90) == 0:
                    print(f"  [{now:%H:%M}] {written:,} samples", flush=True)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        adapter.disconnect()

    print(f"\n{written:,} samples written, {errors} tick errors. "
          f"Summarise with: python scripts/spread_profile.py --report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
