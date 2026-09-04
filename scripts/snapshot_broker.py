"""Snapshot live contract specs and measured spreads to config/broker.json.

Replaces the fixture specs with what this broker actually offers. Spreads are
sampled repeatedly rather than read once, because a single reading taken during
a quiet minute understates what you pay at the open.

    python scripts/snapshot_broker.py --samples 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--interval", type=float, default=0.4)
    ap.add_argument("--out", default="config/broker.json")
    args = ap.parse_args()

    from execution.mt5_adapter import MT5Adapter

    inst = InstrumentConfig.load()
    adapter = MT5Adapter(aliases=inst.aliases)
    adapter.connect()
    try:
        acct = adapter.account()
        snapshot = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "account_currency": acct.currency,
            "specs": {},
            "spreads": {},
        }
        samples: dict[str, list[float]] = {s: [] for s in inst.symbols}

        # Select every symbol into Market Watch first. Without this, metals and
        # indices return no tick and silently produce no measurement.
        for sym in inst.symbols:
            adapter.mt5.symbol_select(adapter.broker_symbol(sym), True)
        time.sleep(1.0)

        for _ in range(args.samples):
            for sym in inst.symbols:
                try:
                    samples[sym].append(adapter.tick(sym).spread)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(args.interval)

        print(f"\n{'symbol':<9}{'point':>10}{'tick_val':>10}{'min_lot':>9}"
              f"{'$/1.0':>12}{'spread(pts)':>13}{'swap L/S':>18}")
        print("-" * 82)
        for sym in inst.symbols:
            spec = adapter.spec(sym)
            snapshot["specs"][sym] = asdict(spec)
            obs = samples[sym]
            if obs:
                snapshot["spreads"][sym] = {
                    "median": statistics.median(obs),
                    "max": max(obs),
                    "n": len(obs),
                }
            med = statistics.median(obs) if obs else float("nan")
            print(f"{sym:<9}{spec.point:>10.5g}{spec.tick_value:>10.4g}{spec.volume_min:>9.3g}"
                  f"{spec.value_per_price_unit:>12,.1f}{med / spec.point:>13.1f}"
                  f"{spec.swap_long:>9.1f}/{spec.swap_short:<8.1f}")

        unmeasured = [s for s in inst.symbols if s not in snapshot["spreads"]]
        if unmeasured:
            print(f"\n  WARNING: no tick for {unmeasured} - market closed, or the")
            print("  symbol is not tradeable on this account. Backtests on these")
            print("  symbols will REFUSE to run rather than use placeholder costs.")

        Path(args.out).write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        print(f"\nwritten to {args.out}")
        print("Specs are now read from this file, not from fixtures.")
        return 0
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
