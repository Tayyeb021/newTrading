"""Run the live trading loop.

    python scripts/run_live.py --dry-run          # startup checks only, no loop
    python scripts/run_live.py --symbols EURUSD   # trade

Stops cleanly on Ctrl+C: it drains the execution queue, saves state, disconnects.
It does NOT close positions on shutdown - stopping the software is not the same
as wanting to be flat. To go flat, use scripts/flatten_all.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig, RiskProfile  # noqa: E402
from live.runner import Runner  # noqa: E402
from live.state import StateStore  # noqa: E402
from ops.journal import Journal  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.killswitch import KillFile  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--timeframe", default="D1")
    ap.add_argument("--profile", default="challenge")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true", help="startup checks, then exit")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    from execution.mt5_adapter import MT5Adapter

    instruments = InstrumentConfig.load()
    symbols = args.symbols or instruments.symbols
    profile = RiskProfile.load(args.profile)

    adapter = MT5Adapter(aliases=instruments.aliases)
    adapter.connect()
    specs = {s: adapter.spec(s) for s in symbols}

    engine = build_engine(profile, adapter.account().equity, specs)
    runner = Runner(
        adapter=adapter,
        risk=engine,
        strategies={s: TrendFollowing() for s in symbols},
        specs=specs,
        timeframe=args.timeframe,
        poll_seconds=args.poll,
        state=StateStore(),
        journal=Journal(),
        kill=KillFile(),
    )

    print(f"\nSTARTUP - {profile.name} profile, {', '.join(symbols)} on {args.timeframe}\n")
    for note in runner.start():
        print(f"  {note}")

    if args.dry_run:
        print("\n  DRY RUN - startup verified, not entering the loop")
        runner.shutdown()
        return 0

    print(f"\n  running, polling every {args.poll:g}s. Ctrl+C to stop.")
    print("  emergency: python scripts/flatten_all.py\n")
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
