"""Emergency: close everything and stop the system.

Deliberately standalone. It connects to the broker on its own and imports nothing
from the runner, so it works when the runner is hung, crashed, or wedged in a
retry loop — which is precisely when you need it.

    python scripts/flatten_all.py --dry-run     # show what would close
    python scripts/flatten_all.py               # close it, engage the kill switch
    python scripts/flatten_all.py --no-kill     # close without stopping the system

Engaging the kill switch is the default and is the right default. If you needed
to flatten by hand, you do not want a running process reopening the position
thirty seconds later. Clear it deliberately with `--clear-kill` once you have
dealt with whatever caused this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig  # noqa: E402
from execution.oms import OrderManager  # noqa: E402
from risk.killswitch import KillFile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-kill", action="store_true", help="do not engage the kill switch")
    ap.add_argument("--clear-kill", action="store_true", help="clear the kill switch and exit")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--state", default="state/KILL")
    args = ap.parse_args()

    kill = KillFile(args.state)

    if args.clear_kill:
        record = kill.read()
        kill.clear()
        print(f"kill switch cleared (was: {record.reason or 'not engaged'})")
        return 0

    from execution.mt5_adapter import MT5Adapter

    adapter = MT5Adapter(aliases=InstrumentConfig.load().aliases)
    adapter.connect()
    try:
        account = adapter.account()
        positions = adapter.positions(args.symbol)

        print(f"\nequity {account.equity:,.2f} {account.currency}, "
              f"{len(positions)} open position(s)\n")
        if not positions:
            print("  nothing to close")
        for pos in positions:
            tick = adapter.tick(pos.symbol)
            spec = adapter.spec(pos.symbol)
            price = tick.bid if pos.side.name == "BUY" else tick.ask
            pnl = pos.unrealized(price, spec)
            print(f"  {pos.symbol:<10} {pos.side.name:<5} {pos.volume:>7g} lots  "
                  f"ticket {pos.ticket:<12} unrealised {pnl:>+12,.2f}")

        if args.dry_run:
            print("\n  DRY RUN - nothing closed")
            return 0

        if positions:
            print()
            oms = OrderManager(adapter)
            for result in oms.close_all(args.symbol):
                mark = "closed " if result.ok else "FAILED "
                print(f"  {mark} ticket {result.ticket}: "
                      f"{result.fill_price if result.ok else result.reason}")

            remaining = adapter.positions(args.symbol)
            if remaining:
                print(f"\n  WARNING: {len(remaining)} position(s) STILL OPEN. "
                      f"Close them in the terminal by hand, now.")
                return 1
            print("\n  account is flat")

        if not args.no_kill:
            record = kill.engage("flatten_all invoked manually", by="operator")
            print(f"  {record}")
            print("  clear it with: python scripts/flatten_all.py --clear-kill")
        return 0
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
