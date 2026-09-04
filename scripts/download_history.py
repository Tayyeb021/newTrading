"""Download historical bars into the Parquet store.

    python scripts/download_history.py --years 5 --timeframes M1 H1 D1
    python scripts/download_history.py --report            # what is already stored
    python scripts/download_history.py --sessions EURUSD   # verify the broker clock

Resumable: rerunning fetches only what is missing, with a day of deliberate
overlap at the seam. Safe to interrupt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig  # noqa: E402
from data.download import download_all, session_profile  # noqa: E402
from data.store import BarStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--timeframes", nargs="+", default=["M1", "H1", "D1"])
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--root", default="data/bars")
    ap.add_argument("--report", action="store_true", help="show coverage, download nothing")
    ap.add_argument("--sessions", metavar="SYMBOL", help="show when bars actually exist, by UTC hour")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    instruments = InstrumentConfig.load()
    symbols = args.symbols or instruments.symbols
    store = BarStore(args.root)

    if args.sessions:
        print(session_profile(store, args.sessions))
        return 0

    if args.report:
        print("\nSTORE COVERAGE\n")
        for symbol in symbols:
            for tf in args.timeframes:
                print(store.report(symbol, tf))
        return 0

    from execution.mt5_adapter import MT5Adapter

    adapter = MT5Adapter(aliases=instruments.aliases)
    adapter.connect()
    print(f"\nDownloading {args.years:g}y of {', '.join(args.timeframes)} "
          f"for {', '.join(symbols)}\n")
    try:
        results = download_all(adapter, store, symbols, args.timeframes, years=args.years)
    finally:
        adapter.disconnect()

    print("\nRESULT\n")
    for symbol in symbols:
        for tf in args.timeframes:
            added = results.get((symbol, tf), 0)
            print(f"{store.report(symbol, tf)}   (+{added:,} this run)")

    print("\nNext: check the broker clock before writing any session filter --")
    print(f"  python scripts/download_history.py --sessions {symbols[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
