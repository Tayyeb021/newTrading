"""Fetch CFTC Commitments of Traders history into data/cot/.

    python scripts/download_cot.py --years 15
    python scripts/download_cot.py --markets GC ES --years 5

Free, weekly, no key. The loader tolerates the CFTC renaming markets between
years and reports exactly which name it matched, because a silent mismatch is
how you end up with a gold signal built on gold *spreads*.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cot import COTStore, features, fetch_range  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["GC", "ES", "NQ", "6E", "CL"])
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--root", default="data/cot")
    args = ap.parse_args()

    this_year = date.today().year
    store = COTStore(args.root)
    print(f"fetching {args.markets} for {this_year - args.years + 1}-{this_year}\n")
    for r in fetch_range(store, args.markets, range(this_year - args.years + 1, this_year + 1)):
        if "error" in r.markets_matched:
            print(f"  {r.year}: FAILED {r.markets_matched['error']}")
            continue
        missing = [m for m in args.markets if m not in r.markets_matched]
        flag = f"   missing {missing}" if missing else ""
        print(f"  {r.year}: {r.rows:>7,} rows, matched {len(r.markets_matched)}/{len(args.markets)}{flag}")

    print("\nstored:")
    for m in args.markets:
        df = store.read(m)
        if df.empty:
            print(f"  {m}: nothing"); continue
        f = features(df)
        print(f"  {m}: {len(f)} weeks {f['as_of'].min():%Y-%m-%d} -> {f['as_of'].max():%Y-%m-%d}   "
              f"spec net {f['spec_net'].iloc[-1]:+,.0f} ({f['spec_net_pct'].iloc[-1]:+.1%} of OI)")
    print("\nnext: python research/cot_screen.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
