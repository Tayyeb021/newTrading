"""What each level of capital unlocks, at the profile's risk per trade.

    python scripts/capital_ladder.py
    python scripts/capital_ladder.py --risk 0.0075 --levels 5000 15000 40000

Capital is a variable in this system, not a constraint: every limit is a
fraction, sizing reads live equity, and nothing anywhere hardcodes an account
size. What DOES depend on capital is which instrument-and-horizon pairs can
express the risk limit in one minimum lot or one contract. This prints that
schedule so you know what a deposit buys before you make it.

Stop distances are representative (2.5x daily ATR, ~1.5x H4 ATR, a tight
intraday stop). Replace with measured values as they change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from risk.sizing import minimum_viable_equity, size_position  # noqa: E402

# (instrument, horizon) -> stop distance in price units. Representative.
STOPS = {
    ("MES", "intraday"): 12.0,  ("MES", "swing H4"): 45.0,  ("MES", "daily"): 125.0,
    ("MNQ", "intraday"): 40.0,  ("MNQ", "swing H4"): 150.0, ("MNQ", "daily"): 300.0,
    ("MGC", "intraday"): 6.0,   ("MGC", "swing H4"): 15.0,  ("MGC", "daily"): 36.0,
    ("M6E", "intraday"): 0.0012, ("M6E", "swing H4"): 0.0040, ("M6E", "daily"): 0.0090,
    ("MCL", "intraday"): 0.40,  ("MCL", "swing H4"): 1.00,  ("MCL", "daily"): 2.00,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--risk", type=float, default=0.005)
    ap.add_argument("--levels", nargs="+", type=float, default=[5_000, 10_000, 25_000, 50_000, 100_000, 250_000])
    args = ap.parse_args()

    print(f"\nWHAT CAPITAL UNLOCKS at {args.risk:.2%} risk per trade  (micro futures, one contract minimum)\n")
    head = f"  {'instrument':<12}{'horizon':<11}{'min equity':>11}  " + "".join(f"{int(l/1000):>7}k" for l in args.levels)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for (root, horizon), stop in STOPS.items():
        spec = MICRO_UNIVERSE[root].to_spec()
        need = minimum_viable_equity(spec, args.risk, stop)
        cells = ""
        for eq in args.levels:
            r = size_position(spec, eq, args.risk, stop)
            cells += f"{('%dx' % r.volume) if r.tradeable else '-':>8}"
        print(f"  {root:<12}{horizon:<11}{need:>11,.0f}  {cells}")
    print("\n  cell = contracts sized at that equity; '-' = one contract would exceed the risk limit.")
    print("  Raise capital and rows unlock; nothing in the code needs to change when they do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
