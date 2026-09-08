"""Fetch trade prints for one futures root from Databento. Research entry 016.

    python scripts/download_ticks.py --dry-run                 # cost estimate, spends nothing
    python scripts/download_ticks.py --root ES --months 3
    python scripts/download_ticks.py --root ES --months 3 --max-cost 45

Writes `data/ticks/<ROOT>/<CONTRACT>.parquet`, one file per contract, with the
columns `features/orderflow.py` expects: ts, price, size, side. Databento's
trades schema carries the aggressor side directly, so the tick rule is not
needed and is not used - an inferred side is a different and worse quantity
than a known one, and mixing them silently would be the kind of error this
repository exists to avoid.

Only the contracts that were actually front month during the window are
fetched. Asking for every listed code multiplied the S&P bill by five for data
nobody would have traded.

Tick data is large: three months of ES is roughly a gigabyte uncompressed. The
cost estimate is printed before anything is bought and `--max-cost` refuses
above a number.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.contracts import ALL_ROOTS  # noqa: E402

DATASET = "GLBX.MDP3"
OUT = ROOT / "data" / "ticks"


def front_sequence(root, start: date, end: date) -> list[str]:
    """Only the contracts that were front month at some point in the window."""
    seen: list[str] = []
    d = start
    while d <= end:
        code = root.code(*root.front(d))
        if code not in seen:
            seen.append(code)
        d += timedelta(days=7)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="ES")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--end", default=None,
                    help="YYYY-MM-DD; default two days back. The most recent session needs a "
                         "live subscription this account does not have, and asking for it fails "
                         "the whole request rather than trimming it")
    ap.add_argument("--schema", default="trades", choices=["trades", "tbbo"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.root not in ALL_ROOTS:
        print(f"unknown root {args.root!r}")
        return 1
    root = ALL_ROOTS[args.root]
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=2)
    start = end - timedelta(days=int(args.months * 30.44))
    codes = front_sequence(root, start, end)

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY is not set. In your own shell:")
        print('  PowerShell:  $env:DATABENTO_API_KEY = "db-..."')
        return 2

    import databento as db
    client = db.Historical(key)

    print(f"\n{args.root} {args.schema}, {start} -> {end}")
    print(f"  front-month contracts: {' '.join(codes)}")
    try:
        cost = float(client.metadata.get_cost(
            dataset=DATASET, symbols=codes, stype_in="raw_symbol", schema=args.schema,
            start=start.isoformat(), end=end.isoformat()))
        size = float(client.metadata.get_billable_size(
            dataset=DATASET, symbols=codes, stype_in="raw_symbol", schema=args.schema,
            start=start.isoformat(), end=end.isoformat()))
        print(f"  Databento cost estimate: ${cost:,.2f}   ({size / 1e9:.2f} GB uncompressed)")
        if args.max_cost is not None and cost > args.max_cost:
            print(f"  exceeds --max-cost ${args.max_cost:,.2f}; nothing downloaded")
            return 3
    except Exception as exc:  # noqa: BLE001
        print(f"  cost estimate unavailable: {type(exc).__name__}: {exc}")
        if args.dry_run or args.max_cost is not None:
            return 1
    if args.dry_run:
        print("  dry run: nothing downloaded\n")
        return 0

    folder = Path(args.out) / args.root
    folder.mkdir(parents=True, exist_ok=True)
    data = client.timeseries.get_range(
        dataset=DATASET, symbols=codes, stype_in="raw_symbol", schema=args.schema,
        start=start.isoformat(), end=end.isoformat())
    df = data.to_df().reset_index()
    if df.empty:
        print("  nothing returned")
        return 1

    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    # Databento marks the AGGRESSOR: 'A' means the trade lifted the offer, 'B'
    # that it hit the bid. Anything else ('N', or absent) is not classifiable and
    # is dropped rather than guessed - an inferred side is a different quantity
    # from a known one and must not be mixed in silently.
    raw_side = df["side"].astype(str) if "side" in df else pd.Series("N", index=df.index)
    df["signed"] = raw_side.map({"A": 1, "B": -1}).fillna(0).astype(int)
    unclassified = int((df["signed"] == 0).sum())

    total = 0
    for code, grp in df.groupby(df["symbol"].astype(str)):
        keep = grp[grp["signed"] != 0]
        out = pd.DataFrame({
            "ts": keep["ts"].to_numpy(),
            "price": keep["price"].astype(float).to_numpy(),
            "size": keep["size"].astype(float).to_numpy(),
            "side": keep["signed"].to_numpy(),
        }).sort_values("ts").reset_index(drop=True)
        if out.empty:
            continue
        out.to_parquet(folder / f"{code}.parquet", index=False)
        total += len(out)
        print(f"  {code:<10}{len(out):>12,} prints   {out['ts'].min():%Y-%m-%d} -> {out['ts'].max():%Y-%m-%d}")

    print(f"\n  {total:,} classified prints written to {folder.relative_to(ROOT)}")
    if unclassified:
        print(f"  {unclassified:,} prints ({unclassified / len(df):.2%}) had no aggressor side and were dropped")
    print(f"  next: python research/orderflow_016.py --root {args.root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
