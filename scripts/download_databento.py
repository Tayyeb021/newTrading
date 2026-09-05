"""Fetch per-expiry daily bars for the futures research universe from Databento.

    $env:DATABENTO_API_KEY = "db-..."     (PowerShell, this session)
    setx DATABENTO_API_KEY "db-..."       (Windows, every future shell)
    export DATABENTO_API_KEY=db-...       (POSIX)
    python scripts/download_databento.py --dry-run                     # cost estimate only, spends nothing
    python scripts/download_databento.py --universe full --since 2010  # all 33 research markets
    python scripts/download_databento.py --roots ES GC ZN

Writes data/futures/<ROOT>/<YYYYMM>.parquet, one file per contract month, in
the same validated bar format the rest of the system uses. Stitching into a
continuous series happens at backtest time so the roll rule can change without
re-downloading.

History is always fetched for the DATA root: a micro (MES) is stored under its
full-size parent (ES), which has the longer history and trades the same price
to within a tick. `backtest_futures.py` resolves a micro to that folder and
sizes with the micro's contract. Micro contracts only list from 2019 (MES,
MNQ), 2010 (MGC), 2009 (M6E), 2021 (MCL); the parents go back to the start of
the dataset, June 2010.

Every run prints Databento's own cost estimate for the request before spending
anything. Without a key the script explains itself and exits 2. It never falls
back to a placeholder dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import ALL_ROOTS, CODE_MONTHS, FULL_UNIVERSE, MICRO_UNIVERSE, data_root  # noqa: E402
from data.store import validate  # noqa: E402

DATASET_START = date(2010, 6, 6)  # GLBX.MDP3 begins here


def contracts_from_family(df: pd.DataFrame, symbol_root: str, end_year: int):
    """Split a parent-symbol response into (year, month, bars) per contract.

    Raw symbols carry one year digit (ESH1), so a 2011 and a 2021 contract share
    a name. Group by symbol, then split a group wherever the bars jump by more
    than a year, and read the decade off each segment's last bar.
    """
    df = df.sort_values("ts")
    for sym, grp in df.groupby("symbol", sort=False):
        code = str(sym).replace(symbol_root, "", 1)
        if "-" in str(sym) or len(code) < 2 or code[0] not in CODE_MONTHS or not code[1:].isdigit():
            continue  # spreads, options, or something we do not model
        month, digit = CODE_MONTHS[code[0]], int(code[1:]) % 10
        ts = grp["ts"]
        breaks = ts.diff().dt.days.fillna(0) > 365
        for _, seg in grp.groupby(breaks.cumsum()):
            last_year = int(seg["ts"].iloc[-1].year)
            year = last_year + ((digit - last_year) % 10)  # first year >= last bar with that digit
            if year > end_year + 2:
                year -= 10
            yield year, month, seg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=None, help="any research or micro root")
    ap.add_argument("--universe", choices=["micro", "full"], default=None,
                    help="every tradeable micro's parent, or all 33 research markets")
    ap.add_argument("--since", type=int, default=DATASET_START.year)
    ap.add_argument("--out", default="data/futures")
    ap.add_argument("--dataset", default="GLBX.MDP3")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    args = ap.parse_args()

    if args.roots and args.universe:
        print("give --roots or --universe, not both"); return 1
    names = args.roots or list(FULL_UNIVERSE if args.universe == "full" else MICRO_UNIVERSE)
    unknown = [n for n in names if n not in ALL_ROOTS]
    if unknown:
        print(f"unknown roots {unknown}; known: {', '.join(sorted(ALL_ROOTS))}"); return 1
    # de-duplicate to data roots: MES and ES are the same history
    targets = sorted({data_root(n).root for n in names})

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY is not set.")
        print("  Get a key at databento.com (new accounts carry free credit), then, in your own shell:")
        print('    PowerShell:  $env:DATABENTO_API_KEY = "db-..."      cmd:  set DATABENTO_API_KEY=db-...')
        print("  Never paste the key into a chat or a file; the script reads it from the environment only.")
        print("  Daily bars are metered per gigabyte; the whole research universe is on the order of")
        print("  a hundred megabytes, well inside a new account's free credit. --dry-run prints the exact cost.")
        return 2

    import databento as db

    client = db.Historical(key)
    start = max(date(args.since, 1, 1), DATASET_START)
    end = date.today()

    print(f"\n{len(targets)} data roots on {args.dataset}, {start} -> {end}, schema ohlcv-1d")
    print("  " + " ".join(targets))
    try:
        total = 0.0
        for t in targets:
            total += float(client.metadata.get_cost(
                dataset=args.dataset, symbols=[f"{t}.FUT"], stype_in="parent", schema="ohlcv-1d",
                start=start.isoformat(), end=end.isoformat()))
        print(f"  Databento cost estimate: ${total:,.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  cost estimate unavailable: {type(exc).__name__}: {exc}")
        if args.dry_run:
            return 1
    if args.dry_run:
        print("  dry run: nothing downloaded"); return 0

    for t in targets:
        folder = Path(args.out) / t
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n{t} ({t}.FUT)")
        data = client.timeseries.get_range(
            dataset=args.dataset, symbols=[f"{t}.FUT"], stype_in="parent",
            schema="ohlcv-1d", start=start.isoformat(), end=end.isoformat(),
        )
        df = data.to_df().reset_index()
        if df.empty:
            print("  nothing returned"); continue
        df["symbol"] = df["symbol"].astype(str)
        df["ts"] = pd.to_datetime(df["ts_event"], utc=True)

        written = 0
        for year, month, seg in contracts_from_family(df, t, end.year):
            bars = pd.DataFrame({
                "ts": seg["ts"].to_numpy(),
                "open": seg["open"].astype(float).to_numpy(), "high": seg["high"].astype(float).to_numpy(),
                "low": seg["low"].astype(float).to_numpy(), "close": seg["close"].astype(float).to_numpy(),
                "volume": seg["volume"].astype(float).to_numpy(),
            }).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            label = f"{t}{year}{month:02d}"
            try:
                validate(bars, label)
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: refused - {exc}"); continue
            bars.to_parquet(folder / f"{year}{month:02d}.parquet", index=False)
            written += 1
        span = f"{df['ts'].min():%Y-%m-%d} -> {df['ts'].max():%Y-%m-%d}"
        print(f"  {written} contract months, {len(df):,} bars, {span}")

    print("\nnext: python scripts/backtest_futures.py --roots", " ".join(targets), "--profile research")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
