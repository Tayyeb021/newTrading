"""Fetch per-expiry daily bars for the micro universe from Databento.

    set DATABENTO_API_KEY=db-...          (Windows)   export DATABENTO_API_KEY=db-...   (POSIX)
    python scripts/download_databento.py --roots MES MGC --since 2018

Writes data/futures/<ROOT>/<YYYYMM>.parquet, one file per contract month, in
the same validated bar format the rest of the system uses. Stitching into a
continuous series happens at backtest time so the roll rule can change without
re-downloading.

Micro contracts only list from 2019 (MES, MNQ), 2010 (MGC), 2009 (M6E), 2021
(MCL). For longer history the full-size parent (ES, GC, 6E, CL) trades the same
price to within a tick; pass `--parent` to fetch those instead and the roots
still map to the micro specs for sizing.

Without a key the script explains itself and exits 2. It never falls back to a
placeholder dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import MICRO_UNIVERSE  # noqa: E402
from data.store import validate  # noqa: E402

PARENT = {"MES": "ES", "MNQ": "NQ", "MGC": "GC", "M6E": "6E", "MCL": "CL", "ZN": "ZN"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["MES", "MGC"])
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--parent", action="store_true", help="fetch the full-size parent contract's history")
    ap.add_argument("--out", default="data/futures")
    ap.add_argument("--dataset", default="GLBX.MDP3")
    args = ap.parse_args()

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY is not set.")
        print("  Get a key at databento.com, then:  set DATABENTO_API_KEY=db-...")
        print("  Historical CME daily bars are pay-as-you-go; a few years of one root costs cents.")
        return 2

    import databento as db

    client = db.Historical(key)
    start = date(args.since, 1, 1)
    end = date.today()

    for name in args.roots:
        root = MICRO_UNIVERSE[name]
        symbol_root = PARENT[name] if args.parent else root.root
        folder = Path(args.out) / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n{name} ({symbol_root}.FUT on {args.dataset}) {start} -> {end}")

        # One request for the whole family; Databento returns every expiry with
        # its own raw symbol (e.g. MESZ5), which we split into per-month files.
        data = client.timeseries.get_range(
            dataset=args.dataset, symbols=[f"{symbol_root}.FUT"], stype_in="parent",
            schema="ohlcv-1d", start=start.isoformat(), end=end.isoformat(),
        )
        df = data.to_df().reset_index()
        if df.empty:
            print("  nothing returned"); continue

        df["symbol"] = df["symbol"].astype(str)
        for sym, grp in df.groupby("symbol"):
            code = sym.replace(symbol_root, "")
            if len(code) < 2:
                continue
            month = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}.get(code[0])
            year_digit = code[1:]
            if month is None or not year_digit.isdigit():
                continue
            year = 2020 + int(year_digit) if len(year_digit) == 1 else int(year_digit)
            if len(year_digit) == 1 and year > end.year + 2:
                year -= 10
            bars = pd.DataFrame({
                "ts": pd.to_datetime(grp["ts_event"], utc=True),
                "open": grp["open"].astype(float), "high": grp["high"].astype(float),
                "low": grp["low"].astype(float), "close": grp["close"].astype(float),
                "volume": grp["volume"].astype(float),
            }).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            try:
                validate(bars, f"{name}{code}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {sym}: refused - {exc}"); continue
            bars.to_parquet(folder / f"{year}{month:02d}.parquet", index=False)
            print(f"  {sym:<8} {len(bars):>5} bars  {bars['ts'].iloc[0]:%Y-%m-%d} -> {bars['ts'].iloc[-1]:%Y-%m-%d}")

    print("\nnext: python scripts/backtest_futures.py --roots", " ".join(args.roots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
