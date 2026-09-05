"""Fetch per-expiry daily bars for the futures research universe from Databento.

    $env:DATABENTO_API_KEY = "db-..."     (PowerShell, this session)
    setx DATABENTO_API_KEY "db-..."       (Windows, every future shell)
    export DATABENTO_API_KEY=db-...       (POSIX)
    python scripts/download_databento.py --dry-run                     # cost estimate only, spends nothing
    python scripts/download_databento.py --universe full --since 2010  # all 33 research markets
    python scripts/download_databento.py --roots ES GC ZN
    python scripts/download_databento.py --reparse --universe full     # rebuild files from the raw store, no API
    python scripts/download_databento.py --check --universe full       # which contract months are missing

Two layers on disk, both under data/futures (gitignored):

- `_raw/<ROOT>.parquet` is the download exactly as received - one row per
  raw symbol and day. It is the thing that cost money, so it is kept and never
  edited; every later fix to the parsing below is a free `--reparse`.
- `<ROOT>/<YYYYMM>.parquet` is one validated file per contract month, in the
  same bar format the rest of the system uses. Stitching into a continuous
  series happens at backtest time so the roll rule can change without
  touching either layer.

Three things the raw feed does that the parser has to undo, each learned from
a real file:

- **Tickers carry one year digit, or two.** NQZ9 is December 2019 and
  December 2029. July 2025 natural gas was NGN25 for its whole life while
  June 2024 was NGM4 throughout: the exchange's choice is not predictable, so
  both forms are requested and a two-digit code is taken literally. A
  one-digit row is assigned by the exchange calendar: the first year at or
  after the row's date with that digit, unless the row falls after that
  contract's last trade day, in which case it is the next decade's. A gap
  heuristic used before glued June 2014 gas onto June 2024, because the 2024
  contract began printing within a year of the 2014 one expiring.
- **Bars are UTC days, not exchange sessions.** Index, rate, FX, metal and
  energy contracts open Sunday evening in Chicago, which is a Sunday UTC
  stub bar of an hour or two - a third of the weekday range, 300 bars a
  year. Each stub is merged into the Monday that follows it.
- **Prices can be zero or negative.** WTI May 2020 settled at -37.63. Files
  are validated with that allowed; the CFD side stays strict.

History is always fetched for the DATA root: a micro (MES) is stored under its
full-size parent (ES). Every run prints Databento's own cost estimate before
spending anything, and `--max-cost` makes it refuse above a number. Without a
key the script explains itself and exits 2. It never falls back to a placeholder.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (  # noqa: E402
    ALL_ROOTS, CODE_MONTHS, FULL_UNIVERSE, MICRO_UNIVERSE, MONTH_CODES, FuturesRoot, data_root,
)
from data.store import validate  # noqa: E402

DATASET_START = date(2010, 6, 6)  # GLBX.MDP3 begins here
TWO_DIGIT_YEARS = range(10, 40)  # 2010..2039: as far out as anything we care about lists
EXPIRY_SLACK = timedelta(days=7)  # holidays are not modelled; a bar this close to last trade is still that contract


def outright_codes(root: FuturesRoot) -> list[str]:
    """Every outright ticker the root can carry, in both year forms.

    Asked for by raw symbol on purpose. The parent symbol (CL.FUT) also
    resolves to every calendar spread, butterfly and inter-commodity spread the
    exchange lists - thousands of instruments for crude - and a dry run priced
    that at five times the outrights alone.
    """
    one = {f"{root.root}{MONTH_CODES[m]}{d}" for m in root.months for d in range(10)}
    two = {f"{root.root}{MONTH_CODES[m]}{yy:02d}" for m in root.months for yy in TWO_DIGIT_YEARS}
    return sorted(one | two)


def contract_year(root: FuturesRoot, month: int, digits: str, day: date) -> int:
    """Which year's contract a row of ticker `<root><month code><digits>` is."""
    if len(digits) >= 2:
        return 2000 + int(digits[-2:])
    year = day.year + ((int(digits) - day.year) % 10)  # first year at/after the row with that digit
    if root.last_trade(year, month) + EXPIRY_SLACK < day:
        year += 10  # printed after that contract died: it is the next decade's
    return year


def merge_sunday_stubs(bars: pd.DataFrame) -> pd.DataFrame:
    """Fold each Sunday UTC stub into the Monday that follows it."""
    if bars.empty:
        return bars
    b = bars.sort_values("ts").reset_index(drop=True)
    wd = b["ts"].dt.dayofweek.to_numpy()
    days = b["ts"].dt.normalize().to_numpy()
    keep = [True] * len(b)
    for i in range(len(b) - 1):
        if wd[i] == 6 and (days[i + 1] - days[i]) == pd.Timedelta(days=1):
            b.loc[i + 1, "open"] = b.loc[i, "open"]
            b.loc[i + 1, "high"] = max(b.loc[i, "high"], b.loc[i + 1, "high"])
            b.loc[i + 1, "low"] = min(b.loc[i, "low"], b.loc[i + 1, "low"])
            b.loc[i + 1, "volume"] = b.loc[i, "volume"] + b.loc[i + 1, "volume"]
            keep[i] = False
    return b[keep].reset_index(drop=True)


def contracts_from_family(df: pd.DataFrame, root: FuturesRoot):
    """Split a raw download into (year, month, bars) per contract."""
    parts: dict[tuple[int, int], list[pd.DataFrame]] = {}
    for sym, grp in df.groupby("symbol", sort=False):
        code = str(sym).replace(root.root, "", 1)
        if "-" in str(sym) or ":" in str(sym) or " " in str(sym) or len(code) < 2 \
                or code[0] not in CODE_MONTHS or not code[1:].isdigit():
            continue  # spreads, options, or something we do not model
        month, digits = CODE_MONTHS[code[0]], code[1:]
        years = grp["ts"].dt.date.map(lambda d: contract_year(root, month, digits, d))
        for year, seg in grp.groupby(years.to_numpy()):
            parts.setdefault((int(year), month), []).append(seg)
    for (year, month), segs in sorted(parts.items()):
        yield year, month, pd.concat(segs).sort_values("ts").drop_duplicates("ts")


def write_contracts(raw: pd.DataFrame, root: FuturesRoot, folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for year, month, seg in contracts_from_family(raw, root):
        bars = pd.DataFrame({
            "ts": seg["ts"].to_numpy(),
            "open": seg["open"].astype(float).to_numpy(), "high": seg["high"].astype(float).to_numpy(),
            "low": seg["low"].astype(float).to_numpy(), "close": seg["close"].astype(float).to_numpy(),
            "volume": seg["volume"].astype(float).to_numpy(),
        }).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
        bars = merge_sunday_stubs(bars)
        label = f"{root.root}{year}{month:02d}"
        try:
            validate(bars, label, allow_nonpositive=True)  # futures can print below zero; CL did
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: refused - {exc}"); continue
        bars.to_parquet(folder / f"{year}{month:02d}.parquet", index=False)
        written += 1
    return written


def missing_months(root: FuturesRoot, folder: Path, start: date, end: date) -> list[str]:
    """Contract months the calendar says traded in [start, end] with no file."""
    have = {f.stem for f in folder.glob("*.parquet")}
    out = []
    for y, m in root.listed(start, end):
        if start <= root.last_trade(y, m) <= end + timedelta(days=400) and f"{y}{m:02d}" not in have:
            out.append(f"{y}{m:02d}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=None, help="any research or micro root")
    ap.add_argument("--universe", choices=["micro", "full"], default=None,
                    help="every tradeable micro's parent, or all 33 research markets")
    ap.add_argument("--since", type=int, default=DATASET_START.year)
    ap.add_argument("--out", default="data/futures")
    ap.add_argument("--dataset", default="GLBX.MDP3")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="refuse to download if Databento's estimate exceeds this many dollars")
    ap.add_argument("--reparse", action="store_true",
                    help="rebuild the per-contract files from _raw/ without touching the API")
    ap.add_argument("--check", action="store_true", help="list contract months the calendar expects but has no file for")
    args = ap.parse_args()

    if args.roots and args.universe:
        print("give --roots or --universe, not both"); return 1
    names = args.roots or list(FULL_UNIVERSE if args.universe == "full" else MICRO_UNIVERSE)
    unknown = [n for n in names if n not in ALL_ROOTS]
    if unknown:
        print(f"unknown roots {unknown}; known: {', '.join(sorted(ALL_ROOTS))}"); return 1
    # de-duplicate to data roots: MES and ES are the same history
    targets = sorted({data_root(n).root for n in names})
    out = Path(args.out)
    raw_dir = out / "_raw"
    start = max(date(args.since, 1, 1), DATASET_START)
    end = date.today()

    if args.check:
        total = 0
        for t in targets:
            miss = missing_months(ALL_ROOTS[t], out / t, start + timedelta(days=45), end)
            total += len(miss)
            if miss:
                print(f"  {t:<4} missing {len(miss):>3}: {' '.join(miss[:12])}{' ...' if len(miss) > 12 else ''}")
        print(f"{total} contract months missing across {len(targets)} roots")
        return 0 if total == 0 else 4

    if args.reparse:
        for t in targets:
            path = raw_dir / f"{t}.parquet"
            if not path.exists():
                print(f"{t}: no raw download at {path}"); continue
            raw = pd.read_parquet(path)
            n = write_contracts(raw, ALL_ROOTS[t], out / t)
            print(f"{t:<4} {n:>4} contract months from {len(raw):,} raw bars")
        return 0

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY is not set.")
        print("  Get a key at databento.com (new accounts carry free credit), then, in your own shell:")
        print('    PowerShell:  $env:DATABENTO_API_KEY = "db-..."      cmd:  set DATABENTO_API_KEY=db-...')
        print("  Never paste the key into a chat or a file; the script reads it from the environment only.")
        print("  Daily bars are metered per gigabyte; --dry-run prints Databento's exact cost before anything is bought.")
        return 2

    import databento as db

    client = db.Historical(key)

    print(f"\n{len(targets)} data roots on {args.dataset}, {start} -> {end}, schema ohlcv-1d, outrights only")
    print("  " + " ".join(targets))
    try:
        total = 0.0
        for t in targets:
            cost = float(client.metadata.get_cost(
                dataset=args.dataset, symbols=outright_codes(ALL_ROOTS[t]), stype_in="raw_symbol",
                schema="ohlcv-1d", start=start.isoformat(), end=end.isoformat()))
            total += cost
            if args.dry_run:
                print(f"    {t:<4} ${cost:6.2f}")
        print(f"  Databento cost estimate: ${total:,.2f}")
        if args.max_cost is not None and total > args.max_cost:
            print(f"  exceeds --max-cost ${args.max_cost:,.2f}; nothing downloaded"); return 3
    except Exception as exc:  # noqa: BLE001
        print(f"  cost estimate unavailable: {type(exc).__name__}: {exc}")
        if args.dry_run or args.max_cost is not None:
            return 1
    if args.dry_run:
        print("  dry run: nothing downloaded"); return 0

    raw_dir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        codes = outright_codes(ALL_ROOTS[t])
        print(f"\n{t} ({len(codes)} outright tickers)")
        data = client.timeseries.get_range(
            dataset=args.dataset, symbols=codes, stype_in="raw_symbol",
            schema="ohlcv-1d", start=start.isoformat(), end=end.isoformat(),
        )
        df = data.to_df().reset_index()
        if df.empty:
            print("  nothing returned"); continue
        raw = pd.DataFrame({
            "symbol": df["symbol"].astype(str), "ts": pd.to_datetime(df["ts_event"], utc=True),
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float), "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
        })
        raw.to_parquet(raw_dir / f"{t}.parquet", index=False)  # the thing that cost money: keep it
        n = write_contracts(raw, ALL_ROOTS[t], out / t)
        print(f"  {n} contract months, {len(raw):,} bars, {raw['ts'].min():%Y-%m-%d} -> {raw['ts'].max():%Y-%m-%d}")

    print("\nnext: python scripts/download_databento.py --check --universe full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
