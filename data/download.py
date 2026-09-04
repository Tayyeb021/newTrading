"""Historical download from the broker.

Two things make this less trivial than it looks.

**Chunking.** Five years of M1 is roughly 1.8 million bars per symbol. The MT5
terminal caps how much history it will hand over in one call — the limit is a
terminal setting, not an API constant — so a single wide request silently returns
a truncated series rather than failing. Requesting month by month keeps every call
comfortably inside any plausible cap.

**Broker time is not UTC.** Most MT5 servers run GMT+2/+3 with daylight saving, so
the terminal's bar timestamps are shifted. The adapter converts on the way in by
treating server epochs as UTC; whether that is correct for *your* broker is an
empirical question, and `--check-session` answers it by showing when the trading
week actually starts and ends in the data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from data.store import BarStore, bars_to_frame
from execution.base import ExecutionAdapter, ExecutionError

log = logging.getLogger(__name__)


def month_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into calendar-month windows."""
    out: list[tuple[datetime, datetime]] = []
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor <= end:
        if cursor.month == 12:
            nxt = cursor.replace(year=cursor.year + 1, month=1)
        else:
            nxt = cursor.replace(month=cursor.month + 1)
        out.append((max(cursor, start), min(nxt - timedelta(seconds=1), end)))
        cursor = nxt
    return out


def download(
    adapter: ExecutionAdapter,
    store: BarStore,
    symbol: str,
    timeframe: str,
    years: float = 5.0,
    end: datetime | None = None,
    resume: bool = True,
    progress: bool = True,
) -> int:
    """Fetch history month by month and merge into the store.

    With `resume=True`, starts from the newest stored bar minus one day of
    overlap. The overlap is deliberate: the merge is idempotent, and re-fetching a
    day is far cheaper than discovering a one-bar hole at the seam later.
    """
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365.25))

    if resume:
        last = store.last_timestamp(symbol, timeframe)
        if last is not None and last > start:
            start = last - timedelta(days=1)
            if progress:
                print(f"  resuming {symbol} {timeframe} from {start:%Y-%m-%d}")

    if start >= end:
        if progress:
            print(f"  {symbol} {timeframe} already current")
        return 0

    total_added = 0
    windows = month_ranges(start, end)

    for i, (window_start, window_end) in enumerate(windows, 1):
        try:
            bars = _fetch_range(adapter, symbol, timeframe, window_start, window_end)
        except ExecutionError as exc:
            # A month with no data is normal at the edges of a broker's history.
            log.warning("%s %s %s: %s", symbol, timeframe, window_start.date(), exc)
            continue

        if not bars:
            continue
        added = store.write(symbol, timeframe, bars)
        total_added += added

        if progress:
            pct = i / len(windows)
            print(
                f"\r  {symbol} {timeframe} [{'#' * int(pct * 24):<24}] "
                f"{pct:>4.0%}  {window_start:%Y-%m}  +{total_added:,} bars",
                end="",
                flush=True,
            )

    if progress:
        print()
    return total_added


def _fetch_range(
    adapter: ExecutionAdapter,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list:
    """One month of bars.

    `bars()` on the adapter takes a count and an end. A month of M1 is at most
    ~44,640 bars; ask for a comfortable ceiling and filter to the window, which
    keeps the adapter interface small rather than growing a range variant for one
    caller.
    """
    from data.store import TIMEFRAME_MINUTES

    minutes = TIMEFRAME_MINUTES[timeframe]
    span_minutes = (end - start).total_seconds() / 60
    count = int(span_minutes / minutes) + 64

    bars = adapter.bars(symbol, timeframe, count=count, end=end)
    return [b for b in bars if start <= b.ts <= end]


def download_all(
    adapter: ExecutionAdapter,
    store: BarStore,
    symbols: list[str],
    timeframes: list[str],
    years: float = 5.0,
) -> dict[tuple[str, str], int]:
    results: dict[tuple[str, str], int] = {}
    for symbol in symbols:
        for tf in timeframes:
            try:
                results[(symbol, tf)] = download(adapter, store, symbol, tf, years=years)
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not stop the run
                print(f"  ! {symbol} {tf}: {exc}")
                results[(symbol, tf)] = 0
    return results


def session_profile(store: BarStore, symbol: str, timeframe: str = "M1") -> str:
    """When does this symbol's data actually exist, by UTC hour and weekday?

    The empirical answer to "is my broker's clock what I think it is". If the
    trading week appears to start Sunday at 22:00 rather than 21:00, the server
    runs an offset you have not accounted for, and every session filter you write
    on top of this data will be an hour wrong.
    """
    df = store.read(symbol, timeframe)
    if df.empty:
        return f"{symbol}: no data"

    ts = df["ts"]
    by_hour = ts.dt.hour.value_counts().sort_index()
    by_dow = ts.dt.dayofweek.value_counts().sort_index()
    peak = int(by_hour.idxmax())

    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines = [f"{symbol} {timeframe} - {len(df):,} bars"]

    lines.append("  bars by UTC hour:")
    top = by_hour.max()
    for hour, n in by_hour.items():
        bar = "#" * int(24 * n / top)
        lines.append(f"    {hour:02d}  {bar:<24} {n:>8,}")

    lines.append("  bars by weekday:")
    for dow, n in by_dow.items():
        lines.append(f"    {names[int(dow)]}  {n:>10,}")

    week_start = ts[ts.dt.dayofweek == 6]
    week_end = ts[ts.dt.dayofweek == 4]
    if not week_start.empty:
        lines.append(f"  earliest Sunday bar: {week_start.dt.hour.min():02d}:00 UTC")
    if not week_end.empty:
        lines.append(f"  latest Friday bar:   {week_end.dt.hour.max():02d}:00 UTC")
    lines.append(f"  busiest hour: {peak:02d}:00 UTC")
    return "\n".join(lines)
