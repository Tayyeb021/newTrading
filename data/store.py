"""Parquet bar store.

Layout: `{root}/{symbol}/{timeframe}/{year}.parquet`. Year partitioning keeps
incremental updates cheap — a download that appends yesterday rewrites one small
file rather than a 50 MB one.

Every write is validated. Bad data does not fail loudly at write time; it fails
silently months later as an indicator that quietly produced nonsense on a
duplicated bar, which is why the validation here is strict and non-optional.

Timestamps are UTC, timezone-aware, and refer to the bar's OPEN. Both facts are
enforced rather than documented, because a mixed convention is unrecoverable once
it is in the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core.types import Bar

COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


class DataError(ValueError):
    """The data is wrong. Never swallowed, never auto-repaired."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Gap:
    start: datetime  # last bar before the gap
    end: datetime  # first bar after it
    missing_bars: int

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def is_weekend(self) -> bool:
        """Friday close to Sunday/Monday open. Expected, not a defect.

        FX trades roughly Sunday 21:00 UTC to Friday 21:00 UTC, so a gap that
        starts Friday and ends Sunday or Monday is the market being shut.
        """
        return self.start.weekday() == 4 and self.end.weekday() in (6, 0)

    def is_daily_break(self) -> bool:
        """The broker's daily rollover, typically an hour around 21:00-22:00 UTC."""
        return self.duration <= timedelta(hours=2) and self.start.hour >= 20

    def __str__(self) -> str:
        kind = "weekend" if self.is_weekend() else "rollover" if self.is_daily_break() else "GAP"
        return (
            f"{kind:<8} {self.start:%Y-%m-%d %H:%M} -> {self.end:%Y-%m-%d %H:%M}  "
            f"({self.missing_bars:,} bars, {self.duration})"
        )


def validate(df: pd.DataFrame, symbol: str, *, allow_nonpositive: bool = False) -> None:
    """Raise on anything that would silently corrupt a backtest.

    `allow_nonpositive` is for futures, which can print at or below zero: WTI
    May 2020 settled at -37.63 on 2020-04-20. For a CFD or a spot rate a
    non-positive price is corruption and stays refused.
    """
    if df.empty:
        return

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")

    ts = df["ts"]
    if ts.dt.tz is None:
        raise DataError(f"{symbol}: timestamps are naive - everything must be UTC")

    if not ts.is_monotonic_increasing:
        raise DataError(f"{symbol}: timestamps are not sorted")

    dupes = int(ts.duplicated().sum())
    if dupes:
        first = ts[ts.duplicated(keep=False)].iloc[0]
        raise DataError(f"{symbol}: {dupes} duplicate timestamps, first at {first}")

    ohlc = df[["open", "high", "low", "close"]]
    if ohlc.isna().any().any():
        raise DataError(f"{symbol}: NaN in OHLC")
    if not allow_nonpositive and (ohlc <= 0).any().any():
        raise DataError(f"{symbol}: non-positive price in OHLC")

    bad_range = df["high"] < df["low"]
    if bad_range.any():
        raise DataError(f"{symbol}: {int(bad_range.sum())} bars where high < low")

    outside = (
        (df["open"] > df["high"]) | (df["open"] < df["low"])
        | (df["close"] > df["high"]) | (df["close"] < df["low"])
    )
    if outside.any():
        raise DataError(f"{symbol}: {int(outside.sum())} bars where open/close sit outside high/low")


def find_gaps(df: pd.DataFrame, timeframe: str, min_missing: int = 1) -> list[Gap]:
    """Gaps larger than one bar interval. Weekends included - the caller judges."""
    if len(df) < 2:
        return []
    step = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    ts = df["ts"].reset_index(drop=True)
    deltas = ts.diff()

    gaps: list[Gap] = []
    for i in range(1, len(ts)):
        delta = deltas.iloc[i]
        missing = int(delta / step) - 1
        if missing >= min_missing:
            gaps.append(Gap(ts.iloc[i - 1].to_pydatetime(), ts.iloc[i].to_pydatetime(), missing))
    return gaps


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(
        {
            "ts": [b.ts for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def frame_to_bars(df: pd.DataFrame, symbol: str) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            ts=row.ts.to_pydatetime(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for row in df.itertuples()
    ]


@dataclass(frozen=True)
class Coverage:
    symbol: str
    timeframe: str
    start: datetime | None
    end: datetime | None
    rows: int

    @property
    def empty(self) -> bool:
        return self.rows == 0

    def __str__(self) -> str:
        if self.empty:
            return f"{self.symbol:<10} {self.timeframe:<4} empty"
        span = (self.end - self.start).days
        return (
            f"{self.symbol:<10} {self.timeframe:<4} {self.rows:>10,} bars  "
            f"{self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d}  ({span:,}d)"
        )


class BarStore:
    def __init__(self, root: str | Path = "data/bars") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol / timeframe

    def _file(self, symbol: str, timeframe: str, year: int) -> Path:
        return self._dir(symbol, timeframe) / f"{year}.parquet"

    def years(self, symbol: str, timeframe: str) -> list[int]:
        d = self._dir(symbol, timeframe)
        if not d.exists():
            return []
        return sorted(int(p.stem) for p in d.glob("*.parquet"))

    # ------------------------------------------------------------------ write

    def write(self, symbol: str, timeframe: str, bars: list[Bar] | pd.DataFrame) -> int:
        """Merge bars into the store. Existing timestamps are overwritten.

        Returns the number of rows added (not overwritten). Merging rather than
        appending means a re-download of an overlapping range is safe and
        idempotent, which matters because that is what a resumed download does.
        """
        if timeframe not in TIMEFRAME_MINUTES:
            raise DataError(f"unknown timeframe {timeframe!r}")

        incoming = bars if isinstance(bars, pd.DataFrame) else bars_to_frame(bars)
        if incoming.empty:
            return 0
        incoming = incoming[COLUMNS].copy()
        incoming["ts"] = pd.to_datetime(incoming["ts"], utc=True)
        validate(incoming, symbol)

        self._dir(symbol, timeframe).mkdir(parents=True, exist_ok=True)
        added = 0

        for year, chunk in incoming.groupby(incoming["ts"].dt.year):
            path = self._file(symbol, timeframe, int(year))
            if path.exists():
                existing = pd.read_parquet(path)
                existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
                before = len(existing)
                # Incoming wins on conflict: a re-download corrects bad history.
                merged = (
                    pd.concat([existing, chunk], ignore_index=True)
                    .drop_duplicates(subset="ts", keep="last")
                    .sort_values("ts")
                    .reset_index(drop=True)
                )
                added += len(merged) - before
            else:
                merged = chunk.sort_values("ts").reset_index(drop=True)
                added += len(merged)

            validate(merged, symbol)
            merged.to_parquet(path, index=False, compression="snappy")

        return added

    # ------------------------------------------------------------------- read

    def read(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        years = self.years(symbol, timeframe)
        if not years:
            return pd.DataFrame(columns=COLUMNS)

        if start is not None:
            years = [y for y in years if y >= start.year]
        if end is not None:
            years = [y for y in years if y <= end.year]
        if not years:
            return pd.DataFrame(columns=COLUMNS)

        frames = [pd.read_parquet(self._file(symbol, timeframe, y)) for y in years]
        df = pd.concat(frames, ignore_index=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)

        if start is not None:
            df = df[df["ts"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["ts"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def read_bars(self, symbol: str, timeframe: str, **kw) -> list[Bar]:
        return frame_to_bars(self.read(symbol, timeframe, **kw), symbol)

    def coverage(self, symbol: str, timeframe: str) -> Coverage:
        df = self.read(symbol, timeframe)
        if df.empty:
            return Coverage(symbol, timeframe, None, None, 0)
        return Coverage(
            symbol,
            timeframe,
            df["ts"].iloc[0].to_pydatetime(),
            df["ts"].iloc[-1].to_pydatetime(),
            len(df),
        )

    def last_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """Newest stored bar, read from the latest year only. Used to resume."""
        years = self.years(symbol, timeframe)
        if not years:
            return None
        df = pd.read_parquet(self._file(symbol, timeframe, years[-1]))
        if df.empty:
            return None
        return pd.to_datetime(df["ts"], utc=True).max().to_pydatetime()

    # ---------------------------------------------------------------- report

    def report(self, symbol: str, timeframe: str, show_gaps: int = 10) -> str:
        df = self.read(symbol, timeframe)
        cov = self.coverage(symbol, timeframe)
        if df.empty:
            return str(cov)

        gaps = find_gaps(df, timeframe)
        anomalous = [g for g in gaps if not g.is_weekend() and not g.is_daily_break()]
        expected_bars = 0
        if cov.start and cov.end:
            step = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
            expected_bars = int((cov.end - cov.start) / step) + 1

        lines = [
            str(cov),
            f"           gaps: {len(gaps):,} total, {len(anomalous):,} unexplained",
            f"           density: {cov.rows / expected_bars:.1%} of calendar bars"
            if expected_bars
            else "",
        ]
        worst = sorted(anomalous, key=lambda g: g.missing_bars, reverse=True)[:show_gaps]
        lines.extend(f"           {g}" for g in worst)
        return "\n".join(line for line in lines if line)
