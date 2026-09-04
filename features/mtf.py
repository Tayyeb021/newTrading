"""Multi-timeframe alignment.

Top-down trading reads direction from a higher timeframe and times entries on a
lower one. Backtesting it correctly turns on a single question:

**At 14:07, what is the current H4 bar?**

The H4 bar stamped 12:00 covers 12:00-16:00 and has not closed. It knows the
16:00 price. Joining it to a 14:07 execution bar hands the strategy four hours of
future information, and the resulting equity curve is spectacular and completely
untradeable. This is the most common bug in multi-timeframe backtests, and it is
invisible: nothing errors, the trades look sensible, and the strategy simply
stops working live.

So alignment here is on the higher-timeframe bar's **close time**, not its open
time, and a bar is only visible once `open + duration <= now`. At 14:07 the most
recent usable H4 bar is the one stamped 08:00, which closed at 12:00.

`tests/test_mtf.py::test_htf_alignment_has_no_lookahead` proves it on data
constructed so that any leak is arithmetically detectable.
"""

from __future__ import annotations

import pandas as pd

TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    except KeyError:
        raise ValueError(f"unknown timeframe {timeframe!r}") from None


def align_higher_timeframe(
    exec_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_timeframe: str,
    columns: list[str] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Attach higher-timeframe columns to execution bars, without leaking.

    Only bars that have **closed** at or before the execution bar's open time are
    visible. Returns `exec_df` with the requested columns added, prefixed by the
    timeframe (``h4_bias``, ``d1_ema``) so the source of every value is obvious
    at the point of use.

    Both frames need a timezone-aware `ts` column.
    """
    if exec_df.empty or htf_df.empty:
        return exec_df.copy()

    prefix = (prefix or htf_timeframe).lower() + "_"
    columns = columns or [c for c in htf_df.columns if c != "ts"]
    missing = [c for c in columns if c not in htf_df.columns]
    if missing:
        raise KeyError(f"higher-timeframe frame is missing {missing}")

    right = htf_df[["ts", *columns]].copy()
    right["ts"] = pd.to_datetime(right["ts"], utc=True)
    # The instant the bar becomes knowable. This one line is the whole point.
    right["available_at"] = right["ts"] + timeframe_delta(htf_timeframe)
    right = right.rename(columns={c: prefix + c for c in columns})
    right = right.drop(columns=["ts"]).sort_values("available_at")

    left = exec_df.copy()
    left["ts"] = pd.to_datetime(left["ts"], utc=True)
    left = left.sort_values("ts")

    merged = pd.merge_asof(
        left,
        right,
        left_on="ts",
        right_on="available_at",
        direction="backward",  # most recent bar that has already closed
        allow_exact_matches=True,
    )
    return merged.drop(columns=["available_at"]).reset_index(drop=True)


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Build higher-timeframe bars from lower ones.

    Useful when the broker's history is deeper on one frame than another - here
    M30 reaches back four years while M5 reaches eight months, so an M30 series
    resampled from M5 would be *shallower*, not deeper. Prefer the broker's own
    higher-timeframe series; use this only to fill a gap.
    """
    rule = f"{TIMEFRAME_MINUTES[timeframe]}min"
    out = (
        df.set_index(pd.to_datetime(df["ts"], utc=True))
        .resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
        .rename(columns={"index": "ts"})
    )
    return out


def session_mask(
    ts: pd.Series,
    start_hour: int,
    end_hour: int,
) -> pd.Series:
    """Bars inside a UTC trading window, handling windows that cross midnight.

    Intraday strategies live or die on this. Outside the liquid sessions you pay
    a wider spread for less movement, which is the fastest way to turn a real
    edge into a losing system.
    """
    hours = pd.to_datetime(ts, utc=True).dt.hour
    if start_hour <= end_hour:
        return (hours >= start_hour) & (hours < end_hour)
    return (hours >= start_hour) | (hours < end_hour)


def bars_until(ts: pd.Series, hour: int, timeframe: str) -> pd.Series:
    """Execution bars remaining until a given UTC hour.

    Drives the flat-before-rollover rule. On this broker a gold long costs
    -512.30 cents per lot per night, which is more than four M5 ranges of
    movement - so an intraday system that drifts into the overnight session pays
    away more than it can plausibly make.
    """
    t = pd.to_datetime(ts, utc=True)
    target = t.dt.normalize() + pd.Timedelta(hours=hour)
    target = target.where(target > t, target + pd.Timedelta(days=1))
    minutes = (target - t).dt.total_seconds() / 60.0
    return (minutes / TIMEFRAME_MINUTES[timeframe]).astype(int)
