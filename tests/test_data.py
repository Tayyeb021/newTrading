"""Tests for the bar store.

The store's job is to refuse bad data rather than absorb it. Most of these tests
therefore assert that something raises — a duplicate timestamp or a naive datetime
must not reach disk, because neither is recoverable once a year of research has
been built on top of it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Bar  # noqa: E402
from data.download import month_ranges  # noqa: E402
from data.store import (  # noqa: E402
    BarStore,
    DataError,
    bars_to_frame,
    find_gaps,
    frame_to_bars,
    validate,
)

START = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)  # a Monday


def make_bars(n: int, symbol: str = "EURUSD", start: datetime = START, minutes: int = 1) -> list[Bar]:
    bars = []
    price = 1.08000
    for i in range(n):
        price += 0.00005 * (1 if i % 3 else -1)
        bars.append(
            Bar(
                symbol=symbol,
                ts=start + timedelta(minutes=minutes * i),
                open=price,
                high=price + 0.00020,
                low=price - 0.00020,
                close=price + 0.00005,
                volume=100 + i,
            )
        )
    return bars


@pytest.fixture()
def store(tmp_path: Path) -> BarStore:
    return BarStore(tmp_path / "bars")


# ------------------------------------------------------------------ round trip

def test_write_then_read(store: BarStore):
    bars = make_bars(500)
    added = store.write("EURUSD", "M1", bars)
    assert added == 500

    df = store.read("EURUSD", "M1")
    assert len(df) == 500
    assert df["ts"].is_monotonic_increasing
    assert str(df["ts"].dt.tz) == "UTC"


def test_bars_survive_the_round_trip(store: BarStore):
    original = make_bars(50)
    store.write("EURUSD", "M1", original)
    restored = store.read_bars("EURUSD", "M1")

    assert len(restored) == len(original)
    assert restored[0].ts == original[0].ts
    assert restored[-1].close == pytest.approx(original[-1].close)
    assert all(b.ts.tzinfo is not None for b in restored)


def test_writes_are_idempotent(store: BarStore):
    bars = make_bars(200)
    assert store.write("EURUSD", "M1", bars) == 200
    assert store.write("EURUSD", "M1", bars) == 0  # same data, nothing new
    assert len(store.read("EURUSD", "M1")) == 200


def test_overlapping_write_merges_without_duplicates(store: BarStore):
    store.write("EURUSD", "M1", make_bars(100))
    store.write("EURUSD", "M1", make_bars(100, start=START + timedelta(minutes=50)))

    df = store.read("EURUSD", "M1")
    assert len(df) == 150
    assert not df["ts"].duplicated().any()


def test_redownload_corrects_bad_history(store: BarStore):
    """A re-download overwrites what is already stored, so bad history is fixable.

    The replacement must itself be valid: the store will not accept a "correction"
    whose close sits outside its own high/low, which is the right behaviour and
    worth asserting in both directions.
    """
    store.write("EURUSD", "M1", make_bars(10))

    fixed = [
        Bar(b.symbol, b.ts, b.open, b.high + 0.005, b.low, b.high + 0.004, b.volume)
        if i == 5 else b
        for i, b in enumerate(make_bars(10))
    ]
    store.write("EURUSD", "M1", fixed)
    df = store.read("EURUSD", "M1")
    assert df["close"].iloc[5] == pytest.approx(fixed[5].close)  # incoming wins
    assert len(df) == 10  # corrected, not appended

    # An invalid correction is refused rather than absorbed.
    broken = bars_to_frame(make_bars(10))
    broken.loc[5, "close"] = broken.loc[5, "high"] + 1.0
    with pytest.raises(DataError, match="outside"):
        store.write("EURUSD", "M1", broken)


def test_year_partitioning(store: BarStore):
    across = make_bars(3, start=datetime(2025, 12, 31, 23, 58, tzinfo=timezone.utc))
    store.write("EURUSD", "M1", across)
    assert store.years("EURUSD", "M1") == [2025, 2026]
    assert len(store.read("EURUSD", "M1")) == 3


def test_date_range_read(store: BarStore):
    store.write("EURUSD", "M1", make_bars(1000))
    lo = START + timedelta(minutes=100)
    hi = START + timedelta(minutes=200)
    df = store.read("EURUSD", "M1", start=lo, end=hi)
    assert len(df) == 101
    assert df["ts"].iloc[0] == pd.Timestamp(lo)


# ------------------------------------------------------------------ validation

def test_naive_timestamps_are_rejected():
    df = pd.DataFrame({
        "ts": [datetime(2026, 1, 5, 0, 0)],
        "open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [10.0],
    })
    with pytest.raises(DataError, match="naive"):
        validate(df, "EURUSD")


def test_duplicate_timestamps_are_rejected():
    bars = make_bars(5)
    df = bars_to_frame(bars + [bars[2]]).sort_values("ts").reset_index(drop=True)
    with pytest.raises(DataError, match="duplicate"):
        validate(df, "EURUSD")


def test_unsorted_timestamps_are_rejected():
    df = bars_to_frame(make_bars(5)).iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataError, match="not sorted"):
        validate(df, "EURUSD")


def test_high_below_low_is_rejected():
    df = bars_to_frame(make_bars(3))
    df.loc[1, "high"] = df.loc[1, "low"] - 0.001
    with pytest.raises(DataError, match="high < low"):
        validate(df, "EURUSD")


def test_close_outside_the_bar_is_rejected():
    df = bars_to_frame(make_bars(3))
    df.loc[1, "close"] = df.loc[1, "high"] + 0.01
    with pytest.raises(DataError, match="outside"):
        validate(df, "EURUSD")


def test_bad_data_never_reaches_disk(store: BarStore):
    good = make_bars(10)
    store.write("EURUSD", "M1", good)
    bad = bars_to_frame(make_bars(10))
    bad.loc[3, "high"] = 0.0
    with pytest.raises(DataError):
        store.write("EURUSD", "M1", bad)
    assert len(store.read("EURUSD", "M1")) == 10  # untouched


# ----------------------------------------------------------------------- gaps

def test_gaps_are_found_and_measured():
    bars = make_bars(10) + make_bars(10, start=START + timedelta(minutes=60))
    gaps = find_gaps(bars_to_frame(bars), "M1")
    assert len(gaps) == 1
    assert gaps[0].missing_bars == 50


def test_weekend_gap_is_classified_as_expected():
    friday = datetime(2026, 1, 9, 20, 55, tzinfo=timezone.utc)
    sunday = datetime(2026, 1, 11, 22, 0, tzinfo=timezone.utc)
    bars = make_bars(5, start=friday) + make_bars(5, start=sunday)
    gaps = find_gaps(bars_to_frame(bars), "M1")
    assert len(gaps) == 1
    assert gaps[0].is_weekend()


def test_midweek_gap_is_not_excused():
    wednesday = datetime(2026, 1, 7, 10, 0, tzinfo=timezone.utc)
    bars = make_bars(5, start=wednesday) + make_bars(
        5, start=wednesday + timedelta(hours=4)
    )
    gaps = find_gaps(bars_to_frame(bars), "M1")
    assert len(gaps) == 1
    assert not gaps[0].is_weekend()
    assert not gaps[0].is_daily_break()


# ------------------------------------------------------------------- coverage

def test_coverage_and_resume_point(store: BarStore):
    assert store.coverage("EURUSD", "M1").empty
    assert store.last_timestamp("EURUSD", "M1") is None

    bars = make_bars(300)
    store.write("EURUSD", "M1", bars)

    cov = store.coverage("EURUSD", "M1")
    assert cov.rows == 300
    assert cov.start == bars[0].ts
    assert cov.end == bars[-1].ts
    assert store.last_timestamp("EURUSD", "M1") == bars[-1].ts


def test_report_runs_on_real_shaped_data(store: BarStore):
    bars = make_bars(100) + make_bars(100, start=START + timedelta(hours=5))
    store.write("EURUSD", "M1", bars)
    text = store.report("EURUSD", "M1")
    assert "EURUSD" in text and "gaps" in text


def test_symbols_and_timeframes_are_isolated(store: BarStore):
    store.write("EURUSD", "M1", make_bars(10))
    store.write("XAUUSD", "M1", make_bars(20, symbol="XAUUSD"))
    store.write("EURUSD", "H1", make_bars(5, minutes=60))

    assert len(store.read("EURUSD", "M1")) == 10
    assert len(store.read("XAUUSD", "M1")) == 20
    assert len(store.read("EURUSD", "H1")) == 5
    assert store.read("GBPUSD", "M1").empty


def test_unknown_timeframe_is_rejected(store: BarStore):
    with pytest.raises(DataError, match="unknown timeframe"):
        store.write("EURUSD", "M7", make_bars(3))


# ------------------------------------------------------------------ download

def test_month_ranges_tile_the_span_without_overlap():
    start = datetime(2025, 11, 15, tzinfo=timezone.utc)
    end = datetime(2026, 2, 10, tzinfo=timezone.utc)
    windows = month_ranges(start, end)

    assert len(windows) == 4
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for (_, a_end), (b_start, _) in zip(windows, windows[1:]):
        assert a_end < b_start


def test_month_ranges_crosses_the_year_boundary():
    windows = month_ranges(
        datetime(2025, 12, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 31, tzinfo=timezone.utc),
    )
    assert len(windows) == 2
    assert windows[0][0].year == 2025
    assert windows[1][0].year == 2026


def test_frame_to_bars_preserves_symbol():
    df = bars_to_frame(make_bars(3))
    bars = frame_to_bars(df, "XAUUSD")
    assert all(b.symbol == "XAUUSD" for b in bars)
