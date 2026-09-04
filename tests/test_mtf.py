"""Tests for multi-timeframe alignment.

The first test is the reason this file exists. Everything else supports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Side  # noqa: E402
from features.mtf import (  # noqa: E402
    align_higher_timeframe,
    bars_until,
    resample,
    session_mask,
    timeframe_delta,
)
from strategies.mtf_pullback import MTFPullback, bias_frame  # noqa: E402


def m5_frame(hours: int = 48, start="2026-03-02 00:00") -> pd.DataFrame:
    n = hours * 12
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(5)
    close = 1.08 + np.cumsum(rng.normal(0, 0.0002, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    pad = np.abs(rng.normal(0, 0.0001, n))
    return pd.DataFrame({
        "ts": ts, "open": open_,
        "high": np.maximum(open_, close) + pad,
        "low": np.minimum(open_, close) - pad,
        "close": close, "volume": np.full(n, 10.0),
    })


# ==========================================================================
# The look-ahead test
# ==========================================================================

def test_htf_alignment_has_no_lookahead():
    """An H4 bar must not be visible until it has closed.

    The H4 series is built so each bar's `marker` equals its own index. If
    alignment leaked, an execution bar inside the 12:00-16:00 H4 window would
    carry that window's marker. Correct behaviour is the PREVIOUS window's.
    """
    exec_ts = pd.date_range("2026-03-02 00:00", periods=96, freq="15min", tz="UTC")
    exec_df = pd.DataFrame({"ts": exec_ts, "close": np.arange(96, dtype=float)})

    htf_ts = pd.date_range("2026-03-02 00:00", periods=6, freq="4h", tz="UTC")
    htf = pd.DataFrame({"ts": htf_ts, "marker": np.arange(6, dtype=float)})

    merged = align_higher_timeframe(exec_df, htf, "H4", columns=["marker"])

    for row in merged.itertuples():
        ts = row.ts
        window = int((ts - htf_ts[0]) // pd.Timedelta(hours=4))
        expected = float(window - 1) if window >= 1 else np.nan

        if np.isnan(expected):
            assert pd.isna(row.h4_marker), f"{ts}: saw a bar before any had closed"
        else:
            assert row.h4_marker == expected, (
                f"{ts}: got marker {row.h4_marker}, expected {expected}. "
                f"Marker {window} belongs to the bar still forming - that is the leak."
            )


def test_alignment_boundary_is_exact_at_the_close():
    """At exactly 04:00 the 00:00-04:00 bar has just closed and IS usable."""
    exec_df = pd.DataFrame({
        "ts": pd.to_datetime(["2026-03-02 03:55", "2026-03-02 04:00", "2026-03-02 04:05"], utc=True),
        "close": [1.0, 1.0, 1.0],
    })
    htf = pd.DataFrame({
        "ts": pd.to_datetime(["2026-03-02 00:00", "2026-03-02 04:00"], utc=True),
        "marker": [10.0, 20.0],
    })
    merged = align_higher_timeframe(exec_df, htf, "H4", columns=["marker"])

    assert pd.isna(merged["h4_marker"].iloc[0])  # 03:55 - nothing closed yet
    assert merged["h4_marker"].iloc[1] == 10.0   # 04:00 - the 00:00 bar just closed
    assert merged["h4_marker"].iloc[2] == 10.0   # 04:05 - still the 00:00 bar


def test_daily_bias_does_not_leak_into_the_same_day():
    exec_ts = pd.date_range("2026-03-02 00:00", periods=48, freq="1h", tz="UTC")
    exec_df = pd.DataFrame({"ts": exec_ts, "close": 1.0})
    d1 = pd.DataFrame({
        "ts": pd.to_datetime(["2026-03-02", "2026-03-03"], utc=True),
        "bias": [1.0, -1.0],
    })
    merged = align_higher_timeframe(exec_df, d1, "D1", columns=["bias"])

    day_one = merged[merged["ts"] < "2026-03-03"]
    assert day_one["d1_bias"].isna().all(), "today's daily bar leaked into today"

    day_two = merged[merged["ts"] >= "2026-03-03"]
    assert (day_two["d1_bias"] == 1.0).all(), "day two should see day one's bar"


def test_strategy_signals_are_unchanged_by_future_data():
    """End-to-end: truncating the data must not change earlier signals."""
    exec_df = m5_frame(hours=200)
    h4 = resample(exec_df, "H4")
    strategy = MTFPullback(execution_timeframe="M5", bias_frames={"H4": h4},
                           bias_timeframes=("H4",))

    full = strategy.prepare(exec_df.copy()).reset_index(drop=True)
    checked = 0
    for i in range(strategy.warmup, len(exec_df) - 1, 53):
        cut = exec_df.iloc[: i + 1].copy()
        partial_strategy = MTFPullback(
            execution_timeframe="M5",
            bias_frames={"H4": resample(cut, "H4")},
            bias_timeframes=("H4",),
        )
        partial = partial_strategy.prepare(cut).reset_index(drop=True)

        a = strategy.evaluate(full, i, None)
        b = partial_strategy.evaluate(partial, i, None)
        assert a.side == b.side, f"bar {i}: signal changed when future data was removed"
        checked += 1
    assert checked > 5


# ==========================================================================
# Helpers
# ==========================================================================

def test_timeframe_delta():
    assert timeframe_delta("M5") == pd.Timedelta(minutes=5)
    assert timeframe_delta("H4") == pd.Timedelta(hours=4)
    assert timeframe_delta("D1") == pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="unknown timeframe"):
        timeframe_delta("M7")


def test_session_mask_normal_and_overnight():
    ts = pd.Series(pd.date_range("2026-03-02 00:00", periods=24, freq="1h", tz="UTC"))
    london = session_mask(ts, 7, 20)
    assert london.sum() == 13
    assert not london.iloc[3] and london.iloc[10]

    overnight = session_mask(ts, 22, 6)
    assert overnight.sum() == 8
    assert overnight.iloc[23] and overnight.iloc[2] and not overnight.iloc[12]


def test_bars_until_counts_down_to_the_target_hour():
    ts = pd.Series(pd.to_datetime(
        ["2026-03-02 18:00", "2026-03-02 19:30", "2026-03-02 21:00"], utc=True))
    remaining = bars_until(ts, hour=20, timeframe="M15")
    assert remaining.iloc[0] == 8    # 2 hours
    assert remaining.iloc[1] == 2    # 30 minutes
    assert remaining.iloc[2] == 92   # rolls to tomorrow


def test_resample_builds_correct_higher_bars():
    df = m5_frame(hours=24)
    h1 = resample(df, "H1")
    assert len(h1) == 24

    first = df[df["ts"] < df["ts"].iloc[0] + pd.Timedelta(hours=1)]
    assert h1["open"].iloc[0] == pytest.approx(first["open"].iloc[0])
    assert h1["close"].iloc[0] == pytest.approx(first["close"].iloc[-1])
    assert h1["high"].iloc[0] == pytest.approx(first["high"].max())
    assert h1["low"].iloc[0] == pytest.approx(first["low"].min())


def test_bias_frame_is_ternary():
    df = m5_frame(hours=400)
    h4 = resample(df, "H4")
    bias = bias_frame(h4)
    assert set(bias["bias"].unique()) <= {-1, 0, 1}
    assert len(bias) == len(h4)


# ==========================================================================
# Strategy rules
# ==========================================================================

def _prepared(**kwargs):
    exec_df = m5_frame(hours=300)
    h4 = resample(exec_df, "H4")
    s = MTFPullback(execution_timeframe="M5", bias_frames={"H4": h4},
                    bias_timeframes=("H4",), **kwargs)
    return s, s.prepare(exec_df.copy()).reset_index(drop=True)


def test_no_trade_outside_the_session():
    s, df = _prepared(session_start=7, session_end=20)
    outside = df[(~df["in_session"]) & (df.index > s.warmup)]
    assert not outside.empty
    for i in outside.index[:60]:
        assert s.evaluate(df, int(i), None).flat, "traded outside the session window"


def test_no_trade_without_higher_timeframe_agreement():
    s, df = _prepared()
    neutral = df[(df["bias"] == 0) & (df.index > s.warmup)]
    for i in neutral.index[:60]:
        assert s.evaluate(df, int(i), None).flat, "traded with no bias"


def test_extended_price_is_not_chased():
    s, df = _prepared(max_extension_atr=0.5)
    extended = df[(df["ext_atr"].abs() > 0.5) & (df["bias"] != 0)
                  & df["in_session"] & (df.index > s.warmup)]
    for i in extended.index[:60]:
        assert s.evaluate(df, int(i), None).flat, "chased an extended move"


def test_position_is_closed_before_rollover():
    """The gold-swap rule. Holding overnight costs more than the edge."""
    from core.types import Position
    from datetime import datetime, timezone

    s, df = _prepared(flat_by_hour=20, min_bars_before_flat=4)
    late = df[(df["bars_to_flat"] <= 4) & (df.index > s.warmup)]
    assert not late.empty

    held = Position("EURUSD", Side.BUY, 0.1, 1.08,
                    datetime(2026, 3, 2, tzinfo=timezone.utc), stop_loss=1.07)
    for i in late.index[:40]:
        assert s.evaluate(df, int(i), held).flat, "held a position into rollover"


def test_exit_when_the_higher_timeframe_turns():
    from core.types import Position
    from datetime import datetime, timezone

    s, df = _prepared()
    against = df[(df["bias"] == -1) & (df["bars_to_flat"] > 20) & (df.index > s.warmup)]
    if against.empty:
        pytest.skip("no opposing-bias bars in this sample")

    held = Position("EURUSD", Side.BUY, 0.1, 1.08,
                    datetime(2026, 3, 2, tzinfo=timezone.utc), stop_loss=1.07)
    i = int(against.index[0])
    assert s.evaluate(df, i, held).flat, "held through a bias flip"


def test_every_entry_carries_a_stop_scaled_to_atr():
    s, df = _prepared()
    entries = 0
    for i in range(s.warmup, len(df) - 1):
        intent = s.evaluate(df, i, None)
        if intent.flat:
            continue
        entries += 1
        expected = float(df.iloc[i]["atr"]) * s.atr_stop_multiple
        assert intent.stop_distance == pytest.approx(expected)
    assert entries > 0, "the strategy never traded on this sample"
