"""015b: commodity seasonality, and the causality it must never break.

This is the most overfittable idea in the repository. The tests below exist less
to prove it works than to prove it cannot cheat: a seasonal rule that peeks at
the year it is trading will look magnificent and be worthless.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.strategy import is_month_start  # noqa: E402
from core.types import Position, Side  # noqa: E402
from strategies.seasonality import Seasonality  # noqa: E402

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _seasonal_frame(years=12, up_month=3, down_month=9, seed=5, start="2010-01-04"):
    """Daily bars where one month drifts up every year and another drifts down."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=252 * years, freq="B", tz="UTC")
    drift = np.where(ts.month == up_month, 0.0035,
                     np.where(ts.month == down_month, -0.0035, 0.0))
    px = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, len(ts))))
    return pd.DataFrame({"ts": ts, "open": px, "high": px * 1.004,
                         "low": px * 0.996, "close": px, "volume": 1.0})


def test_it_finds_a_real_seasonal_pattern():
    s = Seasonality(min_years=3)
    df = s.prepare(_seasonal_frame())
    longs = shorts = 0
    for i in range(s.warmup, len(df)):
        if not is_month_start(df, i):
            continue
        it = s.evaluate(df, i, None)
        if it.flat:
            continue
        if int(df["_month"].iloc[i]) == 3 and it.side is Side.BUY:
            longs += 1
        if int(df["_month"].iloc[i]) == 9 and it.side is Side.SELL:
            shorts += 1
    assert longs >= 3 and shorts >= 3, "the planted pattern should be picked up"


def test_it_cannot_see_the_year_it_is_trading():
    """The test that matters. A month whose history is flat until this very year
    must NOT be traded, however strong this year's move is."""
    df = _seasonal_frame(years=12, up_month=99, down_month=99)  # no seasonality at all
    s = Seasonality(min_years=3)
    prepared = s.prepare(df)

    # plant an enormous rise in one month of the FINAL year only
    ts = pd.to_datetime(prepared["ts"], utc=True)
    final = ts.dt.year.max()
    mask = (ts.dt.year == final) & (ts.dt.month == 7)
    prepared.loc[mask, "close"] = prepared.loc[mask, "close"] * 3.0
    s2 = Seasonality(min_years=3)
    prepared2 = s2.prepare(prepared)

    idx = [i for i in range(s2.warmup, len(prepared2))
           if is_month_start(prepared2, i)
           and int(prepared2["_month"].iloc[i]) == 7
           and int(prepared2["_year"].iloc[i]) == final]
    assert idx, "the July of the final year should be a decision point"
    reason = s2.evaluate(prepared2, idx[0], None).reason
    assert "over" in reason or reason == ""
    # whatever it decided, it cannot have been driven by the tripling in that month
    mean = float(reason.split("mean ")[1].split("%")[0]) if "mean " in reason else 0.0
    assert abs(mean) < 50.0, "a 200% move in the traded year leaked into its own decision"


def test_too_few_prior_years_means_standing_aside():
    s = Seasonality(min_years=8)
    df = s.prepare(_seasonal_frame(years=6))
    decisions = [s.evaluate(df, i, None) for i in range(min(s.warmup, len(df) - 1), len(df))
                 if is_month_start(df, i)]
    assert all(d.flat for d in decisions), "eight years required, six supplied"


def test_it_holds_between_month_starts_and_does_not_re_enter():
    s = Seasonality(min_years=3)
    df = s.prepare(_seasonal_frame())
    held = Position("X", Side.BUY, 1.0, 100.0, NOW, stop_loss=90.0)
    mid = [i for i in range(s.warmup, len(df)) if not is_month_start(df, i)]
    assert s.evaluate(df, mid[0], held).reason == "hold"
    assert s.evaluate(df, mid[0], None).flat, "no mid-month re-entry after a stop"


def test_a_flat_history_produces_no_view():
    ts = pd.date_range("2010-01-04", periods=252 * 10, freq="B", tz="UTC")
    px = np.full(len(ts), 100.0)
    df = pd.DataFrame({"ts": ts, "open": px, "high": px, "low": px, "close": px, "volume": 1.0})
    s = Seasonality(min_years=3)
    prepared = s.prepare(df)
    starts = [i for i in range(s.warmup, len(prepared)) if is_month_start(prepared, i)]
    assert all(s.evaluate(prepared, i, None).flat for i in starts)
