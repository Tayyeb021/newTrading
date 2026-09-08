"""016: the order-flow screen, and the four ways it could lie without failing."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import ALL_ROOTS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "orderflow_016", Path(__file__).resolve().parent.parent / "research" / "orderflow_016.py")
of = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(of)


def _prints(n: int, start: str = "2026-06-15 14:00", freq: str = "1s",
            spread: float = 0.25, base: float = 6000.0) -> pd.DataFrame:
    """Alternating buy and sell aggressors one spread apart, the simplest real book."""
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    side = np.where(np.arange(n) % 2 == 0, 1, -1)
    price = base + np.where(side > 0, spread, 0.0)
    return pd.DataFrame({"ts": ts, "price": price, "size": np.ones(n), "side": side})


def test_measured_spread_recovers_the_spread_that_was_put_in():
    """The cost threshold is built on this number; an assumed 0.25 would pass
    silently on a market whose real spread is four ticks."""
    assert of.measured_spread(_prints(5000, spread=0.25), sample=5000) == 0.25
    assert of.measured_spread(_prints(5000, spread=1.00), sample=5000) == 1.00


def test_spread_ignores_prints_that_are_not_a_crossing():
    """Two trades an hour apart are not a bid and an offer being crossed."""
    p = _prints(5000, freq="1h", spread=2.0)
    assert np.isnan(of.measured_spread(p, sample=5000))


def test_front_windows_never_gives_a_day_to_the_back_month(tmp_path):
    """Databento returns the full range for every symbol asked for. Scoring the
    back month would be scoring a book nobody was trading."""
    folder = tmp_path / "ES"
    folder.mkdir()
    for code in ("ESM6", "ESU6"):
        _prints(10, start="2026-06-05 14:00", freq="1D").to_parquet(folder / f"{code}.parquet")
    windows = of.front_windows(ALL_ROOTS["ES"], sorted(folder.glob("*.parquet")))

    assert set(windows) == {"ESM6", "ESU6"}
    m_lo, m_hi = windows["ESM6"]
    u_lo, u_hi = windows["ESU6"]
    assert m_hi < u_lo, "the two front-month windows must not overlap"
    assert m_lo == date(2026, 6, 5)


def test_divergence_signal_is_short_when_price_makes_a_high_without_delta():
    """The library returns +1 for a BEARISH divergence. If the screen forgets to
    flip it, a losing rule scores as a winning one and nothing else notices."""
    n = 60
    bars = pd.DataFrame({
        "ts": pd.date_range("2026-06-15", periods=n, freq="5min", tz="UTC"),
        "open": 6000.0 + np.arange(n), "high": 6001.0 + np.arange(n),
        "low": 5999.0 + np.arange(n), "close": 6000.0 + np.arange(n),
        "volume": 100.0, "delta": -1.0,      # price climbing on selling pressure
    })
    bars["cum_delta"] = bars["delta"].cumsum()

    from features.orderflow import delta_divergence
    raw = delta_divergence(bars, 20)
    assert (raw == 1).any(), "the fixture must actually produce a bearish divergence"

    obs = of.bar_observations(bars, -raw.to_numpy(dtype=float))
    assert (obs["sig"] == -1).any(), "a bearish divergence must be scored as a short"
    assert not (obs["sig"] == 1).any()


def test_last_bar_of_a_contract_has_no_forward_return():
    """Pooling per contract is the whole defence against measuring the roll gap
    as a delta effect."""
    n = 40
    bars = pd.DataFrame({
        "ts": pd.date_range("2026-06-15", periods=n, freq="5min", tz="UTC"),
        "open": 6000.0, "high": 6001.0, "low": 5999.0, "close": 6000.0,
        "volume": 100.0, "delta": 1.0,
    })
    obs = of.bar_observations(bars, np.ones(n))
    assert len(obs) < n
    assert obs["ts"].max() < bars["ts"].iloc[-1]


def _session_bars(days: list[str], drop: str | None = None) -> pd.DataFrame:
    rows = []
    for d in days:
        # Each day starts at midnight, as ES bars really do, so the 14:00 entry
        # bar has a warm ATR. With only the cash session the very first day is
        # dropped for want of 14 bars - which is what the real run does too.
        ts = pd.date_range(f"{d} 00:00", f"{d} 20:00", freq="5min", tz="UTC")
        if drop:
            ts = ts[ts != pd.Timestamp(f"{d} {drop}", tz="UTC")]
        rows.append(pd.DataFrame({
            "ts": ts, "open": 6000.0, "high": 6002.0, "low": 5998.0,
            "close": 6000.0 + np.arange(len(ts)), "volume": 100.0, "delta": 5.0}))
    b = pd.concat(rows, ignore_index=True)
    b["cum_delta"] = b["delta"].cumsum()
    return b


def test_session_window_is_measured_by_the_clock_not_by_bar_count():
    """A bar-count horizon lands on the wrong day the first time a bar is
    missing, and reports a number either way."""
    days = [f"2026-06-{d:02d}" for d in (15, 16, 17, 18, 19)]
    full = of.session_observations(_session_bars(days))
    holed = of.session_observations(_session_bars(days, drop="15:00"))

    assert len(full) == len(days)
    assert len(holed) == len(days), "a missing mid-session bar must not lose the session"
    # The close is still the 20:00 close, so the measured move is unchanged
    # except for the one bar of drift the hole removes.
    assert (holed["fwd"] > 0).all()


def test_session_without_a_close_is_dropped_rather_than_guessed():
    b = _session_bars(["2026-06-15", "2026-06-16"])
    b = b[~((b["ts"].dt.date == date(2026, 6, 16)) & (b["ts"].dt.hour >= 18))]
    assert len(of.session_observations(b)) == 1


def test_a_real_edge_that_does_not_cover_costs_is_not_a_pass():
    """The result entry 016 actually produced. A rule can be statistically real,
    consistent month by month, and still be a losing trade."""
    rng = np.random.default_rng(7)
    n = 4000
    obs = pd.DataFrame({
        "ts": pd.date_range("2026-06-01", periods=n, freq="30min", tz="UTC"),
        "sig": rng.choice([-1.0, 1.0], n),
    })
    obs["fwd"] = obs["sig"] * 0.04 + rng.normal(0, 0.5, n)   # a genuine 0.04 ATR edge

    cheap = of.score("cheap", obs, trials=9, cost_atr=0.001)
    dear = of.score("dear", obs, trials=9, cost_atr=0.15)
    assert cheap["bonferroni"] and cheap["verdict"] == "SURVIVES"
    assert dear["bonferroni"] and dear["verdict"] == "real but does not pay"


def test_scoring_demeans_against_the_whole_sample_not_the_signalled_bars():
    """A rule that only fires in an up-trending window must not be paid for the
    trend it happened to sit in."""
    n = 2000
    obs = pd.DataFrame({
        "ts": pd.date_range("2026-06-01", periods=n, freq="30min", tz="UTC"),
        "sig": np.where(np.arange(n) % 4 == 0, 1.0, 0.0),
        "fwd": 0.3,                      # every bar drifts up by the same amount
    })
    scored = of.score("all drift", obs, trials=9, cost_atr=0.001)
    assert abs(scored["edge_atr"]) < 1e-9, "pure drift must score as zero edge"
