"""011: book-level volatility targeting, sized once a month, held in between."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from backtest.portfolio import PortfolioBacktester  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.contracts import FULL_UNIVERSE  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from core.strategy import is_month_start  # noqa: E402
from core.types import Side  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.voltarget import LegIntent, VolTarget  # noqa: E402
from strategies.tsmom import TSMOM  # noqa: E402

EQUITY = 1_000_000.0


def _feed(vt: VolTarget, series: dict[str, np.ndarray], start=date(2025, 1, 1)):
    n = max(len(v) for v in series.values())
    for k in range(n):
        vt.observe(start + timedelta(days=k), {s: float(v[k]) for s, v in series.items() if k < len(v)})


def _leg(key, symbol, side=Side.BUY, stop=10.0, value=100.0, held=None):
    return LegIntent(key, symbol, side, stop, value, held)


def test_two_independent_markets_share_the_target_equally():
    rng = np.random.default_rng(1)
    vt = VolTarget(target_annual_vol=0.12, max_risk_fraction=1.0)
    _feed(vt, {"A": rng.normal(0, 100, 130), "B": rng.normal(0, 100, 130)})
    out = vt.allocate(EQUITY, [_leg("a", "A"), _leg("b", "B")])
    target = vt.target_daily_cash_vol(EQUITY)
    a, b = out["a"], out["b"]
    # each leg carries c of daily cash vol, with c^2 * (2 + 2 rho') = target^2 and rho' ~ 0
    c_a, c_b = a.contracts * a.cash_vol_per_contract, b.contracts * b.cash_vol_per_contract
    assert c_a == pytest.approx(c_b, rel=1e-9)
    assert c_a == pytest.approx(target / np.sqrt(2), rel=0.15)
    assert a.risk_fraction == pytest.approx(a.contracts * 10.0 * 100.0 / EQUITY)


def test_correlated_markets_get_less_and_a_hedged_pair_gets_more():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 100, 130)
    vt = VolTarget(target_annual_vol=0.12, max_risk_fraction=1.0)
    _feed(vt, {"A": x, "B": x})  # identical: correlation 1, shrunk to 0.5
    target = vt.target_daily_cash_vol(EQUITY)
    same = vt.allocate(EQUITY, [_leg("a", "A"), _leg("b", "B")])
    c_same = same["a"].contracts * same["a"].cash_vol_per_contract
    assert c_same == pytest.approx(target / np.sqrt(3.0), rel=1e-6)  # 2 + 2*0.5
    hedged = vt.allocate(EQUITY, [_leg("a", "A"), _leg("b", "B", side=Side.SELL)])
    c_hedged = hedged["a"].contracts * hedged["a"].cash_vol_per_contract
    assert c_hedged == pytest.approx(target / np.sqrt(1.0), rel=1e-6)  # 2 - 2*0.5


def test_per_position_cap_and_missing_history():
    rng = np.random.default_rng(3)
    vt = VolTarget(target_annual_vol=0.12, max_risk_fraction=0.01)
    _feed(vt, {"A": rng.normal(0, 1.0, 130), "NEW": rng.normal(0, 1.0, 10)})  # A: tiny vol per contract
    out = vt.allocate(EQUITY, [_leg("a", "A", stop=10.0, value=100.0), _leg("n", "NEW")])
    assert out["a"].risk_fraction == pytest.approx(0.01) and "capped" in out["a"].note
    assert out["a"].contracts == pytest.approx(0.01 * EQUITY / (10.0 * 100.0))
    assert out["n"].risk_fraction is None and "history" in out["n"].note
    assert not vt.ready("NEW") and vt.ready("A")


def test_a_book_already_at_target_adds_nothing():
    rng = np.random.default_rng(4)
    vt = VolTarget(target_annual_vol=0.12, max_risk_fraction=1.0)
    _feed(vt, {"A": rng.normal(0, 100, 130), "B": rng.normal(0, 100, 130)})
    target = vt.target_daily_cash_vol(EQUITY)
    huge = target * 5 / 100.0  # contracts of A worth five targets of vol, held
    out = vt.allocate(EQUITY, [_leg("a", "A", held=huge), _leg("b", "B")])
    assert "a" not in out and out["b"].contracts == 0.0 and "at target" in out["b"].note


def test_window_is_bounded_and_target_scales_with_equity():
    vt = VolTarget(window=50, min_history=10)
    _feed(vt, {"A": np.ones(200)})
    assert vt.history_length("A") == 50
    assert vt.target_daily_cash_vol(2e6) == pytest.approx(2 * vt.target_daily_cash_vol(1e6))
    with pytest.raises(ValueError):
        VolTarget(target_annual_vol=1.5)


# ------------------------------------------------------- the backtester

def _walk(n, sigma, seed, start="2019-01-02"):
    """A random walk; `sigma` may be a scalar or a per-bar array, so a
    volatility regime change can be built in."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="B", tz="UTC")
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
    px = 1000.0 + np.cumsum(rng.normal(0, 1.0, n) * sig)
    return pd.DataFrame({"ts": ts, "open": px, "high": px + sig, "low": px - sig, "close": px, "volume": 1.0})


def test_book_is_resized_only_on_decision_days_and_lands_near_target():
    names = ("ES", "ZN")  # two real specs, independent synthetic walks
    specs = {n: FULL_UNIVERSE[n].to_spec(n) for n in names}
    # ES triples its volatility half way through: a book at target must cut
    # the position on a later decision day, and only on a decision day.
    es_sigma = np.r_[np.full(450, 20.0), np.full(450, 60.0)]
    bars = {"ES": _walk(900, es_sigma, 7), "ZN": _walk(900, 0.4, 8)}
    sleeve = Sleeve("tsmom120", lambda s: TSMOM(lookback=120, monthly_resize=True), names, timeframe="D1")
    # With two markets a 1% per-position cap holds the book to a third of the
    # target (the cap is sized for thirty); lift it, in the allocator AND in
    # the engine that judges what the allocator asks for.
    from dataclasses import replace
    profile = replace(RiskProfile.load("research"), max_risk_per_trade=0.05)
    engine = build_engine(profile, 10_000_000.0, specs, [sleeve])
    vt = VolTarget(target_annual_vol=0.12, max_risk_fraction=0.05)
    pb = PortfolioBacktester([sleeve], specs, engine, CostModel.for_futures({n: FULL_UNIVERSE[n] for n in names}),
                             starting_equity=10_000_000.0, allocator=vt)
    res = pb.run({("tsmom120", n): bars[n] for n in names})

    rebalances = [t for t in res.trades if t.exit_reason == "rebalance"]
    assert res.rebalances > 0 and rebalances, "monthly resizes must happen"
    for t in rebalances:  # decided on the month's first bar, filled at the next bar's open
        df = bars[t.symbol]
        i = int(df.index[df["ts"] == pd.Timestamp(t.exit_ts)][0])
        assert is_month_start(df, i - 1), t
    realised = vt.realised_book_vol(res.equity)
    assert 0.04 < realised < 0.30, f"realised book vol {realised:.1%} is nowhere near the 12% target"
