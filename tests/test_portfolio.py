"""Tests for the portfolio layer.

The first test is the one that makes the rest trustworthy: a portfolio of ONE
sleeve on ONE symbol must produce exactly the trades the single-symbol
backtester produces. If those ever diverge, one of them is wrong and every
portfolio result is suspect.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from backtest.engine import Backtester  # noqa: E402
from backtest.portfolio import (  # noqa: E402
    PortfolioBacktester,
    attribution,
    diversification_ratio,
    flag_correlated,
    sleeve_correlation,
)
from core.config import RiskProfile  # noqa: E402
from core.sleeve import Sleeve, sleeve_of, tag  # noqa: E402
from core.strategy import FLAT, Intent, Strategy  # noqa: E402
from core.types import Position, Side  # noqa: E402
from execution.paper import FIXTURE_SPECS, PaperAdapter, make_tick  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.limits import ProposedTrade, RiskState, SleeveBudget  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402

NOW = datetime(2026, 9, 4, 14, 0, tzinfo=timezone.utc)


def frame(n=600, seed=5, start="2022-01-03", level=1.08, step=0.004):
    rng = np.random.default_rng(seed)
    closes = np.maximum(level + np.cumsum(rng.normal(0, step, n)), level * 0.3)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    pad = np.abs(rng.normal(0, step / 2, n))
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq="B", tz="UTC"),
        "open": opens, "high": np.maximum(opens, closes) + pad,
        "low": np.minimum(opens, closes) - pad, "close": closes, "volume": 100.0,
    })


def profile():
    return RiskProfile.load("challenge")


# ======================================================================
# The equivalence test
# ======================================================================

def test_single_sleeve_matches_single_backtester():
    df = frame()
    spec = FIXTURE_SPECS["EURUSD"]
    costs = CostModel()

    single = Backtester(TrendFollowing(), spec, build_engine(profile(), 100_000.0, {"EURUSD": spec}),
                        costs, starting_equity=100_000.0).run(df)

    sleeve = Sleeve("trend", lambda sym: TrendFollowing(), ("EURUSD",))
    port = PortfolioBacktester([sleeve], {"EURUSD": spec},
                               build_engine(profile(), 100_000.0, {"EURUSD": spec}, [sleeve]),
                               costs, starting_equity=100_000.0).run({("trend", "EURUSD"): df})

    assert len(port.trades) == len(single.trades) > 5
    for a, b in zip(single.trades, port.trades):
        assert a.entry_ts == b.entry_ts and a.exit_ts == b.exit_ts
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_price == pytest.approx(b.exit_price)
        assert a.volume == pytest.approx(b.volume)
        assert a.net_pnl == pytest.approx(b.net_pnl, abs=1e-6)
        assert a.exit_reason == b.exit_reason
    assert port.final_equity == pytest.approx(single.final_equity, abs=1e-4)


# ======================================================================
# Sleeves
# ======================================================================

def test_sleeve_name_must_fit_the_order_comment():
    with pytest.raises(ValueError, match="1-12 characters"):
        Sleeve("this_name_is_far_too_long", lambda s: TrendFollowing(), ("EURUSD",))
    with pytest.raises(ValueError, match="'#'"):
        Sleeve("bad#name", lambda s: TrendFollowing(), ("EURUSD",))


def test_sleeve_builds_a_renamed_strategy():
    s = Sleeve("carry", lambda sym: TrendFollowing(), ("EURUSD",))
    strat = s.build("EURUSD")
    assert strat.name == "carry"
    assert TrendFollowing.name == "S1_trend", "class attribute must not be mutated"


def test_position_is_attributed_by_comment_prefix():
    p = Position("EURUSD", Side.BUY, 0.1, 1.08, NOW, stop_loss=1.07, comment=tag("trend"))
    assert sleeve_of(p) == "trend"
    assert sleeve_of(Position("EURUSD", Side.BUY, 0.1, 1.08, NOW)) == ""


# ======================================================================
# Two sleeves on one symbol
# ======================================================================

class AlwaysLong(Strategy):
    name = "always_long"
    warmup = 3

    def evaluate(self, df, i, position):
        return Intent(side=Side.BUY, stop_distance=0.02)


class AlwaysShort(Strategy):
    name = "always_short"
    warmup = 3

    def evaluate(self, df, i, position):
        return Intent(side=Side.SELL, stop_distance=0.02)


def test_two_sleeves_can_hold_the_same_symbol_in_opposite_directions():
    df = frame(200, step=0.0005)
    spec = FIXTURE_SPECS["EURUSD"]
    long_s = Sleeve("long", lambda s: AlwaysLong(), ("EURUSD",))
    short_s = Sleeve("short", lambda s: AlwaysShort(), ("EURUSD",))
    engine = build_engine(profile(), 100_000.0, {"EURUSD": spec}, [long_s, short_s])
    res = PortfolioBacktester([long_s, short_s], {"EURUSD": spec}, engine, CostModel(),
                              starting_equity=100_000.0).run(
        {("long", "EURUSD"): df, ("short", "EURUSD"): df})

    sleeves = {t.sleeve for t in res.trades}
    assert sleeves == {"long", "short"}
    assert any(t.side is Side.BUY for t in res.trades if t.sleeve == "long")
    assert any(t.side is Side.SELL for t in res.trades if t.sleeve == "short")


# ======================================================================
# The allocator
# ======================================================================

def _state(positions):
    return RiskState(
        equity=100_000.0, balance=100_000.0, margin_level=float("inf"),
        day_start_equity=100_000.0, high_water_equity=100_000.0, starting_equity=100_000.0,
        positions=positions, specs=FIXTURE_SPECS,
    )


def _pos(sleeve, stop_pts=0.0100, volume=0.5):
    # 0.5 lots x 100 pts x $10/pt... EURUSD: 0.5 x 0.0100 x 100000 = $500 = 0.5%
    return Position("EURUSD", Side.BUY, volume, 1.0800, NOW,
                    stop_loss=1.0800 - stop_pts, comment=tag(sleeve))


def test_sleeve_budget_rejects_a_sleeve_over_its_share():
    limit = SleeveBudget(caps={"trend": 0.01, "carry": 0.01}, portfolio_cap=0.02)
    s = _state([_pos("trend"), _pos("trend")])  # trend already holds 1.0%
    trade = ProposedTrade("EURUSD", 0.5, 500.0, 0.005, strategy="trend")
    breach = limit.check(s, trade)
    assert breach is not None and "sleeve 'trend'" in breach.message

    other = ProposedTrade("EURUSD", 0.5, 500.0, 0.005, strategy="carry")
    assert limit.check(s, other) is None, "carry's budget is untouched by trend's positions"


def test_portfolio_cap_binds_across_sleeves():
    limit = SleeveBudget(caps={"a": 0.015, "b": 0.015}, portfolio_cap=0.02)
    s = _state([_pos("a"), _pos("a"), _pos("b")])  # 1.5% open across the book
    trade = ProposedTrade("EURUSD", 0.5, 600.0, 0.006, strategy="b")  # b alone is fine
    breach = limit.check(s, trade)
    assert breach is not None and "book would hold" in breach.message


def test_sleeve_cap_cannot_exceed_portfolio_cap():
    with pytest.raises(ValueError):
        SleeveBudget(caps={"a": 0.05}, portfolio_cap=0.02)


def test_engine_with_sleeves_carries_the_allocator():
    sl = [Sleeve("a", lambda s: TrendFollowing(), ("EURUSD",)),
          Sleeve("b", lambda s: TrendFollowing(), ("EURUSD",), weight=3.0)]
    engine = build_engine(profile(), 100_000.0, dict(FIXTURE_SPECS), sl)
    alloc = [l for l in engine.limits if isinstance(l, SleeveBudget)]
    assert len(alloc) == 1
    assert alloc[0].caps["b"] == pytest.approx(alloc[0].caps["a"] * 3)
    assert sum(alloc[0].caps.values()) == pytest.approx(profile().max_open_risk)


def test_engine_without_sleeves_has_no_allocator():
    engine = build_engine(profile(), 100_000.0, dict(FIXTURE_SPECS))
    assert not any(isinstance(l, SleeveBudget) for l in engine.limits)


# ======================================================================
# Attribution and correlation
# ======================================================================

def _two_sleeve_result(seed_a=5, seed_b=5):
    spec = FIXTURE_SPECS["EURUSD"]
    a = Sleeve("a", lambda s: TrendFollowing(lookback=40), ("EURUSD",))
    b = Sleeve("b", lambda s: TrendFollowing(lookback=80), ("EURUSD",))
    engine = build_engine(profile(), 100_000.0, {"EURUSD": spec}, [a, b])
    return PortfolioBacktester([a, b], {"EURUSD": spec}, engine, CostModel(),
                               starting_equity=100_000.0).run(
        {("a", "EURUSD"): frame(seed=seed_a), ("b", "EURUSD"): frame(seed=seed_b)})


def test_attribution_sums_to_the_total():
    res = _two_sleeve_result()
    att = attribution(res)
    assert set(att.index) == {"a", "b"}
    assert att["net_pnl"].sum() == pytest.approx(sum(t.net_pnl for t in res.trades), abs=1e-6)
    assert att["share"].sum() == pytest.approx(1.0, abs=1e-9) or att["net_pnl"].sum() == 0


def test_identical_sleeves_are_flagged_as_one_strategy():
    spec = FIXTURE_SPECS["EURUSD"]
    a = Sleeve("a", lambda s: TrendFollowing(), ("EURUSD",))
    b = Sleeve("b", lambda s: TrendFollowing(), ("EURUSD",))
    engine = build_engine(profile(), 100_000.0, {"EURUSD": spec}, [a, b])
    df = frame()
    res = PortfolioBacktester([a, b], {"EURUSD": spec}, engine, CostModel(),
                              starting_equity=100_000.0).run({("a", "EURUSD"): df, ("b", "EURUSD"): df})
    corr = sleeve_correlation(res)
    flags = flag_correlated(corr, 0.6)
    assert flags and flags[0][2] > 0.95, "two copies of one strategy must correlate ~1"
    assert diversification_ratio(corr, res.weights) < 1.1, "no diversification from a clone"


def test_diversification_ratio_math():
    corr = pd.DataFrame(np.eye(4), index=list("abcd"), columns=list("abcd"))
    assert diversification_ratio(corr, {k: 0.25 for k in "abcd"}) == pytest.approx(2.0)
    # Build the correlated matrix directly: mutating a DataFrame's .values in
    # place hits pandas' read-only copy-on-write view.
    m = np.full((4, 4), 0.3); np.fill_diagonal(m, 1.0)
    corr = pd.DataFrame(m, index=list("abcd"), columns=list("abcd"))
    assert diversification_ratio(corr, {k: 0.25 for k in "abcd"}) == pytest.approx(np.sqrt(4 / 1.9))


# ======================================================================
# Runner
# ======================================================================

def test_runner_evaluates_every_sleeve_leg(tmp_path):
    from live.runner import Runner
    from live.state import StateStore
    from ops.journal import Journal
    from risk.killswitch import KillFile
    from core.types import Bar

    adapter = PaperAdapter(FIXTURE_SPECS); adapter.connect()
    adapter.feed_tick(make_tick("EURUSD", 1.08, 0.0001, datetime.now(timezone.utc)))
    bars = [Bar("EURUSD", NOW - timedelta(days=12 - i), 1.08, 1.085, 1.075, 1.08 + i * 0.001, 100.0)
            for i in range(12)]
    adapter.feed_bars("EURUSD", "D1", bars)

    sl = [Sleeve("long", lambda s: AlwaysLong(), ("EURUSD",)),
          Sleeve("short", lambda s: AlwaysShort(), ("EURUSD",))]
    engine = build_engine(profile(), 100_000.0, {"EURUSD": FIXTURE_SPECS["EURUSD"]}, sl)
    runner = Runner(adapter=adapter, risk=engine, specs={"EURUSD": FIXTURE_SPECS["EURUSD"]},
                    sleeves=sl, state=StateStore(tmp_path / "s.json"),
                    journal=Journal(tmp_path / "j.jsonl"), kill=KillFile(tmp_path / "K"), poll_seconds=0)
    runner.start()
    try:
        runner.tick(); runner.worker.stop()
        decided = {d["strategy"] for d in runner.journal.read("decision")}
        assert decided == {"long", "short"}, f"both sleeves must be evaluated, got {decided}"
        opened = {sleeve_of(p) for p in adapter.positions()}
        assert "long" in opened and "short" in opened
    finally:
        runner.shutdown()
