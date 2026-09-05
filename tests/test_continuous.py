"""008: continuous forecasts, volatility targeting, position inertia.

The invariant under test is the architecture's: a strategy re-proposes a view,
only the risk engine turns it into lots, an increase is new risk that must pass
every limit, a reduction never needs permission, and a stop only tightens.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from backtest.engine import Backtester  # noqa: E402
from backtest.portfolio import PortfolioBacktester  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from core.strategy import FLAT, Intent, Strategy, forecast_to_confidence  # noqa: E402
from core.types import Bar, Position, Side, Signal  # noqa: E402
from execution.paper import FIXTURE_SPECS, PaperAdapter, make_tick  # noqa: E402
from live.runner import Runner  # noqa: E402
from live.state import StateStore  # noqa: E402
from ops.journal import Journal  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.killswitch import KillFile  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402

NOW = datetime(2026, 9, 4, 14, 0, tzinfo=timezone.utc)
SPEC = FIXTURE_SPECS["EURUSD"]


# ------------------------------------------------------------ forecast mapping

def test_forecast_maps_to_capped_floored_confidence():
    assert forecast_to_confidence(0.0) == 0.0
    assert forecast_to_confidence(float("nan")) == 0.0
    assert forecast_to_confidence(0.1) == 0.25  # the floor: no dust positions
    assert forecast_to_confidence(1.0) == 0.5
    assert forecast_to_confidence(-1.0) == 0.5  # sign is the side's business
    assert forecast_to_confidence(5.0) == 1.0  # the cap


def _trending(n=400, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    px = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, n)))
    return pd.DataFrame({"ts": ts, "open": px, "high": px * 1.005, "low": px * 0.995,
                         "close": px, "volume": 1.0})


def test_discrete_trend_is_unchanged_and_continuous_scales_size_not_direction():
    base, cont = TrendFollowing(), TrendFollowing(continuous=True)
    assert base.rebalances is False and cont.rebalances is True
    dfb, dfc = base.prepare(_trending()), cont.prepare(_trending())
    seen = set()
    for i in range(base.warmup, len(dfb)):
        ib, ic = base.evaluate(dfb, i, None), cont.evaluate(dfc, i, None)
        assert ib.flat == ic.flat
        if ib.flat:
            continue
        assert ib.confidence == 1.0
        assert ib.side is ic.side and ib.stop_distance == ic.stop_distance
        assert 0.25 <= ic.confidence <= 1.0
        seen.add(round(ic.confidence, 2))
    assert len(seen) > 3, "continuous mode must actually vary the size"


# --------------------------------------------------------------- the engine

def _engine(profile="research", equity=100_000.0, **over):
    p = RiskProfile.load(profile)
    if over:
        p = replace(p, **over)
    e = build_engine(p, equity, {"EURUSD": SPEC})
    e.book.trading_day = NOW.date()
    return e


def _state(e, equity, positions, price):
    return e.snapshot(equity=equity, balance=equity, margin_level=float("inf"),
                      positions=positions, now=NOW, current_price={"EURUSD": price})


def _held(volume, entry=1.0800, stop=1.0700):
    return Position("EURUSD", Side.BUY, volume, entry, NOW, stop_loss=stop, ticket=1, comment="trend#1")


def _sig(confidence=1.0, stop_distance=0.0100):
    return Signal("EURUSD", Side.BUY, stop_distance, confidence, strategy="trend", ts=NOW)


def test_resize_holds_inside_inertia_and_ratchets_the_stop_only_tighter():
    e = _engine()
    pos = _held(0.5)  # exactly the 0.5% target: $500 / (0.01 * 100,000)

    d = e.resize(_sig(0.9), _state(e, 100_000, [pos], 1.0800), pos)
    assert d.action == "hold" and d.delta == 0.0 and d.stop_loss == 1.0700  # -10% is inside 25%

    # price ran up: a fresh 1-cent stop sits higher, so the stop ratchets up
    d = e.resize(_sig(1.0), _state(e, 100_000, [pos], 1.0900), pos)
    assert d.action == "hold" and d.stop_loss == pytest.approx(1.0800)

    # a WIDER proposed stop must never loosen the one we have
    d = e.resize(_sig(1.0, stop_distance=0.0300), _state(e, 100_000, [pos], 1.0800), pos)
    assert d.stop_loss == 1.0700 and d.action == "hold"


def test_resize_reduces_without_asking_anyone():
    e = _engine()
    pos = _held(0.5)
    d = e.resize(_sig(0.4), _state(e, 100_000, [pos], 1.0800), pos)
    assert d.action == "reduce" and d.delta == pytest.approx(-0.30) and d.order is None
    assert d.target_volume == pytest.approx(0.20)


def test_resize_increase_is_new_risk_but_not_a_new_position():
    # one position allowed, and we already hold it: an add must still pass
    e = _engine(max_concurrent_positions=1)
    pos = _held(0.25)
    d = e.resize(_sig(1.0), _state(e, 100_000, [pos], 1.0800), pos)
    assert d.action == "increase", d.note
    assert d.delta == pytest.approx(0.25) and d.order.volume == pytest.approx(0.25)
    assert d.order.stop_loss == 1.0700 and d.order.intent.endswith(":add")


def test_resize_increase_is_refused_by_a_limit_like_any_order():
    # challenge profile: 3.4% already lost today, soft limit 3.5% - the add's
    # risk would cross it, so it is refused; what is held stays held
    e = _engine("challenge")
    e.book.day_start_equity = 100_000.0
    pos = _held(0.25)
    d = e.resize(_sig(1.0), _state(e, 96_600, [pos], 1.0800), pos)
    assert d.action == "refused" and d.delta == 0.0
    assert any(b.limit == "daily_loss" for b in d.breaches)


def test_resize_holds_when_the_target_cannot_be_sized():
    e = _engine(equity=1_000.0)
    pos = _held(0.01)
    d = e.resize(_sig(0.25), _state(e, 1_000, [pos], 1.0800), pos)
    assert d.action == "hold" and d.note.startswith("target unsizeable")


# ------------------------------------------------------ portfolio backtester

class Scripted(Strategy):
    """One confidence per bar index; 0 means flat. Always long, 1-cent stop."""
    name = "scripted"
    rebalances = True
    warmup = 0

    def __init__(self, confidences):
        self.confidences = list(confidences)

    def evaluate(self, df, i, position):
        c = self.confidences[i] if i < len(self.confidences) else 1.0
        return FLAT if c == 0 else Intent(Side.BUY, stop_distance=0.0100, confidence=c)


def _flat_frame(n=8, level=1.08):
    ts = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": level, "high": level, "low": level, "close": level, "volume": 1.0})


def _portfolio(confidences):
    sleeve = Sleeve("scripted", lambda s: Scripted(confidences), ("EURUSD",), timeframe="D1")
    engine = build_engine(RiskProfile.load("research"), 100_000.0, {"EURUSD": SPEC}, [sleeve])
    pb = PortfolioBacktester([sleeve], {"EURUSD": SPEC}, engine, CostModel(), starting_equity=100_000.0)
    return pb.run({("scripted", "EURUSD"): _flat_frame()})


def test_portfolio_reduces_then_adds_and_books_each_slice():
    res = _portfolio([1.0, 1.0, 0.4, 0.4, 1.0, 1.0, 1.0, 0])
    assert res.rebalances == 2
    kinds = [(t.exit_reason, round(t.volume, 2)) for t in res.trades]
    assert kinds == [("rebalance", 0.3), ("end_of_data", 0.5)]
    # every fill paid its way: the flat market can only lose the spread
    assert all(t.costs > 0 for t in res.trades)
    assert res.final_equity < 100_000.0


def test_portfolio_position_inertia_ignores_small_changes():
    res = _portfolio([1.0, 1.0, 0.85, 0.85, 1.0, 1.0, 1.0, 0])  # 0.5 -> 0.42 is 16%
    assert res.rebalances == 0
    assert [t.exit_reason for t in res.trades] == ["end_of_data"]


def test_single_symbol_backtester_refuses_continuous_strategies():
    engine = build_engine(RiskProfile.load("research"), 100_000.0, {"EURUSD": SPEC})
    bt = Backtester(Scripted([1.0, 1.0, 0.4, 0.4]), SPEC, engine, CostModel(), starting_equity=100_000.0)
    with pytest.raises(NotImplementedError, match="PortfolioBacktester"):
        bt.run(_flat_frame())


# ------------------------------------------------------------- live runner

class LiveScripted(Strategy):
    """One confidence per new closed bar, in order."""
    name = "scripted"
    rebalances = True
    warmup = 3

    def __init__(self, plan):
        self.plan = list(plan)

    def evaluate(self, df, i, position):
        c = self.plan.pop(0) if self.plan else 1.0
        return FLAT if c == 0 else Intent(Side.BUY, stop_distance=0.0100, confidence=c)


def _bars(n):
    """n daily bars ending later as n grows, so each feed adds a NEW closed bar."""
    base = NOW - timedelta(days=12)
    return [Bar("EURUSD", base + timedelta(days=i), 1.08, 1.085, 1.075, 1.08, 100.0) for i in range(n)]


def _tick_fresh(adapter, price):
    adapter.feed_tick(make_tick("EURUSD", price, 0.00010, datetime.now(timezone.utc)))


def _step(runner, adapter, n_bars, price=1.0800):
    """Feed a new closed bar and a fresh tick, run one iteration, wait for fills."""
    adapter.feed_bars("EURUSD", "D1", _bars(n_bars))
    _tick_fresh(adapter, price)
    runner.tick()
    runner.worker.stop()
    runner.worker.start()


def test_live_runner_resizes_through_the_worker(tmp_path: Path):
    adapter = PaperAdapter(FIXTURE_SPECS)
    adapter.connect()
    _tick_fresh(adapter, 1.0800)
    engine = build_engine(RiskProfile.load("research"), 100_000.0, {"EURUSD": SPEC})
    runner = Runner(
        adapter=adapter, risk=engine, strategies={"EURUSD": LiveScripted([1.0, 0.4, 1.0, 1.0])},
        specs={"EURUSD": SPEC}, state=StateStore(tmp_path / "s.json"),
        journal=Journal(tmp_path / "j.jsonl"), kill=KillFile(tmp_path / "KILL"), poll_seconds=0.0,
    )
    list(runner.start())
    try:
        _step(runner, adapter, 12)  # entry at full size
        assert [p.volume for p in adapter.positions()] == [pytest.approx(0.5)]

        # Sizing rounds DOWN, always: with a few dollars of costs off equity the
        # 0.4 target is 0.1999 lots and the sizer says 0.19. One step of slack.
        _step(runner, adapter, 13)  # confidence 0.4 -> partial close of ~0.3
        assert sum(p.volume for p in adapter.positions()) == pytest.approx(0.2, abs=0.011)
        assert runner.journal.read("resize")[-1]["action"] == "reduce"

        _step(runner, adapter, 14)  # back to 1.0 -> an add, as its own ticket
        assert sum(p.volume for p in adapter.positions()) == pytest.approx(0.5, abs=0.011)
        assert runner.journal.read("resize")[-1]["action"] == "increase"
        assert len(adapter.positions()) == 2

        _step(runner, adapter, 15, price=1.0900)  # price up 1c: hold, stops ratchet to 1.08
        assert runner.journal.read("resize")[-1]["action"] == "hold"
        # the tick's mid is 1.09005 (1-pip spread), so the ratcheted stop is 1.08005
        assert all(p.stop_loss == pytest.approx(1.0800, abs=1e-4) for p in adapter.positions())
        assert runner.journal.read("stop_ratchet")
    finally:
        runner.worker.stop()
