"""Tests for the research harness.

The first test in this file is the one that matters. Everything else here checks
that the backtester is pessimistic in the right places; `test_no_lookahead` checks
that it is answering a question about the past at all.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel, SymbolCosts  # noqa: E402
from backtest.engine import Backtester  # noqa: E402
from backtest.metrics import compute  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.strategy import FLAT, Intent, Strategy  # noqa: E402
from core.types import Position, Side  # noqa: E402
from execution.paper import FIXTURE_SPECS  # noqa: E402
from features.indicators import atr, donchian, ema, rolling_return  # noqa: E402
from risk.build import build_engine  # noqa: E402
from strategies.trend import BuyAndHold, TrendFollowing  # noqa: E402

EURUSD = FIXTURE_SPECS["EURUSD"]


def frame(closes: list[float], start="2024-01-01", spread: float = 0.002) -> pd.DataFrame:
    n = len(closes)
    closes = np.array(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq="D", tz="UTC"),
        "open": opens,
        "high": np.maximum(opens, closes) + spread,
        "low": np.minimum(opens, closes) - spread,
        "close": closes,
        "volume": np.full(n, 100.0),
    })


def random_frame(n: int = 600, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 1.08 + np.cumsum(rng.normal(0, 0.004, n))
    return frame(list(np.maximum(closes, 0.5)))


def make_backtester(strategy, costs: CostModel | None = None, equity: float = 100_000.0):
    profile = RiskProfile.load("challenge")
    engine = build_engine(profile, equity, {"EURUSD": EURUSD})
    return Backtester(strategy, EURUSD, engine, costs or CostModel(), starting_equity=equity)


# ===========================================================================
# The look-ahead test
# ===========================================================================

def test_no_lookahead_in_signals():
    """A signal at bar i must not change when future bars are removed.

    This is the empirical version of "all indicators are causal". If any of them
    used a centred window, a negative shift, or a whole-series statistic, the
    signal computed with the full frame would differ from the one computed on
    data truncated at i -- and this test would fail.

    It is the single most valuable test in the harness, because a look-ahead bug
    does not crash, does not look wrong, and produces a superb equity curve.
    """
    df = random_frame(400)
    strategy = TrendFollowing()

    full = strategy.prepare(df.copy()).reset_index(drop=True)

    checked = 0
    for i in range(strategy.warmup, len(df), 7):
        truncated = strategy.prepare(df.iloc[: i + 1].copy()).reset_index(drop=True)

        a = strategy.evaluate(full, i, None)
        b = strategy.evaluate(truncated, i, None)

        assert a.side == b.side, f"bar {i}: side differs with future data ({a.side} vs {b.side})"
        assert a.stop_distance == pytest.approx(b.stop_distance, rel=1e-9), (
            f"bar {i}: stop distance differs with future data"
        )
        checked += 1

    assert checked > 20, "the test did not actually exercise enough bars"


def test_indicators_are_causal():
    """Same property, one level down: indicator values must not change."""
    df = random_frame(300)
    for name, fn in [
        ("atr", lambda d: atr(d, 14)),
        ("ema", lambda d: ema(d["close"], 20)),
        ("mom", lambda d: rolling_return(d["close"], 60)),
        ("donchian_hi", lambda d: donchian(d, 20)[0]),
    ]:
        full = fn(df)
        for cut in (150, 220, 280):
            partial = fn(df.iloc[:cut])
            assert full.iloc[cut - 1] == pytest.approx(partial.iloc[-1], rel=1e-12, nan_ok=True), (
                f"{name} at bar {cut - 1} changed when future data was added"
            )


def test_donchian_excludes_the_current_bar():
    """Including the current bar makes a breakout trivially true when tested."""
    df = frame([1.0, 1.1, 1.2, 1.3, 5.0])
    upper, _ = donchian(df, 3)
    assert upper.iloc[4] < 5.0  # the spike must not be inside its own channel


# ===========================================================================
# Execution realism
# ===========================================================================

class AlwaysLong(Strategy):
    name = "always_long"
    warmup = 2

    def __init__(self, stop_distance: float = 0.01) -> None:
        self.stop_distance = stop_distance

    def evaluate(self, df, i, position):
        return Intent(side=Side.BUY, stop_distance=self.stop_distance)


class OneShot(Strategy):
    """Long on exactly one bar, then flat. Isolates a single trade."""

    name = "one_shot"
    warmup = 2

    def __init__(self, at: int, stop_distance: float = 0.01) -> None:
        self.at = at
        self.stop_distance = stop_distance

    def evaluate(self, df, i, position):
        if position is not None:
            return Intent(side=Side.BUY, stop_distance=self.stop_distance)
        return Intent(side=Side.BUY, stop_distance=self.stop_distance) if i == self.at else FLAT


def test_entry_happens_at_the_next_bar_open():
    """A signal from bar i's close cannot fill at bar i's price."""
    df = frame([1.00, 1.00, 1.00, 1.00, 1.50, 1.50, 1.50], spread=0.0)
    bt = make_backtester(OneShot(at=3), CostModel(costs={"EURUSD": SymbolCosts(spread=0.0)}))
    result = bt.run(df)

    assert result.trades
    trade = result.trades[0]
    # Signal on bar 3 (close 1.00) -> fill at bar 4's OPEN, which is bar 3's close.
    assert trade.entry_ts == df["ts"].iloc[4]
    assert trade.entry_price == pytest.approx(1.00, abs=1e-9)


def test_stop_is_respected_and_capped_at_one_r():
    df = frame([1.00] * 4 + [0.90, 0.90], spread=0.0)
    bt = make_backtester(
        OneShot(at=2, stop_distance=0.02),
        CostModel(costs={"EURUSD": SymbolCosts(spread=0.0)}),
    )
    result = bt.run(df)
    trade = result.trades[0]
    assert trade.exit_reason in ("stop", "gap_through_stop")
    assert trade.r_multiple <= -0.99


def test_gap_through_stop_fills_at_the_open_not_the_stop():
    """A broker fills the gap, not your wish. The loss exceeds 1R.

    Needs a real opening gap: bar 4 must OPEN below the stop, not merely trade
    through it intraday. The `frame` helper chains open to the previous close, so
    this one is built by hand -- a weekend gap has no such continuity.
    """
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC"),
        "open":  [1.00, 1.00, 1.00, 1.00, 0.50, 0.50],
        "high":  [1.00, 1.00, 1.00, 1.00, 0.52, 0.52],
        "low":   [1.00, 1.00, 1.00, 1.00, 0.48, 0.48],
        "close": [1.00, 1.00, 1.00, 1.00, 0.50, 0.50],
        "volume": [100.0] * 6,
    })
    bt = make_backtester(
        OneShot(at=2, stop_distance=0.02),
        CostModel(costs={"EURUSD": SymbolCosts(spread=0.0)}),
    )
    result = bt.run(df)
    trade = result.trades[0]
    assert trade.exit_reason == "gap_through_stop"
    assert trade.exit_price == pytest.approx(0.50), "must fill at the open, not the stop"
    assert trade.r_multiple < -1.5, "a gap must be able to lose more than 1R"


def test_costs_are_charged_on_both_fills():
    flat = frame([1.00] * 12, spread=0.0)

    free = make_backtester(AlwaysLong(), CostModel(costs={"EURUSD": SymbolCosts(spread=0.0)}))
    charged = make_backtester(
        AlwaysLong(), CostModel(costs={"EURUSD": SymbolCosts(spread=0.001, slippage=0.0005)})
    )
    a, b = free.run(flat), charged.run(flat)

    assert a.final_equity == pytest.approx(100_000.0, abs=1.0)
    assert b.final_equity < a.final_equity, "a spread must cost something on flat prices"


def test_cost_stress_is_monotonic():
    """Doubling costs cannot improve a result. The gauntlet's sensitivity gate."""
    df = random_frame(500)
    equities = []
    for stress in (1.0, 2.0, 4.0):
        bt = make_backtester(TrendFollowing(), CostModel().stressed(stress))
        equities.append(bt.run(df).final_equity)
    assert equities[0] >= equities[1] >= equities[2]


def test_backtest_uses_the_real_risk_engine():
    """Sizing in the backtest must obey the same YAML limits as live."""
    profile = RiskProfile.load("challenge")
    df = random_frame(400)
    result = make_backtester(TrendFollowing()).run(df)

    for trade in result.trades:
        risk_fraction = trade.risk_cash / 100_000.0
        assert risk_fraction <= profile.max_risk_per_trade + 1e-9, (
            f"trade risked {risk_fraction:.4%}, above the profile ceiling"
        )


def test_risk_engine_rejections_are_counted_not_swallowed():
    """A tiny account cannot size anything; that must be visible, not silent."""
    tiny = make_backtester(TrendFollowing(), equity=200.0)
    result = tiny.run(random_frame(400))
    assert not result.trades
    assert result.rejections, "rejections must be reported so 'no trades' is explainable"


def test_every_trade_carries_a_stop():
    result = make_backtester(TrendFollowing()).run(random_frame(400))
    assert result.trades
    assert all(t.stop_price > 0 for t in result.trades)
    assert all(t.risk_cash > 0 for t in result.trades)


# ===========================================================================
# Metrics
# ===========================================================================

def test_r_multiple_and_excursions_are_populated():
    result = make_backtester(TrendFollowing()).run(random_frame(600))
    assert len(result.trades) > 10
    assert any(t.mfe > 0 for t in result.trades), "MFE never recorded"
    assert any(t.mae < 0 for t in result.trades), "MAE never recorded"
    assert all(t.mae <= 0 <= t.mfe for t in result.trades)


def test_cost_drag_reflects_spread_not_just_commission():
    """Regression: charging only commission reported 0% drag on a spread broker."""
    df = random_frame(500)
    bt = make_backtester(
        TrendFollowing(),
        CostModel(costs={"EURUSD": SymbolCosts(spread=0.0005, slippage=0.0002)}),
    )
    result = bt.run(df)
    assert result.trades
    assert all(t.costs > 0 for t in result.trades), "spread cost vanished from accounting"
    assert compute(result).total_costs > 0


def test_metrics_on_a_known_series():
    curve = pd.Series(
        [100.0, 110.0, 105.0, 120.0, 90.0],
        index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
    )
    from backtest.engine import BacktestResult
    from backtest.metrics import _drawdown

    max_dd, _ = _drawdown(curve)
    assert max_dd == pytest.approx(0.25)  # 120 -> 90

    empty = BacktestResult(starting_equity=100.0)
    assert compute(empty).trades == 0


def test_trend_loses_roughly_costs_on_a_random_walk():
    """Sanity anchor: no trend to follow means no edge, and friction is the result.

    If this ever showed a strong positive result, the harness would be lying --
    a random walk has nothing for a trend system to find.
    """
    results = []
    for seed in range(6):
        bt = make_backtester(TrendFollowing())
        results.append(compute(bt.run(random_frame(700, seed=seed))).expectancy_r)
    assert np.mean(results) < 0.15, f"suspicious edge on random data: {np.mean(results):+.3f}R"


def test_buy_and_hold_control_actually_trades():
    """Regression: at 100x ATR the control could never be sized and never traded."""
    result = make_backtester(BuyAndHold()).run(random_frame(400))
    assert result.trades, "the control benchmark must produce at least one trade"


def test_edge_ratio_matches_the_research_thresholds():
    costs = CostModel(costs={"EURUSD": SymbolCosts(spread=0.0002, slippage=0.0001)})
    # round trip = 2 * (0.0001 + 0.0001) = 0.0004
    assert costs.edge_ratio("EURUSD", EURUSD, 0.0008) == pytest.approx(2.0)  # dead
    assert costs.edge_ratio("EURUSD", EURUSD, 0.0016) == pytest.approx(4.0)  # fragile
    assert costs.edge_ratio("EURUSD", EURUSD, 0.0040) == pytest.approx(10.0)  # workable


def test_unknown_symbol_never_trades_for_free():
    with pytest.raises(KeyError, match="never let a symbol trade for free"):
        CostModel().for_symbol("NOTASYMBOL")


def test_calibration_uses_the_median_not_the_mean():
    costs = CostModel()
    fills = [{"spread": 0.0001, "slippage": 0.00002}] * 5 + [
        {"spread": 0.0050, "slippage": 0.00300}  # one news fill
    ]
    fitted = costs.calibrate("EURUSD", fills)
    assert fitted.spread == pytest.approx(0.0001)
    assert costs.calibrated is True
