"""Tests for the risk layer.

Every limit gets a test that forces it to breach. A limit nobody has watched fire
is a limit you are trusting on faith, and the whole point of this layer is that it
is the one part of the system you do not have to trust on faith.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import OrderStatus, Position, Side, Signal  # noqa: E402
from execution.paper import FIXTURE_SPECS, PaperAdapter, make_tick  # noqa: E402
from risk.engine import RiskEngine, SessionBook  # noqa: E402
from risk.limits import (  # noqa: E402
    ConsecutiveLosses,
    CorrelatedBucket,
    DailyLoss,
    KillSwitch,
    MaxDrawdown,
    ProposedTrade,
    RiskPerTrade,
    RiskState,
    Severity,
    SpreadGuard,
    UnstoppedPosition,
)
from risk.sizing import SizingOutcome, minimum_viable_equity, size_position  # noqa: E402

NOW = datetime(2026, 9, 4, 13, 30, tzinfo=timezone.utc)
EURUSD = FIXTURE_SPECS["EURUSD"]
XAUUSD = FIXTURE_SPECS["XAUUSD"]
US30 = FIXTURE_SPECS["US30"]


def state(**kw) -> RiskState:
    base = dict(
        equity=100_000.0,
        balance=100_000.0,
        margin_level=float("inf"),
        day_start_equity=100_000.0,
        high_water_equity=100_000.0,
        starting_equity=100_000.0,
        specs=FIXTURE_SPECS,
    )
    base.update(kw)
    return RiskState(**base)


# ------------------------------------------------------------------ sizing

def test_sizing_never_exceeds_the_budget():
    r = size_position(EURUSD, equity=100_000, risk_fraction=0.005, stop_distance=0.0070)
    assert r.outcome is SizingOutcome.OK
    assert r.risk_cash <= 100_000 * 0.005 + 1e-6
    assert r.risk_fraction <= 0.005 + 1e-9


def test_sizing_rounds_down_not_nearest():
    # Budget lands between two steps; rounding to nearest would exceed the limit.
    r = size_position(EURUSD, equity=10_000, risk_fraction=0.005, stop_distance=0.0035)
    assert r.volume == EURUSD.round_volume(r.volume)
    assert r.risk_cash <= 50.0 + 1e-6


def test_gold_is_blocked_on_a_small_account():
    """The finding that decides v1 scope. Minimum lot risks 2.5% on a 5k account."""
    stop = 50.0 * 2.5  # 2.5 x daily ATR
    r = size_position(XAUUSD, equity=5_000, risk_fraction=0.005, stop_distance=stop)
    assert r.outcome is SizingOutcome.BELOW_MINIMUM
    assert r.volume == 0.0
    assert "too small" in r.reason

    # And it is not blocked once the account is large enough.
    big = size_position(XAUUSD, equity=100_000, risk_fraction=0.005, stop_distance=stop)
    assert big.tradeable


def test_eurusd_is_fine_on_a_small_account():
    r = size_position(EURUSD, equity=5_000, risk_fraction=0.005, stop_distance=0.0070 * 2.5)
    assert r.tradeable
    assert r.volume >= EURUSD.volume_min


def test_minimum_viable_equity_matches_sizing():
    stop = 50.0 * 2.5
    needed = minimum_viable_equity(XAUUSD, 0.005, stop)
    assert size_position(XAUUSD, needed * 1.01, 0.005, stop).tradeable
    assert not size_position(XAUUSD, needed * 0.99, 0.005, stop).tradeable


def test_index_min_lot_drives_viability():
    stop = 450.0 * 2.5
    assert not size_position(US30, 5_000, 0.005, stop).tradeable
    assert size_position(US30, 100_000, 0.005, stop).tradeable


# ------------------------------------------------------------------ limits

def test_kill_switch_flattens():
    breach = KillSwitch().check(state(killed=True), None)
    assert breach is not None and breach.severity is Severity.FLATTEN


def test_daily_loss_soft_then_hard():
    limit = DailyLoss(soft=0.035, hard=0.05)
    assert limit.check(state(equity=98_000), None) is None  # -2%, inside

    soft = limit.check(state(equity=96_400), None)  # -3.6%
    assert soft is not None and soft.severity is Severity.HALT

    hard = limit.check(state(equity=94_900), None)  # -5.1%
    assert hard is not None and hard.severity is Severity.FLATTEN


def test_daily_loss_rejects_a_trade_that_would_cross_the_soft_limit():
    limit = DailyLoss(soft=0.035, hard=0.05)
    s = state(equity=96_800)  # already -3.2%
    trade = ProposedTrade("EURUSD", 0.5, risk_cash=800.0, risk_fraction=0.008)
    breach = limit.check(s, trade)
    assert breach is not None and breach.severity is Severity.REJECT
    assert "would put daily loss" in breach.message


def test_static_vs_trailing_drawdown_differ():
    s = state(equity=95_000, high_water_equity=104_000, starting_equity=100_000)
    static = MaxDrawdown(0.07, 0.10, trailing=False).check(s, None)
    trailing = MaxDrawdown(0.07, 0.10, trailing=True).check(s, None)
    assert static is None  # 5% below start
    assert trailing is not None  # 8.65% below high-water


def test_risk_per_trade_ceiling():
    trade = ProposedTrade("EURUSD", 1.0, risk_cash=900.0, risk_fraction=0.009)
    assert RiskPerTrade(0.0075).check(state(), trade) is not None
    assert RiskPerTrade(0.010).check(state(), trade) is None


def test_correlated_bucket_counts_open_positions():
    limit = CorrelatedBucket({"us_indices": ["US30", "US500"]}, maximum=0.01)
    open_us30 = Position(
        symbol="US30", side=Side.BUY, volume=1.0, entry_price=44_000.0,
        opened_at=NOW, stop_loss=44_000.0 - 700.0,
    )
    # 1.0 lot x 700 points x $1/point = $700 = 0.7% of equity.
    s = state(positions=[open_us30])

    small = ProposedTrade("US500", 0.1, risk_cash=200.0, risk_fraction=0.002)
    assert limit.check(s, small) is None  # 0.9% total

    big = ProposedTrade("US500", 0.4, risk_cash=500.0, risk_fraction=0.005)
    breach = limit.check(s, big)  # 1.2% total
    assert breach is not None and "us_indices" in breach.message


def test_correlated_bucket_ignores_unrelated_symbols():
    limit = CorrelatedBucket({"us_indices": ["US30", "US500"]}, maximum=0.01)
    s = state(positions=[
        Position("EURUSD", Side.BUY, 1.0, 1.0800, NOW, stop_loss=1.0700),
    ])
    trade = ProposedTrade("US500", 0.4, risk_cash=900.0, risk_fraction=0.009)
    assert limit.check(s, trade) is None


def test_unstopped_position_halts():
    s = state(positions=[Position("EURUSD", Side.BUY, 0.5, 1.0800, NOW, stop_loss=None)])
    breach = UnstoppedPosition().check(s, None)
    assert breach is not None and breach.severity is Severity.HALT


def test_spread_guard():
    limit = SpreadGuard(max_multiple=2.0)
    trade = ProposedTrade("EURUSD", 0.1, 50.0, 0.0005)
    wide = state(current_spread={"EURUSD": 0.00030}, median_spread={"EURUSD": 0.00010})
    tight = state(current_spread={"EURUSD": 0.00015}, median_spread={"EURUSD": 0.00010})
    assert limit.check(wide, trade) is not None
    assert limit.check(tight, trade) is None


def test_consecutive_losses_pauses_only_that_strategy():
    limit = ConsecutiveLosses(maximum=4)
    s = state(consecutive_losses={"trend": 4, "orb": 1})
    assert limit.check(s, ProposedTrade("EURUSD", 0.1, 50.0, 0.0005, strategy="trend")) is not None
    assert limit.check(s, ProposedTrade("EURUSD", 0.1, 50.0, 0.0005, strategy="orb")) is None


# ------------------------------------------------------------------ engine

@pytest.fixture()
def engine() -> RiskEngine:
    from core.config import RiskProfile
    from risk.build import build_engine

    profile = RiskProfile.load("challenge")
    return build_engine(profile, starting_equity=100_000.0, specs=dict(FIXTURE_SPECS))


def test_engine_approves_a_clean_signal(engine: RiskEngine):
    s = engine.snapshot(
        equity=100_000, balance=100_000, margin_level=float("inf"),
        positions=[], current_price={"EURUSD": 1.0800},
        current_spread={"EURUSD": 0.00010}, median_spread={"EURUSD": 0.00010},
    )
    signal = Signal("EURUSD", Side.BUY, stop_distance=0.0175, strategy="trend")
    decision = engine.evaluate(signal, s)

    assert decision.approved, decision.explain()
    assert decision.order is not None
    assert decision.order.stop_loss is not None
    assert decision.order.stop_loss < 1.0800  # long stop sits below entry
    assert decision.size.risk_fraction <= 0.005 + 1e-9


def test_engine_refuses_without_a_price(engine: RiskEngine):
    s = engine.snapshot(equity=100_000, balance=100_000, margin_level=float("inf"), positions=[])
    decision = engine.evaluate(Signal("EURUSD", Side.BUY, 0.0175, strategy="trend"), s)
    assert not decision.approved
    assert "cannot place a stop" in decision.note


def test_engine_blocks_everything_once_halted(engine: RiskEngine):
    engine.book.kill("manual stop")
    s = engine.snapshot(
        equity=100_000, balance=100_000, margin_level=float("inf"),
        positions=[], current_price={"EURUSD": 1.0800},
    )
    decision = engine.evaluate(Signal("EURUSD", Side.BUY, 0.0175, strategy="trend"), s)
    assert not decision.approved
    assert decision.must_flatten


def test_engine_blocks_gold_on_a_small_account():
    """A 5k account cannot express 0.5% risk in gold's minimum lot.

    Note the engine must be *opened* at 5k. Handing a 100k engine an equity of 5k
    trips the drawdown limit first, which is correct behaviour and a different
    test — see test_account_limits_short_circuit_before_sizing below.
    """
    from core.config import RiskProfile
    from risk.build import build_engine

    small = build_engine(RiskProfile.load("challenge"), 5_000.0, dict(FIXTURE_SPECS))
    s = small.snapshot(
        equity=5_000, balance=5_000, margin_level=float("inf"),
        positions=[], current_price={"XAUUSD": 3_300.0},
    )
    decision = small.evaluate(Signal("XAUUSD", Side.BUY, 125.0, strategy="trend"), s)
    assert not decision.approved
    assert decision.size is not None
    assert decision.size.outcome is SizingOutcome.BELOW_MINIMUM

    # Same account, same profile, EURUSD: fine.
    s2 = small.snapshot(
        equity=5_000, balance=5_000, margin_level=float("inf"),
        positions=[], current_price={"EURUSD": 1.0800},
    )
    ok = small.evaluate(Signal("EURUSD", Side.BUY, 0.0175, strategy="trend"), s2)
    assert ok.approved, ok.explain()


def test_account_limits_short_circuit_before_sizing(engine: RiskEngine):
    """A halted account never reaches the sizing step. Cheapest check runs first."""
    s = engine.snapshot(
        equity=5_000, balance=5_000, margin_level=float("inf"),
        positions=[], current_price={"EURUSD": 1.0800},
    )
    decision = engine.evaluate(Signal("EURUSD", Side.BUY, 0.0175, strategy="trend"), s)
    assert not decision.approved
    assert decision.size is None
    assert decision.must_flatten
    assert any(b.limit == "max_drawdown" for b in decision.breaches)


def test_session_book_resets_daily_equity_but_not_high_water():
    book = SessionBook.open(100_000.0, day=NOW.date())
    book.observe_equity(103_000.0, today=NOW.date())
    assert book.high_water_equity == 103_000.0

    book.observe_equity(101_000.0, today=(NOW + timedelta(days=1)).date())
    assert book.day_start_equity == 101_000.0
    assert book.high_water_equity == 103_000.0
    assert book.starting_equity == 100_000.0


def test_consecutive_loss_tracking():
    book = SessionBook.open(100_000.0)
    for _ in range(3):
        book.record_close("trend", -100.0)
    assert book.consecutive_losses["trend"] == 3
    book.record_close("trend", 250.0)
    assert book.consecutive_losses["trend"] == 0


# ------------------------------------------------------------ paper adapter

def test_paper_adapter_round_trip():
    adapter = PaperAdapter(FIXTURE_SPECS)
    adapter.connect()
    adapter.feed_tick(make_tick("EURUSD", 1.08000, 0.00010, NOW))

    from core.types import OrderRequest

    result = adapter.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.07000))
    assert result.ok, result.reason
    assert result.fill_price > 1.08010  # crossed the spread plus slippage
    assert len(adapter.positions()) == 1

    adapter.feed_tick(make_tick("EURUSD", 1.08500, 0.00010, NOW))
    closed = adapter.close(result.ticket)
    assert closed.ok
    assert adapter.positions() == []
    assert adapter.realized_pnl > 0


def test_paper_adapter_rejects_sub_minimum_volume():
    adapter = PaperAdapter(FIXTURE_SPECS)
    adapter.connect()
    adapter.feed_tick(make_tick("XAUUSD", 3300.00, 0.30, NOW))

    from core.types import OrderRequest

    result = adapter.submit(OrderRequest("XAUUSD", Side.BUY, 0.004, stop_loss=3200.0))
    assert result.status is OrderStatus.REJECTED
    assert "below minimum" in result.reason


def test_paper_slippage_always_hurts():
    adapter = PaperAdapter(FIXTURE_SPECS)
    adapter.connect()
    adapter.feed_tick(make_tick("EURUSD", 1.08000, 0.00020, NOW))

    from core.types import OrderRequest

    buy = adapter.submit(OrderRequest("EURUSD", Side.BUY, 0.10, stop_loss=1.07000))
    assert buy.slippage() > 0  # paid more than the ask

    adapter.feed_tick(make_tick("EURUSD", 1.08000, 0.00020, NOW))
    sell = adapter.submit(OrderRequest("EURUSD", Side.SELL, 0.10, stop_loss=1.09000))
    assert sell.slippage() > 0  # received less than the bid


# ---------------------------------------------------------------- config

def test_profile_rejects_a_soft_limit_above_hard(tmp_path: Path):
    from core.config import ConfigError, RiskProfile

    with pytest.raises(ConfigError):
        RiskProfile(
            name="broken", risk_per_trade=0.005, max_risk_per_trade=0.0075,
            daily_loss_soft=0.06, daily_loss_hard=0.05,
            max_drawdown_soft=0.07, max_drawdown_hard=0.10, drawdown_trailing=False,
            max_concurrent_positions=3, max_bucket_risk=0.01, consecutive_losses=4,
            consecutive_loss_pause_hours=24.0,
            min_margin_level=3.0, max_spread_multiple=2.0, max_feed_age_seconds=10.0,
            atr_period=14, atr_stop_multiple=2.5, buckets={},
        )


def test_shipped_profiles_load_and_validate():
    from core.config import RiskProfile

    for name in ("challenge", "funded"):
        profile = RiskProfile.load(name)
        assert profile.daily_loss_soft < profile.daily_loss_hard
        assert profile.max_drawdown_soft < profile.max_drawdown_hard
        assert profile.risk_per_trade <= profile.max_risk_per_trade


def test_consecutive_loss_pause_expires():
    """Regression: a pause with no expiry is a permanent stop.

    Once the limit blocks every new trade, the strategy can never record the win
    that resets its streak. The first synthetic backtest took 4 trades in 1,500
    bars and rejected 1,033 signals on this limit alone.
    """
    from datetime import datetime, timedelta, timezone

    limit = ConsecutiveLosses(maximum=4, pause_hours=24.0)
    loss_at = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    trade = ProposedTrade("EURUSD", 0.1, 50.0, 0.0005, strategy="trend")

    blocked = state(
        consecutive_losses={"trend": 4},
        last_loss_ts={"trend": loss_at},
        now=loss_at + timedelta(hours=2),
    )
    assert limit.check(blocked, trade) is not None

    cooled = state(
        consecutive_losses={"trend": 4},
        last_loss_ts={"trend": loss_at},
        now=loss_at + timedelta(hours=25),
    )
    assert limit.check(cooled, trade) is None


def test_a_win_clears_the_streak_and_the_timestamp():
    book = SessionBook.open(100_000.0)
    ts = datetime(2026, 3, 2, tzinfo=timezone.utc)
    for _ in range(4):
        book.record_close("trend", -100.0, ts)
    assert book.consecutive_losses["trend"] == 4
    assert "trend" in book.last_loss_ts

    book.record_close("trend", 300.0, ts)
    assert book.consecutive_losses["trend"] == 0
    assert "trend" not in book.last_loss_ts
