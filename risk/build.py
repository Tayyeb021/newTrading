"""Turn a `RiskProfile` into a live `RiskEngine`.

The ordering of limits here is deliberate: cheapest and most final first, so a
killed or halted system stops before anything else is computed.
"""

from __future__ import annotations

from core.config import RiskProfile
from core.types import SymbolSpec
from risk.engine import RiskEngine, SessionBook
from risk.limits import (
    ConsecutiveLosses,
    CorrelatedBucket,
    DailyLoss,
    FeedHeartbeat,
    KillSwitch,
    Limit,
    MarginLevel,
    MaxConcurrentPositions,
    MaxDrawdown,
    RiskPerTrade,
    SpreadGuard,
    UnstoppedPosition,
)


def build_limits(profile: RiskProfile) -> list[Limit]:
    return [
        KillSwitch(),
        DailyLoss(profile.daily_loss_soft, profile.daily_loss_hard),
        MaxDrawdown(
            profile.max_drawdown_soft,
            profile.max_drawdown_hard,
            trailing=profile.drawdown_trailing,
        ),
        UnstoppedPosition(),
        FeedHeartbeat(profile.max_feed_age_seconds),
        MarginLevel(profile.min_margin_level),
        RiskPerTrade(profile.max_risk_per_trade),
        CorrelatedBucket(profile.buckets, profile.max_bucket_risk),
        MaxConcurrentPositions(profile.max_concurrent_positions),
        SpreadGuard(profile.max_spread_multiple),
        ConsecutiveLosses(profile.consecutive_losses, profile.consecutive_loss_pause_hours),
    ]


def build_engine(
    profile: RiskProfile,
    starting_equity: float,
    specs: dict[str, SymbolSpec] | None = None,
) -> RiskEngine:
    return RiskEngine(
        limits=build_limits(profile),
        book=SessionBook.open(starting_equity),
        risk_per_trade=profile.risk_per_trade,
        specs=specs,
    )
