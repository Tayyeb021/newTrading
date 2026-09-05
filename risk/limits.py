"""The limits register.

Each limit is an independent object with one `check`. They compose, they are
individually testable, and adding one never touches the others. The engine runs
every limit and collects every breach rather than short-circuiting, because
knowing you broke three rules at once is diagnostic information you want in the
journal.

Soft and hard limits are deliberately separate. On a prop evaluation the firm's
number is the hard limit and you must never reach it; the soft limit is where you
stop trading, and the gap between them is the buffer that keeps the account alive
through one bad fill. Configure soft strictly inside hard, always.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum

from core.types import Position, SymbolSpec


class Severity(Enum):
    """What a breach does, in ascending order of finality."""

    REJECT = "reject"  # this order does not go through; system keeps trading
    PAUSE = "pause"  # strategy or symbol stands down for a while
    HALT = "halt"  # stop opening; existing positions managed to their exits
    FLATTEN = "flatten"  # close everything now


@dataclass(frozen=True, slots=True)
class Breach:
    limit: str
    severity: Severity
    message: str
    observed: float
    threshold: float

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.limit}: {self.message}"


@dataclass
class RiskState:
    """Everything a limit is allowed to look at. One snapshot, no hidden reads."""

    equity: float
    balance: float
    margin_level: float
    day_start_equity: float
    high_water_equity: float
    starting_equity: float
    positions: list[Position] = field(default_factory=list)
    # Specs live on the state because limits need them to value open risk, and
    # threading them through every check signature for the benefit of one limit
    # is worse than putting them in the snapshot everything already receives.
    specs: dict[str, SymbolSpec] = field(default_factory=dict)
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    current_price: dict[str, float] = field(default_factory=dict)
    current_spread: dict[str, float] = field(default_factory=dict)
    median_spread: dict[str, float] = field(default_factory=dict)
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_loss_ts: dict[str, datetime] = field(default_factory=dict)
    trading_day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    killed: bool = False
    feed_age_seconds: float = 0.0

    @property
    def daily_pnl_fraction(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.equity - self.day_start_equity) / self.day_start_equity

    def drawdown_fraction(self, trailing: bool) -> float:
        """Positive number meaning 'how far below the reference we are'."""
        reference = self.high_water_equity if trailing else self.starting_equity
        if reference <= 0:
            return 0.0
        return max((reference - self.equity) / reference, 0.0)


@dataclass(frozen=True, slots=True)
class ProposedTrade:
    """The order under consideration, after sizing and before submission."""

    symbol: str
    volume: float
    risk_cash: float
    risk_fraction: float
    strategy: str = ""


class Limit(ABC):
    """One rule. Returns a Breach when violated, None when satisfied."""

    name: str = "limit"
    severity: Severity = Severity.REJECT

    @abstractmethod
    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None: ...

    def _breach(self, message: str, observed: float, threshold: float) -> Breach:
        return Breach(self.name, self.severity, message, observed, threshold)


# --------------------------------------------------------------------------- #
# Account-level limits — these can halt the system
# --------------------------------------------------------------------------- #


class KillSwitch(Limit):
    name = "kill_switch"
    severity = Severity.FLATTEN

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if state.killed:
            return self._breach("kill switch is engaged", 1.0, 0.0)
        return None


class DailyLoss(Limit):
    """Loss since the session's starting equity.

    Measured against day-start equity, checked continuously — which is how a prop
    firm's server evaluates it, and why an end-of-day-only check is not enough.
    """

    name = "daily_loss"

    def __init__(self, soft: float, hard: float, severity: Severity = Severity.HALT) -> None:
        if not 0 < soft < hard < 1:
            raise ValueError(f"require 0 < soft ({soft}) < hard ({hard}) < 1")
        self.soft = soft
        self.hard = hard
        self.severity = severity

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        loss = -min(state.daily_pnl_fraction, 0.0)
        if loss >= self.hard:
            return Breach(
                self.name, Severity.FLATTEN,
                f"daily loss {loss:.2%} reached the HARD limit {self.hard:.2%}",
                loss, self.hard,
            )
        if loss >= self.soft:
            return self._breach(
                f"daily loss {loss:.2%} reached the soft limit {self.soft:.2%}", loss, self.soft
            )
        # Would this trade, at its full stop, take us past the soft limit?
        if trade is not None and state.day_start_equity > 0:
            projected = loss + trade.risk_cash / state.day_start_equity
            if projected >= self.soft:
                return Breach(
                    self.name, Severity.REJECT,
                    f"trade risking {trade.risk_cash:,.2f} would put daily loss at "
                    f"{projected:.2%}, past the soft limit {self.soft:.2%}",
                    projected, self.soft,
                )
        return None


class MaxDrawdown(Limit):
    name = "max_drawdown"

    def __init__(
        self,
        soft: float,
        hard: float,
        trailing: bool = False,
        severity: Severity = Severity.HALT,
    ) -> None:
        if not 0 < soft < hard < 1:
            raise ValueError(f"require 0 < soft ({soft}) < hard ({hard}) < 1")
        self.soft = soft
        self.hard = hard
        self.trailing = trailing
        self.severity = severity

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        dd = state.drawdown_fraction(self.trailing)
        mode = "trailing" if self.trailing else "static"
        if dd >= self.hard:
            return Breach(
                self.name, Severity.FLATTEN,
                f"{mode} drawdown {dd:.2%} reached the HARD limit {self.hard:.2%}",
                dd, self.hard,
            )
        if dd >= self.soft:
            return self._breach(
                f"{mode} drawdown {dd:.2%} reached the soft limit {self.soft:.2%}", dd, self.soft
            )
        return None


class MarginLevel(Limit):
    name = "margin_level"
    severity = Severity.REJECT

    def __init__(self, minimum: float = 3.0) -> None:
        self.minimum = minimum

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if state.margin_level < self.minimum:
            return self._breach(
                f"margin level {state.margin_level:.2f} below minimum {self.minimum:.2f}",
                state.margin_level, self.minimum,
            )
        return None


class FeedHeartbeat(Limit):
    name = "feed_heartbeat"
    severity = Severity.HALT

    def __init__(self, max_age_seconds: float = 10.0) -> None:
        self.max_age = max_age_seconds

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if state.feed_age_seconds > self.max_age:
            return self._breach(
                f"price feed is {state.feed_age_seconds:.1f}s stale, limit {self.max_age:.1f}s",
                state.feed_age_seconds, self.max_age,
            )
        return None


# --------------------------------------------------------------------------- #
# Trade-level limits — these reject one order
# --------------------------------------------------------------------------- #


class RiskPerTrade(Limit):
    name = "risk_per_trade"
    severity = Severity.REJECT

    def __init__(self, maximum: float) -> None:
        self.maximum = maximum

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if trade is None:
            return None
        if trade.risk_fraction > self.maximum + 1e-9:
            return self._breach(
                f"trade risks {trade.risk_fraction:.3%}, maximum {self.maximum:.3%}",
                trade.risk_fraction, self.maximum,
            )
        return None


class CorrelatedBucket(Limit):
    """Total risk across instruments that are really one bet.

    US30 and US500 correlate around 0.9. Running independent risk on each is a
    single position wearing a disguise, and this limit is what stops it.
    """

    name = "correlated_bucket"
    severity = Severity.REJECT

    def __init__(self, buckets: dict[str, list[str]], maximum: float) -> None:
        self.maximum = maximum
        self._of: dict[str, str] = {}
        for bucket, symbols in buckets.items():
            for symbol in symbols:
                self._of[symbol] = bucket

    def bucket_of(self, symbol: str) -> str | None:
        return self._of.get(symbol)

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if trade is None:
            return None
        bucket = self.bucket_of(trade.symbol)
        if bucket is None:
            return None

        # Open risk in this bucket, approximated by position stop distance where
        # known. Positions without a stop count their full notional risk budget.
        open_risk = 0.0
        for pos in state.positions:
            if self.bucket_of(pos.symbol) != bucket:
                continue
            open_risk += _position_risk_fraction(pos, state)

        projected = open_risk + trade.risk_fraction
        if projected > self.maximum + 1e-9:
            return self._breach(
                f"bucket {bucket!r} would carry {projected:.3%} risk, maximum {self.maximum:.3%}",
                projected, self.maximum,
            )
        return None


class MaxConcurrentPositions(Limit):
    name = "max_positions"
    severity = Severity.REJECT

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if trade is None:
            return None
        if len(state.positions) >= self.maximum:
            return self._breach(
                f"{len(state.positions)} positions open, maximum {self.maximum}",
                float(len(state.positions)), float(self.maximum),
            )
        return None


class SpreadGuard(Limit):
    name = "spread_guard"
    severity = Severity.REJECT

    def __init__(self, max_multiple: float = 2.0) -> None:
        self.max_multiple = max_multiple

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if trade is None:
            return None
        current = state.current_spread.get(trade.symbol)
        median = state.median_spread.get(trade.symbol)
        if current is None or not median:
            return None
        multiple = current / median
        if multiple > self.max_multiple:
            return self._breach(
                f"{trade.symbol} spread is {multiple:.1f}x median, limit {self.max_multiple:.1f}x",
                multiple, self.max_multiple,
            )
        return None


class ConsecutiveLosses(Limit):
    """Stand a strategy down after a losing streak, for a bounded time.

    The time bound is the whole point. A pause with no expiry is a permanent
    stop: once the limit blocks every new trade, the strategy can never record
    the win that would reset its streak, and it is dead for the life of the
    process. That deadlock is easy to write and invisible until a backtest shows
    four trades where it should show four hundred.
    """

    name = "consecutive_losses"
    severity = Severity.PAUSE

    def __init__(self, maximum: int, pause_hours: float = 24.0) -> None:
        self.maximum = maximum
        self.pause_hours = pause_hours

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if trade is None:
            return None
        streak = state.consecutive_losses.get(trade.strategy, 0)
        if streak < self.maximum:
            return None

        last = state.last_loss_ts.get(trade.strategy)
        if last is not None:
            elapsed = (state.now - last).total_seconds() / 3600.0
            if elapsed >= self.pause_hours:
                return None  # cooled off; the streak no longer blocks

        return self._breach(
            f"strategy {trade.strategy!r} has {streak} consecutive losses, "
            f"maximum {self.maximum} (pause {self.pause_hours:g}h)",
            float(streak), float(self.maximum),
        )


class SleeveBudget(Limit):
    """Each sleeve gets a share of the book's open-risk budget; the book has a cap.

    Six sleeves each risking 0.5% per trade are not six independent bets when
    they share one drawdown limit. This is the allocator: sleeve A may hold at
    most its share of the portfolio's open risk, and the portfolio as a whole may
    not exceed the cap regardless of how it is split. Open risk is measured from
    stops, which is why every position must carry one.

    A position is attributed to a sleeve by its order-comment prefix, which is
    the same mechanism that identifies positions after a crash.
    """

    name = "sleeve_budget"
    severity = Severity.REJECT

    def __init__(self, caps: dict[str, float], portfolio_cap: float) -> None:
        if any(c > portfolio_cap + 1e-12 for c in caps.values()):
            raise ValueError("a sleeve cap cannot exceed the portfolio cap")
        self.caps = dict(caps)
        self.portfolio_cap = portfolio_cap

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        if trade is None:
            return None
        from core.sleeve import sleeve_of

        open_by: dict[str, float] = {}
        for pos in state.positions:
            open_by[sleeve_of(pos)] = open_by.get(sleeve_of(pos), 0.0) + _position_risk_fraction(pos, state)
        total = sum(open_by.values())
        mine = trade.strategy
        cap = self.caps.get(mine)

        if cap is not None and open_by.get(mine, 0.0) + trade.risk_fraction > cap + 1e-9:
            return self._breach(
                f"sleeve {mine!r} would hold {open_by.get(mine, 0.0) + trade.risk_fraction:.3%} "
                f"open risk, budget {cap:.3%}",
                open_by.get(mine, 0.0) + trade.risk_fraction, cap,
            )
        if total + trade.risk_fraction > self.portfolio_cap + 1e-9:
            return self._breach(
                f"book would hold {total + trade.risk_fraction:.3%} open risk, cap {self.portfolio_cap:.3%}",
                total + trade.risk_fraction, self.portfolio_cap,
            )
        return None


class UnstoppedPosition(Limit):
    """No position may exist without a stop.

    An unstopped position has undefined risk, which means every downstream limit
    that aggregates risk is quietly wrong while one is open. Treat it as a system
    fault rather than a style preference.
    """

    name = "unstopped_position"
    severity = Severity.HALT

    def check(self, state: RiskState, trade: ProposedTrade | None) -> Breach | None:
        naked = [p.symbol for p in state.positions if p.stop_loss is None]
        if naked:
            return self._breach(
                f"positions without a stop loss: {', '.join(sorted(set(naked)))}",
                float(len(naked)), 0.0,
            )
        return None


def _position_risk_fraction(pos: Position, state: RiskState) -> float:
    """Risk still on the table for an open position, as a fraction of equity.

    A position without a stop has undefined risk. `UnstoppedPosition` catches that
    as a fault; here we return zero rather than invent a number, and the fault is
    what stops trading.
    """
    spec = state.specs.get(pos.symbol)
    if spec is None or pos.stop_loss is None or state.equity <= 0:
        return 0.0
    distance = abs(pos.entry_price - pos.stop_loss)
    return spec.risk_for(pos.volume, distance) / state.equity
