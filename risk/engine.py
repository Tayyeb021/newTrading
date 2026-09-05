"""The risk gate.

Every order in the system passes through `RiskEngine.evaluate`. Strategies do not
call the adapter; they produce a `Signal`, and this decides whether it becomes an
order and how large. That separation is the whole architecture — a model can
change the proposal, never the limit.

The engine runs every limit and collects every breach rather than stopping at the
first. Knowing you tripped three rules simultaneously is diagnostic information,
and it costs nothing to gather.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from core.types import OrderRequest, Position, Signal, SymbolSpec
from risk.limits import (
    Breach,
    Limit,
    ProposedTrade,
    RiskState,
    Severity,
)
from risk.sizing import SizeResult, SizingOutcome, size_position

_ACTIONABLE = (Severity.HALT, Severity.FLATTEN)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    order: OrderRequest | None
    size: SizeResult | None
    breaches: tuple[Breach, ...] = ()
    note: str = ""

    @property
    def severity(self) -> Severity | None:
        if not self.breaches:
            return None
        return max((b.severity for b in self.breaches), key=lambda s: _SEVERITY_ORDER[s])

    @property
    def must_flatten(self) -> bool:
        return any(b.severity is Severity.FLATTEN for b in self.breaches)

    @property
    def must_halt(self) -> bool:
        return any(b.severity in _ACTIONABLE for b in self.breaches)

    def explain(self) -> str:
        if self.approved:
            assert self.size is not None
            return (
                f"APPROVED {self.order.symbol} {self.order.side.name} "
                f"{self.order.volume:g} lots, risking {self.size.risk_fraction:.3%}"
            )
        lines = [f"REJECTED: {self.note}" if self.note else "REJECTED"]
        lines.extend(f"  {b}" for b in self.breaches)
        return "\n".join(lines)


_SEVERITY_ORDER = {
    Severity.REJECT: 0,
    Severity.PAUSE: 1,
    Severity.HALT: 2,
    Severity.FLATTEN: 3,
}


@dataclass
class SessionBook:
    """Mutable bookkeeping the limits read but never write.

    Kept separate from `RiskState` so the snapshot handed to limits stays
    immutable in spirit: limits observe, the engine records.
    """

    starting_equity: float
    day_start_equity: float
    high_water_equity: float
    trading_day: date
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    last_loss_ts: dict[str, datetime] = field(default_factory=dict)
    killed: bool = False
    kill_reason: str = ""

    @classmethod
    def open(cls, equity: float, day: date | None = None) -> "SessionBook":
        return cls(
            starting_equity=equity,
            day_start_equity=equity,
            high_water_equity=equity,
            trading_day=day or datetime.now(timezone.utc).date(),
        )

    def observe_equity(self, equity: float, today: date | None = None) -> None:
        """Call on every loop iteration, before evaluating anything."""
        today = today or datetime.now(timezone.utc).date()
        if today != self.trading_day:
            self.trading_day = today
            self.day_start_equity = equity
        self.high_water_equity = max(self.high_water_equity, equity)

    def record_close(self, strategy: str, pnl: float, ts: datetime | None = None) -> None:
        if pnl < 0:
            self.consecutive_losses[strategy] = self.consecutive_losses.get(strategy, 0) + 1
            self.last_loss_ts[strategy] = ts or datetime.now(timezone.utc)
        else:
            self.consecutive_losses[strategy] = 0
            self.last_loss_ts.pop(strategy, None)

    def kill(self, reason: str) -> None:
        self.killed = True
        self.kill_reason = reason

    def revive(self) -> None:
        self.killed = False
        self.kill_reason = ""


class RiskEngine:
    def __init__(
        self,
        limits: list[Limit],
        book: SessionBook,
        risk_per_trade: float,
        specs: dict[str, SymbolSpec] | None = None,
    ) -> None:
        self.limits = list(limits)
        self.book = book
        self.risk_per_trade = risk_per_trade
        self.specs: dict[str, SymbolSpec] = dict(specs or {})

    # ------------------------------------------------------------------ state

    def snapshot(
        self,
        *,
        equity: float,
        balance: float,
        margin_level: float,
        positions: list[Position],
        now: datetime | None = None,
        current_price: dict[str, float] | None = None,
        current_spread: dict[str, float] | None = None,
        median_spread: dict[str, float] | None = None,
        feed_age_seconds: float = 0.0,
    ) -> RiskState:
        return RiskState(
            equity=equity,
            balance=balance,
            margin_level=margin_level,
            day_start_equity=self.book.day_start_equity,
            high_water_equity=self.book.high_water_equity,
            starting_equity=self.book.starting_equity,
            positions=list(positions),
            specs=self.specs,
            now=now or datetime.now(timezone.utc),
            last_loss_ts=dict(self.book.last_loss_ts),
            consecutive_losses=dict(self.book.consecutive_losses),
            current_price=dict(current_price or {}),
            current_spread=dict(current_spread or {}),
            median_spread=dict(median_spread or {}),
            trading_day=self.book.trading_day,
            killed=self.book.killed,
            feed_age_seconds=feed_age_seconds,
        )

    # ------------------------------------------------------------- evaluation

    def check_account(self, state: RiskState) -> tuple[Breach, ...]:
        """Account-level check with no trade in hand. Run this every iteration.

        This is what catches a daily-loss breach caused by an open position moving
        against you while you are not trying to trade — the exact case an
        order-time-only check misses.
        """
        return tuple(b for lim in self.limits if (b := lim.check(state, None)) is not None)

    def evaluate(
        self,
        signal: Signal,
        state: RiskState,
        *,
        regime_scalar: float = 1.0,
        max_volume: float | None = None,
        risk_fraction: float | None = None,
    ) -> RiskDecision:
        spec = self.specs.get(signal.symbol)
        if spec is None:
            return RiskDecision(False, None, None, (), f"no spec loaded for {signal.symbol}")

        # Account-level limits first: if we are halted, size is irrelevant.
        account_breaches = self.check_account(state)
        blocking = tuple(b for b in account_breaches if b.severity in _ACTIONABLE)
        if blocking:
            return RiskDecision(False, None, None, account_breaches, "account-level limit engaged")

        size = size_position(
            spec,
            equity=state.equity,
            risk_fraction=risk_fraction if risk_fraction is not None else self.risk_per_trade,
            stop_distance=signal.stop_distance,
            regime_scalar=regime_scalar,
            confidence=signal.confidence,
            max_volume=max_volume,
        )
        if not size.tradeable:
            return RiskDecision(
                False, None, size, account_breaches,
                f"sizing returned {size.outcome.value}: {size.reason}",
            )

        trade = ProposedTrade(
            symbol=signal.symbol,
            volume=size.volume,
            risk_cash=size.risk_cash,
            risk_fraction=size.risk_fraction,
            strategy=signal.strategy,
        )
        trade_breaches = tuple(
            b for lim in self.limits if (b := lim.check(state, trade)) is not None
        )
        if trade_breaches:
            return RiskDecision(False, None, size, trade_breaches, "trade-level limit")

        # A stop price requires a reference price. Without one we cannot attach a
        # stop, and an order with no stop has undefined risk — so we refuse rather
        # than submit one and hope to attach it afterwards.
        entry = state.current_price.get(signal.symbol)
        if entry is None:
            return RiskDecision(
                False, None, size, (),
                f"no current price for {signal.symbol}; cannot place a stop",
            )
        stop = spec.normalize_price(self._stop_price(signal, entry))
        take_raw = self._take_price(signal, entry)
        take = spec.normalize_price(take_raw) if take_raw is not None else None

        if spec.min_stop_distance and abs(entry - stop) < spec.min_stop_distance:
            return RiskDecision(
                False, None, size, (),
                f"stop {abs(entry - stop):.5f} inside broker minimum "
                f"{spec.min_stop_distance:.5f} for {signal.symbol}",
            )

        order = OrderRequest(
            symbol=signal.symbol,
            side=signal.side,
            volume=size.volume,
            stop_loss=stop,
            take_profit=take,
            comment=signal.strategy[:31],
            intent=f"{signal.strategy}:{signal.side.name}",
        )
        return RiskDecision(True, order, size, ())

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _stop_price(signal: Signal, entry: float) -> float:
        return entry - signal.stop_distance * signal.side.sign

    @staticmethod
    def _take_price(signal: Signal, entry: float) -> float | None:
        if signal.take_profit_distance is None:
            return None
        return entry + signal.take_profit_distance * signal.side.sign
