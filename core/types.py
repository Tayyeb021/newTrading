"""Domain types shared by every layer.

Nothing in here imports a broker library. Strategies, risk and research all speak
these types; only the execution adapters know what a broker is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(Enum):
    BUY = 1
    SELL = -1

    @property
    def sign(self) -> int:
        return self.value

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    ts: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: datetime  # bar OPEN time, always UTC
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError(f"{self.symbol} bar timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Contract specification, read from the broker at startup. Never hardcoded.

    `value_per_price_unit` is the number that actually matters for sizing: how much
    cash one lot gains or loses per 1.0 of price movement. Getting this wrong by a
    factor of ten is the most expensive bug available in retail algo trading, which
    is why it is derived here once and asserted rather than assumed per strategy.
    """

    symbol: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    stops_level_points: int = 0
    swap_long: float = 0.0
    swap_short: float = 0.0
    currency_profit: str = "USD"

    def __post_init__(self) -> None:
        for name in ("tick_size", "tick_value", "volume_min", "volume_step"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{self.symbol}: {name} must be positive, got {getattr(self, name)}")
        if self.volume_max < self.volume_min:
            raise ValueError(f"{self.symbol}: volume_max < volume_min")

    @property
    def value_per_price_unit(self) -> float:
        """Cash P&L per lot per 1.0 of price movement."""
        return self.tick_value / self.tick_size

    @property
    def min_stop_distance(self) -> float:
        """Broker-enforced minimum distance between price and a stop, in price."""
        return self.stops_level_points * self.point

    def round_volume(self, volume: float) -> float:
        """Round DOWN to the nearest valid step.

        Down, not nearest: rounding up silently exceeds the risk limit the caller
        just computed, and a limit you can round past is not a limit.
        """
        if volume <= 0:
            return 0.0
        steps = math.floor(volume / self.volume_step + 1e-9)
        return round(steps * self.volume_step, 8)

    def normalize_price(self, price: float) -> float:
        return round(price, self.digits)

    def risk_for(self, volume: float, stop_distance: float) -> float:
        """Cash at risk for a position of `volume` lots with `stop_distance` in price."""
        return volume * stop_distance * self.value_per_price_unit


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    side: Side
    volume: float
    entry_price: float
    opened_at: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    ticket: int | None = None
    comment: str = ""

    @property
    def signed_volume(self) -> float:
        return self.volume * self.side.sign

    def unrealized(self, price: float, spec: SymbolSpec) -> float:
        move = (price - self.entry_price) * self.side.sign
        return move * self.volume * spec.value_per_price_unit


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    side: Side
    volume: float
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str = ""
    # Set by the risk engine; carried through so the journal can reconstruct why.
    intent: str = ""


@dataclass(frozen=True, slots=True)
class OrderResult:
    status: OrderStatus
    request: OrderRequest
    ticket: int | None = None
    fill_price: float | None = None
    filled_volume: float = 0.0
    reason: str = ""
    requested_price: float | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def ok(self) -> bool:
        return self.status is OrderStatus.FILLED

    def slippage(self) -> float | None:
        """Signed slippage in price. Positive means the fill was worse than asked."""
        if self.fill_price is None or self.requested_price is None:
            return None
        return (self.fill_price - self.requested_price) * self.request.side.sign


@dataclass(frozen=True, slots=True)
class AccountState:
    equity: float
    balance: float
    margin_used: float
    margin_free: float
    currency: str = "USD"
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def margin_level(self) -> float:
        """Equity / margin used. Infinite when flat."""
        if self.margin_used <= 0:
            return math.inf
        return self.equity / self.margin_used


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's proposal. Note what is absent: a lot size.

    Direction and conviction are the strategy's business. Size is the risk
    engine's, and this type is the boundary that enforces it.
    """

    symbol: str
    side: Side
    stop_distance: float  # in price, normally an ATR multiple
    confidence: float = 1.0  # 0..1, used for adaptive sizing later
    strategy: str = ""
    take_profit_distance: float | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.stop_distance <= 0:
            raise ValueError(f"{self.symbol}: stop_distance must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.symbol}: confidence must be in [0, 1]")
