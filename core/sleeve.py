"""A sleeve: one strategy, the symbols it trades, and its share of the risk budget.

The unit a professional book is built from. A sleeve is validated on its own,
sized on its own, journaled on its own, and switched off on its own. The
portfolio is the sum of sleeves; nobody chooses between them at trade time.

Two sleeves may trade the same symbol. Trend and carry can both hold EURUSD,
in opposite directions if they disagree -- the risk engine nets the exposure
and the correlated-bucket limit caps the total. That is the whole point of
separating them.

The 12-character limit on the name is not cosmetic. It is the prefix of the
broker's 31-character order comment, which is how a live position is traced
back to the sleeve that opened it after a crash. Longer names would be
truncated and become ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.strategy import Strategy
from core.types import Position

#: Builds a fresh strategy instance for one symbol. Strategies carry per-symbol
#: state (prepared frames, last bar seen), so every (sleeve, symbol) pair gets
#: its own.
StrategyFactory = Callable[[str], Strategy]

MAX_NAME = 12


@dataclass(frozen=True)
class Sleeve:
    name: str
    factory: StrategyFactory
    symbols: tuple[str, ...]
    #: Share of the portfolio's open-risk budget. Normalised across sleeves, so
    #: weights of 2 and 1 mean two thirds and one third.
    weight: float = 1.0
    timeframe: str = "D1"
    #: Overrides the profile's per-trade risk for this sleeve only. None means
    #: use the profile's number.
    risk_per_trade: float | None = None

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > MAX_NAME:
            raise ValueError(
                f"sleeve name {self.name!r} must be 1-{MAX_NAME} characters: it is the "
                f"order-comment prefix that identifies positions after a restart"
            )
        if "#" in self.name:
            raise ValueError("sleeve name may not contain '#' - it separates name from order id")
        if self.weight <= 0:
            raise ValueError(f"sleeve {self.name}: weight must be positive")
        if not self.symbols:
            raise ValueError(f"sleeve {self.name}: needs at least one symbol")

    def build(self, symbol: str) -> Strategy:
        """A strategy instance for one symbol, renamed to the sleeve so every
        journal entry, order comment and loss streak is attributed here."""
        strategy = self.factory(symbol)
        strategy.name = self.name  # instance attribute shadows the class default
        return strategy


def normalise_weights(sleeves: list[Sleeve]) -> dict[str, float]:
    total = sum(s.weight for s in sleeves)
    return {s.name: s.weight / total for s in sleeves}


def sleeve_of(position: Position) -> str:
    """Which sleeve opened this position, from its order comment prefix."""
    return position.comment.split("#", 1)[0] if position.comment else ""


def tag(sleeve_name: str, suffix: str = "bt") -> str:
    """The comment a sleeve's position carries. Same shape live and in backtest."""
    return f"{sleeve_name}#{suffix}"
