"""The strategy boundary.

A strategy answers one question per bar: **what position should I hold in this
instrument right now?** It returns an `Intent` — a direction and a stop distance.

Note what an `Intent` does not contain: a lot size. Sizing belongs to the risk
engine, and keeping it out of this type is what stops a strategy from ever being
able to risk more than the register allows. A model can change the intent; it can
never change the limit.

Strategies precompute their indicators once in `prepare()` and read index `i` in
`evaluate()`. That is safe only because every indicator in `features.indicators`
is causal — see that module, and the look-ahead test that verifies it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from core.types import Position, Side


@dataclass(frozen=True, slots=True)
class Intent:
    """Desired position. `side=None` means flat."""

    side: Side | None
    stop_distance: float = 0.0
    confidence: float = 1.0
    reason: str = ""

    @property
    def flat(self) -> bool:
        return self.side is None

    def __post_init__(self) -> None:
        if self.side is not None and self.stop_distance <= 0:
            raise ValueError("a directional intent must carry a positive stop distance")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


FLAT = Intent(side=None)


def forecast_to_confidence(forecast: float, cap: float = 2.0, floor: float = 0.25) -> float:
    """Map a signed forecast in volatility units to the sizer's confidence.

    |forecast| at or beyond `cap` is full size. Below it the position scales in
    proportion, but never below `floor` of full size: a dust position pays the
    same spread as a real one and cannot be sized on a small account anyway.
    Zero or NaN means no view, and the strategy should go flat rather than ask
    for a zero-size position.
    """
    if forecast != forecast:  # NaN
        return 0.0
    strength = min(abs(forecast) / cap, 1.0) if cap > 0 else 1.0
    if strength <= 0:
        return 0.0
    return max(strength, floor)


def is_month_start(df: pd.DataFrame, i: int) -> bool:
    """True on the first bar of a calendar month. A decision cadence defined by
    the calendar rather than by row position, so it is identical in a backtest
    over a full frame and live over a rolling window."""
    if i <= 0:
        return True
    ts = df["ts"]
    return ts.iloc[i].month != ts.iloc[i - 1].month or ts.iloc[i].year != ts.iloc[i - 1].year


class Strategy(ABC):
    """One instrument, one strategy. Portfolio composition happens above this."""

    name: str = "strategy"

    #: Bars required before the first signal can be trusted. The engine skips
    #: this many bars, so an indicator's NaN warmup never reaches a decision.
    warmup: int = 0

    #: Continuous strategies re-propose confidence and stop distance on every
    #: bar while a position is open, and the engines resize toward the new
    #: target through `RiskEngine.resize`. Discrete strategies hold what they
    #: opened until the signal flips or the stop fires. The pre-registered
    #: baselines are discrete.
    rebalances: bool = False
    #: Position inertia: the smallest relative change in target volume worth a
    #: trade. Below it the new target is noted and nothing goes to the broker.
    inertia: float = 0.25

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute indicator columns once, over the whole series.

        Safe only with causal indicators. Return the frame with columns added;
        the engine keeps it and passes the row index to `evaluate`.
        """
        return df

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        """Target position given data up to and including bar `i`.

        `df` is the full prepared frame. Reading beyond `i` is a look-ahead bug,
        not a shortcut — the engine cannot detect it for you, but the truncation
        test in the suite can.
        """

    def describe(self) -> str:
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{self.name}({rendered})"
