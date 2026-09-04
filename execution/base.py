"""The broker boundary.

Every layer above this file talks to `ExecutionAdapter` and nothing else. No
strategy, no risk check and no research code may import `MetaTrader5`, a cTrader
client, or any other venue library. That rule is what makes the broker a
swappable detail rather than a rewrite, and it is worth enforcing in review.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from core.types import (
    AccountState,
    Bar,
    OrderRequest,
    OrderResult,
    Position,
    SymbolSpec,
    Tick,
)


class ExecutionError(RuntimeError):
    """Adapter could not complete an operation. Never raised for a rejected order."""


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Minimal surface every venue must implement.

    Deliberately small. Anything that can be computed above the adapter (position
    sizing, netting, signal state) is computed above the adapter, so a new venue
    is a few hundred lines rather than a project.
    """

    name: str

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def account(self) -> AccountState: ...

    def spec(self, symbol: str) -> SymbolSpec:
        """Contract specification, fetched live. Cached by the caller, not here."""
        ...

    def tick(self, symbol: str) -> Tick: ...

    def bars(self, symbol: str, timeframe: str, count: int, end: datetime | None = None) -> list[Bar]: ...

    def positions(self, symbol: str | None = None) -> list[Position]: ...

    def submit(self, request: OrderRequest) -> OrderResult: ...

    def modify(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult: ...

    def close(self, ticket: int, volume: float | None = None) -> OrderResult: ...


def reconcile(
    adapter: ExecutionAdapter,
    expected: dict[str, float],
    tolerance: float = 1e-6,
) -> dict[str, tuple[float, float]]:
    """Compare intended net exposure against the broker's actual positions.

    Returns {symbol: (expected_lots, actual_lots)} for every symbol that disagrees.
    An empty dict is the only acceptable steady state.

    Run this on startup, after every reconnect, and on a timer. The failure this
    catches — bot believes it is flat while the broker holds a position — is the
    one that turns a bad day into a blown account, and it is invisible until you
    look for it.
    """
    actual: dict[str, float] = {}
    for pos in adapter.positions():
        actual[pos.symbol] = actual.get(pos.symbol, 0.0) + pos.signed_volume

    drift: dict[str, tuple[float, float]] = {}
    for symbol in set(expected) | set(actual):
        want = expected.get(symbol, 0.0)
        have = actual.get(symbol, 0.0)
        if abs(want - have) > tolerance:
            drift[symbol] = (want, have)
    return drift
