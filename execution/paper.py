"""Paper adapter - a broker that exists entirely in memory.

Its job is to be pessimistic. Fills cross the spread, slippage is charged against
you, and stop levels are enforced exactly as a real server enforces them. A paper
adapter that flatters your strategy is worse than no paper adapter at all, because
it launders bugs into confidence.

Every test in `tests/` runs against this, which is why the whole system can be
verified without a broker connection or a Windows terminal.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.types import (
    AccountState,
    Bar,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    Side,
    SymbolSpec,
    Tick,
)
from execution.base import ExecutionError


@dataclass
class PaperConfig:
    starting_balance: float = 100_000.0
    # Slippage charged on every market fill, as a multiple of the spread.
    slippage_spread_multiple: float = 0.25
    # Random extra slippage, in price units, drawn uniformly in [0, jitter].
    slippage_jitter: float = 0.0
    seed: int | None = 7


class PaperAdapter:
    """In-memory broker. Drive it by feeding ticks."""

    name = "paper"

    def __init__(
        self,
        specs: dict[str, SymbolSpec],
        config: PaperConfig | None = None,
    ) -> None:
        self._specs = dict(specs)
        self._cfg = config or PaperConfig()
        self._rng = random.Random(self._cfg.seed)
        self._connected = False
        self._tickets = itertools.count(1)

        self._ticks: dict[str, Tick] = {}
        self._bars: dict[tuple[str, str], list[Bar]] = {}
        self._positions: dict[int, Position] = {}

        self.balance = self._cfg.starting_balance
        self.realized_pnl = 0.0
        self.fills: list[OrderResult] = []

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _require_connection(self) -> None:
        if not self._connected:
            raise ExecutionError("paper adapter is not connected")

    # ------------------------------------------------------------------ feeding

    def feed_tick(self, tick: Tick) -> None:
        """Advance the market. Everything else keys off this."""
        self._ticks[tick.symbol] = tick

    def feed_bars(self, symbol: str, timeframe: str, bars: list[Bar]) -> None:
        self._bars[(symbol, timeframe)] = list(bars)

    # -------------------------------------------------------------------- reads

    def account(self) -> AccountState:
        self._require_connection()
        floating = 0.0
        margin = 0.0
        for pos in self._positions.values():
            tick = self._ticks.get(pos.symbol)
            spec = self._specs[pos.symbol]
            if tick is not None:
                close_price = tick.bid if pos.side is Side.BUY else tick.ask
                floating += pos.unrealized(close_price, spec)
                margin += pos.volume * spec.contract_size * tick.mid / 100.0  # 1:100 proxy
        equity = self.balance + floating
        return AccountState(
            equity=equity,
            balance=self.balance,
            margin_used=margin,
            margin_free=max(equity - margin, 0.0),
        )

    def spec(self, symbol: str) -> SymbolSpec:
        try:
            return self._specs[symbol]
        except KeyError:
            raise ExecutionError(f"unknown symbol {symbol!r}") from None

    def tick(self, symbol: str) -> Tick:
        self._require_connection()
        try:
            return self._ticks[symbol]
        except KeyError:
            raise ExecutionError(f"no tick for {symbol!r} - feed one first") from None

    def bars(self, symbol: str, timeframe: str, count: int, end: datetime | None = None) -> list[Bar]:
        series = self._bars.get((symbol, timeframe), [])
        if end is not None:
            series = [b for b in series if b.ts <= end]
        return series[-count:]

    def positions(self, symbol: str | None = None) -> list[Position]:
        out = list(self._positions.values())
        return [p for p in out if symbol is None or p.symbol == symbol]

    # ------------------------------------------------------------------- writes

    def submit(self, request: OrderRequest) -> OrderResult:
        self._require_connection()
        spec = self.spec(request.symbol)
        tick = self.tick(request.symbol)

        if request.order_type is not OrderType.MARKET:
            return self._reject(request, "paper adapter supports market orders only")

        volume = spec.round_volume(request.volume)
        if volume < spec.volume_min:
            return self._reject(
                request,
                f"volume {request.volume:.4f} rounds to {volume:.4f}, "
                f"below minimum {spec.volume_min:.4f}",
            )
        if volume > spec.volume_max:
            return self._reject(request, f"volume {volume:.4f} above maximum {spec.volume_max:.4f}")

        reference = tick.ask if request.side is Side.BUY else tick.bid
        fill = self._apply_slippage(reference, tick, request.side)
        fill = spec.normalize_price(fill)

        if not self._stops_valid(request, fill, spec):
            return self._reject(request, "stop or target inside broker stops level")

        ticket = next(self._tickets)
        self._positions[ticket] = Position(
            symbol=request.symbol,
            side=request.side,
            volume=volume,
            entry_price=fill,
            opened_at=tick.ts,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            ticket=ticket,
            comment=request.comment,
        )
        result = OrderResult(
            status=OrderStatus.FILLED,
            request=request,
            ticket=ticket,
            fill_price=fill,
            filled_volume=volume,
            requested_price=reference,
            ts=tick.ts,
        )
        self.fills.append(result)
        return result

    def modify(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        self._require_connection()
        pos = self._positions.get(ticket)
        if pos is None:
            raise ExecutionError(f"no open position with ticket {ticket}")

        updated = Position(
            symbol=pos.symbol,
            side=pos.side,
            volume=pos.volume,
            entry_price=pos.entry_price,
            opened_at=pos.opened_at,
            stop_loss=stop_loss if stop_loss is not None else pos.stop_loss,
            take_profit=take_profit if take_profit is not None else pos.take_profit,
            ticket=ticket,
            comment=pos.comment,
        )
        self._positions[ticket] = updated
        request = OrderRequest(
            symbol=pos.symbol, side=pos.side, volume=pos.volume, comment="modify"
        )
        return OrderResult(status=OrderStatus.FILLED, request=request, ticket=ticket)

    def close(self, ticket: int, volume: float | None = None) -> OrderResult:
        self._require_connection()
        pos = self._positions.get(ticket)
        if pos is None:
            raise ExecutionError(f"no open position with ticket {ticket}")

        spec = self.spec(pos.symbol)
        tick = self.tick(pos.symbol)
        closing = pos.volume if volume is None else spec.round_volume(volume)
        closing = min(closing, pos.volume)

        reference = tick.bid if pos.side is Side.BUY else tick.ask
        fill = spec.normalize_price(self._apply_slippage(reference, tick, pos.side.opposite()))

        move = (fill - pos.entry_price) * pos.side.sign
        self.realized_pnl += move * closing * spec.value_per_price_unit
        self.balance += move * closing * spec.value_per_price_unit

        remaining = round(pos.volume - closing, 8)
        if remaining >= spec.volume_step:
            self._positions[ticket] = Position(
                symbol=pos.symbol,
                side=pos.side,
                volume=remaining,
                entry_price=pos.entry_price,
                opened_at=pos.opened_at,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                ticket=ticket,
                comment=pos.comment,
            )
        else:
            del self._positions[ticket]

        request = OrderRequest(
            symbol=pos.symbol, side=pos.side.opposite(), volume=closing, comment="close"
        )
        result = OrderResult(
            status=OrderStatus.FILLED,
            request=request,
            ticket=ticket,
            fill_price=fill,
            filled_volume=closing,
            requested_price=reference,
            ts=tick.ts,
        )
        self.fills.append(result)
        return result

    # ------------------------------------------------------------------ helpers

    def _apply_slippage(self, reference: float, tick: Tick, side: Side) -> float:
        """Slippage always moves against the taker. Never in your favour."""
        adverse = tick.spread * self._cfg.slippage_spread_multiple
        if self._cfg.slippage_jitter:
            adverse += self._rng.uniform(0.0, self._cfg.slippage_jitter)
        return reference + adverse * side.sign

    def _stops_valid(self, request: OrderRequest, fill: float, spec: SymbolSpec) -> bool:
        min_dist = spec.min_stop_distance
        if min_dist <= 0:
            return True
        if request.stop_loss is not None and abs(fill - request.stop_loss) < min_dist:
            return False
        if request.take_profit is not None and abs(fill - request.take_profit) < min_dist:
            return False
        return True

    @staticmethod
    def _reject(request: OrderRequest, reason: str) -> OrderResult:
        return OrderResult(status=OrderStatus.REJECTED, request=request, reason=reason)


def spec_from_dict(symbol: str, raw: dict) -> SymbolSpec:
    """Build a spec from a plain dict - used by tests and by the offline fixtures."""
    return SymbolSpec(symbol=symbol, **raw)


# Fixtures matching common broker values, for tests and for the sizing sanity
# checks in scripts/check_specs.py. These are *plausible*, not authoritative:
# the real ones come from the broker at startup, always.
FIXTURE_SPECS: dict[str, SymbolSpec] = {
    "EURUSD": SymbolSpec(
        symbol="EURUSD", digits=5, point=0.00001, tick_size=0.00001, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.01, contract_size=100_000.0,
        stops_level_points=0,
    ),
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD", digits=2, point=0.01, tick_size=0.01, tick_value=1.0,
        volume_min=0.01, volume_max=50.0, volume_step=0.01, contract_size=100.0,
        stops_level_points=0,
    ),
    "US30": SymbolSpec(
        symbol="US30", digits=1, point=0.1, tick_size=0.1, tick_value=0.1,
        volume_min=0.1, volume_max=50.0, volume_step=0.1, contract_size=1.0,
        stops_level_points=0,
    ),
    "US500": SymbolSpec(
        symbol="US500", digits=1, point=0.1, tick_size=0.1, tick_value=0.1,
        volume_min=0.1, volume_max=50.0, volume_step=0.1, contract_size=1.0,
        stops_level_points=0,
    ),
}


def make_tick(symbol: str, bid: float, spread: float, ts: datetime | None = None) -> Tick:
    return Tick(
        symbol=symbol,
        ts=ts or datetime.now(timezone.utc),
        bid=bid,
        ask=bid + spread,
    )
