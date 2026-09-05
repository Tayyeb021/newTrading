"""MetaTrader 5 adapter.

Three hard constraints of the MT5 Python package shape this file:

1. It is Windows-only and needs a running terminal on the same machine.
2. One terminal connection per process.
3. `order_send` is **synchronous and blocking** — there is no async variant.

Point 3 is the one with architectural consequences. A slow fill on gold during a
news spike will stall whatever thread calls it, so in production this adapter runs
in its own worker thread behind a queue and never on the event loop. That wiring
belongs to the runner in phase 3; this file simply stays honest about blocking and
keeps every call timeout-bounded.

The import is lazy so that the rest of the system — tests, research, the paper
adapter — works on any platform without MetaTrader5 installed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

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
from execution.brokertime import ClockCheck, server_epoch_to_utc, verify_offset

log = logging.getLogger(__name__)

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
}

# MT5 retcode for a completed deal.
TRADE_RETCODE_DONE = 10009

#: Terminal-side ceiling on a single history request. Above this the call
#: returns nothing at all rather than a partial series. 50,000 is the common
#: default; raise it in the terminal to get deeper intraday history.
MAX_BARS_PER_REQUEST = 50_000


def _import_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # noqa: N813
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise ExecutionError(
            "MetaTrader5 package is not available. It is Windows-only and requires "
            "a running MT5 terminal. Install with `pip install MetaTrader5`, or use "
            "PaperAdapter for offline work."
        ) from exc
    return mt5


class MT5Adapter:
    name = "mt5"

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        terminal_path: str | None = None,
        deviation_points: int = 20,
        magic: int = 770001,
        aliases: dict[str, str] | None = None,
        clock_symbol: str = "EURUSD",
    ) -> None:
        self._mt5: Any = None
        self._login = login
        self._password = password
        self._server = server
        self._path = terminal_path
        self.deviation = deviation_points
        self.magic = magic
        self._aliases = dict(aliases or {})
        self._spec_cache: dict[str, SymbolSpec] = {}
        #: Symbol used to probe the server clock on connect.
        self._clock_symbol = clock_symbol
        self.clock_status = None
        self.clock_message = ""

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        mt5 = _import_mt5()
        kwargs: dict[str, Any] = {}
        if self._path:
            kwargs["path"] = self._path
        if self._login is not None:
            kwargs.update(login=self._login, password=self._password, server=self._server)

        if not mt5.initialize(**kwargs):
            raise ExecutionError(f"MT5 initialize failed: {mt5.last_error()}")
        self._mt5 = mt5

        # Verify the server timezone before anything reads a timestamp. A broker
        # changing its clock must stop the system, not silently shift a year of data.
        clock = self.broker_symbol(self._clock_symbol)
        mt5.symbol_select(clock, True)
        probe = mt5.symbol_info_tick(clock)
        if probe is not None and probe.time:
            status, message = verify_offset(probe.time)
            if status is ClockCheck.MISMATCH:
                raise ExecutionError(message)
            (log.warning if status is ClockCheck.UNVERIFIABLE else log.info)(message)
            self.clock_status = status
            self.clock_message = message
        else:
            log.warning("could not verify the server clock - no tick available")

        info = mt5.terminal_info()
        if info is not None and not info.trade_allowed:
            log.warning(
                "MT5 terminal reports trade_allowed=False - enable algorithmic "
                "trading in the terminal or every order will be rejected"
            )

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    def is_connected(self) -> bool:
        if self._mt5 is None:
            return False
        return self._mt5.terminal_info() is not None

    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            raise ExecutionError("MT5Adapter is not connected - call connect() first")
        return self._mt5

    def broker_symbol(self, symbol: str) -> str:
        return self._aliases.get(symbol, symbol)

    # -------------------------------------------------------------------- reads

    def account(self) -> AccountState:
        info = self.mt5.account_info()
        if info is None:
            raise ExecutionError(f"account_info failed: {self.mt5.last_error()}")
        return AccountState(
            equity=float(info.equity),
            balance=float(info.balance),
            margin_used=float(info.margin),
            margin_free=float(info.margin_free),
            currency=str(info.currency),
        )

    def spec(self, symbol: str, refresh: bool = False) -> SymbolSpec:
        if not refresh and symbol in self._spec_cache:
            return self._spec_cache[symbol]

        broker_name = self.broker_symbol(symbol)
        info = self.mt5.symbol_info(broker_name)
        if info is None:
            # A symbol absent from Market Watch returns None even when the broker
            # offers it. Select it and retry once before declaring failure.
            if not self.mt5.symbol_select(broker_name, True):
                raise ExecutionError(
                    f"symbol {broker_name!r} not found. Check the alias map in "
                    f"config/instruments.yaml - brokers rename instruments."
                )
            info = self.mt5.symbol_info(broker_name)
        if info is None:
            raise ExecutionError(f"symbol_info({broker_name!r}) returned None")

        spec = SymbolSpec(
            symbol=symbol,
            digits=int(info.digits),
            point=float(info.point),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            contract_size=float(info.trade_contract_size),
            stops_level_points=int(getattr(info, "trade_stops_level", 0)),
            swap_long=float(getattr(info, "swap_long", 0.0)),
            swap_short=float(getattr(info, "swap_short", 0.0)),
            currency_profit=str(getattr(info, "currency_profit", "USD")),
        )
        self._spec_cache[symbol] = spec
        return spec

    def tick(self, symbol: str) -> Tick:
        broker_name = self.broker_symbol(symbol)
        raw = self.mt5.symbol_info_tick(broker_name)
        if raw is None:
            raise ExecutionError(f"no tick for {broker_name!r}: {self.mt5.last_error()}")
        return Tick(
            symbol=symbol,
            ts=server_epoch_to_utc(raw.time),  # server clock -> true UTC
            bid=float(raw.bid),
            ask=float(raw.ask),
        )

    def bars(
        self, symbol: str, timeframe: str, count: int, end: datetime | None = None
    ) -> list[Bar]:
        if timeframe not in TIMEFRAMES:
            raise ExecutionError(f"unknown timeframe {timeframe!r}, expected one of {list(TIMEFRAMES)}")
        tf = getattr(self.mt5, TIMEFRAMES[timeframe])
        broker_name = self.broker_symbol(symbol)

        # Two hard-won constraints, both terminal-side rather than API-side:
        #
        # 1. `copy_rates_range` returns "Invalid params" on some terminals no
        #    matter how the datetimes are built, and `copy_rates_from` rejects
        #    timezone-aware ones. Position-based paging is the only reliable path.
        # 2. Requests are capped by the terminal's "Max bars in chart" setting.
        #    Above it the call returns nothing rather than a truncated series --
        #    a silent failure, so it is probed and reported instead of assumed.
        count = min(count, MAX_BARS_PER_REQUEST)
        rates = self.mt5.copy_rates_from_pos(broker_name, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise ExecutionError(
                f"no bars for {broker_name} {timeframe} (requested {count}): "
                f"{self.mt5.last_error()}. If this is a large request, raise "
                f"Tools > Options > Charts > 'Max bars in chart' in the terminal."
            )

        bars = [
            Bar(
                symbol=symbol,
                ts=server_epoch_to_utc(int(r["time"])),  # server clock -> true UTC
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["tick_volume"]),
            )
            for r in rates
        ]
        if end is not None:
            cutoff = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
            bars = [b for b in bars if b.ts <= cutoff]
            if not bars:
                raise ExecutionError(f"no bars for {broker_name} {timeframe} before {end}")
        return bars

    def positions(self, symbol: str | None = None) -> list[Position]:
        raw = (
            self.mt5.positions_get(symbol=self.broker_symbol(symbol))
            if symbol
            else self.mt5.positions_get()
        )
        if raw is None:
            return []
        reverse = {v: k for k, v in self._aliases.items()}
        out: list[Position] = []
        for p in raw:
            canonical = reverse.get(p.symbol, p.symbol)
            out.append(
                Position(
                    symbol=canonical,
                    side=Side.BUY if p.type == 0 else Side.SELL,
                    volume=float(p.volume),
                    entry_price=float(p.price_open),
                    opened_at=server_epoch_to_utc(p.time),
                    stop_loss=float(p.sl) or None,
                    take_profit=float(p.tp) or None,
                    ticket=int(p.ticket),
                    comment=str(p.comment),
                )
            )
        return out

    # ------------------------------------------------------------------- writes

    def submit(self, request: OrderRequest) -> OrderResult:
        """BLOCKING. Never call this from the event loop - see the module docstring."""
        if request.order_type is not OrderType.MARKET:
            return OrderResult(
                OrderStatus.REJECTED, request, reason="only market orders are implemented"
            )

        mt5 = self.mt5
        spec = self.spec(request.symbol)
        broker_name = self.broker_symbol(request.symbol)
        tick = self.tick(request.symbol)

        volume = spec.round_volume(request.volume)
        if volume < spec.volume_min:
            return OrderResult(
                OrderStatus.REJECTED, request,
                reason=f"volume {request.volume:.4f} rounds below minimum {spec.volume_min:g}",
            )

        price = tick.ask if request.side is Side.BUY else tick.bid
        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_name,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if request.side is Side.BUY else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": request.comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(broker_name),
        }
        if request.stop_loss is not None:
            payload["sl"] = spec.normalize_price(request.stop_loss)
        if request.take_profit is not None:
            payload["tp"] = spec.normalize_price(request.take_profit)

        # order_check catches margin and stop-level problems before the order is
        # ever sent, which keeps rejects out of the broker's log and out of any
        # prop firm's view of your account activity.
        check = mt5.order_check(payload)
        if check is not None and check.retcode not in (0, TRADE_RETCODE_DONE):
            return OrderResult(
                OrderStatus.REJECTED, request,
                reason=f"order_check retcode {check.retcode}: {check.comment}",
                requested_price=price,
            )

        result = mt5.order_send(payload)
        if result is None:
            return OrderResult(
                OrderStatus.REJECTED, request,
                reason=f"order_send returned None: {mt5.last_error()}",
                requested_price=price,
            )
        if result.retcode != TRADE_RETCODE_DONE:
            return OrderResult(
                OrderStatus.REJECTED, request,
                reason=f"retcode {result.retcode}: {result.comment}",
                requested_price=price,
            )

        return OrderResult(
            status=OrderStatus.FILLED,
            request=request,
            ticket=int(result.order),
            fill_price=float(result.price),
            filled_volume=float(result.volume),
            requested_price=price,
        )

    def modify(
        self, ticket: int, stop_loss: float | None = None, take_profit: float | None = None
    ) -> OrderResult:
        mt5 = self.mt5
        current = [p for p in self.positions() if p.ticket == ticket]
        if not current:
            raise ExecutionError(f"no open position with ticket {ticket}")
        pos = current[0]
        spec = self.spec(pos.symbol)

        payload = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.broker_symbol(pos.symbol),
            "sl": spec.normalize_price(stop_loss if stop_loss is not None else (pos.stop_loss or 0.0)),
            "tp": spec.normalize_price(
                take_profit if take_profit is not None else (pos.take_profit or 0.0)
            ),
        }
        request = OrderRequest(pos.symbol, pos.side, pos.volume, comment="modify")
        result = mt5.order_send(payload)
        if result is None or result.retcode != TRADE_RETCODE_DONE:
            reason = f"modify failed: {mt5.last_error() if result is None else result.comment}"
            return OrderResult(OrderStatus.REJECTED, request, ticket=ticket, reason=reason)
        return OrderResult(OrderStatus.FILLED, request, ticket=ticket)

    def close(self, ticket: int, volume: float | None = None) -> OrderResult:
        mt5 = self.mt5
        current = [p for p in self.positions() if p.ticket == ticket]
        if not current:
            raise ExecutionError(f"no open position with ticket {ticket}")
        pos = current[0]
        spec = self.spec(pos.symbol)
        broker_name = self.broker_symbol(pos.symbol)
        tick = self.tick(pos.symbol)

        closing = pos.volume if volume is None else min(spec.round_volume(volume), pos.volume)
        closing_side = pos.side.opposite()
        price = tick.bid if pos.side is Side.BUY else tick.ask

        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_name,
            "volume": closing,
            "type": mt5.ORDER_TYPE_SELL if pos.side is Side.BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(broker_name),
        }
        request = OrderRequest(pos.symbol, closing_side, closing, comment="close")
        result = mt5.order_send(payload)
        if result is None or result.retcode != TRADE_RETCODE_DONE:
            reason = f"close failed: {mt5.last_error() if result is None else result.comment}"
            return OrderResult(OrderStatus.REJECTED, request, ticket=ticket, reason=reason)
        return OrderResult(
            OrderStatus.FILLED, request, ticket=ticket,
            fill_price=float(result.price), filled_volume=float(result.volume),
            requested_price=price,
        )

    # ------------------------------------------------------------------ helpers

    def _filling_mode(self, broker_name: str) -> int:
        """Pick a filling mode the symbol actually accepts.

        Getting this wrong produces retcode 10030 ("unsupported filling mode"),
        which is a common and confusing first failure on a new broker.
        """
        mt5 = self.mt5
        info = mt5.symbol_info(broker_name)
        modes = int(getattr(info, "filling_mode", 0)) if info else 0
        if modes & 2:
            return mt5.ORDER_FILLING_IOC
        if modes & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN
