"""Interactive Brokers futures adapter.

Implements `ExecutionAdapter` for micro futures through `ib_async`. Everything
above this file is unchanged: the same `Signal`, the same risk engine, the same
OMS with its idempotent client ids, the same runner. That boundary is what made
this a new file rather than a rewrite.

Three things are different from the CFD adapter and worth knowing:

- **A symbol is a root, not a contract.** The strategy says "MES"; this file
  resolves it to the live front month (MESZ5, then MESH6) and rolls positions
  before expiry. See `core.contracts`.
- **Stops are separate orders.** IB attaches a child stop to the parent market
  order. A position's stop is found by looking up its child order, not read
  off the position. `positions()` does that join so the runner still sees a
  `Position` with a `stop_loss`.
- **Timestamps are UTC and the exchange's.** IB serves epochs in UTC and there
  is a central limit order book, so the server-clock problem and the bid-bar
  artifact from the CFD side do not exist here. The clock is still verified on
  connect, because trusting it is how the last bug happened.

The IB client is injected. Tests use `execution.ib_fake.FakeIB`; production
passes nothing and `ib_async` is imported. TWS paper trading listens on 7497,
IB Gateway paper on 4002.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any

from core.contracts import MICRO_UNIVERSE, FuturesRoot
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

log = logging.getLogger(__name__)

BAR_SIZE = {"M1": "1 min", "M5": "5 mins", "M15": "15 mins", "M30": "30 mins",
            "H1": "1 hour", "H4": "4 hours", "D1": "1 day", "W1": "1 week"}
BAR_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600,
               "H4": 14400, "D1": 86400, "W1": 604800}


def _import_ib():
    try:
        import ib_async  # noqa: F401
        return ib_async
    except ImportError as exc:  # pragma: no cover
        raise ExecutionError(
            "ib_async is not installed. `pip install ib_async`, and run TWS or IB "
            "Gateway with API access enabled."
        ) from exc


class IBAdapter:
    name = "ib"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 7,
        roots: dict[str, FuturesRoot] | None = None,
        ib: Any = None,
        today: date | None = None,
        fill_timeout: float = 15.0,
    ) -> None:
        self.host, self.port, self.client_id = host, port, client_id
        self.roots = dict(roots or MICRO_UNIVERSE)
        self._ib = ib
        self._today = today
        self.fill_timeout = fill_timeout
        self._contracts: dict[str, Any] = {}
        self._spec_cache: dict[str, SymbolSpec] = {}
        self._orders: dict[int, dict] = {}  # ticket -> {symbol, side, stop_order}
        self.clock_message = ""

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        if self._ib is None:
            ib_async = _import_ib()
            self._ib = ib_async.IB()
        if not self._ib.isConnected():
            self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=20)

        server_now = self._ib.reqCurrentTime()
        if server_now.tzinfo is None:
            server_now = server_now.replace(tzinfo=timezone.utc)
        drift = abs((server_now - datetime.now(timezone.utc)).total_seconds())
        self.clock_message = f"IB server clock drift {drift:.1f}s"
        if drift > 120:
            raise ExecutionError(
                f"IB server time is {drift:.0f}s from local UTC. Fix the machine "
                f"clock before trading; every timestamp depends on it."
            )
        log.info(self.clock_message)

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    @property
    def ib(self) -> Any:
        if self._ib is None or not self._ib.isConnected():
            raise ExecutionError("IBAdapter is not connected - call connect() first")
        return self._ib

    def today(self) -> date:
        return self._today or datetime.now(timezone.utc).date()

    # ---------------------------------------------------------------- contracts

    def root(self, symbol: str) -> FuturesRoot:
        try:
            return self.roots[symbol]
        except KeyError:
            raise ExecutionError(f"{symbol!r} is not a configured futures root") from None

    def front_month(self, symbol: str) -> tuple[int, int]:
        return self.root(symbol).front(self.today())

    def contract(self, symbol: str, month: tuple[int, int] | None = None) -> Any:
        """The qualified IB contract for a root's front (or given) month."""
        r = self.root(symbol)
        year, mon = month or self.front_month(symbol)
        key = f"{symbol}:{r.ib_month(year, mon)}"
        if key in self._contracts:
            return self._contracts[key]
        ib_async = _import_ib() if self._ib.__class__.__name__ != "FakeIB" else None
        if ib_async is not None:
            c = ib_async.Future(symbol=r.root, lastTradeDateOrContractMonth=r.ib_month(year, mon),
                                exchange=r.exchange, currency=r.currency)
        else:
            c = self._ib.make_future(r.root, r.ib_month(year, mon), r.exchange, r.currency)
        qualified = self.ib.qualifyContracts(c)
        if not qualified:
            raise ExecutionError(f"IB could not qualify {key}")
        self._contracts[key] = qualified[0]
        return qualified[0]

    # -------------------------------------------------------------------- reads

    def account(self) -> AccountState:
        rows = {(v.tag): v for v in self.ib.accountSummary()}

        def val(tag: str, default: float = 0.0) -> float:
            v = rows.get(tag)
            return float(v.value) if v is not None else default

        equity = val("NetLiquidation")
        margin = val("MaintMarginReq")
        return AccountState(
            equity=equity, balance=val("TotalCashValue", equity),
            margin_used=margin, margin_free=val("AvailableFunds", max(equity - margin, 0.0)),
            currency=rows["NetLiquidation"].currency if "NetLiquidation" in rows else "USD",
        )

    def spec(self, symbol: str) -> SymbolSpec:
        if symbol in self._spec_cache:
            return self._spec_cache[symbol]
        r = self.root(symbol)
        details = self.ib.reqContractDetails(self.contract(symbol))
        if not details:
            raise ExecutionError(f"no contract details for {symbol}")
        d = details[0]
        min_tick = float(d.minTick)
        multiplier = float(d.contract.multiplier)
        if abs(min_tick - r.tick_size) > 1e-9 or abs(multiplier - r.multiplier) > 1e-9:
            # The exchange's word beats the config. Log loudly and use the exchange.
            log.warning("%s: config says tick %s x %s, exchange says %s x %s - using the exchange",
                        symbol, r.tick_size, r.multiplier, min_tick, multiplier)
        spec = SymbolSpec(
            symbol=symbol, digits=r.digits, point=min_tick, tick_size=min_tick,
            tick_value=min_tick * multiplier, volume_min=1.0, volume_max=10_000.0,
            volume_step=1.0, contract_size=multiplier, stops_level_points=0,
            swap_long=0.0, swap_short=0.0, currency_profit=r.currency, swap_mode=0,
        )
        self._spec_cache[symbol] = spec
        return spec

    def tick(self, symbol: str) -> Tick:
        c = self.contract(symbol)
        t = self.ib.reqMktData(c, "", True, False)
        self.ib.sleep(0.5)
        bid, ask = float(t.bid or 0), float(t.ask or 0)
        if bid <= 0 or ask <= 0:
            last = float(t.last or t.close or 0)
            if last <= 0:
                raise ExecutionError(f"no quote for {symbol}")
            bid = ask = last
        ts = t.time if getattr(t, "time", None) else datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return Tick(symbol=symbol, ts=ts, bid=bid, ask=ask)

    def bars(self, symbol: str, timeframe: str, count: int, end: datetime | None = None) -> list[Bar]:
        if timeframe not in BAR_SIZE:
            raise ExecutionError(f"unknown timeframe {timeframe!r}")
        c = self.contract(symbol)
        seconds = BAR_SECONDS[timeframe] * count
        duration = f"{max(1, seconds // 86400 + 1)} D" if seconds < 86400 * 365 else f"{seconds // (86400 * 365) + 1} Y"
        end_str = "" if end is None else end.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S UTC")
        raw = self.ib.reqHistoricalData(
            c, endDateTime=end_str, durationStr=duration, barSizeSetting=BAR_SIZE[timeframe],
            whatToShow="TRADES", useRTH=False, formatDate=2,
        )
        out: list[Bar] = []
        for b in raw:
            ts = b.date
            if isinstance(ts, date) and not isinstance(ts, datetime):
                ts = datetime(ts.year, ts.month, ts.day, tzinfo=timezone.utc)
            elif ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            out.append(Bar(symbol, ts, float(b.open), float(b.high), float(b.low),
                           float(b.close), float(b.volume)))
        return out[-count:]

    def positions(self, symbol: str | None = None) -> list[Position]:
        stops = self._open_stops()
        out: list[Position] = []
        for p in self.ib.positions():
            root = p.contract.symbol
            if root not in self.roots or (symbol and root != symbol):
                continue
            qty = float(p.position)
            if qty == 0:
                continue
            side = Side.BUY if qty > 0 else Side.SELL
            ticket, stop_px, comment = self._parent_for(root, side)
            out.append(Position(
                symbol=root, side=side, volume=abs(qty), entry_price=float(p.avgCost) / self.root(root).multiplier
                if p.avgCost and self.root(root).multiplier else float(p.avgCost),
                opened_at=datetime.now(timezone.utc), stop_loss=stop_px, ticket=ticket, comment=comment,
            ))
        return out

    # ------------------------------------------------------------------- writes

    def submit(self, request: OrderRequest) -> OrderResult:
        """BLOCKING until fill or timeout. Runs on the OMS worker thread."""
        if request.order_type is not OrderType.MARKET:
            return OrderResult(OrderStatus.REJECTED, request, reason="only market orders are implemented")
        qty = int(round(request.volume))
        if qty < 1:
            return OrderResult(OrderStatus.REJECTED, request, reason="volume below one contract")

        c = self.contract(request.symbol)
        tick = self.tick(request.symbol)
        reference = tick.ask if request.side is Side.BUY else tick.bid
        action = "BUY" if request.side is Side.BUY else "SELL"

        parent = self.ib.make_market_order(action, qty)
        parent.orderRef = request.comment[:31]
        parent.transmit = request.stop_loss is None
        trade = self.ib.placeOrder(c, parent)

        stop_order = None
        if request.stop_loss is not None:
            stop_order = self.ib.make_stop_order("SELL" if action == "BUY" else "BUY", qty, request.stop_loss)
            stop_order.parentId = parent.orderId
            stop_order.orderRef = request.comment[:31]
            stop_order.transmit = True
            self.ib.placeOrder(c, stop_order)

        # Registered NOW, before the fill is confirmed. The first version did
        # this only on a confirmed fill, so a lost reply left the position with
        # no attribution, the OMS could not recognise its own order, and the
        # retry doubled the position - the exact failure idempotency exists for.
        self._orders[parent.orderId] = {"symbol": request.symbol, "side": request.side,
                                        "stop": stop_order, "comment": request.comment}

        deadline = time.time() + self.fill_timeout
        while time.time() < deadline and not trade.isDone():
            self.ib.sleep(0.25)
        status = trade.orderStatus
        if status.status != "Filled":
            reason = f"{status.status}: {getattr(trade, 'log', [''])[-1] if getattr(trade, 'log', None) else 'not filled'}"
            if status.status in ("PendingSubmit", "Submitted", "PreSubmitted"):
                reason = "connection timeout - no fill confirmation"
            return OrderResult(OrderStatus.REJECTED, request, reason=reason, requested_price=reference)

        return OrderResult(
            status=OrderStatus.FILLED, request=request, ticket=parent.orderId,
            fill_price=float(status.avgFillPrice), filled_volume=float(status.filled),
            requested_price=reference,
        )

    def modify(self, ticket: int, stop_loss: float | None = None, take_profit: float | None = None) -> OrderResult:
        meta = self._orders.get(ticket)
        if meta is None or meta["stop"] is None:
            raise ExecutionError(f"no stop order attached to ticket {ticket}")
        req = OrderRequest(meta["symbol"], meta["side"], 0.0, comment="modify")
        if stop_loss is not None:
            meta["stop"].auxPrice = stop_loss
            self.ib.placeOrder(self.contract(meta["symbol"]), meta["stop"])  # re-place = modify
        return OrderResult(OrderStatus.FILLED, req, ticket=ticket)

    def close(self, ticket: int, volume: float | None = None) -> OrderResult:
        meta = self._orders.get(ticket)
        if meta is None:
            raise ExecutionError(f"unknown ticket {ticket}")
        symbol, side = meta["symbol"], meta["side"]
        pos = next((p for p in self.positions(symbol)), None)
        if pos is None:
            raise ExecutionError(f"no open position on {symbol} to close")
        qty = int(round(pos.volume if volume is None else min(volume, pos.volume)))

        if meta["stop"] is not None:
            self.ib.cancelOrder(meta["stop"])
        c = self.contract(symbol)
        tick = self.tick(symbol)
        reference = tick.bid if side is Side.BUY else tick.ask
        order = self.ib.make_market_order("SELL" if side is Side.BUY else "BUY", qty)
        order.orderRef = "close"
        trade = self.ib.placeOrder(c, order)
        deadline = time.time() + self.fill_timeout
        while time.time() < deadline and not trade.isDone():
            self.ib.sleep(0.25)
        req = OrderRequest(symbol, side.opposite(), float(qty), comment="close")
        if trade.orderStatus.status != "Filled":
            return OrderResult(OrderStatus.REJECTED, req, ticket=ticket, reason=trade.orderStatus.status)
        if qty >= pos.volume:
            self._orders.pop(ticket, None)
        return OrderResult(OrderStatus.FILLED, req, ticket=ticket,
                           fill_price=float(trade.orderStatus.avgFillPrice),
                           filled_volume=float(qty), requested_price=reference)

    # --------------------------------------------------------------------- roll

    def roll_due(self, symbol: str) -> bool:
        """True if a position on `symbol` sits in a contract whose roll date has passed."""
        r = self.root(symbol)
        for p in self.ib.positions():
            if p.contract.symbol != r.root or float(p.position) == 0:
                continue
            ym = str(p.contract.lastTradeDateOrContractMonth)[:6]
            year, month = int(ym[:4]), int(ym[4:6])
            if r.roll_date(year, month) <= self.today():
                return True
        return False

    def roll(self, symbol: str) -> list[OrderResult]:
        """Close the expiring contract, reopen the same side and size in the front.

        Two market orders, journaled by the caller. The stop is re-attached at the
        same price on the new contract; the caller may re-size it after.
        """
        r = self.root(symbol)
        results: list[OrderResult] = []
        for p in list(self.ib.positions()):
            if p.contract.symbol != r.root or float(p.position) == 0:
                continue
            ym = str(p.contract.lastTradeDateOrContractMonth)[:6]
            old = (int(ym[:4]), int(ym[4:6]))
            if r.roll_date(*old) > self.today():
                continue
            qty = int(abs(float(p.position)))
            side = Side.BUY if float(p.position) > 0 else Side.SELL
            ticket = next((t for t, m in self._orders.items() if m["symbol"] == symbol), None)
            stop_px = self._orders[ticket]["stop"].auxPrice if ticket is not None and self._orders[ticket]["stop"] else None
            comment = self._orders[ticket]["comment"] if ticket is not None else "roll"

            # Close on the old contract explicitly (not via front resolution).
            old_c = self.contract(symbol, old)
            if ticket is not None and self._orders[ticket]["stop"] is not None:
                self.ib.cancelOrder(self._orders[ticket]["stop"])
            closing = self.ib.make_market_order("SELL" if side is Side.BUY else "BUY", qty)
            closing.orderRef = "roll-close"
            t1 = self.ib.placeOrder(old_c, closing)
            self.ib.sleep(0.25)
            results.append(OrderResult(OrderStatus.FILLED if t1.orderStatus.status == "Filled" else OrderStatus.REJECTED,
                                       OrderRequest(symbol, side.opposite(), qty, comment="roll-close"),
                                       ticket=ticket, fill_price=float(t1.orderStatus.avgFillPrice or 0)))
            self._orders.pop(ticket, None)

            # Reopen in the front month through the normal path so the stop is attached.
            reopened = self.submit(OrderRequest(symbol, side, qty, stop_loss=stop_px, comment=comment))
            results.append(reopened)
        return results

    # ------------------------------------------------------------------ helpers

    def _open_stops(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in self.ib.openTrades():
            o = t.order
            if getattr(o, "orderType", "") == "STP" and getattr(o, "parentId", 0):
                out[int(o.parentId)] = float(o.auxPrice)
        return out

    def _parent_for(self, root: str, side: Side) -> tuple[int | None, float | None, str]:
        stops = self._open_stops()
        for ticket, meta in self._orders.items():
            if meta["symbol"] == root and meta["side"] is side:
                stop_px = stops.get(ticket)
                if stop_px is None and meta["stop"] is not None:
                    stop_px = float(meta["stop"].auxPrice)
                return ticket, stop_px, meta["comment"]

        # Local memory has nothing - a restart, or a reply that never arrived.
        # The broker still has the order and its orderRef (our client id).
        want = "BUY" if side is Side.BUY else "SELL"
        for t in reversed(list(self.ib.trades())):
            o = t.order
            if (t.contract.symbol == root and getattr(o, "orderType", "") == "MKT"
                    and o.action == want and t.orderStatus.status == "Filled"
                    and getattr(o, "orderRef", "")):
                ticket = int(o.orderId)
                self._orders.setdefault(ticket, {"symbol": root, "side": side, "stop": None,
                                                 "comment": o.orderRef})
                return ticket, stops.get(ticket), o.orderRef
        return None, None, ""
