"""A stand-in for the ib_async client, for tests and for dry runs without TWS.

It implements exactly the subset of the IB client that `IBAdapter` touches, with
deterministic fills at a settable price and a settable clock. Like the paper
adapter, it is pessimistic: market orders fill through the spread.

Nothing here talks to a network. If a test passes against this and fails
against TWS, the difference is IB's behaviour, which is the useful thing to
learn.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class _Contract:
    symbol: str
    lastTradeDateOrContractMonth: str
    exchange: str
    currency: str
    multiplier: str = ""
    conId: int = 0


@dataclass
class _Details:
    contract: _Contract
    minTick: float


@dataclass
class _Order:
    action: str
    totalQuantity: int
    orderType: str
    orderId: int
    auxPrice: float = 0.0
    parentId: int = 0
    transmit: bool = True
    orderRef: str = ""


@dataclass
class _Status:
    status: str = "Submitted"
    filled: float = 0.0
    avgFillPrice: float = 0.0


@dataclass
class _Trade:
    contract: _Contract
    order: _Order
    orderStatus: _Status = field(default_factory=_Status)

    def isDone(self) -> bool:
        return self.orderStatus.status in ("Filled", "Cancelled")


@dataclass
class _Position:
    contract: _Contract
    position: float
    avgCost: float


@dataclass
class _Summary:
    tag: str
    value: str
    currency: str = "USD"


@dataclass
class _Ticker:
    bid: float
    ask: float
    last: float
    close: float
    time: datetime


@dataclass
class _HistBar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class FakeIB:
    """Deterministic IB client double."""

    def __init__(self, roots: dict, prices: dict[str, float], spread_ticks: float = 1.0,
                 equity: float = 25_000.0, now: datetime | None = None) -> None:
        self.roots = roots
        self.prices = dict(prices)
        self.spread_ticks = spread_ticks
        self.equity = equity
        self.now = now or datetime.now(timezone.utc)
        self._connected = False
        self._ids = itertools.count(100)
        self._trades: list[_Trade] = []
        self._positions: dict[str, _Position] = {}  # key: symbol:month
        self.history: dict[str, list[_HistBar]] = {}
        self.fail_next_fill = False  # simulate a lost reply

    # --------------------------------------------------------------- session

    def connect(self, host, port, clientId, timeout=20) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def isConnected(self) -> bool:
        return self._connected

    def reqCurrentTime(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        pass

    # ------------------------------------------------------------- contracts

    def make_future(self, root, month, exchange, currency) -> _Contract:
        r = self.roots[root]
        return _Contract(root, month, exchange, currency, multiplier=str(r.multiplier), conId=hash((root, month)) & 0xFFFF)

    def qualifyContracts(self, c: _Contract) -> list[_Contract]:
        return [c]

    def reqContractDetails(self, c: _Contract) -> list[_Details]:
        return [_Details(c, self.roots[c.symbol].tick_size)]

    # --------------------------------------------------------------- account

    def accountSummary(self) -> list[_Summary]:
        margin = sum(abs(p.position) * self.roots[p.contract.symbol].margin_day for p in self._positions.values())
        return [
            _Summary("NetLiquidation", f"{self.equity:.2f}"),
            _Summary("TotalCashValue", f"{self.equity:.2f}"),
            _Summary("MaintMarginReq", f"{margin:.2f}"),
            _Summary("AvailableFunds", f"{self.equity - margin:.2f}"),
        ]

    # ---------------------------------------------------------------- market

    def _quote(self, symbol: str) -> tuple[float, float]:
        mid = self.prices[symbol]
        half = self.roots[symbol].tick_size * self.spread_ticks / 2
        return mid - half, mid + half

    def reqMktData(self, c: _Contract, generic="", snapshot=True, regulatory=False) -> _Ticker:
        bid, ask = self._quote(c.symbol)
        return _Ticker(bid, ask, self.prices[c.symbol], self.prices[c.symbol], self.now)

    def reqHistoricalData(self, c, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate=2):
        return list(self.history.get(c.symbol, []))

    # ---------------------------------------------------------------- orders

    def make_market_order(self, action: str, qty: int) -> _Order:
        return _Order(action, qty, "MKT", next(self._ids))

    def make_stop_order(self, action: str, qty: int, stop: float) -> _Order:
        return _Order(action, qty, "STP", next(self._ids), auxPrice=stop)

    def placeOrder(self, c: _Contract, o: _Order) -> _Trade:
        existing = next((t for t in self._trades if t.order.orderId == o.orderId), None)
        if existing is not None:
            existing.order = o  # modify
            return existing
        trade = _Trade(c, o)
        self._trades.append(trade)
        if o.orderType == "MKT":
            if self.fail_next_fill:
                self.fail_next_fill = False
                # fill it but leave the status looking unconfirmed - the lost reply
                self._fill(trade, record=True)
                trade.orderStatus.status = "Submitted"
            else:
                self._fill(trade, record=True)
        return trade

    def _fill(self, trade: _Trade, record: bool) -> None:
        bid, ask = self._quote(trade.contract.symbol)
        px = ask if trade.order.action == "BUY" else bid
        trade.orderStatus = _Status("Filled", trade.order.totalQuantity, px)
        if record:
            key = f"{trade.contract.symbol}:{trade.contract.lastTradeDateOrContractMonth}"
            signed = trade.order.totalQuantity * (1 if trade.order.action == "BUY" else -1)
            mult = self.roots[trade.contract.symbol].multiplier
            if key in self._positions:
                p = self._positions[key]
                new = p.position + signed
                if new == 0:
                    del self._positions[key]
                else:
                    p.position = new
            else:
                self._positions[key] = _Position(trade.contract, signed, px * mult)

    def cancelOrder(self, o: _Order) -> None:
        for t in self._trades:
            if t.order.orderId == o.orderId:
                t.orderStatus.status = "Cancelled"

    def trades(self) -> list[_Trade]:
        return list(self._trades)

    def openTrades(self) -> list[_Trade]:
        return [t for t in self._trades if t.order.orderType == "STP" and t.orderStatus.status not in ("Cancelled", "Filled")]

    def positions(self) -> list[_Position]:
        return list(self._positions.values())
