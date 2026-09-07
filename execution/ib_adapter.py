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
        #: Which entitlement actually answered: 1 live, 3 delayed, 4 delayed
        #: frozen. None until the first quote. Anything but 1 means prices are
        #: 10-15 minutes old and may not be used to measure execution quality.
        self.market_data_type: int | None = None
        #: (contract, timeframe, count, end) -> (fetched_at, bars). See _history:
        #: this is a rate limit, not a speed optimisation.
        self._bar_cache: dict[tuple, tuple[float, list]] = {}
        self.bar_cache_seconds = 600.0

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
        # Symbol AND multiplier. IB lists Micro Silver as "SI", the same symbol
        # as the 5,000oz contract, and separates the two only by multiplier;
        # sending the size is what makes a shared symbol unambiguous.
        c = _import_ib().Future(symbol=r.ib_symbol, lastTradeDateOrContractMonth=r.ib_month(year, mon),
                                exchange=r.exchange, currency=r.currency, multiplier=r.ib_multiplier)
        qualified = self.ib.qualifyContracts(c)
        qualified = [q for q in qualified if q is not None]
        if not qualified:
            raise ExecutionError(
                f"IB could not qualify {key} (sent symbol={r.ib_symbol!r} "
                f"multiplier={r.ib_multiplier!r} on {r.exchange}). The contract may not "
                f"list that month, or the broker uses a different symbol for this root."
            )
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

    #: IB market data types. 1 is the live subscription; 3 is the delayed feed
    #: every account gets for free; 4 is the last delayed value when the market
    #: is shut. A free-trial account has no live entitlement and returns NaN on
    #: type 1 with error 354, which is a subscription problem wearing the
    #: costume of a broken quote.
    LIVE, DELAYED, DELAYED_FROZEN = 1, 3, 4

    @staticmethod
    def market_order(action: str, qty: int):
        """`ib_async.MarketOrder`, explicitly routed and timed.

        Built here, not on the client, because the client has no such factory -
        an earlier version called `ib.make_market_order`, which existed only on
        the test double and blew up the first time it met the real library.

        `outsideRth=True` because a CME equity-index future trades nearly around
        the clock while IB's "regular trading hours" are 09:30-16:15 New York.
        Left at the default of False, every order this system sends outside that
        window is refused in a market that is plainly open - which is what
        happened at 04:35 New York on 2026-09-07, error 10349.
        """
        return _import_ib().MarketOrder(action, qty, tif="DAY", outsideRth=True)

    @staticmethod
    def stop_order(action: str, qty: int, stop_price: float):
        """A protective stop, GOOD TILL CANCELLED and valid outside regular hours.

        This is a risk control, not a trading preference. `ib_async` leaves
        `tif` empty and IB's order preset fills in DAY, so the stop would be
        **cancelled at the end of the session**. Every strategy here holds
        overnight - the monthly rules hold for a month - so the position would
        wake up unprotected while the risk engine still believed a stop was
        attached, and every aggregate risk number computed from stops would be
        quietly wrong. `UnstoppedPosition` would not catch it either: the
        position did have a stop when it was last checked.
        """
        return _import_ib().StopOrder(action, qty, stop_price, tif="GTC", outsideRth=True)

    def _quote_once(self, contract, data_type: int, wait: float):
        self.ib.reqMarketDataType(data_type)
        t = self.ib.reqMktData(contract, "", True, False)
        self.ib.sleep(wait)
        bid, ask = float(t.bid or 0), float(t.ask or 0)
        if bid > 0 and ask > 0:
            return bid, ask, t
        last = float(t.last or t.close or 0)
        if last > 0:
            return last, last, t  # delayed feeds often carry only a last price
        return 0.0, 0.0, t

    def _best_quote(self, contract, label: str) -> tuple[float, float]:
        """Bid and ask for one specific contract, down the entitlement ladder.

        Live first, because a real subscription is what production uses. On an
        account without one, IB answers with NaN rather than an error the client
        can catch, so the fallback is by value, not by exception.
        """
        for data_type, wait in ((self.LIVE, 0.5), (self.DELAYED, 2.0), (self.DELAYED_FROZEN, 2.0)):
            if self.market_data_type is not None and data_type < self.market_data_type:
                continue  # a previous call already established there is no entitlement
            bid, ask, _ = self._quote_once(contract, data_type, wait)
            if bid > 0 and ask > 0:
                if self.market_data_type != data_type:
                    self.market_data_type = data_type
                    log.info("%s: market data type %d (%s)", label, data_type,
                             {1: "live", 3: "delayed", 4: "delayed frozen"}[data_type])
                return bid, ask
        raise ExecutionError(
            f"no quote for {label} on live, delayed or frozen data. The contract may "
            f"not be trading, or the account has no entitlement at all."
        )

    def tick(self, symbol: str) -> Tick:
        """Best quote for the front contract. `market_data_type` records which
        entitlement answered; anything but live means the price is 10-15 minutes
        old and must not be used to measure execution quality."""
        bid, ask = self._best_quote(self.contract(symbol), symbol)
        return Tick(symbol=symbol, ts=datetime.now(timezone.utc), bid=bid, ask=ask)

    def _await_fill(self, trade) -> bool:
        """Block until the trade is done or `fill_timeout` expires.

        Used by every order path. `roll()` used to sleep a quarter of a second
        and judge, so a fill that arrived a moment later was reported REJECTED -
        which on 2026-09-07 marked a roll as failed after it had actually
        worked. IB may also park a trade in `ValidationError` while it processes
        a warning (2109, outside-RTH ignored for this order type); that is not
        terminal and `isDone()` correctly keeps waiting through it.
        """
        deadline = time.time() + self.fill_timeout
        while time.time() < deadline and not trade.isDone():
            self.ib.sleep(0.25)
        return trade.orderStatus.status == "Filled"

    def _history(self, contract, timeframe: str, count: int, end: datetime | None, symbol: str) -> list[Bar]:
        """Raw history for one specific contract, cached briefly.

        The cache is not an optimisation, it is a rate limit. IB allows sixty
        historical requests per ten minutes. A four-sleeve book on thirteen
        markets asks for the same thirteen series fifty-two times per poll, and
        carry doubles that again by needing the next delivery month too. Without
        this, the book pages IB out of its own quota within one tick.
        """
        key = (getattr(contract, "conId", None) or id(contract), timeframe, count,
               end.isoformat() if end else "")
        hit = self._bar_cache.get(key)
        now = time.time()
        if hit is not None and now - hit[0] < self.bar_cache_seconds:
            return hit[1]

        seconds = BAR_SECONDS[timeframe] * count
        duration = f"{max(1, seconds // 86400 + 1)} D" if seconds < 86400 * 365 else f"{seconds // (86400 * 365) + 1} Y"
        end_str = "" if end is None else end.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S UTC")
        raw = self.ib.reqHistoricalData(
            contract, endDateTime=end_str, durationStr=duration, barSizeSetting=BAR_SIZE[timeframe],
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
        out = out[-count:]
        self._bar_cache[key] = (now, out)
        return out

    def carry_series(self, symbol: str, count: int, timeframe: str = "D1") -> dict[datetime, float]:
        """Annualised roll yield per day: (front - next) / front, scaled by the
        days between the two expiries.

        Positive is backwardation, which pays a long as the contract rolls up
        toward spot. This is the same quantity `data/continuous.stitch` writes
        during a backtest, computed here from two live histories instead, so the
        carry rule sees the identical number live and in research.

        Empty when the next month has no history of its own - a market with one
        listed contract has no curve, and inventing a number for it would be
        worse than reading flat.
        """
        r = self.root(symbol)
        front = self.front_month(symbol)
        try:
            nxt = r.next_after(*front)
        except (ValueError, IndexError):
            return {}
        days = (r.last_trade(*nxt) - r.last_trade(*front)).days
        if days <= 0:
            return {}
        try:
            f_bars = self._history(self.contract(symbol, front), timeframe, count, None, symbol)
            n_bars = self._history(self.contract(symbol, nxt), timeframe, count, None, symbol)
        except Exception as exc:  # noqa: BLE001 - a deferred month may not trade yet
            log.warning("%s: no carry series (%s: %s)", symbol, type(exc).__name__, exc)
            return {}
        next_by_day = {b.ts.date(): b.close for b in n_bars}
        out: dict[datetime, float] = {}
        for b in f_bars:
            nc = next_by_day.get(b.ts.date())
            if nc is None or b.close == 0:
                continue
            out[b.ts] = (b.close - nc) / abs(b.close) * (365.0 / days)
        return out

    def bar_extras(self, symbol: str, timeframe: str, count: int,
                   end: datetime | None = None) -> dict[str, list]:
        """Columns beyond OHLCV that a strategy may need, aligned to `bars`.

        `carry` and `raw_close` are what the carry rule reads. The runner merges
        whatever this returns, so a venue with no curve simply returns nothing
        and the rule reads flat rather than erroring.
        """
        if timeframe != "D1":
            return {}
        bars = self.bars(symbol, timeframe, count, end)
        carry = self.carry_series(symbol, count, timeframe)
        if not carry:
            return {}
        return {
            "carry": [carry.get(b.ts, float("nan")) for b in bars],
            "raw_close": [b.close for b in bars],
        }

    def bars(self, symbol: str, timeframe: str, count: int, end: datetime | None = None) -> list[Bar]:
        if timeframe not in BAR_SIZE:
            raise ExecutionError(f"unknown timeframe {timeframe!r}")
        return self._history(self.contract(symbol), timeframe, count, end, symbol)

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

        parent = self.market_order(action, qty)
        parent.orderRef = request.comment[:31]
        parent.transmit = request.stop_loss is None
        trade = self.ib.placeOrder(c, parent)

        stop_order = None
        if request.stop_loss is not None:
            stop_order = self.stop_order("SELL" if action == "BUY" else "BUY", qty, request.stop_loss)
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

        self._await_fill(trade)
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
        order = self.market_order("SELL" if side is Side.BUY else "BUY", qty)
        order.orderRef = "close"
        trade = self.ib.placeOrder(c, order)
        self._await_fill(trade)
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

    def roll_basis(self, symbol: str, old: tuple[int, int]) -> float:
        """New contract's price minus the expiring one's, at this moment.

        Two delivery months of the same future are not the same price. On
        2026-09-07 the September S&P micro traded at 7722 and December at 7785:
        sixty-three points of carry between them.
        """
        old_bid, old_ask = self._best_quote(self.contract(symbol, old), f"{symbol} {old[0]}{old[1]:02d}")
        new_bid, new_ask = self._best_quote(self.contract(symbol), symbol)
        return (new_bid + new_ask) / 2 - (old_bid + old_ask) / 2

    def roll(self, symbol: str) -> list[OrderResult]:
        """Close the expiring contract, reopen the same side and size in the front.

        Two market orders, journaled by the caller.

        **The stop is shifted by the basis, not carried across at its face
        value.** Two delivery months trade at different prices, so reattaching
        the old stop price to the new contract changes the risk by the whole
        carry. Measured live on 2026-09-07: a long in the September S&P micro at
        7722 with its stop at 7714.75, seven points away, rolled into December
        at 7785 and kept a stop at 7714.75 - seventy points away, nine times the
        intended risk. A short would have been worse: the stop would have landed
        on the far side of the market and liquidated the position on arrival.
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
            # Measure the basis BEFORE closing, while both contracts still quote.
            new_stop = None
            if stop_px is not None:
                basis = self.roll_basis(symbol, old)
                spec = self.spec(symbol)
                new_stop = spec.normalize_price(stop_px + basis)
                log.info("%s roll: basis %+.4f, stop %s -> %s", symbol, basis, stop_px, new_stop)

            closing = self.market_order("SELL" if side is Side.BUY else "BUY", qty)
            closing.orderRef = "roll-close"
            t1 = self.ib.placeOrder(old_c, closing)
            filled = self._await_fill(t1)
            results.append(OrderResult(OrderStatus.FILLED if filled else OrderStatus.REJECTED,
                                       OrderRequest(symbol, side.opposite(), qty, comment="roll-close"),
                                       ticket=ticket, fill_price=float(t1.orderStatus.avgFillPrice or 0),
                                       reason="" if filled else t1.orderStatus.status))
            self._orders.pop(ticket, None)
            if not filled:
                # The old contract is still open. Reopening now would double the
                # position in an expiring month, which is the worst of both.
                log.error("%s roll: close leg did not fill (%s); not reopening", symbol, t1.orderStatus.status)
                return results

            # Reopen in the front month through the normal path so the stop is attached.
            reopened = self.submit(OrderRequest(symbol, side, qty, stop_loss=new_stop, comment=comment))
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
