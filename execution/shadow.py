"""Shadow execution: a live venue for reads, a paper book for writes.

Prices, bars, specs and the account come from the real adapter. Orders never
leave the process. The split is the whole point: everything upstream of
`submit` is the production code path, exercised against real market state,
while nothing reaches the broker.

**The contract-roll problem, and why this file is not a two-line wrapper.**

A futures root is not one price series. "MES" is the September contract until
it rolls, then the December one, and the two trade at different prices - sixty
three points apart on 2026-09-07. A paper book that simply marks a position to
whatever the front month currently quotes would record a phantom gain or loss
of exactly that basis on every roll. Over a year of quarterly rolls that is
four fictitious trades per market, all in the same direction as the term
structure, which is the same error the back-adjusted series exists to remove
from backtests.

So when the front month changes under an open position, this shifts the paper
position's entry price by the measured basis. The position's unrealised P&L is
then continuous across the roll, which is what actually happened to the trader:
they closed one contract and opened another at a different price, ending up
economically where they started, minus two spreads. The roll's real cost is
charged separately as commission and spread, exactly as `roll_cost_cash` models
it in the backtest.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from core.types import Bar, SymbolSpec, Tick
from execution.paper import PaperAdapter, PaperConfig

log = logging.getLogger(__name__)


class ShadowAdapter(PaperAdapter):
    """Live reads, paper writes, with futures rolls handled honestly."""

    name = "shadow"

    def __init__(self, live: Any, specs: dict[str, SymbolSpec], config: PaperConfig | None = None,
                 charge_roll_cost: bool = True) -> None:
        super().__init__(specs, config or PaperConfig(starting_balance=live.account().equity))
        self._live = live
        self.charge_roll_cost = charge_roll_cost
        #: symbol -> the front month we last saw, so a change can be detected
        self._front: dict[str, tuple[int, int]] = {}
        self.rolls: list[dict] = []

    # ------------------------------------------------------------------ reads

    def tick(self, symbol: str) -> Tick:
        self._check_roll(symbol)
        t = self._live.tick(symbol)
        self.feed_tick(t)  # keep the paper book marked to real prices
        return t

    def bars(self, symbol: str, timeframe: str, count: int, end: datetime | None = None) -> list[Bar]:
        return self._live.bars(symbol, timeframe, count, end)

    def bar_extras(self, symbol: str, timeframe: str, count: int,
                   end: datetime | None = None) -> dict[str, list]:
        """Relay the venue's extra columns, so a rule reading the curve behaves
        identically whether fills are shadowed or real."""
        fn = getattr(self._live, "bar_extras", None)
        return fn(symbol, timeframe, count, end) if fn is not None else {}

    def spec(self, symbol: str) -> SymbolSpec:
        return self._live.spec(symbol)

    @property
    def market_data_type(self):
        """Whichever entitlement the live adapter is actually using, so a caller
        can tell a live quote from a delayed one."""
        return getattr(self._live, "market_data_type", None)

    # ------------------------------------------------------------------- roll

    def _check_roll(self, symbol: str) -> None:
        """Detect a front-month change and re-base any open paper position.

        Without this the position would appear to gain or lose the basis at
        every roll, which is fiction. See the module docstring.
        """
        front_month = getattr(self._live, "front_month", None)
        if front_month is None:
            return  # a venue without delivery months: nothing to roll
        try:
            now_front = front_month(symbol)
        except Exception:  # noqa: BLE001 - a symbol this venue does not know
            return

        was = self._front.get(symbol)
        self._front[symbol] = now_front
        if was is None or was == now_front:
            return

        held = [(t, p) for t, p in self._positions.items() if p.symbol == symbol]
        if not held:
            log.info("%s: front month %s -> %s, no position to roll", symbol, was, now_front)
            return

        basis = self._roll_basis(symbol, was)
        if basis is None:
            log.error("%s: front month changed but the basis could not be measured; "
                      "position P&L across this roll is NOT trustworthy", symbol)
            return

        spec = self.spec(symbol)
        charged = 0.0
        for ticket, pos in held:
            shifted = replace(pos, entry_price=pos.entry_price + basis,
                              stop_loss=(pos.stop_loss + basis) if pos.stop_loss is not None else None)
            self._positions[ticket] = shifted
            if self.charge_roll_cost:
                charged += self._roll_cost(symbol, spec, pos.volume)
            log.info("%s roll %s -> %s: basis %+.4f, entry %s -> %s, stop %s -> %s",
                     symbol, was, now_front, basis, pos.entry_price, shifted.entry_price,
                     pos.stop_loss, shifted.stop_loss)
        if charged:
            self.balance -= charged
            self.realized_pnl -= charged
        self.rolls.append({
            "ts": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "from": f"{was[0]}{was[1]:02d}", "to": f"{now_front[0]}{now_front[1]:02d}",
            "basis": basis, "positions": len(held), "cost": charged,
        })

    def _roll_cost(self, symbol: str, spec: SymbolSpec, volume: float) -> float:
        """What leaving one contract and entering the next actually costs:
        two commissions and two spreads. Free rolls are the second way a paper
        book flatters a futures strategy, after the phantom basis P&L."""
        roots = getattr(self._live, "roots", {})
        root = roots.get(symbol)
        commission = 2 * float(getattr(root, "commission_per_side", 1.0)) * volume
        spread_cash = 2 * spec.tick_value * volume  # one tick each way, the micro's normal spread
        return commission + spread_cash

    def _roll_basis(self, symbol: str, old: tuple[int, int]) -> float | None:
        fn = getattr(self._live, "roll_basis", None)
        if fn is None:
            return None
        try:
            return float(fn(symbol, old))
        except Exception as exc:  # noqa: BLE001
            log.error("%s: roll basis failed: %s", symbol, exc)
            return None
