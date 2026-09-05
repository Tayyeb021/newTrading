"""Transaction costs.

The most important module in the backtest, and the one most often waved through.
The documented decay is severe: a strategy showing Sharpe 3.0 frictionless can
land at 0.5 or below once real costs apply. A backtest with an optimistic cost
model is not a slightly wrong backtest; it is a different experiment.

Three components, all charged against you:

- **spread** — you buy at ask, sell at bid, always
- **commission** — per lot per side on raw-spread accounts
- **slippage** — the gap between the price you asked for and the one you got

Defaults here are deliberately pessimistic placeholders. Replace them with
measured values from `scripts/verify_roundtrip.py` output as soon as you have
them; `CostModel.calibrate()` takes those fills directly. Until then, every
result carries `calibrated=False` and the report says so, because an uncalibrated
cost model is a guess wearing a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import median

from core.types import SymbolSpec


@dataclass(frozen=True)
class SymbolCosts:
    """Costs for one instrument, in price units unless stated otherwise."""

    spread: float  # typical half-to-half spread
    commission_per_lot_per_side: float = 0.0  # account currency
    slippage: float = 0.0  # additional adverse price movement per fill
    spread_multiple_at_open: float = 1.0  # widening at session open / news
    #: Overnight financing per lot per night, in account currency, signed as the
    #: broker reports it (negative = you pay). Ignoring this is not a rounding
    #: error: XAUUSDc charges -512.30 cents per lot per night on longs, which is
    #: more than the average M5 range moves. A backtest without it hands every
    #: overnight position free leverage.
    swap_long: float = 0.0
    swap_short: float = 0.0
    triple_swap_weekday: int = 2  # Wednesday, where 3 days are charged at once

    def entry_cost_price(self, at_session_open: bool = False) -> float:
        """Adverse price movement on one fill: half-spread plus slippage."""
        spread = self.spread * (self.spread_multiple_at_open if at_session_open else 1.0)
        return spread / 2.0 + self.slippage

    def round_trip_price(self, at_session_open: bool = False) -> float:
        return 2.0 * self.entry_cost_price(at_session_open)


# Pessimistic placeholders. Retail CFD, not raw-spread institutional pricing.
DEFAULT_COSTS: dict[str, SymbolCosts] = {
    "EURUSD": SymbolCosts(spread=0.00012, slippage=0.00003, spread_multiple_at_open=1.5),
    "GBPUSD": SymbolCosts(spread=0.00018, slippage=0.00004, spread_multiple_at_open=1.5),
    "USDJPY": SymbolCosts(spread=0.014, slippage=0.004, spread_multiple_at_open=1.5),
    "XAUUSD": SymbolCosts(spread=0.28, slippage=0.08, spread_multiple_at_open=2.5),
    "US30": SymbolCosts(spread=3.5, slippage=1.2, spread_multiple_at_open=3.0),
    "US500": SymbolCosts(spread=0.55, slippage=0.20, spread_multiple_at_open=3.0),
}


@dataclass
class CostModel:
    costs: dict[str, SymbolCosts] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    calibrated: bool = False
    #: Multiplier applied to every cost. Set to 2.0 for the sensitivity run that
    #: the validation gauntlet requires — if the edge dies at 2x, it was never an
    #: edge, it was a rounding error in this file.
    stress: float = 1.0

    def for_symbol(self, symbol: str) -> SymbolCosts:
        try:
            base = self.costs[symbol]
        except KeyError:
            raise KeyError(
                f"no cost model for {symbol!r}. Add one to DEFAULT_COSTS or "
                f"calibrate from measured fills - never let a symbol trade for free."
            ) from None
        if self.stress == 1.0:
            return base
        return replace(
            base,
            spread=base.spread * self.stress,
            slippage=base.slippage * self.stress,
            commission_per_lot_per_side=base.commission_per_lot_per_side * self.stress,
        )  # swap is scaled in swap_cash, not here, to avoid double-counting

    def entry_price(
        self, symbol: str, reference: float, sign: int, at_session_open: bool = False
    ) -> float:
        """Reference price moved against you. `sign` is +1 to buy, -1 to sell."""
        return reference + self.for_symbol(symbol).entry_cost_price(at_session_open) * sign

    def commission(self, symbol: str, volume: float) -> float:
        """One side, in account currency."""
        return self.for_symbol(symbol).commission_per_lot_per_side * volume

    def swap_cash(
        self,
        symbol: str,
        volume: float,
        entry_ts,
        exit_ts,
        is_long: bool,
        spec: SymbolSpec | None = None,
        price: float = 0.0,
    ) -> float:
        """Financing charged between two timestamps. Returns a positive COST.

        Counts each 21:00-UTC rollover crossed, with a triple charge on the
        broker's 3-day weekday (Wednesday for FX, Friday for index CFDs here).
        Returns 0.0 for an intraday trade, which is exactly why flattening
        before rollover is worth enforcing.

        When a spec is supplied the broker's swap_mode is honoured - points,
        currency, or annual percent - instead of assuming currency per lot.
        """
        import pandas as pd

        c = self.for_symbol(symbol)
        if spec is not None:
            rate = spec.swap_cash_per_lot_night(is_long, price)
            triple_day = spec.swap_triple_weekday
        else:
            rate = c.swap_long if is_long else c.swap_short
            triple_day = c.triple_swap_weekday
        if rate == 0.0:
            return 0.0

        start = pd.Timestamp(entry_ts).tz_convert("UTC")
        end = pd.Timestamp(exit_ts).tz_convert("UTC")
        rollover = start.normalize() + pd.Timedelta(hours=21)
        if rollover <= start:
            rollover += pd.Timedelta(days=1)

        nights = 0
        while rollover <= end:
            nights += 3 if rollover.weekday() == triple_day else 1
            rollover += pd.Timedelta(days=1)

        return -rate * volume * nights * self.stress

    def round_trip_cash(self, symbol: str, spec: SymbolSpec, volume: float) -> float:
        """Total cost of opening and closing `volume` lots, in account currency."""
        c = self.for_symbol(symbol)
        price_cost = c.round_trip_price() * volume * spec.value_per_price_unit
        return price_cost + 2.0 * self.commission(symbol, volume)

    def edge_ratio(self, symbol: str, spec: SymbolSpec, expected_move: float) -> float:
        """Expected gross move divided by round-trip cost, both in price units.

        The number from the strategy research that kills most ideas before any
        code is written: below 3 the strategy is dead, 3-5 is fragile, above 5 is
        workable. Compute it before building, not after.
        """
        cost = self.for_symbol(symbol).round_trip_price()
        return expected_move / cost if cost > 0 else float("inf")

    # ------------------------------------------------------------- calibration

    def calibrate(self, symbol: str, fills: list[dict]) -> SymbolCosts:
        """Fit costs from measured round trips.

        Each fill: {"spread": float, "slippage": float} in price units, taken
        straight from `verify_roundtrip.py` output. Uses the median rather than
        the mean — one bad news fill should inform your spread guard, not your
        baseline cost assumption.
        """
        if not fills:
            raise ValueError("calibration needs at least one measured fill")

        spreads = [f["spread"] for f in fills if f.get("spread") is not None]
        slips = [abs(f["slippage"]) for f in fills if f.get("slippage") is not None]
        base = self.costs.get(symbol, SymbolCosts(spread=0.0))

        fitted = replace(
            base,
            spread=median(spreads) if spreads else base.spread,
            slippage=median(slips) if slips else base.slippage,
        )
        self.costs[symbol] = fitted
        self.calibrated = True
        return fitted

    def stressed(self, factor: float) -> "CostModel":
        return CostModel(costs=dict(self.costs), calibrated=self.calibrated, stress=factor)

    def summary(self, specs: dict[str, SymbolSpec] | None = None) -> str:
        lines = [
            f"cost model ({'calibrated' if self.calibrated else 'UNCALIBRATED - placeholder values'}"
            + (f", stress x{self.stress:g}" if self.stress != 1.0 else "")
            + ")"
        ]
        header = f"  {'symbol':<10}{'spread':>12}{'slippage':>12}{'round trip':>13}"
        if specs:
            header += f"{'$/rt/lot':>12}"
        lines.append(header)
        for symbol in sorted(self.costs):
            c = self.for_symbol(symbol)
            row = (
                f"  {symbol:<10}{c.spread:>12.6g}{c.slippage:>12.6g}"
                f"{c.round_trip_price():>13.6g}"
            )
            if specs and symbol in specs:
                row += f"{self.round_trip_cash(symbol, specs[symbol], 1.0):>12,.2f}"
            lines.append(row)
        return "\n".join(lines)
