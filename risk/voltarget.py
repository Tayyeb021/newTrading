"""Portfolio-level volatility targeting. Research entry 011.

The third way to size a book, and the one the industry actually uses.

- Risk per trade (everything before 010): each position risks a fixed
  fraction of equity to its stop. Simple, honest, and the book's volatility
  then depends on how many signals happen to be on: three positions is a
  quiet book, thirty is a wild one. The 47% drawdown in 010 is that.
- Position-level resizing (008): each position chases its own target every
  day. On a noisy forecast that is churn, and it was.
- **Book-level targeting (this):** on the monthly decision day, every position
  is sized so that it contributes the same volatility, and the whole set is
  scaled so the book's ex-ante annualised volatility equals a target. Between
  decision days nothing moves. Harvey, Hoyle, Korgaonkar, Rattray, Sargaison
  and Van Hemert (2018) show that this raises the Sharpe of trend and of
  risk assets and cuts their drawdowns; Moskowitz, Ooi and Pedersen's own
  strategy scales every position by its ex-ante volatility.

What it needs: a covariance estimate. Each market's daily cash P&L per contract
over a trailing window, shrunk half-way toward the diagonal so a 33-by-33
matrix from 126 days cannot invent structure. Given the signs the signals
want, solve for the one scalar that puts the book at target, hand each
position its contract count and the risk fraction that count implies, and let
the risk engine apply its limits exactly as it would to any other order. The
per-position cap is the profile's `max_risk_per_trade`; the allocator never
asks for more than that, so a book with two signals on does not become two
enormous bets.

Nothing here sees a price it should not: the window is fed with completed
bars, and a decision taken on a close is sized from history up to that close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Hashable

import numpy as np
import pandas as pd

from core.types import Side


@dataclass(frozen=True)
class LegIntent:
    """One position the book wants, or holds, on a decision day."""

    key: Hashable  # (sleeve, symbol)
    symbol: str
    side: Side
    stop_distance: float
    value_per_unit: float
    #: None: this leg is being sized now. A number: held as is, and its
    #: volatility still counts toward the book's total.
    held_contracts: float | None = None


@dataclass(frozen=True)
class Allocation:
    contracts: float
    risk_fraction: float | None  # None: no history for this market; use the default
    cash_vol_per_contract: float
    note: str = ""


class VolTarget:
    def __init__(
        self,
        target_annual_vol: float = 0.12,
        window: int = 126,
        min_history: int = 63,
        shrink: float = 0.5,
        max_risk_fraction: float = 0.01,
        periods_per_year: int = 252,
    ) -> None:
        if not 0 < target_annual_vol < 1:
            raise ValueError("target_annual_vol must be a fraction in (0, 1)")
        if not 0 <= shrink <= 1:
            raise ValueError("shrink must be in [0, 1]")
        self.target = target_annual_vol
        self.window = window
        self.min_history = min_history
        self.shrink = shrink
        self.max_risk_fraction = max_risk_fraction
        self.periods_per_year = periods_per_year
        self._history: dict[str, dict[date, float]] = {}

    # ---------------------------------------------------------------- feed

    def observe(self, day: date, changes: dict[str, float]) -> None:
        """Record one completed bar per symbol: the cash P&L of ONE contract."""
        for symbol, change in changes.items():
            if change is None or not np.isfinite(change):
                continue
            hist = self._history.setdefault(symbol, {})
            hist[day] = float(change)
            if len(hist) > self.window:
                for old in sorted(hist)[: len(hist) - self.window]:
                    del hist[old]

    def ready(self, symbol: str) -> bool:
        return len(self._history.get(symbol, {})) >= self.min_history

    def history_length(self, symbol: str) -> int:
        return len(self._history.get(symbol, {}))

    # ---------------------------------------------------------- estimation

    def _stats(self, symbols: list[str]) -> tuple[pd.Series, pd.DataFrame]:
        frame = pd.DataFrame({s: pd.Series(self._history[s]) for s in symbols}).sort_index().tail(self.window)
        frame = frame.fillna(0.0)  # a day a market did not trade is a day it did not move
        sigma = frame.std(ddof=1)
        r = frame.corr().fillna(0.0).to_numpy(copy=True)
        np.fill_diagonal(r, 1.0)
        shrunk = (1.0 - self.shrink) * r + self.shrink * np.eye(len(symbols))
        return sigma, pd.DataFrame(shrunk, index=symbols, columns=symbols)

    def target_daily_cash_vol(self, equity: float) -> float:
        return self.target * equity / np.sqrt(self.periods_per_year)

    # ---------------------------------------------------------- allocation

    def allocate(self, equity: float, legs: list[LegIntent]) -> dict[Hashable, Allocation]:
        """Contracts and risk fraction for every leg being sized now.

        Solves, for the free legs, the scalar c such that each gets c of daily
        cash volatility (contracts = c / sigma) and the whole book, held legs
        included, sits at the target.
        """
        out: dict[Hashable, Allocation] = {}
        free = [l for l in legs if l.held_contracts is None and self.ready(l.symbol)]
        for l in legs:
            if l.held_contracts is None and not self.ready(l.symbol):
                out[l.key] = Allocation(0.0, None, 0.0,
                                        f"{l.symbol}: {self.history_length(l.symbol)} days of history, "
                                        f"need {self.min_history}; default risk applies")
        fixed = [l for l in legs if l.held_contracts is not None and self.ready(l.symbol)]
        if not free:
            return out

        symbols = sorted({l.symbol for l in free + fixed})
        sigma, corr = self._stats(symbols)
        r = corr.to_numpy()
        idx = {s: i for i, s in enumerate(symbols)}

        s_free = np.array([l.side.sign for l in free], dtype=float)
        rf = np.array([[r[idx[a.symbol], idx[b.symbol]] for b in free] for a in free])
        a = float(s_free @ rf @ s_free)

        b = d = 0.0
        if fixed:
            x_fixed = np.array([l.side.sign * l.held_contracts * float(sigma[l.symbol]) for l in fixed])
            rfx = np.array([[r[idx[a_.symbol], idx[b_.symbol]] for b_ in fixed] for a_ in free])
            rxx = np.array([[r[idx[a_.symbol], idx[b_.symbol]] for b_ in fixed] for a_ in fixed])
            b = float(2.0 * s_free @ rfx @ x_fixed)
            d = float(x_fixed @ rxx @ x_fixed)

        target = self.target_daily_cash_vol(equity)
        c = 0.0
        if a > 0:
            disc = b * b - 4.0 * a * (d - target * target)
            if disc >= 0:
                c = max((-b + np.sqrt(disc)) / (2.0 * a), 0.0)

        for l in free:
            sig = float(sigma[l.symbol])
            if sig <= 0 or c <= 0:
                out[l.key] = Allocation(0.0, 0.0, sig, "book already at target; nothing to add" if c <= 0 else "zero volatility")
                continue
            contracts = c / sig
            risk_cash = contracts * l.stop_distance * l.value_per_unit
            risk_fraction = risk_cash / equity
            note = f"{contracts:.2f} contracts for {c:,.0f}/day of vol"
            if risk_fraction > self.max_risk_fraction:
                risk_fraction = self.max_risk_fraction
                contracts = risk_fraction * equity / (l.stop_distance * l.value_per_unit)
                note += f"; capped at {self.max_risk_fraction:.2%} risk"
            out[l.key] = Allocation(contracts, risk_fraction, sig, note)
        return out

    def realised_book_vol(self, equity_curve: pd.Series) -> float:
        """Annualised volatility of an equity curve, for checking the target held."""
        r = equity_curve.pct_change().dropna()
        return float(r.std(ddof=1) * np.sqrt(self.periods_per_year)) if len(r) > 1 else 0.0
