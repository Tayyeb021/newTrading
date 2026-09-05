"""Futures contracts: roots, expiry rules, and which month is the front.

A CFD is one symbol forever. A future is a family of contracts that expire, and
the thing you trade changes every one to three months. Three consequences the
rest of the system has to know about:

- **The front month moves.** "MES" on the 10th of December is MESZ5; on the
  20th it is MESH6. The adapter resolves the root to the live contract.
- **You must roll before expiry.** Hold through last trade and you are
  delivered a cash settlement, or worse a physical one. Roll a few business days
  early, journal the cost, move on.
- **History is a stitched series.** Every expiry is its own price series. A
  backtest needs one continuous series with the roll gaps removed, or every
  roll shows up as a fake trade.

Expiry rules follow CME's published contract specs. Holidays are not modelled
here -- the calendar is business days only -- which is why `roll_days_before`
defaults to a buffer wide enough that a holiday cannot push a roll past last
trade. Verify the roll date against the exchange calendar before going live on
a new root.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from core.types import SymbolSpec

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


def _busday(d: date) -> bool:
    return d.weekday() < 5


def _add_busdays(d: date, n: int) -> date:
    step = 1 if n > 0 else -1
    remaining = abs(n)
    while remaining:
        d += timedelta(days=step)
        if _busday(d):
            remaining -= 1
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_busday(year: int, month: int) -> date:
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    while not _busday(d):
        d -= timedelta(days=1)
    return d


def last_trade_date(rule: str, year: int, month: int) -> date:
    """Last trading day of the contract for `year`/`month` under `rule`."""
    if rule == "third_friday":
        # Equity index: MES, MNQ, ES, NQ.
        return _nth_weekday(year, month, 4, 3)
    if rule == "gold":
        # COMEX gold: third-to-last business day of the contract month.
        return _add_busdays(_last_busday(year, month), -2)
    if rule == "fx":
        # CME FX: second business day before the third Wednesday.
        return _add_busdays(_nth_weekday(year, month, 2, 3), -2)
    if rule == "crude":
        # NYMEX WTI: three business days before the 25th of the month PRIOR to
        # the contract month (moved earlier if the 25th is not a business day).
        py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
        d = date(py, pm, 25)
        while not _busday(d):
            d -= timedelta(days=1)
        return _add_busdays(d, -3)
    if rule == "treasury":
        # CBOT notes: seventh business day before the last business day.
        return _add_busdays(_last_busday(year, month), -7)
    raise ValueError(f"unknown expiry rule {rule!r}")


@dataclass(frozen=True)
class FuturesRoot:
    root: str
    exchange: str
    name: str
    multiplier: float  # cash per 1.0 of price per contract
    tick_size: float
    months: tuple[int, ...]
    expiry_rule: str
    roll_days_before: int = 5
    currency: str = "USD"
    #: All-in per side: broker commission plus exchange and clearing fees.
    commission_per_side: float = 1.00
    #: Typical intraday margin. Overnight is several times higher.
    margin_day: float = 100.0
    bucket: str = ""

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.multiplier

    @property
    def digits(self) -> int:
        s = f"{self.tick_size:.10f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0

    def to_spec(self, symbol: str | None = None) -> SymbolSpec:
        """Futures as a SymbolSpec. No swap, whole contracts, mode 0."""
        return SymbolSpec(
            symbol=symbol or self.root,
            digits=self.digits,
            point=self.tick_size,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            volume_min=1.0,
            volume_max=10_000.0,
            volume_step=1.0,
            contract_size=self.multiplier,
            stops_level_points=0,
            swap_long=0.0,
            swap_short=0.0,
            currency_profit=self.currency,
            swap_mode=0,
            swap_triple_weekday=2,
        )

    # ------------------------------------------------------------ calendar

    def last_trade(self, year: int, month: int) -> date:
        return last_trade_date(self.expiry_rule, year, month)

    def roll_date(self, year: int, month: int) -> date:
        return _add_busdays(self.last_trade(year, month), -self.roll_days_before)

    def listed(self, start: date, end: date) -> list[tuple[int, int]]:
        out = []
        for year in range(start.year - 1, end.year + 2):
            for month in self.months:
                if start - timedelta(days=400) <= self.last_trade(year, month) <= end + timedelta(days=400):
                    out.append((year, month))
        return sorted(out)

    def front(self, as_of: date) -> tuple[int, int]:
        """The contract to trade on `as_of`: nearest month whose roll date is
        still ahead. On the roll date itself you are already in the next one."""
        for year, month in self.listed(as_of, as_of):
            if self.roll_date(year, month) > as_of:
                return year, month
        raise ValueError(f"{self.root}: no listed contract after {as_of}")

    def next_after(self, year: int, month: int) -> tuple[int, int]:
        seq = self.listed(date(year, 1, 1), date(year + 1, 12, 31))
        i = seq.index((year, month))
        return seq[i + 1]

    def schedule(self, start: date, end: date) -> list["ContractWindow"]:
        """Every contract active between start and end, with its window."""
        out: list[ContractWindow] = []
        prev_roll: date | None = None
        for year, month in self.listed(start, end):
            roll = self.roll_date(year, month)
            if roll < start:
                prev_roll = roll
                continue
            begins = prev_roll if prev_roll is not None else start
            out.append(ContractWindow(self.root, year, month, begins, roll, self.last_trade(year, month)))
            prev_roll = roll
            if roll > end:
                break
        return out

    # ------------------------------------------------------------- symbols

    def code(self, year: int, month: int) -> str:
        """Exchange-style ticker: MESZ5."""
        return f"{self.root}{MONTH_CODES[month]}{year % 10}"

    def ib_month(self, year: int, month: int) -> str:
        """IB's lastTradeDateOrContractMonth: 202512."""
        return f"{year}{month:02d}"


@dataclass(frozen=True)
class ContractWindow:
    root: str
    year: int
    month: int
    active_from: date  # previous contract's roll date
    roll_on: date  # this contract's roll date
    last_trade: date


# --------------------------------------------------------------------------- #
# The micro universe
# --------------------------------------------------------------------------- #

MICRO_UNIVERSE: dict[str, FuturesRoot] = {
    "MES": FuturesRoot("MES", "CME", "Micro E-mini S&P 500", 5.0, 0.25, (3, 6, 9, 12), "third_friday",
                       roll_days_before=5, commission_per_side=0.85, margin_day=50.0, bucket="us_indices"),
    "MNQ": FuturesRoot("MNQ", "CME", "Micro E-mini Nasdaq-100", 2.0, 0.25, (3, 6, 9, 12), "third_friday",
                       roll_days_before=5, commission_per_side=0.85, margin_day=100.0, bucket="us_indices"),
    "MGC": FuturesRoot("MGC", "COMEX", "Micro Gold", 10.0, 0.10, (2, 4, 6, 8, 10, 12), "gold",
                       roll_days_before=7, commission_per_side=1.00, margin_day=150.0, bucket="metals"),
    "M6E": FuturesRoot("M6E", "CME", "Micro EUR/USD", 12_500.0, 0.0001, (3, 6, 9, 12), "fx",
                       roll_days_before=5, commission_per_side=0.85, margin_day=100.0, bucket="usd_majors"),
    "MCL": FuturesRoot("MCL", "NYMEX", "Micro WTI Crude", 100.0, 0.01, tuple(range(1, 13)), "crude",
                       roll_days_before=5, commission_per_side=1.00, margin_day=200.0, bucket="energy"),
    "ZN": FuturesRoot("ZN", "CBOT", "10-Year T-Note", 1000.0, 0.015625, (3, 6, 9, 12), "treasury",
                      roll_days_before=10, commission_per_side=1.50, margin_day=500.0, bucket="rates"),
}
