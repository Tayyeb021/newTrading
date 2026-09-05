"""Futures contracts: roots, expiry rules, first notice, and which month is the front.

A CFD is one symbol forever. A future is a family of contracts that expire, and
the thing you trade changes every one to three months. Three consequences the
rest of the system has to know about:

- **The front month moves.** "MES" on the 10th of December is MESZ5; on the
  20th it is MESH6. The adapter resolves the root to the live contract.
- **You must roll before expiry -- and before first notice.** Cash-settled
  contracts (equity index, lean hogs) can be held to last trade. Physically
  delivered ones (grains, metals, cattle, treasuries) send a delivery notice
  from *first notice day*, which is BEFORE last trade; hold a long past it and
  you can be assigned 5,000 bushels of corn. The roll anchors on whichever
  comes first.
- **History is a stitched series.** Every expiry is its own price series. A
  backtest needs one continuous series with the roll gaps removed, or every
  roll shows up as a fake trade.

Expiry rules follow CME Group's published contract specs (CME, CBOT, NYMEX,
COMEX). Holidays are not modelled -- the calendar is business days only --
which is why `roll_days_before` is a buffer wide enough that a holiday cannot
push a roll past the anchor. Verify the roll date against the exchange
calendar before going live on a new root.

Two universes:

- `FULL_UNIVERSE`: 33 full-size CME Group markets across seven sectors. This
  is the RESEARCH universe -- longest history, same price as the micro to
  within a tick, and the breadth that trend-following evidence says the
  strategy needs.
- `MICRO_UNIVERSE`: what a small account can actually trade. `MICRO_OF` maps a
  full root to its micro; `tradeable()` returns the smallest contract for a
  root so a backtest can read full-size history and size with the micro.

Prices are in the exchange's quote unit (index points, dollars, cents per
bushel, cents per pound); `multiplier` is the cash value of 1.0 of that unit.
`margin_day` is an approximate intraday margin, refreshed by hand; it feeds the
margin-level guard only, never sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from core.types import SymbolSpec

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
CODE_MONTHS = {v: k for k, v in MONTH_CODES.items()}

QUARTERLY = (3, 6, 9, 12)
MONTHLY = tuple(range(1, 13))


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


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = _last_calendar_day(year, month)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _last_calendar_day(year: int, month: int) -> date:
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    return nxt - timedelta(days=1)


def _last_busday(year: int, month: int) -> date:
    d = _last_calendar_day(year, month)
    while not _busday(d):
        d -= timedelta(days=1)
    return d


def _nth_busday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    while not _busday(d):
        d += timedelta(days=1)
    return _add_busdays(d, n - 1)


def _prior_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def last_trade_date(rule: str, year: int, month: int) -> date:
    """Last trading day of the contract for `year`/`month` under `rule`."""
    if rule == "third_friday":
        # CME equity index (ES, NQ, YM, RTY and micros): third Friday, cash settled.
        return _nth_weekday(year, month, 4, 3)
    if rule in ("gold", "metals"):
        # COMEX/NYMEX metals (GC, SI, HG, PL): third-to-last business day of the month.
        return _add_busdays(_last_busday(year, month), -2)
    if rule == "fx":
        # CME FX: second business day before the third Wednesday.
        return _add_busdays(_nth_weekday(year, month, 2, 3), -2)
    if rule == "crude":
        # NYMEX WTI: three business days before the 25th of the month PRIOR to
        # the contract month (moved earlier if the 25th is not a business day).
        py, pm = _prior_month(year, month)
        d = date(py, pm, 25)
        while not _busday(d):
            d -= timedelta(days=1)
        return _add_busdays(d, -3)
    if rule == "natgas":
        # NYMEX Henry Hub: third-to-last business day of the month PRIOR to the contract month.
        py, pm = _prior_month(year, month)
        return _add_busdays(_last_busday(py, pm), -2)
    if rule == "refined":
        # NYMEX RBOB / heating oil: last business day of the month PRIOR to the contract month.
        py, pm = _prior_month(year, month)
        return _last_busday(py, pm)
    if rule == "treasury":
        # CBOT ZN, ZB, UB: seventh business day before the last business day.
        return _add_busdays(_last_busday(year, month), -7)
    if rule == "last_busday":
        # CBOT ZT, ZF; CME live cattle: last business day of the contract month.
        return _last_busday(year, month)
    if rule == "grains":
        # CBOT corn, soy complex, wheat: business day before the 15th.
        return _add_busdays(date(year, month, 15), -1)
    if rule == "lean_hogs":
        # CME lean hogs: tenth business day of the contract month, cash settled.
        return _nth_busday(year, month, 10)
    if rule == "feeder":
        # CME feeder cattle: last Thursday of the contract month, cash settled.
        return _last_weekday(year, month, 3)
    raise ValueError(f"unknown expiry rule {rule!r}")


def first_notice_date(rule: str, year: int, month: int) -> date:
    """First day a long can be assigned delivery. Roll before it."""
    if rule == "prior_month_end":
        # Grains, metals, treasuries (first position day), live cattle: the
        # last business day of the month before the contract month, or earlier.
        py, pm = _prior_month(year, month)
        return _last_busday(py, pm)
    raise ValueError(f"unknown first-notice rule {rule!r}")


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
    #: Approximate intraday margin. Overnight is several times higher.
    margin_day: float = 100.0
    bucket: str = ""
    #: None for cash-settled contracts and for those whose last trade precedes
    #: the delivery month (energy). Otherwise the roll anchors on this.
    first_notice_rule: str | None = None

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.multiplier

    @property
    def digits(self) -> int:
        s = f"{self.tick_size:.10f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0

    @property
    def physically_delivered(self) -> bool:
        return self.first_notice_rule is not None

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

    def first_notice(self, year: int, month: int) -> date | None:
        if self.first_notice_rule is None:
            return None
        return first_notice_date(self.first_notice_rule, year, month)

    def roll_anchor(self, year: int, month: int) -> date:
        """The day you must be out by: first notice if there is one, else last trade."""
        ltd = self.last_trade(year, month)
        fnd = self.first_notice(year, month)
        return min(ltd, fnd) if fnd is not None else ltd

    def roll_date(self, year: int, month: int) -> date:
        return _add_busdays(self.roll_anchor(year, month), -self.roll_days_before)

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
            out.append(ContractWindow(self.root, year, month, begins, roll,
                                      self.last_trade(year, month), self.first_notice(year, month)))
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
    first_notice: date | None = None


# --------------------------------------------------------------------------- #
# The research universe: 33 full-size CME Group markets, seven sectors
# --------------------------------------------------------------------------- #

def _r(root, exchange, name, multiplier, tick, months, rule, *, roll=5, comm=2.50, margin=500.0,
       bucket="", fnd=None) -> FuturesRoot:
    return FuturesRoot(root, exchange, name, multiplier, tick, months, rule, roll_days_before=roll,
                       commission_per_side=comm, margin_day=margin, bucket=bucket, first_notice_rule=fnd)


FULL_UNIVERSE: dict[str, FuturesRoot] = {
    # equity index -- cash settled, third Friday
    "ES":  _r("ES", "CME", "E-mini S&P 500", 50.0, 0.25, QUARTERLY, "third_friday", margin=1500.0, bucket="us_indices"),
    "NQ":  _r("NQ", "CME", "E-mini Nasdaq-100", 20.0, 0.25, QUARTERLY, "third_friday", margin=2000.0, bucket="us_indices"),
    "YM":  _r("YM", "CBOT", "E-mini Dow", 5.0, 1.0, QUARTERLY, "third_friday", margin=1000.0, bucket="us_indices"),
    "RTY": _r("RTY", "CME", "E-mini Russell 2000", 50.0, 0.10, QUARTERLY, "third_friday", margin=800.0, bucket="us_indices"),
    # rates -- physically delivered; roll off first position day at the prior month end
    "ZT":  _r("ZT", "CBOT", "2-Year T-Note", 2000.0, 0.00390625, QUARTERLY, "last_busday", roll=3, comm=1.50, margin=400.0, bucket="rates", fnd="prior_month_end"),
    "ZF":  _r("ZF", "CBOT", "5-Year T-Note", 1000.0, 0.0078125, QUARTERLY, "last_busday", roll=3, comm=1.50, margin=400.0, bucket="rates", fnd="prior_month_end"),
    "ZN":  _r("ZN", "CBOT", "10-Year T-Note", 1000.0, 0.015625, QUARTERLY, "treasury", roll=3, comm=1.50, margin=500.0, bucket="rates", fnd="prior_month_end"),
    "ZB":  _r("ZB", "CBOT", "30-Year T-Bond", 1000.0, 0.03125, QUARTERLY, "treasury", roll=3, comm=1.50, margin=800.0, bucket="rates", fnd="prior_month_end"),
    "UB":  _r("UB", "CBOT", "Ultra T-Bond", 1000.0, 0.03125, QUARTERLY, "treasury", roll=3, comm=1.50, margin=1200.0, bucket="rates", fnd="prior_month_end"),
    # FX -- last trade precedes delivery, no first-notice constraint
    "6E":  _r("6E", "CME", "Euro FX", 125_000.0, 0.00005, QUARTERLY, "fx", margin=800.0, bucket="usd_majors"),
    "6J":  _r("6J", "CME", "Japanese Yen", 12_500_000.0, 0.0000005, QUARTERLY, "fx", margin=800.0, bucket="usd_majors"),
    "6B":  _r("6B", "CME", "British Pound", 62_500.0, 0.0001, QUARTERLY, "fx", margin=800.0, bucket="usd_majors"),
    "6A":  _r("6A", "CME", "Australian Dollar", 100_000.0, 0.00005, QUARTERLY, "fx", margin=600.0, bucket="usd_majors"),
    "6C":  _r("6C", "CME", "Canadian Dollar", 100_000.0, 0.00005, QUARTERLY, "fx", margin=500.0, bucket="usd_majors"),
    "6S":  _r("6S", "CME", "Swiss Franc", 125_000.0, 0.0001, QUARTERLY, "fx", margin=1000.0, bucket="usd_majors"),
    "6N":  _r("6N", "CME", "New Zealand Dollar", 100_000.0, 0.00005, QUARTERLY, "fx", margin=600.0, bucket="usd_majors"),
    # metals -- physically delivered; roll before the prior month end
    "GC":  _r("GC", "COMEX", "Gold", 100.0, 0.10, (2, 4, 6, 8, 10, 12), "metals", roll=5, margin=2500.0, bucket="metals", fnd="prior_month_end"),
    "SI":  _r("SI", "COMEX", "Silver", 5000.0, 0.005, (3, 5, 7, 9, 12), "metals", roll=5, margin=3000.0, bucket="metals", fnd="prior_month_end"),
    "HG":  _r("HG", "COMEX", "Copper", 25_000.0, 0.0005, (3, 5, 7, 9, 12), "metals", roll=5, margin=1500.0, bucket="metals", fnd="prior_month_end"),
    "PL":  _r("PL", "NYMEX", "Platinum", 50.0, 0.10, (1, 4, 7, 10), "metals", roll=5, margin=1200.0, bucket="metals", fnd="prior_month_end"),
    # energy -- last trade precedes the delivery month
    "CL":  _r("CL", "NYMEX", "WTI Crude", 1000.0, 0.01, MONTHLY, "crude", margin=2000.0, bucket="energy"),
    "NG":  _r("NG", "NYMEX", "Henry Hub Natural Gas", 10_000.0, 0.001, MONTHLY, "natgas", margin=1500.0, bucket="energy"),
    "RB":  _r("RB", "NYMEX", "RBOB Gasoline", 42_000.0, 0.0001, MONTHLY, "refined", margin=2000.0, bucket="energy"),
    "HO":  _r("HO", "NYMEX", "NY Harbor ULSD", 42_000.0, 0.0001, MONTHLY, "refined", margin=2000.0, bucket="energy"),
    # grains -- cents per bushel (ZM: dollars per short ton; ZL: cents per pound); physically delivered
    "ZC":  _r("ZC", "CBOT", "Corn", 50.0, 0.25, (3, 5, 7, 9, 12), "grains", roll=5, margin=600.0, bucket="grains", fnd="prior_month_end"),
    "ZS":  _r("ZS", "CBOT", "Soybeans", 50.0, 0.25, (1, 3, 5, 7, 8, 9, 11), "grains", roll=5, margin=1200.0, bucket="grains", fnd="prior_month_end"),
    "ZW":  _r("ZW", "CBOT", "Chicago Wheat", 50.0, 0.25, (3, 5, 7, 9, 12), "grains", roll=5, margin=900.0, bucket="grains", fnd="prior_month_end"),
    "KE":  _r("KE", "CBOT", "KC Hard Red Winter Wheat", 50.0, 0.25, (3, 5, 7, 9, 12), "grains", roll=5, margin=900.0, bucket="grains", fnd="prior_month_end"),
    "ZM":  _r("ZM", "CBOT", "Soybean Meal", 100.0, 0.10, (1, 3, 5, 7, 8, 9, 10, 12), "grains", roll=5, margin=900.0, bucket="grains", fnd="prior_month_end"),
    "ZL":  _r("ZL", "CBOT", "Soybean Oil", 600.0, 0.01, (1, 3, 5, 7, 8, 9, 10, 12), "grains", roll=5, margin=800.0, bucket="grains", fnd="prior_month_end"),
    # meats -- cents per pound
    "LE":  _r("LE", "CME", "Live Cattle", 400.0, 0.025, (2, 4, 6, 8, 10, 12), "last_busday", roll=5, margin=800.0, bucket="meats", fnd="prior_month_end"),
    "HE":  _r("HE", "CME", "Lean Hogs", 400.0, 0.025, (2, 4, 5, 6, 7, 8, 10, 12), "lean_hogs", roll=5, margin=800.0, bucket="meats"),
    "GF":  _r("GF", "CME", "Feeder Cattle", 500.0, 0.025, (1, 3, 4, 5, 8, 9, 10, 11), "feeder", roll=5, margin=1000.0, bucket="meats"),
}

SECTORS: dict[str, tuple[str, ...]] = {
    "us_indices": ("ES", "NQ", "YM", "RTY"),
    "rates": ("ZT", "ZF", "ZN", "ZB", "UB"),
    "usd_majors": ("6E", "6J", "6B", "6A", "6C", "6S", "6N"),
    "metals": ("GC", "SI", "HG", "PL"),
    "energy": ("CL", "NG", "RB", "HO"),
    "grains": ("ZC", "ZS", "ZW", "KE", "ZM", "ZL"),
    "meats": ("LE", "HE", "GF"),
}

# --------------------------------------------------------------------------- #
# The tradeable universe: micros where they exist
# --------------------------------------------------------------------------- #

MICRO_UNIVERSE: dict[str, FuturesRoot] = {
    "MES": _r("MES", "CME", "Micro E-mini S&P 500", 5.0, 0.25, QUARTERLY, "third_friday", comm=0.85, margin=50.0, bucket="us_indices"),
    "MNQ": _r("MNQ", "CME", "Micro E-mini Nasdaq-100", 2.0, 0.25, QUARTERLY, "third_friday", comm=0.85, margin=100.0, bucket="us_indices"),
    "MYM": _r("MYM", "CBOT", "Micro E-mini Dow", 0.5, 1.0, QUARTERLY, "third_friday", comm=0.85, margin=50.0, bucket="us_indices"),
    "M2K": _r("M2K", "CME", "Micro E-mini Russell 2000", 5.0, 0.10, QUARTERLY, "third_friday", comm=0.85, margin=50.0, bucket="us_indices"),
    "MGC": _r("MGC", "COMEX", "Micro Gold", 10.0, 0.10, (2, 4, 6, 8, 10, 12), "metals", roll=7, comm=1.00, margin=150.0, bucket="metals", fnd="prior_month_end"),
    "SIL": _r("SIL", "COMEX", "Micro Silver", 1000.0, 0.005, (3, 5, 7, 9, 12), "metals", roll=7, comm=1.00, margin=200.0, bucket="metals", fnd="prior_month_end"),
    "MHG": _r("MHG", "COMEX", "Micro Copper", 2500.0, 0.0005, (3, 5, 7, 9, 12), "metals", roll=7, comm=1.00, margin=100.0, bucket="metals", fnd="prior_month_end"),
    "M6E": _r("M6E", "CME", "Micro EUR/USD", 12_500.0, 0.0001, QUARTERLY, "fx", comm=0.85, margin=100.0, bucket="usd_majors"),
    "M6A": _r("M6A", "CME", "Micro AUD/USD", 10_000.0, 0.0001, QUARTERLY, "fx", comm=0.85, margin=50.0, bucket="usd_majors"),
    "M6B": _r("M6B", "CME", "Micro GBP/USD", 6250.0, 0.0001, QUARTERLY, "fx", comm=0.85, margin=50.0, bucket="usd_majors"),
    "MCL": _r("MCL", "NYMEX", "Micro WTI Crude", 100.0, 0.01, MONTHLY, "crude", comm=1.00, margin=200.0, bucket="energy"),
    # No micro exists; the full contract is the smallest. Kept here so the
    # tradeable list has a rates leg for accounts large enough to hold it.
    "ZN": FULL_UNIVERSE["ZN"],
}

#: Full-size research root -> the micro a small account trades.
MICRO_OF: dict[str, str] = {
    "ES": "MES", "NQ": "MNQ", "YM": "MYM", "RTY": "M2K",
    "GC": "MGC", "SI": "SIL", "HG": "MHG",
    "6E": "M6E", "6A": "M6A", "6B": "M6B",
    "CL": "MCL",
}
PARENT_OF: dict[str, str] = {v: k for k, v in MICRO_OF.items()}

ALL_ROOTS: dict[str, FuturesRoot] = {**FULL_UNIVERSE, **MICRO_UNIVERSE}


def tradeable(root: str) -> FuturesRoot:
    """The smallest contract for a root: its micro if one exists, else itself."""
    if root in MICRO_OF:
        return MICRO_UNIVERSE[MICRO_OF[root]]
    return ALL_ROOTS[root]


def data_root(root: str) -> FuturesRoot:
    """The contract whose history to research on: the full-size parent for a
    micro (longer history, same price to within a tick), else itself."""
    return FULL_UNIVERSE[PARENT_OF[root]] if root in PARENT_OF else ALL_ROOTS[root]
