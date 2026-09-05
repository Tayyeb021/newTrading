"""Continuous futures series from individual expiries.

Each contract month is its own price series. Chain them naively and every roll
is a price jump that a backtest reads as a trade: gold December at 2,650, gold
February at 2,662 the next morning, and a strategy that was long "made" twelve
dollars it never could have. Over fifteen years of quarterly rolls that is
sixty fake trades, all in the same direction as the term structure.

The **back-adjusted** ("Panama canal") series fixes this by shifting every price
before each roll by the gap at that roll, so the stitched series has no jump.
Prices far back in history end up offset from what actually printed -- gold in
2010 might read 1,180 instead of 1,220 -- but *returns* are right, which is
what a strategy trades on.

Two things are preserved and reported rather than hidden: the roll dates, and
the gap at each one. The gap is the roll's economic cost or benefit (the term
structure); the dates are where a live system will have to pay a round trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from core.contracts import FuturesRoot


@dataclass(frozen=True)
class Roll:
    on: date
    from_contract: str
    to_contract: str
    gap: float  # next.close - front.close on the roll date
    front_close: float
    next_close: float


def stitch(
    root: FuturesRoot,
    expiries: dict[tuple[int, int], pd.DataFrame],
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, list[Roll]]:
    """Back-adjusted continuous bars from per-expiry frames.

    `expiries` maps (year, month) -> OHLCV frame with a tz-aware `ts` column.
    Returns the continuous frame and the roll log. Bars on the roll date itself
    belong to the NEXT contract, matching `FuturesRoot.front`.
    """
    if not expiries:
        raise ValueError("no expiries supplied")

    frames = {k: _prep(v) for k, v in expiries.items()}
    all_dates = pd.concat([f["day"] for f in frames.values()])
    start = start or all_dates.min()
    end = end or all_dates.max()

    windows = root.schedule(start, end)
    windows = [w for w in windows if (w.year, w.month) in frames]
    if not windows:
        raise ValueError(f"{root.root}: no expiry frames overlap {start}..{end}")

    pieces: list[pd.DataFrame] = []
    rolls: list[Roll] = []

    for i, w in enumerate(windows):
        f = frames[(w.year, w.month)]
        lo = w.active_from
        hi = w.roll_on
        piece = f[(f["day"] >= lo) & (f["day"] < hi)] if i < len(windows) - 1 else f[f["day"] >= lo]
        piece = piece.assign(contract=root.code(w.year, w.month), carry=float("nan"))

        if i < len(windows) - 1:
            nxt = windows[i + 1]
            g = frames[(nxt.year, nxt.month)]
            piece["carry"] = _carry(piece, g, w.last_trade, nxt.last_trade)
            gap, fc, nc = _gap_on(f, g, hi)
            rolls.append(Roll(hi, root.code(w.year, w.month), root.code(nxt.year, nxt.month), gap, fc, nc))
        pieces.append(piece)

    # Back-adjust: walk rolls newest -> oldest, accumulating the shift.
    shift = 0.0
    adjusted = []
    for i in range(len(pieces) - 1, -1, -1):
        p = pieces[i].copy()
        for col in ("open", "high", "low", "close"):
            p[col] = p[col] + shift
        adjusted.append(p)
        if i > 0:
            shift += rolls[i - 1].gap
    out = pd.concat(reversed(adjusted), ignore_index=True)
    out = out.drop(columns=["day"]).sort_values("ts").reset_index(drop=True)
    return out, rolls


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f["day"] = f["ts"].dt.date
    return f.sort_values("ts").reset_index(drop=True)


def _carry(front: pd.DataFrame, nxt: pd.DataFrame, front_expiry: date, next_expiry: date):
    """Annualised roll yield of holding the front against the next contract,
    measured on every day both printed:

        (front - next) / front * 365 / days between the two expiries

    Positive is backwardation, which pays a long as the contract rolls up the
    curve toward spot; negative is contango, which pays a short. This is the
    carry signal of Koijen, Moskowitz, Pedersen and Vrugt, read straight off
    the curve, and it is not affected by the back-adjustment because it is
    taken from the raw closes before any shift. Forward-filled across days the
    next contract did not print; NaN before it first did.
    """
    days = max((next_expiry - front_expiry).days, 1)
    next_close = nxt.groupby("day")["close"].last()
    matched = front["day"].map(next_close)
    carry = (front["close"] - matched) / front["close"] * (365.0 / days)
    return carry.ffill().to_numpy()


def _gap_on(front: pd.DataFrame, nxt: pd.DataFrame, roll_day: date) -> tuple[float, float, float]:
    """Close-to-close gap on the last day both contracts traded before the roll."""
    common = sorted(set(front["day"]) & set(nxt["day"]))
    candidates = [d for d in common if d < roll_day] or common
    if not candidates:
        raise ValueError(f"no overlapping session to measure the roll gap before {roll_day}")
    d = candidates[-1]
    fc = float(front.loc[front["day"] == d, "close"].iloc[-1])
    nc = float(nxt.loc[nxt["day"] == d, "close"].iloc[-1])
    return nc - fc, fc, nc


def roll_cost_cash(root: FuturesRoot, contracts: float, spread_ticks: float = 1.0) -> float:
    """What a live roll costs: close one, open the next, both crossing a spread."""
    friction = 2 * root.commission_per_side * contracts
    friction += 2 * spread_ticks * root.tick_value * contracts
    return friction


def annual_roll_drag(rolls: list[Roll], root: FuturesRoot, years: float) -> float:
    """Average yearly term-structure gap in price units, signed. Positive means
    the next contract was priced above the front (contango): a long pays it."""
    if not rolls or years <= 0:
        return 0.0
    return sum(r.gap for r in rolls) / years
