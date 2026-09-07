"""Commodity seasonality, strictly causal. Research entry 015b.

Grains have a harvest, heating oil has a winter, cattle have a herd cycle. Those
are physical mechanisms that repeat on the calendar, and they are the only
reason to expect a seasonal effect at all.

They are also the most overfittable idea in this repository. Twelve months
across forty-six markets is five hundred and fifty-two chances to find a
pattern in noise, and a seasonal backtest that picks its months after the fact
will always look magnificent. So this rule is deliberately given no choices:

- **Only markets with a mechanism.** The caller passes the grains, energy and
  meats sectors. A seasonal effect in the Nasdaq would have no cause, and
  finding one would be evidence of a bug rather than a signal.
- **Expanding window.** On the first trading day of month m in year Y, the mean
  return of month m is computed from years strictly BEFORE Y. The rule at any
  date has seen only what a trader standing on that date had seen.
- **No month selection.** Every month is traded on the sign of its own history.
  Trading only the months that worked is the post-hoc choice this log refuses.
- **A minimum sample.** Under five prior observations the market stands aside
  rather than acting on two coincidences.

The position is opened on the month's first bar and held to the month end, with
a wide disaster stop, so turnover is twelve round trips a year at most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent, Strategy, is_month_start
from core.types import Position, Side
from features.indicators import atr


class Seasonality(Strategy):
    name = "seasonality"

    def __init__(
        self,
        min_years: int = 5,
        atr_period: int = 14,
        atr_stop_multiple: float = 4.0,
    ) -> None:
        self.min_years = min_years
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        # a decade of daily bars before the first decision can have five prior
        # observations of any given month
        self.warmup = 252 * min_years + 30
        self._decided: Side | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = atr(df, self.atr_period)
        ts = pd.to_datetime(df["ts"], utc=True)
        df["_year"] = ts.dt.year.to_numpy()
        df["_month"] = ts.dt.month.to_numpy()

        # Monthly returns, stamped on the LAST bar of each month, then shifted
        # forward so that a decision on the first bar of a month can only ever
        # read months that have already finished.
        close = df["close"].astype(float)
        key = df["_year"] * 100 + df["_month"]
        first = close.groupby(key).transform("first")
        last = close.groupby(key).transform("last")
        monthly = (last - first) / first.abs().replace(0.0, np.nan)

        ends = key != key.shift(-1)
        hist = pd.DataFrame({"ym": key[ends], "year": df["_year"][ends],
                             "month": df["_month"][ends], "ret": monthly[ends]})
        self._history = hist.dropna(subset=["ret"]).reset_index(drop=True)
        return df

    def _prior_mean(self, month: int, year: int) -> tuple[float, int]:
        """Mean return of this calendar month across strictly earlier years."""
        h = self._history
        past = h[(h["month"] == month) & (h["year"] < year)]
        if len(past) < self.min_years:
            return float("nan"), len(past)
        return float(past["ret"].mean()), len(past)

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        row = df.iloc[i]
        atr_now = row["atr"]
        if pd.isna(atr_now) or atr_now <= 0:
            return FLAT
        stop = float(atr_now) * self.atr_stop_multiple

        if not is_month_start(df, i):
            if position is None:
                return FLAT  # no re-entry mid-month after a stop
            side = self._decided if self._decided is position.side else position.side
            return Intent(side=side, stop_distance=stop, confidence=1.0, reason="hold")

        mean, n = self._prior_mean(int(row["_month"]), int(row["_year"]))
        if not np.isfinite(mean) or mean == 0:
            self._decided = None
            return FLAT
        self._decided = Side.BUY if mean > 0 else Side.SELL
        return Intent(side=self._decided, stop_distance=stop, confidence=1.0,
                      reason=f"month {int(row['_month'])} mean {mean:+.3%} over {n} prior years")
