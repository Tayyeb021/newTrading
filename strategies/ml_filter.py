"""The AI layer as a wrapper around ANY base rule. Research entry 013.

`TrendML` bolted the models onto the daily trend rule that entry 007 killed.
This does the same job for whatever rule is alive - today the monthly
time-series momentum of entry 010 - without touching it:

- the base rule decides direction, stop and calendar exactly as before;
- on a NEW ENTRY the meta model is asked "given that this rule just fired,
  is this trade likely to work?", and may skip it or shrink it;
- "hold" intents between decisions, and re-decisions on a position already
  held, pass through untouched. A filter judges entries; letting it veto a
  running position would turn it into a second exit rule and reopen the
  turnover wound of 007-009;
- the regime model, if given, scales confidence down and never up.

Confidence can only fall below the base rule's. The risk engine still sizes
and still applies every limit. The worst a broken model can do is trade less.

`event_features` is the ONE place features are built, for training and for
live, so the model can never be asked a question in a different dialect from
the one it learned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent, Strategy
from core.types import Position
from ml.models import MetaLabelModel, RegimeModel, meta_features


def event_features(prepared: pd.DataFrame, base: Strategy) -> pd.DataFrame:
    """Meta features plus what the base rule knows at decision time: its own
    forecast strength and its speed. Same function at training and live."""
    feats = meta_features(prepared)
    feats["base_forecast"] = prepared["forecast"].to_numpy(dtype=float) if "forecast" in prepared else np.nan
    feats["lookback"] = float(getattr(base, "lookback", np.nan))
    return feats


def signal_events(base: Strategy, prepared: pd.DataFrame) -> pd.DataFrame:
    """Every bar where the base rule would open a position from flat.

    Meta-labelling trains on the trades the rule actually proposes, not on
    every bar. For a monthly rule that is its decision days."""
    rows = []
    for i in range(base.warmup, len(prepared) - 1):
        intent = base.evaluate(prepared, i, None)
        if not intent.flat:
            rows.append({"i": i, "side": intent.side, "stop_distance": intent.stop_distance})
    return pd.DataFrame(rows)


class MLFiltered(Strategy):
    def __init__(
        self,
        base: Strategy,
        meta: MetaLabelModel | None = None,
        regime: RegimeModel | None = None,
        min_confidence: float = 0.05,
    ) -> None:
        self.base = base
        self.meta = meta
        self.regime = regime
        self.min_confidence = min_confidence
        self.name = f"{base.name}_ml"
        self.rebalances = base.rebalances
        self.inertia = base.inertia
        self.warmup = max(base.warmup, 265)  # the features carry a 252-bar percentile
        self._features: pd.DataFrame | None = None
        self._scalars: pd.Series | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.base.prepare(df)
        if self.meta is not None:
            self._features = event_features(df, self.base)
        if self.regime is not None:
            self._scalars = self.regime.scalars(df)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        intent = self.base.evaluate(df, i, position)
        if intent.flat or intent.reason == "hold" or position is not None:
            return intent  # not an entry: not the filter's business

        confidence = intent.confidence
        if self._scalars is not None:
            scalar = self._scalars.iloc[i]
            if np.isfinite(scalar):
                confidence *= min(float(scalar), 1.0)  # a regime may only shrink

        if self._features is not None:
            row = self._features.iloc[[i]]
            if row.isna().to_numpy().any():
                return FLAT  # no features, no opinion, no trade
            conviction = float(self.meta.confidence(row)[0])
            if conviction <= 0.0:
                return FLAT  # the filter rejected it
            confidence *= conviction

        if confidence <= self.min_confidence:
            return FLAT
        return Intent(
            side=intent.side,
            stop_distance=intent.stop_distance,
            confidence=float(min(confidence, intent.confidence)),
            reason=f"{intent.reason} ml={confidence:.2f}",
            resize=intent.resize,
        )

    def describe(self) -> str:
        return f"{self.name}[{self.base.describe()}]"
