"""S1 with the AI layer bolted on. Tiers 1-3 of the build plan.

The base strategy is unchanged and still decides direction. The models only
change two things:

- **whether** the signal is taken (tier 2 filter)
- **how large** it is, via `Intent.confidence` (tiers 1 and 3)

`TrendML` deliberately subclasses `TrendFollowing` rather than reimplementing it,
so the comparison in the gauntlet is genuinely like-for-like: identical direction
logic, identical stops, and the only difference is the layer under test. If the
ML version wins, that win is attributable.

Neither model can widen a limit. `confidence` feeds the sizing formula, which the
risk engine still caps at `max_risk_per_trade`. The worst a broken model can do is
size to zero or refuse to trade -- never risk more than the register allows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import FLAT, Intent
from core.types import Position
from ml.models import MetaLabelModel, RegimeModel, meta_features
from strategies.trend import TrendFollowing


class TrendML(TrendFollowing):
    name = "S1_trend_ml"

    def __init__(
        self,
        regime: RegimeModel | None = None,
        meta: MetaLabelModel | None = None,
        use_regime: bool = True,
        use_meta: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.regime = regime
        self.meta = meta
        self.use_regime = use_regime and regime is not None
        self.use_meta = use_meta and meta is not None
        self._features: pd.DataFrame | None = None
        self._scalars: pd.Series | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        # Precomputed once over the whole frame. Safe only because every feature
        # is causal - `tests/test_ml.py::test_ml_features_are_causal` asserts it.
        if self.use_meta:
            self._features = meta_features(df)
        if self.use_regime:
            self._scalars = self.regime.scalars(df)
        return df

    def evaluate(self, df: pd.DataFrame, i: int, position: Position | None) -> Intent:
        base = super().evaluate(df, i, position)
        if base.flat:
            return base

        confidence = base.confidence

        if self.use_regime and self._scalars is not None:
            scalar = self._scalars.iloc[i]
            if np.isfinite(scalar):
                # Regime scales exposure but is capped at 1.0 here: sizing above
                # the configured risk is the risk engine's decision, never a
                # model's, and clamping at the source keeps that true.
                confidence *= min(float(scalar), 1.0)

        if self.use_meta and self._features is not None:
            row = self._features.iloc[[i]]
            if row.isna().to_numpy().any():
                return FLAT  # no features, no opinion, no trade
            conviction = float(self.meta.confidence(row)[0])
            if conviction <= 0.0:
                return FLAT  # the filter rejected it
            confidence *= conviction

        if confidence <= 0.05:
            return FLAT

        return Intent(
            side=base.side,
            stop_distance=base.stop_distance,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reason=f"{base.reason} conf={confidence:.2f}",
        )


def signal_events(strategy: TrendFollowing, df: pd.DataFrame) -> pd.DataFrame:
    """Every bar where the BASE strategy would have fired, for labelling.

    Meta-labelling trains on the trades the base strategy actually proposes, not
    on every bar. Sampling anything else answers a different question than the
    one the model is deployed to answer.
    """
    prepared = strategy.prepare(df.copy()).reset_index(drop=True)
    rows = []
    for i in range(strategy.warmup, len(prepared) - 1):
        intent = strategy.evaluate(prepared, i, None)
        if not intent.flat:
            rows.append({"i": i, "side": intent.side, "stop_distance": intent.stop_distance})
    return pd.DataFrame(rows)
