"""Tiers 1-3: regime classification, meta-labelling, adaptive sizing.

The ordering from the build plan, by return on effort:

- **Tier 1, regime** -- which strategy is allowed on right now. Unsupervised, so
  it cannot overfit to a target it never sees. Low risk, immediate value.
- **Tier 2, meta-label** -- given that the strategy fired, does this one work?
  The highest-value model available at this scale.
- **Tier 3, sizing** -- calibrated probability becomes a position multiplier.

Every model here outputs a *scalar or a filter*. None of them outputs an order,
a direction, or a lot size. A model can change the proposal; the risk register
still decides the limit. That boundary is the reason this layer can be added
without adding a new way to blow up.

Probabilities are **calibrated** (isotonic, cross-validated). An uncalibrated
gradient-boosting score is not a probability, and feeding one into a position
multiplier means sizing on a number that does not mean what it says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from features.indicators import atr, ema, realized_vol, rolling_percentile, rolling_return


class Regime(Enum):
    QUIET_TREND = "quiet_trend"
    VOLATILE_TREND = "volatile_trend"
    QUIET_RANGE = "quiet_range"
    VOLATILE_RANGE = "volatile_range"
    UNKNOWN = "unknown"

    @property
    def scalar(self) -> float:
        """Position multiplier. Trend systems want trend and hate volatile ranges."""
        return {
            Regime.QUIET_TREND: 1.3,
            Regime.VOLATILE_TREND: 1.0,
            Regime.QUIET_RANGE: 0.6,
            Regime.VOLATILE_RANGE: 0.4,
            Regime.UNKNOWN: 1.0,
        }[self]


# --------------------------------------------------------------------------- #
# Tier 1 -- regime
# --------------------------------------------------------------------------- #


def regime_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Two causal axes: how much it moves, and how directionally.

    Deliberately small. A regime model with thirty features is a regime model
    that has memorised the sample.
    """
    out = pd.DataFrame(index=df.index)
    a = atr(df, 14)
    out["vol"] = rolling_percentile(a / df["close"], 252)
    out["vol_change"] = a / a.rolling(period, min_periods=period).mean() - 1.0

    # Efficiency ratio: net movement over total path. High means trending.
    net = (df["close"] - df["close"].shift(period)).abs()
    path = df["close"].diff().abs().rolling(period, min_periods=period).sum()
    out["efficiency"] = net / path.replace(0, np.nan)
    out["trend"] = (df["close"] / ema(df["close"], period) - 1.0).abs()
    return out


@dataclass
class RegimeModel:
    """Unsupervised regime clustering.

    Unsupervised on purpose: there is no ground-truth regime label to overfit to,
    which makes this the safest place to start using machine learning at all.
    """

    n_states: int = 4
    period: int = 20
    _gmm: GaussianMixture | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _mapping: dict[int, Regime] = field(default_factory=dict, repr=False)

    def fit(self, df: pd.DataFrame) -> "RegimeModel":
        feats = regime_features(df, self.period).dropna()
        if len(feats) < self.n_states * 30:
            raise ValueError(f"need at least {self.n_states * 30} clean rows to fit regimes")

        self._scaler = StandardScaler().fit(feats.to_numpy())
        x = self._scaler.transform(feats.to_numpy())
        self._gmm = GaussianMixture(
            n_components=self.n_states, covariance_type="full",
            random_state=13, n_init=4,
        ).fit(x)

        # Name the clusters by where their centres sit, so the labels mean
        # something rather than being cluster 0 through 3.
        centres = self._scaler.inverse_transform(self._gmm.means_)
        vol_median = np.median(centres[:, 0])
        eff_median = np.median(centres[:, 2])
        for k, centre in enumerate(centres):
            volatile = centre[0] > vol_median
            trending = centre[2] > eff_median
            self._mapping[k] = (
                Regime.VOLATILE_TREND if volatile and trending
                else Regime.QUIET_TREND if trending
                else Regime.VOLATILE_RANGE if volatile
                else Regime.QUIET_RANGE
            )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._gmm is None or self._scaler is None:
            raise RuntimeError("RegimeModel is not fitted")
        feats = regime_features(df, self.period)
        valid = feats.dropna()
        out = pd.Series(Regime.UNKNOWN, index=df.index, dtype=object)
        if valid.empty:
            return out
        states = self._gmm.predict(self._scaler.transform(valid.to_numpy()))
        out.loc[valid.index] = [self._mapping.get(int(s), Regime.UNKNOWN) for s in states]
        return out

    def scalars(self, df: pd.DataFrame) -> pd.Series:
        return self.predict(df).map(lambda r: r.scalar).astype(float)


# --------------------------------------------------------------------------- #
# Tier 2 -- meta-labelling
# --------------------------------------------------------------------------- #


def meta_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """State of the world at signal time.

    Not price levels and not raw returns -- those do not generalise across the
    years you are training over. Everything here is scale-free: percentiles,
    ratios and distances measured in ATR.
    """
    out = pd.DataFrame(index=df.index)
    a = atr(df, 14)

    out["atr_pct"] = rolling_percentile(a, 252)
    out["vol_ratio"] = a / a.rolling(period, min_periods=period).mean()
    out["realized_vol"] = realized_vol(df["close"], period)

    for lb in (5, 20, 60):
        out[f"mom_{lb}"] = rolling_return(df["close"], lb)

    out["dist_ema_atr"] = (df["close"] - ema(df["close"], 50)) / a
    out["range_ratio"] = (df["high"] - df["low"]) / a

    net = (df["close"] - df["close"].shift(period)).abs()
    path = df["close"].diff().abs().rolling(period, min_periods=period).sum()
    out["efficiency"] = net / path.replace(0, np.nan)

    body = (df["close"] - df["open"]).abs()
    out["body_ratio"] = body / (df["high"] - df["low"]).replace(0, np.nan)
    out["dow"] = pd.to_datetime(df["ts"]).dt.dayofweek
    return out


@dataclass
class MetaLabelModel:
    """Predicts P(this signal works), then filters and sizes on it.

    Note what it is *not* asked: which way price will move. The strategy already
    decided that, and the sample is already restricted to the cases that matter,
    which is why this question is learnable when direction is not.
    """

    threshold: float = 0.50
    min_samples: int = 150
    _model: CalibratedClassifierCV | None = field(default=None, repr=False)
    _columns: list[str] = field(default_factory=list, repr=False)

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        cv=None,
    ) -> "MetaLabelModel":
        if len(x) < self.min_samples:
            raise ValueError(
                f"{len(x)} samples is below the {self.min_samples} minimum. A model "
                f"fitted on fewer is memorising, not learning."
            )
        if len(np.unique(y)) < 2:
            raise ValueError("labels are all one class - nothing to learn")

        self._columns = list(x.columns)
        base = HistGradientBoostingClassifier(
            max_iter=180, max_depth=3, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=1.0, random_state=7,
        )
        # Isotonic calibration on a purged splitter where one is supplied. An
        # uncalibrated score is not a probability, and tier 3 sizes on it.
        self._model = CalibratedClassifierCV(base, method="isotonic", cv=cv or 3)
        self._model.fit(x.to_numpy(), y, sample_weight=sample_weight)
        return self

    def probability(self, x: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MetaLabelModel is not fitted")
        aligned = x[self._columns].to_numpy()
        return self._model.predict_proba(aligned)[:, 1]

    def confidence(self, x: pd.DataFrame) -> np.ndarray:
        """Map probability to a position multiplier in [0, 1].

        Below the threshold the answer is zero, not a small size: a signal the
        model does not believe should be skipped, and sizing it at a hair just
        pays the spread for the privilege of being unsure.
        """
        p = self.probability(x)
        span = max(1.0 - self.threshold, 1e-9)
        return np.where(p < self.threshold, 0.0, np.clip((p - self.threshold) / span, 0.0, 1.0))


@dataclass(frozen=True)
class CalibrationReport:
    bins: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    counts: np.ndarray
    brier: float

    def __str__(self) -> str:
        lines = [f"calibration (Brier {self.brier:.4f} - lower is better)",
                 f"  {'predicted':>10}{'observed':>10}{'n':>8}"]
        for p, o, n in zip(self.predicted, self.observed, self.counts):
            if n == 0:
                continue
            flag = "  <-- overconfident" if p - o > 0.10 else ""
            lines.append(f"  {p:>10.1%}{o:>10.1%}{int(n):>8}{flag}")
        return "\n".join(lines)


def calibration_report(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10):
    """Do predicted probabilities match observed frequencies?

    If the model says 70% and is right 45% of the time, tier 3 is sizing up on a
    lie. This is the check that catches it.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)

    predicted = np.zeros(bins)
    observed = np.zeros(bins)
    counts = np.zeros(bins)
    for b in range(bins):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b]:
            predicted[b] = p[mask].mean()
            observed[b] = y[mask].mean()

    return CalibrationReport(
        bins=edges, predicted=predicted, observed=observed, counts=counts,
        brier=float(np.mean((p - y) ** 2)),
    )
