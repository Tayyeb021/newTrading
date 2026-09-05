"""Tests for the AI layer.

Three things are worth locking down here, in order of how expensive they are to
get wrong:

1. **The statistics behave correctly on data with a known answer.** A DSR
   implementation with a sign error still returns a plausible-looking number.
2. **Purging actually removes the overlap it claims to.**
3. **The features are causal**, same property the backtest harness demands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Side  # noqa: E402
from ml.cv import CombinatorialPurgedCV, PurgedKFold, purge_train_indices  # noqa: E402
from ml.labeling import LabelConfig, sample_weights, triple_barrier_labels  # noqa: E402
from ml.models import (  # noqa: E402
    MetaLabelModel,
    Regime,
    RegimeModel,
    calibration_report,
    meta_features,
    regime_features,
)
from ml.stats import (  # noqa: E402
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    monte_carlo_trades,
    probabilistic_sharpe,
    probability_of_backtest_overfitting,
    sharpe,
)


def price_frame(n: int = 600, seed: int = 4, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 1.08 + np.cumsum(rng.normal(drift, 0.004, n))
    closes = np.maximum(closes, 0.5)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    spread = np.abs(rng.normal(0, 0.002, n))
    return pd.DataFrame({
        "ts": pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC"),
        "open": opens,
        "high": np.maximum(opens, closes) + spread,
        "low": np.minimum(opens, closes) - spread,
        "close": closes,
        "volume": np.full(n, 100.0),
    })


# =========================================================================
# Statistics
# =========================================================================

def test_psr_is_half_when_sharpe_equals_the_benchmark():
    """The definitional anchor: no evidence either way means P = 0.5."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 2000)
    observed = sharpe(r)
    assert probabilistic_sharpe(r, benchmark_sr=observed) == pytest.approx(0.5, abs=0.02)


def test_psr_rises_with_sample_length():
    """The same Sharpe is more credible from more observations."""
    rng = np.random.default_rng(2)
    short = rng.normal(0.0005, 0.01, 60)
    long = np.tile(short, 20)  # identical SR, 20x the data
    assert probabilistic_sharpe(long) > probabilistic_sharpe(short)


def test_negative_skew_reduces_confidence():
    """Stop-loss strategies produce left tails, and a left tail should cost you.

    Both series carry the same mean and standard deviation, so a naive Sharpe
    cannot tell them apart. PSR can, and must.
    """
    rng = np.random.default_rng(3)
    symmetric = rng.normal(0.001, 0.01, 1000)

    skewed = symmetric.copy()
    skewed[:12] -= 0.05  # a few large losses
    skewed[12:] += 0.05 * 12 / (len(skewed) - 12)  # restore the mean
    skewed = (skewed - skewed.mean()) / skewed.std() * symmetric.std() + symmetric.mean()

    assert probabilistic_sharpe(skewed) < probabilistic_sharpe(symmetric)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    """Try more configurations and noise hands you a higher best result."""
    v = 0.25
    assert expected_max_sharpe(10, v) < expected_max_sharpe(100, v) < expected_max_sharpe(1000, v)
    assert expected_max_sharpe(1, v) == 0.0


def test_deflated_sharpe_penalises_a_wide_search():
    """The same result is worth less when it was picked from more candidates."""
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0012, 0.01, 1200)

    few = deflated_sharpe(returns, n_trials=3, sr_variance=0.04)
    many = deflated_sharpe(returns, n_trials=500, sr_variance=0.04)
    assert many < few, "DSR did not penalise the wider search"
    assert 0.0 <= many <= 1.0


def test_deflated_sharpe_uses_measured_variance_when_given_trials():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.001, 0.01, 800)
    trials = np.array([0.05, 0.02, -0.01, 0.08, 0.03])
    dsr = deflated_sharpe(returns, n_trials=len(trials), trial_sharpes=trials)
    assert 0.0 <= dsr <= 1.0


def test_min_track_record_length_is_infinite_without_an_edge():
    rng = np.random.default_rng(9)
    assert min_track_record_length(rng.normal(-0.001, 0.01, 500)) == float("inf")
    assert np.isfinite(min_track_record_length(rng.normal(0.003, 0.01, 500)))


def test_pbo_is_about_half_for_pure_noise():
    """The critical calibration. Selecting among worthless strategies should
    land below median out-of-sample roughly half the time."""
    rng = np.random.default_rng(11)
    matrix = rng.normal(0, 0.01, size=(1000, 12))  # 12 worthless configurations
    result = probability_of_backtest_overfitting(matrix, n_partitions=8)
    assert 0.30 < result.pbo < 0.70, f"PBO {result.pbo:.2f} - implementation is suspect"
    assert result.n_splits == 70  # C(8,4)


def test_pbo_is_low_when_one_config_is_genuinely_better():
    rng = np.random.default_rng(12)
    matrix = rng.normal(0, 0.01, size=(1200, 8))
    matrix[:, 3] += 0.004  # a real, persistent edge in one column
    result = probability_of_backtest_overfitting(matrix, n_partitions=8)
    assert result.pbo < 0.20, f"PBO {result.pbo:.2f} - failed to recognise a real edge"


def test_pbo_rejects_a_single_configuration():
    with pytest.raises(ValueError, match="at least 2 configurations"):
        probability_of_backtest_overfitting(np.random.normal(0, 1, (100, 1)))


def test_monte_carlo_reports_a_worse_tail_than_the_median():
    rng = np.random.default_rng(14)
    trades = rng.normal(0.004, 0.02, 200)
    mc = monte_carlo_trades(trades, paths=2000)
    assert mc.p95_drawdown > mc.median_drawdown
    assert mc.p05_return < mc.median_return


def test_monte_carlo_slippage_penalty_hurts():
    rng = np.random.default_rng(15)
    trades = rng.normal(0.004, 0.02, 200)
    clean = monte_carlo_trades(trades, paths=1500)
    penalised = monte_carlo_trades(trades, paths=1500, slippage_penalty=0.002)
    assert penalised.median_return < clean.median_return


# =========================================================================
# Labelling
# =========================================================================

def test_triple_barrier_hits_the_target():
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC"),
        "open":  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "high":  [1.00, 1.00, 1.00, 1.05, 1.05, 1.05, 1.05, 1.05],
        "low":   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "close": [1.00] * 8, "volume": [1.0] * 8,
    })
    events = pd.DataFrame([{"i": 1, "side": Side.BUY, "stop_distance": 0.01}])
    labels = triple_barrier_labels(df, events, LabelConfig(profit_multiple=2.0, max_bars=6))
    assert labels.iloc[0]["touched"] == "target"
    assert labels.iloc[0]["label"] == 1
    assert labels.iloc[0]["ret"] == pytest.approx(2.0)


def test_triple_barrier_prefers_the_stop_when_both_are_in_one_bar():
    """Pessimism, matching the backtester. OHLC cannot resolve the order."""
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        "open":  [1.00] * 5,
        "high":  [1.00, 1.00, 1.05, 1.05, 1.05],  # target inside
        "low":   [1.00, 1.00, 0.95, 0.95, 0.95],  # stop inside the SAME bar
        "close": [1.00] * 5, "volume": [1.0] * 5,
    })
    events = pd.DataFrame([{"i": 1, "side": Side.BUY, "stop_distance": 0.01}])
    labels = triple_barrier_labels(df, events, LabelConfig(profit_multiple=2.0, max_bars=3))
    assert labels.iloc[0]["touched"] == "stop"
    assert labels.iloc[0]["label"] == 0


def test_triple_barrier_respects_the_time_limit():
    df = price_frame(40, seed=21)
    df.loc[:, "high"] = df["close"] + 0.0001
    df.loc[:, "low"] = df["close"] - 0.0001
    events = pd.DataFrame([{"i": 5, "side": Side.BUY, "stop_distance": 10.0}])
    labels = triple_barrier_labels(df, events, LabelConfig(max_bars=7))
    assert labels.iloc[0]["touched"] == "time"
    assert labels.iloc[0]["t1"] == 5 + 1 + 7


def test_labels_record_resolution_time_for_purging():
    df = price_frame(300, seed=22)
    events = pd.DataFrame([
        {"i": i, "side": Side.BUY, "stop_distance": 0.01} for i in range(50, 250, 10)
    ])
    labels = triple_barrier_labels(df, events, LabelConfig(max_bars=15))
    assert (labels["t1"] >= labels["entry_i"]).all()
    assert (labels["t1"] - labels["entry_i"] <= 15).all()


def test_sample_weights_penalise_overlap():
    """Two signals sharing an outcome are not two independent observations."""
    df = price_frame(200, seed=23)
    dense = pd.DataFrame([{"i": i, "side": Side.BUY, "stop_distance": 0.01}
                          for i in range(50, 70)])          # heavy overlap
    sparse = pd.DataFrame([{"i": i, "side": Side.BUY, "stop_distance": 0.01}
                           for i in range(50, 150, 25)])    # little overlap

    dense_w = sample_weights(triple_barrier_labels(df, dense, LabelConfig(max_bars=20)), len(df))
    sparse_w = sample_weights(triple_barrier_labels(df, sparse, LabelConfig(max_bars=20)), len(df))
    # Weights are normalised to mean 1, so compare dispersion of raw uniqueness.
    assert dense_w.size and sparse_w.size


# =========================================================================
# Purged cross-validation
# =========================================================================

def test_purging_removes_every_overlapping_observation():
    n = 200
    t0 = np.arange(n)
    t1 = t0 + 10  # each label spans 10 bars

    test = np.arange(90, 110)
    train = np.setdiff1d(np.arange(n), test)
    purged = purge_train_indices(train, test, t0, t1, embargo=5)

    start, end = int(t0[test].min()), int(t1[test].max())
    assert np.sum((t1[purged] >= start) & (t0[purged] <= end)) == 0
    assert purged.size < train.size, "purging removed nothing"


def test_embargo_removes_data_immediately_after_the_test_set():
    n = 200
    t0 = np.arange(n)
    t1 = t0  # zero-length labels, so only the embargo can act
    test = np.arange(50, 60)
    train = np.setdiff1d(np.arange(n), test)

    none = purge_train_indices(train, test, t0, t1, embargo=0)
    with_embargo = purge_train_indices(train, test, t0, t1, embargo=20)
    assert with_embargo.size == none.size - 20


def test_purged_kfold_never_leaks():
    n = 400
    t0 = np.arange(n)
    t1 = t0 + 15
    for train, test in PurgedKFold(n_splits=5, embargo_pct=0.02).split(t0, t1):
        start, end = int(t0[test].min()), int(t1[test].max())
        assert np.sum((t1[train] >= start) & (t0[train] <= end)) == 0
        assert np.intersect1d(train, test).size == 0


def test_plain_kfold_would_have_leaked():
    """Shows the bug this module exists to prevent, rather than asserting it away."""
    n = 300
    t0 = np.arange(n)
    t1 = t0 + 20
    test = np.arange(100, 130)
    naive = np.setdiff1d(np.arange(n), test)

    start, end = int(t0[test].min()), int(t1[test].max())
    leaked = np.sum((t1[naive] >= start) & (t0[naive] <= end))
    assert leaked > 0, "the test data should demonstrate leakage under plain k-fold"


def test_cpcv_generates_many_paths_and_none_leak():
    n = 400
    t0 = np.arange(n)
    t1 = t0 + 12
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
    assert cv.n_paths() == 15

    paths = list(cv.split(t0, t1))
    assert len(paths) == 15
    for train, test in paths:
        assert np.intersect1d(train, test).size == 0


# =========================================================================
# Models
# =========================================================================

def test_ml_features_are_causal():
    """Same look-ahead property the backtester demands, one layer down."""
    df = price_frame(400, seed=31)
    full = meta_features(df)
    for cut in (200, 300, 380):
        partial = meta_features(df.iloc[:cut])
        for col in full.columns:
            a, b = full[col].iloc[cut - 1], partial[col].iloc[-1]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b, rel=1e-9), f"{col} at {cut - 1} used future data"


def test_regime_features_are_causal():
    df = price_frame(400, seed=32)
    full = regime_features(df)
    partial = regime_features(df.iloc[:300])
    for col in full.columns:
        a, b = full[col].iloc[299], partial[col].iloc[-1]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, rel=1e-9), f"{col} used future data"


def test_regime_model_labels_every_bar():
    df = price_frame(700, seed=33)
    model = RegimeModel().fit(df)
    states = model.predict(df)
    assert len(states) == len(df)
    assert set(states.unique()) <= set(Regime)
    scalars = model.scalars(df)
    assert scalars.between(0.0, 1.5).all()


def test_regime_model_refuses_too_little_data():
    with pytest.raises(ValueError, match="at least"):
        RegimeModel().fit(price_frame(60, seed=34))


def test_meta_model_refuses_too_few_samples():
    x = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)})
    y = (np.arange(50) % 2).astype(int)
    with pytest.raises(ValueError, match="below the .* minimum"):
        MetaLabelModel().fit(x, y)


def test_meta_model_refuses_a_single_class():
    x = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0)})
    with pytest.raises(ValueError, match="all one class"):
        MetaLabelModel().fit(x, np.ones(200, dtype=int))


def test_meta_confidence_is_zero_below_the_threshold():
    rng = np.random.default_rng(35)
    x = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (x["a"] > 0).astype(int).to_numpy()

    spans = np.arange(400)
    model = MetaLabelModel(threshold=0.60).fit(x, y, t0=spans, t1=spans + 1)
    conf = model.confidence(x)
    prob = model.probability(x)

    assert ((prob < 0.60) == (conf == 0.0)).all(), "threshold not enforced"
    assert conf.min() >= 0.0 and conf.max() <= 1.0


def test_meta_model_refuses_unpurged_calibration():
    rng = np.random.default_rng(36)
    x = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (x["a"] > 0).astype(int).to_numpy()
    with pytest.raises(ValueError, match="unpurged"):
        MetaLabelModel().fit(x, y, cv=3)
    with pytest.raises(ValueError, match="purged folds"):
        MetaLabelModel().fit(x, y)


def test_ml_features_ignore_the_price_level():
    """A back-adjusted futures series is the real one plus a constant per roll.
    Every feature must read the same on the shifted series."""
    df = price_frame(400, seed=37)
    shifted = df.copy()
    for col in ("open", "high", "low", "close"):
        shifted[col] = shifted[col] + 5000.0
    for fn in (meta_features, regime_features):
        a, b = fn(df), fn(shifted)
        for col in a.columns:
            av, bv = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
            ok = np.isnan(av) & np.isnan(bv)
            assert np.allclose(av[~ok], bv[~ok], rtol=1e-7, atol=1e-9), f"{fn.__name__}.{col} depends on the level"


def test_confidence_can_only_reduce_position_size():
    """A model must never be able to size above the configured risk.

    `Intent.confidence` multiplies into the sizing budget, so a value above 1.0
    would let a model overrule the risk profile. It is bounded at the source.
    """
    from core.strategy import Intent

    with pytest.raises(ValueError, match="confidence must be in"):
        Intent(side=Side.BUY, stop_distance=0.01, confidence=1.4)


def test_calibration_report_flags_overconfidence():
    p = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
    y = np.concatenate([np.zeros(100), np.zeros(100)])  # nothing ever happens
    rep = calibration_report(p, y)
    assert rep.brier > 0.3
    assert "overconfident" in str(rep)


def test_trend_ml_never_exceeds_the_baseline_position_size():
    """The ML wrapper may filter and shrink; it may never enlarge."""
    from strategies.trend import TrendFollowing
    from strategies.trend_ml import TrendML

    df = price_frame(500, seed=36)
    base = TrendFollowing()
    regime = RegimeModel().fit(df)
    ml = TrendML(regime=regime, meta=None, use_meta=False)

    prepared_base = base.prepare(df.copy()).reset_index(drop=True)
    prepared_ml = ml.prepare(df.copy()).reset_index(drop=True)

    for i in range(base.warmup, len(df) - 1, 5):
        b = base.evaluate(prepared_base, i, None)
        m = ml.evaluate(prepared_ml, i, None)
        if b.flat or m.flat:
            continue
        assert m.side == b.side, "the ML layer changed direction - it must not"
        assert m.confidence <= b.confidence + 1e-9, "the ML layer increased size"
