"""Phase 4: train tiers 1-3, then put both versions through the gauntlet.

    python scripts/train_and_validate.py --symbol EURUSD --synthetic
    python scripts/train_and_validate.py --symbol EURUSD --timeframe D1

The comparison is the point. The ML version is trained on the first 60% of the
data and both versions are then scored on the untouched remainder, so the baseline
is a real benchmark rather than a formality.

If the ML version does not beat the baseline out-of-sample after costs, **ship the
baseline**. That is a genuine result, not a failure, and it is the most likely
outcome at this scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from backtest.metrics import compute, report  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from data.store import BarStore  # noqa: E402
from execution.paper import FIXTURE_SPECS  # noqa: E402
from ml.cv import CombinatorialPurgedCV, PurgedKFold, leakage_report  # noqa: E402
from ml.labeling import LabelConfig, label_summary, sample_weights, triple_barrier_labels  # noqa: E402
from ml.models import MetaLabelModel, RegimeModel, calibration_report, meta_features  # noqa: E402
from research.gauntlet import GauntletThresholds, run_gauntlet  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402
from strategies.trend_ml import TrendML, signal_events  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="D1")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--root", default="data/bars")
    ap.add_argument("--train-fraction", type=float, default=0.60)
    ap.add_argument("--trials", type=int, default=1,
                    help="honest count of configurations tried across ALL research")
    ap.add_argument("--profile", default="challenge")
    args = ap.parse_args()

    if args.synthetic:
        from scripts.backtest import synthetic
        df = synthetic(args.symbol, bars=2200)
        source = "SYNTHETIC"
    else:
        df = BarStore(args.root).read(args.symbol, args.timeframe)
        source = f"{args.root}/{args.symbol}/{args.timeframe}"
        if df.empty:
            print(f"No data at {source}. Run download_history.py, or use --synthetic.")
            return 1

    spec = FIXTURE_SPECS[args.symbol]
    profile = RiskProfile.load(args.profile)
    costs = CostModel()

    split = int(len(df) * args.train_fraction)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    print(f"\ndata   : {source}, {len(df):,} bars")
    print(f"split  : train {len(train_df):,} ({train_df['ts'].iloc[0]:%Y-%m-%d} -> "
          f"{train_df['ts'].iloc[-1]:%Y-%m-%d})")
    print(f"         test  {len(test_df):,} ({test_df['ts'].iloc[0]:%Y-%m-%d} -> "
          f"{test_df['ts'].iloc[-1]:%Y-%m-%d})  <- never seen during training")

    # ---------------------------------------------------------------- tier 1
    print("\n" + "=" * 84)
    print("TIER 1  regime model (unsupervised)")
    print("=" * 84)
    regime = RegimeModel().fit(train_df)
    states = regime.predict(train_df)
    counts = states.value_counts()
    for state, n in counts.items():
        print(f"  {state.value:<18}{n:>7,} bars  {n / len(states):>6.1%}   "
              f"scalar {state.scalar:.2f}")

    # ---------------------------------------------------------------- labels
    print("\n" + "=" * 84)
    print("TIER 2  meta-labelling")
    print("=" * 84)
    base = TrendFollowing()
    events = signal_events(base, train_df)
    if events.empty:
        print("  base strategy produced no signals on the training set; nothing to label")
        return 1

    prepared = base.prepare(train_df.copy()).reset_index(drop=True)
    labels = triple_barrier_labels(prepared, events, LabelConfig(profit_multiple=2.0, max_bars=20))
    print(label_summary(labels))

    if len(labels) < 150:
        print(f"\n  only {len(labels)} labels - below the 150 minimum. More data, or a")
        print("  strategy that signals more often, before any of this means anything.")
        return 1

    feats = meta_features(prepared)
    x = feats.iloc[labels["i"].to_numpy()].reset_index(drop=True)
    y = labels["label"].to_numpy()

    keep = ~x.isna().any(axis=1).to_numpy()
    x, y, labels = x[keep].reset_index(drop=True), y[keep], labels[keep].reset_index(drop=True)
    print(f"  {len(x):,} usable samples x {x.shape[1]} features")

    # ------------------------------------------------------------ purged CV
    t0 = labels["entry_i"].to_numpy()
    t1 = labels["t1"].to_numpy()
    print("\n  purged cross-validation")
    for line in leakage_report(t0, t1, PurgedKFold(n_splits=5, embargo_pct=0.01)).splitlines():
        print(f"    {line}")

    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    print(f"    CPCV generates {cpcv.n_paths()} train/test paths")

    weights = sample_weights(labels, len(prepared))
    splits = list(PurgedKFold(n_splits=4, embargo_pct=0.01).split(t0, t1))
    meta = MetaLabelModel(threshold=0.55).fit(x, y, sample_weight=weights, cv=splits)

    # Out-of-fold calibration: in-sample probabilities would look perfect and
    # mean nothing.
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
            continue
        fold = MetaLabelModel(threshold=0.55, min_samples=20).fit(
            x.iloc[train_idx], y[train_idx], sample_weight=weights[train_idx],
            t0=t0[train_idx], t1=t1[train_idx],  # purged inside the fold too
        )
        oof[test_idx] = fold.probability(x.iloc[test_idx])

    valid = np.isfinite(oof)
    if valid.sum() > 30:
        print()
        for line in str(calibration_report(oof[valid], y[valid])).splitlines():
            print(f"  {line}")

    # ---------------------------------------------------------- the comparison
    print("\n" + "=" * 84)
    print("OUT-OF-SAMPLE COMPARISON  (test set, never seen in training)")
    print("=" * 84)

    def make_base():
        return TrendFollowing()

    def make_ml():
        return TrendML(regime=regime, meta=meta)

    from research.gauntlet import _run

    base_result = _run(make_base(), test_df, spec, profile, costs)
    ml_result = _run(make_ml(), test_df, spec, profile, costs)
    bm, mm = compute(base_result), compute(ml_result)

    print(report(base_result, bm))
    print(report(ml_result, mm))

    print("\n" + "-" * 84)
    print(f"  {'metric':<22}{'baseline':>14}{'with ML':>14}{'verdict':>16}")
    print("-" * 84)
    for name, b, v, better_high in [
        ("trades", bm.trades, mm.trades, None),
        ("expectancy (R)", bm.expectancy_r, mm.expectancy_r, True),
        ("sharpe", bm.sharpe, mm.sharpe, True),
        ("max drawdown", bm.max_drawdown, mm.max_drawdown, False),
        ("cost drag", bm.cost_drag, mm.cost_drag, False),
    ]:
        if better_high is None:
            verdict = ""
        elif better_high:
            verdict = "ML better" if v > b else "baseline better"
        else:
            verdict = "ML better" if v < b else "baseline better"
        fmt = "{:>14.3f}" if abs(b) < 100 else "{:>14,.0f}"
        print(f"  {name:<22}" + fmt.format(b) + fmt.format(v) + f"{verdict:>16}")

    ml_wins = mm.expectancy_r > bm.expectancy_r and mm.sharpe > bm.sharpe
    print("-" * 84)
    if ml_wins:
        print("  ML beats the baseline out-of-sample. Now prove it is not luck.")
    else:
        print("  ML does NOT beat the baseline out-of-sample.")
        print("  Ship the baseline. This is a real result, and the common one.")

    # ---------------------------------------------------------------- gauntlet
    winner, label = (make_ml, "ML") if ml_wins else (make_base, "baseline")
    variants = [
        lambda: TrendFollowing(lookback=40),
        lambda: TrendFollowing(lookback=80),
        lambda: TrendFollowing(atr_stop_multiple=2.0),
        lambda: TrendFollowing(atr_stop_multiple=3.0),
    ]
    print(f"\nrunning the gauntlet on the {label}")
    outcome = run_gauntlet(
        winner, test_df, spec, profile, costs,
        thresholds=GauntletThresholds(),
        n_trials_tested=max(args.trials, len(variants) + 1),
        variant_factories=variants,
    )
    print(outcome.report())
    return 0 if outcome.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
