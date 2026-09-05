"""Entry 013: meta-labelling the monthly trend book, judged out of sample.

    python research/meta_label_013.py                # wide universe, split 2021-01-01
    python research/meta_label_013.py --universe core

The question is the one the AI layer was built to answer: given that the
monthly trend rule just fired, is THIS trade likely to work, and does skipping
the ones it doubts improve the book after costs on data it never saw?

Steps, in order, none of which looks forward:

1. For every market and speed, collect the rule's decision-day entries.
2. Label each by whether the trade made money by the next decision (21 bars)
   or the 4-ATR stop, whichever came first. The same label the book trades.
3. Features at decision time, shift-invariant, plus the rule's own forecast
   strength and speed. One function, shared with the live wrapper.
4. Train on events RESOLVED before the split, calibrate on purged folds,
   score on events ENTERED after it. Events that straddle the split are dropped.
5. Run the plain book and the filtered book on the same window, same costs,
   same equity, and judge from the split date onward.

Thresholds are in RESEARCH_LOG 013 and were written before this ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "research"))
sys.path.append(str(ROOT / "scripts"))

from backtest.metrics import _drawdown  # noqa: E402
from backtest.portfolio import portfolio_report  # noqa: E402
from core.contracts import CORE_UNIVERSE, FULL_UNIVERSE  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from futures_gauntlet import load_universe, run_book  # noqa: E402
from ml.cv import PurgedKFold, leakage_report  # noqa: E402
from ml.labeling import LabelConfig, sample_weights, triple_barrier_labels  # noqa: E402
from ml.models import MetaLabelModel, calibration_report  # noqa: E402
from ml.stats import sharpe  # noqa: E402
from strategies.ml_filter import MLFiltered, event_features, signal_events  # noqa: E402
from strategies.tsmom import TSMOM  # noqa: E402

HORIZON_BARS = 21  # one month: the rule holds to the next decision


def collect_events(bars: dict[str, pd.DataFrame], lookbacks: list[int]):
    """Every decision-day entry across markets and speeds, labelled and featured."""
    labelled, featured = [], []
    for name, df in bars.items():
        for lb in lookbacks:
            base = TSMOM(lookback=lb)
            prepared = base.prepare(df.copy()).reset_index(drop=True)
            events = signal_events(base, prepared)
            if events.empty:
                continue
            labels = triple_barrier_labels(
                prepared, events, LabelConfig(profit_multiple=1e6, max_bars=HORIZON_BARS, min_return=0.0))
            if labels.empty:
                continue
            feats = event_features(prepared, base).iloc[labels["i"].to_numpy()].reset_index(drop=True)
            days = pd.to_datetime(prepared["ts"]).dt.date.to_numpy()
            labels = labels.assign(
                market=name, lookback=lb,
                t0_date=days[labels["i"].to_numpy()], t1_date=days[labels["t1"].to_numpy()],
                weight=sample_weights(labels, len(prepared)),
            )
            labelled.append(labels)
            featured.append(feats)
    events = pd.concat(labelled, ignore_index=True)
    feats = pd.concat(featured, ignore_index=True)
    # TIME order, not market order. The purged splitter takes contiguous blocks
    # of rows as folds; pooled by market, every block spanned the whole decade
    # and purging removed the entire training set.
    order = np.argsort(np.array([d.toordinal() for d in events["t0_date"]]), kind="stable")
    return events.iloc[order].reset_index(drop=True), feats.iloc[order].reset_index(drop=True)


def book_metrics(result, split: date) -> dict:
    eq = result.equity.dropna()
    eq = eq[eq.index >= pd.Timestamp(split, tz="UTC")]
    rets = eq.pct_change().dropna().to_numpy()
    trades = [t for t in result.trades if pd.Timestamp(t.exit_ts) >= pd.Timestamp(split, tz="UTC")]
    gross = sum(t.gross_pnl for t in trades)
    friction = sum(t.costs for t in trades)
    return {
        "net_sharpe": float(sharpe(rets) * np.sqrt(252)) if rets.size > 1 else 0.0,
        "max_drawdown": float(_drawdown(eq)[0]) if len(eq) else 0.0,
        "trades": len(trades),
        "gross_pnl": gross, "friction": friction,
        "friction_share": friction / abs(gross) if gross else float("inf"),
        "final_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["core", "wide"], default="wide")
    ap.add_argument("--lookbacks", nargs="+", type=int, default=[60, 120, 250])
    ap.add_argument("--split", default="2021-01-01")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--equity", type=float, default=20_000_000.0)
    ap.add_argument("--stress", type=float, default=2.0)
    ap.add_argument("--profile", default="research")
    ap.add_argument("--data", default="data/futures")
    args = ap.parse_args()

    split = date.fromisoformat(args.split)
    names = list(CORE_UNIVERSE if args.universe == "core" else FULL_UNIVERSE)
    print(f"{len(names)} markets ({args.universe}), speeds {args.lookbacks}, split {split}, "
          f"threshold {args.threshold}, costs x{args.stress:g}")

    # ------------------------------------------------------------- events
    bars, specs, trade = load_universe(2011, Path(args.data), "full", names)
    events, x = collect_events(bars, args.lookbacks)
    valid = x.notna().all(axis=1).to_numpy()
    y = (events["ret"] > 0).astype(int).to_numpy()
    train = valid & (events["t1_date"] < split).to_numpy()
    test = valid & (events["t0_date"] >= split).to_numpy()
    print(f"events: {len(events):,} total, {valid.sum():,} with features, "
          f"{train.sum():,} train (resolved before {split}), {test.sum():,} test (entered after)")
    print(f"base rate P(trade makes money): train {y[train].mean():.3f}, test {y[test].mean():.3f}")

    # ------------------------------------------------------------- training
    t0 = np.array([d.toordinal() for d in events["t0_date"]])
    t1 = np.array([d.toordinal() for d in events["t1_date"]])
    splits = list(PurgedKFold(n_splits=4, embargo_pct=0.01).split(t0[train], t1[train]))
    for line in leakage_report(t0[train], t1[train], PurgedKFold(n_splits=4, embargo_pct=0.01)).splitlines():
        print(f"  {line}")
    model = MetaLabelModel(threshold=args.threshold).fit(
        x[train], y[train], sample_weight=events["weight"].to_numpy()[train], cv=splits)

    # ------------------------------------------------------ out of sample
    p = model.probability(x[test])
    yt = y[test]
    brier = float(np.mean((p - yt) ** 2))
    brier_base = float(np.mean((y[train].mean() - yt) ** 2))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yt, p))
    except Exception:  # noqa: BLE001
        auc = float("nan")
    kept = float(np.mean(p >= args.threshold))
    print(f"\nout of sample: Brier {brier:.4f} vs base-rate Brier {brier_base:.4f}, AUC {auc:.3f}, "
          f"entries kept {kept:.0%}")
    for line in str(calibration_report(p, yt)).splitlines():
        print(f"  {line}")

    # ---------------------------------------------------------- the books
    bars19, specs19, trade19 = load_universe(2019, Path(args.data), "full", names)
    plain = [Sleeve(f"tsmom{lb}", (lambda s, lb=lb: TSMOM(lookback=lb)), tuple(names), timeframe="D1")
             for lb in args.lookbacks]
    filtered = [Sleeve(f"mltsmom{lb}", (lambda s, lb=lb: MLFiltered(TSMOM(lookback=lb), meta=model)),
                       tuple(names), timeframe="D1") for lb in args.lookbacks]
    res_plain = run_book(bars19, specs19, trade19, plain, args.equity, args.profile, args.stress)
    res_ml = run_book(bars19, specs19, trade19, filtered, args.equity, args.profile, args.stress)
    print(portfolio_report(res_plain))
    print(portfolio_report(res_ml))
    mp, mm = book_metrics(res_plain, split), book_metrics(res_ml, split)

    rows = [
        ("1. out-of-sample Brier < base-rate Brier", brier < brier_base, f"{brier:.4f} vs {brier_base:.4f}"),
        ("2. filtered net Sharpe >= plain + 0.05 from the split", mm["net_sharpe"] >= mp["net_sharpe"] + 0.05,
         f"{mm['net_sharpe']:.2f} vs {mp['net_sharpe']:.2f}"),
        ("3. filtered max drawdown <= plain's", mm["max_drawdown"] <= mp["max_drawdown"],
         f"{mm['max_drawdown']:.1%} vs {mp['max_drawdown']:.1%}"),
        ("4. filter keeps >= 50% of entries", kept >= 0.5, f"{kept:.0%}"),
        ("family bar, for the record: filtered net Sharpe >= 0.40", mm["net_sharpe"] >= 0.40, f"{mm['net_sharpe']:.2f}"),
    ]
    print(f"\nVERDICT, entry 013 (from {split})")
    print("=" * 70)
    for label, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL':<5} {label:<58} {detail}")
    passed = all(ok for label, ok, _ in rows if label[0].isdigit())
    print(f"\n  {'ALL THRESHOLDS PASSED' if passed else 'FAILED - ship the plain rule'}")

    out = {"plain": mp, "filtered": mm, "brier": brier, "brier_base": brier_base, "auc": auc, "kept": kept,
           "events": {"total": int(len(events)), "train": int(train.sum()), "test": int(test.sum())},
           "verdict": [{"test": l, "pass": bool(ok), "detail": d} for l, ok, d in rows], "passed": passed,
           "args": vars(args)}
    Path("state").mkdir(exist_ok=True)
    (Path("state") / "gauntlet_013.json").write_text(json.dumps(out, indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
