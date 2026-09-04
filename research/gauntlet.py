"""The validation gauntlet -- the thing that tells you whether you have anything.

Ten gates from the build plan, run in order, each cheaper than the one after it.
The point is not to produce a score. It is to make it hard to fool yourself, and
the ordering reflects where self-deception is cheapest to catch.

The most important property: **thresholds are declared before the run**, in
`GauntletThresholds`. After the fact, every threshold looks negotiable, and the
version of you reading a promising equity curve is not the person who should be
setting the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.engine import Backtester
from backtest.metrics import Metrics, compute
from core.config import RiskProfile
from core.strategy import Strategy
from core.types import SymbolSpec
from ml.stats import (
    PBOResult,
    deflated_sharpe,
    monte_carlo_trades,
    probability_of_backtest_overfitting,
    probabilistic_sharpe,
    sharpe,
)
from risk.build import build_engine


@dataclass(frozen=True)
class GauntletThresholds:
    """Declare these BEFORE looking at any result. That is the whole discipline."""

    min_trades: int = 30
    min_expectancy_r: float = 0.02
    min_sharpe: float = 0.50
    max_pbo: float = 0.20
    min_deflated_sharpe: float = 0.90  # P(true SR > expected max under the null)
    max_cost_drag: float = 0.50
    min_stress_survival: float = 0.50  # fraction of edge surviving 2x costs
    max_mc_p95_drawdown: float = 0.25
    min_walk_forward_efficiency: float = 0.40


@dataclass
class Gate:
    name: str
    passed: bool
    observed: str
    threshold: str
    note: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name:<34}{self.observed:>18}  vs {self.threshold:<14}{self.note}"


@dataclass
class GauntletResult:
    gates: list[Gate] = field(default_factory=list)
    metrics: Metrics | None = None
    stress_metrics: Metrics | None = None
    pbo: PBOResult | None = None
    walk_forward: list[Metrics] = field(default_factory=list)
    trial_sharpes: list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)

    def report(self) -> str:
        lines = ["", "VALIDATION GAUNTLET", "=" * 84]
        lines.extend(str(g) for g in self.gates)
        lines.append("=" * 84)
        n_pass = sum(1 for g in self.gates if g.passed)
        lines.append(f"  {n_pass}/{len(self.gates)} gates passed")
        if self.passed:
            lines.append("\n  All gates cleared. That is permission to forward-test on demo,")
            lines.append("  not permission to fund anything.")
        else:
            failed = [g.name for g in self.gates if not g.passed]
            lines.append(f"\n  BLOCKED: {', '.join(failed)}")
        return "\n".join(lines)


def _run(strategy: Strategy, df: pd.DataFrame, spec: SymbolSpec,
         profile: RiskProfile, costs: CostModel, equity: float = 100_000.0):
    engine = build_engine(profile, equity, {spec.symbol: spec})
    return Backtester(strategy, spec, engine, costs, starting_equity=equity).run(df)


def run_gauntlet(
    strategy_factory,
    df: pd.DataFrame,
    spec: SymbolSpec,
    profile: RiskProfile,
    costs: CostModel | None = None,
    thresholds: GauntletThresholds | None = None,
    n_trials_tested: int = 1,
    walk_forward_folds: int = 6,
    variant_factories: list | None = None,
) -> GauntletResult:
    """Run every gate.

    `strategy_factory` is a zero-argument callable, because several gates need a
    *fresh* strategy -- a fitted model reused across folds would leak.

    `n_trials_tested` must be the honest count of configurations you have tried
    across this whole research effort, not the number in this run. The deflated
    Sharpe is only meaningful if that number is true.
    """
    costs = costs or CostModel()
    th = thresholds or GauntletThresholds()
    result = GauntletResult()

    # ---- gates 1-2: does it trade, and does it make money after costs --------
    base = _run(strategy_factory(), df, spec, profile, costs)
    m = compute(base)
    result.metrics = m

    result.gates.append(Gate(
        "sample size", m.trades >= th.min_trades,
        f"{m.trades} trades", f">= {th.min_trades}",
    ))
    result.gates.append(Gate(
        "expectancy after costs", m.expectancy_r >= th.min_expectancy_r,
        f"{m.expectancy_r:+.3f}R", f">= {th.min_expectancy_r:+.3f}R",
    ))
    result.gates.append(Gate(
        "sharpe", m.sharpe >= th.min_sharpe,
        f"{m.sharpe:.2f}", f">= {th.min_sharpe:.2f}",
    ))
    result.gates.append(Gate(
        "cost drag", m.cost_drag <= th.max_cost_drag,
        f"{m.cost_drag:.1%}", f"<= {th.max_cost_drag:.0%}",
    ))

    if m.trades < 2:
        result.gates.append(Gate("remaining gates", False, "skipped", "-",
                                 "too few trades to evaluate"))
        return result

    # ---- gate 3: cost sensitivity -------------------------------------------
    stressed = compute(_run(strategy_factory(), df, spec, profile, costs.stressed(2.0)))
    result.stress_metrics = stressed
    survival = (
        stressed.expectancy_r / m.expectancy_r if m.expectancy_r > 0 else 0.0
    )
    result.gates.append(Gate(
        "survives 2x costs", survival >= th.min_stress_survival,
        f"{survival:.0%} of edge", f">= {th.min_stress_survival:.0%}",
        "" if survival >= th.min_stress_survival else "it was friction, not an edge",
    ))

    # ---- gate 4: walk-forward ------------------------------------------------
    folds = np.array_split(np.arange(len(df)), walk_forward_folds)
    fold_metrics: list[Metrics] = []
    for fold in folds[1:]:  # first fold is warmup only
        window = df.iloc[: int(fold[-1]) + 1]
        if len(window) < 120:
            continue
        try:
            fm = compute(_run(strategy_factory(), window, spec, profile, costs))
            if fm.trades >= 3:
                fold_metrics.append(fm)
        except ValueError:
            continue
    result.walk_forward = fold_metrics

    if fold_metrics:
        positive = sum(1 for f in fold_metrics if f.expectancy_r > 0)
        efficiency = positive / len(fold_metrics)
        result.gates.append(Gate(
            "walk-forward stability", efficiency >= th.min_walk_forward_efficiency,
            f"{positive}/{len(fold_metrics)} folds +ve", f">= {th.min_walk_forward_efficiency:.0%}",
        ))

    # ---- gate 5: deflated sharpe --------------------------------------------
    trade_returns = np.array([t.r_multiple for t in base.trades])
    equity_returns = base.equity.pct_change().dropna().to_numpy()

    variants = variant_factories or []
    trial_sharpes = [sharpe(trade_returns)]
    for factory in variants:
        try:
            variant = _run(factory(), df, spec, profile, costs)
            if variant.trades:
                trial_sharpes.append(sharpe(np.array([t.r_multiple for t in variant.trades])))
        except Exception:  # noqa: BLE001 - a broken variant must not stop the gauntlet
            continue
    result.trial_sharpes = trial_sharpes

    n_trials = max(n_trials_tested, len(trial_sharpes))
    if len(trial_sharpes) > 1:
        dsr = deflated_sharpe(equity_returns, n_trials, trial_sharpes=np.array(trial_sharpes))
    else:
        # No variance measured; fall back to PSR against zero and say so.
        dsr = probabilistic_sharpe(equity_returns, benchmark_sr=0.0)

    result.gates.append(Gate(
        "deflated sharpe", dsr >= th.min_deflated_sharpe,
        f"{dsr:.3f}", f">= {th.min_deflated_sharpe:.2f}",
        f"({n_trials} trials)" if len(trial_sharpes) > 1 else "(PSR only - no variants tested)",
    ))

    # ---- gate 6: PBO ---------------------------------------------------------
    if len(variants) >= 3:
        matrix = _variant_matrix([strategy_factory, *variants], df, spec, profile, costs)
        if matrix is not None and matrix.shape[1] >= 2:
            pbo = probability_of_backtest_overfitting(matrix, n_partitions=8)
            result.pbo = pbo
            result.gates.append(Gate(
                "probability of overfitting", pbo.pbo <= th.max_pbo,
                f"{pbo.pbo:.1%}", f"<= {th.max_pbo:.0%}", pbo.verdict,
            ))

    # ---- gate 7: Monte Carlo -------------------------------------------------
    mc = monte_carlo_trades(trade_returns * 0.01, paths=5_000)
    result.gates.append(Gate(
        "Monte Carlo 95th pct DD", mc.p95_drawdown <= th.max_mc_p95_drawdown,
        f"{mc.p95_drawdown:.1%}", f"<= {th.max_mc_p95_drawdown:.0%}",
        "the drawdown you must be able to sit through",
    ))

    return result


def _variant_matrix(factories, df, spec, profile, costs) -> np.ndarray | None:
    """Per-bar returns for each configuration, aligned. Input to the PBO test."""
    series: list[pd.Series] = []
    for factory in factories:
        try:
            run = _run(factory(), df, spec, profile, costs)
            if len(run.equity) > 10:
                series.append(run.equity.pct_change().fillna(0.0))
        except Exception:  # noqa: BLE001
            continue
    if len(series) < 2:
        return None
    return pd.concat(series, axis=1).dropna().to_numpy()
