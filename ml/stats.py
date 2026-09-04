"""Statistics that correct for the fact that you looked more than once.

This module exists because of one asymmetry: it is easy to produce a backtest
with a Sharpe of 2, and hard to produce one that means anything. Test enough
configurations and a strategy with no edge will hand you a beautiful curve --
the more variants you try, the more certain that becomes.

Three tools, in the order you should apply them:

- **PSR** -- is this Sharpe distinguishable from a benchmark, given the sample
  length and the non-normality of the returns?
- **DSR** -- same question, after penalising you for the number of configurations
  you tried. This is the single most useful number in the module.
- **PBO** -- across many train/test splits, how often does the configuration that
  looked best in-sample land below median out-of-sample? Above ~20% and your
  selection procedure is fitting noise.

References: Bailey & Lopez de Prado, *The Deflated Sharpe Ratio* (2014), and
*The Probability of Backtest Overfitting* (2015).

Every Sharpe here is **per-observation**, not annualised. Mixing the two is the
most common way to get these formulas wrong: annualising inflates SR by sqrt(252)
while T stays in the same units, and the result is nonsense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe(returns: np.ndarray, ddof: int = 1) -> float:
    """Per-observation Sharpe. Not annualised - see the module docstring."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=ddof)
    return float(r.mean() / sd) if sd > 0 else 0.0


def annualise(sr: float, periods_per_year: int = 252) -> float:
    return sr * math.sqrt(periods_per_year)


def probabilistic_sharpe(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    observed_sr: float | None = None,
) -> float:
    """P(true SR > benchmark), correcting for sample length, skew and kurtosis.

    Negative skew and fat tails - exactly what a stop-loss strategy produces -
    make a given Sharpe *less* trustworthy, and this is where that shows up.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = r.size
    if t < 3:
        return 0.0

    sr = observed_sr if observed_sr is not None else sharpe(r)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0

    z = (r - r.mean()) / sd
    skew = float((z**3).mean())
    kurt = float((z**4).mean())  # non-excess

    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom <= 0:
        return 0.0
    return float(norm.cdf((sr - benchmark_sr) * math.sqrt(t - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Highest Sharpe you would expect from `n_trials` worthless strategies.

    This is the bar a real result has to clear. Try 500 configurations and pure
    noise will hand you something respectable; this says how respectable.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - g) * a + g * b))


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    sr_variance: float | None = None,
    trial_sharpes: np.ndarray | None = None,
) -> float:
    """PSR against the expected-maximum-Sharpe benchmark. Report this, not SR.

    Pass `trial_sharpes` -- every configuration you tested, including the losers
    -- and the variance is measured rather than assumed. That count must be
    honest: a DSR computed against 3 trials when you really tried 300 is a
    number that flatters you and means nothing.
    """
    if trial_sharpes is not None:
        arr = np.asarray(trial_sharpes, dtype=float)
        arr = arr[np.isfinite(arr)]
        sr_variance = float(arr.var(ddof=1)) if arr.size > 1 else 0.0
        n_trials = max(n_trials, arr.size)
    if sr_variance is None:
        raise ValueError("provide either sr_variance or trial_sharpes")

    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe(returns, benchmark_sr=sr0)


def min_track_record_length(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed before this Sharpe could be called significant.

    Useful in the other direction: if the answer exceeds the data you have, the
    honest report is "not yet knowable", not a p-value.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("inf")

    sr = sharpe(r)
    if sr <= benchmark_sr:
        return float("inf")

    sd = r.std(ddof=1)
    z = (r - r.mean()) / sd
    skew = float((z**3).mean())
    kurt = float((z**4).mean())

    denom = (sr - benchmark_sr) ** 2
    numer = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    return float(1.0 + numer / denom * norm.ppf(confidence) ** 2)


# --------------------------------------------------------------------------- #
# Probability of backtest overfitting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_splits: int
    n_configs: int
    oos_ranks: np.ndarray
    is_best_oos_sharpe: np.ndarray

    @property
    def verdict(self) -> str:
        if self.pbo < 0.10:
            return "selection looks sound"
        if self.pbo < 0.20:
            return "acceptable"
        if self.pbo < 0.50:
            return "OVERFIT - your selection procedure is fitting noise"
        return "SEVERELY OVERFIT - in-sample rank is anti-correlated with out-of-sample"

    def __str__(self) -> str:
        return (
            f"PBO {self.pbo:.1%} over {self.n_splits:,} splits of {self.n_configs} "
            f"configurations - {self.verdict}"
        )


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_partitions: int = 8,
) -> PBOResult:
    """CSCV: how often does the in-sample winner underperform out-of-sample?

    `returns_matrix` is T observations x N configurations. Every configuration you
    tried belongs in it, losers included -- feeding it only the survivors is
    exactly the selection bias this is meant to measure.

    Split T into `n_partitions` blocks, take every way of choosing half of them as
    in-sample, find the best config there, then check its rank on the complement.
    A strategy selection process with no skill puts that rank below median about
    half the time, so PBO near 0.5 means your search found nothing but noise.
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2-D: observations x configurations")
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least 2 configurations to rank")
    if n_partitions % 2 or n_partitions < 4:
        raise ValueError("n_partitions must be even and at least 4")
    if t < n_partitions * 2:
        raise ValueError(f"need at least {n_partitions * 2} observations, got {t}")

    blocks = np.array_split(np.arange(t), n_partitions)
    half = n_partitions // 2

    logits: list[float] = []
    ranks: list[float] = []
    best_oos: list[float] = []

    for combo in combinations(range(n_partitions), half):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_partitions) if i not in combo])

        is_sr = np.array([sharpe(m[is_idx, c]) for c in range(n)])
        oos_sr = np.array([sharpe(m[oos_idx, c]) for c in range(n)])

        best = int(np.nanargmax(is_sr))
        best_oos.append(float(oos_sr[best]))

        # Relative rank of the in-sample winner within the OOS distribution.
        order = np.argsort(oos_sr)
        rank = float(np.where(order == best)[0][0] + 1) / (n + 1)
        ranks.append(rank)
        logits.append(math.log(rank / (1.0 - rank)))

    logit_arr = np.array(logits)
    return PBOResult(
        pbo=float((logit_arr <= 0).mean()),
        logits=logit_arr,
        n_splits=len(logits),
        n_configs=n,
        oos_ranks=np.array(ranks),
        is_best_oos_sharpe=np.array(best_oos),
    )


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MonteCarloResult:
    median_return: float
    p05_return: float
    p95_drawdown: float
    median_drawdown: float
    worst_drawdown: float
    prob_loss: float
    paths: int

    def __str__(self) -> str:
        return (
            f"Monte Carlo over {self.paths:,} paths:\n"
            f"  return   median {self.median_return:>8.1%}   5th pct {self.p05_return:>8.1%}\n"
            f"  drawdown median {self.median_drawdown:>8.1%}   95th pct {self.p95_drawdown:>8.1%}"
            f"   worst {self.worst_drawdown:>7.1%}\n"
            f"  probability of a losing run: {self.prob_loss:.1%}"
        )


def monte_carlo_trades(
    trade_returns: np.ndarray,
    paths: int = 10_000,
    slippage_penalty: float = 0.0,
    seed: int = 17,
) -> MonteCarloResult:
    """Reshuffle the trade sequence to see what else could have happened.

    The realised equity curve is one draw from a distribution. Read the 95th
    percentile drawdown, not the average - that is the number you have to be able
    to sit through, and the one that ends evaluations.

    `slippage_penalty` subtracts a constant from every trade, which is the cheap
    version of the cost-sensitivity gate.
    """
    r = np.asarray(trade_returns, dtype=float)
    r = r[np.isfinite(r)] - slippage_penalty
    if r.size < 2:
        raise ValueError("need at least 2 trades")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(paths, r.size))
    sampled = r[idx]

    equity = np.cumprod(1.0 + sampled, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    drawdowns = ((peak - equity) / peak).max(axis=1)
    finals = equity[:, -1] - 1.0

    return MonteCarloResult(
        median_return=float(np.median(finals)),
        p05_return=float(np.percentile(finals, 5)),
        p95_drawdown=float(np.percentile(drawdowns, 95)),
        median_drawdown=float(np.median(drawdowns)),
        worst_drawdown=float(drawdowns.max()),
        prob_loss=float((finals < 0).mean()),
        paths=paths,
    )
