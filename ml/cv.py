"""Cross-validation for data where observations overlap in time.

Ordinary k-fold is wrong here, and not subtly. A label at bar 100 that resolves at
bar 120 shares its outcome with a label at bar 110. Put one in train and the other
in test and the model has seen the answer. The result is a validation score that
looks excellent and predicts nothing.

Two corrections, both from Lopez de Prado's *Advances in Financial Machine
Learning*:

- **Purging** -- drop from training any observation whose label span [t0, t1]
  overlaps the test set at all.
- **Embargo** -- additionally drop a gap of observations immediately after the
  test set, because serial correlation leaks forward even without overlap.

`CombinatorialPurgedCV` goes further: instead of one train/test path it generates
many, which turns a single Sharpe into a *distribution* of them. That distribution
is what the PBO calculation in `ml.stats` consumes, and it is the difference
between "my backtest scored 1.8" and "my selection procedure is fitting noise".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def purge_train_indices(
    train: np.ndarray,
    test: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    embargo: int = 0,
) -> np.ndarray:
    """Remove training observations that overlap or immediately follow the test set."""
    if test.size == 0:
        return train

    test_start = int(t0[test].min())
    test_end = int(t1[test].max())
    embargo_end = test_end + embargo

    # Keep only spans that finish before the test starts, or begin after the
    # embargo ends. Anything straddling the boundary is contaminated.
    keep = (t1[train] < test_start) | (t0[train] > embargo_end)
    return train[keep]


@dataclass
class PurgedKFold:
    """K-fold over contiguous time blocks, with purging and an embargo."""

    n_splits: int = 5
    embargo_pct: float = 0.01

    def split(self, t0: np.ndarray, t1: np.ndarray):
        t0 = np.asarray(t0, dtype=int)
        t1 = np.asarray(t1, dtype=int)
        n = t0.size
        if n < self.n_splits * 2:
            raise ValueError(f"need at least {self.n_splits * 2} observations, got {n}")

        embargo = int(n * self.embargo_pct)
        indices = np.arange(n)
        for test in np.array_split(indices, self.n_splits):
            train = np.setdiff1d(indices, test, assume_unique=False)
            yield purge_train_indices(train, test, t0, t1, embargo), test

    def get_n_splits(self, *_args) -> int:
        return self.n_splits


@dataclass
class CombinatorialPurgedCV:
    """CPCV -- many train/test paths instead of one.

    Split the observations into `n_groups` contiguous blocks and use every
    combination of `n_test_groups` of them as a test set. With 6 groups and 2 per
    test that is 15 paths, each purged and embargoed, giving 15 out-of-sample
    performance figures rather than a single lucky one.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    embargo_pct: float = 0.01

    def n_paths(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    def split(self, t0: np.ndarray, t1: np.ndarray):
        t0 = np.asarray(t0, dtype=int)
        t1 = np.asarray(t1, dtype=int)
        n = t0.size
        if n < self.n_groups * 2:
            raise ValueError(f"need at least {self.n_groups * 2} observations, got {n}")

        embargo = int(n * self.embargo_pct)
        indices = np.arange(n)
        groups = np.array_split(indices, self.n_groups)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test = np.concatenate([groups[g] for g in combo])
            train = np.setdiff1d(indices, test, assume_unique=False)

            # Purge against every disjoint test block separately; a single
            # min/max span over non-contiguous blocks would purge the gap
            # between them too, throwing away usable training data.
            for g in combo:
                block = groups[g]
                train = purge_train_indices(train, block, t0, t1, embargo)
            yield train, test


def leakage_report(t0: np.ndarray, t1: np.ndarray, splitter) -> str:
    """How much data purging costs, and proof that it removed the overlap.

    Worth printing once per dataset. If purging removes almost nothing, the
    labels probably are not overlapping and a simpler CV would do. If it removes
    most of the training set, the vertical barrier is too long for the sampling
    frequency and the model will be starved.
    """
    t0 = np.asarray(t0, dtype=int)
    t1 = np.asarray(t1, dtype=int)
    n = t0.size

    lines = [f"{n:,} labelled observations, mean span "
             f"{float((t1 - t0).mean()):.1f} bars"]
    kept, naive_total, leaks = [], [], 0

    for train, test in splitter.split(t0, t1):
        naive = np.setdiff1d(np.arange(n), test)
        kept.append(train.size)
        naive_total.append(naive.size)

        test_start, test_end = int(t0[test].min()), int(t1[test].max())
        overlapping = np.sum((t1[naive] >= test_start) & (t0[naive] <= test_end))
        leaks += int(overlapping)
        assert np.sum((t1[train] >= test_start) & (t0[train] <= test_end)) == 0, (
            "purging failed - contaminated observations survived into training"
        )

    lines.append(
        f"purging removed {np.mean(naive_total) - np.mean(kept):,.0f} of "
        f"{np.mean(naive_total):,.0f} training observations per fold "
        f"({1 - np.mean(kept) / np.mean(naive_total):.1%})"
    )
    lines.append(f"{leaks:,} leaking observations would have been used by plain k-fold")
    return "\n".join(lines)
