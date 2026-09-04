"""Triple-barrier labelling.

The question a meta-label model answers is not "which way will price go?" -- that
is close to a coin flip and the model has almost nothing to learn. It is: **given
that my strategy just fired this signal, does it work?**

To ask that you need a label, and the label has to come from the trade the system
would actually have taken. Three barriers define it:

- **profit target** -- an ATR multiple above entry (below, for a short)
- **stop loss** -- the strategy's own stop, so the label matches the real risk
- **time limit** -- a vertical barrier, because a trade held forever is not a trade

Whichever is touched first decides the label. Crucially each label also carries
`t1`, the bar at which it resolved. That column is what makes purged
cross-validation possible: without it you cannot know which training observations
overlap a test set, and ordinary k-fold leaks the answer across the split.

Barriers are checked against the bar's high and low, and when both are inside one
bar the **stop is assumed first** -- the same pessimism the backtester uses, for
the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.types import Side


@dataclass(frozen=True)
class LabelConfig:
    profit_multiple: float = 2.0  # target as a multiple of the stop distance
    max_bars: int = 20  # vertical barrier
    min_return: float = 0.0  # below this, label 0 regardless of direction


def triple_barrier_labels(
    df: pd.DataFrame,
    events: pd.DataFrame,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Label each event by which barrier it touches first.

    `events` needs columns: `i` (bar index of the signal), `side` (Side), and
    `stop_distance` (price). Returns one row per event with:

        label   1 if the target was hit first, 0 otherwise
        ret     realised return of the trade, in R
        t1      bar index at which it resolved  <- required for purging
        touched 'target' | 'stop' | 'time'

    Note `label` is binary and directional accuracy is *not* what it measures.
    The strategy already chose the direction; this measures whether that choice
    paid, which is a far easier and more stable thing to learn.
    """
    cfg = config or LabelConfig()
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    n = len(df)

    rows = []
    for event in events.itertuples():
        i = int(event.i)
        entry_idx = i + 1  # the strategy trades the NEXT bar's open
        if entry_idx >= n:
            continue

        entry = opens[entry_idx]
        stop_distance = float(event.stop_distance)
        if stop_distance <= 0:
            continue

        sign = event.side.sign if isinstance(event.side, Side) else int(event.side)
        stop = entry - stop_distance * sign
        target = entry + stop_distance * cfg.profit_multiple * sign

        last = min(entry_idx + cfg.max_bars, n - 1)
        touched, t1, exit_price = "time", last, df["close"].to_numpy(dtype=float)[last]

        for j in range(entry_idx, last + 1):
            hit_stop = lows[j] <= stop if sign > 0 else highs[j] >= stop
            hit_target = highs[j] >= target if sign > 0 else lows[j] <= target

            # Both inside one bar: assume the stop. OHLC cannot resolve the order,
            # and assuming the favourable one is how models learn to be optimistic.
            if hit_stop:
                touched, t1, exit_price = "stop", j, stop
                break
            if hit_target:
                touched, t1, exit_price = "target", j, target
                break

        r_multiple = (exit_price - entry) * sign / stop_distance
        rows.append({
            "i": i,
            "entry_i": entry_idx,
            "t1": t1,
            "side": sign,
            "entry": entry,
            "exit": exit_price,
            "stop_distance": stop_distance,
            "touched": touched,
            "ret": r_multiple,
            "label": int(r_multiple > cfg.min_return),
        })

    return pd.DataFrame(rows)


def sample_weights(labels: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Down-weight observations whose label spans overlap others.

    Two signals two bars apart, each resolving over twenty bars, share almost all
    of their outcome. Treating them as independent observations tells the model it
    has far more evidence than it does. Weight is the inverse of how many other
    labels each bar is shared with -- Lopez de Prado's uniqueness weighting.
    """
    if labels.empty:
        return np.array([])

    concurrency = np.zeros(n_bars, dtype=float)
    for row in labels.itertuples():
        concurrency[row.entry_i : row.t1 + 1] += 1.0

    weights = np.zeros(len(labels), dtype=float)
    for k, row in enumerate(labels.itertuples()):
        span = concurrency[row.entry_i : row.t1 + 1]
        span = span[span > 0]
        weights[k] = float((1.0 / span).mean()) if span.size else 0.0

    total = weights.sum()
    return weights * len(weights) / total if total > 0 else weights


def label_summary(labels: pd.DataFrame) -> str:
    if labels.empty:
        return "no labels generated"
    counts = labels["touched"].value_counts()
    lines = [
        f"{len(labels):,} labels, {labels['label'].mean():.1%} positive",
        f"  mean {labels['ret'].mean():+.3f}R   median {labels['ret'].median():+.3f}R",
    ]
    for name in ("target", "stop", "time"):
        n = int(counts.get(name, 0))
        lines.append(f"  {name:<8}{n:>7,}  {n / len(labels):>6.1%}")

    # A class balance far from the base rate usually means the barriers are
    # mis-scaled, not that the strategy is unusually good or bad.
    positive = labels["label"].mean()
    if positive < 0.15 or positive > 0.85:
        lines.append(
            f"  ! class balance {positive:.1%} is extreme - check barrier scaling "
            f"before training on this"
        )
    return "\n".join(lines)
