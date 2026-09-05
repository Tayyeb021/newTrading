"""013: the AI filter wrapped around the monthly trend rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.strategy import is_month_start  # noqa: E402
from core.types import Position, Side  # noqa: E402
from ml.labeling import LabelConfig, triple_barrier_labels  # noqa: E402
from ml.models import MetaLabelModel  # noqa: E402
from strategies.ml_filter import MLFiltered, event_features, signal_events  # noqa: E402
from strategies.tsmom import TSMOM  # noqa: E402

NOW = pd.Timestamp("2024-06-03", tz="UTC")


def _frame(n=1000, seed=5, drift=0.0006):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    px = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    return pd.DataFrame({"ts": ts, "open": px, "high": px * 1.006, "low": px * 0.994, "close": px, "volume": 1.0})


def _trained(base, prepared):
    events = signal_events(base, prepared)
    labels = triple_barrier_labels(prepared, events, LabelConfig(profit_multiple=1e6, max_bars=21))
    x = event_features(prepared, base).iloc[labels["i"].to_numpy()].reset_index(drop=True)
    y = (labels["ret"] > 0).astype(int).to_numpy()
    keep = x.notna().all(axis=1).to_numpy()
    return MetaLabelModel(threshold=0.5, min_samples=10).fit(
        x[keep], y[keep], t0=labels["i"].to_numpy()[keep], t1=labels["t1"].to_numpy()[keep])


def test_wrapper_only_touches_new_entries():
    base = TSMOM(lookback=60)
    prepared = base.prepare(_frame())
    model = _trained(TSMOM(lookback=60), TSMOM(lookback=60).prepare(_frame()))
    wrapped = MLFiltered(TSMOM(lookback=60), meta=model)
    p2 = wrapped.prepare(_frame())
    assert wrapped.name == "tsmom_ml" and wrapped.rebalances is False and wrapped.warmup >= 265

    held = Position("X", Side.BUY, 1.0, 100.0, NOW.to_pydatetime(), stop_loss=90.0)
    entries = skips = holds = 0
    for i in range(wrapped.warmup, len(p2)):
        raw = base.evaluate(prepared, i, None)
        filtered = wrapped.evaluate(p2, i, None)
        if not raw.flat:  # an entry: the filter may skip or shrink, never enlarge
            if filtered.flat:
                skips += 1
            else:
                assert filtered.side is raw.side and filtered.stop_distance == raw.stop_distance
                assert filtered.confidence <= raw.confidence and "ml=" in filtered.reason
                entries += 1
        else:
            assert filtered.flat
        # with a position held, whatever the base says passes through unchanged
        base_held = base.evaluate(prepared, i, held)
        wrapped_held = wrapped.evaluate(p2, i, held)
        assert wrapped_held.side is base_held.side and wrapped_held.confidence == base_held.confidence
        assert wrapped_held.reason == base_held.reason
        holds += 1
    assert entries + skips > 5 and holds > 0


def test_signal_events_are_decision_days_only():
    base = TSMOM(lookback=60)
    prepared = base.prepare(_frame())
    events = signal_events(base, prepared)
    assert len(events) > 5
    assert all(is_month_start(prepared, int(i)) for i in events["i"])


def test_event_features_carry_the_base_forecast_and_speed():
    base = TSMOM(lookback=120)
    prepared = base.prepare(_frame())
    f = event_features(prepared, base)
    assert (f["lookback"] == 120.0).all()
    assert np.allclose(f["base_forecast"].dropna(), prepared["forecast"].dropna())
