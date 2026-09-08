"""A strategy deployed live must be *able* to trade.

Found 2026-09-08. The shadow week ran four days, logged 13,283 heartbeats, and
made zero decisions. Not one. The feed was healthy, the process was up, the
journal was full, and every check said green.

`MTFPullback.prepare` builds its higher-timeframe bias from `self.bias_frames`.
With that dict empty it sets `bias = 0`, and `evaluate` returns FLAT on
`bias == 0` before it reads location, trigger, session or anything else. So a
MTFPullback constructed without bias frames returns FLAT on every bar forever -
993,548 bar evaluations across up to 28 years of stored history produced exactly
zero signals on all four live symbols.

Every backtest script passed `bias_frames`. Every test passed `bias_frames`.
`scripts/shadow.py`, the only caller that runs live, passed none - so the rule
was validated in one configuration and deployed in another that cannot trade.

These tests pin the mechanism and the deployment, because nothing else would
have caught it: no exception, no log line, no failing assertion. Just silence
that looks like patience.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.strategy import FLAT  # noqa: E402
from strategies.mtf_pullback import MTFPullback  # noqa: E402


def _trending(n: int = 4000, tf_minutes: int = 15, start: float = 100.0,
              drift: float = 0.03, amp: float = 1.5, period: float = 30.0,
              wick: float = 0.02) -> pd.DataFrame:
    """A clean uptrend with regular shallow pullbacks - about as favourable as a
    pullback rule ever gets. If it will not fire here it will not fire anywhere.

    Three details the first version got wrong, each of which made a working
    strategy look dead:

    - **Length.** The H4 bias needs a 50-period EMA plus slope, so 900 M15 bars
      (56 H4 bars) left the bias undefined almost everywhere. 4,000 gives 250.
    - **Wick size.** The entry trigger is `close > prev.high`. With a 0.35 wick
      against a maximum per-bar move of 0.19 it could never fire, whatever the
      trend did. The wick has to be smaller than the moves.
    - **Open.** Each bar opens at the previous close, as bars do.
    """
    t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
    ts = [t0 + timedelta(minutes=tf_minutes * i) for i in range(n)]
    close = start + np.arange(n) * drift + np.sin(np.arange(n) * 2 * np.pi / period) * amp
    opens = np.empty(n)
    opens[0], opens[1:] = close[0] - wick, close[:-1]
    return pd.DataFrame({
        "ts": ts, "open": opens,
        "high": np.maximum(opens, close) + wick,
        "low": np.minimum(opens, close) - wick,
        "close": close, "volume": 1000.0,
    })


def _signals(strategy: MTFPullback, df: pd.DataFrame) -> int:
    d = strategy.prepare(df.copy())
    fired = 0
    for i in range(strategy.warmup, len(d)):
        s = strategy.evaluate(d, i, None)
        if s is not None and s is not FLAT and getattr(s, "side", None) is not None:
            fired += 1
    return fired


def test_without_bias_frames_the_rule_can_never_signal():
    """The bug itself, pinned. This is not a quirk of one market or one week -
    it is arithmetic, and it holds on data built to make the rule fire."""
    df = _trending()
    naked = MTFPullback(execution_timeframe="M15", bias_timeframes=("H4", "H1"))
    prepared = naked.prepare(df.copy())
    assert (prepared["bias"] == 0).all(), "no bias frames means no bias, on every bar"
    assert _signals(naked, df) == 0, "and a zero bias returns FLAT before anything else is read"


def test_with_bias_frames_the_same_rule_on_the_same_data_does_signal():
    """The control. Same strategy, same bars - only the bias frames differ. If
    this ever returns zero the fixture has gone stale and the test above proves
    nothing."""
    df = _trending()
    h1 = df.set_index("ts").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    h4 = df.set_index("ts").resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()

    fed = MTFPullback(execution_timeframe="M15", bias_timeframes=("H4", "H1"),
                      bias_frames={"H4": h4, "H1": h1})
    prepared = fed.prepare(df.copy())
    assert (prepared["bias"] != 0).any(), "the higher timeframes must produce a direction"
    assert _signals(fed, df) > 0, "and with a direction the rule trades"


def test_shadow_deploys_a_strategy_that_refreshes_its_own_bias():
    """The deployment, not just the mechanism. shadow.py is the only caller
    that runs live, and it is the one that shipped without bias frames."""
    spec = importlib.util.spec_from_file_location("shadow_mod", ROOT / "scripts" / "shadow.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shadow_mod"] = mod
    spec.loader.exec_module(mod)

    assert hasattr(mod, "LiveMTFPullback"), "shadow.py must deploy a bias-refreshing strategy"
    assert issubclass(mod.LiveMTFPullback, MTFPullback)

    src = (ROOT / "scripts" / "shadow.py").read_text(encoding="utf-8")
    body = src.split("runner = Runner(")[1]
    assert "LiveMTFPullback(" in body, "the runner must be given the live variant"
    assert "MTFPullback(execution_timeframe" not in body, (
        "the bare MTFPullback cannot signal live - it must not be constructed here")


def test_the_live_variant_pulls_its_bias_from_the_adapter_every_bar():
    """A bias snapshotted at startup is a different bug with the same shape:
    by Friday the rule is trading Monday's trend."""
    spec = importlib.util.spec_from_file_location("shadow_mod2", ROOT / "scripts" / "shadow.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shadow_mod2"] = mod
    spec.loader.exec_module(mod)

    df = _trending()

    class Bar:
        def __init__(self, r):
            self.ts, self.open, self.high = r.ts, r.open, r.high
            self.low, self.close, self.volume = r.low, r.close, r.volume

    class CountingAdapter:
        def __init__(self):
            self.calls = []

        def bars(self, symbol, timeframe, count=None, end=None):
            self.calls.append(timeframe)
            step = {"H1": 4, "H4": 16}.get(timeframe, 1)
            return [Bar(r) for r in df.iloc[::step].itertuples()]

    a = CountingAdapter()
    s = mod.LiveMTFPullback(a, "EURUSD", execution_timeframe="M15",
                            bias_timeframes=("H4", "H1"))
    s.prepare(df.copy())
    assert "H4" in a.calls and "H1" in a.calls, "both bias frames must be fetched"

    before = len(a.calls)
    s.prepare(df.copy())
    assert len(a.calls) > before, "and re-fetched on the next bar, not cached from startup"


def test_a_missing_higher_timeframe_is_reported_not_swallowed(capsys):
    """If the feed cannot supply H4, the rule goes flat. That must be visible -
    silent flatness is the thing this whole file is about."""
    spec = importlib.util.spec_from_file_location("shadow_mod3", ROOT / "scripts" / "shadow.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shadow_mod3"] = mod
    spec.loader.exec_module(mod)

    class Broken:
        def bars(self, symbol, timeframe, count=None, end=None):
            raise RuntimeError("no such timeframe")

    s = mod.LiveMTFPullback(Broken(), "EURUSD", execution_timeframe="M15",
                            bias_timeframes=("H4", "H1"))
    s.prepare(_trending().copy())
    out = capsys.readouterr().out
    assert "H4" in out and "no such timeframe" in out
