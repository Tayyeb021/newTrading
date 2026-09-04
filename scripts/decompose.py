"""Take the strategy apart and test each component on its own.

Eighteen backtests failed on expectancy, but a backtest only ever answers "did
this whole machine make money". It cannot tell you *which part* is broken. This
does.

The strategy is three claims stacked on top of each other:

1. **BIAS** -- knowing the H4/H1 direction predicts where price goes next.
2. **LOCATION** -- entering on a pullback beats entering anywhere.
3. **TRIGGER** -- waiting for momentum to resume beats entering immediately.

Each is tested here in isolation, with no stops, no targets and no costs, against
the only question that matters: **does the forward return, measured in the
direction the component says to trade, differ from zero?**

If claim 1 is false, claims 2 and 3 are decoration and no amount of entry work
will save the strategy. That is worth knowing before spending a month on entry
work.

Two statistical points, both of which matter:

- Forward returns from consecutive bars overlap heavily and are not independent.
  Sampling every H-th bar gives non-overlapping windows, at the cost of a smaller
  sample. Overlapping windows would inflate every t-statistic here.
- A signal can be real and still untradeable. Every edge is therefore reported in
  ATR units next to the round-trip cost in the same units, because a mean forward
  return of 0.05 ATR against a cost of 0.20 ATR is a true fact you cannot trade.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.store import BarStore  # noqa: E402
from strategies.mtf_pullback import MTFPullback, load_bias_frames  # noqa: E402


@dataclass
class Edge:
    name: str
    n: int
    mean_atr: float   # mean forward return in the signalled direction, in ATR
    t_stat: float
    hit_rate: float
    cost_atr: float

    @property
    def net_atr(self) -> float:
        return self.mean_atr - self.cost_atr

    @property
    def verdict(self) -> str:
        if self.n < 30:
            return "too few samples"
        if abs(self.t_stat) < 2.0:
            return "no signal (t < 2)"
        if self.mean_atr <= 0:
            return "signal INVERTED"
        if self.net_atr <= 0:
            return "real but untradeable - cost exceeds it"
        return "TRADEABLE"

    def __str__(self) -> str:
        return (
            f"  {self.name:<34}{self.n:>7,}{self.mean_atr:>+9.4f}{self.t_stat:>8.2f}"
            f"{self.hit_rate:>8.1%}{self.net_atr:>+10.4f}   {self.verdict}"
        )


def measure(
    signed_returns: np.ndarray,
    name: str,
    cost_atr: float,
) -> Edge:
    r = signed_returns[np.isfinite(signed_returns)]
    if r.size < 2:
        return Edge(name, r.size, 0.0, 0.0, 0.0, cost_atr)
    mean = float(r.mean())
    se = float(r.std(ddof=1) / np.sqrt(r.size))
    return Edge(
        name=name,
        n=r.size,
        mean_atr=mean,
        t_stat=mean / se if se > 0 else 0.0,
        hit_rate=float((r > 0).mean()),
        cost_atr=cost_atr,
    )


def decompose(
    df: pd.DataFrame,
    strategy: MTFPullback,
    horizon: int,
    cost_price: float,
) -> list[Edge]:
    p = strategy.prepare(df.copy()).reset_index(drop=True)

    close = p["close"].to_numpy(dtype=float)
    a = p["atr"].to_numpy(dtype=float)
    bias = p["bias"].to_numpy(dtype=int)
    ext = p["ext_atr"].to_numpy(dtype=float)
    in_session = p["in_session"].to_numpy(dtype=bool)

    n = len(p)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = (close[horizon:] - close[: n - horizon]) / a[: n - horizon]

    # Non-overlapping windows only. Consecutive forward returns share almost all
    # of their content, and using them all would inflate every t-statistic below.
    keep = np.zeros(n, dtype=bool)
    keep[strategy.warmup : n - horizon : horizon] = True

    cost_atr = float(cost_price / np.nanmedian(a))
    edges: list[Edge] = []

    # --- baseline: buy every bar, no conditions ---------------------------
    base = keep & np.isfinite(fwd)
    edges.append(measure(fwd[base], "0. baseline - always long", cost_atr))

    # --- claim 1: does the higher-timeframe bias predict anything? ---------
    directional = base & (bias != 0)
    edges.append(measure(fwd[directional] * bias[directional],
                         "1. BIAS (H4+H1 agree)", cost_atr))

    sess = directional & in_session
    edges.append(measure(fwd[sess] * bias[sess],
                         "1b. BIAS + session filter", cost_atr))

    # --- claim 2: does entering at value beat entering anywhere? ------------
    at_value = sess & (np.abs(ext) <= strategy.max_extension_atr)
    extended = sess & (np.abs(ext) > strategy.max_extension_atr)
    edges.append(measure(fwd[at_value] * bias[at_value],
                         "2. + LOCATION (pulled back)", cost_atr))
    edges.append(measure(fwd[extended] * bias[extended],
                         "2b. control: extended (chasing)", cost_atr))

    # --- claim 3: does waiting for the trigger add anything? ---------------
    o = p["open"].to_numpy(dtype=float)
    h = p["high"].to_numpy(dtype=float)
    lo = p["low"].to_numpy(dtype=float)
    prev_h = np.roll(h, 1)
    prev_l = np.roll(lo, 1)
    up = (close > o) & (close > prev_h)
    down = (close < o) & (close < prev_l)
    fired = np.where(bias > 0, up, np.where(bias < 0, down, False))

    triggered = at_value & fired
    not_triggered = at_value & ~fired
    edges.append(measure(fwd[triggered] * bias[triggered],
                         "3. + TRIGGER (full strategy)", cost_atr))
    edges.append(measure(fwd[not_triggered] * bias[not_triggered],
                         "3b. control: no trigger", cost_atr))
    return edges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "XAUUSD"])
    ap.add_argument("--exec", dest="exec_tf", default="M15")
    ap.add_argument("--horizons", nargs="+", type=int, default=[8, 24])
    args = ap.parse_args()

    sys.argv = ["x"]
    from scripts.backtest_mtf import load_broker

    specs, costs, _ = load_broker()
    store = BarStore("data/bars")

    print(f"\nCOMPONENT DECOMPOSITION - {args.exec_tf} entry, H4+H1 bias")
    print("Forward return in the signalled direction, in ATR units. "
          "No stops, no targets.")
    print("'net' subtracts round-trip cost. t < 2 means no detectable signal.\n")

    for symbol in args.symbols:
        spec = specs[symbol]
        df = store.read(symbol, args.exec_tf)
        bias = load_bias_frames(store, symbol, ("H4", "H1"))
        if df.empty or not bias:
            print(f"{symbol}: no data")
            continue

        strategy = MTFPullback(
            execution_timeframe=args.exec_tf, bias_frames=bias,
            bias_timeframes=("H4", "H1"), stop_timeframe="H4",
        )
        cost_price = costs.for_symbol(symbol).round_trip_price()

        for horizon in args.horizons:
            print(f"{symbol}  horizon {horizon} bars "
                  f"({horizon * {'M5': 5, 'M15': 15, 'M30': 30}[args.exec_tf] / 60:.1f}h)")
            print(f"  {'component':<34}{'n':>7}{'mean':>9}{'t':>8}{'hit':>8}{'net':>10}")
            print("  " + "-" * 86)
            for edge in decompose(df, strategy, horizon, cost_price):
                print(edge)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
