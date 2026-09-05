"""Cross-sectional FX momentum and carry -- the way the published results are built.

Every test so far timed ONE instrument. The evidence for momentum (Moskowitz,
Ooi & Pedersen 2012) and for FX momentum specifically (Menkhoff, Sarno,
Schmeling & Schrinko 2012) is not about timing one pair. It is a PORTFOLIO:
rank many instruments, go long the ones that rose most, short the ones that
fell most, rebalance monthly. The edge lives in the relative ranking. Whether
EURUSD itself goes up next month is irrelevant to it.

Three constructions, each with real literature behind it:

- **XS momentum**: rank the universe by trailing f-month return; long the top
  quintile, short the bottom; hold one month. f in {1, 3, 6, 12}.
- **TSMOM portfolio**: each pair long or short by the sign of its own 12-month
  return, scaled by its own volatility, all averaged. Moskowitz et al.'s exact
  construction, which the single-instrument S1 test was not.
- **Carry**: long the pairs that PAY to hold, short the ones that charge. Only
  the broker's current swaps are available, not fifteen years of rate history,
  so this is tested only from 2023 where the rate structure resembles today's.
  Labelled as the approximation it is.

Disciplines carried over from the screen: long-short by construction (drift
cancels), t-stat on monthly returns, per-year consistency, Bonferroni across
every variant run here, and measured spreads charged on turnover. Menkhoff's
own finding is that FX momentum's profits sit in the pairs with the widest
spreads -- so the cost line is the result, not a footnote.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.store import BarStore  # noqa: E402


@dataclass
class PortfolioStats:
    name: str
    months: int
    ann_return: float
    ann_vol: float
    sharpe: float
    t: float
    max_dd: float
    years_pos: int
    years: int
    cost_drag_pct: float  # annual spread cost as % of gross annual return
    trials: int

    @property
    def verdict(self) -> str:
        if self.months < 24:
            return "too few months"
        bar = stats.norm.ppf(1 - 0.025 / self.trials)
        consistent = self.years >= 6 and self.years_pos / self.years >= 0.65
        if self.t >= bar and consistent and self.sharpe > 0.4:
            return "SURVIVES"
        if self.t >= bar:
            return "significant, inconsistent"
        if abs(self.t) >= 2 and consistent:
            return "consistent, below bar"
        return "-"


def load_panel(store: BarStore, pairs: list[str], start: str = "2010-01-01") -> pd.DataFrame:
    cols = {}
    for p in pairs:
        df = store.read(p, "D1")
        if df.empty:
            continue
        s = df.set_index(pd.to_datetime(df["ts"]).dt.normalize())["close"]
        cols[p] = s[~s.index.duplicated()]
    panel = pd.DataFrame(cols).sort_index()
    panel = panel.loc[start:].ffill(limit=3).dropna(axis=1, thresh=int(len(panel.loc[start:]) * 0.9))
    return panel.dropna()


def month_ends(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.resample("ME").last()


def describe(port: pd.Series, name: str, turnover: pd.Series, spread_pct: pd.Series,
             trials: int) -> PortfolioStats:
    port = port.dropna()
    if len(port) < 24:
        return PortfolioStats(name, len(port), 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, float("inf"), trials)
    gross_ann = port.mean() * 12
    vol = port.std() * np.sqrt(12)
    t, _ = stats.ttest_1samp(port, 0.0)
    eq = (1 + port).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    by_year = port.groupby(port.index.year).sum()
    # cost: each month, the fraction of the book that changes pays a round-trip
    # spread. spread_pct is per pair; use the mean of pairs actually traded.
    monthly_cost = (turnover * spread_pct.mean()).reindex(port.index).fillna(0.0)
    cost_ann = monthly_cost.mean() * 12
    return PortfolioStats(
        name=name, months=len(port), ann_return=float(gross_ann), ann_vol=float(vol),
        sharpe=float(gross_ann / vol) if vol else 0.0, t=float(t), max_dd=float(dd),
        years_pos=int((by_year > 0).sum()), years=int(len(by_year)),
        cost_drag_pct=float(cost_ann / gross_ann) if gross_ann > 0 else float("inf"),
        trials=trials,
    )


def xs_momentum(me: pd.DataFrame, lookback: int, top_n: int):
    rets = me.pct_change()
    signal = me.pct_change(lookback).shift(1)  # known at the START of the holding month
    port, turn, prev = [], [], set()
    for t in rets.index[lookback + 1:]:
        s = signal.loc[t].dropna()
        if len(s) < top_n * 2:
            port.append(np.nan); turn.append(np.nan); continue
        longs = set(s.nlargest(top_n).index); shorts = set(s.nsmallest(top_n).index)
        r = rets.loc[t]
        port.append(r[list(longs)].mean() - r[list(shorts)].mean())
        book = longs | shorts
        turn.append(len(book ^ prev) / max(len(book), 1) if prev else 1.0)
        prev = book
    idx = rets.index[lookback + 1:]
    return pd.Series(port, idx), pd.Series(turn, idx)


def tsmom_portfolio(me: pd.DataFrame, lookback: int = 12, target_vol: float = 0.10):
    rets = me.pct_change()
    sign = np.sign(me.pct_change(lookback)).shift(1)
    vol = rets.rolling(12).std().shift(1) * np.sqrt(12)
    weights = (sign * (target_vol / vol)).clip(-4, 4)
    port = (weights * rets).mean(axis=1)
    turn = weights.diff().abs().sum(axis=1) / weights.abs().sum(axis=1).replace(0, np.nan)
    return port.iloc[lookback + 1:], turn.iloc[lookback + 1:]


def carry_portfolio(me: pd.DataFrame, swaps: dict[str, float], top_n: int, start="2023-01-01"):
    rets = me.pct_change().loc[start:]
    s = pd.Series(swaps).reindex(me.columns).dropna()
    # A tiny universe let nlargest and nsmallest overlap - EURAUD sat on both
    # sides of the book. A pair can be long or short, never both.
    top_n = min(top_n, len(s) // 2)
    longs = list(s.nlargest(top_n).index); shorts = list(s.nsmallest(top_n).index)
    assert not set(longs) & set(shorts), "carry book overlaps"
    port = rets[longs].mean(axis=1) - rets[shorts].mean(axis=1)
    turn = pd.Series(0.0, index=port.index)  # static book
    return port, turn, longs, shorts


def main() -> int:
    store = BarStore("data/bars")
    universe = sys.argv[1] if len(sys.argv) > 1 else "config/fx_universe.json"
    since = sys.argv[2] if len(sys.argv) > 2 else "2010-01-01"
    specs = json.loads(Path(universe).read_text(encoding="utf-8"))
    print(f"universe file: {universe}, since {since}")
    pairs = sorted(specs)
    panel = load_panel(store, pairs, start=since)
    dropped = sorted(set(pairs) - set(panel.columns))
    if dropped:
        print(f"dropped for short history: {dropped}")
    me = month_ends(panel)
    print(f"universe: {panel.shape[1]} pairs, {panel.index[0]:%Y-%m} -> {panel.index[-1]:%Y-%m}, "
          f"{len(me)} month-ends\n")

    # round-trip spread as a fraction of price, per pair
    spread_pct = pd.Series({p: (specs[p]["spread_pts"] or 0) * specs[p]["point"] / panel[p].iloc[-1]
                            for p in panel.columns}) * 2.0

    # swap per lot per night -> % of notional per month, using the broker's own mode
    from core.types import SymbolSpec
    swap_m = {}
    for p in panel.columns:
        sp = {k: v for k, v in specs[p].items() if k != "spread_pts"}
        s = SymbolSpec(**sp)
        px = panel[p].iloc[-1]
        per_night = s.swap_cash_per_lot_night(True, px)   # long side
        notional = s.contract_size * px
        swap_m[p] = per_night / notional * 22            # ~22 rollovers a month
    top_n = max(3, panel.shape[1] // 5)

    runs = []
    for f in (1, 3, 6, 12):
        port, turn = xs_momentum(me, f, top_n)
        runs.append((f"xs_momentum_{f}m", port, turn))
    port, turn = tsmom_portfolio(me, 12)
    runs.append(("tsmom_12m_volscaled", port, turn))
    port, turn, L, S = carry_portfolio(me, swap_m, top_n)
    runs.append(("carry_2023+_APPROX", port, turn))
    trials = len(runs)

    print(f"  {'construction':<22}{'months':>7}{'ann ret':>9}{'vol':>7}{'sharpe':>8}{'t':>7}"
          f"{'maxDD':>8}{'years+':>9}{'cost/ret':>10}  verdict")
    print("  " + "-" * 96)
    results = []
    for name, port, turn in runs:
        r = describe(port, name, turn, spread_pct, trials)
        results.append(r)
        print(f"  {r.name:<22}{r.months:>7}{r.ann_return:>+9.1%}{r.ann_vol:>7.1%}{r.sharpe:>8.2f}{r.t:>7.2f}"
              f"{r.max_dd:>8.1%}{r.years_pos:>5}/{r.years:<3}{r.cost_drag_pct:>10.1%}  {r.verdict}")
    print(f"\n  Bonferroni bar at {trials} trials: t > {stats.norm.ppf(1 - 0.025 / trials):.2f}")
    print(f"  carry book (2023+, current swaps):  long {L}   short {S}")

    best = max(results, key=lambda r: r.t)
    if best.verdict == "SURVIVES":
        print(f"\n  SURVIVOR: {best.name}. Next: full backtest through the risk engine, then the gauntlet.")
    else:
        print(f"\n  No construction clears the bar. Best was {best.name} at t={best.t:.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
