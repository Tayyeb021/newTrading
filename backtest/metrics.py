"""Performance metrics.

Two rules govern what is reported here.

**Everything is net of costs.** A gross figure is not a result; it is an upper
bound on a result. Where gross appears it is only so that cost drag can be shown
as a share of it.

**Cost drag is reported at the top, not buried.** It is the number that decides
whether a strategy is real at your broker, and it is the first thing that should
be checked when a promising backtest meets live trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, Trade

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    trades: int
    win_rate: float
    payoff: float  # avg win / avg loss, in R
    expectancy_r: float
    profit_factor: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_dd_days: float
    calmar: float
    avg_bars_held: float
    gross_pnl: float
    total_costs: float
    cost_drag: float  # costs / gross profit
    best_r: float
    worst_r: float
    avg_mae: float
    avg_mfe: float
    exposure: float

    def verdict(self) -> str:
        """A blunt read, using the thresholds from the strategy research."""
        if self.trades < 30:
            return "INCONCLUSIVE - too few trades to say anything"
        if self.expectancy_r <= 0:
            return "DEAD - negative expectancy after costs"
        if self.cost_drag > 0.5:
            return "COST-DOMINATED - friction eats over half the gross edge"
        if self.sharpe < 0.5:
            return "WEAK - below anything worth trading"
        if self.sharpe < 1.0:
            return "MARGINAL - real but thin; check it survives 2x costs"
        if self.sharpe < 2.0:
            return "WORKABLE - validate properly before believing it"
        return "STRONG - and therefore suspicious; check for look-ahead first"


EMPTY = Metrics(
    trades=0, win_rate=0.0, payoff=0.0, expectancy_r=0.0, profit_factor=0.0,
    total_return=0.0, cagr=0.0, sharpe=0.0, sortino=0.0, max_drawdown=0.0,
    max_dd_days=0.0, calmar=0.0, avg_bars_held=0.0, gross_pnl=0.0, total_costs=0.0,
    cost_drag=0.0, best_r=0.0, worst_r=0.0, avg_mae=0.0, avg_mfe=0.0, exposure=0.0,
)


def _drawdown(equity: pd.Series) -> tuple[float, float]:
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak
    max_dd = float(dd.max())

    longest, current, prev_peak = 0.0, 0.0, equity.iloc[0]
    start = equity.index[0]
    for ts, value in equity.items():
        if value >= prev_peak:
            prev_peak = value
            start = ts
            current = 0.0
        else:
            current = (ts - start).total_seconds() / 86400.0
            longest = max(longest, current)
    return max_dd, longest


def _annualisation(equity: pd.Series) -> float:
    """Periods per year implied by the index spacing."""
    if len(equity) < 3:
        return TRADING_DAYS
    median_gap = pd.Series(equity.index).diff().dt.total_seconds().median()
    if not median_gap or math.isnan(median_gap):
        return TRADING_DAYS
    per_day = 86400.0 / median_gap
    return TRADING_DAYS * max(per_day, 1.0) if per_day > 1 else TRADING_DAYS


def compute(result: BacktestResult) -> Metrics:
    trades: list[Trade] = result.trades
    equity = result.equity.dropna()

    if not trades or equity.empty:
        return EMPTY

    rs = np.array([t.r_multiple for t in trades])
    wins, losses = rs[rs > 0], rs[rs <= 0]

    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(abs(losses.mean())) if losses.size else 0.0
    gross_win = float(sum(t.net_pnl for t in trades if t.net_pnl > 0))
    gross_loss = float(abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0)))

    returns = equity.pct_change().dropna()
    periods = _annualisation(equity)
    ann_ret = float(returns.mean() * periods)
    ann_vol = float(returns.std() * math.sqrt(periods))
    downside = returns[returns < 0]
    down_vol = float(downside.std() * math.sqrt(periods)) if len(downside) > 1 else 0.0

    max_dd, dd_days = _drawdown(equity)
    span_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total_return = float(equity.iloc[-1] / result.starting_equity - 1.0)
    cagr = float((equity.iloc[-1] / result.starting_equity) ** (1 / span_years) - 1.0)

    gross = float(sum(t.gross_pnl for t in trades))
    costs = float(sum(t.costs for t in trades))
    gross_profit_only = float(sum(t.gross_pnl for t in trades if t.gross_pnl > 0))

    held = float(np.mean([t.bars_held for t in trades]))
    exposure = min(sum(t.bars_held for t in trades) / (span_years * 365.25), 1.0)

    return Metrics(
        trades=len(trades),
        win_rate=float(len(wins) / len(rs)),
        payoff=avg_win / avg_loss if avg_loss else float("inf"),
        expectancy_r=float(rs.mean()),
        profit_factor=gross_win / gross_loss if gross_loss else float("inf"),
        total_return=total_return,
        cagr=cagr,
        sharpe=ann_ret / ann_vol if ann_vol else 0.0,
        sortino=ann_ret / down_vol if down_vol else 0.0,
        max_drawdown=max_dd,
        max_dd_days=dd_days,
        calmar=cagr / max_dd if max_dd else 0.0,
        avg_bars_held=held,
        gross_pnl=gross,
        total_costs=costs,
        cost_drag=costs / gross_profit_only if gross_profit_only else float("inf"),
        best_r=float(rs.max()),
        worst_r=float(rs.min()),
        avg_mae=float(np.mean([t.mae for t in trades])),
        avg_mfe=float(np.mean([t.mfe for t in trades])),
        exposure=exposure,
    )


def report(result: BacktestResult, m: Metrics | None = None) -> str:
    m = m or compute(result)
    L: list[str] = []
    L.append(f"\n{result.strategy} on {result.symbol}")
    L.append("=" * 66)

    if not m.trades:
        L.append("  no trades")
        if result.rejections:
            L.append(f"  rejections: {result.rejections}")
        return "\n".join(L)

    span = f"{result.equity.index[0]:%Y-%m-%d} -> {result.equity.index[-1]:%Y-%m-%d}"
    L.append(f"  period            {span}")
    L.append(f"  equity            {result.starting_equity:,.0f} -> {result.final_equity:,.0f}")
    L.append("")
    L.append(f"  {'RETURN':<20}{'RISK':<24}{'TRADES'}")
    L.append(f"  total   {m.total_return:>9.1%}   max DD    {m.max_drawdown:>9.2%}   n         {m.trades:>8,}")
    L.append(f"  CAGR    {m.cagr:>9.1%}   DD days   {m.max_dd_days:>9,.0f}   win rate  {m.win_rate:>8.1%}")
    L.append(f"  Sharpe  {m.sharpe:>9.2f}   Calmar    {m.calmar:>9.2f}   payoff    {m.payoff:>8.2f}R")
    L.append(f"  Sortino {m.sortino:>9.2f}   exposure  {m.exposure:>9.1%}   expectancy{m.expectancy_r:>8.3f}R")
    L.append("")
    L.append(f"  profit factor {m.profit_factor:>7.2f}      best {m.best_r:>7.2f}R      worst {m.worst_r:>7.2f}R")
    L.append(f"  avg hold {m.avg_bars_held:>8.1f}d      MAE  {m.avg_mae:>7.2f}R      MFE   {m.avg_mfe:>7.2f}R")
    L.append("")
    L.append("  COSTS")
    L.append(f"  gross {m.gross_pnl:>12,.0f}   costs {m.total_costs:>10,.0f}   "
             f"drag on gross profit {m.cost_drag:>7.1%}")
    if not result.cost_model_calibrated:
        L.append("  ! cost model is UNCALIBRATED - placeholder values, not your broker's")

    if result.rejections:
        L.append("")
        L.append("  RISK ENGINE REJECTIONS")
        for name, count in sorted(result.rejections.items(), key=lambda kv: -kv[1]):
            L.append(f"    {name:<28}{count:>6,}")

    L.append("")
    L.append(f"  {m.verdict()}")
    return "\n".join(L)


def r_histogram(result: BacktestResult, bins: int = 12, width: int = 40) -> str:
    """Distribution of R-multiples. The shape tells you what you are trading.

    Trend following looks like a wall at -1R with a long thin right tail. A
    mean-reversion system looks like the mirror image. If yours looks like
    neither, you may not be trading what you think you are.
    """
    if not result.trades:
        return "  no trades"
    rs = np.array([t.r_multiple for t in result.trades])
    counts, edges = np.histogram(rs, bins=bins)
    peak = counts.max() or 1

    lines = ["  R-multiple distribution"]
    for count, lo, hi in zip(counts, edges, edges[1:]):
        bar = "#" * int(width * count / peak)
        marker = " <- 0" if lo <= 0 <= hi else ""
        lines.append(f"    {lo:>6.2f}..{hi:>6.2f} {bar:<{width}} {count:>5,}{marker}")
    return "\n".join(lines)
