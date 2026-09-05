"""Portfolio backtester: many sleeves, many symbols, ONE equity, ONE risk engine.

The single-symbol `Backtester` answers "does this strategy work on this
instrument". This answers the question a book actually poses: **do these sleeves
work together**, with every account-level limit acting on the whole position
set at once. Six sleeves each risking 0.5% are not six independent bets when
they share a drawdown limit -- and the only way to find out is to run them
through the same engine on the same clock.

Mechanically it is N legs, one per (sleeve, symbol), stepped in lockstep on a
master timeline built from the union of their bar timestamps. Each leg keeps
its own position and its own pending intent; they share cash, the risk book,
and the position list every limit sees. When leg A asks to open, the risk
state it is evaluated against already contains leg B's open trade.

Per-leg semantics are identical to `Backtester` -- the same pessimism about
fills, gaps and stops -- and `tests/test_portfolio.py` proves it: one sleeve on
one symbol produces the same trades to the cent.

Attribution and sleeve correlation come out of the same run. Correlation above
0.6 between two sleeves means you have one strategy wearing two hats, and the
diversification you are counting on is not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.engine import Trade
from core.sleeve import Sleeve, normalise_weights, tag
from core.strategy import Intent, Strategy
from core.types import Position, Side, Signal, SymbolSpec
from risk.engine import RiskEngine


@dataclass
class Leg:
    sleeve: Sleeve
    symbol: str
    strategy: Strategy
    spec: SymbolSpec
    df: pd.DataFrame
    index_of: dict[datetime, int]
    position: Position | None = None
    entry_ts: datetime | None = None
    stop_price: float = 0.0
    risk_cash: float = 0.0
    exc_hi: float = 0.0
    exc_lo: float = 0.0
    pending: Intent | None = None
    entry_cost: float = 0.0
    cum_pnl: float = 0.0  # realised, for the per-sleeve curve
    last_close: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.sleeve.name, self.symbol)


@dataclass
class PortfolioResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sleeve_equity: dict[str, pd.Series] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    starting_equity: float = 0.0
    halted_at: datetime | None = None
    evaluations_failed: int = 0
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.starting_equity


class PortfolioBacktester:
    def __init__(
        self,
        sleeves: list[Sleeve],
        specs: dict[str, SymbolSpec],
        risk: RiskEngine,
        costs: CostModel,
        starting_equity: float = 100_000.0,
        reset_on_halt: bool = False,
    ) -> None:
        if not sleeves:
            raise ValueError("a portfolio needs at least one sleeve")
        names = [s.name for s in sleeves]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate sleeve names: {names}")
        self.sleeves = sleeves
        self.specs = specs
        self.risk = risk
        self.costs = costs
        self.starting_equity = starting_equity
        self.reset_on_halt = reset_on_halt
        self.weights = normalise_weights(sleeves)

    # ------------------------------------------------------------------ run

    def run(self, bars: dict[tuple[str, str], pd.DataFrame]) -> PortfolioResult:
        """`bars` maps (sleeve name, symbol) -> OHLCV frame on that sleeve's timeframe."""
        legs = self._build_legs(bars)
        equity = self.starting_equity
        self.risk.book.starting_equity = equity
        self.risk.book.day_start_equity = equity
        self.risk.book.high_water_equity = equity

        result = PortfolioResult(starting_equity=equity, weights=dict(self.weights))
        timeline = sorted({ts for leg in legs for ts in leg.index_of})
        curve, stamps = [], []
        sleeve_curves: dict[str, list[float]] = {s.name: [] for s in self.sleeves}

        for ts in timeline:
            self.risk.book.observe_equity(equity, today=ts.date())
            active = [leg for leg in legs if ts in leg.index_of]

            # 1. execute what each leg decided on its previous bar, at this open
            for leg in active:
                if leg.pending is not None:
                    row = leg.df.iloc[leg.index_of[ts]]
                    equity = self._apply_intent(leg, legs, row, ts, equity, result)
                    leg.pending = None

            # 2. resolve open positions against this bar's range
            for leg in active:
                if leg.position is None:
                    continue
                row = leg.df.iloc[leg.index_of[ts]]
                self._track_excursion(leg, row)
                exit_price, reason = self._check_stop(leg, row)
                if exit_price is not None:
                    equity = self._close(leg, exit_price, ts, reason, equity, result)

            # 3. mark to market, portfolio and per sleeve. A leg with no bar at
            #    this timestamp is marked at its last known close.
            for leg in active:
                leg.last_close = float(leg.df.iloc[leg.index_of[ts]]["close"])
            mark = equity
            unreal = {s.name: 0.0 for s in self.sleeves}
            realised = {s.name: 0.0 for s in self.sleeves}
            for leg in legs:
                realised[leg.sleeve.name] += leg.cum_pnl
                if leg.position is not None and leg.last_close is not None:
                    u = leg.position.unrealized(leg.last_close, leg.spec)
                    mark += u
                    unreal[leg.sleeve.name] += u
            curve.append(mark)
            stamps.append(ts)
            for name in unreal:
                sleeve_curves[name].append(realised[name] + unreal[name])

            # 4. decide for the NEXT bar, closed data only
            for leg in active:
                i = leg.index_of[ts]
                if i >= leg.strategy.warmup and i < len(leg.df) - 1:
                    leg.pending = leg.strategy.evaluate(leg.df, i, leg.position)

        idx = pd.DatetimeIndex(stamps)
        result.equity = pd.Series(curve, index=idx, name="equity")
        result.sleeve_equity = {
            name: pd.Series(vals, index=idx, name=name) for name, vals in sleeve_curves.items()
        }

        for leg in legs:
            if leg.position is not None:
                last = leg.df.iloc[-1]
                self._close(leg, float(last["close"]), last["ts"].to_pydatetime(),
                            "end_of_data", equity, result)
        return result

    # ------------------------------------------------------------- building

    def _build_legs(self, bars) -> list[Leg]:
        legs: list[Leg] = []
        for sleeve in self.sleeves:
            for symbol in sleeve.symbols:
                df = bars.get((sleeve.name, symbol))
                if df is None or df.empty:
                    raise ValueError(f"no bars supplied for sleeve {sleeve.name} on {symbol}")
                spec = self.specs.get(symbol)
                if spec is None:
                    raise ValueError(f"no spec for {symbol}")
                strategy = sleeve.build(symbol)
                prepared = strategy.prepare(df.copy()).reset_index(drop=True)
                prepared["ts"] = pd.to_datetime(prepared["ts"], utc=True)
                if len(prepared) < strategy.warmup + 2:
                    raise ValueError(f"{sleeve.name}/{symbol}: {len(prepared)} bars < warmup")
                index_of = {t.to_pydatetime(): i for i, t in enumerate(prepared["ts"])}
                legs.append(Leg(sleeve, symbol, strategy, spec, prepared, index_of))
        return legs

    # ------------------------------------------------------------- per-leg

    def _apply_intent(self, leg: Leg, legs: list[Leg], row, ts, equity, result) -> float:
        intent = leg.pending
        open_price = float(row["open"])

        if leg.position is not None and (intent.flat or intent.side is not leg.position.side):
            fill = self._fill(leg, open_price, leg.position.side.opposite(), ts)
            equity = self._close(leg, fill, ts, "signal_exit", equity, result, open_price)

        if intent.flat or leg.position is not None:
            return equity

        others = [l.position for l in legs if l is not leg and l.position is not None]
        state = self.risk.snapshot(
            equity=equity, balance=equity, margin_level=float("inf"),
            positions=others, now=ts, current_price={leg.symbol: open_price},
        )
        signal = Signal(
            symbol=leg.symbol, side=intent.side, stop_distance=intent.stop_distance,
            confidence=intent.confidence, strategy=leg.strategy.name, ts=ts,
        )
        decision = self.risk.evaluate(
            signal, state, risk_fraction=leg.sleeve.risk_per_trade,
        )
        if not decision.approved:
            key = decision.breaches[0].limit if decision.breaches else decision.note.split(":")[0]
            result.rejections[key] = result.rejections.get(key, 0) + 1
            if decision.must_halt:
                if result.halted_at is None:
                    result.halted_at = ts
                if self.reset_on_halt:
                    result.evaluations_failed += 1
                    self.risk.book.starting_equity = equity
                    self.risk.book.high_water_equity = equity
                    self.risk.book.day_start_equity = equity
            return equity

        fill = self._fill(leg, open_price, intent.side, ts)
        volume = decision.order.volume
        commission = self.costs.commission(leg.symbol, volume)
        equity -= commission
        leg.entry_cost = commission + self._fill_cost_cash(leg, open_price, fill, volume)
        stop = fill - intent.stop_distance * intent.side.sign
        leg.position = Position(
            symbol=leg.symbol, side=intent.side, volume=volume, entry_price=fill,
            opened_at=ts, stop_loss=stop, comment=tag(leg.sleeve.name),
        )
        leg.entry_ts, leg.stop_price = ts, stop
        leg.risk_cash = leg.spec.risk_for(volume, intent.stop_distance)
        leg.exc_hi = leg.exc_lo = 0.0
        return equity

    def _fill(self, leg: Leg, reference: float, side: Side, ts) -> float:
        return self.costs.entry_price(leg.symbol, reference, side.sign, False)

    def _fill_cost_cash(self, leg: Leg, reference, fill, volume) -> float:
        return abs(fill - reference) * volume * leg.spec.value_per_price_unit

    def _check_stop(self, leg: Leg, row):
        pos, stop = leg.position, leg.stop_price
        high, low, open_ = float(row["high"]), float(row["low"]), float(row["open"])
        if pos.side is Side.BUY:
            if open_ <= stop:
                return open_, "gap_through_stop"
            if low <= stop:
                return stop, "stop"
        else:
            if open_ >= stop:
                return open_, "gap_through_stop"
            if high >= stop:
                return stop, "stop"
        return None, ""

    def _track_excursion(self, leg: Leg, row) -> None:
        if leg.risk_cash <= 0:
            return
        pos = leg.position
        best = pos.unrealized(float(row["high"]) if pos.side is Side.BUY else float(row["low"]), leg.spec)
        worst = pos.unrealized(float(row["low"]) if pos.side is Side.BUY else float(row["high"]), leg.spec)
        leg.exc_hi = max(leg.exc_hi, best / leg.risk_cash)
        leg.exc_lo = min(leg.exc_lo, worst / leg.risk_cash)

    def _close(self, leg: Leg, exit_price, ts, reason, equity, result, exit_reference=None) -> float:
        pos = leg.position
        gross = pos.unrealized(exit_price, leg.spec)
        commission = self.costs.commission(leg.symbol, pos.volume)
        equity += gross - commission

        friction = commission
        if exit_reference is not None:
            friction += self._fill_cost_cash(leg, exit_reference, exit_price, pos.volume)
        swap = self.costs.swap_cash(
            leg.symbol, pos.volume, leg.entry_ts, ts,
            is_long=pos.side is Side.BUY, spec=leg.spec, price=pos.entry_price,
        )
        equity -= swap
        total_costs = leg.entry_cost + friction + swap
        leg.entry_cost = 0.0

        trade = Trade(
            symbol=pos.symbol, strategy=leg.strategy.name, side=pos.side, volume=pos.volume,
            entry_ts=leg.entry_ts, entry_price=pos.entry_price, exit_ts=ts, exit_price=exit_price,
            stop_price=leg.stop_price, exit_reason=reason, gross_pnl=gross, costs=total_costs,
            mae=leg.exc_lo, mfe=leg.exc_hi, risk_cash=leg.risk_cash, sleeve=leg.sleeve.name,
        )
        result.trades.append(trade)
        leg.cum_pnl += trade.net_pnl
        self.risk.book.record_close(leg.strategy.name, trade.net_pnl, ts)
        leg.position = None
        return equity


# --------------------------------------------------------------------------- #
# Attribution and correlation
# --------------------------------------------------------------------------- #


def attribution(result: PortfolioResult) -> pd.DataFrame:
    """Where the P&L came from. Sums to the portfolio's realised total."""
    rows = []
    total = sum(t.net_pnl for t in result.trades) or 1.0
    for name in result.weights:
        ts = [t for t in result.trades if t.sleeve == name]
        if not ts:
            rows.append({"sleeve": name, "trades": 0, "net_pnl": 0.0, "share": 0.0,
                         "expectancy_r": 0.0, "win_rate": 0.0, "weight": result.weights[name]})
            continue
        rs = np.array([t.r_multiple for t in ts])
        pnl = float(sum(t.net_pnl for t in ts))
        rows.append({
            "sleeve": name, "trades": len(ts), "net_pnl": pnl, "share": pnl / total,
            "expectancy_r": float(rs.mean()), "win_rate": float((rs > 0).mean()),
            "weight": result.weights[name],
        })
    return pd.DataFrame(rows).set_index("sleeve")


def sleeve_correlation(result: PortfolioResult, freq: str = "W") -> pd.DataFrame:
    """Correlation of sleeve returns at `freq`. This is the number the whole
    diversification argument rests on, so it is measured, not assumed."""
    if not result.sleeve_equity:
        return pd.DataFrame()
    frame = pd.DataFrame(result.sleeve_equity)
    rets = frame.resample(freq).last().diff().dropna(how="all")
    rets = rets.loc[:, rets.std() > 0]
    return rets.corr()


def flag_correlated(corr: pd.DataFrame, threshold: float = 0.6) -> list[tuple[str, str, float]]:
    """Pairs of sleeves that are really one strategy."""
    out = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = float(corr.loc[a, b])
            if np.isfinite(rho) and rho >= threshold:
                out.append((a, b, rho))
    return sorted(out, key=lambda x: -x[2])


def diversification_ratio(corr: pd.DataFrame, weights: dict[str, float]) -> float:
    """Sharpe multiplier the book earns over its average sleeve: sqrt(n/(1+(n-1)rho)).

    With rho measured rather than hoped for. 1.0 means the sleeves add nothing to
    each other; 2.0 means four uncorrelated sleeves doing what the maths promised.
    """
    names = [n for n in weights if n in corr.columns]
    n = len(names)
    if n < 2:
        return 1.0
    off = [corr.loc[a, b] for i, a in enumerate(names) for b in names[i + 1:]]
    rho = float(np.nanmean(off)) if off else 0.0
    return float(np.sqrt(n / (1 + (n - 1) * max(rho, -1 / (n - 1) + 1e-9))))


def portfolio_report(result: PortfolioResult) -> str:
    from backtest.metrics import _drawdown

    L = ["", "PORTFOLIO", "=" * 70]
    eq = result.equity.dropna()
    if eq.empty:
        return "\n".join(L + ["  no equity curve"])
    ret = eq.pct_change().dropna()
    per_year = 252 if (eq.index[1] - eq.index[0]).days <= 1 else 52
    sharpe = float(ret.mean() / ret.std() * np.sqrt(per_year)) if ret.std() else 0.0
    dd, _ = _drawdown(eq)
    L.append(f"  equity   {result.starting_equity:,.0f} -> {result.final_equity:,.0f}"
             f"   Sharpe {sharpe:.2f}   max DD {dd:.1%}   trades {len(result.trades)}")
    if result.halted_at:
        L.append(f"  ! halted {result.halted_at:%Y-%m-%d}"
                 + (f", evaluation failed {result.evaluations_failed}x" if result.evaluations_failed else ""))

    att = attribution(result)
    L.append("")
    L.append(f"  {'sleeve':<14}{'weight':>8}{'trades':>8}{'net pnl':>12}{'share':>8}{'expect':>9}{'win':>7}")
    L.append("  " + "-" * 66)
    for name, r in att.iterrows():
        L.append(f"  {name:<14}{r.weight:>8.0%}{int(r.trades):>8}{r.net_pnl:>12,.0f}{r.share:>8.0%}"
                 f"{r.expectancy_r:>+9.3f}{r.win_rate:>7.0%}")

    corr = sleeve_correlation(result)
    if len(corr) >= 2:
        L.append("")
        L.append("  sleeve return correlation (weekly)")
        L.append("  " + corr.round(2).to_string().replace("\n", "\n  "))
        flags = flag_correlated(corr)
        dr = diversification_ratio(corr, result.weights)
        L.append(f"\n  diversification ratio: {dr:.2f}x  "
                 f"(the Sharpe multiple the book earns over its average sleeve)")
        for a, b, rho in flags:
            L.append(f"  ! {a} and {b} correlate at {rho:.2f} - one strategy wearing two hats")
    if result.rejections:
        L.append("")
        L.append("  rejections: " + ", ".join(f"{k} {v}" for k, v in
                                              sorted(result.rejections.items(), key=lambda kv: -kv[1])))
    return "\n".join(L)
