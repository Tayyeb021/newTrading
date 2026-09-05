"""Event-driven backtester.

**The property that matters:** this replays bars through the same `RiskEngine`
that runs live. Not a copy of it, not a simplified version — the same object,
the same limits register, the same sizing code, loaded from the same YAML. A
backtest that uses different risk logic than production is testing a system you
will never trade.

Deliberate pessimism throughout:

- Signals are computed on **closed** bars; entry happens at the **next** bar's
  open. Same-bar entry on a signal derived from that bar's close is look-ahead.
- When a bar's range contains both the stop and the target, the **stop** is
  assumed to have hit first. OHLC cannot resolve the order, and assuming the
  favourable one is how backtests invent profits that never existed.
- Costs are charged on both fills, and the spread widens at the session open.
- Gaps through a stop fill at the **open**, not the stop price, which is what a
  broker actually does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from backtest.costs import CostModel
from core.strategy import Intent, Strategy
from core.types import Position, Side, Signal, SymbolSpec
from risk.engine import RiskEngine


@dataclass(frozen=True)
class Trade:
    symbol: str
    strategy: str
    side: Side
    volume: float
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    stop_price: float
    exit_reason: str
    gross_pnl: float
    #: ALL friction in account currency: spread and slippage embedded in both
    #: fills, plus commission on both sides. Charging only commission here made
    #: cost_drag report 0% on a spread-only broker, which is the exact metric
    #: that decides whether a strategy is viable.
    costs: float
    mae: float  # worst unrealised excursion, in R
    mfe: float  # best unrealised excursion, in R
    risk_cash: float

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def r_multiple(self) -> float:
        """Net result in units of initial risk. The only comparable trade metric."""
        return self.net_pnl / self.risk_cash if self.risk_cash else 0.0

    @property
    def bars_held(self) -> float:
        return (self.exit_ts - self.entry_ts).total_seconds() / 86400.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rejections: dict[str, int] = field(default_factory=dict)
    starting_equity: float = 0.0
    symbol: str = ""
    strategy: str = ""
    cost_model_calibrated: bool = False
    #: When an account-level HALT first engaged. Live, a human reviews and
    #: restarts; in a backtest nothing does, so one bad year silently blanks
    #: the next thirteen. Found when US30 showed 17 trades in 14 years with
    #: 2,525 signals rejected by max_drawdown.
    halted_at: datetime | None = None
    #: With `reset_on_halt`, how many times the evaluation would have failed.
    evaluations_failed: int = 0

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.starting_equity


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        spec: SymbolSpec,
        risk: RiskEngine,
        costs: CostModel,
        starting_equity: float = 100_000.0,
        session_open_hour_utc: int | None = None,
        reset_on_halt: bool = False,
    ) -> None:
        self.strategy = strategy
        self.spec = spec
        self.risk = risk
        self.costs = costs
        self.starting_equity = starting_equity
        self.session_open_hour = session_open_hour_utc
        #: Research mode. On an account-level HALT, count a failed evaluation
        #: and re-base the drawdown reference to current equity - simulating
        #: "review, restart a fresh evaluation" - so the strategy's behaviour
        #: over the whole history stays observable. Off by default: the
        #: challenge-mode answer ("you would have failed in 2011") is also real.
        self.reset_on_halt = reset_on_halt

    def run(self, df: pd.DataFrame) -> BacktestResult:
        symbol = self.spec.symbol
        df = self.strategy.prepare(df.copy()).reset_index(drop=True)
        if len(df) < self.strategy.warmup + 2:
            raise ValueError(
                f"{symbol}: {len(df)} bars is not enough for warmup {self.strategy.warmup}"
            )

        equity = self.starting_equity
        self.risk.book.starting_equity = equity
        self.risk.book.day_start_equity = equity
        self.risk.book.high_water_equity = equity

        position: Position | None = None
        entry_ts: datetime | None = None
        stop_price = 0.0
        risk_cash = 0.0
        excursion_hi = 0.0
        excursion_lo = 0.0

        result = BacktestResult(
            starting_equity=equity, symbol=symbol, strategy=self.strategy.name,
            cost_model_calibrated=self.costs.calibrated,
        )
        curve: list[float] = []
        stamps: list[datetime] = []

        pending: Intent | None = None
        self._entry_cost = 0.0

        for i in range(len(df)):
            row = df.iloc[i]
            ts = row["ts"].to_pydatetime()
            self.risk.book.observe_equity(equity, today=ts.date())

            # ---- 1. execute what the previous bar decided, at THIS bar's open
            if pending is not None:
                position, entry_ts, stop_price, risk_cash, equity = self._apply_intent(
                    pending, position, row, ts, equity, result, entry_ts,
                    stop_price, risk_cash, excursion_hi, excursion_lo,
                )
                if position is not None and entry_ts == ts:
                    excursion_hi = excursion_lo = 0.0
                pending = None

            # ---- 2. resolve an open position against this bar's range
            if position is not None:
                excursion_hi, excursion_lo = self._track_excursion(
                    position, row, risk_cash, excursion_hi, excursion_lo
                )
                exit_price, reason = self._check_stop(position, row, stop_price)
                if exit_price is not None:
                    equity = self._close(
                        position, entry_ts, stop_price, risk_cash, exit_price, ts,
                        reason, equity, result, excursion_hi, excursion_lo,
                    )
                    position = None

            # ---- 3. mark to market
            close = float(row["close"])
            mark = equity
            if position is not None:
                mark = equity + position.unrealized(close, self.spec)
            curve.append(mark)
            stamps.append(ts)

            # ---- 4. decide for the NEXT bar, using only closed data
            if i >= self.strategy.warmup and i < len(df) - 1:
                pending = self.strategy.evaluate(df, i, position)

        result.equity = pd.Series(curve, index=pd.DatetimeIndex(stamps), name="equity")

        # An open position at the end is closed at the last close, marked as such
        # so it can be excluded from statistics if you prefer.
        if position is not None:
            last = df.iloc[-1]
            self._close(
                position, entry_ts, stop_price, risk_cash, float(last["close"]),
                last["ts"].to_pydatetime(), "end_of_data", equity, result,
                excursion_hi, excursion_lo,
            )
        return result

    # ------------------------------------------------------------------ helpers

    def _apply_intent(
        self, intent: Intent, position: Position | None, row, ts: datetime,
        equity: float, result: BacktestResult, entry_ts, stop_price, risk_cash,
        mfe: float = 0.0, mae: float = 0.0,
    ):
        open_price = float(row["open"])

        # Close first if the intent contradicts the open position.
        if position is not None and (intent.flat or intent.side is not position.side):
            equity = self._close(
                position, entry_ts, stop_price, risk_cash,
                self._fill(open_price, position.side.opposite(), ts),
                ts, "signal_exit", equity, result, mfe, mae, open_price,
            )
            position = None

        if intent.flat or position is not None:
            return position, entry_ts, stop_price, risk_cash, equity

        state = self.risk.snapshot(
            equity=equity, balance=equity, margin_level=float("inf"),
            positions=[], now=ts, current_price={self.spec.symbol: open_price},
        )
        signal = Signal(
            symbol=self.spec.symbol, side=intent.side,
            stop_distance=intent.stop_distance, confidence=intent.confidence,
            strategy=self.strategy.name, ts=ts,
        )
        decision = self.risk.evaluate(signal, state)
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
            return None, None, 0.0, 0.0, equity

        fill = self._fill(open_price, intent.side, ts)
        volume = decision.order.volume
        commission = self.costs.commission(self.spec.symbol, volume)
        equity -= commission
        self._entry_cost = commission + self._fill_cost_cash(open_price, fill, volume)

        new_stop = fill - intent.stop_distance * intent.side.sign
        return (
            Position(
                symbol=self.spec.symbol, side=intent.side, volume=volume,
                entry_price=fill, opened_at=ts, stop_loss=new_stop,
            ),
            ts,
            new_stop,
            self.spec.risk_for(volume, intent.stop_distance),
            equity,
        )

    def _fill(self, reference: float, side: Side, ts: datetime) -> float:
        at_open = self.session_open_hour is not None and ts.hour == self.session_open_hour
        return self.costs.entry_price(self.spec.symbol, reference, side.sign, at_open)

    def _fill_cost_cash(self, reference: float, fill: float, volume: float) -> float:
        """Cash value of the gap between the price asked for and the one paid."""
        return abs(fill - reference) * volume * self.spec.value_per_price_unit

    def _check_stop(self, position: Position, row, stop_price: float):
        """Did this bar take the stop out? Gaps fill at the open, not the stop."""
        high, low, open_ = float(row["high"]), float(row["low"]), float(row["open"])
        if position.side is Side.BUY:
            if open_ <= stop_price:
                return open_, "gap_through_stop"
            if low <= stop_price:
                return stop_price, "stop"
        else:
            if open_ >= stop_price:
                return open_, "gap_through_stop"
            if high >= stop_price:
                return stop_price, "stop"
        return None, ""

    def _track_excursion(self, position: Position, row, risk_cash: float, hi: float, lo: float):
        if risk_cash <= 0:
            return hi, lo
        best = position.unrealized(
            float(row["high"]) if position.side is Side.BUY else float(row["low"]), self.spec
        )
        worst = position.unrealized(
            float(row["low"]) if position.side is Side.BUY else float(row["high"]), self.spec
        )
        return max(hi, best / risk_cash), min(lo, worst / risk_cash)

    def _close(
        self, position: Position, entry_ts, stop_price, risk_cash, exit_price,
        ts, reason, equity, result, mfe, mae, exit_reference: float | None = None,
    ) -> float:
        gross = position.unrealized(exit_price, self.spec)
        commission = self.costs.commission(self.spec.symbol, position.volume)
        equity += gross - commission

        exit_friction = commission
        if exit_reference is not None:
            exit_friction += self._fill_cost_cash(exit_reference, exit_price, position.volume)

        swap = self.costs.swap_cash(
            self.spec.symbol, position.volume, entry_ts, ts,
            is_long=position.side is Side.BUY, spec=self.spec, price=position.entry_price,
        )
        equity -= swap
        exit_friction += swap
        total_costs = getattr(self, "_entry_cost", commission) + exit_friction
        self._entry_cost = 0.0

        result.trades.append(
            Trade(
                symbol=position.symbol, strategy=self.strategy.name, side=position.side,
                volume=position.volume, entry_ts=entry_ts, entry_price=position.entry_price,
                exit_ts=ts, exit_price=exit_price, stop_price=stop_price,
                exit_reason=reason, gross_pnl=gross, costs=total_costs,
                mae=mae, mfe=mfe, risk_cash=risk_cash,
            )
        )
        self.risk.book.record_close(self.strategy.name, gross - commission, ts)
        return equity
