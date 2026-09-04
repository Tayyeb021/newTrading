"""
Monte Carlo simulation of a prop firm evaluation.

The question this answers is not "is my strategy good?" but the narrower and more
useful one: *given* an edge with a known trade distribution, what risk per trade
maximises the probability of passing, and is the challenge fee positive expected
value at that risk level?

The industry-wide finding that motivates this: roughly 71% of first-phase failures
are daily drawdown breaches, not an inability to reach the profit target. Failure
is dominated by path risk, and path risk is something you can compute.

Usage:
    python prop_challenge_sim.py                 # sweep risk for the default firms
    python prop_challenge_sim.py --paths 100000  # tighter confidence intervals
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np

RNG = np.random.default_rng(20260904)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FirmRules:
    """One phase of an evaluation. Verify every field against the actual rulebook."""

    name: str
    profit_target: float  # fraction of starting balance, e.g. 0.08
    daily_loss_limit: float  # fraction of day-start equity
    max_loss_limit: float  # fraction of starting balance (or high-water if trailing)
    trailing_max_loss: bool  # True = floor rises with equity high-water mark
    min_trading_days: int
    max_days: int | None  # None = no time limit, now the industry norm
    consistency: float | None  # max share of total profit from a single day
    fee: float
    account_size: float

    @property
    def target_cash(self) -> float:
        return self.account_size * self.profit_target


@dataclass(frozen=True)
class Edge:
    """Trade-level distribution. Take these from your backtest, not from hope."""

    name: str
    win_rate: float
    payoff: float  # average win / average loss, in R
    trades_per_day: float
    # Fraction of R lost to slippage on the losing side. Losses run slightly
    # worse than the stop; wins do not run better than the target.
    loss_slippage: float = 0.08

    def expectancy_r(self) -> float:
        loss = 1.0 + self.loss_slippage
        return self.win_rate * self.payoff - (1 - self.win_rate) * loss


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def simulate(
    firm: FirmRules,
    edge: Edge,
    risk_per_trade: float,
    n_paths: int = 20_000,
    hard_day_cap: int = 400,
) -> dict[str, float]:
    """Simulate n_paths evaluations. Returns pass rate and diagnostics.

    Equity is tracked in cash. A path ends when it passes, breaches a limit, or
    runs out of days. Daily loss is measured against day-start equity and checked
    after every trade, which is how a broker-side rule actually bites.
    """
    day_cap = min(firm.max_days or hard_day_cap, hard_day_cap)

    start = firm.account_size
    risk_cash = start * risk_per_trade
    loss_amount = risk_cash * (1.0 + edge.loss_slippage)
    win_amount = risk_cash * edge.payoff

    equity = np.full(n_paths, start)
    high_water = np.full(n_paths, start)
    best_day_profit = np.zeros(n_paths)
    days_traded = np.zeros(n_paths, dtype=np.int32)

    alive = np.ones(n_paths, dtype=bool)
    passed = np.zeros(n_paths, dtype=bool)
    failed_daily = np.zeros(n_paths, dtype=bool)
    failed_max = np.zeros(n_paths, dtype=bool)
    days_to_pass = np.zeros(n_paths, dtype=np.int32)

    static_floor = start * (1.0 - firm.max_loss_limit)

    for day in range(1, day_cap + 1):
        if not alive.any():
            break

        day_start_equity = equity.copy()
        daily_floor = day_start_equity * (1.0 - firm.daily_loss_limit)

        # Poisson trade count keeps the day-to-day variance honest; a fixed
        # trades-per-day understates the tail where several losers land together.
        n_trades = RNG.poisson(edge.trades_per_day, n_paths)
        max_trades = int(n_trades.max()) if n_trades.size else 0

        traded_today = alive & (n_trades > 0)
        days_traded += traded_today

        for t in range(max_trades):
            active = alive & (n_trades > t)
            if not active.any():
                continue

            wins = RNG.random(n_paths) < edge.win_rate
            pnl = np.where(wins, win_amount, -loss_amount)
            equity = np.where(active, equity + pnl, equity)

            # Daily breach — checked intra-day, as the firm does.
            breach_daily = active & (equity <= daily_floor)
            failed_daily |= breach_daily
            alive &= ~breach_daily

            # Max loss breach.
            floor = high_water * (1.0 - firm.max_loss_limit) if firm.trailing_max_loss else static_floor
            breach_max = alive & active & (equity <= floor)
            failed_max |= breach_max
            alive &= ~breach_max

            high_water = np.maximum(high_water, equity)

        # Track the best single day for the consistency rule.
        day_profit = np.maximum(equity - day_start_equity, 0.0)
        best_day_profit = np.where(alive, np.maximum(best_day_profit, day_profit), best_day_profit)

        # Pass check, end of day.
        total_profit = equity - start
        hit_target = alive & (total_profit >= firm.target_cash)
        met_days = days_traded >= firm.min_trading_days

        if firm.consistency is not None:
            # Best day must not exceed the allowed share of total profit. A path
            # that fails this keeps trading; more profit dilutes the big day.
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(total_profit > 0, best_day_profit / total_profit, 1.0)
            consistent = share <= firm.consistency
        else:
            consistent = np.ones(n_paths, dtype=bool)

        now_passing = hit_target & met_days & consistent
        days_to_pass = np.where(now_passing & ~passed, day, days_to_pass)
        passed |= now_passing
        alive &= ~now_passing

    n = float(n_paths)
    passers = days_to_pass[passed]
    return {
        "risk_per_trade": risk_per_trade,
        "p_pass": passed.sum() / n,
        "p_fail_daily": failed_daily.sum() / n,
        "p_fail_max": failed_max.sum() / n,
        "p_timeout": alive.sum() / n,
        "median_days_to_pass": float(np.median(passers)) if passers.size else float("nan"),
    }


def sweep(
    firm: FirmRules,
    edge: Edge,
    risks: np.ndarray,
    n_paths: int,
) -> list[dict[str, float]]:
    return [simulate(firm, edge, r, n_paths) for r in risks]


# --------------------------------------------------------------------------- #
# Firms and edges
# --------------------------------------------------------------------------- #

# Representative rule shapes, not endorsements. Static and trailing drawdown are
# genuinely different games and are both common, so both are modelled.
STATIC_TWO_PHASE = FirmRules(
    name="Static 10% max / 5% daily, 8% target",
    profit_target=0.08,
    daily_loss_limit=0.05,
    max_loss_limit=0.10,
    trailing_max_loss=False,
    min_trading_days=3,
    max_days=None,
    consistency=None,
    fee=500.0,
    account_size=100_000.0,
)

STATIC_TIGHT = FirmRules(
    name="Static 5% max / 4% daily, 6% target",
    profit_target=0.06,
    daily_loss_limit=0.04,
    max_loss_limit=0.05,
    trailing_max_loss=False,
    min_trading_days=3,
    max_days=None,
    consistency=None,
    fee=350.0,
    account_size=100_000.0,
)

TRAILING_CONSISTENCY = FirmRules(
    name="Trailing 6% max / 4% daily, 8% target, 30% consistency",
    profit_target=0.08,
    daily_loss_limit=0.04,
    max_loss_limit=0.06,
    trailing_max_loss=True,
    min_trading_days=5,
    max_days=None,
    consistency=0.30,
    fee=450.0,
    account_size=100_000.0,
)

FIRMS = [STATIC_TWO_PHASE, STATIC_TIGHT, TRAILING_CONSISTENCY]

# Two edge shapes with almost the same expectancy but very different paths.
# This is the comparison that matters for an evaluation.
TREND = Edge(
    name="Trend following (lumpy)",
    win_rate=0.38,
    payoff=2.20,
    trades_per_day=0.35,
)

GRINDER = Edge(
    name="Higher win rate (smooth)",
    win_rate=0.58,
    payoff=1.00,
    trades_per_day=1.20,
)

NO_EDGE = Edge(
    name="No edge (control)",
    win_rate=0.50,
    payoff=1.00,
    trades_per_day=1.00,
)

EDGES = [TREND, GRINDER, NO_EDGE]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report(n_paths: int) -> None:
    risks = np.array([0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020])

    for edge in EDGES:
        print()
        print("=" * 96)
        print(f"{edge.name}   win {edge.win_rate:.0%}  payoff {edge.payoff:.2f}R  "
              f"{edge.trades_per_day:.2f} trades/day  expectancy {edge.expectancy_r():+.3f}R")
        print("=" * 96)

        for firm in FIRMS:
            print(f"\n  {firm.name}")
            print(f"  {'risk/trade':>11} {'P(pass)':>9} {'daily DD':>9} {'max DD':>8} "
                  f"{'stalled':>8} {'med. days':>10}")
            print("  " + "-" * 60)
            for row in sweep(firm, edge, risks, n_paths):
                days = row["median_days_to_pass"]
                days_s = f"{days:.0f}" if days == days else "--"
                print(f"  {row['risk_per_trade']:>10.2%} {row['p_pass']:>9.1%} "
                      f"{row['p_fail_daily']:>9.1%} {row['p_fail_max']:>8.1%} "
                      f"{row['p_timeout']:>8.1%} {days_s:>10}")


def breakeven(firm: FirmRules, p_pass_phase: float, phases: int = 2) -> None:
    """What a challenge attempt has to be worth to justify the fee."""
    p_total = p_pass_phase**phases
    if p_total <= 0:
        print("  P(pass) is zero — no fee is justified.")
        return
    required = firm.fee / p_total
    print(f"  P(pass one phase) {p_pass_phase:.1%} -> P(funded) {p_total:.1%} over {phases} phases")
    print(f"  Fee {firm.fee:,.0f} is breakeven only if a funded account is worth "
          f">= {required:,.0f} in expected lifetime payout.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=20_000, help="Monte Carlo paths per cell")
    args = ap.parse_args()

    report(args.paths)

    print()
    print("=" * 96)
    print("Fee arithmetic")
    print("=" * 96)
    for p in (0.30, 0.50, 0.70):
        print()
        breakeven(STATIC_TWO_PHASE, p)


if __name__ == "__main__":
    main()
