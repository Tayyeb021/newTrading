"""Can the book that survived the research actually pass a prop evaluation?

    python research/prop_fit.py

**Not a research entry and not a trial.** Nothing is being selected on data:
the book is already fixed, the rulebook is Topstep's published one, and the
single free parameter - how much of the book's natural risk to take - is swept
rather than chosen. The running total stays at 213.

The question is narrower than "is the strategy good", and it is the one that
decides whether the prop route is open at all: *given this book's actual daily
path, what is the probability of reaching the profit target before the trailing
maximum loss limit takes the account away?*

Two methods, deliberately:

- **Historical starts.** Every trading day in the sample is treated as a
  possible evaluation start and the real path is walked forward until it passes
  or breaches. No distributional assumption at all - this is what would have
  happened.
- **Block bootstrap.** Twenty-day blocks, so trends and the serial correlation
  of drawdowns survive resampling. A trend book's losses arrive in runs, and an
  IID bootstrap would quietly delete the exact feature that breaks evaluations.

Both understate the difficulty in the same direction, which is worth saying
plainly: the maximum loss limit is monitored in real time against intraday
equity, and this simulation only sees daily closes. Every intraday excursion is
invisible here, so the real pass rate is lower than whatever prints below.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RNG = np.random.default_rng(20260908)


@dataclass(frozen=True)
class Rules:
    """One evaluation. Verified against the published rulebook on 2026-09-08."""

    name: str
    account: float
    target: float          # cash profit required
    mll: float             # cash maximum loss limit
    trailing: bool         # the floor rises with the end-of-day high-water mark
    daily_limit: float | None
    consistency: float | None   # best single day, as a share of the profit TARGET
    min_days: int
    max_days: int
    winning_day: float = 150.0   # a day must clear this to count toward a payout
    min_winning_days: int = 5

    @property
    def target_pct(self) -> float:
        return self.target / self.account

    @property
    def mll_pct(self) -> float:
        return self.mll / self.account


# Topstep's $50K Trading Combine. The daily loss limit is optional in the
# Combine, so it is modelled both ways further down.
TOPSTEP_50K = Rules("Topstep 50K Combine", 50_000, 3_000, 2_000, True, 1_000, 0.50, 2, 400)


def load_book(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    eq = pd.Series(df["equity"].to_numpy(), index=pd.DatetimeIndex(df["ts"]))
    return eq.pct_change().dropna()


def walk(rets: np.ndarray, rules: Rules, k: float, use_daily_limit: bool = True) -> tuple[str, int]:
    """One evaluation over one path of daily returns scaled by k.

    Returns the outcome and the number of days it took. Equity compounds; the
    trailing floor moves only on end-of-day balances and never moves down,
    which is how Topstep describes it.
    """
    equity = rules.account
    high_water = equity
    best_day = 0.0
    days_traded = 0

    for day, r in enumerate(rets[: rules.max_days], start=1):
        start_of_day = equity
        equity *= 1.0 + k * r
        days_traded += 1

        if use_daily_limit and rules.daily_limit is not None:
            if start_of_day - equity >= rules.daily_limit:
                return "daily_limit", day

        floor = (high_water - rules.mll) if rules.trailing else (rules.account - rules.mll)
        if equity <= floor:
            return "max_loss", day

        high_water = max(high_water, equity)
        best_day = max(best_day, equity - start_of_day)

        profit = equity - rules.account
        if profit >= rules.target and days_traded >= rules.min_days:
            # Two readings of the consistency rule are in circulation: best day
            # under half the profit TARGET (a permanent cap), and best day under
            # half of total PROFIT (dilutable by trading on). The second is the
            # softer one and the one this repo's older simulator already used,
            # so it is the one modelled - being wrong in the direction that
            # flatters the route is the mistake that would matter here.
            if rules.consistency is not None and best_day > rules.consistency * profit:
                continue      # keep trading: more profit dilutes the outsized day
            return "pass", day

    return "timeout", min(len(rets), rules.max_days)


def historical_starts(rets: np.ndarray, rules: Rules, k: float, **kw) -> dict:
    """Every day in the sample as a possible start. No assumptions, real paths."""
    outcomes, days = [], []
    for i in range(len(rets) - rules.min_days):
        o, d = walk(rets[i:], rules, k, **kw)
        outcomes.append(o)
        days.append(d)
    return _tally(outcomes, days, k, len(outcomes))


def bootstrap(rets: np.ndarray, rules: Rules, k: float, paths: int, block: int = 20, **kw) -> dict:
    """Block bootstrap. Twenty-day blocks keep drawdowns arriving in runs."""
    n = len(rets)
    need = rules.max_days
    outcomes, days = [], []
    for _ in range(paths):
        starts = RNG.integers(0, n - block, size=need // block + 1)
        path = np.concatenate([rets[s: s + block] for s in starts])
        o, d = walk(path, rules, k, **kw)
        outcomes.append(o)
        days.append(d)
    return _tally(outcomes, days, k, paths)


def _tally(outcomes: list[str], days: list[int], k: float, n: int) -> dict:
    passed = [d for o, d in zip(outcomes, days) if o == "pass"]
    return {
        "risk_multiple": k,
        "p_pass": outcomes.count("pass") / n,
        "p_fail_max_loss": outcomes.count("max_loss") / n,
        "p_fail_daily": outcomes.count("daily_limit") / n,
        "p_timeout": outcomes.count("timeout") / n,
        "median_days_to_pass": float(np.median(passed)) if passed else float("nan"),
    }


def survive(rets: np.ndarray, rules: Rules, k: float, paths: int, horizon: int = 252,
            block: int = 20) -> dict:
    """Life AFTER funding, which is where the prop route is really decided.

    A funded Topstep account keeps its trailing maximum loss limit for good. The
    floor rises with the high-water mark until the account is up by the limit,
    then locks at the starting balance - so from that point the rule is simply
    "never be down on the account, ever". A book whose own history gave back
    41% meets that rule exactly once.
    """
    n = len(rets)
    alive_at_horizon, payable, lifetimes = 0, 0, []
    for _ in range(paths):
        starts = RNG.integers(0, n - block, size=horizon // block + 1)
        path = np.concatenate([rets[s: s + block] for s in starts])[:horizon]
        equity, high_water, locked = rules.account, rules.account, False
        died, winning_days = None, 0
        for day, r in enumerate(path, start=1):
            start_of_day = equity
            equity *= 1.0 + k * r
            floor = rules.account if locked else high_water - rules.mll
            if equity <= floor:
                died = day
                break
            if equity - start_of_day >= rules.winning_day:
                winning_days += 1
            high_water = max(high_water, equity)
            if high_water >= rules.account + rules.mll:
                locked = True
        if died is None:
            alive_at_horizon += 1
            # Surviving is not the same as being paid: the standard path wants
            # five days of at least $150 before a payout can be requested, and
            # risk low enough to survive is risk too low to ever have one.
            if winning_days >= rules.min_winning_days:
                payable += 1
        else:
            lifetimes.append(died)
    return {
        "risk_multiple": k,
        "p_alive_1y": alive_at_horizon / paths,
        "p_payable_1y": payable / paths,
        "median_days_alive": float(np.median(lifetimes)) if lifetimes else float("nan"),
    }


def synthetic(sharpe: float, vol: float, n: int) -> np.ndarray:
    """A Gaussian strategy with a known Sharpe, for the 'what would it take' table."""
    daily_vol = vol / np.sqrt(252)
    return RNG.normal(sharpe * vol / 252, daily_vol, n)


def _row(r: dict) -> str:
    days = f"{r['median_days_to_pass']:.0f}" if np.isfinite(r["median_days_to_pass"]) else "-"
    return (f"{r['risk_multiple']:>8.2f}{r['p_pass']:>9.1%}{r['p_fail_max_loss']:>11.1%}"
            f"{r['p_fail_daily']:>9.1%}{r['p_timeout']:>10.1%}{days:>9}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="data/book_equity.parquet",
                    help="daily equity curve, from futures_gauntlet.py --dump-returns")
    ap.add_argument("--paths", type=int, default=10_000)
    args = ap.parse_args()

    book = Path(args.book)
    if not book.exists():
        raise SystemExit(
            f"no book path at {book}. Build it with:\n"
            f"  python research/futures_gauntlet.py --entry 010c --with-trend "
            f"--dump-returns {book}")

    rets = load_book(book).to_numpy()
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252))
    ann_ret = float(np.mean(rets) * 252)
    sharpe = ann_ret / ann_vol if ann_vol else 0.0
    eq = np.cumprod(1 + rets)
    max_dd = float(np.max(1 - eq / np.maximum.accumulate(eq)))

    r = TOPSTEP_50K
    print(f"\nTHE BOOK, as measured: {len(rets):,} days")
    print("=" * 78)
    print(f"  annual return {ann_ret:>7.2%}   annual vol {ann_vol:>7.2%}   "
          f"Sharpe {sharpe:>5.2f}   max drawdown {max_dd:>6.1%}")
    print(f"\nTHE RULEBOOK: {r.name}")
    print(f"  profit target {r.target:>8,.0f} ({r.target_pct:.1%})   "
          f"max loss {r.mll:>7,.0f} ({r.mll_pct:.1%}, {'trailing' if r.trailing else 'static'})   "
          f"daily {r.daily_limit:,.0f}")
    print(f"  best day may not exceed {r.consistency:.0%} of the target "
          f"({r.consistency * r.target:,.0f})")
    print(f"\n  The book must gain {r.target_pct:.1%} without ever giving back "
          f"{r.mll_pct:.1%} from a high-water mark.")
    print(f"  Its own history gave back {max_dd:.1%}, which is "
          f"{max_dd / r.mll_pct:.0f}x that limit.")

    # The single free parameter: what fraction of the book's own risk to run.
    risks = np.array([1.0, 0.5, 0.25, 0.10, 0.05, 0.025, 0.01])
    print(f"\nHISTORICAL STARTS - every one of {len(rets) - r.min_days:,} days in the sample "
          f"as an evaluation start")
    print("=" * 78)
    print(f"{'risk x':>8}{'pass':>9}{'max loss':>11}{'daily':>9}{'timeout':>10}{'days':>9}")
    hist = [historical_starts(rets, r, k) for k in risks]
    for row in hist:
        print(_row(row))

    print(f"\nBLOCK BOOTSTRAP - {args.paths:,} paths, 20-day blocks")
    print("=" * 78)
    print(f"{'risk x':>8}{'pass':>9}{'max loss':>11}{'daily':>9}{'timeout':>10}{'days':>9}")
    boot = [bootstrap(rets, r, k, args.paths) for k in risks]
    for row in boot:
        print(_row(row))

    best = max(boot, key=lambda x: x["p_pass"])
    print(f"\n  Best case across every risk level: {best['p_pass']:.1%} at {best['risk_multiple']:.2f}x risk.")
    print(f"  A coin flip on a driftless account would be "
          f"{r.mll_pct / (r.target_pct + r.mll_pct):.1%}.")

    print(f"\nLIFE AFTER FUNDING - the trailing limit does not go away when you pass")
    print("=" * 78)
    print("  The floor rises to the starting balance and locks there, so a funded account")
    print("  must never be down on the day it started. Simulated forward one year:")
    print(f"{'risk x':>8}{'alive at 1y':>14}{'and payable':>14}{'median days':>14}")
    surv = [survive(rets, r, k, max(2000, args.paths // 5)) for k in risks]
    for row in surv:
        alive_days = (f"{row['median_days_alive']:.0f}"
                      if np.isfinite(row["median_days_alive"]) else "-")
        print(f"{row['risk_multiple']:>8.2f}{row['p_alive_1y']:>14.1%}"
              f"{row['p_payable_1y']:>14.1%}{alive_days:>14}")
    print(f"  'payable' also needs {r.min_winning_days} days of "
          f"${r.winning_day:,.0f}+, which is what the low-risk rows cannot produce.")

    print(f"\nWHAT WOULD IT TAKE - the same rules, a strategy with a different Sharpe")
    print("=" * 78)
    print("  Sharpe 0.0 is the honest null: the same volatility, no edge at all. If the")
    print("  book cannot beat it, the pass rate is the barrier geometry, not the strategy.")
    print(f"{'Sharpe':>8}{'risk x':>9}{'pass':>9}{'max loss':>11}{'timeout':>10}{'days':>9}")
    for s in (0.0, 0.3, 0.5, 1.0, 2.0, 3.0):
        # A long synthetic sample and the full path count: at 4,000 days and
        # 2,000 paths this table came out non-monotonic in Sharpe, which was
        # sampling noise being read as a finding.
        syn = synthetic(s, ann_vol, 40_000)
        rows = [bootstrap(syn, r, k, args.paths) for k in risks]
        b = max(rows, key=lambda x: x["p_pass"])
        days = f"{b['median_days_to_pass']:.0f}" if np.isfinite(b["median_days_to_pass"]) else "-"
        print(f"{s:>8.1f}{b['risk_multiple']:>9.2f}{b['p_pass']:>9.1%}"
              f"{b['p_fail_max_loss']:>11.1%}{b['p_timeout']:>10.1%}{days:>9}")

    print(f"\nVERDICT")
    print("=" * 78)
    # The two stages must be judged at the SAME risk level. Reading the best
    # pass rate off one row and the best survival off another describes a
    # trader who does not exist.
    surv_by_risk = {row["risk_multiple"]: row for row in surv}
    end_to_end = [(b["risk_multiple"], b["p_pass"], surv_by_risk[b["risk_multiple"]]["p_payable_1y"])
                  for b in boot]
    print("  At one risk level, end to end - pass, then survive a year and be payable:")
    print(f"{'risk x':>8}{'pass':>9}{'x payable':>12}{'= end to end':>15}")
    for kk, p_pass, p_pay in end_to_end:
        print(f"{kk:>8.2f}{p_pass:>9.1%}{p_pay:>12.1%}{p_pass * p_pay:>15.1%}")
    k_best, p_pass_best, p_pay_best = max(end_to_end, key=lambda x: x[1] * x[2])
    joint = p_pass_best * p_pay_best
    print(f"\n  Best end to end: {joint:.1%} at {k_best:.2f}x risk "
          f"({p_pass_best:.1%} pass, then {p_pay_best:.1%} still funded and payable a year on).")
    if joint < 0.10:
        print(f"  A lottery, not a plan. The pincer is that the risk which reaches a "
              f"{r.target_pct:.0%} target")
        print(f"  in reasonable time is the risk that gives back {r.mll_pct:.0%} on the way, and a")
        print(f"  Sharpe of {sharpe:.2f} is far too small to separate the two. Nothing about the")
        print(f"  book is broken - evaluations select for a different animal entirely.")
    else:
        print(f"  Worth a closer look at {best['risk_multiple']:.2f}x risk.")
    print()

    (ROOT / "state").mkdir(exist_ok=True)
    (ROOT / "state" / "prop_fit.json").write_text(json.dumps({
        "book": {"annual_return": ann_ret, "annual_vol": ann_vol, "sharpe": sharpe,
                 "max_drawdown": max_dd, "days": len(rets)},
        "rules": r.__dict__, "historical": hist, "bootstrap": boot, "survival": surv,
        "best_p_pass": best["p_pass"]}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
