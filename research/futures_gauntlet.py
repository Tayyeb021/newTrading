"""Apply the pre-registered thresholds of RESEARCH_LOG 007-009 to a futures book.

    python research/futures_gauntlet.py --entry 007
    python research/futures_gauntlet.py --entry 008
    python research/futures_gauntlet.py --entry 009            # carry alone
    python research/futures_gauntlet.py --entry 009 --with-trend  # carry beside the trend ensemble

Every threshold below was written into the log before the data was downloaded.
This script computes the numbers and prints PASS or FAIL against each; it does
not decide anything, it reports. The run is at the 2x cost stress the
thresholds specify, at an equity where one contract is always inside the risk
budget, so that what is measured is the signal and not the account.

Outputs the report and writes state/gauntlet_<entry>.json for the log.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # after the root: scripts/backtest.py must not shadow the package

from backtest.costs import CostModel  # noqa: E402
from backtest.metrics import _drawdown  # noqa: E402
from backtest.portfolio import (  # noqa: E402
    PortfolioBacktester, diversification_ratio, portfolio_report, sleeve_correlation,
)
from backtest_futures import load_expiries  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.contracts import FULL_UNIVERSE, data_root, tradeable  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from data.continuous import stitch  # noqa: E402
from ml.stats import deflated_sharpe, probability_of_backtest_overfitting, sharpe  # noqa: E402
from risk.build import build_engine  # noqa: E402
from strategies.carry import Carry  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402
from strategies.tsmom import TSMOM  # noqa: E402

TRIALS_SO_FAR = 186  # RESEARCH_LOG running total after 010 was declared


def load_universe(since: int, folder: Path, size_as: str):
    start, end = date(since, 1, 1), date.today()
    bars, specs, trade = {}, {}, {}
    for name in FULL_UNIVERSE:
        droot = data_root(name)
        exp = load_expiries(name, folder)
        if not exp:
            raise SystemExit(f"{name}: no expiry data under {folder}")
        cont, _ = stitch(droot, exp, start=start, end=end)
        bars[name] = cont
        troot = tradeable(name) if size_as == "micro" else droot
        trade[name] = troot
        specs[name] = troot.to_spec(name)
    return bars, specs, trade


def build_sleeves(names, kinds, lookbacks, continuous):
    sleeves = []
    if "trend" in kinds:
        prefix = "ctrend" if continuous else "trend"
        for lb in lookbacks:
            sleeves.append(Sleeve(f"{prefix}{lb}",
                                  (lambda s, lb=lb: TrendFollowing(lookback=lb, ema_period=lb, continuous=continuous)),
                                  tuple(names), timeframe="D1"))
    if "carry" in kinds:
        sleeves.append(Sleeve("carry", lambda s: Carry(), tuple(names), timeframe="D1"))
    if "tsmom" in kinds:
        for lb in lookbacks:
            sleeves.append(Sleeve(f"tsmom{lb}", (lambda s, lb=lb: TSMOM(lookback=lb)), tuple(names), timeframe="D1"))
    if "carry010" in kinds:
        sleeves.append(Sleeve("carry010", lambda s: Carry.published(), tuple(names), timeframe="D1"))
    return sleeves


def yearly(equity: pd.Series) -> pd.Series:
    """Net P&L per calendar year with at least 200 bars, in cash."""
    counts = equity.groupby(equity.index.year).size()
    ends = equity.groupby(equity.index.year).last()
    starts = equity.groupby(equity.index.year).first()
    pnl = (ends - starts)[counts >= 200]
    return pnl


def evaluate(result, specs, stress: float, trials: int) -> dict:
    eq = result.equity.dropna()
    rets = eq.pct_change().dropna().to_numpy()
    sr_daily = sharpe(rets)
    out = {
        "net_sharpe": sr_daily * np.sqrt(252),
        "max_drawdown": _drawdown(eq)[0],
        "final_equity": float(eq.iloc[-1]),
        "trades": len(result.trades),
        "rebalances": result.rebalances,
        "halted_at": str(result.halted_at.date()) if result.halted_at else None,
        "evaluations_failed": result.evaluations_failed,
        "cost_stress": stress,
    }
    gross = sum(t.gross_pnl for t in result.trades)
    friction = sum(t.costs for t in result.trades)
    out["gross_pnl"], out["friction"] = gross, friction
    out["friction_share"] = friction / abs(gross) if gross else float("inf")

    yr = yearly(eq)
    out["years"] = {int(y): float(v) for y, v in yr.items()}
    out["positive_years"] = int((yr > 0).sum())
    out["n_years"] = int(len(yr))

    by_sector: dict[str, float] = {}
    for t in result.trades:
        sector = FULL_UNIVERSE[t.symbol].bucket
        by_sector[sector] = by_sector.get(sector, 0.0) + t.net_pnl
    out["sectors"] = by_sector
    out["positive_sectors"] = sum(1 for v in by_sector.values() if v > 0)

    # last five years out of sample: nothing was fitted, so this is simply the tail
    tail = eq[eq.index >= eq.index[-1] - pd.Timedelta(days=5 * 365)]
    out["last5y_sharpe"] = sharpe(tail.pct_change().dropna().to_numpy()) * np.sqrt(252)

    # overfitting across the sleeves (the speeds), and the deflated Sharpe of the book
    sleeve_rets = pd.DataFrame(result.sleeve_equity).diff().dropna(how="all").fillna(0.0)
    sleeve_rets = sleeve_rets.loc[:, sleeve_rets.std() > 0]
    if sleeve_rets.shape[1] >= 2:
        pbo = probability_of_backtest_overfitting(sleeve_rets.to_numpy(), n_partitions=8)
        out["pbo"] = float(pbo.pbo)
    else:
        out["pbo"] = None
    trial_srs = [sharpe(sleeve_rets[c].to_numpy()) for c in sleeve_rets.columns] + [sr_daily]
    out["deflated_sharpe"] = float(deflated_sharpe(rets, n_trials=trials, trial_sharpes=np.array(trial_srs)))

    corr = sleeve_correlation(result)
    out["diversification_ratio"] = float(diversification_ratio(corr, result.weights)) if len(corr) >= 2 else 1.0
    out["sleeve_correlation"] = corr.round(2).to_dict() if len(corr) >= 2 else {}
    out["per_sleeve"] = {}
    for name in result.weights:
        ts = [t for t in result.trades if t.sleeve == name]
        out["per_sleeve"][name] = {"trades": len(ts), "net_pnl": float(sum(t.net_pnl for t in ts))}
    return out


def verdict_007(m: dict) -> list[tuple[str, bool, str]]:
    return [
        ("1. ensemble net Sharpe >= 0.40 at 2x costs", m["net_sharpe"] >= 0.40, f"{m['net_sharpe']:.2f}"),
        ("2. positive in >= 70% of years", m["positive_years"] >= 0.7 * m["n_years"],
         f"{m['positive_years']}/{m['n_years']}"),
        ("3. PBO across speeds < 0.50", (m["pbo"] is not None and m["pbo"] < 0.5), f"{m['pbo']}"),
        (f"4. deflated Sharpe > 0 given {TRIALS_SO_FAR} trials", m["deflated_sharpe"] > 0,
         f"{m['deflated_sharpe']:.3f} (P(true SR > expected max of noise))"),
        ("5. last five years net Sharpe > 0", m["last5y_sharpe"] > 0, f"{m['last5y_sharpe']:.2f}"),
        ("6. >= 5 of 7 sectors positive", m["positive_sectors"] >= 5, f"{m['positive_sectors']}/7"),
    ]


def verdict_008(m: dict, base: dict) -> list[tuple[str, bool, str]]:
    return [
        ("1. net Sharpe >= 007's - 0.05", m["net_sharpe"] >= base["net_sharpe"] - 0.05,
         f"{m['net_sharpe']:.2f} vs {base['net_sharpe']:.2f}"),
        ("2. friction share <= 007's", m["friction_share"] <= base["friction_share"],
         f"{m['friction_share']:.0%} vs {base['friction_share']:.0%}"),
        ("3. max drawdown <= 007's", m["max_drawdown"] <= base["max_drawdown"],
         f"{m['max_drawdown']:.1%} vs {base['max_drawdown']:.1%}"),
        ("4. positive years >= 007's - 1", m["positive_years"] >= base["positive_years"] - 1,
         f"{m['positive_years']} vs {base['positive_years']}"),
    ]


def verdict_009(m: dict, combined: dict | None, trend: dict | None) -> list[tuple[str, bool, str]]:
    rows = [
        ("1a. carry alone net Sharpe >= 0.30 at 2x costs", m["net_sharpe"] >= 0.30, f"{m['net_sharpe']:.2f}"),
        ("1b. positive in >= 60% of years", m["positive_years"] >= 0.6 * m["n_years"],
         f"{m['positive_years']}/{m['n_years']}"),
        ("1c. >= 5 of 7 sectors positive", m["positive_sectors"] >= 5, f"{m['positive_sectors']}/7"),
    ]
    if combined is not None:
        corr = combined["sleeve_correlation"]
        rho = None
        carry_name = next((k for k in corr if k.startswith("carry")), None)
        if carry_name is not None:
            others = [v for k, v in corr[carry_name].items() if k != carry_name]
            rho = max(others) if others else None
        rows.append(("2. carry vs trend weekly correlation < 0.5", rho is not None and rho < 0.5, f"{rho}"))
        best_alone = max(m["net_sharpe"], trend["net_sharpe"] if trend else -9)
        rows.append(("3. trend + carry net Sharpe > better of the two alone",
                     combined["net_sharpe"] > best_alone, f"{combined['net_sharpe']:.2f} vs {best_alone:.2f}"))
    return rows


def run_book(bars, specs, trade, sleeves, equity, profile_name, stress):
    profile = RiskProfile.load(profile_name)
    engine = build_engine(profile, equity, specs, sleeves)
    costs = CostModel.for_futures(trade).stressed(stress)
    names = list(FULL_UNIVERSE)
    return PortfolioBacktester(sleeves, specs, engine, costs, starting_equity=equity,
                               reset_on_halt=True).run({(s.name, n): bars[n] for s in sleeves for n in names})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", choices=["007", "008", "009", "010", "010c"], required=True,
                    help="010 = published monthly trend, 010c = carry in risk units")
    ap.add_argument("--with-trend", action="store_true", help="009/010c: also run carry beside the trend ensemble")
    ap.add_argument("--lookbacks", nargs="+", type=int, default=None,
                    help="trend speeds; default 20 60 120 for 007/008, 60 120 250 for 010")
    ap.add_argument("--equity", type=float, default=20_000_000.0)
    ap.add_argument("--size-as", choices=["micro", "full"], default="full")
    ap.add_argument("--stress", type=float, default=2.0)
    ap.add_argument("--since", type=int, default=2011)
    ap.add_argument("--profile", default="research")
    ap.add_argument("--data", default="data/futures")
    ap.add_argument("--trials", type=int, default=TRIALS_SO_FAR)
    args = ap.parse_args()

    bars, specs, trade = load_universe(args.since, Path(args.data), args.size_as)
    names = list(FULL_UNIVERSE)
    print(f"{len(names)} markets, {args.since}-{date.today().year}, equity {args.equity:,.0f}, "
          f"costs x{args.stress:g}, profile {args.profile}, sized as {args.size_as}")

    results: dict[str, dict] = {}
    state = Path("state"); state.mkdir(exist_ok=True)
    lookbacks = args.lookbacks or ([60, 120, 250] if args.entry.startswith("010") else [20, 60, 120])

    if args.entry in ("007", "008", "010"):
        kind = "tsmom" if args.entry == "010" else "trend"
        sleeves = build_sleeves(names, [kind], lookbacks, continuous=(args.entry == "008"))
        res = run_book(bars, specs, trade, sleeves, args.equity, args.profile, args.stress)
        print(portfolio_report(res))
        m = evaluate(res, specs, args.stress, args.trials)
        results["book"] = m
        if args.entry == "008":
            base_path = state / "gauntlet_007.json"
            if not base_path.exists():
                raise SystemExit("run --entry 007 first; 008 is judged against it")
            base = json.loads(base_path.read_text())["book"]
            rows = verdict_008(m, base)
        else:
            rows = verdict_007(m)  # 010 is held to the same six thresholds
    else:
        carry_kind = "carry010" if args.entry == "010c" else "carry"
        trend_kind = "tsmom" if args.entry == "010c" else "trend"
        trend_json = state / ("gauntlet_010.json" if args.entry == "010c" else "gauntlet_007.json")
        carry_sleeves = build_sleeves(names, [carry_kind], [], False)
        res = run_book(bars, specs, trade, carry_sleeves, args.equity, args.profile, args.stress)
        print(portfolio_report(res))
        m = evaluate(res, specs, args.stress, args.trials)
        results["carry"] = m
        combined = trend = None
        if args.with_trend:
            trend = json.loads(trend_json.read_text())["book"] if trend_json.exists() else None
            both = build_sleeves(names, [trend_kind, carry_kind], lookbacks, False)
            res2 = run_book(bars, specs, trade, both, args.equity, args.profile, args.stress)
            print(portfolio_report(res2))
            combined = evaluate(res2, specs, args.stress, args.trials)
            results["trend_plus_carry"] = combined
        rows = verdict_009(m, combined, trend)

    print(f"\nVERDICT, entry {args.entry}")
    print("=" * 70)
    for label, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL':<5} {label:<52} {detail}")
    passed = all(ok for _, ok, _ in rows)
    print(f"\n  {'ALL THRESHOLDS PASSED' if passed else 'FAILED - the family is reported dead in this form'}")
    m = results.get("book") or results.get("carry")
    print(f"\n  years: " + ", ".join(f"{y}:{v/1e6:+.2f}M" for y, v in m["years"].items()))
    print(f"  sectors: " + ", ".join(f"{k}:{v/1e6:+.2f}M" for k, v in sorted(m["sectors"].items())))

    results["verdict"] = [{"test": l, "pass": bool(ok), "detail": d} for l, ok, d in rows]
    results["passed"] = passed
    results["args"] = vars(args)
    (state / f"gauntlet_{args.entry}{'_with_trend' if args.with_trend else ''}.json").write_text(
        json.dumps(results, indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
