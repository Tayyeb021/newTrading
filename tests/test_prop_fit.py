"""The evaluation rules, as modelled. Getting these wrong flips the conclusion.

The whole prop-route answer rests on one rule: the maximum loss limit TRAILS.
Modelled as a static floor at 4% below the starting balance it is a mild
constraint; modelled correctly, as a floor that ratchets up with every new
end-of-day high and never comes back down, it is the thing that takes the
account. A silent switch between the two would change the answer from "no" to
"yes" without any test failing, so it is tested first and explicitly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "prop_fit", Path(__file__).resolve().parent.parent / "research" / "prop_fit.py")
pf = importlib.util.module_from_spec(_spec)
sys.modules["prop_fit"] = pf          # dataclasses need the module registered
_spec.loader.exec_module(pf)

R = pf.TOPSTEP_50K


def _daily(pct_moves: list[float]) -> np.ndarray:
    return np.array(pct_moves, dtype=float)


def test_a_steady_climb_to_the_target_passes():
    rets = _daily([0.01] * 10)          # +1% a day, compounding past +6% on day 6
    outcome, day = pf.walk(rets, R, k=1.0)
    assert outcome == "pass"
    assert day == 6


def test_the_max_loss_floor_trails_and_never_comes_back_down():
    """Up 3%, then down 4.1% FROM THERE. The account is still above where it
    started, and a static floor would not have been touched - but Topstep's
    floor moved up with the high-water mark, so this is a breach.

    If this test ever passes with outcome 'pass', the model has quietly become
    a static-drawdown firm and every number in the analysis is wrong.
    """
    rets = _daily([0.03, -0.042])
    outcome, day = pf.walk(rets, R, k=1.0, use_daily_limit=False)
    assert outcome == "max_loss", "the floor must trail the high-water mark"
    assert day == 2

    static = pf.Rules(**{**R.__dict__, "trailing": False})
    assert pf.walk(rets, static, k=1.0, use_daily_limit=False)[0] != "max_loss", \
        "the same path under a static floor is not a breach - that is the difference"


def test_the_daily_limit_fires_on_one_bad_day():
    rets = _daily([-0.021])             # -$1,050 against a $1,000 daily limit
    assert pf.walk(rets, R, k=1.0)[0] == "daily_limit"
    assert pf.walk(rets, R, k=1.0, use_daily_limit=False)[0] != "daily_limit", \
        "the Combine's daily limit is optional, so it must be switchable"


def test_the_consistency_rule_delays_a_pass_it_does_not_grant_one():
    """One day making more than half the profit is not a pass yet. Trading on
    dilutes it - so the path passes later, not never."""
    rets = _daily([0.062] + [0.002] * 40)   # one +$3,100 day, then a slow grind
    outcome, day = pf.walk(rets, R, k=1.0, use_daily_limit=False)
    assert outcome == "pass"
    assert day > 1, "a single day worth the whole target cannot pass on its own"


def test_a_flat_book_times_out_rather_than_passing():
    rets = np.zeros(500)
    outcome, day = pf.walk(rets, R, k=1.0)
    assert outcome == "timeout"
    assert day == R.max_days


def test_scaling_risk_scales_the_path_not_the_rules():
    """The only free parameter is position size. Quarter risk must quarter the
    move while the $3,000 target and $2,000 floor stay put."""
    rets = _daily([-0.042])
    assert pf.walk(rets, R, k=1.0, use_daily_limit=False)[0] == "max_loss"
    assert pf.walk(rets, R, k=0.25, use_daily_limit=False)[0] == "timeout"


def test_surviving_a_year_is_not_the_same_as_being_payable():
    """Risk low enough to never breach is risk too low to ever make a $150 day.
    Reporting survival alone would make the route look open when it is not."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.05 / 252, 0.20 / np.sqrt(252), 3000)

    tiny = pf.survive(rets, R, k=0.01, paths=300)
    assert tiny["p_alive_1y"] > 0.95, "at 1% risk almost nothing can breach"
    assert tiny["p_payable_1y"] < 0.05, "and almost nothing can clear $150 either"

    real = pf.survive(rets, R, k=1.0, paths=300)
    assert real["p_payable_1y"] <= real["p_alive_1y"]


def test_the_funded_floor_locks_at_the_starting_balance():
    """Once the account is up by the limit the floor stops rising, so a funded
    account that gains a lot is not held to a 4% band around its peak."""
    rng = np.random.default_rng(1)
    strong = rng.normal(0.40 / 252, 0.10 / np.sqrt(252), 3000)   # Sharpe 4, unmissable
    out = pf.survive(strong, R, k=1.0, paths=300)
    assert out["p_alive_1y"] > 0.5, "a locked floor must let a strong book run"


def test_historical_starts_and_bootstrap_agree_on_the_same_series():
    """Two methods, no shared assumption beyond the data. A large gap between
    them would mean the block bootstrap had destroyed the path structure."""
    rng = np.random.default_rng(2)
    rets = rng.normal(0.07 / 252, 0.20 / np.sqrt(252), 2500)
    hist = pf.historical_starts(rets, R, k=0.25)
    boot = pf.bootstrap(rets, R, k=0.25, paths=2000)
    assert abs(hist["p_pass"] - boot["p_pass"]) < 0.12
