# Research log

Every idea tested and why it died. Two reasons to keep this: it stops you
re-testing a dead idea in six months having forgotten, and it is the **honest
trial count** that the deflated Sharpe ratio needs. Without that count, no result
here is interpretable.

**Running trial count: 134** (18 backtest configurations + 8 gauntlet variants,
plus the diagnostics below, which test the same hypotheses rather than new ones).

---

## 001 — MTF pullback (H4/H1 bias, M5/M15 entry) — **DEAD**

*Tested 2026-09-04. Exness cent account, real spreads, real data.*

### The idea
Higher timeframe sets direction, lower timeframe finds a pullback entry, momentum
trigger confirms. The standard retail top-down structure.

### How it died

**First pass — 18 configurations, all blocked.** Best was XAUUSD M15 at +0.138R,
Sharpe 1.55. The gauntlet rejected it: dies at 2x costs, deflated Sharpe 0.000
across 17 trials, PBO 40%, Monte Carlo 95th-percentile drawdown 44.7%.

**Implementation error found and fixed.** The stop was sized from the *execution*
timeframe, making an "M5 trade" target a five-minute move against 16 points of
friction — 51% of risk paid as spread. Sizing the stop from H4 instead took
friction to 4.5% and improved every configuration 3–500x. Worth keeping: this is
a real design principle, independent of whether this strategy works.

**Decomposition — the decisive test.** Each component tested alone, forward
returns in ATR units, non-overlapping windows:

| Component | EURUSD | GBPUSD | XAUUSD |
|---|---|---|---|
| Bias (H4+H1) | t = -0.04 | t = -0.64 | t = 2.95 but **no better than always-long** |
| Location | no signal | no signal | +0.42 ATR vs chasing, t = 2.61 |
| Trigger | no signal | no signal | **removed 2/3 of the edge** |

Gold's bias score was drift, not signal: "always long" scored +0.2950 ATR against
the strategy's +0.2922, over a window in which gold rose 86%.

**Replication test — the location effect does not hold.**

| Symbol | TF | Span | Differential | t |
|---|---|---|---|---|
| XAUUSD | M30 | 1,547d | **-0.077** | -0.74 |
| XAUUSD | H1 | 3,084d | **-0.047** | -0.90 |
| XAUUSD | H4 | 4,266d | **-0.038** | -1.00 |
| EURUSD | H1 | 2,927d | -0.016 | -0.29 |
| GBPUSD | H1 | 2,927d | -0.053 | -1.01 |

Negative at every timeframe and every symbol, none significant. By year: positive
in **3 of 9** years on H1, **1 of 5** on M30. A coin flip.

### Verdict
The M15 finding (+0.42 ATR, t=2.61) was noise selected out of many comparisons —
exactly what the PBO of 40% predicted. All three components fail. **Do not revisit
without a materially different entry definition or a different cost structure.**

### What was kept
- Stop distance belongs to the *structure* timeframe, never the entry timeframe.
- The momentum-confirmation trigger is a cost, not a filter.
- Test the differential, not the level, when a sample sits inside one regime.

---

## 002 — S1 time-series momentum, daily bars — **DEAD on all four**

*Tested 2026-09-05. IC Markets, real spreads, real swap (mode-correct), 2010+.*

First run of the trend baseline on real data. 60-day lookback, price vs EMA(60),
2.5x ATR(14) stop, symmetric long/short. Research mode: an account-level halt
re-bases the drawdown reference and counts a failed evaluation.

| Symbol | Trades | Win | Payoff | Expectancy | Sharpe | Cost drag | 2x costs | Evals failed |
|---|---|---|---|---|---|---|---|---|
| EURUSD | 286 | 23.8% | 3.03 | -0.014R | -0.04 | 14% | dead | 0 |
| XAUUSD | 247 | 24.3% | 3.66 | +0.047R | 0.16 | 31% | **-194%** | 0 |
| US30 | 195 | 15.9% | 0.96 | -0.384R | -0.81 | 168% | dead | **4** |
| US500 | 198 | 12.6% | 1.03 | -0.452R | -0.94 | 192% | dead | **5** |

Equal-weight portfolio of all four: Sharpe **-0.71**.

### Reading it
- EURUSD is a coin flip minus costs. The classic no-edge signature.
- Gold is weakly positive and dies when costs double. Not an edge; a rounding
  error in the cost model's favour.
- The indices are a bloodbath: 2010-2026 is a grinding bull market with V-shaped
  corrections, the worst possible regime for a symmetric trend rule. Every short
  gets killed at the V-bottom. Under the challenge profile it would have failed
  the evaluation four and five times.
- The published trend effect (Moskowitz et al.) is a *portfolio* result across
  58 instruments with sophisticated vol scaling, and is notably weaker after
  2010. A single-instrument daily rule on the most liquid markets on earth is
  not the same experiment.

### Two bugs found by this run, both fixed
1. An account-level HALT in a multi-year backtest had no human to restart it, so
   US30 showed 17 trades in 14 years with 2,525 signals rejected. Backtester now
   records `halted_at`, and `reset_on_halt=True` re-bases and counts failures.
2. `swap_mode` is not uniform: FX and gold in POINTS, US30 in margin currency,
   and index CFDs triple-charge on **Friday**, not Wednesday. Conversion now
   honours the broker's mode per symbol.

### Verdict
Two strategy families, 30 configurations, zero survivors. Simple technical rules
on these four instruments at retail costs have now been tested at M5, M15, M30
and D1 with two different signal families. Stop testing indicator rules here.

---

## 003 — Signal screen, 104 hypotheses — **5 survivors, all artifacts**

*Tested 2026-09-05. `research/screen.py`. Drift-removed forward returns in ATR
units, non-overlapping windows, Bonferroni at 104 trials (|t| > 3.49), and
positive in >= 70% of years.*

Eighteen structural hypotheses across four instruments, plus 24 hour-of-day
tests on EURUSD and gold. Everything with a reason to exist: calendar effects,
reversal and momentum at four horizons, range expansion and compression, the US
open both ways, the overnight index drift, the London open, the Asian range
break, gold versus the dollar, gold as a haven.

**Every structural hypothesis died.** The only survivors were five hour-of-day
effects, all in the 20:00-02:00 UTC block -- the rollover and the dead zone
after it. EURUSD hour_21 had t = 5.62 across 13 of 17 years. It looked real.

### Why it is not (`research/verify_hour_effect.py`)

| Check | EURUSD | XAUUSD |
|---|---|---|
| Spread at the signal hour (ticks) | **41-73 pts** vs 5 normally; the edge is ~4 pts | market **closed** 21:00-22:00; reopens at 10 pts |
| Bid-bar signature | hour_20 gap -0.049, hour_21 intrabar +0.063: a V around rollover summing to ~0 | thin hour_21 (n=1959 vs 2700): it is the reopening bar |
| Where in the hour (M5) | concentrated in the **:00 and :55 slots** at bar boundaries, ~0 mid-hour | **one slot, 21:55, carries +388 of +408** -- the reopening gap |

The spread blows out into rollover, bars close on the bid, the close dips and
"recovers" when the spread normalises. No mid-price moved. On gold it is the
gap across a closed hour. Neither is capturable: the spread at the exact moment
of the signal is eight to fifteen times the edge.

### Verdict
Three families, 134 trials, zero tradeable survivors. The funnel worked exactly
as designed: the screen surfaced candidates in a minute and the cost check
killed them in the next. Rollover hours are now excluded from the screen by
default so it cannot rediscover this.

### What a pro would conclude
Simple signals on OHLC bars of the four most liquid instruments on earth do not
survive costs, multiple testing and replication at retail. That is the
literature's answer as well. The published momentum and carry results are
**cross-sectional portfolios** -- rank many instruments, long the top, short the
bottom -- not single-instrument timing. This account offers 7,391 symbols and
four have been used. That is the one structural avenue left untested.
