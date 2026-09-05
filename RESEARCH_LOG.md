# Research log

Every idea tested and why it died. Two reasons to keep this: it stops you
re-testing a dead idea in six months having forgotten, and it is the **honest
trial count** that the deflated Sharpe ratio needs. Without that count, no result
here is interpretable.

**Running trial count: 170** (18 backtest configurations + 8 gauntlet variants,
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

---

## 004 — Portfolio layer built; two-momentum-sleeve book — **works, and both sleeves dead**

*2026-09-05. `core/sleeve.py`, `backtest/portfolio.py`, `SleeveBudget` limit,
sleeve support in the runner. 168 tests.*

The demo book was deliberately a trap: `trend60` and `trend250` on EURUSD,
XAUUSD and US500 since 2015. Two lookbacks look like two strategies.

| sleeve | trades | net | expectancy | win |
|---|---|---|---|---|
| trend60 | 313 | -20,324 | -0.155R | 22% |
| trend250 | 170 | -11,356 | -0.122R | 24% |

Weekly sleeve correlation **0.47**; diversification ratio **1.16x** against a
theoretical 1.41x for two independent sleeves. Read: mostly one bet. Both
sleeves individually dead, consistent with entry 002.

Found by running it: the profile's concurrent-position cap of 3 starved a
2-sleeve x 3-symbol book (4,603 rejections). With the cap at 6, the allocator
became the binding constraint (2,763 `sleeve_budget` rejections) - which is the
intended behaviour.

**Not a trial** for DSR purposes: no new hypothesis was tested, only the
machinery. Trial count stays at 134.

---

## 005 — Cross-sectional FX momentum and carry, G8 universe — **DEAD**

*2026-09-05. `research/cross_sectional.py`. 28 G8 crosses downloaded from IC
Markets; 21 with history back to 2010 (the seven AUD/CAD crosses start after
2014 on this broker). Monthly rebalance, long top quintile / short bottom,
measured spreads on turnover. Run at two windows. 12 trials counted.*

This was the last family with real published evidence that had not been
tested, and the first test in the project built the way that evidence was
generated - as a ranked portfolio rather than single-instrument timing.

| construction | since 2010 | since 2014 |
|---|---|---|
| xs_momentum_1m | -0.3%/yr, t -0.10 | -1.9%, t -0.60 |
| xs_momentum_3m | -3.4%, t -1.05 | -3.3%, t -0.98 |
| xs_momentum_6m | -4.4%, t -1.45 | **-5.3%, t -1.78** |
| xs_momentum_12m | -2.9%, t -0.94 | -2.5%, t -0.76 |
| tsmom_12m_volscaled | -0.4%, t -0.35 | -0.2%, t -0.18 |
| carry_2023+ (approx) | -2.8%, t -0.77 | same |

Bar: t > 2.64. Nothing approaches it in either direction. The closest thing to
a signal is *negative* six-month momentum - developed-market FX has been mildly
mean-reverting since 2014 - and it is not significant either.

### Reading it
This is the literature's answer, not a surprise. Menkhoff et al. found FX
momentum concentrated in emerging and minor currencies with wide spreads, and
weak in G10. Moskowitz's TSMOM result is a 58-instrument, multi-asset-class
portfolio; its FX sleeve alone was never the strong part. A G8-only universe
is where the effect is thinnest, and this broker's tradeable universe is G8.

Carry is the one construction not properly tested: only current swap rates
exist, so the book is ranked on 2026 rates and held over 2023-2026. That is an
approximation and is labelled one. A real carry test needs fifteen years of
rate history the broker does not provide.

### Verdict
Four families, 146 trials, zero survivors. Every family with published
evidence has now been tested in the form the evidence was published in, on
this broker's tradeable universe, and none of it is there at retail costs.

### Data quality, for the record
- Three USD majors failed on first download with a transient terminal error
  and succeeded on retry. A universe missing GBPUSD, USDCHF and USDJPY is not
  a G8 universe; check the pair list before believing a cross-section.
- EURNZD carried exactly one corrupt bar (1999-08-18, low = 0.0). The store's
  validator refused the whole write rather than absorb it. Dropped and rewritten.

---

## 006 — CFTC positioning (COT) as a signal — **DEAD on GC, ES, 6E**

*2026-09-05. `research/cot_screen.py`. Real CFTC legacy reports 2016-2026,
557 weeks per market, joined to daily bars at PUBLICATION time (Friday 21:00
UTC) so no bar sees a report before it existed. 24 trials, bar |t| > 3.08.*

The first hypothesis class that is not a function of price. Four
constructions from the literature - fade speculator extremes, follow the
4-week change in speculative positioning, follow commercial hedgers at their
extremes, fade the spec-vs-commercial gap - on gold, S&P and euro, at 5 and 20
day horizons.

Nothing approaches the bar. The nearest, fading crowded euro longs at 20 days,
sits at t = -1.66 in 3 of 10 years - and the sign is the *opposite* of the
hypothesis: crowded positions continued rather than reverted. On gold, every
construction is negative or flat.

This is Sanders (2004) rather than Wang (2001): no pervasive predictive power,
and nothing that transfers between markets. The literature was split; on these
three markets over this decade it splits toward nothing.

Price proxy: IC Markets daily CFD for the same underlying. Sound for a
signal-level test (demeaned returns, basis far inside one ATR); costs were not
modelled by design. Nothing survived to need them.

Data quality: the loader tracked the CFTC's own renames - ES in 2022, WTI in
2023 - and NQ's e-mini row disappearing behind the micro row, which is now
aliased. 11 years, 5 markets, 152,000 rows.

---

## 007 — Diversified trend following on 33 CME markets — **PRE-REGISTERED, awaiting data**

*2026-09-05. Calendar, research profile and scripts built and tested; the
Databento key is the user's credential and is pending. Thresholds declared here,
before any bar is downloaded.*

Why this hypothesis and not another: every family tested so far (001–006) was
a price pattern on four to eight correlated retail CFDs. The one family with a
century of published evidence — Hurst, Ooi & Pedersen (2017): 67 markets,
1880–2016, Sharpe ≈ 0.4 after costs at the portfolio level — was tested in 002
on four instruments, which is the one setting where the evidence itself says it
is barely visible. A trend book's return comes from breadth, and the
diversification benefit keeps rising past 30 markets (Man Group, 2025).

Universe: `FULL_UNIVERSE` — 33 full-size CME Group markets, 7 sectors (index 4,
rates 5, FX 7, metals 4, energy 4, grains 6, meats 3). Data root = full size;
sizing root = micro where one exists.

Signal: `TrendFollowing` exactly as it stands in `strategies/trend.py`. Three
speeds only — lookback 20, 60 and 120 days — as three sleeves and as their
equal-weight ensemble. **That is four trials.** No other parameter will be
varied. Running total: 174.

Costs: `CostModel.for_futures` — one tick of spread, half a tick of slippage,
commission per side from the root; stressed at 2×; roll friction journaled per
roll; no swap.

Data: Databento GLBX.MDP3, ohlcv-1d, 2011-01-01 to present, per expiry,
stitched by `data/continuous.py` with Panama back-adjustment and rolls before
first notice.

Pass thresholds, declared now:

1. Ensemble net Sharpe ≥ 0.40 over the full sample at the 2× cost stress, at an
   equity where sizing granularity never binds (`--equity 2000000 --size-as full`).
2. Positive net P&L in at least 70% of calendar years.
3. Probability of backtest overfitting (CSCV, 8 partitions) < 0.50 across the
   three speeds — the choice of speed is not what makes it work.
4. Deflated Sharpe > 0 given 174 trials.
5. Walk-forward: the last five years out of sample under parameters fixed on
   the first ten, net Sharpe > 0.
6. At least five of the seven sectors contribute positive net P&L in the
   ensemble. Breadth, not one lucky sector.

Fail any one and the family is reported dead like the others. Pass all six and
the next step is the same book under `--profile challenge` at the capital
ladder's equity levels, then a paper run on IB.

What this cannot show: anything about intraday, anything about the CFD broker.
It is the daily-bar, exchange-cleared version of the only idea left with
evidence behind it.

---

## 008 — Continuous forecasts, volatility targeting, position inertia — **PRE-REGISTERED, awaiting data**

*2026-09-05. Built and tested on synthetic data; runs the day 007's data lands.
Thresholds declared here, before any result.*

What changes and what does not. The direction rule of 007 is untouched. Two
things change. First, the size of a position follows the strength of the
trend: the lookback return in units of the volatility expected over that
horizon, capped at 2σ for full size, floored at a quarter of full size so no
dust position pays a full spread. Second, while the position is open the
strategy re-proposes that size and a fresh 2.5-ATR stop every bar, and the
risk engine (`RiskEngine.resize`) ratchets the stop tighter only, sizes the
target from the real distance to that stop, holds inside a 25% inertia band,
reduces freely, and puts an increase through every limit as new risk. The
constants 2.0, 0.25 and 25% follow Carver's published practice and were not
tuned on anything.

Why it should help, per the literature: volatility scaling raises the Sharpe
of time-series momentum and cuts its drawdowns (Moskowitz, Ooi & Pedersen
2012; Baltas & Kosowski); continuous forecasts with inertia lower turnover,
which is the cost line a small account feels most.

Trials: the three speeds and their ensemble, continuous. **Four.** Running
total 178.

Pass thresholds, against 007's result on the same data, same costs, and an
equity where the confidence floor never meets a minimum contract
(`--equity 10000000 --size-as full`):

1. Ensemble net Sharpe ≥ 007's ensemble net Sharpe − 0.05, at the 2× cost
   stress. Not worse is the bar; better is the hope.
2. Friction as a share of gross P&L ≤ 007's.
3. Maximum drawdown ≤ 007's.
4. Positive years ≥ 007's count − 1.

Pass all four and continuous becomes the standing form of the trend sleeves.
Fail any and 007's discrete form stands, and this entry records why.

---

## 009 — Carry from the futures curve — **PRE-REGISTERED, awaiting data**

*2026-09-05. Built and tested on synthetic curves. Thresholds declared here.*

The first signal in this log that is not a function of price history. Carry is
read off the curve by the stitcher: (front − next) / front, annualised by the
days between the two expiries, on every day both contracts print. Backwardation
pays a long, contango pays a short (Koijen, Moskowitz, Pedersen & Vrugt 2018).
The rule: smooth over 20 days, divide by the 252-day standard deviation of the
market's own carry so a bond and a gas contract are judged on one scale, the
sign is the side, the magnitude capped at 2 is the confidence, flat below 0.25,
2.5-ATR stop. Continuous by construction. Constants fixed, not tuned.

Trials: carry alone; carry with the trend ensemble (007's or 008's, whichever
stands). **Two.** Running total 180.

Pass thresholds:

1. Carry alone: net Sharpe ≥ 0.30 at the 2× cost stress; positive in at least
   60% of years; at least 5 of 7 sectors positive.
2. Weekly return correlation between the carry sleeve and the trend ensemble
   below 0.5. Otherwise it is trend wearing a second hat and adds nothing.
3. Trend + carry: net Sharpe above the better of the two alone. The
   diversification has to show up in the number, not in the story.

Fail 1 and carry is dead here. Pass 1 and fail 2 or 3 and carry is reported
real but redundant, which is also worth knowing.

### 007-009, amendments made before any result was read (2026-09-05)

The download landed and the first stitch pass exposed three things. All were
fixed before a single equity curve was looked at; they are recorded here so
nobody can later claim the definition moved after the fact.

- **Data, parsing.** A raw CME ticker carries one year digit. One stray print
  of NQ December 2029 (NQZ9) was clamped into the present by an assumption
  that no contract lists more than two years out, and overwrote NQ December
  2019 with one row; the stitch then failed at the September 2019 roll. The
  clamp is gone, segments of one contract are merged instead of overwritten,
  and the raw download is kept under `data/futures/_raw` so any future parse
  is free. The full universe was downloaded twice; $21 of the free credit.
- **Data, validation.** CL May 2020 was refused for a non-positive price. It
  printed -37.63 on 2020-04-20; that is history, not corruption. Futures files
  are validated with non-positive prices allowed; CFDs are not.
- **Estimator.** Momentum in `TrendFollowing` was a percentage return and the
  008 forecast divided it by log-return volatility. A back-adjusted series is
  the real one plus a constant per roll: its level is meaningless and can be
  negative, so ratios and logs are undefined on it. Momentum is now the price
  difference over the lookback (same sign on any positive series, so 007's
  rule is unchanged where it was already defined) and the 008 forecast is that
  difference over the standard deviation of one-day price changes times the
  square root of the horizon - the same t-statistic, shift-invariant. The cap
  and floor are unchanged.
- **Not an error.** ES term drag of about +38 index points a year is real: with
  rates above the dividend yield since 2022, the next contract trades above the
  front, and a long pays it. It is the carry signal, seen from the cost side.

---

## 007 — VERDICT: **DEAD in this form.** 008 — **DEAD.** 009 — **DEAD in this form.**

*2026-09-05, real Databento data, 33 markets, 2011-2026, $20M so one contract is
always inside the risk budget, 2x cost stress, research profile with caps wide
enough to hold every signal. `research/futures_gauntlet.py`, numbers in
`state/gauntlet_00{7,8,9}.json`.*

| | 007 discrete trend 20/60/120 | 008 continuous | 009 carry, continuous |
|---|---|---|---|
| net Sharpe | **-0.02** | -0.41 | -0.71 |
| max drawdown | 84% | 96% | 96% |
| positive years | 4 / 15 | 4 / 15 | 2 / 15 |
| positive sectors | 1 / 7 | 0 / 7 | 2 / 7 |
| PBO across speeds | 0.54 | - | - |
| trades (+ resizes) | 28,606 | 56,707 (+ resizes) | 18,648 (+ 37,022) |
| gross P&L | +8.9M | | +10.6M |
| friction | 26.1M = 295% of gross | 177% | 32.5M = 306% |

Five of six thresholds failed for 007 (only the deflated Sharpe "passes", at
0.000, because a negative Sharpe is trivially not above noise). 008 was
judged against 007 and is worse on Sharpe and drawdown. 009 failed all three
of its own.

### What the failure is, and is not

The plumbing is right. The slow rule on gold alone earns +$2.2M on $20M at
Sharpe 0.27, 23% win rate, +0.12R expectancy: the textbook trend shape. The
S&P alone is positive. Both were the largest trends of the period and both
were captured.

The book died of **turnover**, not of the signal's sign. Gross P&L is
positive in every run. Friction is three times gross. The rules as declared
re-decide every day: the trend rule exits on any close through its moving
average and re-enters the next morning, and re-enters immediately after a
stop; the carry rule resizes whenever a daily forecast normalised by its own
standard deviation moves a quarter, which in a market whose carry sits near
zero is most days. That is 10 to 40 round trips a year per market. The
published rules decide once a month and hold.

One diagnostic run, not pre-registered and not a trial for any verdict: carry
with resizing switched off, everything else identical. Sharpe **+0.15**, max
drawdown 39%, **10 of 15 years positive**, 2,035 trades instead of 18,648,
gross +4.6M against 3.5M friction. Same signal, one tenth of the trading,
from -$43M to +$1.1M. That is the whole lesson of this entry.

Not an error, for the record: ES term drag of about +38 points a year (rates
above dividends since 2022) and GC of about +52 dollars a year (contango) are
real and are the cost side of the carry signal.

---

## 010 — Published forms, monthly decisions — **PRE-REGISTERED, declared before running**

*2026-09-05. Declared after reading 007-009's verdicts and the diagnostic
above, and before any 010 number exists.*

**Trend (`strategies/tsmom.py`).** Time-series momentum as in Moskowitz, Ooi
and Pedersen (2012): the sign of the trailing price change over the lookback
decides the side; the decision is taken on the FIRST BAR OF EACH MONTH only
and held until the next; between decisions nothing re-enters, including after
a stop. No moving-average filter. A 4-ATR disaster stop, because the risk
engine will not hold an unstopped position, wide enough that the monthly
decision, not the stop, is what normally closes a trade. Speeds 60, 120 and
250 days (the published 12 months is 250) and their equal-weight ensemble.
Discrete sizing at entry. Four trials.

**Carry (`strategies/carry.py`, `normalise="price_vol"`).** Carry in risk
units, as Carver runs it: the 20-day-smoothed annualised roll yield divided by
the market's annualised price volatility (63-day, from price differences over
the RAW front close, which the stitcher now carries). A quarter of a
standard deviation of carry in a market with none is not a signal; ten percent
of annual volatility is. Threshold 0.10, cap 0.50, decided on the first bar
of each month, discrete, 4-ATR stop. Two trials: alone, and beside the trend
ensemble. Running total 186.

Thresholds are those already declared for the families: 007's six for the
trend ensemble, 009's for carry (1a-1c alone; 2-3 for the combination), at
the 2x cost stress, $20M full-size. No number is lowered for having failed.

---

## 010 — VERDICT: **alive, and below the bar. DEAD as declared.**

*2026-09-05, same data, same $20M, same 2x cost stress, same research
profile. `state/gauntlet_010*.json`.*

| | 010 trend 60/120/250, monthly | 010 carry in risk units, monthly | trend + carry |
|---|---|---|---|
| net Sharpe | **0.28** | **0.21** | **0.31** |
| max drawdown | 47% | 23% | 46% |
| positive years | 7 / 15 | 9 / 15 | |
| positive sectors | 4 / 7 | 4 / 7 | |
| PBO across speeds | **0.00** | - | |
| deflated Sharpe | 0.022 | | |
| last five years Sharpe | **+0.43** | | |
| trades | 4,244 | 824 | 4,738 |
| friction / gross | **21%** | 20% | 21% |
| carry vs trend correlation | | | **0.14-0.21** |

Trend fails thresholds 1, 2 and 6 and passes 3, 4 and 5. Carry fails 1a and
1c, passes 1b, and the combination passes both of its own conditions: the
correlation is a fifth, and the book earns more than either sleeve alone
(diversification ratio 1.34x). The declared rule is all-or-nothing, so the
entry is reported dead in this form. No threshold is lowered.

### What changed between 007 and 010, and what did not

Same 33 markets, same data, same costs. Deciding once a month instead of
once a day took friction from 295% of gross to 21%, the drawdown from 84% to
47%, and the Sharpe from -0.02 to +0.28. The 12-month speed - the one
Moskowitz, Ooi and Pedersen actually published - made +$19M at +0.32R
expectancy across the universe; the 60-day speed lost $5.5M and the ensemble
carried it. The order of the speeds is the same in every in-sample and
out-of-sample half (PBO 0.00), which is what a real, slow effect looks like
and what a fitted one never does.

Yearly signs of the trend book: down 2012, 2016, 2017, 2018, 2019, 2023,
2024; up 2013, 2014, 2015, 2020, 2021, 2022, 2025, with 2014 and 2022 the two
large years. That is the signature of the trend-following industry over the
same period - the large CTAs' index lost money in 2011, 2012, 2016, 2018 and
2023 and made its decade in 2014 and 2022. The machine reproduces what the
professionals lived through, at a Sharpe in the same region as theirs. The
bar of 0.40 came from the century-long average; 2011-2026 was not an average
fifteen years for trend, and the machine says so rather than flattering it.

### What this does and does not license

- **Not licensed:** picking the 250-day speed alone because it won. That
  choice was not declared, and on this data it is not out of sample. It can
  be declared now as entry 011 and tested only on data it has not seen: a
  paper-traded record from here forward, which is the plan anyway.
- **Not licensed:** trading this with the user's capital. Sharpe 0.3 with a
  47% drawdown at $20M full-size is what it is; the capital ladder says what
  a small account can hold of it, and the answer is little.
- **Licensed, and the useful result:** the research machine is validated end
  to end on real exchange data. It finds the effect that is there, at the
  size the industry found it, and it refuses the ones that are not. Every
  earlier family died at the signal; this one is alive at the signal and
  fails only the bar.

Trials: 186. Every family with published evidence has now been tested in
the form the evidence was published in.
