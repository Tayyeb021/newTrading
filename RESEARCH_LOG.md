# Research log

Every idea tested and why it died. Two reasons to keep this: it stops you
re-testing a dead idea in six months having forgotten, and it is the **honest
trial count** that the deflated Sharpe ratio needs. Without that count, no result
here is interpretable.

**Running trial count: 26** (18 backtest configurations + 8 gauntlet variants,
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
