# Trading system

Automated trading for FX, gold and US index CFDs. Python 3.12, broker-agnostic
above the execution adapter.

Design documents:

- [Build plan](https://claude.ai/code/artifact/958ee8d0-0ba3-4cea-bea0-bcaf47015162) — architecture, where AI fits, six phases
- [Strategy research](https://claude.ai/code/artifact/674caf4f-7de1-4ee5-9e0f-5d37fe6b0f1d) — eight families and what the evidence says

## Status

**Broker connected** (IC Markets demo) with 10-28 years of history per symbol,
and the strategy restructured to multi-timeframe: H1/H4 for bias, M5/M15/M30 for
entry. See [BROKER_FINDINGS.md](BROKER_FINDINGS.md) for measured specs, spreads
and the server-timezone correction, and [RESEARCH_LOG.md](RESEARCH_LOG.md) for
what has been tested and why it died.

**Phase 4 — AI layer, feature complete.** Regime model, meta-labelling with
triple-barrier labels, purged/embargoed CV, CPCV, deflated Sharpe, PBO, Monte
Carlo, and the full validation gauntlet. 129 tests passing.

All six phases are built. What remains is not code: real data, a calibrated cost
model, and the demo forward test.

## Layout

```
core/        domain types, config loading, Sleeve, futures contract calendar
execution/   ExecutionAdapter protocol, MT5 + paper adapters
risk/        sizing, limits register, the risk gate
data/        Parquet bar store, validation, gap analysis, downloader, continuous futures, CFTC COT
features/    causal indicators - ATR, EMA, momentum, Donchian
strategies/  S1 trend baseline, buy-and-hold control
backtest/    engine, cost model, metrics, PORTFOLIO (many sleeves, one book)
live/        runner, execution worker, durable session state
ops/         append-only decision journal
ml/          labelling, purged CV, regime + meta models, DSR/PBO/Monte Carlo
config/      risk profiles and instrument map — nothing operational is in code
research/    prop challenge Monte Carlo
scripts/     check_specs, download_history, verify_roundtrip, backtest,
             run_live, flatten_all, chaos_test, train_and_validate
tests/       every limit forced to breach at least once
```

## The one architectural rule

Strategies, risk and research talk to `ExecutionAdapter` and nothing else. No code
above `execution/` may import `MetaTrader5` or any other venue library. That rule
is what makes the broker a swappable detail rather than a rewrite, and it is worth
enforcing in review.

The second rule follows from it: **signal proposes, risk decides.** A `Signal`
carries a direction, a stop distance and a confidence — deliberately not a lot
size. Sizing belongs to the risk engine, and that boundary is what lets machine
learning be added later without adding a new way to blow up.

## Getting started

```bash
pip install numpy pandas pyarrow pyyaml pytest
python -m pytest tests/ -q
python scripts/check_specs.py --offline --equity 5000
```

Everything above runs without a broker. Connecting one:

```bash
pip install MetaTrader5              # Windows only, needs a running terminal
python scripts/check_specs.py --equity 5000
```

Fill in `config/instruments.yaml` aliases from the live output — brokers rename
instruments (`US30.cash`, `DJ30`, `XAUUSD.s`).

## Risk profiles

`challenge` and `funded` are two YAML files, not two code paths.

Each limit has a **soft** and a **hard** threshold. Hard is the firm's published
number, which must never be reached; soft is where the system stops. The gap is
the buffer that survives one bad fill. On an evaluation account this distinction
is the difference between a bad day and a failed attempt.

`risk_per_trade` in the challenge profile is 0.5%, taken from the Monte Carlo
sweep in `research/prop_challenge_sim.py`. For a genuine edge, P(pass) peaks near
0.5% and falls as risk rises. Evaluations no longer carry time limits, so time is
free and drawdown is not.

## Minimum viable equity

`scripts/check_specs.py` computes the smallest account that can express your risk
limit in each instrument's minimum lot. This decides scope, and it is arithmetic
rather than preference.

At 5,000 equity, 0.5% risk, 2.5×ATR stops on daily bars (indicative ATR values):

| Symbol | Min lot risk | % of equity | Min viable equity | |
|---|---|---|---|---|
| EURUSD | 17.50 | 0.35% | 3,500 | tradeable |
| US500 | 15.00 | 0.30% | 3,000 | tradeable |
| US30 | 112.50 | 2.25% | 22,500 | blocked |
| XAUUSD | 125.00 | 2.50% | 25,000 | blocked |

Gold and US30 are not blocked by strategy quality. The smallest position the
broker will accept risks four to five times the limit, and sizing up to reach the
minimum would silently multiply risk — which is why `size_position` returns
`BELOW_MINIMUM` rather than rounding up. The constraint lifts when equity rises or
the stop distance shrinks, and not before.

Rerun this against your own broker before choosing what v1 trades. The table above
uses fixture specs.

## Phase 1 gate

Not met until all four hold:

- [x] Specs load and are asserted; minimum viable equity computed per instrument
- [x] Full test suite green, every limit forced to breach at least once
- [x] Data pipeline built and verified end to end against the paper adapter
- [ ] Five years of M1 stored for every instrument — needs a broker
- [ ] A demo order opened, modified and closed from Python — needs a broker

Both remaining items are one command each once MT5 is connected:

```bash
python scripts/download_history.py --years 5 --timeframes M1 H1 D1
python scripts/verify_roundtrip.py --symbol EURUSD --dry-run
python scripts/verify_roundtrip.py --symbol EURUSD
```

`verify_roundtrip` refuses to run on anything but a demo account. It opens a
position, and a bug in the close path leaves it open.

## Data

Bars live in `{root}/{symbol}/{timeframe}/{year}.parquet`, timestamps UTC and
timezone-aware, referring to the bar's **open**. Both are enforced, not merely
documented — a mixed convention is unrecoverable once research sits on top of it.

Writes are validated and merged. Duplicate timestamps, unsorted series, naive
datetimes, NaN or non-positive prices, `high < low`, and open/close outside the
bar are all refused rather than absorbed. Re-downloading an overlapping range is
idempotent, so an interrupted download resumes safely.

Before writing any session filter, check what your broker's clock actually does:

```bash
python scripts/download_history.py --sessions EURUSD
```

Most MT5 servers run GMT+2/+3 with daylight saving. If the trading week appears
to start at 22:00 UTC rather than 21:00, there is an offset you have not
accounted for, and every session rule built on that data will be an hour wrong.

## Testing

```bash
python -m pytest tests/ -q
```

Every test runs against `PaperAdapter`, which crosses the spread and charges
slippage against you on every fill. A paper broker that flatters the strategy is
worse than none, because it launders bugs into confidence.

## Backtesting

```bash
python scripts/backtest.py --symbol EURUSD --synthetic --compare-hold
python scripts/backtest.py --symbol EURUSD --stress 2
```

The backtester replays bars through the **same `RiskEngine` object** that runs
live — same limits register, same sizing code, same YAML. A backtest with
different risk logic than production is testing a system you will never trade.

Pessimistic by construction:

- Signals come from **closed** bars; entry is at the **next** bar's open
- Gaps through a stop fill at the **open**, not the stop price, so a trade can
  lose more than 1R — because it can
- Spread and slippage are charged on both fills and counted in `cost_drag`
- `--stress 2` doubles all costs. This is the sensitivity gate, not decoration:
  if the edge dies at 2x, it was never an edge

### Look-ahead

`tests/test_backtest.py::test_no_lookahead_in_signals` recomputes every signal on
truncated data and asserts it is unchanged. A look-ahead bug does not crash, does
not look wrong, and produces a superb equity curve — an assertion is the only
thing that catches it.

There is also a sanity anchor: `test_trend_loses_roughly_costs_on_a_random_walk`.
A random walk has no trend to follow, so a trend system must lose about its
friction there. If that test ever showed a strong edge, the harness would be lying.

### Reading a result

`cost_drag` is reported next to returns rather than buried, because it is the
number that decides whether a strategy is viable at your broker. `verdict()`
applies the thresholds from the strategy research: below 30 trades is
inconclusive, drag over 50% is cost-dominated, and a Sharpe above 2 is
"suspicious — check for look-ahead first".

## Running live

```bash
python scripts/run_live.py --dry-run       # startup checks, no loop
python scripts/run_live.py --symbols EURUSD
```

Startup order is not arbitrary — each step depends on the previous one:

1. connect, read equity
2. **restore the session book**, carrying today's daily loss forward
3. **reconcile against the broker**, flagging orphan positions
4. **repair missing stops** — nothing trades until every position has one
5. only then, trade

`order_send` blocks, so it runs on a worker thread behind a queue. A slow fill on
gold during a news spike must not stall ingest for everything else.

Shutdown on Ctrl+C is graceful and **does not close positions**. Stopping the
software is not the same as wanting to be flat.

### Shadow mode

```bash
python scripts/shadow.py --minutes 30                          # attended
python scripts/shadow.py --until 2026-09-11T21:00Z --quiet     # unattended, absolute end time
python scripts/shadow_report.py --hours 24                     # digest of the journal
```

The real runner, risk engine, OMS and journal on live MT5 prices, with orders
routed to the paper adapter. It is the plumbing test a backtest cannot be: live
tick freshness, bar-close detection, the worker thread, state persistence and
every account limit against real account numbers. No order reaches the broker.

Unattended runs must outlive the broker: a failed iteration is journalled as
`loop_error`, three in a row force a reconnect, and the loop carries on. `--until`
is an absolute time so a restarted run ends when the original would have.

**The scheduled week, 2026-09-06 → 09-11.** Three Windows Task Scheduler tasks,
registered 2026-09-05. All run "only when user is logged on", because the MT5
Python API needs the terminal in the same session — disconnect RDP, do not sign out.

| task | when (UTC) | runs |
|---|---|---|
| `TradingShadowWeek` | Sun 21:00 once; 125 h limit; 3 restarts | `scripts/shadow_week.cmd` → `shadow.py --until 2026-09-11T21:00Z --quiet` |
| `TradingShadowReport` | Mon–Fri 21:30 | `scripts/shadow_report.cmd 24` → `state/shadow_reports.md` |
| `TradingShadowWeekReport` | Fri 21:40 once | `scripts/shadow_report.cmd 168` |

Watch `state/shadow_week.log`, `state/shadow_journal.jsonl`, `state/shadow_reports.md`.
Stop early: create `state/SHADOW_KILL` (the runner halts and stays halted), or
`schtasks /End /TN TradingShadowWeek`. Remove everything with
`Get-ScheduledTask -TaskName "TradingShadow*" | Unregister-ScheduledTask -Confirm:$false`.

The strategy in shadow is `MTFPullback` on M15 — a family the research killed. The
week measures the infrastructure, not the edge: expect the digest to show refusals
and heartbeats, not profits.

### Emergency

```bash
python scripts/flatten_all.py --dry-run    # show what would close
python scripts/flatten_all.py              # close all, engage kill switch
python scripts/flatten_all.py --clear-kill # resume, deliberately
```

`flatten_all` connects to the broker independently and imports nothing from the
runner, so it works when the runner is hung or wedged in a retry loop — which is
exactly when you need it.

The kill switch is a file at `state/KILL`. Any process can set it, it survives a
crash, and nothing in the system ever clears its own. An unreadable kill file is
read as *engaged*: the only safe interpretation of a corrupt stop signal is stop.

## Crash recovery

```bash
python scripts/chaos_test.py
```

Five scenarios, all on the paper adapter, no broker needed. The second is the one
that matters:

> Restart at 14:00 after a morning that already lost 3%. If `day_start_equity`
> re-initialises from current equity, the daily-loss limit silently re-arms
> against the lower base and the system will happily lose another 3.5% on a day
> it had already spent its budget. Nothing errors. On an evaluation account that
> is the account.

So risk bookkeeping is persisted to `state/session.json` (atomic write), positions
come from the broker, and startup cross-checks the two. `day_start_equity` is
carried forward within a session; `starting_equity` and `high_water_equity` survive
day rolls, because max-drawdown limits span the whole evaluation.

### Idempotency

Every order carries a deterministic client id derived from strategy, symbol, side
and originating bar, written to the broker's comment field. After an ambiguous
failure — a timeout, where you cannot know whether the order filled — the OMS
looks for a position carrying that id before retrying. Assuming failure and
retrying doubles the position; that is the worst outcome available, and this is
what prevents it.

Retries are classified, not blanket: a requote is retried, "invalid stops" is not,
and an unrecognised reason is not retried at all — investigate it instead.

## The AI layer

```bash
python scripts/train_and_validate.py --symbol EURUSD --synthetic
```

Trains on the first 60% and scores both versions on the untouched remainder.
Three tiers, in order of return on effort:

| Tier | Model | Output |
|---|---|---|
| 1 | Regime (unsupervised GMM) | position scalar |
| 2 | Meta-label (calibrated boosting) | take / skip |
| 3 | Adaptive sizing | `Intent.confidence` |

Every model outputs a **scalar or a filter**. None outputs an order, a direction,
or a lot size. `Intent.confidence` is bounded to `[0, 1]` at construction, so the
worst a broken model can do is size to zero — never above the risk register.
`test_trend_ml_never_exceeds_the_baseline_position_size` asserts it.

Probabilities are isotonic-calibrated. An uncalibrated boosting score is not a
probability, and tier 3 sizes on it.

### Why meta-labelling and not price prediction

The model is never asked which way price will go — that is near a coin flip. It is
asked: *given that the strategy fired, does this one work?* The strategy has
already restricted the sample to the cases that matter, which is why that question
is learnable when direction is not.

Labels come from a triple barrier — profit target, the strategy's own stop, and a
time limit — so the label matches the trade the system would really have taken.
When both barriers sit inside one bar the **stop is assumed first**, the same
pessimism the backtester uses.

### Purged cross-validation

A label at bar 100 resolving at bar 120 shares its outcome with one at bar 110.
Plain k-fold puts them either side of a split and the model has seen the answer.
Every label carries `t1`; training observations overlapping the test set are
purged, and an embargo removes serial correlation at the boundary.

`leakage_report()` prints what purging cost and asserts the overlap is gone.

## The validation gauntlet

```bash
python scripts/train_and_validate.py --symbol EURUSD --synthetic
```

Nine gates, thresholds declared in `GauntletThresholds` **before** the run. After
the fact every threshold looks negotiable, and the version of you reading a
promising equity curve is not the person who should set the bar.

The two that matter most:

- **Deflated Sharpe** — corrects for how many configurations you tried. Test 500
  variants and a Sharpe of 1.5 may be pure selection bias. The trial count must be
  honest across your whole research effort, not just the current run.
- **PBO** — across 70 train/test splits, how often does the in-sample winner land
  below median out-of-sample? Above 20% and your *selection procedure* is fitting
  noise, regardless of how good the winner looks.

Both are calibrated against known answers in `tests/test_ml.py`: PBO returns ≈0.5
on pure noise and <0.2 when one configuration has a real edge; PSR returns exactly
0.5 when the observed Sharpe equals the benchmark.

## The portfolio layer

```bash
python scripts/backtest_portfolio.py --symbols EURUSD XAUUSD US500 --max-positions 6
```

A **sleeve** is one strategy, the symbols it trades, and its share of the risk
budget (`core/sleeve.py`). The book is a list of sleeves. Two sleeves may hold
the same symbol, in opposite directions if they disagree; the risk engine nets
the exposure and the correlated-bucket limit caps the total.

`PortfolioBacktester` runs every (sleeve, symbol) leg in lockstep on one clock,
sharing **one equity curve and one risk engine**. When sleeve A asks to open,
the risk state it is judged against already contains sleeve B's positions. A
portfolio of one sleeve produces trades identical to the single-symbol
backtester — that equivalence is asserted in `tests/test_portfolio.py`.

**The allocator** is a limit like any other: `SleeveBudget` gives each sleeve a
share of `max_open_risk` (2% by default), measured from stops, and caps the
book. It sits in the register beside the others and is journaled like them.

**Attribution and correlation** come out of the same run: per-sleeve P&L, the
weekly return correlation between sleeves, and the *diversification ratio* —
the Sharpe multiple the book earns over its average sleeve, with correlation
measured rather than assumed. Two truly independent sleeves give 1.41×; two
momentum variants with different lookbacks gave **1.16× at ρ = 0.47**. That is
the number the whole "run many strategies" argument rests on, and it is now
measured, not hoped for.

Sleeve names are at most 12 characters because they are the order-comment
prefix that identifies a live position's sleeve after a restart.

One interaction to know: the profile's `max_concurrent_positions` was written
for a single strategy. A book of S sleeves over K symbols can want S×K
positions, and the default cap of 3 starved the allocator before it did
anything — 4,603 rejections in the first run. Size it for the book.

## Futures

The CFD side proved its universe empty. The futures side is the same machine
pointed at exchange-cleared micro contracts, which is where the research says an
edge can exist. Everything above the execution adapter is unchanged.

| Piece | File | Proven by |
|---|---|---|
| Contract calendar: 33 CME Group markets, 7 sectors, expiry **and first-notice** rules, front month, roll dates | `core/contracts.py` | 60 dates checked by hand against the exchange specs; every root's schedule contiguous 2018–2026 |
| Back-adjusted continuous series, roll log, roll cost | `data/continuous.py` | no jump at the roll; returns preserved |
| IB adapter: front-month resolution, child stops, roll | `execution/ib_adapter.py` | 12-step round trip on the test double |
| IB test double | `execution/ib_fake.py` | — |
| CFTC positioning, joined at *publication* time | `data/cot.py` | look-ahead test; 11 years of real data downloaded |
| Order-flow features from trade prints | `features/orderflow.py` | delta arithmetic |
| Futures cost model: commission, ticks, **no swap** | `CostModel.for_futures` | — |

```bash
python scripts/verify_roundtrip_ib.py --fake                       # the whole path, no broker
python scripts/backtest_futures.py --synthetic --universe full     # 33 markets through the stitch and the engine, no data
python scripts/download_cot.py --years 15                          # free, real
python research/cot_screen.py                                      # positioning as a signal
set DATABENTO_API_KEY=db-...
python scripts/download_databento.py --dry-run --universe full     # Databento's own cost estimate, spends nothing
python scripts/download_databento.py --universe full               # daily bars, every expiry, since 2010
python scripts/backtest_futures.py --universe full --equity 2000000 --size-as full --lookbacks 20 60 120               # 007
python scripts/backtest_futures.py --universe full --equity 10000000 --size-as full --continuous --lookbacks 20 60 120 # 008
python scripts/backtest_futures.py --universe full --equity 10000000 --size-as full --sleeves trend carry              # 009
python scripts/verify_roundtrip_ib.py                              # against TWS paper (port 7497), dry run
```

**Two universes.** `FULL_UNIVERSE` is the research universe: 33 full-size
contracts across index, rates, FX, metals, energy, grains and meats — the breadth
that trend-following evidence says the strategy needs. `MICRO_UNIVERSE` is what a
small account can trade; `MICRO_OF` links the two. History is always read from
the full-size contract (longest record, same price to a tick) and positions are
sized with the micro, so a backtest answers "what could *this* account hold"
rather than "what could a fund hold". The `research` risk profile widens the
account-level limits so a nine-year test measures the signal rather than a prop
rule; the same book is then re-run under `challenge` to see what the rules cost.

**Roll before first notice, not before last trade.** Cash-settled contracts can
be held to expiry. Physically delivered ones — grains, metals, cattle, treasuries
— send delivery notices from first-notice day, which falls *before* last trade.
The calendar anchors the roll on whichever comes first. The original ZN entry
rolled off last trade and would have carried a long past first position day.

**Continuous positions (008) and carry (009).** `--continuous` makes the trend
sleeves size by trend strength — the lookback return in units of the volatility
expected over that horizon — and resize open positions through
`RiskEngine.resize`: the stop only ratchets tighter, the target is sized from the
real distance to that stop, a reduction is a partial close that needs no
approval, an increase is new risk that passes every limit without counting as a
new position, and nothing trades inside a 25% inertia band. The live runner does
the same through the worker (partial closes, adds under their own client id,
stop modifications). `--sleeves trend carry` adds the carry sleeve, which reads
the curve's annualised roll yield off the `carry` column the stitcher now writes
on every continuous bar. Both are pre-registered in the research log with
thresholds fixed before any data, and both run end to end on synthetic curves.

Two rules the futures side adds. **A symbol is a root, not a contract**: the
strategy says `MES`, the adapter resolves the live month and rolls before
expiry, journaling the roll. **Stops are child orders at the exchange**: a
position's stop is found by joining to its child order, and attribution falls
back to the broker's own order records so it survives a restart — the lost-reply
double-fill this exposed is now a regression test.

What still needs your credentials: a Databento key for per-expiry history, and
TWS or IB Gateway running with API access for the live dry run. Neither has a
placeholder fallback; both refuse and explain.
