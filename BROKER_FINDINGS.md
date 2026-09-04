# Broker findings — 2026-09-04

Read live from the terminal. Everything here is measured, not assumed.

## Account

| | |
|---|---|
| Broker | Exness Technologies Ltd |
| Server | `Exness-MT5Real34` |
| Login | <redacted> |
| Trade mode | **CONTEST** |
| Balance | **0.00 USC** |
| Currency | USC (US cents) |
| Algo trading | enabled |

**Nothing can be traded.** The balance is zero. Separately, the mode reports
CONTEST on a server named "Real", which is worth resolving with the broker before
depositing anything.

## What this broker actually offers

43 symbols: FX majors and crosses, XAUUSD, XAGUSD, and a little crypto.

**No indices.** US30, US500, NAS100 — none are available. The index half of the
original plan cannot be traded on this account at all.

Every symbol carries a `c` suffix (cent account): `EURUSDc`, `XAUUSDc`. Aliases
are mapped in `config/instruments.yaml`.

## Specs, live

| Symbol | Contract | Min lot | $/1.0 move | Spread | Swap long/short |
|---|---|---|---|---|---|
| EURUSDc | 1,000 | 0.01 | 100,000 USC | 8.0 pts | -5.70 / 0 |
| GBPUSDc | 1,000 | 0.01 | 100,000 USC | 10.0 pts | -1.30 / -1.60 |
| USDJPYc | 1,000 | 0.01 | 639.6 USC | 10.0 pts | 0 / -13.30 |
| XAUUSDc | 1 oz | 0.01 | 100 USC | 260 pts | **-512.30** / 0 |

The cent contract sizes **solve the small-account lot problem entirely**. The
earlier finding that gold and indices were blocked below ~25,000 was based on
standard-account specs; here the minimum position is 1/100th the size and
granularity is no longer a constraint.

`XAUUSDc` financing on longs is -512.30 cents per lot per night. The M5 ATR on
gold is around 400 cents. **One night of financing costs more than four
five-minute ranges of movement**, so gold longs cannot be held overnight here.

## History available

The terminal caps a single request at ~50,000 bars; above that it returns nothing
rather than a truncated series. `copy_rates_range` fails outright on this
terminal. Raise Tools > Options > Charts > "Max bars in chart" to get deeper
intraday history.

| TF | Bars | Span | Validatable? |
|---|---|---|---|
| M5 | 50,000 | **241–258 days** | no — too short for walk-forward |
| M15 | 50,000 | 736–773 days | marginal |
| M30 | 50,000 | 1,467–1,547 days | yes |
| H1 | 50,000 | ~8 years | yes |
| H4 | 16,000 | ~11 years | yes |
| D1 | 3,900 | ~12 years | yes |

## Timeframe sweep — the answer on M5

`MTFPullback`, H4+H1 bias, real spreads, 9 configurations:

| Symbol | Exec | Trades | Win | Payoff | Expectancy | Sharpe | cost/stop |
|---|---|---|---|---|---|---|---|
| EURUSD | M5 | 128 | 15.6% | 3.93 | -0.252 | -0.56 | **22.4%** |
| EURUSD | M15 | 173 | 27.2% | 2.03 | -0.182 | -0.33 | 12.6% |
| EURUSD | M30 | 33 | 18.2% | 1.76 | -0.485 | -0.84 | 6.9% |
| GBPUSD | M5 | 41 | 9.8% | 4.68 | -0.506 | -1.19 | **23.9%** |
| GBPUSD | M15 | 169 | 29.0% | 1.85 | -0.179 | -0.40 | 13.5% |
| GBPUSD | M30 | 325 | 28.0% | 2.16 | -0.116 | -0.22 | 10.1% |
| XAUUSD | M5 | 33 | 12.1% | 3.26 | -0.488 | -1.47 | 5.9% |
| XAUUSD | M15 | 518 | 32.2% | 2.59 | **+0.138** | 1.55 | 4.7% |
| XAUUSD | M30 | 530 | 34.0% | 2.07 | **+0.015** | 0.65 | 5.4% |

`cost/stop` is round-trip friction as a share of the risk on each trade, and it
explains the whole table. **M5 is the worst timeframe on every symbol** — on FX
you pay 22–24% of your risk budget in spread on every trade. Gold survives only
because its spread is 4% of its M15 range, not because the strategy works better
there.

## The survivor did not survive

XAUUSD M15 showed Sharpe 1.56, 518 trades, and 5/5 positive walk-forward folds.
The gauntlet blocked it on four gates:

- **survives 2x costs: -11% of edge** — it is friction, not an edge
- **deflated Sharpe 0.000** across 17 honest trials
- **PBO 40%** — the selection procedure is fitting noise
- **Monte Carlo 95th percentile drawdown 44.7%**

Picking the best of nine timeframe/symbol combinations plus eight variants is
exactly the search the deflated Sharpe exists to penalise.

## Two bugs this found

**MT5 history paging.** `copy_rates_range` returns "Invalid params" regardless of
how the datetimes are built, and `copy_rates_from` rejects timezone-aware ones.
The adapter now uses position-based paging only, with the 50k ceiling made
explicit rather than silently truncating.

**Swap was not modelled at all.** Every overnight position in every backtest had
free financing. Now charged per rollover crossed, with the triple Wednesday
charge. Impact: XAUUSD M15 expectancy 0.148 -> 0.138, M30 0.041 -> 0.015. FX is
unaffected because those trades are intraday.
