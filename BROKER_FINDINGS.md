# Broker findings

Everything here is **measured from the live terminal**, not assumed. Re-run
`scripts/snapshot_broker.py` against your own account before trusting any of it.

> **History note.** This session began on an Exness cent account (43 symbols,
> zero balance, no indices, 8-point EURUSD spread) and the terminal later
> switched to IC Markets. Conclusions drawn on the first account are archived in
> `RESEARCH_LOG.md` and do **not** transfer: cent contracts are 1/100th the size
> and the spreads differ by 8x, so costs from one broker are meaningless for the
> other. That old bar store is kept at `data/bars_exness_archived`.

## Account

| | |
|---|---|
| Broker | Raw Trading Ltd (IC Markets) |
| Server | `ICMarketsSC-Demo` |
| Balance | 103,391.20 USD |
| Leverage | 1:5000 |
| Symbols | 7,391 |
| Trade mode | DEMO (`trade_mode=0`) |
| Algo trading | **enabled** 2026-09-05 |

## Specs and measured spreads

| Symbol | Min lot | $/1.0 move | Spread | Swap long / short |
|---|---|---|---|---|
| EURUSD | 0.01 | 100,000 | 0.5-1.0 pts | -8.17 / +1.45 |
| GBPUSD | 0.01 | 100,000 | 1.0-1.5 pts | -3.83 / -4.18 |
| USDJPY | 0.01 | 639.5 | 1.0-2.0 pts | +7.97 / -16.67 |
| XAUUSD | 0.01 | 100 | 9.0 pts | -57.55 / **+39.88** |
| US30 | 0.10 | 1.0 | 120 pts | -12.24 / -0.56 |
| US500 | 0.10 | 1.0 | 50 pts | -1.77 / -0.08 |

Gold pays you to hold shorts (+39.88) and charges heavily for longs (-57.55).
Any strategy holding gold overnight should know which side it is on.

## Server timezone - read this before writing any time-based rule

**The server runs UTC+3 in US daylight saving and UTC+2 otherwise.** MT5 hands
you epochs from that clock, not UTC, and the Python package does not tell you.

Established from the data rather than assumed: the US cash open is the sharpest
recurring event in the set, and on US30 M15 it sits at **16:30 server time in
both summer and winter**. That can only hold across the DST boundary if the
server clock shifts with US DST, which puts server midnight at the 17:00 New
York close.

`execution/brokertime.py` does the conversion and `verify_offset()` re-checks it
on every connect, because a broker changing its server timezone should stop the
system rather than silently shift a year of research.

After the fix, the cash open lands at 13:30 UTC in summer and 14:30 in winter -
which is correct, and is asserted in `tests/test_brokertime.py`.

## History depth

The terminal caps a request at the "Max bars in chart" setting; above it the call
returns **nothing at all** rather than a truncated series. Raised here to
Unlimited (Tools > Options > Charts).

| TF | EURUSD | XAUUSD | US30 / US500 |
|---|---|---|---|
| M5 | 800k / 10.7y | 680k / 28y | 671k / 14y |
| M15 | 300k / 12y | 235k / 28y | 229k / 14y |
| M30 | 150k / 12y | 123k / 28y | 119k / 14y |
| H1 | 100k / 16y | 68k / 28y | 64k / 14y |
| D1 | 8k / 31y | 7.5k / 28y | 3.6k / 14y |

Before raising the setting, M5 reached back only 241 days - too short to
walk-forward validate anything.

## Cost arithmetic by stop timeframe

`cost/stop` is round-trip friction as a share of the risk on each trade. It is
the number that decides whether a strategy is viable, and it depends far more on
the **size of the move being traded** than on the entry timeframe.

| Symbol | stop from M5 | from H1 | from H4 |
|---|---|---|---|
| EURUSD | 4.8% | 1.4% | 0.6% |
| XAUUSD | 1.4% | 0.3% | 0.1% |
| US30 | 8.0% | 1.9% | 0.7% |
| US500 | **31.2% dead** | 5.6% | 2.2% |

M5 entry is viable here on EURUSD, gold and US30. It was not on the previous
broker, where the same table read 22-24%. That was a spread problem, not a
timeframe problem.
