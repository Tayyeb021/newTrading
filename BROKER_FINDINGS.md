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

---

## Demo fills at the quote: slippage cannot be measured here (2026-09-07)

Four real round trips on IC Markets demo #52946213, 60/60 checks passed, one
minimum lot each, opened, stop attached, stop moved, closed, account reconciled
flat. The live path is proven end to end on EURUSD, XAUUSD, US30 and US500.

**Every one of the eight legs slipped exactly zero.** Across instruments quoting
from 0 to 120 points of spread, including a six-second-old quote on US30, the
fill came back at the quoted price every time, and every operation took 288-300ms.
That is not execution quality; it is a demo server with no liquidity to consume,
filling at whatever it last quoted.

Consequence: **slippage is not measurable on this account.** Writing zero into
the cost model would halve modelled friction. Entry 007 died at 295% of gross,
so halving friction would revive strategies the research has already killed -
the worst error available to this repository. `calibrate_costs.py` therefore
refuses any slippage sample whose median is zero or below 5% of the spread,
keeps the assumed half-spread, and records `slippage_calibrated: false` per
symbol. Only a live account can settle this.

Spreads, being quoted rather than filled, do survive the demo. Measured in the
London pre-open, which is the worst hour:

| symbol | measured spread | previously assumed |
|---|---|---|
| US30 | 120 pts | 350 pts |
| XAUUSD | 8 pts | 28 pts |
| US500 | 50 pts | 55 pts |
| EURUSD | 0 pts | 12 pts |

The assumptions were two to three times *worse* than reality on the indices and
metals, so nothing downstream was flattered by them. EURUSD quoting zero is the
signature of a raw-spread account, where the cost sits in commission instead;
a zero spread is refused too, because a model that charges nothing to cross is
not a cost model, and the commission figure needs verifying separately.

None of this touches entries 007-013: those run on `CostModel.for_futures`,
priced from exchange tick sizes and per-side commission, not from these CFDs.

---

## Interactive Brokers: two order-term defaults that would have cost money (2026-09-07)

First contact with a real IB paper account (DUT097699, free trial) found three
bugs in three attempts. The first two were mine; the third is IB's defaults
meeting a strategy that holds overnight.

**1. The test double invented API.** `make_market_order`, `make_stop_order` and
`make_future` existed only on `FakeIB`. Twelve green checks had validated a
fiction. Now the adapter builds orders from `ib_async` itself and
`tests/test_ib_conformance.py` walks its AST to assert every client call exists
on the real class.

**2. No live market data on a free trial.** IB answers `reqMktData` with NaN and
reports error 354 asynchronously, so the failure arrives as arithmetic, not an
exception, and NaN propagated into the stop distance. The adapter now walks the
entitlement ladder by value (live, delayed, delayed-frozen) and records which
answered, so nothing measures execution quality from a 15-minute-old price.

**3. Error 10349: TIF set to DAY, both orders cancelled.** `ib_async` leaves
`tif` empty and `outsideRth` False; IB's preset filled in DAY. Two faults:

- *Visible:* at 04:35 New York every order is refused, because IB's "regular
  trading hours" are 09:30-16:15 while MES trades nearly around the clock.
- *Invisible, and the serious one:* **a protective stop with DAY time-in-force
  is cancelled at the session close.** Every strategy here holds overnight and
  the monthly rules hold for a month. The position would wake up unprotected
  while the risk engine still believed a stop was attached, and every aggregate
  risk figure computed from stops would be wrong. `UnstoppedPosition` would not
  catch it, because the position *did* have a stop when it was last checked.

Entries are now DAY + outsideRth; protective stops are **GTC + outsideRth**, and
`tests/test_ib_order_terms.py` asserts it at the factory, through the full submit
path, and after a stop modification.

Note this is specific to IB's separate child-stop model. On MT5 the stop is a
field on the position itself and cannot expire, which is why the CFD side never
showed this.

**4. The roll carried the stop across at its face value.** Measured live: a long
in the September S&P micro at 7722.25 with its stop at 7714.75, seven points
away, rolled into December at 7785 and kept a stop of 7714.75 - seventy points
away, nine times the intended risk. Two delivery months of the same future are
not the same price; on that day the carry between them was sixty-three points.
A short would have been worse: its stop would have landed on the far side of the
new contract and liquidated the position the moment it was placed. `roll()` now
measures the basis while both months still quote, shifts the stop by it, and
preserves the risk distance. Backwardation moves it the other way; the sign
follows the curve.

**5. The roll judged its close leg after a quarter of a second.** `submit()` and
`close()` both wait `fill_timeout` for `isDone()`; `roll()` slept 0.25s and
decided, so a fill arriving a moment later was reported REJECTED - which is why
the roll showed FAIL on 2026-09-07 after it had actually worked. Worse, the code
then reopened in the front month regardless, which on a genuinely unfilled close
would leave a doubled position in the month being abandoned. All three paths now
share `_await_fill`, and a close that does not confirm stops the roll instead of
compounding it.

Warning 2109 ("Outside Regular Trading Hours is ignored based on the order type
and destination") is noise: IB parks the trade in `ValidationError` while it
processes the warning, then fills. `isDone()` correctly waits through it.

### Futures gate MET, 2026-09-07 08:50 UTC

Full round trip on IB paper DUT097699, contract MESU6 -> MESZ6:

| step | result |
|---|---|
| front contract from the exchange | MESU6, last trade 2026-09-18 - matches the `third_friday` rule |
| spec from the exchange | tick 0.25 x 5.0 = $1.25/tick - matches the hard-coded spec |
| open with child stop | ticket 36 @ 7722.25, stop 7712.25, slippage 0.0 |
| stop modification | 7712.25 -> 7714.75, confirmed at the broker |
| **roll to the next contract** | **stop 7714.75 -> 7782.5, basis +67.75** |
| close | @ 7789.5 |
| flat | no position |

The roll is the number that matters. Risk before the roll was 7.50 points;
after it, 7.00 - the half point being the market moving between the roll and the
close. On the previous run, with the same trade, it was 70.25 points: the same
position carrying **9.4x the intended risk**, silently.

Six bugs in four attempts, none of them findable by backtesting, and the first
concealed by our own test double. That is the argument for connecting to the
real venue early, and the argument against hand-rolling an adapter when a
maintained one exists.

Note for later: this account has no live data entitlement, so all of the above
ran on a 10-15 minute delayed feed. Fills on a paper account are simulated
regardless, so no slippage figure here is real - see the demo-fill section above.
