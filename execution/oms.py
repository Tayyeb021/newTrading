"""Order management: idempotency, retry, reconciliation.

The problem this solves is narrow and nasty. You call `order_send`. The network
drops. You get a timeout. **Did the order fill?** You cannot know from the return
value, and the two wrong answers are both expensive: assume it failed and retry,
and you may end up with two positions; assume it filled and you may have none.

The fix is a deterministic client id derived from what the order *is* — strategy,
symbol, side, and the bar that produced it. It goes in the broker's comment field.
After any ambiguous failure, look for a position carrying that id before doing
anything else. Same intent, same id, so a retry can recognise its own earlier
attempt.

Retries are classified, not blanket. A requote is worth retrying; "invalid stops"
is not, and retrying it just fills the broker's log with rejects that a prop firm
can see.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from core.types import OrderRequest, OrderResult, OrderStatus, Position
from execution.base import ExecutionAdapter, ExecutionError

log = logging.getLogger(__name__)


class Retryable(Enum):
    YES = "yes"  # transient: requote, price change, busy
    NO = "no"  # terminal: bad volume, bad stops, no money
    AMBIGUOUS = "ambiguous"  # timeout or connection loss - state unknown


# Substrings that classify a broker rejection. Matched case-insensitively against
# the reason text, which is where MT5 puts the useful part.
TRANSIENT = ("requote", "price changed", "off quotes", "busy", "timeout", "too many requests")
AMBIGUOUS = ("timeout", "connection", "no reply", "returned none")
TERMINAL = (
    "invalid volume", "invalid stops", "invalid price", "not enough money",
    "market closed", "trade disabled", "below minimum", "unsupported filling",
)


def classify(reason: str) -> Retryable:
    text = (reason or "").lower()
    for needle in TERMINAL:
        if needle in text:
            return Retryable.NO
    for needle in AMBIGUOUS:
        if needle in text:
            return Retryable.AMBIGUOUS
    for needle in TRANSIENT:
        if needle in text:
            return Retryable.YES
    return Retryable.NO  # unknown reasons are not retried; investigate instead


def client_id(strategy: str, symbol: str, side: str, bar_ts: datetime) -> str:
    """Deterministic, short enough for the MT5 comment field (31 chars).

    Two calls describing the same intent produce the same id, which is what lets
    a retry recognise its own earlier attempt.
    """
    seed = f"{strategy}|{symbol}|{side}|{bar_ts.isoformat()}"
    digest = hashlib.sha1(seed.encode()).hexdigest()[:10]
    return f"{strategy[:12]}#{digest}"


@dataclass
class OMSConfig:
    max_attempts: int = 3
    backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0


class OrderManager:
    def __init__(
        self,
        adapter: ExecutionAdapter,
        config: OMSConfig | None = None,
        sleep=time.sleep,
    ) -> None:
        self.adapter = adapter
        self.cfg = config or OMSConfig()
        self._sleep = sleep  # injectable so tests do not actually wait
        self.attempts: list[OrderResult] = []

    # ------------------------------------------------------------------ submit

    def submit(self, request: OrderRequest, cid: str) -> OrderResult:
        """Submit with retry and idempotency. `cid` must be deterministic."""
        request = self._tag(request, cid)

        # If a previous attempt already landed - including one from a process
        # that died before recording it - adopt it instead of duplicating.
        existing = self.find_by_client_id(request.symbol, cid)
        if existing is not None:
            log.info("adopting existing position %s for %s", existing.ticket, cid)
            return OrderResult(
                status=OrderStatus.FILLED, request=request, ticket=existing.ticket,
                fill_price=existing.entry_price, filled_volume=existing.volume,
                reason="already_open",
            )

        delay = self.cfg.backoff_seconds
        last: OrderResult | None = None

        for attempt in range(1, self.cfg.max_attempts + 1):
            try:
                result = self.adapter.submit(request)
            except ExecutionError as exc:
                result = OrderResult(OrderStatus.REJECTED, request, reason=f"timeout: {exc}")

            self.attempts.append(result)
            last = result
            if result.ok:
                return result

            kind = classify(result.reason)
            log.warning("attempt %d/%d for %s failed (%s): %s",
                        attempt, self.cfg.max_attempts, cid, kind.value, result.reason)

            if kind is Retryable.NO:
                return result

            if kind is Retryable.AMBIGUOUS:
                # The order may have filled. Check before retrying, or risk
                # doubling the position - the worst outcome available here.
                self._sleep(delay)
                landed = self.find_by_client_id(request.symbol, cid)
                if landed is not None:
                    return OrderResult(
                        status=OrderStatus.FILLED, request=request, ticket=landed.ticket,
                        fill_price=landed.entry_price, filled_volume=landed.volume,
                        reason="recovered_after_ambiguous_failure",
                    )

            if attempt < self.cfg.max_attempts:
                self._sleep(delay)
                delay *= self.cfg.backoff_multiplier

        return last or OrderResult(OrderStatus.REJECTED, request, reason="no attempts made")

    def find_by_client_id(self, symbol: str, cid: str) -> Position | None:
        try:
            positions = self.adapter.positions(symbol)
        except ExecutionError:
            return None
        for pos in positions:
            if pos.comment and pos.comment.strip() == cid.strip():
                return pos
        return None

    @staticmethod
    def _tag(request: OrderRequest, cid: str) -> OrderRequest:
        return OrderRequest(
            symbol=request.symbol, side=request.side, volume=request.volume,
            order_type=request.order_type, price=request.price,
            stop_loss=request.stop_loss, take_profit=request.take_profit,
            comment=cid[:31], intent=request.intent,
        )

    # ------------------------------------------------------------------- close

    def close_all(self, symbol: str | None = None) -> list[OrderResult]:
        """Close everything, best effort, reporting each outcome.

        Never raises. This runs in the path where something has already gone
        wrong, and an exception here would leave positions open.
        """
        results: list[OrderResult] = []
        try:
            positions = self.adapter.positions(symbol)
        except ExecutionError as exc:
            log.error("cannot list positions to close: %s", exc)
            return results

        for pos in positions:
            for attempt in range(1, self.cfg.max_attempts + 1):
                try:
                    result = self.adapter.close(pos.ticket)
                except ExecutionError as exc:
                    result = OrderResult(
                        OrderStatus.REJECTED,
                        OrderRequest(pos.symbol, pos.side.opposite(), pos.volume),
                        reason=str(exc),
                    )
                results.append(result)
                if result.ok or classify(result.reason) is Retryable.NO:
                    break
                self._sleep(self.cfg.backoff_seconds * attempt)
        return results

    # ------------------------------------------------------- reconcile / repair

    def reconcile(self, expected: dict[str, float]) -> dict[str, tuple[float, float]]:
        from execution.base import reconcile as _reconcile

        return _reconcile(self.adapter, expected)

    def orphans(self, known_tickets: set[int]) -> list[Position]:
        """Positions at the broker that this system does not know about.

        After a crash these are the trades placed by a process that died before
        journalling them. They are real, they carry real risk, and they must be
        adopted or closed deliberately - never ignored.
        """
        try:
            return [p for p in self.adapter.positions() if p.ticket not in known_tickets]
        except ExecutionError:
            return []

    def ensure_stops(self, default_distance: dict[str, float]) -> list[OrderResult]:
        """Attach a stop to any position missing one.

        A position without a stop has undefined risk, which makes every
        aggregate limit silently wrong while it is open. After a crash this is
        the first repair to make, before anything else is allowed to trade.
        """
        out: list[OrderResult] = []
        for pos in self.adapter.positions():
            if pos.stop_loss:
                continue
            distance = default_distance.get(pos.symbol)
            if distance is None:
                log.error("%s has no stop and no default distance configured", pos.symbol)
                continue
            spec = self.adapter.spec(pos.symbol)
            stop = spec.normalize_price(pos.entry_price - distance * pos.side.sign)
            log.warning("attaching missing stop to ticket %s at %s", pos.ticket, stop)
            out.append(self.adapter.modify(pos.ticket, stop_loss=stop))
        return out
