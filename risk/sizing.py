"""Position sizing.

One rule governs this whole module: **the computed size may never imply more risk
than the caller asked for.** Rounding goes down, never to nearest. Every result is
re-checked against the limit after rounding, because a limit you can round past is
not a limit.

The `BELOW_MINIMUM` outcome is not an error condition to be worked around. It is
the correct, load-bearing answer when the account is too small to express the
requested risk in the instrument's minimum lot - the situation that makes daily
gold untradeable on a small account. Sizing up to reach the minimum would silently
multiply risk by four or five, which is how small accounts die.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.types import SymbolSpec


class SizingOutcome(Enum):
    OK = "ok"
    BELOW_MINIMUM = "below_minimum"
    CAPPED_BY_MAXIMUM = "capped_by_maximum"
    CAPPED_BY_MARGIN = "capped_by_margin"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class SizeResult:
    volume: float
    outcome: SizingOutcome
    risk_cash: float  # actual cash at risk after rounding
    risk_fraction: float  # actual risk as a fraction of equity
    reason: str = ""

    @property
    def tradeable(self) -> bool:
        return self.volume > 0 and self.outcome in (
            SizingOutcome.OK,
            SizingOutcome.CAPPED_BY_MAXIMUM,
            SizingOutcome.CAPPED_BY_MARGIN,
        )


def size_position(
    spec: SymbolSpec,
    equity: float,
    risk_fraction: float,
    stop_distance: float,
    *,
    regime_scalar: float = 1.0,
    confidence: float = 1.0,
    max_volume: float | None = None,
) -> SizeResult:
    """Volatility-scaled position size.

        risk_cash = equity * risk_fraction * regime_scalar * confidence
        volume    = risk_cash / (stop_distance * value_per_price_unit)

    `regime_scalar` comes from the regime model (tier 1); `confidence` from the
    meta-label model (tier 3). Both default to 1.0 so the function is fully usable
    before any machine learning exists — which is the point of the ordering in the
    build plan.
    """
    if equity <= 0:
        return SizeResult(0.0, SizingOutcome.INVALID, 0.0, 0.0, "equity is not positive")
    if stop_distance <= 0:
        return SizeResult(0.0, SizingOutcome.INVALID, 0.0, 0.0, "stop distance is not positive")
    if not 0.0 < risk_fraction < 1.0:
        return SizeResult(0.0, SizingOutcome.INVALID, 0.0, 0.0, f"risk_fraction {risk_fraction} out of range")

    budget = equity * risk_fraction * max(regime_scalar, 0.0) * max(confidence, 0.0)
    if budget <= 0:
        return SizeResult(0.0, SizingOutcome.INVALID, 0.0, 0.0, "risk budget is zero")

    risk_per_lot = stop_distance * spec.value_per_price_unit
    raw = budget / risk_per_lot

    volume = spec.round_volume(raw)
    outcome = SizingOutcome.OK
    reason = ""

    if volume < spec.volume_min:
        needed = spec.volume_min * risk_per_lot
        return SizeResult(
            0.0,
            SizingOutcome.BELOW_MINIMUM,
            0.0,
            0.0,
            f"minimum lot {spec.volume_min:g} risks {needed:,.2f} "
            f"({needed / equity:.2%} of equity) against a budget of {budget:,.2f} "
            f"({budget / equity:.2%}) - account too small for this stop distance",
        )

    if volume > spec.volume_max:
        volume = spec.round_volume(spec.volume_max)
        outcome = SizingOutcome.CAPPED_BY_MAXIMUM
        reason = f"capped at symbol maximum {spec.volume_max:g}"

    if max_volume is not None and volume > max_volume:
        capped = spec.round_volume(max_volume)
        if capped < spec.volume_min:
            return SizeResult(
                0.0,
                SizingOutcome.BELOW_MINIMUM,
                0.0,
                0.0,
                f"margin cap {max_volume:g} is below minimum lot {spec.volume_min:g}",
            )
        volume = capped
        outcome = SizingOutcome.CAPPED_BY_MARGIN
        reason = f"capped by margin at {max_volume:g}"

    actual_risk = spec.risk_for(volume, stop_distance)

    # The invariant. Rounding down should make this impossible; assert it anyway,
    # because the one time it is violated is the one time it matters.
    if actual_risk > budget + 1e-6:
        raise AssertionError(
            f"{spec.symbol}: sized risk {actual_risk:.4f} exceeds budget {budget:.4f} "
            f"- rounding is broken"
        )

    return SizeResult(
        volume=volume,
        outcome=outcome,
        risk_cash=actual_risk,
        risk_fraction=actual_risk / equity,
        reason=reason,
    )


def minimum_viable_equity(
    spec: SymbolSpec,
    risk_fraction: float,
    stop_distance: float,
) -> float:
    """Smallest equity that can trade this instrument at this risk and stop.

    The answer to "why can I not trade gold on a small account?", in one number.
    Run it across your instrument set before choosing what v1 trades.
    """
    return (spec.volume_min * stop_distance * spec.value_per_price_unit) / risk_fraction
