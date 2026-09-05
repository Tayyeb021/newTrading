"""Configuration loading.

Nothing operational is hardcoded anywhere in this codebase: not a lot size, not a
symbol name, not a limit. All of it lives in `config/*.yaml` and arrives here.
That is what makes `challenge` and `funded` two config files rather than two code
paths, and it is what lets you change a limit without a deploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level")
    return data


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"{ctx}: missing required key {key!r}")
    return d[key]


@dataclass(frozen=True)
class RiskProfile:
    """A complete risk configuration. One of these is active at a time."""

    name: str
    risk_per_trade: float
    max_risk_per_trade: float
    daily_loss_soft: float
    daily_loss_hard: float
    max_drawdown_soft: float
    max_drawdown_hard: float
    drawdown_trailing: bool
    max_concurrent_positions: int
    max_bucket_risk: float
    consecutive_losses: int
    consecutive_loss_pause_hours: float
    min_margin_level: float
    max_spread_multiple: float
    max_feed_age_seconds: float
    atr_period: int
    atr_stop_multiple: float
    buckets: dict[str, list[str]]
    #: Total open risk the whole book may carry, across every sleeve. The
    #: allocator splits this between sleeves by weight. Defaulted, so it
    #: must sit after every required field.
    max_open_risk: float = 0.02

    def __post_init__(self) -> None:
        if self.risk_per_trade > self.max_risk_per_trade:
            raise ConfigError(
                f"{self.name}: risk_per_trade {self.risk_per_trade} exceeds "
                f"max_risk_per_trade {self.max_risk_per_trade}"
            )
        # The soft/hard gap is the entire point of the two-tier design. A profile
        # where they are equal offers no buffer, and on an evaluation account that
        # means one bad fill ends the attempt.
        for soft, hard, label in (
            (self.daily_loss_soft, self.daily_loss_hard, "daily_loss"),
            (self.max_drawdown_soft, self.max_drawdown_hard, "max_drawdown"),
        ):
            if not 0 < soft < hard < 1:
                raise ConfigError(
                    f"{self.name}: require 0 < {label}_soft ({soft}) < "
                    f"{label}_hard ({hard}) < 1"
                )
        if self.max_open_risk < self.max_risk_per_trade:
            raise ConfigError(
                f"{self.name}: max_open_risk {self.max_open_risk:.2%} is below "
                f"max_risk_per_trade {self.max_risk_per_trade:.2%} - no trade could ever open"
            )
        if self.daily_loss_soft <= self.max_risk_per_trade:
            raise ConfigError(
                f"{self.name}: daily_loss_soft {self.daily_loss_soft:.2%} is not larger "
                f"than max_risk_per_trade {self.max_risk_per_trade:.2%} - a single "
                f"losing trade would halt the day"
            )

    @classmethod
    def load(cls, name: str) -> "RiskProfile":
        raw = load_yaml(f"risk.{name}.yaml")
        ctx = f"risk.{name}.yaml"
        sizing = _require(raw, "sizing", ctx)
        limits = _require(raw, "limits", ctx)
        return cls(
            name=raw.get("profile", name),
            risk_per_trade=float(_require(sizing, "risk_per_trade", ctx)),
            max_risk_per_trade=float(_require(sizing, "max_risk_per_trade", ctx)),
            atr_period=int(sizing.get("atr_period", 14)),
            atr_stop_multiple=float(sizing.get("atr_stop_multiple", 2.5)),
            daily_loss_soft=float(_require(limits, "daily_loss_soft", ctx)),
            daily_loss_hard=float(_require(limits, "daily_loss_hard", ctx)),
            max_drawdown_soft=float(_require(limits, "max_drawdown_soft", ctx)),
            max_drawdown_hard=float(_require(limits, "max_drawdown_hard", ctx)),
            drawdown_trailing=bool(limits.get("drawdown_trailing", False)),
            max_concurrent_positions=int(limits.get("max_concurrent_positions", 4)),
            max_bucket_risk=float(limits.get("max_bucket_risk", 0.01)),
            consecutive_losses=int(limits.get("consecutive_losses", 4)),
            consecutive_loss_pause_hours=float(limits.get("consecutive_loss_pause_hours", 24.0)),
            max_open_risk=float(limits.get("max_open_risk", 0.02)),
            min_margin_level=float(limits.get("min_margin_level", 3.0)),
            max_spread_multiple=float(limits.get("max_spread_multiple", 2.0)),
            max_feed_age_seconds=float(limits.get("max_feed_age_seconds", 10.0)),
            buckets={k: list(v) for k, v in (raw.get("buckets") or {}).items()},
        )


@dataclass(frozen=True)
class InstrumentConfig:
    symbols: list[str]
    aliases: dict[str, str]
    active: list[str] = field(default_factory=list)  # what the live/shadow runner trades; defaults to all

    def __post_init__(self) -> None:
        if not self.active:
            object.__setattr__(self, "active", list(self.symbols))
        unknown = [s for s in self.active if s not in self.symbols]
        if unknown:
            raise ValueError(f"active symbols not in symbols: {unknown}")

    @classmethod
    def load(cls, path: str = "instruments.yaml") -> "InstrumentConfig":
        raw = load_yaml(path)
        return cls(
            symbols=list(raw.get("symbols") or []),
            aliases={str(k): str(v) for k, v in (raw.get("aliases") or {}).items()},
            active=list(raw.get("active") or []),
        )

    def resolve(self, symbol: str) -> str:
        """Map a canonical name to whatever this broker calls it.

        Brokers rename things - 'US30' here, 'US30.cash' or 'DJ30' there. Strategies
        use the canonical name; only the adapter sees the broker's.
        """
        return self.aliases.get(symbol, symbol)
