"""Shared observed-only cadence decisions for live and offline TCN input."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CadenceDecision:
    action: str
    dt_sec: float | None
    append: bool
    reset: bool


def decide_observation(timestamp: float, last_timestamp: float | None, *, min_interval_sec: float = 0.070, max_interval_sec: float = 0.150) -> CadenceDecision:
    """Classify one real pose observation without manufacturing missing rows."""
    if not 0.0 < min_interval_sec <= max_interval_sec:
        raise ValueError("sampling interval must satisfy 0 < min <= max")
    if last_timestamp is None:
        return CadenceDecision("append", None, True, False)
    dt = float(timestamp) - float(last_timestamp)
    if dt <= 0.0:
        return CadenceDecision("non_monotonic_skip", dt, False, False)
    if dt < min_interval_sec:
        return CadenceDecision("duplicate_skip", dt, False, False)
    if dt > max_interval_sec:
        return CadenceDecision("gap_reset_append", dt, True, True)
    return CadenceDecision("append", dt, True, False)
