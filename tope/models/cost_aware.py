"""Cost-aware event gating for optional ToPE computations.

The controller is deliberately independent of wall-clock timing.  Callers pass
work estimates in one consistent unit (FLOPs, milliseconds, joules, or dollars)
so that a policy can decide whether its own routing overhead is amortized.  It
does not silently approximate a required computation: a cold cache, invalidated
version, forced audit, or uneconomic gate always requests the full path.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable, Optional


@dataclass(frozen=True)
class WorkEstimate:
    """Estimated work for one optional computation in a common cost unit.

    ``avoided_work`` is the work saved on a skipped call.  ``expensive_work`` is
    the full-path work paid on a triggered call.  The remaining fields are paid
    on every call that uses the gate.
    """

    avoided_work: float
    expensive_work: float
    proxy_work: float = 0.0
    routing_work: float = 0.0
    maintenance_work: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.avoided_work,
            self.expensive_work,
            self.proxy_work,
            self.routing_work,
            self.maintenance_work,
        )
        if any(value < 0 for value in values):
            raise ValueError("work estimates must be non-negative")

    @property
    def fixed_overhead(self) -> float:
        return self.proxy_work + self.routing_work + self.maintenance_work

    def projected_savings(self, trigger_probability: float) -> float:
        """Return savings after gate overhead and expected triggered work."""
        if not 0.0 <= trigger_probability <= 1.0:
            raise ValueError("trigger_probability must lie in [0, 1]")
        return (
            self.avoided_work
            - self.fixed_overhead
            - trigger_probability * self.expensive_work
        )


@dataclass(frozen=True)
class CostAwareConfig:
    """Policy knobs for a stateful, hysteretic event gate."""

    trigger_threshold: float
    release_threshold: float
    expected_trigger_probability: float = 0.1
    minimum_savings: float = 0.0
    audit_interval: int = 100
    minimum_run_interval: int = 0
    cache_entries: int = 32

    def __post_init__(self) -> None:
        if self.release_threshold > self.trigger_threshold:
            raise ValueError("release_threshold cannot exceed trigger_threshold")
        if not 0.0 <= self.expected_trigger_probability <= 1.0:
            raise ValueError("expected_trigger_probability must lie in [0, 1]")
        if self.minimum_savings < 0:
            raise ValueError("minimum_savings must be non-negative")
        if self.audit_interval <= 0:
            raise ValueError("audit_interval must be positive")
        if self.minimum_run_interval < 0:
            raise ValueError("minimum_run_interval must be non-negative")
        if self.cache_entries < 0:
            raise ValueError("cache_entries must be non-negative")


@dataclass(frozen=True)
class GateDecision:
    """Auditable result of one routing decision."""

    run: bool
    reason: str
    projected_savings: float
    trigger_probability: float
    cached: bool = False


class CostAwareGate:
    """Decide when an optional expensive path should run.

    The observed trigger rate replaces the configured prior after the first
    decision.  Hysteresis prevents threshold chatter, ``audit_interval`` bounds
    stale reuse, and ``version`` invalidates cached structures after topology or
    model changes.  Cache keys are local to this gate and eviction is LRU.
    """

    def __init__(self, config: CostAwareConfig):
        self.config = config
        self._step = 0
        self._runs = 0
        self._last_run = 0
        self._active = False
        self._cache: OrderedDict[Hashable, tuple[Hashable, Any]] = OrderedDict()

    @property
    def trigger_probability(self) -> float:
        if self._step == 0:
            return self.config.expected_trigger_probability
        return self._runs / self._step

    def clear(self) -> None:
        """Drop reusable values without resetting cost observations."""
        self._cache.clear()

    def get_cached(self, key: Hashable, version: Hashable) -> Any:
        entry = self._cache.get(key)
        if entry is None or entry[0] != version:
            return None
        self._cache.move_to_end(key)
        return entry[1]

    def store(self, key: Hashable, version: Hashable, value: Any) -> None:
        if self.config.cache_entries == 0:
            return
        self._cache[key] = (version, value)
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.cache_entries:
            self._cache.popitem(last=False)

    def decide(
        self,
        score: float,
        estimate: WorkEstimate,
        *,
        cache_key: Optional[Hashable] = None,
        version: Hashable = 0,
        changed: bool = False,
        force: bool = False,
    ) -> GateDecision:
        """Return whether the caller must execute the expensive path.

        ``score`` is a cheap surprise/error proxy.  A skipped decision is only
        returned when a valid cached result exists; otherwise the gate fails
        closed and requests computation.
        """
        if score != score or score in (float("inf"), float("-inf")):
            raise ValueError("score must be finite")

        # Price the decision from history *before* adding the current event.
        # This preserves the configured prior on the first call.
        probability = self.trigger_probability
        self._step += 1
        cached = (
            cache_key is not None
            and self.get_cached(cache_key, version) is not None
            and not changed
        )
        savings = estimate.projected_savings(probability)

        reason: Optional[str] = None
        if force:
            reason = "forced"
        elif changed:
            reason = "changed"
        elif not cached:
            reason = "cold_cache"
        elif savings < self.config.minimum_savings:
            reason = "gate_not_economic"
        elif self._step - self._last_run >= self.config.audit_interval:
            reason = "periodic_audit"
        elif (
            self._runs > 0
            and self._step - self._last_run < self.config.minimum_run_interval
        ):
            reason = None
        else:
            threshold = (
                self.config.release_threshold
                if self._active
                else self.config.trigger_threshold
            )
            if score >= threshold:
                reason = "proxy_trigger"

        run = reason is not None
        if run:
            self._runs += 1
            self._last_run = self._step
            self._active = True
        elif score < self.config.release_threshold:
            self._active = False

        return GateDecision(
            run=run,
            reason=reason or "reuse_cache",
            projected_savings=savings,
            trigger_probability=probability,
            cached=cached and not run,
        )

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "decisions": self._step,
            "runs": self._runs,
            "skips": self._step - self._runs,
            "trigger_probability": self.trigger_probability,
            "cache_entries": len(self._cache),
        }
