"""Cost accounting and safety checks for event-driven optional work."""
import pytest

from tope.models.cost_aware import CostAwareConfig, CostAwareGate, WorkEstimate


def _gate(**updates):
    values = dict(trigger_threshold=0.7, release_threshold=0.3,
                  expected_trigger_probability=0.1, audit_interval=5)
    values.update(updates)
    return CostAwareGate(CostAwareConfig(**values))


def test_work_estimate_accounts_for_overhead_and_triggered_work():
    work = WorkEstimate(avoided_work=100, expensive_work=80,
                        proxy_work=3, routing_work=2, maintenance_work=1)
    assert work.projected_savings(0.25) == 74
    with pytest.raises(ValueError, match="non-negative"):
        WorkEstimate(-1, 1)


def test_cold_cache_change_and_uneconomic_gate_fail_closed():
    work = WorkEstimate(100, 20, proxy_work=1)
    gate = _gate()
    first = gate.decide(0.0, work, cache_key="protein", version=1)
    assert first.run and first.reason == "cold_cache"
    gate.store("protein", 1, "fingerprint")
    assert not gate.decide(0.0, work, cache_key="protein", version=1).run
    changed = gate.decide(0.0, work, cache_key="protein", version=2)
    assert changed.run and changed.reason == "cold_cache"

    costly = _gate(expected_trigger_probability=1.0)
    costly.store("p", 1, object())
    decision = costly.decide(0.0, WorkEstimate(10, 20), cache_key="p", version=1)
    assert decision.run and decision.reason == "gate_not_economic"


def test_hysteresis_interval_and_periodic_audit_bound_recomputation():
    work = WorkEstimate(100, 10)
    gate = _gate(audit_interval=4, minimum_run_interval=2)
    gate.store("p", 1, object())
    assert gate.decide(0.8, work, cache_key="p", version=1).reason == "proxy_trigger"
    # Minimum interval suppresses immediate retriggering despite a high score.
    assert not gate.decide(0.9, work, cache_key="p", version=1).run
    # Active hysteresis uses the lower release threshold after the interval.
    assert gate.decide(0.5, work, cache_key="p", version=1).run
    # Low score releases the gate, then the periodic audit eventually runs.
    assert not gate.decide(0.0, work, cache_key="p", version=1).run
    assert not gate.decide(0.0, work, cache_key="p", version=1).run
    assert not gate.decide(0.0, work, cache_key="p", version=1).run
    assert gate.decide(0.0, work, cache_key="p", version=1).reason == "periodic_audit"


def test_lru_cache_is_bounded_and_reports_diagnostics():
    gate = _gate(cache_entries=2)
    gate.store("a", 1, 1)
    gate.store("b", 1, 2)
    assert gate.get_cached("a", 1) == 1
    gate.store("c", 1, 3)
    assert gate.get_cached("b", 1) is None
    assert gate.diagnostics()["cache_entries"] == 2
