# Cost-aware execution

ToPE's sparse incidences, hierarchical ranks, local candidate sets and compact
representations reduce the amount of work. `CostAwareGate` handles the separate
question of whether an optional expensive update is worth invoking at all.

```python
from tope.models.cost_aware import CostAwareConfig, CostAwareGate, WorkEstimate

gate = CostAwareGate(CostAwareConfig(
    trigger_threshold=0.7,
    release_threshold=0.3,
    expected_trigger_probability=0.1,
    audit_interval=100,
))
work = WorkEstimate(
    avoided_work=120.0,
    expensive_work=100.0,
    proxy_work=2.0,
    routing_work=1.0,
    maintenance_work=1.0,
)

decision = gate.decide(
    cheap_surprise_score,
    work,
    cache_key=protein_id,
    version=(topology_version, checkpoint_id),
    changed=local_topology_changed,
)
if decision.run:
    fingerprint = build_electronic_fingerprint(inputs)
    gate.store(protein_id, (topology_version, checkpoint_id), fingerprint)
else:
    fingerprint = gate.get_cached(
        protein_id, (topology_version, checkpoint_id)
    )
```

All `WorkEstimate` fields must use one unit, such as measured milliseconds or
profiled FLOPs. The policy runs the full path on a cold cache, version change,
explicit change event, forced call, periodic audit, proxy threshold crossing, or
when projected savings do not cover gating overhead. This fail-closed behavior
prevents the controller from silently substituting a missing or stale result.

Hysteresis uses a high trigger threshold and a lower release threshold to avoid
chatter. `minimum_run_interval` suppresses repeated heavy work, while
`audit_interval` bounds how long a cached result can survive without a full
check. The cache is a bounded LRU. Use topology/data/checkpoint identities in
the version token and call `clear()` when external state cannot be versioned.

Caching tensors that retain an autograd graph is unsafe across optimization
steps. Cache detached inference products or immutable structural metadata; for
training, use the decision only to schedule recomputation and keep the current
step's differentiable result in the caller.
