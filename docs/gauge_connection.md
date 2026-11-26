# Electronic gauge connection

`tope.quantum.gauge_connection` aligns electronic representations before
step-to-step prediction, compression, replay comparison, or residual coding.
It handles three distinct redundancies:

- orbital phases, swaps, and rotations by unitary Procrustes transport;
- the global many-body phase;
- MPS virtual-bond gauges by a left-canonical QR sweep followed by bond-wise
  Procrustes transport.

```python
from tope.quantum import (
    GaugeDiagnostics,
    GaugeTriggerPolicy,
    align_orbitals,
    holonomy_deviation,
)

alignment = align_orbitals(orbitals_t, orbitals_next, metric=ao_overlap)
orbitals_next = alignment.aligned

diagnostics = GaugeDiagnostics.from_singular_values(
    alignment.singular_values,
    predictive_error=force_error,
    truncation_weight=discarded_schmidt_weight,
)
policy = GaugeTriggerPolicy()
if policy.should_check_curvature(diagnostics.sigma_min, force_error):
    # Build transports around a small closed nuclear-coordinate plaquette.
    diagnostics = GaugeDiagnostics(
        diagnostics.sigma_min,
        diagnostics.subspace_leakage,
        curvature=float(holonomy_deviation(loop_transports, plaquette_area)),
        predictive_error=force_error,
        truncation_weight=discarded_schmidt_weight,
    )
action = policy.decide(diagnostics)
```

The overlap SVD is the cheap first-stage diagnostic. Curvature is evaluated only
after overlap loss or predictive error crosses its gate, matching ToPE's
cost-aware execution design. Tier 1 realigns and locally refines; Tier 2 can grow
the affected MPS bond when energy/force error or discarded Schmidt weight
corroborates the gauge warning; Tier 3 requests a reference calculation, smaller
time step, and replay update.

`holonomy_deviation` reports `||H-I||_F / area`, a finite-plaquette curvature
proxy rather than a continuum Berry-curvature tensor. It is invariant to a
common conjugating gauge at the loop base point. For replay states reached along
different paths, realign their frames directly before comparing raw orbitals or
MPS tensors.

The implementation does not infer physical accuracy from compression quality.
Validate decisions against reference energies, forces, nonadiabatic couplings,
and controlled Hamiltonian switching. Orbital singular values indicate
single-particle subspace continuity, not many-body entanglement; this is why bond
growth requires corroborating evidence by default.
