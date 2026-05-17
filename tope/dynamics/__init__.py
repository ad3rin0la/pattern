"""
Topological dynamics scaffolding.

Two pieces live here:

* ``nma`` — Anisotropic-ENM normal-mode analysis. Stateless,
  deterministic, numpy/scipy. Gives mode-projected dynamics and a
  harmonic ensemble sampler without committing to an integrator.

* ``free_energy`` — Free-energy predictors (ΔG_unfold, ΔG‡) composed
  from existing ToPE primitives where possible and from explicit
  stubs where not. Built to drop straight into
  ``ThermalCompositeScorer`` as ``delta_g_unfold`` / ``ea`` callables.

This module deliberately does *not* implement a trajectory integrator
on top of the sheaf Laplacian — see the design note in the engineering
discussion: cochain amplitudes vs. atomic positions need disambiguation
before that gets written.
"""

from tope.dynamics.nma import (
    NMAConfig,
    NormalModeAnalysis,
    build_anm_hessian,
    propagate_mode,
    sample_harmonic_ensemble,
    thermal_msf,
)
from tope.dynamics.free_energy import (
    FreeEnergyConfig,
    predict_delta_g_unfold,
    predict_delta_g_dagger,
    DeltaGUnfoldStub,
    DeltaGDaggerStub,
)

__all__ = [
    "NMAConfig",
    "NormalModeAnalysis",
    "build_anm_hessian",
    "propagate_mode",
    "sample_harmonic_ensemble",
    "thermal_msf",
    "FreeEnergyConfig",
    "predict_delta_g_unfold",
    "predict_delta_g_dagger",
    "DeltaGUnfoldStub",
    "DeltaGDaggerStub",
]
