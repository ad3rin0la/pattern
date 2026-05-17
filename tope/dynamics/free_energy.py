"""Free-energy predictors (ΔG_unfold, ΔG‡) — composable stubs.

Wires the API end-to-end with sensible defaults so downstream
consumers (the engineering engine, the thermal composite scorer) can
be developed against a stable interface while the supervised heads —
PSL B-factor predictor, GFN2-xTB-anchored Ea head, folding
cooperativity head — land independently.

Two callables are exported with the exact signatures expected by
``tope.engineering.thermal``:

* ``predict_delta_g_unfold(active_site, T) -> float``   (kcal/mol)
* ``predict_delta_g_dagger(active_site, T) -> float``   (kcal/mol)

The default ``predict_delta_g_unfold`` composes existing primitives:

    ΔG_unfold = ΔH_contacts − T · ΔS_config

where ``ΔH_contacts`` is the integrated SheafENM-equivalent contact
energy (number of Cα contacts × γ × characteristic energy per contact)
and ``ΔS_config`` is the harmonic configurational entropy
``Σ_k (1/2) k_B (1 + ln(k_B T / ω_k²))`` summed over the NMA spectrum.
This is the Schlitter-style upper bound, not the true entropy — it's a
defensible scaffold value, not a load-bearing prediction.

The default ``predict_delta_g_dagger`` is a flat constant returning the
``Ea_kcal_per_mol`` knob; users replace it with a trained head later.

Class-form stubs (``DeltaGUnfoldStub``, ``DeltaGDaggerStub``) wrap the
functional API for users who want config-attached callables, matching
the pattern of ``ThermalCompositeScorer``'s sub-scorers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from tope.data.active_site import ActiveSite
from tope.dynamics.nma import NMAConfig, NormalModeAnalysis, KB_KCAL


@dataclass
class FreeEnergyConfig:
    """Knobs shared by the default ΔG predictors."""

    # ΔH_contacts: characteristic enthalpic cost per native contact (kcal/mol).
    # ~1–2 kcal/mol is the rough magnitude for a single nonbonded contact.
    contact_enthalpy_kcal: float = 1.5

    # ΔS_config: Schlitter prefactor (1/2 in the harmonic formula). Kept
    # as a knob so callers can ablate the entropy term.
    schlitter_prefactor: float = 0.5

    # Unfolded-state entropy baseline, kcal/(mol·K) per residue.
    # Picked to put ΔS_unfold = S_unf − S_fold on the right side of zero
    # for typical compact folds; ~3 cal/(mol·K) per residue is in the
    # ballpark of residue conformational entropy in the random-coil
    # limit. Treated as a constant baseline rather than a prediction.
    s_unfolded_per_residue_kcal_per_K: float = 0.003

    # NMA parameters used internally.
    nma_cfg: NMAConfig = field(default_factory=NMAConfig)

    # ΔG‡: stub-constant activation energy (kcal/mol) until a trained
    # head lands.
    Ea_kcal_per_mol: float = 15.0


# ── ΔG_unfold ─────────────────────────────────────────────────────────────────

def predict_delta_g_unfold(
    active_site: ActiveSite,
    T: float = 298.15,
    cfg: Optional[FreeEnergyConfig] = None,
) -> float:
    """Stub ΔG_unfold (kcal/mol). Composition of existing primitives.

    Returns a positive number for a folded-state-favoured equilibrium.
    Not a trained prediction — use as a defensible default until the
    PSL B-factor predictor and a supervised ΔG_unfold head land.
    """
    cfg = cfg or FreeEnergyConfig()
    coords = active_site.ca_coords_array()
    if coords.shape[0] < 4:
        # Too few residues to define a meaningful ENM; return a flat
        # default that keeps f_folded above 0.5 at room temperature.
        return 5.0

    # ── ΔH_contacts: count Cα contacts within the ANM cutoff ────────────
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    n_contacts = int(((d2 <= cfg.nma_cfg.contact_cutoff ** 2)).sum() // 2)
    dH = cfg.contact_enthalpy_kcal * n_contacts

    # ── ΔS_config: Schlitter harmonic upper bound from NMA spectrum ─────
    nma = NormalModeAnalysis(coords, cfg=cfg.nma_cfg)
    omega, _ = nma.nontrivial_modes()
    omega2 = np.maximum(omega ** 2, cfg.nma_cfg.eigenvalue_floor)
    # S_harmonic ≈ k_B · α · Σ_k (1 + ln(k_B T / ω_k²)) ; truncate negatives.
    log_term = np.log(np.maximum(KB_KCAL * T / omega2, 1e-30))
    s_folded = KB_KCAL * cfg.schlitter_prefactor * (1.0 + log_term).sum()
    s_unfolded = cfg.s_unfolded_per_residue_kcal_per_K * len(active_site.residues)
    dS = s_unfolded - s_folded   # ΔS_unfold = S_unf − S_fold (should be > 0)

    return float(dH - T * dS)


# ── ΔG‡ ───────────────────────────────────────────────────────────────────────

def predict_delta_g_dagger(
    active_site: ActiveSite,
    T: float = 298.15,
    cfg: Optional[FreeEnergyConfig] = None,
) -> float:
    """Stub activation free energy ΔG‡ (kcal/mol).

    Currently a flat constant — replace with a wrapper around
    ``LearnablePLaplacianToPE.predict_arrhenius_slope`` (or a future
    GFN2-xTB-anchored head) when a supervised model is available.
    """
    cfg = cfg or FreeEnergyConfig()
    return float(cfg.Ea_kcal_per_mol)


# ── Class wrappers for ThermalCompositeScorer plug-in ─────────────────────────

@dataclass
class DeltaGUnfoldStub:
    """Callable wrapper: drop-in for ``BoltzmannFoldedFraction.delta_g_unfold``."""

    cfg: FreeEnergyConfig = field(default_factory=FreeEnergyConfig)

    def __call__(self, active_site: ActiveSite, T: float) -> float:
        return predict_delta_g_unfold(active_site, T, self.cfg)


@dataclass
class DeltaGDaggerStub:
    """Callable wrapper: drop-in for ``ArrheniusEaScorer``."""

    cfg: FreeEnergyConfig = field(default_factory=FreeEnergyConfig)

    def __call__(self, active_site: ActiveSite, T: float) -> float:
        return predict_delta_g_dagger(active_site, T, self.cfg)
