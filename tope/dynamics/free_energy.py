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

    ΔG_unfold = ΔH_contacts − T · ΔS_unfold

where ``ΔH_contacts`` is the integrated SheafENM-equivalent contact
energy (Cα contact count × characteristic energy per contact) and
``ΔS_unfold`` is the **intrinsic SPD log-det ratio**

    ΔS_unfold = k_B · ½ log(det Σ_unfold / det Σ_fold)
              = k_B · ½ log(Π λ(H_fold) / Π λ(H_unfold))

between the folded harmonic covariance Σ_fold = H_fold⁺ and the
unfolded Gaussian-chain covariance Σ_unfold (path-graph Laplacian
pseudoinverse, segment variance σ²). Both live on the SPD manifold;
the sign of ΔS_unfold is determined by Loewner order, not by a
hand-set per-residue baseline.

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
from tope.dynamics.spd import (
    gaussian_chain_covariance,
    gaussian_entropy_change_basis_free,
    gaussian_entropy_change_from_hessians,
    gaussian_entropy_change_shape_and_offset,
    gaussian_entropy_change_shape_invariant,
    path_laplacian,
)


@dataclass
class FreeEnergyConfig:
    """Knobs shared by the default ΔG predictors."""

    # ΔH_contacts: characteristic enthalpic cost per native contact (kcal/mol).
    # ~1–2 kcal/mol is the rough magnitude for a single nonbonded contact.
    contact_enthalpy_kcal: float = 1.5

    # Gaussian-chain segment variance (Å²) used to build Σ_unfold via the
    # path-graph Laplacian. ~13 Å² is the textbook ideal-chain value.
    # Replaces the previous hand-tuned per-residue entropy baseline:
    # ΔS now comes from log(det Σ_unf / det Σ_fold), an intrinsic
    # quantity on the SPD manifold.
    segment_var_A2: float = 13.0

    # NMA parameters used internally.
    nma_cfg: NMAConfig = field(default_factory=NMAConfig)

    # How to compute the log-det ratio for ΔS_unfold:
    #   "sorted"     — sorted-paired eigenvalue heuristic. Default.
    #                  Σ²-sensitive (note #1) but has a guaranteed sign
    #                  from Loewner order. ΔS_shape under sorted is
    #                  *identically zero* (the whole signal is offset).
    #   "basis_free" — project both Hessians onto the folded non-rigid
    #                  basis and take log det there. Captures basis
    #                  coupling that sorted misses; still σ²-sensitive.
    #   "shape_only" — geometric-mean-normalised basis-free log-det.
    #                  Invariant under uniform γ/σ² rescaling, but the
    #                  *sign* of the result depends on subspace alignment
    #                  (whether folded modes hit stiff or floppy unfolded
    #                  modes). Do NOT use as ΔG_unfold input until the
    #                  downstream head can handle indeterminate ΔS sign.
    entropy_mode: str = "sorted"

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

    # ── ΔS_unfold via intrinsic log-det ratio on SPD(3N − 6) ────────────
    # Folded Hessian: full ANM. Unfolded Hessian: path-graph Laplacian
    # of an ideal Gaussian chain. Three modes for combining the spectra
    # are selectable via cfg.entropy_mode (see FreeEnergyConfig).
    nma = NormalModeAnalysis(coords, cfg=cfg.nma_cfg)
    n_res = len(active_site.residues)
    H_unfold_1d = (1.0 / cfg.segment_var_A2) * path_laplacian(n_res)
    H_unfold_full = np.kron(H_unfold_1d, np.eye(3))

    if cfg.entropy_mode == "basis_free":
        log_ratio = gaussian_entropy_change_basis_free(
            nma.H, H_unfold_full,
            n_trivial_fold=cfg.nma_cfg.n_trivial_modes,
            n_trivial_unfold=3,
            floor=cfg.nma_cfg.eigenvalue_floor,
        )
    elif cfg.entropy_mode == "shape_only":
        # Geometric-mean-normalised basis-free log-det. Invariant
        # under uniform rescaling of either H_fold or H_unfold ⇒ γ
        # and σ² calibration uncertainty cannot bias this signal.
        log_ratio = gaussian_entropy_change_shape_invariant(
            nma.H, H_unfold_full,
            n_trivial_fold=cfg.nma_cfg.n_trivial_modes,
            floor=cfg.nma_cfg.eigenvalue_floor,
        )
    elif cfg.entropy_mode == "sorted":
        H_fold_eigs = np.linalg.eigvalsh(nma.H)
        H_fold_nonzero = H_fold_eigs[H_fold_eigs > cfg.nma_cfg.eigenvalue_floor]
        H_unfold_eigs = np.linalg.eigvalsh(H_unfold_full)
        H_unfold_nonzero = H_unfold_eigs[H_unfold_eigs > cfg.nma_cfg.eigenvalue_floor]
        log_ratio = gaussian_entropy_change_from_hessians(
            H_fold_nonzero, H_unfold_nonzero,
            floor=cfg.nma_cfg.eigenvalue_floor,
        )
    else:
        raise ValueError(
            f"entropy_mode must be one of "
            f"{{'sorted', 'basis_free', 'shape_only'}}, got {cfg.entropy_mode!r}"
        )

    dS = KB_KCAL * log_ratio  # kcal/(mol·K). Sign from Loewner order.

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
