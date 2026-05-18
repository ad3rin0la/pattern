"""Verify the resolutions to the calibration findings.

Each test pins a specific resolution claim from the design discussion:

1. ΔΔG cancels the unit-offset bias for matched-N mutations
   (the within-protein use case where γ uncalibration doesn't matter).
2. basis_free entropy_mode is sign-guaranteed on globular geometry
   (justifying the default switch from sorted to basis_free).
3. basis_coupling_residual is unit-offset-invariant (whatever γ and σ²
   are, this signal is calibration-free).
4. per_mode_entropy_change kills the O(N) cross-protein scaling.
5. DynamicsScorer plugs into the engineering engine end-to-end with
   the ΔΔG signal.
"""

import numpy as np
import pytest

from tope.data.active_site import ActiveSite, AtomRecord, ResidueRecord
from tope.dynamics import (
    FreeEnergyConfig,
    NMAConfig,
    NormalModeAnalysis,
    basis_coupling_residual,
    delta_delta_g_unfold,
    gaussian_entropy_change_basis_free,
    gaussian_entropy_change_from_hessians,
    path_laplacian,
    per_mode_entropy_change,
    predict_delta_g_unfold,
)


def _helix(n: int = 16) -> np.ndarray:
    r, rise = 2.3, 1.5
    twist = np.deg2rad(100.0)
    i = np.arange(n)
    return np.stack(
        [r * np.cos(twist * i), r * np.sin(twist * i), rise * i], axis=1
    )


def _site(coords, name="x", aa="ALA"):
    res = [
        ResidueRecord(
            chain_id="A", residue_name=aa, residue_number=100 + i,
            ca_coord=c, n_atoms=1,
            atoms=[AtomRecord("CA", "C", c, "A", aa, 100 + i)],
        )
        for i, c in enumerate(coords)
    ]
    return ActiveSite(pdb_id=name, residues=res, ec_number="")


# ── #1: ΔΔG cancels the unit offset for matched-N proteins ───────────────────

def test_ddg_unit_offset_cancels_under_segment_var_rescale():
    """For matched-N WT and mutant, ΔΔG is invariant under σ² rescaling
    even in 'sorted' mode (where ΔG individually is σ²-sensitive)."""
    coords = _helix(16)
    wt = _site(coords)
    mut_coords = coords.copy()
    mut_coords[5] += np.array([0.3, 0.0, 0.0])  # small perturbation
    mut = _site(mut_coords)

    for mode in ["sorted", "basis_free"]:
        cfg_a = FreeEnergyConfig(entropy_mode=mode, segment_var_A2=13.0)
        cfg_b = FreeEnergyConfig(entropy_mode=mode, segment_var_A2=1.3)
        ddg_a = delta_delta_g_unfold(wt, mut, 298.15, cfg_a)
        ddg_b = delta_delta_g_unfold(wt, mut, 298.15, cfg_b)
        np.testing.assert_allclose(
            ddg_a, ddg_b, atol=1e-6,
            err_msg=f"ΔΔG drifted under σ² rescale in mode {mode!r}",
        )


def test_ddg_zero_for_unchanged_geometry():
    """ΔΔG(WT, WT) = 0 to numerical noise."""
    wt = _site(_helix(20))
    cfg = FreeEnergyConfig()
    ddg = delta_delta_g_unfold(wt, wt, 298.15, cfg)
    assert abs(ddg) < 1e-9


# ── #2: basis_free entropy_mode is sign-guaranteed and is the new default ────

def test_basis_free_is_default_entropy_mode():
    assert FreeEnergyConfig().entropy_mode == "basis_free"


def test_basis_free_positive_entropy_on_globular_fold():
    """For a folded helix, basis_free should give ΔS > 0 (folded state
    is stiffer than the Gaussian chain on the folded subspace)."""
    site = _site(_helix(20))
    cfg = FreeEnergyConfig(entropy_mode="basis_free")
    dG_280 = predict_delta_g_unfold(site, T=280.0, cfg=cfg)
    dG_380 = predict_delta_g_unfold(site, T=380.0, cfg=cfg)
    # ΔG_unfold monotone decreasing iff ΔS_unfold > 0.
    assert dG_380 < dG_280


# ── #3: basis_coupling_residual is unit-offset-invariant ─────────────────────

def test_basis_coupling_residual_invariant_under_uniform_rescale():
    """Multiplying H_fold by α or H_unfold by β must not change the
    residual: both shift sorted and basis_free identically."""
    coords = _helix(16)
    nma = NormalModeAnalysis(coords, cfg=NMAConfig())
    H_unfold = np.kron((1.0 / 13.0) * path_laplacian(16), np.eye(3))

    r_base = basis_coupling_residual(nma.H, H_unfold)
    r_scale_f = basis_coupling_residual(7.3 * nma.H, H_unfold)
    r_scale_u = basis_coupling_residual(nma.H, 0.04 * H_unfold)

    np.testing.assert_allclose(r_base, r_scale_f, atol=1e-9)
    np.testing.assert_allclose(r_base, r_scale_u, atol=1e-9)


def test_basis_coupling_residual_matches_basis_free_minus_sorted():
    """By definition the residual is basis_free − sorted."""
    coords = _helix(16)
    nma = NormalModeAnalysis(coords, cfg=NMAConfig())
    H_unfold = np.kron((1.0 / 13.0) * path_laplacian(16), np.eye(3))

    eigs_f = np.linalg.eigvalsh(nma.H)
    eigs_u = np.linalg.eigvalsh(H_unfold)
    sorted_val = gaussian_entropy_change_from_hessians(
        eigs_f[eigs_f > 1e-8], eigs_u[eigs_u > 1e-8],
    )
    bf_val = gaussian_entropy_change_basis_free(nma.H, H_unfold)

    expected = bf_val - sorted_val
    actual = basis_coupling_residual(nma.H, H_unfold)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


# ── #4: per-mode normalisation kills O(N) scaling ────────────────────────────

def test_per_mode_entropy_change_size_invariant():
    """ΔS / mode should be roughly N-independent on the same fold class.
    Total ΔS scales as O(N), per-mode does not."""
    sigma = 13.0
    per_mode_values = []
    totals = []
    for N in (12, 18, 24, 30):
        coords = _helix(N)
        nma = NormalModeAnalysis(coords, cfg=NMAConfig())
        H_unfold = np.kron((1.0 / sigma) * path_laplacian(N), np.eye(3))
        log_ratio = gaussian_entropy_change_basis_free(nma.H, H_unfold)
        k = 3 * N - 6
        per_mode_values.append(per_mode_entropy_change(log_ratio, k))
        totals.append(log_ratio)

    # Totals grow with N; per-mode stays in a tight band.
    assert totals[-1] / totals[0] > 1.5, (
        f"Total ΔS didn't scale with N as expected: {totals}"
    )
    spread = (max(per_mode_values) - min(per_mode_values)) / abs(
        np.mean(per_mode_values)
    )
    assert spread < 0.25, (
        f"Per-mode entropy varied too much across N: {per_mode_values}, "
        f"spread = {spread:.3f}"
    )


# ── #5: DynamicsScorer wires into the engineering engine ─────────────────────

def test_dynamics_scorer_in_engine():
    from tope.engineering import (
        DynamicsScorer, EngineConfig, EngineeringEngine, Mutation,
        SaturationScan,
    )

    site = _site(_helix(12))
    engine = EngineeringEngine(
        scorer=DynamicsScorer(),
        search=SaturationScan(top_k=3),
        cfg=EngineConfig(
            target_amino_acids=["GLY", "VAL"],
            saturation_top_k=3,
        ),
    )
    out = engine.propose(site)
    assert len(out) == 3
    # Every score should be a finite ΔΔG.
    assert all(np.isfinite(s.score) for s in out)
    # Scoring WT against itself should be zero.
    direct = engine.score(site, [])
    assert abs(direct.score) < 1e-9
