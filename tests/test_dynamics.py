"""Smoke + validation tests for tope.dynamics."""

import numpy as np
import pytest

from tope.data.active_site import ActiveSite, AtomRecord, ResidueRecord
from tope.dynamics import (
    DeltaGDaggerStub,
    DeltaGUnfoldStub,
    FreeEnergyConfig,
    NMAConfig,
    NormalModeAnalysis,
    build_anm_hessian,
    predict_delta_g_dagger,
    predict_delta_g_unfold,
    sample_harmonic_ensemble,
    thermal_msf,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _toy_cluster(n: int = 6, seed: int = 0) -> np.ndarray:
    """Random 3D cluster of Cα atoms — non-degenerate ANM geometry.

    A perfectly collinear chain has zero transverse stiffness in ANM
    (the dyad r⊗r/r² has rank 1 along the chain axis), which produces
    a huge spectrum of near-zero modes and breaks the harmonic
    sampler. A random cluster avoids this.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(scale=4.0, size=(n, 3))


def _toy_site(n: int = 6) -> ActiveSite:
    coords = _toy_cluster(n)
    residues = []
    for i in range(n):
        a = AtomRecord("CA", "C", coords[i], "A", "ALA", 100 + i)
        residues.append(ResidueRecord(
            chain_id="A", residue_name="ALA", residue_number=100 + i,
            ca_coord=coords[i], n_atoms=1, atoms=[a],
        ))
    return ActiveSite(pdb_id="toy", residues=residues, ec_number="3.4.21.1")


# ── ANM Hessian ──────────────────────────────────────────────────────────────

def test_anm_hessian_symmetric_psd():
    coords = _toy_cluster(5)
    H = build_anm_hessian(coords, cutoff=5.0, force_constant=1.0)
    assert H.shape == (15, 15)
    # Symmetric.
    assert np.allclose(H, H.T)
    # PSD (rigid modes are 0 eigenvalues, rest non-negative).
    eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
    assert eigs.min() > -1e-8


def test_anm_has_six_rigid_modes():
    """3D rigid body in 3D space ⇒ at least 6 near-zero eigenvalues."""
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(8, 3))
    H = build_anm_hessian(coords, cutoff=10.0)
    eigs = np.sort(np.linalg.eigvalsh(0.5 * (H + H.T)))
    n_zero = (np.abs(eigs) < 1e-6).sum()
    assert n_zero >= 6


# ── Normal mode analysis ─────────────────────────────────────────────────────

def test_nma_frequencies_real_nonneg():
    nma = NormalModeAnalysis(
        _toy_cluster(6),
        cfg=NMAConfig(contact_cutoff=5.0, n_modes=5),
    )
    omega, modes = nma.nontrivial_modes()
    assert omega.shape == (5,)
    assert modes.shape == (5, 18)
    assert (omega >= 0).all()


def test_thermal_msf_positive_per_residue():
    nma = NormalModeAnalysis(
        _toy_cluster(8),
        cfg=NMAConfig(contact_cutoff=6.0, n_modes=10, temperature_K=300.0),
    )
    msf = thermal_msf(nma)
    assert msf.shape == (8,)
    assert (msf > 0).all()


def test_msf_scales_linearly_with_T():
    """MSF ∝ T in the harmonic regime."""
    cfg = NMAConfig(contact_cutoff=6.0, n_modes=10)
    nma = NormalModeAnalysis(_toy_cluster(8), cfg=cfg)
    msf_300 = thermal_msf(nma, temperature_K=300.0)
    msf_600 = thermal_msf(nma, temperature_K=600.0)
    np.testing.assert_allclose(msf_600 / msf_300, 2.0, rtol=1e-6)


def test_propagate_mode_returns_trajectory():
    nma = NormalModeAnalysis(
        _toy_cluster(6),
        cfg=NMAConfig(contact_cutoff=5.0, n_modes=4),
    )
    traj = nma.propagate_mode(mode_idx=0, amplitude=0.5, n_steps=20)
    assert traj.shape == (20, 6, 3)
    # First frame ≈ baseline (cos(0) = 1, so r0 + A·v).
    assert not np.allclose(traj[0], nma.coords)
    # Middle of period: cos(π) = −1, so symmetric about baseline.
    np.testing.assert_allclose(traj[0] + traj[10], 2 * nma.coords, atol=1e-6)


def test_harmonic_ensemble_variance_matches_msf():
    """Empirical per-residue variance (summed over xyz) should agree
    with the analytic thermal MSF formula in the large-sample limit."""
    nma = NormalModeAnalysis(
        _toy_cluster(8, seed=1),
        cfg=NMAConfig(contact_cutoff=8.0, n_modes=12, temperature_K=300.0),
    )
    rng = np.random.default_rng(42)
    samples = sample_harmonic_ensemble(nma, n_samples=4000, rng=rng)
    assert samples.shape == (4000, 8, 3)

    deltas = samples - nma.coords[None, :, :]
    empirical = (deltas ** 2).sum(axis=-1).mean(axis=0)  # (N,)
    analytic = thermal_msf(nma)

    # Allow 25% relative error — softest modes have huge variance and
    # sample noise is significant at n=4000.
    np.testing.assert_allclose(empirical, analytic, rtol=0.25)


# ── Free-energy stubs ────────────────────────────────────────────────────────

def test_delta_g_unfold_positive_for_compact_site():
    site = _toy_site(8)
    dG = predict_delta_g_unfold(site, T=298.15)
    # Contact enthalpy dominates over harmonic entropy for a compact fold.
    assert dG > 0


def test_delta_g_unfold_temperature_dependence():
    """ΔG_unfold should decrease as T increases (TΔS term grows)."""
    site = _toy_site(10)
    cfg = FreeEnergyConfig()
    dG_low = predict_delta_g_unfold(site, T=280.0, cfg=cfg)
    dG_high = predict_delta_g_unfold(site, T=380.0, cfg=cfg)
    assert dG_high < dG_low


def test_delta_g_dagger_stub_constant():
    site = _toy_site(8)
    cfg = FreeEnergyConfig(Ea_kcal_per_mol=18.0)
    assert predict_delta_g_dagger(site, T=298.15, cfg=cfg) == 18.0


def test_stubs_plug_into_thermal_composite_scorer():
    """The whole point: DeltaGUnfoldStub/DeltaGDaggerStub fit the
    ThermalCompositeScorer callable contracts without adapters."""
    from tope.engineering import (
        BoltzmannFoldedFraction, ThermalCompositeScorer, Mutation,
    )

    site = _toy_site(8)
    scorer = ThermalCompositeScorer(
        folded=BoltzmannFoldedFraction(delta_g_unfold=DeltaGUnfoldStub()),
        ea=DeltaGDaggerStub(),
        T=310.0,
    )
    # Mutate one of the toy residues to something that exists in the
    # mutation engine's standard alphabet.
    res = scorer.score(site, [Mutation("A", 103, "GLY")])
    # Stub Ea is identical for WT and mutant ⇒ Arrhenius term is zero.
    assert abs(res.breakdown["log_arrhenius_gain"]) < 1e-9
    # Score is finite.
    assert np.isfinite(res.score)
