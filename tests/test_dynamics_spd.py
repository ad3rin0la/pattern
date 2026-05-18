"""Tests for the SPD-manifold dynamics primitives."""

import numpy as np
import pytest

from tope.dynamics import (
    LogEuclideanContact,
    NMAConfig,
    NormalModeAnalysis,
    build_anm_hessian,
    gaussian_chain_covariance,
    gaussian_entropy_change_from_hessians,
    log_det_ratio_from_eigvals,
    path_laplacian,
    relative_entropy_spd,
)


# ── Log-Euclidean contact tensors ────────────────────────────────────────────

def test_log_euclidean_isotropic_recovers_scalar_spring():
    """expm(log(γ)·I) = γ·I — the isotropic limit must round-trip."""
    c = LogEuclideanContact.isotropic(2.5)
    G = c.tensor
    np.testing.assert_allclose(G, 2.5 * np.eye(3), atol=1e-12)


def test_log_euclidean_is_spd():
    """Random Sym(3) parameters give SPD matrices."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(3, 3))
        L = 0.5 * (A + A.T)
        G = LogEuclideanContact(L=L).tensor
        eigs = np.linalg.eigvalsh(G)
        assert (eigs > 0).all()


def test_log_euclidean_from_spd_round_trip():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    spd = A @ A.T + 0.1 * np.eye(3)
    c = LogEuclideanContact.from_spd(spd)
    np.testing.assert_allclose(c.tensor, spd, atol=1e-10)


# ── Collinear-chain fix: tensor-ANM has no transverse degeneracy ──────────────

def test_scalar_anm_has_collinear_chain_degeneracy():
    """Baseline: scalar ANM on a straight chain has many zero modes."""
    coords = np.stack(
        [np.arange(8) * 3.8, np.zeros(8), np.zeros(8)], axis=1,
    )
    H = build_anm_hessian(coords, cutoff=5.0)
    eigs = np.sort(np.linalg.eigvalsh(0.5 * (H + H.T)))
    # Rigid translations (6) plus 2N transverse zero modes ⇒ many zeros.
    n_zero = int((np.abs(eigs) < 1e-8).sum())
    assert n_zero > 6


def test_tensor_anm_resolves_collinear_chain_degeneracy():
    """SPD contact tensors give full transverse stiffness on a chain."""
    coords = np.stack(
        [np.arange(8) * 3.8, np.zeros(8), np.zeros(8)], axis=1,
    )
    iso = LogEuclideanContact.isotropic(1.0)

    def tensors(i, j):
        return iso.tensor   # γ·I — full rank, no rank-1 collapse

    H = build_anm_hessian(coords, cutoff=5.0, contact_tensors=tensors)
    eigs = np.sort(np.linalg.eigvalsh(0.5 * (H + H.T)))
    n_zero = int((np.abs(eigs) < 1e-8).sum())
    # A generic 3D rigid body has 6 zero modes — but a perfectly linear
    # chain still has rotational symmetry around its axis, so 1 extra
    # zero is allowed. The crucial thing is no transverse degeneracy.
    assert n_zero <= 7


def test_tensor_anm_nma_succeeds_on_chain():
    """End-to-end: NMA on the collinear chain produces a real spectrum."""
    coords = np.stack(
        [np.arange(8) * 3.8, np.zeros(8), np.zeros(8)], axis=1,
    )
    iso = LogEuclideanContact.isotropic(1.0)

    nma = NormalModeAnalysis(
        coords,
        cfg=NMAConfig(contact_cutoff=5.0, n_modes=10, n_trivial_modes=7),
        contact_tensors=lambda i, j: iso.tensor,
    )
    omega, modes = nma.nontrivial_modes()
    assert omega.shape == (10,)
    assert (omega >= 0).all()
    # Most of the kept modes should be non-trivial (> 1e-4).
    assert (omega > 1e-4).sum() >= 7


# ── Gaussian-chain unfolded covariance ────────────────────────────────────────

def test_path_laplacian_known_eigenvalues():
    """L_path on n nodes has eigenvalues 2(1 − cos(πk/n))."""
    n = 8
    L = path_laplacian(n)
    eigs = np.sort(np.linalg.eigvalsh(L))
    expected = np.sort(2.0 * (1.0 - np.cos(np.pi * np.arange(n) / n)))
    np.testing.assert_allclose(eigs, expected, atol=1e-10)


def test_gaussian_chain_covariance_is_spd_after_translation_strip():
    n = 6
    sigma = gaussian_chain_covariance(n, segment_var_A2=13.0,
                                      exclude_translation=True)
    assert sigma.shape == (3 * n, 3 * n)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)
    eigs = np.sort(np.linalg.eigvalsh(sigma))
    n_zero = int((np.abs(eigs) < 1e-8).sum())
    # 3 translational modes projected out; the rest of the SPSD kernel
    # (the path-Laplacian zero mode is the same constant vector, so
    # nothing extra) must be SPD.
    assert n_zero == 3
    assert (eigs[3:] > 0).all()


# ── Intrinsic entropy machinery ──────────────────────────────────────────────

def test_relative_entropy_zero_for_identical_covariances():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(5, 5))
    sigma = A @ A.T + 0.5 * np.eye(5)
    assert abs(relative_entropy_spd(sigma, sigma)) < 1e-9


def test_relative_entropy_nonneg_for_distinct_covariances():
    rng = np.random.default_rng(8)
    A1 = rng.normal(size=(5, 5))
    A2 = rng.normal(size=(5, 5))
    s1 = A1 @ A1.T + 0.5 * np.eye(5)
    s2 = A2 @ A2.T + 0.5 * np.eye(5)
    # KL(s1 || s2) is non-negative for SPD covariances.
    assert relative_entropy_spd(s1, s2) >= -1e-9
    assert relative_entropy_spd(s2, s1) >= -1e-9


def test_log_det_ratio_sign():
    eigs_big = np.array([10.0, 5.0, 2.0])
    eigs_small = np.array([1.0, 0.5, 0.2])
    # Σ_unf has larger eigenvalues ⇒ Σ_fold⁻¹ has the smaller ones ⇒
    # log_det_ratio(big, small) > 0.
    assert log_det_ratio_from_eigvals(eigs_big, eigs_small) > 0
    assert log_det_ratio_from_eigvals(eigs_small, eigs_big) < 0


def test_gaussian_entropy_change_positive_for_floppier_unfolded():
    """Folded state has stiffer Hessian than unfolded ⇒ ΔS_unfold > 0."""
    fold = np.array([100.0, 80.0, 60.0, 40.0])
    unfold = np.array([1.0, 0.8, 0.6, 0.4])
    dS = gaussian_entropy_change_from_hessians(fold, unfold)
    assert dS > 0


# ── End-to-end: ΔG_unfold via SPD log-det ratio ──────────────────────────────

def test_predict_delta_g_unfold_uses_spd_pathway():
    """ΔG_unfold should still be positive and T-decreasing under the
    new SPD-based ΔS formula."""
    from tope.dynamics import predict_delta_g_unfold, FreeEnergyConfig
    from tope.data.active_site import ActiveSite, AtomRecord, ResidueRecord

    rng = np.random.default_rng(2)
    coords = rng.normal(scale=4.0, size=(10, 3))
    residues = []
    for i in range(10):
        a = AtomRecord("CA", "C", coords[i], "A", "ALA", 100 + i)
        residues.append(ResidueRecord(
            chain_id="A", residue_name="ALA", residue_number=100 + i,
            ca_coord=coords[i], n_atoms=1, atoms=[a],
        ))
    site = ActiveSite(pdb_id="x", residues=residues, ec_number="")

    cfg = FreeEnergyConfig()
    dG_low = predict_delta_g_unfold(site, T=280.0, cfg=cfg)
    dG_high = predict_delta_g_unfold(site, T=380.0, cfg=cfg)
    assert dG_low > 0
    assert dG_high < dG_low
