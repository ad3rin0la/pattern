"""Validation suite for tope.topology.hsh_restriction_maps.

Mirrors §5 of the HSH Restriction Maps handoff doc.  Imports the module
directly by file path to avoid the (pre-existing, unrelated) import error
in the top-level ``tope`` package __init__.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


# ── Module loader ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HSH_PATH = _REPO_ROOT / "tope" / "topology" / "hsh_restriction_maps.py"


def _load_module():
    if "hsh_restriction_maps" in sys.modules:
        return sys.modules["hsh_restriction_maps"]
    spec = importlib.util.spec_from_file_location(
        "hsh_restriction_maps", str(_HSH_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hsh_restriction_maps"] = module
    spec.loader.exec_module(module)
    return module


hsh = _load_module()


# ── Test 1: Basis shape ──────────────────────────────────────────────────────

def test_basis_shape_d8():
    basis, labels = hsh.compute_hsh_basis_matrices(d=8, lambda_max=2)
    assert basis.shape == (36, 8, 8)
    assert len(labels) == 36


def test_n_hsh_counts():
    assert hsh.n_hsh(8, 0) == 1
    assert hsh.n_hsh(8, 1) == 8
    assert hsh.n_hsh(8, 2) == 35
    assert hsh.n_hsh(8, 3) == 112


# ── Test 2: Frobenius orthonormality ─────────────────────────────────────────

def test_frobenius_orthonormality():
    basis, _ = hsh.compute_hsh_basis_matrices(d=8, lambda_max=2)
    gram = np.einsum("kij,lij->kl", basis, basis)
    assert np.allclose(gram, np.eye(basis.shape[0]), atol=1e-12)


# ── Test 3: Parseval completeness ────────────────────────────────────────────

def test_parseval_completeness():
    basis, _ = hsh.compute_hsh_basis_matrices(d=8, lambda_max=2)
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((100, 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for v in vecs:
        assert abs(hsh.parseval_check(v, basis) - 1.0) < 1e-12


# ── Test 4: Isotropic limit (lambda_max=0) ───────────────────────────────────

def test_isotropic_limit_proportional_to_identity():
    basis_iso, _ = hsh.compute_hsh_basis_matrices(d=8, lambda_max=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = rng.standard_normal(8)
        v /= np.linalg.norm(v)
        c = hsh.hsh_coefficients(v, basis_iso)
        F = np.einsum("k,kij->ij", c, basis_iso)
        off = F - np.diag(np.diag(F))
        assert np.max(np.abs(off)) < 1e-12
        # All diagonal entries equal
        assert np.var(np.diag(F)) < 1e-24


# ── Test 5: Gegenbauer values at t = 1, 0, -1 for α = 3 ──────────────────────

def test_gegenbauer_C2_alpha3():
    assert abs(hsh.gegenbauer(2, 3.0, 1.0) - 21.0) < 1e-12
    assert abs(hsh.gegenbauer(2, 3.0, 0.0) - (-3.0)) < 1e-12
    assert abs(hsh.gegenbauer(2, 3.0, -1.0) - 21.0) < 1e-12


def test_gegenbauer_recursion_low_order():
    # C_0 = 1, C_1 = 2α t; check at α=3, t=0.5
    assert hsh.gegenbauer(0, 3.0, 0.5) == 1.0
    assert hsh.gegenbauer(1, 3.0, 0.5) == 3.0


# ── Test 6: Addition theorem Monte Carlo ─────────────────────────────────────

def test_addition_theorem_lambda0():
    mae = hsh.verify_addition_theorem(d=8, lam=0, n_samples=2000)
    assert mae < 1e-10


def test_addition_theorem_lambda2():
    mae = hsh.verify_addition_theorem(d=8, lam=2, n_samples=5000)
    assert mae < 1e-10


# ── Bonus: rank-1 init reproduces δ̂_G ⊗ δ̂_G  (zero-diff vs. current map) ──

def test_rank1_init_reproduces_outer_product():
    """With init_mode='rank1', restriction map = δ̂_G ⊗ δ̂_G exactly."""
    cfg = hsh.HSHConfig(d=8, init_mode="rank1")
    layer = hsh.HSHSheafLaplacian(d=8, config=cfg)

    rng = np.random.default_rng(0)
    for _ in range(20):
        s_i = rng.standard_normal(8)
        s_j = rng.standard_normal(8)
        R = layer.restriction_map(s_i, s_j, normalize=True)

        # Reference: the existing Cerrini construction
        delta = s_i - s_j
        G_delta = delta  # G = I here
        inner = float(delta @ G_delta) + 1e-10
        delta_hat_G = G_delta / math.sqrt(inner)
        delta_hat = delta_hat_G / np.linalg.norm(delta_hat_G)
        R_ref = np.outer(delta_hat, delta_hat)
        R_ref = R_ref / np.linalg.norm(R_ref)

        assert np.allclose(R, R_ref, atol=1e-10), \
            f"max diff = {np.max(np.abs(R - R_ref)):.2e}"


def test_isotropic_init_is_identity_proportional():
    cfg = hsh.HSHConfig(d=8, init_mode="isotropic")
    layer = hsh.HSHSheafLaplacian(d=8, config=cfg)
    rng = np.random.default_rng(1)
    s_i = rng.standard_normal(8)
    s_j = rng.standard_normal(8)
    R = layer.restriction_map(s_i, s_j)
    # Should be I/√d (off-diagonals zero, diagonals equal)
    off = R - np.diag(np.diag(R))
    assert np.max(np.abs(off)) < 1e-12
    assert np.var(np.diag(R)) < 1e-24


# ── Aggregate suite via run_validation() ─────────────────────────────────────

def test_run_validation_passes():
    assert hsh.run_validation(verbose=False)


# ── PyTorch path (skipped if torch unavailable) ──────────────────────────────

torch = pytest.importorskip("torch")


def test_torch_forward_matches_numpy_rank1():
    cfg = hsh.HSHConfig(d=8, init_mode="rank1")
    np_layer = hsh.HSHSheafLaplacian(d=8, config=cfg)
    pt_layer = hsh.HSHRestrictionMap(d=8, config=cfg)

    rng = np.random.default_rng(7)
    for _ in range(5):
        s_i_np = rng.standard_normal(8)
        s_j_np = rng.standard_normal(8)
        R_np = np_layer.restriction_map(s_i_np, s_j_np)
        s_i = torch.from_numpy(s_i_np.astype(np.float32))
        s_j = torch.from_numpy(s_j_np.astype(np.float32))
        R_pt = pt_layer(s_i, s_j).detach().cpu().numpy().astype(np.float64)
        assert np.allclose(R_np, R_pt, atol=1e-5), \
            f"numpy/torch diff = {np.max(np.abs(R_np - R_pt)):.2e}"


def test_sheafenm_integration_zero_diff_at_rank1_init():
    """SheafENM with use_hsh=True and rank-1 init must produce a sheaf
    Laplacian identical (to 1e-10) to the unflagged baseline when G=I.

    Spec §3.4 "Backward compatibility": the HSH expansion with all weights
    = 1 reproduces δ̂⊗δ̂ exactly by basis completeness, so the per-edge
    restriction map matches the existing Cerrini outer-product map when
    the descriptor metric is the identity.
    """
    # Avoid the pre-existing broken `tope.data` import in the top-level
    # `tope/__init__.py` by manually wiring shim packages into sys.modules
    # and loading the two relevant modules by file path.  The lazy
    # `from tope.topology.hsh_restriction_maps import ...` inside
    # SheafENM.__init__ will resolve via sys.modules without re-executing
    # the broken package init.
    import types
    if "tope" not in sys.modules:
        sys.modules["tope"] = types.ModuleType("tope")
        sys.modules["tope"].__path__ = [str(_REPO_ROOT / "tope")]
    if "tope.topology" not in sys.modules:
        topo_pkg = types.ModuleType("tope.topology")
        topo_pkg.__path__ = [str(_REPO_ROOT / "tope" / "topology")]
        sys.modules["tope.topology"] = topo_pkg
        sys.modules["tope"].topology = topo_pkg  # type: ignore[attr-defined]

    # Pre-load hsh_restriction_maps under the dotted name
    if "tope.topology.hsh_restriction_maps" not in sys.modules:
        spec_h = importlib.util.spec_from_file_location(
            "tope.topology.hsh_restriction_maps", str(_HSH_PATH)
        )
        m_h = importlib.util.module_from_spec(spec_h)
        sys.modules["tope.topology.hsh_restriction_maps"] = m_h
        spec_h.loader.exec_module(m_h)

    # Load phonon_topology by file path
    if "tope.topology.phonon_topology" not in sys.modules:
        spec_p = importlib.util.spec_from_file_location(
            "tope.topology.phonon_topology",
            str(_REPO_ROOT / "tope" / "topology" / "phonon_topology.py"),
        )
        m_p = importlib.util.module_from_spec(spec_p)
        sys.modules["tope.topology.phonon_topology"] = m_p
        spec_p.loader.exec_module(m_p)

    SheafENM = sys.modules["tope.topology.phonon_topology"].SheafENM

    # Build a minimal triangle PCC: 3 nodes, 3 edges.
    from scipy.sparse import csr_matrix
    B_01 = csr_matrix(np.array([
        [-1,  0,  1],   # node 0 in edges 0 (out), 2 (in)
        [ 1, -1,  0],   # node 1 in edges 0 (in),  1 (out)
        [ 0,  1, -1],   # node 2 in edges 1 (in),  2 (out)
    ], dtype=float))
    pcc = {"boundary_0_1": B_01}

    rng = np.random.default_rng(123)
    sections = {i: rng.standard_normal(8) for i in range(3)}

    baseline = SheafENM(pcc, sheaf_dim=8)
    hsh_enm = SheafENM(pcc, sheaf_dim=8, use_hsh=True)  # default rank1 init

    L_base = baseline.sheaf_laplacian(rank=0, sheaf_sections=sections)
    L_hsh = hsh_enm.sheaf_laplacian(rank=0, sheaf_sections=sections)

    diff = float(np.max(np.abs(L_base - L_hsh)))
    assert diff < 1e-10, f"sheaf Laplacian diff = {diff:.2e}"


def test_torch_isotropy_fraction_starts_at_one_for_rank1():
    """Sanity: rank-1 init puts equal weight on all components, so
    isotropy_fraction is exactly 1/n_basis = 1/36."""
    cfg = hsh.HSHConfig(d=8, init_mode="rank1")
    layer = hsh.HSHSheafLaplacian(d=8, config=cfg)
    assert abs(layer.isotropy_fraction() - 1.0 / 36) < 1e-12

    cfg = hsh.HSHConfig(d=8, init_mode="isotropic")
    layer = hsh.HSHSheafLaplacian(d=8, config=cfg)
    assert abs(layer.isotropy_fraction() - 1.0) < 1e-12
