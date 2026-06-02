"""Tests for the §5 tangent-space gyro sheaf Laplacian (SheafENM extensions).

Verifies the spec's degree-0 guarantees: the γ-weighted ``L = δ*δ`` is
self-adjoint w.r.t. the γ-measure inner product and PSD (real ≥ 0 spectrum)
unconditionally, and that the transport rotation ``R_xy`` genuinely enters
(dropping it — the "metric-flattening" — changes the operator and the
holonomy FrustIndex).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_pkg(name: str, path: Path) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod._gyro_test_stub = True  # tagged so conftest can purge it (test isolation)
        sys.modules[name] = mod


def _load(name: str, path: Path):
    if name in sys.modules and getattr(sys.modules[name], "__file__", None):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_pkg("tope", _REPO_ROOT / "tope")
_stub_pkg("tope.topology", _REPO_ROOT / "tope" / "topology")
gm = _load("tope.topology.gyro_memory", _REPO_ROOT / "tope" / "topology" / "gyro_memory.py")
pt = _load("tope.topology.phonon_topology", _REPO_ROOT / "tope" / "topology" / "phonon_topology.py")


def _triangle_pcc():
    """3 atoms, 3 bonds — oriented incidence of a triangle graph."""
    # B_01[atom, bond] = ±1.  Bonds: (0,1), (1,2), (2,0).
    B01 = np.array([
        [-1.0, 0.0, 1.0],   # atom 0 in bonds 0 (−) and 2 (+)
        [1.0, -1.0, 0.0],   # atom 1 in bonds 0 (+) and 1 (−)
        [0.0, 1.0, -1.0],   # atom 2 in bonds 1 (+) and 2 (−)
    ])
    return {"boundary_0_1": B01}


def _make_sheaf(d=3, seed=0):
    enzyme_pcc = _triangle_pcc()
    sheaf = pt.SheafENM(enzyme_pcc, sheaf_dim=d)
    rng = np.random.default_rng(seed)
    # Descriptor sections (will be lifted to the ball via exp_o internally).
    sections = {i: rng.normal(scale=0.4, size=d) for i in range(3)}
    return sheaf, sections, d


def test_gyro_sheaf_laplacian_is_psd_and_self_adjoint():
    sheaf, sections, d = _make_sheaf()
    L = sheaf.gyro_sheaf_laplacian(0, sections, ball_radius=1.0)
    N = 3
    assert L.shape == (N * d, N * d)

    # Recover the γ-measure D_0 to check self-adjointness ⟨·,·⟩_{D_0}.
    h = sheaf._hyperbolic_features(sections, None, 1.0)
    mu = np.array([float(gm.conformal_gamma(h[i], s=1.0)) ** (2 * d) for i in range(N)])
    D0 = np.diag(np.repeat(mu, d))
    DL = D0 @ L
    np.testing.assert_allclose(DL, DL.T, atol=1e-9)  # D_0 L symmetric ⇒ self-adjoint

    eig = np.linalg.eigvalsh(0.5 * (DL + DL.T))  # spectrum w.r.t. D_0 metric
    assert eig.min() > -1e-9, f"not PSD: min eig {eig.min():.2e}"


def test_transport_changes_the_operator():
    """Dropping R_xy (features → origin) is the metric-flattening; it must
    produce a different Laplacian when features are genuinely curved."""
    sheaf, sections, d = _make_sheaf(seed=1)
    L_curved = sheaf.gyro_sheaf_laplacian(0, sections, ball_radius=1.0)
    # Flat field: all features at the origin → R_xy = I, μ = 1.
    flat = {i: np.zeros(d) for i in sections}
    L_flat = sheaf.gyro_sheaf_laplacian(0, sections, hyperbolic_features=flat)
    assert np.abs(L_curved - L_flat).max() > 1e-6


def test_gyro_restriction_maps_equal_K_when_flat():
    sheaf, sections, d = _make_sheaf(seed=2)
    K = sheaf.build_restriction_maps(0, sections)
    flat = {i: np.zeros(d) for i in sections}
    F = sheaf.gyro_restriction_maps(0, sections, hyperbolic_features=flat)
    for key in K:
        np.testing.assert_allclose(F[key], K[key], atol=1e-9)


def test_gyro_restriction_maps_differ_when_curved():
    sheaf, sections, d = _make_sheaf(seed=3)
    K = sheaf.build_restriction_maps(0, sections)
    F = sheaf.gyro_restriction_maps(0, sections, ball_radius=1.0)
    diffs = [np.abs(F[k] - K[k]).max() for k in K]
    assert max(diffs) > 1e-6


def test_holonomy_frust_index_nontrivial_for_curved_loop():
    sheaf, sections, d = _make_sheaf(seed=4)
    res = sheaf.holonomy_frust_index([[0, 1, 2]], sections, ball_radius=1.0)
    assert res["frust_total"] > 1e-6
    assert not res["is_flat"]


def test_holonomy_frust_index_flat_for_origin_loop():
    sheaf, sections, d = _make_sheaf(seed=5)
    flat = {i: np.zeros(d) for i in sections}
    res = sheaf.holonomy_frust_index([[0, 1, 2]], sections, hyperbolic_features=flat)
    assert res["frust_total"] < 1e-8
    assert res["is_flat"]
