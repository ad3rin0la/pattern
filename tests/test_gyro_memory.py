"""Tests for tope.topology.gyro_memory — verifies the closed-form gyration
against its defining identity and checks the documented properties (spec §4).

Imports the module directly by file path to avoid the (pre-existing,
unrelated) import error in the top-level ``tope`` package __init__.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GM_PATH = _REPO_ROOT / "tope" / "topology" / "gyro_memory.py"


def _load_module():
    if "gyro_memory" in sys.modules:
        return sys.modules["gyro_memory"]
    spec = importlib.util.spec_from_file_location("gyro_memory", str(_GM_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["gyro_memory"] = module
    spec.loader.exec_module(module)
    return module


gm = _load_module()


def _mobius_add_np(a, b, s=1.0):
    s2 = s * s
    ab = np.dot(a, b) / s2
    aa = np.dot(a, a) / s2
    bb = np.dot(b, b) / s2
    num = (1.0 + 2.0 * ab + bb) * a + (1.0 - aa) * b
    den = 1.0 + 2.0 * ab + aa * bb
    return num / den


def _rand_ball(d, rng, s=1.0, rmax=0.9):
    """Uniform-ish random point strictly inside B^d_s."""
    x = rng.normal(size=d)
    x = x / np.linalg.norm(x)
    r = rng.uniform(0.0, rmax) * s
    return r * x


@pytest.mark.parametrize("d", [2, 3, 4, 5, 8])
def test_gyr_matches_defining_identity(d):
    """gyr[a,b]v = ⊖(a⊕b) ⊕ (a ⊕ (b⊕v))  (Ungar).

    The spec reports < 2e-13 over its verified regime; sampling up to radius
    0.9 here pushes nearer the boundary where the small ``D`` denominator
    amplifies float round-off, so the closed form is asserted correct to a
    still-stringent 1e-11 (machine-precision agreement away from the edge).
    """
    rng = np.random.default_rng(d)
    max_err = 0.0
    for _ in range(2000):
        a, b, v = (_rand_ball(d, rng) for _ in range(3))
        lhs = gm.gyr(a, b, v)
        rhs = _mobius_add_np(-_mobius_add_np(a, b), _mobius_add_np(a, _mobius_add_np(b, v)))
        max_err = max(max_err, float(np.linalg.norm(lhs - rhs)))
    assert max_err < 1e-11, f"d={d}: max identity error {max_err:.2e}"


@pytest.mark.parametrize("d", [2, 3, 5])
def test_gyr_is_orthogonal_det_plus_one(d):
    """gyr[a,b] is a rotation: RᵀR = I, det = +1, fixes span{a,b}^⊥."""
    rng = np.random.default_rng(100 + d)
    for _ in range(50):
        a, b = _rand_ball(d, rng), _rand_ball(d, rng)
        R = gm.transport_rotation(b, -a)  # arbitrary a,b via transport wrapper
        # Build the raw gyr matrix directly: columns = gyr[a,b] e_k.
        R = np.stack([gm.gyr(a, b, np.eye(d)[:, k]) for k in range(d)], axis=-1)
        np.testing.assert_allclose(R.T @ R, np.eye(d), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        # Vectors orthogonal to span{a,b} are fixed.
        basis = np.stack([a, b])
        # find a vector orthogonal to both
        for _try in range(10):
            w = rng.normal(size=d)
            w = w - basis.T @ np.linalg.lstsq(basis.T, w, rcond=None)[0]
            if np.linalg.norm(w) > 1e-6:
                break
        np.testing.assert_allclose(R @ w, w, atol=1e-9)


def test_general_s_scaling_consistency():
    """gyr is scale-equivariant: gyr_s[sa,sb](sv) = s·gyr_1[a,b]v."""
    rng = np.random.default_rng(7)
    d, s = 4, 2.5
    for _ in range(200):
        a, b, v = (_rand_ball(d, rng, s=1.0) for _ in range(3))
        out_s = gm.gyr(s * a, s * b, s * v, s=s)
        out_1 = s * gm.gyr(a, b, v, s=1.0)
        np.testing.assert_allclose(out_s, out_1, atol=1e-10)


def test_flat_loop_holonomy_is_identity_for_colinear_origin():
    """A loop through the origin with colinear points has trivial holonomy."""
    d = 3
    feats = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0]])
    res = gm.frust_index_gyro(feats, [[0, 1, 2]])
    # colinear points → all transport in a common line → flat
    assert res["frust_total"] < 1e-8
    assert res["is_flat"]


def test_curved_loop_holonomy_nontrivial():
    """A genuinely 2-D triangle of features has nonzero holonomy (curvature)."""
    rng = np.random.default_rng(3)
    feats = np.stack([_rand_ball(3, rng) for _ in range(3)])
    res = gm.frust_index_gyro(feats, [[0, 1, 2]])
    assert res["frust_total"] > 1e-6
    assert not res["is_flat"]


def test_torch_gyr_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    d = 5
    A = np.stack([_rand_ball(d, rng) for _ in range(32)])
    B = np.stack([_rand_ball(d, rng) for _ in range(32)])
    V = np.stack([_rand_ball(d, rng) for _ in range(32)])
    out_np = gm.gyr(A, B, V)
    out_t = gm.gyr_torch(torch.tensor(A), torch.tensor(B), torch.tensor(V)).numpy()
    np.testing.assert_allclose(out_np, out_t, atol=1e-12)


def test_gyrobary_stays_in_ball_and_limits_to_mean():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    pts = torch.tensor(np.stack([_rand_ball(4, rng) for _ in range(6)]))[None]  # (1,6,4)
    w = torch.rand(1, 6)
    out = gm.gyrobary(w, pts)
    assert out.norm(dim=-1).item() < 1.0
    # large s → Euclidean weighted mean
    out_big = gm.gyrobary(w, pts, s=1e4)
    mean = (w.unsqueeze(-1) * pts).sum(-2) / w.sum(-1, keepdim=True)
    np.testing.assert_allclose(out_big.numpy(), mean.numpy(), atol=1e-3)


def test_mobius_matmul_identity_roundtrip():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    x = torch.tensor(np.stack([_rand_ball(4, rng) for _ in range(10)]))
    out = gm.mobius_matmul(torch.eye(4, dtype=x.dtype), x)
    np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-9)
