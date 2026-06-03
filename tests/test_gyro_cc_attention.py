"""Tests for the §3 gyro-CCANN attention block (GyroCCAttentionBlock).

Loads ``cc_attention`` and its ``tope.topology`` dependencies by file path,
registering lightweight package stubs in ``sys.modules`` so the intra-package
imports resolve without triggering the (pre-existing, unrelated) breakage in
the top-level ``tope`` package __init__.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # mark as package
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


# Stub the packages, then load the real sibling modules under their dotted names
# so `from tope.topology.gyro_memory import ...` inside cc_attention resolves.
_stub_pkg("tope", _REPO_ROOT / "tope")
_stub_pkg("tope.topology", _REPO_ROOT / "tope" / "topology")
_stub_pkg("tope.models", _REPO_ROOT / "tope" / "models")
gm = _load("tope.topology.gyro_memory", _REPO_ROOT / "tope" / "topology" / "gyro_memory.py")
_load("tope.topology.g_structure", _REPO_ROOT / "tope" / "topology" / "g_structure.py")
cca = _load("tope.models.cc_attention", _REPO_ROOT / "tope" / "models" / "cc_attention.py")


@pytest.fixture(autouse=True)
def _seed():
    # Deterministic inputs so the threshold-based assertions never flake on an
    # unlucky random draw (e.g. features landing where gyration ≈ identity).
    torch.manual_seed(0)


def test_gyro_available():
    assert cca.HAS_GYRO, "gyro_memory import failed inside cc_attention"


def _toy_incidence(N_s, N_t, rng):
    """Random bipartite incidence (2, E) with every t-cell covered."""
    edges = []
    for i in range(N_t):
        for j in rng.choice(N_s, size=2, replace=False):
            edges.append((int(j), i))
    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    tgt = torch.tensor([e[1] for e in edges], dtype=torch.long)
    return torch.stack([src, tgt])


@pytest.mark.parametrize("isotropic", [False, True])
def test_forward_shapes_and_ball_membership(isotropic):
    rng = np.random.default_rng(0)
    N_s, N_t, d_in, d_out = 6, 4, 8, 8
    blk = cca.GyroCCAttentionBlock(d_in, d_in, d_out, n_heads=2, isotropic=isotropic)
    H_s = torch.randn(N_s, d_in)
    H_t = torch.randn(N_t, d_in)
    B = _toy_incidence(N_s, N_t, rng)
    K_t, K_s = blk(H_s, H_t, B)
    assert K_t.shape == (N_t, d_out)
    assert K_s.shape == (N_s, d_out)
    s = blk.ball_radius
    hd = blk.head_dim
    # Each head's output must lie strictly inside its head_dim-ball.
    for K, N in [(K_t, N_t), (K_s, N_s)]:
        heads = K.view(N, blk.n_heads, hd)
        assert (heads.norm(dim=-1) < s + 1e-5).all()


def test_gradients_flow():
    rng = np.random.default_rng(1)
    blk = cca.GyroCCAttentionBlock(8, 8, 8, n_heads=2, isotropic=False)
    H_s = torch.randn(6, 8, requires_grad=True)
    H_t = torch.randn(4, 8, requires_grad=True)
    B = _toy_incidence(6, 4, rng)
    K_t, K_s = blk(H_s, H_t, B)
    (K_t.sum() + K_s.sum()).backward()
    assert blk.a.grad is not None and torch.isfinite(blk.a.grad).all()
    assert blk.log_s.grad is not None and torch.isfinite(blk.log_s.grad).all()
    assert blk.W_s.weight.grad is not None


def test_frame_correction_is_the_true_reverse_coordinate():
    """The mandatory correction −gyr[q,⊖p]·v equals log_o((⊖q)⊕p) (spec §3.1).

    This is the identity that makes the anisotropic block gyro-adjoint; the
    spec's literal gyr[p,⊖q] order fails it (kept here as a guard).
    """
    rng = np.random.default_rng(2)
    max_correct_err = 0.0
    max_wrong_err = 0.0
    for _ in range(500):
        d = 6
        p = torch.tensor(_rand_ball(d, rng))
        q = torch.tensor(_rand_ball(d, rng))
        v = gm.log_o(gm.mobius_add(-p, q))
        true_rev = gm.log_o(gm.mobius_add(-q, p))
        correct = -gm.gyr_torch(q, -p, v)
        wrong = -gm.gyr_torch(p, -q, v)
        max_correct_err = max(max_correct_err, (true_rev - correct).norm().item())
        max_wrong_err = max(max_wrong_err, (true_rev - wrong).norm().item())
    # The correct order is exact everywhere; the spec's literal gyr[p,⊖q] order
    # is materially wrong (the two collapse only as p,q → origin).
    assert max_correct_err < 1e-10, max_correct_err
    assert max_wrong_err > 1e-1, max_wrong_err


def test_anisotropic_uses_frame_correction():
    """Anisotropic reverse output must change if the gyration is dropped."""
    rng = np.random.default_rng(3)
    blk = cca.GyroCCAttentionBlock(8, 8, 8, n_heads=1, isotropic=False)
    H_s = torch.randn(6, 8)
    H_t = torch.randn(4, 8)
    B = _toy_incidence(6, 4, rng)
    _, K_s = blk(H_s, H_t, B)
    # Monkeypatch gyr to identity → drops the frame correction.
    orig = cca._gyr_torch
    try:
        cca._gyr_torch = lambda a, b, vv, s=1.0: vv
        _, K_s_noG = blk(H_s, H_t, B)
    finally:
        cca._gyr_torch = orig
    assert (K_s - K_s_noG).abs().max().item() > 1e-6


def _rand_ball(d, rng, rmax=0.8):
    x = rng.normal(size=d)
    x = x / np.linalg.norm(x)
    return (rng.uniform(0.0, rmax) * x).astype(np.float64)
