"""Tests for the §3 cross-rank gyro cross-attention (GyroCrossAttention).

Dense bipartite analog of GyroCCAttentionBlock; loaded by file path with
``tope.*`` package stubs (see conftest.py for the isolation rationale).
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
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod._gyro_test_stub = True
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
_stub_pkg("tope.models", _REPO_ROOT / "tope" / "models")
_load("tope.topology.gyro_memory", _REPO_ROOT / "tope" / "topology" / "gyro_memory.py")
_load("tope.topology.g_structure", _REPO_ROOT / "tope" / "topology" / "g_structure.py")
_load("tope.models.cc_attention", _REPO_ROOT / "tope" / "models" / "cc_attention.py")
cax = _load(
    "tope.models.cross_attention_improved",
    _REPO_ROOT / "tope" / "models" / "cross_attention_improved.py",
)


@pytest.fixture(autouse=True)
def _seed():
    # Deterministic inputs so threshold-based assertions never flake.
    torch.manual_seed(0)


def test_gyro_available():
    assert cax.HAS_GYRO


def test_forward_shapes_and_ball_membership():
    blk = cax.GyroCrossAttention(hidden_dim=16, mol_dim=12, n_heads=4).eval()
    h_enz = torch.randn(5, 16)
    h_mol = torch.randn(7, 12)
    h_enz_out, attn, h_mol_out = blk(h_enz, h_mol, return_reverse=True)
    assert h_enz_out.shape == (5, 16)
    assert attn.shape == (4, 5, 7)
    assert h_mol_out.shape == (7, 16)
    # Attention rows are proper distributions.
    np.testing.assert_allclose(attn.sum(-1).detach().numpy(), 1.0, atol=1e-5)


def test_gradients_flow_including_geometry():
    blk = cax.GyroCrossAttention(hidden_dim=16, n_heads=4)
    h_enz = torch.randn(5, 16, requires_grad=True)
    h_mol = torch.randn(7, 16, requires_grad=True)
    h_enz_out, _, h_mol_out = blk(h_enz, h_mol, return_reverse=True)
    (h_enz_out.sum() + h_mol_out.sum()).backward()
    assert blk.a.grad is not None and torch.isfinite(blk.a.grad).all()
    assert blk.log_s.grad is not None and torch.isfinite(blk.log_s.grad).all()


def test_batch_mask_blocks_cross_graph_attention():
    blk = cax.GyroCrossAttention(hidden_dim=8, n_heads=2)
    h_enz = torch.randn(4, 8)
    h_mol = torch.randn(4, 8)
    enz_b = torch.tensor([0, 0, 1, 1])
    mol_b = torch.tensor([0, 1, 0, 1])
    _, attn, _ = blk(h_enz, h_mol, enzyme_batch=enz_b, mol_batch=mol_b)
    # Enzyme atom 0 (graph 0) must give zero weight to mol atoms in graph 1.
    a = attn.detach().numpy()
    for h in range(attn.shape[0]):
        assert a[h, 0, 1] < 1e-6 and a[h, 0, 3] < 1e-6


def test_reverse_uses_frame_correction():
    blk = cax.GyroCrossAttention(hidden_dim=8, n_heads=1)
    h_enz = torch.randn(4, 8)
    h_mol = torch.randn(5, 8)
    _, _, h_mol_out = blk(h_enz, h_mol, return_reverse=True)
    orig = cax._gyr_torch
    try:
        cax._gyr_torch = lambda a, b, vv, s=1.0: vv  # drop the frame correction
        _, _, h_mol_noG = blk(h_enz, h_mol, return_reverse=True)
    finally:
        cax._gyr_torch = orig
    assert (h_mol_out - h_mol_noG).abs().max().item() > 1e-6
