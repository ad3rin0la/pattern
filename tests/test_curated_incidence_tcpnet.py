"""Tests for the curated atom→residue incidence driving tcpnet's rank-2
boundary operator (B_12).

Verifies the data-driven boundary-operator path: the extractor's
``ActiveSite.atom_residue_incidence()`` (rank-0 → rank-2 membership) flows into
``EnzymeTCPNet`` and constructs ``B_12`` (bonds→residues) directly, instead of
the model re-deriving residues from a synthetic ``face_to_edge`` triangulation.

Loaded by file path with package stubs (see conftest.py for isolation).
"""

from __future__ import annotations

import importlib
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


_stub_pkg("tope", _REPO_ROOT / "tope")
_stub_pkg("tope.topology", _REPO_ROOT / "tope" / "topology")
_stub_pkg("tope.models", _REPO_ROOT / "tope" / "models")
_stub_pkg("tope.data", _REPO_ROOT / "tope" / "data")
importlib.import_module("tope.topology.gyro_memory")
importlib.import_module("tope.topology.g_structure")
importlib.import_module("tope.models.cc_attention")
tcp = importlib.import_module("tope.models.tcpnet")
_active_site = importlib.import_module("tope.data.active_site")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _toy_site():
    """3 atoms in 2 residues: atoms 0,1 ∈ HIS(0); atom 2 ∈ ASP(1)."""
    AtomRecord = _active_site.AtomRecord
    ResidueRecord = _active_site.ResidueRecord
    ActiveSite = _active_site.ActiveSite
    residues = [
        ResidueRecord("A", "HIS", 57, np.zeros(3), n_atoms=2, is_catalytic=True),
        ResidueRecord("A", "ASP", 102, np.array([3.0, 0, 0]), n_atoms=1),
    ]
    atoms = [
        AtomRecord(1, "ND1", "N", "HIS", 57, "A", np.array([0.1, 0, 0])),
        AtomRecord(2, "CE1", "C", "HIS", 57, "A", np.array([0.2, 0.1, 0])),
        AtomRecord(3, "OD1", "O", "ASP", 102, "A", np.array([3.1, 0, 0])),
    ]
    return ActiveSite(pdb_id="T", residues=residues, atoms=atoms)


def test_b12_from_membership_matches_endpoint_residues():
    site = _toy_site()
    atom_residue = torch.tensor(site.atom_residue_incidence())   # [0, 0, 1]
    # Bonds: (0,1) intra-HIS, (1,2) HIS–ASP bridge.
    edge_index = torch.tensor([[0, 1], [1, 2]])
    B_12 = tcp._build_B_12_from_atom_residue(edge_index, atom_residue)

    pairs = {tuple(p) for p in B_12.t().tolist()}
    # bond 0 = (0,1): both in residue 0 → single (0,0) after dedup.
    # bond 1 = (1,2): residues 0 and 1 → (1,0) and (1,1).
    assert pairs == {(0, 0), (1, 0), (1, 1)}
    # No residue index exceeds the residue count.
    assert int(B_12[1].max()) < site.n_residues


def test_b12_skips_absent_parent_residue():
    atom_residue = torch.tensor([0, -1, 1])   # atom 1's residue absent
    edge_index = torch.tensor([[0, 1], [1, 2]])
    B_12 = tcp._build_B_12_from_atom_residue(edge_index, atom_residue)
    pairs = {tuple(p) for p in B_12.t().tolist()}
    # bond 0=(0,1): only atom 0 valid → (0,0). bond 1=(1,2): only atom 2 → (1,1).
    assert pairs == {(0, 0), (1, 1)}


def test_b12_empty_edges():
    B_12 = tcp._build_B_12_from_atom_residue(
        torch.zeros(2, 0, dtype=torch.long), torch.tensor([0, 1])
    )
    assert B_12.shape == (2, 0)


def _enzyme_pcc(site, n_node_feat, n_edge_feat, with_membership):
    N = site.n_atoms
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 0]])  # 3 bonds
    pcc = {
        "node_features": torch.randn(N, n_node_feat),
        "edge_index": edge_index,
        "edge_features": torch.randn(edge_index.size(1), n_edge_feat),
        "pos": torch.tensor(site.atoms_array(), dtype=torch.float32),
    }
    if with_membership:
        pcc["atom_residue"] = torch.tensor(site.atom_residue_incidence())
    return pcc


def test_enzyme_tcpnet_consumes_curated_incidence_end_to_end():
    cfg = tcp.TCPNetConfig(node_feat_dim=8, edge_feat_dim=4, hidden_dim=16, n_layers=2)
    model = tcp.EnzymeTCPNet(cfg).eval()
    site = _toy_site()

    pcc = _enzyme_pcc(site, 8, 4, with_membership=True)
    emb, h_nodes = model(pcc)
    assert emb.shape[-1] == cfg.hidden_dim
    assert h_nodes.shape == (site.n_atoms, cfg.hidden_dim)


def test_membership_path_overrides_face_triangulation():
    """When atom_residue is present it drives B_12 — verified by patching the
    builder and confirming it is the one invoked (not _build_B_12)."""
    cfg = tcp.TCPNetConfig(node_feat_dim=8, edge_feat_dim=4, hidden_dim=16, n_layers=1)
    model = tcp.EnzymeTCPNet(cfg).eval()
    site = _toy_site()
    pcc = _enzyme_pcc(site, 8, 4, with_membership=True)
    # Also supply a face_to_edge; the membership path must take precedence.
    pcc["face_to_edge"] = torch.tensor([[0], [1], [2]])

    called = {"membership": 0, "face": 0}
    orig_m, orig_f = tcp._build_B_12_from_atom_residue, tcp._build_B_12
    try:
        def spy_m(ei, ar):
            called["membership"] += 1
            return orig_m(ei, ar)

        def spy_f(fte):
            called["face"] += 1
            return orig_f(fte)

        tcp._build_B_12_from_atom_residue = spy_m
        tcp._build_B_12 = spy_f
        model(pcc)
    finally:
        tcp._build_B_12_from_atom_residue = orig_m
        tcp._build_B_12 = orig_f

    assert called["membership"] == 1
    assert called["face"] == 0
