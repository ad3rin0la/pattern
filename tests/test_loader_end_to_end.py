"""End-to-end curate→train loop: on-disk arrays → enzyme_pcc → model forward.

Builds a tiny dataset to disk with DatasetBuilder, reconstructs the model input
with ToPEDataset + collate_enzyme_pcc, and forwards a batch through
EnzymeTCPNet — verifying the persisted atom→residue incidence drives B_12 with
no model-side re-derivation, and that multi-graph batching offsets indices
correctly.

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
_features = importlib.import_module("tope.data.features")
_dataset = importlib.import_module("tope.data.dataset")
_loader = importlib.import_module("tope.data.loader")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_site(pdb_id, ec, seed):
    rng = np.random.default_rng(seed)
    AtomRecord = _active_site.AtomRecord
    ResidueRecord = _active_site.ResidueRecord
    ActiveSite = _active_site.ActiveSite
    # 2 residues, 2 atoms each, within a few Å so the radius graph has edges.
    residues, atoms = [], []
    for ri, (rname, rnum, cat) in enumerate([("HIS", 57, True), ("ASP", 102, False)]):
        base = np.array([ri * 2.0, 0.0, 0.0])
        residues.append(ResidueRecord("A", rname, rnum, base, n_atoms=2, is_catalytic=cat))
        for ai, elem in enumerate(["N", "C"]):
            atoms.append(AtomRecord(len(atoms) + 1, elem + str(ai), elem, rname, rnum,
                                    "A", base + rng.normal(scale=0.3, size=3)))
    return ActiveSite(pdb_id=pdb_id, residues=residues, atoms=atoms, ec_number=ec)


def _build_dataset(tmp_path, n=2):
    sites = [_make_site(f"PDB{i}", f"3.4.21.{i}", seed=i) for i in range(n)]
    fc = _features.FeatureComputer(compute_sasa=False, normalise=False)
    feats = [fc.compute(s) for s in sites]
    builder = _dataset.DatasetBuilder(
        output_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        config=_config_json(),
    )
    builder.build(sites, feats, split_ratios=(0.0, 0.0, 1.0),
                  kinetics_map={"PDB0": {"kcat": 1.0, "Km": -2.0, "kcat/Km": 3.0}})
    return tmp_path / "processed" / "dataset_index.json", tmp_path / "features"


def _config_json():
    cfg_mod = importlib.import_module("tope.data.config")
    return cfg_mod.PipelineConfig(output_format="json")


def test_dataset_item_builds_enzyme_pcc(tmp_path):
    index, fdir = _build_dataset(tmp_path, n=1)
    ds = _loader.ToPEDataset.from_index(index, fdir, edge_radius=6.0, edge_feat_dim=16)
    assert len(ds) == 1
    assert ds.node_feat_dim == 8

    item = ds[0]
    pcc = item["enzyme_pcc"]
    N = pcc["node_features"].size(0)
    assert N == 4                                   # 2 residues × 2 atoms
    assert pcc["pos"].shape == (4, 3)
    assert pcc["edge_features"].shape[1] == 16
    assert pcc["edge_index"].shape[0] == 2
    # Bonds derived from coords are symmetric (both directions present).
    assert pcc["edge_index"].shape[1] > 0
    # atom_residue is the persisted incidence: atoms 0,1 → res 0; 2,3 → res 1.
    assert pcc["atom_residue"].tolist() == [0, 0, 1, 1]


def test_collate_offsets_indices_across_graphs(tmp_path):
    index, fdir = _build_dataset(tmp_path, n=2)
    ds = _loader.ToPEDataset.from_index(index, fdir, edge_radius=6.0, edge_feat_dim=16)
    batch = _loader.collate_enzyme_pcc([ds[0], ds[1]])
    pcc = batch["enzyme_pcc"]

    assert pcc["node_features"].size(0) == 8        # 4 + 4 atoms
    # batch membership: first graph 0, second graph 1.
    assert pcc["batch"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # Edge indices of graph 1 are offset by 4 (no edge crosses graphs).
    ei = pcc["edge_index"]
    assert int(ei.max()) < 8
    # atom_residue offset: graph 0 → {0,1}; graph 1 → {2,3}.
    assert pcc["atom_residue"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert pcc["n_residues"] == 4
    # Labels carried through.
    assert batch["labels"]["kinetics"].shape == (2, 3)


def test_full_loop_loader_to_enzyme_tcpnet(tmp_path):
    """The reconstructed batch forwards through the encoder, and the persisted
    incidence drives B_12 (no face_to_edge present)."""
    index, fdir = _build_dataset(tmp_path, n=2)
    ds = _loader.ToPEDataset.from_index(index, fdir, edge_radius=6.0, edge_feat_dim=4)
    batch = _loader.collate_enzyme_pcc([ds[0], ds[1]])

    cfg = tcp.TCPNetConfig(node_feat_dim=ds.node_feat_dim, edge_feat_dim=4,
                           hidden_dim=16, n_layers=2)
    model = tcp.EnzymeTCPNet(cfg).eval()

    called = {"membership": 0}
    orig = tcp._build_B_12_from_atom_residue
    try:
        def spy(ei, ar):
            called["membership"] += 1
            return orig(ei, ar)
        tcp._build_B_12_from_atom_residue = spy
        emb, h_nodes = model(batch["enzyme_pcc"])
    finally:
        tcp._build_B_12_from_atom_residue = orig

    # Pooled to one embedding per graph (2 graphs in the batch).
    assert emb.shape == (2, cfg.hidden_dim)
    assert h_nodes.shape == (8, cfg.hidden_dim)
    assert called["membership"] == 1                # curated incidence drove B_12


def test_split_filter(tmp_path):
    index, fdir = _build_dataset(tmp_path, n=2)
    # All samples were assigned to "test" by split_ratios=(0,0,1).
    test_ds = _loader.ToPEDataset.from_index(index, fdir, split="test")
    train_ds = _loader.ToPEDataset.from_index(index, fdir, split="train")
    assert len(test_ds) == 2
    assert len(train_ds) == 0
