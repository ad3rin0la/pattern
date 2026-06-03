"""Substrate/product molecule graphs: SMILES featurisation → loader → cross-
attention → kinetics/selectivity heads.

Covers the dependency-free SMILES parser, the loader emitting + collating
molecule graphs, SMILES persistence in the dataset, and a full ToPEModel
training step with cross-attention enabled so the kinetics head trains.

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


for _n, _p in [
    ("tope", _REPO_ROOT / "tope"),
    ("tope.topology", _REPO_ROOT / "tope" / "topology"),
    ("tope.models", _REPO_ROOT / "tope" / "models"),
    ("tope.data", _REPO_ROOT / "tope" / "data"),
    ("tope.training", _REPO_ROOT / "tope" / "training"),
]:
    _stub_pkg(_n, _p)
for _dep in ["tope.topology.gyro_memory", "tope.topology.g_structure",
             "tope.models.cc_attention"]:
    importlib.import_module(_dep)
_mol = importlib.import_module("tope.data.molecule")
_active_site = importlib.import_module("tope.data.active_site")
_features = importlib.import_module("tope.data.features")
_dataset = importlib.import_module("tope.data.dataset")
_loader = importlib.import_module("tope.data.loader")
_config = importlib.import_module("tope.data.config")
_labels = importlib.import_module("tope.training.labels")
_losses = importlib.import_module("tope.training.losses")
_tope_model = importlib.import_module("tope.models.tope_model")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ── SMILES parser / featuriser ────────────────────────────────────────────────

@pytest.mark.parametrize("smiles,n_atoms,n_bonds", [
    ("CCO", 3, 2),                  # ethanol
    ("c1ccccc1", 6, 6),             # benzene (ring closure)
    ("CC(=O)O", 4, 3),             # acetic acid (branch + double bond)
    ("C#N", 2, 1),                  # triple bond
    ("[NH4+]", 1, 0),              # bracket atom + charge
    ("c1ccc2ccccc2c1", 10, 11),    # naphthalene (fused rings)
    ("[Na+].[Cl-]", 2, 0),         # disconnected
])
def test_parse_smiles_connectivity(smiles, n_atoms, n_bonds):
    atoms, bonds = _mol.parse_smiles(smiles)
    assert len(atoms) == n_atoms
    assert len(bonds) == n_bonds


def test_smiles_to_graph_shapes_and_undirected():
    feats, ei = _mol.smiles_to_graph("CC(=O)O")
    assert feats.shape == (4, _mol.MOL_FEAT_DIM)
    assert ei.shape == (2, 6)                       # 3 bonds × 2 directions
    # Undirected: the reverse of every edge is present.
    edges = {tuple(c) for c in ei.T.tolist()}
    assert all((j, i) in edges for i, j in edges)


def test_invalid_smiles_degrades_to_dummy():
    feats, ei = _mol.smiles_to_graph("not-a-molecule @@@")
    assert feats.shape == (1, _mol.MOL_FEAT_DIM)
    assert ei.shape == (2, 0)


# ── Loader: molecule emission + collation ─────────────────────────────────────

def _make_site(pdb_id, ec, seed):
    rng = np.random.default_rng(seed)
    AtomRecord = _active_site.AtomRecord
    ResidueRecord = _active_site.ResidueRecord
    ActiveSite = _active_site.ActiveSite
    residues, atoms = [], []
    for ri, (rname, rnum, cat) in enumerate([("HIS", 57, True), ("ASP", 102, False)]):
        base = np.array([ri * 2.0, 0.0, 0.0])
        residues.append(ResidueRecord("A", rname, rnum, base, n_atoms=2, is_catalytic=cat))
        for ai, elem in enumerate(["N", "C"]):
            atoms.append(AtomRecord(len(atoms) + 1, elem + str(ai), elem, rname, rnum,
                                    "A", base + rng.normal(scale=0.3, size=3)))
    return ActiveSite(pdb_id=pdb_id, residues=residues, atoms=atoms, ec_number=ec)


def _build(tmp_path, with_mol=True):
    sites = [_make_site("PDB0", "3.4.21.1", 0), _make_site("PDB1", "1.1.1.1", 1)]
    fc = _features.FeatureComputer(compute_sasa=False, normalise=False)
    feats = [fc.compute(s) for s in sites]
    builder = _dataset.DatasetBuilder(
        output_dir=tmp_path / "processed", features_dir=tmp_path / "features",
        config=_config.PipelineConfig(output_format="json"),
    )
    mol_map = {
        "PDB0": {"substrate": "CCO", "product": "CC=O"},
        "PDB1": {"substrate": "c1ccccc1", "product": ""},   # missing product
    } if with_mol else None
    builder.build(sites, feats, split_ratios=(1.0, 0.0, 0.0),
                  kinetics_map={"PDB0": {"kcat": 1.0, "Km": -2.0, "kcat/Km": 3.0}},
                  molecule_map=mol_map)
    return tmp_path / "processed" / "dataset_index.json", tmp_path / "features"


def test_dataset_persists_smiles(tmp_path):
    import json
    index, _ = _build(tmp_path)
    recs = json.loads(Path(index).read_text())
    by_id = {r["pdb_id"]: r for r in recs}
    assert by_id["PDB0"]["substrate_smiles"] == "CCO"
    assert by_id["PDB0"]["product_smiles"] == "CC=O"
    assert by_id["PDB1"]["product_smiles"] == ""


def test_loader_emits_and_collates_molecule_graphs(tmp_path):
    index, fdir = _build(tmp_path)
    ds = _loader.ToPEDataset.from_index(index, fdir, edge_radius=6.0, edge_feat_dim=4,
                                        with_molecules=True)
    assert ds.mol_feat_dim == _mol.MOL_FEAT_DIM
    item = ds[0]
    assert item["substrate"]["node_features"].shape == (3, _mol.MOL_FEAT_DIM)  # CCO
    assert item["product"]["node_features"].shape[0] == 3                       # CC=O

    batch = _loader.collate_enzyme_pcc([ds[0], ds[1]])
    sub = batch["substrate"]
    # CCO (3) + benzene (6) = 9 substrate atoms across 2 graphs.
    assert sub["node_features"].size(0) == 9
    assert sub["batch"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    # Graph-2 edges are offset by 3 (no edge crosses molecules).
    assert int(sub["edge_index"].max()) < 9
    # Missing product (PDB1) degrades to a dummy atom, so product batch has 3+1.
    assert batch["product"]["node_features"].size(0) == 4


def test_loader_without_molecules_omits_keys(tmp_path):
    index, fdir = _build(tmp_path)
    ds = _loader.ToPEDataset.from_index(index, fdir, with_molecules=False)
    assert "substrate" not in ds[0]


# ── Full ToPEModel step with cross-attention + kinetics head ──────────────────

def test_full_model_step_with_kinetics_head(tmp_path):
    index, fdir = _build(tmp_path)
    ds = _loader.ToPEDataset.from_index(index, fdir, edge_radius=6.0, edge_feat_dim=4,
                                        with_molecules=True)
    batch = _loader.collate_enzyme_pcc([ds[0], ds[1]])
    vocab = _labels.ECVocabulary.build([r["ec_number"] for r in ds.records])

    cfg = _tope_model.ToPEConfig(
        node_feat_dim=ds.node_feat_dim, edge_feat_dim=4, hidden_dim=16,
        n_tcpnet_layers=2, n_heads=2, ec_levels=vocab.level_sizes,
        head_hidden_dim=16, use_cross_attention=True, use_e3nn=False,
        mol_feat_dim=ds.mol_feat_dim, mol_hidden_dim=16, mol_n_layers=2,
    )
    model = _tope_model.ToPEModel(cfg)
    loss_fn = _losses.MultiTaskLoss()
    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)

    model.train()
    predictions = model(batch)
    # Cross-attention ran → kinetics head produced predictions.
    assert predictions["kinetics"] is not None
    assert predictions["kinetics"].shape == (2, 3)

    targets = _labels.encode_targets(batch["labels"], vocab)
    active = _labels.active_tasks_for(targets)
    assert active["kinetics"] is True                     # PDB0 has kinetics
    out = loss_fn(predictions, targets, active_tasks=active)
    assert torch.isfinite(out["total"])
    assert out["kinetics"].item() > 0                      # kinetics head supervised

    opt.zero_grad()
    out["total"].backward()
    # Gradients reached the substrate molecule GNN (the new path).
    sub_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.cross_attention.parameters())
    assert sub_grad
    opt.step()
