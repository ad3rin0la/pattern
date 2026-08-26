"""Label encoding + a full ToPEModel training step.

Closes the loop to a real optimisation step: curated arrays → ToPEDataset →
collate → ToPEModel.forward → encode_targets (EC vocab + kinetics) →
MultiTaskLoss → backward → optimizer.step.

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


# ── EC vocabulary unit tests ──────────────────────────────────────────────────

def test_ec_vocabulary_hierarchical_indices_and_sizes():
    ECVocabulary = _labels.ECVocabulary
    vocab = ECVocabulary.build(["3.4.21.1", "3.4.21.4", "1.1.1.1"])
    # Level 0: {3, 1} → 2 tokens + <unk> = 3.  Level 3: 3 distinct full numbers.
    assert vocab.level_sizes == [3, 3, 3, 4]   # [unk+{1,3}, unk+{1.1,3.4}, ...]

    i = vocab.encode("3.4.21.1")
    assert len(i) == 4 and all(idx > 0 for idx in i)        # all known
    # Shared prefix → shared level-0/1/2 indices with 3.4.21.4.
    j = vocab.encode("3.4.21.4")
    assert i[:3] == j[:3] and i[3] != j[3]


def test_ec_vocabulary_unknown_and_partial_degrade_to_unk():
    vocab = _labels.ECVocabulary.build(["2.7.11.1"])
    # Fully unseen EC → <unk> (0) at every level.
    assert vocab.encode("9.9.9.9") == [0, 0, 0, 0]
    # Partial ("2.7" then '-') → known at levels 0,1; <unk> deeper.
    enc = vocab.encode("2.7.-.-")
    assert enc[0] > 0 and enc[1] > 0 and enc[2] == 0 and enc[3] == 0


def test_encode_targets_builds_kinetics_mask():
    vocab = _labels.ECVocabulary.build(["3.4.21.1", "1.1.1.1"])
    labels = {
        "ec_number": ["3.4.21.1", "1.1.1.1"],
        "kinetics": torch.tensor([[1.0, float("nan"), 3.0],
                                  [float("nan"), float("nan"), float("nan")]]),
    }
    t = _labels.encode_targets(labels, vocab)
    assert len(t["ec_levels"]) == 4
    assert t["ec_levels"][0].tolist() == [vocab.encode("3.4.21.1")[0],
                                          vocab.encode("1.1.1.1")[0]]
    # Mask marks present values; missing → 0 in both mask and (zeroed) target.
    assert t["kinetics_mask"].tolist() == [[1, 0, 1], [0, 0, 0]]
    assert t["kinetics"][0, 1].item() == 0.0          # NaN replaced
    assert _labels.active_tasks_for(t) == {"ec": True, "kinetics": True,
                                           "selectivity": False}


def test_kinetics_targets_flow_into_loss():
    """The encoded kinetics target + mask drive MultiTaskLoss._kinetics_loss."""
    vocab = _labels.ECVocabulary.build(["3.4.21.1"])
    labels = {"ec_number": ["3.4.21.1"],
              "kinetics": torch.tensor([[2.0, float("nan"), 1.0]])}
    targets = _labels.encode_targets(labels, vocab)

    loss_fn = _losses.MultiTaskLoss()
    preds = {
        "ec_logits": [torch.zeros(1, n, requires_grad=True) for n in vocab.level_sizes],
        "selectivity": None,
        "kinetics": torch.zeros(1, 3, requires_grad=True),
    }
    out = loss_fn(preds, targets, active_tasks={"ec": True, "kinetics": True,
                                                "selectivity": False})
    assert torch.isfinite(out["total"])
    assert out["kinetics"].item() > 0          # masked MSE over the 2 present values
    out["total"].backward()
    assert preds["kinetics"].grad is not None


# ── Full ToPEModel training step ──────────────────────────────────────────────

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


def _build_loader(tmp_path):
    sites = [_make_site("PDB0", "3.4.21.1", 0), _make_site("PDB1", "1.1.1.1", 1)]
    fc = _features.FeatureComputer(compute_sasa=False, normalise=False)
    feats = [fc.compute(s) for s in sites]
    builder = _dataset.DatasetBuilder(
        output_dir=tmp_path / "processed", features_dir=tmp_path / "features",
        config=_config.PipelineConfig(output_format="json"),
    )
    builder.build(sites, feats, split_ratios=(1.0, 0.0, 0.0),
                  kinetics_map={"PDB0": {"kcat": 1.0, "Km": -2.0, "kcat/Km": 3.0}})
    return _loader.ToPEDataset.from_index(
        tmp_path / "processed" / "dataset_index.json", tmp_path / "features",
        edge_radius=6.0, edge_feat_dim=4,
    )


def test_full_tope_model_training_step(tmp_path):
    ds = _build_loader(tmp_path)
    batch = _loader.collate_enzyme_pcc([ds[0], ds[1]])

    # Fit the EC vocabulary over the dataset; it sizes the EC head.
    vocab = _labels.ECVocabulary.build([r["ec_number"] for r in ds.records])

    cfg = _tope_model.ToPEConfig(
        node_feat_dim=ds.node_feat_dim, edge_feat_dim=4, hidden_dim=16,
        n_tcpnet_layers=2, n_heads=2, ec_levels=vocab.level_sizes,
        head_hidden_dim=16, use_cross_attention=False, use_e3nn=False,
    )
    with pytest.warns(DeprecationWarning, match="active-site crop"):
        model = _tope_model.ToPEModel(cfg)
    loss_fn = _losses.MultiTaskLoss()
    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)

    model.train()
    predictions = model(batch)
    # EC logits exist at every level, sized by the fitted vocabulary.
    assert [t.shape[-1] for t in predictions["ec_logits"]] == vocab.level_sizes
    assert predictions["ec_logits"][0].shape[0] == 2          # batch of 2 graphs

    targets = _labels.encode_targets(batch["labels"], vocab)
    out = loss_fn(predictions, targets, active_tasks=_labels.active_tasks_for(targets))
    assert torch.isfinite(out["total"]) and out["total"].item() > 0

    opt.zero_grad()
    out["total"].backward()
    # Gradients reached the encoder (curated complex) and the EC head.
    enc_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.encoder.parameters())
    assert enc_grad
    opt.step()
