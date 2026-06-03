"""Regression tests for the incomplete-restructure import bugs.

Two symbols were referenced across ``tope.data`` but missing after the
atom-level → residue-level ``ActiveSite`` refactor (commit 9594424) and the
package restructure (b8a2c8d):

  * ``AtomRecord`` — deleted from ``active_site.py`` but still imported and used
    by ``features.py`` (which is atom-level);
  * ``CurationConfig`` — referenced by the CLI / package ``__init__`` / docstring
    examples but never defined (``config.py`` only had ``PipelineConfig``).

Both broke ``import tope`` outright.  These tests load the two modules by file
path (the rest of the package needs the optional ``torch_scatter`` dependency,
unrelated to these fixes) and lock in the contracts the consumers rely on.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_pkg(name: str, path: Path) -> None:
    """Register a package stub with a real __path__ so intra-package absolute
    imports resolve to the real source files without executing the top-level
    ``tope`` package __init__ (which needs the optional torch_scatter dep)."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod._gyro_test_stub = True  # tagged so conftest purges it (see conftest.py)
        sys.modules[name] = mod


_stub_pkg("tope", _REPO_ROOT / "tope")
_stub_pkg("tope.data", _REPO_ROOT / "tope" / "data")
_config = importlib.import_module("tope.data.config")
_active_site = importlib.import_module("tope.data.active_site")
_features = importlib.import_module("tope.data.features")
_dataset = importlib.import_module("tope.data.dataset")


def test_atomrecord_exists_with_feature_contract():
    """features.FeatureComputer reads .element/.residue_name/.residue_id/
    .is_catalytic/.coord on each atom — AtomRecord must provide them."""
    AtomRecord = _active_site.AtomRecord
    a = AtomRecord(
        serial=1, name="NZ", element="N", residue_name="LYS",
        residue_number=42, chain_id="A", coord=np.zeros(3),
    )
    assert a.element == "N"
    assert a.residue_name == "LYS"
    assert a.residue_id == "A:LYS42"        # property used as a SASA-map key
    assert a.is_catalytic is False
    assert a.coord.shape == (3,)


def test_curationconfig_is_pipelineconfig_with_output_dir():
    CurationConfig = _config.CurationConfig
    PipelineConfig = _config.PipelineConfig

    # Subclass → drop-in for CurationPipeline (which reads PipelineConfig fields).
    c = CurationConfig(output_dir="./run42")
    assert isinstance(c, PipelineConfig)
    # output_dir bridges onto data_root (the field the pipeline actually uses).
    assert str(c.data_root) == "run42" or c.data_root == Path("./run42")

    # No output_dir → unchanged default.
    assert CurationConfig().data_root == PipelineConfig().data_root

    # Accepts pipeline fields and the CLI's **config_dict construction.
    e = CurationConfig(**{"output_dir": "./y", "max_resolution": 2.0})
    assert e.max_resolution == 2.0
    assert e.data_root == Path("./y")


# ── Simultaneous atom + residue representation (combinatorial complex) ─────────

def _toy_site():
    """Hand-build an ActiveSite carrying both ranks: 3 atoms across 2 residues."""
    AtomRecord = _active_site.AtomRecord
    ResidueRecord = _active_site.ResidueRecord
    ActiveSite = _active_site.ActiveSite

    residues = [
        ResidueRecord(chain_id="A", residue_name="HIS", residue_number=57,
                      ca_coord=np.array([0.0, 0.0, 0.0]), n_atoms=2, is_catalytic=True),
        ResidueRecord(chain_id="A", residue_name="ASP", residue_number=102,
                      ca_coord=np.array([3.0, 0.0, 0.0]), n_atoms=1, is_catalytic=False),
    ]
    atoms = [
        AtomRecord(1, "ND1", "N", "HIS", 57, "A", np.array([0.1, 0.0, 0.0]), is_catalytic=True),
        AtomRecord(2, "CE1", "C", "HIS", 57, "A", np.array([0.2, 0.1, 0.0]), is_catalytic=True),
        AtomRecord(3, "OD1", "O", "ASP", 102, "A", np.array([3.1, 0.0, 0.0]), is_catalytic=False),
    ]
    return ActiveSite(pdb_id="TEST", residues=residues, atoms=atoms,
                      catalytic_residue_ids={"A:HIS57"}, ec_number="3.4.21.1")


def test_active_site_carries_both_ranks_with_incidence():
    site = _toy_site()
    # Both ranks present simultaneously.
    assert site.n_residues == 2
    assert site.n_atoms == 3                      # uses the materialised atoms
    assert site.atoms_array().shape == (3, 3)
    assert site.ca_coords_array().shape == (2, 3)

    # Atom→residue incidence (rank 0 → rank 2 membership) is exact.
    b = site.atom_residue_incidence()
    assert list(b) == [0, 0, 1]                   # 2 atoms in HIS(0), 1 in ASP(1)
    # Membership keys on residue_id, consistent across both ranks.
    assert site.atoms[0].residue_id == site.residues[0].residue_id == "A:HIS57"


def test_n_atoms_falls_back_to_residue_summary_when_atoms_absent():
    ResidueRecord = _active_site.ResidueRecord
    ActiveSite = _active_site.ActiveSite
    site = ActiveSite(pdb_id="R", residues=[
        ResidueRecord("A", "GLY", 1, np.zeros(3), n_atoms=4),
    ])  # no atoms populated
    assert site.n_atoms == 4                      # backward-compatible fallback


def test_feature_computer_consumes_the_atom_rank():
    """FeatureComputer.compute() reads active_site.atoms — the rank-0 cells —
    so the simultaneous representation makes the previously-broken path run."""
    site = _toy_site()
    fc = _features.FeatureComputer(compute_sasa=False, normalise=False)
    result = fc.compute(site)
    assert result.n_atoms == 3
    assert result.feature_matrix.shape == (3, 8)  # 8 Ioffe properties per atom
    assert result.coords.shape == (3, 3)
    assert result.catalytic_mask.tolist() == [True, True, False]
    # Atom-level feature records preserve the residue linkage.
    assert result.atom_features[0].residue_id == "A:HIS57"


# ── Restored dataset-layer members (refactor debt) ────────────────────────────

def _metal_site():
    AtomRecord = _active_site.AtomRecord
    ResidueRecord = _active_site.ResidueRecord
    ActiveSite = _active_site.ActiveSite
    residues = [ResidueRecord("A", "CYS", 10, np.zeros(3), n_atoms=2, is_catalytic=True)]
    atoms = [
        AtomRecord(1, "SG", "S", "CYS", 10, "A", np.array([0.0, 0.0, 0.0])),
        AtomRecord(2, "FE", "FE", "FES", 11, "A", np.array([2.0, 0.0, 0.0])),
    ]
    return ActiveSite(pdb_id="MET", residues=residues, atoms=atoms)


def test_elements_and_has_metal():
    site = _metal_site()
    assert site.elements == {"S", "FE"}
    assert site.has_metal is True
    # A non-metal site
    plain = _toy_site()
    assert plain.has_metal is False


def test_distance_matrix_and_filtration_adjacencies():
    site = _toy_site()  # 3 atoms
    D = site.distance_matrix()
    assert D.shape == (3, 3)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
    np.testing.assert_allclose(D, D.T)  # symmetric

    adjs = _active_site.ActiveSiteExtractor.compute_filtration_adjacencies(
        site, radii=[0.05, 10.0]
    )
    assert set(adjs) == {0.05, 10.0}
    # Tiny radius → only self-adjacency; large radius → fully connected.
    assert adjs[0.05].dtype == np.int32
    assert int(adjs[0.05].sum()) == 3        # diagonal only
    assert (adjs[10.0] == 1).all()


def test_dataset_build_persists_atom_residue(tmp_path):
    """Full DatasetBuilder.build round-trip writes atom_residue.npy aligned with
    the saved atom features, and records its path — so the incidence survives to
    disk and can drive tcpnet's B_12 downstream."""
    site = _toy_site()
    feats = _features.FeatureComputer(compute_sasa=False, normalise=False).compute(site)

    PipelineConfig = _config.PipelineConfig
    builder = _dataset.DatasetBuilder(
        output_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        config=PipelineConfig(output_format="json"),
    )
    records = builder.build([site], [feats], split_ratios=(0.0, 0.0, 1.0))
    assert len(records) == 1
    rec = records[0]

    # The record carries the new path, and it points at a real array.
    assert rec.atom_residue_path
    arr_path = (tmp_path / "features") / rec.atom_residue_path
    assert arr_path.exists()
    loaded = np.load(arr_path)
    np.testing.assert_array_equal(loaded, site.atom_residue_incidence())

    # Aligned 1:1 with the saved atom features / coords.
    assert loaded.shape[0] == feats.feature_matrix.shape[0] == site.n_atoms
    # Restored summary fields populate the record.
    assert rec.n_atoms == 3 and rec.n_residues == 2
    assert rec.has_metal is False

    # Serialised dict includes the new field.
    d = _dataset.DatasetBuilder._record_to_dict(rec)
    assert "atom_residue_path" in d
