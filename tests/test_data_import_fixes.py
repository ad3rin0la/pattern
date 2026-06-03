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
