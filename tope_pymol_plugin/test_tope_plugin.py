"""
test_tope_plugin.py
====================
Standalone test for ToPE plugin logic that does NOT require a live PyMOL
session.  Tests:

  1. JSON output schema (load_tope_output)
  2. NPZ round-trip (save → load)
  3. Score normalisation (apply_residue_scores skipped — needs PyMOL)
  4. Mock live-run smoke test (skipped if ToPE not installed)

Run with:
    python test_tope_plugin.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import os
import unittest
import numpy as np

# Make sure plugin directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tope_viz.runner import load_tope_output, _validate_schema, _load_npz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_JSON = {
    "pdb_id": "4AKE",
    "chain": "A",
    "n_residues": 5,
    "voip": {
        "residue_scores": [8.2, 9.4, 12.1, 7.8, 14.3],
        "raw_voip": [8.2, 9.4, 12.1, 7.8, 14.3],
        "thresholds": {"high": 15.0, "low": 10.0}
    },
    "attention": {
        "residue_scores": [0.1, 0.5, 0.9, 0.2, 0.7],
        "substrate_smiles": "C1=CC=CC=C1",
    },
    "frustration": {
        "residue_scores": [0.3, 0.8, 0.2, 0.95, 0.1],
        "holonomy_norm": 0.95,
        "frustrated_residues": [1, 3],
    },
    "attribution": {
        "residue_scores": [0.01, 0.22, 0.88, 0.15, 0.44],
        "zone_importance": {"zone1": 0.6, "zone2": 0.3, "zone3": 0.1},
        "filtration_importance": {},
        "pathways": [[0, 2, 4]],
        "catalytic_residues": [2],
        "task": "kinetics",
        "mcsa_overlap": 0.85,
    },
    "kinetics": {
        "log_kcat": 3.2,
        "log_km": -5.1,
        "log_kcat_km": 8.3,
    }
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadJSON(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(MOCK_JSON, self.tmp)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_load_returns_dict(self):
        data = load_tope_output(self.tmp.name)
        self.assertIsInstance(data, dict)

    def test_voip_scores_are_floats(self):
        data = load_tope_output(self.tmp.name)
        scores = data["voip"]["residue_scores"]
        self.assertEqual(len(scores), 5)
        for s in scores:
            self.assertIsInstance(s, float)

    def test_all_layers_present(self):
        data = load_tope_output(self.tmp.name)
        for layer in ("voip", "attention", "frustration", "attribution"):
            self.assertIn(layer, data)

    def test_kinetics_present(self):
        data = load_tope_output(self.tmp.name)
        self.assertIn("kinetics", data)
        self.assertAlmostEqual(data["kinetics"]["log_kcat"], 3.2)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_tope_output("/nonexistent/path/tope.json")

    def test_unsupported_format_raises(self):
        # Copy the json to a .xyz path so the file exists but format is wrong
        import shutil
        xyz_path = self.tmp.name.replace(".json", ".xyz")
        shutil.copy(self.tmp.name, xyz_path)
        try:
            with self.assertRaises(ValueError):
                load_tope_output(xyz_path)
        finally:
            os.unlink(xyz_path)


class TestLoadNPZ(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        self.tmp.close()

        arrays = {
            "voip_residue_scores":        np.array([8.2, 9.4, 12.1, 7.8, 14.3]),
            "attention_residue_scores":   np.array([0.1, 0.5, 0.9, 0.2, 0.7]),
            "frustration_residue_scores": np.array([0.3, 0.8, 0.2, 0.95, 0.1]),
            "frustration_holonomy_norm":  np.array(0.95),
            "frustration_frustrated_residues": np.array([1, 3]),
            "attribution_residue_scores": np.array([0.01, 0.22, 0.88, 0.15, 0.44]),
            "attribution_catalytic_residues": np.array([2]),
        }
        np.savez(self.tmp.name, **arrays)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_load_npz(self):
        from pathlib import Path
        data = _load_npz(Path(self.tmp.name))
        self.assertIn("voip", data)
        self.assertEqual(len(data["voip"]["residue_scores"]), 5)

    def test_frustration_extras(self):
        from pathlib import Path
        data = _load_npz(Path(self.tmp.name))
        self.assertAlmostEqual(data["frustration"]["holonomy_norm"], 0.95)
        self.assertIn(1, data["frustration"]["frustrated_residues"])


class TestScoreNormalisation(unittest.TestCase):

    def test_validate_schema_converts_to_float(self):
        raw = {
            "voip": {"residue_scores": [1, 2, 3]},
            "attention": {"residue_scores": [0.1, "0.5", 0.9]},
        }
        out = _validate_schema(raw)
        for s in out["voip"]["residue_scores"]:
            self.assertIsInstance(s, float)
        for s in out["attention"]["residue_scores"]:
            self.assertIsInstance(s, float)


class TestLiveRunSkipped(unittest.TestCase):

    def test_tope_import_check(self):
        """run_tope_live raises ImportError if ToPE not installed."""
        try:
            import tope  # noqa: F401
            self.skipTest("ToPE is installed — skipping ImportError check.")
        except ImportError:
            pass

        from tope_viz.runner import run_tope_live
        with self.assertRaises(ImportError):
            run_tope_live("/fake.pdb", layers=["voip"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ToPE Plugin — standalone unit tests")
    print("=" * 60)
    unittest.main(verbosity=2)
