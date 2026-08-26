"""
Dataset assembly and storage.

Collects processed active-site features into structured dataset formats
suitable for downstream topological encoding (Phase 2) and model training
(Phase 3). Supports Parquet, CSV, and HDF5 output, plus a native NumPy
archive for direct consumption by PyTorch / PyG data loaders.

The dataset schema follows the specificity-preserving ToPE convention:
    - One row per enzyme–substrate–condition observation (not per protein)
    - Atomic-level data stored as variable-length arrays within each row
    - Metadata columns for PDB ID, EC number, split, and kinetics labels
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tope.data.active_site import ActiveSite
from tope.data.config import EC_TOP_LEVEL, PROCESSED_DIR, FEATURES_DIR, PipelineConfig
from tope.data.features import ActiveSiteFeatures

logger = logging.getLogger(__name__)


def _observations_for_site(
    kinetics_map: Dict[Any, Dict[str, Any]], pdb_id: str,
) -> List[Tuple[Any, Dict[str, Any]]]:
    """Select canonical observations for one structure.

    String-keyed maps remain supported as a legacy single-observation format.
    New maps are keyed by ``ObservationKey`` and therefore expand one enzyme
    structure into one dataset row per substrate/condition assay.
    """
    if pdb_id in kinetics_map:
        return [(None, kinetics_map[pdb_id])]
    pid = pdb_id.upper()
    return [
        (key, values)
        for key, values in kinetics_map.items()
        if getattr(key, "enzyme_id", "").upper() == pid
    ]

# Optional heavy I/O libraries.
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


# ── Dataset record ────────────────────────────────────────────────────────────

@dataclass
class DatasetRecord:
    """One enzyme–substrate assay row linked to processed structure arrays."""

    pdb_id: str
    ec_number: str
    ec_top_level: str
    n_atoms: int
    n_residues: int
    n_catalytic_residues: int
    has_metal: bool
    elements: List[str]

    # Paths to per-sample numpy files (filled during serialisation)
    coords_path: str = ""
    features_path: str = ""
    mask_path: str = ""
    adjacency_dir: str = ""
    atom_residue_path: str = ""   # rank-0 → rank-2 incidence (drives B_12)

    # Kinetics labels (log10 values; None if unavailable)
    log_kcat: Optional[float] = None       # log10(kcat / s⁻¹)
    log_km: Optional[float] = None         # log10(Km / mM)
    log_kcat_km: Optional[float] = None    # log10(kcat/Km)

    # Reaction molecules (SMILES; empty if unavailable). Featurised on load into
    # the substrate/product graphs the cross-attention + kinetics head consume.
    substrate_smiles: str = ""
    product_smiles: str = ""

    # Canonical enzyme–substrate assay identity. Multiple records may share
    # the same structure arrays but must never share distinct substrate labels.
    enzyme_id: str = ""
    substrate_id: str = ""
    product_id: str = ""
    temperature: Optional[float] = None
    ph: Optional[float] = None
    ionic_conditions: str = ""
    mutation: str = "WT"
    detected: Optional[bool] = None
    detection_limit_log: Optional[float] = None

    # Train / val / test split
    split: str = ""


# ── Dataset builder ───────────────────────────────────────────────────────────

class DatasetBuilder:
    """Assemble and store the curated ToPE dataset."""

    def __init__(
        self,
        output_dir: Path = PROCESSED_DIR,
        features_dir: Path = FEATURES_DIR,
        config: Optional[PipelineConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.features_dir = Path(features_dir)
        self.config = config or PipelineConfig()

        # Subdirectories for per-sample arrays
        self.arrays_dir = self.features_dir / "arrays"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build from processed features ────────────────────────────────────

    def build(
        self,
        active_sites: List[ActiveSite],
        features_list: List[ActiveSiteFeatures],
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        kinetics_map: Optional[Dict[Any, Dict[str, Any]]] = None,
        molecule_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[DatasetRecord]:
        """Build the full dataset from active sites and their features.

        Parameters
        ----------
        active_sites : list of ActiveSite
        features_list : list of ActiveSiteFeatures
            Must be aligned 1:1 with active_sites.
        split_ratios : tuple of float
            (train, val, test) fractions. Must sum to 1.
        seed : int
            Random seed for split assignment.
        kinetics_map : dict, optional
            Mapping PDB ID → {param_type: log10_value} from the kinetics
            aggregator. Used to attach kcat/Km labels to each sample.

        Returns
        -------
        list of DatasetRecord
        """
        assert len(active_sites) == len(features_list)

        if kinetics_map is None:
            kinetics_map = {}

        # Assign splits
        n = len(active_sites)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        split_labels = np.empty(n, dtype=object)
        split_labels[indices[:n_train]] = "train"
        split_labels[indices[n_train:n_train + n_val]] = "val"
        split_labels[indices[n_train + n_val:]] = "test"

        records: List[DatasetRecord] = []
        molecule_map = molecule_map or {}

        for i, (site, feats) in enumerate(zip(active_sites, features_list)):
            # Save per-sample arrays
            sample_dir = self.arrays_dir / site.pdb_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            coords_path = sample_dir / "coords.npy"
            features_path = sample_dir / "features.npy"
            mask_path = sample_dir / "catalytic_mask.npy"
            atom_residue_path = sample_dir / "atom_residue.npy"

            np.save(str(coords_path), feats.coords)
            np.save(str(features_path), feats.feature_matrix)
            np.save(str(mask_path), feats.catalytic_mask)
            # Persist the atom→residue incidence (rank-0 → rank-2 membership)
            # so the curated complex — not the model — defines B_12 downstream.
            # Aligned 1:1 with the saved atom coords / features ordering.
            np.save(str(atom_residue_path), site.atom_residue_incidence())

            # Save filtration adjacency matrices
            adj_dir = sample_dir / "adjacency"
            adj_dir.mkdir(exist_ok=True)
            from tope.data.active_site import ActiveSiteExtractor
            adjs = ActiveSiteExtractor.compute_filtration_adjacencies(site)
            for radius, adj in adjs.items():
                np.save(str(adj_dir / f"adj_{radius:.1f}.npy"), adj)

            ec_top = site.ec_number.split(".")[0] if site.ec_number else ""

            observations = _observations_for_site(kinetics_map, site.pdb_id)
            if not observations:
                observations = [(None, {})]

            for observation, kin in observations:
                # A molecule map may be keyed by the full observation or by PDB
                # for legacy single-substrate datasets.
                mol = molecule_map.get(observation, molecule_map.get(site.pdb_id, {}))
                record = DatasetRecord(
                    pdb_id=site.pdb_id,
                    ec_number=site.ec_number,
                    ec_top_level=ec_top,
                    n_atoms=site.n_atoms,
                    n_residues=site.n_residues,
                    n_catalytic_residues=site.n_catalytic_residues,
                    has_metal=site.has_metal,
                    elements=sorted(site.elements),
                    coords_path=str(coords_path.relative_to(self.features_dir)),
                    features_path=str(features_path.relative_to(self.features_dir)),
                    mask_path=str(mask_path.relative_to(self.features_dir)),
                    adjacency_dir=str(adj_dir.relative_to(self.features_dir)),
                    atom_residue_path=str(atom_residue_path.relative_to(self.features_dir)),
                    log_kcat=kin.get("kcat"),
                    log_km=kin.get("Km"),
                    log_kcat_km=kin.get("kcat/Km"),
                    substrate_smiles=str(mol.get("substrate", "")),
                    product_smiles=str(mol.get("product", "")),
                    enzyme_id=getattr(observation, "enzyme_id", site.pdb_id),
                    substrate_id=getattr(observation, "substrate_id", ""),
                    product_id=getattr(observation, "product_id", ""),
                    temperature=getattr(observation, "temperature", None),
                    ph=getattr(observation, "ph", None),
                    ionic_conditions=getattr(observation, "ionic_conditions", ""),
                    mutation=getattr(observation, "mutation", "WT"),
                    detected=kin.get("detected"),
                    detection_limit_log=kin.get("detection_limit_log"),
                    split=str(split_labels[i]),
                )
                records.append(record)

            if (i + 1) % 100 == 0 or (i + 1) == n:
                logger.info("Dataset assembly: %d / %d", i + 1, n)

        kin_count = sum(
            1 for r in records
            if r.log_kcat is not None or r.log_km is not None
        )
        logger.info(
            "Dataset built: %d samples (train=%d, val=%d, test=%d), "
            "%d with kinetics labels",
            len(records),
            sum(1 for r in records if r.split == "train"),
            sum(1 for r in records if r.split == "val"),
            sum(1 for r in records if r.split == "test"),
            kin_count,
        )

        # Save index
        self._save_index(records)

        return records

    # ── Export formats ───────────────────────────────────────────────────

    def _save_index(self, records: List[DatasetRecord]) -> None:
        """Save the dataset index in the configured format."""
        fmt = self.config.output_format

        if fmt == "parquet" and HAS_ARROW and HAS_PANDAS:
            self._save_parquet(records)
        elif fmt == "csv" and HAS_PANDAS:
            self._save_csv(records)
        else:
            # Always save JSON as fallback
            self._save_json(records)

        # Always save JSON alongside for easy inspection
        if fmt != "json":
            self._save_json(records)

    def _save_parquet(self, records: List[DatasetRecord]) -> None:
        """Save dataset index as Parquet."""
        df = pd.DataFrame([self._record_to_dict(r) for r in records])
        path = self.output_dir / "dataset_index.parquet"
        df.to_parquet(str(path), index=False)
        logger.info("Saved dataset index → %s", path)

    def _save_csv(self, records: List[DatasetRecord]) -> None:
        """Save dataset index as CSV."""
        df = pd.DataFrame([self._record_to_dict(r) for r in records])
        path = self.output_dir / "dataset_index.csv"
        df.to_csv(str(path), index=False)
        logger.info("Saved dataset index → %s", path)

    def _save_json(self, records: List[DatasetRecord]) -> None:
        """Save dataset index as JSON."""
        data = [self._record_to_dict(r) for r in records]
        path = self.output_dir / "dataset_index.json"
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved dataset index → %s", path)

    # ── Summary statistics ───────────────────────────────────────────────

    @staticmethod
    def summarise(records: List[DatasetRecord]) -> Dict[str, Any]:
        """Compute summary statistics for the dataset."""
        n = len(records)
        if n == 0:
            return {"total": 0}

        ec_dist: Dict[str, int] = {}
        split_dist: Dict[str, int] = {}
        atom_counts = []
        metal_count = 0
        has_kcat = 0
        has_km = 0
        has_both = 0

        for r in records:
            ec_dist[r.ec_top_level] = ec_dist.get(r.ec_top_level, 0) + 1
            split_dist[r.split] = split_dist.get(r.split, 0) + 1
            atom_counts.append(r.n_atoms)
            if r.has_metal:
                metal_count += 1
            r_has_kcat = r.log_kcat is not None
            r_has_km = r.log_km is not None
            if r_has_kcat:
                has_kcat += 1
            if r_has_km:
                has_km += 1
            if r_has_kcat and r_has_km:
                has_both += 1

        atom_arr = np.array(atom_counts)

        # Map EC numbers to names
        ec_named = {}
        for ec_id, count in sorted(ec_dist.items()):
            name = EC_TOP_LEVEL.get(ec_id, "Unknown")
            ec_named[f"EC {ec_id} ({name})"] = count

        return {
            "total_samples": n,
            "split_distribution": split_dist,
            "ec_distribution": ec_named,
            "atom_count_stats": {
                "mean": float(atom_arr.mean()),
                "std": float(atom_arr.std()),
                "min": int(atom_arr.min()),
                "max": int(atom_arr.max()),
                "median": float(np.median(atom_arr)),
            },
            "metalloenzyme_fraction": metal_count / n,
            "kinetics_coverage": {
                "has_kcat": has_kcat,
                "has_km": has_km,
                "has_both": has_both,
                "fraction_with_any": (has_kcat + has_km - has_both) / n if n > 0 else 0,
            },
        }

    # ── Loading (for downstream consumers) ───────────────────────────────

    @classmethod
    def load_index(cls, index_path: Path) -> List[Dict[str, Any]]:
        """Load a dataset index from JSON/CSV/Parquet."""
        path = Path(index_path)

        if path.suffix == ".json":
            return json.loads(path.read_text())
        elif path.suffix == ".csv" and HAS_PANDAS:
            df = pd.read_csv(str(path))
            return df.to_dict(orient="records")
        elif path.suffix == ".parquet" and HAS_PANDAS and HAS_ARROW:
            df = pd.read_parquet(str(path))
            return df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported index format: {path.suffix}")

    @staticmethod
    def load_sample(
        features_dir: Path,
        record: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load per-sample arrays (coords, features, mask) from disk.

        Returns
        -------
        (coords, features, catalytic_mask)
            coords: (N, 3) float64
            features: (N, 8) float64
            catalytic_mask: (N,) bool
        """
        base = Path(features_dir)
        coords = np.load(str(base / record["coords_path"]))
        features = np.load(str(base / record["features_path"]))
        mask = np.load(str(base / record["mask_path"]))
        return coords, features, mask

    @staticmethod
    def load_adjacencies(
        features_dir: Path,
        record: Dict[str, Any],
    ) -> Dict[float, np.ndarray]:
        """Load filtration adjacency matrices for a sample.

        Returns
        -------
        dict mapping filtration radius → (N, N) adjacency matrix
        """
        adj_dir = Path(features_dir) / record["adjacency_dir"]
        result: Dict[float, np.ndarray] = {}
        for npy_path in sorted(adj_dir.glob("adj_*.npy")):
            # Parse radius from filename "adj_4.0.npy"
            radius_str = npy_path.stem.replace("adj_", "")
            radius = float(radius_str)
            result[radius] = np.load(str(npy_path))
        return result

    @staticmethod
    def load_kinetics(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Extract kinetics labels from a dataset record.

        Returns
        -------
        dict with keys "log_kcat", "log_km", "log_kcat_km"
        """
        return {
            "log_kcat": record.get("log_kcat"),
            "log_km": record.get("log_km"),
            "log_kcat_km": record.get("log_kcat_km"),
        }

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_dict(r: DatasetRecord) -> Dict[str, Any]:
        """Convert a DatasetRecord to a serialisable dict."""
        return {
            "pdb_id": r.pdb_id,
            "ec_number": r.ec_number,
            "ec_top_level": r.ec_top_level,
            "n_atoms": r.n_atoms,
            "n_residues": r.n_residues,
            "n_catalytic_residues": r.n_catalytic_residues,
            "has_metal": r.has_metal,
            "elements": r.elements,
            "coords_path": r.coords_path,
            "features_path": r.features_path,
            "mask_path": r.mask_path,
            "adjacency_dir": r.adjacency_dir,
            "atom_residue_path": r.atom_residue_path,
            "log_kcat": r.log_kcat,
            "log_km": r.log_km,
            "log_kcat_km": r.log_kcat_km,
            "substrate_smiles": r.substrate_smiles,
            "product_smiles": r.product_smiles,
            "enzyme_id": r.enzyme_id,
            "substrate_id": r.substrate_id,
            "product_id": r.product_id,
            "temperature": r.temperature,
            "ph": r.ph,
            "ionic_conditions": r.ionic_conditions,
            "mutation": r.mutation,
            "detected": r.detected,
            "detection_limit_log": r.detection_limit_log,
            "split": r.split,
        }
