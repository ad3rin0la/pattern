"""PDB B-factor extraction and curation for γ calibration.

Pulls per-Cα isotropic B-factors out of PDB/CIF structures, applies
the resolution/method filters that ENM benchmarks (PSL, GNM) use, and
splits the resulting dataset into calibration/validation folds with
optional size stratification.

Designed to live alongside the existing ``active_site`` extractor: the
parser reuses BioPython, the filters mirror the curation pipeline's
``PipelineConfig`` knobs, and the dataset can be cached to parquet so
calibration re-runs don't repeat structure parsing.

This module is *bridge-agnostic*: it returns B-factors and Cα
coordinates, not stiffness tensors. The calibration consumer
(``tope.dynamics.calibration``) is responsible for mapping coords →
stiffness.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from Bio.PDB import MMCIFParser, PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ── Records and configuration ─────────────────────────────────────────────────

@dataclass
class BFactorRecord:
    """One protein's worth of B-factor / coordinate data for calibration."""

    pdb_id: str
    chain_id: str                          # primary chain extracted
    n_residues: int
    coords: np.ndarray                     # (N, 3) Cα coords
    b_factors: np.ndarray                  # (N,) isotropic B-factors (Å²)
    residue_names: List[str] = field(default_factory=list)
    residue_numbers: List[int] = field(default_factory=list)
    resolution_A: Optional[float] = None
    temperature_K: Optional[float] = None
    method: str = ""                       # X-RAY DIFFRACTION, ELECTRON MICROSCOPY, …

    def size_bucket(self) -> str:
        if self.n_residues < 100:
            return "small"
        if self.n_residues < 250:
            return "medium"
        return "large"

    def log_b(self, floor: float = 1e-6) -> np.ndarray:
        return np.log(np.maximum(self.b_factors, floor))


@dataclass
class BFactorFilter:
    """Quality filter mirroring ENM benchmarking practice (PSL, GNM)."""

    max_resolution_A: float = 2.5
    min_residues: int = 50
    max_residues: int = 800
    allowed_methods: Tuple[str, ...] = ("X-RAY DIFFRACTION",)
    reject_anisotropic: bool = True
    min_b_factor: float = 1e-3             # filter out pathological B=0 entries
    max_b_factor: float = 200.0            # reject disordered tail residues

    def accept_structure(
        self,
        resolution_A: Optional[float],
        method: str,
        n_residues: int,
    ) -> Tuple[bool, str]:
        if self.allowed_methods and method.upper() not in {m.upper() for m in self.allowed_methods}:
            return False, f"method={method!r} not allowed"
        if resolution_A is not None and resolution_A > self.max_resolution_A:
            return False, f"resolution {resolution_A:.2f} Å > {self.max_resolution_A}"
        if n_residues < self.min_residues:
            return False, f"too small (n={n_residues})"
        if n_residues > self.max_residues:
            return False, f"too large (n={n_residues})"
        return True, "ok"


@dataclass
class BFactorDataset:
    """A curated set of B-factor records plus a calibration/validation split."""

    records: List[BFactorRecord] = field(default_factory=list)
    calibration_ids: List[str] = field(default_factory=list)
    validation_ids: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def get(self, pdb_id: str) -> Optional[BFactorRecord]:
        for r in self.records:
            if r.pdb_id == pdb_id:
                return r
        return None

    def calibration(self) -> List[BFactorRecord]:
        cal = set(self.calibration_ids)
        return [r for r in self.records if r.pdb_id in cal]

    def validation(self) -> List[BFactorRecord]:
        val = set(self.validation_ids)
        return [r for r in self.records if r.pdb_id in val]

    def summary(self) -> Dict[str, int]:
        buckets: Dict[str, int] = {"small": 0, "medium": 0, "large": 0}
        for r in self.records:
            buckets[r.size_bucket()] += 1
        return {
            "n_total": len(self.records),
            "n_calibration": len(self.calibration_ids),
            "n_validation": len(self.validation_ids),
            "n_residues_total": sum(r.n_residues for r in self.records),
            **buckets,
        }


# ── PDB extraction ────────────────────────────────────────────────────────────

def extract_bfactors_from_structure(
    structure_path: Path,
    pdb_id: str = "",
    chain_id: Optional[str] = None,
    filter_cfg: Optional[BFactorFilter] = None,
) -> Optional[BFactorRecord]:
    """Parse one PDB/CIF file → BFactorRecord, returning None if filtered out."""
    if not HAS_BIOPYTHON:
        raise ImportError("BioPython required: pip install biopython")
    filt = filter_cfg or BFactorFilter()

    path = Path(structure_path)
    is_cif = path.suffix.lower() in (".cif", ".mmcif")
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id or "X", str(path))
    except Exception as exc:
        logger.warning("%s: parse failed: %s", pdb_id, exc)
        return None

    resolution, temperature, method = _read_header(structure, is_cif=is_cif)

    coords_l: List[np.ndarray] = []
    b_l: List[float] = []
    names: List[str] = []
    numbers: List[int] = []
    used_chain = chain_id or ""

    model = structure[0]
    for chain in model:
        cid = chain.get_id()
        if chain_id is not None and cid != chain_id:
            continue
        if chain_id is None and not used_chain:
            used_chain = cid

        for residue in chain:
            het_flag = residue.get_id()[0]
            if het_flag.strip() not in ("", "H_MSE"):
                continue
            try:
                ca = residue["CA"]
            except KeyError:
                continue
            b = float(ca.get_bfactor())
            if b < filt.min_b_factor or b > filt.max_b_factor:
                continue
            if filt.reject_anisotropic and ca.is_disordered():
                continue
            coords_l.append(np.array(ca.get_vector().get_array(), dtype=np.float64))
            b_l.append(b)
            names.append(residue.get_resname().strip())
            numbers.append(int(residue.get_id()[1]))

        if chain_id is None and coords_l:
            break  # primary-chain mode: take the first chain with data

    if not coords_l:
        return None

    n = len(coords_l)
    ok, reason = filt.accept_structure(resolution, method, n)
    if not ok:
        logger.info("%s: rejected (%s)", pdb_id, reason)
        return None

    return BFactorRecord(
        pdb_id=pdb_id or path.stem,
        chain_id=used_chain,
        n_residues=n,
        coords=np.array(coords_l, dtype=np.float64),
        b_factors=np.array(b_l, dtype=np.float64),
        residue_names=names,
        residue_numbers=numbers,
        resolution_A=resolution,
        temperature_K=temperature,
        method=method,
    )


def _read_header(structure, is_cif: bool) -> Tuple[Optional[float], Optional[float], str]:
    """Best-effort resolution / temperature / method extraction."""
    method = ""
    resolution: Optional[float] = None
    temperature: Optional[float] = None

    hdr = getattr(structure, "header", {}) or {}
    method = str(hdr.get("structure_method", "")).upper()
    res_str = hdr.get("resolution")
    if res_str is not None:
        try:
            resolution = float(res_str)
        except (TypeError, ValueError):
            resolution = None

    # Temperature lives in REMARK 200 for X-ray; BioPython doesn't normalize it.
    # Conservative default: leave as None; calibration falls back to a config T.
    return resolution, temperature, method


def ingest_directory(
    directory: Path,
    filter_cfg: Optional[BFactorFilter] = None,
    extensions: Tuple[str, ...] = (".pdb", ".cif"),
) -> List[BFactorRecord]:
    """Walk a directory of structures and produce records for everything
    that survives the filter."""
    directory = Path(directory)
    out: List[BFactorRecord] = []
    for ext in extensions:
        for path in sorted(directory.glob(f"*{ext}")):
            rec = extract_bfactors_from_structure(
                path, pdb_id=path.stem, filter_cfg=filter_cfg,
            )
            if rec is not None:
                out.append(rec)
    return out


# ── Stratified calibration / validation split ─────────────────────────────────

def stratified_split(
    records: Sequence[BFactorRecord],
    calibration_fraction: float = 0.6,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split records into (calibration_ids, validation_ids).

    Stratified by size_bucket so each split sees a representative mix of
    small/medium/large proteins. Returns IDs only; caller can re-key
    into the records list.
    """
    if not records:
        return [], []
    by_bucket: Dict[str, List[str]] = {}
    for r in records:
        by_bucket.setdefault(r.size_bucket(), []).append(r.pdb_id)

    rng = random.Random(seed)
    cal: List[str] = []
    val: List[str] = []
    for bucket, ids in by_bucket.items():
        rng.shuffle(ids)
        cut = max(1, int(round(calibration_fraction * len(ids))))
        cal.extend(ids[:cut])
        val.extend(ids[cut:])
    return cal, val


def build_dataset(
    records: Sequence[BFactorRecord],
    calibration_fraction: float = 0.6,
    seed: int = 0,
) -> BFactorDataset:
    """Assemble a BFactorDataset with a stratified train/val split."""
    cal_ids, val_ids = stratified_split(records, calibration_fraction, seed)
    return BFactorDataset(
        records=list(records),
        calibration_ids=cal_ids,
        validation_ids=val_ids,
    )


# ── Parquet cache (optional) ──────────────────────────────────────────────────

def save_parquet(dataset: BFactorDataset, path: Path) -> None:
    """Round-trip the dataset to parquet for fast re-loads.

    Stores per-(protein, residue) rows so a single file holds the whole
    benchmark. Coordinates are stored as separate x/y/z columns; metadata
    is repeated per row (denormalised) for simplicity.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "pdb_id": [],
        "chain_id": [],
        "residue_idx": [],
        "residue_name": [],
        "residue_number": [],
        "x": [],
        "y": [],
        "z": [],
        "b_factor": [],
        "resolution_A": [],
        "method": [],
        "split": [],
    }
    cal = set(dataset.calibration_ids)
    val = set(dataset.validation_ids)

    for r in dataset.records:
        split = "calibration" if r.pdb_id in cal else "validation" if r.pdb_id in val else "unassigned"
        for i in range(r.n_residues):
            rows["pdb_id"].append(r.pdb_id)
            rows["chain_id"].append(r.chain_id)
            rows["residue_idx"].append(i)
            rows["residue_name"].append(r.residue_names[i] if i < len(r.residue_names) else "")
            rows["residue_number"].append(r.residue_numbers[i] if i < len(r.residue_numbers) else -1)
            rows["x"].append(float(r.coords[i, 0]))
            rows["y"].append(float(r.coords[i, 1]))
            rows["z"].append(float(r.coords[i, 2]))
            rows["b_factor"].append(float(r.b_factors[i]))
            rows["resolution_A"].append(r.resolution_A if r.resolution_A is not None else float("nan"))
            rows["method"].append(r.method)
            rows["split"].append(split)

    pq.write_table(pa.table(rows), str(path))


def load_parquet(path: Path) -> BFactorDataset:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    df = table.to_pandas()
    records: Dict[str, dict] = {}
    cal_ids: List[str] = []
    val_ids: List[str] = []

    for pdb_id, group in df.groupby("pdb_id"):
        group = group.sort_values("residue_idx")
        rec = BFactorRecord(
            pdb_id=str(pdb_id),
            chain_id=str(group["chain_id"].iloc[0]),
            n_residues=len(group),
            coords=group[["x", "y", "z"]].to_numpy(dtype=np.float64),
            b_factors=group["b_factor"].to_numpy(dtype=np.float64),
            residue_names=group["residue_name"].astype(str).tolist(),
            residue_numbers=group["residue_number"].astype(int).tolist(),
            resolution_A=float(group["resolution_A"].iloc[0]) if not np.isnan(group["resolution_A"].iloc[0]) else None,
            method=str(group["method"].iloc[0]),
        )
        records[str(pdb_id)] = rec
        split = str(group["split"].iloc[0])
        if split == "calibration":
            cal_ids.append(str(pdb_id))
        elif split == "validation":
            val_ids.append(str(pdb_id))

    return BFactorDataset(
        records=list(records.values()),
        calibration_ids=cal_ids,
        validation_ids=val_ids,
    )
