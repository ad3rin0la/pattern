#!/usr/bin/env python3
"""
TopEC → ToPE Data Ingestion Pipeline
======================================

Ingests the TopEC dataset (van der Weg et al., Nat. Commun. 2025) into
the ToPE multi-scale pipeline, enriching enzyme structures with kinetic
parameters from BRENDA/SABIO-RK/ChEMBL for curriculum-based training.

TopEC data sources:
    - GitHub:  https://github.com/IBG4-CBCLab/TopEC/tree/main
    - HHU:    https://researchdata.hhu.de/items/e70684b7-d1f1-45d0-8dcd-7954a142a8c0
    - Sciebo:  PDB files: https://fz-juelich.sciebo.de/s/7cOPiXC0iqlh3c9
               H5 dataset: https://fz-juelich.sciebo.de/s/zvnTIm0TdJmPwdd

Output:
    - Unified Parquet index mapping enzyme → EC + binding site + kinetics + zones
    - PDB files symlinked/copied into ToPE directory structure
    - Ready for MultiScaleProteinGraph.build() and curriculum training

Usage:
    python topec_to_tope_ingestion.py \
        --topec-csv-dir  /path/to/TopEC/data/csv/ \
        --topec-pdb-dir  /path/to/TopEC/structures/ \
        --tope-data-dir  ./data \
        --enrich-kinetics \
        --workers 8

Phase A pretraining (EC only):
    Uses ALL TopEC structures — no kinetics required.

Phase B fine-tuning (kinetics):
    Uses the kinetics-enriched subset for kcat/Km regression.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Optional imports — gracefully degrade
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestionConfig:
    """Configuration for TopEC → ToPE data ingestion."""

    # ── Input paths ──────────────────────────────────────────────────────
    topec_csv_dir: Path = Path("TopEC/data/csv")
    topec_pdb_dir: Path = Path("TopEC/structures")
    topec_h5_path: Optional[Path] = None  # Optional: pre-built H5 dataset

    # ── Output paths ─────────────────────────────────────────────────────
    tope_data_dir: Path = Path("data")
    output_index: Path = Path("data/processed/topec_tope_index.parquet")
    output_json: Path = Path("data/processed/topec_tope_index.json")

    # ── Zone boundaries (Å) ──────────────────────────────────────────────
    zone1_radius: float = 8.0
    zone2_radius: float = 20.0

    # ── Kinetics enrichment ──────────────────────────────────────────────
    enrich_kinetics: bool = True
    kinetics_cache_dir: Path = Path("data/raw/kinetics")
    chembl_enrichment: bool = True  # Use ChEMBL MCP for additional data
    brenda_cache: Optional[Path] = None
    sabio_cache: Optional[Path] = None

    # ── Fold split ───────────────────────────────────────────────────────
    use_topec_splits: bool = True  # Preserve TopEC's fold-split assignments
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # ── Processing ───────────────────────────────────────────────────────
    n_workers: int = 8
    min_residues: int = 20  # Skip tiny fragments
    max_residues: int = 5000  # Skip enormous complexes
    skip_existing: bool = True
    verbose: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: TopEC CSV Parsing
# ══════════════════════════════════════════════════════════════════════════════

def discover_topec_csvs(csv_dir: Path) -> Dict[str, Path]:
    """Find all TopEC dataset CSV files.

    TopEC uses separate CSVs for different dataset configurations:
        - PDB300: ~300 EC classes from experimental PDB structures
        - AF703:  ~703 EC classes including AlphaFold2 predictions
        - Combined: PDB300 + AF703

    Returns
    -------
    dict mapping dataset_name → csv_path
    """
    csvs = {}
    if not csv_dir.exists():
        logger.warning("TopEC CSV directory not found: %s", csv_dir)
        return csvs

    for f in sorted(csv_dir.glob("*.csv")):
        name = f.stem.lower()
        csvs[name] = f
        logger.info("  Found TopEC CSV: %s (%s)", f.name, name)

    return csvs


def parse_topec_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse a single TopEC CSV into standardised records.

    TopEC CSV format (essential columns):
        enzyme_name  — UniProt AC or PDB ID
        centers      — binding site center as "(x, y, z)" string
        hierarchical — integer EC class label

    Additional columns may include:
        ec_number, split, fold_id, uniprot_ac, pdb_id, source, ...

    Returns
    -------
    list of dicts, each with keys:
        enzyme_id, binding_center, ec_class_idx, ec_number,
        source_dataset, pdb_id, uniprot_ac, split
    """
    if HAS_PANDAS:
        return _parse_csv_pandas(csv_path)
    else:
        return _parse_csv_stdlib(csv_path)


def _parse_csv_pandas(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse using pandas (preferred)."""
    df = pd.read_csv(csv_path)
    records = []

    # Normalise column names
    col_map = {c.lower().strip(): c for c in df.columns}

    enzyme_col = col_map.get("enzyme_name", col_map.get("enzyme", None))
    center_col = col_map.get("centers", col_map.get("center", None))
    ec_idx_col = col_map.get("hierarchical", col_map.get("ec_class", None))

    if enzyme_col is None or center_col is None:
        logger.error("CSV missing required columns (enzyme_name, centers): %s",
                     list(df.columns))
        return records

    # Optional columns
    ec_num_col = col_map.get("ec_number", col_map.get("ec", None))
    split_col = col_map.get("split", col_map.get("fold_split", None))
    pdb_col = col_map.get("pdb_id", col_map.get("pdb", None))
    uniprot_col = col_map.get("uniprot_ac", col_map.get("uniprot", None))
    source_col = col_map.get("source", None)

    dataset_name = csv_path.stem

    for idx, row in df.iterrows():
        enzyme_id = str(row[enzyme_col]).strip()
        center_str = str(row[center_col]).strip()

        # Parse binding site center: "(x, y, z)" → [x, y, z]
        binding_center = _parse_center_string(center_str)
        if binding_center is None:
            logger.debug("Skipping row %d: unparseable center '%s'", idx, center_str)
            continue

        rec = {
            "enzyme_id": enzyme_id,
            "binding_center": binding_center,
            "ec_class_idx": int(row[ec_idx_col]) if ec_idx_col and pd.notna(row.get(ec_idx_col)) else -1,
            "ec_number": str(row.get(ec_num_col, "")).strip() if ec_num_col else "",
            "source_dataset": dataset_name,
            "pdb_id": str(row.get(pdb_col, enzyme_id)).strip()[:4].upper() if pdb_col else _infer_pdb_id(enzyme_id),
            "uniprot_ac": str(row.get(uniprot_col, "")).strip() if uniprot_col else "",
            "split": str(row.get(split_col, "")).strip().lower() if split_col else "",
            "source_type": str(row.get(source_col, "")).strip() if source_col else _infer_source(enzyme_id),
        }
        records.append(rec)

    logger.info("Parsed %d records from %s", len(records), csv_path.name)
    return records


def _parse_csv_stdlib(csv_path: Path) -> List[Dict[str, Any]]:
    """Fallback CSV parser using stdlib only."""
    import csv
    records = []
    dataset_name = csv_path.stem

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Try to find the right columns (case-insensitive)
            row_lower = {k.lower().strip(): v for k, v in row.items()}

            enzyme_id = row_lower.get("enzyme_name", row_lower.get("enzyme", "")).strip()
            center_str = row_lower.get("centers", row_lower.get("center", "")).strip()
            ec_idx = row_lower.get("hierarchical", row_lower.get("ec_class", "-1"))

            binding_center = _parse_center_string(center_str)
            if not enzyme_id or binding_center is None:
                continue

            rec = {
                "enzyme_id": enzyme_id,
                "binding_center": binding_center,
                "ec_class_idx": int(ec_idx) if ec_idx.lstrip("-").isdigit() else -1,
                "ec_number": row_lower.get("ec_number", row_lower.get("ec", "")),
                "source_dataset": dataset_name,
                "pdb_id": _infer_pdb_id(enzyme_id),
                "uniprot_ac": row_lower.get("uniprot_ac", row_lower.get("uniprot", "")),
                "split": row_lower.get("split", row_lower.get("fold_split", "")),
                "source_type": row_lower.get("source", _infer_source(enzyme_id)),
            }
            records.append(rec)

    logger.info("Parsed %d records from %s (stdlib)", len(records), csv_path.name)
    return records


def _parse_center_string(s: str) -> Optional[List[float]]:
    """Parse TopEC binding center string → [x, y, z].

    Handles formats:
        "(1.23, 4.56, 7.89)"
        "1.23 4.56 7.89"
        "[1.23, 4.56, 7.89]"
    """
    s = s.strip()
    if not s or s.lower() == "nan":
        return None

    # Try ast.literal_eval for tuple/list formats
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 3:
            return [float(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass

    # Try space-separated
    parts = s.replace(",", " ").replace("(", "").replace(")", "").replace("[", "").replace("]", "").split()
    if len(parts) == 3:
        try:
            return [float(x) for x in parts]
        except ValueError:
            pass

    return None


def _infer_pdb_id(enzyme_id: str) -> str:
    """Infer PDB ID from enzyme identifier."""
    clean = enzyme_id.strip().upper()
    # 4-char PDB ID
    if len(clean) == 4 and clean.isalnum():
        return clean
    # PDB ID with chain (e.g., "1ABC_A")
    if len(clean) >= 4 and clean[:4].isalnum():
        return clean[:4]
    return clean


def _infer_source(enzyme_id: str) -> str:
    """Infer whether structure is experimental or AlphaFold."""
    eid = enzyme_id.strip()
    # AlphaFold IDs typically start with AF- or are longer UniProt ACs
    if eid.startswith("AF-") or eid.startswith("af-"):
        return "alphafold"
    if len(eid) == 4:
        return "pdb_experimental"
    if len(eid) >= 6 and eid[0].isalpha():
        return "alphafold"  # Likely UniProt AC → AF2 structure
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: EC Number Resolution
# ══════════════════════════════════════════════════════════════════════════════

def build_ec_class_mapping(records: List[Dict]) -> Dict[int, str]:
    """Build mapping from TopEC integer class → EC number string.

    TopEC encodes EC numbers as sequential integers in the 'hierarchical'
    column. We reconstruct the mapping from records that have both
    ec_class_idx and ec_number populated.

    Returns
    -------
    dict mapping ec_class_idx → ec_number (e.g., 0 → "1.1.1.1")
    """
    mapping = {}
    for rec in records:
        idx = rec.get("ec_class_idx", -1)
        ec = rec.get("ec_number", "").strip()
        if idx >= 0 and ec and re.match(r"^\d+\.\d+", ec):
            if idx not in mapping:
                mapping[idx] = ec
            elif mapping[idx] != ec:
                logger.warning("EC class %d maps to both '%s' and '%s'",
                               idx, mapping[idx], ec)

    logger.info("Resolved %d EC class → EC number mappings", len(mapping))
    return mapping


def resolve_ec_from_pdb(pdb_id: str) -> Optional[str]:
    """Query RCSB PDB for EC number associated with a structure.

    Fallback when TopEC CSV doesn't include EC number strings.
    """
    try:
        import urllib.request
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            # EC numbers are in struct.pdbx_descriptor or
            # entity[].rcsb_entity_source_organism[].rcsb_ec_lineage
            enzymes = data.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0)
            if enzymes > 0:
                # Try polymer entities
                for entity in data.get("polymer_entities", []):
                    ec_list = entity.get("rcsb_polymer_entity", {}).get("ec_numbers", [])
                    if ec_list:
                        return ec_list[0]
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Zone Assignment from Binding Centers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneStats:
    """Statistics about zone assignments for a single enzyme."""
    enzyme_id: str
    binding_center: List[float]
    n_zone1: int = 0  # 0–8 Å
    n_zone2: int = 0  # 8–20 Å
    n_zone3: int = 0  # >20 Å
    n_total: int = 0
    max_distance: float = 0.0


def compute_zone_stats_from_pdb(
    pdb_path: Path,
    binding_center: List[float],
    zone1_radius: float = 8.0,
    zone2_radius: float = 20.0,
) -> Optional[ZoneStats]:
    """Compute residue counts per zone for a PDB structure.

    Uses the TopEC binding site center as the Zone 1 origin, then
    assigns each residue's Cα to Zone 1/2/3 based on distance.
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        logger.warning("BioPython required for zone stats; skipping %s", pdb_path)
        return None

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("enzyme", str(pdb_path))
    except Exception as e:
        logger.debug("Failed to parse %s: %s", pdb_path, e)
        return None

    center = np.array(binding_center)
    stats = ZoneStats(
        enzyme_id=pdb_path.stem,
        binding_center=binding_center,
    )

    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip non-amino acids
                if residue.id[0] != " ":
                    continue
                # Get Cα position
                if "CA" not in residue:
                    continue
                ca_coord = residue["CA"].get_vector().get_array()
                dist = np.linalg.norm(ca_coord - center)

                stats.n_total += 1
                stats.max_distance = max(stats.max_distance, dist)

                if dist <= zone1_radius:
                    stats.n_zone1 += 1
                elif dist <= zone2_radius:
                    stats.n_zone2 += 1
                else:
                    stats.n_zone3 += 1
        break  # First model only

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Kinetics Enrichment
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KineticsRecord:
    """Kinetic parameters for a single enzyme-substrate pair."""
    ec_number: str = ""
    substrate: str = ""
    log_kcat: Optional[float] = None     # log10(kcat / s^-1)
    log_km: Optional[float] = None       # log10(Km / M)
    log_efficiency: Optional[float] = None  # log10(kcat/Km / M^-1 s^-1)
    kcat_source: str = ""
    km_source: str = ""
    organism: str = ""
    ph: Optional[float] = None
    temperature: Optional[float] = None  # °C
    confidence: str = "low"  # low / medium / high


def enrich_with_kinetics(
    records: List[Dict],
    kinetics_cache_dir: Path,
    use_chembl: bool = True,
) -> Dict[str, List[KineticsRecord]]:
    """Enrich TopEC enzyme records with kinetic parameters.

    Data sources (in priority order):
        1. BRENDA — gold standard, manually curated
        2. SABIO-RK — reaction kinetics database
        3. ChEMBL — bioactivity data (via MCP or API)

    Returns
    -------
    dict mapping enzyme_id → list of KineticsRecord
    """
    kinetics_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = kinetics_cache_dir / "kinetics_enrichment_cache.json"

    # Load cache
    kinetics_map: Dict[str, List[KineticsRecord]] = {}
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            for eid, entries in cached.items():
                kinetics_map[eid] = [KineticsRecord(**e) for e in entries]
            logger.info("Loaded %d cached kinetics entries", len(kinetics_map))
        except Exception:
            pass

    # Collect EC numbers to query
    ec_to_enzymes: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        eid = rec["enzyme_id"]
        ec = rec.get("ec_number", "")
        if ec and eid not in kinetics_map:
            ec_to_enzymes[ec].append(eid)

    n_missing = sum(len(v) for v in ec_to_enzymes.values())
    if n_missing == 0:
        logger.info("All enzymes have cached kinetics data")
        return kinetics_map

    logger.info("Querying kinetics for %d EC classes (%d enzymes)",
                len(ec_to_enzymes), n_missing)

    # ── Source 1: BRENDA (via SOAP or cached flat files) ─────────────
    brenda_hits = _query_brenda_kinetics(ec_to_enzymes, kinetics_cache_dir)
    for eid, entries in brenda_hits.items():
        kinetics_map.setdefault(eid, []).extend(entries)

    # ── Source 2: SABIO-RK (REST API) ────────────────────────────────
    remaining_ecs = {
        ec: eids for ec, eids in ec_to_enzymes.items()
        if not all(eid in kinetics_map for eid in eids)
    }
    if remaining_ecs:
        sabio_hits = _query_sabio_kinetics(remaining_ecs, kinetics_cache_dir)
        for eid, entries in sabio_hits.items():
            kinetics_map.setdefault(eid, []).extend(entries)

    # ── Source 3: ChEMBL (for remaining gaps) ────────────────────────
    if use_chembl:
        still_missing = {
            ec: [eid for eid in eids if eid not in kinetics_map]
            for ec, eids in ec_to_enzymes.items()
        }
        still_missing = {ec: eids for ec, eids in still_missing.items() if eids}
        if still_missing:
            logger.info("Querying ChEMBL for %d remaining EC classes", len(still_missing))
            chembl_hits = _query_chembl_kinetics(still_missing, kinetics_cache_dir)
            for eid, entries in chembl_hits.items():
                kinetics_map.setdefault(eid, []).extend(entries)

    # Save cache
    serialisable = {}
    for eid, entries in kinetics_map.items():
        serialisable[eid] = [
            {k: v for k, v in e.__dict__.items()} for e in entries
        ]
    with open(cache_file, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    n_with_kinetics = sum(1 for v in kinetics_map.values() if v)
    logger.info("Kinetics enrichment: %d / %d enzymes have kinetic data (%.1f%%)",
                n_with_kinetics, len(records),
                100 * n_with_kinetics / max(1, len(records)))

    return kinetics_map


def _query_brenda_kinetics(
    ec_to_enzymes: Dict[str, List[str]],
    cache_dir: Path,
) -> Dict[str, List[KineticsRecord]]:
    """Query BRENDA for kcat/Km data by EC number.

    Uses the BRENDA SOAP API if available, otherwise falls back to
    cached flat files from the ToPE CurationPipeline.
    """
    results: Dict[str, List[KineticsRecord]] = {}

    # Check for pre-cached BRENDA data from ToPE pipeline
    brenda_cache = cache_dir / "brenda"
    if brenda_cache.exists():
        for ec, enzyme_ids in ec_to_enzymes.items():
            ec_file = brenda_cache / f"{ec.replace('.', '_')}.json"
            if ec_file.exists():
                try:
                    with open(ec_file) as f:
                        data = json.load(f)
                    for entry in data:
                        for eid in enzyme_ids:
                            kr = KineticsRecord(
                                ec_number=ec,
                                substrate=entry.get("substrate", ""),
                                log_kcat=_safe_log10(entry.get("kcat")),
                                log_km=_safe_log10(entry.get("km")),
                                kcat_source="BRENDA",
                                km_source="BRENDA",
                                organism=entry.get("organism", ""),
                                ph=entry.get("ph"),
                                temperature=entry.get("temperature"),
                                confidence="high",
                            )
                            if kr.log_kcat is not None or kr.log_km is not None:
                                kr.log_efficiency = _compute_efficiency(kr.log_kcat, kr.log_km)
                                results.setdefault(eid, []).append(kr)
                except Exception as e:
                    logger.debug("Error reading BRENDA cache %s: %s", ec_file, e)

    logger.info("BRENDA: found kinetics for %d enzymes", len(results))
    return results


def _query_sabio_kinetics(
    ec_to_enzymes: Dict[str, List[str]],
    cache_dir: Path,
) -> Dict[str, List[KineticsRecord]]:
    """Query SABIO-RK REST API for kinetic parameters."""
    results: Dict[str, List[KineticsRecord]] = {}
    sabio_base = "http://sabiork.h-its.org/sabioRestWebServices"

    for ec, enzyme_ids in ec_to_enzymes.items():
        try:
            import urllib.request
            import urllib.parse

            # SABIO-RK query by EC number
            query_url = (
                f"{sabio_base}/searchKineticLaws/sbml"
                f"?q=ECNumber:{ec}"
                f"&fields[]=parameter"
                f"&fields[]=enzymename"
            )

            req = urllib.request.Request(query_url)
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    entries = data if isinstance(data, list) else data.get("results", [])
                    for entry in entries:
                        for eid in enzyme_ids:
                            kcat_val = entry.get("kcat", {}).get("value")
                            km_val = entry.get("km", {}).get("value")
                            kr = KineticsRecord(
                                ec_number=ec,
                                substrate=entry.get("substrate", {}).get("name", ""),
                                log_kcat=_safe_log10(kcat_val),
                                log_km=_safe_log10(km_val),
                                kcat_source="SABIO-RK",
                                km_source="SABIO-RK",
                                organism=entry.get("organism", ""),
                                ph=entry.get("ph"),
                                temperature=entry.get("temperature"),
                                confidence="medium",
                            )
                            if kr.log_kcat is not None or kr.log_km is not None:
                                kr.log_efficiency = _compute_efficiency(kr.log_kcat, kr.log_km)
                                results.setdefault(eid, []).append(kr)
        except Exception as e:
            logger.debug("SABIO-RK query failed for EC %s: %s", ec, e)

        time.sleep(0.5)  # Rate limiting

    logger.info("SABIO-RK: found kinetics for %d enzymes", len(results))
    return results


def _query_chembl_kinetics(
    ec_to_enzymes: Dict[str, List[str]],
    cache_dir: Path,
) -> Dict[str, List[KineticsRecord]]:
    """Query ChEMBL for bioactivity data relevant to enzyme kinetics.

    Looks for IC50, Ki, Kd, and kcat measurements against enzyme targets.
    This is a fallback source — data quality is lower than BRENDA/SABIO-RK.

    NOTE: In the full ToPE pipeline, this uses the ChEMBL MCP server.
    Here we provide a standalone REST API fallback.
    """
    results: Dict[str, List[KineticsRecord]] = {}
    chembl_base = "https://www.ebi.ac.uk/chembl/api/data"

    for ec, enzyme_ids in ec_to_enzymes.items():
        try:
            import urllib.request

            # Search for targets by EC number
            target_url = (
                f"{chembl_base}/target/search.json"
                f"?q={ec}&limit=5"
            )

            req = urllib.request.Request(target_url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                targets = data.get("targets", [])

                for target in targets:
                    target_chembl_id = target.get("target_chembl_id", "")
                    if not target_chembl_id:
                        continue

                    # Get bioactivities for this target
                    activity_url = (
                        f"{chembl_base}/activity.json"
                        f"?target_chembl_id={target_chembl_id}"
                        f"&standard_type__in=IC50,Ki,Kd"
                        f"&limit=20"
                    )

                    act_req = urllib.request.Request(activity_url)
                    with urllib.request.urlopen(act_req, timeout=15) as act_resp:
                        act_data = json.loads(act_resp.read())
                        activities = act_data.get("activities", [])

                        for act in activities:
                            value = act.get("standard_value")
                            units = act.get("standard_units", "")
                            act_type = act.get("standard_type", "")

                            if value is None:
                                continue

                            # Convert to approximate kinetic parameters
                            # IC50/Ki → approximate Km (very rough!)
                            log_val = _safe_log10(float(value))
                            if log_val is None:
                                continue

                            # Unit conversion: nM → M
                            if units == "nM":
                                log_val -= 9
                            elif units == "uM":
                                log_val -= 6

                            for eid in enzyme_ids:
                                kr = KineticsRecord(
                                    ec_number=ec,
                                    substrate=act.get("molecule_chembl_id", ""),
                                    log_km=log_val if act_type in ("IC50", "Ki", "Kd") else None,
                                    kcat_source="",
                                    km_source=f"ChEMBL ({act_type})",
                                    confidence="low",
                                )
                                results.setdefault(eid, []).append(kr)

        except Exception as e:
            logger.debug("ChEMBL query failed for EC %s: %s", ec, e)

        time.sleep(0.3)  # Rate limiting

    logger.info("ChEMBL: found kinetics for %d enzymes", len(results))
    return results


def _safe_log10(value) -> Optional[float]:
    """Safely compute log10, returning None for invalid values."""
    if value is None:
        return None
    try:
        v = float(value)
        if v > 0:
            return float(np.log10(v))
    except (ValueError, TypeError):
        pass
    return None


def _compute_efficiency(
    log_kcat: Optional[float],
    log_km: Optional[float],
) -> Optional[float]:
    """Compute log10(kcat/Km) from individual log values."""
    if log_kcat is not None and log_km is not None:
        return log_kcat - log_km
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Structure File Management
# ══════════════════════════════════════════════════════════════════════════════

def locate_pdb_files(
    records: List[Dict],
    topec_pdb_dir: Path,
) -> Dict[str, Path]:
    """Map enzyme IDs to their PDB file paths.

    TopEC stores structures as:
        structures/{pdb_id}.pdb
        structures/{uniprot_ac}.pdb
        structures/AF-{uniprot_ac}-F1-model_v4.pdb  (AlphaFold)
    """
    pdb_map: Dict[str, Path] = {}

    if not topec_pdb_dir.exists():
        logger.warning("TopEC PDB directory not found: %s", topec_pdb_dir)
        return pdb_map

    # Index all available PDB files
    available_files: Dict[str, Path] = {}
    for ext in ("*.pdb", "*.cif", "*.ent"):
        for f in topec_pdb_dir.rglob(ext):
            stem = f.stem.upper()
            available_files[stem] = f
            # Also index without AF prefix
            if stem.startswith("AF-"):
                parts = stem.split("-")
                if len(parts) >= 2:
                    available_files[parts[1]] = f

    logger.info("Found %d structure files in %s", len(available_files), topec_pdb_dir)

    for rec in records:
        eid = rec["enzyme_id"].upper()
        pdb_id = rec.get("pdb_id", "").upper()

        # Try exact match, then PDB ID, then partial
        for key in (eid, pdb_id, eid[:4]):
            if key and key in available_files:
                pdb_map[rec["enzyme_id"]] = available_files[key]
                break

    logger.info("Mapped %d / %d enzymes to PDB files (%.1f%%)",
                len(pdb_map), len(records),
                100 * len(pdb_map) / max(1, len(records)))

    return pdb_map


def symlink_structures(
    pdb_map: Dict[str, Path],
    tope_pdb_dir: Path,
) -> Dict[str, Path]:
    """Create symlinks in the ToPE directory structure.

    ToPE expects structures at:
        data/raw/pdb/{PDB_ID}.pdb
    """
    tope_pdb_dir.mkdir(parents=True, exist_ok=True)
    new_paths: Dict[str, Path] = {}

    for enzyme_id, src_path in pdb_map.items():
        dst = tope_pdb_dir / f"{enzyme_id}.pdb"
        if dst.exists():
            new_paths[enzyme_id] = dst
            continue

        try:
            if src_path.suffix == ".pdb":
                dst.symlink_to(src_path.resolve())
            else:
                # Copy non-PDB formats with conversion note
                shutil.copy2(src_path, dst)
            new_paths[enzyme_id] = dst
        except OSError as e:
            # Symlink may fail on some filesystems; fall back to copy
            try:
                shutil.copy2(src_path, dst)
                new_paths[enzyme_id] = dst
            except Exception:
                logger.debug("Failed to link/copy %s → %s: %s", src_path, dst, e)

    logger.info("Linked %d structures to %s", len(new_paths), tope_pdb_dir)
    return new_paths


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Fold Split Management
# ══════════════════════════════════════════════════════════════════════════════

def assign_splits(
    records: List[Dict],
    use_topec_splits: bool = True,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> List[Dict]:
    """Assign train/val/test splits.

    If use_topec_splits is True and TopEC split info exists, use it.
    Otherwise, create a random stratified split by EC class.
    """
    if use_topec_splits:
        n_with_split = sum(1 for r in records if r.get("split") in ("train", "val", "test"))
        if n_with_split > len(records) * 0.5:
            logger.info("Using TopEC fold splits (%d pre-assigned)", n_with_split)
            # Fill in any unassigned records
            for rec in records:
                if rec.get("split") not in ("train", "val", "test"):
                    rec["split"] = "train"
            return records

    # Random stratified split
    logger.info("Creating stratified split (val=%.1f%%, test=%.1f%%)",
                val_frac * 100, test_frac * 100)

    rng = np.random.RandomState(seed)

    # Group by EC class for stratification
    ec_groups: Dict[int, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        ec_groups[rec.get("ec_class_idx", -1)].append(i)

    for ec_idx, indices in ec_groups.items():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        for i, idx in enumerate(indices):
            if i < n_test:
                records[idx]["split"] = "test"
            elif i < n_test + n_val:
                records[idx]["split"] = "val"
            else:
                records[idx]["split"] = "train"

    split_counts = defaultdict(int)
    for rec in records:
        split_counts[rec["split"]] += 1
    logger.info("Split: train=%d, val=%d, test=%d",
                split_counts["train"], split_counts["val"], split_counts["test"])

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Unified Index Assembly
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToPERecord:
    """Unified record for the ToPE training pipeline."""

    # Identity
    enzyme_id: str
    pdb_id: str
    uniprot_ac: str
    pdb_path: str

    # Binding site (from TopEC)
    binding_center_x: float
    binding_center_y: float
    binding_center_z: float

    # EC classification
    ec_number: str
    ec_class_idx: int
    ec_level1: int = -1  # First digit (1-7)
    ec_level2: int = -1  # Second digit
    ec_level3: int = -1  # Third digit
    ec_level4: int = -1  # Fourth digit

    # Kinetics (NaN if unavailable)
    log_kcat: float = float("nan")
    log_km: float = float("nan")
    log_efficiency: float = float("nan")
    kinetics_source: str = ""
    kinetics_confidence: str = ""
    has_kinetics: bool = False

    # Zone statistics
    n_zone1_residues: int = 0
    n_zone2_residues: int = 0
    n_zone3_residues: int = 0
    n_total_residues: int = 0

    # Split / provenance
    split: str = "train"
    source_dataset: str = ""
    source_type: str = ""  # pdb_experimental / alphafold

    # Training phase eligibility
    phase_a_eligible: bool = True   # EC pretraining (all structures)
    phase_b_eligible: bool = False  # Kinetics fine-tuning (needs kcat/Km)


def assemble_index(
    records: List[Dict],
    pdb_map: Dict[str, Path],
    kinetics_map: Dict[str, List[KineticsRecord]],
    ec_mapping: Dict[int, str],
    zone_stats: Dict[str, ZoneStats],
    cfg: IngestionConfig,
) -> List[ToPERecord]:
    """Assemble the unified ToPE training index."""
    tope_records = []

    for rec in records:
        eid = rec["enzyme_id"]

        # Skip if no structure available
        if eid not in pdb_map:
            continue

        # EC number resolution
        ec_num = rec.get("ec_number", "")
        if not ec_num and rec.get("ec_class_idx", -1) >= 0:
            ec_num = ec_mapping.get(rec["ec_class_idx"], "")

        ec_levels = _parse_ec_levels(ec_num)

        # Best kinetics entry (highest confidence, prefer kcat)
        best_kinetics = _select_best_kinetics(kinetics_map.get(eid, []))

        # Zone stats
        zs = zone_stats.get(eid)

        center = rec["binding_center"]

        tr = ToPERecord(
            enzyme_id=eid,
            pdb_id=rec.get("pdb_id", ""),
            uniprot_ac=rec.get("uniprot_ac", ""),
            pdb_path=str(pdb_map[eid]),
            binding_center_x=center[0],
            binding_center_y=center[1],
            binding_center_z=center[2],
            ec_number=ec_num,
            ec_class_idx=rec.get("ec_class_idx", -1),
            ec_level1=ec_levels[0],
            ec_level2=ec_levels[1],
            ec_level3=ec_levels[2],
            ec_level4=ec_levels[3],
            log_kcat=best_kinetics.log_kcat if best_kinetics and best_kinetics.log_kcat is not None else float("nan"),
            log_km=best_kinetics.log_km if best_kinetics and best_kinetics.log_km is not None else float("nan"),
            log_efficiency=best_kinetics.log_efficiency if best_kinetics and best_kinetics.log_efficiency is not None else float("nan"),
            kinetics_source=best_kinetics.kcat_source or best_kinetics.km_source if best_kinetics else "",
            kinetics_confidence=best_kinetics.confidence if best_kinetics else "",
            has_kinetics=best_kinetics is not None and (best_kinetics.log_kcat is not None or best_kinetics.log_km is not None),
            n_zone1_residues=zs.n_zone1 if zs else 0,
            n_zone2_residues=zs.n_zone2 if zs else 0,
            n_zone3_residues=zs.n_zone3 if zs else 0,
            n_total_residues=zs.n_total if zs else 0,
            split=rec.get("split", "train"),
            source_dataset=rec.get("source_dataset", ""),
            source_type=rec.get("source_type", ""),
            phase_a_eligible=True,
            phase_b_eligible=best_kinetics is not None and (best_kinetics.log_kcat is not None or best_kinetics.log_km is not None),
        )
        tope_records.append(tr)

    logger.info("Assembled %d ToPE records", len(tope_records))
    return tope_records


def _parse_ec_levels(ec_str: str) -> List[int]:
    """Parse EC number string into 4 integer levels."""
    levels = [-1, -1, -1, -1]
    if not ec_str:
        return levels
    parts = ec_str.strip().split(".")
    for i, p in enumerate(parts[:4]):
        try:
            levels[i] = int(p)
        except ValueError:
            break
    return levels


def _select_best_kinetics(entries: List[KineticsRecord]) -> Optional[KineticsRecord]:
    """Select the best kinetics entry by confidence and completeness."""
    if not entries:
        return None

    # Priority: high confidence with both kcat+Km > high with kcat only > medium > low
    def score(kr: KineticsRecord) -> Tuple[int, int, int]:
        conf_score = {"high": 3, "medium": 2, "low": 1}.get(kr.confidence, 0)
        completeness = (1 if kr.log_kcat is not None else 0) + (1 if kr.log_km is not None else 0)
        has_efficiency = 1 if kr.log_efficiency is not None else 0
        return (conf_score, completeness, has_efficiency)

    return max(entries, key=score)


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Export
# ══════════════════════════════════════════════════════════════════════════════

def export_index(
    tope_records: List[ToPERecord],
    output_parquet: Path,
    output_json: Path,
):
    """Export the unified index in Parquet and JSON formats."""
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts
    rows = [r.__dict__ for r in tope_records]

    # JSON export (always works)
    with open(output_json, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("Exported JSON index: %s (%d records)", output_json, len(rows))

    # Parquet export (if pandas available)
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_parquet(output_parquet, index=False)
        logger.info("Exported Parquet index: %s", output_parquet)

        # Print summary statistics
        _print_summary(df)
    else:
        logger.warning("pandas not available; skipping Parquet export")


def _print_summary(df):
    """Print dataset summary statistics."""
    print("\n" + "=" * 70)
    print("ToPE Dataset Summary (TopEC Ingestion)")
    print("=" * 70)

    print(f"\nTotal records:          {len(df):,}")
    print(f"Unique enzymes:         {df['enzyme_id'].nunique():,}")
    print(f"Unique PDB IDs:         {df['pdb_id'].nunique():,}")
    print(f"Unique EC numbers:      {df['ec_number'].nunique():,}")

    print(f"\n── Source breakdown ──")
    for src, count in df["source_type"].value_counts().items():
        print(f"  {src:25s} {count:>6,}")

    print(f"\n── Split breakdown ──")
    for split, count in df["split"].value_counts().items():
        print(f"  {split:25s} {count:>6,}")

    print(f"\n── Kinetics coverage ──")
    n_kin = df["has_kinetics"].sum()
    print(f"  With kinetics:        {n_kin:>6,} ({100*n_kin/len(df):.1f}%)")
    print(f"  Phase A eligible:     {df['phase_a_eligible'].sum():>6,}")
    print(f"  Phase B eligible:     {df['phase_b_eligible'].sum():>6,}")

    if n_kin > 0:
        kin_df = df[df["has_kinetics"]]
        for col, label in [("log_kcat", "log₁₀(kcat)"), ("log_km", "log₁₀(Km)"), ("log_efficiency", "log₁₀(kcat/Km)")]:
            valid = kin_df[col].dropna()
            if len(valid) > 0:
                print(f"  {label:25s} μ={valid.mean():.2f}, σ={valid.std():.2f}, "
                      f"range=[{valid.min():.1f}, {valid.max():.1f}], n={len(valid)}")

    print(f"\n── Zone statistics ──")
    for zone, col in [("Zone 1 (0-8Å)", "n_zone1_residues"),
                      ("Zone 2 (8-20Å)", "n_zone2_residues"),
                      ("Zone 3 (>20Å)", "n_zone3_residues")]:
        valid = df[col][df[col] > 0]
        if len(valid) > 0:
            print(f"  {zone:25s} μ={valid.mean():.1f} residues, "
                  f"range=[{valid.min()}, {valid.max()}]")

    # Kinetics source breakdown
    if n_kin > 0:
        print(f"\n── Kinetics sources ──")
        for src, count in kin_df["kinetics_source"].value_counts().items():
            print(f"  {src:25s} {count:>6,}")

    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Section 9: Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion(cfg: IngestionConfig) -> List[ToPERecord]:
    """Execute the full TopEC → ToPE ingestion pipeline.

    Steps:
        1. Discover and parse TopEC CSV files
        2. Build EC class → EC number mapping
        3. Locate PDB structure files
        4. Symlink structures into ToPE directory
        5. Optionally enrich with kinetic parameters
        6. Compute zone statistics (if structures available)
        7. Assign train/val/test splits
        8. Assemble unified index
        9. Export as Parquet + JSON

    Returns
    -------
    list of ToPERecord
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("TopEC → ToPE Data Ingestion Pipeline")
    logger.info("=" * 70)

    # ── 1. Parse TopEC CSVs ──────────────────────────────────────────
    logger.info("─── Step 1: Discovering TopEC datasets ───")
    csvs = discover_topec_csvs(cfg.topec_csv_dir)
    if not csvs:
        logger.error("No TopEC CSV files found in %s", cfg.topec_csv_dir)
        sys.exit(1)

    all_records: List[Dict] = []
    seen_enzymes: Set[str] = set()

    for dataset_name, csv_path in csvs.items():
        records = parse_topec_csv(csv_path)
        # Deduplicate across datasets (prefer first occurrence)
        for rec in records:
            if rec["enzyme_id"] not in seen_enzymes:
                seen_enzymes.add(rec["enzyme_id"])
                all_records.append(rec)

    logger.info("Total unique enzyme records: %d", len(all_records))

    # ── 2. Build EC mapping ──────────────────────────────────────────
    logger.info("─── Step 2: Building EC class mapping ───")
    ec_mapping = build_ec_class_mapping(all_records)

    # Fill in missing EC numbers from mapping
    for rec in all_records:
        if not rec.get("ec_number") and rec.get("ec_class_idx", -1) >= 0:
            rec["ec_number"] = ec_mapping.get(rec["ec_class_idx"], "")

    # ── 3. Locate PDB files ──────────────────────────────────────────
    logger.info("─── Step 3: Locating PDB structure files ───")
    pdb_map = locate_pdb_files(all_records, cfg.topec_pdb_dir)

    # ── 4. Symlink into ToPE directory ───────────────────────────────
    logger.info("─── Step 4: Linking structures to ToPE directory ───")
    tope_pdb_dir = cfg.tope_data_dir / "raw" / "pdb"
    pdb_map = symlink_structures(pdb_map, tope_pdb_dir)

    # ── 5. Kinetics enrichment ───────────────────────────────────────
    kinetics_map: Dict[str, List[KineticsRecord]] = {}
    if cfg.enrich_kinetics:
        logger.info("─── Step 5: Enriching with kinetic parameters ───")
        kinetics_map = enrich_with_kinetics(
            all_records, cfg.kinetics_cache_dir, cfg.chembl_enrichment
        )
    else:
        logger.info("─── Step 5: Skipping kinetics enrichment ───")

    # ── 6. Zone statistics ───────────────────────────────────────────
    logger.info("─── Step 6: Computing zone statistics ───")
    zone_stats: Dict[str, ZoneStats] = {}
    for rec in all_records:
        eid = rec["enzyme_id"]
        if eid in pdb_map:
            zs = compute_zone_stats_from_pdb(
                pdb_map[eid], rec["binding_center"],
                cfg.zone1_radius, cfg.zone2_radius,
            )
            if zs:
                zone_stats[eid] = zs

    logger.info("Computed zone stats for %d / %d enzymes",
                len(zone_stats), len(pdb_map))

    # ── 7. Assign splits ─────────────────────────────────────────────
    logger.info("─── Step 7: Assigning train/val/test splits ───")
    all_records = assign_splits(
        all_records, cfg.use_topec_splits,
        cfg.val_fraction, cfg.test_fraction,
    )

    # ── 8. Assemble index ────────────────────────────────────────────
    logger.info("─── Step 8: Assembling unified ToPE index ───")
    tope_records = assemble_index(
        all_records, pdb_map, kinetics_map, ec_mapping, zone_stats, cfg,
    )

    # ── 9. Export ────────────────────────────────────────────────────
    logger.info("─── Step 9: Exporting index ───")
    export_index(tope_records, cfg.output_index, cfg.output_json)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Ingestion complete: %d records in %.1f s", len(tope_records), elapsed)
    logger.info("=" * 70)

    return tope_records


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TopEC → ToPE Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion (EC labels only, no kinetics)
  python topec_to_tope_ingestion.py \\
      --topec-csv-dir TopEC/data/csv \\
      --topec-pdb-dir TopEC/structures

  # Full ingestion with kinetics enrichment
  python topec_to_tope_ingestion.py \\
      --topec-csv-dir TopEC/data/csv \\
      --topec-pdb-dir TopEC/structures \\
      --enrich-kinetics --chembl

  # Using pre-downloaded H5 dataset
  python topec_to_tope_ingestion.py \\
      --topec-csv-dir TopEC/data/csv \\
      --topec-h5 TopEC/enzyme_dataset.h5
        """,
    )

    parser.add_argument("--topec-csv-dir", type=Path, required=True,
                        help="Path to TopEC/data/csv directory")
    parser.add_argument("--topec-pdb-dir", type=Path, default=None,
                        help="Path to TopEC PDB structures directory")
    parser.add_argument("--topec-h5", type=Path, default=None,
                        help="Path to TopEC pre-built H5 dataset")
    parser.add_argument("--tope-data-dir", type=Path, default=Path("data"),
                        help="ToPE data output directory (default: ./data)")
    parser.add_argument("--enrich-kinetics", action="store_true",
                        help="Enrich with kinetic parameters from BRENDA/SABIO-RK")
    parser.add_argument("--chembl", action="store_true",
                        help="Also query ChEMBL for kinetics (slower)")
    parser.add_argument("--zone1-radius", type=float, default=8.0,
                        help="Zone 1 radius in Å (default: 8.0)")
    parser.add_argument("--zone2-radius", type=float, default=20.0,
                        help="Zone 2 radius in Å (default: 20.0)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--no-topec-splits", action="store_true",
                        help="Ignore TopEC fold splits; create new random splits")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = IngestionConfig(
        topec_csv_dir=args.topec_csv_dir,
        topec_pdb_dir=args.topec_pdb_dir or args.topec_csv_dir.parent.parent / "structures",
        topec_h5_path=args.topec_h5,
        tope_data_dir=args.tope_data_dir,
        output_index=args.tope_data_dir / "processed" / "topec_tope_index.parquet",
        output_json=args.tope_data_dir / "processed" / "topec_tope_index.json",
        zone1_radius=args.zone1_radius,
        zone2_radius=args.zone2_radius,
        enrich_kinetics=args.enrich_kinetics,
        chembl_enrichment=args.chembl,
        kinetics_cache_dir=args.tope_data_dir / "raw" / "kinetics",
        use_topec_splits=not args.no_topec_splits,
        n_workers=args.workers,
        verbose=args.verbose,
    )

    records = run_ingestion(cfg)

    # Quick validation
    n_phase_a = sum(1 for r in records if r.phase_a_eligible)
    n_phase_b = sum(1 for r in records if r.phase_b_eligible)
    print(f"\n✓ Phase A (EC pretraining):    {n_phase_a:,} structures ready")
    print(f"✓ Phase B (kinetics fine-tune): {n_phase_b:,} structures with kinetics")
    print(f"  Kinetics coverage: {100*n_phase_b/max(1,n_phase_a):.1f}%")


if __name__ == "__main__":
    main()
