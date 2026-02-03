"""
M-CSA (Mechanism and Catalytic Site Atlas) client.

Fetches catalytic site annotations — which residues participate in catalysis,
their mechanistic roles, and the associated EC numbers — for every curated
enzyme entry. This information seeds the active-site extraction step and
provides ground-truth labels for the inverse-attribution validation
(Phase 4 of the ToPE roadmap).

Reference: Ribeiro et al., Nucleic Acids Res. 46 (2018) D618–D623.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from data_curation.config import (
    MCSA_CSV_URL,
    MCSA_DIR,
    MCSA_ENTRIES_URL,
    MCSA_RESIDUES_URL,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CatalyticResidue:
    """A single residue annotated as part of a catalytic mechanism."""

    chain_id: str
    residue_name: str          # three-letter code, e.g. "HIS"
    residue_number: int
    role: str                  # e.g. "Proton donor", "Nucleophile", "Electrostatic stabiliser"
    chemical_function: str     # e.g. "proton_shuttle", "covalent_catalysis"

    @property
    def residue_id(self) -> str:
        return f"{self.chain_id}:{self.residue_name}{self.residue_number}"


@dataclass
class MCSAEntry:
    """One M-CSA curated enzyme entry."""

    mcsa_id: int
    pdb_id: str
    ec_number: str
    enzyme_name: str
    organism: str
    catalytic_residues: List[CatalyticResidue] = field(default_factory=list)

    @property
    def ec_top_level(self) -> str:
        return self.ec_number.split(".")[0] if self.ec_number else ""

    @property
    def catalytic_chain_ids(self) -> Set[str]:
        return {r.chain_id for r in self.catalytic_residues}

    @property
    def catalytic_residue_numbers(self) -> Dict[str, List[int]]:
        """Map chain_id → sorted list of catalytic residue sequence numbers."""
        result: Dict[str, List[int]] = {}
        for r in self.catalytic_residues:
            result.setdefault(r.chain_id, []).append(r.residue_number)
        return {k: sorted(v) for k, v in result.items()}


# ── Client ────────────────────────────────────────────────────────────────────

class MCSAClient:
    """Fetch and cache M-CSA catalytic-site annotations."""

    def __init__(
        self,
        cache_dir: Path = MCSA_DIR,
        request_delay: float = 0.25,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ToPE-DataCuration/1.0",
        })

    # ── Bulk download ────────────────────────────────────────────────────

    def fetch_all_entries_csv(self, force: bool = False) -> Path:
        """Download the full M-CSA entries table as CSV."""
        csv_path = self.cache_dir / "mcsa_entries.csv"
        if csv_path.exists() and not force:
            logger.info("M-CSA CSV already cached at %s", csv_path)
            return csv_path

        logger.info("Downloading M-CSA entries CSV …")
        resp = self._session.get(MCSA_CSV_URL, timeout=120)
        resp.raise_for_status()
        csv_path.write_text(resp.text, encoding="utf-8")
        logger.info("Saved %d bytes → %s", len(resp.text), csv_path)
        return csv_path

    def load_entries_from_csv(self, csv_path: Optional[Path] = None) -> List[MCSAEntry]:
        """Parse the bulk CSV into MCSAEntry objects (without residue details)."""
        if csv_path is None:
            csv_path = self.fetch_all_entries_csv()

        entries: List[MCSAEntry] = []
        text = csv_path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            entry = MCSAEntry(
                mcsa_id=int(row.get("mcsa_id") or row.get("id", 0)),
                pdb_id=row.get("pdb_id", "").strip().upper(),
                ec_number=row.get("ec_number", "").strip(),
                enzyme_name=row.get("enzyme_name", "").strip(),
                organism=row.get("organism", "").strip(),
            )
            if entry.pdb_id and len(entry.pdb_id) == 4:
                entries.append(entry)

        logger.info("Parsed %d M-CSA entries from CSV", len(entries))
        return entries

    # ── Per-entry residue detail ─────────────────────────────────────────

    def fetch_residues(self, mcsa_id: int) -> List[CatalyticResidue]:
        """Fetch catalytic residue annotations for a single M-CSA entry."""
        cache_path = self.cache_dir / "residues" / f"{mcsa_id}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            data = json.loads(cache_path.read_text())
        else:
            url = f"{MCSA_RESIDUES_URL}?entry_id={mcsa_id}&format=json"
            time.sleep(self.request_delay)
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cache_path.write_text(json.dumps(data, indent=2))

        return self._parse_residues(data)

    def enrich_entry(self, entry: MCSAEntry) -> MCSAEntry:
        """Populate an entry's catalytic_residues list from the API."""
        if not entry.catalytic_residues:
            entry.catalytic_residues = self.fetch_residues(entry.mcsa_id)
        return entry

    # ── Batch operations ─────────────────────────────────────────────────

    def fetch_all_with_residues(
        self,
        entries: Optional[List[MCSAEntry]] = None,
        max_entries: Optional[int] = None,
    ) -> List[MCSAEntry]:
        """Bulk-fetch residue annotations for all entries.

        Parameters
        ----------
        entries : list, optional
            Pre-loaded entries. If None, loads from the CSV first.
        max_entries : int, optional
            Cap the number of entries to fetch (useful for development).
        """
        if entries is None:
            entries = self.load_entries_from_csv()

        if max_entries is not None:
            entries = entries[:max_entries]

        total = len(entries)
        enriched: List[MCSAEntry] = []
        for i, entry in enumerate(entries, 1):
            try:
                self.enrich_entry(entry)
                enriched.append(entry)
            except requests.RequestException as exc:
                logger.warning(
                    "Failed to fetch residues for M-CSA %d (%s): %s",
                    entry.mcsa_id, entry.pdb_id, exc,
                )
            if i % 100 == 0 or i == total:
                logger.info("Fetched residues: %d / %d", i, total)

        logger.info(
            "Enriched %d / %d entries with catalytic residue annotations",
            len(enriched), total,
        )
        return enriched

    # ── Filtering helpers ────────────────────────────────────────────────

    @staticmethod
    def filter_by_ec(entries: List[MCSAEntry], ec_prefix: str) -> List[MCSAEntry]:
        """Filter entries whose EC number starts with a given prefix (e.g. '3.4')."""
        return [e for e in entries if e.ec_number.startswith(ec_prefix)]

    @staticmethod
    def unique_pdb_ids(entries: List[MCSAEntry]) -> List[str]:
        """Deduplicated list of PDB IDs across entries."""
        seen: Set[str] = set()
        result: List[str] = []
        for e in entries:
            if e.pdb_id not in seen:
                seen.add(e.pdb_id)
                result.append(e.pdb_id)
        return result

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_residues(data) -> List[CatalyticResidue]:
        """Convert raw M-CSA JSON residue data into CatalyticResidue objects."""
        residues: List[CatalyticResidue] = []

        items = data if isinstance(data, list) else data.get("results", data.get("residues", []))

        for item in items:
            try:
                residues.append(CatalyticResidue(
                    chain_id=str(item.get("chain_id", item.get("chain", "A"))).strip(),
                    residue_name=str(item.get("residue_name", item.get("code", "UNK"))).strip().upper(),
                    residue_number=int(item.get("residue_number", item.get("resid", 0))),
                    role=str(item.get("role", item.get("function", ""))).strip(),
                    chemical_function=str(item.get("chemical_function", "")).strip(),
                ))
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed residue record: %s", exc)

        return residues

    # ── Summary ──────────────────────────────────────────────────────────

    @staticmethod
    def summarise(entries: List[MCSAEntry]) -> Dict:
        """Produce a summary dict of the loaded dataset."""
        ec_counts: Dict[str, int] = {}
        total_residues = 0
        for e in entries:
            ec_top = e.ec_top_level or "unknown"
            ec_counts[ec_top] = ec_counts.get(ec_top, 0) + 1
            total_residues += len(e.catalytic_residues)

        return {
            "total_entries": len(entries),
            "unique_pdb_ids": len({e.pdb_id for e in entries}),
            "total_catalytic_residues": total_residues,
            "ec_distribution": dict(sorted(ec_counts.items())),
        }
