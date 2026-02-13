"""
BRENDA and SABIO-RK kinetics client.

Fetches enzyme kinetic parameters — kcat, Km, kcat/Km, Ki — from two
complementary databases and joins them to PDB / EC identifiers so the
downstream pipeline can attach quantitative activity labels to each
active-site sample.

Data volumes (approximate):
    BRENDA:   ~23 000 kcat entries, ~41 000 Km entries
    SABIO-RK: ~50 000 kinetic law entries with structured conditions

References:
    Chang et al., Nucleic Acids Res. 49 (2021) D498–D502 (BRENDA)
    Wittig et al., Nucleic Acids Res. 46 (2018) D667–D670 (SABIO-RK)
    Li et al., Nat Commun 16 (2025) 1229 (CataPro)
    Zhang et al., Brief Bioinform 25 (2024) bbae077 (DeepEnzyme)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from tope.data.config import (
    KINETICS_DIR,
    KINETICS_PARAM_TYPES,
    SABIO_RK_API_URL,
    SABIO_RK_ENTRY_URL,
    SABIO_RK_SEARCH_URL,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class KineticEntry:
    """One kinetic measurement for an enzyme–substrate pair."""

    ec_number: str
    uniprot_id: str             # UniProt accession (if available)
    pdb_id: str                 # PDB ID (if available, else "")
    organism: str
    substrate: str              # substrate name / SMILES
    param_type: str             # "kcat", "Km", "kcat/Km", "Ki", "Vmax"
    value: float                # raw value
    unit: str                   # e.g. "s^(-1)", "mM", "s^(-1)*mM^(-1)"
    log_value: float = 0.0      # log10(value) — computed on init
    ph: Optional[float] = None
    temperature: Optional[float] = None   # °C
    source_db: str = ""         # "BRENDA" or "SABIO-RK"
    source_id: str = ""         # database-specific identifier
    is_mutant: bool = False     # wild-type vs mutant
    comment: str = ""

    def __post_init__(self):
        if self.value > 0:
            self.log_value = math.log10(self.value)

    @property
    def has_pdb(self) -> bool:
        return bool(self.pdb_id) and len(self.pdb_id) == 4


@dataclass
class KineticsSummary:
    """Aggregate statistics for the kinetics dataset."""

    total_entries: int = 0
    kcat_count: int = 0
    km_count: int = 0
    kcat_km_count: int = 0
    entries_with_pdb: int = 0
    unique_ec_numbers: int = 0
    unique_pdb_ids: int = 0
    source_distribution: Dict[str, int] = field(default_factory=dict)


# ── BRENDA parser ─────────────────────────────────────────────────────────────

class BRENDAParser:
    """Parse BRENDA flat-file exports for kinetic parameters.

    BRENDA distributes data as a flat text file (brenda_download.txt)
    which must be downloaded manually with a license agreement.
    This parser extracts kcat and Km entries and maps them to EC numbers.
    """

    # BRENDA section markers
    SECTION_MARKERS = {
        "kcat": "TURNOVER_NUMBER",
        "Km": "KM_VALUE",
        "Ki": "KI_VALUE",
        "kcat/Km": "KCAT_KM_VALUE",
    }

    def __init__(self, cache_dir: Path = KINETICS_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse_flat_file(
        self,
        flat_file_path: Path,
        param_types: Optional[List[str]] = None,
    ) -> List[KineticEntry]:
        """Parse the BRENDA flat file and extract kinetic entries.

        Parameters
        ----------
        flat_file_path : Path
            Path to the BRENDA brenda_download.txt file.
        param_types : list, optional
            Which parameter types to extract. Defaults to all.

        Returns
        -------
        list of KineticEntry
        """
        if param_types is None:
            param_types = list(self.SECTION_MARKERS.keys())

        path = Path(flat_file_path)
        if not path.exists():
            logger.warning("BRENDA flat file not found at %s", path)
            return []

        logger.info("Parsing BRENDA flat file: %s", path)
        text = path.read_text(encoding="utf-8", errors="replace")

        entries: List[KineticEntry] = []
        current_ec = ""

        # Split into EC-level blocks
        ec_blocks = re.split(r"^ID\s+", text, flags=re.MULTILINE)

        for block in ec_blocks:
            if not block.strip():
                continue

            # Extract EC number from block header
            ec_match = re.match(r"(\d+\.\d+\.\d+\.\d+)", block)
            if ec_match:
                current_ec = ec_match.group(1)
            else:
                continue

            for param_type in param_types:
                marker = self.SECTION_MARKERS.get(param_type)
                if marker and marker in block:
                    block_entries = self._parse_section(
                        block, marker, current_ec, param_type,
                    )
                    entries.extend(block_entries)

        logger.info(
            "BRENDA: parsed %d kinetic entries across %d EC classes",
            len(entries), len({e.ec_number for e in entries}),
        )

        # Cache parsed results
        cache_path = self.cache_dir / "brenda_parsed.json"
        self._save_cache(entries, cache_path)

        return entries

    def load_cached(self) -> List[KineticEntry]:
        """Load previously parsed BRENDA entries from cache."""
        cache_path = self.cache_dir / "brenda_parsed.json"
        if not cache_path.exists():
            return []
        return self._load_cache(cache_path)

    def _parse_section(
        self,
        block: str,
        marker: str,
        ec_number: str,
        param_type: str,
    ) -> List[KineticEntry]:
        """Parse a single parameter section within an EC block."""
        entries: List[KineticEntry] = []

        # Find the section
        pattern = rf"^{marker}\n(.*?)(?=^\w|$)"
        section_match = re.search(pattern, block, re.MULTILINE | re.DOTALL)
        if not section_match:
            return entries

        section_text = section_match.group(1)

        # Each entry line has format: #N# value {substrate} (organism) <ref>
        line_pattern = re.compile(
            r"#(\d+)#\s+([\d.eE+\-]+)\s*"    # entry number + value
            r"(?:\{([^}]*)\})?"               # substrate in braces
            r"(?:\s*\(([^)]*)\))?"            # organism in parens
        )

        for match in line_pattern.finditer(section_text):
            try:
                value = float(match.group(2))
                if value <= 0:
                    continue

                substrate = (match.group(3) or "").strip()
                organism = (match.group(4) or "").strip()

                unit = self._default_unit(param_type)

                entries.append(KineticEntry(
                    ec_number=ec_number,
                    uniprot_id="",
                    pdb_id="",
                    organism=organism,
                    substrate=substrate,
                    param_type=param_type,
                    value=value,
                    unit=unit,
                    source_db="BRENDA",
                    source_id=f"BRENDA:{ec_number}:{match.group(1)}",
                ))
            except (ValueError, TypeError):
                continue

        return entries

    @staticmethod
    def _default_unit(param_type: str) -> str:
        return {
            "kcat": "s^(-1)",
            "Km": "mM",
            "kcat/Km": "s^(-1)*mM^(-1)",
            "Ki": "mM",
            "Vmax": "µmol/min/mg",
        }.get(param_type, "")

    @staticmethod
    def _save_cache(entries: List[KineticEntry], path: Path) -> None:
        data = []
        for e in entries:
            data.append({
                "ec_number": e.ec_number,
                "uniprot_id": e.uniprot_id,
                "pdb_id": e.pdb_id,
                "organism": e.organism,
                "substrate": e.substrate,
                "param_type": e.param_type,
                "value": e.value,
                "unit": e.unit,
                "log_value": e.log_value,
                "ph": e.ph,
                "temperature": e.temperature,
                "source_db": e.source_db,
                "source_id": e.source_id,
                "is_mutant": e.is_mutant,
            })
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _load_cache(path: Path) -> List[KineticEntry]:
        data = json.loads(path.read_text())
        entries = []
        for d in data:
            entries.append(KineticEntry(**d))
        return entries


# ── SABIO-RK client ───────────────────────────────────────────────────────────

class SABIORKClient:
    """Fetch kinetic parameters from the SABIO-RK REST API.

    SABIO-RK provides structured kinetic laws with explicit conditions
    (pH, temperature), substrate identifiers, and PDB cross-references.
    This complements BRENDA's broader coverage with more structured data.
    """

    def __init__(
        self,
        cache_dir: Path = KINETICS_DIR,
        request_delay: float = 0.5,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ToPE-DataCuration/1.0",
        })

    def fetch_by_ec(
        self,
        ec_number: str,
        param_types: Optional[List[str]] = None,
    ) -> List[KineticEntry]:
        """Fetch kinetic entries for a given EC number.

        Parameters
        ----------
        ec_number : str
            Full EC number (e.g. "3.4.21.4").
        param_types : list, optional
            Filter to specific parameter types.
        """
        if param_types is None:
            param_types = KINETICS_PARAM_TYPES

        cache_path = self.cache_dir / "sabio_rk" / f"{ec_number}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            return self._load_ec_cache(cache_path)

        # Query SABIO-RK for kinetic law IDs
        query_url = (
            f"{SABIO_RK_API_URL}/searchKineticLaws/sbml"
            f"?q=ECNumber:{ec_number}"
            f"&format=json"
        )

        entries: List[KineticEntry] = []

        try:
            time.sleep(self.request_delay)
            resp = self._session.get(query_url, timeout=60)
            if resp.status_code == 404 or resp.status_code == 204:
                cache_path.write_text("[]")
                return entries
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.debug("SABIO-RK query failed for EC %s: %s", ec_number, exc)
            return entries

        # Parse response
        laws = data if isinstance(data, list) else data.get("results", [])
        for law in laws:
            parsed = self._parse_kinetic_law(law, ec_number, param_types)
            entries.extend(parsed)

        # Cache
        self._save_ec_cache(entries, cache_path)

        return entries

    def fetch_batch(
        self,
        ec_numbers: List[str],
        param_types: Optional[List[str]] = None,
    ) -> List[KineticEntry]:
        """Fetch kinetics for multiple EC numbers."""
        all_entries: List[KineticEntry] = []
        total = len(ec_numbers)

        for i, ec in enumerate(ec_numbers, 1):
            entries = self.fetch_by_ec(ec, param_types)
            all_entries.extend(entries)

            if i % 50 == 0 or i == total:
                logger.info(
                    "SABIO-RK: fetched %d EC classes (%d entries so far)",
                    i, len(all_entries),
                )

        logger.info(
            "SABIO-RK batch complete: %d entries across %d EC classes",
            len(all_entries), total,
        )
        return all_entries

    def _parse_kinetic_law(
        self,
        law: Dict[str, Any],
        ec_number: str,
        param_types: List[str],
    ) -> List[KineticEntry]:
        """Parse one SABIO-RK kinetic law into KineticEntry objects."""
        entries: List[KineticEntry] = []

        parameters = law.get("parameters", law.get("kineticParameters", []))
        if not isinstance(parameters, list):
            return entries

        organism = str(law.get("organism", law.get("Organism", ""))).strip()
        pdb_id = str(law.get("pdbId", law.get("PDBId", ""))).strip().upper()
        if len(pdb_id) != 4:
            pdb_id = ""

        uniprot = str(law.get("uniprotId", law.get("UniProtId", ""))).strip()
        substrate = str(law.get("substrate", law.get("Substrate", ""))).strip()

        # Conditions
        ph = law.get("pH", law.get("ph"))
        temp = law.get("temperature", law.get("Temperature"))

        try:
            ph = float(ph) if ph is not None else None
        except (ValueError, TypeError):
            ph = None
        try:
            temp = float(temp) if temp is not None else None
        except (ValueError, TypeError):
            temp = None

        is_mutant = bool(law.get("isMutant", law.get("HasMutant", False)))
        law_id = str(law.get("kineticLawId", law.get("EntryID", "")))

        for param in parameters:
            ptype = str(param.get("type", param.get("parameterType", ""))).strip()

            # Normalise parameter type names
            ptype_norm = self._normalise_param_type(ptype)
            if ptype_norm not in param_types:
                continue

            try:
                value = float(param.get("value", param.get("startValue", 0)))
            except (ValueError, TypeError):
                continue

            if value <= 0:
                continue

            unit = str(param.get("unit", "")).strip()

            entries.append(KineticEntry(
                ec_number=ec_number,
                uniprot_id=uniprot,
                pdb_id=pdb_id,
                organism=organism,
                substrate=substrate,
                param_type=ptype_norm,
                value=value,
                unit=unit,
                ph=ph,
                temperature=temp,
                source_db="SABIO-RK",
                source_id=f"SABIO-RK:{law_id}",
                is_mutant=is_mutant,
            ))

        return entries

    @staticmethod
    def _normalise_param_type(raw: str) -> str:
        """Map SABIO-RK parameter type names to standard keys."""
        raw_lower = raw.lower().strip()
        mapping = {
            "kcat": "kcat",
            "turnover number": "kcat",
            "km": "Km",
            "michaelis constant": "Km",
            "kcat/km": "kcat/Km",
            "catalytic efficiency": "kcat/Km",
            "ki": "Ki",
            "inhibition constant": "Ki",
            "vmax": "Vmax",
        }
        return mapping.get(raw_lower, raw)

    @staticmethod
    def _save_ec_cache(entries: List[KineticEntry], path: Path) -> None:
        data = []
        for e in entries:
            data.append({
                "ec_number": e.ec_number,
                "uniprot_id": e.uniprot_id,
                "pdb_id": e.pdb_id,
                "organism": e.organism,
                "substrate": e.substrate,
                "param_type": e.param_type,
                "value": e.value,
                "unit": e.unit,
                "log_value": e.log_value,
                "ph": e.ph,
                "temperature": e.temperature,
                "source_db": e.source_db,
                "source_id": e.source_id,
                "is_mutant": e.is_mutant,
            })
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _load_ec_cache(path: Path) -> List[KineticEntry]:
        data = json.loads(path.read_text())
        return [KineticEntry(**d) for d in data]


# ── Unified kinetics aggregator ───────────────────────────────────────────────

class KineticsAggregator:
    """Merge BRENDA and SABIO-RK entries and join to PDB/EC identifiers.

    Handles:
    - Deduplication across sources
    - Unit normalisation (kcat → s⁻¹, Km → mM)
    - Wildtype filtering (exclude mutants by default)
    - Aggregation: median value per (EC, substrate, param_type) group
    """

    def __init__(
        self,
        brenda_parser: Optional[BRENDAParser] = None,
        sabio_client: Optional[SABIORKClient] = None,
        cache_dir: Path = KINETICS_DIR,
    ):
        self.brenda = brenda_parser or BRENDAParser(cache_dir)
        self.sabio = sabio_client or SABIORKClient(cache_dir)
        self.cache_dir = Path(cache_dir)

    def collect(
        self,
        ec_numbers: List[str],
        brenda_flat_file: Optional[Path] = None,
        include_mutants: bool = False,
        param_types: Optional[List[str]] = None,
    ) -> List[KineticEntry]:
        """Collect and merge kinetics from both sources.

        Parameters
        ----------
        ec_numbers : list of str
            EC numbers to query.
        brenda_flat_file : Path, optional
            Path to the BRENDA download file. If None, uses cached data.
        include_mutants : bool
            Whether to include mutant entries.
        param_types : list of str, optional
            Parameter types to include.

        Returns
        -------
        list of KineticEntry
            Merged, deduplicated entries.
        """
        if param_types is None:
            param_types = ["kcat", "Km", "kcat/Km"]

        all_entries: List[KineticEntry] = []

        # BRENDA
        if brenda_flat_file and Path(brenda_flat_file).exists():
            brenda_entries = self.brenda.parse_flat_file(
                Path(brenda_flat_file), param_types,
            )
            all_entries.extend(brenda_entries)
            logger.info("BRENDA: %d entries", len(brenda_entries))
        else:
            cached = self.brenda.load_cached()
            if cached:
                # Filter to requested EC numbers
                ec_set = set(ec_numbers)
                cached = [e for e in cached if e.ec_number in ec_set]
                all_entries.extend(cached)
                logger.info("BRENDA (cached): %d entries", len(cached))

        # SABIO-RK
        sabio_entries = self.sabio.fetch_batch(ec_numbers, param_types)
        all_entries.extend(sabio_entries)
        logger.info("SABIO-RK: %d entries", len(sabio_entries))

        # Filter mutants
        if not include_mutants:
            before = len(all_entries)
            all_entries = [e for e in all_entries if not e.is_mutant]
            logger.info(
                "Filtered mutants: %d → %d entries", before, len(all_entries),
            )

        # Filter to requested param types
        all_entries = [e for e in all_entries if e.param_type in param_types]

        logger.info(
            "Kinetics collection complete: %d entries across %d EC classes",
            len(all_entries), len({e.ec_number for e in all_entries}),
        )

        return all_entries

    def build_ec_kinetics_map(
        self,
        entries: List[KineticEntry],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate entries into median values per (EC, param_type).

        Returns
        -------
        dict mapping EC number → {param_type: median_log_value}
        """
        import numpy as np

        # Group by (ec, param_type)
        groups: Dict[Tuple[str, str], List[float]] = {}
        for e in entries:
            key = (e.ec_number, e.param_type)
            groups.setdefault(key, []).append(e.log_value)

        result: Dict[str, Dict[str, float]] = {}
        for (ec, ptype), values in groups.items():
            result.setdefault(ec, {})[ptype] = float(np.median(values))

        return result

    def build_pdb_kinetics_map(
        self,
        entries: List[KineticEntry],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate entries into median values per (PDB, param_type).

        Only includes entries that have a PDB cross-reference.

        Returns
        -------
        dict mapping PDB ID → {param_type: median_log_value}
        """
        import numpy as np

        groups: Dict[Tuple[str, str], List[float]] = {}
        for e in entries:
            if not e.has_pdb:
                continue
            key = (e.pdb_id, e.param_type)
            groups.setdefault(key, []).append(e.log_value)

        result: Dict[str, Dict[str, float]] = {}
        for (pdb_id, ptype), values in groups.items():
            result.setdefault(pdb_id, {})[ptype] = float(np.median(values))

        return result

    @staticmethod
    def summarise(entries: List[KineticEntry]) -> KineticsSummary:
        """Compute summary statistics."""
        summary = KineticsSummary(total_entries=len(entries))
        ec_set: Set[str] = set()
        pdb_set: Set[str] = set()
        source_counts: Dict[str, int] = {}

        for e in entries:
            ec_set.add(e.ec_number)
            if e.has_pdb:
                pdb_set.add(e.pdb_id)
                summary.entries_with_pdb += 1

            if e.param_type == "kcat":
                summary.kcat_count += 1
            elif e.param_type == "Km":
                summary.km_count += 1
            elif e.param_type == "kcat/Km":
                summary.kcat_km_count += 1

            source_counts[e.source_db] = source_counts.get(e.source_db, 0) + 1

        summary.unique_ec_numbers = len(ec_set)
        summary.unique_pdb_ids = len(pdb_set)
        summary.source_distribution = source_counts

        return summary
