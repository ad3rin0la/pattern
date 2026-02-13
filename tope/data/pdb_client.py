"""
RCSB PDB client for structure search and download.

Uses the RCSB Search API v2 to query for enzyme structures matching
resolution, experimental method, and chain-length criteria, then
downloads coordinate files (mmCIF) for local processing.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from tope.data.config import (
    PDB_DIR,
    RCSB_DATA_URL,
    RCSB_DOWNLOAD_URL,
    RCSB_SEARCH_URL,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


class PDBClient:
    """Search RCSB PDB and download structure files."""

    def __init__(
        self,
        cache_dir: Path = PDB_DIR,
        request_delay: float = 0.25,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "ToPE-DataCuration/1.0",
        })

    # ── RCSB Search API ──────────────────────────────────────────────────

    def search_enzymes(
        self,
        config: Optional[PipelineConfig] = None,
        ec_prefix: Optional[str] = None,
        pdb_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Query RCSB for enzyme PDB IDs matching quality filters.

        Parameters
        ----------
        config : PipelineConfig, optional
            Resolution and method filters. Uses defaults if not given.
        ec_prefix : str, optional
            Restrict to a specific EC class (e.g. "3.4" for hydrolase peptidases).
        pdb_ids : list, optional
            If provided, skip the search and validate these IDs instead.

        Returns
        -------
        list of str
            Deduplicated, uppercased PDB IDs.
        """
        if pdb_ids is not None:
            return [pid.upper().strip() for pid in pdb_ids if len(pid.strip()) == 4]

        if config is None:
            config = PipelineConfig()

        query = self._build_search_query(config, ec_prefix)
        logger.info("Querying RCSB Search API for enzyme structures …")

        results: List[str] = []
        batch_start = 0
        batch_size = 500

        while True:
            query["request_options"] = {
                "paginate": {"start": batch_start, "rows": batch_size},
                "results_content_type": ["experimental"],
                "sort": [{"sort_by": "score", "direction": "desc"}],
            }
            query["return_type"] = "entry"

            time.sleep(self.request_delay)
            resp = self._session.post(RCSB_SEARCH_URL, json=query, timeout=60)

            if resp.status_code == 204:
                # No results
                break

            resp.raise_for_status()
            data = resp.json()
            hits = data.get("result_set", [])

            if not hits:
                break

            for hit in hits:
                pdb_id = hit.get("identifier", "").upper().strip()
                if pdb_id and len(pdb_id) == 4:
                    results.append(pdb_id)

            total = data.get("total_count", 0)
            batch_start += batch_size
            logger.info("  … fetched %d / %d", min(batch_start, total), total)

            if batch_start >= total:
                break

        # Deduplicate preserving order
        seen: Set[str] = set()
        deduped: List[str] = []
        for pid in results:
            if pid not in seen:
                seen.add(pid)
                deduped.append(pid)

        logger.info("Search returned %d unique PDB IDs", len(deduped))
        return deduped

    # ── Structure download ───────────────────────────────────────────────

    def download_structure(
        self,
        pdb_id: str,
        fmt: str = "cif",
        force: bool = False,
    ) -> Optional[Path]:
        """Download a structure file from RCSB.

        Parameters
        ----------
        pdb_id : str
            Four-character PDB identifier.
        fmt : str
            File format: "cif" (mmCIF) or "pdb".
        force : bool
            Re-download even if cached.

        Returns
        -------
        Path or None
            Path to the downloaded file, or None on failure.
        """
        pdb_id = pdb_id.upper().strip()
        ext = ".cif" if fmt == "cif" else ".pdb"
        local_path = self.cache_dir / f"{pdb_id}{ext}"

        if local_path.exists() and not force:
            return local_path

        suffix = ".cif" if fmt == "cif" else ".pdb"
        url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}{suffix}.gz"

        try:
            time.sleep(self.request_delay)
            resp = self._session.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            gz_path = self.cache_dir / f"{pdb_id}{suffix}.gz"
            with open(gz_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)

            # Decompress
            with gzip.open(gz_path, "rb") as f_in, open(local_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()

            logger.debug("Downloaded %s → %s", pdb_id, local_path)
            return local_path

        except requests.RequestException as exc:
            logger.warning("Failed to download %s: %s", pdb_id, exc)
            return None

    def download_batch(
        self,
        pdb_ids: List[str],
        fmt: str = "cif",
        force: bool = False,
    ) -> Dict[str, Optional[Path]]:
        """Download multiple structures, returning a map of PDB ID → path."""
        results: Dict[str, Optional[Path]] = {}
        total = len(pdb_ids)

        for i, pdb_id in enumerate(pdb_ids, 1):
            results[pdb_id] = self.download_structure(pdb_id, fmt=fmt, force=force)
            if i % 100 == 0 or i == total:
                downloaded = sum(1 for p in results.values() if p is not None)
                logger.info("Downloaded %d / %d structures (%d OK)", i, total, downloaded)

        downloaded = sum(1 for p in results.values() if p is not None)
        logger.info("Batch complete: %d / %d structures downloaded", downloaded, total)
        return results

    # ── Entry metadata ───────────────────────────────────────────────────

    def fetch_entry_metadata(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """Fetch core entry metadata (resolution, method, title, EC)."""
        pdb_id = pdb_id.upper().strip()
        cache_path = self.cache_dir / "metadata" / f"{pdb_id}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            return json.loads(cache_path.read_text())

        url = f"{RCSB_DATA_URL}/{pdb_id}"
        try:
            time.sleep(self.request_delay)
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cache_path.write_text(json.dumps(data, indent=2))
            return data
        except requests.RequestException as exc:
            logger.warning("Failed to fetch metadata for %s: %s", pdb_id, exc)
            return None

    def get_resolution(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Extract resolution in Å from entry metadata."""
        try:
            refine = metadata.get("rcsb_entry_info", {})
            return refine.get("resolution_combined", [None])[0]
        except (IndexError, TypeError):
            return None

    def get_ec_numbers(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract EC numbers from entry metadata."""
        ec_numbers: List[str] = []
        try:
            polymers = metadata.get("rcsb_entry_container_identifiers", {})
            for poly_id in polymers.get("polymer_entity_ids", []):
                # Enzyme classification is at the polymer entity level
                pass
        except (KeyError, TypeError):
            pass

        # Fallback: struct_keywords
        try:
            keywords = metadata.get("struct_keywords", {})
            text = keywords.get("pdbx_keywords", "") + " " + keywords.get("text", "")
            # EC numbers follow the pattern N.N.N.N
            import re
            ec_numbers.extend(re.findall(r"\b(\d+\.\d+\.\d+\.\d+)\b", text))
        except (KeyError, TypeError):
            pass

        return list(set(ec_numbers))

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_search_query(
        config: PipelineConfig,
        ec_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an RCSB Search API v2 query JSON.

        Filters:
        - Has enzyme classification (EC number assigned)
        - Resolution ≤ config.max_resolution
        - Experimental method in config.experimental_methods
        - Polymer entity length ≥ config.min_chain_length
        """
        nodes = []

        # Must have an EC number
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                "operator": "exists",
                "negation": False,
            },
        })

        # Resolution filter
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": config.max_resolution,
                "negation": False,
            },
        })

        # Experimental method
        if config.experimental_methods:
            nodes.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "in",
                    "value": config.experimental_methods,
                    "negation": False,
                },
            })

        # Chain length
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": "Protein",
                "negation": False,
            },
        })

        # EC prefix filter
        if ec_prefix:
            nodes.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                    "operator": "starts_with",
                    "value": ec_prefix,
                    "negation": False,
                },
            })

        query: Dict[str, Any] = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            },
        }
        return query
