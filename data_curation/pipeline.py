"""
Main pipeline orchestrator for ToPE data curation.

Ties together:
    1. M-CSA → catalytic site annotations
    2. RCSB PDB → structure download
    3. Active-site extraction (BioPython)
    4. Physicochemical feature computation (Ioffe descriptors)
    5. Dataset assembly and storage

Usage
-----
    from data_curation import CurationPipeline
    from data_curation.config import PipelineConfig

    config = PipelineConfig(min_structures=5000, max_resolution=2.5)
    pipeline = CurationPipeline(config)
    records = pipeline.run()

Or from the command line:

    python -m data_curation.pipeline --min-structures 5000 --max-resolution 2.5
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_curation.active_site import ActiveSite, ActiveSiteExtractor
from data_curation.config import (
    DATA_ROOT,
    FEATURES_DIR,
    PDB_DIR,
    PROCESSED_DIR,
    PipelineConfig,
)
from data_curation.dataset import DatasetBuilder, DatasetRecord
from data_curation.features import ActiveSiteFeatures, FeatureComputer
from data_curation.mcsa_client import MCSAClient, MCSAEntry
from data_curation.pdb_client import PDBClient

logger = logging.getLogger(__name__)


class CurationPipeline:
    """End-to-end data curation for the ToPE project.

    Implements Phase 1 of the ToPE roadmap:
        - Curate enzyme active-site dataset from M-CSA + PDB (≥5 000 structures)
        - Compute Ioffe-style physicochemical descriptors
        - Produce filtration-ready adjacency matrices
        - Export as Parquet / NumPy arrays for downstream topological encoding
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialise sub-components
        self.mcsa = MCSAClient(
            cache_dir=self.config.data_root / "raw" / "mcsa",
            request_delay=self.config.request_delay,
        )
        self.pdb = PDBClient(
            cache_dir=self.config.data_root / "raw" / "pdb",
            request_delay=self.config.request_delay,
        )
        self.extractor = ActiveSiteExtractor(config=self.config)
        self.featuriser = FeatureComputer(
            compute_sasa=self.config.compute_sasa,
            normalise=True,
        )
        self.dataset_builder = DatasetBuilder(
            output_dir=self.config.data_root / "processed",
            features_dir=self.config.data_root / "features",
            config=self.config,
        )

    def run(
        self,
        ec_prefix: Optional[str] = None,
        max_entries: Optional[int] = None,
        skip_download: bool = False,
    ) -> List[DatasetRecord]:
        """Execute the full curation pipeline.

        Parameters
        ----------
        ec_prefix : str, optional
            Restrict to a specific EC class (e.g. "3" for hydrolases).
        max_entries : int, optional
            Cap entries for development / debugging runs.
        skip_download : bool
            If True, skip PDB download (assumes files already cached).

        Returns
        -------
        list of DatasetRecord
            The assembled dataset index.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("ToPE Data Curation Pipeline — Starting")
        logger.info("=" * 60)
        logger.info("Config: min_structures=%d, resolution≤%.1f Å, radius=%.1f Å",
                     self.config.min_structures, self.config.max_resolution,
                     self.config.active_site_radius)

        # ── Step 1: Fetch M-CSA annotations ──────────────────────────────
        logger.info("─" * 40)
        logger.info("Step 1/5: Fetching M-CSA catalytic site annotations")
        mcsa_entries = self._step_fetch_mcsa(ec_prefix, max_entries)

        # ── Step 2: Resolve PDB IDs ──────────────────────────────────────
        logger.info("─" * 40)
        logger.info("Step 2/5: Resolving PDB structures")
        pdb_ids = self._step_resolve_pdb_ids(mcsa_entries)

        # ── Step 3: Download structures ──────────────────────────────────
        logger.info("─" * 40)
        logger.info("Step 3/5: Downloading PDB structures")
        structure_paths = self._step_download_structures(pdb_ids, skip_download)

        # ── Step 4: Extract active sites + compute features ──────────────
        logger.info("─" * 40)
        logger.info("Step 4/5: Extracting active sites and computing features")
        active_sites, features_list = self._step_extract_and_featurise(
            mcsa_entries, structure_paths,
        )

        # ── Step 5: Assemble dataset ─────────────────────────────────────
        logger.info("─" * 40)
        logger.info("Step 5/5: Assembling dataset")
        records = self._step_assemble_dataset(active_sites, features_list)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("Pipeline complete: %d samples in %.1f s", len(records), elapsed)
        logger.info("=" * 60)

        # Save summary
        summary = DatasetBuilder.summarise(records)
        summary_path = self.config.data_root / "processed" / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Summary → %s", summary_path)
        self._log_summary(summary)

        return records

    # ── Pipeline steps ───────────────────────────────────────────────────

    def _step_fetch_mcsa(
        self,
        ec_prefix: Optional[str],
        max_entries: Optional[int],
    ) -> List[MCSAEntry]:
        """Step 1: Load M-CSA entries and fetch residue annotations."""
        entries = self.mcsa.load_entries_from_csv()

        if ec_prefix:
            entries = self.mcsa.filter_by_ec(entries, ec_prefix)
            logger.info("Filtered to EC prefix '%s': %d entries", ec_prefix, len(entries))

        if max_entries is not None:
            entries = entries[:max_entries]
            logger.info("Capped at %d entries (development mode)", max_entries)

        entries = self.mcsa.fetch_all_with_residues(entries)
        summary = self.mcsa.summarise(entries)
        logger.info("M-CSA: %d entries, %d unique PDB IDs, %d catalytic residues",
                     summary["total_entries"], summary["unique_pdb_ids"],
                     summary["total_catalytic_residues"])
        return entries

    def _step_resolve_pdb_ids(
        self, mcsa_entries: List[MCSAEntry]
    ) -> List[str]:
        """Step 2: Get deduplicated PDB IDs, optionally supplemented by RCSB search."""
        pdb_ids = self.mcsa.unique_pdb_ids(mcsa_entries)
        logger.info("PDB IDs from M-CSA: %d", len(pdb_ids))

        # If we don't have enough, supplement with an RCSB search
        if len(pdb_ids) < self.config.min_structures:
            shortfall = self.config.min_structures - len(pdb_ids)
            logger.info(
                "Need %d more structures to reach target of %d — querying RCSB",
                shortfall, self.config.min_structures,
            )
            extra_ids = self.pdb.search_enzymes(config=self.config)
            # Merge, keeping M-CSA entries first (they have annotations)
            existing = set(pdb_ids)
            for pid in extra_ids:
                if pid not in existing:
                    pdb_ids.append(pid)
                    existing.add(pid)
                    if len(pdb_ids) >= self.config.min_structures:
                        break

        logger.info("Total PDB IDs to process: %d", len(pdb_ids))
        return pdb_ids

    def _step_download_structures(
        self,
        pdb_ids: List[str],
        skip: bool = False,
    ) -> Dict[str, Optional[Path]]:
        """Step 3: Download mmCIF files from RCSB."""
        if skip:
            # Build path map from cache
            result = {}
            for pid in pdb_ids:
                path = self.pdb.cache_dir / f"{pid}.cif"
                result[pid] = path if path.exists() else None
            cached = sum(1 for p in result.values() if p is not None)
            logger.info("Skip download: found %d / %d cached structures", cached, len(pdb_ids))
            return result

        return self.pdb.download_batch(pdb_ids, fmt="cif")

    def _step_extract_and_featurise(
        self,
        mcsa_entries: List[MCSAEntry],
        structure_paths: Dict[str, Optional[Path]],
    ) -> tuple[List[ActiveSite], List[ActiveSiteFeatures]]:
        """Step 4: Extract active sites and compute Ioffe features."""
        # Build lookup: PDB ID → MCSAEntry
        entry_map: Dict[str, MCSAEntry] = {}
        for e in mcsa_entries:
            entry_map.setdefault(e.pdb_id, e)

        active_sites: List[ActiveSite] = []
        features_list: List[ActiveSiteFeatures] = []
        processed = 0
        total = len(structure_paths)

        for pdb_id, path in structure_paths.items():
            if path is None or not path.exists():
                continue

            entry = entry_map.get(pdb_id)
            cat_residues = entry.catalytic_residues if entry else []
            ec = entry.ec_number if entry else ""

            site = self.extractor.extract(
                structure_path=path,
                catalytic_residues=cat_residues,
                pdb_id=pdb_id,
                ec_number=ec,
            )
            if site is None or site.n_atoms == 0:
                continue

            feats = self.featuriser.compute(site)
            active_sites.append(site)
            features_list.append(feats)
            processed += 1

            if processed % 200 == 0:
                logger.info("Extracted + featurised: %d / %d", processed, total)

        logger.info(
            "Extraction complete: %d active sites from %d structures",
            len(active_sites), total,
        )
        return active_sites, features_list

    def _step_assemble_dataset(
        self,
        active_sites: List[ActiveSite],
        features_list: List[ActiveSiteFeatures],
    ) -> List[DatasetRecord]:
        """Step 5: Build the final dataset with train/val/test splits."""
        return self.dataset_builder.build(active_sites, features_list)

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(summary: Dict[str, Any]) -> None:
        logger.info("  Total samples:  %d", summary.get("total_samples", 0))
        logger.info("  Splits:         %s", summary.get("split_distribution", {}))
        logger.info("  EC distribution:")
        for ec, count in summary.get("ec_distribution", {}).items():
            logger.info("    %s: %d", ec, count)
        stats = summary.get("atom_count_stats", {})
        if stats:
            logger.info("  Atoms/site:     mean=%.0f, median=%.0f, range=[%d, %d]",
                         stats.get("mean", 0), stats.get("median", 0),
                         stats.get("min", 0), stats.get("max", 0))
        logger.info("  Metalloenzymes: %.1f%%",
                     summary.get("metalloenzyme_fraction", 0) * 100)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ToPE Data Curation Pipeline — Phase 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-structures", type=int, default=5000,
        help="Target number of enzyme structures",
    )
    parser.add_argument(
        "--max-resolution", type=float, default=3.0,
        help="Maximum X-ray resolution in Å",
    )
    parser.add_argument(
        "--active-site-radius", type=float, default=8.0,
        help="Extraction radius around catalytic residues in Å",
    )
    parser.add_argument(
        "--ec-prefix", type=str, default=None,
        help="Restrict to EC class prefix (e.g. '3.4')",
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Cap entries for development runs",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip PDB download (use cached files)",
    )
    parser.add_argument(
        "--output-format", choices=["parquet", "csv", "json"],
        default="parquet", help="Dataset index output format",
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root directory for all data",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        min_structures=args.min_structures,
        max_resolution=args.max_resolution,
        active_site_radius=args.active_site_radius,
        output_format=args.output_format,
        data_root=Path(args.data_root),
        n_workers=args.workers,
    )

    pipeline = CurationPipeline(config)
    records = pipeline.run(
        ec_prefix=args.ec_prefix,
        max_entries=args.max_entries,
        skip_download=args.skip_download,
    )

    print(f"\nDone — {len(records)} samples written to {config.data_root / 'processed'}")


if __name__ == "__main__":
    main()
