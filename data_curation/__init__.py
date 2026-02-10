"""
ToPE Data Curation Pipeline
============================
Curate enzyme active-site datasets from M-CSA + PDB + BRENDA/SABIO-RK
for topological pattern recognition in enzyme catalysis.

Supports multiple data sources:
    - M-CSA: Mechanism and Catalytic Site Atlas
    - PDB: Protein Data Bank structures
    - BRENDA/SABIO-RK: Kinetic parameters
    - TopEC: Pre-curated enzyme classification dataset

Usage:
    from data_curation import CurationPipeline, CurationConfig

    config = CurationConfig(output_dir="./data")
    pipeline = CurationPipeline(config)
    dataset = pipeline.run()

    # Or ingest from TopEC dataset:
    from data_curation import run_ingestion, IngestionConfig

    cfg = IngestionConfig(topec_csv_dir="TopEC/data/csv")
    records = run_ingestion(cfg)
"""

from data_curation.pipeline import CurationPipeline
from data_curation.config import CurationConfig
from data_curation.kinetics_client import KineticsAggregator
from data_curation.topec_ingestion import (
    IngestionConfig,
    ToPERecord,
    KineticsRecord,
    ZoneStats,
    run_ingestion,
    discover_topec_csvs,
    parse_topec_csv,
    build_ec_class_mapping,
    enrich_with_kinetics,
    locate_pdb_files,
    compute_zone_stats_from_pdb,
    assign_splits,
    assemble_index,
    export_index,
)

__all__ = [
    # Core pipeline
    "CurationPipeline",
    "CurationConfig",
    "KineticsAggregator",
    # TopEC ingestion
    "IngestionConfig",
    "ToPERecord",
    "KineticsRecord",
    "ZoneStats",
    "run_ingestion",
    "discover_topec_csvs",
    "parse_topec_csv",
    "build_ec_class_mapping",
    "enrich_with_kinetics",
    "locate_pdb_files",
    "compute_zone_stats_from_pdb",
    "assign_splits",
    "assemble_index",
    "export_index",
]
