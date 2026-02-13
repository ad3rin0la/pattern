"""
ToPE Data Pipeline
==================
Curate enzyme active-site datasets from M-CSA + PDB + BRENDA/SABIO-RK
for topological pattern recognition in enzyme catalysis.

Supports multiple data sources:
    - M-CSA: Mechanism and Catalytic Site Atlas
    - PDB: Protein Data Bank structures
    - BRENDA/SABIO-RK: Kinetic parameters
    - TopEC: Pre-curated enzyme classification dataset

Usage:
    from tope.data import CurationPipeline, CurationConfig

    config = CurationConfig(output_dir="./data")
    pipeline = CurationPipeline(config)
    dataset = pipeline.run()
"""

from tope.data.pipeline import CurationPipeline
from tope.data.config import CurationConfig, PipelineConfig
from tope.data.kinetics_client import KineticsAggregator
from tope.data.active_site import (
    ResidueRecord,
    ActiveSite,
    ActiveSiteExtractor,
)
from tope.data.mcsa_client import MCSAClient, MCSAEntry, CatalyticResidue
from tope.data.pdb_client import PDBClient
from tope.data.features import FeatureComputer, ActiveSiteFeatures
from tope.data.dataset import DatasetBuilder, DatasetRecord
from tope.data.topec_ingestion import (
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
    "PipelineConfig",
    "KineticsAggregator",
    # Active site extraction
    "ResidueRecord",
    "ActiveSite",
    "ActiveSiteExtractor",
    # Data sources
    "MCSAClient",
    "MCSAEntry",
    "CatalyticResidue",
    "PDBClient",
    # Features
    "FeatureComputer",
    "ActiveSiteFeatures",
    # Dataset
    "DatasetBuilder",
    "DatasetRecord",
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
