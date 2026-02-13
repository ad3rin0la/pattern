"""Command-line interface for ToPE data curation."""

import argparse
from pathlib import Path


def curate():
    """Run the data curation pipeline."""
    parser = argparse.ArgumentParser(
        description="ToPE: Curate enzyme dataset from M-CSA + PDB + BRENDA"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory for curated dataset",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of M-CSA entries to process",
    )

    args = parser.parse_args()

    from tope.data import CurationPipeline, CurationConfig

    if args.config:
        import yaml

        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = CurationConfig(**config_dict)
    else:
        config = CurationConfig(output_dir=str(args.output_dir))

    pipeline = CurationPipeline(config)
    dataset = pipeline.run()
    print(f"Curated {len(dataset)} enzyme entries to {args.output_dir}")


def ingest_topec():
    """Ingest TopEC pre-curated dataset."""
    parser = argparse.ArgumentParser(
        description="ToPE: Ingest TopEC dataset"
    )
    parser.add_argument(
        "topec_dir",
        type=Path,
        help="Path to TopEC CSV directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Path to PDB structure files",
    )

    args = parser.parse_args()

    from tope.data import IngestionConfig, run_ingestion

    config = IngestionConfig(
        topec_csv_dir=str(args.topec_dir),
        output_dir=str(args.output_dir),
    )
    if args.pdb_dir:
        config.pdb_dir = str(args.pdb_dir)

    records = run_ingestion(config)
    print(f"Ingested {len(records)} records from TopEC")


if __name__ == "__main__":
    curate()
