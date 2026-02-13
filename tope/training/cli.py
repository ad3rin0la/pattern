"""Command-line interface for ToPE model training."""

import argparse
from pathlib import Path


def main():
    """Train a ToPE model."""
    parser = argparse.ArgumentParser(description="ToPE: Train enzyme function prediction model")
    parser.add_argument("config", type=Path, help="Path to training config YAML")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to curated dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")

    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["n_epochs"] = args.epochs

    from tope.training import ToPETrainer

    trainer = ToPETrainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
