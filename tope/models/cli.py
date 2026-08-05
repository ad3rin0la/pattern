"""Command-line interface for ToPE model inference."""

import argparse
from pathlib import Path


def predict():
    """Run ToPE model prediction on an enzyme structure."""
    parser = argparse.ArgumentParser(description="ToPE: Predict enzyme function from structure")
    parser.add_argument("pdb_file", type=Path, help="Path to PDB structure file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--task",
        type=str,
        choices=["ec", "kinetics", "selectivity", "all"],
        default="all",
        help="Prediction task",
    )

    args = parser.parse_args()

    import json
    import torch

    from tope.models import CompleteToPEModel, CompleteToPEConfig

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", CompleteToPEConfig())
    model = CompleteToPEModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Run prediction
    with torch.no_grad():
        predictions = model.predict(args.pdb_file, task=args.task)

    results = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in predictions.items()}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    predict()
