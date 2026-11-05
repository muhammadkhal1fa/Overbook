"""Train the cancellation classifier on H2.csv.

Usage (from the repository root):
    python classification/train.py --data H2.csv
    python classification/train.py --data H2.csv --sample 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.pipeline import train_classification  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the H2 IsCanceled classifier.")
    parser.add_argument(
        "--data",
        default="H2.csv",
        help="Path to H2.csv (31 columns). Default: H2.csv in the current directory.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional row sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "artifacts"),
        help="Directory for the joblib artifact and manifest.json.",
    )
    parser.add_argument("--no-save", action="store_true", help="Evaluate without saving.")
    args = parser.parse_args()

    result = train_classification(
        data_path=args.data,
        sample=args.sample,
        seed=args.seed,
        save=not args.no_save,
        out_dir=args.out_dir,
    )
    summary = {
        "task": result["task"],
        "target": result["target"],
        "model": result["model"],
        "rows": result["rows"],
        "n_features": result["n_features"],
        "threshold": result["threshold"],
        "split_sizes": result["split_sizes"],
        "validation_f1_tuned": result["validation"]["at_tuned_threshold"]["f1"],
        "test": result["test"],
        "artifact": result.get("artifact_path"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
