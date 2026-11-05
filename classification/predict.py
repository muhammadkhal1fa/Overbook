"""Score bookings with the saved cancellation classifier. Never retrains.

Usage (from the repository root):
    python classification/predict.py --input bookings.csv
    python classification/predict.py --json '{"LeadTime": 10, ...}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from classification.pipeline import predict_cancel  # noqa: E402

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "logistic_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict H2 booking cancellations.")
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_ARTIFACT),
        help="Path to the saved classification artifact.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="CSV of bookings (raw H2 columns or engineered features).")
    group.add_argument("--json", help="One booking as a JSON object.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved threshold.")
    args = parser.parse_args()

    if args.json:
        rows = json.loads(args.json)
    else:
        rows = pd.read_csv(args.input)

    predictions = predict_cancel(args.artifact, rows, threshold=args.threshold)
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
