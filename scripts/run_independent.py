#!/usr/bin/env python
"""Run the independent rule-based evaluator."""

from __future__ import annotations

import argparse

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.methods.independent.predict import predict_many


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        default="data/dummy.jsonl",
        help="Input RagSample JSONL dataset.",
    )

    parser.add_argument(
        "--output",
        default="results/independent_predictions.jsonl",
        help="Output predictions JSONL file.",
    )

    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data}")

    samples = load_jsonl(args.data)

    print(f"Loaded {len(samples)} samples")

    if args.limit is not None:
        samples = samples[: args.limit]

    predictions = predict_many(
        samples,
        faithfulness_threshold=args.faithfulness_threshold,
        relevance_threshold=args.relevance_threshold,
    )

    save_jsonl(predictions, args.output)

    reliable_count = sum(
        prediction.reliable_pred
        for prediction in predictions
    )

    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(f"Predicted reliable answers: {reliable_count}/{len(predictions)}")


if __name__ == "__main__":
    main()