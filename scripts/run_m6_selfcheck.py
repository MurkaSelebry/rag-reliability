#!/usr/bin/env python
"""Run Method 6 from precomputed SelfCheck-style features."""

from __future__ import annotations

import argparse

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.methods.m6.predict import load_features, predictions_from_feature_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--features", required=True, help="JSONL with Method 6 features keyed by id")
    parser.add_argument("--output", default="results/m6_selfcheck_predictions.jsonl")
    parser.add_argument("--contradiction-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=1.0)
    parser.add_argument("--relevance-threshold", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    features = load_features(args.features)
    predictions = predictions_from_feature_rows(
        samples,
        features,
        contradiction_threshold=args.contradiction_threshold,
        entropy_threshold=args.entropy_threshold,
        relevance_threshold=args.relevance_threshold,
    )
    save_jsonl(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
