#!/usr/bin/env python
"""Evaluate a predictions jsonl against a labeled dataset.

Example:
    python scripts/evaluate.py \
        --data data/dummy.jsonl \
        --predictions results/dummy_predictions.jsonl \
        --output results/dummy_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_reliability.dataset import load_jsonl
from rag_reliability.metrics import evaluate_predictions
from rag_reliability.schema import Prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Labeled dataset (jsonl)")
    parser.add_argument("--predictions", required=True, help="Predictions file (jsonl)")
    parser.add_argument("--output", default="results/metrics.json", help="Where to save metrics")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    return parser.parse_args()


def apply_limit(items: list, limit: int | None) -> list:
    return items if limit is None else items[:limit]


def load_predictions(path: str | Path) -> list[Prediction]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path.resolve()}")
    predictions: list[Prediction] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(Prediction.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid prediction at {path}:{line_no}: {exc}") from exc
    return predictions


def main() -> None:
    args = parse_args()
    samples = apply_limit(load_jsonl(args.data), args.limit)
    predictions = load_predictions(args.predictions)

    result = evaluate_predictions(samples, predictions)
    payload = result.model_dump(exclude_none=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
