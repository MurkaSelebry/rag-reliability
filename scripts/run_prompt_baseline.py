#!/usr/bin/env python
"""Run a prompt-based judge baseline (dummy or MLX backend) over a dataset.

Example:
    python scripts/run_prompt_baseline.py \
        --data data/dummy.jsonl \
        --output results/dummy_predictions.jsonl \
        --mode direct --backend dummy --dummy-strategy always_reliable
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.dummy_model import STRATEGIES, DummyPredictor
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.parsing import parse_prediction
from rag_reliability.prompts import build_direct_prompt, build_marker_prompt
from rag_reliability.schema import RagSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Input dataset (jsonl)")
    parser.add_argument(
        "--output",
        default="results/prompt_baseline_predictions.jsonl",
        help="Output predictions (jsonl)",
    )
    parser.add_argument("--mode", choices=["direct", "marker"], default="direct")
    parser.add_argument("--backend", choices=["dummy", "mlx"], default="dummy")
    parser.add_argument("--dummy-strategy", choices=list(STRATEGIES), default="always_reliable")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--max-tokens", type=int, default=64, help="MLX generation budget")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples: list[RagSample] = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples from {args.data}")

    build_prompt = build_direct_prompt if args.mode == "direct" else build_marker_prompt

    if args.backend == "dummy":
        predictor = DummyPredictor(strategy=args.dummy_strategy, mode=args.mode)
        generate_fn = None
    else:
        predictor = None
        generate_fn = make_generate_fn(args.model, args.max_tokens)

    predictions = []
    for sample in tqdm(samples, desc=f"{args.backend}/{args.mode}"):
        if predictor is not None:
            raw_output = predictor.predict(sample)
        else:
            raw_output = generate_fn(build_prompt(sample))
        predictions.append(
            parse_prediction(raw_output, sample.id, expect_marker=(args.mode == "marker"))
        )

    save_jsonl(predictions, args.output)
    invalid = sum(p.invalid_output for p in predictions)
    print(f"Wrote {len(predictions)} predictions to {Path(args.output)} (invalid outputs: {invalid})")


if __name__ == "__main__":
    main()
