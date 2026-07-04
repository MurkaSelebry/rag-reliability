#!/usr/bin/env python
"""Run inference with an MLX model (optionally with a trained LoRA adapter).

Same output format as run_prompt_baseline.py, so results are evaluated with
scripts/evaluate.py.

Example:
    python scripts/infer.py \
        --data data/dummy.jsonl \
        --output results/direct_lora_predictions.jsonl \
        --mode direct \
        --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
        --adapter-path results/adapters_direct
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.parsing import parse_prediction
from rag_reliability.prompts import build_direct_prompt, build_marker_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Input dataset (jsonl)")
    parser.add_argument("--output", required=True, help="Output predictions (jsonl)")
    parser.add_argument("--mode", choices=["direct", "marker"], default="direct")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default=None, help="Path to trained LoRA adapters")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    build_prompt = build_direct_prompt if args.mode == "direct" else build_marker_prompt

    generate_fn = make_generate_fn(args.model, args.max_tokens, adapter_path=args.adapter_path)

    predictions = []
    for sample in tqdm(samples, desc=f"infer/{args.mode}"):
        raw_output = generate_fn(build_prompt(sample))
        predictions.append(
            parse_prediction(raw_output, sample.id, expect_marker=(args.mode == "marker"))
        )

    save_jsonl(predictions, args.output)
    invalid = sum(p.invalid_output for p in predictions)
    print(f"Wrote {len(predictions)} predictions to {Path(args.output)} (invalid outputs: {invalid})")


if __name__ == "__main__":
    main()
