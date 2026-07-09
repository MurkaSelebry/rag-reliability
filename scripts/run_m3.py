#!/usr/bin/env python
"""Run Method 3 prompt judge through the shared predictions contract."""

from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.dummy_model import STRATEGIES, DummyPredictor
from rag_reliability.methods.m3 import build_system_prompt, build_user_prompt, parse_m3_prediction
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.parsing import parse_prediction
from rag_reliability.schema import Prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--output", default="results/m3_zero_shot_predictions.jsonl")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "gepa"], default="zero_shot")
    parser.add_argument("--examples", default=None, help="YAML examples for --mode few_shot")
    parser.add_argument("--prompt-file", default=None, help="Prompt text for --mode gepa")
    parser.add_argument("--backend", choices=["dummy", "mlx", "openai"], default="mlx")
    parser.add_argument("--dummy-strategy", choices=list(STRATEGIES), default="always_reliable")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.backend == "dummy":
        predictor = DummyPredictor(strategy=args.dummy_strategy, mode="direct")
        generate_fn = None
        chat_client = None
    elif args.backend == "openai":
        from rag_reliability.methods.m3.openai_client import CachedChatClient  # noqa: PLC0415

        predictor = None
        generate_fn = None
        chat_client = CachedChatClient(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            cache_dir=args.cache_dir,
        )
    else:
        predictor = None
        generate_fn = make_generate_fn(args.model, args.max_tokens)
        chat_client = None

    system_prompt = build_system_prompt(
        args.mode,
        examples_path=args.examples,
        prompt_file=args.prompt_file,
    )

    predictions: list[Prediction] = []
    for sample in tqdm(samples, desc=f"m3/{args.backend}"):
        if predictor is not None:
            raw_output = predictor.predict(sample)
            prediction = parse_prediction(raw_output, sample.id)
        else:
            user_prompt = build_user_prompt(sample, args.max_context_chars)
            if chat_client is not None:
                raw_output = chat_client.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=args.max_tokens,
                )
            else:
                prompt = f"{system_prompt}\n\n{user_prompt}"
                raw_output = generate_fn(prompt)
            prediction = parse_m3_prediction(raw_output, sample.id)
        predictions.append(prediction)

    save_jsonl(predictions, args.output)
    invalid = sum(prediction.invalid_output for prediction in predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output} (invalid outputs: {invalid})")


if __name__ == "__main__":
    main()
