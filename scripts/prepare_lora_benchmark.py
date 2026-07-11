#!/usr/bin/env python
"""Prepare held-out LoRA benchmark files for direct or marker judge modes."""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
from typing import NamedTuple

import yaml

from rag_reliability.dataset import load_jsonl, save_jsonl, split_samples, write_training_jsonl
from rag_reliability.formatting import build_chat_training_record
from rag_reliability.schema import RagSample


class BenchmarkPreparation(NamedTuple):
    mode: str
    output_dir: Path
    mlx_data_dir: Path
    raw_test_path: Path
    train_total: int
    validation_total: int
    test_total: int
    iters: int
    test_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/organizers.jsonl", help="Input RagSample jsonl")
    parser.add_argument("--mode", choices=["direct", "marker"], required=True)
    parser.add_argument("--config", default=None, help="Training config; defaults to configs/<mode>_lora.yaml")
    parser.add_argument("--output-dir", default="results/organizer_lora", help="Where to write split files")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-question-chars", type=int, default=2000)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance-train-labels",
        action="store_true",
        help="Balance the four (faithfulness, relevance) training-label pairs",
    )
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=None,
        help="Examples per training-label pair; required with --balance-train-labels",
    )
    return parser.parse_args()


def load_config(mode: str, config_path: str | None) -> dict:
    path = Path(config_path) if config_path is not None else Path(f"configs/{mode}_lora.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prepare_benchmark(
    samples: list[RagSample],
    mode: str,
    output_dir: str | Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    epochs: int,
    batch_size: int,
    max_question_chars: int | None = 2000,
    max_context_chars: int | None = 5000,
    balance_train_labels: bool = False,
    target_per_label: int | None = None,
) -> BenchmarkPreparation:
    if epochs < 1 or batch_size < 1:
        raise ValueError(f"epochs and batch_size must be >= 1, got {epochs}, {batch_size}")
    if balance_train_labels and target_per_label is None:
        raise ValueError("target_per_label is required when balance_train_labels is enabled")

    train, val, test = split_samples(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    output_path = Path(output_dir)
    mlx_data_dir = output_path / f"lora_{mode}"
    raw_test_path = output_path / f"{mode}_test_samples.jsonl"

    if balance_train_labels:
        train = balance_training_labels(train, seed, target_per_label)

    sft_train = truncate_samples_for_sft(train, max_question_chars, max_context_chars)
    sft_val = truncate_samples_for_sft(val, max_question_chars, max_context_chars)
    sft_test = truncate_samples_for_sft(test, max_question_chars, max_context_chars)

    write_training_jsonl(sft_train, output_path / f"train_{mode}.jsonl", mode)
    save_jsonl((build_chat_training_record(sample, mode) for sample in sft_train), mlx_data_dir / "train.jsonl")
    save_jsonl((build_chat_training_record(sample, mode) for sample in sft_val), mlx_data_dir / "valid.jsonl")
    save_jsonl((build_chat_training_record(sample, mode) for sample in sft_test), mlx_data_dir / "test.jsonl")
    save_jsonl(test, raw_test_path)

    iters = max(1, len(train) * epochs // batch_size)
    return BenchmarkPreparation(
        mode=mode,
        output_dir=output_path,
        mlx_data_dir=mlx_data_dir,
        raw_test_path=raw_test_path,
        train_total=len(train),
        validation_total=len(val),
        test_total=len(test),
        iters=iters,
        test_ids=[sample.id for sample in test],
    )


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars]


def balance_training_labels(
    samples: list[RagSample],
    seed: int,
    target_per_label: int,
) -> list[RagSample]:
    """Return a deterministic equal-size sample for each direct judgement pair."""
    if target_per_label < 1:
        raise ValueError(f"target_per_label must be >= 1, got {target_per_label}")

    labels = ((0, 0), (0, 1), (1, 0), (1, 1))
    grouped = {label: [] for label in labels}
    for sample in samples:
        grouped[(sample.faithfulness, sample.relevance)].append(sample)

    empty_labels = [label for label, group in grouped.items() if not group]
    if empty_labels:
        raise ValueError(f"cannot balance absent training labels: {empty_labels}")

    rng = Random(seed)
    balanced: list[RagSample] = []
    for label in labels:
        group = grouped[label]
        rng.shuffle(group)
        if len(group) >= target_per_label:
            balanced.extend(group[:target_per_label])
        else:
            balanced.extend(rng.choice(group) for _ in range(target_per_label))
    rng.shuffle(balanced)
    return balanced


def truncate_samples_for_sft(
    samples: list[RagSample],
    max_question_chars: int | None,
    max_context_chars: int | None,
) -> list[RagSample]:
    return [
        sample.model_copy(
            update={
                "question": truncate_text(sample.question, max_question_chars),
                "context": truncate_text(sample.context, max_context_chars),
            }
        )
        for sample in samples
    ]


def render_lora_command(result: BenchmarkPreparation, config: dict) -> str:
    return f"""mlx_lm.lora \\
    --model {config["model"]} \\
    --train \\
    --data {result.mlx_data_dir} \\
    --batch-size {config["batch_size"]} \\
    --grad-accumulation-steps {config["grad_accumulation_steps"]} \\
    --iters {result.iters} \\
    --learning-rate {config["learning_rate"]} \\
    --max-seq-length {config["max_seq_length"]} \\
    --mask-prompt \\
    --adapter-path results/adapters_{result.mode}"""


def render_infer_command(result: BenchmarkPreparation, config: dict) -> str:
    predictions_path = result.output_dir / f"{result.mode}_lora_test_predictions.jsonl"
    return f"""python scripts/infer.py \\
    --data {result.raw_test_path} \\
    --output {predictions_path} \\
    --mode {result.mode} \\
    --model {config["model"]} \\
    --adapter-path results/adapters_{result.mode}"""


def render_evaluate_command(result: BenchmarkPreparation) -> str:
    predictions_path = result.output_dir / f"{result.mode}_lora_test_predictions.jsonl"
    metrics_path = result.output_dir / f"{result.mode}_lora_test_metrics.json"
    return f"""python scripts/evaluate.py \\
    --data {result.raw_test_path} \\
    --predictions {predictions_path} \\
    --output {metrics_path}"""


def main() -> None:
    args = parse_args()
    config = load_config(args.mode, args.config)
    samples = load_jsonl(args.data)
    result = prepare_benchmark(
        samples=samples,
        mode=args.mode,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        max_question_chars=args.max_question_chars,
        max_context_chars=args.max_context_chars,
        balance_train_labels=args.balance_train_labels,
        target_per_label=args.target_per_label,
    )

    print(
        f"Split {len(samples)} samples -> train={result.train_total} "
        f"val={result.validation_total} test={result.test_total}"
    )
    print(f"Wrote MLX SFT records to {result.mlx_data_dir}/")
    print(f"Wrote held-out evaluation samples to {result.raw_test_path}")
    print("\nTo fine-tune:\n")
    print(render_lora_command(result, config))
    print("\nThen infer on held-out test split:\n")
    print(render_infer_command(result, config))
    print("\nThen evaluate:\n")
    print(render_evaluate_command(result))


if __name__ == "__main__":
    main()
