#!/usr/bin/env python
"""Prepare SFT training files for the MARKER-aware judge and print the mlx_lm.lora command.

Skeleton: no actual training here — it writes {"prompt", "completion"} jsonl
splits and the exact command to launch LoRA fine-tuning with mlx-lm.

Example:
    python scripts/train_marker_lora.py --data data/dummy.jsonl --config configs/marker_lora.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rag_reliability.dataset import load_jsonl, split_samples, write_training_jsonl

MODE = "marker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Labeled dataset (jsonl)")
    parser.add_argument("--config", default=f"configs/{MODE}_lora.yaml", help="Training config")
    parser.add_argument("--output-dir", default="results", help="Where to write training files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    samples = load_jsonl(args.data)
    train, val, test = split_samples(samples)
    print(f"Split {len(samples)} samples -> train={len(train)} val={len(val)} test={len(test)}")

    output_dir = Path(args.output_dir)
    # Flat file per project convention + mlx-lm data dir (expects train/valid.jsonl).
    write_training_jsonl(train, output_dir / f"train_{MODE}.jsonl", MODE)
    mlx_data_dir = output_dir / f"lora_{MODE}"
    write_training_jsonl(train, mlx_data_dir / "train.jsonl", MODE)
    write_training_jsonl(val, mlx_data_dir / "valid.jsonl", MODE)
    write_training_jsonl(test, mlx_data_dir / "test.jsonl", MODE)
    print(f"Wrote training records to {output_dir / f'train_{MODE}.jsonl'} and {mlx_data_dir}/")

    epochs, batch_size = int(config["epochs"]), int(config["batch_size"])
    if epochs < 1 or batch_size < 1:
        raise ValueError(f"epochs and batch_size must be >= 1, got {epochs}, {batch_size}")
    iters = max(1, len(train) * epochs // batch_size)
    command = f"""mlx_lm.lora \\
    --model {config["model"]} \\
    --train \\
    --data {mlx_data_dir} \\
    --batch-size {batch_size} \\
    --grad-accumulation-steps {config["grad_accumulation_steps"]} \\
    --iters {iters} \\
    --learning-rate {config["learning_rate"]} \\
    --max-seq-length {config["max_seq_length"]} \\
    --mask-prompt \\
    --adapter-path results/adapters_{MODE}"""
    print("\nTo fine-tune (requires `pip install -e \".[mlx]\"`):\n")
    print(command)


if __name__ == "__main__":
    main()
