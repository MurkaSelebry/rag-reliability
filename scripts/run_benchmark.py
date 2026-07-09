#!/usr/bin/env python
"""Run supported reliability methods through one predictions -> metrics contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


METHODS = (
    "dummy_direct",
    "dummy_marker",
    "prompt_direct",
    "prompt_marker",
    "lora_direct",
    "lora_marker",
    "lettucedetect",
    "encoder",
)


@dataclass(frozen=True)
class MethodRun:
    name: str
    predictions_path: Path
    metrics_path: Path
    run_command: list[str]
    evaluate_command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Input RagSample JSONL")
    parser.add_argument("--output-dir", default="results/benchmark", help="Where to write runs")
    parser.add_argument(
        "--methods",
        default="dummy_direct,dummy_marker",
        help=f"Comma-separated methods. Available: {', '.join(METHODS)}",
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--direct-adapter-path", default="results/adapters_direct")
    parser.add_argument("--marker-adapter-path", default="results/adapters_marker")
    parser.add_argument("--lettucedetect-model", default="results/lettucedetect/classifier.joblib")
    parser.add_argument("--encoder-model", default="deepvk/RuModernBERT-base")
    parser.add_argument("--encoder-output-dir", default=None)
    parser.add_argument("--encoder-max-length", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--encoder-epochs", type=float, default=3)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--encoder-pos-weight-mode", choices=["balanced", "none"], default="none")
    return parser.parse_args()


def parse_methods(raw_methods: str) -> list[str]:
    methods = [method.strip() for method in raw_methods.split(",") if method.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available: {METHODS}")
    return methods


def build_evaluate_command(
    python: str,
    data: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> list[str]:
    return [
        python,
        "scripts/evaluate.py",
        "--data",
        str(data),
        "--predictions",
        str(predictions_path),
        "--output",
        str(metrics_path),
    ]


def build_method_run(  # noqa: PLR0913, PLR0912
    method: str,
    data: Path,
    output_dir: Path,
    python: str = sys.executable,
    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    max_tokens: int = 64,
    direct_adapter_path: str = "results/adapters_direct",
    marker_adapter_path: str = "results/adapters_marker",
    lettucedetect_model: str = "results/lettucedetect/classifier.joblib",
    encoder_model: str = "deepvk/RuModernBERT-base",
    encoder_output_dir: str | None = None,
    encoder_max_length: int = 512,
    encoder_batch_size: int = 4,
    encoder_epochs: float = 3,
    encoder_learning_rate: float = 2e-5,
    encoder_pos_weight_mode: str = "none",
) -> MethodRun:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    run_dir = output_dir / method
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"

    if method.startswith("dummy_"):
        mode = method.removeprefix("dummy_")
        command = [
            python,
            "scripts/run_prompt_baseline.py",
            "--data",
            str(data),
            "--output",
            str(predictions_path),
            "--mode",
            mode,
            "--backend",
            "dummy",
            "--dummy-strategy",
            "keyword" if mode == "marker" else "always_reliable",
        ]
    elif method.startswith("prompt_"):
        mode = method.removeprefix("prompt_")
        command = [
            python,
            "scripts/run_prompt_baseline.py",
            "--data",
            str(data),
            "--output",
            str(predictions_path),
            "--mode",
            mode,
            "--backend",
            "mlx",
            "--model",
            model,
            "--max-tokens",
            str(max_tokens),
        ]
    elif method.startswith("lora_"):
        mode = method.removeprefix("lora_")
        adapter_path = direct_adapter_path if mode == "direct" else marker_adapter_path
        command = [
            python,
            "scripts/infer.py",
            "--data",
            str(data),
            "--output",
            str(predictions_path),
            "--mode",
            mode,
            "--model",
            model,
            "--adapter-path",
            adapter_path,
            "--max-tokens",
            str(max_tokens),
        ]
    elif method == "lettucedetect":
        command = [
            python,
            "scripts/infer_lettucedetect.py",
            "--data",
            str(data),
            "--model",
            lettucedetect_model,
            "--output",
            str(predictions_path),
        ]
    else:
        checkpoint_dir = encoder_output_dir or str(run_dir / "checkpoints")
        command = [
            python,
            "scripts/train_encoder_baseline.py",
            "--data",
            str(data),
            "--output",
            str(run_dir / "encoder_binary_metrics.json"),
            "--predictions-output",
            str(predictions_path),
            "--model",
            encoder_model,
            "--output-dir",
            checkpoint_dir,
            "--max-length",
            str(encoder_max_length),
            "--batch-size",
            str(encoder_batch_size),
            "--epochs",
            str(encoder_epochs),
            "--learning-rate",
            str(encoder_learning_rate),
            "--pos-weight-mode",
            encoder_pos_weight_mode,
        ]

    return MethodRun(
        name=method,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        run_command=command,
        evaluate_command=build_evaluate_command(python, data, predictions_path, metrics_path),
    )


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    output_dir = Path(args.output_dir)
    runs = [
        build_method_run(
            method=method,
            data=data,
            output_dir=output_dir,
            model=args.model,
            max_tokens=args.max_tokens,
            direct_adapter_path=args.direct_adapter_path,
            marker_adapter_path=args.marker_adapter_path,
            lettucedetect_model=args.lettucedetect_model,
            encoder_model=args.encoder_model,
            encoder_output_dir=args.encoder_output_dir,
            encoder_max_length=args.encoder_max_length,
            encoder_batch_size=args.encoder_batch_size,
            encoder_epochs=args.encoder_epochs,
            encoder_learning_rate=args.encoder_learning_rate,
            encoder_pos_weight_mode=args.encoder_pos_weight_mode,
        )
        for method in parse_methods(args.methods)
    ]

    summary = {}
    for method_run in runs:
        method_run.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(method_run.run_command)
        run_command(method_run.evaluate_command)
        summary[method_run.name] = {
            "predictions": str(method_run.predictions_path),
            "metrics": str(method_run.metrics_path),
        }

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
