#!/usr/bin/env python
"""Run a trained LettuceDetect logistic-regression classifier.

Outputs repository-compatible Prediction JSONL, so scripts/evaluate.py can be
used unchanged.
"""

from __future__ import annotations

import argparse
import json

import joblib

from common import FeatureConfig, REPO_ROOT, extract_features, make_detector, select_split

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.schema import Prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Input RagSample JSONL")
    parser.add_argument(
        "--model",
        default="results/lettucedetect/classifier.joblib",
        help="Trained classifier from train_classifier.py",
    )
    parser.add_argument(
        "--output",
        default="results/lettucedetect/predictions.jsonl",
        help="Where to write Prediction JSONL",
    )
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=None, help="Override saved LettuceDetect model path")
    parser.add_argument("--threshold", type=float, default=None, help="Override saved threshold")
    parser.add_argument("--device", default=None, help="Override saved device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact = joblib.load(REPO_ROOT / args.model)
    pipeline = artifact["pipeline"]
    saved_config = artifact.get("feature_config", {})
    config = FeatureConfig(
        model_path=args.model_path or saved_config.get("model_path") or FeatureConfig.model_path,
        threshold=(
            args.threshold if args.threshold is not None else saved_config.get("threshold", 0.5)
        ),
        device=args.device if args.device is not None else saved_config.get("device"),
    )

    samples = load_jsonl(REPO_ROOT / args.data)
    samples = select_split(
        samples,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Loaded {len(samples)} sample(s) for split={args.split}")

    detector = make_detector(config)
    features = extract_features(samples, detector, config.threshold, desc=f"features/{args.split}")
    pred_y = pipeline.predict(features)

    predictions: list[Prediction] = []
    for sample, row, feature_row in zip(samples, pred_y, features, strict=True):
        predictions.append(
            Prediction(
                id=sample.id,
                faithfulness_pred=int(row[0]),
                relevance_pred=int(row[1]),
                marker_pred=None,
                raw_output=json.dumps(
                    {
                        "max_prob": float(feature_row[0]),
                        "mean_prob": float(feature_row[1]),
                        "frac_prob_gt_threshold": float(feature_row[2]),
                    },
                    ensure_ascii=False,
                ),
                invalid_output=False,
            )
        )

    output = REPO_ROOT / args.output
    save_jsonl(predictions, output)
    print(f"Wrote {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
