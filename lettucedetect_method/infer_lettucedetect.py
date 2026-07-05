#!/usr/bin/env python
"""Run a trained LettuceDetect logistic-regression classifier."""

from __future__ import annotations

import argparse

import joblib

from _bootstrap import REPO_ROOT, add_repo_src_to_path

add_repo_src_to_path()

from classifier import predictions_from_outputs
from features import FeatureConfig, extract_features, make_detector
from rag_reliability.dataset import load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Input RagSample JSONL")
    parser.add_argument(
        "--model",
        default="results/lettucedetect/classifier.joblib",
        help="Trained classifier from train_lettucedetect.py",
    )
    parser.add_argument(
        "--output",
        default="results/lettucedetect/predictions.jsonl",
        help="Where to write Prediction JSONL",
    )
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
    print(f"Loaded {len(samples)} sample(s) from {args.data}")

    detector = make_detector(config)
    features = extract_features(samples, detector, config.threshold, desc="features")
    pred_y = pipeline.predict(features)
    predictions = predictions_from_outputs(samples, pred_y, features)

    output = REPO_ROOT / args.output
    save_jsonl(predictions, output)
    print(f"Wrote {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
