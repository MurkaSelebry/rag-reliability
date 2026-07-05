#!/usr/bin/env python
"""Train logistic regression on LettuceDetect aggregate features.

The script reuses the repository dataset schema and split logic. It does not
modify existing project files; model artifacts and split files are written under
results/lettucedetect/ by default.
"""

from __future__ import annotations

import argparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FeatureConfig, extract_features, make_detector, targets_from_samples
from common import REPO_ROOT

from rag_reliability.dataset import load_jsonl, save_jsonl, split_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Labeled RagSample JSONL")
    parser.add_argument(
        "--output",
        default="results/lettucedetect/classifier.joblib",
        help="Where to save the sklearn pipeline",
    )
    parser.add_argument(
        "--split-dir",
        default="results/lettucedetect/splits",
        help="Where to write RagSample train/val/test splits",
    )
    parser.add_argument("--model-path", default=FeatureConfig.model_path)
    parser.add_argument("--threshold", type=float, default=FeatureConfig.threshold)
    parser.add_argument("--device", default=None, help="LettuceDetect device: cuda, cpu, etc.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def validate_targets(y_train) -> None:
    for idx, name in enumerate(("faithfulness", "relevance")):
        if len(set(y_train[:, idx].tolist())) < 2:
            raise ValueError(
                f"Training split has only one class for {name}. "
                "Use a larger dataset or adjust the split parameters."
            )


def main() -> None:
    args = parse_args()

    samples = load_jsonl(REPO_ROOT / args.data)
    train, val, test = split_samples(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Split {len(samples)} samples -> train={len(train)} val={len(val)} test={len(test)}")

    split_dir = REPO_ROOT / args.split_dir
    save_jsonl(train, split_dir / "train.jsonl")
    save_jsonl(val, split_dir / "val.jsonl")
    save_jsonl(test, split_dir / "test.jsonl")
    print(f"Wrote RagSample splits to {split_dir}")

    config = FeatureConfig(
        model_path=args.model_path,
        threshold=args.threshold,
        device=args.device,
    )
    detector = make_detector(config)

    train_x = extract_features(train, detector, args.threshold, desc="features/train")
    train_y = targets_from_samples(train)
    validate_targets(train_y)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MultiOutputClassifier(LogisticRegression(max_iter=args.max_iter)),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)

    if val:
        val_x = extract_features(val, detector, args.threshold, desc="features/val")
        val_y = targets_from_samples(val)
        val_pred = pipeline.predict(val_x)
        faithfulness_f1 = f1_score(val_y[:, 0], val_pred[:, 0], average="macro", zero_division=0)
        relevance_f1 = f1_score(val_y[:, 1], val_pred[:, 1], average="macro", zero_division=0)
        reliable_true = (val_y[:, 0] & val_y[:, 1]).astype(int)
        reliable_pred = (val_pred[:, 0] & val_pred[:, 1]).astype(int)
        reliable_f1 = f1_score(reliable_true, reliable_pred, average="macro", zero_division=0)
        print(
            "Validation macro-F1: "
            f"reliable={reliable_f1:.4f} "
            f"faithfulness={faithfulness_f1:.4f} "
            f"relevance={relevance_f1:.4f}"
        )

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_config": config.as_dict(),
            "target_names": ["faithfulness", "relevance"],
            "split": {
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
            },
        },
        output,
    )
    print(f"Saved classifier to {output}")


if __name__ == "__main__":
    main()
