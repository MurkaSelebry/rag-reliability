#!/usr/bin/env python
"""Train Independent Evaluator V2 using lightweight engineered features.

Pipeline:
RagSample JSONL
    -> deterministic feature extraction
    -> two logistic-regression classifiers
       (faithfulness and relevance)
    -> validation threshold tuning
    -> held-out test predictions
    -> repository-compatible Prediction JSONL + metrics JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.methods.independent.features import extract_feature_rows
from rag_reliability.schema import Prediction, RagSample


FEATURE_NAMES = [
    "context_coverage",
    "full_question_coverage",
    "latest_question_coverage",
    "context_answer_jaccard",
    "question_answer_jaccard",
    "latest_question_answer_jaccard",
    "number_support",
    "answer_token_count",
    "question_token_count",
    "context_token_count",
    "answer_question_length_ratio",
    "false_verification",
    "redirect_only",
    "reveals_ai_identity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        default="data/organizers.jsonl",
        help="Input RagSample JSONL.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/independent_v2",
        help="Directory for predictions and metrics.",
    )

    # Explicit shared split files
    parser.add_argument(
        "--train-data",
        default=None,
        help="Explicit training split JSONL.",
    )

    parser.add_argument(
        "--val-data",
        default=None,
        help="Explicit validation split JSONL.",
    )

    parser.add_argument(
        "--test-data",
        default=None,
        help="Explicit held-out test split JSONL.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def feature_matrix(samples: list[RagSample]) -> np.ndarray:
    rows = extract_feature_rows(samples)

    return np.asarray(
        [
            [float(row[name]) for name in FEATURE_NAMES]
            for row in rows
        ],
        dtype=float,
    )


def split_samples(
    samples: list[RagSample],
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[list[RagSample], list[RagSample], list[RagSample]]:
    reliability = [sample.reliable for sample in samples]

    train_validation, test = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=reliability,
    )

    train_validation_reliability = [
        sample.reliable for sample in train_validation
    ]

    relative_validation_size = validation_size / (1.0 - test_size)

    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=train_validation_reliability,
    )

    return train, validation, test


def build_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def best_threshold(
    gold: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    best_value = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = (probabilities >= threshold).astype(int)

        score = f1_score(
            gold,
            prediction,
            average="macro",
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_value = float(threshold)

    return best_value, best_f1


def macro_f1(
    gold: list[int],
    predicted: list[int],
) -> float:
    return float(
        f1_score(
            gold,
            predicted,
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    args = parse_args()

    # Use explicit shared splits when all three are provided.
    if args.train_data and args.val_data and args.test_data:
        train_samples = load_jsonl(args.train_data)
        validation_samples = load_jsonl(args.val_data)
        test_samples = load_jsonl(args.test_data)
    else:
        samples = load_jsonl(args.data)

        train_samples, validation_samples, test_samples = split_samples(
            samples,
            test_size=args.test_size,
            validation_size=args.validation_size,
            seed=args.seed,
        )

    print(
        f"Split: train={len(train_samples)}, "
        f"validation={len(validation_samples)}, "
        f"test={len(test_samples)}"
    )

    x_train = feature_matrix(train_samples)
    x_validation = feature_matrix(validation_samples)
    x_test = feature_matrix(test_samples)

    y_train_faith = np.asarray(
        [sample.faithfulness for sample in train_samples],
        dtype=int,
    )
    y_train_rel = np.asarray(
        [sample.relevance for sample in train_samples],
        dtype=int,
    )

    y_validation_faith = np.asarray(
        [sample.faithfulness for sample in validation_samples],
        dtype=int,
    )
    y_validation_rel = np.asarray(
        [sample.relevance for sample in validation_samples],
        dtype=int,
    )

    faith_classifier = build_classifier(args.seed)
    relevance_classifier = build_classifier(args.seed)

    faith_classifier.fit(x_train, y_train_faith)
    relevance_classifier.fit(x_train, y_train_rel)

    faith_validation_prob = faith_classifier.predict_proba(
        x_validation
    )[:, 1]

    relevance_validation_prob = relevance_classifier.predict_proba(
        x_validation
    )[:, 1]

    faith_threshold, faith_validation_f1 = best_threshold(
        y_validation_faith,
        faith_validation_prob,
    )

    relevance_threshold, relevance_validation_f1 = best_threshold(
        y_validation_rel,
        relevance_validation_prob,
    )

    print(
        f"Faithfulness threshold: {faith_threshold:.2f} "
        f"(val macro-F1={faith_validation_f1:.4f})"
    )

    print(
        f"Relevance threshold: {relevance_threshold:.2f} "
        f"(val macro-F1={relevance_validation_f1:.4f})"
    )

    faith_test_prob = faith_classifier.predict_proba(x_test)[:, 1]
    relevance_test_prob = relevance_classifier.predict_proba(x_test)[:, 1]

    faith_test_pred = (
        faith_test_prob >= faith_threshold
    ).astype(int)

    relevance_test_pred = (
        relevance_test_prob >= relevance_threshold
    ).astype(int)

    reliable_test_pred = (
        (faith_test_pred == 1)
        & (relevance_test_pred == 1)
    ).astype(int)

    gold_faith = [sample.faithfulness for sample in test_samples]
    gold_rel = [sample.relevance for sample in test_samples]
    gold_reliable = [sample.reliable for sample in test_samples]

    metrics = {
        "method": "independent_v2",
        "seed": args.seed,
        "train_size": len(train_samples),
        "validation_size": len(validation_samples),
        "test_size": len(test_samples),
        "faithfulness_threshold": faith_threshold,
        "relevance_threshold": relevance_threshold,
        "faithfulness_f1_macro": macro_f1(
            gold_faith,
            faith_test_pred.tolist(),
        ),
        "relevance_f1_macro": macro_f1(
            gold_rel,
            relevance_test_pred.tolist(),
        ),
        "reliable_f1_macro": macro_f1(
            gold_reliable,
            reliable_test_pred.tolist(),
        ),
        "invalid_output_rate": 0.0,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
    {
        "faith_classifier": faith_classifier,
        "relevance_classifier": relevance_classifier,
        "faith_threshold": faith_threshold,
        "relevance_threshold": relevance_threshold,
        "feature_names": FEATURE_NAMES,
    },
    output_dir / "model.joblib",
)

    metrics_path = output_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions: list[Prediction] = []

    for (
        sample,
        faith_probability,
        relevance_probability,
        faith_prediction,
        relevance_prediction,
    ) in zip(
        test_samples,
        faith_test_prob,
        relevance_test_prob,
        faith_test_pred,
        relevance_test_pred,
        strict=True,
    ):
        raw_output = json.dumps(
            {
                "method": "independent_v2",
                "faithfulness_probability": float(
                    faith_probability
                ),
                "relevance_probability": float(
                    relevance_probability
                ),
                "faithfulness_threshold": faith_threshold,
                "relevance_threshold": relevance_threshold,
            },
            ensure_ascii=False,
        )

        predictions.append(
            Prediction(
                id=sample.id,
                faithfulness_pred=int(faith_prediction),
                relevance_pred=int(relevance_prediction),
                raw_output=raw_output,
                invalid_output=False,
            )
        )

    predictions_path = output_dir / "predictions.jsonl"

    save_jsonl(
        predictions,
        predictions_path,
    )

    print()
    print("TEST RESULTS")
    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()