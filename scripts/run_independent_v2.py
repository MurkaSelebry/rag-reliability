#!/usr/bin/env python
"""Run Independent Evaluator V2 using a trained model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

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
        default="data/dummy.jsonl",
        help="Input RagSample JSONL.",
    )

    parser.add_argument(
        "--model",
        default="results/independent_v2/model.joblib",
        help="Trained Independent V2 model artifact.",
    )

    parser.add_argument(
        "--output",
        default="results/independent_v2_predictions.jsonl",
        help="Output Prediction JSONL.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
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


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Independent V2 model not found: {model_path.resolve()}"
        )

    artifact = joblib.load(model_path)

    faith_classifier = artifact["faith_classifier"]
    relevance_classifier = artifact["relevance_classifier"]

    faith_threshold = float(artifact["faith_threshold"])
    relevance_threshold = float(artifact["relevance_threshold"])

    samples = load_jsonl(args.data)

    if args.limit is not None:
        samples = samples[: args.limit]

    x = feature_matrix(samples)

    faith_probabilities = faith_classifier.predict_proba(x)[:, 1]
    relevance_probabilities = relevance_classifier.predict_proba(x)[:, 1]

    predictions: list[Prediction] = []

    for sample, faith_prob, relevance_prob in zip(
        samples,
        faith_probabilities,
        relevance_probabilities,
        strict=True,
    ):
        faith_pred = int(faith_prob >= faith_threshold)
        relevance_pred = int(relevance_prob >= relevance_threshold)

        raw_output = json.dumps(
            {
                "method": "independent_v2",
                "faithfulness_probability": float(faith_prob),
                "relevance_probability": float(relevance_prob),
                "faithfulness_threshold": faith_threshold,
                "relevance_threshold": relevance_threshold,
            },
            ensure_ascii=False,
        )

        predictions.append(
    Prediction(
        id=sample.id,
        faithfulness_pred=faith_pred,
        relevance_pred=relevance_pred,
        faithfulness_prob=float(faith_prob),
        relevance_prob=float(relevance_prob),
        prob_method="independent_v2_logreg",
        raw_output=raw_output,
        invalid_output=False,
    )
)

    save_jsonl(predictions, args.output)

    reliable_count = sum(
        prediction.reliable_pred
        for prediction in predictions
    )

    print(f"Loaded Independent V2 model from {model_path}")
    print(
        f"Thresholds: faithfulness={faith_threshold:.2f}, "
        f"relevance={relevance_threshold:.2f}"
    )
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(
        f"Predicted reliable answers: "
        f"{reliable_count}/{len(predictions)}"
    )


if __name__ == "__main__":
    main()