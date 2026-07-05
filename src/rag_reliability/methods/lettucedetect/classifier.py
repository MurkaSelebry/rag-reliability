"""Classifier utilities for LettuceDetect aggregate features."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from rag_reliability.schema import Prediction, RagSample


def targets_from_samples(samples: list["RagSample"]) -> np.ndarray:
    """Return [faithfulness, relevance] binary targets for every sample."""
    return np.asarray([[s.faithfulness, s.relevance] for s in samples], dtype=np.int64)


def validate_targets(y_train: np.ndarray) -> None:
    """Fail early when logistic regression cannot be trained for a target."""
    for idx, name in enumerate(("faithfulness", "relevance")):
        if len(set(y_train[:, idx].tolist())) < 2:
            raise ValueError(
                f"Training split has only one class for {name}. "
                "Use a larger dataset or adjust the split parameters."
            )


def build_classifier(max_iter: int = 1000) -> Pipeline:
    """Build the sklearn model used by the LettuceDetect method."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", MultiOutputClassifier(LogisticRegression(max_iter=max_iter))),
        ]
    )


def train_feature_classifier(train_x: np.ndarray, train_y: np.ndarray, max_iter: int = 1000):
    """Fit a multi-output logistic regression on already extracted features."""
    validate_targets(train_y)
    pipeline = build_classifier(max_iter=max_iter)
    pipeline.fit(train_x, train_y)
    return pipeline


def predictions_from_outputs(
    samples: list["RagSample"],
    pred_y: np.ndarray,
    features: np.ndarray,
) -> list["Prediction"]:
    """Build repository-compatible Prediction objects from classifier outputs."""
    from rag_reliability.schema import Prediction

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
    return predictions
