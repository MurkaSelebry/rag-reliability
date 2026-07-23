"""Convert precomputed Method 6 features to repository-wide predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_reliability.schema import Prediction, RagSample


def load_features(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Method 6 features file not found: {path.resolve()}")
    features: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid features JSON at {path}:{line_no}: {exc}") from exc
            if "id" not in row:
                raise ValueError(f"Feature row at {path}:{line_no} has no id")
            features[str(row["id"])] = row
    return features


def prediction_from_features(
    sample: RagSample,
    features: dict[str, Any],
    *,
    contradiction_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    relevance_threshold: float = 0.25,
) -> Prediction:
    """Map SelfCheck-style features to binary labels with explicit thresholds."""
    contradiction = float(features.get("selfcheck_contra_mean", 0.0))
    entropy = float(features.get("semantic_entropy", 0.0))
    cosine = float(features.get("cos_q_a", 1.0))
    faithfulness = int(contradiction <= contradiction_threshold and entropy <= entropy_threshold)
    relevance = int(cosine >= relevance_threshold)
    p_faith = max(0.0, min(1.0, 1.0 - contradiction))
    p_rel = max(0.0, min(1.0, cosine))
    return Prediction(
        id=sample.id,
        faithfulness_pred=faithfulness,
        relevance_pred=relevance,
        raw_output=json.dumps(features, ensure_ascii=False),
        invalid_output=False,
        faithfulness_prob=p_faith,
        relevance_prob=p_rel,
        prob_method="m6_features",
    )


def predictions_from_feature_rows(
    samples: list[RagSample],
    features_by_id: dict[str, dict[str, Any]],
    *,
    contradiction_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    relevance_threshold: float = 0.25,
) -> list[Prediction]:
    missing = [sample.id for sample in samples if sample.id not in features_by_id]
    if missing:
        raise ValueError(f"Missing Method 6 features for {len(missing)} sample(s): {missing[:5]}")
    return [
        prediction_from_features(
            sample,
            features_by_id[sample.id],
            contradiction_threshold=contradiction_threshold,
            entropy_threshold=entropy_threshold,
            relevance_threshold=relevance_threshold,
        )
        for sample in samples
    ]
