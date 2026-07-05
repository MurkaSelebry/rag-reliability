"""Compatibility imports for the isolated LettuceDetect method prototype."""

from __future__ import annotations

from classifier import (
    build_classifier,
    predictions_from_outputs,
    targets_from_samples,
    train_feature_classifier,
    validate_targets,
)
from features import (
    DEFAULT_MODEL_PATH,
    FeatureConfig,
    aggregate_token_scores,
    extract_features,
    make_detector,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "FeatureConfig",
    "aggregate_token_scores",
    "build_classifier",
    "extract_features",
    "make_detector",
    "predictions_from_outputs",
    "targets_from_samples",
    "train_feature_classifier",
    "validate_targets",
]
