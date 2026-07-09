"""Tests for the supervised encoder baseline helpers."""

import importlib.util
from pathlib import Path

from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "train_encoder_baseline",
    Path(__file__).parents[1] / "scripts" / "train_encoder_baseline.py",
)
assert _SPEC is not None
encoder_baseline = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(encoder_baseline)


def make_sample(faithfulness: int = 1, relevance: int = 1) -> RagSample:
    return RagSample(
        id="organizer_000001",
        question="Клиент: Как подключить Alfa Pay?",
        context="[CHUNK 1]\nПодключите Alfa Pay в настройках карты.",
        answer="Откройте карту и выберите оплату смартфоном.",
        faithfulness=faithfulness,
        relevance=relevance,
        marker="none" if faithfulness and relevance else "unknown",
    )


def test_build_encoder_text_uses_dialog_answer_and_context() -> None:
    text = encoder_baseline.build_encoder_text(make_sample())

    assert text == (
        "dialog: Клиент: Как подключить Alfa Pay?\n"
        "answer: Откройте карту и выберите оплату смартфоном.\n"
        "context: [CHUNK 1]\nПодключите Alfa Pay в настройках карты."
    )


def test_reliability_labels_are_derived_from_both_binary_labels() -> None:
    samples = [
        make_sample(1, 1),
        make_sample(1, 0),
        make_sample(0, 1),
        make_sample(0, 0),
    ]

    assert encoder_baseline.reliability_labels(samples) == [1, 0, 0, 0]


def test_compute_binary_metrics_thresholds_probabilities() -> None:
    metrics = encoder_baseline.compute_binary_metrics(
        y_true=[1, 0, 1, 0],
        y_prob=[0.9, 0.7, 0.6, 0.1],
        threshold=0.5,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.8
    assert metrics["f1_macro"] == 0.7333333333333334
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_find_best_threshold_maximizes_macro_f1() -> None:
    threshold, metrics = encoder_baseline.find_best_threshold(
        y_true=[1, 0, 1, 0],
        y_prob=[0.9, 0.8, 0.7, 0.2],
        thresholds=[0.5, 0.85],
    )

    assert threshold == 0.5
    assert metrics["f1_macro"] == 0.7333333333333334


def test_split_samples_uses_train_validation_and_test_sets() -> None:
    samples = [make_sample(1, 1) for _ in range(12)] + [make_sample(0, 1) for _ in range(8)]

    train_samples, validation_samples, test_samples = encoder_baseline.split_samples(
        samples,
        test_size=0.2,
        validation_size=0.2,
        seed=42,
    )

    assert len(train_samples) == 12
    assert len(validation_samples) == 4
    assert len(test_samples) == 4
    assert sum(encoder_baseline.reliability_labels(validation_samples)) in {2, 3}
    assert sum(encoder_baseline.reliability_labels(test_samples)) in {2, 3}


def test_compute_pos_weight_can_be_disabled() -> None:
    assert encoder_baseline.compute_pos_weight([1, 1, 1, 0], mode="none") == 1.0


def test_compute_pos_weight_balances_positive_majority() -> None:
    assert encoder_baseline.compute_pos_weight([1, 1, 1, 0], mode="balanced") == 1 / 3
