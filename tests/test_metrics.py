"""Tests for evaluation metrics."""

import pytest

from rag_reliability.metrics import evaluate_predictions
from rag_reliability.schema import Prediction, RagSample


def make_sample(
    sample_id: str, faithfulness: int, relevance: int, marker: str | None = None
) -> RagSample:
    return RagSample(
        id=sample_id,
        question="q",
        context="c",
        answer="a",
        faithfulness=faithfulness,
        relevance=relevance,
        marker=marker,
    )


def make_prediction(
    sample_id: str,
    faithfulness: int,
    relevance: int,
    invalid: bool = False,
    marker: str | None = None,
) -> Prediction:
    return Prediction(
        id=sample_id,
        faithfulness_pred=faithfulness,
        relevance_pred=relevance,
        marker_pred=marker,
        invalid_output=invalid,
    )


def test_reliable_is_and_of_labels() -> None:
    samples = [
        make_sample("a", 1, 1),  # reliable
        make_sample("b", 1, 0),  # not reliable
        make_sample("c", 0, 1),  # not reliable
        make_sample("d", 0, 0),  # not reliable
    ]
    # Predict faithfulness perfectly, relevance always 1:
    # reliable_pred = 1 for a (correct) and c... no: c has f_pred=0 -> reliable_pred=0.
    predictions = [
        make_prediction("a", 1, 1),
        make_prediction("b", 1, 1),  # wrong reliable (pred 1, true 0)
        make_prediction("c", 0, 1),
        make_prediction("d", 0, 1),
    ]
    result = evaluate_predictions(samples, predictions)
    assert result.total == 4
    assert result.faithfulness_f1_macro == 1.0
    assert result.reliable_f1_macro < 1.0  # sample b broke reliable


def test_perfect_predictions() -> None:
    samples = [make_sample("a", 1, 1), make_sample("b", 0, 0), make_sample("c", 1, 0)]
    predictions = [
        make_prediction("a", 1, 1),
        make_prediction("b", 0, 0),
        make_prediction("c", 1, 0),
    ]
    result = evaluate_predictions(samples, predictions)
    assert result.reliable_f1_macro == 1.0
    assert result.faithfulness_f1_macro == 1.0
    assert result.relevance_f1_macro == 1.0
    assert result.invalid_output_rate == 0.0


def test_invalid_output_rate() -> None:
    samples = [make_sample("a", 1, 1), make_sample("b", 0, 0)]
    predictions = [
        make_prediction("a", 0, 0, invalid=True),
        make_prediction("b", 0, 0),
    ]
    result = evaluate_predictions(samples, predictions)
    assert result.invalid_count == 1
    assert result.invalid_output_rate == 0.5


def test_single_class_no_crash() -> None:
    # All samples reliable, all predictions unreliable: F1 must not raise.
    samples = [make_sample("a", 1, 1), make_sample("b", 1, 1)]
    predictions = [make_prediction("a", 0, 0), make_prediction("b", 0, 0)]
    result = evaluate_predictions(samples, predictions)
    assert 0.0 <= result.reliable_f1_macro <= 1.0


def test_duplicate_prediction_ids_raise() -> None:
    # Dict-collapse would silently keep only the last duplicate.
    samples = [make_sample("a", 1, 1), make_sample("b", 0, 0)]
    predictions = [
        make_prediction("a", 1, 1),
        make_prediction("a", 0, 0),
        make_prediction("b", 0, 0),
    ]
    with pytest.raises(ValueError, match="Duplicate prediction id"):
        evaluate_predictions(samples, predictions)


def test_missing_prediction_raises() -> None:
    samples = [make_sample("a", 1, 1), make_sample("b", 0, 0)]
    predictions = [make_prediction("a", 1, 1)]
    with pytest.raises(ValueError, match="Missing predictions"):
        evaluate_predictions(samples, predictions)


def test_empty_samples_raise() -> None:
    with pytest.raises(ValueError, match="No samples"):
        evaluate_predictions([], [])


def test_marker_metrics_absent_without_marker_predictions() -> None:
    # Direct mode: no prediction carries a marker -> marker metrics stay None.
    samples = [make_sample("a", 1, 1), make_sample("b", 0, 0)]
    predictions = [make_prediction("a", 1, 1), make_prediction("b", 0, 0)]
    result = evaluate_predictions(samples, predictions)
    assert result.marker_f1_macro is None
    assert result.marker_per_class_f1 is None
    assert result.marker_confusion is None


def test_marker_metrics_perfect() -> None:
    samples = [
        make_sample("a", 1, 1, marker="none"),
        make_sample("b", 0, 1, marker="hallucination"),
        make_sample("c", 1, 0, marker="off_topic_answer"),
    ]
    predictions = [
        make_prediction("a", 1, 1, marker="none"),
        make_prediction("b", 0, 1, marker="hallucination"),
        make_prediction("c", 1, 0, marker="off_topic_answer"),
    ]
    result = evaluate_predictions(samples, predictions)
    assert result.marker_f1_macro == 1.0
    assert result.marker_per_class_f1 == {
        "none": 1.0,
        "hallucination": 1.0,
        "off_topic_answer": 1.0,
    }
    assert result.marker_confusion == {
        "none": {"none": 1},
        "hallucination": {"hallucination": 1},
        "off_topic_answer": {"off_topic_answer": 1},
    }


def test_marker_confusion_counts_mistakes() -> None:
    samples = [
        make_sample("a", 0, 1, marker="hallucination"),
        make_sample("b", 0, 1, marker="hallucination"),
    ]
    predictions = [
        make_prediction("a", 0, 1, marker="hallucination"),
        make_prediction("b", 0, 1, marker="contradiction"),
    ]
    result = evaluate_predictions(samples, predictions)
    assert result.marker_confusion == {
        "hallucination": {"hallucination": 1, "contradiction": 1},
    }
    assert result.marker_per_class_f1 is not None
    assert result.marker_per_class_f1["hallucination"] == pytest.approx(2 / 3)
    assert result.marker_per_class_f1["contradiction"] == 0.0


def test_marker_gold_derived_when_sample_has_no_marker() -> None:
    # Gold marker falls back to none (reliable) / unknown (unreliable),
    # same rule as training targets.
    samples = [make_sample("a", 1, 1), make_sample("b", 0, 0)]
    predictions = [
        make_prediction("a", 1, 1, marker="none"),
        make_prediction("b", 0, 0, marker="unknown"),
    ]
    result = evaluate_predictions(samples, predictions)
    assert result.marker_f1_macro == 1.0


def test_marker_pred_missing_falls_back_to_unknown() -> None:
    # One marker-mode prediction lost its marker (e.g. parse fallback):
    # treated as "unknown", metrics still computed.
    samples = [
        make_sample("a", 0, 1, marker="hallucination"),
        make_sample("b", 0, 0, marker="contradiction"),
    ]
    predictions = [
        make_prediction("a", 0, 1, marker="hallucination"),
        make_prediction("b", 0, 0, invalid=True),  # marker_pred=None
    ]
    result = evaluate_predictions(samples, predictions)
    assert result.marker_confusion == {
        "hallucination": {"hallucination": 1},
        "contradiction": {"unknown": 1},
    }
