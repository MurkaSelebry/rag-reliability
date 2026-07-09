"""Tests for schema validation."""

import pytest
from pydantic import ValidationError

from rag_reliability.schema import ALLOWED_MARKERS, Prediction, RagSample


def make_sample(**overrides) -> RagSample:
    base = dict(
        id="s1", question="q", context="c", answer="a", faithfulness=1, relevance=1
    )
    base.update(overrides)
    return RagSample(**base)


def test_valid_markers_accepted() -> None:
    for marker in ALLOWED_MARKERS:
        assert make_sample(marker=marker).marker == marker
    assert make_sample(marker=None).marker is None


def test_invalid_marker_rejected() -> None:
    with pytest.raises(ValidationError, match="marker"):
        make_sample(marker="halucination")  # typo must not reach training data


def test_official_organizer_markers_accepted() -> None:
    assert make_sample(marker="reason_hallucinated_fact").marker == "reason_hallucinated_fact"
    assert make_sample(marker="reason_missed_chunk_conditions").marker == (
        "reason_missed_chunk_conditions"
    )


def test_prediction_marker_free_form() -> None:
    # Model output markers stay free-form: bad values must show up in
    # metrics/confusion, not crash the pipeline.
    pred = Prediction(id="s1", faithfulness_pred=0, relevance_pred=0, marker_pred="whatever")
    assert pred.marker_pred == "whatever"


def test_reliable_derivation() -> None:
    assert make_sample(faithfulness=1, relevance=1).reliable == 1
    assert make_sample(faithfulness=1, relevance=0).reliable == 0
    assert make_sample(faithfulness=0, relevance=1).reliable == 0
