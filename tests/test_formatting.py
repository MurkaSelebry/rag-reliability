"""Tests for target and training-record builders."""

import json

import pytest

from rag_reliability.formatting import (
    build_direct_target,
    build_marker_target,
    build_training_record,
)
from rag_reliability.schema import RagSample


def make_sample(
    faithfulness: int = 1, relevance: int = 1, marker: str | None = None
) -> RagSample:
    return RagSample(
        id="s1",
        question="Какая комиссия?",
        context="Комиссия 1%.",
        answer="Комиссия составляет 1%.",
        faithfulness=faithfulness,
        relevance=relevance,
        marker=marker,
    )


def test_direct_target_keys() -> None:
    target = json.loads(build_direct_target(make_sample(1, 0)))
    assert target == {"faithfulness": 1, "relevance": 0}
    assert "marker" not in target


def test_marker_target_contains_marker() -> None:
    target = json.loads(build_marker_target(make_sample(0, 1, marker="hallucination")))
    assert target == {"marker": "hallucination", "faithfulness": 0, "relevance": 1}


def test_marker_fallback_none_for_reliable() -> None:
    target = json.loads(build_marker_target(make_sample(1, 1, marker=None)))
    assert target["marker"] == "none"


def test_marker_fallback_unknown_for_unreliable() -> None:
    for faithfulness, relevance in [(0, 1), (1, 0), (0, 0)]:
        target = json.loads(build_marker_target(make_sample(faithfulness, relevance, marker=None)))
        assert target["marker"] == "unknown"


def test_training_record_direct() -> None:
    record = build_training_record(make_sample(1, 1), mode="direct")
    assert set(record) == {"prompt", "completion"}
    assert "[QUESTION]" in record["prompt"]
    assert "[CONTEXT]" in record["prompt"]
    assert "[ANSWER]" in record["prompt"]
    completion = json.loads(record["completion"])
    assert set(completion) == {"faithfulness", "relevance"}


def test_training_record_marker() -> None:
    record = build_training_record(make_sample(0, 1, marker="contradiction"), mode="marker")
    completion = json.loads(record["completion"])
    assert set(completion) == {"marker", "faithfulness", "relevance"}
    assert completion["marker"] == "contradiction"
    assert "marker" in record["prompt"].lower()


def test_training_record_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        build_training_record(make_sample(), mode="cot")
