"""Tests for Method 6 pure feature logic and prediction mapping."""

import math
import json

import numpy as np
import pytest

from rag_reliability.methods.m6.features import (
    build_feature_row,
    entropy_features,
    load_sample_cache,
    selfcheck_scores,
    semantic_clusters,
    sentences,
)
from rag_reliability.methods.m6.predict import prediction_from_features
from rag_reliability.schema import RagSample


def make_sample(sample_id: str) -> RagSample:
    return RagSample(
        id=sample_id,
        question="q",
        context="c",
        answer="a",
        faithfulness=1,
        relevance=1,
    )


class StubNLI:
    def __init__(self, table):
        self.table = table

    def score(self, pairs):
        return [self.table.get(pair, {"entail": 0.0, "contra": 0.0}) for pair in pairs]


def test_sentences_and_blank_fallback() -> None:
    assert sentences("Первое предложение. Второе!") == ["Первое предложение.", "Второе!"]
    assert sentences("   ") == ["   "]


def test_selfcheck_mean_max() -> None:
    answer = "Ставка 5%. Срок 30 дней."
    samples = ["s1", "s2"]
    answer_sentences = sentences(answer)
    table = {
        ("s1", answer_sentences[0]): {"entail": 0.0, "contra": 0.8},
        ("s2", answer_sentences[0]): {"entail": 0.0, "contra": 0.6},
        ("s1", answer_sentences[1]): {"entail": 0.0, "contra": 0.1},
        ("s2", answer_sentences[1]): {"entail": 0.0, "contra": 0.3},
    }

    output = selfcheck_scores(answer, samples, StubNLI(table))

    assert output["selfcheck_contra_mean"] == pytest.approx((0.7 + 0.2) / 2)
    assert output["selfcheck_contra_max"] == pytest.approx(0.7)


def _symmetric(left: str, right: str, entail: float) -> dict[tuple[str, str], dict[str, float]]:
    return {
        (left, right): {"entail": entail, "contra": 0.0},
        (right, left): {"entail": entail, "contra": 0.0},
    }


def test_semantic_clusters_union_find() -> None:
    table = {**_symmetric("a", "b", 0.9), **_symmetric("c", "d", 0.2)}
    labels = semantic_clusters(["a", "b", "c", "d"], StubNLI(table), threshold=0.5)

    assert labels[0] == labels[1]
    assert len(set(labels)) == 3


def test_entropy_features_known_distribution() -> None:
    output = entropy_features("answer", ["s1", "s2", "s3"], StubNLI(_symmetric("answer", "s1", 0.9)), 0.5)
    expected = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))

    assert output["semantic_entropy"] == pytest.approx(expected)
    assert output["n_clusters"] == 3
    assert output["answer_in_top_cluster"] == 1.0


def test_prediction_from_features_thresholds() -> None:
    sample = RagSample(
        id="s1",
        question="q",
        context="c",
        answer="a",
        faithfulness=1,
        relevance=1,
    )

    prediction = prediction_from_features(
        sample,
        {
            "selfcheck_contra_mean": 0.2,
            "semantic_entropy": 0.3,
            "cos_q_a": 0.8,
        },
    )

    assert prediction.faithfulness_pred == 1
    assert prediction.relevance_pred == 1
    assert prediction.invalid_output is False


def test_prediction_emits_probabilities() -> None:
    sample = make_sample("x")
    features = {"selfcheck_contra_mean": 0.3, "semantic_entropy": 0.2, "cos_q_a": 0.8}

    pred = prediction_from_features(sample, features)

    assert pred.faithfulness_prob == pytest.approx(0.7)
    assert pred.relevance_prob == pytest.approx(0.8)
    assert pred.prob_method == "m6_features"


def test_probabilities_clipped_to_unit_interval() -> None:
    sample = make_sample("x")
    features = {"selfcheck_contra_mean": 1.4, "semantic_entropy": 0.0, "cos_q_a": -0.2}

    pred = prediction_from_features(sample, features)

    assert pred.faithfulness_prob == 0.0
    assert pred.relevance_prob == 0.0


def test_binary_fields_unchanged_by_prob_addition() -> None:
    sample = make_sample("x")
    features = {"selfcheck_contra_mean": 0.3, "semantic_entropy": 0.2, "cos_q_a": 0.8}

    pred = prediction_from_features(sample, features)

    assert pred.faithfulness_pred == 1 and pred.relevance_pred == 1


def test_load_sample_cache_reads_cached_generation(tmp_path) -> None:
    cache_file = tmp_path / "s1.json"
    cache_file.write_text(
        json.dumps({"id": "s1", "samples": ["ответ 1", "ответ 2"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_sample_cache(tmp_path, "s1") == ["ответ 1", "ответ 2"]


class StubEmbedder:
    def encode(self, texts, normalize_embeddings=True):  # noqa: ARG002
        assert len(texts) == 2
        return np.array([[1.0, 0.0], [1.0, 0.0]])


def test_build_feature_row_combines_selfcheck_entropy_and_similarity() -> None:
    sample = RagSample(
        id="s1",
        question="Как подключить услугу?",
        context="Услуга подключается в кабинете.",
        answer="Откройте кабинет.",
        faithfulness=1,
        relevance=1,
    )
    table = {
        ("sample one", "Откройте кабинет."): {"entail": 0.9, "contra": 0.1},
        ("sample two", "Откройте кабинет."): {"entail": 0.8, "contra": 0.3},
        ("Откройте кабинет.", "sample one"): {"entail": 0.9, "contra": 0.0},
        ("Откройте кабинет.", "sample two"): {"entail": 0.2, "contra": 0.0},
        ("sample one", "sample two"): {"entail": 0.2, "contra": 0.0},
        ("sample two", "sample one"): {"entail": 0.2, "contra": 0.0},
    }

    row = build_feature_row(
        sample,
        ["sample one", "sample two"],
        nli=StubNLI(table),
        embedder=StubEmbedder(),
        entail_threshold=0.5,
    )

    assert row["id"] == "s1"
    assert row["selfcheck_contra_mean"] == pytest.approx(0.2)
    assert row["n_clusters"] == 2
    assert row["cos_q_a"] == pytest.approx(1.0)
