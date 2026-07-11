"""Чистая логика фич m6 на стабе NLI (docs/04 этап 3): кластеры, энтропия, selfcheck."""

import math

import pytest

from rag_reliability_m3m6.methods.m6.features import (
    entropy_features,
    selfcheck_scores,
    semantic_clusters,
    sentences,
)


class StubNLI:
    """Скорер по заранее заданным парам; незнакомая пара -> entail=0, contra=0."""

    def __init__(self, table):
        self.table = table

    def score(self, pairs):
        return [self.table.get(p, {"entail": 0.0, "contra": 0.0}) for p in pairs]


def test_sentences_razdel_and_fallback():
    assert sentences("Первое предложение. Второе!") == ["Первое предложение.", "Второе!"]
    assert sentences("   ") == ["   "]  # пустой текст не теряется


def test_selfcheck_mean_max():
    answer = "Ставка 5%. Срок 30 дней."
    samples = ["s1", "s2"]
    sents = sentences(answer)
    table = {
        (("s1", sents[0])): {"entail": 0, "contra": 0.8},
        (("s2", sents[0])): {"entail": 0, "contra": 0.6},
        (("s1", sents[1])): {"entail": 0, "contra": 0.1},
        (("s2", sents[1])): {"entail": 0, "contra": 0.3},
    }
    out = selfcheck_scores(answer, samples, StubNLI(table))
    assert out["selfcheck_contra_mean"] == pytest.approx((0.7 + 0.2) / 2)
    assert out["selfcheck_contra_max"] == pytest.approx(0.7)


def _sym(a, b, entail):
    return {(a, b): {"entail": entail, "contra": 0.0}, (b, a): {"entail": entail, "contra": 0.0}}


def test_semantic_clusters_union_find():
    texts = ["a", "b", "c", "d"]
    table = {**_sym("a", "b", 0.9), **_sym("c", "d", 0.2)}  # только a~b эквивалентны
    labels = semantic_clusters(texts, StubNLI(table), thr=0.5)
    assert labels[0] == labels[1] and len(set(labels)) == 3


def test_entropy_features_known_distribution():
    """answer+3 сэмпла: кластер {answer, s1} и два синглтона -> p=[.5,.25,.25]."""
    table = {**_sym("ans", "s1", 0.9)}
    out = entropy_features("ans", ["s1", "s2", "s3"], StubNLI(table), thr=0.5)
    expected = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert out["semantic_entropy"] == pytest.approx(expected)
    assert out["n_clusters"] == 3
    assert out["answer_in_top_cluster"] == 1.0
