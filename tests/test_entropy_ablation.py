"""Абляция: кластеры при нескольких порогах и N из одной NLI-матрицы."""

from rag_reliability.methods.m6.entropy_ablation import cluster_features_multi


class StubNLI:
    def __init__(self, table):
        self.table = table

    def score(self, pairs):
        return [self.table.get(p, {"entail": 0.0, "contra": 0.0}) for p in pairs]


def _sym(a, b, e):
    return {(a, b): {"entail": e, "contra": 0.0}, (b, a): {"entail": e, "contra": 0.0}}


def test_thresholds_split_clusters():
    """Пара с entail=0.45: при thr=0.4 — один кластер, при thr=0.5 — два."""
    table = {**_sym("ans", "s1", 0.45)}
    res = cluster_features_multi("ans", ["s1"], StubNLI(table), thresholds=[0.4, 0.5], ns=[1])
    assert res[(0.4, 1)]["n_clusters"] == 1
    assert res[(0.5, 1)]["n_clusters"] == 2


def test_n_slice_uses_prefix():
    """N=1 использует только первый сэмпл; NLI-матрица считается один раз (для max N)."""
    calls = []

    class CountingNLI(StubNLI):
        def score(self, pairs):
            calls.append(len(pairs))
            return super().score(pairs)

    table = {**_sym("ans", "s1", 0.9), **_sym("ans", "s2", 0.9), **_sym("s1", "s2", 0.9)}
    res = cluster_features_multi(
        "ans", ["s1", "s2"], CountingNLI(table), thresholds=[0.5], ns=[1, 2]
    )
    assert res[(0.5, 2)]["n_clusters"] == 1
    assert res[(0.5, 1)]["n_clusters"] == 1  # ans~s1
    assert len(calls) == 1  # один батч на кейс
