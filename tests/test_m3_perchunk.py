"""Пофрагментная верификация Метода 3 на dummy batch-бэкенде."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_reliability.methods.m3.perchunk import score_per_chunk
from rag_reliability.schema import RagSample


class DummyBatchJudge:
    """Детерминированный batch-бэкенд без модели и сети."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.network_calls = 0
        self.systems: list[str] = []
        self.users: list[str] = []

    def __call__(self, system: str, users: Sequence[str]) -> list[float]:
        self.network_calls += 1
        self.systems.append(system)
        self.users.extend(users)
        if len(users) != len(self.scores):
            raise AssertionError(
                f"Dummy got {len(users)} prompt(s), but has {len(self.scores)} score(s)"
            )
        return list(self.scores)


def make_sample(n_chunks: int) -> RagSample:
    context = "\n\n".join(
        f"[CHUNK {index}]\nУНИКАЛЬНЫЙ_ЧАНК_{index}" for index in range(1, n_chunks + 1)
    )
    return RagSample(
        id=f"case-{n_chunks}",
        question="Как выполнить операцию?",
        context=context,
        answer="Выполните указанные шаги.",
        faithfulness=1,
        relevance=1,
        marker="none",
    )


@pytest.mark.parametrize("n_chunks", [5, 8])
def test_one_prompt_per_chunk_is_sent_in_one_batch(n_chunks: int) -> None:
    judge = DummyBatchJudge([0.6] * n_chunks)

    score_per_chunk(make_sample(n_chunks), judge)

    assert len(judge.users) == n_chunks
    assert judge.network_calls == 1


def test_each_prompt_contains_only_its_own_chunk() -> None:
    judge = DummyBatchJudge([0.2, 0.5, 0.8])

    score_per_chunk(make_sample(3), judge)

    for index, user in enumerate(judge.users, start=1):
        assert f"УНИКАЛЬНЫЙ_ЧАНК_{index}" in user
        other_chunks = {
            f"УНИКАЛЬНЫЙ_ЧАНК_{other}"
            for other in range(1, 4)
            if other != index
        }
        assert all(chunk not in user for chunk in other_chunks)
