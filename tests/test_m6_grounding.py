"""Тесты NLI-grounding на мок-NLI: весов и GPU не нужно.

Мок — детерминированная таблица (premise, hypothesis) -> (entail, contra), так
что проверяются свойства агрегации, а не поведение конкретной NLI-модели.
"""

from __future__ import annotations

import math

import pytest

from rag_reliability.methods.m6.grounding import (
    GROUNDING_KEYS,
    compute_grounding,
    grounding_features,
    score_matrix,
)


class TableNLI:
    """Мок-NLI: явная таблица пар, дефолт — «не подкреплено, но и не противоречит»."""

    def __init__(self, table: dict[tuple[str, str], tuple[float, float]]) -> None:
        self.table = table
        self.calls = 0

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        self.calls += 1
        rows = []
        for pair in pairs:
            entail, contra = self.table.get(pair, (0.05, 0.05))
            rows.append({"entail": entail, "contra": contra})
        return rows


def split_lines(text: str) -> list[str]:
    """Простой сплиттер для тестов: одна строка — одно «предложение»."""
    return [line.strip() for line in text.splitlines() if line.strip()]


SUPPORTED = "Комиссия составляет 100 рублей."
SUPPORTED_2 = "Срок перевода — 5 дней."
UNSUPPORTED = "Кэшбэк начисляется 30 числа."
CHUNK_A = "chunk about комиссия"
CHUNK_B = "chunk about сроки"


def make_nli(**overrides: tuple[float, float]) -> TableNLI:
    table = {
        (CHUNK_A, SUPPORTED): (0.90, 0.02),
        (CHUNK_B, SUPPORTED): (0.10, 0.05),
        (CHUNK_A, SUPPORTED_2): (0.20, 0.05),
        (CHUNK_B, SUPPORTED_2): (0.85, 0.03),
        (CHUNK_A, UNSUPPORTED): (0.05, 0.40),
        (CHUNK_B, UNSUPPORTED): (0.08, 0.60),
    }
    for key, value in overrides.items():
        sentence, chunk = key.split("__")
        table[({"a": CHUNK_A, "b": CHUNK_B}[chunk], sentence)] = value
    return TableNLI(table)


# --------------------------------------------------------------------------- #
# Контракт фич
# --------------------------------------------------------------------------- #


def test_returns_exactly_eight_declared_finite_features() -> None:
    features = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}",
        [CHUNK_A, CHUNK_B],
        make_nli(),
        sentence_splitter=split_lines,
    )

    assert set(features) == set(GROUNDING_KEYS)
    assert len(GROUNDING_KEYS) == 8
    assert all(math.isfinite(value) for value in features.values())
    assert all(key.startswith("m6.") for key in features)


def test_matrix_shape_is_sentences_by_chunks() -> None:
    entail, contra = score_matrix([SUPPORTED, UNSUPPORTED], [CHUNK_A, CHUNK_B], make_nli())

    assert entail.shape == (2, 2)
    assert contra.shape == (2, 2)
    assert entail[0, 0] == pytest.approx(0.90)
    assert contra[1, 1] == pytest.approx(0.60)


def test_one_nli_call_per_case() -> None:
    """Матрица считается одним батчем: покейсовые вызовы — это стоимость ветки."""
    nli = make_nli()

    compute_grounding(
        f"{SUPPORTED}\n{SUPPORTED_2}\n{UNSUPPORTED}",
        [CHUNK_A, CHUNK_B],
        nli,
        sentence_splitter=split_lines,
    )

    assert nli.calls == 1


# --------------------------------------------------------------------------- #
# min_entail — целевая фича
# --------------------------------------------------------------------------- #


def test_min_entail_reacts_to_a_single_unsupported_sentence() -> None:
    """Слабейшее звено: одно неподкреплённое предложение среди подкреплённых."""
    chunks = [CHUNK_A, CHUNK_B]
    all_supported = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}", chunks, make_nli(), sentence_splitter=split_lines
    )
    one_bad = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}\n{UNSUPPORTED}",
        chunks,
        make_nli(),
        sentence_splitter=split_lines,
    )

    assert one_bad["m6.min_entail"] < all_supported["m6.min_entail"]
    assert one_bad["m6.min_entail"] == pytest.approx(0.08)
    # среднее размывает сигнал сильнее, чем минимум, — ради этого фича и введена
    mean_drop = all_supported["m6.mean_entail"] - one_bad["m6.mean_entail"]
    min_drop = all_supported["m6.min_entail"] - one_bad["m6.min_entail"]
    assert min_drop > mean_drop


def test_min_entail_is_monotone_in_the_weakest_sentence() -> None:
    previous = -1.0
    for entail in (0.05, 0.3, 0.6, 0.95):
        nli = make_nli(**{f"{UNSUPPORTED}__a": (entail, 0.1)})
        features = grounding_features(
            f"{SUPPORTED}\n{UNSUPPORTED}", [CHUNK_A, CHUNK_B], nli, sentence_splitter=split_lines
        )
        assert features["m6.min_entail"] > previous
        previous = features["m6.min_entail"]


def test_max_entail_ignores_the_weak_sentence() -> None:
    features = grounding_features(
        f"{SUPPORTED}\n{UNSUPPORTED}", [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )

    assert features["m6.max_entail"] == pytest.approx(0.90)


def test_frac_unsupported_counts_sentences_below_threshold() -> None:
    features = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}\n{UNSUPPORTED}",
        [CHUNK_A, CHUNK_B],
        make_nli(),
        sentence_splitter=split_lines,
        entail_threshold=0.5,
    )

    assert features["m6.frac_unsupported"] == pytest.approx(1 / 3)
    assert features["m6.n_sentences"] == pytest.approx(3.0)


def test_contra_features_take_the_worst_chunk() -> None:
    features = grounding_features(
        UNSUPPORTED, [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )

    assert features["m6.max_contra"] == pytest.approx(0.60)
    assert features["m6.mean_contra"] == pytest.approx(0.60)


# --------------------------------------------------------------------------- #
# chunk_spread — детектор смешения источников
# --------------------------------------------------------------------------- #


def test_chunk_spread_grows_when_sentences_lean_on_different_chunks() -> None:
    one_source = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED}", [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )
    two_sources = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}", [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )

    assert one_source["m6.chunk_spread"] == pytest.approx(1.0)
    assert two_sources["m6.chunk_spread"] == pytest.approx(2.0)


def test_source_chunk_ids_follow_argmax_entail() -> None:
    result = compute_grounding(
        f"{SUPPORTED}\n{SUPPORTED_2}",
        [CHUNK_A, CHUNK_B],
        make_nli(),
        sentence_splitter=split_lines,
    )

    assert result.source_chunk_ids == {0, 1}
    assert result.features["m6.chunk_spread"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Границы
# --------------------------------------------------------------------------- #


def test_single_chunk_case_is_valid() -> None:
    features = grounding_features(
        SUPPORTED, [CHUNK_A], make_nli(), sentence_splitter=split_lines
    )

    assert features["m6.chunk_spread"] == pytest.approx(1.0)
    assert features["m6.n_sentences"] == pytest.approx(1.0)


def test_empty_chunks_raise_instead_of_producing_a_perfect_case() -> None:
    with pytest.raises(ValueError, match="no context chunks"):
        grounding_features(SUPPORTED, [], make_nli(), sentence_splitter=split_lines)


def test_blank_answer_still_produces_one_hypothesis() -> None:
    features = grounding_features(
        "   ", [CHUNK_A], make_nli(), sentence_splitter=lambda text: [text]
    )

    assert features["m6.n_sentences"] == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in features.values())


def test_pairs_are_built_premise_chunk_hypothesis_sentence() -> None:
    """Направление пары — часть определения метода, перепутать его нечем поймать."""
    seen: list[tuple[str, str]] = []

    class Recorder:
        def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
            seen.extend(pairs)
            return [{"entail": 0.5, "contra": 0.1} for _ in pairs]

    grounding_features(SUPPORTED, [CHUNK_A, CHUNK_B], Recorder(), sentence_splitter=split_lines)

    assert seen == [(CHUNK_A, SUPPORTED), (CHUNK_B, SUPPORTED)]
