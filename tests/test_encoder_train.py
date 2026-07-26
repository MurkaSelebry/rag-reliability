"""OOF-раскладка энкодера и контроль схлопывания — на подставном тренере.

Ни torch, ни transformers, ни единого скачанного веса: обучение фолда приходит
параметром, а тест подставляет функцию, которая записывает, что ей дали.
"""

from __future__ import annotations

import logging

import pytest

from rag_reliability.methods.encoder.train import (
    FoldOutcome,
    FoldRequest,
    OofResult,
    TrainConfig,
    compute_pos_weight,
    decisions_from_logits,
    is_collapsed,
    train_oof,
    train_oof_detailed,
)
from rag_reliability.methods.surface.oof import Folds
from rag_reliability.schema import RagSample


def make_sample(sample_id: str, faithfulness: int = 1, relevance: int = 1) -> RagSample:
    return RagSample(
        id=sample_id,
        question=f"Клиент: вопрос {sample_id}",
        context=f"[CHUNK 1]\nконтекст {sample_id}",
        answer=f"ответ {sample_id}",
        faithfulness=faithfulness,
        relevance=relevance,
        marker="none" if faithfulness and relevance else "unknown",
    )


def make_corpus(n: int = 20) -> list[RagSample]:
    # Каждый пятый ненадёжен: без обоих классов диагностика схлопывания слепа.
    return [make_sample(f"case_{index:03d}", relevance=int(index % 5 != 0)) for index in range(n)]


def make_folds(samples: list[RagSample], *, n_folds: int = 5, n_repeats: int = 2) -> Folds:
    assignment = {
        sample.id: [(index + repeat) % n_folds for repeat in range(n_repeats)]
        for index, sample in enumerate(samples)
    }
    return Folds(
        assignment=assignment,
        n_folds=n_folds,
        n_repeats=n_repeats,
        corpus_n=len(samples),
        sha256="0" * 64,
    )


class RecordingTrainer:
    """Подставной тренер: помнит, что видел, и честно зовёт хук после эпохи."""

    def __init__(self, *, logit: float | None = None, epochs_to_report: int | None = None) -> None:
        self.seen: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
        self.logit = logit
        self.epochs_to_report = epochs_to_report

    def __call__(self, request: FoldRequest) -> FoldOutcome:
        train_ids = tuple(sample.id for sample in request.train_samples)
        test_ids = tuple(sample.id for sample in request.test_samples)
        self.seen.append((request.fold, train_ids, test_ids))
        # Логит по умолчанию зависит от кейса, чтобы прогон не выглядел схлопнувшимся.
        logits = [
            self.logit if self.logit is not None else (1.0 if index % 2 else -1.0)
            for index in range(len(test_ids))
        ]
        n_epochs = (
            self.epochs_to_report
            if self.epochs_to_report is not None
            else request.config.n_epochs
        )
        for epoch in range(1, n_epochs + 1):
            request.on_epoch_end(epoch, logits)
        return FoldOutcome(logits=tuple(logits), checkpoint=f"ckpt/fold{request.fold}.pt")


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #


def test_config_rejects_unknown_class_weighting() -> None:
    with pytest.raises(ValueError, match="pos_weight_mode"):
        TrainConfig(pos_weight_mode="inverse")


def test_config_rejects_warmup_ratio_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="warmup_ratio"):
        TrainConfig(warmup_ratio=1.0)


def test_config_rounds_fractional_epochs_up_for_diagnostics() -> None:
    assert TrainConfig(epochs=2.5).n_epochs == 3
    assert TrainConfig(epochs=3.0).n_epochs == 3


def test_pos_weight_balances_the_positive_majority() -> None:
    assert compute_pos_weight([1, 1, 1, 0], mode="balanced") == 1 / 3
    assert compute_pos_weight([1, 1, 1, 0], mode="none") == 1.0


# --------------------------------------------------------------------------- #
# Изоляция фолдов
# --------------------------------------------------------------------------- #


def test_train_oof_never_shows_a_fold_its_own_cases() -> None:
    samples = make_corpus()
    folds = make_folds(samples)
    trainer = RecordingTrainer()

    train_oof(samples, folds, TrainConfig(), repeat=0, train_fold=trainer)

    assert len(trainer.seen) == folds.n_folds
    for fold, train_ids, test_ids in trainer.seen:
        assert set(train_ids) & set(test_ids) == set()
        expected = {
            sample.id for sample in samples if folds.assignment[sample.id][0] == fold
        }
        assert set(test_ids) == expected
        assert set(train_ids) | set(test_ids) == {sample.id for sample in samples}


def test_train_oof_uses_the_requested_repeat() -> None:
    samples = make_corpus()
    folds = make_folds(samples)
    trainer = RecordingTrainer()

    train_oof(samples, folds, TrainConfig(), repeat=1, train_fold=trainer)

    for fold, _train_ids, test_ids in trainer.seen:
        expected = {sample.id for sample in samples if folds.assignment[sample.id][1] == fold}
        assert set(test_ids) == expected


def test_train_oof_scores_every_case_exactly_once() -> None:
    samples = make_corpus()

    logits = train_oof(samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer())

    assert set(logits) == {sample.id for sample in samples}


def test_train_oof_rejects_a_repeat_folds_json_does_not_have() -> None:
    samples = make_corpus()
    folds = make_folds(samples, n_repeats=1)

    with pytest.raises(ValueError, match="repeat must be in"):
        train_oof(samples, folds, TrainConfig(), repeat=3, train_fold=RecordingTrainer())


def test_train_oof_rejects_cases_without_a_fold_assignment() -> None:
    samples = make_corpus()
    folds = make_folds(samples)
    orphan = make_sample("case_999")

    with pytest.raises(ValueError, match="no fold assignment"):
        train_oof([*samples, orphan], folds, TrainConfig(), train_fold=RecordingTrainer())


def test_train_oof_rejects_a_trainer_that_returns_the_wrong_number_of_logits() -> None:
    samples = make_corpus()

    def short_trainer(request: FoldRequest) -> FoldOutcome:
        for epoch in range(1, request.config.n_epochs + 1):
            request.on_epoch_end(epoch, [0.5] * len(request.test_samples))
        return FoldOutcome(logits=(0.5,))

    with pytest.raises(ValueError, match="logit"):
        train_oof(samples, make_folds(samples), TrainConfig(), train_fold=short_trainer)


# --------------------------------------------------------------------------- #
# Контроль схлопывания
# --------------------------------------------------------------------------- #


def test_diagnostics_run_after_every_epoch() -> None:
    samples = make_corpus()
    folds = make_folds(samples)
    config = TrainConfig(epochs=3)

    result = train_oof_detailed(samples, folds, config, train_fold=RecordingTrainer())

    assert len(result.epochs) == folds.n_folds * config.n_epochs
    assert sorted({log.epoch for log in result.epochs}) == [1, 2, 3]
    for log in result.epochs:
        assert 0.0 < log.const_share <= 1.0


def test_a_trainer_that_skips_the_epoch_hook_fails_the_run() -> None:
    """Диагностика обязательна: молча пропущенная эпоха — это снова прогон 1024."""
    samples = make_corpus()

    with pytest.raises(ValueError, match="degenerate_rate is mandatory"):
        train_oof(
            samples,
            make_folds(samples),
            TrainConfig(epochs=3),
            train_fold=RecordingTrainer(epochs_to_report=1),
        )


def test_epoch_diagnostics_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    samples = make_corpus()

    with caplog.at_level(logging.INFO, logger="rag_reliability.methods.encoder.train"):
        train_oof(samples, make_folds(samples), TrainConfig(epochs=1), train_fold=RecordingTrainer())

    assert any("const_share" in record.getMessage() for record in caplog.records)


def test_a_run_that_predicts_one_class_everywhere_is_marked_collapsed() -> None:
    samples = make_corpus()

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer(logit=5.0)
    )

    assert result.collapsed is True
    assert result.diagnostics()["collapsed"] is True
    assert result.diagnostics()["const_share"] == 1.0
    assert all(log.is_degenerate for log in result.epochs)


def test_a_run_with_a_mixed_output_is_not_collapsed() -> None:
    samples = make_corpus()

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer()
    )

    assert result.collapsed is False
    assert result.diagnostics()["const_share"] < 0.98


def test_collapse_is_detected_at_the_threshold_from_metrics() -> None:
    """Порог 0.98 не дублируется в пакете — он приходит из metrics.degenerate_rate."""
    below = {f"case_{index}": (5.0 if index >= 2 else -5.0) for index in range(100)}
    above = {f"case_{index}": (5.0 if index >= 1 else -5.0) for index in range(100)}

    assert is_collapsed(below) is False  # const_share = 0.98, порог строгий
    assert is_collapsed(above) is True  # const_share = 0.99


def test_decisions_from_logits_binarize_at_zero() -> None:
    decisions = decisions_from_logits({"a": 0.1, "b": -0.1, "c": 0.0})

    assert [prediction.reliable_pred for prediction in decisions] == [1, 0, 1]


def test_decisions_from_an_empty_run_are_an_error_not_a_silent_pass() -> None:
    with pytest.raises(ValueError, match="empty"):
        decisions_from_logits({})


def test_diagnostics_record_the_single_repeat_explicitly() -> None:
    """n_repeats: 1 обязано быть видно при сравнении с методами, у которых 5."""
    samples = make_corpus()

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), repeat=1, train_fold=RecordingTrainer()
    )

    diagnostics = result.diagnostics()
    assert diagnostics["n_repeats"] == 1
    assert diagnostics["repeat"] == 1
    assert diagnostics["n_scored"] == len(samples)


def test_checkpoints_are_recorded_per_fold() -> None:
    samples = make_corpus()
    folds = make_folds(samples)

    result = train_oof_detailed(samples, folds, TrainConfig(), train_fold=RecordingTrainer())

    assert sorted(result.checkpoints) == list(range(folds.n_folds))


def test_oof_result_of_an_empty_run_cannot_claim_a_verdict() -> None:
    with pytest.raises(ValueError, match="empty"):
        OofResult(logits={}).collapsed  # noqa: B018
