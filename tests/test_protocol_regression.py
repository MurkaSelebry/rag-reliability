"""Главный тест B1: новый контур изменил протокол, а не арифметику.

В режиме ``--legacy-holdout`` ``evaluate_cv`` воспроизводит опубликованные числа
из ``predictions/alfa/**/report_{val,test}.json``. Если бы он их не
воспроизводил, разницу между старыми и новыми числами нельзя было бы объяснить
сменой протокола: она могла бы быть и ошибкой в новой реализации метрики.

Здесь же проверяется канонизация корпуса: id из ``data/alfa.jsonl`` обязаны
покрывать все 446 строк исторических сплитов. Реконструкция id с ``strip()``
даёт 422 из 446 и все десять чисел ломает.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
CORPUS = REPO / "data" / "alfa.jsonl"


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "evaluate_cv_regression", REPO / "scripts" / "evaluate_cv.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_cv_regression"] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


@dataclass(frozen=True)
class PublishedRun:
    """Прогон, чьи числа опубликованы под старым протоколом."""

    name: str
    directory: Path
    #: Порог из report_*.json. None — скор константный, и любой порог ниже
    #: константы даёт одни и те же предсказания (majority): сравнивать нечего.
    compare_thresholds: bool = True


PUBLISHED = [
    PublishedRun("majority", REPO / "predictions/alfa/baselines/majority", False),
    PublishedRun("surface", REPO / "predictions/alfa/baselines/surface"),
    PublishedRun("surface_e5", REPO / "predictions/alfa/baselines/surface_e5"),
    PublishedRun("m3_zero_shot", REPO / "predictions/alfa/m3/zero_shot"),
    PublishedRun("m3_few_shot", REPO / "predictions/alfa/m3/few_shot"),
]


def _run_legacy(run: PublishedRun, tmp_path: Path) -> dict:
    output = tmp_path / f"{run.name}.json"
    assert (
        cli.main(
            [
                "--data", str(CORPUS),
                "--scores", str(run.directory / "scores.jsonl"),
                "--legacy-holdout",
                "--legacy-val", str(run.directory / "val.jsonl"),
                "--legacy-test", str(run.directory / "test.jsonl"),
                "--output", str(output),
            ]
        )
        == 0
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _published(run: PublishedRun, split: str) -> dict:
    return json.loads((run.directory / f"report_{split}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("run", PUBLISHED, ids=lambda run: run.name)
def test_legacy_mode_reproduces_published_numbers(run: PublishedRun, tmp_path: Path) -> None:
    """Оба опубликованных числа прогона — до 1e-6."""
    reproduced = _run_legacy(run, tmp_path)

    for split in ("val", "test"):
        published = _published(run, split)
        assert reproduced[split]["n"] == published["n"]
        assert reproduced[split]["f1_macro_reliable"] == pytest.approx(
            published["f1_macro_reliable"], abs=1e-6
        ), f"{run.name}/{split}"


@pytest.mark.parametrize("run", PUBLISHED, ids=lambda run: run.name)
def test_legacy_mode_reproduces_published_axes_and_thresholds(
    run: PublishedRun, tmp_path: Path
) -> None:
    """Оси и порог — та же арифметика, что стояла за первичным числом."""
    reproduced = _run_legacy(run, tmp_path)

    for split in ("val", "test"):
        published = _published(run, split)
        for field in ("f1_macro_faith", "f1_macro_rel"):
            assert reproduced[split][field] == pytest.approx(
                published[field], abs=1e-6
            ), f"{run.name}/{split}/{field}"

    if run.compare_thresholds:
        published = _published(run, "val")
        assert reproduced["t_faith"] == pytest.approx(published["t_faith"], abs=1e-9)
        assert reproduced["t_rel"] == pytest.approx(published["t_rel"], abs=1e-9)


def test_all_ten_published_numbers_are_covered() -> None:
    """Карточка требует все десять; список не должен молча усохнуть."""
    numbers = [
        (run.name, split, _published(run, split)["f1_macro_reliable"])
        for run in PUBLISHED
        for split in ("val", "test")
    ]
    assert len(numbers) == 10
    assert len({value for _, _, value in numbers}) == 10


def test_canonical_corpus_covers_every_historical_split_id() -> None:
    """Канонический корпус обязан покрывать исторические сплиты целиком."""
    corpus_ids = {json.loads(line)["id"] for line in CORPUS.read_text(encoding="utf-8").splitlines()}
    assert len(corpus_ids) == 2233

    for run in PUBLISHED:
        split_ids = [
            sample_id
            for split in ("val", "test")
            for sample_id in cli.read_ids(run.directory / f"{split}.jsonl")
        ]
        assert len(split_ids) == 446
        missing = [sample_id for sample_id in split_ids if sample_id not in corpus_ids]
        assert not missing, f"{run.name}: {len(missing)} id(s) missing, e.g. {missing[:3]}"


def test_legacy_holdout_requires_both_split_files(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--data", str(CORPUS),
                "--scores", str(PUBLISHED[0].directory / "scores.jsonl"),
                "--legacy-holdout",
                "--legacy-val", str(PUBLISHED[0].directory / "val.jsonl"),
                "--output", str(tmp_path / "out.json"),
            ]
        )
