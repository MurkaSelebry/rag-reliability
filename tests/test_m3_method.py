"""Tests for Method 3 prompt and parser integration."""

from pathlib import Path

import pytest
import yaml

from rag_reliability.methods.m3 import (
    SEED_INSTRUCTION,
    build_few_shot_system,
    build_system_prompt,
    build_user_prompt,
    parse_m3_prediction,
)
from rag_reliability.schema import RagSample


def test_seed_instruction_keeps_axis_independence_rule() -> None:
    assert "оси независимы" in SEED_INSTRUCTION
    assert "ТОЛЬКО против [CTX]" in SEED_INSTRUCTION
    assert SEED_INSTRUCTION.index("оси независимы") < SEED_INSTRUCTION.index(
        "FAITHFULNESS: PASS или FAIL"
    )


def test_build_user_prompt_uses_shared_sample_schema() -> None:
    sample = RagSample(
        id="s1",
        question="Как подключить услугу?",
        context="Услуга подключается в личном кабинете.",
        answer="Откройте личный кабинет.",
        faithfulness=1,
        relevance=1,
        marker="none",
    )

    prompt = build_user_prompt(sample)

    assert "[Q]" in prompt
    assert "Как подключить услугу?" in prompt
    assert "[CTX]" in prompt
    assert "Услуга подключается" in prompt
    assert "[A]" in prompt


def test_parse_m3_prediction_pass_fail_contract() -> None:
    prediction = parse_m3_prediction(
        "Анализ...\nFAITHFULNESS: PASS\nRELEVANCE: FAIL",
        "s1",
    )

    assert prediction.faithfulness_pred == 1
    assert prediction.relevance_pred == 0
    assert prediction.invalid_output is False


def test_parse_m3_prediction_conservative_fallback() -> None:
    prediction = parse_m3_prediction("no verdict", "s1")

    assert prediction.faithfulness_pred == 0
    assert prediction.relevance_pred == 0
    assert prediction.invalid_output is True


def test_build_few_shot_system_appends_examples() -> None:
    system = build_few_shot_system(
        [
            {
                "q": "q",
                "ctx": "ctx",
                "a": "a",
                "analysis": "analysis",
                "faith": "PASS",
                "rel": "FAIL",
            }
        ]
    )

    assert system.startswith(SEED_INSTRUCTION)
    assert "Пример 1." in system
    assert "FAITHFULNESS: PASS" in system
    assert "RELEVANCE: FAIL" in system


def test_build_system_prompt_loads_few_shot_yaml(tmp_path: Path) -> None:
    examples_path = tmp_path / "few_shot.yaml"
    examples_path.write_text(
        yaml.safe_dump(
            {
                "examples": [
                    {
                        "q": "q",
                        "ctx": "ctx",
                        "a": "a",
                        "analysis": "analysis",
                        "faith": "PASS",
                        "rel": "PASS",
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    system = build_system_prompt("few_shot", examples_path=examples_path)

    assert "Пример 1." in system
    assert "FAITHFULNESS: PASS" in system


def test_build_system_prompt_reads_gepa_prompt_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("custom evolved prompt", encoding="utf-8")

    assert build_system_prompt("gepa", prompt_file=prompt_file) == "custom evolved prompt"


def test_gepa_prompt_missing_raises_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_gepa"):
        build_system_prompt("gepa", prompt_file=str(tmp_path / "absent.txt"))
