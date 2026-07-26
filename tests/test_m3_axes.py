"""Одноосевой путь Метода 3: сборка промптов, парсинг MARKER, вероятности вердикта."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rag_reliability.methods.m3.axes import (
    AXES,
    AXIS_FAITHFULNESS,
    AXIS_RELEVANCE,
    axis_anchor,
    axis_pass_prob,
    build_axis_prompt,
    build_axis_system,
    extract_axis_verdict,
    load_axis_prompt,
    parse_axis_verdict,
    parse_marker,
)
from rag_reliability.methods.m3.logprobs import extract_verdict_probs
from rag_reliability.schema import RagSample

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "configs" / "prompts"

# Файлы чужих задач: A4 (logprobs/parsing/judge_client) и D2 (gepa*).
FOREIGN_FILES = (
    "src/rag_reliability/methods/m3/logprobs.py",
    "src/rag_reliability/methods/m3/parsing.py",
    "src/rag_reliability/methods/m3/judge_client.py",
    "src/rag_reliability/methods/m3/gepa.py",
    "src/rag_reliability/methods/m3/gepa_report.py",
)


def _sample() -> RagSample:
    return RagSample(
        id="s1",
        question="Как отключить автоплатёж?",
        context="[CHUNK 1] Удаление автоплатежа: раздел «Мои платежи».",
        answer="Откройте «Мои платежи» и удалите автоплатёж.",
        faithfulness=1,
        relevance=1,
        marker="none",
    )


def _verdict_tokens(axis: str, pass_logprob: float, fail_logprob: float) -> list[dict]:
    """Поток токенов формата ANALYSIS/MARKER/<AXIS>, как его вернёт судья."""
    anchor = axis_anchor(axis)
    return [
        {"token": "ANALYSIS", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {"token": " разбор", "logprob": -0.1, "top": {}},
        {"token": "\nMARKER", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {"token": " reason", "logprob": -0.1, "top": {}},
        {"token": "_off", "logprob": -0.1, "top": {}},
        {"token": "_topic", "logprob": -0.1, "top": {}},
        {"token": "_answer", "logprob": -0.1, "top": {}},
        {"token": f"\n{anchor}", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {
            "token": " PASS",
            "logprob": pass_logprob,
            "top": {" PASS": pass_logprob, " FAIL": fail_logprob},
        },
    ]


@pytest.mark.parametrize("axis", AXES)
def test_build_axis_prompt_returns_system_and_user(axis: str) -> None:
    system, user = build_axis_prompt(_sample(), axis, prompts_dir=PROMPTS_DIR)

    assert axis_anchor(axis) in system
    assert _sample().answer in user


def test_axes_do_not_see_each_other_criteria() -> None:
    faith_system, _ = build_axis_prompt(_sample(), AXIS_FAITHFULNESS, prompts_dir=PROMPTS_DIR)
    rel_system, _ = build_axis_prompt(_sample(), AXIS_RELEVANCE, prompts_dir=PROMPTS_DIR)

    assert axis_anchor(AXIS_RELEVANCE) not in faith_system
    assert axis_anchor(AXIS_FAITHFULNESS) not in rel_system


def test_unknown_axis_and_mode_fail_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown axis"):
        load_axis_prompt("truthiness", prompts_dir=PROMPTS_DIR)
    with pytest.raises(ValueError, match="Unknown Method 3 mode"):
        build_axis_system(AXIS_RELEVANCE, mode="vibes", prompts_dir=PROMPTS_DIR)


def test_few_shot_reuses_the_joint_example_file() -> None:
    """configs/few_shot.yaml размечен вручную по обеим осям — переиспользуем его."""
    examples = [
        {
            "q": "Как отключить автоплатёж?",
            "ctx": "[CHUNK 1] Удаление автоплатежа.",
            "a": "Откройте «Мои платежи».",
            "analysis": "Шаги совпадают с чанком.",
            "faith": "PASS",
            "rel": "FAIL",
        }
    ]

    faith = build_axis_system(
        AXIS_FAITHFULNESS, mode="few_shot", examples=examples, prompts_dir=PROMPTS_DIR
    )
    rel = build_axis_system(
        AXIS_RELEVANCE, mode="few_shot", examples=examples, prompts_dir=PROMPTS_DIR
    )

    assert "Пример 1." in faith
    assert "FAITHFULNESS: PASS" in faith
    assert "MARKER: none" in faith
    assert "RELEVANCE: FAIL" in rel
    assert "[CTX]" not in rel  # ось relevance не получает чанки и в примерах


def test_few_shot_without_examples_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no examples"):
        build_axis_system(AXIS_RELEVANCE, mode="few_shot", prompts_dir=PROMPTS_DIR)


def test_gepa_prompt_must_keep_the_verdict_anchor(tmp_path: Path) -> None:
    prompt_file = tmp_path / "evolved.txt"
    prompt_file.write_text("Оцени ответ и напиши вывод.", encoding="utf-8")

    with pytest.raises(ValueError, match="FAITHFULNESS"):
        build_axis_system(
            AXIS_FAITHFULNESS, mode="gepa", prompt_file=prompt_file, prompts_dir=PROMPTS_DIR
        )

    prompt_file.write_text("Оцени ответ.\nFAITHFULNESS: PASS или FAIL", encoding="utf-8")
    assert "FAITHFULNESS" in build_axis_system(
        AXIS_FAITHFULNESS, mode="gepa", prompt_file=prompt_file, prompts_dir=PROMPTS_DIR
    )


def test_braces_in_the_answer_do_not_break_rendering() -> None:
    sample = _sample().model_copy(update={"answer": 'Ответ: {"limit": 100} — см. {раздел}'})

    _, user = build_axis_prompt(sample, AXIS_RELEVANCE, prompts_dir=PROMPTS_DIR)

    assert '{"limit": 100}' in user


def test_marker_line_parses_and_does_not_hide_the_verdict() -> None:
    text = "ANALYSIS: ответ уводит в статью\nMARKER: reason_wrong_navigation\nRELEVANCE: FAIL"

    assert parse_marker(text) == "reason_wrong_navigation"
    assert parse_axis_verdict(text, AXIS_RELEVANCE) == 0
    assert parse_axis_verdict(text, AXIS_FAITHFULNESS) is None


def test_marker_none_and_missing_marker_line() -> None:
    assert parse_marker("MARKER: none\nRELEVANCE: PASS") == "none"
    assert parse_marker("RELEVANCE: PASS") is None


def test_verdict_taken_from_the_last_mention() -> None:
    """ANALYSIS может процитировать ось — вердикт всегда последний."""
    text = "ANALYSIS: соблазн поставить RELEVANCE: PASS\nMARKER: none\nRELEVANCE: FAIL"

    assert parse_axis_verdict(text, AXIS_RELEVANCE) == 0


@pytest.mark.parametrize("axis", AXES)
def test_axis_pass_prob_is_monotone_in_the_verdict_logprob(axis: str) -> None:
    confident = axis_pass_prob(_verdict_tokens(axis, -0.05, -3.0), axis)
    unsure = axis_pass_prob(_verdict_tokens(axis, -0.7, -0.7), axis)
    doubtful = axis_pass_prob(_verdict_tokens(axis, -3.0, -0.05), axis)

    assert confident is not None and unsure is not None and doubtful is not None
    assert doubtful < unsure < confident
    assert unsure == pytest.approx(0.5)


def test_axis_pass_prob_matches_the_two_axis_extractor() -> None:
    """Одноосевой путь обязан совпадать с двухосевым на общем потоке токенов."""
    tokens = _verdict_tokens(AXIS_FAITHFULNESS, -0.1, -2.4) + [
        {"token": "\nRELEVANCE", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {"token": " FAIL", "logprob": -0.2, "top": {" FAIL": -0.2, " PASS": -1.7}},
    ]

    joint = extract_verdict_probs(tokens)

    assert joint is not None
    assert axis_pass_prob(tokens, AXIS_FAITHFULNESS) == pytest.approx(joint[0])
    assert axis_pass_prob(tokens, AXIS_RELEVANCE) == pytest.approx(joint[1])


def test_axis_pass_prob_without_anchor_is_none() -> None:
    assert axis_pass_prob([], AXIS_RELEVANCE) is None
    assert axis_pass_prob(_verdict_tokens(AXIS_FAITHFULNESS, -0.1, -2.0), AXIS_RELEVANCE) is None


def test_extract_axis_verdict_fallback_chain() -> None:
    tokens = _verdict_tokens(AXIS_RELEVANCE, -0.1, -2.4)
    text = "ANALYSIS: ок\nMARKER: none\nRELEVANCE: PASS"

    probability, meta = extract_axis_verdict(text, tokens, AXIS_RELEVANCE)
    assert meta["method"] == "logprobs"
    assert probability > 0.5
    assert meta["marker"] == "none"

    probability, meta = extract_axis_verdict(text, [], AXIS_RELEVANCE)
    assert (probability, meta["method"]) == (0.9, "regex")

    probability, meta = extract_axis_verdict("бессвязный текст", [], AXIS_RELEVANCE)
    assert (probability, meta["method"]) == (0.5, "default")

    _, meta = extract_axis_verdict(text, [], AXIS_RELEVANCE, finish_reason="length")
    assert meta["truncated"] is True


def test_foreign_m3_files_are_untouched() -> None:
    """Владение файлами: C3 не имеет права менять код A4 и D2."""
    base = subprocess.run(
        ["git", "merge-base", "HEAD", "integration"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if base.returncode != 0:
        pytest.skip("no 'integration' ref to diff against")
    changed = subprocess.run(
        ["git", "diff", "--name-only", base.stdout.strip(), "--", *FOREIGN_FILES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert changed.stdout.strip() == ""
