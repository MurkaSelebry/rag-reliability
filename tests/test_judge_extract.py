"""4 сценария извлечения вероятностей вердикта (docs/03, docs/05 этап 0) + BPE-склейка."""
import math

import pytest

from src.common.llm_client import (JudgeClient, _pass_prob, _verdict_positions,
                                   extract_verdict_probs)


def tok(token, logprob=-0.1, top=None):
    return {"token": token, "logprob": logprob, "top": top or {}}


def test_whole_tokens_both_axes():
    """Сценарий 1: обе позиции — целые токены PASS/FAIL с top-парой."""
    tokens = [tok("FAITHFULNESS"), tok(":"),
              tok(" PASS", top={" PASS": -0.1, " FAIL": -2.4}),
              tok("\n"), tok("RELEVANCE"), tok(":"),
              tok(" FAIL", top={" FAIL": -0.2, " PASS": -1.7})]
    res = extract_verdict_probs(tokens)
    assert res is not None
    p_f, p_r = res
    assert p_f == pytest.approx(1 / (1 + math.exp(-2.3)), abs=1e-6)   # 0.909
    assert p_r == pytest.approx(1 - 1 / (1 + math.exp(-1.5)), abs=1e-6)  # 0.182


def test_bpe_subtokens_merged():
    """Сценарий 2: PASS разбит на 'PA'+'SS' — позиция ищется склейкой,
    вероятность — по префиксам в top первой сабтокен-позиции."""
    tokens = [tok("FAITH"), tok("FULNESS"), tok(":"),
              tok(" PA", top={" PA": -0.2, " FA": -1.8}), tok("SS"),
              tok("\nRELEVANCE"), tok(":"),
              tok(" FA", top={" FA": -0.3, " PA": -1.4}), tok("IL")]
    positions = _verdict_positions(tokens)
    assert positions == [3, 7]
    p_f, p_r = extract_verdict_probs(tokens)
    assert p_f == pytest.approx(1 / (1 + math.exp(-1.6)), abs=1e-6)
    assert p_r == pytest.approx(1 - 1 / (1 + math.exp(-1.1)), abs=1e-6)


def test_label_tokens_not_matched_as_verdicts():
    """'FAITHFULNESS' из метки не должен считаться вердиктом."""
    tokens = [tok("FA"), tok("ITHFULNESS"), tok(":"), tok(" PASS"), tok(" RELEVANCE"),
              tok(":"), tok(" FAIL")]
    assert _verdict_positions(tokens) == [3, 6]


def test_one_side_visible_sigmoid():
    """В top видна только одна сторона пары — сигмоида по её logprob."""
    p = _pass_prob(tok(" PASS", logprob=-0.1, top={" PASS": -0.1}))
    assert p == pytest.approx(1 / (1 + math.exp(0.1)), abs=1e-6)


def test_regex_fallback(monkeypatch):
    """Сценарий 3: logprobs нет — вердикты из текста, 0.9/0.1."""
    c = JudgeClient.__new__(JudgeClient)
    c.cache_dir = None
    monkeypatch.setattr(JudgeClient, "_chat_judge", lambda self, s, u, mt, case=None: (
        "Анализ...\nFAITHFULNESS: PASS\nRELEVANCE: FAIL", []), raising=False)
    p_f, p_r, meta = c.judge("sys", "usr")
    assert (p_f, p_r) == (0.9, 0.1) and meta["method"] == "regex"


def test_unparseable_gives_half(monkeypatch):
    """Сценарий 4: ни logprobs, ни regex — 0.5/0.5, кейс не теряется."""
    c = JudgeClient.__new__(JudgeClient)
    c.cache_dir = None
    monkeypatch.setattr(JudgeClient, "_chat_judge",
                        lambda self, s, u, mt, case=None: ("бессвязный текст", []), raising=False)
    p_f, p_r, meta = c.judge("sys", "usr")
    assert (p_f, p_r) == (0.5, 0.5) and meta["method"] == "default"


def test_exact_token_preferred_over_prefix():
    """Целый ' PASS' в top: префикс ' PA' не должен добавлять массу."""
    t = tok(" PASS", top={" PASS": -0.1, " PA": -0.5, " FAIL": -2.4})
    p_with_prefix = _pass_prob(t)
    t2 = tok(" PASS", top={" PASS": -0.1, " FAIL": -2.4})
    assert p_with_prefix == pytest.approx(_pass_prob(t2), abs=1e-9)
