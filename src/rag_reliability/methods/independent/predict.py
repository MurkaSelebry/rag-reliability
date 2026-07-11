"""Independent rule-based evaluator for RAG responses."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from rag_reliability.schema import Prediction, RagSample


# Common Russian function words that should not dominate lexical overlap.
RUSSIAN_STOPWORDS: set[str] = {
    "а",
    "без",
    "более",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "быть",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всего",
    "вы",
    "где",
    "да",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "есть",
    "ещё",
    "же",
    "за",
    "и",
    "из",
    "или",
    "их",
    "к",
    "как",
    "когда",
    "который",
    "ли",
    "на",
    "над",
    "не",
    "но",
    "о",
    "об",
    "он",
    "она",
    "они",
    "от",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "также",
    "то",
    "у",
    "уже",
    "что",
    "чтобы",
    "это",
    "этот",
    "я",
}


def tokenize(text: str) -> set[str]:
    """Convert text into normalized content-word tokens."""
    if not text:
        return set()

    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())

    return {
        token
        for token in tokens
        if len(token) > 2 and token not in RUSSIAN_STOPWORDS
    }


def lexical_coverage(source: str, target: str) -> float:
    """
    Calculate the fraction of target tokens found in the source.

    The target is normally the chatbot answer.
    """
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)

    if not target_tokens:
        return 0.0

    return len(source_tokens.intersection(target_tokens)) / len(target_tokens)


def numerical_consistency(context: str, answer: str) -> float:
    """
    Check whether numbers appearing in the answer are present in the context.

    Returns 1.0 when the answer contains no numbers or all its numbers are
    supported by the context.
    """
    answer_numbers = set(
        re.findall(r"\d+(?:[.,]\d+)?", answer)
    )

    if not answer_numbers:
        return 1.0

    context_numbers = set(
        re.findall(r"\d+(?:[.,]\d+)?", context)
    )

    supported = answer_numbers.intersection(context_numbers)

    return len(supported) / len(answer_numbers)


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    """Return True when any phrase occurs in the text."""
    normalized = text.lower()
    return any(pattern.lower() in normalized for pattern in patterns)


def predict_independent(
    sample: RagSample,
    *,
    faithfulness_threshold: float = 0.20,
    relevance_threshold: float = 0.10,
) -> Prediction:
    """
    Predict faithfulness and relevance independently.

    This first prototype is intentionally transparent and rule-based.
    It combines lexical coverage, numerical consistency, and several
    conservative error rules.
    """
    context_coverage = lexical_coverage(
        source=sample.context,
        target=sample.answer,
    )

    question_coverage = lexical_coverage(
        source=sample.question,
        target=sample.answer,
    )

    number_support = numerical_consistency(
        context=sample.context,
        answer=sample.answer,
    )

    unsupported_verification_phrases = (
        "я проверил",
        "мы проверили",
        "проверено",
        "я рассчитал",
        "мы рассчитали",
        "я выполнил",
        "мы выполнили",
    )

    redirect_only_phrases = (
        "посмотрите в статье",
        "перейдите по ссылке",
        "информация доступна по ссылке",
        "обратитесь к оператору",
    )

    reveals_ai_phrases = (
        "я искусственный интеллект",
        "я являюсь ии",
        "как языковая модель",
        "я чат-бот",
    )

    false_verification = contains_any(
        sample.answer,
        unsupported_verification_phrases,
    )

    redirect_only = contains_any(
        sample.answer,
        redirect_only_phrases,
    )

    reveals_ai = contains_any(
        sample.answer,
        reveals_ai_phrases,
    )

    faithfulness_score = (
        0.70 * context_coverage
        + 0.30 * number_support
    )

    relevance_score = question_coverage

    faithfulness = int(
        faithfulness_score >= faithfulness_threshold
        and not false_verification
    )

    relevance = int(
        relevance_score >= relevance_threshold
        and not redirect_only
        and not reveals_ai
    )

    diagnostics = {
        "method": "independent_rule_based",
        "context_coverage": round(context_coverage, 4),
        "question_coverage": round(question_coverage, 4),
        "number_support": round(number_support, 4),
        "faithfulness_score": round(faithfulness_score, 4),
        "relevance_score": round(relevance_score, 4),
        "false_verification": false_verification,
        "redirect_only": redirect_only,
        "reveals_ai": reveals_ai,
        "faithfulness_threshold": faithfulness_threshold,
        "relevance_threshold": relevance_threshold,
    }

    return Prediction(
        id=sample.id,
        faithfulness_pred=faithfulness,
        relevance_pred=relevance,
        raw_output=json.dumps(
            diagnostics,
            ensure_ascii=False,
        ),
        invalid_output=False,
    )


def predict_many(
    samples: list[RagSample],
    *,
    faithfulness_threshold: float = 0.20,
    relevance_threshold: float = 0.10,
) -> list[Prediction]:
    """Run the independent evaluator on multiple samples."""
    return [
        predict_independent(
            sample,
            faithfulness_threshold=faithfulness_threshold,
            relevance_threshold=relevance_threshold,
        )
        for sample in samples
    ]