"""Feature extraction for Independent Evaluator V2.

The V2 evaluator keeps the original transparent rule-based signals but turns
them into features that can later be consumed by learned classifiers.

Features are intentionally lightweight and deterministic so that they can be
computed without GPU access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from rag_reliability.schema import RagSample


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return " ".join(text.lower().strip().split())


def tokenize(text: str) -> list[str]:
    """Simple Russian/English word tokenizer."""
    return _WORD_RE.findall(normalize_text(text))


def extract_numbers(text: str) -> set[str]:
    """Extract normalized numeric expressions."""
    return {x.replace(",", ".") for x in _NUMBER_RE.findall(text)}


def extract_last_client_turn(dialogue: str) -> str:
    """Extract the latest client/user utterance from a multi-turn dialogue.

    Organizer examples commonly use Russian labels such as "Клиент:".
    English labels are supported as a fallback.
    """
    patterns = (
        r"(?:Клиент|Пользователь|User|Client)\s*:\s*(.+?)(?=(?:Ассистент|Assistant|Клиент|Пользователь|User|Client)\s*:|$)",
    )

    matches: list[str] = []

    for pattern in patterns:
        matches.extend(
            match.strip()
            for match in re.findall(
                pattern,
                dialogue,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match.strip()
        )

    if matches:
        return matches[-1]

    # If the schema contains only the question rather than a full dialogue,
    # simply use the whole field.
    return dialogue.strip()


def coverage(source: str, target: str) -> float:
    """Fraction of target tokens also appearing in source."""
    source_tokens = set(tokenize(source))
    target_tokens = set(tokenize(target))

    if not target_tokens:
        return 0.0

    return len(source_tokens & target_tokens) / len(target_tokens)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-set Jaccard similarity."""
    a = set(tokenize(text_a))
    b = set(tokenize(text_b))

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def numeric_support(context: str, answer: str) -> float:
    """Fraction of answer numbers that are supported by the context."""
    answer_numbers = extract_numbers(answer)

    if not answer_numbers:
        return 1.0

    context_numbers = extract_numbers(context)

    return len(answer_numbers & context_numbers) / len(answer_numbers)


def _contains_any(text: str, phrases: tuple[str, ...]) -> int:
    normalized = normalize_text(text)
    return int(any(phrase in normalized for phrase in phrases))


@dataclass(frozen=True)
class IndependentFeatures:
    """Feature vector used by Independent Evaluator V2."""

    context_coverage: float
    full_question_coverage: float
    latest_question_coverage: float

    context_answer_jaccard: float
    question_answer_jaccard: float
    latest_question_answer_jaccard: float

    number_support: float

    answer_token_count: int
    question_token_count: int
    context_token_count: int

    answer_question_length_ratio: float

    false_verification: int
    redirect_only: int
    reveals_ai_identity: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


_FALSE_VERIFICATION_PHRASES = (
    "я проверил",
    "я проверила",
    "мы проверили",
    "проверил информацию",
    "проверила информацию",
    "i checked",
    "i verified",
)

_REDIRECT_PHRASES = (
    "обратитесь в поддержку",
    "обратитесь к оператору",
    "свяжитесь с поддержкой",
    "позвоните в банк",
    "contact support",
    "contact an operator",
)

_AI_IDENTITY_PHRASES = (
    "я искусственный интеллект",
    "я языковая модель",
    "как ии",
    "как искусственный интеллект",
    "as an ai",
    "language model",
)


def extract_features(sample: RagSample) -> IndependentFeatures:
    """Extract deterministic V2 features from one RAG sample."""

    question = sample.question or ""
    context = sample.context or ""
    answer = sample.answer or ""

    latest_question = extract_last_client_turn(question)

    answer_tokens = tokenize(answer)
    question_tokens = tokenize(question)
    context_tokens = tokenize(context)

    question_length = len(question_tokens)

    if question_length:
        length_ratio = len(answer_tokens) / question_length
    else:
        length_ratio = 0.0

    return IndependentFeatures(
        context_coverage=coverage(context, answer),
        full_question_coverage=coverage(question, answer),
        latest_question_coverage=coverage(latest_question, answer),

        context_answer_jaccard=jaccard_similarity(context, answer),
        question_answer_jaccard=jaccard_similarity(question, answer),
        latest_question_answer_jaccard=jaccard_similarity(
            latest_question,
            answer,
        ),

        number_support=numeric_support(context, answer),

        answer_token_count=len(answer_tokens),
        question_token_count=len(question_tokens),
        context_token_count=len(context_tokens),

        answer_question_length_ratio=length_ratio,

        false_verification=_contains_any(
            answer,
            _FALSE_VERIFICATION_PHRASES,
        ),
        redirect_only=_contains_any(
            answer,
            _REDIRECT_PHRASES,
        ),
        reveals_ai_identity=_contains_any(
            answer,
            _AI_IDENTITY_PHRASES,
        ),
    )


def extract_feature_rows(samples: list[RagSample]) -> list[dict[str, float | int]]:
    """Extract features for a collection of samples."""
    return [extract_features(sample).to_dict() for sample in samples]