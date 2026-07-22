import pytest

from rag_reliability.formatting import build_chat_training_record
from rag_reliability.nb_format import build_sft_messages
from rag_reliability.schema import RagSample

SAMPLES = [
    RagSample(
        id="ok",
        question="Сколько стоит обслуживание?",
        context="Обслуживание карты «Классика» составляет 149 рублей.",
        answer="149 рублей.",
        faithfulness=1,
        relevance=1,
        marker="none",
    ),
    RagSample(
        id="bad",
        question="Какой суточный лимит?",
        context="Лимит 100 000 рублей.",
        answer="500 000 рублей.",
        faithfulness=0,
        relevance=1,
        marker="hallucination",
    ),
]


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("mode", ["direct", "marker"])
def test_nb_format_matches_repo(sample: RagSample, mode: str) -> None:
    assert build_sft_messages(sample, mode) == build_chat_training_record(sample, mode)
