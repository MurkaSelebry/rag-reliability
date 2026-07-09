"""Tests for converting organizer-provided data into RagSample records."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).parents[1] / "scripts" / "prepare_data.py"
)
assert _SPEC is not None
prepare_data = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prepare_data)

convert_organizer_record = prepare_data.convert_organizer_record
detect_format = prepare_data.detect_format


def test_detect_format_treats_zip_as_csv() -> None:
    assert detect_format("from_organizators/data/data.zip", "auto") == "csv"


def test_convert_organizer_record_builds_rag_sample_fields() -> None:
    record = {
        "full_dialog": "Клиент: Как подключить Alfa Pay?\nАссистент: Сейчас помогу.",
        "answer": "Откройте карту и выберите оплату смартфоном.",
        "chunk_1": "Название статьи: Alfa Pay\nСодержание: Откройте карту.",
        "chunk_2": "",
        "chunk_3": "Содержание: Выберите оплату смартфоном.",
        "chunk_4": "",
        "chunk_5": "",
        "chunk_6": "",
        "chunk_7": "",
        "chunk_8": "",
        "binary_relevancy": "True",
        "binary_faithfulness": "False",
        "markers": "['reason_incomplete_answer', 'reason_irrelevant_chunk_used']",
    }

    sample = convert_organizer_record(record, row_number=7)

    assert sample.id == "organizer_000007"
    assert sample.question == record["full_dialog"]
    assert sample.answer == record["answer"]
    assert sample.context == (
        "[CHUNK 1]\nНазвание статьи: Alfa Pay\nСодержание: Откройте карту.\n\n"
        "[CHUNK 3]\nСодержание: Выберите оплату смартфоном."
    )
    assert sample.relevance == 1
    assert sample.faithfulness == 0
    assert sample.marker == "reason_incomplete_answer"


def test_convert_organizer_record_uses_none_marker_for_reliable_empty_marker() -> None:
    record = {
        "full_dialog": "Клиент: Сколько стоит карта?",
        "answer": "Карта стоит 590 рублей за первый год.",
        "chunk_1": "Карта стоит 590 рублей за первый год.",
        "binary_relevancy": "True",
        "binary_faithfulness": "True",
        "markers": "",
    }

    sample = convert_organizer_record(record, row_number=1)

    assert sample.faithfulness == 1
    assert sample.relevance == 1
    assert sample.marker == "none"


def test_convert_organizer_record_overrides_reliable_marker_to_none() -> None:
    record = {
        "full_dialog": "Клиент: Как подключить Alfa Pay?",
        "answer": "Откройте карту и выберите оплату смартфоном.",
        "chunk_1": "Откройте карту и выберите оплату смартфоном.",
        "binary_relevancy": "True",
        "binary_faithfulness": "True",
        "markers": "['reason_hallucinated_fact']",
    }

    sample = convert_organizer_record(record, row_number=2)

    assert sample.marker == "none"
