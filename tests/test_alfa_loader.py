"""Адаптер схемы кураторов: id, извлечение query, чанки, маркеры, edge-кейсы."""

import pandas as pd

from rag_reliability.data.alfa_loader import extract_last_client_turn, load_alfa

ROW = {
    "full_dialog": "Ассистент: Приветствую!\nКлиент: Какая ставка по вкладу?",
    "answer": "Ставка 19.84%.",
    "chunk_1": "Вклад Максимальный: 19.84% на 62 дня.",
    "chunk_2": None,
    "chunk_3": None,
    "chunk_4": None,
    "chunk_5": None,
    "chunk_6": None,
    "chunk_7": None,
    "chunk_8": None,
    "binary_relevancy": True,
    "binary_faithfulness": False,
    "markers": "['reason_hallucinated_fact']",
}


def _csv(tmp_path, rows):
    p = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_basic_mapping(tmp_path):
    cases = load_alfa(_csv(tmp_path, [ROW]))
    c = cases[0]
    assert c.id.startswith("alfa_") and not c.id.startswith("pseudo_")
    assert c.query == "Какая ставка по вкладу?"
    assert c.context == ["Вклад Максимальный: 19.84% на 62 дня."]
    assert (c.faith, c.rel) == (0, 1) and c.reliable == 0
    assert c.markers == ["reason_hallucinated_fact"]
    assert c.meta["synthetic"] is False
    assert "Ассистент: Приветствую!" in "\n".join(c.dialog)


def test_id_deterministic_and_unique(tmp_path):
    row2 = dict(ROW, answer="Другой ответ.")
    a = load_alfa(_csv(tmp_path, [ROW, row2]))
    b = load_alfa(_csv(tmp_path, [ROW, row2]))
    assert [c.id for c in a] == [c.id for c in b]
    assert a[0].id != a[1].id


def test_no_client_turn_fallback(tmp_path):
    row = dict(ROW, full_dialog="Ассистент: Приветствую! Чем помочь?")
    c = load_alfa(_csv(tmp_path, [row]))[0]
    assert c.query == "" and c.meta["no_client_turn"] is True


def test_markers_absent_and_malformed(tmp_path):
    rows = [dict(ROW, markers=None), dict(ROW, answer="x", markers="не список")]
    cases = load_alfa(_csv(tmp_path, rows))
    assert cases[0].markers == []
    assert cases[1].markers == [] and cases[1].meta["markers_parse_error"] is True


def test_extract_last_client_turn_multi():
    d = "Клиент: раз\nАссистент: ответ\nКлиент: два\nОператор: и оператор"
    assert extract_last_client_turn(d) == "два"
