"""Тесты структур данных и jsonl IO."""

import json

from src.common.schemas import Case, Pred, load_cases, save_preds


def test_load_cases_full(tmp_path):
    rec = {
        "id": "pseudo_00001",
        "query": "q?",
        "context": ["c1", "c2"],
        "answer": "a",
        "dialog": ["клиент: привет"],
        "faith": 1,
        "rel": 0,
        "markers": ["off_topic_answer"],
        "meta": {"kind": "off_topic_answer", "synthetic": True},
    }
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    cases = load_cases(p)
    assert len(cases) == 1
    c = cases[0]
    assert c.id == "pseudo_00001" and c.meta["synthetic"] is True
    assert c.reliable == 0  # faith=1, rel=0


def test_context_str_becomes_list(tmp_path):
    rec = {"id": "1", "query": "q", "context": "один чанк", "answer": "a"}
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    c = load_cases(p)[0]
    assert c.context == ["один чанк"]
    assert c.faith is None and c.reliable is None


def test_ctx_truncation():
    c = Case(id="1", query="q", context=["x" * 100], answer="a")
    t = c.ctx_text(max_chars=30)
    assert t.endswith("[контекст усечён]") and len(t) < 100


def test_q_text_with_dialog():
    c = Case(id="1", query="тек. вопрос", context=["c"], answer="a", dialog=["клиент: раньше"])
    assert "История диалога" in c.q_text() and "тек. вопрос" in c.q_text()


def test_save_preds_roundtrip(tmp_path):
    preds = [Pred(id="1", p_faith=0.9, p_rel=0.1, meta={"m": "zero_shot"})]
    out = tmp_path / "sub" / "val.jsonl"
    save_preds(preds, out)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d == {"id": "1", "p_faith": 0.9, "p_rel": 0.1, "meta": {"m": "zero_shot"}}
