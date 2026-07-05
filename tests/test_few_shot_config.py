"""few_shot.yaml: структура, покрытие типов, якоря обеих осей (итог B1/B2)."""

import yaml

REQUIRED_FIELDS = {"q", "ctx", "a", "analysis", "faith", "rel"}


def _load():
    return yaml.safe_load(open("configs/few_shot.yaml", encoding="utf-8"))["examples"]


def test_examples_count_and_fields():
    ex = _load()
    assert 6 <= len(ex) <= 8
    for e in ex:
        assert REQUIRED_FIELDS <= set(e), f"нет полей: {REQUIRED_FIELDS - set(e)}"
        assert e["faith"] in ("PASS", "FAIL") and e["rel"] in ("PASS", "FAIL")
        assert len(e["analysis"].strip()) >= 20  # анализ содержательный, не заглушка


def test_off_topic_anchor_present():
    """Цель, которую B1 не взял промптом: off_topic = faith PASS + rel FAIL."""
    assert any(e["faith"] == "PASS" and e["rel"] == "FAIL" for e in _load())


def test_incomplete_priority():
    """B2: минимум 2 примера с faith=FAIL, rel=PASS и словами про упущенную деталь."""
    ex = [e for e in _load() if e["faith"] == "FAIL" and e["rel"] == "PASS"]
    assert (
        sum("опущ" in e["analysis"].lower() or "неполн" in e["analysis"].lower() for e in ex) >= 2
    )


def test_strict_rel_fail_anchor():
    """Противовес размытию rel после B1: есть пример с безоговорочным rel=FAIL в анализе."""
    assert any(
        e["rel"] == "FAIL" and "не отвечает на вопрос" in e["analysis"].lower() for e in _load()
    )


def test_both_pass_present():
    assert any(e["faith"] == "PASS" and e["rel"] == "PASS" for e in _load())
