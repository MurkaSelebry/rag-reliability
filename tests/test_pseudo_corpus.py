"""Чистая логика генератора псевдо-корпуса (docs/07.2): микс, контекст, сплиты."""
import random
from collections import Counter

from tools.make_pseudo_corpus import LABELS, build_context, plan_kinds, split_ids


def test_plan_kinds_proportion_2111():
    kinds = plan_kinds(10)
    c = Counter(kinds)
    assert c == {"clean": 4, "hallucination": 2, "incomplete_answer": 2, "off_topic_answer": 2}


def test_labels_table_matches_spec():
    assert LABELS["clean"] == {"faith": 1, "rel": 1, "markers": []}
    assert LABELS["hallucination"] == {"faith": 0, "rel": 1, "markers": ["hallucination"]}
    assert LABELS["incomplete_answer"] == {"faith": 0, "rel": 1, "markers": ["incomplete_answer"]}
    assert LABELS["off_topic_answer"] == {"faith": 1, "rel": 0, "markers": ["off_topic_answer"]}


def test_build_context_has_source_and_distractors():
    rng = random.Random(0)
    pool = [f"абзац {i}" for i in range(50)]
    ctx = build_context(rng, "источник", pool)
    assert "источник" in ctx
    assert 2 <= len(ctx) <= 3                      # источник + 1–2 дистрактора
    assert all(c == "источник" or c in pool for c in ctx)
    assert len(set(ctx)) == len(ctx)               # без дублей


def test_build_context_shuffles():
    """Источник не обязан быть первым чанком (имитация retrieved-контекста)."""
    rng = random.Random(1)
    pool = [f"абзац {i}" for i in range(50)]
    positions = {build_context(rng, "источник", pool).index("источник") for _ in range(30)}
    assert len(positions) > 1


def test_split_ids_80_10_10_deterministic():
    ids = [f"pseudo_{i:05d}" for i in range(300)]
    s1, s2 = split_ids(ids, seed=42), split_ids(ids, seed=42)
    assert s1 == s2                                 # детерминизм
    assert sorted(s1["train"] + s1["val"] + s1["test"]) == sorted(ids)
    assert len(s1["val"]) == 30 and len(s1["test"]) == 30
    assert set(s1["train"]).isdisjoint(s1["val"])
