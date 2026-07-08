"""Сплиты: group-aware по нормализованному запросу, стратификация, curator-режим."""

from src.common.schemas import Case
from src.data.make_splits import group_key, make_splits


def _c(i, q, faith=1, rel=1):
    return Case(
        id=f"alfa_{i:012d}",
        query=q,
        context=["c"],
        answer=f"a{i}",
        faith=faith,
        rel=rel,
        meta={"synthetic": False},
    )


def test_group_key_normalizes():
    assert group_key(_c(1, " Какая  СТАВКА?! ")) == group_key(_c(2, "какая ставка"))


def test_groups_do_not_cross_splits():
    cases = [_c(i, f"вопрос {i % 20}", faith=i % 3 > 0) for i in range(200)]
    sp = make_splits(cases, seed=42, mode="group")
    key2split = {}
    for name, ids in sp.items():
        by_id = {c.id: c for c in cases}
        for cid in ids:
            k = group_key(by_id[cid])
            assert key2split.setdefault(k, name) == name


def test_deterministic_and_partition():
    cases = [_c(i, f"q{i}") for i in range(100)]
    a, b = make_splits(cases, seed=42), make_splits(cases, seed=42)
    assert a == b
    all_ids = sorted(a["train"] + a["val"] + a["test"])
    assert all_ids == sorted(c.id for c in cases)


def test_curator_mode_matches_sklearn():
    """Режим воспроизведения бейзлайна: 80/20 stratify(reliable) seed 42, без val."""
    cases = [_c(i, f"q{i}", faith=i % 4 > 0) for i in range(100)]
    sp = make_splits(cases, seed=42, mode="curator")
    assert len(sp["test"]) == 20 and len(sp["val"]) == 0
