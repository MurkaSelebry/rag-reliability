"""Тесты стэкинга разнородных сигналов."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag_reliability.stacking.collect import collect_features


def _write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_collect_features_preserves_sample_order(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    m3 = tmp_path / "m3.jsonl"
    _write_scores(
        surface,
        [
            {"id": "b", "scores": {"surf.p_faith": 0.8, "surf.p_rel": 0.7}},
            {"id": "a", "scores": {"surf.p_faith": 0.2, "surf.p_rel": 0.3}},
        ],
    )
    _write_scores(
        m3,
        [
            {"id": "a", "scores": {"m3.p_faith": 0.4}},
            {"id": "b", "scores": {"m3.p_faith": 0.9}},
        ],
    )

    matrix, names = collect_features(
        {"surf": surface, "m3": m3},
        ["a", "b"],
        ["surf.p_faith", "surf.p_rel", "m3.p_faith"],
    )

    assert names == ["surf.p_faith", "surf.p_rel", "m3.p_faith"]
    np.testing.assert_allclose(matrix, [[0.2, 0.3, 0.4], [0.8, 0.7, 0.9]])


def test_collect_features_lists_missing_ids(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [{"id": "a", "scores": {"surf.p_faith": 0.2}}],
    )

    with pytest.raises(ValueError, match=r"Missing 2 sample id\(s\).*b.*c"):
        collect_features(
            {"surf": surface},
            ["a", "b", "c"],
            ["surf.p_faith"],
        )


def test_collect_features_reports_missing_key(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [
            {"id": "a", "scores": {"surf.p_faith": 0.2}},
            {"id": "b", "scores": {"surf.p_rel": 0.7}},
        ],
    )

    with pytest.raises(ValueError, match=r"Missing feature 'surf\.p_faith'.*b"):
        collect_features(
            {"surf": surface},
            ["a", "b"],
            ["surf.p_faith"],
        )


def test_collect_features_skips_absent_optional_source(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [{"id": "a", "scores": {"surf.p_faith": 0.2}}],
    )

    matrix, names = collect_features(
        {"surf": surface, "enc": tmp_path / "missing.jsonl"},
        ["a"],
        ["surf.p_faith", "enc.logit"],
        required=False,
    )

    assert names == ["surf.p_faith"]
    np.testing.assert_allclose(matrix, [[0.2]])


def test_collect_features_rejects_duplicate_ids(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [
            {"id": "a", "scores": {"surf.p_faith": 0.2}},
            {"id": "a", "scores": {"surf.p_faith": 0.3}},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate id 'a'"):
        collect_features({"surf": surface}, ["a"], ["surf.p_faith"])
