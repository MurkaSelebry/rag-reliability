"""Добор сэмплов m6: сколько догенерировать под текущий n_samples."""

import json

from src.m6.sample import _need_samples


def test_need_samples_topup(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"id": "x", "samples": ["a", "b"]}), encoding="utf-8")
    need, existing = _need_samples(f, 5)
    assert need == 3 and existing == ["a", "b"]


def test_need_samples_full(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"id": "x", "samples": list("abcde")}), encoding="utf-8")
    assert _need_samples(f, 5) == (0, list("abcde"))


def test_need_samples_missing(tmp_path):
    assert _need_samples(tmp_path / "none.json", 5) == (5, [])
