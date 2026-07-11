"""results_index: сбор прогонов m3/m6 в DataFrame'ы."""

import json

from rag_reliability.common.results_index import build_index


def _mk_variant(root, method, variant, split, preds, report=None, run=None):
    d = root / "predictions" / "cloud" / method / variant
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds), encoding="utf-8"
    )
    if report:
        (d / f"report_{split}.json").write_text(json.dumps(report), encoding="utf-8")
    if run:
        (d / "run.yaml").write_text("profile: cloud\nseed: 0\n", encoding="utf-8")


def test_index_collects_runs_and_predictions(tmp_path):
    _mk_variant(
        tmp_path,
        "m3",
        "few_shot",
        "val",
        [{"id": "pseudo_1", "p_faith": 0.9, "p_rel": 0.8, "meta": {"method": "logprobs"}}],
        report={"f1_macro_reliable": 0.82, "f1_macro_faith": 0.66, "f1_macro_rel": 0.80},
        run=True,
    )
    _mk_variant(
        tmp_path,
        "m3",
        "gepa_markers_s0",
        "val",
        [{"id": "pseudo_1", "p_faith": 0.5, "p_rel": 0.5, "meta": {}}],
    )
    idx = build_index(tmp_path)
    assert set(idx.runs["variant"]) == {"few_shot", "gepa_markers_s0"}
    row = idx.runs[idx.runs["variant"] == "few_shot"].iloc[0]
    assert row["f1_macro_reliable"] == 0.82 and row["method"] == "m3" and row["split"] == "val"
    assert len(idx.predictions) == 2
    p = idx.predictions[(idx.predictions["variant"] == "few_shot")].iloc[0]
    assert p["p_faith"] == 0.9 and p["extract_method"] == "logprobs"


def test_predictions_join_cases_kind(tmp_path):
    data = tmp_path / "data" / "processed"
    data.mkdir(parents=True)
    (data / "pseudo_dev_val.jsonl").write_text(
        json.dumps(
            {
                "id": "pseudo_1",
                "query": "q",
                "context": ["c"],
                "answer": "a",
                "faith": 1,
                "rel": 0,
                "markers": [],
                "meta": {"kind": "off_topic_answer", "synthetic": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _mk_variant(
        tmp_path,
        "m3",
        "few_shot",
        "val",
        [{"id": "pseudo_1", "p_faith": 0.9, "p_rel": 0.1, "meta": {}}],
    )
    idx = build_index(tmp_path)
    p = idx.predictions.iloc[0]
    assert p["kind"] == "off_topic_answer" and p["faith"] == 1 and p["rel"] == 0


def test_missing_dirs_give_empty_frames(tmp_path):
    idx = build_index(tmp_path)
    assert idx.runs.empty and idx.predictions.empty


def test_empty_data_file_and_smoke_ignored(tmp_path):
    # 0-байтовый файл данных не должен ронять индекс
    data = tmp_path / "data" / "processed"
    data.mkdir(parents=True)
    (data / "pseudo_dev_val__smoke5.jsonl").write_text("", encoding="utf-8")
    _mk_variant(
        tmp_path,
        "m3",
        "few_shot",
        "val",
        [{"id": "pseudo_1", "p_faith": 0.9, "p_rel": 0.1, "meta": {}}],
    )
    # smoke-версия предсказаний тоже игнорируется
    d = tmp_path / "predictions" / "cloud" / "m3" / "few_shot"
    (d / "val__smoke5.jsonl").write_text("", encoding="utf-8")
    idx = build_index(tmp_path)
    assert len(idx.predictions) == 1
    assert set(idx.runs["variant"]) == {"few_shot"}
