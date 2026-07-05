"""Логика отчёта: сводная таблица вариантов, матрица ошибок, данные эволюции GEPA."""
import pandas as pd
import pytest

from tools.make_report import confusion_counts, gepa_evolution_frame, variant_summary


def test_variant_summary_orders_and_gaps():
    runs = pd.DataFrame([
        {"method": "m3", "variant": "few_shot", "split": "val", "f1_macro_reliable": 0.82},
        {"method": "m3", "variant": "few_shot", "split": "test", "f1_macro_reliable": 0.69},
        {"method": "m3", "variant": "zero_shot", "split": "val", "f1_macro_reliable": 0.79},
        {"method": "m3", "variant": "zero_shot", "split": "test", "f1_macro_reliable": 0.66},
    ])
    s = variant_summary(runs)
    assert list(s["variant"]) == ["few_shot", "zero_shot"]        # сортировка по val desc
    assert s.iloc[0]["gap_val_test"] == pytest.approx(0.13)


def test_confusion_counts_reliable():
    preds = pd.DataFrame([
        {"p_faith": 0.9, "p_rel": 0.9, "faith": 1, "rel": 1},   # TP reliable
        {"p_faith": 0.9, "p_rel": 0.1, "faith": 1, "rel": 1},   # FN
        {"p_faith": 0.9, "p_rel": 0.9, "faith": 0, "rel": 1},   # FP
        {"p_faith": 0.1, "p_rel": 0.1, "faith": 0, "rel": 0},   # TN
    ])
    c = confusion_counts(preds, t_faith=0.5, t_rel=0.5)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)


def test_gepa_evolution_frame_best_flag():
    cands = pd.DataFrame([
        {"variant": "markers", "seed": 0, "cand_idx": 0, "val_score": 0.70},
        {"variant": "markers", "seed": 0, "cand_idx": 2, "val_score": 0.717, "is_best": True},
    ])
    ev = gepa_evolution_frame(cands)
    assert ev[ev["cand_idx"] == 2]["is_best"].iloc[0]
