"""Сигналы по реальным маркерам: recall unreliable в разрезе 13 маркеров и ячеек 2x2."""

import pandas as pd

from tools.marker_signals import cell_2x2_table, per_marker_recall


def _preds():
    return pd.DataFrame(
        [
            {
                "id": "a",
                "p_faith": 0.1,
                "p_rel": 0.9,
                "faith": 0,
                "rel": 1,
                "markers": ["reason_hallucinated_fact"],
            },
            {
                "id": "b",
                "p_faith": 0.9,
                "p_rel": 0.9,
                "faith": 0,
                "rel": 1,
                "markers": ["reason_hallucinated_fact"],
            },  # пропущенный
            {
                "id": "c",
                "p_faith": 0.9,
                "p_rel": 0.1,
                "faith": 1,
                "rel": 0,
                "markers": ["reason_answer_for_operator"],
            },
        ]
    )


def test_per_marker_recall():
    r = per_marker_recall(_preds(), t_faith=0.5, t_rel=0.5)
    row = r[r["marker"] == "reason_hallucinated_fact"].iloc[0]
    assert row["n"] == 2 and row["recall_unreliable"] == 0.5  # пойман 1 из 2
    op = r[r["marker"] == "reason_answer_for_operator"].iloc[0]
    assert op["recall_unreliable"] == 1.0


def test_cell_2x2_mean_probs():
    t = cell_2x2_table(_preds())
    cell = t[(t.faith == 1) & (t.rel == 0)].iloc[0]
    assert cell["mean_p_faith"] == 0.9 and cell["n"] == 1
