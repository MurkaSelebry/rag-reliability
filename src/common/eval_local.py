"""Локальная оценка для разработки. Финальные числа — только через общий
замороженный evaluate.py платформы; этот модуль повторяет его логику для
быстрых итераций внутри веток 3 и 6."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from .schemas import Case, Pred


def _align(cases: list[Case], preds: list[Pred]):
    by_id = {p.id: p for p in preds}
    y_f, y_r, p_f, p_r = [], [], [], []
    for c in cases:
        if c.faith is None or c.id not in by_id:
            continue
        y_f.append(c.faith)
        y_r.append(c.rel)
        p_f.append(by_id[c.id].p_faith)
        p_r.append(by_id[c.id].p_rel)
    return map(np.asarray, (y_f, y_r, p_f, p_r))


def fit_thresholds(cases: list[Case], preds: list[Pred], step: float = 0.01):
    """Сетка порогов (t_faith, t_rel), максимизирующая f1-macro(reliable) на val."""
    y_f, y_r, p_f, p_r = _align(cases, preds)
    y_rel_joint = (y_f == 1) & (y_r == 1)
    grid = np.arange(step, 1.0, step)
    best = (0.5, 0.5, -1.0)
    for tf in grid:
        pred_f = p_f >= tf
        for tr in grid:
            pred_joint = pred_f & (p_r >= tr)
            f1 = f1_score(y_rel_joint, pred_joint, average="macro")
            if f1 > best[2]:
                best = (float(tf), float(tr), float(f1))
    return best  # (t_faith, t_rel, val_f1)


def evaluate(cases: list[Case], preds: list[Pred], t_faith: float, t_rel: float) -> dict:
    y_f, y_r, p_f, p_r = _align(cases, preds)
    pred_f, pred_r = p_f >= t_faith, p_r >= t_rel
    y_joint = (y_f == 1) & (y_r == 1)
    pred_joint = pred_f & pred_r
    return {
        "f1_macro_reliable": float(f1_score(y_joint, pred_joint, average="macro")),
        "f1_macro_faith": float(f1_score(y_f, pred_f, average="macro")),
        "f1_macro_rel": float(f1_score(y_r, pred_r, average="macro")),
        "t_faith": t_faith,
        "t_rel": t_rel,
        "n": int(len(y_f)),
    }
