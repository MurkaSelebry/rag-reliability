"""Метод 6, этап 3: калибровка и предсказания.

p_faith — изотоническая калибровка комбинированного consistency-скора на val
          (unsupervised-сигнал + минимальная supervised-калибровка, как в
          статье: «threshold calibration on val»);
p_rel   — логрег на тех же фичах + cos_q_a (честно ожидаем слабый результат,
          это и есть проверка H4).

Также пишет стратификацию качества по n_clusters — анализ провала
alignment-collapse (кейсы с 1 кластером, где sampling-методы слепы).

Запуск (после sample+features на train/val/test):
  python -m src.m6_selfcheck.calibrate_predict --config configs/config.yaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..common.schemas import load_cases, load_yaml, Pred, save_preds
from ..common.eval_local import fit_thresholds, evaluate

FEATS = ["selfcheck_contra_mean", "selfcheck_contra_max",
         "semantic_entropy", "n_clusters", "answer_in_top_cluster", "cos_q_a"]


def load_feats(path: Path) -> dict[str, dict]:
    return {d["id"]: d for l in open(path, encoding="utf-8") if (d := json.loads(l))}


def matrix(cases, feats) -> np.ndarray:
    return np.array([[feats[c.id][f] for f in FEATS] for c in cases])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    m6 = cfg["m6"]
    fdir = Path(m6["features_cache"])

    splits = {s: load_cases(cfg["data"][s]) for s in ("train", "val", "test")}
    feats = {s: load_feats(fdir / f"{s}.jsonl") for s in splits}

    X = {s: matrix(splits[s], feats[s]) for s in splits}
    y_f = {s: np.array([c.faith for c in splits[s]]) for s in ("train", "val")}
    y_r = {s: np.array([c.rel for c in splits[s]]) for s in ("train", "val")}

    # --- p_faith: consistency-скор -> изотоника на val ---------------------
    # скор «ненадёжности»: z-нормированная сумма contra_mean и sem_entropy
    scaler = StandardScaler().fit(X["train"][:, [0, 2]])
    def raw_unfaith(Xs):
        z = scaler.transform(Xs[:, [0, 2]])
        return z.sum(axis=1)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(-raw_unfaith(X["val"]), y_f["val"])  # минус: больше скор -> меньше P(faith)

    # --- p_rel: логрег на train ---------------------------------------------
    sc_all = StandardScaler().fit(X["train"])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc_all.transform(X["train"]), y_r["train"])

    out_root = Path(m6["out_pred_dir"])
    preds_by_split = {}
    for s in splits:
        p_faith = iso.predict(-raw_unfaith(X[s]))
        p_rel = lr.predict_proba(sc_all.transform(X[s]))[:, 1]
        preds = [Pred(id=c.id, p_faith=float(pf), p_rel=float(pr),
                      meta={"n_clusters": int(feats[s][c.id]["n_clusters"])})
                 for c, pf, pr in zip(splits[s], p_faith, p_rel)]
        save_preds(preds, out_root / f"{s}.jsonl")
        preds_by_split[s] = preds

    # --- отчёт + стратификация по n_clusters --------------------------------
    tf, tr, _ = fit_thresholds(splits["val"], preds_by_split["val"])
    report = evaluate(splits["test"], preds_by_split["test"], tf, tr)
    collapse = [c for c, p in zip(splits["test"], preds_by_split["test"])
                if p.meta["n_clusters"] == 1]
    report["share_single_cluster_test"] = round(len(collapse) / max(len(splits["test"]), 1), 3)
    if len(collapse) > 20:
        sub_preds = [p for p in preds_by_split["test"] if p.meta["n_clusters"] == 1]
        report["single_cluster_subset"] = evaluate(collapse, sub_preds, tf, tr)
    (out_root / "report_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
