"""Метод 6, этап 3: калибровка и предсказания (изотоника val + логрег train).

Запуск (после sample+features на train/val/test):
  python -m src.m6.predict --config configs/config.cloud.yaml --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..common.config import load_config
from ..common.eval_local import evaluate, fit_thresholds
from ..common.run_meta import save_run_yaml
from ..common.schemas import Pred, load_cases, save_preds

FEATS = [
    "selfcheck_contra_mean",
    "selfcheck_contra_max",
    "semantic_entropy",
    "n_clusters",
    "answer_in_top_cluster",
    "cos_q_a",
]


def load_feats(path: Path) -> dict[str, dict]:
    return {d["id"]: d for l in open(path, encoding="utf-8") if (d := json.loads(l))}


def matrix(cases, feats) -> np.ndarray:
    return np.array([[feats[c.id][f] for f in FEATS] for c in cases])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument(
        "--limit", type=int, default=None, help="обрезать каждый сплит до первых N кейсов"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    m6 = cfg["m6"]
    fdir = Path(m6["features_cache"])

    splits = {s: load_cases(cfg["data"][s])[: args.limit] for s in ("train", "val", "test")}
    feats = {s: load_feats(fdir / f"{s}.jsonl") for s in splits}

    # Guard: все кейсы должны иметь фичи
    for s, cs in splits.items():
        missing = [c.id for c in cs if c.id not in feats[s]]
        if missing:
            raise SystemExit(
                f"нет фич для {len(missing)} кейсов сплита {s} "
                f"(первые: {missing[:3]}) — прогони sample+features"
            )

    X = {s: matrix(splits[s], feats[s]) for s in splits}
    y_f = {s: np.array([c.faith for c in splits[s]]) for s in ("train", "val")}
    y_r = {s: np.array([c.rel for c in splits[s]]) for s in ("train", "val")}

    # --- p_faith: consistency-скор -> изотоника на val ---------------------
    # скор «ненадёжности»: z-нормированная сумма contra_mean и sem_entropy
    scaler = StandardScaler().fit(X["train"][:, [0, 2]])

    def raw_unfaith(Xs):
        z = scaler.transform(Xs[:, [0, 2]])
        return z.sum(axis=1)

    if len(np.unique(y_f["val"])) < 2:
        raise SystemExit("val не содержит обоих классов faith — увеличь --limit или проверь данные")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(-raw_unfaith(X["val"]), y_f["val"])  # минус: больше скор -> меньше P(faith)

    # --- p_rel: логрег на train ---------------------------------------------
    sc_all = StandardScaler().fit(X["train"])
    if len(np.unique(y_r["train"])) < 2:
        raise SystemExit("train не содержит обоих классов rel — увеличь --limit или проверь данные")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc_all.transform(X["train"]), y_r["train"])

    out_root = Path(m6["out_pred_dir"])
    preds_by_split = {}
    for s in splits:
        p_faith = iso.predict(-raw_unfaith(X[s]))
        p_rel = lr.predict_proba(sc_all.transform(X[s]))[:, 1]
        preds = [
            Pred(
                id=c.id,
                p_faith=float(pf),
                p_rel=float(pr),
                meta={"n_clusters": int(feats[s][c.id]["n_clusters"])},
            )
            for c, pf, pr in zip(splits[s], p_faith, p_rel)
        ]
        save_preds(preds, out_root / f"{s}.jsonl")
        preds_by_split[s] = preds

    # --- отчёт + стратификация по n_clusters --------------------------------
    tf, tr, _ = fit_thresholds(splits["val"], preds_by_split["val"])
    report = evaluate(splits["test"], preds_by_split["test"], tf, tr)
    collapse = [
        c for c, p in zip(splits["test"], preds_by_split["test"]) if p.meta["n_clusters"] == 1
    ]
    report["share_single_cluster_test"] = round(len(collapse) / max(len(splits["test"]), 1), 3)
    if len(collapse) > 20:
        sub_preds = [p for p in preds_by_split["test"] if p.meta["n_clusters"] == 1]
        report["single_cluster_subset"] = evaluate(collapse, sub_preds, tf, tr)
    (out_root / "report_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Сохраняем метаданные прогона после успешного завершения
    save_run_yaml(out_root, cfg, split="all", limit=args.limit, method="m6")

    # mlflow-трекинг (локальный file-store), только при tracking.enabled
    tr_cfg = cfg.get("tracking") or {}
    if tr_cfg.get("enabled"):
        from ..common.tracking import log_run

        log_run(
            tracking_uri=tr_cfg.get("uri", "file:./mlruns"),
            experiment="m6",
            run_name="m6/test",
            cfg=cfg,
            metrics={k: float(v) for k, v in report.items() if isinstance(v, (int, float))},
            artifacts=[out_root / "report_test.json", out_root / "run.yaml"],
            tags={"split": "test", "variant": "m6", "method": "m6"},
        )


if __name__ == "__main__":
    main()
