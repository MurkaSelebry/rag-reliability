"""Разбор ошибок по реальным маркерам корпуса (reason_* taxonomy).

Для каждого варианта предсказаний считает recall unreliable в разрезе маркеров
(какие типы ошибок метод ловит, какие пропускает) и таблицу 2x2 по золотым
(faith, rel): размер ячейки и средние p_faith/p_rel. Главный вывод — матрица
«variant × marker» recall_unreliable.

Запуск:
  python scripts/marker_signals.py --config configs/config.alfa_cloud.yaml --split val \
      --pred predictions/alfa_openrouter/m3/zero_shot/val.jsonl \
      --pred predictions/alfa_openrouter/m3/few_shot/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rag_reliability.common.config import load_config
from rag_reliability.common.schemas import load_cases

ALL_ROW = "__all_unreliable__"  # сводная строка: recall по всем ненадёжным кейсам


def per_marker_recall(preds: pd.DataFrame, t_faith: float, t_rel: float) -> pd.DataFrame:
    """Recall unreliable по маркерам.

    predicted_unreliable = NOT (p_faith >= t_faith AND p_rel >= t_rel).
    Для каждого маркера: n — кейсов с ним, recall_unreliable — доля пойманных.
    """
    df = preds.copy()
    df["pred_unrel"] = ~((df["p_faith"] >= t_faith) & (df["p_rel"] >= t_rel))
    ex = df.explode("markers").dropna(subset=["markers"])
    g = ex.groupby("markers")["pred_unrel"].agg(n="size", recall_unreliable="mean")
    out = g.reset_index().rename(columns={"markers": "marker"})
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def cell_2x2_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Таблица по четырём ячейкам золотых (faith, rel): n и средние вероятности."""
    g = preds.groupby(["faith", "rel"]).agg(
        n=("id", "size"), mean_p_faith=("p_faith", "mean"), mean_p_rel=("p_rel", "mean")
    )
    return g.reset_index()


def _load_preds_with_gold(pred_path: Path, gold: pd.DataFrame) -> pd.DataFrame:
    """Читает predictions jsonl и присоединяет золотые faith/rel/markers по id."""
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    p = pd.DataFrame([{"id": r["id"], "p_faith": r["p_faith"], "p_rel": r["p_rel"]} for r in rows])
    df = p.merge(gold, on="id", how="inner")
    if len(df) < len(p):
        print(f"!! {pred_path}: {len(p) - len(df)} предсказаний без золота — выброшены из анализа")
    return df


def _load_thresholds(pred_path: Path, split: str) -> tuple[float, float]:
    """Пороги из report_{split}.json рядом с predictions; fallback 0.5/0.5."""
    rep = pred_path.parent / f"report_{split}.json"
    if rep.exists():
        d = json.loads(rep.read_text(encoding="utf-8"))
        return float(d["t_faith"]), float(d["t_rel"])
    print(f"!! {rep} не найден — пороги по умолчанию 0.5/0.5")
    return 0.5, 0.5


def _recall_all_unreliable(df: pd.DataFrame, t_faith: float, t_rel: float) -> tuple[int, float]:
    """Recall по ВСЕМ золотым unreliable-кейсам (с маркерами и без)."""
    unrel = df[(df["faith"] == 0) | (df["rel"] == 0)]
    if unrel.empty:
        return 0, float("nan")
    caught = ~((unrel["p_faith"] >= t_faith) & (unrel["p_rel"] >= t_rel))
    return len(unrel), float(caught.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Анализ ошибок по маркерам reason_*")
    ap.add_argument("--config", default="configs/config.alfa_cloud.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument(
        "--pred", action="append", required=True, help="predictions jsonl (можно несколько)"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    cases = load_cases(cfg["data"][args.split])
    gold = pd.DataFrame(
        [
            {"id": c.id, "faith": c.faith, "rel": c.rel, "markers": c.markers}
            for c in cases
            if c.faith is not None and c.rel is not None
        ]
    )

    result: dict = {"split": args.split, "variants": {}}
    matrix: dict[str, dict[str, float]] = {}  # variant -> {marker: recall}
    marker_n: dict[str, int] = {}
    all_unrel_n = 0

    for pred in args.pred:
        pred_path = Path(pred)
        variant = pred_path.parent.name
        t_faith, t_rel = _load_thresholds(pred_path, args.split)
        df = _load_preds_with_gold(pred_path, gold)

        pm = per_marker_recall(df, t_faith, t_rel)
        cells = cell_2x2_table(df)
        n_unrel, rec_all = _recall_all_unreliable(df, t_faith, t_rel)

        print(f"\n== {variant}  (t_faith={t_faith}, t_rel={t_rel}, n={len(df)})")
        print(pm.to_string(index=False))
        print(f"{ALL_ROW}: n={n_unrel}, recall_unreliable={rec_all:.3f}")
        print("-- 2x2 (gold faith x rel):")
        print(cells.to_string(index=False))

        matrix[variant] = dict(zip(pm["marker"], pm["recall_unreliable"]))
        matrix[variant][ALL_ROW] = rec_all
        all_unrel_n = max(all_unrel_n, n_unrel)
        for _, r in pm.iterrows():
            marker_n[r["marker"]] = max(marker_n.get(r["marker"], 0), int(r["n"]))
        result["variants"][variant] = {
            "t_faith": t_faith,
            "t_rel": t_rel,
            "n": len(df),
            "per_marker": pm.to_dict(orient="records"),
            "recall_all_unreliable": {"n": n_unrel, "recall_unreliable": rec_all},
            "cell_2x2": cells.to_dict(orient="records"),
        }

    variants = list(matrix)
    markers = sorted(marker_n, key=lambda m: marker_n[m], reverse=True)
    print("\n== МАТРИЦА variant x marker: recall_unreliable (пороги из report)")
    header = f"{'marker':<34}{'n':>5}" + "".join(f"{v:>16}" for v in variants)
    print(header)
    rows_json = []
    for m in markers + [ALL_ROW]:
        n = marker_n.get(m, all_unrel_n)
        vals = {v: matrix[v].get(m) for v in variants}
        cells_str = "".join(
            f"{vals[v]:>16.3f}" if vals[v] is not None else f"{'-':>16}" for v in variants
        )
        print(f"{m:<34}{n:>5}" + cells_str)
        rows_json.append({"marker": m, "n": n, **vals})
    result["matrix"] = rows_json

    out = Path("artifacts/alfa_or") / f"marker_signals_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
