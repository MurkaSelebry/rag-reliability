"""Сбор всех артефактов прогонов (m3/m6) в pandas-DataFrame'ы.

Единый коллектор данных: HTML-отчёт и Streamlit-эксплорер читают ТОЛЬКО его
вывод, не дублируя парсинг. Здесь никакого рисования — только данные.
Всё устойчиво к отсутствующим частям: любая пропавшая директория/файл даёт
пустой DataFrame, исключения не пробрасываются.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# Ключи метрик в report_*.json
_REPORT_KEYS = ("f1_macro_reliable", "f1_macro_faith", "f1_macro_rel", "t_faith", "t_rel", "n")
_RUN_SPLITS = ("val", "test", "train")
# Признаки, извлекаемые Методом 6
_M6_FEATURE_KEYS = (
    "selfcheck_contra_mean", "selfcheck_contra_max", "semantic_entropy",
    "n_clusters", "answer_in_top_cluster", "cos_q_a",
)
_RUN_COLS = (
    "method", "variant", "split", *_REPORT_KEYS,
    "profile", "seed", "git_hash", "share_single_cluster_test", "path",
)
_PRED_COLS = (
    "id", "method", "variant", "split", "p_faith", "p_rel",
    "extract_method", "meta_raw", "faith", "rel", "kind", "markers",
)


@dataclass
class ResultsIndex:
    """Набор таблиц по всем прогонам проекта."""
    runs: pd.DataFrame
    predictions: pd.DataFrame
    m6_features: pd.DataFrame
    gepa: pd.DataFrame
    entropy_ablation: pd.DataFrame


def stats_to_pred_dir(variant: str, seed: int | str) -> str:
    """Имя dir предсказаний GEPA: `{variant}_seed{seed}` → `gepa_{variant}_s{seed}`."""
    return f"gepa_{variant}_s{seed}"


def _read_jsonl(path: Path) -> list[dict]:
    """Читает jsonl, пропуская пустые строки; при ошибке — пустой список."""
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _read_json(path: Path) -> dict | None:
    """Читает json-объект; None при отсутствии/ошибке."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_run_yaml(path: Path) -> dict:
    """Парсит run.yaml, вытягивая top-level profile/seed/git_hash."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data.get(k) for k in ("profile", "seed", "git_hash")}


def _is_smoke(name: str) -> bool:
    """Файл-остаток smoke-прогона вида `*__smoke*.jsonl`."""
    return "__smoke" in name


def _empty(cols) -> pd.DataFrame:
    return pd.DataFrame(columns=list(cols))


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
def _collect_runs(pred_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    # m3 — структура с поддиректориями-вариантами
    m3_root = pred_root / "m3"
    if m3_root.is_dir():
        for vdir in sorted(p for p in m3_root.iterdir() if p.is_dir()):
            run_meta = _read_run_yaml(vdir / "run.yaml")
            for split in _RUN_SPLITS:
                if not (vdir / f"{split}.jsonl").is_file():
                    continue
                report = _read_json(vdir / f"report_{split}.json") or {}
                rows.append({
                    "method": "m3", "variant": vdir.name, "split": split,
                    **{k: report.get(k) for k in _REPORT_KEYS},
                    "profile": run_meta.get("profile"),
                    "seed": run_meta.get("seed"),
                    "git_hash": run_meta.get("git_hash"),
                    "share_single_cluster_test": None,
                    "path": str(vdir),
                })
    # m6 — плоская директория, variant="m6", отчёт только для test
    m6_root = pred_root / "m6"
    if m6_root.is_dir():
        run_meta = _read_run_yaml(m6_root / "run.yaml")
        for split in _RUN_SPLITS:
            if not (m6_root / f"{split}.jsonl").is_file():
                continue
            report = _read_json(m6_root / f"report_{split}.json") or {}
            rows.append({
                "method": "m6", "variant": "m6", "split": split,
                **{k: report.get(k) for k in _REPORT_KEYS},
                "profile": run_meta.get("profile"),
                "seed": run_meta.get("seed"),
                "git_hash": run_meta.get("git_hash"),
                "share_single_cluster_test": report.get("share_single_cluster_test"),
                "path": str(m6_root),
            })
    if not rows:
        return _empty(_RUN_COLS)
    return pd.DataFrame(rows, columns=list(_RUN_COLS))


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
def _load_gold(data_root: Path) -> dict[str, dict[str, dict]]:
    """Голд-метки по сплитам: {split: {id: {faith,rel,kind,markers}}}."""
    gold: dict[str, dict[str, dict]] = {}
    for split in _RUN_SPLITS:
        path = data_root / f"pseudo_dev_{split}.jsonl"
        if not path.is_file():
            continue
        try:
            from src.common.schemas import load_cases
            cases = load_cases(path)
        except Exception:
            continue
        gold[split] = {
            c.id: {
                "faith": c.faith, "rel": c.rel,
                "kind": (c.meta or {}).get("kind"), "markers": c.markers,
            }
            for c in cases
        }
    return gold


def _iter_pred_dirs(pred_root: Path):
    """Возвращает (method, variant, dir) для всех вариантов предсказаний."""
    m3_root = pred_root / "m3"
    if m3_root.is_dir():
        for vdir in sorted(p for p in m3_root.iterdir() if p.is_dir()):
            yield "m3", vdir.name, vdir
    m6_root = pred_root / "m6"
    if m6_root.is_dir():
        yield "m6", "m6", m6_root


def _collect_predictions(pred_root: Path, data_root: Path) -> pd.DataFrame:
    gold = _load_gold(data_root)
    rows: list[dict] = []
    for method, variant, vdir in _iter_pred_dirs(pred_root):
        for jf in sorted(vdir.glob("*.jsonl")):
            name = jf.name
            if _is_smoke(name):
                continue
            split = name[:-len(".jsonl")]
            if split not in _RUN_SPLITS:
                continue
            gold_split = gold.get(split, {})
            for rec in _read_jsonl(jf):
                meta = rec.get("meta") or {}
                raw = str(meta.get("raw", ""))[:400]
                g = gold_split.get(str(rec.get("id")), {})
                rows.append({
                    "id": rec.get("id"),
                    "method": method, "variant": variant, "split": split,
                    "p_faith": rec.get("p_faith"), "p_rel": rec.get("p_rel"),
                    "extract_method": meta.get("method"),
                    "meta_raw": raw,
                    "faith": g.get("faith"), "rel": g.get("rel"),
                    "kind": g.get("kind"), "markers": g.get("markers"),
                })
    if not rows:
        return _empty(_PRED_COLS)
    return pd.DataFrame(rows, columns=list(_PRED_COLS))


# ---------------------------------------------------------------------------
# m6_features
# ---------------------------------------------------------------------------
def _collect_m6_features(art_root: Path) -> pd.DataFrame:
    cols = ["id", "split", "feature_set", *_M6_FEATURE_KEYS]
    rows: list[dict] = []
    if art_root.is_dir():
        for fdir in sorted(p for p in art_root.iterdir()
                           if p.is_dir() and p.name.startswith("m6_features")):
            for split in _RUN_SPLITS:
                jf = fdir / f"{split}.jsonl"
                if not jf.is_file():
                    continue
                for rec in _read_jsonl(jf):
                    rows.append({
                        "id": rec.get("id"), "split": split, "feature_set": fdir.name,
                        **{k: rec.get(k) for k in _M6_FEATURE_KEYS},
                    })
    if not rows:
        return _empty(cols)
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# gepa
# ---------------------------------------------------------------------------
def _collect_gepa(art_root: Path) -> pd.DataFrame:
    cols = ["variant", "seed", "cand_idx", "val_score", "is_best",
            "task_lm_calls", "reflection_lm_calls", "pred_dir"]
    rows: list[dict] = []
    if art_root.is_dir():
        for sf in sorted(art_root.glob("m3_gepa_stats_*.json")):
            d = _read_json(sf)
            if not isinstance(d, dict):
                continue
            variant = d.get("variant")
            seed = d.get("seed")
            dr = d.get("detailed_results") or {}
            cands = dr.get("candidates") or []
            scores = dr.get("val_aggregate_scores") or []
            best_idx = dr.get("best_idx")
            for i in range(len(cands)):
                rows.append({
                    "variant": variant, "seed": seed, "cand_idx": i,
                    "val_score": scores[i] if i < len(scores) else None,
                    "is_best": (i == best_idx),
                    "task_lm_calls": d.get("task_lm_calls"),
                    "reflection_lm_calls": d.get("reflection_lm_calls"),
                    "pred_dir": stats_to_pred_dir(variant, seed),
                })
    if not rows:
        return _empty(cols)
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# entropy_ablation
# ---------------------------------------------------------------------------
def _parse_thr_n(key: str) -> tuple[float | None, int | None]:
    """Разбирает ключ вида `thr=0.3,n=5` в (0.3, 5)."""
    thr: float | None = None
    n: int | None = None
    try:
        for part in key.split(","):
            k, _, v = part.partition("=")
            k = k.strip()
            if k == "thr":
                thr = float(v)
            elif k == "n":
                n = int(v)
    except ValueError:
        return None, None
    return thr, n


def _collect_entropy_ablation(art_root: Path) -> pd.DataFrame:
    cols = ["split", "thr", "n", "delta_semantic_entropy", "delta_n_clusters"]
    rows: list[dict] = []
    if art_root.is_dir():
        for af in sorted(art_root.glob("m6_entropy_ablation_*.json")):
            d = _read_json(af)
            if not isinstance(d, dict):
                continue
            split = d.get("split")
            results = d.get("results") or {}
            for key, val in results.items():
                if not isinstance(val, dict):
                    continue
                thr, n = _parse_thr_n(key)
                rows.append({
                    "split": split, "thr": thr, "n": n,
                    "delta_semantic_entropy": val.get("delta_semantic_entropy"),
                    "delta_n_clusters": val.get("delta_n_clusters"),
                })
    if not rows:
        return _empty(cols)
    return pd.DataFrame(rows, columns=cols)


def build_index(root: str | Path = ".") -> ResultsIndex:
    """Собирает все артефакты прогонов из `root` в набор DataFrame'ов."""
    root = Path(root)
    pred_root = root / "predictions" / "cloud"
    data_root = root / "data" / "processed"
    art_root = root / "artifacts" / "cloud"
    return ResultsIndex(
        runs=_collect_runs(pred_root),
        predictions=_collect_predictions(pred_root, data_root),
        m6_features=_collect_m6_features(art_root),
        gepa=_collect_gepa(art_root),
        entropy_ablation=_collect_entropy_ablation(art_root),
    )
