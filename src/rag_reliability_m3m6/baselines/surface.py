"""Локальные не-LLM бейзлайны: majority, logreg на поверхностных фичах, +e5-косинусы.

Первые «честные» строки таблицы для реального корпуса. Никаких LLM-вызовов;
единственная тяжёлая зависимость — sentence-transformers для e5 (импорт лениво).

CLI: python scripts/m3m6/run_surface_baseline.py --config configs/config.yaml
     [--no-embeddings] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rag_reliability_m3m6.common.config import load_config
from rag_reliability_m3m6.common.eval_local import evaluate, fit_thresholds
from rag_reliability_m3m6.common.run_meta import save_run_yaml
from rag_reliability_m3m6.common.schemas import Case, Pred, load_cases, save_preds

_WORD_RE = re.compile(r"\w+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# URL/навигационные паттерны — прокси для маркеров wrong_navigation/answer_for_operator
_NAV_RE = re.compile(
    r"https?://|www\.|в\s+приложени\w*|в\s+раздел\w*|по\s+ссылк\w*|стать[еия]\w*",
    re.IGNORECASE,
)

# Фиксированный порядок фич для матрицы (детерминизм)
_FEATURE_KEYS = [
    "len_answer",
    "len_ctx",
    "len_query",
    "n_chunks",
    "overlap_ans_ctx_1",
    "overlap_ans_ctx_2",
    "overlap_ans_q_1",
    "digit_match_ratio",
    "url_or_nav_ratio",
]

_OUT_ROOT = Path("predictions/local/baselines")
_EMB_CACHE_DIR = Path("artifacts/alfa_emb")


def _tokens(s: str) -> list[str]:
    """Токенизация: lower + все \\w+ последовательности."""
    return _WORD_RE.findall(s.lower())


def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    """Доля словных n-грамм строки a, встречающихся среди n-грамм строки b.

    Если в a меньше n слов — откат к n=1; пустая a → 0.0. Результат в [0, 1].
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    if len(ta) < n:
        n = 1
    a_ngrams = [tuple(ta[i : i + n]) for i in range(len(ta) - n + 1)]
    b_ngrams = {tuple(tb[i : i + n]) for i in range(len(tb) - n + 1)} if len(tb) >= n else set()
    hits = sum(1 for g in a_ngrams if g in b_ngrams)
    return hits / max(1, len(a_ngrams))


def _norm_num(s: str) -> str:
    """Нормализация числа как строки: запятая → точка, без хвостовых нулей дроби."""
    s = s.replace(",", ".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def surface_features(case: Case) -> dict[str, float]:
    """Поверхностные фичи кейса: длины, n-грамм overlap, совпадение чисел, URL/навигация."""
    ans_t = _tokens(case.answer)
    q_t = _tokens(case.query)
    ctx = " ".join(case.context)
    ctx_t = _tokens(ctx)

    ans_nums = [_norm_num(m) for m in _NUM_RE.findall(case.answer)]
    ctx_nums = {_norm_num(m) for m in _NUM_RE.findall(ctx)}
    if ans_nums:
        digit_ratio = sum(1 for x in ans_nums if x in ctx_nums) / max(1, len(ans_nums))
    else:
        digit_ratio = 1.0  # нет чисел в ответе — нечему противоречить

    nav_hits = len(_NAV_RE.findall(case.answer))
    return {
        "len_answer": float(len(ans_t)),
        "len_ctx": float(len(ctx_t)),
        "len_query": float(len(q_t)),
        "n_chunks": float(len(case.context)),
        "overlap_ans_ctx_1": ngram_overlap(case.answer, ctx, n=1),
        "overlap_ans_ctx_2": ngram_overlap(case.answer, ctx, n=2),
        "overlap_ans_q_1": ngram_overlap(case.answer, case.query, n=1),
        "digit_match_ratio": float(digit_ratio),
        "url_or_nav_ratio": nav_hits / max(1, len(ans_t)),
    }


class E5Embedder:
    """Ленивая обёртка e5 с пофайловым кэшем косинусов в artifacts/alfa_emb/{id}.json."""

    def __init__(self, model_name: str, cache_dir: str | Path = _EMB_CACHE_DIR) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            # тяжёлый импорт — только при реальном вычислении (кэш-хиты бесплатны)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, normalize_embeddings=True)

    def cosines(self, case: Case) -> tuple[float, float]:
        """(cos(query, answer), cos(ctx, answer)); повторный вызов читает кэш."""
        path = self.cache_dir / f"{case.id}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return float(d["cos_q_ans"]), float(d["cos_ctx_ans"])
        q, ans, ctx = self._encode(
            [
                f"query: {case.q_text()}",
                f"passage: {case.answer}",
                f"passage: {case.ctx_text()}",
            ]
        )
        cos_q, cos_c = float(np.dot(q, ans)), float(np.dot(ctx, ans))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cos_q_ans": cos_q, "cos_ctx_ans": cos_c}), encoding="utf-8")
        os.replace(tmp, path)  # атомарная запись
        return cos_q, cos_c


def _matrix(cases: list[Case], embedder: E5Embedder | None = None) -> np.ndarray:
    """Матрица фич в фиксированном порядке; +2 косинуса e5, если embedder задан."""
    rows: list[list[float]] = []
    for c in cases:
        f = surface_features(c)
        row = [f[k] for k in _FEATURE_KEYS]
        if embedder is not None:
            row.extend(embedder.cosines(c))
        rows.append(row)
    if not rows:
        return np.zeros((0, len(_FEATURE_KEYS)))
    return np.asarray(rows, dtype=float)


def _mean_label(labels: list[int | None]) -> float:
    vals = [v for v in labels if v is not None]
    return float(np.mean(vals)) if vals else 0.5


def _fit_head(x: np.ndarray, y: np.ndarray, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Одна голова (faith или rel): scaler + logreg; при одном классе — константа."""
    classes = set(y.tolist())
    if len(classes) < 2:
        const = float(next(iter(classes))) if classes else 0.5
        return lambda xn: np.full(len(xn), const)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(x), y)
    return lambda xn: clf.predict_proba(scaler.transform(xn))[:, 1]


def build_detector(
    name: str, train: list[Case], embedder: E5Embedder | None, seed: int
) -> Callable[[list[Case]], list[Pred]]:
    """Обучает детектор на train и возвращает функцию cases → preds (две головы)."""
    meta = {"variant": name, "heads": "two"}

    if name == "majority":
        p_f = _mean_label([c.faith for c in train])
        p_r = _mean_label([c.rel for c in train])

        def predict_const(cases: list[Case]) -> list[Pred]:
            return [Pred(id=c.id, p_faith=p_f, p_rel=p_r, meta=dict(meta)) for c in cases]

        return predict_const

    labeled = [c for c in train if c.faith is not None and c.rel is not None]
    x = _matrix(labeled, embedder)
    head_f = _fit_head(x, np.asarray([c.faith for c in labeled]), seed)
    head_r = _fit_head(x, np.asarray([c.rel for c in labeled]), seed)

    def predict(cases: list[Case]) -> list[Pred]:
        xn = _matrix(cases, embedder)
        pf, pr = head_f(xn), head_r(xn)
        return [
            Pred(id=c.id, p_faith=float(a), p_rel=float(b), meta=dict(meta))
            for c, a, b in zip(cases, pf, pr)
        ]

    return predict


def _maybe_track(cfg: dict, name: str, split: str, metrics: dict) -> None:
    """MLflow-лог, если включён tracking; любая ошибка не валит прогон."""
    if not cfg.get("tracking", {}).get("enabled"):
        return
    try:
        from rag_reliability_m3m6.common.tracking import log_run

        log_run(cfg["tracking"]["uri"], "alfa", f"{name}/{split}", cfg, metrics)
    except Exception as e:  # noqa: BLE001 — трекинг не должен ронять прогон
        print(f"[tracking] пропущено ({name}/{split}): {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Не-LLM бейзлайны на поверхностных фичах")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-embeddings", action="store_true", help="без варианта surface_e5")
    parser.add_argument("--limit", type=int, default=None, help="обрезать каждый сплит до N")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = {s: load_cases(cfg["data"][s]) for s in ("train", "val", "test")}
    if args.limit:
        splits = {s: cs[: args.limit] for s, cs in splits.items()}
    seed = int(cfg.get("alfa", {}).get("seed", 42))

    variants = ["majority", "surface"] + ([] if args.no_embeddings else ["surface_e5"])
    embedder = E5Embedder(cfg["m6"]["embed_model"]) if "surface_e5" in variants else None

    summary: list[tuple[str, float, float]] = []
    for name in variants:
        predict = build_detector(
            name, splits["train"], embedder if name == "surface_e5" else None, seed
        )
        out_dir = _OUT_ROOT / name
        preds_by_split: dict[str, list[Pred]] = {}
        for split in ("val", "test"):
            preds = predict(splits[split])
            save_preds(preds, out_dir / f"{split}.jsonl")
            save_run_yaml(out_dir, cfg, split=split, method="baseline", variant=name)
            preds_by_split[split] = preds

        t_faith, t_rel, _ = fit_thresholds(splits["val"], preds_by_split["val"])
        reports: dict[str, dict] = {}
        for split in ("val", "test"):
            rep = evaluate(splits[split], preds_by_split[split], t_faith, t_rel)
            (out_dir / f"report_{split}.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            reports[split] = rep
            _maybe_track(cfg, name, split, rep)
        summary.append(
            (name, reports["val"]["f1_macro_reliable"], reports["test"]["f1_macro_reliable"])
        )

    print(f"{'variant':<12} {'f1_rel(val)':>12} {'f1_rel(test)':>13}")
    for name, f1_val, f1_test in summary:
        print(f"{name:<12} {f1_val:>12.4f} {f1_test:>13.4f}")


if __name__ == "__main__":
    main()
