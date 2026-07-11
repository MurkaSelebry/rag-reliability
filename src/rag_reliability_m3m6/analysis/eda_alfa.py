"""EDA реального корпуса кураторов (data/raw/alfa/data.csv) для docs/12.

Ключевые числа пишутся в artifacts/alfa_eda.json (единственный источник
правды — docs/12 пишется ПО этому json, не руками), фигуры — в
artifacts/figs/alfa_*.png, человекочитаемые таблицы — в stdout.
Полные тексты диалогов в stdout не печатаются (данные конфиденциальные).

Запуск:
  python scripts/m3m6/eda_alfa.py --config configs/config.yaml [--limit N] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_reliability_m3m6.common.config import load_config
from rag_reliability_m3m6.common.schemas import Case, load_cases
from rag_reliability_m3m6.data.alfa_loader import load_alfa
from rag_reliability_m3m6.data.make_splits import group_key

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PSEUDO_FILES = [
    "data/processed/pseudo_dev_train.jsonl",
    "data/processed/pseudo_dev_val.jsonl",
    "data/processed/pseudo_dev_test.jsonl",
]
# лимиты контекста для вердиктов «влезает ли кейс»
LIMIT_16K = 16384  # vLLM max-model-len (минус 512 на генерацию)
LIMIT_8K = 8096  # RuModernBERT
GEN_RESERVE = 512


def _get_token_counter() -> tuple[Callable[[str], int], str]:
    """Токенайзер Qwen, если доступен; иначе оценка словами ×1.5."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(QWEN_MODEL)

        def count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        return count, "qwen_tokenizer"
    except Exception as exc:  # нет сети/кэша — деградируем в оценку по словам
        print(f"[warn] токенайзер Qwen недоступен ({exc}); оценка словами ×1.5")

        def count(text: str) -> int:
            return int(len(text.split()) * 1.5)

        return count, "fallback_words"


def _pcts(values: list[int]) -> dict:
    """p50/p95/max для списка длин."""
    arr = np.asarray(values)
    return {
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def _share(n: int, total: int) -> float:
    return round(n / total, 4) if total else 0.0


def _balance(cases: list[Case]) -> dict:
    """Секция 1: баланс меток и таблица 2×2 faith×rel."""
    n = len(cases)
    cells = Counter((c.faith, c.rel) for c in cases)
    return {
        "n": n,
        "rel1_share": _share(sum(c.rel == 1 for c in cases), n),
        "faith1_share": _share(sum(c.faith == 1 for c in cases), n),
        "reliable_share": _share(sum(c.reliable == 1 for c in cases), n),
        "cells": {
            "faith1_rel1": cells[(1, 1)],
            "faith1_rel0": cells[(1, 0)],  # реальный аналог off_topic
            "faith0_rel1": cells[(0, 1)],
            "faith0_rel0": cells[(0, 0)],
        },
    }


def _markers(cases: list[Case]) -> dict:
    """Секция 2: частоты маркеров, аномалии, привязка маркеров к осям."""
    freq = Counter(m for c in cases for m in c.markers)
    anomaly = [{"id": c.id, "markers": c.markers} for c in cases if c.reliable == 1 and c.markers]
    marker_axis: dict[str, dict] = {}
    for m in freq:
        with_m = [c for c in cases if m in c.markers]
        marker_axis[m] = {
            "n": len(with_m),
            "faith0_share": _share(sum(c.faith == 0 for c in with_m), len(with_m)),
            "rel0_share": _share(sum(c.rel == 0 for c in with_m), len(with_m)),
        }
    return {
        "freq": dict(freq.most_common()),
        "n_cases_with_markers": sum(1 for c in cases if c.markers),
        "n_unreliable": sum(1 for c in cases if c.reliable == 0),
        "n_unreliable_with_markers": sum(1 for c in cases if c.reliable == 0 and c.markers),
        "anomaly_markers_reliable1": anomaly,
        "marker_axis": marker_axis,
    }


def _lengths(cases: list[Case], count: Callable[[str], int], method: str) -> tuple[dict, list]:
    """Секция 3: длины в токенах (dialog/context/answer/total) и вердикты."""
    parts = {"dialog": [], "context": [], "answer": [], "total": []}
    for c in cases:
        d = count("\n".join(c.dialog))
        ctx = count("\n".join(c.context))
        a = count(c.answer)
        parts["dialog"].append(d)
        parts["context"].append(ctx)
        parts["answer"].append(a)
        parts["total"].append(d + ctx + a)
    totals = parts["total"]
    section = {
        "token_len_method": method,
        **{k: _pcts(v) for k, v in parts.items()},
        "fits_16k_share": _share(sum(t + GEN_RESERVE <= LIMIT_16K for t in totals), len(totals)),
        "fits_8k_share": _share(sum(t + GEN_RESERVE <= LIMIT_8K for t in totals), len(totals)),
    }
    return section, totals


def _duplicates(cases: list[Case]) -> dict:
    """Секция 4: точные дубликаты, группы near-dup по запросу, шаблонные приветствия."""
    exact = Counter(("\n".join(c.dialog), c.answer) for c in cases)
    n_exact_dups = sum(n - 1 for n in exact.values() if n > 1)

    groups = Counter(group_key(c) for c in cases)
    multi = {k: n for k, n in groups.items() if n > 1}
    top = sorted(multi.items(), key=lambda kv: -kv[1])[:5]

    template_n = sum(
        1
        for c in cases
        if c.dialog and c.dialog[0].startswith("Ассистент:") and "Альфа-Помощник" in c.dialog[0]
    )
    return {
        "exact_dup_extra_rows": n_exact_dups,
        "n_query_groups_gt1": len(multi),
        "top_query_groups": [{"key_prefix": k[:60], "n": n} for k, n in top],
        "template_greeting_n": template_n,
    }


def _pseudo_compare(alfa: list[Case], pseudo: list[Case]) -> dict:
    """Секция 6: сравнение метрик альфа vs псевдо-корпус."""

    def block(cs: list[Case]) -> dict:
        n = len(cs)
        return {
            "n": n,
            "faith1": _share(sum(c.faith == 1 for c in cs), n),
            "rel1": _share(sum(c.rel == 1 for c in cs), n),
            "reliable": _share(sum(c.reliable == 1 for c in cs), n),
            "answer_words_p50": int(np.median([len(c.answer.split()) for c in cs])) if n else 0,
        }

    return {"alfa": block(alfa), "pseudo": block(pseudo)}


def _print_report(r: dict) -> None:
    """Компактные таблицы в stdout (только числа и id, без текстов)."""
    b = r["balance"]
    print(f"\n=== 1. Баланс (n={b['n']}) ===")
    print(
        f"rel=1: {b['rel1_share']:.3f}  faith=1: {b['faith1_share']:.3f}  "
        f"reliable: {b['reliable_share']:.3f}"
    )
    c = b["cells"]
    print("2x2 faith×rel:            rel=1   rel=0")
    print(f"  faith=1              {c['faith1_rel1']:6d}  {c['faith1_rel0']:6d}")
    print(f"  faith=0              {c['faith0_rel1']:6d}  {c['faith0_rel0']:6d}")
    print(f"  (faith=1, rel=0) = {c['faith1_rel0']} — реальный аналог off_topic")

    m = r["markers"]
    print(f"\n=== 2. Маркеры (кейсов с маркерами: {m['n_cases_with_markers']}) ===")
    print(f"{'маркер':38s} {'n':>4s} {'faith0':>7s} {'rel0':>6s}")
    for name, n in m["freq"].items():
        ax = m["marker_axis"][name]
        print(f"{name:38s} {n:4d} {ax['faith0_share']:7.2f} {ax['rel0_share']:6.2f}")
    print(f"среди unreliable ({m['n_unreliable']}): с маркерами {m['n_unreliable_with_markers']}")
    anom = m["anomaly_markers_reliable1"]
    print(f"АНОМАЛИЯ: маркеры при reliable=1 — {len(anom)} кейсов:")
    for a in anom:
        print(f"  {a['id']}: {', '.join(a['markers'])}")

    ln = r["lengths"]
    print(f"\n=== 3. Длины в токенах ({ln['token_len_method']}) ===")
    print(f"{'поле':8s} {'p50':>6s} {'p95':>6s} {'max':>7s}")
    for k in ("dialog", "context", "answer", "total"):
        print(f"{k:8s} {ln[k]['p50']:6d} {ln[k]['p95']:6d} {ln[k]['max']:7d}")
    print(f"влезает в 16k (total+512): {ln['fits_16k_share']:.3f}")
    print(f"влезает в 8k  (total+512): {ln['fits_8k_share']:.3f}")

    d = r["duplicates"]
    print("\n=== 4. Дубликаты ===")
    print(f"точных дублей (лишних строк по (dialog, answer)): {d['exact_dup_extra_rows']}")
    print(f"групп near-dup по нормализованному запросу (>1): {d['n_query_groups_gt1']}")
    print("топ-5 групп:")
    for g in d["top_query_groups"]:
        print(f"  n={g['n']:3d}  «{g['key_prefix']}…»")
    print(f"шаблонное приветствие «Альфа-Помощник» первой строкой: {d['template_greeting_n']}")

    print(f"\n=== 5. Кейсы без реплики клиента: {len(r['no_client_turn_ids'])} ===")
    print("  " + ", ".join(r["no_client_turn_ids"]))

    p = r["pseudo_compare"]
    print("\n=== 6. Сравнение с псевдо-корпусом ===")
    print(f"{'':14s} {'alfa':>8s} {'pseudo':>8s}")
    for k in ("n", "faith1", "rel1", "reliable", "answer_words_p50"):
        print(f"{k:14s} {p['alfa'][k]:8} {p['pseudo'][k]:8}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA реального корпуса кураторов (docs/12)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="первые N кейсов (smoke)")
    ap.add_argument("--out", default="artifacts/alfa_eda.json")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    cases = load_alfa(cfg["alfa"]["raw_csv"])
    if args.limit:
        cases = cases[: args.limit]

    pseudo: list[Case] = []
    for path in PSEUDO_FILES:
        pseudo.extend(load_cases(path))
    if args.limit:
        pseudo = pseudo[: args.limit]

    count, method = _get_token_counter()
    lengths, totals = _lengths(cases, count, method)

    report = {
        "n": len(cases),
        "balance": _balance(cases),
        "markers": _markers(cases),
        "lengths": lengths,
        "duplicates": _duplicates(cases),
        "no_client_turn_ids": [c.id for c in cases if c.meta.get("no_client_turn")],
        "pseudo_compare": _pseudo_compare(cases, pseudo),
    }
    # плоские вердикты-дубликаты на верхнем уровне (по ним пишется docs/12)
    report["token_len_method"] = method
    report["fits_16k_share"] = lengths["fits_16k_share"]
    report["fits_8k_share"] = lengths["fits_8k_share"]
    report["anomaly_markers_reliable1"] = report["markers"]["anomaly_markers_reliable1"]
    report["marker_axis"] = report["markers"]["marker_axis"]
    report["top_query_groups"] = report["duplicates"]["top_query_groups"]
    report["template_greeting_n"] = report["duplicates"]["template_greeting_n"]

    _print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    figs_dir = Path("artifacts/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    # (a) частоты маркеров — горизонтальный бар
    freq = report["markers"]["freq"]
    names = list(freq)[::-1]  # most_common → снизу вверх по убыванию
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, [freq[m] for m in names])
    ax.set_xlabel("число кейсов")
    ax.set_title(f"Частоты маркеров reason_* (alfa, n={len(cases)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs_dir / "alfa_marker_freq.png", dpi=150)
    plt.close(fig)

    # (b) гистограмма суммарных токенов
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(totals, bins=50)
    ax.axvline(LIMIT_16K - GEN_RESERVE, color="red", ls="--", label="16k − 512")
    ax.axvline(LIMIT_8K - GEN_RESERVE, color="orange", ls="--", label="8k − 512")
    ax.set_xlabel(f"токены dialog+context+answer ({method})")
    ax.set_ylabel("число кейсов")
    ax.set_title(f"Длина кейса в токенах (alfa, n={len(cases)})", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "alfa_len_tokens.png", dpi=150)
    plt.close(fig)

    # (c) длины ответов в словах: alfa vs pseudo
    alfa_words = [len(c.answer.split()) for c in cases]
    pseudo_words = [len(c.answer.split()) for c in pseudo]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.histogram_bin_edges(alfa_words + pseudo_words, bins=40)
    ax.hist(alfa_words, bins=bins, alpha=0.6, density=True, label=f"alfa (n={len(cases)})")
    ax.hist(pseudo_words, bins=bins, alpha=0.6, density=True, label=f"pseudo (n={len(pseudo)})")
    ax.set_xlabel("длина ответа, слов")
    ax.set_ylabel("плотность")
    ax.set_title("Длина ответа: alfa vs псевдо-корпус", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "alfa_vs_pseudo_answer_len.png", dpi=150)
    plt.close(fig)

    print(f"\njson: {out_path}")
    print(
        f"фигуры: {figs_dir}/alfa_marker_freq.png, alfa_len_tokens.png, "
        "alfa_vs_pseudo_answer_len.png"
    )


if __name__ == "__main__":
    main()
