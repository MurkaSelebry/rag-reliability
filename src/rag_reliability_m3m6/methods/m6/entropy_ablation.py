"""Абляция semantic entropy по порогу entailment и числу сэмплов N — из кэша
сэмплов Метода 6, без единого LLM-вызова. NLI-матрица считается один раз на кейс
(для max N), кластеры при всех (thr, N) — срезами этой матрицы.

Запуск:
  python scripts/m3m6/entropy_ablation.py --config configs/config.cloud.yaml --split val \
      --thresholds 0.3 0.4 0.5 --ns 3 5 10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag_reliability_m3m6.common.config import load_config
from rag_reliability_m3m6.common.schemas import load_cases


def _entropy_from_labels(labels: list[int]) -> dict:
    uniq, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return {
        "semantic_entropy": float(-(p * np.log(p)).sum()),
        "n_clusters": int(len(uniq)),
        "answer_in_top_cluster": float(labels[0] == uniq[counts.argmax()]),
    }


def cluster_features_multi(
    answer: str, samples: list[str], nli, thresholds: list[float], ns: list[int]
) -> dict:
    """Фичи кластеров при каждом (thr, n): один батч NLI на все пары [answer]+samples.

    -> {(thr, n): {semantic_entropy, n_clusters, answer_in_top_cluster}}
    """
    n_max = min(max(ns), len(samples))
    texts = [answer] + list(samples[:n_max])
    pairs, idx = [], []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs += [(texts[i], texts[j]), (texts[j], texts[i])]
            idx.append((i, j))
    res = nli.score(pairs) if pairs else []
    entail = {ij: (res[2 * k]["entail"], res[2 * k + 1]["entail"]) for k, ij in enumerate(idx)}

    out: dict[tuple[float, int], dict] = {}
    for thr in thresholds:
        for n in ns:
            size = 1 + min(n, len(samples))
            parent = list(range(size))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for (i, j), (e1, e2) in entail.items():
                if i < size and j < size and e1 > thr and e2 > thr:
                    parent[find(i)] = find(j)
            out[(thr, n)] = _entropy_from_labels([find(i) for i in range(size)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="абляция энтропии по (thr, N) из кэша сэмплов")
    ap.add_argument("--config", default="configs/config.cloud.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    ap.add_argument("--ns", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--limit", type=int, default=None, help="smoke: первые N кейсов")
    ap.add_argument(
        "--by",
        choices=["kind", "faith"],
        default="kind",
        help="группировка: kind (псевдо-корпус) или метка faith (реальный корпус)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    m6 = cfg["m6"]
    cases = load_cases(cfg["data"][args.split])
    if args.limit is not None:
        cases = cases[: args.limit]
    cache = Path(m6["samples_cache"]) / args.split

    from rag_reliability_m3m6.methods.m6.nli import NLIScorer  # ленивый тяжёлый импорт (torch)

    nli = NLIScorer(m6["nli_model"])

    from tqdm import tqdm

    per_kind: dict[tuple[float, int], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_cases = 0
    for c in tqdm(cases, desc=f"ablation/{args.split}"):
        f = cache / f"{c.id}.json"
        if not f.exists():
            print(f"!! {c.id}: нет кэша сэмплов ({f}) — пропуск")
            continue
        samples = json.loads(f.read_text(encoding="utf-8"))["samples"]
        feats = cluster_features_multi(c.answer, samples, nli, args.thresholds, args.ns)
        if args.by == "faith":
            kind = "hallucination" if c.faith == 0 else "clean"  # faith0≈halluc, faith1≈clean
        else:
            kind = c.meta.get("kind", "?")
        for key, v in feats.items():
            per_kind[key][kind].append(v)
        n_cases += 1

    # таблица Δ (hallucination − clean) по каждой (thr, N)
    fields = ["semantic_entropy", "n_clusters"]
    result: dict[str, dict] = {}
    print(f"\nΔ (hallucination − clean), {n_cases} кейсов, split={args.split}")
    print(
        f"{'thr':>5} {'N':>3}"
        + "".join(f"{'Δ ' + f:>22}" for f in fields)
        + "".join(f"{'clean ' + f:>22}" for f in fields)
    )
    for thr in args.thresholds:
        for n in args.ns:
            groups = per_kind[(thr, n)]
            means = {
                k: {f: float(np.mean([x[f] for x in xs])) for f in fields}
                for k, xs in groups.items()
            }
            row: dict = {"means_by_kind": means}
            line = f"{thr:>5.2f} {n:>3}"
            for f in fields:
                if {"clean", "hallucination"} <= means.keys():
                    d = means["hallucination"][f] - means["clean"][f]
                    row[f"delta_{f}"] = d
                    line += f"{d:>+22.3f}"
                else:
                    line += f"{'n/a':>22}"
            for f in fields:
                line += f"{means['clean'][f]:>22.3f}" if "clean" in means else f"{'n/a':>22}"
            print(line)
            result[f"thr={thr},n={n}"] = row

    out_path = Path(m6["samples_cache"]).parent / f"m6_entropy_ablation_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "n_cases": n_cases,
                "thresholds": args.thresholds,
                "ns": args.ns,
                "results": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\njson: {out_path}")


if __name__ == "__main__":
    main()
