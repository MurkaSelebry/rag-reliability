"""Метод 6, этап 2: фичи консистентности из кэша сэмплов.

Фичи на кейс:
  selfcheck_contra_mean / _max — SelfCheckGPT-NLI: средняя/максимальная по
      предложениям ответа вероятность противоречия сэмплам;
  semantic_entropy — энтропия кластеров сэмплов (двунаправленный entailment);
  n_clusters — число семантических кластеров среди сэмплов;
  answer_in_top_cluster — принадлежит ли исходный ответ крупнейшему кластеру;
  cos_q_a — косинус эмбеддингов вопроса и ответа (relevance-сигнал).

Запуск:
  python -m src.m6_selfcheck.features --config configs/config.yaml --split val
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
from razdel import sentenize
from tqdm import tqdm

from ..common.schemas import load_cases, load_yaml


def sentences(text: str) -> list[str]:
    return [s.text.strip() for s in sentenize(text) if s.text.strip()] or [text]


def selfcheck_scores(answer: str, samples: list[str], nli) -> dict:
    sents = sentences(answer)
    pairs = [(smp, sent) for sent in sents for smp in samples]  # premise=сэмпл, hyp=предложение
    res = nli.score(pairs)
    contra = np.array([r["contra"] for r in res]).reshape(len(sents), len(samples))
    per_sent = contra.mean(axis=1)  # средняя противоречивость предложения сэмплам
    return {"selfcheck_contra_mean": float(per_sent.mean()),
            "selfcheck_contra_max": float(per_sent.max())}


def semantic_clusters(texts: list[str], nli, thr: float) -> list[int]:
    """Кластеризация по двунаправленному entailment -> метка кластера каждому тексту."""
    n = len(texts)
    pairs, idx = [], []
    for i in range(n):
        for j in range(i + 1, n):
            pairs += [(texts[i], texts[j]), (texts[j], texts[i])]
            idx.append((i, j))
    res = nli.score(pairs) if pairs else []
    # union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for k, (i, j) in enumerate(idx):
        if res[2 * k]["entail"] > thr and res[2 * k + 1]["entail"] > thr:
            parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def entropy_features(answer: str, samples: list[str], nli, thr: float) -> dict:
    texts = [answer] + samples          # исходный ответ — элемент 0
    labels = semantic_clusters(texts, nli, thr)
    uniq, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    se = float(-(p * np.log(p)).sum())
    top = uniq[counts.argmax()]
    return {"semantic_entropy": se,
            "n_clusters": int(len(uniq)),
            "answer_in_top_cluster": float(labels[0] == top)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    m6 = cfg["m6"]

    from .nli import NLIScorer
    nli = NLIScorer(m6["nli_model"])
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(m6["embed_model"])

    cases = load_cases(cfg["data"][args.split])
    cache = Path(m6["samples_cache"]) / args.split
    out_dir = Path(m6["features_cache"]); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.jsonl"

    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}

    with open(out_path, "a", encoding="utf-8") as fout:
        for c in tqdm(cases, desc=f"m6/features/{args.split}"):
            if c.id in done:
                continue
            samples = json.loads((cache / f"{c.id}.json").read_text(encoding="utf-8"))["samples"]
            feats = {"id": c.id}
            feats.update(selfcheck_scores(c.answer, samples, nli))
            feats.update(entropy_features(c.answer, samples, nli, m6["entail_threshold"]))
            v = emb.encode([f"query: {c.q_text()}", f"passage: {c.answer}"],
                           normalize_embeddings=True)
            feats["cos_q_a"] = float(v[0] @ v[1])
            fout.write(json.dumps(feats, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
