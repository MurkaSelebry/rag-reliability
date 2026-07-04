"""Сигналы работоспособности пайплайнов на псевдо-корпусе (docs/07.3 п.4).

Не качество, а «пайплайн жив»: средние p_faith/p_rel и фичи m6 по типам кейсов.
Ожидания: clean -> в основном PASS/PASS; hallucination -> p_faith ниже clean;
off_topic -> p_rel ниже clean; у m6 на hallucination entropy/contra выше clean.

Запуск:
  python -m tools.check_signals --config configs/config.cloud.yaml --split val \
      --m3-pred predictions/cloud/m3/zero_shot/val.jsonl \
      --m6-features artifacts/cloud/m6_features/val.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from src.common.config import load_config
from src.common.schemas import load_cases


def group_means(values: dict[str, dict[str, float]], by_kind: dict[str, str],
                fields: list[str]) -> dict[str, dict[str, float]]:
    """Средние значений полей по kind; kind без данных пропускается."""
    acc: dict[str, list[dict]] = defaultdict(list)
    for cid, v in values.items():
        if cid in by_kind:
            acc[by_kind[cid]].append(v)
    return {k: {f: sum(x[f] for x in xs) / len(xs) for f in fields}
            for k, xs in acc.items()}


def _load_jsonl(path: str) -> dict[str, dict]:
    return {d["id"]: d for l in open(path, encoding="utf-8") if (d := json.loads(l))}


def _print_table(title: str, means: dict[str, dict[str, float]], fields: list[str]) -> None:
    print(f"\n== {title}")
    print(f"{'kind':<20}" + "".join(f"{f:>24}" for f in fields))
    for kind in ("clean", "hallucination", "incomplete_answer", "off_topic_answer"):
        if kind in means:
            print(f"{kind:<20}" + "".join(f"{means[kind][f]:>24.3f}" for f in fields))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.cloud.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--m3-pred", default=None)
    ap.add_argument("--m6-features", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cases = load_cases(cfg["data"][args.split])
    by_kind = {c.id: c.meta.get("kind", "?") for c in cases}

    if args.m3_pred:
        preds = _load_jsonl(args.m3_pred)
        m = group_means(preds, by_kind, ["p_faith", "p_rel"])
        _print_table(f"m3 ({args.m3_pred})", m, ["p_faith", "p_rel"])
        if {"clean", "hallucination"} <= m.keys():
            d = m["clean"]["p_faith"] - m["hallucination"]["p_faith"]
            print(f"сигнал faith (clean - hallucination): {d:+.3f} "
                  + ("OK" if d > 0 else "!! нет сигнала"))
        if {"clean", "off_topic_answer"} <= m.keys():
            d = m["clean"]["p_rel"] - m["off_topic_answer"]["p_rel"]
            print(f"сигнал rel (clean - off_topic):       {d:+.3f} "
                  + ("OK" if d > 0 else "!! нет сигнала"))
        if {"clean", "incomplete_answer"} <= m.keys():
            d = m["clean"]["p_faith"] - m["incomplete_answer"]["p_faith"]
            status = "OK" if d > 0.2 else ("слабый — ожидаемо трудный тип" if d > 0 else "!! нет сигнала")
            print(f"сигнал faith (clean - incomplete):    {d:+.3f} {status}")

    if args.m6_features:
        feats = _load_jsonl(args.m6_features)
        fields = ["selfcheck_contra_mean", "semantic_entropy", "n_clusters", "cos_q_a"]
        m = group_means(feats, by_kind, fields)
        _print_table(f"m6 features ({args.m6_features})", m, fields)
        if {"clean", "hallucination"} <= m.keys():
            d_se = m["hallucination"]["semantic_entropy"] - m["clean"]["semantic_entropy"]
            d_c = m["hallucination"]["selfcheck_contra_mean"] - m["clean"]["selfcheck_contra_mean"]
            print(f"сигнал m6 (halluc - clean): entropy {d_se:+.3f}, contra {d_c:+.3f} "
                  + ("OK" if (d_se > 0 or d_c > 0) else "!! нет сигнала"))
        if {"clean", "incomplete_answer"} <= m.keys():
            d_c = m["incomplete_answer"]["selfcheck_contra_mean"] - m["clean"]["selfcheck_contra_mean"]
            d_se = m["incomplete_answer"]["semantic_entropy"] - m["clean"]["semantic_entropy"]
            print(f"инфо m6 (incomplete - clean): contra {d_c:+.3f}, entropy {d_se:+.3f} "
                  "(consistency слепа к неполноте — ожидаемо, H4)")


if __name__ == "__main__":
    main()
