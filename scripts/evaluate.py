#!/usr/bin/env python
"""Локальная оценка predictions против кейсов со снятой разметкой.

Повторяет логику замороженного evaluate.py платформы (через
`rag_reliability.common.eval_local`). Пороги: либо явные --t-faith/--t-rel,
либо подбор сеткой на переданном файле (использовать ТОЛЬКО для val).

Запуск:
  python scripts/evaluate.py --cases data/processed/pseudo_dev_val.jsonl \
      --preds predictions/cloud/m3/zero_shot/val.jsonl --fit
  python scripts/evaluate.py --cases data/processed/pseudo_dev_test.jsonl \
      --preds predictions/cloud/m3/zero_shot/test.jsonl --t-faith 0.62 --t-rel 0.55
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_reliability.common.eval_local import evaluate, fit_thresholds
from rag_reliability.common.schemas import Pred, load_cases


def load_preds(path: str | Path) -> list[Pred]:
    preds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            preds.append(
                Pred(
                    id=str(d["id"]),
                    p_faith=float(d["p_faith"]),
                    p_rel=float(d["p_rel"]),
                    meta=d.get("meta") or {},
                )
            )
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="оценка predictions (f1-macro reliable/faith/rel)")
    ap.add_argument("--cases", required=True, help="jsonl кейсов с разметкой faith/rel")
    ap.add_argument("--preds", required=True, help="jsonl предсказаний {id, p_faith, p_rel}")
    ap.add_argument("--t-faith", type=float, default=None)
    ap.add_argument("--t-rel", type=float, default=None)
    ap.add_argument("--fit", action="store_true", help="подобрать пороги сеткой (только val!)")
    ap.add_argument("--out", default=None, help="куда записать json-отчёт (иначе stdout)")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    preds = load_preds(args.preds)

    if args.fit:
        t_faith, t_rel, _ = fit_thresholds(cases, preds)
    elif args.t_faith is not None and args.t_rel is not None:
        t_faith, t_rel = args.t_faith, args.t_rel
    else:
        raise SystemExit("нужно --fit ИЛИ оба --t-faith и --t-rel")

    report = evaluate(cases, preds, t_faith, t_rel)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
