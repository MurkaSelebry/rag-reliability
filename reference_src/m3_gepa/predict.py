"""Метод 3, инференс: судья с выбранным промптом -> predictions/{split}.jsonl.

Режимы (m3.mode в конфиге):
  zero_shot — SEED_INSTRUCTION как есть (ступень 1);
  few_shot  — SEED_INSTRUCTION + рукописные примеры (ступень 2);
  gepa      — инструкция из artifacts/m3_optimized_prompt.txt (ступень 3).

Вероятности PASS снимаются из logprobs через общий JudgeClient — единая
схема с Методами 1–2, что делает H1-сравнение честным.

Запуск:
  python -m src.m3_gepa.predict --config configs/config.yaml --split val
  python -m src.m3_gepa.predict --config configs/config.yaml --split test
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ..common.schemas import load_cases, save_preds, load_yaml, Pred
from ..common.judge_client import JudgeClient
from ..common.eval_local import fit_thresholds, evaluate
from .prompts import SEED_INSTRUCTION, build_user_prompt, build_few_shot_system

FEW_SHOT_EXAMPLES: list[dict] = []  # заполнить 6–8 кейсами из dev-train (см. prompts.py)


def get_system_prompt(cfg) -> str:
    mode = cfg["m3"]["mode"]
    if mode == "zero_shot":
        return SEED_INSTRUCTION
    if mode == "few_shot":
        assert FEW_SHOT_EXAMPLES, "заполни FEW_SHOT_EXAMPLES"
        return build_few_shot_system(FEW_SHOT_EXAMPLES)
    if mode == "gepa":
        return Path(cfg["m3"]["out_prompt"]).read_text(encoding="utf-8")
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    client = JudgeClient(cfg["llm"]["api_base"], cfg["llm"]["api_key"], cfg["llm"]["model"])
    system = get_system_prompt(cfg)
    cases = load_cases(cfg["data"][args.split])

    preds = []
    for c in tqdm(cases, desc=f"m3/{cfg['m3']['mode']}/{args.split}"):
        user = build_user_prompt(c, cfg["llm"]["max_ctx_chars"])
        p_f, p_r, raw = client.judge(system, user)
        preds.append(Pred(id=c.id, p_faith=p_f, p_rel=p_r,
                          meta={"mode": cfg["m3"]["mode"], "raw": raw[-400:]}))

    out_dir = Path(cfg["m3"]["out_pred_dir"]) / cfg["m3"]["mode"]
    save_preds(preds, out_dir / f"{args.split}.jsonl")

    # dev-оценка: пороги на val, метрики на текущем сплите
    if args.split in ("val", "test"):
        val_cases = load_cases(cfg["data"]["val"])
        val_path = out_dir / "val.jsonl"
        if val_path.exists():
            from ..common.schemas import Case  # noqa
            val_preds = [Pred(**json.loads(l)) for l in open(val_path, encoding="utf-8")]
            tf, tr, _ = fit_thresholds(val_cases, val_preds)
            report = evaluate(cases, preds, tf, tr)
            (out_dir / f"report_{args.split}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
