"""Метод 3, инференс: судья с выбранным промптом -> predictions/{variant}/{split}.jsonl.

Запуск:
  python -m src.m3.predict --mode zero_shot --split val --limit 10
  python -m src.m3.predict --config configs/config.cloud.yaml --mode zero_shot --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ..common.config import load_config
from ..common.eval_local import evaluate, fit_thresholds
from ..common.guard import assert_cloud_safe
from ..common.llm_client import JudgeClient
from ..common.run_meta import save_run_yaml
from ..common.schemas import Pred, load_cases, save_preds
from .prompts import SEED_INSTRUCTION, build_few_shot_system, build_user_prompt


def get_system_prompt(cfg: dict) -> str:
    mode = cfg["m3"]["mode"]
    if mode == "zero_shot":
        return SEED_INSTRUCTION
    if mode == "few_shot":
        import yaml
        examples = yaml.safe_load(open("configs/few_shot.yaml", encoding="utf-8"))["examples"]
        return build_few_shot_system(examples)
    if mode == "gepa":
        return Path(cfg["m3"]["out_prompt"]).read_text(encoding="utf-8")
    raise ValueError(f"неизвестный mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mode", choices=["zero_shot", "few_shot", "gepa"], default=None,
                    help="переопределяет m3.mode из конфига")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke: первые N кейсов")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["m3"]["mode"] = args.mode
    profile = cfg.get("profile", "local")

    cases = load_cases(cfg["data"][args.split])
    assert_cloud_safe(cases, profile)          # guard на весь файл до первого запроса
    if args.limit is not None:
        cases = cases[: args.limit]

    client = JudgeClient(cfg, cache_dir=cfg["m3"]["judge_cache"])
    system = get_system_prompt(cfg)

    preds = []
    for c in tqdm(cases, desc=f"m3/{cfg['m3']['mode']}/{args.split}"):
        user = build_user_prompt(c, cfg["llm"]["max_ctx_chars"])
        p_f, p_r, meta = client.judge(system, user, case=c)
        preds.append(Pred(id=c.id, p_faith=p_f, p_rel=p_r,
                          meta={"mode": cfg["m3"]["mode"], **meta}))

    out_dir = Path(cfg["m3"]["out_pred_dir"]) / cfg["m3"]["mode"]
    is_smoke = args.limit is not None
    fname = f"{args.split}__smoke{args.limit}.jsonl" if is_smoke else f"{args.split}.jsonl"
    save_preds(preds, out_dir / fname)
    save_run_yaml(out_dir, cfg, seed=cfg["m3"]["seed"], split=args.split,
                  limit=args.limit, method="m3")

    # dev-оценка: пороги с val (если val-предсказания уже есть), метрики на текущем сплите
    if not is_smoke and args.split in ("val", "test") and any(c.faith is not None for c in cases):
        val_path = out_dir / "val.jsonl"
        if val_path.exists():
            val_cases = load_cases(cfg["data"]["val"])
            with open(val_path, encoding="utf-8") as fh:
                val_preds = [Pred(**json.loads(l)) for l in fh]
            tf, tr, _ = fit_thresholds(val_cases, val_preds)
            report = evaluate(cases, preds, tf, tr)
            (out_dir / f"report_{args.split}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
