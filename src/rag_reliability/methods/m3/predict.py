"""Метод 3, инференс: судья с выбранным промптом -> predictions/{variant}/{split}.jsonl.

Запуск:
  python scripts/run_m3.py --mode zero_shot --split val --limit 10
  python scripts/run_m3.py --config configs/config.cloud.yaml --mode zero_shot --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rag_reliability.common.config import load_config
from rag_reliability.common.eval_local import evaluate, fit_thresholds
from rag_reliability.common.guard import assert_cloud_safe
from rag_reliability.common.llm_client import JudgeClient
from rag_reliability.common.run_meta import save_run_yaml
from rag_reliability.common.schemas import Pred, load_cases, save_preds
from rag_reliability.methods.m3.prompts import (
    SEED_INSTRUCTION,
    build_few_shot_system,
    build_user_prompt,
)


def get_system_prompt(cfg: dict, prompt_file: str | None = None) -> str:
    mode = cfg["m3"]["mode"]
    if mode == "zero_shot":
        return SEED_INSTRUCTION
    if mode == "few_shot":
        import yaml

        examples = yaml.safe_load(open("configs/few_shot.yaml", encoding="utf-8"))["examples"]
        return build_few_shot_system(examples)
    if mode == "gepa":
        return Path(prompt_file or cfg["m3"]["out_prompt"]).read_text(encoding="utf-8")
    raise ValueError(f"неизвестный mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot", "gepa"],
        default=None,
        help="переопределяет m3.mode из конфига",
    )
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke: первые N кейсов")
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="только mode=gepa: путь к txt эволюционированной инструкции",
    )
    ap.add_argument(
        "--variant-name",
        default=None,
        help="имя поддиректории predictions (default = mode); нужен, чтобы "
        "прогоны gepa markers/plain × seed не перетирали друг друга",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="параллельные запросы судьи (1 = старый синхронный путь)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["m3"]["mode"] = args.mode
    if args.prompt_file and cfg["m3"]["mode"] != "gepa":
        ap.error("--prompt-file имеет смысл только при --mode gepa")
    profile = cfg.get("profile", "local")

    allow_real = bool((cfg.get("guard") or {}).get("allow_real_data"))
    cases = load_cases(cfg["data"][args.split])
    assert_cloud_safe(cases, profile, allow_real=allow_real)  # guard до первого запроса
    if args.limit is not None:
        cases = cases[: args.limit]

    system = get_system_prompt(cfg, prompt_file=args.prompt_file)
    variant_name = args.variant_name or cfg["m3"]["mode"]

    if args.concurrency > 1:
        # async-путь: тот же кэш и цепочка fallback, что у синхронного клиента
        import asyncio

        from rag_reliability.common.async_llm import AsyncJudgeClient

        aclient = AsyncJudgeClient(
            cfg, cache_dir=cfg["m3"]["judge_cache"], concurrency=args.concurrency
        )
        items = [(c, build_user_prompt(c, cfg["llm"]["max_ctx_chars"])) for c in cases]
        results = asyncio.run(aclient.judge_many(system, items))
        preds = [
            Pred(
                id=c.id,
                p_faith=p_f,
                p_rel=p_r,
                meta={"mode": cfg["m3"]["mode"], "variant": variant_name, **meta},
            )
            for c, (p_f, p_r, meta) in zip(cases, results)
        ]
    else:
        client = JudgeClient(cfg, cache_dir=cfg["m3"]["judge_cache"])
        preds = []
        for c in tqdm(cases, desc=f"m3/{variant_name}/{args.split}"):
            user = build_user_prompt(c, cfg["llm"]["max_ctx_chars"])
            p_f, p_r, meta = client.judge(system, user, case=c)
            preds.append(
                Pred(
                    id=c.id,
                    p_faith=p_f,
                    p_rel=p_r,
                    meta={"mode": cfg["m3"]["mode"], "variant": variant_name, **meta},
                )
            )

    out_dir = Path(cfg["m3"]["out_pred_dir"]) / variant_name
    is_smoke = args.limit is not None
    fname = f"{args.split}__smoke{args.limit}.jsonl" if is_smoke else f"{args.split}.jsonl"
    save_preds(preds, out_dir / fname)
    save_run_yaml(
        out_dir,
        cfg,
        seed=cfg["m3"]["seed"],
        split=args.split,
        limit=args.limit,
        method="m3",
        variant=variant_name,
        prompt_file=args.prompt_file,
    )

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
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))

            # mlflow-трекинг (локальный file-store), только при tracking.enabled
            tr_cfg = cfg.get("tracking") or {}
            if tr_cfg.get("enabled"):
                from rag_reliability.common.tracking import log_run

                log_run(
                    tracking_uri=tr_cfg.get("uri", "file:./mlruns"),
                    experiment="m3",
                    run_name=f"{variant_name}/{args.split}",
                    cfg=cfg,
                    metrics={k: float(v) for k, v in report.items() if isinstance(v, (int, float))},
                    artifacts=[out_dir / f"report_{args.split}.json", out_dir / "run.yaml"],
                    tags={"split": args.split, "variant": variant_name, "method": "m3"},
                )


if __name__ == "__main__":
    main()
