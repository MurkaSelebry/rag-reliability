"""Метод 3, ступень 3: GEPA-оптимизация инструкции судьи (DSPy).

DSPy используется ТОЛЬКО для оптимизации; инференс — rag_reliability.methods.m3.predict
(JudgeClient, вероятности из logprobs). Варианты markers/plain отличаются
ИСКЛЮЧИТЕЛЬНО наличием расшифровок маркеров в feedback — их разница при
равном бюджете и seed = проверка механики H5 (docs/03).

Запуск:
  python scripts/run_gepa.py --config configs/config.cloud.yaml --variant markers --seed 0
  python scripts/run_gepa.py --config configs/config.cloud.yaml --variant plain --seed 1 \
      --train-size 100 --auto light
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Literal

import yaml

from rag_reliability.common.config import load_config
from rag_reliability.common.guard import assert_cloud_safe
from rag_reliability.common.run_meta import git_hash
from rag_reliability.common.schemas import Case, load_cases
from rag_reliability.methods.m3.prompts import SEED_INSTRUCTION


def load_marker_gloss(path: str | Path) -> dict[str, str]:
    """Глосс маркеров из configs/markers.yaml (НЕ хардкод: на реальном корпусе
    файл заменится словарём кураторов без правок кода)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in data.items()}


def make_metric(use_markers: bool, gloss: dict[str, str]):
    """Метрика GEPA: score = (ok_faith + ok_rel) / 2; feedback при ошибке —
    истинные метки, в варианте markers дополненные глоссом маркеров кейса."""
    import dspy

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        ok_f = str(getattr(pred, "faithfulness", "")).strip().upper() == gold.faithfulness
        ok_r = str(getattr(pred, "relevance", "")).strip().upper() == gold.relevance
        score = (int(ok_f) + int(ok_r)) / 2
        if ok_f and ok_r:
            feedback = "Обе оценки верны."
        else:
            feedback = (
                f"Ошибка. Правильный ответ: FAITHFULNESS={gold.faithfulness}, "
                f"RELEVANCE={gold.relevance}."
            )
            markers = list(getattr(gold, "markers", None) or [])
            if use_markers and markers:
                lines = [f"- {m}: {gloss[m]}" if m in gloss else f"- {m}" for m in markers]
                feedback += "\nМаркеры ошибки от кураторов:\n" + "\n".join(lines)
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def build_program():
    """ChainOfThought-судья; инструкция = АКТУАЛЬНЫЙ SEED_INSTRUCTION
    (включает правило независимости осей из правки B1)."""
    import dspy

    class Judge(dspy.Signature):
        """(инструкция подставляется через with_instructions ниже)"""

        query: str = dspy.InputField(desc="вопрос клиента (с историей диалога)")
        context: str = dspy.InputField(desc="фрагменты документации")
        answer: str = dspy.InputField(desc="ответ ассистента")
        faithfulness: Literal["PASS", "FAIL"] = dspy.OutputField()
        relevance: Literal["PASS", "FAIL"] = dspy.OutputField()

    return dspy.ChainOfThought(Judge.with_instructions(SEED_INSTRUCTION))


def build_examples(cases: list[Case], max_ctx_chars: int | None) -> list:
    """dspy.Example из Case: те же поля, что видит судья на инференсе."""
    import dspy

    return [
        dspy.Example(
            query=c.q_text(),
            context=c.ctx_text(max_ctx_chars),
            answer=c.answer,
            faithfulness="PASS" if c.faith == 1 else "FAIL",
            relevance="PASS" if c.rel == 1 else "FAIL",
            markers=list(c.markers),
        ).with_inputs("query", "context", "answer")
        for c in cases
    ]


def _extract_instruction(program) -> str:
    """Инструкция единственного предиктора оптимизированной программы."""
    _, predictor = next(iter(program.named_predictors()))
    return predictor.signature.instructions


def _serialize_detailed(dr) -> dict:
    """track_stats -> json-совместимая выжимка (кандидаты, val-скоры, счётчики)."""
    if dr is None:
        return {}
    try:
        return json.loads(json.dumps(dr.to_dict(), default=str, ensure_ascii=False))
    except Exception:
        return {
            "val_aggregate_scores": getattr(dr, "val_aggregate_scores", None),
            "best_idx": getattr(dr, "best_idx", None),
            "total_metric_calls": getattr(dr, "total_metric_calls", None),
            "candidates": [
                {k: str(v) for k, v in c.items()} if isinstance(c, dict) else str(c)
                for c in (getattr(dr, "candidates", None) or [])
            ],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="GEPA-оптимизация инструкции судьи (m3)")
    ap.add_argument("--config", default="configs/config.cloud.yaml")
    ap.add_argument("--variant", choices=["markers", "plain"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--train-size", type=int, default=None, help="переопределяет m3.gepa.train_size"
    )
    ap.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default=None,
        help="переопределяет m3.gepa.auto (medium — только по решению пользователя)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    llm, gcfg = cfg["llm"], cfg["m3"]["gepa"]
    vcfg = gcfg["variants"][args.variant]
    train_size = args.train_size if args.train_size is not None else gcfg["train_size"]
    auto = args.auto or gcfg["auto"]
    profile = cfg.get("profile", "local")

    train_cases = load_cases(cfg["data"]["train"])
    val_cases = load_cases(cfg["data"]["val"])
    allow_real = bool((cfg.get("guard") or {}).get("allow_real_data"))
    assert_cloud_safe(train_cases, profile, allow_real=allow_real)  # guard до 1-го вызова
    assert_cloud_safe(val_cases, profile, allow_real=allow_real)

    # детерминированная подвыборка train (seed прогона); val — целиком.
    # На реальном корпусе маркеры разреженные -> поднимаем их долю до
    # gepa.train_marker_share (решение одинаково для markers и plain — H5 честен).
    rng = random.Random(args.seed)
    if train_size < len(train_cases):
        share = float(gcfg.get("train_marker_share", 0.0))
        marked = [c for c in train_cases if c.markers]
        if share > 0 and marked:
            n_marked = min(len(marked), int(round(train_size * share)))
            rest_pool = [c for c in train_cases if not c.markers]
            picked = rng.sample(marked, n_marked) + rng.sample(rest_pool, train_size - n_marked)
            rng.shuffle(picked)
            train_cases = picked
            print(f"[gepa] подвыборка {train_size}: маркерных {n_marked} ({share:.0%})")
        else:
            train_cases = rng.sample(train_cases, train_size)

    import dspy

    extra_body = dict(llm.get("openrouter_extra_body") or {})
    task_lm = dspy.LM(
        f"openai/{llm['model']}",
        api_base=llm["api_base"],
        api_key=llm["api_key"],
        temperature=0.0,
        max_tokens=600,
        **({"extra_body": extra_body} if extra_body else {}),
    )
    refl = gcfg["reflection"]
    reflection_lm = dspy.LM(
        f"openai/{refl['model']}",
        api_base=refl.get("api_base", llm["api_base"]),
        api_key=llm["api_key"],
        temperature=1.0,
        max_tokens=refl.get("max_tokens", 8000),
    )
    dspy.configure(lm=task_lm)

    program = build_program()
    metric = make_metric(
        use_markers=bool(vcfg["use_marker_feedback"]),
        gloss=load_marker_gloss("configs/markers.yaml"),
    )
    trainset = build_examples(train_cases, llm.get("max_ctx_chars"))
    valset = build_examples(val_cases, llm.get("max_ctx_chars"))

    gepa = dspy.GEPA(
        metric=metric, auto=auto, reflection_lm=reflection_lm, track_stats=True, seed=args.seed
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    # --- сохранение: инструкция (txt), программа (json), статистика эволюции ---
    instruction = _extract_instruction(optimized)
    out_prompt = Path(vcfg["out_prompt"].format(seed=args.seed))
    out_prompt.parent.mkdir(parents=True, exist_ok=True)
    out_prompt.write_text(instruction, encoding="utf-8")
    optimized.save(str(out_prompt.with_suffix(".program.json")))

    stats = {
        "variant": args.variant,
        "seed": args.seed,
        "auto": auto,
        "train_size": len(trainset),
        "val_size": len(valset),
        "use_marker_feedback": bool(vcfg["use_marker_feedback"]),
        "task_model": llm["model"],
        "reflection_model": refl["model"],
        "task_lm_calls": len(getattr(task_lm, "history", []) or []),
        "reflection_lm_calls": len(getattr(reflection_lm, "history", []) or []),
        "profile": profile,
        "git_hash": git_hash(),
        "seed_instruction": SEED_INSTRUCTION,
        "best_instruction": instruction,
        "detailed_results": _serialize_detailed(getattr(optimized, "detailed_results", None)),
    }
    stats_path = out_prompt.parent / f"m3_gepa_stats_{args.variant}_seed{args.seed}.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"промпт: {out_prompt}\nстатистика: {stats_path}\n"
        f"вызовы: task={stats['task_lm_calls']}, reflection={stats['reflection_lm_calls']}"
    )


if __name__ == "__main__":
    main()
