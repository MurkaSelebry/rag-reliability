"""Метод 3, ступень 3: GEPA-оптимизация judge-промпта через DSPy.

Архитектура:
  1. DSPy-программа = один ChainOfThought-предиктор с сигнатурой Judge.
  2. Метрика возвращает score (вклад в accuracy по двум осям) + текстовый
     feedback; при use_marker_feedback=True в feedback попадают маркеры
     аннотатора — это проверка H5.
  3. GEPA эволюционирует ИНСТРУКЦИЮ предиктора; итог сохраняется в txt и
     используется скриптом predict.py через общий JudgeClient (чтобы снять
     вероятности из logprobs, чего DSPy не даёт из коробки).

Запуск:
  python -m src.m3_gepa.run_gepa --config configs/config.yaml
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Literal

import dspy

from ..common.schemas import load_cases, load_yaml
from .prompts import SEED_INSTRUCTION

# Человекочитаемые расшифровки 13 маркеров — ЗАПОЛНИТЬ по словарю кураторов.
MARKER_GLOSS = {
    "hallucination": "в ответе есть факты, отсутствующие в контексте",
    "incomplete_answer": "ответ упускает важные детали из контекста",
    "off_topic_answer": "ответ не на тот вопрос, который задал клиент",
    # ... остальные 10 маркеров из словаря кураторов
}


class Judge(dspy.Signature):
    """placeholder — заменяется SEED_INSTRUCTION ниже"""
    query: str = dspy.InputField(desc="вопрос клиента и история диалога")
    context: str = dspy.InputField(desc="фрагменты документации из поиска")
    answer: str = dspy.InputField(desc="ответ RAG-ассистента")
    faithfulness: Literal["PASS", "FAIL"] = dspy.OutputField()
    relevance: Literal["PASS", "FAIL"] = dspy.OutputField()


Judge.__doc__ = SEED_INSTRUCTION  # сид-инструкция для эволюции


def case_to_example(c) -> dspy.Example:
    return dspy.Example(
        query=c.q_text(), context=c.ctx_text(), answer=c.answer,
        faithfulness="PASS" if c.faith == 1 else "FAIL",
        relevance="PASS" if c.rel == 1 else "FAIL",
        markers=list(c.markers),
    ).with_inputs("query", "context", "answer")


def make_metric(use_markers: bool):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        ok_f = getattr(pred, "faithfulness", None) == gold.faithfulness
        ok_r = getattr(pred, "relevance", None) == gold.relevance
        score = (ok_f + ok_r) / 2.0
        fb = []
        if ok_f and ok_r:
            fb.append("Обе оценки верны.")
        else:
            fb.append(
                f"Ошибка. Истинные метки: FAITHFULNESS={gold.faithfulness}, "
                f"RELEVANCE={gold.relevance}."
            )
            if use_markers and gold.markers:
                gloss = "; ".join(
                    f"{m} ({MARKER_GLOSS.get(m, 'см. словарь маркеров')})" for m in gold.markers
                )
                fb.append(f"Аннотатор пометил типы ошибок в ответе: {gloss}. "
                          f"Инструкция должна помогать обнаруживать такие ошибки.")
        return dspy.Prediction(score=score, feedback=" ".join(fb))
    return metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    m3, llm = cfg["m3"], cfg["llm"]
    random.seed(m3["seed"])

    task_lm = dspy.LM(f"openai/{llm['model']}", api_base=llm["api_base"],
                      api_key=llm["api_key"], temperature=0.0, max_tokens=600)
    reflection_lm = dspy.LM(f"openai/{llm['reflection_model']}",
                            api_base=llm["reflection_api_base"],
                            api_key=llm["api_key"], temperature=1.0, max_tokens=8000)
    dspy.configure(lm=task_lm)

    train = [case_to_example(c) for c in load_cases(cfg["data"]["train"])]
    val = [case_to_example(c) for c in load_cases(cfg["data"]["val"])]
    random.shuffle(train)
    train = train[: m3["gepa_train_size"]]

    program = dspy.ChainOfThought(Judge)
    gepa = dspy.GEPA(
        metric=make_metric(m3["use_marker_feedback"]),
        auto=m3["gepa_auto"],
        reflection_lm=reflection_lm,
        track_stats=True,
        seed=m3["seed"],
    )
    optimized = gepa.compile(program, trainset=train, valset=val)

    # Достаём эволюционированную инструкцию и сохраняем как финальный промпт
    instructions = optimized.predictors()[0].signature.instructions
    out = Path(m3["out_prompt"]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(instructions, encoding="utf-8")
    optimized.save(str(out.with_suffix(".program.json")))
    detailed = getattr(optimized, "detailed_results", None)
    if detailed is not None:
        (out.parent / "m3_gepa_stats.json").write_text(
            json.dumps(getattr(detailed, "to_dict", lambda: {})(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    print(f"[m3] optimized instruction -> {out}")


if __name__ == "__main__":
    main()
