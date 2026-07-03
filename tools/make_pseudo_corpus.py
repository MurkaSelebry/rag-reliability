"""Генератор псевдо-корпуса для отладки пайплайнов (docs/07.2).

~300 русскоязычных кейсов с известными синтетическими метками из SberQuAD.
Метки синтетические: числа на псевдо-корпусе не доказывают гипотезы и не идут
в отчёт. Генерации кэшируются поэлементно, скрипт можно прерывать/продолжать.

Запуск (только cloud-профиль или локальный vLLM; данные публичные):
  python -m tools.make_pseudo_corpus --config configs/config.cloud.yaml --limit 20
  python -m tools.make_pseudo_corpus --config configs/config.cloud.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.common.config import load_config
from src.common.llm_client import LLMClient

# --- метки по таблице docs/07.2 ---------------------------------------------
LABELS: dict[str, dict] = {
    "clean":             {"faith": 1, "rel": 1, "markers": []},
    "hallucination":     {"faith": 0, "rel": 1, "markers": ["hallucination"]},
    "incomplete_answer": {"faith": 0, "rel": 1, "markers": ["incomplete_answer"]},
    "off_topic_answer":  {"faith": 1, "rel": 0, "markers": ["off_topic_answer"]},
}

# микс 2/1/1/1 (clean/halluc/incomplete/off-topic)
_KIND_CYCLE = ["clean", "hallucination", "clean", "incomplete_answer", "off_topic_answer"]

GEN_SYSTEM = ("Ты помогаешь готовить синтетические данные для тестирования систем "
              "проверки ответов. Выводи только текст ответа, без пояснений, преамбул "
              "и кавычек.")

GEN_USER: dict[str, str] = {
    "clean": ("Абзац:\n{par}\n\nВопрос: {q}\n\n"
              "Дай точный ответ на вопрос строго по абзацу (1–3 предложения)."),
    "hallucination": ("Абзац:\n{par}\n\nВопрос: {q}\n\n"
                      "Дай ответ на вопрос по абзацу (1–3 предложения), но намеренно "
                      "подмени ровно ОДИН факт — число, дату, имя или условие — на "
                      "правдоподобный, но неверный. Всё остальное оставь верным. "
                      "Никак не отмечай подмену."),
    "incomplete_answer": ("Абзац:\n{par}\n\nВопрос: {q}\n\n"
                          "Дай верный, но намеренно НЕПОЛНЫЙ ответ на вопрос: опусти "
                          "одну важную деталь или оговорку из абзаца, без которой ответ "
                          "неполон. Не упоминай, что что-то опущено."),
    "off_topic_answer": ("Абзац:\n{par}\n\nВопрос: {q}\n\n"
                         "Напиши верный по абзацу ответ (1–3 предложения) про ДРУГОЙ "
                         "аспект абзаца, который НЕ отвечает на заданный вопрос. "
                         "Сам вопрос не упоминай."),
}


def plan_kinds(n: int) -> list[str]:
    """Последовательность типов кейсов в пропорции 2/1/1/1."""
    return [_KIND_CYCLE[i % len(_KIND_CYCLE)] for i in range(n)]


def build_context(rng: random.Random, paragraph: str, pool: list[str]) -> list[str]:
    """Абзац-источник + 1–2 дистрактора из пула, порядок перемешан (docs/07.2)."""
    distractors = rng.sample([p for p in pool if p != paragraph], k=rng.randint(1, 2))
    ctx = [paragraph, *distractors]
    rng.shuffle(ctx)
    return ctx


def split_ids(ids: list[str], seed: int) -> dict[str, list[str]]:
    """Детерминированный сплит 80/10/10."""
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val, n_test = round(n * 0.1), round(n * 0.1)
    return {"train": ids[: n - n_val - n_test],
            "val": ids[n - n_val - n_test: n - n_test],
            "test": ids[n - n_test:]}
