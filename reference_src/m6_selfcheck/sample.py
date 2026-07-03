"""Метод 6, этап 1: сэмплирование N ответов на (Q, CTX) и кэширование.

Самая дорогая часть метода — выполняется один раз, результат кэшируется
по id кейса. Промпт сэмплера должен максимально повторять промпт реального
бота (уточнить у кураторов); ниже — разумный дефолт RAG-ассистента.

Запуск:
  python -m src.m6_selfcheck.sample --config configs/config.yaml --split val
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ..common.schemas import load_cases, load_yaml
from ..common.judge_client import JudgeClient

BOT_SYSTEM = ("Ты — ассистент банка для корпоративных клиентов. Отвечай на вопрос "
              "клиента, используя только предоставленные фрагменты документации. "
              "Если ответа в фрагментах нет, скажи об этом.")
BOT_USER = "Фрагменты документации:\n{ctx}\n\nВопрос клиента: {q}\n\nОтвет:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    m6, llm = cfg["m6"], cfg["llm"]

    client = JudgeClient(llm["api_base"], llm["api_key"], llm["model"])
    cache_dir = Path(m6["samples_cache"]) / args.split
    cache_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cfg["data"][args.split])
    for c in tqdm(cases, desc=f"m6/sample/{args.split}"):
        out = cache_dir / f"{c.id}.json"
        if out.exists():
            continue
        messages = [
            {"role": "system", "content": BOT_SYSTEM},
            {"role": "user", "content": BOT_USER.format(
                ctx=c.ctx_text(llm["max_ctx_chars"]), q=c.q_text())},
        ]
        choices = client.chat(messages, temperature=m6["temperature"],
                              top_p=m6["top_p"], n=m6["n_samples"],
                              max_tokens=m6["max_new_tokens"])
        out.write_text(json.dumps(
            {"id": c.id, "samples": [ch["text"] for ch in choices]},
            ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
