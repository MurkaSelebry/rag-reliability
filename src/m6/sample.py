"""Метод 6, этап 1: N сэмплов ответа «бота» на (Q, CTX), поэлементный кэш.

Запуск:
  python -m src.m6.sample --config configs/config.cloud.yaml --split val --limit 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ..common.config import load_config
from ..common.guard import assert_cloud_safe
from ..common.llm_client import LLMClient
from ..common.schemas import load_cases

# прокси-промпт реального бота (дефолт из docs/04; уточнить у кураторов)
BOT_SYSTEM = ("Ты — ассистент банка для корпоративных клиентов. Отвечай на вопрос "
              "клиента, используя только предоставленные фрагменты документации. "
              "Если ответа в фрагментах нет, скажи об этом.")
BOT_USER = "Фрагменты документации:\n{ctx}\n\nВопрос клиента: {q}\n\nОтвет:"


def _need_samples(cache_file: Path, target: int) -> tuple[int, list[str]]:
    """Сколько ещё сэмплов добрать под target и что уже лежит в кэше."""
    existing: list[str] = []
    if cache_file.exists():
        existing = json.loads(cache_file.read_text(encoding="utf-8"))["samples"]
    return max(0, target - len(existing)), existing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=None,
                    help="переопределяет m6.n_samples (напр. добор до 10 для абляции N)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m6, llm = cfg["m6"], cfg["llm"]
    if args.n is not None:
        m6 = {**m6, "n_samples": args.n}
    cases = load_cases(cfg["data"][args.split])
    assert_cloud_safe(cases, cfg.get("profile", "local"))
    if args.limit:
        cases = cases[: args.limit]

    client = LLMClient(cfg)
    cache_dir = Path(m6["samples_cache"]) / args.split
    cache_dir.mkdir(parents=True, exist_ok=True)

    for c in tqdm(cases, desc=f"m6/sample/{args.split}"):
        out = cache_dir / f"{c.id}.json"
        need, existing = _need_samples(out, m6["n_samples"])
        if need == 0:
            continue  # кэш уже полон под текущий n_samples
        messages = [{"role": "system", "content": BOT_SYSTEM},
                    {"role": "user", "content": BOT_USER.format(
                        ctx=c.ctx_text(llm["max_ctx_chars"]), q=c.q_text())}]
        choices = client.chat(messages, temperature=m6["temperature"],
                              top_p=m6["top_p"], n=need,
                              max_tokens=m6["max_new_tokens"], case=c)
        if not choices:
            raise RuntimeError(f"{c.id}: провайдер вернул 0 сэмплов — не кэширую")
        samples = existing + [ch["text"] for ch in choices]
        payload = json.dumps({"id": c.id, "samples": samples}, ensure_ascii=False)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(out)  # атомарная замена — обрыв не оставит битый кэш


if __name__ == "__main__":
    main()
