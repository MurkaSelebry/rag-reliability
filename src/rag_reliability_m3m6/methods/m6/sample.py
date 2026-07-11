"""Метод 6, этап 1: N сэмплов ответа «бота» на (Q, CTX), поэлементный кэш.

Запуск:
  python scripts/m3m6/prepare_m6_samples.py --config configs/config.cloud.yaml --split val --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rag_reliability_m3m6.common.config import load_config
from rag_reliability_m3m6.common.guard import assert_cloud_safe
from rag_reliability_m3m6.common.llm_client import LLMClient
from rag_reliability_m3m6.common.schemas import load_cases

# прокси-промпт реального бота (дефолт из docs/04; уточнить у кураторов)
BOT_SYSTEM = (
    "Ты — ассистент банка для корпоративных клиентов. Отвечай на вопрос "
    "клиента, используя только предоставленные фрагменты документации. "
    "Если ответа в фрагментах нет, скажи об этом."
)
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
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="переопределяет m6.n_samples (напр. добор до 10 для абляции N)",
    )
    ap.add_argument(
        "--concurrency", type=int, default=1, help="параллельные кейсы (1 = старый синхронный путь)"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    m6, llm = cfg["m6"], cfg["llm"]
    if args.n is not None:
        m6 = {**m6, "n_samples": args.n}
    cases = load_cases(cfg["data"][args.split])
    assert_cloud_safe(
        cases,
        cfg.get("profile", "local"),
        allow_real=bool((cfg.get("guard") or {}).get("allow_real_data")),
    )
    if args.limit:
        cases = cases[: args.limit]

    cache_dir = Path(m6["samples_cache"]) / args.split
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _messages(c):
        return [
            {"role": "system", "content": BOT_SYSTEM},
            {
                "role": "user",
                "content": BOT_USER.format(ctx=c.ctx_text(llm["max_ctx_chars"]), q=c.q_text()),
            },
        ]

    def _write_cache(out: Path, cid: str, samples: list[str]) -> None:
        payload = json.dumps({"id": cid, "samples": samples}, ensure_ascii=False)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(out)  # атомарная замена — обрыв не оставит битый кэш

    # кейсы, которым нужен добор (cache-skip сохраняется в обоих путях)
    todo = []
    for c in cases:
        out = cache_dir / f"{c.id}.json"
        need, existing = _need_samples(out, m6["n_samples"])
        if need > 0:
            todo.append((c, out, need, existing))

    if args.concurrency > 1:
        import asyncio

        from rag_reliability_m3m6.common.async_llm import AsyncLLMClient

        aclient = AsyncLLMClient(cfg)

        async def _one(c, out, need, existing, sem):
            async with sem:
                choices = await aclient.chat(
                    _messages(c),
                    temperature=m6["temperature"],
                    top_p=m6["top_p"],
                    n=need,
                    max_tokens=m6["max_new_tokens"],
                    case=c,
                )
            if not choices:
                raise RuntimeError(f"{c.id}: провайдер вернул 0 сэмплов — не кэширую")
            _write_cache(out, c.id, existing + [ch["text"] for ch in choices])

        async def _run():
            sem = asyncio.Semaphore(args.concurrency)
            await asyncio.gather(*[_one(c, out, need, ex, sem) for c, out, need, ex in todo])

        asyncio.run(_run())
    else:
        client = LLMClient(cfg)
        for c, out, need, existing in tqdm(todo, desc=f"m6/sample/{args.split}"):
            choices = client.chat(
                _messages(c),
                temperature=m6["temperature"],
                top_p=m6["top_p"],
                n=need,
                max_tokens=m6["max_new_tokens"],
                case=c,
            )
            if not choices:
                raise RuntimeError(f"{c.id}: провайдер вернул 0 сэмплов — не кэширую")
            _write_cache(out, c.id, existing + [ch["text"] for ch in choices])


if __name__ == "__main__":
    main()
