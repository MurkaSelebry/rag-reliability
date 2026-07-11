"""Генератор псевдо-корпуса для отладки пайплайнов (docs/07.2).

~300 русскоязычных кейсов с известными синтетическими метками из SberQuAD.
Метки синтетические: числа на псевдо-корпусе не доказывают гипотезы и не идут
в отчёт. Генерации кэшируются поэлементно, скрипт можно прерывать/продолжать.

Запуск (только cloud-профиль или локальный vLLM; данные публичные):
  python scripts/m3m6/make_pseudo_corpus.py --config configs/config.cloud.yaml --limit 20
  python scripts/m3m6/make_pseudo_corpus.py --config configs/config.cloud.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rag_reliability_m3m6.common.config import load_config
from rag_reliability_m3m6.common.llm_client import LLMClient

# --- метки по таблице docs/07.2 ---------------------------------------------
LABELS: dict[str, dict] = {
    "clean": {"faith": 1, "rel": 1, "markers": []},
    "hallucination": {"faith": 0, "rel": 1, "markers": ["hallucination"]},
    "incomplete_answer": {"faith": 0, "rel": 1, "markers": ["incomplete_answer"]},
    "off_topic_answer": {"faith": 1, "rel": 0, "markers": ["off_topic_answer"]},
}

# микс 2/1/1/1 (clean/halluc/incomplete/off-topic)
_KIND_CYCLE = ["clean", "hallucination", "clean", "incomplete_answer", "off_topic_answer"]

GEN_SYSTEM = (
    "Ты помогаешь готовить синтетические данные для тестирования систем "
    "проверки ответов. Выводи только текст ответа, без пояснений, преамбул "
    "и кавычек. Отвечай ТОЛЬКО на русском языке. Не повторяй вопрос в ответе."
)

GEN_USER: dict[str, str] = {
    "clean": (
        "Абзац:\n{par}\n\nВопрос: {q}\n\n"
        "Дай точный ответ на вопрос строго по абзацу (1–3 предложения)."
    ),
    "hallucination": (
        "Абзац:\n{par}\n\nВопрос: {q}\n\n"
        "Дай ответ на вопрос по абзацу (1–3 предложения), но намеренно "
        "подмени ровно ОДИН факт — число, дату, имя или условие — на "
        "правдоподобный, но неверный. Всё остальное оставь верным. "
        "ОБЯЗАТЕЛЬНО включи подмену: ответ без изменённого факта недопустим. "
        "Никак не отмечай подмену."
    ),
    "incomplete_answer": (
        "Абзац:\n{par}\n\nВопрос: {q}\n\n"
        "Дай верный, но намеренно НЕПОЛНЫЙ ответ на вопрос: опусти "
        "одну важную деталь или оговорку из абзаца, без которой ответ "
        "неполон. Не упоминай, что что-то опущено."
    ),
    "off_topic_answer": (
        "Абзац:\n{par}\n\nВопрос: {q}\n\n"
        "Напиши верный по абзацу ответ (1–3 предложения) про ДРУГОЙ "
        "аспект абзаца, который НЕ отвечает на заданный вопрос. "
        "Сам вопрос не упоминай. "
        "НЕ давай прямой ответ на вопрос ни явно, ни косвенно: "
        "не называй факт, дату, имя или число, которое является ответом."
    ),
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
    return {
        "train": ids[: n - n_val - n_test],
        "val": ids[n - n_val - n_test : n - n_test],
        "test": ids[n - n_test :],
    }


def load_pairs(source: str, n: int, seed: int) -> list[dict]:
    """Пары (question, paragraph): SberQuAD (не более одного кейса на абзац) или свой jsonl."""
    rng = random.Random(seed)
    if source == "sberquad":
        from datasets import load_dataset  # ленивый импорт (тяжёлый)

        ds = load_dataset("kuznetsoffandrey/sberquad", split="train")
        by_par: dict[str, dict] = {}
        for row in ds:
            by_par.setdefault(
                row["context"], {"question": row["question"], "paragraph": row["context"]}
            )
        pool = list(by_par.values())
    else:
        with open(source, encoding="utf-8") as fh:
            pool = [json.loads(l) for l in fh if l.strip()]
    if len(pool) < n:
        raise SystemExit(f"в источнике {len(pool)} абзацев, нужно {n}")
    return rng.sample(pool, n)


def generate_case(
    client: LLMClient,
    cache: Path,
    rng: random.Random,
    i: int,
    kind: str,
    pair: dict,
    par_pool: list[str],
) -> dict:
    """Один кейс канонического формата (docs/01) + meta; генерация кэшируется."""
    cid = f"pseudo_{i:05d}"
    cache_file = cache / f"{cid}.json"
    cached = None
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None  # повреждённый кэш (обрыв записи) — перегенерируем
    if cached is not None:
        if cached.get("kind") != kind:
            raise ValueError(f"{cid}: кэш kind={cached.get('kind')}, ожидается {kind} — удали кэш")
        answer = cached["answer"]
    else:
        messages = [
            {"role": "system", "content": GEN_SYSTEM},
            {
                "role": "user",
                "content": GEN_USER[kind].format(par=pair["paragraph"], q=pair["question"]),
            },
        ]
        # публичные данные (SberQuAD), не корпус кураторов — флаг public_data=True
        answer = client.chat(messages, temperature=0.7, max_tokens=300, public_data=True)[0][
            "text"
        ].strip()
        if not answer:
            raise ValueError(f"{cid}: пустой ответ от модели — не кэширую")
        tmp = cache_file.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"id": cid, "kind": kind, "answer": answer}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(cache_file)  # атомарная замена — обрыв не оставит битый кэш
    return {
        "id": cid,
        "query": pair["question"],
        "context": build_context(rng, pair["paragraph"], par_pool),
        "answer": answer,
        **LABELS[kind],
        "meta": {"kind": kind, "synthetic": True},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.cloud.yaml")
    ap.add_argument("--n", type=int, default=None, help="размер корпуса (дефолт из конфига)")
    ap.add_argument("--limit", type=int, default=None, help="smoke: только первые N кейсов")
    ap.add_argument("--source", default=None, help="sberquad | путь к jsonl {question, paragraph}")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ps = cfg["pseudo"]
    n = args.n if args.n is not None else ps["n"]
    seed = args.seed if args.seed is not None else ps["seed"]
    source = args.source or ps["source"]

    client = LLMClient(cfg, model=ps["gen_model"])
    cache = Path(ps["cache"])
    cache.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(source, n, seed)
    par_pool = [p["paragraph"] for p in pairs]
    kinds = plan_kinds(n)
    todo = min(n, args.limit) if args.limit else n

    from tqdm import tqdm

    rng = random.Random(seed)  # один rng на весь прогон -> детерминированные контексты
    cases = [
        generate_case(client, cache, rng, i, kinds[i], pairs[i], par_pool)
        for i in tqdm(range(todo), desc="pseudo")
    ]

    splits = split_ids([c["id"] for c in cases], seed)
    by_id = {c["id"]: c for c in cases}
    out_dir = Path(ps["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # smoke-прогон (--limit) пишет в суффиксные файлы, чтобы не затирать полный корпус
    is_smoke = args.limit is not None
    suffix = f"__smoke{args.limit}" if is_smoke else ""
    if is_smoke:
        print(
            f"smoke-режим (--limit {args.limit}): пишу в pseudo_dev_*{suffix}.jsonl, "
            "полный корпус не трогаю"
        )
    for split, ids in splits.items():
        path = out_dir / f"pseudo_dev_{split}{suffix}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for cid in ids:
                if cid in by_id:
                    f.write(json.dumps(by_id[cid], ensure_ascii=False) + "\n")
        count = sum(1 for cid in ids if cid in by_id)
        print(f"{path}: {count} кейсов")


if __name__ == "__main__":
    main()
