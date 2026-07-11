#!/usr/bin/env python
"""Generate Method 6 answer samples into a per-sample JSON cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.schema import RagSample

BOT_SYSTEM = (
    "Ты — ассистент банка для корпоративных клиентов. Отвечай на вопрос клиента, "
    "используя только предоставленные фрагменты документации. Если ответа в "
    "фрагментах нет, скажи об этом."
)
BOT_USER = "Фрагменты документации:\n{context}\n\nВопрос клиента: {question}\n\nОтвет:"


def build_bot_prompt(sample: RagSample, max_context_chars: int | None = None) -> str:
    context = sample.context
    if max_context_chars is not None and len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[контекст усечён]"
    return f"{BOT_SYSTEM}\n\n{BOT_USER.format(context=context, question=sample.question)}"


def need_samples(cache_file: Path, target: int) -> tuple[int, list[str]]:
    existing: list[str] = []
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        raw_samples = payload.get("samples") or []
        if not isinstance(raw_samples, list) or not all(
            isinstance(sample, str) for sample in raw_samples
        ):
            raise ValueError(f"Invalid Method 6 sample cache at {cache_file}")
        existing = raw_samples
    return max(0, target - len(existing)), existing


def write_sample_cache(output_dir: str | Path, sample_id: str, samples: list[str]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = output_path / f"{sample_id}.json"
    payload = json.dumps({"id": sample_id, "samples": samples}, ensure_ascii=False)
    tmp = cache_file.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(cache_file)


def build_sample_cache(
    samples: list[RagSample],
    *,
    output_dir: str | Path,
    generate_fn: Callable[[str], str],
    n_samples: int,
    max_context_chars: int | None = None,
) -> None:
    output_path = Path(output_dir)
    for sample in tqdm(samples, desc="m6/samples"):
        cache_file = output_path / f"{sample.id}.json"
        needed, existing = need_samples(cache_file, n_samples)
        if needed == 0:
            continue
        prompt = build_bot_prompt(sample, max_context_chars=max_context_chars)
        generated = [generate_fn(prompt) for _ in range(needed)]
        write_sample_cache(output_path, sample.id, existing + generated)


def dummy_generate(prompt: str) -> str:
    return prompt.rsplit("Ответ:", maxsplit=1)[-1].strip() or "dummy answer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--output-dir", default="results/m6/samples")
    parser.add_argument("--backend", choices=["dummy", "mlx"], default="mlx")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.backend == "dummy":
        generate_fn = dummy_generate
    else:
        generate_fn = make_generate_fn(args.model, args.max_tokens)

    build_sample_cache(
        samples,
        output_dir=args.output_dir,
        generate_fn=generate_fn,
        n_samples=args.n_samples,
        max_context_chars=args.max_context_chars,
    )
    print(f"Prepared Method 6 samples for {len(samples)} case(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
