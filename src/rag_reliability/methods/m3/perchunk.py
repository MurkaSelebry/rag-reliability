"""Пофрагментная верификация faithfulness для Метода 3.

Каждый запрос видит ровно один retrieved-чанк. Все запросы одного кейса
передаются batch-функции одновременно: так вызывающая сторона может отправить
их через асинхронный клиент с семафором, не делая последовательный цикл сети.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Sequence

from rag_reliability.methods.m3.axes import AXIS_FAITHFULNESS, build_axis_prompt
from rag_reliability.methods.surface.features import split_chunks
from rag_reliability.schema import RagSample

BatchJudgeResult = Sequence[float] | Awaitable[Sequence[float]]
BatchJudgeFn = Callable[[str, Sequence[str]], BatchJudgeResult]

SUPPORT_THRESHOLD = 0.5


def _chunk_prompts(sample: RagSample, axis: str) -> tuple[str, list[str]]:
    """Собрать общий system и по одному изолированному user-промпту на чанк."""
    chunks = split_chunks(sample.context)
    if not chunks:
        raise ValueError(f"Sample {sample.id!r} has no context chunks for per-chunk scoring")

    system: str | None = None
    users: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_sample = sample.model_copy(update={"context": f"[CHUNK {index}]\n{chunk}"})
        chunk_system, user = build_axis_prompt(chunk_sample, axis)
        if system is None:
            system = chunk_system
        elif chunk_system != system:
            raise RuntimeError(
                f"Faithfulness system prompt changed within sample {sample.id!r} at chunk {index}"
            )
        users.append(user)

    if system is None:  # pragma: no cover — непустота chunks проверена выше
        raise RuntimeError(f"Failed to build per-chunk prompts for sample {sample.id!r}")
    return system, users


def _validated_scores(
    raw_scores: Sequence[float], *, sample_id: str, expected: int
) -> list[float]:
    scores = [float(score) for score in raw_scores]
    if len(scores) != expected:
        raise ValueError(
            f"Batch judge returned {len(scores)} score(s) for sample {sample_id!r}; "
            f"expected one score for each of {expected} chunk(s)"
        )
    invalid = [score for score in scores if not math.isfinite(score) or not 0.0 <= score <= 1.0]
    if invalid:
        raise ValueError(
            f"Batch judge returned invalid probability score(s) for sample {sample_id!r}: "
            f"{invalid[:5]}"
        )
    return scores


def _resolve_batch(result: BatchJudgeResult, *, sample_id: str) -> Sequence[float]:
    """Синхронный фасад принимает и обычный, и async batch-адаптер."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    if inspect.iscoroutine(result):
        result.close()
    raise RuntimeError(
        f"Async batch judge for sample {sample_id!r} cannot be awaited from a running event loop; "
        "call score_per_chunk from synchronous orchestration"
    )


def score_per_chunk(
    sample: RagSample,
    judge_fn: BatchJudgeFn,
    *,
    axis: str = AXIS_FAITHFULNESS,
) -> dict[str, float]:
    """Оценить ответ отдельно по каждому чанку и вернуть пять ``m3.*`` фич.

    ``judge_fn`` получает общий system-промпт и все user-промпты кейса одним
    batch-вызовом. Поддерживается только faithfulness: relevance по контракту C3
    не получает чанки и потому не имеет пофрагментного варианта.
    """
    if axis != AXIS_FAITHFULNESS:
        raise ValueError(
            f"Per-chunk scoring supports only axis={AXIS_FAITHFULNESS!r}, got {axis!r}"
        )

    system, users = _chunk_prompts(sample, axis)
    scores = _validated_scores(
        _resolve_batch(judge_fn(system, users), sample_id=sample.id),
        sample_id=sample.id,
        expected=len(users),
    )

    ranked = sorted(scores, reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else top1
    argmax = max(range(len(scores)), key=scores.__getitem__) + 1
    return {
        "m3.max_chunk_score": top1,
        "m3.mean_chunk_score": math.fsum(scores) / len(scores),
        "m3.chunk_disagreement": top1 - top2,
        "m3.n_supporting": float(sum(score > SUPPORT_THRESHOLD for score in scores)),
        "m3.argmax_chunk": float(argmax),
    }
