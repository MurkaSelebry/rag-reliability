"""AsyncLLMClient: конкуррентность ограничена семафором, guard и порядок результатов."""

import asyncio

import pytest

from src.common.async_llm import AsyncJudgeClient
from src.common.guard import DataLeakError
from src.common.schemas import Case

CFG = {
    "profile": "cloud",
    "llm": {"api_base": "http://x", "api_key": "k", "model": "m", "max_ctx_chars": 1000},
}


def _mk_client(monkeypatch, delay=0.01):
    c = AsyncJudgeClient(CFG, cache_dir=None, concurrency=2)
    seen = {"active": 0, "max_active": 0, "calls": 0}

    async def fake(self, system, user, max_tokens):
        seen["active"] += 1
        seen["max_active"] = max(seen["max_active"], seen["active"])
        seen["calls"] += 1
        await asyncio.sleep(delay)
        seen["active"] -= 1
        return f"FAITHFULNESS: PASS\nRELEVANCE: FAIL ({user[-6:]})", []

    monkeypatch.setattr(AsyncJudgeClient, "_chat_judge_async", fake, raising=False)
    return c, seen


def test_semaphore_caps_concurrency(monkeypatch):
    client, seen = _mk_client(monkeypatch)
    cases = [Case(id=f"pseudo_{i}", query="q", context=["c"], answer="a") for i in range(6)]
    out = asyncio.run(client.judge_many("sys", [(c, f"user_{i:03d}") for i, c in enumerate(cases)]))
    assert len(out) == 6 and seen["calls"] == 6
    assert seen["max_active"] <= 2  # семафор работает
    assert [o[2]["raw"][-4:-1] for o in out] == [f"{i:03d}" for i in range(6)]  # порядок сохранён


def test_guard_fires_before_any_call(monkeypatch):
    client, seen = _mk_client(monkeypatch)
    bad = Case(id="case_1", query="q", context=["c"], answer="a")
    with pytest.raises(DataLeakError):
        asyncio.run(client.judge_many("sys", [(bad, "u")]))
    assert seen["calls"] == 0
