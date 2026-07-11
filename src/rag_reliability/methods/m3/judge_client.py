"""OpenAI-compatible judge clients with logprob probabilities (ported from m3-m6).

Sync and async clients over one OpenAI-compatible chat endpoint (local vLLM or
a cloud provider). Both return per-sample probabilities with the fallback chain
logprobs -> regex (0.9/0.1) -> default (0.5/0.5): a sample is never lost.

The sync and async judges share the same sha256 cache key, so their file
caches are interchangeable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rag_reliability.methods.m3.logprobs import extract_verdict_probs
from rag_reliability.methods.m3.parsing import parse_m3_prediction

Messages = list[dict[str, str]]
# choice: {"text": str, "tokens": [{"token", "logprob", "top": {tok: lp}}]}
Choice = dict[str, Any]

_IMPORT_HINT = 'Install OpenAI-compatible backend deps with: uv pip install -e ".[cloud]"'


def _choices_from_response(resp: Any, logprobs: bool) -> list[Choice]:
    if resp.choices is None:
        raise RuntimeError("provider returned choices=None (request parameters unsupported)")
    out: list[Choice] = []
    for ch in resp.choices:
        item: Choice = {"text": ch.message.content or "", "tokens": []}
        if logprobs and ch.logprobs and ch.logprobs.content:
            for t in ch.logprobs.content:
                item["tokens"].append(
                    {
                        "token": t.token,
                        "logprob": t.logprob,
                        "top": {tt.token: tt.logprob for tt in (t.top_logprobs or [])},
                    }
                )
        out.append(item)
    return out


def _judge_verdict(text: str, tokens: list[dict]) -> tuple[float, float, dict]:
    """Fallback chain logprobs -> regex -> default; returns (p_faith, p_rel, meta)."""
    probs = extract_verdict_probs(tokens)
    if probs is not None:
        return probs[0], probs[1], {"method": "logprobs", "raw": text[-400:]}
    parsed = parse_m3_prediction(text, sample_id="_")
    if not parsed.invalid_output:
        p_f = 0.9 if parsed.faithfulness_pred == 1 else 0.1
        p_r = 0.9 if parsed.relevance_pred == 1 else 0.1
        return p_f, p_r, {"method": "regex", "raw": text[-400:]}
    return 0.5, 0.5, {"method": "default", "raw": text[-400:]}


class LLMClient:
    """Sync client: retries and n-fallback over one chat endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        extra_body: dict | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.extra_body = dict(extra_body or {})
        self._client: Any = None

    @property
    def client(self) -> Any:
        # Lazy: constructing the client must not require the optional openai dep.
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(_IMPORT_HINT) from exc
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def _request(
        self,
        *,
        messages: Messages,
        temperature: float,
        n: int,
        max_tokens: int,
        top_p: float,
        logprobs: bool,
        retries: int = 3,
    ) -> list[Choice]:
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=logprobs,
                    top_logprobs=20 if logprobs else None,
                    extra_body=self.extra_body or None,
                )
                return _choices_from_response(resp, logprobs)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def chat(
        self,
        messages: Messages,
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        top_p: float = 1.0,
        logprobs: bool = False,
    ) -> list[Choice]:
        """The only public request method.

        If the provider rejects ``n > 1``, the client degrades to sequential
        single-choice requests until ``n`` choices are collected.
        """
        try:
            choices = self._request(
                messages=messages,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=logprobs,
            )
        except Exception:
            if n <= 1:
                raise
            choices = []  # provider cannot do n>1 — collect with single requests below

        while len(choices) < n:
            new = self._request(
                messages=messages,
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=logprobs,
            )
            if not new:
                raise RuntimeError("provider returned 0 choices at n=1 — cannot backfill")
            choices += new
        return choices[:n]


def _cache_path(cache_dir: Path, model: str, system: str, user: str) -> Path:
    """Shared sync/async cache key."""
    key = hashlib.sha256("\x00".join((model, system, user)).encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _cache_read(cp: Path | None) -> tuple[float, float, dict] | None:
    if cp is None or not cp.exists():
        return None
    try:
        d = json.loads(cp.read_text(encoding="utf-8"))
        return d["p_faith"], d["p_rel"], d["meta"]
    except (json.JSONDecodeError, KeyError):
        return None  # truncated/corrupted cache entry counts as a miss


def _cache_write(cp: Path | None, p_f: float, p_r: float, meta: dict) -> None:
    if cp is None:
        return
    tmp = cp.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"p_faith": p_f, "p_rel": p_r, "meta": meta}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(cp)  # atomic replace — an interrupt cannot leave a broken entry


class JudgeClient(LLMClient):
    """Judge: one T=0 generation with logprobs; fallback logprobs -> regex -> 0.5/0.5."""

    def __init__(self, *, cache_dir: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _chat_judge(self, system: str, user: str, max_tokens: int) -> tuple[str, list]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            choices = self.chat(msgs, temperature=0.0, max_tokens=max_tokens, logprobs=True)
        except RuntimeError:
            # provider cannot do logprobs — degrade to plain text (regex path)
            choices = self.chat(msgs, temperature=0.0, max_tokens=max_tokens, logprobs=False)
        return choices[0]["text"], choices[0]["tokens"]

    def judge(self, system: str, user: str, max_tokens: int = 400) -> tuple[float, float, dict]:
        """-> (p_faith, p_rel, meta). A sample is never lost (fallback to 0.5/0.5)."""
        cp = (
            _cache_path(self.cache_dir, self.model, system, user)
            if self.cache_dir is not None
            else None
        )
        cached = _cache_read(cp)
        if cached is not None:
            return cached
        text, tokens = self._chat_judge(system, user, max_tokens)
        p_f, p_r, meta = _judge_verdict(text, tokens)
        _cache_write(cp, p_f, p_r, meta)
        return p_f, p_r, meta


class AsyncLLMClient:
    """Async twin of LLMClient: same retries and n-fallback."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        extra_body: dict | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.extra_body = dict(extra_body or {})
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(_IMPORT_HINT) from exc
            self._client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    async def _request(
        self,
        *,
        messages: Messages,
        temperature: float,
        n: int,
        max_tokens: int,
        top_p: float,
        logprobs: bool,
        retries: int = 3,
    ) -> list[Choice]:
        for attempt in range(retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=logprobs,
                    top_logprobs=20 if logprobs else None,
                    extra_body=self.extra_body or None,
                )
                return _choices_from_response(resp, logprobs)
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
        raise RuntimeError("unreachable")

    async def chat(
        self,
        messages: Messages,
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        top_p: float = 1.0,
        logprobs: bool = False,
    ) -> list[Choice]:
        try:
            choices = await self._request(
                messages=messages,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=logprobs,
            )
        except Exception:
            if n <= 1:
                raise
            choices = []

        while len(choices) < n:
            new = await self._request(
                messages=messages,
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=logprobs,
            )
            if not new:
                raise RuntimeError("provider returned 0 choices at n=1 — cannot backfill")
            choices += new
        return choices[:n]


class AsyncJudgeClient(AsyncLLMClient):
    """Async judge: a batch of prompts behind a semaphore, cache shared with JudgeClient."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.concurrency = concurrency
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def _chat_judge_async(self, system: str, user: str, max_tokens: int) -> tuple[str, list]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            choices = await self.chat(msgs, temperature=0.0, max_tokens=max_tokens, logprobs=True)
        except RuntimeError:
            choices = await self.chat(
                msgs, temperature=0.0, max_tokens=max_tokens, logprobs=False
            )
        return choices[0]["text"], choices[0]["tokens"]

    async def judge_one(
        self,
        system: str,
        user: str,
        sem: asyncio.Semaphore,
        max_tokens: int = 400,
    ) -> tuple[float, float, dict]:
        cp = (
            _cache_path(self.cache_dir, self.model, system, user)
            if self.cache_dir is not None
            else None
        )
        cached = _cache_read(cp)
        if cached is not None:
            return cached
        async with sem:
            text, tokens = await self._chat_judge_async(system, user, max_tokens)
        p_f, p_r, meta = _judge_verdict(text, tokens)
        _cache_write(cp, p_f, p_r, meta)
        return p_f, p_r, meta

    async def judge_many(
        self,
        system: str,
        users: list[str],
        max_tokens: int = 400,
    ) -> list[tuple[float, float, dict]]:
        """Batch judging; input order preserved."""
        sem = asyncio.Semaphore(self.concurrency)  # bound to the current event loop
        return list(
            await asyncio.gather(*[self.judge_one(system, user, sem, max_tokens) for user in users])
        )
