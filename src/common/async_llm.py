"""Асинхронный LLM-клиент с ограничением конкуррентности (семафор).

Ставится РЯДОМ с синхронным `llm_client.py`, а не вместо него: для будущего
масштаба (5k кейсов × 6 вариантов) нужны параллельные запросы к vLLM.

Переиспользует извлечение вердикта (`extract_verdict_probs`, `VERDICT_RE`) и
схему ключа кэша синхронного `JudgeClient`, поэтому async и sync прогоны делят
одни и те же файлы кэша и дают идентичные вероятности.

Guard (A2) и n-fallback (A1) — те же, что в синхронном клиенте.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from openai import AsyncOpenAI

from .guard import DataLeakError, assert_case_cloud_safe, assert_cloud_safe
from .llm_client import VERDICT_RE, extract_verdict_probs
from .schemas import Case


class AsyncLLMClient:
    """Асинхронный аналог LLMClient: тот же конфиг, тот же формат ответа."""

    def __init__(self, cfg: dict, model: str | None = None):
        llm = cfg["llm"]
        self.profile: str = cfg.get("profile", "local")
        self.model: str = model or llm["model"]
        self.extra_body: dict = dict(llm.get("openrouter_extra_body") or {})
        self.client = AsyncOpenAI(base_url=llm["api_base"], api_key=llm["api_key"])

    # ---------- низкоуровневый запрос с retry ----------

    async def _request(self, *, messages: list[dict], temperature: float, n: int,
                       max_tokens: int, top_p: float, logprobs: bool,
                       retries: int = 3) -> list[dict]:
        """-> choices: [{text, tokens: [{token, logprob, top: {tok: lp}}]}]"""
        for attempt in range(retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                    n=n, max_tokens=max_tokens, top_p=top_p,
                    logprobs=logprobs, top_logprobs=20 if logprobs else None,
                    extra_body=self.extra_body or None,
                )
                out = []
                for ch in resp.choices:
                    item = {"text": ch.message.content or "", "tokens": []}
                    if logprobs and ch.logprobs and ch.logprobs.content:
                        for t in ch.logprobs.content:
                            item["tokens"].append({
                                "token": t.token,
                                "logprob": t.logprob,
                                "top": {tt.token: tt.logprob for tt in (t.top_logprobs or [])},
                            })
                    out.append(item)
                return out
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    # ---------- публичный chat с guard (A2) и n-fallback (A1) ----------

    async def chat(self, messages: list[dict], *, temperature: float = 0.0, n: int = 1,
                   max_tokens: int = 512, top_p: float = 1.0, logprobs: bool = False,
                   case: Case | None = None, public_data: bool = False) -> list[dict]:
        """Единственный публичный метод отправки запросов (guard A2 + n-fallback A1)."""
        # --- guard (A2) ---
        if case is not None:
            assert_case_cloud_safe(case, self.profile)
        elif self.profile == "cloud" and not public_data:
            raise DataLeakError(
                "cloud-профиль: вызов без case и без public_data=True запрещён — "
                "пометь запрос по публичным данным явно или передай case")

        # --- первый запрос; при n>1 и ошибке провайдера — деградация (A1) ---
        try:
            choices = await self._request(messages=messages, temperature=temperature, n=n,
                                          max_tokens=max_tokens, top_p=top_p, logprobs=logprobs)
        except Exception:
            if n <= 1:
                raise
            choices = []  # провайдер не умеет n>1 — добираем одиночными запросами ниже

        # --- добор недостающих сэмплов одиночными запросами ---
        while len(choices) < n:
            new = await self._request(messages=messages, temperature=temperature, n=1,
                                      max_tokens=max_tokens, top_p=top_p, logprobs=logprobs)
            if not new:
                raise RuntimeError("провайдер вернул 0 choices при n=1 — добор невозможен")
            choices += new
        return choices[:n]


class AsyncJudgeClient(AsyncLLMClient):
    """Асинхронный судья: батч кейсов через семафор, кэш общий с sync JudgeClient."""

    def __init__(self, cfg: dict, cache_dir: str | Path | None = None, concurrency: int = 8):
        super().__init__(cfg)
        self.concurrency = concurrency
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, system: str, user: str) -> Path:
        """Тот же sha256-ключ, что у sync JudgeClient — кэш общий."""
        key = hashlib.sha256(
            "\x00".join((self.model, system, user)).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    async def _chat_judge_async(self, system: str, user: str,
                                max_tokens: int) -> tuple[str, list]:
        """Одна генерация T=0 с logprobs. Per-case guard уже отработал в judge_many,
        поэтому здесь public_data=True — case не нужен."""
        choices = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=max_tokens, logprobs=True, public_data=True)
        return choices[0]["text"], choices[0]["tokens"]

    async def judge_one(self, system: str, user: str, case: Case,
                        sem: asyncio.Semaphore,
                        max_tokens: int = 400) -> tuple[float, float, dict]:
        """-> (p_faith, p_rel, meta). Кейс не теряется никогда (fallback до 0.5/0.5)."""
        cp = self._cache_path(system, user) if self.cache_dir else None
        if cp is not None and cp.exists():
            try:
                d = json.loads(cp.read_text(encoding="utf-8"))
                return d["p_faith"], d["p_rel"], d["meta"]
            except (json.JSONDecodeError, KeyError):
                pass  # повреждённый кэш (обрыв записи) — трактуем как промах
        async with sem:
            text, tokens = await self._chat_judge_async(system, user, max_tokens)
        probs = extract_verdict_probs(tokens)
        if probs is not None:
            p_f, p_r = probs
            meta = {"method": "logprobs", "raw": text[-400:]}
        else:
            m = VERDICT_RE.search(text)
            if m:
                p_f = 0.9 if m.group(1).upper() == "PASS" else 0.1
                p_r = 0.9 if m.group(2).upper() == "PASS" else 0.1
                meta = {"method": "regex", "raw": text[-400:]}
            else:
                p_f, p_r, meta = 0.5, 0.5, {"method": "default", "raw": text[-400:]}
        if cp is not None:
            tmp = cp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"p_faith": p_f, "p_rel": p_r, "meta": meta},
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(cp)  # атомарная замена — обрыв не оставит битый кэш
        return p_f, p_r, meta

    async def judge_many(self, system: str, items: list[tuple[Case, str]],
                         max_tokens: int = 400) -> list[tuple[float, float, dict]]:
        """Батч кейсов; guard по всему списку ДО любого запроса; порядок входа сохранён."""
        assert_cloud_safe([c for c, _ in items], self.profile)
        sem = asyncio.Semaphore(self.concurrency)  # семафор биндится к текущему loop
        return await asyncio.gather(
            *[self.judge_one(system, user, case, sem, max_tokens) for case, user in items])
