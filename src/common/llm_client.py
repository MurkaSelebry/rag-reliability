"""Общий LLM-клиент: OpenAI-совместимый API (vLLM локально / OpenRouter в облаке).

Код не знает о провайдере ничего, кроме конфига: api_base/api_key/model +
опциональный llm.openrouter_extra_body, который пробрасывается в extra_body
каждого запроса (docs/07.3 п.2).

Guard (A2): при cloud-профиле КАЖДЫЙ вызов обязан явно пометить источник данных —
либо передать `case` (тогда срабатывает assert_case_cloud_safe), либо поднять
флаг `public_data=True` (публичные / синтетические данные). Забытый case = ошибка.

N-fallback (A1): если провайдер бросает ошибку на n>1, клиент автоматически
деградирует до последовательных одиночных запросов.
"""
from __future__ import annotations

import time

from openai import OpenAI

from .guard import DataLeakError, assert_case_cloud_safe
from .schemas import Case


class LLMClient:
    def __init__(self, cfg: dict, model: str | None = None):
        llm = cfg["llm"]
        self.profile: str = cfg.get("profile", "local")
        self.model: str = model or llm["model"]
        self.extra_body: dict = llm.get("openrouter_extra_body") or {}
        self.client = OpenAI(base_url=llm["api_base"], api_key=llm["api_key"])

    # ---------- низкоуровневый запрос с retry ----------

    def _request(self, *, messages: list[dict], temperature: float, n: int,
                 max_tokens: int, top_p: float, logprobs: bool,
                 retries: int = 3) -> list[dict]:
        """-> choices: [{text, tokens: [{token, logprob, top: {tok: lp}}]}]"""
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
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
                time.sleep(2 ** attempt)

    # ---------- публичный chat с guard (A2) и n-fallback (A1) ----------

    def chat(self, messages: list[dict], *, temperature: float = 0.0, n: int = 1,
             max_tokens: int = 512, top_p: float = 1.0, logprobs: bool = False,
             case: Case | None = None, public_data: bool = False) -> list[dict]:
        """Единственный публичный метод отправки запросов.

        Guard (A2): при cloud-профиле необходимо явно передать либо `case`
        (проверяется assert_case_cloud_safe), либо `public_data=True`.
        Локальный профиль не ограничен.

        N-fallback (A1): если первый запрос на n>1 бросает исключение,
        клиент деградирует до одиночных запросов и добирает нужное количество.
        """
        # --- guard (A2) ---
        if case is not None:
            assert_case_cloud_safe(case, self.profile)
        elif self.profile == "cloud" and not public_data:
            raise DataLeakError(
                "cloud-профиль: вызов без case и без public_data=True запрещён — "
                "пометь запрос по публичным данным явно или передай case")

        # --- первый запрос; при n>1 и ошибке провайдера — деградация (A1) ---
        try:
            choices = self._request(messages=messages, temperature=temperature, n=n,
                                    max_tokens=max_tokens, top_p=top_p, logprobs=logprobs)
        except Exception:
            if n <= 1:
                raise
            choices = []  # провайдер не умеет n>1 — добираем одиночными запросами ниже

        # --- добор недостающих сэмплов одиночными запросами ---
        while len(choices) < n:
            choices += self._request(messages=messages, temperature=temperature, n=1,
                                     max_tokens=max_tokens, top_p=top_p, logprobs=logprobs)
        return choices[:n]
