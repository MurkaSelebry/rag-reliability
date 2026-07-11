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

import hashlib
import json
import math
import re
import time
from pathlib import Path

from openai import OpenAI

from rag_reliability.common.guard import DataLeakError, assert_case_cloud_safe
from rag_reliability.common.schemas import Case


class LLMClient:
    def __init__(self, cfg: dict, model: str | None = None):
        llm = cfg["llm"]
        self.profile: str = cfg.get("profile", "local")
        self.model: str = model or llm["model"]
        self.extra_body: dict = dict(llm.get("openrouter_extra_body") or {})
        # явный opt-in владельца данных (guard.allow_real_data) — см. guard.py
        self.allow_real: bool = bool((cfg.get("guard") or {}).get("allow_real_data"))
        self.client = OpenAI(base_url=llm["api_base"], api_key=llm["api_key"])

    # ---------- низкоуровневый запрос с retry ----------

    def _request(
        self,
        *,
        messages: list[dict],
        temperature: float,
        n: int,
        max_tokens: int,
        top_p: float,
        logprobs: bool,
        retries: int = 3,
    ) -> list[dict]:
        """-> choices: [{text, tokens: [{token, logprob, top: {tok: lp}}]}]"""
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
                if resp.choices is None:
                    raise RuntimeError("провайдер вернул choices=None (не тянет параметры запроса)")
                out = []
                for ch in resp.choices:
                    item = {"text": ch.message.content or "", "tokens": []}
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
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)

    # ---------- публичный chat с guard (A2) и n-fallback (A1) ----------

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        top_p: float = 1.0,
        logprobs: bool = False,
        case: Case | None = None,
        public_data: bool = False,
    ) -> list[dict]:
        """Единственный публичный метод отправки запросов.

        Guard (A2): при cloud-профиле необходимо явно передать либо `case`
        (проверяется assert_case_cloud_safe), либо `public_data=True`.
        Локальный профиль не ограничен.

        N-fallback (A1): если первый запрос на n>1 бросает исключение,
        клиент деградирует до одиночных запросов и добирает нужное количество.
        """
        # --- guard (A2) ---
        if case is not None:
            assert_case_cloud_safe(case, self.profile, allow_real=self.allow_real)
        elif self.profile == "cloud" and not public_data:
            raise DataLeakError(
                "cloud-профиль: вызов без case и без public_data=True запрещён — "
                "пометь запрос по публичным данным явно или передай case"
            )

        # --- первый запрос; при n>1 и ошибке провайдера — деградация (A1) ---
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
            choices = []  # провайдер не умеет n>1 — добираем одиночными запросами ниже

        # --- добор недостающих сэмплов одиночными запросами ---
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
                raise RuntimeError("провайдер вернул 0 choices при n=1 — добор невозможен")
            choices += new
        return choices[:n]


# ---------- извлечение вероятностей вердикта (Метод 3 и все judge-методы) ----------

VERDICT_RE = re.compile(r"FAITHFULNESS:\s*(PASS|FAIL).*?RELEVANCE:\s*(PASS|FAIL)", re.S | re.I)
_WORDS = ("PASS", "FAIL")


def _verdict_positions(tokens: list[dict]) -> list[int]:
    """Позиции начала PASS/FAIL: целый токен или склейка подряд идущих BPE-подтокенов."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        t = tokens[i]["token"].strip().upper()
        if t in _WORDS:
            out.append(i)
            i += 1
            continue
        matched = False
        for w in _WORDS:
            if t and t != w and w.startswith(t):
                acc, j = t, i + 1
                while j < n and acc != w and w.startswith(acc):
                    piece = tokens[j]["token"].strip().upper()
                    if not piece:
                        break  # пробельный подтокен разрывает склейку
                    acc += piece
                    j += 1
                if acc == w:
                    out.append(i)
                    i = j
                    matched = True
                    break
        if not matched:
            i += 1
    return out


def _side_logprob(top: dict[str, float], word: str) -> float | None:
    """logprob стороны word на позиции: точные совпадения приоритетнее префиксов.

    Префиксы (>=2 симв.) используются только когда целого токена в top нет —
    иначе масса префикса (включающая чужие продолжения) завышает сторону.
    """
    exact, prefix = [], []
    for t, v in top.items():
        s = t.strip().upper()
        if s == word:
            exact.append(v)
        elif len(s) >= 2 and word.startswith(s):
            prefix.append(v)
    lps = exact or prefix
    if not lps:
        return None
    m = max(lps)
    return m + math.log(sum(math.exp(v - m) for v in lps))


def _pass_prob(tok: dict) -> float:
    """P(PASS) на позиции вердикта: softmax пары; одна сторона -> сигмоида; пусто -> 0.5."""
    top = tok["top"] or {tok["token"]: tok["logprob"]}
    lp_pass, lp_fail = _side_logprob(top, "PASS"), _side_logprob(top, "FAIL")
    if lp_pass is None and lp_fail is None:
        return 0.5
    if lp_pass is None:
        return 1.0 - 1.0 / (1.0 + math.exp(-lp_fail))
    if lp_fail is None:
        return 1.0 / (1.0 + math.exp(-lp_pass))
    m = max(lp_pass, lp_fail)
    e_p, e_f = math.exp(lp_pass - m), math.exp(lp_fail - m)
    return e_p / (e_p + e_f)


def extract_verdict_probs(tokens: list[dict]) -> tuple[float, float] | None:
    """(p_faith, p_rel) по 1-й и 2-й позициям вердикта; None, если позиций < 2."""
    pos = _verdict_positions(tokens)
    if len(pos) < 2:
        return None
    return _pass_prob(tokens[pos[0]]), _pass_prob(tokens[pos[1]])


class JudgeClient(LLMClient):
    """Судья: одна генерация T=0 с logprobs; цепочка fallback logprobs -> regex -> 0.5/0.5."""

    def __init__(self, cfg: dict, cache_dir: str | Path | None = None):
        super().__init__(cfg)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, system: str, user: str) -> Path:
        key = hashlib.sha256("\x00".join((self.model, system, user)).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _chat_judge(
        self, system: str, user: str, max_tokens: int, case: Case | None = None
    ) -> tuple[str, list]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            choices = self.chat(
                msgs, temperature=0.0, max_tokens=max_tokens, logprobs=True, case=case
            )
        except RuntimeError:
            # провайдер не тянет logprobs — деградация: текст без токенов (regex-путь)
            choices = self.chat(
                msgs, temperature=0.0, max_tokens=max_tokens, logprobs=False, case=case
            )
        return choices[0]["text"], choices[0]["tokens"]

    def judge(
        self, system: str, user: str, case: Case | None = None, max_tokens: int = 400
    ) -> tuple[float, float, dict]:
        """-> (p_faith, p_rel, meta). Кейс не теряется никогда (fallback до 0.5/0.5)."""
        cp = self._cache_path(system, user) if self.cache_dir else None
        if cp is not None and cp.exists():
            try:
                d = json.loads(cp.read_text(encoding="utf-8"))
                return d["p_faith"], d["p_rel"], d["meta"]
            except (json.JSONDecodeError, KeyError):
                pass  # повреждённый кэш (обрыв записи) — трактуем как промах
        text, tokens = self._chat_judge(system, user, max_tokens, case=case)
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
            tmp.write_text(
                json.dumps({"p_faith": p_f, "p_rel": p_r, "meta": meta}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(cp)  # атомарная замена — обрыв не оставит битый кэш
        return p_f, p_r, meta
