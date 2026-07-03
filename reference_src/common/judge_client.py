"""Клиент LLM-судьи поверх vLLM (OpenAI-compatible).

Ключевая функция — judge(): одна генерация, из которой извлекаются
вероятности PASS/FAIL для faithfulness и relevance из logprobs токенов.
Схема Lynx: вердикт токенами PASS/FAIL, вероятность = softmax(logprob_PASS,
logprob_FAIL) на позиции вердикта.
"""
from __future__ import annotations
import math
import re
import time
from openai import OpenAI

VERDICT_RE = re.compile(r"FAITHFULNESS:\s*(PASS|FAIL).*?RELEVANCE:\s*(PASS|FAIL)", re.S | re.I)


class JudgeClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model

    # ---------- низкоуровневые примитивы ----------

    def chat(self, messages: list[dict], temperature: float = 0.0, n: int = 1,
             max_tokens: int = 512, top_p: float = 1.0, logprobs: bool = False,
             retries: int = 3) -> list[dict]:
        """Возвращает список choices: {text, tokens:[{token, logprob, top:{tok:lp}}]}"""
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                    n=n, max_tokens=max_tokens, top_p=top_p,
                    logprobs=logprobs, top_logprobs=20 if logprobs else None,
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

    # ---------- судья ----------

    @staticmethod
    def _pass_prob(tok: dict) -> float:
        """P(PASS) из top-logprobs позиции, где сгенерирован PASS|FAIL."""
        top = tok["top"] or {tok["token"]: tok["logprob"]}
        def lp(word):
            best = None
            for t, v in top.items():
                if t.strip().upper() == word:
                    best = v if best is None else max(best, v)
            return best
        lp_pass, lp_fail = lp("PASS"), lp("FAIL")
        if lp_pass is None and lp_fail is None:
            return 0.5
        if lp_pass is None:
            return 1.0 - 1.0 / (1.0 + math.exp(lp_fail))  # только FAIL виден
        if lp_fail is None:
            return 1.0 / (1.0 + math.exp(-lp_pass))
        m = max(lp_pass, lp_fail)
        e_p, e_f = math.exp(lp_pass - m), math.exp(lp_fail - m)
        return e_p / (e_p + e_f)

    def judge(self, system_prompt: str, user_prompt: str,
              max_tokens: int = 400) -> tuple[float, float, str]:
        """-> (p_faith, p_rel, raw_text). p_* — вероятность PASS по оси."""
        choices = self.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=max_tokens, logprobs=True,
        )
        text, tokens = choices[0]["text"], choices[0]["tokens"]
        # позиции вердиктов: токены PASS/FAIL в порядке появления
        verdict_positions = [t for t in tokens
                             if t["token"].strip().upper() in ("PASS", "FAIL")]
        if len(verdict_positions) >= 2:
            return self._pass_prob(verdict_positions[0]), self._pass_prob(verdict_positions[1]), text
        # fallback: парсим текст без вероятностей (0.9/0.1 как «жёсткая» уверенность)
        m = VERDICT_RE.search(text)
        if m:
            pf = 0.9 if m.group(1).upper() == "PASS" else 0.1
            pr = 0.9 if m.group(2).upper() == "PASS" else 0.1
            return pf, pr, text
        return 0.5, 0.5, text  # не распарсили — максимум неопределённости
