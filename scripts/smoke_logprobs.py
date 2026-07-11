"""Smoke-тест logprobs провайдера (первая задача cloud-режима, docs/07.1 и 07.3 п.4).

Проверяет:
  (а) top-логпробы реально приходят (logprobs=true, top_logprobs=20);
  (б) токены PASS/FAIL восстановимы — целиком или склейкой BPE-подтокенов;
  (в) при повторных запросах отвечает один и тот же провайдер.

Запуск: python scripts/smoke_logprobs.py --config configs/config.cloud.yaml -n 3
Код возврата 0 = провайдер пригоден для Метода 3; 1 = сменить provider.order.
"""

from __future__ import annotations

import argparse
import sys

from openai import OpenAI

from rag_reliability.common.config import load_config

SYSTEM = "Отвечай строго в заданном формате, без лишних слов."
USER = "Выведи дословно две строки:\nFAITHFULNESS: PASS\nRELEVANCE: FAIL"


def recoverable(tokens: list[str], word: str) -> bool:
    """word восстановим: целый токен или склейка подряд идущих подтокенов."""
    ups = [t.strip().upper() for t in tokens]
    if word in ups:
        return True
    for i in range(len(ups)):
        acc = ""
        for j in range(i, len(ups)):
            acc += ups[j]
            if acc == word:
                return True
            if not word.startswith(acc):
                break
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.cloud.yaml")
    ap.add_argument("-n", type=int, default=3, help="число повторов (стабильность провайдера)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    llm = cfg["llm"]
    client = OpenAI(base_url=llm["api_base"], api_key=llm["api_key"])
    extra = llm.get("openrouter_extra_body") or {}

    providers: set[str | None] = set()
    ok = True
    for i in range(args.n):
        resp = client.chat.completions.create(
            model=llm["model"],
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
            temperature=0.0,
            max_tokens=40,
            logprobs=True,
            top_logprobs=20,
            extra_body=extra,
        )
        provider = getattr(resp, "provider", None) or (resp.model_extra or {}).get("provider")
        providers.add(provider)
        ch = resp.choices[0]
        content = ch.logprobs.content if ch.logprobs else None
        print(f"\n=== запрос {i + 1}/{args.n} | provider={provider} | model={resp.model}")
        print("текст:", repr(ch.message.content))
        if not content:
            print("!! top-логпробы НЕ пришли — провайдер непригоден для Метода 3")
            ok = False
            continue
        toks = [t.token for t in content]
        print("токены:", toks)
        for t in content:
            up = t.token.strip().upper()
            if up and ("PASS".startswith(up) or "FAIL".startswith(up)):
                alts = {a.token: round(a.logprob, 3) for a in (t.top_logprobs or [])}
                print(f"  позиция вердикта {t.token!r}: top-альтернативы {alts}")
                if not alts:
                    print("  !! top_logprobs пуст на позиции вердикта")
                    ok = False
        for w in ("PASS", "FAIL"):
            r = recoverable(toks, w)
            print(f"  {w} восстановим: {r}")
            ok = ok and r

    print(f"\nпровайдеры за {args.n} запросов: {providers}")
    if len(providers) > 1:
        print("!! провайдер меняется между запросами — зафиксировать provider.order в конфиге")
        ok = False
    print("ИТОГ:", "OK — провайдер пригоден" if ok else "FAIL — см. docs/07.3 п.4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
