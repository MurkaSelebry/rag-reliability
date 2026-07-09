"""Optional OpenAI-compatible chat client with file cache for Method 3."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

Messages = list[dict[str, str]]
Transport = Callable[..., str]


def default_transport(
    messages: Messages,
    *,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError('Install OpenAI-compatible backend deps with: uv pip install -e ".[cloud]"') from exc

    client = OpenAI(base_url=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


class CachedChatClient:
    """Cache chat completions by model, endpoint and messages."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        cache_dir: str | Path | None = None,
        transport: Transport = default_transport,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transport = transport
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, messages: Messages, max_tokens: int, temperature: float) -> str:
        payload: dict[str, Any] = {
            "api_base": self.api_base,
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def chat(self, messages: Messages, *, max_tokens: int = 400, temperature: float = 0.0) -> str:
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{self.cache_key(messages, max_tokens, temperature)}.json"
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return str(payload["content"])

        content = self.transport(
            messages,
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if cache_path is not None:
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache_path)
        return content
