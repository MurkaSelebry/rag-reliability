"""Загрузка конфига: yaml + чтение .env + подстановка ${VAR} из окружения.

Правило проекта: код знает о профиле только api_base/api_key/model из конфига;
ключи не хардкодятся, а приходят через переменные окружения (docs/07.3 п.3).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def load_dotenv(path: str | Path = ".env") -> None:
    """Минимальный парсер .env (KEY=VALUE). Не перетирает уже установленные переменные."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(x) for x in obj]
    if isinstance(obj, str):
        def sub(m: re.Match) -> str:
            val = os.environ.get(m.group(1))
            if val is None:
                raise KeyError(f"переменная окружения {m.group(1)} не установлена (нужна конфигу)")
            return val
        return _VAR_RE.sub(sub, obj)
    return obj


def load_config(path: str | Path) -> dict:
    """Читает yaml, подставляет ${VAR}; .env в cwd подхватывается автоматически."""
    load_dotenv()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand(cfg)
