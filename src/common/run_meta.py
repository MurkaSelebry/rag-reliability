"""Метаданные прогона: run.yaml рядом с predictions (правило детерминизма CLAUDE.md)."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import yaml


def git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _redact(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    if isinstance(c.get("llm"), dict) and "api_key" in c["llm"]:
        c["llm"]["api_key"] = "***"
    return c


def save_run_yaml(out_dir: str | Path, cfg: dict, **extra) -> None:
    payload = {
        "config": _redact(cfg),
        "git_hash": git_hash(),
        "profile": cfg.get("profile", "local"),
        **extra,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
