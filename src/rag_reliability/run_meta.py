"""Run provenance: CLI args + git hash + timestamp next to every predictions file."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path


def git_short_hash() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_run_meta(path: str | Path, args: argparse.Namespace) -> None:
    payload = {
        "args": {key: str(value) for key, value in sorted(vars(args).items())},
        "git_hash": git_short_hash(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
