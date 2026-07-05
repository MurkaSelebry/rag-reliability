"""Local bootstrap helpers for the isolated LettuceDetect method directory."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def add_repo_src_to_path() -> None:
    """Allow scripts in this directory to import the local rag_reliability package."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
