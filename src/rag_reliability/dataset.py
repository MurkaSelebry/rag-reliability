"""JSONL IO, splitting and training-file preparation."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from rag_reliability.formatting import build_training_record
from rag_reliability.schema import RagSample


def load_jsonl(path: str | Path) -> list[RagSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")
    samples: list[RagSample] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(RagSample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid record at {path}:{line_no}: {exc}") from exc
    return samples


def save_jsonl(records: Iterable[dict | BaseModel], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = record.model_dump() if isinstance(record, BaseModel) else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def split_samples(
    samples: list[RagSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[RagSample], list[RagSample], list[RagSample]]:
    """Split into (train, val, test), stratified by the reliable label."""
    if not 0 < train_ratio + val_ratio < 1:
        raise ValueError("train_ratio + val_ratio must be in (0, 1)")

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in (0, 1):
        group = [s for s in samples if s.reliable == label]
        rng.shuffle(group)
        n_train = round(len(group) * train_ratio)
        n_val = round(len(group) * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_training_jsonl(samples: list[RagSample], path: str | Path, mode: str) -> None:
    """Write {"prompt", "completion"} SFT records for the given mode."""
    save_jsonl((build_training_record(s, mode) for s in samples), path)
