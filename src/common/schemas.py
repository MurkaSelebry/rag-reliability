"""Общие структуры данных и чтение/запись jsonl."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Case:
    id: str
    query: str
    context: list[str]  # список чанков
    answer: str
    dialog: list[str] = field(default_factory=list)
    faith: int | None = None  # 1 = faithful (PASS), 0 = FAIL; None на приватном тесте
    rel: int | None = None
    markers: list[str] = field(default_factory=list)
    meta: dict = field(
        default_factory=dict
    )  # напр. {"kind": ..., "synthetic": true} у псевдо-корпуса

    @property
    def reliable(self) -> int | None:
        if self.faith is None or self.rel is None:
            return None
        return int(self.faith == 1 and self.rel == 1)

    def ctx_text(self, max_chars: int | None = None) -> str:
        parts = [f"[Чанк {i + 1}] {c}" for i, c in enumerate(self.context)]
        text = "\n".join(parts)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n[контекст усечён]"
        return text

    def q_text(self) -> str:
        if self.dialog:
            hist = "\n".join(self.dialog)
            return f"История диалога:\n{hist}\nТекущий вопрос: {self.query}"
        return self.query


@dataclass
class Pred:
    id: str
    p_faith: float
    p_rel: float
    meta: dict = field(default_factory=dict)


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cases.append(
                Case(
                    id=str(d["id"]),
                    query=d["query"],
                    context=d["context"] if isinstance(d["context"], list) else [d["context"]],
                    answer=d["answer"],
                    dialog=d.get("dialog") or [],
                    faith=d.get("faith"),
                    rel=d.get("rel"),
                    markers=d.get("markers") or [],
                    meta=d.get("meta") or {},
                )
            )
    return cases


def save_preds(preds: list[Pred], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
