"""Адаптер реальной схемы кураторов (data/raw/alfa/data.csv) → канонический Case.

id всегда с префиксом ``alfa_`` — по нему guard (src/common/guard.py)
не пускает реальные данные в cloud-режим; в meta пишется synthetic=False.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from rag_reliability.common.schemas import Case

_N_CHUNKS = 8
_CLIENT_RE = re.compile(r"Клиент:\s*(.*?)(?=\n(?:Ассистент|Оператор|Клиент):|\Z)", re.S)


def make_case_id(dialog: str, answer: str) -> str:
    """Детерминированный id от содержимого строки (переживает переупорядочивание)."""
    digest = hashlib.sha1((dialog + "\x00" + answer).encode("utf-8")).hexdigest()
    return "alfa_" + digest[:12]


def extract_last_client_turn(full_dialog: str) -> str | None:
    """Последняя реплика «Клиент:» до следующей роли или конца; None, если её нет."""
    matches = _CLIENT_RE.findall(full_dialog)
    if not matches:
        return None
    return matches[-1].strip()


def _to01(v: object) -> int:
    """Булево значение из CSV (bool / 'True' / 1 / '1') → 0/1."""
    if isinstance(v, str):
        s = v.strip().lower()
        return int(s == "true" or s == "1")
    return int(bool(v))


def _is_na(v: object) -> bool:
    """NaN/None/пустая строка."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    return isinstance(v, str) and not v.strip()


def _parse_markers(raw: object) -> tuple[list[str], bool]:
    """Строковый repr списка маркеров → (список, флаг ошибки парсинга)."""
    if _is_na(raw):
        return [], False
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return [], True
    if not isinstance(parsed, list):
        return [], True
    return [str(m) for m in parsed], False


def load_alfa(csv_path: str | Path) -> list[Case]:
    """Читает CSV кураторов и возвращает список Case в порядке строк файла."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    cases: list[Case] = []
    for _, row in df.iterrows():
        full_dialog = "" if _is_na(row.get("full_dialog")) else str(row["full_dialog"])
        answer = "" if _is_na(row.get("answer")) else str(row["answer"])

        query_raw = extract_last_client_turn(full_dialog)
        no_client_turn = query_raw is None

        context = [
            str(row[f"chunk_{i}"]).strip()
            for i in range(1, _N_CHUNKS + 1)
            if not _is_na(row.get(f"chunk_{i}"))
        ]

        markers, parse_error = _parse_markers(row.get("markers"))

        meta: dict = {
            "synthetic": False,
            "source": "alfa",
            "no_client_turn": bool(no_client_turn),
        }
        if parse_error:
            meta["markers_parse_error"] = True

        cases.append(
            Case(
                id=make_case_id(full_dialog, answer),
                query=query_raw or "",
                context=context,
                answer=answer,
                dialog=full_dialog.splitlines(),
                faith=_to01(row.get("binary_faithfulness")),
                rel=_to01(row.get("binary_relevancy")),
                markers=markers,
                meta=meta,
            )
        )
    return cases
