"""Robust parsing of model outputs into Prediction objects."""

from __future__ import annotations

import json
import re
from typing import Any

from rag_reliability.schema import Prediction

_FAITHFULNESS_RE = re.compile(r"faithfulness\"?\s*[:=]\s*\"?([01])\"?", re.IGNORECASE)
_RELEVANCE_RE = re.compile(r"relevance\"?\s*[:=]\s*\"?([01])\"?", re.IGNORECASE)
_MARKER_RE = re.compile(r"marker\"?\s*[:=]\s*\"([a-z_]+)\"", re.IGNORECASE)


def extract_json_object(text: str) -> str | None:
    """Return the first balanced, json-parseable {...} substring, or None."""
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try the next opening brace
                    return candidate
    return None


def normalize_binary(value: Any) -> int | None:
    """Coerce 0/1, "0"/"1", true/false (and string forms) to int 0/1."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("0", "false"):
            return 0
        if lowered in ("1", "true"):
            return 1
    return None


def parse_prediction(raw_output: str, sample_id: str, expect_marker: bool = False) -> Prediction:
    """Parse a raw LLM output. Falls back JSON -> regex -> conservative default.

    Conservative fallback predicts (0, 0), i.e. "unreliable", and sets
    invalid_output=True.
    """
    faithfulness: int | None = None
    relevance: int | None = None
    marker: str | None = None

    json_text = extract_json_object(raw_output)
    if json_text is not None:
        data = json.loads(json_text)
        if isinstance(data, dict):
            faithfulness = normalize_binary(data.get("faithfulness"))
            relevance = normalize_binary(data.get("relevance"))
            if expect_marker:
                raw_marker = data.get("marker")
                marker = raw_marker if isinstance(raw_marker, str) else None

    if faithfulness is None or relevance is None:
        faith_match = _FAITHFULNESS_RE.search(raw_output)
        rel_match = _RELEVANCE_RE.search(raw_output)
        if faith_match and rel_match:
            faithfulness = int(faith_match.group(1))
            relevance = int(rel_match.group(1))
            if expect_marker and marker is None:
                marker_match = _MARKER_RE.search(raw_output)
                marker = marker_match.group(1) if marker_match else None

    if faithfulness is None or relevance is None:
        return Prediction(
            id=sample_id,
            faithfulness_pred=0,
            relevance_pred=0,
            marker_pred=None,
            raw_output=raw_output,
            invalid_output=True,
        )

    return Prediction(
        id=sample_id,
        faithfulness_pred=faithfulness,
        relevance_pred=relevance,
        marker_pred=marker if expect_marker else None,
        raw_output=raw_output,
        invalid_output=False,
    )
