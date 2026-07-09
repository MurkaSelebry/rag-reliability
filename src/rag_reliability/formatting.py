"""Target/completion builders for SFT training records."""

from __future__ import annotations

import json

from rag_reliability.prompts import build_direct_prompt, build_marker_prompt
from rag_reliability.schema import RagSample

MODES = ("direct", "marker")


def resolve_marker(sample: RagSample) -> str:
    """Marker for training targets; falls back when the sample has none."""
    if sample.marker:
        return sample.marker
    if sample.faithfulness == 1 and sample.relevance == 1:
        return "none"
    return "unknown"


def build_direct_target(sample: RagSample) -> str:
    return json.dumps(
        {"faithfulness": sample.faithfulness, "relevance": sample.relevance},
        ensure_ascii=False,
        separators=(", ", ": "),
    )


def build_marker_target(sample: RagSample) -> str:
    return json.dumps(
        {
            "marker": resolve_marker(sample),
            "faithfulness": sample.faithfulness,
            "relevance": sample.relevance,
        },
        ensure_ascii=False,
        separators=(", ", ": "),
    )


def build_training_record(sample: RagSample, mode: str) -> dict[str, str]:
    """One SFT record: {"prompt": ..., "completion": ...}."""
    if mode == "direct":
        return {"prompt": build_direct_prompt(sample), "completion": build_direct_target(sample)}
    if mode == "marker":
        return {"prompt": build_marker_prompt(sample), "completion": build_marker_target(sample)}
    raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")


def build_chat_training_record(sample: RagSample, mode: str) -> dict[str, list[dict[str, str]]]:
    """One SFT chat record compatible with mlx_lm's ChatDataset."""
    record = build_training_record(sample, mode)
    return {
        "messages": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["completion"]},
        ]
    }
