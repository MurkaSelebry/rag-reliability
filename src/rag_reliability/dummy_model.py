"""Pseudo-LLM predictors for testing the pipeline without a real model.

Each strategy returns a raw string (as an LLM would), so outputs flow through
the real parser in parsing.py.
"""

from __future__ import annotations

import json

from rag_reliability.formatting import build_direct_target, build_marker_target
from rag_reliability.schema import RagSample

STRATEGIES = ("always_reliable", "keyword", "echo_gold")

_OFF_TOPIC_HINTS = ("кстати", "рекомендую также")
_INCOMPLETE_HINTS = ("обратитесь в отделение", "не могу сказать точно")
_HALLUCINATION_HINTS = ("3%", "как известно", "обычно")


class DummyPredictor:
    """Deterministic stand-in for an LLM judge.

    Strategies:
    - always_reliable: constant {"faithfulness": 1, "relevance": 1}.
    - keyword: crude keyword heuristics over the answer text.
    - echo_gold: returns the gold labels. Pipeline smoke test ONLY — never
      report it as a baseline; on real data it is label leakage.
    """

    def __init__(self, strategy: str = "always_reliable", mode: str = "direct") -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}, expected one of {STRATEGIES}")
        if mode not in ("direct", "marker"):
            raise ValueError(f"Unknown mode {mode!r}, expected 'direct' or 'marker'")
        self.strategy = strategy
        self.mode = mode

    def predict(self, sample: RagSample) -> str:
        if self.strategy == "always_reliable":
            return self._render(faithfulness=1, relevance=1, marker="none")
        if self.strategy == "keyword":
            return self._keyword_predict(sample)
        # echo_gold
        if self.mode == "marker":
            return build_marker_target(sample)
        return build_direct_target(sample)

    def _keyword_predict(self, sample: RagSample) -> str:
        answer = sample.answer.lower()
        if any(hint in answer for hint in _OFF_TOPIC_HINTS):
            return self._render(faithfulness=1, relevance=0, marker="off_topic_answer")
        if any(hint in answer for hint in _INCOMPLETE_HINTS):
            return self._render(faithfulness=1, relevance=0, marker="incomplete_answer")
        if any(hint in answer for hint in _HALLUCINATION_HINTS):
            return self._render(faithfulness=0, relevance=1, marker="hallucination")
        return self._render(faithfulness=1, relevance=1, marker="none")

    def _render(self, faithfulness: int, relevance: int, marker: str) -> str:
        payload: dict[str, object] = {}
        if self.mode == "marker":
            payload["marker"] = marker
        payload["faithfulness"] = faithfulness
        payload["relevance"] = relevance
        return json.dumps(payload, ensure_ascii=False)
