"""Prompt builders for the LLM-as-judge task.

Instructions are in English (instruct models follow English instructions more
reliably); QUESTION/CONTEXT/ANSWER payloads may be in Russian.
"""

from __future__ import annotations

from rag_reliability.schema import ALLOWED_MARKERS, RagSample

__all__ = ["ALLOWED_MARKERS", "build_direct_prompt", "build_marker_prompt"]

_DEFINITIONS = """Definitions:
Faithfulness:
1 = the answer is fully supported by the provided context.
0 = the answer contains unsupported claims, contradicts the context, mixes facts incorrectly, or omits essential context.

Relevance:
1 = the answer directly addresses the user's question.
0 = the answer is off-topic, incomplete, or does not answer the question."""


def _payload(sample: RagSample) -> str:
    return f"""[QUESTION]
{sample.question}

[CONTEXT]
{sample.context}

[ANSWER]
{sample.answer}"""


def build_direct_prompt(sample: RagSample) -> str:
    """Prompt asking only for faithfulness/relevance JSON."""
    return f"""You are a strict evaluator of RAG answers.

Evaluate whether the ANSWER is faithful to the CONTEXT and relevant to the QUESTION.

{_DEFINITIONS}

Return only valid JSON with exactly these keys:
{{"faithfulness": 0 or 1, "relevance": 0 or 1}}

{_payload(sample)}"""


def build_marker_prompt(sample: RagSample) -> str:
    """Prompt asking for an error marker in addition to the binary labels."""
    markers = ", ".join(ALLOWED_MARKERS)
    return f"""You are a strict evaluator of RAG answers.

Evaluate whether the ANSWER is faithful to the CONTEXT and relevant to the QUESTION.
Also classify the error type of the answer with a marker.

{_DEFINITIONS}

Marker:
Choose exactly one marker from this list: {markers}.
Use "none" only when the answer is both faithful and relevant.
Use "unknown" when the answer is unreliable but no other marker fits.

Return only valid JSON with exactly these keys:
{{"marker": "...", "faithfulness": 0 or 1, "relevance": 0 or 1}}

{_payload(sample)}"""
