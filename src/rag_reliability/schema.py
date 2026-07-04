"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ALLOWED_MARKERS: tuple[str, ...] = (
    "none",
    "unknown",
    "hallucination",
    "off_topic_answer",
    "incomplete_answer",
    "context_mixing",
    "contradiction",
    "unsupported_claim",
)


class RagSample(BaseModel):
    """One labeled (question, context, answer) triple."""

    id: str
    question: str
    context: str
    answer: str
    faithfulness: int = Field(ge=0, le=1)
    relevance: int = Field(ge=0, le=1)
    marker: str | None = None

    @field_validator("marker")
    @classmethod
    def _marker_allowed(cls, value: str | None) -> str | None:
        # Gold labels only; predicted markers (Prediction.marker_pred) stay
        # free-form so bad model outputs surface in metrics, not crashes.
        if value is not None and value not in ALLOWED_MARKERS:
            raise ValueError(f"marker must be one of {ALLOWED_MARKERS}, got {value!r}")
        return value

    @property
    def reliable(self) -> int:
        return int(self.faithfulness == 1 and self.relevance == 1)


class Prediction(BaseModel):
    """Parsed model output for one sample."""

    id: str
    faithfulness_pred: int = Field(ge=0, le=1)
    relevance_pred: int = Field(ge=0, le=1)
    marker_pred: str | None = None
    raw_output: str | None = None
    invalid_output: bool = False

    @property
    def reliable_pred(self) -> int:
        return int(self.faithfulness_pred == 1 and self.relevance_pred == 1)


class EvaluationResult(BaseModel):
    """Aggregate metrics over a prediction set.

    Marker fields are populated only when at least one prediction carries a
    marker (marker mode); in direct mode they stay None.
    """

    reliable_f1_macro: float
    faithfulness_f1_macro: float
    relevance_f1_macro: float
    invalid_output_rate: float
    total: int
    invalid_count: int
    marker_f1_macro: float | None = None
    marker_per_class_f1: dict[str, float] | None = None
    marker_confusion: dict[str, dict[str, int]] | None = None
