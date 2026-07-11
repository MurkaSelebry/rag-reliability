"""Evaluation metrics: reliable/faithfulness/relevance F1-macro + invalid rate."""

from __future__ import annotations

from sklearn.metrics import f1_score

from rag_reliability.formatting import resolve_marker
from rag_reliability.schema import EvaluationResult, Prediction, RagSample


def _marker_metrics(
    samples: list[RagSample], pred_by_id: dict[str, Prediction]
) -> tuple[float, dict[str, float], dict[str, dict[str, int]]]:
    """Per-marker F1 and gold->pred confusion counts.

    Gold markers fall back like training targets (none/unknown); predictions
    without a marker (parse fallback) count as "unknown".
    """
    gold = [resolve_marker(s) for s in samples]
    pred = [pred_by_id[s.id].marker_pred or "unknown" for s in samples]

    labels = sorted(set(gold) | set(pred))
    per_class = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    macro = float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))

    confusion: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, pred, strict=True):
        row = confusion.setdefault(g, {})
        row[p] = row.get(p, 0) + 1

    return macro, {label: float(v) for label, v in zip(labels, per_class, strict=True)}, confusion


def evaluate_predictions(
    samples: list[RagSample], predictions: list[Prediction]
) -> EvaluationResult:
    """Join predictions to samples by id and compute macro-F1 metrics.

    reliable = faithfulness AND relevance, on both gold and predicted sides.
    Invalid outputs keep their conservative (0, 0) predictions and are counted
    in invalid_output_rate.
    """
    if not samples:
        raise ValueError("No samples to evaluate")

    pred_by_id: dict[str, Prediction] = {}
    for p in predictions:
        if p.id in pred_by_id:
            raise ValueError(f"Duplicate prediction id: {p.id!r}")
        pred_by_id[p.id] = p
    missing = [s.id for s in samples if s.id not in pred_by_id]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} sample(s): {missing[:5]}...")

    faithfulness_true, faithfulness_pred = [], []
    relevance_true, relevance_pred = [], []
    reliable_true, reliable_pred = [], []
    invalid_count = 0

    for sample in samples:
        pred = pred_by_id[sample.id]
        faithfulness_true.append(sample.faithfulness)
        relevance_true.append(sample.relevance)
        reliable_true.append(sample.reliable)
        faithfulness_pred.append(pred.faithfulness_pred)
        relevance_pred.append(pred.relevance_pred)
        reliable_pred.append(pred.reliable_pred)
        if pred.invalid_output:
            invalid_count += 1

    def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    marker_macro: float | None = None
    marker_per_class: dict[str, float] | None = None
    marker_confusion: dict[str, dict[str, int]] | None = None
    if any(p.marker_pred is not None for p in pred_by_id.values()):
        marker_macro, marker_per_class, marker_confusion = _marker_metrics(samples, pred_by_id)

    total = len(samples)
    return EvaluationResult(
        reliable_f1_macro=macro_f1(reliable_true, reliable_pred),
        faithfulness_f1_macro=macro_f1(faithfulness_true, faithfulness_pred),
        relevance_f1_macro=macro_f1(relevance_true, relevance_pred),
        invalid_output_rate=invalid_count / total,
        total=total,
        invalid_count=invalid_count,
        marker_f1_macro=marker_macro,
        marker_per_class_f1=marker_per_class,
        marker_confusion=marker_confusion,
    )
