"""Feature extraction for the LettuceDetect-based reliability method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from rag_reliability.schema import RagSample


DEFAULT_MODEL_PATH = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


@dataclass(frozen=True)
class FeatureConfig:
    model_path: str = DEFAULT_MODEL_PATH
    threshold: float = 0.5
    device: str | None = None

    def as_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


def make_detector(config: FeatureConfig):
    """Create the LettuceDetect detector lazily, with a useful error message."""
    try:
        import torch
        from lettucedetect.models.inference import HallucinationDetector
    except ImportError as exc:
        raise ImportError(
            "LettuceDetect dependencies are not installed. Run: "
            "uv pip install -r lettucedetect_method/requirements.txt"
        ) from exc

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return HallucinationDetector(
        method="transformer",
        model_path=config.model_path,
        device=device,
    )


def aggregate_token_scores(token_predictions: list[dict], threshold: float) -> list[float]:
    """Convert token-level LettuceDetect scores to [max, mean, fraction_above_threshold]."""
    probs = [float(item["prob"]) for item in token_predictions if "prob" in item]
    if not probs:
        return [0.0, 0.0, 0.0]

    max_prob = max(probs)
    mean_prob = sum(probs) / len(probs)
    fraction_unsupported = sum(prob > threshold for prob in probs) / len(probs)
    return [max_prob, mean_prob, fraction_unsupported]


def extract_features(
    samples: list["RagSample"],
    detector,
    threshold: float,
    desc: str = "lettucedetect",
) -> np.ndarray:
    """Extract three LettuceDetect features for every sample."""
    rows: list[list[float]] = []
    for sample in tqdm(samples, desc=desc):
        token_predictions = detector.predict(
            question=sample.question,
            context=[sample.context],
            answer=sample.answer,
            output_format="tokens",
        )
        rows.append(aggregate_token_scores(token_predictions, threshold))
    return np.asarray(rows, dtype=np.float32)
