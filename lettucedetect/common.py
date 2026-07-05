"""Shared utilities for the LettuceDetect feature-extractor method.

Run scripts in this directory as files, for example:
    python lettucedetect/train_classifier.py ...

The directory name intentionally matches the third-party package name requested
by the project layout. Avoid `python -m lettucedetect...`, because that would
make Python resolve this local directory instead of the installed package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_reliability.schema import RagSample  # noqa: E402


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
            "uv pip install -r lettucedetect/requirements.txt"
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
    samples: list[RagSample],
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


def targets_from_samples(samples: list[RagSample]) -> np.ndarray:
    """Return the two binary targets expected by the downstream classifier."""
    return np.asarray([[s.faithfulness, s.relevance] for s in samples], dtype=np.int64)


def select_split(
    samples: list[RagSample],
    split: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[RagSample]:
    """Select all/train/val/test using the repository's existing split helper."""
    if split == "all":
        return samples

    from rag_reliability.dataset import split_samples

    train, val, test = split_samples(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    return {"train": train, "val": val, "test": test}[split]
