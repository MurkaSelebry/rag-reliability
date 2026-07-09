"""Pure feature logic for Method 6 SelfCheck-style scoring.

This module is intentionally dependency-light. Heavy generation, NLI and
embedding model calls stay outside the unified benchmark runner; the runner
consumes precomputed feature JSONL and emits standard ``Prediction`` records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from rag_reliability.schema import RagSample

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?|[^\s]+", re.UNICODE)


def sentences(text: str) -> list[str]:
    """Split text into rough sentence spans, preserving a fallback for blanks."""
    stripped = text.strip()
    if not stripped:
        return [text]
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def selfcheck_scores(answer: str, samples: list[str], nli) -> dict[str, float]:
    """Return average/max contradiction of answer sentences against sampled answers."""
    if not samples:
        raise ValueError("empty samples list")
    answer_sentences = sentences(answer)
    pairs = [(sample, sentence) for sentence in answer_sentences for sample in samples]
    scored = nli.score(pairs)
    contra = np.array([row["contra"] for row in scored]).reshape(
        len(answer_sentences),
        len(samples),
    )
    per_sentence = contra.mean(axis=1)
    return {
        "selfcheck_contra_mean": float(per_sentence.mean()),
        "selfcheck_contra_max": float(per_sentence.max()),
    }


def semantic_clusters(texts: list[str], nli, threshold: float) -> list[int]:
    """Cluster texts by bidirectional entailment using a tiny union-find."""
    n = len(texts)
    pairs: list[tuple[str, str]] = []
    pair_indexes: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.extend([(texts[i], texts[j]), (texts[j], texts[i])])
            pair_indexes.append((i, j))

    scored = nli.score(pairs) if pairs else []
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for result_index, (left, right) in enumerate(pair_indexes):
        if (
            scored[2 * result_index]["entail"] > threshold
            and scored[2 * result_index + 1]["entail"] > threshold
        ):
            parent[find(left)] = find(right)
    return [find(index) for index in range(n)]


def entropy_features(answer: str, samples: list[str], nli, threshold: float) -> dict[str, float]:
    """Return semantic entropy and cluster metadata for answer + sampled answers."""
    texts = [answer] + samples
    labels = semantic_clusters(texts, nli, threshold)
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    top_label = unique_labels[counts.argmax()]
    return {
        "semantic_entropy": entropy,
        "n_clusters": int(len(unique_labels)),
        "answer_in_top_cluster": float(labels[0] == top_label),
    }


def load_sample_cache(cache_dir: str | Path, sample_id: str) -> list[str]:
    """Read generated answer samples for one case from ``{cache_dir}/{id}.json``."""
    path = Path(cache_dir) / f"{sample_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Method 6 sample cache not found: {path.resolve()}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not all(isinstance(sample, str) for sample in samples):
        raise ValueError(f"Invalid Method 6 sample cache at {path}: expected string samples")
    return samples


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity, guarding zero vectors."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(left @ right / denominator)


def build_feature_row(
    sample: RagSample,
    samples: list[str],
    *,
    nli,
    embedder,
    entail_threshold: float,
) -> dict[str, float | int | str]:
    """Build one Method 6 feature row for the shared RagSample schema."""
    features: dict[str, float | int | str] = {"id": sample.id}
    features.update(selfcheck_scores(sample.answer, samples, nli))
    features.update(entropy_features(sample.answer, samples, nli, entail_threshold))
    embeddings = embedder.encode(
        [f"query: {sample.question}", f"passage: {sample.answer}"],
        normalize_embeddings=True,
    )
    features["cos_q_a"] = cosine_similarity(np.asarray(embeddings[0]), np.asarray(embeddings[1]))
    return features
