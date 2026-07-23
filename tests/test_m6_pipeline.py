"""End-to-end tests for the resumable Method 6 pipeline."""

import json
import subprocess
import sys
from pathlib import Path

from rag_reliability.dataset import load_jsonl
from rag_reliability.metrics import evaluate_predictions
from rag_reliability.schema import Prediction

REPO = Path(__file__).resolve().parents[1]


def _write_data(tmp_path: Path) -> Path:
    rows = [
        {
            "id": f"s{i}",
            "question": "вопрос?",
            "context": "контекст.",
            "answer": "ответ.",
            "faithfulness": i % 2,
            "relevance": 1,
        }
        for i in range(4)
    ]
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path


def _run_pipeline(tmp_path: Path, data: Path, extra: tuple[str, ...] = ()) -> Path:
    output = tmp_path / "preds.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_m6_pipeline.py",
            "--data",
            str(data),
            "--samples-dir",
            str(tmp_path / "samples"),
            "--features",
            str(tmp_path / "features.jsonl"),
            "--output",
            str(output),
            "--backend",
            "dummy",
            "--features-backend",
            "dummy",
            "--n-samples",
            "3",
            *extra,
        ],
        check=True,
        cwd=REPO,
    )
    return output


def test_pipeline_dummy_end_to_end(tmp_path: Path) -> None:
    data = _write_data(tmp_path)

    output = _run_pipeline(tmp_path, data)

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    predictions = [Prediction.model_validate_json(line) for line in lines]
    assert len(predictions) == 4
    assert all(prediction.faithfulness_prob is not None for prediction in predictions)
    result = evaluate_predictions(load_jsonl(data), predictions)
    assert result.total == 4


def test_pipeline_reuses_existing_features(tmp_path: Path) -> None:
    data = _write_data(tmp_path)
    _run_pipeline(tmp_path, data)
    features = tmp_path / "features.jsonl"
    mtime = features.stat().st_mtime_ns

    _run_pipeline(tmp_path, data)

    assert features.stat().st_mtime_ns == mtime
