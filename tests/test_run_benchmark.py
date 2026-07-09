"""Tests for the unified benchmark runner."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_benchmark",
    Path(__file__).parents[1] / "scripts" / "run_benchmark.py",
)
assert _SPEC is not None
run_benchmark = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["run_benchmark"] = run_benchmark
_SPEC.loader.exec_module(run_benchmark)


def test_build_method_commands_for_dummy_and_evaluation(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="dummy_marker",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
    )

    assert run.name == "dummy_marker"
    assert run.predictions_path == tmp_path / "dummy_marker" / "predictions.jsonl"
    assert run.metrics_path == tmp_path / "dummy_marker" / "metrics.json"
    assert run.run_command == [
        "python",
        "scripts/run_prompt_baseline.py",
        "--data",
        "data/dummy.jsonl",
        "--output",
        str(tmp_path / "dummy_marker" / "predictions.jsonl"),
        "--mode",
        "marker",
        "--backend",
        "dummy",
        "--dummy-strategy",
        "keyword",
    ]
    assert run.evaluate_command[-4:] == [
        "--predictions",
        str(tmp_path / "dummy_marker" / "predictions.jsonl"),
        "--output",
        str(tmp_path / "dummy_marker" / "metrics.json"),
    ]


def test_build_method_commands_for_encoder_export_predictions(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="encoder",
        data=Path("data/organizers.jsonl"),
        output_dir=tmp_path,
        python="python",
    )

    assert run.run_command[:3] == ["python", "scripts/train_encoder_baseline.py", "--data"]
    assert "--predictions-output" in run.run_command
    assert str(tmp_path / "encoder" / "predictions.jsonl") in run.run_command
    assert run.evaluate_command[0:2] == ["python", "scripts/evaluate.py"]


def test_build_method_commands_for_lora_and_lettucedetect(tmp_path: Path) -> None:
    lora = run_benchmark.build_method_run(
        method="lora_direct",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
    )
    lettuce = run_benchmark.build_method_run(
        method="lettucedetect",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
    )

    assert lora.run_command[1:4] == ["scripts/infer.py", "--data", "data/dummy.jsonl"]
    assert "--adapter-path" in lora.run_command
    assert lettuce.run_command[1:4] == [
        "scripts/infer_lettucedetect.py",
        "--data",
        "data/dummy.jsonl",
    ]
    assert "--model" in lettuce.run_command
