"""Tests for the manual demo UI backend helpers."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "serve_demo",
    Path(__file__).parents[1] / "scripts" / "serve_demo.py",
)
assert _SPEC is not None
serve_demo = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["serve_demo"] = serve_demo
_SPEC.loader.exec_module(serve_demo)


def test_build_manual_sample_uses_stable_manual_id() -> None:
    sample = serve_demo.build_manual_sample(
        question="Как подключить Alfa Pay?",
        context="Alfa Pay подключается в настройках карты.",
        answer="Откройте карту и выберите оплату смартфоном.",
        faithfulness=1,
        relevance=1,
        marker="none",
    )

    assert sample.id == "manual_000001"
    assert sample.reliable == 1


def test_run_manual_method_returns_dummy_prediction() -> None:
    result = serve_demo.run_manual_method(
        method="dummy_direct",
        question="Вопрос",
        context="Контекст",
        answer="Ответ",
        faithfulness=1,
        relevance=1,
        marker="none",
    )

    assert result["available"] is True
    assert result["prediction"]["faithfulness_pred"] == 1
    assert result["prediction"]["relevance_pred"] == 1
    assert result["prediction"]["reliable_pred"] == 1
    assert result["gold"]["reliable"] == 1


def test_run_manual_method_reports_disabled_encoder() -> None:
    result = serve_demo.run_manual_method(
        method="encoder",
        question="Вопрос",
        context="Контекст",
        answer="Ответ",
        faithfulness=None,
        relevance=None,
        marker=None,
    )

    assert result["available"] is False
    assert "not wired for single-example inference" in result["error"]


def test_method_statuses_include_artifact_paths() -> None:
    statuses = serve_demo.method_statuses()

    assert statuses["dummy_direct"]["available"] is True
    assert statuses["lora_direct"]["artifact"] == "results/adapters_direct"
    assert statuses["lettucedetect"]["artifact"] == "results/lettucedetect/classifier.joblib"


def test_method_choice_labels_show_availability() -> None:
    labels = serve_demo.method_choice_labels(
        {
            "dummy_direct": {"available": True, "artifact": None},
            "lora_direct": {"available": False, "artifact": "results/adapters_direct"},
        }
    )

    assert labels == [
        "dummy_direct — available",
        "lora_direct — missing: results/adapters_direct",
    ]
    assert serve_demo.method_from_choice(labels[0]) == "dummy_direct"


def test_example_choices_and_loading(tmp_path: Path) -> None:
    data_path = tmp_path / "samples.jsonl"
    data_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"s1","question":"q1","context":"c1","answer":"a1",'
                    '"faithfulness":1,"relevance":1,"marker":"none"}'
                ),
                (
                    '{"id":"s2","question":"q2","context":"c2","answer":"a2",'
                    '"faithfulness":0,"relevance":1,"marker":"reason_other"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    choices = serve_demo.example_choices(str(data_path), limit=1)
    loaded = serve_demo.load_example_choice(choices[0], str(data_path))

    assert choices == ["0 — s1 — reliable=1"]
    assert loaded == ("q1", "c1", "a1", 1, 1, "none")


def test_run_manual_methods_returns_table_summary_and_raw_outputs() -> None:
    result = serve_demo.run_manual_methods(
        methods=["dummy_direct", "dummy_marker"],
        question="Вопрос",
        context="Контекст",
        answer="Ответ",
        faithfulness=1,
        relevance=1,
        marker="none",
    )

    assert len(result["rows"]) == 2
    assert result["rows"][0]["method"] == "dummy_direct"
    assert "dummy_direct: reliable=1" in result["summary"]
    assert result["raw_outputs"]["dummy_direct"].startswith("{")


def test_history_keeps_latest_rows() -> None:
    history = serve_demo.update_history(
        [{"method": "old", "reliable": 0}],
        [{"method": "new", "reliable": 1}],
        max_rows=1,
    )

    assert history == [{"method": "new", "reliable": 1}]


def test_batch_command_uses_selected_methods() -> None:
    command = serve_demo.build_batch_command(
        data_path="data/organizers.jsonl",
        methods=["dummy_direct", "dummy_marker"],
        output_dir="results/demo_batch",
    )

    assert command == (
        "python scripts/run_benchmark.py --data data/organizers.jsonl "
        "--methods dummy_direct,dummy_marker --output-dir results/demo_batch"
    )
