"""Tests for preparing LoRA benchmark splits."""

import importlib.util
import json
from pathlib import Path

from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "prepare_lora_benchmark",
    Path(__file__).parents[1] / "scripts" / "prepare_lora_benchmark.py",
)
assert _SPEC is not None
prepare_lora_benchmark = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prepare_lora_benchmark)


def make_sample(index: int, reliable: bool, relevance: int = 1) -> RagSample:
    return RagSample(
        id=f"sample_{index:03d}",
        question=f"Вопрос {index}",
        context=f"Контекст {index}",
        answer=f"Ответ {index}",
        faithfulness=1 if reliable else 0,
        relevance=relevance,
        marker="none" if reliable else "reason_hallucinated_fact",
    )


def test_prepare_lora_benchmark_writes_sft_and_raw_test_files(tmp_path: Path) -> None:
    samples = [make_sample(i, reliable=True) for i in range(10)]
    samples += [make_sample(i + 10, reliable=False) for i in range(10)]
    output_dir = tmp_path / "benchmark"

    result = prepare_lora_benchmark.prepare_benchmark(
        samples=samples,
        mode="direct",
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.1,
        seed=123,
        epochs=2,
        batch_size=2,
    )

    assert result.train_total == 14
    assert result.validation_total == 2
    assert result.test_total == 4
    assert result.iters == 14
    assert result.mlx_data_dir == output_dir / "lora_direct"
    assert (output_dir / "lora_direct" / "train.jsonl").exists()
    assert (output_dir / "lora_direct" / "valid.jsonl").exists()
    assert (output_dir / "lora_direct" / "test.jsonl").exists()

    raw_test_rows = [
        json.loads(line)
        for line in (output_dir / "direct_test_samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sft_test_rows = [
        json.loads(line)
        for line in (output_dir / "lora_direct" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert {row["id"] for row in raw_test_rows} == set(result.test_ids)
    assert len(sft_test_rows) == len(raw_test_rows)
    assert set(sft_test_rows[0]) == {"messages"}
    assert [message["role"] for message in sft_test_rows[0]["messages"]] == ["user", "assistant"]


def test_render_lora_command_targets_mode_specific_adapter(tmp_path: Path) -> None:
    result = prepare_lora_benchmark.BenchmarkPreparation(
        mode="marker",
        output_dir=tmp_path,
        mlx_data_dir=tmp_path / "lora_marker",
        raw_test_path=tmp_path / "marker_test_samples.jsonl",
        train_total=8,
        validation_total=1,
        test_total=1,
        iters=16,
        test_ids=["sample_001"],
    )

    command = prepare_lora_benchmark.render_lora_command(
        result,
        config={
            "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "batch_size": 1,
            "grad_accumulation_steps": 8,
            "learning_rate": 0.0001,
            "max_seq_length": 2048,
        },
    )

    assert "--data " + str(tmp_path / "lora_marker") in command
    assert "--iters 16" in command
    assert "--adapter-path results/adapters_marker" in command


def test_prepare_lora_benchmark_truncates_long_dialog_and_context(tmp_path: Path) -> None:
    sample = make_sample(1, reliable=False).model_copy(
        update={
            "question": "q" * 20,
            "context": "c" * 30,
        }
    )

    prepare_lora_benchmark.prepare_benchmark(
        samples=[sample],
        mode="marker",
        output_dir=tmp_path,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        epochs=1,
        batch_size=1,
        max_question_chars=5,
        max_context_chars=7,
    )

    row = json.loads((tmp_path / "lora_marker" / "train.jsonl").read_text(encoding="utf-8"))
    user_message = row["messages"][0]["content"]

    assert "[QUESTION]\nqqqqq\n\n[CONTEXT]\nccccccc" in user_message
    assert "qqqqqq" not in user_message
    assert "cccccccc" not in user_message


def test_balance_training_labels_equalizes_four_judgement_pairs() -> None:
    samples = [make_sample(i, reliable=True, relevance=1) for i in range(5)]
    samples += [make_sample(i + 10, reliable=False, relevance=1) for i in range(3)]
    samples += [make_sample(i + 20, reliable=False, relevance=0) for i in range(2)]
    samples += [make_sample(30, reliable=True, relevance=0)]

    balanced = prepare_lora_benchmark.balance_training_labels(
        samples,
        seed=7,
        target_per_label=3,
    )

    pairs = [(sample.faithfulness, sample.relevance) for sample in balanced]
    assert len(balanced) == 12
    assert {pair: pairs.count(pair) for pair in set(pairs)} == {
        (1, 1): 3,
        (0, 1): 3,
        (0, 0): 3,
        (1, 0): 3,
    }
    assert [sample.id for sample in balanced] == [sample.id for sample in prepare_lora_benchmark.balance_training_labels(samples, seed=7, target_per_label=3)]
