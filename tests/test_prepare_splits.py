"""Tests for materializing deterministic dataset splits."""

import importlib.util
from pathlib import Path

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "prepare_splits", Path(__file__).parents[1] / "scripts" / "prepare_splits.py"
)
assert _SPEC is not None
prepare_splits = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prepare_splits)


def make_sample(index: int) -> RagSample:
    """Build a synthetic sample with alternating reliability labels."""
    reliable = index % 2 == 0
    return RagSample(
        id=f"sample-{index:03d}",
        question=f"Question {index}",
        context=f"Context {index}",
        answer=f"Answer {index}",
        faithfulness=int(reliable),
        relevance=int(reliable),
    )


def test_prepare_splits_writes_three_disjoint_deterministic_files(tmp_path: Path) -> None:
    """The CLI materializes reproducible train, validation, and test JSONL files."""
    data_path = tmp_path / "organizers.jsonl"
    output_dir = tmp_path / "splits"
    save_jsonl([make_sample(index) for index in range(50)], data_path)

    args = ["--data", str(data_path), "--output-dir", str(output_dir), "--seed", "17"]
    prepare_splits.main(args)

    split_paths = [output_dir / f"{name}.jsonl" for name in ("train", "val", "test")]
    first_run = {path.name: path.read_bytes() for path in split_paths}
    split_ids = [{sample.id for sample in load_jsonl(path)} for path in split_paths]

    assert all(path.exists() for path in split_paths)
    assert len(split_ids[0] | split_ids[1] | split_ids[2]) == 50
    assert not (split_ids[0] & split_ids[1])
    assert not (split_ids[0] & split_ids[2])
    assert not (split_ids[1] & split_ids[2])

    prepare_splits.main(args)

    assert {path.name: path.read_bytes() for path in split_paths} == first_run
