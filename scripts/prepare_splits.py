"""Materialize deterministic train, validation, and test dataset splits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rag_reliability.dataset import load_jsonl, save_jsonl, split_samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for split materialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/organizers.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load labeled samples and write reproducible stratified splits."""
    args = parse_args(argv)
    train, val, test = split_samples(load_jsonl(args.data), seed=args.seed)
    save_jsonl(train, args.output_dir / "train.jsonl")
    save_jsonl(val, args.output_dir / "val.jsonl")
    save_jsonl(test, args.output_dir / "test.jsonl")


if __name__ == "__main__":
    main()
