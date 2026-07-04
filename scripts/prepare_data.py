#!/usr/bin/env python
"""Convert a raw dataset into the project's RagSample jsonl schema.

Currently a skeleton: jsonl passthrough with optional column renaming.
CSV support is a TODO until the real dataset format is known.

Example:
    python scripts/prepare_data.py \
        --input raw_dataset.jsonl \
        --output data/processed.jsonl \
        --column-map '{"query": "question", "passage": "context"}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_reliability.dataset import save_jsonl
from rag_reliability.schema import RagSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw dataset file")
    parser.add_argument("--output", default="data/processed.jsonl", help="Output jsonl")
    parser.add_argument("--format", choices=["auto", "csv", "json", "jsonl"], default="auto")
    parser.add_argument(
        "--column-map",
        default=None,
        help='JSON string mapping source field -> target field, e.g. \'{"query": "question"}\'',
    )
    return parser.parse_args()


def detect_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("csv", "json", "jsonl"):
        return suffix
    raise ValueError(f"Cannot auto-detect format for {path}; pass --format explicitly")


def remap(record: dict, column_map: dict[str, str]) -> dict:
    return {column_map.get(key, key): value for key, value in record.items()}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    fmt = detect_format(input_path, args.format)
    column_map: dict[str, str] = json.loads(args.column_map) if args.column_map else {}

    if fmt == "csv":
        # TODO: implement once the real dataset arrives (csv.DictReader + remap
        # + type coercion for faithfulness/relevance).
        raise NotImplementedError("CSV support is not implemented yet; convert to jsonl first")
    if fmt == "json":
        raw_records = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(raw_records, list):
            raise ValueError("JSON input must be a list of records")
    else:  # jsonl
        raw_records = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    samples = [RagSample.model_validate(remap(record, column_map)) for record in raw_records]
    save_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
