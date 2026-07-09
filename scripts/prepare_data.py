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
import ast
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from rag_reliability.dataset import save_jsonl
from rag_reliability.schema import RagSample

ORGANIZER_CHUNK_COLUMNS = tuple(f"chunk_{i}" for i in range(1, 9))
ORGANIZER_REQUIRED_COLUMNS = {
    "full_dialog",
    "answer",
    "binary_relevancy",
    "binary_faithfulness",
}


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


def detect_format(path: str | Path, fmt: str) -> str:
    path = Path(path)
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "zip":
        return "csv"
    if suffix in ("csv", "json", "jsonl"):
        return suffix
    raise ValueError(f"Cannot auto-detect format for {path}; pass --format explicitly")


def remap(record: dict, column_map: dict[str, str]) -> dict:
    return {column_map.get(key, key): value for key, value in record.items()}


def parse_bool_label(value: object, *, field: str, row_number: int) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise ValueError(f"Invalid boolean label {field}={value!r} at row {row_number}")


def parse_marker_list(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str) and item]
    return [text]


def organizer_context(record: dict[str, object]) -> str:
    chunks: list[str] = []
    for index, column in enumerate(ORGANIZER_CHUNK_COLUMNS, start=1):
        chunk = str(record.get(column, "") or "").strip()
        if chunk:
            chunks.append(f"[CHUNK {index}]\n{chunk}")
    return "\n\n".join(chunks)


def convert_organizer_record(record: dict[str, object], *, row_number: int) -> RagSample:
    relevance = parse_bool_label(
        record.get("binary_relevancy"), field="binary_relevancy", row_number=row_number
    )
    faithfulness = parse_bool_label(
        record.get("binary_faithfulness"), field="binary_faithfulness", row_number=row_number
    )
    markers = parse_marker_list(record.get("markers"))
    if faithfulness == 1 and relevance == 1:
        marker = "none"
    else:
        marker = markers[0] if markers else "unknown"
    return RagSample(
        id=f"organizer_{row_number:06d}",
        question=str(record.get("full_dialog", "") or "").strip(),
        context=organizer_context(record),
        answer=str(record.get("answer", "") or "").strip(),
        faithfulness=faithfulness,
        relevance=relevance,
        marker=marker,
    )


def is_organizer_record(record: dict[str, object]) -> bool:
    return ORGANIZER_REQUIRED_COLUMNS.issubset(record)


def load_csv_records(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside {path}")
            with archive.open(csv_names[0]) as raw:
                text = (line.decode("utf-8-sig") for line in raw)
                return list(csv.DictReader(text))
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    fmt = detect_format(input_path, args.format)
    column_map: dict[str, str] = json.loads(args.column_map) if args.column_map else {}

    if fmt == "csv":
        raw_records = load_csv_records(input_path)
        if raw_records and is_organizer_record(raw_records[0]) and not column_map:
            samples = [
                convert_organizer_record(record, row_number=index)
                for index, record in enumerate(raw_records, start=1)
            ]
        else:
            samples = [
                RagSample.model_validate(remap(record, column_map)) for record in raw_records
            ]
        save_jsonl(samples, args.output)
        print(f"Wrote {len(samples)} samples to {args.output}")
        return
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
