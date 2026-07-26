"""Энкодер-классификатор надёжности: построение входа, OOF-обучение, инференс.

Пакет существует потому, что логика жила в ``scripts/train_encoder_baseline.py``
и была непроверяемой: собственный сплит, отсутствие контроля схлопывания и
формат входа, отличный от того, на котором получены опубликованные числа.
Здесь всё это разнесено на тестируемые части, а скрипт остаётся обёрткой.
"""

from __future__ import annotations

from rag_reliability.methods.encoder.data import (
    EncodedInput,
    EncoderExample,
    EncoderSegments,
    build_encoder_text,
    build_segments,
    encode,
    make_examples,
    parse_chunks,
    split_dialog,
)

__all__ = [
    "EncodedInput",
    "EncoderExample",
    "EncoderSegments",
    "build_encoder_text",
    "build_segments",
    "encode",
    "make_examples",
    "parse_chunks",
    "split_dialog",
]
