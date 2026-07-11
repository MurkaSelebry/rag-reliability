"""Метод 3: LLM-as-judge с вердиктами из logprobs и промпт-оптимизацией GEPA/DSPy."""

from rag_reliability.methods.m3.prompts import (
    SEED_INSTRUCTION,
    build_few_shot_system,
    build_user_prompt,
)

__all__ = ["SEED_INSTRUCTION", "build_few_shot_system", "build_user_prompt"]
