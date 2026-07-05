"""SEED_INSTRUCTION обязан содержать правило независимости осей (итог этапа −1)."""

from src.m3.prompts import SEED_INSTRUCTION


def test_seed_instruction_has_axis_independence_rule():
    assert "оси независимы" in SEED_INSTRUCTION
    assert "ТОЛЬКО против [CTX]" in SEED_INSTRUCTION
    # правило стоит ДО формата вывода, чтобы влиять на рассуждение
    assert SEED_INSTRUCTION.index("оси независимы") < SEED_INSTRUCTION.index(
        "FAITHFULNESS: PASS или FAIL"
    )
