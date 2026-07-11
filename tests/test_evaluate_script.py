"""Tests for evaluate.py helper behavior."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_script",
    Path(__file__).parents[1] / "scripts" / "evaluate.py",
)
assert _SPEC is not None
evaluate_script = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["evaluate_script"] = evaluate_script
_SPEC.loader.exec_module(evaluate_script)


def test_apply_limit_keeps_first_n_samples() -> None:
    assert evaluate_script.apply_limit([1, 2, 3], None) == [1, 2, 3]
    assert evaluate_script.apply_limit([1, 2, 3], 2) == [1, 2]
