# tests/test_registry.py
import re
from dataclasses import replace
from pathlib import Path

import pytest

from rag_reliability.methods import registry


def _ctx(tmp_path: Path) -> registry.CommandContext:
    run_dir = tmp_path / "m"
    return registry.CommandContext(
        data=Path("data/dummy.jsonl"),
        run_dir=run_dir,
        predictions_path=run_dir / "predictions.jsonl",
        python="python",
    )


def test_registry_has_fifteen_methods() -> None:
    assert len(registry.METHODS) == 15
    assert set(registry.all_method_names()) == set(registry.METHODS)


def test_resolve_all_returns_every_method() -> None:
    assert registry.resolve_names("all") == list(registry.all_method_names())


def test_resolve_unknown_raises_with_available_set() -> None:
    try:
        registry.resolve_names("dummy_direct,nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_dummy_marker_build_command_matches_legacy_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    argv = registry.get("dummy_marker").build_command(ctx)
    assert argv == [
        "python",
        "scripts/run_prompt_baseline.py",
        "--data",
        "data/dummy.jsonl",
        "--output",
        str(ctx.predictions_path),
        "--mode",
        "marker",
        "--backend",
        "dummy",
        "--dummy-strategy",
        "keyword",
    ]


def test_independent_build_command(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    argv = registry.get("independent").build_command(ctx)
    assert argv[0:2] == ["python", "scripts/run_independent.py"]
    assert "--output" in argv
    assert str(ctx.predictions_path) in argv


def test_every_spec_builds_a_nonempty_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    for name in registry.all_method_names():
        argv = registry.get(name).build_command(ctx)
        assert argv and argv[0] == "python"


def test_m3_gepa_default_prompt_is_committed_path(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    argv = registry.get("m3_gepa").build_command(ctx)

    assert "configs/m3_gepa_prompt.txt" in argv
    assert "configs/m3_gepa_prompt.txt" in registry.get("m3_gepa").requires


def test_named_m3_mode_with_openai_judge_backend_gets_api_args(tmp_path: Path) -> None:
    ctx = replace(_ctx(tmp_path), m3_backend="openai_judge")

    argv = registry.get("m3_zero_shot").build_command(ctx)

    assert "--api-base" in argv
    assert "--cache-dir" in argv
    assert "--concurrency" in argv


def test_m6_build_command_uses_single_command_pipeline(tmp_path: Path) -> None:
    ctx = replace(
        _ctx(tmp_path),
        model="remote-model",
        m6_backend="openai",
        m6_samples_dir="cache/m6",
        m6_n_samples=7,
        m6_api_base="https://example.test/v1",
    )

    argv = registry.get("m6_selfcheck").build_command(ctx)

    assert argv[0:2] == ["python", "scripts/run_m6_pipeline.py"]
    assert argv[argv.index("--samples-dir") + 1] == "cache/m6"
    assert argv[argv.index("--backend") + 1] == "openai"
    assert argv[argv.index("--n-samples") + 1] == "7"
    assert argv[argv.index("--model") + 1] == "remote-model"
    assert argv[argv.index("--api-base") + 1] == "https://example.test/v1"


def test_demo_runner_keys_are_known(tmp_path: Path) -> None:
    allowed = {"dummy", "prompt", "lora", "lettucedetect", "encoder", "m3", "independent"}
    for spec in registry.METHODS.values():
        assert spec.demo_runner is None or spec.demo_runner in allowed


# --------------------------------------------------------------------------- #
# Контракт scores: инварианты, а не захардкоженные примеры.
# --------------------------------------------------------------------------- #


def test_corpus_wide_methods_declare_score_keys() -> None:
    """Метод, участвующий в корпус-wide протоколе, обязан объявить свои сигналы.

    Дамми — единственное исключение: он существует ради смоука пайплайна.
    """
    for spec in registry.METHODS.values():
        if not spec.corpus_wide or spec.name in registry.DUMMY_METHODS:
            continue
        assert spec.score_keys, f"{spec.name} is corpus_wide but declares no score_keys"


def test_methods_without_score_keys_are_parked_for_wave_three() -> None:
    """Пустые score_keys допустимы только у дамми и у явно отложенных методов."""
    for spec in registry.METHODS.values():
        if spec.score_keys or spec.name in registry.DUMMY_METHODS:
            continue
        assert not spec.corpus_wide, f"{spec.name} has no score_keys but claims corpus_wide"
        assert spec.name in registry.WAVE3_OWNER, f"{spec.name} has no wave-3 owner recorded"


def test_score_keys_use_registered_method_prefixes() -> None:
    for spec in registry.METHODS.values():
        for key in spec.score_keys:
            assert key.startswith(registry.SCORE_PREFIXES), f"{spec.name}: bad prefix in {key!r}"
            assert key.count(".") == 1, f"{spec.name}: {key!r} must be '<method>.<signal>'"


def test_default_score_expr_uses_only_declared_keys() -> None:
    """Выражение по умолчанию не может ссылаться на сигнал, которого метод не даёт."""
    identifier = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)+")
    for spec in registry.METHODS.values():
        if spec.default_score_expr is None:
            continue
        assert spec.score_keys, f"{spec.name} has default_score_expr but no score_keys"
        referenced = set(identifier.findall(spec.default_score_expr))
        unknown = referenced - set(spec.score_keys)
        assert not unknown, f"{spec.name}: default_score_expr references undeclared {unknown}"


def test_every_corpus_wide_method_has_a_scorer() -> None:
    for spec in registry.METHODS.values():
        assert (spec.build_scorer is not None) == spec.corpus_wide, (
            f"{spec.name}: corpus_wide={spec.corpus_wide} but "
            f"build_scorer={'set' if spec.build_scorer else 'None'}"
        )


def test_build_scorer_refuses_parked_methods_with_wave_three_pointer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C2"):
        registry.build_scorer("encoder", _ctx(tmp_path))


def test_binary_only_methods_are_excluded_from_corpus_wide() -> None:
    for name in registry.BINARY_ONLY_METHODS:
        assert not registry.get(name).corpus_wide


def test_list_methods_prints_the_new_contract_fields() -> None:
    """rag-judge остаётся окном в реестр: новые поля должны быть видны оператору."""
    from typer.testing import CliRunner

    from rag_reliability.cli import app

    result = CliRunner().invoke(app, ["list-methods"])

    assert result.exit_code == 0
    assert "corpus-wide" in result.output
    assert "split-only" in result.output
    assert "m3.p_faith" in result.output
    assert "ind.faith_score" in result.output


def test_m3_mode_and_backend_shared_by_command_and_scorer(tmp_path: Path) -> None:
    """Одна точка разбора имени: subprocess и score.py не должны разъехаться."""
    ctx = replace(_ctx(tmp_path), m3_backend="mlx")
    assert registry.m3_mode_and_backend("m3_few_shot", ctx) == ("few_shot", "mlx")
    assert registry.m3_mode_and_backend("m3_openai_judge", ctx) == ("zero_shot", "openai_judge")

    argv = registry.get("m3_few_shot").build_command(ctx)
    assert argv[argv.index("--mode") + 1] == "few_shot"
