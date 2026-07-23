# tests/test_registry.py
from pathlib import Path

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


def test_demo_runner_keys_are_known(tmp_path: Path) -> None:
    allowed = {"dummy", "prompt", "lora", "lettucedetect", "encoder", "m3", "independent"}
    for spec in registry.METHODS.values():
        assert spec.demo_runner is None or spec.demo_runner in allowed
