from typer.testing import CliRunner

from rag_reliability.cli import app

runner = CliRunner()


def test_global_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "benchmark", "eval", "serve", "list-methods"):
        assert command in result.output


def test_benchmark_help_documents_methods_option() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--methods" in result.output


def test_list_methods_prints_all_methods() -> None:
    result = runner.invoke(app, ["list-methods"])
    assert result.exit_code == 0
    assert "independent" in result.output
    assert "m3_openai_judge" in result.output


def test_run_rejects_unknown_method() -> None:
    result = runner.invoke(app, ["run", "--method", "nope", "--data", "data/dummy.jsonl"])
    assert result.exit_code != 0


def test_run_rejects_multiple_methods() -> None:
    result = runner.invoke(app, ["run", "--method", "all", "--data", "data/dummy.jsonl"])
    assert result.exit_code != 0
