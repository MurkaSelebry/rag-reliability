"""Тесты загрузчика конфига: подстановка ${VAR}, чтение .env."""
import os
import pytest

from src.common.config import load_config, load_dotenv


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_expand_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    p = _write(tmp_path, "c.yaml", 'llm:\n  api_key: "${MY_KEY}"\n  n: 5\n')
    cfg = load_config(p)
    assert cfg["llm"]["api_key"] == "secret123"
    assert cfg["llm"]["n"] == 5  # не-строки не трогаем


def test_missing_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("NO_SUCH_VAR_XYZ", raising=False)
    p = _write(tmp_path, "c.yaml", 'k: "${NO_SUCH_VAR_XYZ}"\n')
    with pytest.raises(KeyError):
        load_config(p)


def test_expand_in_list_and_nested(tmp_path, monkeypatch):
    monkeypatch.setenv("V1", "a")
    p = _write(tmp_path, "c.yaml", 'xs:\n  - "${V1}"\n  - plain\nd:\n  e: "${V1}/b"\n')
    cfg = load_config(p)
    assert cfg["xs"] == ["a", "plain"]
    assert cfg["d"]["e"] == "a/b"


def test_dotenv_setdefault(tmp_path, monkeypatch):
    monkeypatch.delenv("FROM_DOTENV", raising=False)
    monkeypatch.setenv("ALREADY", "keep")
    env = _write(tmp_path, ".env", 'FROM_DOTENV="hello"\nALREADY=lose\n# comment\n')
    load_dotenv(env)
    assert os.environ["FROM_DOTENV"] == "hello"
    assert os.environ["ALREADY"] == "keep"  # не перетирает установленные
