"""Тесты LLMClient на стабах (без сети)."""
import pytest

from src.common.guard import DataLeakError
from src.common.llm_client import LLMClient
from src.common.schemas import Case

CFG = {"profile": "cloud",
       "llm": {"api_base": "http://x", "api_key": "k", "model": "m",
               "max_ctx_chars": 1000,
               "openrouter_extra_body": {"provider": {"require_parameters": True}}}}


def _client(monkeypatch, fake_request):
    c = LLMClient(CFG)
    monkeypatch.setattr(c, "_request", fake_request)
    return c


def _choice(text):
    return {"text": text, "tokens": []}


# ---- base 4 ----

def test_guard_blocks_real_case(monkeypatch):
    c = _client(monkeypatch, lambda **kw: [_choice("ok")])
    real = Case(id="case_1", query="q", context=["c"], answer="a")
    with pytest.raises(DataLeakError):
        c.chat([{"role": "user", "content": "x"}], case=real)


def test_guard_allows_pseudo(monkeypatch):
    c = _client(monkeypatch, lambda **kw: [_choice("ok")])
    pseudo = Case(id="pseudo_1", query="q", context=["c"], answer="a")
    assert c.chat([{"role": "user", "content": "x"}], case=pseudo)[0]["text"] == "ok"


def test_n_fallback_tops_up(monkeypatch):
    """Провайдер игнорирует n>1 — клиент добирает отдельными запросами (docs/07.3 п.5)."""
    calls = []

    def fake(**kw):
        calls.append(kw["n"])
        return [_choice(f"s{len(calls)}")]  # всегда один сэмпл на запрос

    c = _client(monkeypatch, fake)
    out = c.chat([{"role": "user", "content": "x"}], n=3, temperature=0.8, public_data=True)
    assert [o["text"] for o in out] == ["s1", "s2", "s3"]
    assert len(calls) == 3


def test_model_override():
    c = LLMClient(CFG, model="other-model")
    assert c.model == "other-model"


# ---- A1: fallback on provider error for n>1 ----

def test_n_fallback_on_provider_error(monkeypatch):
    """Провайдер бросает ошибку на n>1 — клиент деградирует до одиночных запросов."""
    calls = []

    def fake(**kw):
        calls.append(kw["n"])
        if kw["n"] > 1:
            raise RuntimeError("400: n>1 not supported")
        return [_choice(f"s{sum(1 for c in calls if c == 1)}")]

    c = _client(monkeypatch, fake)
    out = c.chat([{"role": "user", "content": "x"}], n=3, temperature=0.8, public_data=True)
    assert [o["text"] for o in out] == ["s1", "s2", "s3"]
    assert calls == [3, 1, 1, 1]  # одна неудачная попытка n=3, затем одиночные


# ---- A2: guard-by-default ----

def test_cloud_without_case_blocked(monkeypatch):
    """cloud-профиль: вызов без case и без public_data=True запрещён."""
    c = _client(monkeypatch, lambda **kw: [_choice("ok")])
    with pytest.raises(DataLeakError):
        c.chat([{"role": "user", "content": "x"}])


def test_cloud_public_data_allowed(monkeypatch):
    c = _client(monkeypatch, lambda **kw: [_choice("ok")])
    assert c.chat([{"role": "user", "content": "x"}], public_data=True)[0]["text"] == "ok"


def test_local_without_case_allowed(monkeypatch):
    cfg = {**CFG, "profile": "local"}
    c = LLMClient(cfg)
    monkeypatch.setattr(c, "_request", lambda **kw: [_choice("ok")])
    assert c.chat([{"role": "user", "content": "x"}])[0]["text"] == "ok"
