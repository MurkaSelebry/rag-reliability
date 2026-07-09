"""Tests for the optional OpenAI-compatible Method 3 client."""

from rag_reliability.methods.m3.openai_client import CachedChatClient


def test_cached_chat_client_reuses_cached_response(tmp_path) -> None:
    calls = []

    def transport(messages, *, model, api_base, api_key, max_tokens, temperature):  # noqa: ARG001
        calls.append(messages)
        return "FAITHFULNESS: PASS\nRELEVANCE: FAIL"

    client = CachedChatClient(
        model="judge-model",
        api_base="https://example.test/v1",
        api_key="secret",
        cache_dir=tmp_path,
        transport=transport,
    )
    messages = [{"role": "user", "content": "judge this"}]

    first = client.chat(messages, max_tokens=32)
    second = client.chat(messages, max_tokens=32)

    assert first == "FAITHFULNESS: PASS\nRELEVANCE: FAIL"
    assert second == first
    assert len(calls) == 1


def test_cached_chat_client_cache_key_includes_model(tmp_path) -> None:
    calls = []

    def transport(messages, *, model, api_base, api_key, max_tokens, temperature):  # noqa: ARG001
        calls.append(model)
        return f"model={model}"

    first_client = CachedChatClient(
        model="model-a",
        api_base="https://example.test/v1",
        api_key="secret",
        cache_dir=tmp_path,
        transport=transport,
    )
    second_client = CachedChatClient(
        model="model-b",
        api_base="https://example.test/v1",
        api_key="secret",
        cache_dir=tmp_path,
        transport=transport,
    )
    messages = [{"role": "user", "content": "same prompt"}]

    assert first_client.chat(messages) == "model=model-a"
    assert second_client.chat(messages) == "model=model-b"
    assert calls == ["model-a", "model-b"]
