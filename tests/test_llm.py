"""Tests for the multi-backend LLM client (app.llm).

No network is used: httpx.AsyncClient is replaced with fakes, and the
persisted-state file is redirected to a temp location.
"""

from __future__ import annotations

import json

import pytest

from app import llm


@pytest.fixture(autouse=True)
def llm_state_tmp(monkeypatch, tmp_path):
    """Point the state file at a temp path and clear any selection."""
    state_file = tmp_path / "llm_state.json"
    monkeypatch.setattr(llm, "_DEFAULT_STATE_FILE", state_file)
    return state_file


def _fake_async_client(monkeypatch, response_data=None, status_code=200, raise_error=None):
    """Replace httpx.AsyncClient with a fake returning ``response_data``."""
    http_status = status_code

    class FakeResponse:
        status_code = http_status

        def json(self):
            return response_data if response_data is not None else {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            if raise_error:
                raise raise_error
            return FakeResponse()

        async def get(self, url):
            if raise_error:
                raise raise_error
            return FakeResponse()

    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    return FakeClient


class TestConfiguredBackends:
    def test_legacy_fallback_requires_model(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", "")
        monkeypatch.setattr(llm.settings, "ollama_model", "my-model")
        monkeypatch.setattr(llm.settings, "ollama_url", "http://x:11434")
        backends = llm._configured_backends()
        assert backends == [{
            "name": "ollama", "type": "ollama",
            "url": "http://x:11434", "model": "my-model",
        }]

    def test_legacy_fallback_empty_model_dropped(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", "")
        monkeypatch.setattr(llm.settings, "ollama_model", "")
        assert llm._configured_backends() == []

    def test_json_backends(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
            {"name": "l1", "type": "openai", "url": "http://a:8080", "model": "m1"},
            {"name": "l2", "type": "ollama", "url": "http://b:11434", "model": "m2"},
        ]))
        backends = llm._configured_backends()
        assert len(backends) == 2
        assert backends[0]["type"] == "openai"
        assert backends[1]["type"] == "ollama"

    def test_invalid_json_falls_back_to_legacy(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", "{not json")
        monkeypatch.setattr(llm.settings, "ollama_model", "legacy-model")
        assert llm._configured_backends()[0]["model"] == "legacy-model"


class TestSelection:
    def test_first_backend_is_default(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
            {"name": "a", "type": "openai", "url": "http://a:1", "model": "m1"},
            {"name": "b", "type": "openai", "url": "http://b:1", "model": "m2"},
        ]))

        async def run():
            return await llm.current_backend()

        assert asyncio_run(run())["name"] == "a"

    def test_select_persists_and_switches(self, monkeypatch, llm_state_tmp):
        monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
            {"name": "a", "type": "openai", "url": "http://a:1", "model": "m1"},
            {"name": "b", "type": "ollama", "url": "http://b:1", "model": "m2"},
        ]))

        async def run():
            selected = await llm.select_backend("b", "m2")
            assert selected["name"] == "b"
            cur = await llm.current_backend()
            assert cur["name"] == "b"
            state = json.loads(llm_state_tmp.read_text())
            assert state["selected"] == "b"
            assert state["models"]["b"] == "m2"
            # Fresh module reload (simulated by reading state anew) keeps it.
            return await llm.current_backend()

        assert asyncio_run(run())["name"] == "b"

    def test_model_override_applied(self, monkeypatch, llm_state_tmp):
        monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
            {"name": "a", "type": "openai", "url": "http://a:1", "model": "m1"},
        ]))

        async def run():
            await llm.select_backend("a", "other-model")
            cur = await llm.current_backend()
            return cur["model"]

        assert asyncio_run(run()) == "other-model"

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", "[]")

        async def run():
            await llm.select_backend("nope")

        with pytest.raises(ValueError):
            asyncio_run(run())


class TestChat:
    def _chat(self, monkeypatch, llm_state_tmp, response_data, **kw):
        _fake_async_client(monkeypatch, response_data, **kw)
        monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
            {"name": "a", "type": "openai", "url": "http://a:1", "model": "m1"},
        ]))

        async def run():
            return await llm.chat([{"role": "user", "content": "hi"}])

        return asyncio_run(run())

    def test_openai_response(self, monkeypatch, llm_state_tmp):
        out = self._chat(monkeypatch, llm_state_tmp, {
            "choices": [{"message": {"content": "hi there", "reasoning_content": "thinking"}}],
            "model": "m1",
        })
        assert out["text"] == "hi there"
        assert out["model"] == "m1"

    def test_ollama_response(self, monkeypatch, llm_state_tmp):
        monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
            {"name": "o", "type": "ollama", "url": "http://o:1", "model": "m9"},
        ]))
        _fake_async_client(monkeypatch, {"message": {"content": "hello", "reasoning": "r"}})

        async def run():
            return await llm.chat([{"role": "user", "content": "hi"}])

        out = asyncio_run(run())
        assert out["text"] == "hello"
        assert out["model"] == "m9"

    def test_empty_config_returns_message(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", "")
        monkeypatch.setattr(llm.settings, "ollama_model", "")

        async def run():
            return await llm.chat([{"role": "user", "content": "hi"}])

        out = asyncio_run(run())
        assert "No LLM backend" in out["text"]

    def test_http_error_raises(self, monkeypatch, llm_state_tmp):
        with pytest.raises(RuntimeError):
            self._chat(monkeypatch, llm_state_tmp, {}, status_code=500)


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
