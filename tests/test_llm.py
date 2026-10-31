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

        async def post(self, url, json=None, headers=None):
            if raise_error:
                raise raise_error
            return FakeResponse()

        async def get(self, url, headers=None):
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


class TestChatTemplateKwargs:
    """The vLLM-native ``chat_template_kwargs`` are OPT-IN per backend.

    Regression pin for the 2026-10-31 live smoke: the LAN LiteLLM proxy
    (routing deepseek-v4.1-flash-mistral) answered HTTP 422
    ``extra_forbidden ... body.chat_template_kwargs`` for the
    unconditional field, so every sell-layer consult fail-safed; the clean
    body (no field) answered 200. Only a backend that DECLARES the kwargs
    sends them.
    """

    def _capture_body(self, monkeypatch, backends_json, effort="",
                      temperature=None):
        captured: dict = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}],
                        "model": "m1"}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                captured["body"] = json
                return FakeResponse()

            async def get(self, url, headers=None):
                return FakeResponse()

        monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(llm.settings, "llm_backends", backends_json)
        monkeypatch.setattr(llm.settings, "llm_reasoning_effort", effort)

        async def run():
            return await llm.chat([{"role": "user", "content": "hi"}],
                                  temperature=temperature)

        asyncio_run(run())
        return captured["body"]

    def test_temperature_sent_only_when_declared(self, monkeypatch, llm_state_tmp):
        """The frozen-constant fix: the wire must carry the temperature every
        consult record declares — None (the default) leaves the server's
        default untouched."""
        b1 = self._capture_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]))
        assert "temperature" not in b1
        b2 = self._capture_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]), temperature=0.0)
        assert b2["temperature"] == 0.0

    def test_undeclared_backend_sends_the_clean_body(self, monkeypatch, llm_state_tmp):
        body = self._capture_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]))
        assert "chat_template_kwargs" not in body  # the LiteLLM 422 regression

    def test_declared_backend_carries_the_kwargs(self, monkeypatch, llm_state_tmp):
        body = self._capture_body(monkeypatch, json.dumps([
            {"name": "vllm", "type": "openai", "url": "http://b:1",
             "model": "m", "chat_template_kwargs": {"enable_thinking": True}},
        ]))
        assert body["chat_template_kwargs"]["enable_thinking"] is True

    def test_effort_budget_rides_only_on_a_declared_backend(self, monkeypatch, llm_state_tmp):
        body = self._capture_body(monkeypatch, json.dumps([
            {"name": "vllm", "type": "openai", "url": "http://b:1",
             "model": "m", "chat_template_kwargs": {"enable_thinking": True}},
        ]), effort="high")
        assert body["chat_template_kwargs"]["thinking_budget"] == 8192

    def test_effort_on_a_strict_backend_sends_reasoning_effort(self, monkeypatch, llm_state_tmp):
        """The strict-schema path (probe-verified 2026-10-31): the LiteLLM
        proxy answers ``reasoning_effort`` with HTTP 200 + reasoning_content,
        while the vLLM-only field 422s — so the effort travels as the
        OpenAI-style field, never chat_template_kwargs."""
        body = self._capture_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]), effort="high")
        assert body["reasoning_effort"] == "high"
        assert "chat_template_kwargs" not in body

    def test_xhigh_rides_verbatim_on_the_strict_path(self, monkeypatch, llm_state_tmp):
        """Probe-verified 2026-10-31: the proxy's Mistral route SUPPORTS
        xhigh (supported values ['none', 'low', 'high', 'xhigh', 'max']) —
        so it rides verbatim; the app does not curate the server's
        vocabulary (an earlier pin kept it vLLM-only, contradicting the
        probe)."""
        body = self._capture_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]), effort="xhigh")
        assert body["reasoning_effort"] == "xhigh"
        assert "chat_template_kwargs" not in body

    def test_max_rides_verbatim_outside_the_vllm_table(self, monkeypatch, llm_state_tmp):
        """``max`` is outside the vLLM thinking-budget table yet supported
        by the Mistral route — it rides verbatim, proving no curation;
        the SERVER validates the value (the route rejects "medium" with a
        400 listing its supported values, by design the app's fail-safe
        then records it)."""
        body = self._capture_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]), effort="max")
        assert body["reasoning_effort"] == "max"
        assert "chat_template_kwargs" not in body

    def test_no_effort_sends_no_reasoning_fields(self, monkeypatch, llm_state_tmp):
        body = self._capture_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]), effort="")
        assert "reasoning_effort" not in body
        assert "chat_template_kwargs" not in body

    # --- the streaming variant (chat_stream — the live chat UI's wire) ---

    def _capture_stream_body(self, monkeypatch, backends_json, effort=""):
        """Drive llm.chat_stream against a fake SSE stream; return the
        request body it sent (and prove the stream parsed to a done)."""
        captured: dict = {}

        class FakeStreamResp:
            status_code = 200

            async def aiter_lines(self):
                yield ('data: {"model": "m", "choices": '
                       '[{"delta": {"content": "ok"}}]}')
                yield "data: [DONE]"

            async def aread(self):
                return b""

        class FakeStreamCtx:
            async def __aenter__(self):
                return FakeStreamResp()

            async def __aexit__(self, *a):
                return False

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, method, url, json=None, headers=None):
                captured["body"] = json
                return FakeStreamCtx()

        monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(llm.settings, "llm_backends", backends_json)
        monkeypatch.setattr(llm.settings, "llm_reasoning_effort", effort)

        async def run():
            events = []
            async for evt in llm.chat_stream(
                    [{"role": "user", "content": "hi"}]):
                events.append(evt)
            return events

        events = asyncio_run(run())
        assert events[-1]["type"] == "done"
        assert events[-1]["text"] == "ok"
        return captured["body"]

    def test_stream_undeclared_backend_sends_the_clean_body(self, monkeypatch, llm_state_tmp):
        """The streaming 422 regression: chat_stream (the live chat UI's
        wire — the chat endpoint + the sim chat) must send the same clean
        body as the non-streaming path on an undeclared backend."""
        body = self._capture_stream_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]))
        assert "chat_template_kwargs" not in body
        assert "reasoning_effort" not in body

    def test_stream_declared_backend_carries_the_kwargs(self, monkeypatch, llm_state_tmp):
        body = self._capture_stream_body(monkeypatch, json.dumps([
            {"name": "vllm", "type": "openai", "url": "http://b:1",
             "model": "m", "chat_template_kwargs": {"enable_thinking": True}},
        ]))
        assert body["chat_template_kwargs"]["enable_thinking"] is True

    def test_stream_effort_rides_verbatim_on_the_strict_path(self, monkeypatch, llm_state_tmp):
        body = self._capture_stream_body(monkeypatch, json.dumps([
            {"name": "proxy", "type": "openai", "url": "http://a:1",
             "model": "m"},
        ]), effort="max")
        assert body["reasoning_effort"] == "max"
        assert "chat_template_kwargs" not in body

    def test_stream_effort_budget_rides_only_on_a_declared_backend(self, monkeypatch, llm_state_tmp):
        body = self._capture_stream_body(monkeypatch, json.dumps([
            {"name": "vllm", "type": "openai", "url": "http://b:1",
             "model": "m", "chat_template_kwargs": {"enable_thinking": True}},
        ]), effort="high")
        assert body["chat_template_kwargs"]["thinking_budget"] == 8192
        assert "reasoning_effort" not in body


class TestBackendEntryRedaction:
    """The invalid-entry log must never carry an inline api_key — the app's
    own never-log-the-key invariant (a malformed entry is exactly the case
    where an operator pastes a config with the key inline)."""

    def test_invalid_entry_log_omits_the_key(self, monkeypatch, llm_state_tmp, caplog):
        import logging
        monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
            {"name": "bad", "type": "nope", "url": "http://a:1", "model": "m",
             "api_key": "sk-NEVER-LOG-THIS-xyz"},
        ]))
        with caplog.at_level(logging.WARNING):
            assert llm._configured_backends() == []
        assert "Skipping invalid LLM backend entry" in caplog.text
        assert "NEVER-LOG-THIS-xyz" not in caplog.text


# ---------------------------------------------------------------------------
# chat_stream — streaming SSE/NDJSON parser
# ---------------------------------------------------------------------------

class _FakeStreamResp:
    """Mimics the relevant subset of httpx.Response for streaming tests."""
    def __init__(self, lines=None, status_code=200, body=b""):
        self._lines = lines or []
        self.status_code = status_code
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


class _FakeStreamClient:
    """``httpx.AsyncClient`` substitute that returns a _FakeStreamResp from
    ``stream()``. Configure the response lines via ``configure()`` before
    running the test. Each ``AsyncClient(...)`` instantiation returns the same
    singleton-like instance so httpx's async-context-manager pattern works."""
    _configured_lines: list = []
    _configured_status: int = 200
    _configured_body: bytes = b""

    def __init__(self, *a, **k):
        # httpx.AsyncClient(timeout=...) calls us with kwargs we ignore.
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method, url, json=None, headers=None):
        return _FakeStreamResp(
            lines=self._configured_lines,
            status_code=self._configured_status,
            body=self._configured_body,
        )

    @classmethod
    def configure(cls, lines=None, status_code=200, body=b""):
        cls._configured_lines = lines or []
        cls._configured_status = status_code
        cls._configured_body = body


def _configure_streaming_backend(monkeypatch, backend_type="openai", model="m1"):
    """Wire up a single backend and a fake streaming httpx client."""
    monkeypatch.setattr(llm.settings, "llm_backends", json.dumps([
        {"name": "a", "type": backend_type, "url": "http://x:1", "model": model},
    ]))
    monkeypatch.setattr(llm.httpx, "AsyncClient", _FakeStreamClient)


class TestChatStream:
    def test_openai_sse_streaming(self, monkeypatch, llm_state_tmp):
        lines = [
            'data: ' + json.dumps({'choices': [{'delta': {'content': 'Hello'}}], 'model': 'm1'}),
            'data: ' + json.dumps({'choices': [{'delta': {'content': ' world'}}], 'model': 'm1'}),
            'data: ' + json.dumps({'choices': [{'delta': {'content': '!'}}], 'model': 'm1'}),
            'data: [DONE]',
        ]
        _FakeStreamClient.configure(lines=lines)
        _configure_streaming_backend(monkeypatch, "openai", "m1")

        async def run():
            out = []
            async for evt in llm.chat_stream([{"role": "user", "content": "hi"}]):
                out.append(evt)
            return out

        events = asyncio_run(run())
        deltas = [e for e in events if e["type"] == "delta"]
        done = [e for e in events if e["type"] == "done"]
        assert [d["text"] for d in deltas] == ["Hello", " world", "!"]
        assert len(done) == 1
        assert done[0]["text"] == "Hello world!"
        assert done[0]["model"] == "m1"
        assert done[0]["backend"] == "a"

    def test_ollama_ndjson_streaming(self, monkeypatch, llm_state_tmp):
        lines = [
            json.dumps({'message': {'content': 'Oll'}, 'model': 'm9'}),
            json.dumps({'message': {'content': 'ama'}, 'model': 'm9'}),
            json.dumps({'message': {'content': ' rules'}, 'model': 'm9', 'done': True}),
        ]
        _FakeStreamClient.configure(lines=lines)
        _configure_streaming_backend(monkeypatch, "ollama", "m9")

        async def run():
            out = []
            async for evt in llm.chat_stream([{"role": "user", "content": "hi"}]):
                out.append(evt)
            return out

        events = asyncio_run(run())
        deltas = [e for e in events if e["type"] == "delta"]
        done = [e for e in events if e["type"] == "done"]
        assert [d["text"] for d in deltas] == ["Oll", "ama", " rules"]
        assert done[0]["text"] == "Ollama rules"
        assert done[0]["model"] == "m9"

    def test_http_error_emits_error_event(self, monkeypatch, llm_state_tmp):
        _FakeStreamClient.configure(lines=[], status_code=500, body=b"oops")
        _configure_streaming_backend(monkeypatch, "openai", "m1")

        async def run():
            out = []
            async for evt in llm.chat_stream([{"role": "user", "content": "hi"}]):
                out.append(evt)
            return out

        events = asyncio_run(run())
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "HTTP 500" in events[0]["text"]
        assert "oops" in events[0]["text"]

    def test_no_backend_emits_error(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "llm_backends", "")
        monkeypatch.setattr(llm.settings, "ollama_model", "")

        async def run():
            out = []
            async for evt in llm.chat_stream([{"role": "user", "content": "hi"}]):
                out.append(evt)
            return out

        events = asyncio_run(run())
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "No LLM backend" in events[0]["text"]


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
