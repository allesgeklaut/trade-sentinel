"""Multi-backend LLM client for Trade Sentinel.

Supports two wire protocols, both discovered and selectable at runtime:

- ``ollama`` — Ollama's native ``/api/chat`` (also used for model listing via
  ``/api/tags``).
- ``openai``  — OpenAI-compatible ``/v1/chat/completions`` (used by
  llama.cpp's ``llama-server``, vLLM, LiteLLM, OpenAI itself, ...).

Backends are configured through the ``LLM_BACKENDS`` JSON env var, e.g.::

    LLM_BACKENDS=[
      {"name":"local llama","type":"openai","url":"http://your-server-ip:4000","model":"qwen3.8-27b","api_key":"sk-..."},
      {"name":"ollama","type":"ollama","url":"http://host.docker.internal:11434","model":"deepseek-v4-flash:0731-cloud"}
    ]

A backend may declare ``chat_template_kwargs`` (a dict) for vLLM-native
chat-template overrides (e.g. ``{"enable_thinking": true}`` plus the
``LLM_REASONING_EFFORT`` thinking budget). The field is OPT-IN: strict
OpenAI-schema servers reject it (HTTP 422 ``extra_forbidden`` on
``body.chat_template_kwargs``, probe-verified 2026-10-31 through the LAN
LiteLLM proxy routing to Mistral), so it is only sent when the backend
declares it.

The actively selected backend is persisted to a small JSON file so the choice
survives restarts.  For backwards compatibility the legacy ``OLLAMA_URL`` /
``OLLAMA_MODEL`` settings are honoured when ``LLM_BACKENDS`` is not set.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger("trade_sentinel.llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_STATE_FILE = Path(settings.llm_state_file)

# API key sources, in priority order: the backend's own ``api_key`` (set in
# LLM_BACKENDS), then a dedicated key file (LITELLM_API_KEY_FILE — single
# source of truth, e.g. /opt/secrets/litellm.key), then the legacy
# LITELLM_API_KEY env var.  Preferring the file means the key is not
# duplicated into per-service env files.
_LLM_API_KEY_FILE = os.environ.get("LITELLM_API_KEY_FILE", "")


def _api_key_from_file() -> str:
    """Read the bare API key from ``LITELLM_API_KEY_FILE`` (or ``""``)."""
    path = _LLM_API_KEY_FILE.strip()
    if not path:
        return ""
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        logger.warning("Could not read LLM API key file %s: %s", path, e)
        return ""


def _auth_headers(backend: dict[str, Any]) -> dict[str, str]:
    """Authorization headers for an OpenAI-compatible backend.

    Uses the backend's own ``api_key`` if set, otherwise the key from
    ``LITELLM_API_KEY_FILE``, otherwise the legacy ``LITELLM_API_KEY`` env var.
    """
    key = (
        backend.get("api_key")
        or _api_key_from_file()
        or os.environ.get("LITELLM_API_KEY")
        or ""
    ).strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def _configured_backends() -> list[dict[str, Any]]:
    """Backends from LLM_BACKENDS, or the legacy OLLAMA_* settings."""
    legacy = [
        {
            "name": "ollama",
            "type": "ollama",
            "url": settings.ollama_url,
            "model": settings.ollama_model,
        }
    ]
    raw = (settings.llm_backends or "").strip()
    if not raw:
        return [b for b in legacy if b["model"]]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("LLM_BACKENDS is not valid JSON: %s", e)
        return [b for b in legacy if b["model"]]
    if not isinstance(parsed, list):
        logger.error("LLM_BACKENDS must be a JSON list")
        return [b for b in legacy if b["model"]]
    out = []
    for b in parsed:
        if not isinstance(b, dict):
            continue
        name = str(b.get("name") or "").strip()
        btype = str(b.get("type") or "openai").strip().lower()
        url = str(b.get("url") or "").strip().rstrip("/")
        model = str(b.get("model") or "").strip()
        api_key = str(b.get("api_key") or "").strip()
        # vLLM-native chat-template overrides, OPT-IN per backend: strict
        # OpenAI-schema servers reject the field with HTTP 422
        # "extra_forbidden" (probe-verified 2026-10-31 through the LAN
        # LiteLLM proxy -> Mistral), so it is only sent when declared.
        ctk = b.get("chat_template_kwargs")
        chat_kwargs = dict(ctk) if isinstance(ctk, dict) else {}
        if btype not in ("ollama", "openai") or not url or not name:
            # Redacted on purpose: a malformed entry may carry an inline
            # api_key — the key must NEVER reach the logs (the app's own
            # invariant, pinned by test).
            logger.warning("Skipping invalid LLM backend entry: %r",
                           {k: v for k, v in b.items() if k != "api_key"})
            continue
        out.append({"name": name, "type": btype, "url": url, "model": model,
                    "api_key": api_key, "chat_template_kwargs": chat_kwargs})
    return out


# ---------------------------------------------------------------------------
# Persistent selection
# ---------------------------------------------------------------------------

def _load_state() -> dict[str, Any]:
    try:
        return json.loads(_DEFAULT_STATE_FILE.read_text())
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("Could not read LLM state file: %s", e)
        return {}


def _save_state(state: dict[str, Any]) -> None:
    try:
        _DEFAULT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DEFAULT_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.warning("Could not write LLM state file %s: %s", _DEFAULT_STATE_FILE, e)


# ---------------------------------------------------------------------------
# Backend / model discovery
# ---------------------------------------------------------------------------

async def list_backends() -> list[dict[str, Any]]:
    """Discover every configured backend and the models it currently serves."""
    out: list[dict[str, Any]] = []
    for b in _configured_backends():
        models: list[str] = []
        error: str | None = None
        try:
            models = await _fetch_models(b)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning("Model discovery failed for %s (%s): %s",
                           b["name"], b["url"], e)
        out.append({
            "name": b["name"],
            "type": b["type"],
            "url": b["url"],
            "models": models,
            "error": error,
        })
    return out


async def _fetch_models(b: dict[str, Any]) -> list[str]:
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    headers = _auth_headers(b)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if b["type"] == "ollama":
            resp = await client.get(b["url"] + "/api/tags")
            resp.raise_for_status()
            data = resp.json()
            names = [m.get("name", "") for m in data.get("models", [])]
        else:
            resp = await client.get(b["url"] + "/v1/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            names = [m.get("id", "") for m in data.get("data", [])]
    return [n for n in names if n]


async def current_backend() -> dict[str, Any]:
    """The active backend (persisted selection, else the first configured)."""
    backends = _configured_backends()
    if not backends:
        return {}
    state = _load_state()
    selected = state.get("selected")
    for b in backends:
        if b["name"] == selected:
            override = state.get("models", {}).get(b["name"])
            if override:
                b = {**b, "model": override}
            return b
    return backends[0]


async def current_model_label() -> str:
    """Human label for the active model, e.g. ``llama-server · Qwen3.8-27B``."""
    b = await current_backend()
    if not b:
        return "no LLM configured"
    return f"{b['name']} · {b['model']}"


async def select_backend(name: str, model: str | None = None) -> dict[str, Any]:
    """Switch the active backend by name and optionally the model within it.

    Persisted across restarts. ``model`` may be an empty string to clear a
    previous runtime override for that backend.
    """
    backends = _configured_backends()
    match = next((b for b in backends if b["name"] == name), None)
    if match is None:
        raise ValueError(f"unknown LLM backend: {name!r}")
    state = _load_state()
    state["selected"] = name
    if model is not None:
        state.setdefault("models", {})
        if model:
            state["models"][name] = model
        else:
            state["models"].pop(name, None)
    _save_state(state)
    return await current_backend()


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------

# Requests carry no decoding options: the cloud model backend does not honor
# Ollama seed/temperature, so pinning would be ignored anyway and only adds
# false reproducibility expectations to backtests.

async def chat(messages: list[dict[str, str]], *,
               temperature: float | None = None,
               timeout_s: float | None = None) -> dict[str, Any]:
    """Send a chat request to the active backend.

    ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts (system
    messages included). ``temperature`` is sent ONLY when set (None leaves
    the server's default) — the sell layer passes its frozen recorded
    temperature so the wire matches what every consult record declares.
    ``timeout_s`` overrides the read bound per call. Returns ``{"text": ...,
    "backend": ..., "model": ..., "reasoning": ...}``.
    """
    backend = await current_backend()
    if not backend:
        return {
            "text": "No LLM backend configured",
            "backend": "",
            "model": "",
            "reasoning": "",
        }
    text, reasoning, model = await _post_chat(backend, messages,
                                              temperature=temperature,
                                              timeout_s=timeout_s)
    return {
        "text": text,
        "backend": backend["name"],
        "model": model,
        "reasoning": reasoning,
    }


async def chat_stream(messages: list[dict[str, str]]):
    """Stream a chat request to the active backend.

    Yields dicts:
      - {"type": "delta", "text": "..."} for each content chunk
      - {"type": "done", "text": full, "reasoning": ..., "backend": ...,
         "model": ...} at the end
      - {"type": "error", "text": "..."} if the request fails before any text

    For OpenAI-compatible backends, parses the SSE stream and extracts
    ``choices[0].delta.content``. For Ollama, parses the NDJSON stream and
    extracts ``message.content``. ``reasoning_content`` / ``reasoning`` is
    collected but not streamed (kept out of the visible text).
    """
    backend = await current_backend()
    if not backend:
        yield {"type": "error", "text": "No LLM backend configured"}
        return
    timeout = httpx.Timeout(
        connect=10.0,
        read=settings.ollama_timeout_seconds,
        write=30.0,
        pool=10.0,
    )
    full_parts: list[str] = []
    reasoning_parts: list[str] = []
    model_id = ""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if backend["type"] == "ollama":
                req_body = {
                    "model": backend["model"],
                    "messages": messages,
                    "stream": True,
                }
                async with client.stream(
                    "POST", backend["url"] + "/api/chat", json=req_body,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise RuntimeError(
                            f"LLM {backend['name']} HTTP {resp.status_code}: "
                            f"{body.decode(errors='replace')[:200]}"
                        )
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        msg = chunk.get("message") or {}
                        delta = msg.get("content") or ""
                        r = msg.get("reasoning") or msg.get("reasoning_content") or ""
                        if delta:
                            full_parts.append(delta)
                            yield {"type": "delta", "text": delta}
                        elif r and not full_parts:
                            yield {"type": "thinking"}
                        if r:
                            reasoning_parts.append(r)
                        if chunk.get("model"):
                            model_id = chunk["model"]
                        if chunk.get("done"):
                            break
            else:
                payload: dict[str, Any] = {
                    "model": backend["model"] or "default",
                    "messages": messages,
                    "stream": True,
                }
                # Same opt-in contract as _post_chat (the non-streaming
                # path): the vLLM-native chat_template_kwargs ride ONLY on
                # a declared backend — strict OpenAI-schema servers reject
                # the field with 422 "extra_forbidden" (probe-verified
                # 2026-10-31 through the LAN LiteLLM proxy -> Mistral) —
                # and on the strict path the effort rides as the
                # OpenAI-style reasoning_effort (the operator's value,
                # case-normalized — never a curated list; the server
                # validates it). This is
                # the live chat UI's wire (the chat endpoint + the sim
                # chat), so the streaming variant must not regress it.
                chat_kwargs = dict(backend.get("chat_template_kwargs") or {})
                effort = str(settings.llm_reasoning_effort or "").strip().lower()
                if chat_kwargs:
                    if effort in ("low", "medium", "high", "xhigh"):
                        chat_kwargs["thinking_budget"] = {
                            "low": 512,
                            "medium": 2048,
                            "high": 8192,
                            "xhigh": 32768,
                        }[effort]
                    payload["chat_template_kwargs"] = chat_kwargs
                elif effort:
                    payload["reasoning_effort"] = effort
                async with client.stream(
                    "POST", backend["url"] + "/v1/chat/completions",
                    json=payload, headers=_auth_headers(backend),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise RuntimeError(
                            f"LLM {backend['name']} HTTP {resp.status_code}: "
                            f"{body.decode(errors='replace')[:200]}"
                        )
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: "):]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        if chunk.get("model"):
                            model_id = chunk["model"]
                        choice = (chunk.get("choices") or [{}])[0]
                        if not isinstance(choice, dict):
                            continue
                        delta_obj = choice.get("delta") or {}
                        delta = delta_obj.get("content") or ""
                        r = delta_obj.get("reasoning_content") or delta_obj.get("reasoning") or ""
                        if delta:
                            full_parts.append(delta)
                            yield {"type": "delta", "text": delta}
                        elif r and not full_parts:
                            # The model is producing reasoning_content (thinking)
                            # but no visible content yet — emit a thinking event
                            # so the frontend can keep the typing indicator alive
                            # instead of looking frozen (Qwen3 does this for minutes).
                            yield {"type": "thinking"}
                        if r:
                            reasoning_parts.append(r)
        yield {
            "type": "done",
            "text": "".join(full_parts),
            "reasoning": "".join(reasoning_parts),
            "backend": backend["name"],
            "model": model_id or backend["model"],
        }
    except httpx.RequestError as e:
        yield {"type": "error", "text": f"Network error: {e}"}
    except Exception as e:
        # If we already streamed some text, surface the partial plus the error.
        # Otherwise emit an error message as the assistant reply.
        if full_parts:
            yield {
                "type": "done",
                "text": "".join(full_parts),
                "reasoning": "".join(reasoning_parts),
                "backend": backend["name"],
                "model": model_id or backend["model"],
                "error": str(e),
            }
        else:
            yield {"type": "error", "text": f"{type(e).__name__}: {e}"}


async def _post_chat(
    backend: dict[str, Any], messages: list[dict[str, str]],
    *, temperature: float | None = None, timeout_s: float | None = None
) -> tuple[str, str, str]:
    """Raw POST for one backend; returns (text, reasoning, model_id)."""
    timeout = httpx.Timeout(
        connect=10.0,
        read=(timeout_s if timeout_s is not None
              else settings.ollama_timeout_seconds),
        write=30.0,
        pool=10.0,
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        if backend["type"] == "ollama":
            req_body: dict[str, Any] = {
                "model": backend["model"], "messages": messages, "stream": False,
            }
            if temperature is not None:
                req_body["options"] = {"temperature": temperature}
            resp = await client.post(backend["url"] + "/api/chat", json=req_body)
        else:
            payload: dict[str, Any] = {
                "model": backend["model"] or "default",
                "messages": messages,
                "stream": False,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            # The vLLM-native kwargs ride ONLY when the backend declared
            # them: strict OpenAI-schema servers (LiteLLM routing to
            # Mistral/OpenAI/...) reject the field with 422
            # "extra_forbidden" (probe-verified 2026-10-31), so the clean
            # body is the default. On the strict path the reasoning effort
            # travels as the OpenAI-style ``reasoning_effort`` (the
            # operator's value, case-normalized — never a curated list):
            # the supported vocabulary is SERVER-SPECIFIC — the LAN proxy's
            # Mistral route answers none/low/high/xhigh/max and rejects
            # "medium" (HTTP 400 "reasoning_effort 'medium' is not supported
            # for this model; supported values: ['none', 'low', 'high',
            # 'xhigh', 'max']"; low/high answer 200 with
            # ``reasoning_content`` + ``usage.reasoning_tokens``), while
            # OpenAI itself takes "medium" but not xhigh/max — so the app
            # does NOT curate the list; the operator sets a value their
            # server supports and the server validates it (all probe-
            # verified 2026-10-31).
            chat_kwargs = dict(backend.get("chat_template_kwargs") or {})
            effort = str(settings.llm_reasoning_effort or "").strip().lower()
            if chat_kwargs:
                if effort in ("low", "medium", "high", "xhigh"):
                    chat_kwargs["thinking_budget"] = {
                        "low": 512,
                        "medium": 2048,
                        "high": 8192,
                        "xhigh": 32768,
                    }[effort]
                payload["chat_template_kwargs"] = chat_kwargs
            elif effort:
                payload["reasoning_effort"] = effort
            resp = await client.post(
                backend["url"] + "/v1/chat/completions",
                json=payload, headers=_auth_headers(backend),
            )
        if getattr(resp, "status_code", 200) != 200:
            raise RuntimeError(
                f"LLM backend {backend['name']} returned HTTP {resp.status_code}"
            )
        data = resp.json()
    return _extract_content(data, backend)


def _extract_content(data: Any, backend: dict[str, Any]) -> tuple[str, str, str]:
    """Pull (text, reasoning, model) from either an OpenAI-compatible or an
    Ollama chat response shape, whichever is present."""
    if isinstance(data, dict):
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") if isinstance(choice, dict) else None
        content = (msg or {}).get("content") or ""
        reasoning = (msg or {}).get("reasoning_content") or (msg or {}).get("reasoning") or ""
        model = data.get("model") or ""
        if content:
            return content, reasoning, model
    msg = data.get("message") if isinstance(data, dict) else None
    content = (msg or {}).get("content") or (data.get("response") if isinstance(data, dict) else "") or ""
    reasoning = (msg or {}).get("reasoning") or ""
    return content, reasoning, backend["model"]
