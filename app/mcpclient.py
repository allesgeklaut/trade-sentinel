"""The thin MCP client for the type-D material-news agent.

Speaks just enough JSON-RPC over the Streamable HTTP transport for
``app/newsagent.py``: the initialize handshake, ``tools/list`` and
``tools/call``. Deliberately hand-rolled over httpx (the design note's
lean-client choice, docs/sell-news-agent-design-2026-11-01.md): the shared
searxng MCP server (supergateway, stateless) serves every request with a
FRESH child, so there is no session state to keep — the client holds only
the ``mcp-session-id`` the server hands back (sent on later requests;
ignored by the stateless server, honored by a stateful one).

Fail-safe discipline: ANY failure raises :class:`McpError` — the agent
records a fail-safe row and the next night retries; a middleman outage
never touches the book. No retries, no fallbacks: one path, one error.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger("trade_sentinel.mcpclient")

# The protocol version both sides accepted in the live step-0 pin
# (2026-11-01: supergateway 4.2.1 serving mcp-searxng 2.9.0).
_PROTOCOL_VERSION = "2025-03-26"


class McpError(RuntimeError):
    """Any MCP transport/protocol failure. The caller records a fail-safe
    row (never raises into the book)."""


class McpClient:
    """A minimal Streamable-HTTP MCP client for ONE tool server."""

    def __init__(self, base_url: str, *, timeout_s: float | None = None):
        self._base = str(base_url).rstrip("/")
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=(timeout_s if timeout_s is not None
                  else settings.searxng_mcp_timeout_s),
            write=30.0, pool=10.0)
        self._headers = {"content-type": "application/json",
                         "accept": "application/json, text/event-stream"}
        self._client: httpx.AsyncClient | None = None
        self._initialized = False
        self._id = 0

    # -- internals ------------------------------------------------------

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _post(self, payload: dict[str, Any], *,
                    notification: bool = False) -> dict[str, Any]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        try:
            resp = await self._client.post(self._base, headers=self._headers,
                                           json=payload)
        except httpx.RequestError as e:
            raise McpError(f"mcp transport error: {type(e).__name__}: {e}") from e
        if resp.status_code not in (200, 202):
            raise McpError(f"mcp http {resp.status_code}: {resp.text[:200]}")
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self._headers["mcp-session-id"] = sid
        if notification or not resp.text.strip():
            return {}
        return _parse_response(resp, payload.get("id"))

    async def _ensure_session(self) -> None:
        """The initialize handshake, once per client. A stateless server
        (supergateway) answers anyway; a session id, when offered, is kept
        and sent on later requests."""
        if self._initialized:
            return
        await self._post({
            "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
            "params": {"protocolVersion": _PROTOCOL_VERSION,
                       "capabilities": {},
                       "clientInfo": {"name": "trade-sentinel-newsagent",
                                      "version": "1"}},
        })
        await self._post({"jsonrpc": "2.0",
                          "method": "notifications/initialized"},
                         notification=True)
        self._initialized = True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- the three operations -------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """``tools/list`` → the raw tool definitions (name, description,
        inputSchema)."""
        await self._ensure_session()
        body = await self._post({"jsonrpc": "2.0", "id": self._next_id(),
                                 "method": "tools/list", "params": {}})
        tools = body.get("result", {}).get("tools")
        if not isinstance(tools, list):
            raise McpError("mcp tools/list: no tools array in the result")
        return tools

    async def openai_tools(self) -> list[dict[str, Any]]:
        """The tools translated into the OpenAI ``tools`` format — the
        native-function-calling bridge for ``llm.chat``."""
        out: list[dict[str, Any]] = []
        for t in await self.list_tools():
            schema = t.get("inputSchema")
            out.append({
                "type": "function",
                "function": {
                    "name": str(t.get("name") or ""),
                    "description": str(t.get("description") or ""),
                    "parameters": (schema if isinstance(schema, dict)
                                   else {"type": "object", "properties": {}}),
                },
            })
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """``tools/call`` → the tool's text content (text blocks joined
        with a blank line). An ``isError`` result is an :class:`McpError` —
        the tool itself failed."""
        await self._ensure_session()
        body = await self._post({"jsonrpc": "2.0", "id": self._next_id(),
                                 "method": "tools/call",
                                 "params": {"name": name,
                                            "arguments": arguments or {}}})
        result = body.get("result")
        if not isinstance(result, dict):
            raise McpError("mcp tools/call: no result object")
        parts = [str(b.get("text") or "")
                 for b in (result.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "text"]
        text = "\n\n".join(p for p in parts if p)
        if result.get("isError"):
            raise McpError(f"mcp tool error: {text[:200]}")
        return text


def _parse_response(resp: httpx.Response, expected_id: Any) -> dict[str, Any]:
    """One Streamable-HTTP response body: a JSON object or an SSE stream
    of JSON-RPC messages (the response is the message whose ``id`` matches
    the request; log notifications in the stream are skipped). A JSON-RPC
    error object raises :class:`McpError`."""
    ct = resp.headers.get("content-type", "")
    bodies: list[Any]
    if "text/event-stream" in ct:
        bodies = []
        for ln in resp.text.splitlines():
            if ln.startswith("data:"):
                try:
                    bodies.append(json.loads(ln[len("data:"):].strip()))
                except (json.JSONDecodeError, ValueError):
                    continue
    else:
        try:
            bodies = [resp.json()]
        except ValueError as e:
            raise McpError(f"mcp: response is not JSON: {e}") from e
    match: dict[str, Any] | None = None
    for b in bodies:
        if isinstance(b, dict) and b.get("id") == expected_id:
            match = b
    if match is None:
        dicts = [b for b in bodies if isinstance(b, dict)]
        if not dicts:
            raise McpError("mcp: no JSON-RPC message in the response")
        match = dicts[-1]
    if "error" in match:
        err = match["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise McpError(f"mcp protocol error: {str(msg)[:200]}")
    return match
