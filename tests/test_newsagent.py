"""Offline tests for the type-D material-news agent (app/newsagent.py).

Zero network: the MCP client and the LLM turn are fakes injected into
``run_agent`` (the agent's injection points), the llm tools-surface
helpers are pure functions, and the MCP client's protocol flow is tested
against a scripted httpx replacement.
"""

import json

import httpx
import pytest

from app import llm as llm_mod
from app import newsagent
from app.mcpclient import McpClient, McpError, _parse_response

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_TOOLS = [{"name": "searxng_web_search", "description": "web search",
           "inputSchema": {"type": "object",
                          "properties": {"query": {"type": "string"}},
                          "required": ["query"]}}]


class FakeMcp:
    """The agent's MCP surface, scripted: fails by name or wholesale."""

    def __init__(self, *, results=None, fail_names=(), fail_all=False,
                 tools=None):
        self.results = results or {}
        self.fail_names = set(fail_names)
        self.fail_all = fail_all
        self.tools = tools if tools is not None else _TOOLS
        self.calls = []

    async def openai_tools(self):
        return [{"type": "function",
                 "function": {"name": t["name"],
                              "description": t.get("description", ""),
                              "parameters": t.get("inputSchema")}}
                for t in self.tools]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail_all or name in self.fail_names:
            raise McpError("mcp tool boom")
        return self.results.get(name, "search result text")

    async def close(self):
        pass


def transport_turns(turns):
    """A scripted transport: each call returns the next llm.chat-shaped dict;
    every (messages, tools) it is called with is captured."""
    seen = []
    it = iter(turns)

    async def _transport(messages, tools):
        seen.append((list(messages), tools))
        return next(it)
    _transport.seen = seen
    return _transport


def _ctx():
    return {"ticker": "ACME", "stated_thesis_condition": "ROE > 0",
            "months_held": 4.2}


def _tool_call(name="searxng_web_search", args='{"query": "ACME news"}',
               cid="call_1"):
    return {"id": cid, "name": name, "arguments": json.loads(args),
            "arguments_raw": args}


def _turn(text="", calls=None, backend="llama-server",
          model="deepseek-v4.1-flash-mistral"):
    return {"text": text, "tool_calls": calls or [], "backend": backend,
            "model": model, "reasoning": ""}


_HOLD = '{"decision": "HOLD", "citations": [], "rationale": "nothing material"}'
_EXIT = ('{"decision": "EXIT", "citations": [{"title": "ACME fraud probe", '
         '"source": "reuters", "date": "2026-10-31"}], '
         '"rationale": "fraud accusation"}')


# ---------------------------------------------------------------------------
# run_agent — the loop
# ---------------------------------------------------------------------------

async def test_search_then_hold_records_trace_and_replay():
    mcp = FakeMcp()
    t = transport_turns([_turn(calls=[_tool_call()]), _turn(text=_HOLD)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=mcp, transport=t)
    assert rec["decision"] == "HOLD" and rec["parsed"] is True
    assert rec["fail_safe"] is False and rec["fail_reason"] is None
    assert rec["tool_calls"] == 1 and rec["turns"] == 2
    assert len(rec["trace"]) == 1
    assert rec["trace"][0]["name"] == "searxng_web_search"
    assert rec["trace"][0]["ok"] is True
    assert rec["temperature"] == newsagent.TEMPERATURE == 0.6
    assert rec["news_source"] == "searxng-mcp"
    assert rec["event_type"] == "D"
    assert rec["prompt_version"] == "sellshadow/1-d"
    assert rec["backend"] == "llama-server"
    assert rec["model"] == "deepseek-v4.1-flash-mistral"
    assert rec["schema"] == "newsagent-consult/1"
    # the second turn saw the replayed assistant tool_calls + the tool result
    msgs2 = t.seen[1][0]
    assert any(m.get("role") == "assistant" and m.get("tool_calls")
               and m["tool_calls"][0]["function"]["name"] == "searxng_web_search"
               for m in msgs2)
    assert any(m.get("role") == "tool"
               and m.get("content") == "search result text" for m in msgs2)


async def test_exit_with_citation_parses():
    t = transport_turns([_turn(text=_EXIT)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=FakeMcp(), transport=t)
    assert rec["decision"] == "EXIT" and rec["parsed"] is True
    assert rec["citations"][0]["title"] == "ACME fraud probe"


async def test_exit_without_citations_is_fail_safe_hold():
    t = transport_turns([_turn(
        text='{"decision": "EXIT", "citations": [], "rationale": "fraud"}')])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=FakeMcp(), transport=t)
    assert rec["decision"] == "HOLD" and rec["parsed"] is False
    assert rec["fail_safe"] is True
    assert "EXIT without citations" in rec["fail_reason"]


async def test_malformed_final_is_fail_safe_hold():
    t = transport_turns([_turn(text="I cannot answer that.")])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=FakeMcp(), transport=t)
    assert rec["decision"] == "HOLD" and rec["fail_safe"] is True
    assert rec["fail_reason"]


async def test_zero_search_verdict_is_accepted_no_floor():
    # The model decides when it has enough: a verdict with zero searches is
    # a legitimate record (auditable via the empty trace), not a harness
    # intervention — there is NO floor (the design note).
    t = transport_turns([_turn(text=_HOLD)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=FakeMcp(), transport=t)
    assert rec["decision"] == "HOLD" and rec["parsed"] is True
    assert rec["tool_calls"] == 0 and rec["trace"] == []


async def test_all_tool_calls_failed_fail_safes_even_on_model_verdict():
    mcp = FakeMcp(fail_all=True)
    t = transport_turns([_turn(calls=[_tool_call()]), _turn(text=_HOLD)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=mcp, transport=t)
    assert rec["fail_safe"] is True and rec["parsed"] is False
    assert "mcp unavailable" in rec["fail_reason"]
    # the failed call's error text WAS returned to the model (loop semantics)
    assert any(m.get("role") == "tool" and "tool error" in m.get("content", "")
               for m in t.seen[1][0])


async def test_partial_failure_returns_error_to_model_and_continues():
    mcp = FakeMcp(fail_names={"boom"})
    calls = [_tool_call(name="boom", cid="b"), _tool_call(cid="s")]
    t = transport_turns([_turn(calls=calls), _turn(text=_HOLD)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=mcp, transport=t)
    assert rec["parsed"] is True and rec["fail_safe"] is False
    assert rec["tool_calls"] == 2
    assert [e["ok"] for e in rec["trace"]] == [False, True]
    assert [e["kind"] for e in rec["trace"]] == ["mcp", "ok"]


async def test_unparsable_tool_arguments_returned_as_error():
    call = {"id": "c", "name": "searxng_web_search",
            "arguments": "not json", "arguments_raw": "not json"}
    t = transport_turns([_turn(calls=[call]), _turn(text=_HOLD)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=FakeMcp(), transport=t)
    # the model's OWN failed call is loop semantics, not an outage: the
    # error was returned to the model, which then verdicted — the verdict
    # stands (the same acceptance as a zero-search verdict)
    assert rec["parsed"] is True and rec["fail_safe"] is False
    assert rec["trace"][0]["ok"] is False
    assert rec["trace"][0]["kind"] == "args"
    assert "not a JSON object" in t.seen[1][0][-1]["content"]


async def test_mixed_args_and_mcp_failures_no_success_fail_safes():
    # one call garbled by the model (args), one killed by the middleman
    # (mcp), none succeeded: the MCP error makes it an outage record —
    # the fail-safe fires even though the model verdicted
    mcp = FakeMcp(fail_all=True)
    calls = [{"id": "a", "name": "searxng_web_search",
              "arguments": "not json", "arguments_raw": "not json"},
             _tool_call(cid="s")]
    t = transport_turns([_turn(calls=calls), _turn(text=_HOLD)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=mcp, transport=t)
    assert rec["fail_safe"] is True and rec["parsed"] is False
    assert "mcp unavailable" in rec["fail_reason"]
    assert [e["kind"] for e in rec["trace"]] == ["args", "mcp"]


async def test_hang_guard_is_an_anomaly_fail_safe(monkeypatch):
    monkeypatch.setattr(newsagent, "HANG_GUARD_TURNS", 3)
    # a transport that NEVER produces a final answer
    def _endless():
        while True:
            yield _turn(calls=[_tool_call(cid="c")])
    t = transport_turns(_endless())
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=FakeMcp(), transport=t)
    assert rec["fail_safe"] is True
    assert "hang guard" in rec["fail_reason"]
    assert rec["tool_calls"] == 3  # the ceiling, not a budget: 3 turns ran


async def test_transport_error_fail_safes():
    async def boom(messages, tools):
        raise RuntimeError("proxy down")
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=FakeMcp(), transport=boom)
    assert rec["fail_safe"] is True and rec["decision"] == "HOLD"
    assert rec["error_class"] == "RuntimeError"
    # when an exception was caught, fail_reason IS the root cause
    # (error_class: error), not the derivative why
    assert rec["fail_reason"] == "RuntimeError: proxy down"


async def test_mcp_handshake_failure_fail_safes():
    # the outage-at-openai_tools path: the MCP never even listed its tools
    class DeadMcp(FakeMcp):
        async def openai_tools(self):
            raise McpError("mcp transport error: ConnectError")
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=DeadMcp(),
                                    transport=transport_turns([]))
    assert rec["fail_safe"] is True and rec["decision"] == "HOLD"
    assert rec["error_class"] == "McpError"
    assert rec["fail_reason"] == \
        "McpError: mcp transport error: ConnectError"
    assert rec["turns"] == 0 and rec["tool_calls"] == 0 and rec["trace"] == []


async def test_no_mcp_url_is_config_fail_safe(monkeypatch):
    monkeypatch.setattr(newsagent.settings, "searxng_mcp_url", "")
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01")
    assert rec["fail_safe"] is True and rec["turns"] == 0
    assert "not configured" in rec["fail_reason"]


async def test_concurrent_calls_in_one_turn_execute_all():
    mcp = FakeMcp()
    calls = [_tool_call(cid="a", args='{"query": "q1"}'),
             _tool_call(cid="b", args='{"query": "q2"}')]
    t = transport_turns([_turn(calls=calls), _turn(text=_HOLD)])
    rec = await newsagent.run_agent(_ctx(), event_id="D:ACME:2026-11-01",
                                    mcp=mcp, transport=t)
    assert rec["tool_calls"] == 2 and len(mcp.calls) == 2
    assert [args["query"] for _n, args in mcp.calls] == ["q1", "q2"]
    assert [e["arguments"]["query"] for e in rec["trace"]] == ["q1", "q2"]


# ---------------------------------------------------------------------------
# _describe_tools — the instance-truth schema augmentation
# ---------------------------------------------------------------------------

def test_describe_tools_appends_instance_note_to_engines_param():
    tools = [{"type": "function", "function": {
        "name": "searxng_web_search", "description": "search",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "engines": {"type": "string",
                        "description": "generic text"}}}}}]
    out = newsagent._describe_tools(tools)
    desc = (out[0]["function"]["parameters"]["properties"]["engines"]
            ["description"])
    assert desc.startswith("generic text")
    assert "exaapi" in desc
    assert newsagent._INSTANCE_ENGINES_NOTE in desc
    assert "tinyfish news" in desc
    assert "exa keyless" in desc
    assert "Those scrapers support time_range" in desc
    assert "No engine on this instance supports time_range" not in desc


def test_describe_tools_is_a_noop_on_foreign_or_missing_shapes():
    # a non-searxng tool passes through untouched
    foreign = [{"type": "function", "function": {
        "name": "web_url_read", "description": "read",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}}}}]
    assert newsagent._describe_tools(foreign) == foreign
    # a searxng tool without an engines property: no crash, untouched
    bare = [{"type": "function", "function": {
        "name": "searxng_web_search", "description": "search",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}}}}]
    assert newsagent._describe_tools(bare) == bare


# ---------------------------------------------------------------------------
# parse_verdict units
# ---------------------------------------------------------------------------

def test_parse_verdict_exit_requires_citations():
    out = newsagent.parse_verdict(_EXIT)
    assert out["decision"] == "EXIT" and out["parsed"] is True
    bad = newsagent.parse_verdict('{"decision": "EXIT", "citations": []}')
    assert bad["fail_safe"] is True and bad["decision"] == "HOLD"


def test_parse_verdict_hold_accepts_empty_citations():
    out = newsagent.parse_verdict(_HOLD)
    assert out["decision"] == "HOLD" and out["citations"] == []


def test_parse_verdict_lowercase_uppered():
    out = newsagent.parse_verdict('{"decision": "hold", "citations": []}')
    assert out["decision"] == "HOLD" and out["parsed"] is True


def test_parse_verdict_garbage_is_fail_safe():
    for text in ("", "no json", "{broken", '["not", "object"]',
                 '{"decision": "MAYBE"}'):
        out = newsagent.parse_verdict(text)
        assert out["fail_safe"] is True and out["decision"] == "HOLD"


def test_parse_verdict_non_list_citations_treated_as_empty():
    out = newsagent.parse_verdict(
        '{"decision": "EXIT", "citations": "oops", "rationale": "x"}')
    assert out["fail_safe"] is True  # EXIT without a real citations list


# ---------------------------------------------------------------------------
# llm tools surface (the wire the agent rides)
# ---------------------------------------------------------------------------

def test_parse_tool_calls_normalizes_arguments_json():
    raw = [{"id": "1", "type": "function",
            "function": {"name": "f", "arguments": '{"a": 1}'}}]
    out = llm_mod._parse_tool_calls(raw)
    assert out == [{"id": "1", "name": "f", "arguments": {"a": 1},
                    "arguments_raw": '{"a": 1}'}]


def test_parse_tool_calls_unparsable_arguments_kept_raw():
    raw = [{"id": "1", "function": {"name": "f", "arguments": "not json"}}]
    out = llm_mod._parse_tool_calls(raw)
    assert out[0]["arguments"] == "not json"
    assert out[0]["arguments_raw"] == "not json"


def test_parse_tool_calls_skips_malformed_entries():
    assert llm_mod._parse_tool_calls("nope") == []
    assert llm_mod._parse_tool_calls([{"function": {"name": ""}}]) == []
    assert llm_mod._parse_tool_calls([{"function": {"name": "ok",
                                                   "arguments": "{}"}}])[0]["name"] == "ok"


def test_extract_content_tool_call_turn_with_empty_text():
    data = {"model": "m", "choices": [{"message": {
        "content": "",
        "tool_calls": [{"id": "9", "type": "function",
                        "function": {"name": "f",
                                     "arguments": '{"q": 1}'}}]}}]}
    text, _reasoning, model, calls = llm_mod._extract_content(
        data, {"model": "fallback"})
    assert text == "" and model == "m"
    assert calls[0]["name"] == "f" and calls[0]["arguments"] == {"q": 1}


def test_extract_content_text_turn_still_works():
    data = {"model": "m",
            "choices": [{"message": {"content": "plain answer"}}]}
    text, _r, model, calls = llm_mod._extract_content(data, {"model": "fb"})
    assert text == "plain answer" and model == "m" and calls == []


def test_chat_no_backend_includes_empty_tool_calls(monkeypatch):
    async def none_backend():
        return {}
    monkeypatch.setattr(llm_mod, "current_backend", none_backend)
    import asyncio
    out = asyncio.run(llm_mod.chat([{"role": "user", "content": "x"}]))
    assert out["tool_calls"] == [] and out["backend"] == ""


# ---------------------------------------------------------------------------
# mcpclient — the protocol flow, offline
# ---------------------------------------------------------------------------

def _resp(status=200, ct="application/json", body=None, **headers):
    return httpx.Response(status, headers={"content-type": ct, **headers},
                          content=(body if isinstance(body, bytes)
                                   else str(body or "").encode()))


class _FakeHttpx:
    """A scripted httpx.AsyncClient: pops one Response per post."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    async def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "headers": dict(headers or {}),
                           "json": json})
        return self.responses.pop(0)

    async def aclose(self):
        pass


async def test_mcp_client_handshake_tools_and_call():
    init = _resp(body=json.dumps({"jsonrpc": "2.0", "id": 1,
                                  "result": {"protocolVersion": "2025-03-26",
                                             "capabilities": {},
                                             "serverInfo": {"name": "x"}}}),
                 **{"mcp-session-id": "sid-1"})
    notif = _resp(status=202, body="")
    tools = _resp(body=json.dumps({
        "jsonrpc": "2.0", "id": 3,
        "result": {"tools": [{
            "name": "searxng_web_search",
            "description": "search",
            "inputSchema": {"type": "object", "properties": {}},
        }]},
    }))
    call = _resp(body=json.dumps(
        {"jsonrpc": "2.0", "id": 4,
         "result": {"content": [{"type": "text", "text": "headline A"}],
                    "isError": False}}))
    c = McpClient("http://mcp:8808/mcp")
    c._client = _FakeHttpx([init, notif, tools, call])
    try:
        oai = await c.openai_tools()
        assert oai == [{"type": "function",
                        "function": {"name": "searxng_web_search",
                                     "description": "search",
                                     "parameters": {"type": "object",
                                                    "properties": {}}}}]
        assert await c.call_tool("searxng_web_search", {"query": "x"}) == \
            "headline A"
        methods = [p["json"]["method"] for p in c._client.posts]
        assert methods == ["initialize", "notifications/initialized",
                           "tools/list", "tools/call"]
        # the offered session id rides on every later request
        assert all(p["headers"].get("mcp-session-id") == "sid-1"
                   for p in c._client.posts[1:])
        # initialize happens once, even across three operations
        assert methods.count("initialize") == 1
    finally:
        await c.close()


async def test_mcp_client_iserror_result_raises():
    init = _resp(body='{"jsonrpc":"2.0","id":1,"result":{}}')
    notif = _resp(status=202, body="")
    bad = _resp(body=json.dumps(
        {"jsonrpc": "2.0", "id": 4,
         "result": {"content": [{"type": "text", "text": "boom"}],
                    "isError": True}}))
    c = McpClient("http://mcp:8808/mcp")
    c._client = _FakeHttpx([init, notif, bad])
    try:
        with pytest.raises(McpError, match="mcp tool error"):
            await c.call_tool("searxng_web_search", {})
    finally:
        await c.close()


def test_parse_response_json_and_sse_and_error():
    ok = _resp(body='{"jsonrpc":"2.0","id":7,"result":{"x":1}}')
    assert _parse_response(ok, 7)["result"] == {"x": 1}
    sse = _resp(ct="text/event-stream",
                body='data: {"jsonrpc":"2.0","method":"notifications/message"}\n'
                     '\n'
                     'data: {"jsonrpc":"2.0","id":7,"result":{"y":2}}\n')
    # the id-matching message wins, log notifications in the stream are skipped
    assert _parse_response(sse, 7)["result"] == {"y": 2}
    with pytest.raises(McpError, match="mcp protocol error"):
        _parse_response(_resp(
            body='{"jsonrpc":"2.0","id":7,"error":{"message":"nope"}}'), 7)
    with pytest.raises(McpError, match="no JSON-RPC message"):
        _parse_response(_resp(ct="text/event-stream", body=""), 7)
