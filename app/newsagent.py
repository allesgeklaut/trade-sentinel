"""The type-D material-news agent — the sell program's news step.

A REAL agent (docs/sell-news-agent-design-2026-11-01.md): the model plans
the research — what to search, what to read, how deep — and decides when
it has enough; the loop runs to the model's final answer. The harness
executes tool calls, records the trace, and converts outages into
fail-safe rows. It does NOT shape the research: no call caps, no token
limits, no search floors, no query templates, no wall-clock budgets. The
one limit is HANG_GUARD_TURNS, a runaway ceiling whose firing is an
anomaly to investigate (recorded as a fail-safe row), never a budget —
the researched pattern: termination is TYPED (the model's final answer),
framework turn limits are safety ceilings.

Wire: the app's ONE LLM wire (``llm.chat`` with the OpenAI tools surface)
at TEMPERATURE (the DeepSeek family's own documented recommendation,
0.6 — chosen to prevent the greedy-repetition pathology; recorded in
every row) and the operator's reasoning-effort knob (recorded too).
Tools: the shared searxng MCP via :mod:`app.mcpclient` — the ONLY news
path for D (no direct-SearXNG client, the owner's call).

Shadow-first: every run is recorded in the ``LlmSellDecision`` ledger
(event_type "D", engine "monthly", the fill columns NULL forever in the
shadow phase) with NO execution authority; the promotion rule lives in
the design note. Nightly cadence: one run per monthly-book name, right
after the type-C step (``python -m app.newsagent news``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text

from .config import settings
from .db import Candle, Fundamental, MonthlyPosition, Session
from .llm import chat as llm_chat
from .mcpclient import McpClient, McpError
from .selllayer import _record_decision as _record_ledger_row
from .sellreview import _THESIS_BREAK_CONDITION

logger = logging.getLogger("trade_sentinel.newsagent")

PROMPT_VERSION = "sellshadow/1-d"
SCHEMA_RECORD = "newsagent-consult/1"
SCHEMA_RUN = "newsagent-run/1"
NEWS_SOURCE = "searxng-mcp"
ERROR_TEXT_MAX = 200
# The DeepSeek family's documented recommendation ("Set the temperature
# within the range of 0.5-0.7 (0.6 is recommended) to prevent endless
# repetitions or incoherent outputs" — the R1-series usage guidance; their
# own product runs 0.6). See the design note's researched-pattern section.
TEMPERATURE = 0.6
# A RUNAWAY ceiling, not a budget: firing it is an anomaly to investigate
# (recorded as a fail-safe row), never normal operation.
HANG_GUARD_TURNS = 50
# The counterfactual every D row carries (the engine's own rule: the live
# book waits for the month-end rebuild; the shadow never acts).
ENGINE_BRANCH = {
    "rule": "hold to next month-end rebuild",
    "note": "shadow: no execution authority; the fill columns stay NULL "
            "by design (docs/sell-news-agent-design-2026-11-01.md)",
}
# The fundamental tags that anchor the stated thesis (the same facts the
# C step's break condition evaluates — see sellreview's predeclared
# condition; D judges NEWS against that thesis, it does not re-audit the
# filings, which is C's job).
_FUNDAMENTAL_TAGS = (
    "NetIncomeLoss",
    "StockholdersEquity",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "CommonStockSharesOutstanding",
)

SYSTEM_PROMPT = (
    "You are the news investigator for one holding of a paper-trading "
    "monthly book.\n"
    "TASK: decide from recent news whether this position's stated thesis "
    "still holds. Fraud, guidance cuts, structural breaks, delisting or "
    "bankruptcy risk, and major litigation are the invalidating kind. "
    "Price moves alone are not news.\n"
    'OUTPUT: reply with a single JSON object and nothing else — '
    '{"decision": "EXIT" or "HOLD", "citations": [{"title": ..., '
    '"source": ..., "date": ...}], "rationale": ...}. EXIT requires at '
    "least one citation, its article read and verified; HOLD takes an "
    "empty list."
)


# ---------------------------------------------------------------------------
# The frozen prompt + its fail-safe verdict parser (pure, versioned)
# ---------------------------------------------------------------------------

def render_prompt(context: dict) -> list[dict]:
    """The initial message list: the frozen instruction + the position
    context rendered as stable sorted-keys JSON (byte-identical for
    identical context)."""
    user = (f"prompt-version: {PROMPT_VERSION}\n"
            f"position context (JSON, sorted keys):\n"
            + json.dumps(context, sort_keys=True, indent=2, allow_nan=False))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _fail_safe(why: str) -> dict:
    """The type-D fail-safe: HOLD (the engine's rule — the live book waits
    for month-end)."""
    return {"decision": "HOLD", "rationale": f"fail-safe: {why}",
            "citations": [], "parsed": False, "fail_safe": True, "why": why}


def parse_verdict(text: str) -> dict:
    """The fail-safe verdict parser. Accepts exactly one of EXIT or HOLD.
    EXIT requires a non-empty ``citations`` list — an EXIT the trace
    cannot audit is worthless (the design note's citation rule). Malformed
    output, refusal, or missing JSON -> HOLD (fail-safe to the engine's
    current rule)."""
    if not isinstance(text, str) or not text.strip():
        return _fail_safe("empty output")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return _fail_safe("no JSON object")
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return _fail_safe("unparsable JSON")
    if not isinstance(obj, dict):
        return _fail_safe("JSON is not an object")
    decision = str(obj.get("decision", "")).strip().upper()
    rationale = str(obj.get("rationale") or "")
    citations = obj.get("citations")
    if not isinstance(citations, list):
        citations = []
    if decision == "EXIT":
        if not citations:
            return _fail_safe("EXIT without citations")
        return {"decision": "EXIT", "rationale": rationale,
                "citations": citations, "parsed": True, "fail_safe": False,
                "why": None}
    if decision == "HOLD":
        return {"decision": "HOLD", "rationale": rationale,
                "citations": citations, "parsed": True, "fail_safe": False,
                "why": None}
    return _fail_safe(f"unknown decision {decision!r}")


# ---------------------------------------------------------------------------
# The agent loop — model-owned termination; a runaway ceiling, not a budget
# ---------------------------------------------------------------------------

async def _default_transport(messages: list[dict],
                             tools: list[dict]) -> dict:
    """One LLM turn on the app's ONE wire: temperature TEMPERATURE (the
    frozen recorded constant — the wire matches what every D row
    declares), the operator's reasoning effort (the global knob rides the
    wire as everywhere), and the OpenAI tools surface. Returns the
    ``llm.chat`` dict (text + tool_calls + backend + model)."""
    return await llm_chat(messages, temperature=TEMPERATURE, tools=tools)


async def _execute_calls(mcp, calls: list[dict]) -> list[dict]:
    """Execute one turn's tool calls CONCURRENTLY (the tools are read-only
    and the MCP server is stateless) and return each call's outcome. A
    FAILED call returns its error text as the call's result — standard
    agent-loop semantics: the model sees tool errors and adapts. ``kind``
    records WHICH class a failed call was: ``"mcp"`` (the middleman
    errored — run_agent's outage fail-safe keys on this) or ``"args"``
    (the model's own unparsable arguments — loop semantics; the model saw
    the error, its verdict stands, the same acceptance as a zero-search
    verdict)."""
    async def one(call: dict) -> dict:
        args = call["arguments"]
        if not isinstance(args, dict):
            return {"ok": False, "kind": "args",
                    "text": "tool error: arguments are not a JSON object: "
                            + str(call["arguments_raw"])[:120]}
        try:
            return {"ok": True, "kind": "ok",
                    "text": await mcp.call_tool(call["name"], args)}
        except McpError as e:
            return {"ok": False, "kind": "mcp",
                    "text": f"tool error: {str(e)[:ERROR_TEXT_MAX]}"}
    return list(await asyncio.gather(*(one(c) for c in calls)))


# The tool surface the model receives is the MCP server's generic text,
# written for arbitrary SearXNG instances — its `engines` examples
# ('google,bing,ddg') name engines dead on OUR instance, which invited
# engine-selection flailing in the second live smoke (the model burned
# turns discovering dead engines by hand). The agent appends the INSTANCE
# truth to that parameter before handing the surface to the model:
# honest environment documentation, not research steering — the model
# still owns every search decision (prompt sellshadow/1-d stays frozen).
_INSTANCE_ENGINES_NOTE = (
    "Live on this instance: exaapi (Exa neural web search — by far the "
    "highest-quality engine here) and wikipedia. Omitting `engines` uses "
    "the default set: exaapi + wikipedia for general, exaapi for news — "
    "this is best. Other engine names are either ignored (unregistered — "
    "the default set is used) or off the default set. No engine on this "
    "instance supports time_range: omit it."
)


def _describe_tools(tools: list[dict]) -> list[dict]:
    """Augment the searxng_web_search tool description with the instance's
    live-engine truth (see _INSTANCE_ENGINES_NOTE). No-op on any tool
    shape that does not match — the surface passes through unchanged."""
    for t in tools:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if fn.get("name") != "searxng_web_search":
            continue
        props = (fn.get("parameters") or {}).get("properties", {})
        eng = props.get("engines")
        if isinstance(eng, dict):
            eng["description"] = (
                str(eng.get("description") or "").strip()
                + " " + _INSTANCE_ENGINES_NOTE).strip()
    return tools


async def run_agent(context: dict, *, event_id: str,
                    mcp=None, transport=None) -> dict:
    """One agent run for one position. Returns the ``newsagent-consult/1``
    record (the CALLER records the ledger row). Never raises: every
    failure — MCP outage, transport error, hang guard, unparsable final —
    is the fail-safe record, retried the next night.

    ``mcp`` (anything with ``openai_tools()`` / ``call_tool(name, args)``
    coroutines) and ``transport(messages, tools)`` are injectable so tests
    run fully offline; the defaults are the real MCP client (built from
    ``settings.searxng_mcp_url``) and the real LLM wire.
    """
    started = time.perf_counter()
    own_client = mcp is None
    if own_client and not settings.searxng_mcp_url:
        return _build_record(context, event_id,
                             _fail_safe("searxng_mcp_url not configured"),
                             trace=[], turns=0, tool_calls=0,
                             backend="", model="", error=None,
                             error_class="ConfigError",
                             latency_ms=_latency_ms(started))
    trace: list[dict] = []
    turns = 0
    tool_calls = 0
    any_success = False
    any_mcp_error = False
    backend = model = ""
    error: str | None = None
    error_class: str | None = None
    parsed: dict = {}
    if own_client:
        mcp = McpClient(settings.searxng_mcp_url)
    try:
        tools = _describe_tools(await mcp.openai_tools())
        messages = render_prompt(context)
        while True:
            turns += 1
            if turns > HANG_GUARD_TURNS:
                # The runaway ceiling: an anomaly to investigate, never a
                # budget (the exit is the model's final answer).
                parsed = _fail_safe(
                    f"hang guard: {HANG_GUARD_TURNS} turns without a "
                    f"final answer")
                break
            result = await (transport or _default_transport)(messages, tools)
            backend = str(result.get("backend") or "") or backend
            model = str(result.get("model") or "") or model
            calls = result.get("tool_calls") or []
            if calls:
                messages.append({
                    "role": "assistant",
                    "content": str(result.get("text") or ""),
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"],
                                      "arguments": c["arguments_raw"]}}
                        for c in calls],
                })
                outcomes = await _execute_calls(mcp, calls)
                for call, outcome in zip(calls, outcomes):
                    tool_calls += 1
                    any_success = any_success or outcome["ok"]
                    any_mcp_error = any_mcp_error or outcome["kind"] == "mcp"
                    trace.append({"name": call["name"],
                                  "arguments": call["arguments"],
                                  "ok": outcome["ok"],
                                  "kind": outcome["kind"],
                                  "result_chars": len(outcome["text"]),
                                  "result_head": outcome["text"][:200]})
                    messages.append({"role": "tool",
                                     "tool_call_id": call["id"],
                                     "content": outcome["text"]})
                continue
            # The model's final answer — the typed exit.
            parsed = parse_verdict(str(result.get("text") or ""))
            if tool_calls > 0 and not any_success and any_mcp_error:
                # Every call failed AND the middleman errored: the
                # research never obtained any information while the news
                # path was failing — the consult did not happen (a
                # middleman outage never touches the book; the next night
                # retries). Calls that failed through the model's OWN
                # unparsable arguments are loop semantics — the model saw
                # the error and owns its verdict, like a zero-search one.
                parsed = _fail_safe("mcp unavailable (all tool calls failed)")
            break
    except McpError as e:
        error_class, error = "McpError", str(e)[:ERROR_TEXT_MAX]
        logger.warning("newsagent %s: mcp failure: %s", event_id, error)
        parsed = _fail_safe(f"mcp failure ({error_class})")
    except Exception as e:  # noqa: BLE001 — ANY failure is the fail-safe
        error_class, error = type(e).__name__, str(e)[:ERROR_TEXT_MAX]
        logger.warning("newsagent %s: agent failure (%s): %s",
                       event_id, error_class, error)
        parsed = _fail_safe(f"agent failure ({error_class})")
    finally:
        if own_client and mcp is not None:
            await mcp.close()
    return _build_record(context, event_id, parsed, trace=trace,
                         turns=turns, tool_calls=tool_calls,
                         backend=backend, model=model, error=error,
                         error_class=error_class,
                         latency_ms=_latency_ms(started))


def _latency_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _build_record(context: dict, event_id: str, parsed: dict, *, trace: list,
                  turns: int, tool_calls: int, backend: str, model: str,
                  error: str | None, error_class: str | None,
                  latency_ms: float) -> dict:
    fail_reason = None
    if parsed.get("fail_safe"):
        fail_reason = (f"{error_class}: {error}"
                       if error else
                       str(parsed.get("why") or "unparsable response")
                       [:ERROR_TEXT_MAX])
    return {
        "schema": SCHEMA_RECORD,
        "event_id": event_id,
        "event_type": "D",
        "prompt_version": PROMPT_VERSION,
        "temperature": TEMPERATURE,
        "reasoning_effort": str(settings.llm_reasoning_effort or ""),
        "decision": parsed["decision"],
        "rationale": str(parsed.get("rationale") or ""),
        "citations": parsed.get("citations") or [],
        "parsed": bool(parsed.get("parsed")),
        "fail_safe": bool(parsed.get("fail_safe")),
        "fail_reason": fail_reason,
        "error": error,
        "error_class": error_class,
        "latency_ms": latency_ms,
        "turns": turns,
        "tool_calls": tool_calls,
        "trace": trace,
        "news_source": NEWS_SOURCE,
        "backend": backend,
        "model": model,
    }


# ---------------------------------------------------------------------------
# The nightly driver — one agent run per monthly-book name
# ---------------------------------------------------------------------------

async def _last_session_date() -> str | None:
    """The DB's last stored candle session (read-only SQL) — the as-of
    default, the same anchor the shadow command uses."""
    async with Session() as s:
        last = await s.scalar(text("SELECT MAX(timestamp) FROM candles"))
    return str(last)[:10] if last is not None else None


async def _latest_fund(s, ticker: str, tag: str, as_of: str) -> dict | None:
    """The newest filed fundamental for (ticker, tag) PUBLIC as of the
    decision session (``filed <= S`` and ``end < S`` — the same
    filing-date PIT rule the engines apply)."""
    stmt = (select(Fundamental.end, Fundamental.filed, Fundamental.val)
            .where(Fundamental.ticker == ticker,
                   Fundamental.tag == tag,
                   Fundamental.filed <= as_of,
                   Fundamental.end < as_of)
            .order_by(Fundamental.filed.desc())
            .limit(1))
    row = (await s.execute(stmt)).first()
    if row is None:
        return None
    return {"end": row.end, "filed": row.filed, "val": float(row.val)}


async def _book_contexts(as_of: str,
                         limit: int | None = None) -> list[dict]:
    """The monthly book's live positions (read-only) + per-name context:
    the position facts and the SAME predeclared thesis condition the C
    step breaks on, anchored by the latest filed fundamentals. The news
    itself is NOT pre-fetched — that is the agent's job."""
    out: list[dict] = []
    async with Session() as s:
        positions = (await s.scalars(
            select(MonthlyPosition).order_by(MonthlyPosition.ticker))).all()
        if limit:
            positions = positions[:limit]
        for p in positions:
            close = await s.scalar(
                select(Candle.close)
                .where(Candle.ticker == p.ticker,
                       func.date(Candle.timestamp) <= as_of)
                .order_by(Candle.timestamp.desc()).limit(1))
            opened = (p.opened_at.date()
                      if isinstance(p.opened_at, datetime)
                      else date.fromisoformat(str(p.opened_at)[:10]))
            days = (date.fromisoformat(as_of) - opened).days
            out.append({
                "ticker": p.ticker,
                "shares_held": float(p.shares),
                "entry": {"date": str(opened), "price": float(p.avg_cost)},
                "current": {"date": as_of,
                            "close": (float(close) if close is not None
                                      else None)},
                "pnl_pct": (round((float(close) / float(p.avg_cost) - 1.0)
                                  * 100.0, 2)
                            if close is not None else None),
                "months_held": (round(days / 30.44, 1) if days >= 0 else None),
                "stated_thesis_condition": _THESIS_BREAK_CONDITION,
                "latest_fundamentals": {
                    tag: await _latest_fund(s, p.ticker, tag, as_of)
                    for tag in _FUNDAMENTAL_TAGS},
            })
    return out


def _log(record: dict, path: Path) -> None:
    """Append one JSON line per agent run."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False,
                           separators=(",", ":")) + "\n")


async def _cmd_news(args) -> int:
    as_of = args.as_of or await _last_session_date()
    if as_of is None:
        print("no stored sessions in the DB (candles empty) — nothing to "
              "scan; aborting")
        return 1
    if not args.dry_run and not settings.searxng_mcp_url:
        # The knob-off convention: the step is DISABLED, not broken —
        # exit 0 with an explicit message (no ledger rows).
        print("SEARXNG_MCP_URL is not set — the D step is disabled "
              "(the knob-off convention)")
        return 0
    contexts = await _book_contexts(as_of, args.limit)
    generated_at = args.generated_at or datetime.now(UTC).isoformat()
    print(f"material-news scan: {len(contexts)} monthly-book name(s), "
          f"as-of {as_of}, dry_run={args.dry_run}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"news-consult-{as_of}.jsonl"
    runs: list[dict] = []
    for ctx in contexts:
        if args.dry_run:
            print(json.dumps(ctx, sort_keys=True, indent=2, allow_nan=False))
            continue
        event_id = f"D:{ctx['ticker']}:{as_of}"
        rec = await run_agent(ctx, event_id=event_id)
        rec["session_date"] = as_of
        _log(rec, log_path)
        # The ledger row — the shadow record with NO execution authority
        # (the fill columns stay NULL by design; one writer, no drift:
        # selllayer's append-only recorder).
        await _record_ledger_row(
            rec=rec, engine="monthly", session_date=as_of,
            ticker=ctx["ticker"],
            context={"position": ctx, "citations": rec["citations"],
                     "trace": rec["trace"], "temperature": rec["temperature"],
                     "reasoning_effort": rec["reasoning_effort"],
                     "tool_calls": rec["tool_calls"], "turns": rec["turns"],
                     "news_source": rec["news_source"]},
            engine_branch=ENGINE_BRANCH)
        runs.append(rec)
        print(f"{ctx['ticker']}: {rec['decision']} parsed={rec['parsed']} "
              f"fail_safe={rec['fail_safe']} tool_calls={rec['tool_calls']} "
              f"turns={rec['turns']} latency={rec['latency_ms']:.0f}ms")
    artifact = {"schema": SCHEMA_RUN, "as_of": as_of,
                "generated_at": generated_at, "dry_run": bool(args.dry_run),
                "knob_searxng_mcp_url": bool(settings.searxng_mcp_url),
                "runs": runs}
    artifact_path = out_dir / f"news-run-{as_of}.json"
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8")
    print(f"artifact: {artifact_path}")
    if not args.dry_run and runs:
        print(f"ledger rows appended: {len(runs)} (event_type D, engine "
              f"monthly, shadow — fills NULL); log: {log_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.newsagent",
        description="The type-D material-news agent: a real agent with "
                    "native function calling (the model plans the research "
                    "and decides when it has enough). The `news` command is "
                    "the nightly shadow scan — NO execution authority.")
    sub = p.add_subparsers(dest="command", required=True)
    news = sub.add_parser(
        "news", help="the nightly material-news scan: one agent run per "
                     "monthly-book name (shadow; the ledger rows carry the "
                     "hold-to-rebuild counterfactual)")
    news.add_argument("--as-of", default=None,
                      help="decision session YYYY-MM-DD (default: the DB's "
                           "last stored session)")
    news.add_argument("--limit", type=int, default=None,
                      help="cap on the number of names (a smoke-run "
                           "convenience; NOT an agent limit)")
    news.add_argument("--dry-run", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="contexts only, ZERO consults (default); "
                           "--no-dry-run runs the agent")
    news.add_argument("--out", default="news-out",
                      help="output directory for the JSONL consult log and "
                           "the run artifact")
    news.add_argument("--generated-at", default=None,
                      help="caller-supplied ISO timestamp recorded in the "
                           "artifact (default: wall-clock UTC)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "news":
        return asyncio.run(_cmd_news(args))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
