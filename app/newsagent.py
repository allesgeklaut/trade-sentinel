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
the design note. Nightly cadence: one run per NAME in the UNION of both
paper books (monthly + daily-core — one context per name, so a name held
in both books is consulted ONCE), right after the type-C step
(``monthly.run_monthly_cycle`` calls ``run_news_scan``; the
``python -m app.newsagent news`` CLI is a thin wrapper over the same
scan core).

Beside the nightly scan there is the MANUAL consult — ``run_manual_consult``
(one name, the UI/API trigger) and the CLI's ``--fresh`` mode. Manual looks
carry the 4-part event_id ``D:{ticker}:{as_of}:m`` (the ``:m`` marker): they
are NEVER deduped (against nightly rows or against themselves — every manual
run consults again) and the 4-part form keeps them OUT of the nightly
≥6-month sample, which counts only the exact 3-part ``D:{ticker}:{as_of}``
rows the nightly scan writes.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import func, select, text

from .config import settings
from .db import (Candle, DailyCorePosition, Fundamental, LlmSellDecision,
                 MonthlyPosition, Session)
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
# The MANUAL-look marker: manual consults (run_manual_consult / the CLI's
# --fresh) use the 4-part event_id ``D:{ticker}:{as_of}:m`` — never deduped
# (every manual run consults again) and kept out of the nightly ≥6-month
# sample, which counts only the exact 3-part ``D:{ticker}:{as_of}`` rows.
MANUAL_EVENT_SUFFIX = "m"


def _event_id(ticker: str, as_of: str, *, fresh: bool = False) -> str:
    """The event_id for one consult: the nightly 3-part
    ``D:{ticker}:{as_of}`` (deduped per night) or the manual 4-part
    ``D:{ticker}:{as_of}:m`` (never deduped, never in the nightly sample)."""
    base = f"D:{ticker}:{as_of}"
    return f"{base}:{MANUAL_EVENT_SUFFIX}" if fresh else base
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
                for call, outcome in zip(calls, outcomes, strict=False):
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
# The nightly driver — one agent run per name in the UNION of both books
# ---------------------------------------------------------------------------


async def _notify_progress(on_progress, payload: dict) -> None:
    """Invoke the optional ``on_progress`` callback with one name's progress
    payload. A callback exception NEVER breaks the scan — progress is
    observation only (the default ``None`` keeps the scan byte-identical
    to the pre-callback behavior)."""
    if on_progress is None:
        return
    try:
        result = on_progress(payload)
        if inspect.isawaitable(result):
            await result
    except Exception as e:  # noqa: BLE001 — observation must never break the scan
        logger.warning("news scan: on_progress callback failed (%s: %s) — ignored",
                       type(e).__name__, e)

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
    """The UNION of both paper books' live positions (read-only) +
    per-name context: the position facts and the SAME predeclared thesis
    condition the C step breaks on, anchored by the latest filed
    fundamentals. The news itself is NOT pre-fetched — that is the
    agent's job.

    The union is MonthlyPosition + DailyCorePosition (identical shape:
    ticker, shares, avg_cost, opened_at). ONE context per NAME: a name
    held in BOTH books yields a single context (never two), so a name
    held in both books is consulted ONCE per night — the judgment is
    name-level and the stated thesis condition is book-agnostic. The
    context carries the attribution: ``books`` lists the books holding
    the name (sorted, only the books that hold it) and ``positions`` the
    per-book detail (shares_held, entry, pnl_pct, months_held — the same
    computations as the flat fields, per position). The FLAT legacy
    fields (ticker, shares_held, entry, current, pnl_pct, months_held,
    stated_thesis_condition, latest_fundamentals) stay byte-compatible
    with the pre-union shape and are computed from the PRIMARY position:
    monthly if the name is held there, else daily_core. ``limit`` caps
    the union list ordered by ticker (a smoke-run convenience)."""
    out: list[dict] = []
    async with Session() as s:
        monthly = (await s.scalars(
            select(MonthlyPosition).order_by(MonthlyPosition.ticker))).all()
        daily = (await s.scalars(
            select(DailyCorePosition).order_by(DailyCorePosition.ticker))).all()
        books_by_ticker: dict[str, dict[str, object]] = {}
        for p in daily:
            books_by_ticker.setdefault(p.ticker, {})["daily_core"] = p
        for p in monthly:
            books_by_ticker.setdefault(p.ticker, {})["monthly"] = p
        for ticker in sorted(books_by_ticker):
            if limit and len(out) >= limit:
                break
            books = books_by_ticker[ticker]
            close = await s.scalar(
                select(Candle.close)
                .where(Candle.ticker == ticker,
                       func.date(Candle.timestamp) <= as_of)
                .order_by(Candle.timestamp.desc()).limit(1))
            positions: dict[str, dict] = {}
            for book in sorted(books):
                p = books[book]
                opened = (p.opened_at.date()
                          if isinstance(p.opened_at, datetime)
                          else date.fromisoformat(str(p.opened_at)[:10]))
                days = (date.fromisoformat(as_of) - opened).days
                positions[book] = {
                    "shares_held": float(p.shares),
                    "entry": {"date": str(opened), "price": float(p.avg_cost)},
                    "pnl_pct": (round((float(close) / float(p.avg_cost) - 1.0)
                                      * 100.0, 2)
                                if close is not None else None),
                    "months_held": (round(days / 30.44, 1)
                                    if days >= 0 else None),
                }
            # The PRIMARY position (monthly if held there, else daily_core)
            # feeds the flat legacy fields — byte-compatible with the
            # pre-union monthly-only shape.
            primary = positions["monthly"] if "monthly" in positions \
                else positions["daily_core"]
            out.append({
                "ticker": ticker,
                "books": sorted(books),
                "positions": positions,
                "shares_held": primary["shares_held"],
                "entry": dict(primary["entry"]),
                "current": {"date": as_of,
                            "close": (float(close) if close is not None
                                      else None)},
                "pnl_pct": primary["pnl_pct"],
                "months_held": primary["months_held"],
                "stated_thesis_condition": _THESIS_BREAK_CONDITION,
                "latest_fundamentals": {
                    tag: await _latest_fund(s, ticker, tag, as_of)
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


async def run_news_scan(*, as_of: str | None = None, limit: int | None = None,
                        out_dir: str = "news-out", dry_run: bool = False,
                        generated_at: str | None = None, transport=None,
                        mcp=None, fresh: bool = False,
                        on_progress=None) -> dict:
    """The nightly type-D scan core (extracted from the ``news`` CLI command,
    which stays a thin wrapper; wired into ``monthly.run_monthly_cycle``):
    one agent run per NAME in the UNION of both paper books (monthly +
    daily-core — ``_book_contexts`` returns ONE context per name, so a name
    held in both books is consulted ONCE; the dedupe below stays
    name-level ``D:{ticker}:{as_of}``, which IS the not-twice guarantee)
    for the session ``as_of`` (default: the DB's last stored candle
    session), recorded as shadow ledger rows (event_type "D", engine
    "monthly" — the fill columns stay NULL forever in the shadow phase,
    NO execution authority). The scan covers the union of both books;
    the ledger ``engine`` stays "monthly" (the scan is driven by the
    monthly cycle; the context's ``books`` carries the attribution).

    Semantics (unchanged from the CLI driver):

    * no stored sessions -> status "no-sessions", nothing written;
    * knob off (``not dry_run and not settings.searxng_mcp_url``) ->
      status "disabled": the knob-off convention — DISABLED, not broken,
      zero DB writes, no error;
    * scan-time dedupe (mirrors type C, selllayer.py): a CONSULTED row
      (parsed AND NOT fail-safe) with event_id ``D:{ticker}:{as_of}``
      silences the name for that night (counted in ``n_deduped``); a
      fail-safe row never silences — a rerun after a fail-safe consults
      again;
    * ``fresh=True`` (the CLI's ``--fresh``, the MANUAL mode): the event_id
      becomes the 4-part ``D:{ticker}:{as_of}:m`` and the dedupe check is
      SKIPPED — a manual look is never deduped against nightly rows or
      against itself (every ``--fresh`` run consults again), and the
      4-part form keeps manual looks out of the nightly ≥6-month sample,
      which counts only the exact 3-part nightly ids. The nightly path
      (``fresh=False``, as called by ``monthly.run_monthly_cycle``) is
      UNCHANGED;
    * per-name error isolation: each name's (run_agent + _log +
      _record_ledger_row) runs in its own try/except — one name's failure
      is logged and collected, the loop continues, the scan NEVER raises;
    * the artifact writes (the JSONL consult log + the run artifact) are
      best-effort: a failure warns and continues (the ledger row is the
      source of truth);
    * ``on_progress`` (optional, default None = the pre-callback behavior,
      byte-identical results): called after EACH name with
      ``{"ticker", "n_done", "n_total", "decision", "fail_safe",
      "event_id"}`` (plus ``"error"`` for a name whose pipeline failed;
      ``decision``/``fail_safe`` are None for a deduped or failed name).
      A callback exception NEVER breaks the scan (see _notify_progress) —
      it feeds the background full-scan route's status state.

    ``transport`` / ``mcp`` are the same injection points ``run_agent``
    already has, passed through per name, so DB-level tests reuse the
    existing fakes. Returns the run summary: status ("ok" | "disabled" |
    "no-sessions"), as_of, n_contexts, n_ledger_rows, n_deduped, n_errors,
    errors, log_path, artifact_path (plus the CLI's contexts/runs payloads).
    """
    as_of = as_of or await _last_session_date()
    base = {"as_of": as_of, "n_contexts": 0, "n_ledger_rows": 0,
            "n_deduped": 0, "n_errors": 0, "errors": [],
            "log_path": None, "artifact_path": None,
            "contexts": [], "runs": []}
    if as_of is None:
        return {"status": "no-sessions", **base}
    if not dry_run and not settings.searxng_mcp_url:
        # The knob-off convention: the step is DISABLED, not broken —
        # zero ledger rows, no error.
        logger.info("news scan: SEARXNG_MCP_URL is not set — the D step is "
                    "disabled (the knob-off convention)")
        return {"status": "disabled", **base}
    contexts = await _book_contexts(as_of, limit)
    generated_at = generated_at or datetime.now(UTC).isoformat()
    out_dir = Path(out_dir)
    log_path = out_dir / f"news-consult-{as_of}.jsonl"
    artifact_path = out_dir / f"news-run-{as_of}.json"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # noqa: BLE001 — the artifacts are best-effort
        logger.warning("news scan: cannot create out dir %s (%s: %s) — "
                       "artifact writes skipped, the ledger rows stand",
                       out_dir, type(e).__name__, e)
    runs: list[dict] = []
    errors: list[dict] = []
    n_deduped = 0
    n_total = len(contexts)
    for n_done, ctx in enumerate(contexts, start=1):
        if dry_run:
            continue
        ticker = ctx["ticker"]
        event_id = _event_id(ticker, as_of, fresh=fresh)
        # Scan-time dedupe (mirrors type C): only a CONSULTED row silences
        # the name — a transient fail-safe row never does. MANUAL (--fresh)
        # looks are NEVER deduped: every --fresh run consults again. The
        # dedupe key is NAME-level (D:{ticker}:{as_of}) over the union — a
        # name held in both books is consulted at most once per night.
        if not fresh:
            async with Session() as s:
                consulted = await s.scalar(
                    select(func.count()).select_from(LlmSellDecision)
                    .where(LlmSellDecision.event_id == event_id,
                           LlmSellDecision.parsed.is_(True),
                           LlmSellDecision.fail_safe.is_(False)))
            if consulted:
                n_deduped += 1
                logger.info("news scan: %s already consulted for %s (%s) — "
                            "skipped", ticker, as_of, event_id)
                await _notify_progress(on_progress, {
                    "ticker": ticker, "n_done": n_done, "n_total": n_total,
                    "decision": None, "fail_safe": None, "event_id": event_id})
                continue
        try:
            rec = await run_agent(ctx, event_id=event_id, mcp=mcp,
                                  transport=transport)
            rec["session_date"] = as_of
            try:
                _log(rec, log_path)
            except Exception as e:  # noqa: BLE001 — best-effort artifact
                logger.warning("news scan: JSONL log write failed for %s "
                               "(%s: %s) — the ledger row stands",
                               event_id, type(e).__name__, e)
            # The ledger row — the shadow record with NO execution authority
            # (the fill columns stay NULL by design; one writer, no drift:
            # selllayer's append-only recorder).
            await _record_ledger_row(
                rec=rec, engine="monthly", session_date=as_of,
                ticker=ticker,
                context={"position": ctx, "citations": rec["citations"],
                         "trace": rec["trace"], "temperature": rec["temperature"],
                         "reasoning_effort": rec["reasoning_effort"],
                         "tool_calls": rec["tool_calls"], "turns": rec["turns"],
                         "news_source": rec["news_source"]},
                engine_branch=ENGINE_BRANCH)
            runs.append(rec)
            await _notify_progress(on_progress, {
                "ticker": ticker, "n_done": n_done, "n_total": n_total,
                "decision": rec["decision"], "fail_safe": bool(rec["fail_safe"]),
                "event_id": event_id})
        except Exception as e:  # noqa: BLE001 — one name never breaks the scan
            errors.append({"ticker": ticker, "event_id": event_id,
                           "error": f"{type(e).__name__}: {e}"})
            logger.error("news scan: %s (%s) failed (%s: %s) — continuing "
                         "with the remaining names", ticker, event_id,
                         type(e).__name__, e, exc_info=True)
            await _notify_progress(on_progress, {
                "ticker": ticker, "n_done": n_done, "n_total": n_total,
                "decision": None, "fail_safe": None, "event_id": event_id,
                "error": f"{type(e).__name__}: {e}"})
    artifact = {"schema": SCHEMA_RUN, "as_of": as_of,
                "generated_at": generated_at, "dry_run": bool(dry_run),
                "knob_searxng_mcp_url": bool(settings.searxng_mcp_url),
                "runs": runs}
    try:
        artifact_path.write_text(
            json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — best-effort artifact
        logger.warning("news scan: run artifact write failed for %s (%s: %s) "
                       "— the ledger rows stand", artifact_path,
                       type(e).__name__, e)
    return {"status": "ok", "as_of": as_of, "n_contexts": len(contexts),
            "n_ledger_rows": len(runs), "n_deduped": n_deduped,
            "n_errors": len(errors), "errors": errors,
            "log_path": str(log_path), "artifact_path": str(artifact_path),
            "contexts": contexts, "runs": runs}


async def run_manual_consult(ticker: str, *, as_of: str | None = None,
                             out_dir: str | None = None, transport=None,
                             mcp=None) -> dict:
    """ONE manual material-news consult for ONE name of the paper books —
    the UI/API trigger (``POST /api/news/consult``) and the heart of the
    CLI's ``--fresh`` mode.

    Fresh semantics: the event_id is the 4-part ``D:{ticker}:{as_of}:m``
    (the ``:m`` manual marker) and there is NO dedupe — every call consults
    again (never against nightly rows, never against itself), and the
    4-part form stays out of the nightly ≥6-month sample.

    The context is the same ``_book_contexts`` block the nightly scan uses
    (the UNION of both paper books, one context per name), filtered to the
    ticker (an honest error dict when the name is held in neither book —
    monthly nor daily-core); ``as_of`` defaults to
    ``_last_session_date()``. The run
    goes through the same ``run_agent``, the ledger row is appended via the
    same recorder (``selllayer._record_decision`` — shadow: the fill columns
    stay NULL), and the artifacts (the JSONL log line) are best-effort.
    NEVER raises. Returns ``{"ok": bool, "reason"?, "record"?}`` where
    ``record`` is the ``run_agent`` record (which carries the ledger
    event_id); a fail-safe record is a valid ``ok: True`` result.
    """
    try:
        ticker = str(ticker or "").strip().upper()
        if not ticker:
            return {"ok": False, "reason": "ticker must not be empty"}
        as_of = as_of or await _last_session_date()
        if as_of is None:
            return {"ok": False,
                    "reason": "no stored sessions in the DB (candles empty) — "
                              "nothing to consult"}
        if not settings.searxng_mcp_url:
            # The knob-off convention: DISABLED, not broken — zero rows.
            logger.info("manual news consult: SEARXNG_MCP_URL is not set — "
                        "the D step is disabled (the knob-off convention)")
            return {"ok": False,
                    "reason": "the D step is disabled — SEARXNG_MCP_URL is "
                              "not set"}
        contexts = await _book_contexts(as_of)
        ctx = next((c for c in contexts if c["ticker"] == ticker), None)
        if ctx is None:
            return {"ok": False,
                    "reason": f"{ticker} is not held in either paper book "
                              f"(monthly / daily-core, as of {as_of})"}
        event_id = _event_id(ticker, as_of, fresh=True)
        rec = await run_agent(ctx, event_id=event_id, mcp=mcp,
                              transport=transport)
        rec["session_date"] = as_of
        # Best-effort artifact: the same JSONL consult log the scan writes.
        out = Path(out_dir) if out_dir else Path(settings.sim_news_out_dir)
        try:
            _log(rec, out / f"news-consult-{as_of}.jsonl")
        except Exception as e:  # noqa: BLE001 — best-effort artifact
            logger.warning("manual news consult: JSONL log write failed for "
                           "%s (%s: %s) — the ledger row stands",
                           event_id, type(e).__name__, e)
        # The ledger row — the shadow record with NO execution authority
        # (the fill columns stay NULL by design; one writer, no drift:
        # selllayer's append-only recorder).
        await _record_ledger_row(
            rec=rec, engine="monthly", session_date=as_of,
            ticker=ticker,
            context={"position": ctx, "citations": rec["citations"],
                     "trace": rec["trace"], "temperature": rec["temperature"],
                     "reasoning_effort": rec["reasoning_effort"],
                     "tool_calls": rec["tool_calls"], "turns": rec["turns"],
                     "news_source": rec["news_source"]},
            engine_branch=ENGINE_BRANCH)
        return {"ok": True, "record": rec}
    except Exception as e:  # noqa: BLE001 — a manual consult never raises
        logger.warning("manual news consult %s failed (%s: %s)",
                       ticker, type(e).__name__, e)
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


async def _cmd_news(args) -> int:
    """The ``news`` CLI command — a thin wrapper over :func:`run_news_scan`
    (the scan core the nightly cycle calls)."""
    result = await run_news_scan(as_of=args.as_of, limit=args.limit,
                                 out_dir=args.out, dry_run=args.dry_run,
                                 generated_at=args.generated_at,
                                 fresh=args.fresh)
    if result["status"] == "no-sessions":
        print("no stored sessions in the DB (candles empty) — nothing to "
              "scan; aborting")
        return 1
    if result["status"] == "disabled":
        # The knob-off convention: the step is DISABLED, not broken —
        # exit 0 with an explicit message (no ledger rows).
        print("SEARXNG_MCP_URL is not set — the D step is disabled "
              "(the knob-off convention)")
        return 0
    as_of = result["as_of"]
    mode = ("fresh MANUAL scan — event_id marker :m, NO dedupe (every "
            "--fresh run consults again)" if args.fresh else "nightly scan")
    print(f"material-news scan [{mode}]: {result['n_contexts']} name(s) in the "
          f"union of both paper books (monthly + daily-core, one consult per "
          f"name), as-of {as_of}, dry_run={args.dry_run}")
    if args.dry_run:
        for ctx in result["contexts"]:
            print(json.dumps(ctx, sort_keys=True, indent=2, allow_nan=False))
    else:
        for rec in result["runs"]:
            ticker = rec["event_id"].split(":")[1]
            print(f"{ticker}: {rec['decision']} parsed={rec['parsed']} "
                  f"fail_safe={rec['fail_safe']} tool_calls={rec['tool_calls']} "
                  f"turns={rec['turns']} latency={rec['latency_ms']:.0f}ms")
            # The response itself, so a terminal run is readable end-to-end:
            # the verdict's rationale and each citation (title + URL).
            if rec.get("rationale"):
                print(f"  rationale: {rec['rationale']}")
            for c in rec.get("citations") or []:
                if isinstance(c, dict):
                    title = str(c.get("title") or c.get("source") or "citation")
                    url = str(c.get("url") or "")
                else:
                    title, url = str(c), ""
                print(f"  citation: {title} — {url}" if url
                      else f"  citation: {title}")
    if result["artifact_path"]:
        print(f"artifact: {result['artifact_path']}")
    if not args.dry_run and result["runs"]:
        print(f"ledger rows appended: {len(result['runs'])} (event_type D, engine "
              f"monthly, shadow — fills NULL); log: {result['log_path']}")
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
        "news", help="the material-news scan: one agent run per name in the "
                     "union of both paper books, monthly + daily-core (one "
                     "consult per name, never twice; shadow — the ledger rows "
                     "carry the hold-to-rebuild counterfactual). Default = the "
                     "nightly 3-part event_id D:{ticker}:{as_of} (deduped per "
                     "night); --fresh = manual consults with the 4-part :m "
                     "marker, never deduped")
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
    news.add_argument("--fresh", action="store_true",
                      help="MANUAL consult mode: each name's event_id becomes "
                           "'D:{ticker}:{as_of}:m' (the :m manual marker) — "
                           "never deduped against nightly rows or itself "
                           "(every --fresh run consults again), and the "
                           "4-part form keeps manual looks OUT of the nightly "
                           "≥6-month sample, which counts only the exact "
                           "3-part 'D:{ticker}:{as_of}' rows")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "news":
        return asyncio.run(_cmd_news(args))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
