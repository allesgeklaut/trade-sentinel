"""Phase 5C slices 5C-A/B/C: the unified reviewer's prospective consult runner,
the type-B (monthly band-drop) and type-C (intra-month thesis-break) event
adapters, the daily ``shadow`` command and the ``assess`` command (offline,
analysis-only).

NO LLM calls on history, ever — contexts are for prompt design and prospective
evaluation windows only; the first and only origin of real LLM calls is the
``consult`` runner below, invoked PROSPECTIVELY on live/forward events; the
authorized model is recorded as a constant and was verified reachable (LAN
LiteLLM proxy http://192.168.0.46:4000, OpenAI-compatible; key at
/opt/secrets/litellm.key or the LITELLM_API_KEY env — the key is NEVER stored,
logged, or printed).

Purpose
-------
``docs/sell-exit-research-2026-10-31.md`` §4b designs ONE LLM sell reviewer over
candidate exit events from every engine, sharing one context-schema family, one
prompt/versioning discipline, one fail-safe (the engine's own rule stands), and
one counterfactual-ledger pattern. Slice 5A-1 (``app/sellshadow``) built the
type-A machinery; slice 5C-A/B built the runner + type-B adapter; this slice
(5C-C) builds:

1. ``consult`` — the prospective consult runner, event-type-agnostic and the
   ONLY place an LLM call can originate in this module. Transport-injectable:
   ``transport`` is an ``async (messages, model_id, temperature, timeout_s) ->
   str`` callable; the DEFAULT transport is the real proxy call via httpx (ONE
   attempt, pinned timeout, key read AT CALL TIME, never logged). ANY failure
   (network error, non-200, timeout, empty body, unparsable response) -> the
   per-type fail-safe (CONFIRM for A/B, HOLD for C) with the error class
   recorded. Returns the ``sell-consult/1`` record; ``log_consult`` appends one
   JSON line per call (``shadow-consult-log/1``). The runner is LIBRARY-ONLY:
   no CLI subcommand ever constructs a real LLM call.
2. The type-B adapter (offline, deterministic, ZERO LLM calls):
   ``banddrop_contexts`` — every band-drop release of an M0
   (``optimize._daily_core_backtest`` at the "monthly/none" defaults, the 5B
   flat cost convention) run, schema ``banddrop-contexts/1``, per-event PIT
   context in the 5A-1 schema family plus the §4b factor fields (ROE, P/FCF,
   12-1 momentum + its percentile rank in the frame, band distance, days
   since the last fundamental filing, benchmark regime, sessions to the next
   rebalance), with the 5A-1-style self-check;
   ``banddrop_ledger`` — the DEFER-retention counterfactual ledger, schema
   ``banddrop-ledger/1``: per band-drop event, keeping the name to the NEXT
   rebalance vs the engine's actual use of the freed capital (the equal-weight
   mean of the names newly added at the same rebuild; cash yield when there are
   no additions), both legs net of the engine's one-way cost convention;
   the frozen type-B prompt ``sellshadow-b/1`` and its fail-safe parser
   ``parse_response_b`` (DEFER is parameter-free for B — retention has no
   level — which is why it splits from ``sellshadow.parse_response``).
3. The type-C adapter (offline, deterministic, ZERO LLM calls):
   ``thesisbreak_contexts`` — every thesis-break exit the M5 hook records
   (``optimize._daily_core_backtest`` at the M0 defaults WITH
   ``thesis_break_exit=True``; the predeclared condition imported from
   ``optimize._thesis_break_fundamentals``: record missing OR equity <= 0 OR
   TTM NI missing OR shares missing OR ROE = TTM NI/equity <= 0, evaluated on
   the fundamentals public as of the decision session S; F/P-FCF alone does
   NOT trigger; price-side gates NOT re-checked), schema
   ``thesisbreak-contexts/1``, per-event PIT context in the same factor-field
   family as type B plus days since the BREAKING filing and sessions to the
   next month-end rebuild, with the established self-check (held at the prior
   session; the engine marked the (ticker, fill) a thesis-break exit; the
   condition really holds at S on the public fundamentals; the fill price
   matches the ledger);
   ``thesisbreak_ledger`` — the EXIT counterfactual, schema
   ``thesisbreak-ledger/1``: per event the early exit's per-unit path (sell at
   the next open after the break, proceeds at the declared cash yield to the
   next rebuild, net of the one-way cost at the early exit) vs the engine's
   CURRENT default (hold to the next month-end rebuild fill, net of the
   one-way cost there), delta = early - default, aggregated overall + by year;
   the frozen type-C prompt ``sellshadow-c/1`` and its fail-safe parser
   ``parse_response_c`` (EXIT is parameter-free for C; fail-safe HOLD — the
   live book still waits for month-end unless a policy survives validation,
   research doc §4b).
4. The daily ``shadow`` command — on an as-of date, detect the day's CANDIDATE
   exit events from the LIVE state READ-ONLY (type A sim stop-fire candidates
   from ``sim_positions`` + candles; type B band-drop candidates at a
   month-end rebuild and type C thesis-break candidates from the live monthly
   book, ``monthly_positions`` when present, else the declared M0
   reconstruction), consult the reviewer per event unless ``--dry-run`` (the
   default), and append decisions to a JSONL log. Fail-safe everywhere; never
   raises on event-level problems.
5. The ``assess`` command — read the accumulated consult JSONL log(s), score
   each logged decision whose PREDECLARED forward window has COMPLETED (no
   mid-window peeking) with the same forward-path math as the per-type
   ledgers on realized data, and emit ``shadow-assessment/1`` with the
   reference bars from the committed ledgers as declared constants.

MEASURED PRIOR (declared in every output): type B's bar is HIGH — blanket
retention of band-dropped names is measured negative-edge (~-1%/quarter excess
vs the universe, research doc §1.2); the reviewer must find the specific subset
where retention wins.

It is ANALYSIS-ONLY and OFFLINE-ONLY:

* it never writes the database — ``reval.point_at_database`` opens the target
  read-only (SQLite ``PRAGMA query_only=ON``), and the test suite proves every
  table's row counts are unchanged;
* the contexts/ledger/shadow/assess CLI paths never call a market-data provider
  or an LLM (``shadow`` consults ONLY without ``--dry-run``, through
  ``consult``'s default transport, which the deployment's env enables); every
  engine runs from the stored candles of the ``--db`` snapshot. This slice's
  build and tests make ZERO network and ZERO LLM calls — the default transport
  is never invoked by any test or CLI path.

Determinism: outputs are sorted-keys JSON (``allow_nan=False``, NaN/inf ->
null); two runs with identical inputs and identical ``generated_at`` produce
byte-identical JSON (``shadow-run/1`` additionally excludes the ``latency_ms``
fields and the wall-clock; ``shadow-assessment/1`` excludes ``generated_at``
and ``wall_clock_s`` — declared in each payload).

CLI (offline)::

    python -m app.sellreview contexts --db sqlite+aiosqlite:////tmp/.../db \
        --universe sp500 --universes-dir ... --start 2022-01-01 --end 2024-12-31 --out DIR
    python -m app.sellreview ledger   ... (same args)
    python -m app.sellreview tcontexts ... (same args; thesis-break contexts)
    python -m app.sellreview tledger   ... (same args; thesis-break ledger)
    python -m app.sellreview shadow --db URL [--universe sp500]
        [--universes-dir D] [--as-of YYYY-MM-DD] [--dry-run|--no-dry-run]
        [--out DIR] [--max-events N]
    python -m app.sellreview assess --log PATH[,...] --db URL
        [--as-of YYYY-MM-DD] [--out DIR]

Library-first: everything is callable as plain functions from tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from datetime import datetime, UTC
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from . import optimize, reval, sellshadow
from .monthly import _qv_order
from .sellshadow import (
    CASH_YIELD_ANNUAL,
    CONTRIBUTION,
    MODEL_ID,
    PROMPT_VERSION,  # noqa: F401 — stable type-A prompt version, re-exported for the A/B/C routing
    TEMPERATURE,
    CoreWindow,
    _jsonable,
    defer_forward,
    load_core_window,
    parse_response,
    recompute_atr,
)

SCHEMA_CONSULT = "sell-consult/1"
SCHEMA_CONSULT_LOG = "shadow-consult-log/1"
SCHEMA_CONTEXTS_B = "banddrop-contexts/1"
SCHEMA_LEDGER_B = "banddrop-ledger/1"
SCHEMA_CONTEXTS_C = "thesisbreak-contexts/1"
SCHEMA_LEDGER_C = "thesisbreak-ledger/1"
SCHEMA_SHADOW_DRYRUN = "shadow-dryrun/1"
SCHEMA_SHADOW_RUN = "shadow-run/1"
SCHEMA_ASSESSMENT = "shadow-assessment/1"
PROMPT_VERSION_B = "sellshadow-b/1"
PROMPT_VERSION_C = "sellshadow-c/1"

# The authorized model/temperature travel with every consult record (imported
# from sellshadow). The proxy endpoint and key location are environment facts,
# recorded here; the key itself is read AT CALL TIME and never stored/logged.
DEFAULT_LLM_BASE_URL = "http://192.168.0.46:4000"
DEFAULT_KEY_FILE = "/opt/secrets/litellm.key"
MAX_TOKENS = 512
ERROR_TEXT_MAX = 200          # error text is truncated to this many chars
DEFAULT_TIMEOUT_S = 30.0

BENCHMARK = "URTH"
DEFAULT_START = "2015-01-02"
DEFAULT_END = "2026-10-30"

# Event types routed to the per-type parsers/fail-safes (the A/B/C parser split
# is genuine: A DEFER requires rearm_mult; B DEFER is parameter-free retention;
# C EXIT is a parameter-free early exit).
_B_EVENT_TYPES = {"B", "banddrop", "band_drop", "band-drop"}
_C_EVENT_TYPES = {"C", "thesisbreak", "thesis_break", "thesis-break"}

SYSTEM_PROMPT_B = (
    f"You are the sell-decider in a paper-trading research harness (prompt-version: {PROMPT_VERSION_B}).\n"
    "ROLE: you review ONE holding that just dropped out of the hysteresis band at the month-end "
    "rebuild. The live engine ALWAYS releases it at the rebuild fill; your answer only steers the "
    "shadow counterfactual branch.\n"
    "DECISION: reply with exactly one of CONFIRM (release the name at the rebuild) or DEFER (retain "
    "it to the next rebalance — retention has no level, so DEFER carries NO parameters) and a short "
    "rationale. CONFIRM may carry a rationale.\n"
    "OUTPUT FORMAT: a single JSON object, e.g. {\"decision\": \"DEFER\", \"rationale\": \"...\"} or "
    "{\"decision\": \"CONFIRM\", \"rationale\": \"...\"}. Nothing else.\n"
    "FAIL-SAFE: any malformed output, refusal, or missing JSON counts as CONFIRM — the engine's own "
    "rule always stands. You see point-in-time context only (decision-day information); never ask "
    "for or assume future data."
)

SYSTEM_PROMPT_C = (
    f"You are the sell-decider in a paper-trading research harness (prompt-version: {PROMPT_VERSION_C}).\n"
    "ROLE: you review ONE holding whose newly filed fundamentals BREAK its predeclared thesis "
    "condition (record missing, equity <= 0, TTM net income missing, shares missing, or "
    "ROE = TTM NI / equity <= 0), discovered intra-month. The live engine (today) WAITS for the "
    "month-end rebuild; your answer only steers the shadow counterfactual branch.\n"
    "DECISION: reply with exactly one of EXIT (sell at the next session's open — the early exit) or "
    "HOLD (wait for the month-end rebuild, the engine's current rule) and a short rationale. EXIT "
    "carries NO parameters (the early exit has no level).\n"
    "OUTPUT FORMAT: a single JSON object, e.g. {\"decision\": \"EXIT\", \"rationale\": \"...\"} or "
    "{\"decision\": \"HOLD\", \"rationale\": \"...\"}. Nothing else.\n"
    "FAIL-SAFE: any malformed output, refusal, or missing JSON counts as HOLD — the engine's current "
    "rule stands: the live book still waits for month-end unless a policy survives validation "
    "(research doc §4b). You see point-in-time context only (decision-day information); never ask "
    "for or assume future data."
)


# ---------------------------------------------------------------------------
# Small deterministic helpers (local re-derivations; the parallel-owned
# sellshadow internals are NOT imported)
# ---------------------------------------------------------------------------

def _dumps(obj) -> str:
    return json.dumps(_jsonable(obj), sort_keys=True, indent=2, allow_nan=False)


def _fail_safe_b(why: str) -> dict:
    """The type-B fail-safe: CONFIRM, parameter-free (retention has no level)."""
    return {"decision": "CONFIRM", "rationale": f"fail-safe: {why}",
            "parsed": False, "fail_safe": True}


def _fail_safe_c(why: str) -> dict:
    """The type-C fail-safe: HOLD (the engine's current rule — the live book
    waits for month-end unless a policy survives validation, research doc §4b)."""
    return {"decision": "HOLD", "rationale": f"fail-safe: {why}",
            "parsed": False, "fail_safe": True}


# ---------------------------------------------------------------------------
# 1. The prospective consult runner (the ONLY origin of real LLM calls)
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str | None:
    """Read the LLM key AT CALL TIME from LITELLM_API_KEY, else the file named
    by LITELLM_API_KEY_FILE, else /opt/secrets/litellm.key. The key is never
    stored on the module, logged, or printed — only its PRESENCE is reported."""
    key = os.environ.get("LITELLM_API_KEY")
    if key:
        return key
    key_file = os.environ.get("LITELLM_API_KEY_FILE") or DEFAULT_KEY_FILE
    try:
        text = Path(key_file).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


async def _default_transport(messages: list[dict], model_id: str, temperature: float,
                             timeout_s: float) -> str:
    """The real proxy call: ONE attempt, pinned timeout, via httpx POST to
    ``{LLM_BASE_URL or default}/v1/chat/completions``. The Authorization header
    carries the key read at call time; headers are never logged, and an httpx
    failure's text never contains the key (assert-safe)."""
    import httpx

    key = _resolve_api_key()
    if not key:
        raise RuntimeError(
            "no LLM API key available (LITELLM_API_KEY / LITELLM_API_KEY_FILE / "
            f"{DEFAULT_KEY_FILE})")
    base = os.environ.get("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL
    payload = {"model": model_id, "messages": messages,
               "temperature": temperature, "max_tokens": MAX_TOKENS}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(f"{base}/v1/chat/completions", json=payload,
                                 headers={"Authorization": f"Bearer {key}"})
    if resp.status_code != 200:
        raise RuntimeError(f"LLM proxy returned HTTP {resp.status_code}")
    try:
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"malformed LLM proxy response body: {type(e).__name__}") from e
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("empty LLM response content")
    return text


# The per-type parser map: A -> sellshadow.parse_response (DEFER requires
# rearm_mult), B -> parse_response_b (parameter-free DEFER retention),
# C -> parse_response_c (parameter-free EXIT).
_PARSERS = {
    "A": parse_response,
    "B": lambda text: parse_response_b(text),   # noqa: E731 — keyed map below
    "C": lambda text: parse_response_c(text),   # noqa: E731 — defined below
}


def _parser_for(event_type: str):
    """The event-type-appropriate fail-safe parser (the A/B/C split is genuine:
    A DEFER requires rearm_mult; B DEFER is parameter-free retention; C EXIT is
    a parameter-free early exit)."""
    et = str(event_type)
    if et in _C_EVENT_TYPES:
        return _PARSERS["C"]
    if et in _B_EVENT_TYPES:
        return _PARSERS["B"]
    return _PARSERS["A"]


def _fail_safe_for(event_type: str, why: str) -> dict:
    """The per-type fail-safe on consult failure: CONFIRM for A/B (the engine's
    own rule stands), HOLD for C (the live book waits for month-end)."""
    if str(event_type) in _C_EVENT_TYPES:
        return _fail_safe_c(why)
    return _fail_safe_b(why)


async def consult(event_id: str, event_type: str, prompt_version: str,
                  messages: list[dict], *, transport=None,
                  timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Consult the unified sell reviewer on ONE prospective candidate exit
    event. This is the only function in the module that can originate a real
    LLM call, and only when ``transport`` is left at its default (the proxy);
    tests and CLI paths always inject a transport and stay offline.

    ONE attempt, pinned timeout. ANY failure — transport exception, non-200,
    timeout, empty body, malformed/refused/unparsable response — returns the
    per-type fail-safe (CONFIRM for A/B, HOLD for C) with the error recorded
    (``str(e)`` truncated to ``ERROR_TEXT_MAX`` chars; the error text of an
    httpx failure never contains the key, and headers are never logged). The
    returned ``sell-consult/1`` record carries ``key_present`` (presence ONLY —
    never the key)."""
    started = time.perf_counter()
    key_present = False
    error: str | None = None
    error_class: str | None = None
    if transport is None:
        key_present = _resolve_api_key() is not None
        transport = _default_transport
    try:
        text = await transport(messages, MODEL_ID, TEMPERATURE, timeout_s)
        parsed = _parser_for(event_type)(text)
    except Exception as e:  # noqa: BLE001 — ANY failure is the fail-safe
        error_class = type(e).__name__
        error = str(e)[:ERROR_TEXT_MAX]
        parsed = _fail_safe_for(event_type, f"consult failure ({error_class})")
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return {
        "schema": SCHEMA_CONSULT,
        "event_id": event_id,
        "event_type": event_type,
        "prompt_version": prompt_version,
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "decision": parsed["decision"],
        "params": {k: parsed[k] for k in ("rearm_mult",) if parsed.get(k) is not None},
        "rationale": parsed["rationale"],
        "parsed": parsed["parsed"],
        "fail_safe": parsed["fail_safe"],
        "error": error,
        "error_class": error_class,
        "latency_ms": latency_ms,
        "key_present": key_present,
    }


def log_consult(record: dict, path: str | Path) -> None:
    """Append one JSON line per consult call (``shadow-consult-log/1`` line
    schema, sorted keys, compact separators)."""
    line = {**record, "schema": SCHEMA_CONSULT_LOG}  # the log line schema wins
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(line), sort_keys=True, allow_nan=False,
                           separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# 2. The frozen type-B prompt + its fail-safe parser (pure, versioned)
# ---------------------------------------------------------------------------

def render_prompt_b(context: dict) -> list[dict]:
    """The exact message list the LLM will receive (``sellshadow-b/1``).
    System: role, the decision, the fail-safe, the output format. User: the
    context rendered as stable, versioned text (sorted-keys JSON under a
    prompt-version header). Byte-identical for identical context."""
    user = (f"prompt-version: {PROMPT_VERSION_B}\n"
            f"context (JSON, sorted keys):\n"
            + json.dumps(_jsonable(context), sort_keys=True, indent=2, allow_nan=False))
    return [{"role": "system", "content": SYSTEM_PROMPT_B},
            {"role": "user", "content": user}]


def parse_response_b(text: str) -> dict:
    """The type-B fail-safe parser. Accepts exactly one of CONFIRM or DEFER;
    DEFER is PARAMETER-FREE (retention to the next rebalance has no level —
    unlike type A's rearm_mult, which is why this splits from
    ``sellshadow.parse_response``). Malformed output, refusal, missing JSON,
    or anything unparsable -> CONFIRM (fail-safe to the engine's own rule)."""
    if not isinstance(text, str) or not text.strip():
        return _fail_safe_b("empty output")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return _fail_safe_b("no JSON object")
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return _fail_safe_b("unparsable JSON")
    if not isinstance(obj, dict):
        return _fail_safe_b("JSON is not an object")
    decision = str(obj.get("decision", "")).strip().upper()
    rationale = obj.get("rationale")
    rationale = str(rationale) if rationale is not None else ""
    if decision == "CONFIRM":
        return {"decision": "CONFIRM", "rationale": rationale,
                "parsed": True, "fail_safe": False}
    if decision == "DEFER":
        # parameter-free by design: no rearm_mult (or any other parameter) is
        # read, required, or clamped for type B
        return {"decision": "DEFER", "rationale": rationale,
                "parsed": True, "fail_safe": False}
    return _fail_safe_b(f"unknown decision {decision!r}")


def render_prompt_c(context: dict) -> list[dict]:
    """The exact message list the LLM will receive (``sellshadow-c/1``).
    System: role, the decision, the fail-safe, the output format. User: the
    context rendered as stable, versioned text (sorted-keys JSON under a
    prompt-version header). Byte-identical for identical context."""
    user = (f"prompt-version: {PROMPT_VERSION_C}\n"
            f"context (JSON, sorted keys):\n"
            + json.dumps(_jsonable(context), sort_keys=True, indent=2, allow_nan=False))
    return [{"role": "system", "content": SYSTEM_PROMPT_C},
            {"role": "user", "content": user}]


def parse_response_c(text: str) -> dict:
    """The type-C fail-safe parser. Accepts exactly one of EXIT or HOLD; EXIT
    is PARAMETER-FREE (the early exit at the next open has no level — unlike
    type A's rearm_mult and type B's retention, which is why C splits into its
    own parser). Malformed output, refusal, missing JSON, or anything
    unparsable -> HOLD (fail-safe to the engine's current rule: the live book
    waits for month-end unless a policy survives validation, research doc §4b)."""
    if not isinstance(text, str) or not text.strip():
        return _fail_safe_c("empty output")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return _fail_safe_c("no JSON object")
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return _fail_safe_c("unparsable JSON")
    if not isinstance(obj, dict):
        return _fail_safe_c("JSON is not an object")
    decision = str(obj.get("decision", "")).strip().upper()
    rationale = obj.get("rationale")
    rationale = str(rationale) if rationale is not None else ""
    if decision == "EXIT":
        # parameter-free by design: no parameter is read, required, or clamped
        return {"decision": "EXIT", "rationale": rationale,
                "parsed": True, "fail_safe": False}
    if decision == "HOLD":
        return {"decision": "HOLD", "rationale": rationale,
                "parsed": True, "fail_safe": False}
    return _fail_safe_c(f"unknown decision {decision!r}")


def render_prompt_for(event_type: str, context: dict) -> list[dict]:
    """The per-type frozen prompt for one candidate event: A -> sellshadow/1
    (sellshadow.render_prompt), B -> sellshadow-b/1, C -> sellshadow-c/1."""
    from . import sellshadow
    et = str(event_type)
    if et in _C_EVENT_TYPES:
        return render_prompt_c(context)
    if et in _B_EVENT_TYPES:
        return render_prompt_b(context)
    return sellshadow.render_prompt(context)


def prompt_version_for(event_type: str) -> str:
    """The per-type frozen prompt version string (A/B/C routing)."""
    et = str(event_type)
    if et in _C_EVENT_TYPES:
        return PROMPT_VERSION_C
    if et in _B_EVENT_TYPES:
        return PROMPT_VERSION_B
    return PROMPT_VERSION


# ---------------------------------------------------------------------------
# 3. The type-B adapter: window load + M0 run + band-drop derivation (local)
# ---------------------------------------------------------------------------

def _m0_knobs() -> dict:
    """The M0 defaults re-derived locally (the 5B M-set's baseline):
    ``dca="monthly"``, ``entry_gate="none"``, ``exit_cadence="monthly"``,
    ``crash_pause=False``, every risk overlay off — the engine's documented
    reduction to the monthly qv-mom backtest."""
    return {"dca": "monthly", "entry_gate": "none", "exit_cadence": "monthly",
            "crash_pause": False, "vol_weight": None, "target_vol": None,
            "portfolio_stop_pct": None, "exposure_trend_days": None,
            "trailing_stop_pct": None}


async def _run_m0(win: CoreWindow, costs: reval.CostConfig | None = None,
                  contribution: float = CONTRIBUTION,
                  thesis_break_exit: bool = False):
    """Run ``optimize._daily_core_backtest`` at the M0 defaults over the window
    on the shared frame cache, with the 5B flat cost convention (flat one-way
    stock bps through the engine's ``cost_oneway`` wrapper). With
    ``thesis_break_exit=True`` this is the M5 run: the same M0 defaults PLUS
    the predeclared thesis-break hook (the events it records are the type-C
    candidate events)."""
    import io
    from contextlib import redirect_stdout

    costs = costs or reval.CostConfig()
    stock_rate = costs.stock_oneway_bps / 1e4
    knobs = _m0_knobs()
    with reval._settings_overrides(
            sim_daily_core_mom_variant="raw", sim_daily_core_lowvol_tilt=False,
            sim_daily_core_vol_weight=False, sim_daily_core_target_vol=0.0,
            sim_daily_core_portfolio_stop=0.0, sim_daily_core_exposure_trend=0,
            sim_daily_core_trailing_stop=0.0, sim_daily_core_basket_trend=0,
            sim_daily_core_basket_confirm=3, sim_daily_core_basket_threshold=0.0,
            sim_daily_core_basket_drawdown=0.0, sim_daily_core_basket_er_min=0.0,
            sim_daily_core_basket_good_times=False, sim_daily_core_basket_arm_sma=200,
            sim_monthly_rank_boost=0.0):
        with redirect_stdout(io.StringIO()):
            summary = await optimize._daily_core_backtest(
                win.start, win.end, win.universe["name"], contribution, False,
                knobs["entry_gate"], knobs["exit_cadence"], knobs["crash_pause"],
                knobs["dca"],
                frame_cache=win.frame_cache, close_preloaded=(win.close, win.vol),
                fund_preloaded=win.fund, open_preloaded=win.open_px,
                cost_oneway=stock_rate, with_baseline=False,
                vol_weight=knobs.get("vol_weight"),
                target_vol=knobs.get("target_vol"),
                portfolio_stop_pct=knobs.get("portfolio_stop_pct"),
                exposure_trend_days=knobs.get("exposure_trend_days"),
                trailing_stop_pct=knobs.get("trailing_stop_pct"),
                thesis_break_exit=thesis_break_exit)
    if summary is None:
        raise ValueError("daily-core M0 backtest returned no summary")
    return summary


def _prior_month_end(full_idx: pd.DatetimeIndex, first_day: pd.Timestamp) -> pd.Timestamp | None:
    """The engine's pre-window ranking month-end (optimize.py's
    ``prior_month_end`` rule, mirrored locally)."""
    from . import monthly as monthly_mod
    prev_y, prev_m = ((first_day.year - 1, 12) if first_day.month == 1
                      else (first_day.year, first_day.month - 1))
    pme_sched = pd.Timestamp(monthly_mod.month_last_trading_day(datetime(prev_y, prev_m, 1)))
    prev_idx = full_idx[full_idx <= pme_sched]
    return prev_idx[-1] if len(prev_idx) else None


def _band_days(win: CoreWindow, hold_band: int) -> list[dict]:
    """Per window session: the decision band (``order[:hold_band]`` of the
    month-end frame the engine itself used, from the shared frame cache), the
    governing month-end, and whether the day is a rebuild day (first session
    after a completed month-end). Re-derived locally from the frame cache —
    ``band=None`` marks a dead frame day (no ranking -> no release pass)."""
    from . import monthly as monthly_mod
    full_idx = pd.DatetimeIndex(win.close.index)
    lo, hi = pd.Timestamp(win.start), pd.Timestamp(win.end)
    days_all = full_idx[(full_idx >= lo) & (full_idx <= hi)]
    months = monthly_mod.completed_month_ends(win.close.index, days_all[0], days_all[-1])
    month_set = set(months)
    locs = full_idx.get_indexer(pd.DatetimeIndex(days_all))
    prior = _prior_month_end(full_idx, days_all[0])
    band_of: dict[pd.Timestamp, set | None] = {}
    for me in set(months) | ({prior} if prior is not None else set()):
        frame = win.frame_cache.get(me)
        if frame is None or not frame["eligible"].any():
            band_of[me] = None
        else:
            _elig, order = _qv_order(frame)
            band_of[me] = set(order[:hold_band])
    out = []
    for i, d in enumerate(days_all):
        li = locs[i]
        s_prev = full_idx[li - 1] if li > 0 else None
        rebuild = s_prev is not None and s_prev in month_set
        if s_prev is None:
            me = prior
        else:
            m = [m for m in months if m <= s_prev]
            me = m[-1] if m else prior
        out.append({"date": d.strftime("%Y-%m-%d"), "band": band_of.get(me),
                    "me": me.strftime("%Y-%m-%d") if me is not None else None,
                    "rebuild": rebuild})
    return out


def _days_since_filing(fund_rec: dict | None, decision_date: str) -> int | None:
    """Days from the decision day to the name's last fundamental FILING
    (max of ``filed`` <= decision day across all tags); None when the name has
    no filing at/before the decision day (declared)."""
    filed = [str(e["filed"]) for entries in (fund_rec or {}).values()
             for e in entries if e.get("filed")]
    prior = [f for f in filed if f <= decision_date]
    if not prior:
        return None
    return (pd.Timestamp(decision_date) - pd.Timestamp(max(prior))).days


def _bench_regime(bench: pd.DataFrame | None, all_days: list[str],
                  s_idx: int) -> tuple[float | None, float | None]:
    """Benchmark drawdown-from-peak (%) and 20d annualized realized vol at the
    decision day, from benchmark closes carried forward over the timeline.
    PIT only: bench data <= decision day. (Local re-derivation of the 5A-1
    helper.)"""
    if bench is None or bench.empty:
        return None, None
    bclose = dict(zip(bench["time"].tolist(), bench["close"].tolist(), strict=True))
    carried: list[float] = []
    last: float | None = None
    for d in all_days[: s_idx + 1]:
        if d in bclose:
            last = float(bclose[d])
        if last is not None:
            carried.append(last)
    if not carried:
        return None, None
    peak = max(carried)
    dd = (carried[-1] / peak - 1.0) * 100.0 if peak > 0 else None
    rets = [carried[i] / carried[i - 1] - 1.0 for i in range(1, len(carried))
            if carried[i - 1] > 0]
    vol = None
    if len(rets) >= 20:
        w = rets[-20:]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / (len(w) - 1)
        vol = math.sqrt(var) * math.sqrt(252.0) if var > 0 else 0.0
    return dd, vol


def _core_equity_series(summary, win: CoreWindow, contribution: float,
                        stock_rate: float) -> dict[str, list]:
    """Per-session account reconstruction from the engine's per-leg trade
    ledger, mirroring ``reval._run_daily_core_arm``'s EOD mark policy exactly
    (fresh forward-filled close else cost basis, same contribution days, same
    flat stock rate at fill). Local re-derivation; only the fields the type-B
    contexts need are kept (dates, equity, cash)."""
    from . import monthly as monthly_mod
    from . import price_freshness
    from .config import settings

    close = win.close
    idx_all = pd.DatetimeIndex(close.index)
    lo, hi = pd.Timestamp(win.start), pd.Timestamp(win.end)
    days = idx_all[(idx_all >= lo) & (idx_all <= hi)]
    contrib_days = optimize._contribution_days(days, {(d.year, d.month) for d in days})
    fresh = price_freshness.freshness_mask(close, None, settings.factor_price_max_age_days)
    close_val = close.ffill()

    def eod_mark(t: str, d: pd.Timestamp, cost_basis: dict[str, float]) -> float:
        try:
            is_fresh = bool(fresh.at[d, t])
        except KeyError:
            is_fresh = False
        if not is_fresh:
            return cost_basis.get(t, 0.0) or 0.0
        p = monthly_mod.px_at(close_val, d, t)
        return p if p is not None else 0.0

    trades_by_date: dict[str, list[dict]] = {}
    for t in summary.trades:
        trades_by_date.setdefault(t["date"], []).append(t)

    shares: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    cash = 0.0
    dates: list[str] = []
    equity_raw: list[float] = []
    cash_s: list[float] = []
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        if d in contrib_days:
            cash += contribution
        for tr in trades_by_date.get(ds, []):
            t, notional = tr["ticker"], float(tr["notional"])
            if tr["side"] == "BUY":
                cash -= notional * (1.0 + stock_rate)
                prev = shares.get(t, 0.0)
                shares[t] = prev + float(tr["shares"])
                cost_basis[t] = ((prev * cost_basis.get(t, 0.0) + notional) / shares[t]
                                 if prev > 0 else float(tr["price"]))
            else:
                cash += notional * (1.0 - stock_rate)
                shares.pop(t, None)
                cost_basis.pop(t, None)
        eq = cash + sum(shares[t] * eod_mark(t, d, cost_basis) for t in sorted(shares))
        dates.append(ds)
        equity_raw.append(eq)
        cash_s.append(cash)
    return {"dates": dates, "equity_raw": equity_raw, "cash": cash_s}


def _exit_px_at(win: CoreWindow, ticker: str, day: pd.Timestamp) -> tuple[float | None, str]:
    """The exit fill convention at a rebuild/window-end day: the name's strict
    stored OPEN at that day when it has a bar there, else its forward-filled
    CLOSE (declared fallback); None when neither exists."""
    try:
        op = win.open_px.at[day, ticker]
    except KeyError:
        op = None
    if op is not None and pd.notna(op):
        return float(op), "open"
    try:
        c = win.close.ffill().at[day, ticker]
    except KeyError:
        return None, "none"
    if c is not None and pd.notna(c):
        return float(c), "ffilled_close"
    return None, "none"


def _next_rebuild(days_all: list[pd.Timestamp], months: list[pd.Timestamp],
                  me: pd.Timestamp, f_idx: int) -> tuple[pd.Timestamp, int]:
    """The next rebuild day after the fill: the first session after the next
    completed month-end after ``me``; the window end when there is none."""
    later = [m for m in months if m > me]
    if later:
        me2 = later[0]
        f2 = next((d for d in days_all[f_idx + 1:] if d > me2), days_all[-1])
    else:
        f2 = days_all[-1]
    return f2, days_all.index(f2) - f_idx


def _factor_fields(frame, elig, order, tk: str, hold_band: int,
                   fund_rec: dict | None, decision_date: str) -> dict:
    """The shared per-event factor-field family (research doc §4b): ROE, P/FCF,
    12-1 momentum + its percentile rank in the eligible frame, band distance
    (1 = first name outside the hold band), eligibility flag, and days since
    the last fundamental filing. Used by BOTH the type-B and the type-C
    context extractors and by the shadow command's per-type contexts, so all
    three carry the identical field family."""
    in_frame = frame is not None and tk in frame.index
    in_elig = elig is not None and tk in elig.index
    roe = float(frame.loc[tk, "roe"]) if in_frame and pd.notna(frame.loc[tk, "roe"]) else None
    p_fcf = (float(frame.loc[tk, "p_fcf"]) if in_frame and pd.notna(frame.loc[tk, "p_fcf"])
             else None)
    mom = float(frame.loc[tk, "mom"]) if in_frame and pd.notna(frame.loc[tk, "mom"]) else None
    mom_rank = (float(elig.loc[tk, "mom_rank"]) if in_elig and pd.notna(elig.loc[tk, "mom_rank"])
                else None)
    band_dist = (order.index(tk) - hold_band + 1) if tk in order and order.index(tk) >= hold_band else None
    return {"roe": roe, "p_fcf": p_fcf, "mom_12_1": mom,
            "mom_rank_pct_in_frame": mom_rank,
            "band_distance_ranks": band_dist,  # 1 = first name outside the hold band
            "eligible_in_frame": bool(in_elig),
            "days_since_filing": _days_since_filing(fund_rec, decision_date)}


def _extract_banddrop_contexts(win: CoreWindow, bench: pd.DataFrame | None, summary,
                               start: str, end: str,
                               costs: reval.CostConfig | None = None) -> dict:
    """Extract EVERY band-drop release of the M0 run into the
    ``banddrop-contexts/1`` payload (deterministic; per-event PIT context at
    the governing month-end, decision-day information only).

    A band-drop release is a SELL trade on a rebuild day whose ticker the
    fill day's decision band (the month-end frame's top ``hold_band`` by the
    qv order, from the shared frame cache) excludes; the fill is the rebuild
    fill from the engine's trade ledger. The self-check asserts per event: the
    name was held at the prior session, the frame's band really excluded it,
    and the fill price matches the ledger."""
    from .config import settings

    costs = costs or reval.CostConfig()
    hold_band = int(settings.sim_monthly_hold_band)
    target_n = int(settings.sim_monthly_target_n)
    stock_rate = costs.stock_oneway_bps / 1e4

    full_idx = pd.DatetimeIndex(win.close.index)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    days_all = list(full_idx[(full_idx >= lo) & (full_idx <= hi)])
    day_strs = [d.strftime("%Y-%m-%d") for d in days_all]
    day_idx = {d: i for i, d in enumerate(day_strs)}
    from . import monthly as monthly_mod
    months = monthly_mod.completed_month_ends(win.close.index, days_all[0], days_all[-1])

    band_days = _band_days(win, hold_band)
    band_at = {d["date"]: d["band"] for d in band_days}
    me_at = {d["date"]: d["me"] for d in band_days}
    rebuild_at = {d["date"]: d["rebuild"] for d in band_days}

    close_val = win.close.ffill()
    close_map = {t: dict(zip(day_strs, close_val.loc[days_all, t].tolist(), strict=True))
                 for t in win.close.columns}
    eq_series = _core_equity_series(summary, win, CONTRIBUTION, stock_rate)
    eq_at = dict(zip(eq_series["dates"], eq_series["equity_raw"], strict=True))

    # walk the ledger: stint entry (date, price), shares held, holdings
    entry: dict[str, tuple[str, float]] = {}
    shares_held: dict[str, float] = {}
    events: list[dict] = []
    n_checks = 0
    n_pass = 0
    n_other_sells = 0
    for t in summary.trades:
        tk = t["ticker"]
        if t["side"] == "BUY":
            if tk not in shares_held or shares_held[tk] <= 0:
                entry[tk] = (t["date"], float(t["price"]))
            shares_held[tk] = shares_held.get(tk, 0.0) + float(t["shares"])
            continue
        F = t["date"]
        fill_price = float(t["price"])
        was_held = shares_held.get(tk, 0.0) > 0
        shares_held[tk] = shares_held.get(tk, 0.0) - float(t["shares"])
        e_date, e_px = entry.pop(tk, ("", 0.0))  # the stint's entry (read before discard)
        if shares_held[tk] <= 1e-9:
            shares_held.pop(tk, None)
        band = band_at.get(F)
        if not (rebuild_at.get(F) and band is not None and tk not in band):
            n_other_sells += 1
            continue
        me_str = me_at[F]
        me = pd.Timestamp(me_str)
        f_idx = day_idx[F]
        me_idx = day_idx[me_str]
        frame = win.frame_cache.get(me)
        elig, order = _qv_order(frame) if frame is not None else (None, [])
        # --- self-check: held at the prior session, band really excludes, fill matches
        held_prior = was_held
        band_excludes = tk not in band
        fill_matches = fill_price == float(t["price"])
        n_checks += 1
        if held_prior and band_excludes and fill_matches:
            n_pass += 1

        ff = _factor_fields(frame, elig, order, tk, hold_band, win.fund.get(tk), me_str)

        c_me = close_map.get(tk, {}).get(me_str)
        peak = None
        if e_date in day_idx:
            peak = max((c for c in (close_map.get(tk, {}).get(day_strs[i])
                                    for i in range(day_idx[e_date], me_idx + 1))
                        if c is not None), default=None)
        giveback = ((c_me / peak - 1.0) * 100.0 if (c_me is not None and peak) else None)
        pnl = ((fill_price / e_px - 1.0) * 100.0 if e_px > 0 else None)
        sessions_held = (f_idx - day_idx[e_date]) if e_date in day_idx else None

        eq_me = eq_at.get(me_str)
        sh = float(t["shares"])
        weight = (sh * c_me / eq_me) if (c_me is not None and eq_me) else None

        bench_dd, bench_vol = _bench_regime(bench, day_strs, me_idx)
        f2, sessions_to_next = _next_rebuild(days_all, months, me, f_idx)

        events.append({
            "ticker": tk,
            "decision_date": me_str,           # the governing month-end (PIT decision day)
            "fill_date": F,                    # the rebuild session (the release fill)
            "fill_price": fill_price,
            "entry_date": e_date,
            "entry_price": e_px,
            "sessions_held": sessions_held,
            "pnl_pct": pnl,
            "giveback_from_peak_pct": giveback,
            "shares": sh,
            "position_weight": weight,
            **ff,
            "bench_drawdown_pct": bench_dd,
            "bench_vol_ann_20d": bench_vol,
            "sessions_to_next_rebalance": sessions_to_next,
            "next_rebuild_date": f2.strftime("%Y-%m-%d"),
            "selfcheck": {"held_at_prior_session": held_prior,
                          "band_excludes_name": band_excludes,
                          "fill_price_matches_ledger": fill_matches},
        })

    events.sort(key=lambda e: (e["fill_date"], e["ticker"]))
    return {
        "schema": SCHEMA_CONTEXTS_B,
        "prompt_version": PROMPT_VERSION_B,
        "window": {"start": start, "end": end, "sessions": len(days_all)},
        "config": {
            "model_id": MODEL_ID, "temperature": TEMPERATURE,
            "note": "recorded constants only — this slice makes zero LLM/network calls",
            "engine": ("optimize._daily_core_backtest at the M0 defaults (dca=monthly, "
                       "entry_gate=none, exit_cadence=monthly, crash_pause=False), one "
                       "shared eligibility-frame cache (mom_variant=raw), 5B flat "
                       "one-way cost convention"),
            "target_n": target_n, "hold_band": hold_band,
            "contribution": CONTRIBUTION, "cash_yield_annual": CASH_YIELD_ANNUAL,
            "cost_rule": "flat one-way stock bps through the engine's cost_oneway wrapper",
            "measured_prior": ("type B's bar is HIGH: blanket retention of band-dropped names "
                               "is measured negative-edge (~-1%/quarter excess vs the universe, "
                               "research doc §1.2); the reviewer must find the subset where "
                               "retention wins"),
        },
        "selfcheck": {
            "events": n_checks,
            "checks_passed": n_pass,
            "match_rate": (n_pass / n_checks) if n_checks else None,
            "other_sells_not_band_drops": n_other_sells,
            "assertions": ["held at the prior session", "the frame's band excludes the name",
                           "fill price matches the ledger"],
        },
        "events": events,
    }


# ---------------------------------------------------------------------------
# 4. The type-B retention counterfactual ledger (banddrop-ledger/1)
# ---------------------------------------------------------------------------

def _retention_counterfactual(fill_price: float, retention_exit_price: float,
                              additions: list[tuple[str, float, float]],
                              sessions_window: int, rate: float) -> dict:
    """PURE per-event math of the DEFER-retention counterfactual (unit-tested
    with hand-computed constants).

    * retention leg: per-unit return of the kept name over
      [release fill -> next rebuild fill], net of the one-way cost paid at the
      NEXT rebalance: ``defer_fwd = r_hold - rate``;
    * release leg (the engine's actual use of the freed capital): the
      equal-weight mean per-unit return of the names NEWLY ADDED at the same
      rebuild over the same window, net of the one-way cost paid AT THE
      RELEASE: ``confirm_fwd = r_repl - rate``; when there are no additions the
      freed cash earns the declared cash yield over the window instead;
    * primary delta (predeclared): ``delta = defer_fwd - confirm_fwd``.
    """
    r_hold = retention_exit_price / fill_price - 1.0
    if additions:
        r_repl = sum(exit_px / buy_px - 1.0 for _t, buy_px, exit_px in additions) / len(additions)
        replacement_kind = "equal_weight_new_additions"
    else:
        r_repl = CASH_YIELD_ANNUAL * sessions_window / 252.0
        replacement_kind = "cash_yield_no_additions"
    defer_fwd = r_hold - rate
    confirm_fwd = r_repl - rate
    return {"r_hold": r_hold, "r_replacement": r_repl,
            "replacement_kind": replacement_kind,
            "oneway_rate": rate, "defer_fwd_net": defer_fwd,
            "confirm_fwd_net": confirm_fwd,
            "delta_defer_minus_confirm": defer_fwd - confirm_fwd}


def _agg_deltas(rows: list[dict], key: str) -> dict:
    ds = sorted(r[key] for r in rows if r.get(key) is not None)
    if not ds:
        return {"n": len(rows), "mean": None, "median": None, "p5": None,
                "p50": None, "p95": None, "share_positive": None}
    n = len(ds)
    mean = sum(ds) / n

    def _pct(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return ds[int(k)]
        return ds[f] + (ds[c] - ds[f]) * (k - f)

    return {"n": n, "mean": mean, "median": _pct(0.5), "p5": _pct(0.05),
            "p50": _pct(0.50), "p95": _pct(0.95),
            "share_positive": sum(1 for x in ds if x > 0) / n}


LEDGER_CAVEATS = [
    "single-name what-if: the DEFER branch holds the dropped name in isolation and "
    "ignores the portfolio-level interaction of retaining it",
    "no re-entry modeling: the next rebalance's own band may re-admit the name — the "
    "engine's real behavior — this ledger keeps it regardless",
    "both-branch cost convention: the release leg pays the engine's one-way cost at the "
    "release fill, the retention leg at the next rebalance (one cost per leg, on the "
    "event name)",
    "survivorship caveat: universes are current-membership snapshots (survivorship/selection "
    "bias in absolute terms; the ledger is a per-event measure on one universe and one window)",
]


def _build_banddrop_ledger(win: CoreWindow, summary, contexts: dict,
                           start: str, end: str,
                           costs: reval.CostConfig | None = None) -> dict:
    """The ``banddrop-ledger/1`` payload: per band-drop event the DEFER-retention
    counterfactual (keep the name to the NEXT rebalance) vs the engine's actual
    release, per the predeclared primary delta, aggregated overall and by
    calendar year, plus the universe-baseline drift of the dropped names as
    context (the §1.2 measure)."""
    costs = costs or reval.CostConfig()
    full_idx = pd.DatetimeIndex(win.close.index)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    days_all = list(full_idx[(full_idx >= lo) & (full_idx <= hi)])

    # BUY trades per day + holdings before each day, for the "newly added at
    # the same rebuild" leg
    buys_by_date: dict[str, list[dict]] = {}
    for t in summary.trades:
        if t["side"] == "BUY":
            buys_by_date.setdefault(t["date"], []).append(t)
    held: set[str] = set()
    held_before: dict[str, set[str]] = {}
    for t in summary.trades:
        d = t["date"]
        if d not in held_before:
            held_before[d] = set(held)
        if t["side"] == "BUY":
            held.add(t["ticker"])
        else:
            held.discard(t["ticker"])

    records: list[dict] = []
    for ev in contexts["events"]:
        tk = ev["ticker"]
        F = ev["fill_date"]
        f2 = pd.Timestamp(ev["next_rebuild_date"])
        sessions_window = ev["sessions_to_next_rebalance"]
        exit_px, exit_kind = _exit_px_at(win, tk, f2)
        additions = [(b["ticker"], float(b["price"]), _exit_px_at(win, b["ticker"], f2)[0])
                     for b in buys_by_date.get(F, [])
                     if b["ticker"] not in held_before.get(F, set())]
        additions = [(t, bp, xp) for t, bp, xp in additions if xp is not None]
        rate = costs.rate_for(tk, "stock")
        if exit_px is None:
            cf = {"r_hold": None, "r_replacement": None, "replacement_kind": None,
                  "oneway_rate": rate, "defer_fwd_net": None, "confirm_fwd_net": None,
                  "delta_defer_minus_confirm": None}
        else:
            cf = _retention_counterfactual(ev["fill_price"], exit_px, additions,
                                           sessions_window, rate)
        records.append({
            "ticker": tk,
            "decision_date": ev["decision_date"],
            "fill_date": F,
            "fill_price": ev["fill_price"],
            "next_rebuild_date": ev["next_rebuild_date"],
            "retention_exit_price": exit_px,
            "retention_exit_kind": exit_kind if exit_px is not None else "none",
            "sessions_retention_window": sessions_window,
            "n_additions": len(additions),
            "additions": [{"ticker": t, "buy_price": bp,
                           "fwd_return": (xp / bp - 1.0) if bp > 0 else None}
                          for t, bp, xp in additions],
            **cf,
        })

    by_year: dict[str, list[dict]] = {}
    for r in records:
        by_year.setdefault(r["decision_date"][:4], []).append(r)

    return {
        "schema": SCHEMA_LEDGER_B,
        "prompt_version": PROMPT_VERSION_B,
        "window": {"start": start, "end": end, "sessions": len(days_all)},
        "config": {
            "model_id": MODEL_ID, "temperature": TEMPERATURE,
            "note": "recorded constants only — this slice makes zero LLM/network calls",
            "primary_delta": ("the retained name's per-unit return over [release fill -> next "
                              "rebuild fill] minus the equal-weight mean return of the names "
                              "newly added at the same rebuild (cash yield over the window when "
                              "no additions), both legs net of the engine's one-way cost "
                              "convention (release leg pays at the release, retention leg at "
                              "the next rebalance)"),
            "cash_yield_annual": CASH_YIELD_ANNUAL,
            "cost_rule": "one-way CostConfig.rate_for(ticker,'stock') per leg, on the event name",
            "stock_oneway_bps": costs.stock_oneway_bps, "fx_oneway_bps": costs.fx_oneway_bps,
            "measured_prior": ("blanket retention of band-dropped names is measured negative-edge "
                               "(~-1%/quarter excess vs the universe, §1.2); the baseline_drift "
                               "block measures it per event on this window"),
        },
        "selfcheck_match_rate": contexts["selfcheck"]["match_rate"],
        "aggregate": _agg_deltas(records, "delta_defer_minus_confirm"),
        "by_year": {y: _agg_deltas(rows, "delta_defer_minus_confirm")
                    for y, rows in sorted(by_year.items())},
        "baseline_drift": {
            "method": ("the §1.2 context measure: the dropped names' raw per-unit forward "
                       "returns over [release fill -> next rebuild fill], before any "
                       "replacement comparison"),
            "aggregate": _agg_deltas(records, "r_hold"),
            "by_year": {y: _agg_deltas(rows, "r_hold")
                        for y, rows in sorted(by_year.items())},
        },
        "caveats": LEDGER_CAVEATS,
        "events": records,
    }


# ---------------------------------------------------------------------------
# 5. The type-C adapter: M5 thesis-break events -> contexts + ledger (local)
# ---------------------------------------------------------------------------

# The predeclared M5 thesis-break condition, verbatim (implemented by
# optimize._thesis_break_fundamentals, imported here so the C event definition
# can never drift from the engine's hook): the fundamentals record is missing,
# OR equity <= 0, OR TTM NI missing, OR shares missing, OR ROE = TTM NI /
# equity <= 0, evaluated on the fundamentals PUBLIC as of the decision session
# S (filing-date PIT via monthly._asof_entries; restatements replace
# originals). F/P-FCF changes alone do NOT trigger (the value leg is a rank,
# not a gate); price-side gates (mcap, dollar_vol, momentum, staleness) are
# explicitly NOT re-checked intraday.
_THESIS_BREAK_CONDITION = (
    "record missing OR equity <= 0 OR TTM NI missing OR shares missing OR "
    "ROE = TTM NI / equity <= 0 (fundamentals public as of the decision session "
    "S, filing-date PIT); F/P-FCF alone does NOT trigger; price-side gates NOT "
    "re-checked"
)


def _breaking_filing(fund_rec: dict | None, decision_date: str) -> str | None:
    """The newest ``filed`` date among the ticker's entries PUBLIC as of the
    decision day (``filed <= S`` and ``end < S`` — the same filing-date PIT
    rule ``monthly._asof_entries`` applies); None when no entry is public."""
    filed = [str(e["filed"]) for entries in (fund_rec or {}).values()
             for e in entries
             if e.get("filed") and e["filed"] <= decision_date
             and e.get("end") and e["end"] < decision_date]
    return max(filed) if filed else None


def _extract_thesisbreak_contexts(win: CoreWindow, bench: pd.DataFrame | None, summary,
                                  start: str, end: str,
                                  costs: reval.CostConfig | None = None) -> dict:
    """Extract EVERY thesis-break exit the M5 hook recorded (``summary.
    thesis_break_exits``) into the ``thesisbreak-contexts/1`` payload
    (deterministic; per-event PIT context at the decision session S — the
    session whose PUBLIC fundamentals broke the predeclared thesis condition;
    the fill is the next session F's strict open).

    The self-check asserts per event: the name was held at the prior session,
    the engine marked the (ticker, fill) a thesis-break exit, the predeclared
    condition really holds at S on the public fundamentals
    (``optimize._thesis_break_fundamentals``), and the fill price matches the
    trade ledger."""
    from .config import settings

    costs = costs or reval.CostConfig()
    hold_band = int(settings.sim_monthly_hold_band)
    target_n = int(settings.sim_monthly_target_n)
    stock_rate = costs.stock_oneway_bps / 1e4

    full_idx = pd.DatetimeIndex(win.close.index)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    days_all = list(full_idx[(full_idx >= lo) & (full_idx <= hi)])
    day_strs = [d.strftime("%Y-%m-%d") for d in days_all]
    day_idx = {d: i for i, d in enumerate(day_strs)}
    from . import monthly as monthly_mod
    months = monthly_mod.completed_month_ends(win.close.index, days_all[0], days_all[-1])

    band_days = _band_days(win, hold_band)
    me_at = {d["date"]: d["me"] for d in band_days}

    close_val = win.close.ffill()
    close_map = {t: dict(zip(day_strs, close_val.loc[days_all, t].tolist(), strict=True))
                 for t in win.close.columns}
    eq_series = _core_equity_series(summary, win, CONTRIBUTION, stock_rate)
    eq_at = dict(zip(eq_series["dates"], eq_series["equity_raw"], strict=True))

    thesis_set = {(e["ticker"], e["date"]) for e in summary.thesis_break_exits}

    # walk the ledger once: stint entry (date, price), shares held, holdings
    entry: dict[str, tuple[str, float]] = {}
    shares_held: dict[str, float] = {}
    events: list[dict] = []
    n_checks = 0
    n_pass = 0
    for t in summary.trades:
        tk = t["ticker"]
        if t["side"] == "BUY":
            if tk not in shares_held or shares_held[tk] <= 0:
                entry[tk] = (t["date"], float(t["price"]))
            shares_held[tk] = shares_held.get(tk, 0.0) + float(t["shares"])
            continue
        F = t["date"]
        if (tk, F) not in thesis_set:
            shares_held[tk] = shares_held.get(tk, 0.0) - float(t["shares"])
            if shares_held.get(tk, 0.0) <= 1e-9:
                shares_held.pop(tk, None)
            continue
        fill_price = float(t["price"])
        was_held = shares_held.get(tk, 0.0) > 0
        shares_held[tk] = shares_held.get(tk, 0.0) - float(t["shares"])
        e_date, e_px = entry.pop(tk, ("", 0.0))
        if shares_held.get(tk, 0.0) <= 1e-9:
            shares_held.pop(tk, None)

        S = next((e.get("decision_date") for e in summary.thesis_break_exits
                  if e["ticker"] == tk and e["date"] == F), None)
        s_str = S or day_strs[max(0, day_idx[F] - 1)]
        s_idx = day_idx[s_str]
        f_idx = day_idx[F]
        me_str = me_at.get(F) or me_at.get(s_str)
        me = pd.Timestamp(me_str) if me_str else None
        frame = win.frame_cache.get(me) if me is not None else None
        elig, order = _qv_order(frame) if frame is not None else (None, [])

        # --- self-check: held at the prior session, engine-marked, the
        # predeclared condition really holds at S, fill matches the ledger
        held_prior = was_held
        engine_marked = (tk, F) in thesis_set
        condition_holds = bool(
            optimize._thesis_break_fundamentals(win.fund.get(tk), pd.Timestamp(s_str)))
        fill_matches = fill_price == float(t["price"])
        n_checks += 1
        if held_prior and engine_marked and condition_holds and fill_matches:
            n_pass += 1

        c_s = close_map.get(tk, {}).get(s_str)
        peak = None
        if e_date in day_idx:
            peak = max((c for c in (close_map.get(tk, {}).get(day_strs[i])
                                    for i in range(day_idx[e_date], s_idx + 1))
                        if c is not None), default=None)
        giveback = ((c_s / peak - 1.0) * 100.0 if (c_s is not None and peak) else None)
        pnl = ((fill_price / e_px - 1.0) * 100.0 if e_px > 0 else None)
        sessions_held = (f_idx - day_idx[e_date]) if e_date in day_idx else None

        eq_s = eq_at.get(s_str)
        sh = float(t["shares"])
        weight = (sh * c_s / eq_s) if (c_s is not None and eq_s) else None

        bench_dd, bench_vol = _bench_regime(bench, day_strs, s_idx)
        f2, sessions_to_next = _next_rebuild(days_all, months, me, f_idx)
        brk_filed = _breaking_filing(win.fund.get(tk), s_str)

        events.append({
            "ticker": tk,
            "decision_date": s_str,            # S: the session whose public fundamentals broke
            "fill_date": F,                     # the next session's strict open (the early exit)
            "fill_price": fill_price,
            "entry_date": e_date,
            "entry_price": e_px,
            "sessions_held": sessions_held,
            "pnl_pct": pnl,
            "giveback_from_peak_pct": giveback,
            "shares": sh,
            "position_weight": weight,
            **_factor_fields(frame, elig, order, tk, hold_band, win.fund.get(tk), s_str),
            "days_since_breaking_filing": ((pd.Timestamp(s_str) - pd.Timestamp(brk_filed)).days
                                           if brk_filed else None),
            "breaking_filing_date": brk_filed,
            "thesis_break_condition": _THESIS_BREAK_CONDITION,
            "bench_drawdown_pct": bench_dd,
            "bench_vol_ann_20d": bench_vol,
            "sessions_to_next_rebuild": sessions_to_next,
            "next_rebuild_date": f2.strftime("%Y-%m-%d"),
            "selfcheck": {"held_at_prior_session": held_prior,
                          "engine_marked_thesis_break": engine_marked,
                          "condition_holds_at_S": condition_holds,
                          "fill_price_matches_ledger": fill_matches},
        })

    events.sort(key=lambda e: (e["fill_date"], e["ticker"]))
    return {
        "schema": SCHEMA_CONTEXTS_C,
        "prompt_version": PROMPT_VERSION_C,
        "window": {"start": start, "end": end, "sessions": len(days_all)},
        "config": {
            "model_id": MODEL_ID, "temperature": TEMPERATURE,
            "note": "recorded constants only — this slice makes zero LLM/network calls",
            "engine": ("optimize._daily_core_backtest at the M0 defaults WITH "
                       "thesis_break_exit=True (the M5 run: dca=monthly, entry_gate=none, "
                       "exit_cadence=monthly, crash_pause=False), one shared "
                       "eligibility-frame cache (mom_variant=raw), 5B flat one-way cost "
                       "convention; the C events are summary.thesis_break_exits"),
            "thesis_break_condition": _THESIS_BREAK_CONDITION,
            "target_n": target_n, "hold_band": hold_band,
            "contribution": CONTRIBUTION, "cash_yield_annual": CASH_YIELD_ANNUAL,
            "cost_rule": "flat one-way stock bps through the engine's cost_oneway wrapper",
            "measured_prior": ("the M5 hook's portfolio-level measurement: M5 vs M0 "
                               "+0.05pp/yr, 7 thesis-break events/decade (research doc "
                               "§4a); this contexts file is the per-event view of the same "
                               "events"),
        },
        "selfcheck": {
            "events": n_checks,
            "checks_passed": n_pass,
            "match_rate": (n_pass / n_checks) if n_checks else None,
            "assertions": ["held at the prior session",
                           "the engine marked the (ticker, fill) a thesis-break exit",
                           "the predeclared condition holds at S on the public fundamentals",
                           "fill price matches the ledger"],
        },
        "events": events,
    }


def _thesisbreak_counterfactual(fill_price: float, default_exit_price: float | None,
                                sessions_window: int, rate: float) -> dict:
    """PURE per-event math of the EXIT counterfactual (unit-tested with
    hand-computed constants).

    Both legs are per-unit paths over [early-exit fill F -> next rebuild fill],
    both starting from the F open (``fill_price``):

    * early-exit leg (a): sell at F's open net of the one-way cost there, and
      deploy the proceeds at the declared cash yield to the next rebuild (the
      engine would not re-buy intra-month — declared):
      ``r_early = (1 - rate) * (1 + CASH_YIELD_ANNUAL * sessions/252) - 1``;
    * default leg (b) — the engine's CURRENT rule (M0): hold the name to the
      next month-end rebuild fill, net of the one-way cost there:
      ``r_default = default_exit_price / fill_price - 1 - rate``;
    * primary delta (predeclared): ``delta = r_early - r_default``.
    """
    r_early = (1.0 - rate) * (1.0 + CASH_YIELD_ANNUAL * sessions_window / 252.0) - 1.0
    if default_exit_price is None:
        return {"r_early_exit": r_early, "r_default_hold": None,
                "oneway_rate": rate, "early_fwd_net": r_early, "default_fwd_net": None,
                "delta_exit_minus_hold": None}
    r_default = default_exit_price / fill_price - 1.0 - rate
    return {"r_early_exit": r_early, "r_default_hold": r_default,
            "oneway_rate": rate, "early_fwd_net": r_early, "default_fwd_net": r_default,
            "delta_exit_minus_hold": r_early - r_default}


LEDGER_CAVEATS_C = [
    "single-name what-if: the EXIT branch exits the broken name in isolation and "
    "ignores the portfolio-level interaction of the early exit",
    "no re-entry modeling: the next rebuild's band may re-admit or re-drop the name "
    "— the engine's real behavior — this ledger keeps the early exit regardless",
    "the engine would not re-buy intra-month (declared): the early exit's proceeds "
    "sit at the declared cash yield to the next rebuild instead of being redeployed",
    "costs convention: leg (a) pays the one-way cost at the early-exit fill, leg (b) "
    "at the next rebuild fill (one cost per leg, on the event name)",
]


def _build_thesisbreak_ledger(win: CoreWindow, summary, contexts: dict,
                              start: str, end: str,
                              costs: reval.CostConfig | None = None) -> dict:
    """The ``thesisbreak-ledger/1`` payload: per thesis-break event the EXIT
    counterfactual (early exit at the next open, proceeds at the declared cash
    yield to the next rebuild) vs the engine's CURRENT default (hold to the
    next month-end rebuild fill), per the predeclared primary delta, aggregated
    overall and by calendar year. This is the per-event view of what M5-vs-M0
    already measured at portfolio level — the header says so."""
    costs = costs or reval.CostConfig()
    full_idx = pd.DatetimeIndex(win.close.index)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    days_all = list(full_idx[(full_idx >= lo) & (full_idx <= hi)])

    records: list[dict] = []
    for ev in contexts["events"]:
        tk = ev["ticker"]
        F = ev["fill_date"]
        f2 = pd.Timestamp(ev["next_rebuild_date"])
        sessions_window = ev["sessions_to_next_rebuild"]
        exit_px, exit_kind = _exit_px_at(win, tk, f2)
        rate = costs.rate_for(tk, "stock")
        cf = _thesisbreak_counterfactual(ev["fill_price"], exit_px, sessions_window, rate)
        records.append({
            "ticker": tk,
            "decision_date": ev["decision_date"],
            "fill_date": F,
            "fill_price": ev["fill_price"],
            "next_rebuild_date": ev["next_rebuild_date"],
            "default_exit_price": exit_px,
            "default_exit_kind": exit_kind if exit_px is not None else "none",
            "sessions_window": sessions_window,
            **cf,
        })

    by_year: dict[str, list[dict]] = {}
    for r in records:
        by_year.setdefault(r["decision_date"][:4], []).append(r)

    return {
        "schema": SCHEMA_LEDGER_C,
        "prompt_version": PROMPT_VERSION_C,
        "window": {"start": start, "end": end, "sessions": len(days_all)},
        "config": {
            "model_id": MODEL_ID, "temperature": TEMPERATURE,
            "note": "recorded constants only — this slice makes zero LLM/network calls",
            "primary_delta": ("the early exit's per-unit path (sell at the next open after "
                              "the break, proceeds at the declared cash yield to the next "
                              "rebuild, net of the one-way cost at the early exit) minus the "
                              "engine's current default (hold to the next month-end rebuild "
                              "fill, net of the one-way cost there)"),
            "thesis_break_condition": _THESIS_BREAK_CONDITION,
            "cash_yield_annual": CASH_YIELD_ANNUAL,
            "cost_rule": "one-way CostConfig.rate_for(ticker,'stock') per leg, on the event name",
            "stock_oneway_bps": costs.stock_oneway_bps, "fx_oneway_bps": costs.fx_oneway_bps,
            "reconciliation": ("this ledger is the PER-EVENT view of the same comparison "
                               "M5-vs-M0 measured at PORTFOLIO level (M5 +0.05pp/yr, 7 "
                               "events/decade, research doc §4a); its aggregate must "
                               "reconcile with the measured M5-vs-M0 portfolio delta on "
                               "the same window"),
        },
        "selfcheck_match_rate": contexts["selfcheck"]["match_rate"],
        "aggregate": _agg_deltas(records, "delta_exit_minus_hold"),
        "by_year": {y: _agg_deltas(rows, "delta_exit_minus_hold")
                    for y, rows in sorted(by_year.items())},
        "baseline_drift": {
            "method": ("context measure: the default branch's raw per-unit return over "
                       "[early-exit fill -> next rebuild fill] before any cost/cash-yield "
                       "adjustment (what simply holding to month-end earned)"),
            "aggregate": _agg_deltas(records, "r_default_hold"),
            "by_year": {y: _agg_deltas(rows, "r_default_hold")
                        for y, rows in sorted(by_year.items())},
        },
        "caveats": LEDGER_CAVEATS_C,
        "events": records,
    }


# ---------------------------------------------------------------------------
# Window drivers (read-only DB) + CLI
# ---------------------------------------------------------------------------

async def _load_b_window(db_url: str, universe: str, universes_dir: str | None,
                         start: str, end: str,
                         costs: reval.CostConfig | None = None,
                         thesis_break_exit: bool = False):
    """Resolve the universe, point at the snapshot DB READ-ONLY, load the core
    window (one shared frame cache) + the benchmark, and run the M0 backtest
    (M5 - M0 + the thesis-break hook - when ``thesis_break_exit=True``)."""
    uni = reval.resolve_universe(universe, universes_dir)
    with reval.universes_override(universes_dir):
        async with reval.point_at_database(db_url):
            win = await load_core_window(uni, db_url, universes_dir, start, end)
            bench_series = await optimize._load_series(
                [BENCHMARK], start=pd.Timestamp(start).to_pydatetime())
            bench = bench_series.get(BENCHMARK)
            summary = await _run_m0(win, costs, thesis_break_exit=thesis_break_exit)
    return uni, win, bench, summary


def _base_report(db_url: str, uni: dict, start: str, end: str,
                 generated_at: str | None) -> dict:
    return {"generated_at": generated_at, "db": {"url": db_url, "read_only": True},
            "universe": uni, "window": {"start": start, "end": end}}


async def banddrop_contexts(db_url: str, universe: str, universes_dir: str | None,
                           start: str, end: str, out: str | Path | None = None,
                           generated_at: str | None = None,
                           costs: reval.CostConfig | None = None) -> dict:
    """Build the ``banddrop-contexts/1`` payload for the window (offline; the
    DB is opened read-only). When ``out`` is given, also writes
    ``banddrop-contexts.json`` there."""
    uni, win, bench, summary = await _load_b_window(db_url, universe, universes_dir,
                                                    start, end, costs)
    report = _extract_banddrop_contexts(win, bench, summary, start, end, costs)
    report.update(_base_report(db_url, uni, start, end, generated_at))
    if out is not None:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "banddrop-contexts.json").write_text(_dumps(report) + "\n", encoding="utf-8")
    return report


async def banddrop_ledger(db_url: str, universe: str, universes_dir: str | None,
                          start: str, end: str, out: str | Path | None = None,
                          generated_at: str | None = None,
                          costs: reval.CostConfig | None = None) -> dict:
    """Build the ``banddrop-ledger/1`` payload for the window (offline; the DB
    is opened read-only). When ``out`` is given, also writes
    ``banddrop-ledger.json`` there."""
    uni, win, bench, summary = await _load_b_window(db_url, universe, universes_dir,
                                                    start, end, costs)
    contexts = _extract_banddrop_contexts(win, bench, summary, start, end, costs)
    report = _build_banddrop_ledger(win, summary, contexts, start, end, costs)
    report.update(_base_report(db_url, uni, start, end, generated_at))
    if out is not None:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "banddrop-ledger.json").write_text(_dumps(report) + "\n", encoding="utf-8")
    return report


async def thesisbreak_contexts(db_url: str, universe: str, universes_dir: str | None,
                               start: str, end: str, out: str | Path | None = None,
                               generated_at: str | None = None,
                               costs: reval.CostConfig | None = None) -> dict:
    """Build the ``thesisbreak-contexts/1`` payload for the window (offline;
    the DB is opened read-only): the M5 run (M0 defaults + the thesis-break
    hook) over the window, its recorded thesis-break exits as the C events.
    When ``out`` is given, also writes ``thesisbreak-contexts.json`` there."""
    uni, win, bench, summary = await _load_b_window(db_url, universe, universes_dir,
                                                    start, end, costs,
                                                    thesis_break_exit=True)
    report = _extract_thesisbreak_contexts(win, bench, summary, start, end, costs)
    report.update(_base_report(db_url, uni, start, end, generated_at))
    if out is not None:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "thesisbreak-contexts.json").write_text(_dumps(report) + "\n", encoding="utf-8")
    return report


async def thesisbreak_ledger(db_url: str, universe: str, universes_dir: str | None,
                             start: str, end: str, out: str | Path | None = None,
                             generated_at: str | None = None,
                             costs: reval.CostConfig | None = None) -> dict:
    """Build the ``thesisbreak-ledger/1`` payload for the window (offline; the
    DB is opened read-only): the EXIT counterfactual per thesis-break event.
    When ``out`` is given, also writes ``thesisbreak-ledger.json`` there."""
    uni, win, bench, summary = await _load_b_window(db_url, universe, universes_dir,
                                                    start, end, costs,
                                                    thesis_break_exit=True)
    contexts = _extract_thesisbreak_contexts(win, bench, summary, start, end, costs)
    report = _build_thesisbreak_ledger(win, summary, contexts, start, end, costs)
    report.update(_base_report(db_url, uni, start, end, generated_at))
    if out is not None:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "thesisbreak-ledger.json").write_text(_dumps(report) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# 6. The daily shadow command (live-state detection, READ-ONLY; the only
#    subcommand that can consult the reviewer, and only without --dry-run)
# ---------------------------------------------------------------------------

SHADOW_DEFAULT_UNIVERSE = "sp500"


async def _shadow_sessions(after: str | None = None) -> tuple[str | None, str | None]:
    """The last stored session and the first stored session after ``after``
    (default: after the last stored session), from ``candles`` (read-only
    SQL). The as-of default and the type-A "next open" both come from here."""
    from sqlalchemy import text
    from .db import Session

    async with Session() as s:
        last = await s.scalar(text("SELECT MAX(timestamp) FROM candles"))
        # compare on the DATE part: the stored DATETIME strings compare
        # lexically, and "...00" < "...00.000000" would shadow the boundary
        anchor = after if after is not None else last
        nxt = await s.scalar(text(
            "SELECT MIN(timestamp) FROM candles WHERE date(timestamp) > date(:d)"),
            {"d": anchor}) if anchor is not None else None
    fmt = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")  # noqa: E731
    return (fmt(last) if last is not None else None), (fmt(nxt) if nxt is not None else None)


async def _shadow_live_state() -> dict:
    """The live state tables, READ-ONLY: the sim's positions/account
    (``sim_positions``/``sim_account``) and the monthly book's
    positions/account (``monthly_positions``/``monthly_account``). DECLARED:
    legacy sim rows carry ``stop_price=NULL`` — the standing level is derived
    from the position record + candles (the avg_cost fallback), not read from
    a stored column."""
    from sqlalchemy import select
    from .db import MonthlyAccount, MonthlyPosition, Session, SimAccount, SimPosition

    async with Session() as s:
        sims = (await s.scalars(select(SimPosition).order_by(SimPosition.ticker))).all()
        mons = (await s.scalars(select(MonthlyPosition).order_by(MonthlyPosition.ticker))).all()
        sacc = await s.scalar(select(SimAccount))
        macc = await s.scalar(select(MonthlyAccount))
    return {
        "sim_positions": [{"ticker": p.ticker, "shares": float(p.shares),
                           "avg_cost": float(p.avg_cost),
                           "opened_at": pd.Timestamp(p.opened_at).strftime("%Y-%m-%d"),
                           "stop_price": (float(p.stop_price) if p.stop_price is not None else None)}
                          for p in sims],
        "sim_cash": float(sacc.cash) if sacc is not None else None,
        "monthly_positions": [{"ticker": p.ticker, "shares": float(p.shares),
                               "avg_cost": float(p.avg_cost),
                               "opened_at": pd.Timestamp(p.opened_at).strftime("%Y-%m-%d")}
                              for p in mons],
        "monthly_cash": float(macc.cash) if macc is not None else None,
    }


def _exit_px_from_series(df: pd.DataFrame, day: str) -> tuple[float | None, str]:
    """The exit fill convention on one ticker's series: the strict stored OPEN
    at ``day`` when it has a bar there, else its forward-filled CLOSE (declared
    fallback); None when neither exists."""
    if day in df["time"].values:
        op = df.loc[df["time"] == day, "open"].iloc[0]
        if pd.notna(op):
            return float(op), "open"
    c = df["close"].ffill()
    if day in df["time"].values:
        cv = c[df["time"] == day].iloc[0]
        if pd.notna(cv):
            return float(cv), "ffilled_close"
    return None, "none"


def _series_frames(series: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide open/close frames (index=row, one column per ticker, a ``time``
    column) from per-ticker series — the shape ``_exit_px_at`` expects."""
    opens, closes = {}, {}
    for t, df in series.items():
        opens[t] = df["open"].tolist()
        closes[t] = df["close"].tolist()
    times = pd.DatetimeIndex(sorted({d for df in series.values() for d in df["time"].tolist()}))
    n = len(times)
    open_px = pd.DataFrame({t: (v + [None] * (n - len(v))) for t, v in opens.items()}, index=times)
    close = pd.DataFrame({t: (v + [None] * (n - len(v))) for t, v in closes.items()}, index=times)
    return open_px, close


def _detect_type_a(sim_positions: list[dict], series: dict[str, pd.DataFrame],
                   as_of: str, next_session: str | None) -> tuple[list[dict], list[dict]]:
    """Type-A candidates: a live sim position whose ATR trailing-stop standing
    order would fire at the NEXT session's open. The standing level is
    recomputed with the engine's exact formula (5A-1's recompute approach —
    proven to match the recorded levels): the per-bar prior-known Chandelier
    candidate at S, the position-specific monotone ratchet (max candidate over
    completed bars since the entry anchor ``opened_at``), maxed with the frozen
    initial percent stop (``stop_price`` when stored, else the DECLARED
    avg_cost fallback ``avg_cost * (1 - sim_stop_pct/100)``). The candidate =
    the next session's strict open would breach the governing level
    (``open <= level`` -> gap fill at the open)."""
    from .config import settings

    events: list[dict] = []
    skipped: list[dict] = []
    for pos in sim_positions:
        tk = pos["ticker"]
        try:
            df = series.get(tk)
            if df is None:
                skipped.append({"ticker": tk, "reason": "no candle series in the snapshot"})
                continue
            if next_session is None:
                skipped.append({"ticker": tk,
                                "reason": "no stored session after the as-of date — the "
                                          "next open cannot be determined"})
                continue
            atr14_s, atr_stop_s = recompute_atr(df)
            times = df["time"].tolist()
            if as_of not in times:
                skipped.append({"ticker": tk, "reason": f"no bar at the as-of date {as_of}"})
                continue
            s_row = times.index(as_of)
            entry = pos["opened_at"]
            pct_stop = (pos["stop_price"] if pos["stop_price"] is not None
                        else pos["avg_cost"] * (1.0 - settings.sim_stop_pct / 100.0))
            cand_s = float(atr_stop_s.iloc[s_row]) if pd.notna(atr_stop_s.iloc[s_row]) else None
            ratchet_vals = [float(v) for t, v in zip(times, atr_stop_s, strict=True)
                            if entry <= t <= as_of and pd.notna(v)]
            ratchet = max(ratchet_vals) if ratchet_vals else None
            levels = [(v, lbl) for v, lbl in ((pct_stop, "Initial stop"),
                                              (ratchet, "ATR stop ratchet"),
                                              (cand_s, "ATR stop"))
                      if v is not None]
            if not levels:
                skipped.append({"ticker": tk, "reason": "no computable standing level"})
                continue
            level, label = max(levels, key=lambda lv: lv[0])
            f_row = df[df["time"] == next_session]
            if f_row.empty:
                skipped.append({"ticker": tk,
                                "reason": f"no bar at the next session {next_session} — "
                                          "the standing order cannot fill at an open"})
                continue
            next_open = float(f_row["open"].iloc[0])
            if not (next_open <= level):
                continue  # the standing order does not fire at the next open
            close_s = float(df["close"].iloc[s_row])
            entry_row = times.index(entry) if entry in times else None
            peak = (float(df["close"].iloc[entry_row: s_row + 1].max())
                    if entry_row is not None else close_s)
            events.append({
                "ticker": tk, "decision_date": as_of, "fill_date": next_session,
                "fill_price": next_open, "fill_kind": "at_open",
                "stop_level": level, "stop_label": label,
                "atr14": (float(atr14_s.iloc[s_row])
                          if pd.notna(atr14_s.iloc[s_row]) else None),
                "atr_candidate_at_S": cand_s, "atr_ratchet_since_entry": ratchet,
                "frozen_pct_stop": pct_stop,
                "entry_date": entry, "entry_price": pos["avg_cost"],
                "shares": pos["shares"],
                "current_close": close_s,
                "pnl_pct": ((close_s / pos["avg_cost"] - 1.0) * 100.0
                            if pos["avg_cost"] > 0 else None),
                "giveback_pct": ((close_s / peak - 1.0) * 100.0 if peak else None),
                "dist_to_stop_pct": ((close_s - level) / close_s * 100.0
                                     if close_s > 0 else None),
                "stop_price_stored": pos["stop_price"] is not None,
                "derivation": ("stop_price NULL on this row: the frozen initial stop is "
                               "derived as avg_cost * (1 - sim_stop_pct/100) (declared "
                               "fallback, session fill contract rule 5)" if pos["stop_price"] is None
                               else "stop_price read from the live positions table"),
            })
        except Exception as e:  # noqa: BLE001 — event-level problems never raise
            skipped.append({"ticker": tk, "reason": f"{type(e).__name__}: {e}"})
    events.sort(key=lambda e: (e["fill_date"], e["ticker"]))
    return events, skipped


def _monthly_book_context(fund: dict, members: list[str],
                          as_of: str, holdings: list[dict], band: set | None,
                          hold_band: int, next_rebuild: tuple[str, int] | None,
                          frame, elig, order, close_map: dict, day_idx: dict,
                          equity: float | None, bench, day_strs) -> list[dict]:
    """Shared per-holding context builder for the type-B and type-C shadow
    events (same field family as the offline adapters)."""
    out = []
    for h in holdings:
        tk = h["ticker"]
        ff = _factor_fields(frame, elig, order, tk, hold_band, fund.get(tk), as_of)
        brk = _breaking_filing(fund.get(tk), as_of)
        c_s = close_map.get(tk, {}).get(as_of)
        s_idx = day_idx[as_of]
        bench_dd, bench_vol = _bench_regime(bench, day_strs, s_idx)
        out.append({
            "ticker": tk,
            "entry_date": h["opened_at"],
            "entry_price": h["avg_cost"],
            "shares": h["shares"],
            "current_close": c_s,
            "pnl_pct": ((c_s / h["avg_cost"] - 1.0) * 100.0
                        if (c_s is not None and h["avg_cost"] > 0) else None),
            "position_weight": ((h["shares"] * c_s / equity)
                                if (c_s is not None and equity) else None),
            **ff,
            "days_since_breaking_filing": ((pd.Timestamp(as_of) - pd.Timestamp(brk)).days
                                           if brk else None),
            "breaking_filing_date": brk,
            "bench_drawdown_pct": bench_dd,
            "bench_vol_ann_20d": bench_vol,
            "sessions_to_next_rebuild": (next_rebuild[1] if next_rebuild else None),
            "next_rebuild_date": (next_rebuild[0] if next_rebuild else None),
        })
    return out


async def _detect_shadow(db_url: str, universe: str, universes_dir: str | None,
                         as_of: str | None) -> dict:
    """Detect the day's CANDIDATE exit events from the LIVE state (read-only).
    Returns the detection block: per-type events with their PIT contexts +
    assess inputs, the sourcing declarations, and the per-event skips."""
    from . import fundamentals as fundamentals_mod
    from . import monthly as monthly_mod
    from .config import settings

    uni = reval.resolve_universe(universe, universes_dir)
    with reval.universes_override(universes_dir):
        async with reval.point_at_database(db_url):
            as_of = as_of or (await _shadow_sessions())[0]
            if as_of is None:
                raise ValueError("no candles in the snapshot — no as-of date")
            last_sess, next_sess = await _shadow_sessions(as_of)
            members = uni["members"]
            open_px, close, vol = await monthly_mod.load_ohlcv_frames(members)
            fund = await fundamentals_mod.load_fundamentals(members)
            bench_series = await optimize._load_series(
                [BENCHMARK], start=(pd.Timestamp(as_of) - pd.Timedelta(days=400)).to_pydatetime())
            bench = bench_series.get(BENCHMARK)
            live = await _shadow_live_state()
            hold_band = int(settings.sim_monthly_hold_band)
            target_n = int(settings.sim_monthly_target_n)

            full_idx = pd.DatetimeIndex(close.index)
            as_of_ts = pd.Timestamp(as_of)
            days_all = list(full_idx[full_idx <= as_of_ts])
            day_strs = [d.strftime("%Y-%m-%d") for d in days_all]
            day_idx = {d: i for i, d in enumerate(day_strs)}
            months = monthly_mod.completed_month_ends(close.index, days_all[0], days_all[-1])
            month_set = set(months)
            s_prev = full_idx[full_idx < as_of_ts][-1] if (full_idx < as_of_ts).any() else None
            is_rebuild = s_prev is not None and s_prev in month_set
            as_of_is_me = as_of_ts in month_set

            close_map = {t: dict(zip(day_strs, close.ffill().loc[days_all, t].tolist(),
                                     strict=True)) for t in close.columns}

            # the governing month-end frame at the as-of date (PIT)
            me_list = [m for m in months if m <= as_of_ts]
            me = me_list[-1] if me_list else None
            with reval._settings_overrides(sim_daily_core_mom_variant="raw"):
                frame = (await asyncio.to_thread(
                    monthly_mod.eligible_frame, me, close, vol, fund)
                    if me is not None else None)
            elig, order = _qv_order(frame) if frame is not None else (None, [])
            band = set(order[:hold_band]) if frame is not None and frame["eligible"].any() else None

            # next month-end rebuild after the as-of date (month-ends over the
            # FULL stored index, so a rebuild beyond the as-of is still found)
            months_full = monthly_mod.completed_month_ends(close.index, full_idx[0], full_idx[-1])
            full_day_strs = [d.strftime("%Y-%m-%d") for d in full_idx]
            full_day_idx = {d: i for i, d in enumerate(full_day_strs)}
            later = [m for m in months_full if m > as_of_ts]
            f2 = next((d for d in full_idx if d > later[0]), None) if later else None
            next_rebuild = None
            if f2 is not None:
                f2s = f2.strftime("%Y-%m-%d")
                next_rebuild = (f2s, full_day_idx[f2s] - full_day_idx[as_of])

            # ---- type A: live sim stop-fire candidates
            sim_tickers = [p["ticker"] for p in live["sim_positions"]]
            series_a = await optimize._load_series(sim_tickers) if sim_tickers else {}
            ev_a, skip_a = _detect_type_a(live["sim_positions"], series_a, as_of, next_sess)
            sim_equity = None
            if live["sim_cash"] is not None:
                pos_val = sum(p["shares"] * (close_map.get(p["ticker"], {}).get(as_of) or 0.0)
                              for p in live["sim_positions"])
                sim_equity = live["sim_cash"] + pos_val
            for ev in ev_a:
                ev["event_type"] = "A"
                ev["position_weight"] = ((ev["shares"] * ev["current_close"] / sim_equity)
                                         if (ev["current_close"] and sim_equity) else None)
                ev["bench_drawdown_pct"], ev["bench_vol_ann_20d"] = _bench_regime(
                    bench, day_strs, day_idx[as_of])
                ev["sessions_to_month_end"] = sellshadow._sessions_to_month_end(
                    day_strs, day_idx[as_of])
                ev["assess"] = {"ticker": ev["ticker"], "decision_date": as_of,
                                "fill_date": ev["fill_date"], "stop_level": ev["stop_level"],
                                "fill_price": ev["fill_price"],
                                "current_close": ev["current_close"]}

            # ---- the live monthly book's holdings (B/C source)
            if live["monthly_positions"]:
                holdings = live["monthly_positions"]
                book_source = ("live state table monthly_positions "
                               f"({len(holdings)} rows) — the real state is used")
            else:
                holdings = []
                book_source = ("monthly_positions is EMPTY in this snapshot: the M0-default "
                               "engine reconstruction to the as-of date would be the declared "
                               "method (not run here — no holdings to assess)")
            mon_equity = None
            if live["monthly_cash"] is not None:
                pos_val = sum(h["shares"] * (close_map.get(h["ticker"], {}).get(as_of) or 0.0)
                              for h in holdings)
                mon_equity = live["monthly_cash"] + pos_val

            # ---- type B: band-drop candidates at the month-end rebuild
            ev_b, skip_b = [], []
            rebuild_day = None
            if as_of_is_me and next_sess is not None:
                rebuild_day = next_sess          # the rebuild is the next session
            elif is_rebuild and me is not None:
                rebuild_day = as_of              # the as-of IS the rebuild session
            if rebuild_day is not None and band is not None:
                held_tickers = {h["ticker"] for h in holdings}
                for h in holdings:
                    tk = h["ticker"]
                    if tk in band:
                        continue
                    try:
                        ctxs = _monthly_book_context(
                            fund, members, as_of, [h], band, hold_band,
                            next_rebuild, frame, elig, order, close_map, day_idx,
                            mon_equity, bench, day_strs)
                        ctx = ctxs[0]
                        r_open, r_kind = (None, "none")
                        if pd.Timestamp(rebuild_day) in open_px.index and tk in open_px.columns:
                            op = open_px.at[pd.Timestamp(rebuild_day), tk]
                            if op is not None and pd.notna(op):
                                r_open, r_kind = float(op), "open"
                        if r_open is None and pd.Timestamp(rebuild_day) in close.index:
                            # declared fallback: the rebuild-day forward-filled close
                            cv = close.ffill().at[pd.Timestamp(rebuild_day), tk] \
                                if tk in close.columns else None
                            if cv is not None and pd.notna(cv):
                                r_open, r_kind = float(cv), "ffilled_close"
                        ctx.update({
                            "event_type": "B", "decision_date": as_of,
                            "rebuild_date": rebuild_day, "fill_date": rebuild_day,
                            "fill_price": r_open, "fill_kind": r_kind,
                            "band_excludes_name": True,
                        })
                        # the freed capital's replacement leg: the names newly added
                        # at this rebuild = top of the band not already held
                        n_held_in_band = len(held_tickers & band)
                        additions = []
                        for t in order[:hold_band]:
                            if len(additions) >= max(target_n - n_held_in_band, 0):
                                break
                            if t in held_tickers:
                                continue
                            ap = open_px.at[pd.Timestamp(rebuild_day), t] \
                                if t in open_px.columns else None
                            if ap is not None and pd.notna(ap):
                                additions.append({"ticker": t, "buy_price": float(ap)})
                        ctx["assess"] = {
                            "ticker": tk, "fill_date": rebuild_day, "fill_price": r_open,
                            "next_rebuild_date": next_rebuild[0] if next_rebuild else None,
                            "sessions_window": next_rebuild[1] if next_rebuild else None,
                            "additions": additions,
                        }
                        ev_b.append(ctx)
                    except Exception as e:  # noqa: BLE001
                        skip_b.append({"ticker": tk, "reason": f"{type(e).__name__}: {e}"})
            elif rebuild_day is None:
                skip_b.append({"reason": ("the as-of date is neither a completed month-end "
                                          "nor a rebuild session — no band-drop candidates "
                                          "today (the engine's scheduling: "
                                          "monthly.completed_month_ends + the rebuild fill "
                                          "convention)")})

            # ---- type C: thesis-break candidates among the live holdings
            ev_c, skip_c = [], []
            for h in holdings:
                tk = h["ticker"]
                try:
                    if not optimize._thesis_break_fundamentals(fund.get(tk), as_of_ts):
                        continue
                    ctxs = _monthly_book_context(
                        fund, members, as_of, [h], band, hold_band,
                        next_rebuild, frame, elig, order, close_map, day_idx,
                        mon_equity, bench, day_strs)
                    ctx = ctxs[0]
                    f_open, f_kind = (None, "none")
                    if next_sess is not None and tk in open_px.columns:
                        op = open_px.at[pd.Timestamp(next_sess), tk] \
                            if pd.Timestamp(next_sess) in open_px.index else None
                        if op is not None and pd.notna(op):
                            f_open, f_kind = float(op), "open"
                    ctx.update({
                        "event_type": "C", "decision_date": as_of,
                        "fill_date": next_sess, "fill_price": f_open, "fill_kind": f_kind,
                        "thesis_break_condition": _THESIS_BREAK_CONDITION,
                    })
                    ctx["assess"] = {
                        "ticker": tk, "decision_date": as_of, "fill_date": next_sess,
                        "fill_price": f_open,
                        "next_rebuild_date": next_rebuild[0] if next_rebuild else None,
                        "sessions_window": next_rebuild[1] if next_rebuild else None,
                    }
                    ev_c.append(ctx)
                except Exception as e:  # noqa: BLE001
                    skip_c.append({"ticker": tk, "reason": f"{type(e).__name__}: {e}"})

    for ev in ev_b + ev_c:
        ev["event_id"] = f"{ev['event_type']}:{ev['ticker']}:{as_of}"
    for ev in ev_a:
        ev["event_id"] = f"A:{ev['ticker']}:{as_of}"
    ev_a.sort(key=lambda e: e["event_id"])
    ev_b.sort(key=lambda e: e["event_id"])
    ev_c.sort(key=lambda e: e["event_id"])
    return {
        "as_of": as_of, "last_stored_session": last_sess, "next_session": next_sess,
        "universe": uni,
        "sourcing": {
            "type_A": ("live sim_positions + sim_account (state tables) + stored candles; "
                       "the standing level is RECOMPUTED from the position record + candles "
                       "(stop_price NULL on these rows -> the declared avg_cost fallback; "
                       "ATR Chandelier candidate + monotone ratchet since opened_at, "
                       "5A-1 recompute)"),
            "type_B_C_book": book_source,
            "type_B_band": ("the governing month-end's eligibility frame "
                            "(monthly.eligible_frame, mom_variant=raw) from the stored "
                            "candles/fundamentals; hold_band from settings"),
            "read_only": True,
        },
        "events": {"A": ev_a, "B": ev_b, "C": ev_c},
        "skipped": {"A": skip_a, "B": skip_b, "C": skip_c},
    }


async def run_shadow(db_url: str, universe: str = SHADOW_DEFAULT_UNIVERSE,
                     universes_dir: str | None = None, as_of: str | None = None,
                     dry_run: bool = True, out: str | Path | None = None,
                     max_events: int | None = None, generated_at: str | None = None,
                     transport=None) -> dict:
    """The daily shadow command's core: on the as-of date (default: the DB's
    last stored session) detect the day's candidate exit events from the live
    state READ-ONLY, then either (``--dry-run``, the DEFAULT) write the
    detection-only ``shadow-dryrun/1`` payload with ZERO consults, or consult
    the reviewer per event via ``consult()`` (the only path that can originate
    an LLM call, and only with the default transport), ``log_consult`` each
    decision to the JSONL log, and write ``shadow-run/1``.

    Fail-safe everywhere: a failed/missing context field still proceeds with
    what exists; a failed consult returns the per-type fail-safe; the command
    NEVER raises on event-level problems (they are logged + counted; only fatal
    setup errors — e.g. a bad DB URL — raise). ``--max-events`` caps a runaway
    day and is declared in the output."""
    started = time.perf_counter()
    det = await _detect_shadow(db_url, universe, universes_dir, as_of)
    as_of = det["as_of"]
    all_events = ([("A", e) for e in det["events"]["A"]]
                  + [("B", e) for e in det["events"]["B"]]
                  + [("C", e) for e in det["events"]["C"]])
    capped = False
    if max_events is not None and len(all_events) > max_events:
        all_events = all_events[:max_events]
        capped = True
    base = {
        "generated_at": generated_at,
        "db": {"url": db_url, "read_only": True},
        "universe": det["universe"],
        "as_of": as_of,
        "last_stored_session": det["last_stored_session"],
        "next_session": det["next_session"],
        "sourcing": det["sourcing"],
        "max_events": max_events,
        "max_events_applied": capped,
        "contamination_rule": ("NO LLM calls on history — the LLM's decisions are never "
                               "graded on historical events; its skill is measured only on "
                               "the forward log this command accumulates"),
    }
    if dry_run:
        payload = {
            **base,
            "schema": SCHEMA_SHADOW_DRYRUN,
            "dry_run": True,
            "n_candidates": {t: len(det["events"][t]) for t in ("A", "B", "C")},
            "events": [{"event_id": e["event_id"], "event_type": t,
                        "would_consult": True, "context": e}
                       for t, e in all_events],
            "skipped": det["skipped"],
            "consults_attempted": 0,
            "determinism": ("byte-identical for identical inputs and identical "
                            "generated_at; no latency/wall-clock fields exist in this "
                            "payload"),
        }
    else:
        log_path = (Path(out) / "consult-log.jsonl") if out is not None else None
        decisions = []
        errors = []
        for t, e in all_events:
            try:
                messages = render_prompt_for(t, e)
                rec = await consult(e["event_id"], t, prompt_version_for(t), messages,
                                    transport=transport)
                log_rec = {**rec, "event": {"event_type": t, "context": e,
                                            "assess": e.get("assess")}}
                if log_path is not None:
                    log_consult(log_rec, log_path)
                decisions.append({"event_id": rec["event_id"], "event_type": t,
                                  "prompt_version": rec["prompt_version"],
                                  "decision": rec["decision"], "params": rec["params"],
                                  "parsed": rec["parsed"], "fail_safe": rec["fail_safe"],
                                  "error_class": rec["error_class"],
                                  "latency_ms": rec["latency_ms"]})
            except Exception as ex:  # noqa: BLE001 — never raise on event-level problems
                errors.append({"event_id": e["event_id"], "event_type": t,
                               "error": f"{type(ex).__name__}: {str(ex)[:ERROR_TEXT_MAX]}"})
        n_by_type: dict[str, int] = {}
        n_fs: dict[str, int] = {}
        for d in decisions:
            n_by_type[d["event_type"]] = n_by_type.get(d["event_type"], 0) + 1
            if d["fail_safe"]:
                n_fs[d["event_type"]] = n_fs.get(d["event_type"], 0) + 1
        payload = {
            **base,
            "schema": SCHEMA_SHADOW_RUN,
            "dry_run": False,
            "n_candidates": {t: len(det["events"][t]) for t in ("A", "B", "C")},
            "n_decided": n_by_type,
            "n_fail_safe": n_fs,
            "decisions": decisions,
            "errors": errors + [{k: v for k, v in s.items()} for t in ("A", "B", "C")
                                for s in det["skipped"][t]],
            "log_path": str(log_path) if log_path is not None else None,
            "wall_clock_s": round(time.perf_counter() - started, 3),
            "determinism": ("byte-identical for identical inputs and identical "
                            "generated_at EXCEPT the latency_ms fields and wall_clock_s "
                            "(declared exclusions)"),
        }
    if out is not None:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        fname = "shadow-dryrun.json" if dry_run else "shadow-run.json"
        (o / fname).write_text(_dumps(payload) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 7. The assessor: score the accumulated consult log on COMPLETED windows
# ---------------------------------------------------------------------------

# The REFERENCE BARS, quoted as declared constants from the committed ledgers
# and research doc (pointers included); the reviewer must beat these.
REFERENCE_BARS = {
    "A": {
        "blanket_defer_pp_per_event": 0.0293,
        "blanket_defer_source": ("sellshadow defer-ledger/1, full window, research doc "
                                 "§5: blanket deferral +2.93pp/event (positive 9 of 12 "
                                 "years, 61% of defers never re-stopping within a quarter)"),
        "portfolio_bar": ("V1 signal exits +18.5%/yr vs BASE -2.2%/yr on the same buys "
                          "(the entire gap is the exit rule) — research doc §5"),
    },
    "B": {
        "blanket_retention_mean_pp_per_event": -0.0160,
        "blanket_retention_median_pp_per_event": -0.0103,
        "blanket_retention_share_positive": 0.442,
        "n_events_full_window": 165,
        "by_year_mean_pp": {"2016": -0.0285, "2018": -0.0260, "2020": 0.0132,
                            "2022": 0.0318, "2023": -0.1330, "2026_partial": -0.0920},
        "source": ("banddrop-ledger/1, full window (165 events, self-check 165/165), "
                   "research doc §4b: blanket retention is negative-edge; retention WINS "
                   "in bear/choppy years, LOSES when fresh additions rip"),
    },
    "C": {
        "m5_vs_m0_pp_per_year": 0.0005,
        "events_per_decade": 7,
        "source": ("the M5 hook's portfolio-level measurement (M5 = M0 + "
                   "thesis_break_exit), research doc §4a; the thesisbreak-ledger/1 is the "
                   "per-event view of the same comparison and must reconcile with it"),
    },
}

COMPLETED_WINDOW_RULE = (
    "only COMPLETED predeclared forward windows score — no mid-window peeking: "
    "type A scores when the defer branch realized a re-armed-stop breach on/before "
    "the as-of date, or when the as-of date reaches the predeclared window end "
    "(fill + 63 sessions, or the last stored session); type B and C score when the "
    "as-of date reaches the event's next month-end rebuild date"
)

SMALL_N_CAVEAT = ("deltas are not meaningful until n accumulates — small-n samples are "
                  "reported as-is; NO p-values or significance claims are computed")


def _norm_event_type(event_type: str) -> str:
    et = str(event_type)
    if et in _C_EVENT_TYPES:
        return "C"
    if et in _B_EVENT_TYPES:
        return "B"
    return "A"


def _score_event_a(ev: dict, decision: str, params: dict, series: dict[str, pd.DataFrame],
                   as_of_idx: int, n_all: int, rate: float) -> dict | None:
    """Score one type-A logged decision on realized data: the engine branch =
    the stop executed at the fill (CONFIRM); the LLM branch = CONFIRM (same) or
    DEFER (the 5A-1 predeclared defer path realized from the re-armed level
    ``level * (1 - (rearm_mult - 1) * REARM_WIDEN)``). Delta = LLM branch -
    engine branch, net of round-trip one-way costs — the defer ledger's math on
    realized prices. Scoreable only when the predeclared window completed: the
    as-of reaches fill + TIME_CAP_SESSIONS (or the last stored session), or a
    DEFER's re-armed stop realized a breach on/before the as-of date."""
    tk = ev.get("ticker")
    df = series.get(tk)
    if df is None or not all(k in ev for k in ("fill_date", "stop_level", "fill_price",
                                               "current_close")):
        return None
    times = df["time"].tolist()
    if ev["fill_date"] not in times:
        return None
    fill_idx = times.index(ev["fill_date"])
    mark = float(ev["current_close"])
    rate2 = 2.0 * rate
    confirm_fwd = ev["fill_price"] / mark - 1.0 - rate2
    cap_idx = fill_idx + sellshadow.TIME_CAP_SESSIONS
    window_end_reached = as_of_idx >= min(cap_idx, n_all - 1)
    bars = {d: {"open": (float(df["open"].iloc[i]) if pd.notna(df["open"].iloc[i]) else None),
                "low": (float(df["low"].iloc[i]) if pd.notna(df["low"].iloc[i]) else None),
                "close": float(df["close"].iloc[i])}
            for i, d in enumerate(times)}
    if decision == "DEFER":
        rm = params.get("rearm_mult")
        rearmed = (float(ev["stop_level"])
                   * (1.0 - ((float(rm) if rm is not None else 2.0) - 1.0)
                      * sellshadow.REARM_WIDEN))
        branch = defer_forward(bars, times, fill_idx, rearmed, as_of_idx + 1)
        breach_realized = branch["exit_kind"] in ("at_open", "at_level")
        if not (window_end_reached or breach_realized):
            return None
        llm_fwd = branch["exit_price"] / mark - 1.0 - rate2
        return {"delta": llm_fwd - confirm_fwd, "branch": branch["exit_kind"],
                "completed": True}
    if not window_end_reached:
        return None
    return {"delta": 0.0, "branch": "confirm", "completed": True}


def _score_event_bc(ev: dict, decision: str, etype: str, series: dict[str, pd.DataFrame],
                    open_px: pd.DataFrame, close: pd.DataFrame, as_of: str,
                    rate: float) -> dict | None:
    """Score one type-B/C logged decision whose window (to the next month-end
    rebuild) has completed: the same primary-delta math as the per-type
    ledgers on realized CONTRACT fills (strict open else ffilled close)."""
    tk = ev.get("ticker")
    df = series.get(tk)
    nr = ev.get("next_rebuild_date")
    if df is None or not nr or nr > as_of or ev.get("fill_price") is None:
        return None
    win_view = SimpleNamespace(open_px=open_px, close=close)
    if etype == "B":
        exit_px, _kind = _exit_px_at(win_view, tk, pd.Timestamp(nr))
        additions = [(a["ticker"], float(a["buy_price"]),
                      _exit_px_at(win_view, a["ticker"], pd.Timestamp(nr))[0])
                     for a in ev.get("additions", []) if a.get("buy_price")]
        additions = [(t, bp, xp) for t, bp, xp in additions if xp is not None]
        cf = _retention_counterfactual(float(ev["fill_price"]), exit_px or 0.0,
                                       additions, int(ev.get("sessions_window") or 0), rate)
        if exit_px is None:
            return None
        engine_fwd = cf["confirm_fwd_net"]          # CONFIRM = the engine's release
        llm_fwd = cf["defer_fwd_net"] if decision == "DEFER" else engine_fwd
        return {"delta": llm_fwd - engine_fwd, "branch": "defer" if decision == "DEFER"
                else "confirm", "completed": True}
    # type C
    exit_px, _kind = _exit_px_at(win_view, tk, pd.Timestamp(nr))
    cf = _thesisbreak_counterfactual(float(ev["fill_price"]), exit_px,
                                     int(ev.get("sessions_window") or 0), rate)
    if exit_px is None:
        return None
    engine_fwd = cf["default_fwd_net"]              # HOLD = the engine's current rule
    llm_fwd = cf["early_fwd_net"] if decision == "EXIT" else engine_fwd
    return {"delta": llm_fwd - engine_fwd, "branch": "exit" if decision == "EXIT"
            else "hold", "completed": True}


async def run_assess(log_paths: list[str] | str, db_url: str, as_of: str | None = None,
                     out: str | Path | None = None, generated_at: str | None = None,
                     costs: reval.CostConfig | None = None) -> dict:
    """Read the accumulated consult JSONL log(s), and for each logged decision
    whose PREDECLARED forward window has COMPLETED (see ``COMPLETED_WINDOW_RULE``)
    recompute both branches' REALIZED outcomes from the snapshot's stored
    candles (read-only) with the same forward-path math as the per-type ledgers.
    Emits ``shadow-assessment/1``: per type n_decided / n_fail_safe, the
    realized delta aggregates (LLM branch - engine branch) overall + by year,
    the reference bars, and the caveats. Malformed/incomplete log lines are
    counted + skipped — the command never crashes on the log's own
    imperfections; read-only on the DB; offline; deterministic byte-identical
    modulo the declared wall-clock fields."""
    started = time.perf_counter()
    costs = costs or reval.CostConfig()
    if isinstance(log_paths, str):
        log_paths = [p.strip() for p in log_paths.split(",") if p.strip()]
    lines_total = 0
    lines_malformed = 0
    records: list[dict] = []
    for lp in log_paths:
        p = Path(lp)
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            lines_total += 1
            try:
                obj = json.loads(ln)
            except (json.JSONDecodeError, ValueError):
                lines_malformed += 1
                continue
            if not isinstance(obj, dict) or obj.get("schema") not in (
                    SCHEMA_CONSULT_LOG, SCHEMA_CONSULT):
                lines_malformed += 1
                continue
            records.append(obj)

    async with reval.point_at_database(db_url):
        last_sess, _ = await _shadow_sessions()
        as_of = as_of or last_sess
        if as_of is None:
            raise ValueError("no candles in the snapshot — no as-of date")
        tickers = sorted({(obj.get("event") or {}).get("assess", {}).get("ticker")
                          or (obj.get("event") or {}).get("context", {}).get("ticker")
                          for obj in records} - {None})
        for obj in records:  # additions' tickers too (type B)
            for a in ((obj.get("event") or {}).get("assess") or {}).get("additions", []):
                if a.get("ticker"):
                    tickers.append(a["ticker"])
        series = await optimize._load_series(sorted(set(tickers))) if tickers else {}
        open_px, close = _series_frames(series)
        all_days = sorted({d for df in series.values() for d in df["time"].tolist()})
        as_of_idx = all_days.index(as_of) if as_of in all_days else len(all_days) - 1
        n_all = len(all_days)

    per_type: dict[str, dict] = {t: {"records": [], "n_fail_safe": 0,
                                     "n_incomplete": 0, "n_not_completed": 0}
                                 for t in ("A", "B", "C")}
    for obj in records:
        et = _norm_event_type(obj.get("event_type", "A"))
        bucket = per_type[et]
        ev_block = obj.get("event") or {}
        ev = ev_block.get("assess") or ev_block.get("context") or ev_block
        decision = str(obj.get("decision", "")).strip().upper()
        parsed = bool(obj.get("parsed", False))
        if not parsed:
            bucket["n_fail_safe"] += 1
        if not isinstance(ev, dict) or not ev.get("ticker"):
            bucket["n_incomplete"] += 1
            continue
        rate = costs.rate_for(ev["ticker"], "stock")
        try:
            if et == "A":
                scored = _score_event_a(ev, decision, obj.get("params") or {}, series,
                                        as_of_idx, n_all, rate)
            else:
                scored = _score_event_bc(ev, decision, et, series, open_px, close,
                                         as_of, rate)
        except Exception:  # noqa: BLE001 — event-level problems never raise
            bucket["n_incomplete"] += 1
            continue
        if scored is None:
            bucket["n_not_completed"] += 1
            continue
        bucket["records"].append({
            "event_id": obj.get("event_id"),
            "ticker": ev["ticker"],
            "event_date": (ev.get("fill_date") or ev.get("decision_date") or ""),
            "decision": decision, "fail_safe": not parsed,
            **scored,
        })

    types_out = {}
    for t, bucket in per_type.items():
        recs = bucket["records"]
        by_year: dict[str, list[dict]] = {}
        for r in recs:
            by_year.setdefault((r["event_date"] or "")[:4], []).append(r)
        types_out[t] = {
            "n_decided": len(recs),
            "n_fail_safe": bucket["n_fail_safe"],
            "n_incomplete_context": bucket["n_incomplete"],
            "n_window_not_completed": bucket["n_not_completed"],
            "aggregate": _agg_deltas(recs, "delta"),
            "by_year": {y: _agg_deltas(rows, "delta") for y, rows in sorted(by_year.items())},
            "events": recs,
        }
    payload = {
        "schema": SCHEMA_ASSESSMENT,
        "generated_at": generated_at,
        "db": {"url": db_url, "read_only": True},
        "as_of": as_of,
        "logs": [str(p) for p in log_paths],
        "log_lines": {"total": lines_total, "malformed_or_skipped": lines_malformed,
                      "parsed": len(records)},
        "completed_window_rule": COMPLETED_WINDOW_RULE,
        "reference_bars": REFERENCE_BARS,
        "delta_definition": ("realized LLM branch minus engine branch, net of the "
                             "one-way cost convention, per the per-type ledgers' math on "
                             "realized CONTRACT fills (strict open else ffilled close)"),
        "fail_safe_note": ("a high fail-safe rate (parsed=False) is itself a verdict on "
                           "the reviewer — fail-safes score as the engine's own branch"),
        "small_n_caveat": SMALL_N_CAVEAT,
        "types": types_out,
        "wall_clock_s": round(time.perf_counter() - started, 3),
        "determinism": ("byte-identical for identical inputs and identical generated_at "
                        "EXCEPT generated_at and wall_clock_s (declared exclusions)"),
    }
    if out is not None:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "shadow-assessment.json").write_text(_dumps(payload) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# CLI (offline ONLY — no subcommand ever constructs a real LLM call)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.sellreview",
        description="Phase 5C slices 5C-A/B/C: the unified reviewer's consult runner "
                    "(library-only) + the type-B/type-C adapters + the daily shadow and "
                    "assess commands (offline, analysis-only, read-only DB — the offline "
                    "subcommands make zero LLM/network calls)")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in (
            ("contexts", "extract band-drop contexts (banddrop-contexts/1)"),
            ("ledger", "build the retention counterfactual ledger (banddrop-ledger/1)"),
            ("tcontexts", "extract thesis-break contexts (thesisbreak-contexts/1)"),
            ("tledger", "build the early-exit counterfactual ledger (thesisbreak-ledger/1)")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--db", required=True,
                       help="database URL of the snapshot to read (opened read-only)")
        s.add_argument("--universe", required=True, help="universe name (e.g. sp500)")
        s.add_argument("--universes-dir", default=None,
                       help="extra universes dir (shadows the repo universes/ dir)")
        s.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD inclusive window start")
        s.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD inclusive window end")
        s.add_argument("--out", required=True, help="output directory for the JSON artifact")
        s.add_argument("--generated-at", default=None,
                       help="caller-supplied ISO timestamp recorded in the output "
                            "(default: wall-clock UTC)")
    sh = sub.add_parser("shadow", help="the daily shadow command: detect the day's candidate "
                                       "exit events from the live state (read-only) and, "
                                       "unless --dry-run (the default), consult the reviewer")
    sh.add_argument("--db", required=True,
                    help="database URL of the snapshot/live DB to read (opened read-only)")
    sh.add_argument("--universe", default=SHADOW_DEFAULT_UNIVERSE,
                    help="universe name for the factor frames (default: sp500)")
    sh.add_argument("--universes-dir", default=None,
                    help="extra universes dir (shadows the repo universes/ dir)")
    sh.add_argument("--as-of", default=None,
                    help="decision date YYYY-MM-DD (default: the DB's last stored session)")
    sh.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                    help="detection-only, ZERO consults (default); --no-dry-run consults "
                         "via consult()'s default transport only")
    sh.add_argument("--out", default="shadow-out",
                    help="output directory for the JSON artifact + the JSONL consult log")
    sh.add_argument("--max-events", type=int, default=None,
                    help="cap on consulted events (a runaway day); declared in the output")
    sh.add_argument("--generated-at", default=None,
                    help="caller-supplied ISO timestamp recorded in the output "
                         "(default: wall-clock UTC)")
    as_ = sub.add_parser("assess", help="score the accumulated consult JSONL log(s) on "
                                        "COMPLETED forward windows (shadow-assessment/1); "
                                        "never constructs an LLM call")
    as_.add_argument("--log", required=True,
                     help="comma-separated consult JSONL log path(s)")
    as_.add_argument("--db", required=True,
                     help="database URL of the snapshot to read (opened read-only)")
    as_.add_argument("--as-of", default=None,
                     help="assessment date YYYY-MM-DD (default: the DB's last stored session)")
    as_.add_argument("--out", default="shadow-out",
                     help="output directory for shadow-assessment.json")
    as_.add_argument("--generated-at", default=None,
                     help="caller-supplied ISO timestamp recorded in the output "
                          "(default: wall-clock UTC)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    generated_at = args.generated_at or datetime.now(UTC).isoformat()
    try:
        if args.command == "contexts":
            report = asyncio.run(banddrop_contexts(
                args.db, args.universe, args.universes_dir, args.start, args.end,
                args.out, generated_at))
            fname = "banddrop-contexts.json"
        elif args.command == "ledger":
            report = asyncio.run(banddrop_ledger(
                args.db, args.universe, args.universes_dir, args.start, args.end,
                args.out, generated_at))
            fname = "banddrop-ledger.json"
        elif args.command == "tcontexts":
            report = asyncio.run(thesisbreak_contexts(
                args.db, args.universe, args.universes_dir, args.start, args.end,
                args.out, generated_at))
            fname = "thesisbreak-contexts.json"
        elif args.command == "tledger":
            report = asyncio.run(thesisbreak_ledger(
                args.db, args.universe, args.universes_dir, args.start, args.end,
                args.out, generated_at))
            fname = "thesisbreak-ledger.json"
        elif args.command == "shadow":
            report = asyncio.run(run_shadow(
                args.db, args.universe, args.universes_dir, args.as_of,
                dry_run=args.dry_run, out=args.out, max_events=args.max_events,
                generated_at=generated_at))
            fname = "shadow-dryrun.json" if args.dry_run else "shadow-run.json"
        else:
            report = asyncio.run(run_assess(
                args.log, args.db, args.as_of, args.out, generated_at))
            fname = "shadow-assessment.json"
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    if args.command in ("contexts", "tcontexts"):
        sc = report.get("selfcheck", {})
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, "
              f"{sc.get('events')} events, self-check match rate "
              f"{sc.get('match_rate')})")
    elif args.command in ("ledger", "tledger"):
        agg = report["aggregate"]
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, "
              f"{agg['n']} events, mean delta {agg['mean']}, "
              f"share positive {agg['share_positive']})")
    elif args.command == "shadow":
        n = report["n_candidates"]
        extra = (f", decided {report['n_decided']}, fail-safe {report['n_fail_safe']}"
                 if not args.dry_run else ", detection-only (zero consults)")
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, as-of "
              f"{report['as_of']}, candidates A/B/C = {n.get('A')}/{n.get('B')}/{n.get('C')}"
              f"{extra})")
    else:
        tl = report["log_lines"]
        tsum = {t: v["n_decided"] for t, v in report["types"].items()}
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, as-of "
              f"{report['as_of']}, log lines {tl['parsed']} parsed/"
              f"{tl['malformed_or_skipped']} skipped, decided per type {tsum})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
