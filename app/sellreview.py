"""Phase 5C slice 5C-A/B: the unified reviewer's prospective consult runner and
the type-B (monthly band-drop) event adapter (offline, analysis-only).

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
prompt/versioning discipline, one fail-safe (CONFIRM = the engine's own rule
stands), and one counterfactual-ledger pattern. Slice 5A-1 (``app/sellshadow``)
built the type-A machinery; this slice builds:

1. ``consult`` — the prospective consult runner, event-type-agnostic and the
   ONLY place an LLM call can originate in this module. Transport-injectable:
   ``transport`` is an ``async (messages, model_id, temperature, timeout_s) ->
   str`` callable; the DEFAULT transport is the real proxy call via httpx (ONE
   attempt, pinned timeout, key read AT CALL TIME, never logged). ANY failure
   (network error, non-200, timeout, empty body, unparsable response) -> the
   fail-safe CONFIRM with the error class recorded. Returns the
   ``sell-consult/1`` record; ``log_consult`` appends one JSON line per call
   (``shadow-consult-log/1``). The runner is LIBRARY-ONLY: no CLI subcommand
   ever constructs a real LLM call.
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

MEASURED PRIOR (declared in every output): type B's bar is HIGH — blanket
retention of band-dropped names is measured negative-edge (~-1%/quarter excess
vs the universe, research doc §1.2); the reviewer must find the specific subset
where retention wins.

It is ANALYSIS-ONLY and OFFLINE-ONLY:

* it never writes the database — ``reval.point_at_database`` opens the target
  read-only (SQLite ``PRAGMA query_only=ON``), and the test suite proves every
  table's row counts are unchanged;
* the contexts/ledger CLI paths never call a market-data provider or an LLM;
  every engine runs from the stored candles of the ``--db`` snapshot. This
  slice's build and tests make ZERO network and ZERO LLM calls — the default
  transport is never invoked by any test or CLI path.

Determinism: outputs are sorted-keys JSON (``allow_nan=False``, NaN/inf ->
null); two runs with identical inputs and identical ``generated_at`` produce
byte-identical JSON.

CLI (offline)::

    python -m app.sellreview contexts --db sqlite+aiosqlite:////tmp/.../db \
        --universe sp500 --universes-dir ... --start 2022-01-01 --end 2024-12-31 --out DIR
    python -m app.sellreview ledger   ... (same args)

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
from pathlib import Path

import pandas as pd

from . import optimize, reval
from .monthly import _qv_order
from .sellshadow import (
    CASH_YIELD_ANNUAL,
    CONTRIBUTION,
    MODEL_ID,
    PROMPT_VERSION,  # noqa: F401 — stable type-A prompt version, re-exported for the A/B routing
    TEMPERATURE,
    CoreWindow,
    _jsonable,
    load_core_window,
    parse_response,
)

SCHEMA_CONSULT = "sell-consult/1"
SCHEMA_CONSULT_LOG = "shadow-consult-log/1"
SCHEMA_CONTEXTS_B = "banddrop-contexts/1"
SCHEMA_LEDGER_B = "banddrop-ledger/1"
PROMPT_VERSION_B = "sellshadow-b/1"

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

# Event types routed to the type-B parser (DEFER is parameter-free there).
_B_EVENT_TYPES = {"B", "banddrop", "band_drop", "band-drop"}

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


def _parser_for(event_type: str):
    """The event-type-appropriate fail-safe parser: type B (band-drop) DEFER is
    parameter-free, so it splits from sellshadow.parse_response (type A), which
    requires rearm_mult on DEFER."""
    return parse_response_b if str(event_type) in _B_EVENT_TYPES else parse_response


async def consult(event_id: str, event_type: str, prompt_version: str,
                  messages: list[dict], *, transport=None,
                  timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Consult the unified sell reviewer on ONE prospective candidate exit
    event. This is the only function in the module that can originate a real
    LLM call, and only when ``transport`` is left at its default (the proxy);
    tests and CLI paths always inject a transport and stay offline.

    ONE attempt, pinned timeout. ANY failure — transport exception, non-200,
    timeout, empty body, malformed/refused/unparsable response — returns the
    fail-safe CONFIRM with the error recorded (``str(e)`` truncated to
    ``ERROR_TEXT_MAX`` chars; the error text of an httpx failure never contains
    the key, and headers are never logged). The returned ``sell-consult/1``
    record carries ``key_present`` (presence ONLY — never the key)."""
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
        parsed = _fail_safe_b(f"consult failure ({error_class})")
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
                  contribution: float = CONTRIBUTION):
    """Run ``optimize._daily_core_backtest`` at the M0 defaults over the window
    on the shared frame cache, with the 5B flat cost convention (flat one-way
    stock bps through the engine's ``cost_oneway`` wrapper)."""
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
                trailing_stop_pct=knobs.get("trailing_stop_pct"))
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

        in_frame = frame is not None and tk in frame.index
        in_elig = elig is not None and tk in elig.index
        roe = float(frame.loc[tk, "roe"]) if in_frame and pd.notna(frame.loc[tk, "roe"]) else None
        p_fcf = (float(frame.loc[tk, "p_fcf"]) if in_frame and pd.notna(frame.loc[tk, "p_fcf"])
                 else None)
        mom = float(frame.loc[tk, "mom"]) if in_frame and pd.notna(frame.loc[tk, "mom"]) else None
        mom_rank = (float(elig.loc[tk, "mom_rank"]) if in_elig and pd.notna(elig.loc[tk, "mom_rank"])
                    else None)
        band_dist = (order.index(tk) - hold_band + 1) if tk in order and order.index(tk) >= hold_band else None

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
            "roe": roe,
            "p_fcf": p_fcf,
            "mom_12_1": mom,
            "mom_rank_pct_in_frame": mom_rank,
            "band_distance_ranks": band_dist,  # 1 = first name outside the hold band
            "eligible_in_frame": bool(in_elig),
            "days_since_filing": _days_since_filing(win.fund.get(tk), me_str),
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
# Window drivers (read-only DB) + CLI
# ---------------------------------------------------------------------------

async def _load_b_window(db_url: str, universe: str, universes_dir: str | None,
                         start: str, end: str,
                         costs: reval.CostConfig | None = None):
    """Resolve the universe, point at the snapshot DB READ-ONLY, load the core
    window (one shared frame cache) + the benchmark, and run the M0 backtest."""
    uni = reval.resolve_universe(universe, universes_dir)
    with reval.universes_override(universes_dir):
        async with reval.point_at_database(db_url):
            win = await load_core_window(uni, db_url, universes_dir, start, end)
            bench_series = await optimize._load_series(
                [BENCHMARK], start=pd.Timestamp(start).to_pydatetime())
            bench = bench_series.get(BENCHMARK)
            summary = await _run_m0(win, costs)
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


# ---------------------------------------------------------------------------
# CLI (offline ONLY — no subcommand ever constructs a real LLM call)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.sellreview",
        description="Phase 5C slice 5C-A/B: the unified reviewer's consult runner "
                    "(library-only) + the type-B band-drop adapter (offline, "
                    "analysis-only, read-only DB — zero LLM/network calls)")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in (("contexts", "extract band-drop contexts (banddrop-contexts/1)"),
                        ("ledger", "build the retention counterfactual ledger (banddrop-ledger/1)")):
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
        else:
            report = asyncio.run(banddrop_ledger(
                args.db, args.universe, args.universes_dir, args.start, args.end,
                args.out, generated_at))
            fname = "banddrop-ledger.json"
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    sc = report.get("selfcheck", {})
    if args.command == "contexts":
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, "
              f"{sc.get('events')} band-drop events, self-check match rate "
              f"{sc.get('match_rate')})")
    else:
        agg = report["aggregate"]
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, "
              f"{agg['n']} events, mean delta {agg['mean']}, "
              f"share positive {agg['share_positive']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
