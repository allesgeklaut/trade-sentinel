"""Phase 5A slice 5A-1: the sell-shadow research harness (offline, analysis-only).

Purpose
-------
``docs/sell-exit-research-2026-10-31.md`` §4 designs an LLM sell-decider in
PROSPECTIVE shadow mode: consulted on every ATR stop fire with PIT-only
context, returning CONFIRM or DEFER, fail-safe CONFIRM. This slice builds the
deterministic machinery only — **zero LLM calls, zero network calls** (the
authorized model id is a recorded constant, never contacted):

1. ``contexts``   — stop-fire context extractor, schema ``stopfire-context/1``:
   every SELL of a CONTRACT-mode ``optimize._replay`` run whose reason is an
   ATR standing order, with per-event point-in-time context (decision-day
   information only) as deterministic sorted-keys JSON. Includes the ratchet
   self-check: the standing level is recomputed from the raw bars with the
   engine's exact formula and compared to the level recorded in the trade's
   reason string (acceptance: 100% match).
2. prompt         — the frozen, versioned prompt renderer (``sellshadow/1``)
   and the fail-safe response parser (pure; malformed/refusal/missing JSON ->
   CONFIRM). No model is ever called.
3. ``defer``      — the counterfactual defer ledger, schema ``defer-ledger/1``:
   per stop-fire event, the DEFER branch's forward path under the predeclared
   rule (hold with the stop re-armed at ``level * (1 - REARM_WIDEN)``, exit at
   the first of re-armed-stop breach with CONTRACT-honest fills, a time cap of
   ``TIME_CAP_SESSIONS`` sessions, or window end), both branches' forward
   returns net of round-trip one-way costs, and the delta, aggregated overall
   and by year.
4. ``variants``   — five deterministic exit variants (existing
   ``ReplayParams`` knobs only, zero engine changes), each run over the window
   through the honest metric pipeline (reval's pure functions on the unitized
   NAV), reporting TWR/XIRR/maxDD/vol/excess-Sharpe/turnover, trade count,
   sell-mix by reason class, holding-period stats and Odean PGR/PLR.

It is ANALYSIS-ONLY and OFFLINE-ONLY:

* it never writes the database — ``reval.point_at_database`` opens the target
  read-only (SQLite ``PRAGMA query_only=ON``), and the test suite proves every
  table's row counts are unchanged;
* it never calls a market-data provider or an LLM; every engine runs from the
  stored candles of the ``--db`` snapshot.

Declared assumptions (recorded verbatim in every output)
---------------------------------------------------------
* Base replay: ``optimize._replay`` in CONTRACT mode on
  ``optimize._load_series`` output (strict session bars) with
  ``optimize._live_sim_params()``, ``monthly_allowance=1000``,
  ``start_cash=0``, and ``cost_oneway = CostConfig().rate_for(t, "stock")``
  (10 bps stock + 15 bps FX on non-USD listings), mirroring
  ``reval._run_sim_arm``.
* The standing ATR stop level in the replay is the per-bar prior-known
  Chandelier candidate at the decision session S,
  ``atr_stop(S) = max(close[S-20..S-1]) - 2 * atr14[S-1]`` with
  ``atr14 = rolling(14).mean()`` of the true range (``app.analysis``'s exact
  formula); the governing standing order is the max of that candidate and the
  frozen percent stop (``_live_sim_params``: stop_type="percent",
  stop_pct=15). The replay does not pass a per-position ``atr_stop_level``
  ratchet; the monotone max-since-entry of the candidate is reported per event
  as a cross-check field only.
* DEFER branch rule (predeclared): re-arm at ``level * (1 - REARM_WIDEN)``
  (REARM_WIDEN = 0.10), CONTRACT-honest fills (at-open when open <= level,
  else at the level when low <= level, else no fill), time cap
  ``TIME_CAP_SESSIONS = 63`` sessions after the fill, window end as the last
  bound. Time-cap and window-end exits fill at that session's strict open when
  the ticker has a bar, else at its last known close (declared here).
* Forward returns are per-share from the decision-day (S) close mark, net of
  round-trip one-way costs (``rate_for(ticker, "stock") * 2``) on both branches.
* Metrics: reval's pure functions on the unitized NAV of the yield-adjusted
  equity (declared cash yield 3%/yr, post-hoc), rf proxy = the same declared
  cash yield; turnover per ``reval.turnover_metrics``; PGR/PLR per Odean (1998)
  counted by position on each sell day.
* ``configs_tried = 5`` for the variants study (recorded, audit requirement).

Honesty limits (must be stated by any evaluation built on this)
----------------------------------------------------------------
* The counterfactual DEFER branch is a single-name what-if: it holds the
  position in isolation and ignores the freed cash's redeployment, the
  re-entry the engine would actually make, and any interaction with the
  portfolio circuit breaker. It measures the stop rule's per-event opportunity
  cost, not a full portfolio policy.
* The LLM is NOT evaluated here: no prompt is ever sent. The contexts are for
  prompt design and prospective evaluation windows only (no LLM calls on
  history — contamination).
* Universes are current-membership snapshots (survivorship/selection bias in
  absolute terms; the five variants share one universe and one window, so
  their RELATIVE comparison is robust to it).
* The cash yield and the risk-free proxy are post-hoc declared assumptions.

Determinism: outputs are sorted-keys JSON (``allow_nan=False``, NaN/inf ->
null); two runs with identical inputs and identical ``generated_at`` produce
byte-identical JSON. ``generated_at`` is caller-supplied (CLI: ``--generated-at``,
default wall-clock UTC).

CLI (offline)::

    python -m app.sellshadow contexts --db sqlite+aiosqlite:////tmp/opencode/reval/trading_reval.db \
        --universe sp500 --universes-dir /tmp/opencode/reval/universes \
        --start 2022-01-01 --end 2023-12-31 --out DIR
    python -m app.sellshadow defer    ... (same args)
    python -m app.sellshadow variants ... (same args)

Library-first: everything is callable as plain functions from tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd

from . import optimize, reval
from .analysis import ATR_STOP_LOOKBACK, ATR_STOP_MULT
from .optimize import ReplayParams

SCHEMA_CONTEXTS = "stopfire-context/1"
SCHEMA_LEDGER = "defer-ledger/1"
SCHEMA_VARIANTS = "sellshadow-variants/1"
PROMPT_VERSION = "sellshadow/1"

# Recorded configuration constants — the authorized model is NEVER called in
# this slice (offline-only); the id and temperature travel with every output.
MODEL_ID = "deepseek-v4.1-flash-mistral"
TEMPERATURE = 0.0

# Predeclared counterfactual rule (module constants, declared in outputs).
REARM_WIDEN = 0.10          # defer re-arm: level * (1 - REARM_WIDEN)
TIME_CAP_SESSIONS = 63      # defer time cap, sessions after the fill
REARM_MULT_MIN = 1.0        # parser clamp for the LLM's rearm_mult field
REARM_MULT_MAX = 3.0

CASH_YIELD_ANNUAL = 0.03    # declared rf proxy / cash yield (reval default)
CONTRIBUTION = 1000.0       # monthly allowance, USD
START_CASH = 0.0
BENCHMARK = "URTH"
DEFAULT_START = "2015-01-02"
DEFAULT_END = "2026-10-30"
VARIANTS_CONFIGS_TRIED = 5

_STOP_REASON_RE = re.compile(
    r"^ATR stop standing order: (open|low) ([0-9.]+) <= level ([0-9.]+) — filled at (the open|the level)$"
)
_STRENGTH_RE = re.compile(r"\(strength (\d+)\)")

SYSTEM_PROMPT = (
    f"You are the sell-decider in a paper-trading research harness (prompt-version: {PROMPT_VERSION}).\n"
    "ROLE: you review ONE position whose ATR trailing-stop standing order would fire on the "
    "next session. The live engine ALWAYS executes the stop; your answer only decides whether the "
    "shadow counterfactual branch DEFERs it.\n"
    "DECISION: reply with exactly one of CONFIRM (let the stop execute) or DEFER (hold, with the "
    "stop re-armed wider). DEFER must carry a rearm_mult in [1.0, 3.0] (the stop level is re-armed "
    "at level x (1 - (rearm_mult - 1) x widen)) and a short rationale. CONFIRM may carry a rationale.\n"
    "OUTPUT FORMAT: a single JSON object, e.g. {\"decision\": \"DEFER\", \"rearm_mult\": 1.5, "
    "\"rationale\": \"...\"} or {\"decision\": \"CONFIRM\", \"rationale\": \"...\"}. Nothing else.\n"
    "FAIL-SAFE: any malformed output, refusal, or missing JSON counts as CONFIRM — the existing "
    "rule always stands. You see point-in-time context only (decision-day information); never ask "
    "for or assume future data."
)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Recursively convert NaN/inf floats to None so ``allow_nan=False`` JSON
    is always writable and deterministic (same convention as ``reval``)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _dumps(obj) -> str:
    return json.dumps(_jsonable(obj), sort_keys=True, indent=2, allow_nan=False)


def parse_stop_reason(reason: str) -> dict | None:
    """Parse an ATR standing-order SELL reason into its parts.

    Contract-mode reasons look like
    ``"ATR stop standing order: low 45.02 <= level 45.14 — filled at the level"``
    or ``"... open 48.70 <= level 49.16 — filled at the open"``. Returns
    ``{"trigger", "trigger_px", "level", "fill_kind"}`` or None.
    """
    m = _STOP_REASON_RE.match(reason or "")
    if not m:
        return None
    trigger, trig_px, level, where = m.groups()
    return {"trigger": trigger, "trigger_px": float(trig_px), "level": float(level),
            "fill_kind": "at_open" if where == "the open" else "at_level"}


def classify_sell_reason(reason: str) -> str:
    """Reason class of a SELL trade (sell-mix buckets)."""
    r = reason or ""
    if r.startswith("ATR stop standing order") or r.startswith("ATR stop hit"):
        return "atr_stop"
    if r.startswith("Initial stop"):
        return "initial_stop"
    if r.startswith("Trailing stop"):
        return "trailing_pct"
    if r.startswith("Portfolio stop"):
        return "portfolio_stop"
    if r.startswith("SELL signal") or r.startswith("SELL ("):
        return "signal"  # replay adapter stamps signal SELLs "SELL (strength N)"
    return "other"


def recompute_atr(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Recompute ``atr14`` and the prior-known ``atr_stop`` candidate from raw
    bars with ``app.analysis.signal_series``'s EXACT formula (same pandas ops
    on the same values, so bit-identical to the series columns):
    ``tr = max(h-l, |h-c_prev|, |l-c_prev|)``, ``atr14 = tr.rolling(14).mean()``,
    ``atr_stop(t) = (max(close[t-20..t]) - 2*atr14[t]).shift(1)``.
    """
    c = df["close"]
    prev = c.shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_stop = (c.rolling(ATR_STOP_LOOKBACK).max() - ATR_STOP_MULT * atr14).shift(1)
    return atr14, atr_stop


def _timeline(series: dict[str, pd.DataFrame], start: str | None,
              end: str | None) -> tuple[list[str], int, int]:
    """Global session timeline + window bounds, exactly as ``_replay`` builds them."""
    all_days: set[str] = set()
    for df in series.values():
        all_days.update(df["time"].tolist())
    days = sorted(all_days)
    lo, hi = 0, len(days)
    if start:
        lo = next((i for i, d in enumerate(days) if d >= start), len(days))
    if end:
        hi = next((i for i, d in enumerate(days) if d > end), len(days))
    return days, lo, hi


def _pos_maps(series: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """``{ticker: {time: positional row index}}`` for O(1) row lookup."""
    return {t: dict(zip(df["time"].tolist(), range(len(df)), strict=True))
            for t, df in series.items()}


@dataclass
class _Lot:
    """One open position lot rebuilt from the trade ledger (mirrors
    ``PaperPortfolio``'s avg_cost blend and buy_date/stop_price semantics)."""
    shares: float = 0.0
    avg_cost: float = 0.0
    entry_date: str = ""
    entry_reason: str = ""
    entry_price: float = 0.0
    stop_price: float | None = None


def _lot_snapshots(trades: list[dict], params: ReplayParams) -> dict[int, tuple[str, _Lot]]:
    """Walk the trade ledger once; at every SELL trade index, snapshot the
    sold ticker's lot (state before the sell). Mirrors ``PaperPortfolio.buy``
    (frozen entry stop + blended avg cost) and ``PaperPortfolio.sell``."""
    lots: dict[str, _Lot] = {}
    out: dict[int, tuple[str, _Lot]] = {}
    for i, t in enumerate(trades):
        tk = t["ticker"]
        if t["side"] == "BUY":
            lot = lots.get(tk)
            if lot is None:
                stop = None
                if params.stop_type == "percent":
                    stop = float(t["price"]) * (1 - params.stop_pct / 100)
                elif params.stop_type == "atr":
                    stop = None  # needs ATR at entry; replayed via entry_stop below
                lot = _Lot(shares=0.0, avg_cost=0.0, entry_date=t["date"],
                           entry_reason=t["reason"], entry_price=float(t["price"]),
                           stop_price=stop)
                lots[tk] = lot
            # PaperPortfolio.buy blends the average cost on top-ups.
            cost = float(t["shares"]) * float(t["price"])
            lot.avg_cost = (lot.shares * lot.avg_cost + cost) / (lot.shares + float(t["shares"]))
            lot.shares += float(t["shares"])
        else:
            lot = lots.get(tk)
            if lot is not None:
                out[i] = (tk, _Lot(**{**lot.__dict__}))
            if lot is not None:
                lot.shares -= float(t["shares"])
                if lot.shares <= 0.0001:
                    lots.pop(tk, None)
    return out


# ---------------------------------------------------------------------------
# 1. Stop-fire context extractor (stopfire-context/1)
# ---------------------------------------------------------------------------

def _bench_regime(bench: pd.DataFrame | None, all_days: list[str],
                  s_idx: int) -> tuple[float | None, float | None]:
    """Benchmark drawdown-from-peak (%) and annualized realized vol at the
    decision day S, from benchmark closes carried forward over the timeline
    (the replay's own mark convention). PIT only: bench data <= S."""
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


def _sessions_to_month_end(all_days: list[str], s_idx: int) -> int | None:
    """Sessions from S until the next month-end session (the last timeline
    session of S's month if still ahead, else of the next month)."""
    s_day = all_days[s_idx]
    month = s_day[:7]
    later_same = [d for d in all_days[s_idx + 1:] if d[:7] == month]
    if later_same:
        end_day = later_same[-1]
    else:
        nxt = [d for d in all_days[s_idx + 1:] if d[:7] > month]
        if not nxt:
            return None
        end_day = [d for d in nxt if d[:7] == nxt[0][:7]][-1]
    return all_days.index(end_day) - s_idx


def extract_contexts(series: dict[str, pd.DataFrame], bench: pd.DataFrame | None,
                     res, params: ReplayParams, start: str, end: str) -> dict:
    """Extract every ATR stop-fire SELL of a CONTRACT-mode replay into the
    ``stopfire-context/1`` payload (deterministic; per-event PIT context).

    Includes the ratchet self-check: the standing level is recomputed from the
    raw bars with the engine's exact formula and compared to the level parsed
    from the trade's reason string (2-decimal rounding tolerance 0.005).
    """
    all_days, lo, hi = _timeline(series, start, end)
    pos = _pos_maps(series)
    curve_days = [e["time"] for e in res.equity_curve]
    curve_idx = {d: i for i, d in enumerate(curve_days)}
    snapshots = _lot_snapshots(res.trades, params)

    events: list[dict] = []
    n_match = 0
    n_events = 0
    max_col_diff = 0.0
    for i, t in enumerate(res.trades):
        if t["side"] != "SELL":
            continue
        parsed = parse_stop_reason(t["reason"])
        if parsed is None:
            continue  # not an ATR standing-order stop (signal / other stop class)
        tk = t["ticker"]
        fill_date = t["date"]
        f_idx = all_days.index(fill_date)
        s_idx = f_idx - 1  # the decision session (data <= S)
        s_day = all_days[s_idx]
        if tk not in pos or s_day not in pos[tk]:
            continue  # cannot happen for a fired stop (the signal needs S's row)
        n_events += 1
        p = pos[tk][s_day]
        df = series[tk]
        atr14_s, atr_stop_s = recompute_atr(df)
        atr14_col = df["atr14"].iloc[p]
        atr_col = df["atr_stop"].iloc[p]
        # engine-formula cross-check: recompute vs the series' own columns
        if pd.notna(atr14_col):
            max_col_diff = max(max_col_diff, abs(float(atr14_s.iloc[p]) - float(atr14_col)))
        if pd.notna(atr_col):
            max_col_diff = max(max_col_diff, abs(float(atr_stop_s.iloc[p]) - float(atr_col)))
        atr14_val = float(atr14_s.iloc[p]) if pd.notna(atr14_s.iloc[p]) else None
        atr_cand = float(atr_stop_s.iloc[p]) if pd.notna(atr_stop_s.iloc[p]) else None

        lot = snapshots.get(i, (tk, None))[1]
        entry_date = lot.entry_date if lot else ""
        entry_reason = lot.entry_reason if lot else ""
        entry_price = lot.avg_cost if lot else 0.0
        m = _STRENGTH_RE.search(entry_reason)
        entry_strength = int(m.group(1)) if m else None
        pct_stop = None
        if lot is not None and lot.stop_price is not None:
            pct_stop = lot.stop_price
        elif params.stop_type == "percent" and lot is not None:
            pct_stop = lot.avg_cost * (1 - params.stop_pct / 100)
        levels = [v for v in (pct_stop, atr_cand) if v is not None]
        governing = max(levels) if levels else None
        match = (governing is not None
                 and abs(governing - parsed["level"]) <= 0.005 + 1e-12)
        if match:
            n_match += 1

        close_s = float(df["close"].iloc[p])
        p_entry = pos[tk].get(entry_date)
        if p_entry is not None:
            peak_close = float(df["close"].iloc[p_entry: p + 1].max())
        else:
            peak_close = close_s
        giveback = (close_s / peak_close - 1.0) * 100.0 if peak_close > 0 else None
        pnl = (close_s / entry_price - 1.0) * 100.0 if entry_price > 0 else None
        sessions_held = (all_days.index(fill_date)
                         - all_days.index(entry_date)) if entry_date else None

        ci = curve_idx.get(s_day)
        cash_s = float(res.cash_curve[ci]) if ci is not None else None
        eq_s = float(res.equity_curve[ci]["equity"]) if ci is not None else None
        shares = lot.shares if lot else float(t["shares"])
        weight = (shares * close_s / eq_s) if (eq_s and eq_s > 0) else None

        bench_dd, bench_vol = _bench_regime(bench, all_days, s_idx)

        events.append({
            "ticker": tk,
            "decision_date": s_day,
            "fill_date": fill_date,
            "fill_price": float(t["price"]),
            "stop_level": parsed["level"],
            "fill_kind": parsed["fill_kind"],
            "raw_reason": t["reason"],
            "entry_date": entry_date,
            "entry_price": entry_price,
            "entry_reason": entry_reason,
            "entry_strength": entry_strength,
            "sessions_held": sessions_held,
            "current_close": close_s,
            "pnl_pct": pnl,
            "giveback_pct": giveback,
            "shares": shares,
            "position_weight": weight,
            "portfolio_cash": cash_s,
            "portfolio_equity": eq_s,
            "atr14": atr14_val,
            "dist_to_stop_atr": ((close_s - parsed["level"]) / atr14_val
                                 if atr14_val else None),
            "dist_to_stop_pct": ((close_s - parsed["level"]) / close_s * 100.0
                                 if close_s > 0 else None),
            "bench_drawdown_pct": bench_dd,
            "bench_vol_ann": bench_vol,
            "sessions_to_month_end": _sessions_to_month_end(all_days, s_idx),
            "selfcheck": {
                "recomputed_governing_level": governing,
                "recomputed_atr_candidate": atr_cand,
                "frozen_pct_stop": pct_stop,
                "atr_ratchet_since_entry": (
                    float(atr_stop_s.iloc[p_entry: p + 1].max())
                    if p_entry is not None and pd.notna(atr_stop_s.iloc[p_entry: p + 1]).any()
                    else None),
                "level_match": match,
            },
        })

    events.sort(key=lambda e: (e["fill_date"], e["ticker"]))
    return {
        "schema": SCHEMA_CONTEXTS,
        "prompt_version": PROMPT_VERSION,
        "window": {"start": start, "end": end, "sessions": hi - lo},
        "config": {
            "model_id": MODEL_ID, "temperature": TEMPERATURE,
            "note": "recorded constants only — this slice makes zero LLM/network calls",
            "replay": "optimize._replay CONTRACT mode, _live_sim_params()",
            "params": {"use_atr_stop": params.use_atr_stop,
                       "stop_type": params.stop_type, "stop_pct": params.stop_pct,
                       "trailing_stop_pct": params.trailing_stop_pct,
                       "portfolio_stop_pct": params.portfolio_stop_pct,
                       "max_positions": params.max_positions},
        },
        "selfcheck": {
            "events": n_events,
            "level_matches": n_match,
            "match_rate": (n_match / n_events) if n_events else None,
            "recompute_vs_series_column_max_abs_diff": max_col_diff,
            "tolerance": "reason string records the level at 2 decimals (0.005)",
        },
        "events": events,
    }


# ---------------------------------------------------------------------------
# 2. Frozen prompt renderer + fail-safe parser (pure, versioned, no LLM calls)
# ---------------------------------------------------------------------------

def render_prompt(context: dict) -> list[dict]:
    """The exact message list the LLM will receive (``sellshadow/1``).

    System: role, the decision, the fail-safe, the output format. User: the
    context rendered as stable, versioned text (sorted-keys JSON under a
    prompt-version header). Byte-identical for identical context.
    """
    user = (f"prompt-version: {PROMPT_VERSION}\n"
            f"context (JSON, sorted keys):\n"
            + json.dumps(_jsonable(context), sort_keys=True, indent=2, allow_nan=False))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _fail_safe(why: str) -> dict:
    return {"decision": "CONFIRM", "rearm_mult": None,
            "rationale": f"fail-safe: {why}", "parsed": False, "fail_safe": True}


def parse_response(text: str) -> dict:
    """Parse the model output. Accepts exactly one of CONFIRM or DEFER.

    DEFER must carry ``rearm_mult`` (clamped to [1.0, 3.0]) and may carry a
    rationale; CONFIRM may carry a rationale. Malformed output, refusal,
    missing JSON, or anything unparsable -> CONFIRM (fail-safe to the existing
    rule)."""
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
    rationale = obj.get("rationale")
    rationale = str(rationale) if rationale is not None else ""
    if decision == "CONFIRM":
        return {"decision": "CONFIRM", "rearm_mult": None, "rationale": rationale,
                "parsed": True, "fail_safe": False}
    if decision == "DEFER":
        rm = obj.get("rearm_mult")
        if rm is None:
            return _fail_safe("DEFER without rearm_mult")
        try:
            rmf = float(rm)
        except (TypeError, ValueError):
            return _fail_safe("rearm_mult not numeric")
        if not math.isfinite(rmf):
            return _fail_safe("rearm_mult not finite")
        clamped = min(max(rmf, REARM_MULT_MIN), REARM_MULT_MAX)
        return {"decision": "DEFER", "rearm_mult": clamped, "rationale": rationale,
                "parsed": True, "fail_safe": False}
    return _fail_safe(f"unknown decision {decision!r}")


# ---------------------------------------------------------------------------
# 3. Counterfactual defer ledger (defer-ledger/1)
# ---------------------------------------------------------------------------

def defer_forward(ticker_bars: dict[str, dict], all_days: list[str],
                  fill_idx: int, level: float, hi: int) -> dict:
    """The DEFER branch's forward path under the predeclared rule.

    Hold with the stop re-armed at ``level * (1 - REARM_WIDEN)``; exit at the
    FIRST of (a) a re-armed-stop breach with CONTRACT-honest fills (at-open
    when open <= level, else at the level when low <= level, else no fill and
    retry next session — a ticker without a bar gets no fill), (b) the time cap
    of ``TIME_CAP_SESSIONS`` sessions after the fill, (c) window end
    (``all_days[hi-1]``). Time-cap / window-end exits fill at that session's
    strict open when a bar exists, else at the last known close.
    """
    rearmed = level * (1.0 - REARM_WIDEN)
    last_idx = hi - 1
    cap_idx = min(fill_idx + TIME_CAP_SESSIONS, last_idx)
    last_close: float | None = None
    for j in range(fill_idx, cap_idx + 1):
        bar = ticker_bars.get(all_days[j])
        if bar is not None and bar.get("close") is not None:
            last_close = float(bar["close"])
    for j in range(fill_idx + 1, cap_idx + 1):
        day = all_days[j]
        bar = ticker_bars.get(day)
        if bar is None:
            continue
        open_px = optimize._exec_price(bar.get("open"))
        if open_px is None:
            continue
        fill = optimize._standing_stop_fill({"open": open_px, "low": bar.get("low")},
                                            rearmed, None)
        if fill is not None:
            kind = "at_open" if open_px <= rearmed else "at_level"
            trig = f"open {open_px:.2f}" if kind == "at_open" else f"low {bar['low']:.2f}"
            return {"exit_price": fill, "exit_date": day, "exit_kind": kind,
                    "exit_reason": (f"defer re-armed stop (level {level:.2f} x "
                                    f"(1-{REARM_WIDEN:g}) = {rearmed:.2f}): {trig} <= "
                                    f"level {rearmed:.2f} — filled at "
                                    f"{'the open' if kind == 'at_open' else 'the level'}")}
    day = all_days[cap_idx]
    bar = ticker_bars.get(day) or {}
    open_px = optimize._exec_price(bar.get("open"))
    price = open_px if open_px is not None else last_close
    why = (f"defer time cap ({TIME_CAP_SESSIONS} sessions)" if cap_idx < last_idx
           else "window end")
    at = "open" if open_px is not None else "last known close"
    return {"exit_price": price, "exit_date": day,
            "exit_kind": "time_cap" if cap_idx < last_idx else "window_end",
            "exit_reason": f"{why} — exited at the {at} ({price:.2f})"}


def build_defer_ledger(series: dict[str, pd.DataFrame], res, params: ReplayParams,
                       contexts: dict, start: str, end: str,
                       costs: reval.CostConfig | None = None) -> dict:
    """The ``defer-ledger/1`` payload: per stop-fire event the DEFER branch's
    forward path vs the actual CONFIRM (stop) fill, both net of round-trip
    one-way costs, plus aggregates (overall and by decision year)."""
    costs = costs or reval.CostConfig()
    all_days, lo, hi = _timeline(series, start, end)
    pos = _pos_maps(series)
    records: list[dict] = []
    for ev in contexts["events"]:
        tk = ev["ticker"]
        fill_idx = all_days.index(ev["fill_date"])
        df = series[tk]
        ticker_bars = {d: {"open": (float(df["open"].iloc[i]) if pd.notna(df["open"].iloc[i]) else None),
                           "low": (float(df["low"].iloc[i]) if pd.notna(df["low"].iloc[i]) else None),
                           "close": float(df["close"].iloc[i])}
                       for d, i in pos[tk].items()}
        branch = defer_forward(ticker_bars, all_days, fill_idx,
                               ev["stop_level"], hi)
        rate = costs.rate_for(tk, "stock")
        mark = ev["current_close"]
        confirm_fwd = (ev["fill_price"] / mark - 1.0 - 2.0 * rate) if mark else None
        defer_fwd = (branch["exit_price"] / mark - 1.0 - 2.0 * rate) if mark else None
        delta = (defer_fwd - confirm_fwd) if (defer_fwd is not None and confirm_fwd is not None) else None
        records.append({
            "ticker": tk,
            "decision_date": ev["decision_date"],
            "fill_date": ev["fill_date"],
            "stop_level": ev["stop_level"],
            "rearmed_level": ev["stop_level"] * (1.0 - REARM_WIDEN),
            "confirm_fill_price": ev["fill_price"],
            "confirm_fill_kind": ev["fill_kind"],
            "defer_exit_price": branch["exit_price"],
            "defer_exit_date": branch["exit_date"],
            "defer_exit_kind": branch["exit_kind"],
            "defer_exit_reason": branch["exit_reason"],
            "oneway_rate": rate,
            "mark_close_at_decision": mark,
            "confirm_fwd_net": confirm_fwd,
            "defer_fwd_net": defer_fwd,
            "delta_defer_minus_confirm": delta,
            "level_match": ev["selfcheck"]["level_match"],
        })

    def _agg(rows: list[dict]) -> dict:
        ds = sorted(r["delta_defer_minus_confirm"] for r in rows
                    if r["delta_defer_minus_confirm"] is not None)
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

    by_year: dict[str, dict] = {}
    for r in records:
        y = r["decision_date"][:4]
        by_year.setdefault(y, []).append(r)

    return {
        "schema": SCHEMA_LEDGER,
        "prompt_version": PROMPT_VERSION,
        "window": {"start": start, "end": end, "sessions": hi - lo},
        "config": {
            "model_id": MODEL_ID, "temperature": TEMPERATURE,
            "note": "recorded constants only — this slice makes zero LLM/network calls",
            "rearm_widen": REARM_WIDEN, "time_cap_sessions": TIME_CAP_SESSIONS,
            "cost_rule": "forward returns net of round-trip one-way costs "
                         "(CostConfig.rate_for(ticker,'stock') x2), per-share from the decision-day close",
            "stock_oneway_bps": costs.stock_oneway_bps, "fx_oneway_bps": costs.fx_oneway_bps,
        },
        "selfcheck_match_rate": contexts["selfcheck"]["match_rate"],
        "aggregate": _agg(records),
        "by_year": {y: _agg(rows) for y, rows in sorted(by_year.items())},
        "events": records,
    }


# ---------------------------------------------------------------------------
# 4. Deterministic exit variants (existing ReplayParams knobs only)
# ---------------------------------------------------------------------------

def variant_params() -> dict[str, ReplayParams]:
    """The five predeclared variants. BASE mirrors the live configuration;
    each variant changes only existing ``ReplayParams`` knobs."""
    base = optimize._live_sim_params()
    base.monthly_allowance = CONTRIBUTION
    base.start_cash = START_CASH
    return {
        "BASE": base,
        "V1_signal_exits": replace(base, use_atr_stop=False, stop_type="none"),
        "V2_pct_trail_15": replace(base, use_atr_stop=False, stop_type="none",
                                   trailing_stop_pct=15.0),
        "V3_initial_atr_only": replace(base, use_atr_stop=False, stop_type="atr",
                                       stop_atr_mult=2.0),
        "V4_portfolio_breaker": replace(base, portfolio_stop_pct=20.0),
    }


def disposition_metrics(res, series: dict[str, pd.DataFrame]) -> dict:
    """Odean (1998) PGR/PLR, counted by position on each sell day.

    On every SELL event the sold position realizes a gain (RG) or loss (RL)
    vs its average basis; every other position still held that day carries a
    paper gain (PG) or loss (PL) vs basis, marked at that day's close (carried
    forward). PGR = RG/(RG+PG), PLR = RL/(RL+PL).
    """
    all_days, _, _ = _timeline(series, None, None)
    day_idx = {d: i for i, d in enumerate(all_days)}
    close_map = {t: dict(zip(df["time"].tolist(), df["close"].tolist(), strict=True))
                 for t, df in series.items()}
    lots: dict[str, _Lot] = {}
    rg = rl = pg = pl = 0
    for t in res.trades:
        tk = t["ticker"]
        if t["side"] == "BUY":
            lot = lots.get(tk)
            if lot is None:
                lot = _Lot(entry_date=t["date"], entry_reason=t["reason"],
                           entry_price=float(t["price"]))
                lots[tk] = lot
            cost = float(t["shares"]) * float(t["price"])
            lot.avg_cost = (lot.shares * lot.avg_cost + cost) / (lot.shares + float(t["shares"]))
            lot.shares += float(t["shares"])
            continue
        lot = lots.get(tk)
        if lot is None:
            continue
        # realized side of the sold position
        if float(t["price"]) > lot.avg_cost:
            rg += 1
        else:
            rl += 1
        # paper side of everything else still held on this sell day
        d = t["date"]
        for other, ol in lots.items():
            if other == tk:
                continue
            c = None
            for dd in all_days[: day_idx[d] + 1]:
                if dd in close_map.get(other, {}):
                    c = float(close_map[other][dd])
            if c is None:
                continue
            if c > ol.avg_cost:
                pg += 1
            else:
                pl += 1
        lot.shares -= float(t["shares"])
        if lot.shares <= 0.0001:
            lots.pop(tk, None)
    pgr = rg / (rg + pg) if (rg + pg) > 0 else None
    plr = rl / (rl + pl) if (rl + pl) > 0 else None
    return {"RG": rg, "RL": rl, "PG": pg, "PL": pl, "PGR": pgr, "PLR": plr}


def holding_period_stats(res, series: dict[str, pd.DataFrame]) -> dict:
    """Sessions held per closed lot (entry fill date -> sell date), median/mean."""
    all_days, _, _ = _timeline(series, None, None)
    idx = {d: i for i, d in enumerate(all_days)}
    lots: dict[str, str] = {}
    holds: list[int] = []
    for t in res.trades:
        tk = t["ticker"]
        if t["side"] == "BUY":
            lots.setdefault(tk, t["date"])
        else:
            e = lots.pop(tk, None)
            if e is not None and t["date"] in idx and e in idx:
                holds.append(idx[t["date"]] - idx[e])
    if not holds:
        return {"n_closed_lots": 0, "median_sessions": None, "mean_sessions": None}
    s = sorted(holds)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n_closed_lots": n, "median_sessions": med,
            "mean_sessions": sum(holds) / n}


def run_variant(series: dict[str, pd.DataFrame], name: str, params: ReplayParams,
                start: str, end: str, costs: reval.CostConfig | None = None,
                contribution: float = CONTRIBUTION, start_cash: float = START_CASH) -> dict:
    """One full-window variant run through the honest metric pipeline:
    ``_replay`` (CONTRACT, per-ticker cost hook) -> yield-adjusted unitized
    NAV -> reval's pure metric functions. Also sell-mix, holding stats, PGR/PLR."""
    from .performance import unitized_nav
    costs = costs or reval.CostConfig()
    rate_fn = lambda t: costs.rate_for(t, "stock")  # noqa: E731
    res = optimize._replay(series, params, start=start, end=end, cost_oneway=rate_fn)
    dates = [e["time"] for e in res.equity_curve]
    equity = [float(e["equity"]) for e in res.equity_curve]
    cash = [float(c) for c in res.cash_curve]
    invested: list[float] = []
    cum = start_cash
    prev_month: str | None = None
    contrib_by_date: dict[str, float] = {}
    for d in dates:
        if d[:7] != prev_month:
            cum += contribution
            contrib_by_date[d] = contribution
            prev_month = d[:7]
        invested.append(cum)
    equity_adj, _interest = reval.apply_cash_yield(cash, equity, CASH_YIELD_ANNUAL)
    nav = unitized_nav(equity_adj, invested)
    twr = reval.twr_metrics(dates, nav)
    trades = [{**t, "notional": float(t["shares"]) * float(t["price"])} for t in res.trades]
    sells = [t for t in res.trades if t["side"] == "SELL"]
    mix: dict[str, int] = {}
    for t in sells:
        mix[classify_sell_reason(t["reason"])] = mix.get(classify_sell_reason(t["reason"]), 0) + 1
    return {
        "name": name,
        "params": {"use_atr_stop": params.use_atr_stop, "stop_type": params.stop_type,
                   "stop_pct": params.stop_pct, "stop_atr_mult": params.stop_atr_mult,
                   "trailing_stop_pct": params.trailing_stop_pct,
                   "portfolio_stop_pct": params.portfolio_stop_pct,
                   "max_positions": params.max_positions,
                   "monthly_allowance": params.monthly_allowance,
                   "start_cash": params.start_cash},
        "metrics": {
            "twr_annualized": twr["twr_annualized"],
            "twr_cumulative": twr["twr_cumulative"],
            "xirr": reval.xirr_metric(dates, invested, equity_adj[-1] if equity_adj else None),
            "max_drawdown_pct": reval.drawdown_metrics(nav)["max_drawdown_pct"],
            "vol_annualized": reval.annualized_vol(nav),
            "sharpe_excess": reval.excess_sharpe(nav, CASH_YIELD_ANNUAL),
            "rf_annual": CASH_YIELD_ANNUAL,
            "turnover_annualized": reval.turnover_metrics(
                dates, trades, equity_adj, contrib_by_date)["annualized"],
            "n_trades": res.n_trades,
            "n_sells": len(sells),
        },
        "sell_mix": dict(sorted(mix.items())),
        "holding": holding_period_stats(res, series),
        "disposition": disposition_metrics(res, series),
        "final_equity": res.final_equity,
    }


# ---------------------------------------------------------------------------
# Window loading (read-only DB) + async drivers
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """Everything one window needs: universe, series, benchmark, base replay."""
    universe: dict
    series: dict[str, pd.DataFrame]
    bench: pd.DataFrame | None
    res: Any
    params: ReplayParams
    start: str
    end: str
    db_url: str
    universes_dir: str | None = None
    extra: dict = field(default_factory=dict)


async def load_window(db_url: str, universe: str, universes_dir: str | None,
                      start: str, end: str,
                      costs: reval.CostConfig | None = None) -> Window:
    """Resolve the universe, point at the snapshot DB READ-ONLY, load the
    CONTRACT series + benchmark, and run the BASE replay once."""
    costs = costs or reval.CostConfig()
    uni = reval.resolve_universe(universe, universes_dir)
    with reval.universes_override(universes_dir):
        async with reval.point_at_database(db_url):
            series = await optimize._load_series(
                uni["members"], start=pd.Timestamp(start).to_pydatetime())
            if not series:
                raise ValueError("no series with enough candle history in the window")
            bench_series = await optimize._load_series(
                [BENCHMARK], start=pd.Timestamp(start).to_pydatetime())
            bench = bench_series.get(BENCHMARK)
            params = optimize._live_sim_params()
            params.monthly_allowance = CONTRIBUTION
            params.start_cash = START_CASH
            rate_fn = lambda t: costs.rate_for(t, "stock")  # noqa: E731
            import io
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                res = optimize._replay(series, params, start=start, end=end,
                                       cost_oneway=rate_fn)
    return Window(universe=uni, series=series, bench=bench, res=res,
                  params=params, start=start, end=end, db_url=db_url,
                  universes_dir=universes_dir)


async def load_series_only(db_url: str, universe: str, universes_dir: str | None,
                           start: str, end: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Resolve universe + load the CONTRACT series (for the variants sweep,
    which replays each variant on the same loaded series)."""
    uni = reval.resolve_universe(universe, universes_dir)
    with reval.universes_override(universes_dir):
        async with reval.point_at_database(db_url):
            series = await optimize._load_series(
                uni["members"], start=pd.Timestamp(start).to_pydatetime())
            if not series:
                raise ValueError("no series with enough candle history in the window")
    return uni, series


def _base_report(win: Window, generated_at: str | None) -> dict:
    return {"generated_at": generated_at, "db": {"url": win.db_url, "read_only": True},
            "universe": win.universe, "window": {"start": win.start, "end": win.end}}


async def run_contexts(db_url: str, universe: str, universes_dir: str | None,
                       start: str, end: str, generated_at: str | None = None) -> dict:
    win = await load_window(db_url, universe, universes_dir, start, end)
    out = extract_contexts(win.series, win.bench, win.res, win.params, start, end)
    out.update(_base_report(win, generated_at))
    return out


async def run_defer(db_url: str, universe: str, universes_dir: str | None,
                    start: str, end: str, generated_at: str | None = None,
                    costs: reval.CostConfig | None = None) -> dict:
    costs = costs or reval.CostConfig()
    win = await load_window(db_url, universe, universes_dir, start, end, costs)
    contexts = extract_contexts(win.series, win.bench, win.res, win.params, start, end)
    out = build_defer_ledger(win.series, win.res, win.params, contexts, start, end, costs)
    out.update(_base_report(win, generated_at))
    return out


async def run_variants(db_url: str, universe: str, universes_dir: str | None,
                       start: str, end: str, generated_at: str | None = None,
                       costs: reval.CostConfig | None = None) -> dict:
    costs = costs or reval.CostConfig()
    uni, series = await load_series_only(db_url, universe, universes_dir, start, end)
    variants = [run_variant(series, name, params, start, end, costs)
                for name, params in variant_params().items()]
    return {
        "schema": SCHEMA_VARIANTS,
        "generated_at": generated_at,
        "db": {"url": db_url, "read_only": True},
        "universe": uni,
        "window": {"start": start, "end": end},
        "config": {"model_id": MODEL_ID, "temperature": TEMPERATURE,
                   "note": "recorded constants only — this slice makes zero LLM/network calls",
                   "cash_yield_annual": CASH_YIELD_ANNUAL,
                   "contribution": CONTRIBUTION, "start_cash": START_CASH,
                   "cost_rule": "cost_oneway = CostConfig.rate_for(ticker,'stock') at fill, mirroring reval._run_sim_arm",
                   "configs_tried": VARIANTS_CONFIGS_TRIED},
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.sellshadow",
        description="Phase 5A slice 5A-1: sell-shadow research harness "
                    "(analysis-only, read-only DB, offline — zero LLM/network calls)")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in (("contexts", "extract stop-fire contexts (stopfire-context/1)"),
                        ("defer", "build the counterfactual defer ledger (defer-ledger/1)"),
                        ("variants", "run the five deterministic exit variants")):
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
            report = asyncio.run(run_contexts(args.db, args.universe, args.universes_dir,
                                              args.start, args.end, generated_at))
            fname = "stopfire-contexts.json"
        elif args.command == "defer":
            report = asyncio.run(run_defer(args.db, args.universe, args.universes_dir,
                                           args.start, args.end, generated_at))
            fname = "defer-ledger.json"
        else:
            report = asyncio.run(run_variants(args.db, args.universe, args.universes_dir,
                                              args.start, args.end, generated_at))
            fname = "sellshadow-variants.json"
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / fname).write_text(_dumps(report) + "\n")
    if args.command == "contexts":
        sc = report["selfcheck"]
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, "
              f"{sc['events']} stop-fire events, ratchet self-check match rate "
              f"{sc['match_rate']})")
    elif args.command == "defer":
        agg = report["aggregate"]
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, "
              f"{agg['n']} events, mean delta {agg['mean']}, "
              f"share positive {agg['share_positive']})")
    else:
        names = ", ".join(v["name"] for v in report["variants"])
        print(f"{fname} written to {out}/{fname} (schema {report['schema']}, "
              f"configs_tried {report['config']['configs_tried']}, variants: {names})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
