"""The LLM sell layer — slice 1: consult + EXECUTE in the live paper accounts.

The app keeps its deterministic strategies; the LLM becomes the SELL brain of
both live paper accounts at three NARROW per-event decision moments (the
owner-approved concept; the research's frozen prompts and per-type mandates
from ``app/sellreview.py``):

  * **Type A (the sim, ``deterministic`` strategy mode only)** — a position's
    standing stop (ATR/percent) is about to fire. The LLM CONFIRMs (the stop
    executes as today) or DEFERs (the sale is suppressed; the stop's ATR leg
    re-arms at ``level * (1 - (rearm_mult - 1) * REARM_WIDEN)`` — the 5A-1
    rule; the frozen percent stop is NEVER loosened). The defer is persisted
    on the position (``SimPosition.stop_defer_level`` / ``stop_defer_set_at``)
    and capped at ``sell_layer_max_defers`` consecutive defers, counted from
    the decision ledger — at the cap the stop executes regardless ("defer
    cap"). The hybrid/llm whole-cycle strategy modes are RETIRED (a stale
    value degrades to deterministic with a warning in run_cycle), so there
    is no whole-cycle LLM review left to duplicate; NO second LLM pass is
    added anywhere.
  * **Type B (the monthly book)** — at the month-end rebuild a held name falls
    out of the hysteresis band. CONFIRM releases it as today; DEFER retains it
    one more month and the new name it displaced is not bought (capacity stays
    ``target_n``).
  * **Type C (the monthly book)** — a holding's newly available fundamentals
    break the predeclared thesis condition. EXIT sells at the NEXT session's
    strict exact-date open (proceeds park until the next rebuild — declared);
    HOLD is the engine's rule (wait for the month-end).

Every decision is appended to the ``LlmSellDecision`` ledger with the
engine-counterfactual branch, so the scorecard slice can show the
with/without-LLM history.

CONTAMINATION RULE (mechanical, not just discipline): the LLM is NEVER graded
on history and never sees historical events from this code. This module wires
ONLY into the LIVE cycle paths — ``sim.run_cycle()`` (deterministic branch,
between propose and execute) and ``monthly.run_monthly_cycle()`` /
``monthly._run_rebalance_locked()``. The backfill/replay/research paths never
import this module, and ``tests/test_selllayer.py`` proves that a backfill run
with an exploding mock transport makes ZERO consult attempts.

FAIL-SAFE EVERYWHERE: any failure — no backend configured (short-circuit
BEFORE any transport call, fail_reason "no backend"), transport error,
``asyncio.wait_for`` timeout (``sell_layer_timeout_s``), malformed/refusal —
means the ENGINE's own rule stands; the live book never breaks because of
the LLM. With the knobs off the overlays are identity functions: bit-identical
OFF parity.

The owner's explicit caution is honoured: the sim's retired LLM layer
(hybrid/pure-LLM whole-cycle review — retired, degrades to deterministic)
never worked well — nothing here is modelled on it beyond
``SimLlmDecision``'s ledger conventions (append-only,
``decided_at`` frozen at parse, NULL-together fill columns, no pending-order
persistence beyond the ledger rows — the session fill contract, rule 7).

Transport: the overlay's transport is a callable
``async (messages, model_id, temperature, timeout_s) -> str`` wrapping
``app.llm.chat``; the consult is bounded by ``asyncio.wait_for`` at
``sell_layer_timeout_s`` (llm.chat's own timeout is the global 1200s setting,
far too long for a nightly cycle; wait_for cancellation is acceptable). All
prompts/parsers/fail-safes/field builders are REUSED from ``app/sellreview``
(the frozen ``sellshadow/1``, ``sellshadow-b/1``, ``sellshadow-c/1`` prompts
render byte-identically); this module is the ONLY place consult wires into
live execution. The research CLI paths in sellreview are untouched.

Zero network in tests: every test injects a mock transport (and the
no-backend tests need no transport at all).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, UTC

import pandas as pd
from sqlalchemy import func, select

from . import sellreview
from .config import settings
from . import db as db_mod
from .db import LlmSellDecision, MonthlyAccount, MonthlyPosition, SimPosition
from .sellshadow import REARM_WIDEN

logger = logging.getLogger("trade_sentinel.selllayer")


def _dumps(obj) -> str:
    return json.dumps(sellreview._jsonable(obj), sort_keys=True, indent=2,
                      allow_nan=False)


# ---------------------------------------------------------------------------
# The transport + the live consult runner
# ---------------------------------------------------------------------------

async def _llm_transport(messages: list[dict], model_id: str, temperature: float,
                         timeout_s: float) -> str:
    """The default transport: wrap ``app.llm.chat`` (the multi-backend client).

    ``model_id`` is recorded, not forwarded — the backend's own configured
    model serves the request. ``temperature`` IS forwarded (one wire,
    matching the frozen constant every consult record declares) and
    ``timeout_s`` rides through as the per-call read bound; the caller's
    ``asyncio.wait_for`` (``sell_layer_timeout_s``) remains the wall-clock
    bound.
    """
    from . import llm as llm_mod
    result = await llm_mod.chat(messages, temperature=temperature,
                                timeout_s=timeout_s)
    return str(result.get("text") or "")


def _consult_record(event_id: str, event_type: str, prompt_version: str,
                    parsed: dict, *, error: str | None, error_class: str | None,
                    latency_ms: float, fail_reason: str | None,
                    backend: str = "", model: str = "",
                    key_present: bool = True) -> dict:
    """One ``sell-consult/1``-shaped record as extended by the live layer
    (adds ``backend``/``model``/``fail_reason`` for the ledger)."""
    return {
        "schema": sellreview.SCHEMA_CONSULT,
        "event_id": event_id,
        "event_type": event_type,
        "prompt_version": prompt_version,
        "model_id": sellreview.MODEL_ID,
        "temperature": sellreview.TEMPERATURE,
        "decision": parsed["decision"],
        "params": {k: parsed[k] for k in ("rearm_mult",) if parsed.get(k) is not None},
        "rationale": parsed["rationale"],
        "parsed": parsed["parsed"],
        "fail_safe": parsed["fail_safe"],
        "error": error,
        "error_class": error_class,
        "latency_ms": latency_ms,
        "key_present": key_present,
        "backend": backend,
        "model": model,
        "fail_reason": fail_reason,
    }


async def consult_live(event_id: str, event_type: str, context: dict, *,
                       transport=None, timeout_s: float | None = None) -> dict:
    """Consult the sell reviewer on ONE LIVE candidate event.

    Fail-safe everywhere; never raises on an LLM-side problem:

      * no backend configured -> short-circuit BEFORE any transport call:
        the per-type fail-safe with ``fail_reason="no backend"`` (zero calls);
      * the consult runs under ``asyncio.wait_for(..., sell_layer_timeout_s)``
        — a timeout is a fail-safe (``fail_reason="timeout"``);
      * transport errors / malformed / refusal are fail-safed inside
        ``sellreview.consult`` (CONFIRM for A/B, HOLD for C).
    """
    from . import llm as llm_mod

    timeout_s = float(settings.sell_layer_timeout_s if timeout_s is None else timeout_s)
    started = time.perf_counter()
    backend = await llm_mod.current_backend()
    if not backend:
        # No-backend short-circuit: the account behaves exactly as without
        # the layer — the engine's own rule stands, and NO transport call is
        # made (the proof tests count the calls).
        parsed = sellreview._fail_safe_for(event_type, "no backend")
        return _consult_record(
            event_id, event_type, sellreview.prompt_version_for(event_type), parsed,
            error=None, error_class=None,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            fail_reason="no backend", key_present=False)
    messages = sellreview.render_prompt_for(event_type, context)
    prompt_version = sellreview.prompt_version_for(event_type)
    tr = transport if transport is not None else _llm_transport
    try:
        rec = await asyncio.wait_for(
            sellreview.consult(event_id, event_type, prompt_version, messages,
                               transport=tr, timeout_s=timeout_s),
            timeout=timeout_s)
    except TimeoutError:
        parsed = sellreview._fail_safe_for(event_type, "timeout (asyncio.wait_for)")
        rec = _consult_record(
            event_id, event_type, prompt_version, parsed,
            error="asyncio.TimeoutError", error_class="TimeoutError",
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            fail_reason="timeout", backend=backend["name"],
            model=str(backend.get("model") or ""))
    rec["backend"] = backend["name"]
    rec["model"] = str(backend.get("model") or "")
    if rec.get("fail_reason") is None:
        if rec.get("error"):
            rec["fail_reason"] = f"{rec.get('error_class')}: {rec['error']}"
        elif rec.get("fail_safe"):
            rec["fail_reason"] = "unparsable response"
        else:
            rec["fail_reason"] = None
    return rec


async def _record_decision(*, rec: dict, engine: str, session_date: str,
                           ticker: str, context: dict, engine_branch: dict,
                           decided_at: datetime | None = None) -> LlmSellDecision:
    """Append ONE ``LlmSellDecision`` row (append-only). ``decided_at`` is
    frozen at parse — captured once, here, before any order can execute."""
    row = LlmSellDecision(
        decided_at=decided_at or datetime.now(UTC),
        session_date=session_date,
        event_type=str(rec["event_type"]),
        engine=engine,
        event_id=str(rec["event_id"]),
        ticker=ticker,
        decision=str(rec["decision"]),
        rationale=str(rec.get("rationale") or ""),
        prompt_version=str(rec.get("prompt_version") or ""),
        parsed=bool(rec.get("parsed")),
        fail_safe=bool(rec.get("fail_safe")),
        fail_reason=(str(rec["fail_reason"])[:200] if rec.get("fail_reason") else None),
        context=_dumps(context),
        engine_branch=_dumps(engine_branch),
        backend=str(rec.get("backend") or ""),
        model=str(rec.get("model") or ""),
        latency_ms=rec.get("latency_ms"),
    )
    async with db_mod.Session() as s:
        s.add(row)
        await s.commit()
    return row


# ---------------------------------------------------------------------------
# Type A — the sim's standing-stop overlay (wired between propose + execute)
# ---------------------------------------------------------------------------

async def _consecutive_a_defers(ticker: str, opened_at: datetime) -> int:
    """Consecutive A-DEFER ledger rows for the position since its last
    A-CONFIRM or its opening (the defer-cap count)."""
    async with db_mod.Session() as s:
        rows = (await s.scalars(
            select(LlmSellDecision)
            .where(LlmSellDecision.engine == "sim",
                   LlmSellDecision.event_type == "A",
                   LlmSellDecision.ticker == ticker,
                   LlmSellDecision.decided_at >= opened_at)
            .order_by(LlmSellDecision.decided_at))).all()
    n = 0
    for r in rows:
        n = n + 1 if r.decision == "DEFER" else 0
    return n


async def _persist_defer(ticker: str, rearmed_level: float) -> None:
    """Persist the type-A defer on the position (the re-armed ATR leg)."""
    async with db_mod.Session() as s:
        pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
        if pos is None:
            return
        pos.stop_defer_level = float(rearmed_level)
        pos.stop_defer_set_at = datetime.now(UTC)
        await s.commit()


async def overlay_type_a(proposals: list[dict], decision, *,
                         transport=None, series: dict | None = None) -> list[dict]:
    """The type-A overlay: consult per stop-fire SELL proposal (identified
    MACHINE-READABLY via the proposal's ``"stop_fire"`` key), between propose
    and execute. CONFIRM keeps the proposal; DEFER drops it and persists the
    re-armed ATR level on the position. At ``sell_layer_max_defers``
    consecutive defers the stop executes regardless ("defer cap").

    Fail-safe: knob off -> identity (bit-identical OFF); no backend / any
    consult failure -> the engine's rule stands (the proposal is kept).
    Never raises on an LLM-side problem.
    """
    if not settings.sim_llm_sell_layer:
        return proposals
    stop_fires = [p for p in proposals if isinstance(p.get("stop_fire"), dict)]
    if not stop_fires:
        return proposals
    try:
        return await _overlay_type_a_impl(proposals, decision, transport=transport,
                                          series=series)
    except Exception as e:  # noqa: BLE001 — the live book never breaks
        logger.error("sell layer type-A overlay failed (%s: %s) — engine's rule stands",
                     type(e).__name__, e, exc_info=True)
        return proposals


async def _overlay_type_a_impl(proposals: list[dict], decision, *,
                               transport=None, series: dict | None = None) -> list[dict]:
    f_str = pd.Timestamp(decision.f_ts).strftime("%Y-%m-%d")
    s_str = (pd.Timestamp(decision.s_ts).strftime("%Y-%m-%d")
             if decision.s_ts is not None else f_str)
    if series is None:
        from . import optimize
        series = await optimize._load_series(sorted({p["ticker"] for p in proposals
                                                      if p.get("stop_fire")}))
    max_defers = int(settings.sell_layer_max_defers)
    kept: list[dict] = []
    for p in proposals:
        sf = p.get("stop_fire")
        if not isinstance(sf, dict):
            kept.append(p)
            continue
        tk = p["ticker"]
        level = float(sf["level"])
        label = str(sf.get("label") or "")
        async with db_mod.Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == tk))
        if pos is None:
            kept.append(p)  # nothing held — the engine's exec guard handles it
            continue
        pos_dict = {
            "ticker": pos.ticker, "shares": float(pos.shares),
            "avg_cost": float(pos.avg_cost),
            "opened_at": pd.Timestamp(pos.opened_at).strftime("%Y-%m-%d"),
            "stop_price": (float(pos.stop_price) if pos.stop_price is not None else None),
            "stop_defer_level": (float(pos.stop_defer_level)
                                 if pos.stop_defer_level is not None else None),
        }
        # Defer cap: at the cap the stop executes regardless, no consult.
        if await _consecutive_a_defers(tk, pos.opened_at) >= max_defers:
            logger.warning("sell layer A: defer cap (%d) reached for %s — the stop "
                           "executes regardless", max_defers, tk)
            rec = _consult_record(
                f"A:{tk}:{s_str}", "A", sellreview.prompt_version_for("A"),
                sellreview._fail_safe_for("A", "defer cap"),
                error=None, error_class=None, latency_ms=None,
                fail_reason="defer cap")
            await _record_decision(
                rec=rec, engine="sim", session_date=s_str, ticker=tk,
                context={"ticker": tk, "decision_date": s_str, "defer_cap": max_defers},
                engine_branch={"action": "execute standing stop (defer cap)",
                               "level": level, "label": label, "fill_price": p["price"]})
            kept.append(p)
            continue
        # The A context: the shared _detect_type_a field family, built from the
        # cycle's own position/frame data.
        df = (series or {}).get(tk)
        event = None
        if df is not None:
            try:
                event, _skip = sellreview._type_a_event(pos_dict, df, s_str, f_str)
            except Exception as e:  # noqa: BLE001 — fall back to the proposal's facts
                logger.warning("sell layer A: context build failed for %s: %s", tk, e)
        if event is not None:
            context = dict(event)
        else:
            context = {
                "ticker": tk, "decision_date": s_str, "fill_date": f_str,
                "fill_price": p["price"], "fill_kind": "standing_order",
                "stop_level": level, "stop_label": label,
                "entry_date": pos_dict["opened_at"], "entry_price": pos_dict["avg_cost"],
                "shares": pos_dict["shares"], "frozen_pct_stop": pos_dict["stop_price"],
                "stop_defer_level": pos_dict["stop_defer_level"],
                "derivation": "context from the live proposal (no computable series event)",
            }
        context["event_type"] = "A"
        context["event_id"] = f"A:{tk}:{s_str}"
        context["proposal_price"] = p["price"]
        rec = await consult_live(context["event_id"], "A", context, transport=transport)
        rearm_mult = (rec.get("params") or {}).get("rearm_mult")
        engine_branch = {"action": "execute standing stop", "level": level,
                         "label": label, "fill_price": p["price"]}
        if rec["decision"] == "DEFER" and rearm_mult is not None:
            # 5A-1 re-arm: the ATR leg re-arms at level x (1 - (mult-1) x widen);
            # the frozen percent stop is never loosened.
            rearmed = level * (1.0 - (float(rearm_mult) - 1.0) * REARM_WIDEN)
            engine_branch = {"action": "defer", "level": level, "label": label,
                             "rearm_mult": float(rearm_mult), "rearmed_level": rearmed}
            await _record_decision(rec=rec, engine="sim", session_date=s_str, ticker=tk,
                                   context=context, engine_branch=engine_branch)
            await _persist_defer(tk, rearmed)
            logger.info("sell layer A: DEFER %s — stop re-armed to %.4f (was %.4f)",
                        tk, rearmed, level)
            continue  # the proposal is dropped — the sale is suppressed
        await _record_decision(rec=rec, engine="sim", session_date=s_str, ticker=tk,
                               context=context, engine_branch=engine_branch)
        kept.append(p)
    return kept


# ---------------------------------------------------------------------------
# Type B — the monthly rebuild's band-drop overlay
# ---------------------------------------------------------------------------

def _book_plumbing(close: pd.DataFrame, as_of_ts: pd.Timestamp,
                   frame, hold_band: int):
    """The shared plumbing the ``sellreview._monthly_book_context`` field
    family expects: close_map / day_idx / day_strs / band / order / next
    rebuild, derived from the cycle's own frames (PIT: data <= as-of)."""
    from . import monthly as monthly_mod

    full_idx = pd.DatetimeIndex(close.index)
    days_all = list(full_idx[full_idx <= as_of_ts])
    day_strs = [d.strftime("%Y-%m-%d") for d in days_all]
    day_idx = {d: i for i, d in enumerate(day_strs)}
    close_map = {t: dict(zip(day_strs, close.ffill().loc[days_all, t].tolist(),
                             strict=True)) for t in close.columns}
    months_full = monthly_mod.completed_month_ends(close.index, full_idx[0], full_idx[-1])
    later = [m for m in months_full if m > as_of_ts]
    next_rebuild = None
    if later:
        f2 = next((d for d in full_idx if d > later[0]), None)
        if f2 is not None:
            full_strs = [d.strftime("%Y-%m-%d") for d in full_idx]
            full_idx_map = {d: i for i, d in enumerate(full_strs)}
            f2s = f2.strftime("%Y-%m-%d")
            next_rebuild = (f2s, full_idx_map[f2s] - full_idx_map[as_of_ts.strftime("%Y-%m-%d")])
    elig, order = (monthly_mod._qv_order(frame) if frame is not None else (None, []))
    band = set(order[:hold_band]) if frame is not None and frame["eligible"].any() else None
    return {"close_map": close_map, "day_idx": day_idx, "day_strs": day_strs,
            "band": band, "elig": elig, "order": order, "next_rebuild": next_rebuild}


async def _bench_series(as_of_ts: pd.Timestamp):
    """Best-effort benchmark (URTH) series from the stored candles for the
    bench regime fields; None when unavailable (fields render null)."""
    try:
        from . import optimize
        start = (as_of_ts - pd.Timedelta(days=400)).to_pydatetime()
        series = await optimize._load_series([sellreview.BENCHMARK], start=start)
        return series.get(sellreview.BENCHMARK)
    except Exception:  # noqa: BLE001 — bench fields are context, never critical
        return None


async def overlay_type_b(*, held_before: list[str], picks: list[str],
                         fill_map: dict, frame, close: pd.DataFrame, fund: dict,
                         members: list[str], me_ts, f_date,
                         target_n: int, hold_band: int,
                         transport=None) -> tuple[list[str], dict | None]:
    """The type-B overlay: after ``pick_portfolio``, before the SELL loop.

    Candidates = held names NOT in picks with an executable fill price (names
    without a quote stay pending as today — not consulted). CONFIRM -> release
    as today; DEFER -> the name is retained one more month: ``executed_picks``
    = picks minus the LAST-ADDED new names (from the tail of the adds) plus
    the retained names, so ``len(executed_picks) <= target_n`` and at most one
    retained name per displaced add. When the defensive capacity cap releases
    a DEFER-retained name anyway (reachable when a ``target_n`` cut shrinks
    the book below the retained names), the detail records it as
    ``capacity_truncated`` — the DEFER ledger rows stand, and the scorecard
    can attribute the release to capacity, not the LLM's call. The SELL/BUY
    loops then run on
    ``executed_picks`` unchanged.

    Returns ``(executed_picks, sell_detail)`` — ``sell_detail`` is the
    {"engine_picks", "executed_picks"} JSON the MonthlyRebalance audit row
    gains, or None when no candidate was consulted. Fail-safe: knob off or
    any failure -> ``(picks, None)`` (bit-identical OFF).
    """
    if not settings.sim_monthly_llm_sells:
        return list(picks), None
    try:
        return await _overlay_type_b_impl(
            held_before=held_before, picks=picks, fill_map=fill_map, frame=frame,
            close=close, fund=fund, members=members, me_ts=me_ts, f_date=f_date,
            target_n=target_n, hold_band=hold_band, transport=transport)
    except Exception as e:  # noqa: BLE001 — the live book never breaks
        logger.error("sell layer type-B overlay failed (%s: %s) — engine's rule stands",
                     type(e).__name__, e, exc_info=True)
        return list(picks), None


async def _overlay_type_b_impl(*, held_before, picks, fill_map, frame, close, fund,
                               members, me_ts, f_date, target_n, hold_band,
                               transport=None) -> tuple[list[str], dict | None]:
    picks = list(picks)
    held_set = set(held_before)
    candidates = [t for t in held_before if t not in set(picks) and fill_map.get(t)]
    if not candidates:
        return picks, None

    f_ts = pd.Timestamp(f_date)
    if f_ts.tzinfo is not None:
        f_ts = f_ts.tz_localize(None)
    f_ts = f_ts.normalize()
    as_of = f_ts.strftime("%Y-%m-%d")
    plumb = _book_plumbing(close, f_ts, frame, hold_band)
    async with db_mod.Session() as s:
        held_rows = (await s.scalars(select(MonthlyPosition))).all()
        acc = await s.get(MonthlyAccount, 1)
    holdings = {p.ticker: {"ticker": p.ticker, "shares": float(p.shares),
                           "avg_cost": float(p.avg_cost),
                           "opened_at": pd.Timestamp(p.opened_at).strftime("%Y-%m-%d")}
                for p in held_rows}
    equity = None
    if acc is not None:
        pos_val = sum(h["shares"] * (plumb["close_map"].get(t, {}).get(as_of) or 0.0)
                      for t, h in holdings.items())
        equity = float(acc.cash) + pos_val
    bench = await _bench_series(f_ts)

    retained: list[str] = []
    consulted = 0
    for t in candidates:
        h = holdings.get(t)
        if h is None:
            continue
        ctxs = sellreview._monthly_book_context(
            fund, members, as_of, [h], plumb["band"], hold_band, plumb["next_rebuild"],
            frame, plumb["elig"], plumb["order"], plumb["close_map"], plumb["day_idx"],
            equity, bench, plumb["day_strs"])
        ctx = dict(ctxs[0])
        # Derived, never hardcoded: in the capacity edge case (more held names
        # inside the band than target_n slots — _band_fill truncates
        # band[:target_n]) a released held name IS still inside the band.
        band = plumb["band"]
        ctx.update({
            "event_type": "B", "decision_date": as_of,
            "rebuild_date": as_of, "fill_date": as_of,
            "fill_price": float(fill_map[t]), "fill_kind": "open",
            "band_excludes_name": ((t not in band) if band is not None else None),
            "event_id": f"B:{t}:{as_of}",
        })
        rec = await consult_live(ctx["event_id"], "B", ctx, transport=transport)
        consulted += 1
        await _record_decision(
            rec=rec, engine="monthly", session_date=as_of, ticker=t, context=ctx,
            engine_branch={"action": "release at rebuild", "fill_price": float(fill_map[t])})
        if rec["decision"] == "DEFER":
            retained.append(t)
            logger.info("sell layer B: DEFER %s — retained one more month", t)
    if not retained:
        return picks, ({"engine_picks": picks, "executed_picks": picks}
                       if consulted else None)
    # Capacity-preserving retention: drop the LAST-ADDED new names (the tail of
    # the adds) — at most one retained name per displaced add.
    adds = [t for t in picks if t not in held_set]
    drop = set(adds[-len(retained):]) if adds else set()
    executed = [t for t in picks if t not in drop] + sorted(retained)
    truncated: list[str] = []
    if len(executed) > target_n:  # defensive cap — never exceed target_n
        truncated = sorted(executed[target_n:])
        executed = executed[:target_n]
    detail = {"engine_picks": picks, "executed_picks": executed,
              "retained": sorted(retained), "consulted": consulted}
    if truncated:
        # Attribution honesty: the capacity cap can release DEFER-retained
        # names anyway — their DEFER ledger rows stand, and the scorecard
        # must be able to see the release was capacity, not the LLM's call.
        detail["capacity_truncated"] = truncated
        logger.info("sell layer B: capacity cap released %s "
                    "(their DEFER rows stand)", ", ".join(truncated))
    return executed, detail


# ---------------------------------------------------------------------------
# Type C — the monthly book's nightly thesis-break step
# ---------------------------------------------------------------------------

async def run_type_c_step(now: datetime | None = None, *, transport=None,
                          fund: dict | None = None, close: pd.DataFrame | None = None,
                          vol: pd.DataFrame | None = None,
                          frame=None) -> dict:
    """The nightly type-C step, run EVERY trading night by
    ``monthly.run_monthly_cycle()`` BEFORE the rebuild step (declared order).

    1. FILL step: scan unfilled C-EXIT ledger rows for still-held names and
       sell at THIS session's strict exact-date open (``_open_usd_map`` +
       ``_exec_sell``, reason "thesis-break LLM exit"), stamping the row's
       frozen fill columns (``filled_at >= decided_at`` always). A name
       without an executable open this session stays pending (retried at the
       next cycle — no pending-order persistence beyond the ledger).
    2. SCAN step: for each held monthly position, load the latest PIT
       fundamentals (filing-date PIT, the ``_asof_entries`` family), apply
       the predeclared thesis condition; a break is NEW when no CONSULTED
       (non-fail-safe) ledger row exists for that (ticker, latest filing
       date) — a transient fail-safe row never silences a filing. Consult;
       EXIT appends the unfilled pending row (filled at the NEXT cycle),
       HOLD the row only.

    The scan's frame/fund/series loads run over the LIVE UNIVERSE (the same
    loads the rebuild and the daily-core cycle make — declared nightly cost,
    DB-only), so the context's cross-sectional fields are percentiles in the
    real eligible frame, not among the held names alone; the variant is synced
    from the live selection exactly like the rebuild does.

    Fail-safe: knob off -> skipped; any failure -> logged, never raised.
    """
    if not settings.sim_monthly_llm_sells:
        return {"skipped": True, "reason": "sim_monthly_llm_sells off"}
    try:
        return await _run_type_c_step_impl(now, transport=transport, fund=fund,
                                           close=close, vol=vol, frame=frame)
    except Exception as e:  # noqa: BLE001 — the live book never breaks
        logger.error("sell layer type-C step failed (%s: %s)", type(e).__name__, e,
                     exc_info=True)
        return {"error": f"{type(e).__name__}: {e}"}


async def run_type_d_fill_step(now=None) -> dict:
    """The type-D FILL step — the 2026-11-04 promotion (the design note's
    addendum of that date; the owner's declared decision, scope BOTH paper
    books). Fills pending D-EXIT ledger rows at the first session STRICTLY
    AFTER the row's decision date, on whichever book holds the name:

    1. The guard the C step never needed: D verdicts arrive at ANY
       wall-clock (manual scans), so a row fills only when its decided_at
       DATE is strictly before the fill session's date — an airtight
       no-lookback invariant with no market-clock assumptions (no bell
       times, no DST edge). For cycle-decided rows this is identical to
       C's effective cadence: decide on session S, fill at S+1's open.
    2. The price: the session's strict exact-date open
       (``monthly._open_usd_map`` + the book's own ``_exec_sell``),
       exactly the session fill contract — never a stale/forward price.
    3. BOTH books execute through their transaction-level sell primitives,
       under both state locks. Trades, fill columns and per-book lifecycle
       audit commit atomically (app/newsexits.py). Replaced/released/reset
       positions and duplicate verdicts close without selling a later buy.
    4. The frozen fill columns summarize the first book; execution JSON
       records every book's trade/session. The scorecard exposes execution
       separately from the still-pending performance window.

    Runs EVERY trading night (never weekday-gated — the cadence gate owns
    the SCAN, not the fills) via ``monthly.run_monthly_cycle``, declared
    order after the type-C step. Fail-safe: knob off -> the shadow phase
    exactly as before; any failure -> logged, never raised.
    """
    if not settings.sim_news_execute:
        return {"skipped": True, "reason": "SIM_NEWS_EXECUTE off (shadow)"}
    try:
        return await _run_type_d_fill_step_impl(now)
    except Exception as e:  # noqa: BLE001 — the live book never breaks
        logger.error("sell layer type-D fill step failed (%s: %s)", type(e).__name__, e,
                     exc_info=True)
        return {"error": f"{type(e).__name__}: {e}"}


async def _run_type_d_fill_step_impl(now) -> dict:
    from .newsexits import fill_pending
    return await fill_pending(now)


async def news_exit_excluded(me) -> set[str]:
    """Tickers a deployment pass must NOT buy (the re-entry guard, the
    2026-11-04 promotion addendum): names with a FILLED D-EXIT whose fill
    is at/after the governing month-end ``me``. Until the next completed
    month-end re-evaluates the name on fresh data, both books' deploy
    paths exclude it from NEW buys — a news exit is not in any factor, so
    without this guard the rank-first refill would re-buy the name the
    very next session (sell Tuesday, re-bought Wednesday). ``me`` is a
    pandas Timestamp (normalized); held names are never force-released by
    this guard — it blocks new buys only (a name can only be both held
    and excluded via a pre-guard re-buy, and the band hysteresis owns
    held names).
    """
    me_ts = pd.Timestamp(me)
    if me_ts.tzinfo is not None:
        me_ts = me_ts.tz_localize(None)
    me_ts = me_ts.normalize()
    async with db_mod.Session() as s:
        rows = (await s.scalars(
            select(LlmSellDecision.ticker)
            .where(LlmSellDecision.event_type == "D",
                   LlmSellDecision.decision == "EXIT",
                   LlmSellDecision.filled_at.is_not(None),
                   LlmSellDecision.filled_at >= me_ts.to_pydatetime()))).all()
    return set(rows)


async def _run_type_c_step_impl(now, *, transport=None, fund, close, vol, frame) -> dict:
    from . import fundamentals as fundamentals_mod
    from . import monthly as monthly_mod
    from . import optimize

    now = now or monthly_mod._now()
    f_ts = pd.Timestamp(now)
    if f_ts.tzinfo is not None:
        f_ts = f_ts.tz_localize(None)
    f_ts = f_ts.normalize()
    as_of = f_ts.strftime("%Y-%m-%d")

    async with db_mod.Session() as s:
        held_rows = (await s.scalars(select(MonthlyPosition).order_by(MonthlyPosition.ticker))).all()
        pending = (await s.scalars(
            select(LlmSellDecision)
            .where(LlmSellDecision.engine == "monthly",
                   LlmSellDecision.event_type == "C",
                   LlmSellDecision.decision == "EXIT",
                   LlmSellDecision.filled_at.is_(None))
            .order_by(LlmSellDecision.decided_at))).all()
    held = {p.ticker: p for p in held_rows}

    # --- 1. FILL step -------------------------------------------------------
    fills: list[dict] = []
    to_fill = [r for r in pending if r.ticker in held]
    if to_fill:
        open_map = await monthly_mod._open_usd_map(sorted(r.ticker for r in to_fill), f_ts)
        for r in to_fill:
            px = open_map.get(r.ticker)
            if not px:
                continue  # no executable open this session — retried next cycle
            trade = await monthly_mod._exec_sell(r.ticker, px, "thesis-break LLM exit")
            if trade is None:
                continue
            async with db_mod.Session() as s:
                row = await s.get(LlmSellDecision, r.id)
                if row is not None and row.filled_at is None:
                    acc = await s.get(MonthlyAccount, 1)
                    row.filled_at = datetime.now(UTC)
                    row.shares = float(trade["shares"])
                    row.price = float(trade["price"])
                    row.cash_after = float(acc.cash) if acc is not None else None
                    await s.commit()
            fills.append({"ticker": r.ticker, "price": float(px),
                          "event_id": r.event_id, "decided_at": str(r.decided_at)})
            logger.info("sell layer C: EXIT %s filled at %s open %.2f",
                        r.ticker, as_of, px)

    # --- 2. SCAN step -------------------------------------------------------
    if not held:
        return {"fills": fills, "scans": [], "as_of": as_of}
    # The frame/fund/series loads widen to the LIVE UNIVERSE (not just the held
    # names): the context's cross-sectional fields (mom_rank_pct_in_frame,
    # band_distance_ranks via _monthly_book_context/_factor_fields) are
    # percentiles in the ELIGIBLE FRAME over the universe — a percentile among
    # ~10 held names would be meaningless. This mirrors the rebuild's own
    # construction (monthly.py): universe members, the variant synced from the
    # live selection (settings.sim_daily_core_mom_variant = _strategy_variant(),
    # exactly like the rebuild does before its frame math — NO "raw" pin), and
    # the same fund/close/vol loads. Declared cost: the nightly step's loads
    # are the same class as the daily-core cycle's (DB-only). Only the HELD
    # positions are scan candidates. When fund/close/vol are injected (tests),
    # the injected-arguments path is unchanged.
    injected = fund is not None and close is not None and vol is not None
    if injected:
        members = sorted(held)
    else:
        universe = monthly_mod.universe_tickers(monthly_mod._strategy_universe())
        members = sorted(set(universe) | set(held))
        if fund is None:
            fund = await fundamentals_mod.load_fundamentals(members)
        if close is None or vol is None:
            close, vol = await monthly_mod.load_frames(
                members, None, start=monthly_mod._live_history_start())
        # Sync the runtime variant from the live selection EXACTLY like the
        # rebuild (monthly.py: settings.sim_daily_core_mom_variant = _strategy_variant()).
        settings.sim_daily_core_mom_variant = monthly_mod._strategy_variant()
    hold_band = int(settings.sim_monthly_hold_band)
    if frame is None:
        me_list = [m for m in monthly_mod.completed_month_ends(
            close.index, pd.DatetimeIndex(close.index)[0], f_ts) if m <= f_ts]
        me = me_list[-1] if me_list else None
        if me is not None:
            frame = await asyncio.to_thread(monthly_mod.eligible_frame, me, close, vol, fund)
    plumb = _book_plumbing(close, f_ts, frame, hold_band)
    async with db_mod.Session() as s:
        acc = await s.get(MonthlyAccount, 1)
    holdings = [{"ticker": p.ticker, "shares": float(p.shares),
                 "avg_cost": float(p.avg_cost),
                 "opened_at": pd.Timestamp(p.opened_at).strftime("%Y-%m-%d")}
                for p in held_rows]
    equity = None
    if acc is not None:
        pos_val = sum(h["shares"] * (plumb["close_map"].get(h["ticker"], {}).get(as_of) or 0.0)
                      for h in holdings)
        equity = float(acc.cash) + pos_val
    bench = await _bench_series(f_ts)

    scans: list[dict] = []
    for h in holdings:
        tk = h["ticker"]
        rec_fund = fund.get(tk)
        if not optimize._thesis_break_fundamentals(rec_fund, f_ts):
            continue
        filing = sellreview._breaking_filing(rec_fund, as_of) or "none"
        event_id = f"C:{tk}:{filing}"
        async with db_mod.Session() as s:
            exists = await s.scalar(
                select(func.count()).select_from(LlmSellDecision)
                .where(LlmSellDecision.event_id == event_id,
                       LlmSellDecision.fail_safe.is_(False)))
        if exists:
            # dedupe: only a CONSULTED row silences the filing — a transient
            # fail-safe row (no backend / timeout / parse failure) must never
            # permanently suppress the re-consult
            continue
        ctxs = sellreview._monthly_book_context(
            fund, members, as_of, [h], plumb["band"], hold_band, plumb["next_rebuild"],
            frame, plumb["elig"], plumb["order"], plumb["close_map"], plumb["day_idx"],
            equity, bench, plumb["day_strs"])
        ctx = dict(ctxs[0])
        ctx.update({
            "event_type": "C", "decision_date": as_of,
            # the fill is the NEXT session's strict open — not yet known at
            # the nightly decision (declared)
            "fill_date": None, "fill_price": None, "fill_kind": "next_session_open",
            "thesis_break_condition": sellreview._THESIS_BREAK_CONDITION,
            "breaking_filing_date": filing if filing != "none" else None,
            "event_id": event_id,
        })
        rec = await consult_live(event_id, "C", ctx, transport=transport)
        await _record_decision(
            rec=rec, engine="monthly", session_date=as_of, ticker=tk, context=ctx,
            engine_branch={"action": "hold to next month-end rebuild"})
        scans.append({"ticker": tk, "event_id": event_id, "decision": rec["decision"],
                      "fail_safe": rec["fail_safe"]})
        logger.info("sell layer C: %s thesis break (%s) -> %s", tk, event_id,
                    rec["decision"])
    return {"fills": fills, "scans": scans, "as_of": as_of}
