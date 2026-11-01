"""The LLM sell scorecard — slice 2: the with/without-LLM history.

The app's answer to "did the LLM decision improve things?": for every
``LlmSellDecision`` ledger row (app/selllayer.py, slice 1) whose PREDECLARED
forward window has COMPLETED, score BOTH branches' REALIZED outcomes from the
stored prices — the same per-event counterfactual math the research's ledgers
used, applied forward to live events — and aggregate per event type against the
pre-measured reference bars (``sellreview.REFERENCE_BARS``).

  * **Type A (engine "sim")** — window = [decision session → the EARLIER of
    the position's next SELL fill on the ticker (the ``SimTrade`` rows after
    the decision date — the re-armed stop firing or any other exit) and
    ``TIME_CAP_SESSIONS`` (63) sessions after the decision]. LLM branch
    (DEFER) = held through the window, exited at the recorded fill when a sell
    fired, else marked at the window's last available close, net of the one-way
    cost at that exit; engine branch = sold at the decision's fill price, the
    proceeds at the declared cash yield over the window, net of the one-way
    cost at the decision fill. CONFIRM (and every fail-safe, whose implied
    branch IS the engine's path) = no divergence, delta 0 by construction.
  * **Type B (engine "monthly")** — the row joins its rebuild's
    ``MonthlyRebalance.sell_layer`` JSON BY THE REBUILD'S IDENTITY, not by a
    month string: the B row's ``session_date`` is the rebuild's FILL session
    (the first session AFTER the governing month-end — app/selllayer.py writes
    ``as_of = f_date``), while ``MonthlyRebalance.rebal_date`` stores the
    governing month-end (``me_ts``) and ``rebal_month`` its month, so the join
    is the audit row with the GREATEST ``rebal_date`` strictly before the row's
    ``session_date`` (preferring ``sell_layer NOT NULL``, then highest epoch —
    the highest-epoch-wins dedupe). Window = [this rebuild's fill → the next
    rebuild's fill session]. LLM branch (DEFER) = the retained
    name's return over the window; engine branch = the DISPLACED addition's
    return over the same window (``engine_picks − executed_picks``, exact
    name; when it cannot be identified, the declared fallback is the
    equal-weight mean of the names newly added at the same rebuild, and the
    research's cash-yield leg when there are no additions). Fill prices are
    the strict exact-date opens of the governing sessions
    (``monthly._open_usd_map`` — the same convention as the rebuild's fills).
    Both legs of both branches are net of the one-way cost (the costs cancel
    in the delta). CONFIRM = delta 0.
  * **Type C (engine "monthly")** — window = [the row's EXIT fill → the next
    month-end rebuild's fill session]. LLM branch (EXIT) = sold at the row's
    recorded fill, net of the one-way cost there, the proceeds at the declared
    cash yield to the next rebuild; engine branch = the name held to the next
    rebuild's fill (the strict open there, net of the one-way cost — a name
    with no executable price at the next rebuild stays PENDING). HOLD = delta 0.

CONTAMINATION RULE (mechanical, not just discipline): the scorer grades ONLY
live forward-log decisions from the ``LlmSellDecision`` ledger, NEVER history;
it makes NO LLM calls and NO network calls. It is READ-ONLY on the trade
tables: no writes, no new columns — the outcome is computed on demand from the
stored candles/trades/rebalances, deterministic byte-identical modulo the
declared wall-clock field (``generated_at``). It never raises on a single row's
problems — the row is counted + skipped with a reason.

Type-D rows (the shadow material-news scan, app/newsagent.py) appear in the
decision history with status "shadow" but are NEVER scored, NEVER bucketed
into A/B/C and NEVER counted in the per-type or overall stats (the D scorer
does not exist in the shadow phase; ``n_d_shadow`` counts them); any other
unknown event_type is skipped-with-reason the same way.

Costs/cash conventions: ``reval.CostConfig()``'s declared rates (the same
stock bps + cash yield the study used), imported from reval, not redefined.
Cash yield accrues COMPOUNDED per session — ``(1 + cash_yield_annual) **
(w / 252)`` over a window of ``w`` sessions — the declared ``reval``
convention (``apply_cash_yield``'s ``(1+cy)**(1/252)`` per session), so the
deltas and the ``REFERENCE_BARS`` share one convention.

Deltas are fractions of the decision fill (0.0293 = +2.93pp); the reference
bars' ``*_pp`` fields share this unit. AGGREGATES are taken over the DIVERGENT
(non-zero-delta) rows only, overall + by year, so the UI can show "the LLM vs
the blanket rules". NO p-values, NO significance claims: small-n samples are
reported as-is (``sellreview.SMALL_N_CAVEAT`` verbatim in the output).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Awaitable, Callable
from datetime import datetime, UTC

import pandas as pd
from sqlalchemy import func, select

from . import db as db_mod
from . import monthly as monthly_mod
from . import reval
from . import sellreview
from .db import Candle, LlmSellDecision, MonthlyRebalance, SimTrade
from .sellshadow import TIME_CAP_SESSIONS

logger = logging.getLogger("trade_sentinel.sellscore")

SCHEMA = "sellscore/1"
HISTORY_LIMIT = 200
RATIONALE_LEN = 200

# The decisions that DIVERGE from the engine's own rule (everything else —
# CONFIRM/HOLD, and every fail-safe whose implied branch IS the engine's path
# — scores delta 0 by construction).
DIVERGENT_DECISION = {"A": "DEFER", "B": "DEFER", "C": "EXIT"}

# The strict-open loader signature shared by the B/C scorers:
# ``async (day, names) -> {ticker: open | None}`` (monthly._open_usd_map).
OpenFn = Callable[[str, list[str]], Awaitable[dict[str, float | None]]]


def _dumps(obj) -> str:
    return json.dumps(sellreview._jsonable(obj), sort_keys=True, allow_nan=False)


def _json(text: str | None) -> dict:
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


# ---------------------------------------------------------------------------
# Read-only loads
# ---------------------------------------------------------------------------

async def _load_candles(session_factory, tickers: list[str]) -> dict[str, dict[str, dict]]:
    """{ticker: {"YYYY-MM-DD": {"open","low","close"}}} from the stored
    candles (READ-ONLY)."""
    out: dict[str, dict[str, dict]] = {}
    if not tickers:
        return out
    async with session_factory() as s:
        rows = (await s.execute(
            select(Candle.ticker, Candle.timestamp, Candle.open, Candle.low, Candle.close)
            .where(Candle.ticker.in_(sorted(set(tickers))))
            .order_by(Candle.ticker, Candle.timestamp))).all()
    for tk, ts, op, lo, cl in rows:
        d = pd.Timestamp(ts)
        if d.tzinfo is not None:
            d = d.tz_localize(None)
        out.setdefault(tk, {})[d.strftime("%Y-%m-%d")] = {
            "open": (float(op) if op is not None and math.isfinite(op) and op > 0 else None),
            "low": (float(lo) if lo is not None and math.isfinite(lo) else None),
            "close": (float(cl) if cl is not None and math.isfinite(cl) else None),
        }
    return out


async def _session_calendar(session_factory) -> list[str]:
    """The sorted distinct stored session dates (the trading calendar)."""
    async with session_factory() as s:
        rows = await s.scalars(select(func.distinct(func.date(Candle.timestamp)))
                               .order_by(func.date(Candle.timestamp)))
    return [pd.Timestamp(r).strftime("%Y-%m-%d") for r in rows]


async def _next_sell_fills(session_factory,
                           tickers: list[str]) -> dict[str, list[tuple[str, float]]]:
    """{ticker: [(date, price)] of SELL fills, ascending} — the sim's exit
    fills after a decision (READ-ONLY)."""
    out: dict[str, list[tuple[str, float]]] = {}
    if not tickers:
        return out
    async with session_factory() as s:
        rows = (await s.execute(
            select(SimTrade.ticker, SimTrade.price, SimTrade.created_at)
            .where(SimTrade.side == "SELL", SimTrade.ticker.in_(sorted(set(tickers))))
            .order_by(SimTrade.created_at, SimTrade.id))).all()
    for tk, px, ts in rows:
        d = pd.Timestamp(ts)
        if d.tzinfo is not None:
            d = d.tz_localize(None)
        out.setdefault(tk, []).append((d.strftime("%Y-%m-%d"), float(px)))
    return out


def _rebuild_for(session_date: str, rebalance_rows: list) -> object | None:
    """The MonthlyRebalance audit row governing a type-B ledger row.

    The B row's ``session_date`` is the rebuild's FILL session (the first
    session AFTER the governing month-end — app/selllayer.py writes
    ``as_of = f_date``), while ``rebal_date`` stores the governing month-end
    (``me_ts``) and ``rebal_month`` its calendar month — so a month-string join
    (``session_date[:7] == rebal_month``) never matches live rows. The join is
    by the rebuild's identity instead: the row with the GREATEST ``rebal_date``
    (date part) strictly before the row's ``session_date``, preferring rows
    with ``sell_layer NOT NULL``, then the highest epoch, then the highest id
    (the highest-epoch-wins dedupe)."""
    cands = []
    for rb in rebalance_rows:
        rd = pd.Timestamp(rb.rebal_date)
        if rd.tzinfo is not None:
            rd = rd.tz_localize(None)
        rd_str = rd.strftime("%Y-%m-%d")
        if rd_str < session_date:
            cands.append((rd_str, rb))
    if not cands:
        return None
    latest = max(c[0] for c in cands)
    same_day = [rb for rd_str, rb in cands if rd_str == latest]
    # prefer sell_layer NOT NULL, then the highest epoch, then the highest id
    return max(same_day, key=lambda rb: (rb.sell_layer is not None, rb.epoch, rb.id))


def _next_rebuild_fill(days: list[str], anchor: str) -> str | None:
    """The next month-end rebuild's fill session after ``anchor``: the first
    stored session after the first COMPLETED month-end later than the
    month-end governing ``anchor`` (the ``sellreview._next_rebuild`` /
    ``_book_plumbing`` convention). None while the next rebuild's session is
    not stored yet — the window is PENDING."""
    if not days:
        return None
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    months = monthly_mod.completed_month_ends(idx, idx[0], idx[-1])
    a = pd.Timestamp(anchor)
    me0 = max((m for m in months if m <= a), default=None)
    if me0 is None:
        return None
    later = [m for m in months if m > me0]
    if not later:
        return None
    me2_str = later[0].strftime("%Y-%m-%d")
    return next((d for d in days if d > me2_str), None)


# ---------------------------------------------------------------------------
# Per-type scorers (each returns a row result dict; the caller counts + skips
# on exceptions — a single row's problems never raise)
# ---------------------------------------------------------------------------

def _score_a(row, candles: dict, sells: dict, costs: reval.CostConfig) -> dict:
    """Type A: the deferred path vs sold-at-level + cash yield, over
    [decision session → the earlier of the next SELL fill / the 63-session
    cap]."""
    tk = row.ticker
    branch = _json(row.engine_branch)
    ctx = _json(row.context)
    series = candles.get(tk) or {}
    dates = sorted(series)
    s_date = row.session_date
    if s_date not in dates:
        raise ValueError(f"no stored bar at the decision session {s_date} for {tk}")
    fill_px = branch.get("fill_price", ctx.get("fill_price", ctx.get("proposal_price")))
    if fill_px is None:
        raise ValueError("no decision fill price in engine_branch/context")
    fill_px = float(fill_px)
    rate = costs.rate_for(tk, "stock")
    d_idx = dates.index(s_date)
    last_idx = len(dates) - 1
    cap_idx = d_idx + TIME_CAP_SESSIONS
    next_sell = next(((d, p) for d, p in sells.get(tk, []) if d > s_date), None)
    sell_idx = dates.index(next_sell[0]) if next_sell and next_sell[0] in dates else None
    if sell_idx is not None and sell_idx <= cap_idx:
        w_end, exit_px, exit_kind = sell_idx, next_sell[1], "sold at the next SimTrade SELL fill"
    elif last_idx >= cap_idx:
        w_end = cap_idx
        exit_px = next((series[dates[j]]["close"] for j in range(cap_idx, d_idx, -1)
                        if series[dates[j]]["close"] is not None), None)
        exit_kind = (f"time cap ({TIME_CAP_SESSIONS} sessions) — marked at the last "
                     "available close")
    else:
        return {"status": "pending",
                "window": {"start": s_date, "end": None,
                           "reason": "cap session not yet reached and no exit fill yet"}}
    if exit_px is None:
        raise ValueError("no mark at the window end")
    w = w_end - d_idx
    # cash yield accrues COMPOUNDED per session: (1+cy)**(w/252) — the reval
    # convention the REFERENCE_BARS were measured under
    engine_ret = (1.0 - rate) * (1.0 + costs.cash_yield_annual) ** (
        w / reval.TRADING_DAYS_PER_YEAR) - 1.0
    window = {"start": s_date, "end": dates[w_end], "sessions": w,
              "exit_kind": exit_kind, "exit_price": exit_px}
    if row.decision != DIVERGENT_DECISION["A"]:
        return {"status": "scored", "delta": 0.0, "branch": "confirm (engine's path)",
                "window": window}
    llm_ret = exit_px / fill_px - 1.0 - rate
    return {"status": "scored", "delta": llm_ret - engine_ret, "branch": "defer",
            "window": window,
            "detail": {"fill_price": fill_px, "llm_ret": llm_ret, "engine_ret": engine_ret}}


async def _score_b(row, rebalance_rows: list, days: list[str], open_fn: OpenFn,
                   costs: reval.CostConfig) -> dict:
    """Type B: the retained name vs the DISPLACED addition over [this
    rebuild's fill → the next rebuild's fill]."""
    tk = row.ticker
    s_date = row.session_date
    f2 = _next_rebuild_fill(days, s_date)
    if f2 is None:
        return {"status": "pending",
                "window": {"start": s_date, "end": None,
                           "reason": "the next rebuild's fill session is not stored yet"}}
    window = {"start": s_date, "end": f2, "sessions": days.index(f2) - days.index(s_date)}
    if row.decision != DIVERGENT_DECISION["B"]:
        return {"status": "scored", "delta": 0.0, "branch": "confirm (engine's path)",
                "window": window}
    rb = _rebuild_for(s_date, rebalance_rows)
    if rb is None:
        raise ValueError(f"no MonthlyRebalance audit row before the fill session {s_date}")
    layer = _json(rb.sell_layer)
    engine_picks = layer.get("engine_picks") or [t for t in rb.picked.split(",") if t]
    executed_picks = layer.get("executed_picks") or engine_picks
    held_before = {t for t in rb.held_before.split(",") if t}
    displaced = [t for t in engine_picks if t not in set(executed_picks)]
    rate = costs.rate_for(tk, "stock")
    additions = [t for t in engine_picks if t not in held_before]

    async def _opens(names: list[str]) -> dict[str, dict[str, float | None]]:
        e = await open_fn(s_date, names)
        x = await open_fn(f2, names)
        return {t: {"entry": e.get(t), "exit": x.get(t)} for t in names}

    if len(displaced) == 1:
        px = await _opens(sorted({tk, displaced[0]}))
        if px[tk]["entry"] is None or px[tk]["exit"] is None:
            return {"status": "pending",
                    "window": {**window, "reason": "no executable open for the retained name "
                                                  "at a governing session"}}
        if px[displaced[0]]["entry"] is None or px[displaced[0]]["exit"] is None:
            return {"status": "pending",
                    "window": {**window, "reason": f"no executable open for the displaced add "
                                                  f"{displaced[0]}"}}
        llm_net = px[tk]["exit"] / px[tk]["entry"] - 1.0 - 2.0 * rate
        engine_net = (px[displaced[0]]["exit"] / px[displaced[0]]["entry"] - 1.0 - 2.0 * rate)
        engine_kind = f"displaced add {displaced[0]}"
    else:
        # Declared fallback: the equal-weight mean of the names newly added at
        # the same rebuild (engine picks not held before); cash yield when the
        # rebuild added nothing.
        if not additions:
            w = window["sessions"]
            px_tk = await _opens([tk])
            if px_tk[tk]["entry"] is None or px_tk[tk]["exit"] is None:
                return {"status": "pending",
                        "window": {**window, "reason": "no executable open for the retained name "
                                                      "at a governing session"}}
            llm_net = px_tk[tk]["exit"] / px_tk[tk]["entry"] - 1.0 - 2.0 * rate
            engine_net = ((1.0 + costs.cash_yield_annual) ** (w / reval.TRADING_DAYS_PER_YEAR)
                          - 1.0 - rate)
            engine_kind = "cash yield, no additions (declared fallback)"
        else:
            px = await _opens(sorted({tk, *additions}))
            if px[tk]["entry"] is None or px[tk]["exit"] is None:
                return {"status": "pending",
                        "window": {**window, "reason": "no executable open for the retained name "
                                                      "at a governing session"}}
            legs = []
            for t in additions:
                if px[t]["entry"] is None or px[t]["exit"] is None:
                    return {"status": "pending",
                            "window": {**window, "reason": f"no executable open for the fallback "
                                                          f"add {t}"}}
                legs.append(px[t]["exit"] / px[t]["entry"] - 1.0 - 2.0 * rate)
            llm_net = px[tk]["exit"] / px[tk]["entry"] - 1.0 - 2.0 * rate
            engine_net = sum(legs) / len(legs)
            engine_kind = "equal-weight mean of the new adds (declared fallback)"
    return {"status": "scored", "delta": llm_net - engine_net, "branch": "defer",
            "window": window,
            "detail": {"retained": tk, "engine_branch": engine_kind,
                       "llm_net": llm_net, "engine_net": engine_net}}


async def _score_c(row, days: list[str], open_fn: OpenFn,
                   costs: reval.CostConfig) -> dict:
    """Type C: the early exit + cash yield vs held-to-next-rebuild, over
    [the row's EXIT fill → the next month-end rebuild's fill]."""
    tk = row.ticker
    rate = costs.rate_for(tk, "stock")
    if row.decision != DIVERGENT_DECISION["C"]:
        f2 = _next_rebuild_fill(days, row.session_date)
        if f2 is None:
            return {"status": "pending",
                    "window": {"start": row.session_date, "end": None,
                               "reason": "the next rebuild's fill session is not stored yet"}}
        return {"status": "scored", "delta": 0.0, "branch": "hold (engine's path)",
                "window": {"start": row.session_date, "end": f2,
                           "sessions": days.index(f2) - days.index(row.session_date)}}
    if row.price is None or row.filled_at is None:
        return {"status": "pending",
                "window": {"start": row.session_date, "end": None,
                           "reason": "the EXIT has not filled yet (fill columns NULL)"}}
    fill_d = pd.Timestamp(row.filled_at)
    if fill_d.tzinfo is not None:
        fill_d = fill_d.tz_localize(None)
    fill_d = fill_d.strftime("%Y-%m-%d")
    fill_px = float(row.price)
    if fill_d not in days:
        raise ValueError(f"the EXIT fill date {fill_d} has no stored session")
    f2 = _next_rebuild_fill(days, fill_d)
    if f2 is None:
        return {"status": "pending",
                "window": {"start": fill_d, "end": None,
                           "reason": "the next rebuild's fill session is not stored yet"}}
    w = days.index(f2) - days.index(fill_d)
    window = {"start": fill_d, "end": f2, "sessions": w}
    m = await open_fn(f2, [tk])
    exit_px = m.get(tk)
    if exit_px is None:
        return {"status": "pending",
                "window": {**window,
                           "reason": "no executable price at the next rebuild — the held leg "
                                     "is not realizable yet"}}
    llm_ret = (1.0 - rate) * (1.0 + costs.cash_yield_annual) ** (
        w / reval.TRADING_DAYS_PER_YEAR) - 1.0
    engine_ret = exit_px / fill_px - 1.0 - rate
    return {"status": "scored", "delta": llm_ret - engine_ret, "branch": "exit",
            "window": window,
            "detail": {"fill_price": fill_px, "exit_price": exit_px,
                       "llm_ret": llm_ret, "engine_ret": engine_ret}}


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

async def score_sell_decisions(db=None, *, now: datetime | None = None) -> dict:
    """Compute the sell scorecard ON DEMAND from the stored ledger + prices.

    READ-ONLY on the trade tables (no writes, no new columns); deterministic
    byte-identical modulo the wall-clock ``generated_at`` field; never raises
    on a single row's problems (counted + skipped with a reason). ``db`` is an
    optional session factory override (default: the current database's
    ``db.Session``); ``now`` pins the wall-clock field (tests).
    """
    costs = reval.CostConfig()
    session_factory = db if db is not None else db_mod.Session
    generated_at = (now or datetime.now(UTC)).isoformat()

    async with session_factory() as s:
        rows = (await s.scalars(
            select(LlmSellDecision).order_by(LlmSellDecision.decided_at, LlmSellDecision.id))).all()
        rebalance_rows = (await s.scalars(
            select(MonthlyRebalance).order_by(MonthlyRebalance.rebal_date,
                                              MonthlyRebalance.id))).all()

    # The displaced-add join reads each B row's rebuild from
    # MonthlyRebalance.sell_layer (see _rebuild_for — joined by the rebuild's
    # identity: greatest rebal_date strictly before the row's fill session).
    rebalance_names: set[str] = set()
    for rb in rebalance_rows:
        rebalance_names.update(t for t in rb.picked.split(",") if t)
        rebalance_names.update(t for t in rb.held_before.split(",") if t)
        layer = _json(rb.sell_layer)
        for key in ("engine_picks", "executed_picks"):
            rebalance_names.update(layer.get(key) or [])

    tickers = sorted({r.ticker for r in rows} | rebalance_names)
    candles = await _load_candles(session_factory, tickers)
    days = await _session_calendar(session_factory)
    a_tickers = sorted({r.ticker for r in rows if r.event_type == "A"})
    sells = await _next_sell_fills(session_factory, a_tickers)
    data_as_of = days[-1] if days else None

    async def open_fn(day: str, names: list[str]) -> dict[str, float | None]:
        """The strict exact-date USD open of session ``day`` per name —
        ``monthly._open_usd_map``, the rebuild's own fill convention."""
        return await monthly_mod._open_usd_map(sorted(set(names)), pd.Timestamp(day))

    per_type: dict[str, dict] = {t: {"rows": [], "scored": [], "pending": 0,
                                      "skipped": [], "n_fail_safe": 0}
                                  for t in ("A", "B", "C")}
    decisions: list[dict] = []
    n_d_shadow = 0
    for row in rows:
        et = row.event_type
        base = {
            "event_id": row.event_id, "session_date": row.session_date,
            "decided_at": row.decided_at.isoformat(), "account": row.engine,
            "event_type": row.event_type, "ticker": row.ticker,
            "decision": row.decision, "parsed": bool(row.parsed),
            "fail_safe": bool(row.fail_safe), "fail_reason": row.fail_reason,
            "rationale": (row.rationale or "")[:RATIONALE_LEN],
        }
        if et not in per_type:
            # Type-D shadow rows (and any OTHER unknown event_type) are
            # visible and honest in the history but NEVER scored, NEVER
            # bucketed into A/B/C, NEVER counted in the per-type or overall
            # stats — the old silent else-"A" fallback let D rows pollute
            # the A bucket.
            if et == "D":
                n_d_shadow += 1
                decisions.append({**base, "status": "shadow", "delta": None,
                                  "window": None, "branch": None,
                                  "detail": None,
                                  "skip_reason": "type-D shadow row — scoring "
                                                 "pending the D scorer"})
            else:
                decisions.append({**base, "status": "skipped", "delta": None,
                                  "window": None, "branch": None,
                                  "detail": None,
                                  "skip_reason": f"unknown event_type {et!r} — "
                                                 "not scored"})
            continue
        bucket = per_type[et]
        bucket["rows"].append(row)
        if row.fail_safe:
            bucket["n_fail_safe"] += 1
        try:
            if et == "A":
                res = _score_a(row, candles, sells, costs)
            elif et == "B":
                res = await _score_b(row, rebalance_rows, days, open_fn, costs)
            else:
                res = await _score_c(row, days, open_fn, costs)
        except Exception as e:  # noqa: BLE001 — one row's problems never raise
            logger.warning("sellscore: skipping %s (%s: %s)", row.event_id,
                           type(e).__name__, e)
            res = {"status": "skipped", "reason": f"{type(e).__name__}: {e}"}
        entry = {**base, "status": res["status"],
                 "delta": res.get("delta"), "window": res.get("window"),
                 "branch": res.get("branch"), "detail": res.get("detail"),
                 "skip_reason": res.get("reason") or (res.get("window") or {}).get("reason")}
        decisions.append(entry)
        if res["status"] == "scored":
            bucket["scored"].append(entry)
        elif res["status"] == "pending":
            bucket["pending"] += 1
        else:
            bucket["skipped"].append({"event_id": row.event_id,
                                      "reason": res.get("reason", "")})

    def _type_block(bucket: dict, bar: dict) -> dict:
        scored = bucket["scored"]
        divergent = [r for r in scored if r["delta"] is not None and r["delta"] != 0.0]
        by_year: dict[str, list[dict]] = {}
        for r in divergent:
            by_year.setdefault(r["session_date"][:4], []).append(r)
        n_rows = len(bucket["rows"])
        return {
            "n_rows": n_rows,
            "n_decided": len(scored),
            "n_pending": bucket["pending"],
            "n_skipped": len(bucket["skipped"]),
            "skipped": bucket["skipped"],
            "n_fail_safe": bucket["n_fail_safe"],
            "fail_safe_rate": (bucket["n_fail_safe"] / n_rows) if n_rows else None,
            "n_divergent": len(divergent),
            "aggregate": sellreview._agg_deltas(divergent, "delta"),
            "by_year": {y: sellreview._agg_deltas(rs, "delta")
                        for y, rs in sorted(by_year.items())},
            "events": scored,
            "reference_bar": bar,
        }

    types_out = {t: _type_block(per_type[t], sellreview.REFERENCE_BARS[t])
                 for t in ("A", "B", "C")}
    all_rows = [r for t in ("A", "B", "C") for r in per_type[t]["rows"]]
    all_scored = [r for t in ("A", "B", "C") for r in per_type[t]["scored"]]
    overall_bucket = {"rows": all_rows, "scored": all_scored,
                      "pending": sum(per_type[t]["pending"] for t in per_type),
                      "skipped": [s for t in per_type for s in per_type[t]["skipped"]],
                      "n_fail_safe": sum(per_type[t]["n_fail_safe"] for t in per_type)}
    overall = _type_block(overall_bucket, sellreview.REFERENCE_BARS)

    # The recent decision history: the last HISTORY_LIMIT rows, newest first.
    history = sorted(decisions, key=lambda d: (d["decided_at"], d["event_id"]),
                     reverse=True)[:HISTORY_LIMIT]

    return {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "data_as_of": data_as_of,
        "read_only": True,
        "n_d_shadow": n_d_shadow,
        "costs": {
            "stock_oneway_bps": costs.stock_oneway_bps,
            "etf_oneway_bps": costs.etf_oneway_bps,
            "fx_oneway_bps": costs.fx_oneway_bps,
            "cash_yield_annual": costs.cash_yield_annual,
            "convention": ("reval.CostConfig() declared rates, imported from reval — the same "
                           "stock bps + cash yield the study used; cash yield accrues COMPOUNDED "
                           "per session, (1+cash_yield_annual)**(w/252) over a window of w "
                           "sessions (the reval apply_cash_yield convention, shared with the "
                           "REFERENCE_BARS); type A: one-way cost at each branch's own exit, "
                           "engine proceeds at the cash yield over the window; type B: both legs "
                           "of both branches net of the one-way cost (cancels in the delta); "
                           "type C: one-way cost at the early exit and at the next rebuild's "
                           "would-be exit, proceeds at the cash yield"),
        },
        "delta_definition": ("realized LLM branch minus engine branch per completed-window "
                             "decision, as a fraction of the decision fill (0.0293 = +2.93pp; "
                             "the reference bars' *_pp fields share this unit); CONFIRM/HOLD rows "
                             "and fail-safes are no-divergence (delta 0) and are excluded from "
                             "n_divergent and from the delta aggregates"),
        "completed_window_rule": sellreview.COMPLETED_WINDOW_RULE,
        "small_n_caveat": sellreview.SMALL_N_CAVEAT,
        "fail_safe_note": ("the fail-safe rate (fail_safe / n_rows, per type and overall) is "
                           "itself a verdict on the reviewer — fail-safes score as the engine's "
                           "own branch"),
        "reference_bars": sellreview.REFERENCE_BARS,
        "types": types_out,
        "overall": overall,
        "decisions": history,
        "determinism": ("byte-identical for identical stored data and identical now EXCEPT "
                        "generated_at (the declared wall-clock exclusion)"),
    }
