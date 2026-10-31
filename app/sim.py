"""Autonomous paper-trading simulation engine for Trade Sentinel.

The bot receives an imaginary monthly allowance, decides when to buy/sell
assets from a configurable universe using deterministic signals (the LLM
sell layer, app/selllayer.py, overlays the standing-stop exits), and tracks
portfolio performance over time. The former llm/hybrid whole-cycle strategy
modes are retired: a stale SIM_STRATEGY value degrades to deterministic via
the unknown-strategy fallback in run_cycle.

Execution follows the session fill contract — documented in full in
``app/session.py``, the single place it lives. Short version: the scheduled
23:30 UTC cycle decides on data up to the PREVIOUS completed session's close
(S) and fills at the CURRENT session's (F's) open — strict same-session
bars, no stale or forward-filled fills; a stop level known at S's close is
a standing order for F (open-gap fills at the open, intrabar breaches fill
at the level); the percent stop is frozen at entry and persisted; the ATR
stop is a position-specific monotone ratchet from the entry date; marks
use F's close; and a per-session guard makes a second cycle in the same
session a no-op. No real broker, no intraday orders, no shorting. This is
a research/education tool.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from zoneinfo import ZoneInfo
from collections.abc import Callable
from typing import Any

from sqlalchemy import delete, func, or_, select

import pandas as pd

from .analysis import MIN_CANDLES, compute, signal_series
from .config import settings
from . import llm as llm_mod
from . import monthly as monthly_mod
from . import run_config
from . import session as session_mod
from .db import (
    SimAccount,
    SimAllowance,
    SimBenchmarkAccount,
    SimBenchmarkSnapshot,
    SimChatMessage,
    SimLlmDecision,
    SimPosition,
    SimSnapshot,
    SimTrade,
    Session,
    current_db as db_current_db,
)
from .market import candles, latest_close, refresh, refresh_many
from .screener import tickers as universe_tickers
from .strategy import (
    _bar_price,
    StrategyParams,
    propose_trades,
)

logger = logging.getLogger("trade_sentinel.sim")

# The RunConfig registry's engine key for this portfolio (app/run_config.py).
_ENGINE = "sim"

# The run-configuration CONTRACT version. BUMP RULE: bump when the snapshot's
# field set/meaning changes, or when a DECISION CONSTANT/formula changes (the
# snapshot's ``code`` block records the constants; a behavioural change not
# expressible as a value is only visible through this version). A config
# recorded by an older contract must never be read as equivalent to a newer
# one.
_CONFIG_CONTRACT_VERSION = "phase2-v1"

# The LOGICAL RUN's frozen paper stamp ({"mode","config_id","epoch"}).
# run_cycle sets it for its duration (the chat confirm path threads its stamp
# explicitly instead); the deep
# executors (_exec_buy/_exec_sell and the snapshot writer) read it instead of
# resolving per leg. A ContextVar (not a module global) keeps concurrent runs
# isolated and propagates through awaited calls in the same task; an explicit
# writer kwarg always wins, so the public mutators stay callable standalone.
_RUN_STAMP: ContextVar[dict | None] = ContextVar("sim_run_stamp", default=None)

# Stores the raw LLM reasoning text from the most recent whole-cycle LLM call.
# The llm/hybrid whole-cycle modes are retired, so run_cycle resets this to ""
# every cycle and it stays empty; the chat flow never sets it.
_last_llm_reasoning: str = ""
# Stores the prose summary the LLM emitted before its JSON array. Same
# deterministic-only status as _last_llm_reasoning.
_last_llm_summary: str = ""
# Stores the deterministic trades from the most recent cycle (the executed
# trades in the deterministic strategy — the only live strategy).
_last_deterministic_trades: list[dict] = []
# Stores the parsed LLM decisions from the most recent cycle. Empty in the
# deterministic-only world (run_cycle resets it); the chat context reads it.
_last_llm_decisions: list[dict] = []
# Retired with the hybrid mode; kept (always empty) because run_cycle's
# per-cycle state reset and its test pin it.
_last_llm_vetoes: list[dict] = []

# In-progress run-cycle state for the frontend status poller. Cleared at the
# start of each run_cycle() and updated at each stage; read by /api/sim/run-status.
# Shape: {"running": bool, "stage": str, "started_at": iso, "updated_at": iso,
#         "detail": str, "error": str|None}
_run_progress: dict[str, Any] = {"running": False, "stage": "", "started_at": "", "updated_at": "", "detail": "", "error": None}

# Serializes run_cycle() calls so the scheduler and a manual "Run Bot Now"
# click can't execute simultaneously. If a cycle is already running, a
# scheduled/deferred call WAITS (bounded below) instead of dropping the day's
# cycle; the manual HTTP path keeps its own fail-fast pre-check
# (start_run_cycle_background).
_run_cycle_lock: asyncio.Lock = asyncio.Lock()

# Upper bound on how long a scheduled/deferred call waits for the busy lock
# before giving up with the old "skipped" semantics (a DEV backfill can hold
# it for minutes; a wedged holder should not park the scheduler forever).
_TICK_LOCK_WAIT_SECONDS = 7200

# Serialises every STATE/CASH/HISTORY mutation against a reset's boundary (the
# stamp/reset TOCTOU): a stamp captured before a reset is validated against the
# CURRENT epoch INSIDE this lock and its transaction, and refuses without
# writing into the new epoch's book. Lock order is ALWAYS cycle-lock ->
# state-lock (never the reverse); the chat path takes only this lock per leg,
# so a long LLM turn never blocks a reset.
_state_write_lock: asyncio.Lock = asyncio.Lock()
# Holds the asyncio.Task for the currently in-flight run_cycle() (whether
# triggered manually or by the scheduler). Used by the decoupled
# /api/sim/run endpoint so the cycle survives browser disconnects.
_run_cycle_task: asyncio.Task | None = None

# How many candles to refresh for the sim universe (2y is a good balance
# for indicator computation without excessive API load).
_SIM_REFRESH_PERIOD = "2y"

# ---------------------------------------------------------------------------
# Per-session fill guard (session fill contract, rule 2)
# ---------------------------------------------------------------------------
# Exactly ONE decision/fill cycle runs per session: a second cycle in the
# same session (a scheduled retry, a manual "Run Bot Now" after the
# scheduled run, a process restart) must not double-fill. The guard is a
# small JSON state file next to the other sims' state files (e.g.
# daily_core_state.json) rather than a new DB column — the additive
# DB-migration surface stays untouched, and tests can point the file at a
# temp path. It is stamped just BEFORE the decision/fill step (a crash
# mid-cycle must never lead to a second fill pass) and reset by backfill /
# reset_sim, which replace the whole portfolio and must leave the next
# cycle free.
_STATE_FILE = Path("/data/sim_state.json")          # PROD (live book)
_STATE_FILE_DEV = Path("/data/sim_state_dev.json")   # DEV sandbox
_state_lock = threading.Lock()


def _state_file() -> Path:
    """Per-DB state file (same convention as daily_core): the session guard
    is stamped by BOTH the live scheduler and DEV experiments, so a DEV run
    stamping the shared file would make the PROD tick silently skip the
    session (and vice versa). The globals stay patchable — tests point
    ``_STATE_FILE`` at a temp path."""
    return _STATE_FILE_DEV if db_current_db() == "dev" else _STATE_FILE


def _load_state() -> dict:
    path = _state_file()
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("Could not read sim state file %s: %s", path, e)
        return {}


def _save_state(state: dict) -> None:
    path = _state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a concurrent reader never sees a half-written file.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("Could not write sim state file %s: %s", path, e)


def _session_guard_key() -> str:
    """The guard key for the cycle's own session F (the run date, UTC)."""
    return _utcnow().date().isoformat()


def _session_guard_passed(key: str) -> bool:
    with _state_lock:
        return _load_state().get("last_session") != key


def _stamp_session_guard(key: str) -> None:
    with _state_lock:
        state = _load_state()
        state["last_session"] = key
        _save_state(state)


def _reset_session_guard() -> None:
    with _state_lock:
        state = _load_state()
        state.pop("last_session", None)
        _save_state(state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Used only for _current_month(): the monthly allowance is a calendar-month
# concept, so we anchor it to the operator's local timezone. Shared with the
# monthly qv-mom portfolio (allowance_tz) so both contributed figures step on
# the same calendar boundary and stay comparable.
_TZ = ZoneInfo(settings.allowance_tz)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _current_month() -> str:
    return datetime.now(_TZ).strftime("%Y-%m")


def _week_diff(current: str, last: str) -> int:
    """Calendar-week distance between two ISO 'YYYY-Www' keys (>= 0).

    Anchors each key to the Thursday of its ISO week and diffs those
    dates, so year boundaries are counted correctly (e.g.
    2026-W02 - 2025-W50 == 4, not 5). ``last`` is assumed to be <= ``current``.
    """
    cy, cw = (int(x) for x in current.split("-W"))
    ly, lw = (int(x) for x in last.split("-W"))
    jan4 = datetime(cy, 1, 4)
    cur_thu = jan4 - timedelta(days=(jan4.weekday() - 3) % 7) + timedelta(weeks=cw - 1)
    jan4 = datetime(ly, 1, 4)
    last_thu = jan4 - timedelta(days=(jan4.weekday() - 3) % 7) + timedelta(weeks=lw - 1)
    return round((cur_thu - last_thu).days / 7)


async def _latest_close(ticker: str) -> float | None:
    """Return the most recent cached close price for *ticker*, or None.

    Delegates to market.latest_close (single-row query) — the old path loaded
    the ticker's entire history via candles() just to read the last bar."""
    return await latest_close(ticker)


async def held_tickers() -> list[str]:
    """Tickers currently held in the sim portfolio, plus the benchmark ticker."""
    async with Session() as s:
        positions = (await s.scalars(select(SimPosition).order_by(SimPosition.ticker))).all()
    tickers = [p.ticker for p in positions]
    if settings.sim_benchmark_enabled and settings.sim_benchmark_ticker:
        if settings.sim_benchmark_ticker not in tickers:
            tickers.append(settings.sim_benchmark_ticker)
    return tickers


async def _account() -> SimAccount:
    """Return the singleton SimAccount row, creating it if necessary."""
    async with Session() as s:
        acc = await s.get(SimAccount, 1)
        if acc is None:
            acc = SimAccount(id=1, cash=settings.sim_start_cash,
                             last_allowance_month=None,
                             opening_seed=settings.sim_start_cash)
            s.add(acc)
            await s.commit()
        return acc


# ---------------------------------------------------------------------------
# Allowance
# ---------------------------------------------------------------------------

async def deposit_allowance(*, mode: str | None = None, config_id: int | None = None,
                            epoch: int | None = None) -> dict[str, Any]:
    """Deposit the monthly allowance if a new month has begun.

    Returns a dict describing whether a deposit was made and the resulting
    cash balance.

    Forward PAPER record: stamped with the logical run's frozen stamp when one
    is threaded (``run_cycle``), else with a freshly resolved paper stamp —
    the status endpoint calls this standalone, and a standalone deposit is
    still a forward record.

    EPOCH-AWARE GUARD: the month marker is current STATE (a reset clears it),
    so it alone cannot prove the month was funded in THIS epoch — a closed
    epoch's same-month row is preserved on purpose. Before inserting, the
    guard verifies no allowance row exists for the month under the current
    epoch: a marker that lagged its row heals instead of duplicating (the
    rebuilt UNIQUE(month, epoch) is the last-resort backstop), while a
    post-reset same-month deposit writes its own new-epoch row — a NEW
    imaginary allocation in the new epoch's ledger, not a transfer of the old
    epoch's money and not real capital.
    """
    month = _current_month()
    async with _state_write_lock:
        return await _deposit_allowance_locked(month, mode=mode,
                                               config_id=config_id, epoch=epoch)


async def _deposit_allowance_locked(month: str, *, mode: str | None,
                                    config_id: int | None,
                                    epoch: int | None) -> dict[str, Any]:
    stamp = await _resolve_paper_stamp(mode=mode, config_id=config_id, epoch=epoch)
    async with Session() as s:
        if not await _stamp_is_current(s, stamp):
            acc = await s.get(SimAccount, 1)
            logger.warning("Sim deposit refused: stale epoch stamp %s", stamp["epoch"])
            return {"deposited": False, "month": month,
                    "cash": acc.cash if acc is not None else None,
                    "refused": "stale epoch stamp"}
        acc = await s.get(SimAccount, 1)
        if acc is None:
            acc = SimAccount(id=1, cash=settings.sim_start_cash,
                             last_allowance_month=None,
                             opening_seed=settings.sim_start_cash)
            s.add(acc)

        if acc.last_allowance_month == month:
            return {"deposited": False, "month": month, "cash": acc.cash}
        exists = await s.scalar(
            select(func.count()).select_from(SimAllowance).where(
                SimAllowance.month == month,
                run_config.epoch_clause(SimAllowance, stamp["epoch"])))
        if exists:
            acc.last_allowance_month = month
            await s.commit()
            logger.info("Sim allowance for %s already deposited (epoch %s)",
                        month, stamp["epoch"])
            return {"deposited": False, "month": month, "cash": acc.cash}

        acc.cash += settings.sim_monthly_allowance
        acc.last_allowance_month = month
        s.add(SimAllowance(amount=settings.sim_monthly_allowance, month=month,
                           mode=stamp["mode"], config_id=stamp["config_id"],
                           epoch=stamp["epoch"]))
        await s.commit()

        logger.info("Sim allowance deposited: %.2f for %s → cash %.2f",
                     settings.sim_monthly_allowance, month, acc.cash)
        return {"deposited": True, "amount": settings.sim_monthly_allowance, "month": month, "cash": acc.cash}


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

async def valuate() -> dict[str, Any]:
    """Compute the current portfolio valuation.

    Returns::

        {
            "cash": float,
            "positions": [{"ticker","shares","avg_cost","current_price","value","pnl_pct"}],
            "positions_value": float,
            "total_equity": float,
            "allowance_total": float,
        }

    EPOCH-SCOPED contributed total ("seed capital + monthly allowances"):
    ``allowance_total`` = the epoch's opening SEED (the epoch's FROZEN
    ``sim_start_cash`` — see ``_epoch_seed``) + the CURRENT epoch's monthly
    deposits. A nonzero seed must be counted or the contributed baseline is 0
    on the seed-only stretch (undefined XIRR, phantom gain on the seed); it is
    constant per epoch so contribution-adjusted differences cancel, and it is
    FROZEN: changing the setting without a reset does not rewrite an account
    that already holds the old capital. A reset re-seeds with the then-current
    setting, so the fresh book's baseline restarts with its own seed.
    """
    async with Session() as s:
        acc = await s.get(SimAccount, 1)
        if acc is None:
            acc = SimAccount(id=1, cash=settings.sim_start_cash,
                             last_allowance_month=None,
                             opening_seed=settings.sim_start_cash)
            s.add(acc)
            await s.commit()

        positions = (await s.scalars(select(SimPosition).order_by(SimPosition.ticker))).all()
        cur = await run_config.current_epoch(_ENGINE, session=s)
        deposits = (await s.scalar(
            select(func.sum(SimAllowance.amount)).where(
                run_config.epoch_clause(SimAllowance, cur)))) or 0.0
        seed = await _epoch_seed(s)

    positions_value = 0.0
    pos_list: list[dict[str, Any]] = []
    for p in positions:
        price = await _latest_close(p.ticker)
        if price is None:
            # Fallback: use avg cost if no candle data yet
            price = p.avg_cost
        value = p.shares * price
        positions_value += value
        pnl_pct = ((price - p.avg_cost) / p.avg_cost * 100) if p.avg_cost > 0 else 0.0
        pos_list.append({
            "ticker": p.ticker,
            "shares": p.shares,
            "avg_cost": round(p.avg_cost, 4),
            "current_price": round(price, 4),
            "value": round(value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "thesis": p.thesis or "",
            "buy_date": p.opened_at.strftime("%Y-%m-%d") if p.opened_at else "",
        })

    return {
        "cash": round(acc.cash, 2),
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(acc.cash + positions_value, 2),
        # "seed capital + monthly allowances": the epoch's external capital
        # basis (see the docstring). The pre-existing key set stays frozen —
        # the composition is documented, not expanded with new keys.
        "allowance_total": round(seed + float(deposits), 2),
    }


# ---------------------------------------------------------------------------
# Deterministic strategy
# ---------------------------------------------------------------------------

async def _candidate_tickers() -> list[str]:
    """Return the list of tickers the sim should consider.

    If the (runtime or configured) universe is ``watchlist``, use the watchlist
    table; otherwise load from the universe file. The universe is the shared
    Dashboard selection, so it applies to every sim.
    """
    name = _universe()
    if name.lower() == "watchlist":
        from .db import Watchlist
        async with Session() as s:
            rows = (await s.scalars(select(Watchlist).order_by(Watchlist.ticker))).all()
            return [r.ticker for r in rows]
    else:
        return universe_tickers(name)


def _universe() -> str:
    """The runtime-selected sim universe (shared across all sims, persisted in
    the daily-core state file), else the config default. Lazy import avoids any
    import cycle with daily_core."""
    try:
        from . import daily_core
        return daily_core.persisted_universe() or settings.sim_universe
    except Exception:  # noqa: BLE001 — selection must never break a cycle
        return settings.sim_universe


# ---------------------------------------------------------------------------
# Run configuration + provenance stamping (the immutable Phase 2 half)
# ---------------------------------------------------------------------------
#
# RULE (provenance contract): the snapshot must capture EVERY input that
# changes a decision. A configuration that changed a decision but not the
# config hash is a provenance bug. The field list is grep-verified against this
# module's ``settings.*`` reads and the hard-coded strategy constants
# (``code``). ``strategy.py`` reads NO settings itself — every decision
# parameter is passed in from here, so this module's reads are the complete
# sim list.
# Deliberately excluded, with rationale:
#   * ``sim_benchmark_enabled``/``sim_benchmark_ticker`` — the DCA benchmark is
#     a comparison artifact, not a sim decision input.
#   * ``sim_run_hour``/``sim_run_minute`` — scheduler timing.
#   * ``sim_monthly_enabled`` — feature gate.
#   * ``market_fresh_seconds`` — refresh plumbing (the Candle provenance
#     columns cover the data, not the run policy).
#   * ``sim_monthly_rank_boost`` — read only by the research replay
#     (app/optimize.py), never by this live engine.


async def current_config() -> dict:
    """The immutable configuration snapshot for ONE sim run.

    ASYNC ON PURPOSE: the effective universe can be the ``watchlist`` DB table
    (its members ARE the decision universe), so resolving the actual
    membership needs the DB. The builder never raises — an unresolvable
    universe records None members rather than breaking a cycle.
    """
    name = _universe()
    source = "watchlist" if name.lower() == "watchlist" else "file"
    try:
        members = sorted(await _candidate_tickers())
    except Exception:  # noqa: BLE001 — provenance must never break a cycle
        members = None
    universe = {
        "name": name,
        "source": source,
        "members_hash": (hashlib.sha256(",".join(members).encode()).hexdigest()
                         if members is not None else None),
    }
    return {
        "contract_version": _CONFIG_CONTRACT_VERSION,
        "engine": _ENGINE,
        "currency": "USD",
        "fill_basis": "session strict exact-date open (app/session.py rule 2), zero fees",
        "fill_basis_note": (
            "the technical sim's historical replay (app/optimize.py) still "
            "fills each decision at that bar's same close — a documented "
            "legacy approximation, not this engine's live contract"),
        "timezone": settings.allowance_tz,
        # --- effective runtime selection (shared state file) --------------
        # EFFECTIVE only: the raw ``sim_universe`` fallback is not repeated —
        # a persisted selection shadows it, and duplicating it would mint a
        # new "configuration tried" for a setting that changed no decision.
        "universe": universe,
        # The single top-level members key ``run_provenance.summarize`` reads
        # to scope its candle/fundamental provenance query.
        "universe_members": members,
        # --- strategy / risk parameters ----------------------------------
        "sim_strategy": settings.sim_strategy,
        "sim_start_cash": settings.sim_start_cash,
        "sim_monthly_allowance": settings.sim_monthly_allowance,
        "sim_max_position_pct": settings.sim_max_position_pct,
        "sim_min_cash_pct": settings.sim_min_cash_pct,
        "sim_max_positions": settings.sim_max_positions,
        "sim_stop_pct": settings.sim_stop_pct,
        "sim_max_run_5d": settings.sim_max_run_5d,
        "sim_min_run_5d": settings.sim_min_run_5d,
        "sim_max_dist_above": settings.sim_max_dist_above,
        # --- LLM decision controls ---------------------------------------
        "sim_llm_fundamentals_context": settings.sim_llm_fundamentals_context,
        "sim_block_negative_roe": settings.sim_block_negative_roe,
        # --- LLM sell layer (app/selllayer.py) — provenance: these knobs are
        # decision inputs of the live cycle (same rule as above).
        "sim_llm_sell_layer": settings.sim_llm_sell_layer,
        "sell_layer_timeout_s": settings.sell_layer_timeout_s,
        "sell_layer_max_defers": settings.sell_layer_max_defers,
        # --- hard-coded decision constants (see the contract bump rule) ---
        "code": {
            "buy_threshold": 40,
            "sell_threshold": -40,
            "stop_type": "percent",
            "use_atr_stop": True,
            "min_candles": MIN_CANDLES,
        },
    }


async def _paper_stamp(config: dict | None = None, *, record: bool = True) -> dict:
    """Resolve the stamp for ONE forward PAPER run: ``(mode, config_id, epoch)``.

    Resolve ONCE per logical run and thread it (ContextVar or explicit kwargs);
    the fallback path goes through THIS module's ``Session`` factory so a
    monkeypatched in-memory DB is never bypassed.

    ``record=False`` defers the RunExecution to the caller's STABLE INPUT
    POINT: a live cycle refreshes candles after its stamp, so recording at
    stamp time would describe the pre-refresh cache instead of the data the
    decision actually read. The caller must then call ``_record_run`` exactly
    once (standalone writers and the chat turn keep the default: no data
    refresh follows them).
    """
    cfg = await current_config() if config is None else config
    stamp = {
        "mode": "paper",
        "config_id": await run_config.get_or_create(
            _ENGINE, "paper", cfg, session_factory=Session),
        "epoch": await run_config.current_epoch(_ENGINE, session_factory=Session),
    }
    # A stamp scope IS a logical run: record its data provenance exactly once.
    if record:
        await _record_run(stamp, cfg)
    return stamp


async def _record_run(stamp: dict, cfg: dict, *,
                      asof: datetime | None = None,
                      session=None) -> dict:
    """Summarize the run's DATA provenance and append one RunExecution.

    The clock is an explicit ``asof`` or the WALL clock at recording time —
    deliberately NOT the injectable decision clock (``_utcnow``), so recording
    provenance can never consume or perturb a frozen decision/fill timestamp
    (session fill contract rule 7: ``decided_at`` is the FIRST clock capture
    of a decision turn); in production the two coincide. The summary is an
    upper-bounded aggregate of the candle/fundamental cache the run could
    read, unknown provenance preserved — NOT an immutable data snapshot.
    """
    from . import run_provenance
    clock = datetime.now(UTC) if asof is None else asof
    # A caller-owned session keeps the execution row in the caller's
    # transaction (the helpers never commit it): a backfill's rows and its
    # execution record then commit — or roll back on a late forward-record
    # refusal — atomically.
    scope = {"session": session} if session is not None else {"session_factory": Session}
    summary = await run_provenance.summarize(cfg, asof=clock, **scope)
    # The run's capital basis: the account's frozen opening seed, or an
    # explicit UNKNOWN for a legacy account (never an inferred value).
    if session is not None:
        acc = await session.get(SimAccount, 1)
    else:
        async with Session() as meta_s:
            acc = await meta_s.get(SimAccount, 1)
    opening = None if acc is None else acc.opening_seed
    summary["opening_seed"] = {
        "value": opening,
        "known": opening is not None,
        "basis": ("account_opening_seed" if opening is not None
                  else "conservative_zero_legacy_unknown"),
    }
    await run_config.record_execution(
        _ENGINE, stamp["mode"], stamp["config_id"], stamp["epoch"], summary,
        data_asof=clock, **scope)
    return summary


async def _stamp_is_current(s, stamp: dict) -> bool:
    """True when the stamp's epoch is the CURRENT epoch (read in the caller's
    session, i.e. under the state lock + its transaction)."""
    return await run_config.current_epoch(_ENGINE, session=s) == stamp["epoch"]


async def _resolve_paper_stamp(*, mode: str | None = None,
                               config_id: int | None = None,
                               epoch: int | None = None) -> dict:
    """The stamp a writer uses: explicit kwargs win, then the running logical
    run's frozen ContextVar stamp, else a freshly resolved PAPER stamp.

    Standalone public writers (status-endpoint calls, a hand-run executor) are
    forward paper writes and are stamped ``paper`` by default; the all-None
    NULL default exists only for direct model construction. A non-paper mode
    always passes its full stamp (the backfill writes its own rows).
    """
    if mode is not None and config_id is not None and epoch is not None:
        return {"mode": mode, "config_id": config_id, "epoch": epoch}
    ctx = _RUN_STAMP.get() or {}
    out = {
        "mode": mode if mode is not None else ctx.get("mode") or "paper",
        "config_id": config_id if config_id is not None else ctx.get("config_id"),
        "epoch": epoch if epoch is not None else ctx.get("epoch"),
    }
    if out["config_id"] is None or out["epoch"] is None:
        fresh = await _paper_stamp()
        out["config_id"] = fresh["config_id"] if out["config_id"] is None else out["config_id"]
        out["epoch"] = fresh["epoch"] if out["epoch"] is None else out["epoch"]
    return out


async def _config_changed(frozen: dict) -> bool:
    """True when the runtime configuration no longer matches the one this run
    froze at its start (best-effort mid-run guard; see
    ``daily_core._config_changed``). Async because this engine's universe can
    be the watchlist table."""
    return (run_config.config_hash(await current_config())
            != run_config.config_hash(frozen))


async def _epoch_seed(s) -> float:
    """The opening SEED capital of the CURRENT epoch — the account's own
    ``opening_seed``, FROZEN when the account was opened/re-seeded.

    A live-setting read would retroactively rewrite the contributed basis when
    ``sim_start_cash`` changes without a reset (the account still holds the old
    capital). The field is written in the SAME transaction that
    creates/re-seeds the account and by the historical replay (the sim
    backfill sets ``params.start_cash`` — this engine's synthetic baseline
    really is that seed). A LEGACY account (NULL) has an UNKNOWN seed: the
    conservative fallback is 0.0 and ``_record_run`` flags
    ``opening_seed.known = False`` — never fabricate historical funding.
    """
    acc = await s.get(SimAccount, 1)
    seed = None if acc is None else acc.opening_seed
    if seed is not None:
        return float(seed)
    return 0.0


async def _write_snapshot(post: dict, *, mode: str | None = None,
                          config_id: int | None = None,
                          epoch: int | None = None) -> None:
    """Write one equity-curve point stamped with the logical run's stamp.

    The forward writers (run_cycle, and the chat confirm path
    confirm_chat_proposal) all go through here: chat-driven marks are forward
    records too.
    """
    async with _state_write_lock:
        stamp = await _resolve_paper_stamp(mode=mode, config_id=config_id, epoch=epoch)
        async with Session() as s:
            if not await _stamp_is_current(s, stamp):
                logger.warning("Sim snapshot refused: stale epoch stamp")
                return
            s.add(SimSnapshot(
                cash=post["cash"],
                positions_value=post["positions_value"],
                total_equity=post["total_equity"],
                allowance_total=post["allowance_total"],
                mode=stamp["mode"], config_id=stamp["config_id"],
                epoch=stamp["epoch"],
            ))
            await s.commit()


async def _exec_buy(ticker: str, price: float, max_budget: float, reason: str,
                    stop_price: float | None = None, *, mode: str | None = None,
                    config_id: int | None = None,
                    epoch: int | None = None) -> dict | None:
    """Execute a paper BUY.  Returns the trade dict or None if skipped.

    Uses fractional shares (rounded to 4 decimals) so the bot can always
    deploy capital regardless of share price.

    ``stop_price``: the percent stop frozen at entry (session fill
    contract, rule 5). When None, the sim default applies — the entry fill
    price dropped by ``sim_stop_pct`` — so every NEW position carries a
    frozen stop. A top-up NEVER touches the stored level: freezing at
    entry means a lower-priced top-up cannot loosen the stop (re-anchoring
    it to the blended average cost would), and the position-specific
    monotone ATR ratchet is the only mechanism that can tighten it.
    """
    if price <= 0 or max_budget < 1:
        return None
    # Fractional shares — floor (not round) so cost never exceeds the
    # budget; rounding up can push a high-priced ticker cents over the
    # available cash and silently drop the buy.
    shares = math.floor(max_budget / price * 10000) / 10000
    if shares < 0.0001:
        return None
    cost = shares * price
    async with _state_write_lock:
        stamp = await _resolve_paper_stamp(mode=mode, config_id=config_id, epoch=epoch)
        return await _exec_buy_locked(ticker, price, shares, cost, reason,
                                      stop_price, stamp)


async def _exec_buy_locked(ticker: str, price: float, shares: float, cost: float,
                           reason: str, stop_price: float | None,
                           stamp: dict) -> dict | None:
    async with Session() as s:
        if not await _stamp_is_current(s, stamp):
            logger.warning("Sim BUY %s refused: stale epoch stamp", ticker)
            return None
        acc = await s.get(SimAccount, 1)
        if acc is None or acc.cash < cost:
            return None
        acc.cash -= cost

        pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
        if pos:
            # Update weighted average cost. The frozen percent stop is NOT
            # re-anchored here (see docstring): the stored level survives
            # every top-up untouched.
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + cost) / total_shares
            pos.shares = total_shares
        else:
            frozen = stop_price
            if frozen is None:
                frozen = price * (1 - settings.sim_stop_pct / 100)
            s.add(SimPosition(ticker=ticker, shares=shares, avg_cost=price,
                              thesis=reason, stop_price=frozen))

        trade = SimTrade(
            ticker=ticker, side="BUY", shares=shares, price=price,
            cash_after=acc.cash, reason=reason, mode=stamp["mode"],
            config_id=stamp["config_id"], epoch=stamp["epoch"],
        )
        s.add(trade)
        await s.commit()

        logger.info("Sim BUY %s ×%.4f @%.2f — %s", ticker, shares, price, reason)
        return {
            "ticker": ticker, "side": "BUY", "shares": shares, "price": price,
            "cash_after": round(acc.cash, 2), "reason": reason,
        }


async def _exec_sell(ticker: str, price: float, shares: float | None, reason: str,
                     *, mode: str | None = None, config_id: int | None = None,
                     epoch: int | None = None) -> dict | None:
    """Execute a paper SELL.  *shares*=None means sell entire position.

    Returns the trade dict or None if nothing to sell.
    """
    async with _state_write_lock:
        stamp = await _resolve_paper_stamp(mode=mode, config_id=config_id, epoch=epoch)
        return await _exec_sell_locked(ticker, price, shares, reason, stamp)


async def _exec_sell_locked(ticker: str, price: float, shares: float | None,
                            reason: str, stamp: dict) -> dict | None:
    async with Session() as s:
        if not await _stamp_is_current(s, stamp):
            logger.warning("Sim SELL %s refused: stale epoch stamp", ticker)
            return None
        pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
        if pos is None or pos.shares <= 0:
            return None

        sell_shares = pos.shares if shares is None else min(shares, pos.shares)
        if sell_shares <= 0:
            return None
        proceeds = sell_shares * price

        acc = await s.get(SimAccount, 1)
        if acc is None:
            return None
        acc.cash += proceeds
        pos.shares -= sell_shares
        if pos.shares <= 0.0001:
            # The position closes: the row (and with it the sell layer's defer
            # columns stop_defer_level/stop_defer_set_at) is deleted — the
            # defer state is cleared when the position closes.
            await s.delete(pos)

        trade = SimTrade(
            ticker=ticker, side="SELL", shares=sell_shares, price=price,
            cash_after=acc.cash, reason=reason, mode=stamp["mode"],
            config_id=stamp["config_id"], epoch=stamp["epoch"],
        )
        s.add(trade)
        await s.commit()

        logger.info("Sim SELL %s ×%.4f @%.2f — %s", ticker, sell_shares, price, reason)
        return {
            "ticker": ticker, "side": "SELL", "shares": sell_shares, "price": price,
            "cash_after": round(acc.cash, 2), "reason": reason,
        }


async def _quality_map(tickers: list[str]) -> dict[str, dict[str, float | None]]:
    """Point-in-time quality snapshot {ticker: {"roe", "p_fcf"}} for candidates.

    Reuses the monthly portfolio's fundamentals (EDGAR-first facts, already
    refreshed monthly) — no new network calls. Tickers without fundamentals
    (ETFs, thin coverage) are simply absent. DB-only work.
    """
    from .fundamentals import load_fundamentals
    from .monthly import quality_snapshot

    fund = await load_fundamentals(tickers)
    if not fund:
        return {}
    iso_today = _utcnow().strftime("%Y-%m-%d")
    today: pd.Timestamp = pd.Timestamp(iso_today)  # type: ignore[assignment]
    # USD-converted closes so P/FCF market caps are comparable (same
    # convention as the monthly strategy's decision frames).
    close, _ = await monthly_mod.load_frames(tickers, None)
    out: dict[str, dict[str, float | None]] = {}
    for t in fund:
        px = None
        if t in close.columns and len(close):
            series = close[t].dropna()
            if len(series):
                px = float(series.iloc[-1])
        q = quality_snapshot(fund, t, today, px)
        if q is not None:
            out[t] = q
    return out


# ---------------------------------------------------------------------------
# Frozen decision inputs (session fill contract)
# ---------------------------------------------------------------------------

# Warmup bound for the one-query candle load. The nightly refresh stores ~2y
# of daily bars (matches _SIM_REFRESH_PERIOD); the indicator engine needs
# ~206 daily bars, 50 weekly closes and the 52-week high, so ~760 calendar
# days covers the warmup deterministically (a longer cache is simply not
# loaded; a shorter one degrades exactly as the indicator math demands).
_DECISION_WARMUP_DAYS = 760


@dataclass
class _DecisionInputs:
    """One cycle's frozen decision inputs.

    ``signals``/``prices`` come strictly from data <= the previous completed
    session S; ``session_bars`` carries F's bar (strict exact date) for the
    tickers that have one — the only place F's data enters the cycle — and
    ``ratchets`` holds the per-position monotone ATR levels.
    """

    f_ts: pd.Timestamp
    s_ts: pd.Timestamp | None
    signals: dict[str, dict]
    prices: dict[str, float]           # decision closes (on S)
    session_bars: dict[str, dict]      # {ticker: {"open","low","close"}} at F
    ratchets: dict[str, float | None]  # held ticker -> monotone ATR level


def _atr_ratchets(frames: session_mod.SessionFrames,
                  held_entries: dict[str, str | None],
                  s_str: str | None) -> dict[str, float | None]:
    """Position-specific monotone ATR ratchet levels (contract rule 6).

    For each held ticker: the MAX of the prior-known ATR candidate level
    (``analysis``'s atr_stop) over all COMPLETED bars since the position's
    entry anchor. The max over a growing window is naturally monotone —
    recomputing it each cycle from the entry anchor can tighten but never
    loosen the level, so no persistence is needed. The entry anchor is
    ``SimPosition.opened_at``'s date: the actual first-buy session for live
    fills, and the replay's first-buy date after a backfill (the backfill
    writes it there instead of wall-clock now()).
    """
    out: dict[str, float | None] = {}
    for t in held_entries:
        out[t] = None
    if s_str is None:
        return out
    for t, entry in held_entries.items():
        if not entry:
            continue
        rows = session_mod.rows_for(frames, t)
        dec_rows = [r for r in rows if r["time"] <= s_str]
        if not dec_rows:
            continue
        try:
            ser = signal_series(dec_rows)["atr_stop"]
        except Exception as e:
            logger.warning("No ratchet series for %s: %s", t, e)
            continue
        vals: list[float] = []
        for r, v in zip(dec_rows, ser, strict=True):
            if r["time"] < entry or r["time"] > s_str:
                continue  # bars before entry are not this position's levels
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                vals.append(fv)
        if vals:
            out[t] = max(vals)
    return out


async def _decision_inputs(tickers: list[str],
                           held_entries: dict[str, str | None] | None = None,
                           now: datetime | None = None) -> _DecisionInputs:
    """Build the frozen decision inputs for one cycle.

    One candle query for all decision tickers (candidates + held), then:

      * S = the previous completed session strictly before F (the cycle's
        run date), taken from the stored frame index
        (``session.prev_session``);
      * signals = ``analysis.compute`` over each ticker's candles truncated
        to <= S — F's bar is never a decision input, so mutating or
        appending F's candle cannot change a decision;
      * a signal additionally requires the ticker to have ACTUALLY traded
        on S (its last truncated candle is dated S): a ticker whose last
        candle predates S has no current view and gets no decision this
        cycle — its stop standing orders still apply through F's bar;
      * prices = each signal ticker's close ON S (decision-side equity and
        budget math);
      * session_bars = F's strict exact-date open/low/close per ticker
        (``session.open_at`` — no ffill); tickers without an F bar are
        absent, which makes them unfillable this cycle;
      * ratchets = per held ticker the monotone ATR level since its entry
        anchor.
    """
    f_ts = pd.Timestamp((now or _utcnow()).date())
    start = (f_ts - pd.Timedelta(days=_DECISION_WARMUP_DAYS)).to_pydatetime()
    all_tickers = sorted(set(tickers) | set(held_entries or {}))
    frames = await session_mod.load_open_frames(all_tickers, asof=f_ts, start=start)
    s_ts = session_mod.prev_session(frames.index, f_ts)
    s_str = s_ts.strftime("%Y-%m-%d") if s_ts is not None else None

    signals: dict[str, dict] = {}
    prices: dict[str, float] = {}
    session_bars: dict[str, dict] = {}
    for t in all_tickers:
        rows = session_mod.rows_for(frames, t)
        if not rows:
            continue
        # F's bar via the strict exact-date lookup — never a forward-fill.
        open_px = session_mod.open_at(frames.open, f_ts, t)
        if open_px is not None:
            session_bars[t] = {
                "open": open_px,
                "low": session_mod.open_at(frames.low, f_ts, t),
                "close": session_mod.open_at(frames.close, f_ts, t),
            }
        # Decision data: candles truncated to <= S (F's bar excluded).
        if s_str is None:
            continue
        dec_rows = [r for r in rows if r["time"] <= s_str]
        if not dec_rows or dec_rows[-1]["time"] != s_str:
            continue  # did not trade on S — no fresh decision view
        if len(dec_rows) < MIN_CANDLES:
            continue
        try:
            signals[t] = compute(dec_rows)
        except Exception as e:
            logger.warning("No signal for %s: %s", t, e)
            continue
        prices[t] = float(dec_rows[-1]["close"])

    ratchets = _atr_ratchets(frames, held_entries or {}, s_str)
    return _DecisionInputs(f_ts=f_ts, s_ts=s_ts, signals=signals,
                           prices=prices, session_bars=session_bars,
                           ratchets=ratchets)


def _session_fill_open(session_bars: dict[str, dict], ticker: str) -> float | None:
    """F's OPEN for a ticker from the cycle's strict session bars.

    None when the ticker has no F bar — no fill this cycle. Only the open
    is ever returned: cycle fills happen at F's open, never at a stale
    close (the stop standing orders may fill AT their level instead — that
    level comes from the proposals, not from here).
    """
    bar = session_bars.get(ticker) or session_bars.get(ticker.upper())
    if not bar:
        return None
    return _bar_price(bar.get("open"))


async def _deterministic_propose(
    valuation: dict[str, Any],
    decision: _DecisionInputs,
    quality_of: Callable[[str], dict[str, float | None] | None] | None = None,
) -> list[dict]:
    """Propose deterministic trades WITHOUT executing them.

    Thin wrapper over ``strategy.propose_trades`` — loads positions from the
    DB, converts to plain data, and delegates the SELL/BUY logic to the
    shared module. The hybrid strategy passes the proposals to the LLM for
    review/veto before any DB write happens.

    Each proposal carries everything ``_execute_proposed`` needs:
      - SELL: {"ticker", "side": "SELL", "price", "shares": None, "reason"}
      - BUY:  {"ticker", "side": "BUY",  "price", "budget", "reason", "entry_stop"}

    ``decision`` is the cycle's frozen decision inputs — the only path: the
    proposals follow the session fill contract: decision data <= S, fills
    at F's open via the strict session bars, stops as standing orders
    (rule 4), the position's persisted frozen percent stop plus the
    monotone ATR ratchet (rules 5-6) passed through.
    """
    async with Session() as s:
        open_positions = (await s.scalars(select(SimPosition))).all()

    signals = decision.signals
    prices = decision.prices
    session_bars = decision.session_bars
    ratchets = decision.ratchets

    positions = [
        {"ticker": pos.ticker, "shares": pos.shares, "avg_cost": pos.avg_cost,
         # The percent stop frozen at entry — persisted since the
         # stop_price column exists. Legacy rows without it fall back to
         # the avg_cost recompute inside propose_trades (documented).
         "stop_price": pos.stop_price,
         # Position-specific monotone ATR ratchet since the entry anchor
         # (contract rule 6) — None until the position has completed bars.
         "atr_stop_level": ratchets.get(pos.ticker),
         # The LLM sell layer's active type-A DEFER level (app/selllayer.py):
         # while set, strategy._standing_stop_levels uses it as the ATR leg.
         "stop_defer_level": (float(pos.stop_defer_level)
                              if pos.stop_defer_level is not None else None)}
        for pos in open_positions
    ]

    sp = StrategyParams(
        buy_threshold=40, sell_threshold=-40,
        min_cash_pct=settings.sim_min_cash_pct,
        max_position_pct=settings.sim_max_position_pct,
        max_positions=settings.sim_max_positions,
        # Two independent exit controls:
        #  - stop_type="percent": the entry-anchored control, FROZEN at
        #    entry — persisted on the position (SimPosition.stop_price) so
        #    a top-up cannot loosen it (session fill contract, rule 5).
        #  - use_atr_stop: the ATR exit control — a position-specific
        #    monotone ratchet (max candidate level over all completed bars
        #    since entry, contract rule 6) plus the per-bar prior-known
        #    candidate from the snapshot; it can only tighten, never
        #    loosen.
        stop_type="percent",
        stop_pct=settings.sim_stop_pct,
        use_atr_stop=True,
        max_run_5d=settings.sim_max_run_5d,
        # Entry guards: block falling-knife (5d crash) and parabolic
        # (dist_above SMA200) entries — see strategy.py for the measured
        # rationale. Set the env vars to 0 to disable.
        min_run_5d=settings.sim_min_run_5d,
        max_dist_above=settings.sim_max_dist_above,
        # Quality guard: block entries into names with KNOWN non-positive
        # ROE. Missing fundamentals stay neutral (ETFs, thin coverage).
        # Opt-in via SIM_BLOCK_NEGATIVE_ROE until the replay A/B verdict.
        block_negative_roe=settings.sim_block_negative_roe,
        # Relaxed fallback: a HOLD may only be entered when it is a genuine
        # near-threshold BUY — positive net score and the same entry gate as
        # a strict BUY (see strategy._fallback_eligible).
        relaxed_hold_strength=40,
        relaxed_hold_limit=3,
    )
    return propose_trades(positions, valuation["cash"], prices, signals, sp,
                          quality_of=quality_of, session_bars=session_bars)


async def _execute_proposed(proposals: list[dict]) -> list[dict]:
    """Execute a list of proposed trades via _exec_buy / _exec_sell.

    Returns the list of actually-executed trade dicts (same shape as
    _exec_buy/_exec_sell return). Skips any that fail the exec guards
    (e.g. cash shortfall discovered at execution time). Re-checks the
    max-positions cap on each BUY: a veto that freed a slot mid-list is
    honoured, and a BUY for an already-held ticker is allowed through as a
    top-up (the exec guard clamps it to the max-position-% limit).
    """
    executed: list[dict] = []
    for p in proposals:
        if p["side"] == "SELL":
            t = await _exec_sell(p["ticker"], p["price"], p.get("shares"), p["reason"])
            if t:
                executed.append(t)
            continue
        # BUY: re-check the max-positions cap against the live DB state, so
        # a SELL earlier in this same list frees a slot for the next BUY.
        async with Session() as s:
            open_count = await s.scalar(
                select(func.count()).select_from(SimPosition)) or 0
            held = await s.scalar(select(SimPosition).where(SimPosition.ticker == p["ticker"]))
        if open_count >= settings.sim_max_positions and held is None:
            continue  # at cap and this is a new position — skip
        # The frozen percent stop rides on the proposal (entry_stop,
        # anchored to the entry fill); _exec_buy falls back to the sim
        # default (entry price - sim_stop_pct) when absent.
        t = await _exec_buy(p["ticker"], p["price"], p["budget"], p["reason"],
                            stop_price=p.get("entry_stop"))
        if t:
            executed.append(t)
    return executed


async def _deterministic_decide(valuation: dict[str, Any],
                                decision: _DecisionInputs,
                                quality_of: Callable[[str], dict[str, float | None] | None] | None = None) -> list[dict]:
    """Rule-based strategy: propose + execute deterministic trades.

    Thin wrapper over ``_deterministic_propose`` + ``_execute_proposed`` so
    the deterministic-only strategy keeps its original behaviour. The hybrid
    strategy calls the two halves separately so the LLM can review proposals
    before they execute. ``decision`` = the cycle's frozen decision inputs
    (session fill contract).
    """
    proposals = await _deterministic_propose(valuation, quality_of=quality_of,
                                             decision=decision)
    return await _execute_proposed(proposals)


def _quality_lookup(
    quality: dict[str, dict[str, float | None]] | None,
) -> Callable[[str], dict[str, float | None] | None] | None:
    """Turn a quality map into the ``quality_of(ticker)`` callable that
    ``strategy.propose_trades`` expects. None map -> None (feature off)."""
    if quality is None:
        return None

    def _lookup(ticker: str) -> dict[str, float | None] | None:
        return quality.get(ticker.upper())

    return _lookup


# ---------------------------------------------------------------------------
# Hybrid / LLM strategy
# ---------------------------------------------------------------------------

# Shared methodology block injected into both the autonomous-decide and the
# interactive-chat system prompts. Keeps the two LLMs' reasoning aligned with
# the deterministic engine's indicator interpretation without exposing the
# exact scoring weights (so tuning the weights doesn't drift the prompts).
_SIM_METHODOLOGY = (
    "## How to read the signals\n"
    "The snapshot each ticker carries these indicators. Use them to rank "
    "candidates and to judge whether a position should be kept, trimmed, or "
    "sold when the user asks you to clean up or rebalance the portfolio:\n"
    "  - **action / strength**: the deterministic engine's BUY/SELL/HOLD call "
    "and a 0-100 confidence (strength = the bullish or bearish score). Higher "
    "strength = stronger signal. Rank candidates by strength when choosing "
    "what to buy or what to keep.\n"
    "  - **close vs sma50 vs sma200**: a genuine uptrend needs close > sma50 > "
    "sma200 (downtrend is the mirror). Price stuck between the MAs = sideways / "
    "HOLD.\n"
    "  - **adx** (ADX-14): trend *strength*. ADX > 25 = strong, clean trend; "
    "ADX < 20 = weak/choppy. A BUY with high ADX is far more trustworthy than "
    "one with low ADX. When trimming a portfolio down to a position cap, prefer "
    "keeping positions with higher ADX over equal-strength low-ADX ones.\n"
    "  - **rsi** (RSI-14): momentum. 40-55 and rising = pullback turning up "
    "(good entry). 55-65 = moderately strong. > 70 = overbought (the engine "
    "penalizes it — don't chase). < 30 = oversold (the engine penalizes "
    "bearishness there too). Use RSI to avoid buying overbought names and to "
    "spot ones turning up from a pullback.\n"
    "  - **rsi_3d_change**: the 3-day net change in RSI. Positive = RSI is "
    "turning up (momentum recovering); negative = RSI still falling (momentum "
    "deteriorating). This is critical for SELL-gating: a SELL on a ticker with "
    "RSI 35 and rsi_3d_change positive is a pullback *ending*, not deepening — "
    "hold through it. A SELL with RSI 35 and rsi_3d_change negative is still "
    "falling — let it sell.\n"
    "  - **macd / macd_signal / macd_hist**: momentum. macd > macd_signal = "
    "bullish; the *histogram* (macd_hist) rising = momentum is turning up, not "
    "just already up. A rising histogram is a better confirmation than the laggy "
    "boolean crossover alone.\n"
    "  - **macd_hist_3d_change**: the 3-day net change in the MACD histogram. "
    "Positive = histogram is turning up (momentum shifting bullish); negative = "
    "histogram still falling. Use this the same way as rsi_3d_change: a SELL "
    "with macd_hist_3d_change positive is a turn-up signal — hold unless the "
    "trend is genuinely broken (ADX > 25 + price < sma50 < sma200).\n"
    "  - **weekly_trend_up** (when provided): the slower weekly-chart filter. "
    "A daily BUY is only valid when the weekly chart is also up (weekly close > "
    "weekly SMA-50). A daily BUY against a down weekly trend is a bear-market "
    "rally — avoid it. For SELL-gating, weekly_trend_up is the most important "
    "factor: the deterministic SELL signal is wrong ~86% of the time when the "
    "weekly trend is still up — it fires on normal pullbacks within an uptrend, "
    "not just on real trend breaks.\n"
    "  - **vol_surge**: a confirmation bonus, not a primary driver.\n"
    "  - **atr_stop**: volatility exit reference. It is the highest close of "
    "the previous 20 completed bars minus 2x ATR of the previous completed "
    "bar — a level that was already known before today's bar closed. Price "
    "at or below it = the trend broke. The engine holds a position-specific "
    "monotone ratchet of that level since entry (it can only tighten, never "
    "loosen) plus the percent stop FROZEN at the position's entry price — a "
    "later top-up cannot loosen either.\n"
    "\n"
    "## Timing (session fill contract)\n"
    "Signals are computed from candles up to the PREVIOUS completed session's "
    "close; fills execute at the CURRENT session's OPEN, and marks use the "
    "current session's close. A stop level known at the previous close is a "
    "standing order: it fills at the open after an opening gap through it, "
    "otherwise at the level itself when the session's low reaches it.\n"
    "\n"
    "## GATING deterministic BUYs\n"
    "The deterministic BUY signal (net score >= +40 + uptrend confirmed) fires "
    "on any ticker in an uptrend, even when the entry is overextended. Downgrade "
    "a BUY to HOLD when ANY of these signal the entry is too late:\n"
    "  - **run_5d > 15%**: price already spiked more than 15% in the last 5 "
    "days — you are chasing a short-term spike that is prone to reversion.\n"
    "  - **RSI > 70 AND rsi_3d_change < 0**: overbought and turning down — "
    "momentum is fading at the top.\n"
    "  - **ADX < 15 AND macd_hist_3d_change <= 0 AND rsi_3d_change <= 0**: no "
    "real trend AND no momentum turning up — the signal is noise. NOTE: ADX "
    "is a lagging indicator that stays low at the START of trends; a low ADX "
    "with rising MACD histogram or rising RSI is an early-trend entry, not "
    "noise — do NOT veto those.\n"
    "Downgrading a BUY to HOLD is the highest-impact decision you can make: "
    "backtesting showed avoiding 4 catastrophic entries (each losing 15-38% "
    "within 20 days) outweighs missing 8 good entries, for a net +5% return "
    "improvement.\n"
    "\n"
    "## GATING deterministic SELLs\n"
    "The deterministic SELL signal fires on any pullback that briefly crosses "
    "below SMA50, not just on real trend breaks. You may override it to HOLD, "
    "but be conservative: holding traps capital that could be redeployed. Only "
    "override when ALL of these hold:\n"
    "  1. weekly_trend_up is True, AND\n"
    "  2. rsi_3d_change is positive OR macd_hist_3d_change is positive.\n"
    "When in doubt, let the SELL execute — the capital will be redeployed into "
    "the next BUY signal.\n"
    "\n"
    "When the portfolio holds more positions than the max-positions cap (or the "
    "user asks to clean up / trim / take profits), sell the weakest first: "
    "lowest strength, SELL signals, low ADX, RSI overbought, or price below its "
    "ATR stop — and keep the highest-strength, highest-ADX, still-in-uptrend "
    "names.\n"
)

_LLM_SYSTEM_PROMPT = (
    "You are a disciplined portfolio manager reviewing deterministic technical "
    "signals for a paper-trading simulation.\n"
    "You will receive the current portfolio state and a list of candidate "
    "tickers with their technical indicators (signal action, strength, close "
    "price, RSI, MACD).\n"
    "Your job: review the deterministic PROPOSED trades (which have NOT yet "
    "executed) and the signals, then return your own decisions. Your HOLD on "
    "a proposed trade VETOES it — the trade will not execute. Your BUY/SELL "
    "on a proposed trade confirms and executes it. You may also add new "
    "BUY/SELL decisions for tickers not in the proposed list.\n"
    "\n"
    + _SIM_METHODOLOGY +
    "Rules:\n"
    "2. Your primary job is to downgrade overextended BUYs to HOLD. See the "
    "GATING deterministic BUYs section above. This is the highest-impact "
    "decision you make — avoiding catastrophic entries outweighs everything "
    "else. Your HOLD on a proposed BUY will block it from executing. You may "
    "also veto a SELL, but be conservative (see GATING "
    "deterministic SELLs).\n"
    "3. ADAPT TO THE MARKET REGIME shown in the context: in a BULL regime "
    "momentum persists, so do NOT veto strong entries merely for being "
    "slightly overbought (RSI 70-78 or a hot 5-day run is normal in a "
    "bull); reserve vetoes for confirmed reversals. In a BEAR regime be "
    "aggressive — skip weak-trend and overbought entries entirely. In a "
    "MIXED regime judge each entry on its own merits. A one-size-fits-all "
    "RSI>70 veto rule loses money in bull markets.\n"
    "3b. Respect risk management: do not buy if cash is too low; do not over-"
    "concentrate in a single ticker. The engine caps the number of open "
    "positions (it stops buying once the max position count is reached), so "
    "prioritize the strongest candidates.\n"
    "4. The engine enforces a percent stop FROZEN at the position's entry "
    "price (persisted — a top-up cannot loosen it) plus a position-specific "
    "monotone ATR ratchet (the highest candidate ATR level since entry — it "
    "can only tighten). Fills execute at the next session's open, and a stop "
    "known before a session acts as a standing order for it (open-gap fills "
    "at the open, intrabar breaches at the level): positions that hit either "
    "are auto-sold by the deterministic layer. Do not be "
    "surprised if a position disappears between cycles — that is the stop "
    "loss, not a decision you need to replicate.\n"
    "5. Do NOT sell a position to free up cash for another BUY. Selling one "
    "ticker to buy another is portfolio churn — backtesting proved this "
    "reduces returns because the \"stronger opportunity\" has the same "
    "indicator profile as the position being sold. Only SELL when the "
    "deterministic engine proposes a SELL and you agree, or when a position "
    "is clearly broken (price below ATR stop, SELL signal with weekly trend "
    "down).\n"
    "6. Decisions must be grounded in the provided signals and indicators.\n"
    "7. You may receive recent news headlines for supplementary context. News "
    "can explain *why* indicators are moving, but do not make trades based on "
    "news alone — the technical signals and risk rules take priority. Never "
    "reference specific URLs in your output.\n"
    "8. You may specify a partial position size per action using optional fields:\n"
    "   - \"shares\": exact number of shares to trade (e.g. 3.5).\n"
    "   - \"amount\": dollar amount to trade (e.g. 67.43). For SELL this is the"
    " value of shares to sell; for BUY it is the dollars to invest.\n"
    "   If neither is given, SELL sells the entire position and BUY invests the"
    " maximum allowed by risk rules.\n"
    "9. The max position % is a buy-time sizing limit, not a ceiling to enforce "
    "on exits. Do NOT sell a position just because its price rose above it — "
    "let winners run.\n"
    "10. The min cash floor is also a buy-time constraint, not a sell trigger. "
    "Do NOT sell a position solely to restore cash above the floor — the floor "
    "only blocks new BUYs. If cash is below the floor, hold the positions you "
    "have and wait for the next allowance deposit or a stop-out to replenish "
    "cash. (The user can explicitly authorise spending below the floor from "
    "the chat — that override does not apply to autonomous cycles.)\n"
    "11. When the portfolio holds more positions than the max-positions cap, "
    "sell the weakest first: lowest strength, SELL signals, low ADX, RSI "
    "overbought, or price below its ATR stop — and keep the highest-strength, "
    "highest-ADX, still-in-uptrend names. But do not sell just to rotate into "
    "a different ticker with similar indicators.\n"
    "\n"
    "Response format: begin with a 2-4 sentence prose summary of your overall "
    "read and the decisions you made (write this even when you made no "
    "changes), then the JSON array of decisions on a new line. Objects have "
    'the fields "ticker", "action", "reason", and optional "shares" / '
    '"amount". No code fences, no markdown — just the prose, then the JSON.\n'
)


_PURE_LLM_METHODOLOGY = (
    "## How to read the signals\n"
    "The snapshot each ticker carries these indicators. Use them to rank "
    "candidates and to manage open positions:\n"
    "  - **action / strength**: a deterministic 0-100 score of the indicators "
    "(strength = the bullish or bearish reading). Use it as a starting point, "
    "not as a verdict.\n"
    "  - **close vs sma50 vs sma200**: a genuine uptrend needs close > sma50 > "
    "sma200 (downtrend is the mirror). Price stuck between the MAs = sideways.\n"
    "  - **adx** (ADX-14): trend *strength*. ADX > 25 = strong, clean trend; "
    "ADX < 20 = weak/choppy. High ADX makes a BUY far more trustworthy than "
    "low ADX.\n"
    "  - **rsi** (RSI-14): momentum. 40-55 and rising = pullback turning up "
    "(good entry). 55-65 = moderately strong. > 70 = overbought — don't "
    "chase. < 30 = oversold (often a bounce risk).\n"
    "  - **rsi_3d_change**: the 3-day net change in RSI. Positive = momentum "
    "recovering; negative = momentum deteriorating.\n"
    "  - **macd / macd_signal / macd_hist**: momentum. macd > macd_signal = "
    "bullish; macd_hist rising = momentum turning up.\n"
    "  - **macd_hist_3d_change**: the 3-day net change in the MACD histogram. "
    "Positive = histogram rising; negative = histogram falling.\n"
    "  - **weekly_trend_up** (when provided): the slower weekly-chart filter "
    "(weekly close > weekly SMA-50). A BUY against a down weekly trend is a "
    "bear-market rally — risky. A position whose weekly trend has turned "
    "down is structurally weaker.\n"
    "  - **run_5d**: the 5-day run-up %. A BUY after a >15% spike is chasing "
    "a short-term move prone to reversion.\n"
    "  - **atr_stop**: volatility exit reference. Highest close of the "
    "previous 20 completed bars minus 2x ATR of the previous completed bar "
    "— known before today's bar closed. Price at or below it = the trend "
    "broke. The engine tracks a position-specific monotone ratchet of this "
    "level since entry (it can only tighten), and the percent stop is "
    "FROZEN at the entry price.\n"
    "\n"
    "## Managing exits\n"
    "You are the ONLY mechanism that sells positions — no engine will do it "
    "for you. Each cycle, review every open position and sell when the reason "
    "you bought it is gone: the trend broke (price < sma50 < sma200 or weekly "
    "trend down), momentum rolled over (RSI falling, MACD histogram "
    "declining), or price is at/below its stop. Holding a broken position "
    "traps capital that could earn elsewhere; act on your read.\n"
    "\n"
    "## Entry quality\n"
    "Buying is when care matters most. Avoid entries that are too late: a BUY "
    "after a >15% 5-day spike, RSI > 70 with momentum turning down, or ADX < "
    "15 (no trend, just noise). Favor pullbacks that are turning up inside "
    "an uptrend. Missing a move costs less than catching a falling knife.\n"
)

_PURE_LLM_SYSTEM_PROMPT = (
    "You are the sole portfolio manager for a paper-trading simulation. "
    "A deterministic risk floor runs alongside you: it auto-sells positions "
    "that hit their initial stop or the ATR reference stop, but you make all "
    "buy and discretionary sell decisions yourself.\n"
    "You will receive the current portfolio state and a list of candidate "
    "tickers with their technical indicators (signal action, strength, close "
    "price, RSI, MACD). The signal action/strength is a deterministic "
    "scoring of the indicators — use it as a starting point, but you decide "
    "whether to act on it.\n"
    "\n"
    + _PURE_LLM_METHODOLOGY +
    "Rules:\n"
    "2. Decide the portfolio yourself: what to buy, what to hold, what to "
    "sell, how much of the cash to deploy, and how many positions to hold. "
    "Use the indicators and your exit-management principles; do not simply "
    "echo the signal actions.\n"
    "3. You are responsible for stop losses. Each open position shows its "
    "initial stop price (FROZEN at the position's entry price — a later "
    "top-up cannot loosen it) and the signals include an ATR reference stop "
    "(the highest close of the previous 20 completed bars minus 2x ATR of "
    "the previous bar; the engine also tracks a monotone ratchet of it "
    "since entry that can only tighten). Sell any "
    "position whose current price is at or below either stop unless you have "
    "a strong indicator-based reason to override. (The engine also "
    "auto-sells stop hits as a safety net, but do not rely on it — act on "
    "your read.)\n"
    "4. STABILITY RULES — do NOT rotate the portfolio. These are the most "
    "important rules you have:\n"
    "   - Never SELL and BUY in the same cycle (no 'sell X to buy Y' rotation). "
    "Decide holds first; only propose new BUYs from cash that is already free.\n"
    "   - Never re-buy a ticker you sold within the last 10 trading days — the "
    "Recent Trades section shows your own activity. A round-trip (sell Monday, "
    "re-buy Thursday) is churn and loses money on the spread. If you sold it, "
    "you had a reason; that reason has not changed in 3 days.\n"
    "   - Hold through short-term noise. A position that is down 3-5% on a "
    "normal pullback inside an uptrend is NOT a sell — the engine's stop is "
    "your safety net. Do not exit a position just because it is red today.\n"
    "   - Sell when a position's thesis is broken; if the redeployed capital "
    "goes into a stronger name, so be it, but don't manufacture trades.\n"
    "   - The engine blocks re-buying a ticker sold within the last 10 "
    "trading days — don't waste your output proposing those.\n"
    "5. You decide how much cash to keep in reserve and how concentrated the "
    "portfolio should be. The reference min-cash floor and max-position-% in "
    "the portfolio state are guidance, not hard limits — you decide. There is "
    "no limit on the number of positions you may hold.\n"
    "6. Decisions must be grounded in the provided signals and indicators.\n"
    "7. You may receive recent news headlines for supplementary context. News "
    "can explain *why* indicators are moving, but do not make trades based on "
    "news alone — the technical signals and risk rules take priority. Never "
    "reference specific URLs in your output.\n"
    "8. You may specify a partial position size per action using optional fields:\n"
    "   - \"shares\": exact number of shares to trade (e.g. 3.5).\n"
    "   - \"amount\": dollar amount to trade (e.g. 67.43). For SELL this is the"
    " value of shares to sell; for BUY it is the dollars to invest.\n"
    "   If neither is given, SELL sells the entire position and BUY is executed "
    "as follows: when you send several BUYs in one cycle, the available cash is "
    "split evenly between them; when you send a single BUY, it receives the full "
    "available cash. Always use \"amount\" (or \"shares\") when you want a "
    "specific size — otherwise your position sizing is left to the even split.\n"
    "9. You decide position sizing. Do NOT sell a position just because its "
    "price rose — let winners run.\n"
    "10. Do NOT sell a position solely to restore cash — if you want more dry "
    "powder, wait for the next allowance deposit or a stop-out to replenish "
    "cash.\n"
    "\n"
    "Response format: begin with a 2-4 sentence prose summary of your overall "
    "read and the decisions you made (write this even when you made no "
    "changes), then the JSON array of decisions on a new line. Objects have "
    'the fields "ticker", "action", "reason", and optional "shares" / '
    '"amount". No code fences, no markdown — just the prose, then the JSON.\n'
)


_LLM_MINIMAL_SYSTEM_PROMPT = (
    "You are a portfolio manager for a paper-trading simulation. "
    "You will receive the current portfolio state and a table of candidate "
    "tickers with technical indicators.\n"
    "Review the portfolio and the signals, then decide whether to make "
    "adjustments.\n"
    "\n"
    "Actions:\n"
    '- BUY: open or add to a position.\n'
    '- SELL: reduce or close a position.\n'
    '- HOLD: do nothing.\n'
    "You may specify a partial position size per action:\n"
    '   - "shares": exact number of shares to trade (e.g. 3.5).\n'
    '   - "amount": dollar amount to trade (e.g. 67.43). For SELL this is '
    "the value of shares to sell; for BUY it is the dollars to invest.\n"
    "   If neither is given, SELL sells the entire position and BUY invests "
    "the maximum allowed by the risk rules.\n"
    "\n"
    'Return ONLY a JSON array of objects: '
    '{"ticker": "...", "action": "BUY|SELL|HOLD", "reason": "..."}. '
    'Optional "shares" / "amount" fields size the trade. '
    "No markdown, no prose.\n"
)


_LLM_MODE_AWARE_MINIMAL_PROMPT = (
    "You are a portfolio manager for a paper-trading simulation. A "
    "deterministic engine runs the day-to-day trading and manages all exits: "
    "its pending proposed trades (new entries, stop-losses and SELL signals) "
    "execute automatically, so you do not need to confirm or repeat them. "
    "Your only job: add a small number of high-quality BUYs the engine did "
    "not propose, using the free cash and open position slots shown in the "
    "portfolio state. Any SELL you return is ignored, so return none.\n"
    "\n"
    "Only buy when the trend is real and not already extended:\n"
    "  - Prefer: close > sma50 > sma200 (confirmed uptrend), ADX > 25 "
    "(strong trend), RSI 45-65 (momentum without being overbought), MACD "
    "above its signal line with a rising histogram, run_5d < 15% (not chasing "
    "a spike).\n"
    "  - Skip: run_5d > 15% (chasing), RSI > 70 (overbought), ADX < 20 "
    "(chop, no trend), or momentum rolling over (RSI change and MACD histogram "
    "change both negative).\n"
    "Prefer names that are not already in the open positions. Size each BUY "
    "with the optional \"amount\" field (dollars to invest) or \"shares\"; if "
    "omitted the risk rules cap the size.\n"
    "\n"
    'Return ONLY a JSON array of objects: '
    '{"ticker": "...", "action": "BUY", "reason": "..."}. '
    'Optional "shares" / "amount" fields size a BUY. '
    "Return an empty array [] if nothing deserves a new position. "
    "No markdown, no prose.\n"
)


def _build_llm_context(
    valuation: dict[str, Any],
    deterministic_trades: list[dict],
    signals: dict[str, dict],
    news: dict[str, list[dict]] | None = None,
    pure_llm: bool = False,
    trade_history: list[dict] | None = None,
    minimal: bool = False,
    max_positions: int = 0,
    quality: dict[str, dict[str, float | None]] | None = None,
) -> str:
    """Build the compact context string sent to the LLM.

    ``news`` is an optional dict of ``{"market": [...], "TICKER": [...]}``
    headline lists. If empty or None, the news section is omitted.

    ``trade_history`` is an optional list of recent trade dicts
    (``{"date", "side", "ticker", "shares", "price", "reason"}``), newest
    first. When provided, a "Recent Trades" section is included so the LLM
    can see what it did recently and avoid round-trips / repeated mistakes.
    The list is truncated to the last 12 trades to keep the context compact.

    ``pure_llm`` omits the deterministic-proposals section (the LLM is the
    sole decision-maker), drops the max-positions line (no count cap in
    pure-LLM mode), and shows each position's initial stop price so the LLM
    can act on stop-loss hits itself.

    ``max_positions`` overrides the reported position cap (0 = use
    ``settings.sim_max_positions``) so the LLM sees the raised cap when the
    caller grants it extra slots above the deterministic engine's limit.

    ``quality`` is an optional ``{ticker: {"roe": float|None, "p_fcf":
    float|None}}`` map (point-in-time, same math as the monthly qv-mom
    scoring). When provided (SIM_LLM_FUNDAMENTALS_CONTEXT=true), two columns
    are added to the signals table: ``roe`` (TTM net income / equity, in %)
    and ``p/fcf`` (market cap / TTM free cash flow). Missing fundamentals
    render as ``-`` (neutral: ETFs, thin coverage). When None the table is
    unchanged (zero prompt delta when the feature is off).
    """
    from .news import format_news_for_context, format_market_news_for_context

    lines: list[str] = []

    # --- Portfolio state ---
    lines.append("## Current Portfolio State")
    lines.append(f"Cash: {valuation['cash']:.2f}")
    lines.append(f"Positions value: {valuation['positions_value']:.2f}")
    lines.append(f"Total equity: {valuation['total_equity']:.2f}")
    lines.append(f"Cumulative allowance deposited: {valuation['allowance_total']:.2f}")
    # Main-branch parity: hard limits in both modes — the engine enforces
    # min-cash, max-position-% and max-positions on every LLM BUY.
    lines.append(f"Min cash floor (buy-time only, {settings.sim_min_cash_pct}%): {valuation['total_equity'] * settings.sim_min_cash_pct / 100:.2f}")
    lines.append(f"Max position size ({settings.sim_max_position_pct}%): {valuation['total_equity'] * settings.sim_max_position_pct / 100:.2f}")
    lines.append(f"Max open positions: {max_positions if max_positions > 0 else settings.sim_max_positions}")
    lines.append(f"Stop loss: {settings.sim_stop_pct:.0f}% below the position's entry price (frozen at entry — a top-up cannot loosen it; monotone ATR ratchet also applies)")
    lines.append("")

    if valuation["positions"]:
        lines.append("Open positions:")
        for p in valuation["positions"]:
            thesis = p.get("thesis", "")
            buy_date = p.get("buy_date", "")
            # Show the entry thesis (why this position was bought) and the
            # holding period so the LLM can judge "is the thesis still valid?"
            thesis_str = f" | since {buy_date}" if buy_date else ""
            if thesis:
                # Truncate long theses to keep the line readable.
                t = thesis if len(thesis) <= 80 else thesis[:77] + "..."
                thesis_str += f" | thesis: {t}"
            lines.append(
                f"  - {p['ticker']}: {p['shares']} shares @ avg {p['avg_cost']:.2f} "
                f"| current {p['current_price']:.2f} | value {p['value']:.2f} "
                f"| P&L {p['pnl_pct']:+.2f}%{thesis_str}"
            )
    else:
        lines.append("Open positions: none")
    lines.append("")

    # --- Signals summary ---
    # The full signal table is shown — the LLM is the decision-maker in
    # pure-LLM mode and needs to see every candidate, exactly as the good-era
    # runs did. Truncating to "top N" starved it of context.
    shown = sorted(signals.items()) if signals else []
    lines.append("## Signals (all candidate tickers)")

    # --- Market regime (breadth-derived) ---
    # Count candidates above their SMA200 / weekly-up so the LLM can tell a
    # strong bull market (momentum persists — be slow to veto RSI-hot
    # entries) from a fragile one (veto aggressively). Omitted in minimal
    # mode: the experiment is a prompt with no guidance or bias.
    ups, total = 0, 0
    run5s: list[float] = []
    for sig in signals.values():
        snap = sig.get("snapshot", {})
        c = snap.get("close")
        s2 = snap.get("sma200")
        if c is not None and s2 is not None and s2 > 0:
            total += 1
            if c > s2:
                ups += 1
        r = snap.get("run_5d")
        if r is not None:
            run5s.append(r)
    if total > 0 and not minimal:
        pct = ups / total * 100
        med_run5 = 0.0
        if run5s:
            run5s.sort()
            med_run5 = run5s[len(run5s) // 2]
            p25_run5 = run5s[len(run5s) // 4]
            momentum_line = (f" | median 5d run {med_run5:+.1f}% "
                             f"(p25 {p25_run5:+.1f}%)")
        else:
            momentum_line = ""
        # A tape mid-pullback (median 5d run deeply negative) is a fragile
        # regime even when breadth is high — entries made there stop out.
        pullback = bool(run5s) and med_run5 < -2.5
        if pct >= 60 and not pullback:
            regime = "BULL"
            regime_advice = (
                "Broad-market uptrend. The engine's BUY proposals have "
                "already passed its overextension gates (run_5d, RSI+momentum, "
                "ADX) and in this regime the names keep running. Do NOT veto "
                "any deterministic BUY proposal in a BULL regime — approve "
                "them all; vetoing them only strands capital while the market "
                "moves. You may veto deterministic SELLs only when the weekly "
                "trend is clearly up with momentum turning up."
            )
        elif pullback:
            regime = "PULLBACK"
            regime_advice = (
                "Market is mid-pullback (median 5d run is negative across "
                "the tape). Entries made during pullbacks frequently stop "
                "out within days. Be selective: only approve BUYs with "
                "confirmed strength (ADX > 25, strong weekly trend, RSI "
                "recovering from oversold), and veto everything marginal."
            )
        elif pct >= 40:
            regime = "MIXED"
            regime_advice = (
                "Mixed market with a positive undertone. The engine's "
                "proposals are mostly sound here — only veto with clear "
                "evidence: run_5d > 15% AND (RSI > 70 with rsi_3d_change < 0) "
                "AND macd_hist_3d_change <= 0 together. A single overbought "
                "or low-ADX flag is NOT enough — strong names keep running "
                "even in mixed tapes. When in doubt, approve."
            )
        else:
            regime = "BEAR"
            regime_advice = (
                "Broad-market downtrend / fragile tape. Veto aggressively: "
                "skip weak-trend entries (low ADX), RSI > 65, and any BUY "
                "against the weekly trend. Capital preservation comes first."
            )
        lines.append("")
        lines.append(f"## Market Regime: {regime} — {ups}/{total} candidates above SMA200 ({pct:.0f}%){momentum_line}")
        lines.append(regime_advice)
        lines.append("")

    if shown:
        atr_col = " {'atrStop':>9}" if pure_llm else ""
        # Fundamentals columns (SIM_LLM_FUNDAMENTALS_CONTEXT=true): ROE in %,
        # P/FCF as a multiple. "-" = no fundamentals (neutral — ETFs, thin
        # coverage), NOT bad quality.
        q_cols = quality is not None
        roe_hdr = " {'roe%':>7}" if q_cols else ""
        pfcf_hdr = " {'p/fcf':>7}" if q_cols else ""
        lines.append(
            f"{'ticker':<10} {'action':<6} {'strength':>8} "
            f"{'close':>10} {'rsi':>6} {'rsiΔ3':>6} {'adx':>5} {'wk':>3} {'macd':>10} {'mhΔ3':>7} {'run5d':>6} {'run20d':>7} {'run60d':>7} {'hi52d':>6}{roe_hdr}{pfcf_hdr}{atr_col}"
        )
        for ticker, sig in shown:
            snap = sig.get("snapshot", {})
            wk = "up" if snap.get("weekly_trend_up") else "dn"
            rsi_d = snap.get("rsi_3d_change")
            mh_d = snap.get("macd_hist_3d_change")
            run5 = snap.get("run_5d")
            run20 = snap.get("run_20d")
            run60 = snap.get("run_60d")
            dist52 = snap.get("dist_52w_high")
            atr_stop = snap.get("atr_stop")
            rsi_d_s = f"{rsi_d:+.1f}" if rsi_d is not None else "  -  "
            mh_d_s = f"{mh_d:+.2f}" if mh_d is not None else "  -  "
            run5_s = f"{run5:+.1f}%" if run5 is not None else "  -  "
            run20_s = f"{run20:+.1f}%" if run20 is not None else "  -  "
            run60_s = f"{run60:+.1f}%" if run60 is not None else "  -  "
            dist52_s = f"{dist52:+.1f}%" if dist52 is not None else "  -  "
            atr_s = f"{atr_stop:>9.2f}" if pure_llm and atr_stop is not None else ""
            roe_s = pfcf_s = ""
            if q_cols:
                q = (quality or {}).get(ticker) or {}
                roe = q.get("roe")
                pfcf = q.get("p_fcf")
                roe_s = f"{roe * 100:>7.1f}" if roe is not None else "      -"
                pfcf_s = f"{pfcf:>7.1f}" if pfcf is not None else "      -"
            lines.append(
                f"{ticker:<10} {sig['action']:<6} {sig['strength']:>8} "
                f"{(snap.get('close') or 0):>10.2f} {(snap.get('rsi') or 0):>6.1f} "
                f"{rsi_d_s:>6} "
                f"{(snap.get('adx') or 0):>5.0f} {wk:>3} "
                f"{(snap.get('macd') or 0):>10.3f} {mh_d_s:>7} {run5_s:>6} {run20_s:>7} {run60_s:>7} {dist52_s:>6}{roe_s}{pfcf_s}{atr_s}"
            )
    else:
        lines.append("(no signals available)")
    lines.append("")

    # --- Recent news (optional, supplementary) ---
    if news and not minimal:
        lines.append("## Recent News (supplementary context — do not trade on news alone)")
        market_hl = news.get("market", [])
        if market_hl:
            lines.append(format_market_news_for_context(market_hl))
        for ticker in sorted(news.keys()):
            if ticker == "market":
                continue
            hl = news.get(ticker, [])
            if hl:
                lines.append(format_news_for_context(ticker, hl))
        lines.append("")

    # --- Recent Trades (your own recent activity, for continuity) ---
    if trade_history and not minimal:
        lines.append("## Recent Trades (your last 12 actions — use this to avoid round-trips)")
        for t in trade_history[:12]:
            shares = t.get("shares", "?")
            lines.append(
                f"  - {t.get('date', '?')} {t['side']:<4} {t['ticker']:<10} "
                f"×{shares} @ {t.get('price', '?'):.2f} — {t.get('reason', '')[:60]}"
            )
        lines.append("")

    # --- Deterministic proposed trades (pending — NOT yet executed) ---
    if not pure_llm:
        lines.append("## Deterministic Proposed Trades (pending — your HOLD will veto)")
        if deterministic_trades:
            for t in deterministic_trades:
                shares = t.get("shares", "?")
                if t["side"] == "BUY":
                    # Proposals carry a budget; show it so the LLM sees the size
                    shares = f"budget ${t.get('budget', 0):.0f}"
                lines.append(
                    f"  - {t['ticker']} {t['side']} ×{shares} @ "
                    f"{t.get('price', '?'):.2f} — {t.get('reason', '')}"
                )
        else:
            lines.append("(no deterministic trades proposed)")
        lines.append("")

    lines.append("## Your Decisions")
    if pure_llm:
        lines.append(
            "Return ONLY a JSON array of objects: "
            '{"ticker": "...", "action": "BUY|SELL|HOLD", "reason": "..."}. '
            "BUY/SELL will execute; HOLD means do nothing for that ticker. "
            "No markdown, no prose."
        )
    else:
        lines.append(
            "Return ONLY a JSON array of objects: "
            '{"ticker": "...", "action": "BUY|SELL|HOLD", "reason": "..."}. '
            "For tickers listed above, HOLD = veto (block the proposed trade); "
            "BUY/SELL = agree and execute. You may also add new BUY/SELL decisions "
            "for tickers NOT in the proposed list. No markdown, no prose."
        )
    return "\n".join(lines)


def _find_json_array(text: str) -> tuple[int, int] | None:
    """Locate the LLM's JSON array (start, end) inside ``text``.

    The model's response may contain prose before/after the array (the prompt
    asks for a prose summary first), and that prose — or a JSON string inside
    the array itself — can contain square brackets (e.g. "I reviewed [NVDA]"
    or a reason like "see [1]"). A naive ``find("[")/rfind("]")`` pair
    truncates at such a bracket.

    Instead we try parsing at every ``[`` and keep the candidate that (a)
    decodes as a JSON list and (b) looks like a decisions array — it holds at
    least one dict, or is an empty array (a valid "do nothing" decision).
    Among candidates we pick the one that ends latest: the real array is the
    last JSON value in the output per the prompt, and bracket fragments inside
    JSON strings always end before their enclosing array. Prose-only text
    yields None.
    """
    best: tuple[int, int] | None = None
    for cand in range(len(text)):
        if text[cand] != "[":
            continue
        try:
            obj, end = json.JSONDecoder().raw_decode(text[cand:])
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, list):
            continue
        # "[2]" inside prose or a JSON string parses as a list but is not the
        # decisions array. Empty arrays are kept (valid "do nothing").
        if obj and not any(isinstance(d, dict) for d in obj):
            continue
        if best is None or cand + end >= best[1]:
            best = (cand, cand + end)
    return best


def _parse_llm_decisions(content: str) -> list[dict] | None:
    """Parse the LLM JSON array output, tolerating minor formatting issues.

    Returns None if parsing fails so the caller can fall back.
    """
    if not content:
        return None
    text = content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (optionally with language tag)
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    # Locate the real JSON array — prose containing "[" must not confuse us.
    loc = _find_json_array(text)
    if loc is None:
        return None
    start, end = loc
    json_str = text[start:end]

    try:
        decisions = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(decisions, list):
        return None

    # An explicitly empty array is a valid decision: the LLM chose to do
    # nothing. Return [] so the caller doesn't treat it as a parse failure.
    if not decisions:
        return []

    # Validate / normalize entries
    valid = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        ticker = d.get("ticker")
        action = str(d.get("action", "")).upper().strip()
        reason = str(d.get("reason", "")).strip()
        if not ticker or action not in ("BUY", "SELL", "HOLD"):
            continue
        entry: dict[str, Any] = {"ticker": ticker, "action": action, "reason": reason}
        # Optional partial-size fields (validated by the caller against the
        # current position / cash).
        for field in ("shares", "amount"):
            val = d.get(field)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                entry[field] = float(val)
        valid.append(entry)
    return valid if valid else None


# ---------------------------------------------------------------------------
# LLM frozen-decision ledger (session fill contract, rule 7)
# ---------------------------------------------------------------------------
# Every LLM decision is appended to SimLlmDecision with the ``decided_at``
# captured when its response was PARSED, its relation to the deterministic
# proposal, and — when the decision's trade executed — the frozen fill. The
# ledger is ADDITIVE bookkeeping: it never changes an order, a guard, an
# execution order or a returned payload, and its failure never fails a cycle
# or a chat turn (the executed trades are already committed real state).


def _llm_decision_relation(action: str, det_side: str | None, *,
                           vetoed: bool = False) -> str:
    """Classify one LLM decision against the deterministic proposal.

    Single source of truth for the relation vocabulary (the chat flow's
    ledger rows use ``chat_action`` directly; the cycle-mode vocabulary is
    kept for the optimize replay paths, which import this helper):

      * HOLD on a proposed ticker   -> ``veto`` (the proposal is blocked)
      * HOLD on any other ticker    -> ``hold`` (passive)
      * BUY/SELL with no proposal   -> ``addition`` (LLM-initiated)
      * side == the proposal's side -> ``confirmation``
      * side != the proposal's side -> ``change`` (a flip: the proposal is
        vetoed and the flipped order itself is engine-excluded)
    """
    act = str(action or "").upper()
    side = str(det_side or "").upper() or None
    if act == "HOLD":
        return "veto" if vetoed else "hold"
    if side is None:
        return "addition"
    return "confirmation" if act == side else "change"


def _chat_session_date() -> str:
    """The chat turn's operator-local calendar date ("YYYY-MM-DD")."""
    return datetime.now(_TZ).strftime("%Y-%m-%d")


def _executed_index(executed: list[dict]) -> dict[tuple[str, str], dict]:
    """(TICKER, SIDE) -> executed trade dict, for the ledger's fill columns.

    ``_exec_buy`` / ``_exec_sell`` (and therefore ``_execute_proposed``)
    return dicts carrying ticker/side/shares/price/cash_after; matching on
    (ticker, side) attributes a fill to the decision that caused it.
    """
    index: dict[tuple[str, str], dict] = {}
    for t in executed:
        index[(str(t.get("ticker") or "").upper(),
               str(t.get("side") or "").upper())] = t
    return index


def _ledger_row(*, decided_at: datetime, mode: str, session_date: str,
                ticker: str, action: str, reason: str, relation: str,
                det_side: str | None = None, det_reason: str | None = None,
                fill: dict | None = None,
                filled_at: datetime | None = None) -> SimLlmDecision:
    """Build one frozen ledger row (rule 7).

    ``decided_at`` is always the value captured at PARSE time — this helper
    never re-stamps it. ``fill`` is the executed trade dict when the
    decision's order ran; without both ``fill`` and ``filled_at`` the four
    fill columns stay NULL: decided, never filled.
    """
    filled = fill is not None and filled_at is not None
    return SimLlmDecision(
        decided_at=decided_at,
        mode=mode,
        session_date=session_date,
        ticker=str(ticker or ""),
        action=str(action or ""),
        reason=reason or "",
        det_side=det_side,
        det_reason=det_reason,
        relation=relation,
        filled_at=filled_at if filled else None,
        shares=fill.get("shares") if filled else None,
        price=fill.get("price") if filled else None,
        cash_after=fill.get("cash_after") if filled else None,
    )


def _ledger_chat_rows(actions: list[dict], executed: list[dict],
                      decided_at: datetime) -> list[SimLlmDecision]:
    """Rule-7 rows for one chat turn's ``[[ACTION]]`` block: one per action.

    Chat actions are operator-driven LLM decisions with no deterministic
    proposal behind them, so ``det_side``/``det_reason`` are NULL and the
    relation is ``chat_action``. An action that was skipped (no price, cash
    or position guard) keeps NULL fill columns.
    """
    fills = _executed_index(executed)
    filled_at = _utcnow()
    session_date = _chat_session_date()
    rows: list[SimLlmDecision] = []
    for a in actions:
        ticker = str(a.get("ticker") or "")
        action = str(a.get("action") or "")
        rows.append(_ledger_row(
            decided_at=decided_at,
            mode="chat",
            session_date=session_date,
            ticker=ticker,
            action=action,
            reason=str(a.get("reason") or ""),
            relation="chat_action",
            fill=fills.get((ticker.upper(), action.upper())),
            filled_at=filled_at,
        ))
    return rows


async def _ledger_llm_decisions(rows: list[SimLlmDecision]) -> None:
    """Append LLM decisions to the rule-7 ledger — best-effort, never fatal.

    One session/commit for the whole batch. A failure is logged LOUDLY and
    swallowed: the trades these rows describe are ALREADY committed real
    state (``_exec_buy``/``_exec_sell``), and trading must never roll back or
    fail because the ledger write failed — a failed write is a loud audit
    gap, not a trading failure.
    """
    if not rows:
        return
    try:
        async with Session() as s:
            s.add_all(rows)
            await s.commit()
    except Exception:
        logger.error(
            "LLM decision ledger write FAILED — %d decision(s) lost from the "
            "audit trail; the executed trades are unaffected", len(rows),
            exc_info=True)


async def _freeze_llm_decisions(rows: list[SimLlmDecision]) -> None:
    """Freeze a batch of LLM decisions (rule 7) — the hooks' entry point.

    ``_ledger_llm_decisions`` already swallows DB/commit failures; this outer
    guard also covers a bug in the writer itself, so ledger bookkeeping can
    never fail a cycle or a chat turn whose trades are already committed.
    """
    try:
        await _ledger_llm_decisions(rows)
    except Exception:
        logger.error(
            "LLM decision ledger bookkeeping FAILED — the executed trades "
            "and the cycle/chat result are unaffected", exc_info=True)


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

async def _gather_signals(tickers: list[str]) -> dict[str, dict]:
    """Fetch compute() signals for every ticker, skipping failures.

    Kept for direct callers and tests; the scheduled cycle instead uses
    ``_decision_inputs`` (the session fill contract), whose signals are
    frozen to data <= the previous completed session and fill at F's open.
    NOTE: these signals include the LATEST cached bar (possibly the
    session still in progress) and are therefore NOT contract inputs.
    """
    signals: dict[str, dict] = {}
    for t in tickers:
        try:
            rows = await candles(t)
            signals[t] = compute(rows)
        except Exception:
            continue
    return signals


# ---------------------------------------------------------------------------
# Persistent chat history for the sim portfolio manager
# ---------------------------------------------------------------------------

async def _chat_history(limit: int = 20) -> list[dict]:
    """Return the persisted sim chat history, oldest-first.

    Selects the most recent ``limit`` messages (newest-first) then reverses
    them so the returned list is in chronological order.
    """
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SimChatMessage)
                .order_by(SimChatMessage.created_at.desc())
                .limit(limit)
            )
        ).all()
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]


async def _append_chat_message(role: str, content: str) -> None:
    """Persist a single sim chat message."""
    async with Session() as s:
        s.add(SimChatMessage(role=role, content=content))
        await s.commit()


async def _clear_chat_history() -> None:
    """Delete all persisted sim chat messages."""
    async with Session() as s:
        await s.execute(delete(SimChatMessage))
        await s.commit()


async def get_chat_history(limit: int = 40) -> list[dict]:
    """Public: return the persisted sim portfolio-manager chat history."""
    return await _chat_history(limit=limit)


async def clear_chat_history() -> None:
    """Public: clear the persisted sim portfolio-manager chat history."""
    await _clear_chat_history()


def _set_progress(stage: str, detail: str = "", *, running: bool = True,
                  started_at: str | None = None, error: str | None = None) -> None:
    """Update the in-progress run-cycle state for the /api/sim/run-status poller.

    Called from run_cycle() at each stage. ``started_at`` is preserved across
    updates so the frontend can show an elapsed timer.
    """
    now = datetime.now(UTC).isoformat()
    if started_at is None:
        started_at = now
    _run_progress.update({
        "running": running,
        "stage": stage,
        "detail": detail,
        "started_at": started_at,
        "updated_at": now,
        "error": error,
    })


async def run_cycle() -> dict[str, Any]:
    """Run one full sim cycle: deposit allowance → refresh → decide → snapshot.

    This is called by the scheduler or the manual trigger endpoint.

    Concurrency: an asyncio.Lock serialises calls. A call that finds the
    lock held WAITS for the holder (bounded by ``_TICK_LOCK_WAIT_SECONDS``)
    instead of dropping the day's cycle — a DEV backfill may hold the lock
    for minutes, and a scheduled tick delayed by it is still correct, only
    one dropped is not. Only a holder wedged past the bound skips
    (error-logged). The manual HTTP path keeps its fail-fast pre-check in
    ``start_run_cycle_background`` and never queues behind a backfill.
    """
    global _last_deterministic_trades, _last_llm_reasoning, _last_llm_summary
    global _last_llm_decisions, _last_llm_vetoes

    # Bounded manual acquire (was: skip-early + `async with`): the wait is
    # bounded but the HELD time is not, so a long-but-legitimate cycle is
    # never cancelled — only the wait for the previous holder can time out.
    try:
        async with asyncio.timeout(_TICK_LOCK_WAIT_SECONDS):
            await _run_cycle_lock.acquire()
    except TimeoutError:
        logger.error("run_cycle() skipped — the cycle lock was still held after %ds",
                     _TICK_LOCK_WAIT_SECONDS)
        return {"skipped": True, "reason": "already running"}

    try:
        # Per-session guard (session fill contract, rule 2): exactly one
        # decision/fill cycle per session F. A second cycle in the same
        # session — a scheduled retry, a manual click after the scheduled
        # run, a post-restart catch-up — must not double-fill, so it bails
        # out before touching anything.
        session_key = _session_guard_key()
        if not _session_guard_passed(session_key):
            logger.info("run_cycle() skipped — session %s already processed",
                        session_key)
            return {"skipped": True,
                    "reason": f"session {session_key} already processed"}

        # ONE paper stamp for the whole cycle (deposit, trades, snapshot):
        # one logical run, one configuration. The ContextVar carries it to the
        # deep executors without changing their call signatures.
        cfg = await current_config()
        # The stamp is created first (the deposit needs it); its RunExecution
        # is recorded only AFTER the refresh/decision-inputs stage below, so
        # the provenance summary describes the cache the decision read.
        stamp = await _paper_stamp(cfg, record=False)
        stamp_token = _RUN_STAMP.set(stamp)

        # Reset all per-cycle LLM state (the whole-cycle llm/hybrid modes are
        # retired, so nothing sets them anymore); the frontend's llm_called
        # flag reads them, and this keeps a previous cycle's reasoning from
        # being re-reported as if it belonged here.
        _last_llm_reasoning = ""
        _last_llm_summary = ""
        _last_llm_decisions = []
        _last_llm_vetoes = []

        # Progress poller: mark the cycle as running with a start timestamp. Each
        # stage updates _run_progress so the frontend can show what's happening.
        _set_progress("starting", "Starting cycle")
        started_at = _run_progress["started_at"]

        try:
            # 1. Deposit allowance (if new month)
            _set_progress("allowance", "Depositing monthly allowance", started_at=started_at)
            allowance_result = await deposit_allowance(**stamp)

            # 2. Refresh candles for the universe (reuse the nightly prefetch:
            # tickers already fetched within the freshness window are skipped).
            _set_progress("refresh", "Refreshing candle data", started_at=started_at)
            tickers = await _candidate_tickers()
            _, refresh_errors = await refresh_many(
                tickers, _SIM_REFRESH_PERIOD,
                max_age_seconds=settings.market_fresh_seconds)
            detail = f" ({len(refresh_errors)} failed)" if refresh_errors else ""
            _set_progress("refresh", f"Refreshed {len(tickers)} tickers{detail}", started_at=started_at)

            # 2b. Refresh benchmark ticker and run benchmark DCA
            benchmark_result = {"deposited": False, "skipped": True}
            if settings.sim_benchmark_enabled:
                try:
                    await refresh(settings.sim_benchmark_ticker, _SIM_REFRESH_PERIOD)
                except Exception as e:
                    refresh_errors.append(f"{settings.sim_benchmark_ticker}: {e}")
                benchmark_result = await _benchmark_deposit_and_buy()

            # 3. Valuate (marks: held positions at the latest cached close —
            # after the refresh above that is session F's close, the
            # contract's mark basis).
            _set_progress("valuate", "Valuing portfolio", started_at=started_at)
            valuation = await valuate()

            # 3b. Frozen decision inputs (session fill contract): one candle
            # load, decision data truncated to <= the previous completed
            # session S, F's bar kept only as the strict fill reference.
            # Built once so every strategy path decides on identical frozen
            # inputs (bit-identical on a re-run of the same session data).
            _set_progress("session", "Preparing session decision inputs", started_at=started_at)
            held_entries = {p["ticker"]: p.get("buy_date") or None
                            for p in valuation["positions"]}
            decision = await _decision_inputs(tickers, held_entries)

            # Stamp the per-session guard BEFORE decisions/fills run: a
            # crash mid-cycle must never lead to a second fill pass for the
            # same session (contract rule 2). The stages before this point
            # (allowance, refresh, valuation) are safe to retry.
            _stamp_session_guard(session_key)

            # Mid-run configuration guard: the refresh/valuation awaits above
            # give an operator time to change the universe/risk config. The
            # cycle decided under `cfg`; executing/stamping under a changed
            # configuration would misattribute the run. Refuse BEFORE the
            # first decision write; the next cycle runs under the new config.
            if await _config_changed(cfg):
                return {"skipped": True,
                        "reason": "run configuration changed mid-run — retry under the new config"}

            # Stable input point: the refresh/frozen-input stages above wrote
            # the data the decision reads, so the run's provenance belongs
            # here (exactly once) and before any decision/trade write.
            await _record_run(stamp, cfg)

            # 4. Decide & trade
            strategy = settings.sim_strategy.lower()
            # Fundamentals quality map (SIM_LLM_FUNDAMENTALS_CONTEXT /
            # SIM_BLOCK_NEGATIVE_ROE): point-in-time ROE + P/FCF per candidate
            # from the monthly portfolio's EDGAR-first fact store — DB-only,
            # no new network calls. Empty when both features are off.
            quality: dict[str, dict[str, float | None]] | None = None
            if settings.sim_llm_fundamentals_context or settings.sim_block_negative_roe:
                _set_progress("quality", "Loading fundamentals quality map", started_at=started_at)
                quality = await _quality_map(tickers)
            if strategy == "deterministic":
                _set_progress("decide", "Deterministic engine deciding", started_at=started_at)
                proposals = await _deterministic_propose(valuation,
                                                         quality_of=_quality_lookup(quality),
                                                         decision=decision)
                # LLM sell layer, type A (the live LLM path in the sim): wired
                # BETWEEN propose and execute; stop-fire proposals carry the
                # machine-readable "stop_fire" key. Fail-safe: knob off /
                # no backend / any consult failure -> proposals unchanged.
                from . import selllayer
                proposals = await selllayer.overlay_type_a(proposals, decision)
                trades = await _execute_proposed(proposals)
                _last_deterministic_trades = trades
            else:
                # The llm/hybrid whole-cycle modes are retired: a stale
                # SIM_STRATEGY=llm/hybrid value lands here and degrades to
                # the deterministic engine (logged, never a crash).
                logger.warning("Unknown strategy '%s', falling back to deterministic", strategy)
                _set_progress("decide", "Deterministic engine deciding (unknown strategy fallback)", started_at=started_at)
                trades = await _deterministic_decide(valuation, decision=decision)
                _last_deterministic_trades = trades

            # 5. Snapshot for equity curve (marks at F's close — contract
            # rule 3; valuate() reads the latest cached close, which is F's
            # completed bar after the refresh above).
            _set_progress("snapshot", "Snapshotting equity curve", started_at=started_at)
            post_valuation = await valuate()
            await _write_snapshot(post_valuation, **stamp)

            # 5b. Benchmark snapshot
            await _benchmark_snapshot()

            _set_progress("done", f"Done — {len(trades)} trade(s)", started_at=started_at, running=False)
            return {
                "allowance": allowance_result,
                "benchmark": benchmark_result,
                "refresh_errors": refresh_errors,
                "trades": trades,
                "valuation": post_valuation,
                "llm_reasoning": _last_llm_reasoning,
            }
        except Exception as e:
            # Mark the cycle as failed so the frontend can show the error instead
            # of an indeterminate "Running..." state.
            _set_progress("error", f"{type(e).__name__}: {e}", started_at=started_at, running=False, error=str(e))
            raise
        finally:
            # The logical run is over — the frozen stamp must not leak into a
            # later standalone write in the same task.
            _RUN_STAMP.reset(stamp_token)
    finally:
        # Manual release pairs with the bounded acquire above (the body is
        # deliberately NOT re-indented into an `async with`).
        _run_cycle_lock.release()


def start_run_cycle_background() -> dict[str, Any]:
    """Launch run_cycle() as a fire-and-forget background task.

    Used by the /api/sim/run endpoint so the cycle survives browser
    disconnects — the task is not tied to the HTTP request. Returns
    immediately with {started: true} or {started: false, reason: ...}
    if a cycle is already running.
    """
    global _run_cycle_task
    if _run_cycle_lock.locked():
        return {"started": False, "reason": "already running"}

    async def _run_and_clear():
        global _run_cycle_task
        try:
            await run_cycle()
        except Exception:
            # run_cycle() already logged + set _run_progress to error.
            pass
        finally:
            _run_cycle_task = None

    _run_cycle_task = asyncio.create_task(_run_and_clear())
    return {"started": True}


# ---------------------------------------------------------------------------
# Query helpers (for API endpoints)
# ---------------------------------------------------------------------------

def get_run_progress() -> dict[str, Any]:
    """Return the current/last run-cycle progress for the frontend poller."""
    return dict(_run_progress)


def get_last_llm_summary() -> dict[str, Any]:
    """Return the reasoning-panel payload for the frontend (/api/sim/reasoning).

    The llm/hybrid whole-cycle strategy modes are retired: the cycle always
    runs the deterministic engine, so no whole-cycle LLM is ever called and
    this honestly reports ``llm_called=false`` (run_cycle resets the LLM
    state globals at the start of every cycle, so the flag is always False).
    The LLM's live role in the sim is the sell layer (app/selllayer.py) —
    its decisions appear in the SELL SCORECARD, not in this panel.
    """
    return {
        "raw": _last_llm_reasoning,
        "summary": _last_llm_summary,
        "deterministic_trades": _last_deterministic_trades,
        "strategy": settings.sim_strategy,
        "llm_called": bool(_last_llm_reasoning or _last_llm_summary),
    }


async def get_trades(limit: int = 100) -> list[dict]:
    """Raw trade-log export (all epochs, newest first).

    Deliberately NOT epoch-clipped: the audit/history listing.
    """
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SimTrade).order_by(SimTrade.created_at.desc()).limit(limit)
            )
        ).all()
        return [
            {
                "id": r.id,
                "ticker": r.ticker,
                "side": r.side,
                "shares": r.shares,
                "price": r.price,
                "cash_after": r.cash_after,
                "reason": r.reason,
                "at": r.created_at.isoformat(),
            }
            for r in rows
        ]


async def get_allowances() -> list[dict]:
    """The CURRENT epoch's deposit ledger, newest month first (epoch-scoped:
    no read may count the lifetime after a reset; closed epochs' rows stay in
    the table for audit). ``mode``/``config_id``/``epoch`` are additive keys.
    """
    async with Session() as s:
        cur = await run_config.current_epoch(_ENGINE, session=s)
        rows = (
            await s.scalars(
                select(SimAllowance)
                .where(run_config.epoch_clause(SimAllowance, cur))
                .order_by(SimAllowance.month.desc())
            )
        ).all()
        return [
            {"id": r.id, "amount": r.amount, "month": r.month, "at": r.created_at.isoformat(),
             "mode": r.mode, "config_id": r.config_id, "epoch": r.epoch}
            for r in rows
        ]


async def get_equity_curve(limit: int = 365) -> list[dict]:
    """The CURRENT epoch's equity curve, oldest first.

    Epoch-scoped at the SQL level and the limit applies AFTER the clip: a
    reset restarts the curve, so closed-epoch rows stay in the table
    (audit/history) but never leak into the chart. Each row carries its
    provenance additively (``mode``/``config_id``/``epoch``; JSON-safe None on
    legacy rows).
    """
    async with Session() as s:
        cur = await run_config.current_epoch(_ENGINE, session=s)
        rows = (
            await s.scalars(
                select(SimSnapshot)
                .where(run_config.epoch_clause(SimSnapshot, cur))
                .order_by(SimSnapshot.created_at.desc()).limit(limit)
            )
        ).all()
        # Return oldest-first for charting
        rows = list(reversed(rows))
        return [
            {
                "at": r.created_at.isoformat(),
                "cash": r.cash,
                "positions_value": r.positions_value,
                "total_equity": r.total_equity,
                "allowance_total": r.allowance_total,
                "mode": r.mode,
                "config_id": r.config_id,
                "epoch": r.epoch,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Backfill (deterministic synthetic history, no LLM calls)
# ---------------------------------------------------------------------------

async def backfill(start: str | None = None) -> dict[str, Any]:
    """Replay the DETERMINISTIC sim strategy over historical data and REPLACE
    the portfolio state with the replay's end state.

    The live sim's deterministic cycle is the live decision path (the LLM
    sell layer wires into it); ``llm``/``hybrid`` remain selectable
    strategy modes, but the LLM cannot be replayed faithfully (tokens, and
    its cloud behaviour isn't reproducible), so this backfill is a
    DETERMINISTIC APPROXIMATION: the engine's own rules only — the same
    SELL/ATR-stop/BUY logic, the same risk config (max-positions, stops,
    run-up block, cash floor), zero LLM calls. The curve approximates the
    deterministic engine; treat it as context for comparison, not a track
    record.

    Wipes account/positions/trades/allowances/snapshots (NOT the benchmark,
    NOT chat history), then walks every stored trading day from `start` to
    today, depositing the allowance on each month's first trading day and
    applying the engine's decisions at the day's close.

    Honesty note: the underlying replay (``optimize._replay``) still fills
    each decision at that decision bar's close — a different fill timing
    from the live engine's session fill contract (fills at the next
    session's open). Live/replay kernel+fill parity is a separate Phase 1
    item; treat this replay as a deterministic approximation, not a track
    record.

    The persisted end-state positions carry the replay's real entry anchor
    (``opened_at`` = the replay's actual first-buy date — NOT wall-clock
    now(), which would silently anchor the position ratchet to the wrong
    bars) and the percent stop frozen at that entry fill (contract rule 5).

    FORWARD-RECORD PROTECTION (the audit's "do not overwrite a forward
    record"): if the CURRENT epoch already holds a forward paper row, the
    replay is REFUSED before any mutation. A replay regenerates the book's
    ORIGIN; a "bounded re-backfill" (regenerate only the pre-paper synthetic
    segment) was considered and REJECTED because the seam books a spurious
    external flow (``allowance_total`` jumps between the two segments'
    baselines) and the paper book's origin would no longer match the synthetic
    run preceding it. Reset first — a reset preserves the old history as a
    closed epoch. The refusal FAILS CLOSED: a legacy NULL-mode row cannot be
    proven synthetic, so it is protected too.

    The replay's rows are stamped ``mode="backfill"`` in the current epoch; a
    RE-backfill replaces only its own rows (mode ``backfill``/NULL in the same
    epoch) — never a paper row and never a closed epoch's history. The
    benchmark tables are NOT part of the replay (its own backfill_benchmark
    writes them) and are left untouched.

    CAPITAL BASIS: this replay's baseline IS ``params.start_cash`` (the
    configured ``sim_start_cash`` seed) plus its monthly deposits, and its
    snapshot rows carry exactly that explicit contributed total — consistent
    with the live ``valuate`` seed-inclusive basis.

    ``start`` semantics (mirrors monthly.backfill / daily_core.backfill):
      - explicit "YYYY-MM-DD": replay from that day
      - "all": the full stored history (candles to 1980; the replay walks
        every stored trading day and deposits the monthly allowance on the
        first trading day of each month it covers, so the curve starts at the
        earliest stored candle rather than at the first signal)
      - None (default): synched with the other sims — the earliest snapshot
        date of the monthly / daily-core portfolios (the sim's own snapshots
        are wiped by the replay, so the other two anchor the shared window).
    """
    from sqlalchemy import delete as sa_delete
    from .db import (SimAccount as Acc, SimPosition as Pos, SimTrade as Tr,
                     SimAllowance as Al, SimSnapshot as Sn)
    from . import optimize as opt

    def opt_load_series(tickers: list[str], start=None):
        return opt._load_series(tickers, start=start)

    def opt_live_sim_params():
        return opt._live_sim_params()

    def opt_replay(series, params, start):
        return opt._replay(series, params, start=start)

    if _run_cycle_lock.locked():
        return {"skipped": True, "reason": "a sim cycle is already running"}
    async with _run_cycle_lock:
        # --- forward-record guard: BEFORE any mutation ---------------------
        cur = await run_config.current_epoch(_ENGINE, session_factory=Session)
        if await run_config.has_paper_rows(_ENGINE, cur, session_factory=Session):
            return {"ok": False,
                    "error": "a forward paper record exists for this epoch — "
                             "backfill would overwrite the live book's origin; "
                             "reset first (reset preserves history as a closed epoch)"}
        # The replay's OWN configuration snapshot — never a paper stamp (a
        # backfill must not create a paper config row while it prepares).
        cfg = await current_config()
        stamp = {
            "mode": "backfill",
            "config_id": await run_config.get_or_create(
                _ENGINE, "backfill", cfg, session_factory=Session),
            "epoch": cur,
        }
        if start is None:
            start = await _sync_start_date()
        elif start == "all":
            start = None  # full stored history
        start_note = start or "first signal day"

        tickers = await _candidate_tickers()
        series = await opt_load_series(tickers, start)
        if not series:
            return {"ok": False, "error": "no tickers with enough candle history"}

        params = opt_live_sim_params()
        res = opt_replay(series, params, start=start)
        if not res.equity_curve:
            return {"ok": False, "error": "replay produced no days"}

        # The replay's allowance cadence: contribution on the first trading
        # day of each month (the _replay convention).
        days = [e["time"] for e in res.equity_curve]
        replay_months = sorted({d[:7] for d in days})

        # The replay replaces the whole portfolio state, so the per-session
        # fill guard must reset too — the next cycle must be free to run.
        _reset_session_guard()

        # Replay state comes from the result's curves; shares/cash are not
        # exported by ReplayResult, so rebuild the end-state holdings from
        # the trade log (executing it forward from zero). The allowance
        # deposits are NOT in the trade log (_replay adds them straight to
        # pf.cash) — add one deposit per replayed month or cash ends ~$100k
        # negative and the equity curve cliffs on the live point.
        shares: dict[str, float] = {}
        # Entry anchor per END-STATE holding: the replay's actual first-buy
        # date + entry fill price. A full SELL closes the anchor — a later
        # re-buy is a NEW entry (the ratchet must not see the previous
        # holding period's bars). Contract rules 5-6.
        entry_anchor: dict[str, dict] = {}
        cash = params.start_cash
        deposited_months: set[str] = set()
        for tr in res.trades:
            d = tr["date"]
            deposited_months.add(d[:7])
            notional = tr["shares"] * tr["price"]
            if tr["side"] == "BUY":
                shares[tr["ticker"]] = shares.get(tr["ticker"], 0.0) + tr["shares"]
                cash -= notional
                if tr["ticker"] not in entry_anchor:
                    entry_anchor[tr["ticker"]] = {"date": d, "price": tr["price"]}
            else:
                shares[tr["ticker"]] = shares.get(tr["ticker"], 0.0) - tr["shares"]
                if shares[tr["ticker"]] <= 0.0001:
                    shares.pop(tr["ticker"], None)
                    entry_anchor.pop(tr["ticker"], None)
                cash += notional
        # deposits: _replay deposits on the first trading day of every month
        # it processes — including months where the portfolio stayed flat
        # (no trades), so count months from the equity curve, not the trades.
        deposited_months = {e["time"][:7] for e in res.equity_curve}
        cash += params.monthly_allowance * len(deposited_months)

        # The live engine deposits at the START of the month, so it may have
        # already deposited the CURRENT month while the replay window has no
        # candle for it yet (e.g. backfill on the 1st before the market opens).
        # The replay never re-deposits it, so restore the contribution and the
        # row — otherwise the account ends permanently short, and if the
        # marker lagged behind the row the next live deposit would hit the
        # unique allowance constraint. (Mirrors daily_core.backfill.)
        current_month = _current_month()

        # --- persist the end state: ONE transaction under the state lock ---
        # The forward-record guard, the deletions, the inserts, the account
        # re-seed/markers, the RunExecution and the commit are ONE transaction
        # serialized with standalone writers by ``_state_write_lock``: a
        # writer either committed BEFORE it (the guard refuses) or runs AFTER
        # the lock releases (its paper row lands on the new book and is never
        # erased) — it can never slip into the gap. The long replay/loader work
        # above stays OUTSIDE the lock; only the final commit block is held.
        async with _state_write_lock:
            async with Session() as s:
                if await run_config.has_paper_rows(_ENGINE, cur, session=s):
                    await s.rollback()
                    return {"ok": False,
                            "error": "a forward paper record exists for this epoch — "
                                     "backfill would overwrite the live book's origin; "
                                     "reset first (reset preserves history as a closed epoch)"}
                acc = await s.get(Acc, 1)
                live_allowance_month = None if acc is None else acc.last_allowance_month
                # The live engine deposits at the START of the month, so it may
                # already have deposited the CURRENT month while the replay
                # window has no candle for it yet (e.g. backfill on the 1st
                # before the market opens). The replay never re-deposits it, so
                # restore the contribution and the row — otherwise the account
                # ends permanently short. (Mirrors daily_core.backfill.)
                if (live_allowance_month == current_month
                        and current_month not in replay_months):
                    cash += params.monthly_allowance
                    replay_months = sorted(set(replay_months) | {current_month})
                # Only this epoch's replaceable rows go: mode backfill, or
                # legacy NULL-mode rows (cannot be proven forward). A paper row
                # would have been refused above; a CLOSED epoch's history is
                # never touched. Positions are current STATE, cleared outright.
                for tbl in (Tr, Al, Sn):
                    await s.execute(sa_delete(tbl).where(
                        run_config.epoch_clause(tbl, cur),
                        or_(tbl.mode.is_(None), tbl.mode == "backfill")))
                await s.execute(sa_delete(Pos))
                if acc is None:
                    acc = Acc(id=1, cash=0.0, opening_seed=params.start_cash)
                    s.add(acc)
                acc.cash = round(cash, 2)
                acc.last_review_week = None
                # This engine's replay baseline IS params.start_cash (the
                # configured sim_start_cash seed the synthetic book was opened
                # with) — freeze it as the epoch's capital basis.
                acc.opening_seed = params.start_cash
                # Appended in THIS transaction, after the in-tx forward guard
                # and the account seed: a refusal/failure records no execution.
                await _record_run(stamp, cfg, session=s)
                for t, sh in shares.items():
                    # avg_cost not tracked per-fill here — book at the last
                    # close like the other backfills (the equity curve is what
                    # matters).
                    last_px = None
                    df = series.get(t)
                    if df is not None and len(df):
                        last_px = float(df["close"].iloc[-1])
                    anchor = entry_anchor.get(t)
                    # opened_at: the replay's ACTUAL first-buy date (16:00 UTC
                    # on that session), NOT wall-clock now() — the monotone ATR
                    # ratchet's entry anchor (contract rule 6).
                    opened_at = (pd.Timestamp(
                        f"{anchor['date']} 16:00:00+00:00").to_pydatetime()
                        if anchor else None)
                    # stop_price: the percent stop frozen at the ENTRY fill
                    # (contract rule 5).
                    frozen_stop = (anchor["price"] * (1 - params.stop_pct / 100)
                                   if anchor and params.stop_type == "percent" else None)
                    s.add(Pos(ticker=t, shares=round(sh, 6),
                              avg_cost=round(last_px or 0.0, 6),
                              opened_at=opened_at, stop_price=frozen_stop))
                for tr in res.trades:
                    s.add(Tr(ticker=tr["ticker"], side=tr["side"],
                             shares=round(tr["shares"], 6),
                             price=round(tr["price"], 6), cash_after=0.0,
                             reason=f"backfill: {tr['reason']}",
                             mode=stamp["mode"], config_id=stamp["config_id"],
                             epoch=stamp["epoch"],
                             created_at=pd.Timestamp(
                                 f"{tr['date']} 16:00:00+00:00").to_pydatetime()))
                for m in replay_months:
                    s.add(Al(amount=settings.sim_monthly_allowance, month=m,
                             mode=stamp["mode"], config_id=stamp["config_id"],
                             epoch=stamp["epoch"]))
                # Keep the marker consistent: never let it lag the newest
                # inserted row, or the next live deposit would duplicate it.
                last_month = max(live_allowance_month or "",
                                 replay_months[-1] if replay_months else "")
                acc.last_allowance_month = last_month or None
                allowance_running = 0.0
                months_seen: set[str] = set()
                for e in res.equity_curve:
                    d = e["time"]
                    # equity_curve is ascending, so counting distinct months
                    # as we go is O(days), not a per-point rescan.
                    months_seen.add(d[:7])
                    allowance_running = round(
                        params.start_cash
                        + params.monthly_allowance * len(months_seen), 2)
                    s.add(Sn(cash=0.0, positions_value=e["equity"],
                             total_equity=e["equity"],
                             allowance_total=round(allowance_running, 2),
                             mode=stamp["mode"], config_id=stamp["config_id"],
                             epoch=stamp["epoch"],
                             created_at=pd.Timestamp(
                                 f"{d} 16:00:00+00:00").to_pydatetime()))
                await s.commit()

        return {"ok": True, "days": len(res.equity_curve),
                "start": res.equity_curve[0]["time"], "end": res.equity_curve[-1]["time"],
                "requested_start": start_note,
                "final_equity": round(res.final_equity, 2),
                "trades": res.n_trades, "snapshots": len(res.equity_curve),
                "approximation": True}


async def _sync_start_date() -> str | None:
    """Earliest MonthlySnapshot date — the sim backfill's default start so
    the sim / monthly / daily-core curves cover the same window.

    Deliberately NOT epoch-clipped: a fresh backfill may start inside a CLOSED
    epoch's window to model a shared historical experiment. It only READS that
    history; the replay's own rows are stamped into the current epoch, so
    nothing in a closed epoch is rewritten."""
    from .db import MonthlySnapshot
    async with Session() as s:
        d = await s.scalar(select(func.min(MonthlySnapshot.created_at)))
    return d.strftime("%Y-%m-%d") if d else None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

async def reset_sim() -> dict[str, Any]:
    """Close the current epoch and re-seed the book — HISTORY IS PRESERVED.

    The money ledger survives: trades, allowances and snapshots are NOT
    deleted. One ``RunEpoch`` boundary records the reset (with a note counting
    what was preserved), the current positions are cleared and the account is
    re-seeded (cash = start, month/week markers None), all in ONE transaction
    under the ENGINE LOCK (a concurrent cycle can never interleave a trade or
    a boundary with the re-seed). The equity curve restarts at the new epoch
    (``get_equity_curve`` clips to it); preserved rows keep their epoch and
    remain auditable. The LLM decision ledger (SimLlmDecision) is untouched —
    it is append-only by contract.

    CAPITAL BASIS: the new epoch is an INDEPENDENT paper capital baseline. Its
    contribution basis is the configured ``start_cash`` seed (EXTERNAL capital
    the paper book is opened with — not a real transfer) plus the epoch's own
    monthly deposits; a deposit for a month the closed epoch already funded is
    a NEW allocation in the new ledger (UNIQUE(month, epoch) lets both rows
    coexist). Epoch metrics never count the lifetime.

    DERIVED-ARTIFACT EXCEPTION: the DCA benchmark tables are a comparison
    artifact (not a decision book and not evidence of what the portfolio did),
    so ``reset_benchmark()`` still REPLACES them destructively — documented
    here because it is the one deliberate exception to "resets preserve
    history". The per-session fill guard is re-armed for the fresh book.

    Returns the new epoch and the preserved row counts additively; existing
    keys are unchanged.
    """
    async with _run_cycle_lock, _state_write_lock:
        async with Session() as s:
            # ONE transaction: boundary + counts + positions + re-seed. The
            # helpers take the caller's session and never commit behind us.
            cur = await run_config.current_epoch(_ENGINE, session=s)
            preserved = {
                "trades": (await s.scalar(
                    select(func.count()).select_from(SimTrade)
                    .where(run_config.epoch_clause(SimTrade, cur)))) or 0,
                "snapshots": (await s.scalar(
                    select(func.count()).select_from(SimSnapshot)
                    .where(run_config.epoch_clause(SimSnapshot, cur)))) or 0,
                "allowances": (await s.scalar(
                    select(func.count()).select_from(SimAllowance)
                    .where(run_config.epoch_clause(SimAllowance, cur)))) or 0,
            }
            note = (f"reset — {preserved['trades']} trades, "
                    f"{preserved['snapshots']} snapshots, "
                    f"{preserved['allowances']} allowances preserved "
                    f"as epoch {cur}")
            new_epoch = await run_config.record_boundary(_ENGINE, note, session=s)
            await s.execute(delete(SimPosition))
            acc = await s.get(SimAccount, 1)
            if acc is None:
                acc = SimAccount(id=1, cash=settings.sim_start_cash,
                                 last_allowance_month=None,
                                 opening_seed=settings.sim_start_cash)
                s.add(acc)
            else:
                acc.cash = settings.sim_start_cash
                acc.last_allowance_month = None
                acc.last_review_week = None
                # The new epoch's ACTUAL opening capital (external seed).
                acc.opening_seed = settings.sim_start_cash
            await s.commit()

        # A fresh portfolio leaves the next cycle free (per-session fill guard).
        _reset_session_guard()

        # Derived comparison artifact — destructive BY DESIGN (see docstring).
        await reset_benchmark()
    logger.info("Sim reset: cash=%.2f epoch=%s (preserved %s)",
                settings.sim_start_cash, new_epoch, preserved)
    return {"ok": True, "cash": settings.sim_start_cash,
            "epoch": new_epoch, "preserved": preserved}


# ---------------------------------------------------------------------------
# Benchmark (DCA control portfolio)
# ---------------------------------------------------------------------------

async def benchmark_valuate() -> dict[str, Any]:
    """Compute the DCA benchmark portfolio valuation."""
    async with Session() as s:
        acc = await s.get(SimBenchmarkAccount, 1)
        if acc is None:
            acc = SimBenchmarkAccount(id=1, cash=0, shares=0, avg_cost=0, last_allowance_month=None)
            s.add(acc)
            await s.commit()

    price = await _latest_close(settings.sim_benchmark_ticker)
    if price is None:
        price = acc.avg_cost or 0.0
    value = acc.shares * price

    return {
        "ticker": settings.sim_benchmark_ticker,
        "shares": round(acc.shares, 6),
        "avg_cost": round(acc.avg_cost, 4),
        "current_price": round(price, 4),
        "total_equity": round(value, 2),
        # total deposited: acc.cash tracks the cumulative allowances (see
        # _benchmark_deposit_and_buy). The old placeholder 0.0 made every
        # contributed-normalized view of the live bench divide by zero —
        # the pct conversion fell back to the RAW value and the 1Y chart
        # showed the bench at 1986% instead of ~99%.
        "allowance_total": round(acc.cash, 2),
    }


async def _benchmark_deposit_and_buy() -> dict[str, Any]:
    """Deposit the monthly allowance into the benchmark and buy the ETF.

    The benchmark always invests 100% of each allowance immediately
    (dollar-cost averaging).  Uses fractional shares so the full amount
    is always invested.
    """
    if not settings.sim_benchmark_enabled:
        return {"deposited": False, "skipped": True}

    month = _current_month()
    ticker = settings.sim_benchmark_ticker

    price = await _latest_close(ticker)
    if price is None or price <= 0:
        logger.warning("Benchmark: no price for %s, skipping deposit", ticker)
        return {"deposited": False, "skipped": True, "reason": "no price"}

    async with Session() as s:
        acc = await s.get(SimBenchmarkAccount, 1)
        if acc is None:
            acc = SimBenchmarkAccount(id=1, cash=0, shares=0, avg_cost=0, last_allowance_month=None)
            s.add(acc)

        if acc.last_allowance_month == month:
            return {"deposited": False, "skipped": False, "month": month}

        amount = settings.sim_monthly_allowance
        # Fractional shares for the benchmark (full investment)
        new_shares = amount / price
        total_shares = acc.shares + new_shares
        acc.avg_cost = (acc.shares * acc.avg_cost + amount) / total_shares if total_shares > 0 else price
        acc.shares = total_shares
        acc.cash += amount  # track total deposited via cash in/out accounting
        acc.last_allowance_month = month
        await s.commit()

        logger.info("Benchmark DCA: deposited %.2f, bought %.6f shares of %s @ %.2f",
                     amount, new_shares, ticker, price)
        return {
            "deposited": True,
            "amount": amount,
            "month": month,
            "shares": round(new_shares, 6),
            "price": round(price, 4),
        }


async def _benchmark_snapshot() -> None:
    """Take a snapshot of the benchmark portfolio for the equity curve."""
    if not settings.sim_benchmark_enabled:
        return

    val = await benchmark_valuate()
    price = val["current_price"]

    # Compute allowance total from account cash (tracks cumulative deposits)
    async with Session() as s:
        acc = await s.get(SimBenchmarkAccount, 1)
        allowance_total = acc.cash if acc else 0.0

    async with Session() as s:
        snap = SimBenchmarkSnapshot(
            shares=val["shares"],
            price=price,
            total_equity=val["total_equity"],
            allowance_total=round(allowance_total, 2),
        )
        s.add(snap)
        await s.commit()


async def get_benchmark_equity_curve(limit: int = 365) -> list[dict]:
    """Return benchmark snapshots oldest-first for charting."""
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SimBenchmarkSnapshot)
                .order_by(SimBenchmarkSnapshot.created_at.desc())
                .limit(limit)
            )
        ).all()
        rows = list(reversed(rows))
        return [
            {
                "at": r.created_at.isoformat(),
                "shares": r.shares,
                "price": r.price,
                "total_equity": r.total_equity,
                "allowance_total": r.allowance_total,
            }
            for r in rows
        ]


async def reset_benchmark() -> None:
    """Wipe benchmark tables and re-initialize."""
    async with Session() as s:
        await s.execute(delete(SimBenchmarkSnapshot))
        await s.execute(delete(SimBenchmarkAccount))
        acc = SimBenchmarkAccount(id=1, cash=0, shares=0, avg_cost=0, last_allowance_month=None)
        s.add(acc)
        await s.commit()
    logger.info("Benchmark reset")


async def backfill_benchmark(start: str | None = None) -> dict[str, Any]:
    """Replay the DCA benchmark over historical URTH candles and REPLACE the
    benchmark state with the replay's end state.

    The benchmark has no decisions to approximate: each month's allowance is
    deposited on the month's FIRST trading day and fully invested at that
    day's close (the live _benchmark_deposit_and_buy rule). Fetches the
    benchmark ticker's full available candle history first (the cache only
    carries ~2y by default), so the synthetic curve matches the other
    backfills' window.

    ``start``: "YYYY-MM-DD" to replay from that date, "all" for the full
    candle history of the ticker, or omit (default) to synch with the other
    sims (earliest monthly/daily-core snapshot).
    """
    from sqlalchemy import delete as sa_delete
    from .db import Candle, SimBenchmarkAccount as Acc
    from .market import refresh as market_refresh

    async with _run_cycle_lock:
        ticker = settings.sim_benchmark_ticker
        try:
            await market_refresh(ticker, "max")
        except Exception as e:
            logger.warning("Benchmark backfill: could not refresh %s (%s) — using cache", ticker, e)

        async with Session() as s:
            rows = (await s.scalars(select(Candle).where(Candle.ticker == ticker)
                                    .order_by(Candle.timestamp))).all()
        if not rows:
            return {"ok": False, "error": f"no candles for {ticker}"}

        dates = pd.DatetimeIndex([r.timestamp.replace(tzinfo=None) for r in rows])
        closes = pd.Series([r.close for r in rows], index=dates)

        if start is None:
            start = await _sync_start_date()
        elif start == "all":
            start = None
        start_note = start or "first candle"
        if start:
            closes = closes[closes.index >= pd.Timestamp(start)]
        if closes.empty:
            return {"ok": False, "error": f"no candles since {start_note}"}

        # DCA replay: deposit on each month's first trading day, buy at that
        # day's close (same rule as the live engine; no fees for the ETF).
        amount = settings.sim_monthly_allowance
        shares = 0.0
        cash = 0.0  # cumulative deposited (the account's contributed total)
        avg_cost = 0.0
        snaps: list[tuple[datetime, float, float, float, float]] = []
        seen_months: set[str] = set()
        n_buys = 0
        for ts in closes.index:
            px = float(closes.loc[ts])
            day = pd.Timestamp(ts)
            month = day.strftime("%Y-%m")
            if month not in seen_months:
                seen_months.add(month)
                cash += amount
                new_shares = amount / px if px > 0 else 0.0
                avg_cost = (shares * avg_cost + amount) / (shares + new_shares) \
                    if (shares + new_shares) > 0 else px
                shares += new_shares
                n_buys += 1
            snaps.append((day.to_pydatetime(), shares, px,
                          round(shares * px, 2), round(cash, 2)))

        if not snaps:
            return {"ok": False, "error": "no days in window"}

        # The live engine may already have deposited the CURRENT month while
        # the replay window ends before it (e.g. backfill on the 1st) — the
        # wipe would destroy that allowance, so preserve the marker and the
        # deposit (same rule as the other backfills).
        async with Session() as s:
            acc0 = await s.get(Acc, 1)
            live_month = acc0.last_allowance_month if acc0 else None
            current_month = _current_month()
            extra = 0.0
            if live_month == current_month and current_month not in seen_months:
                # no candle for today's month yet: book the live deposit at
                # the last close (the live engine's next cycle will buy it)
                px = float(closes.iloc[-1])
                if px > 0:
                    cash += amount
                    avg_cost = (shares * avg_cost + amount) / (shares + amount / px) \
                        if (shares + amount / px) > 0 else px
                    shares += amount / px
                    extra = amount
            await s.execute(sa_delete(SimBenchmarkSnapshot))
            await s.execute(sa_delete(Acc))
            # Marker = the newest month actually funded (or the live month when
            # it was booked above). Never `current_month` blindly: with no
            # candle for the current month that would suppress the next live
            # deposit forever (and a lagging marker would double-deposit).
            acc = Acc(id=1, cash=round(cash, 2), shares=round(shares, 6),
                      avg_cost=round(avg_cost, 6),
                      last_allowance_month=max(live_month or "", max(seen_months)))
            s.add(acc)
            for d, sh, _px, eq, contrib in snaps:
                s.add(SimBenchmarkSnapshot(
                    shares=round(sh, 6), price=0.0, total_equity=eq,
                    allowance_total=contrib,
                    created_at=d))
            await s.commit()

        return {"ok": True, "ticker": ticker, "days": len(snaps),
                "start": snaps[0][0].strftime("%Y-%m-%d"),
                "end": snaps[-1][0].strftime("%Y-%m-%d"),
                "requested_start": start or "first candle",
                "contributed": round(cash, 2), "final_equity": snaps[-1][3],
                "n_buys": n_buys, "extra_current_month": round(extra, 2)}


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

_scheduler_task: asyncio.Task | None = None
_daily_snapshot_task: asyncio.Task | None = None

# --- Deferred scheduled retries -------------------------------------------
#
# A scheduled tick that finds its portfolio lock held by a DEV backfill or a
# manual run must DELAY the scheduled action, never silently drop it. The
# tick keeps its immediate pre-check (skip the stage this wake so the later
# stages still run), and schedules this fire-and-forget retry: the first
# attempt waits _DEFERRED_RETRY_DELAY_SECONDS (long enough that a
# test-driven tick never wakes its retry inside the test), then polls the
# lock until it frees, bounded by _DEFERRED_RETRY_WAIT_SECONDS. The action
# reference is captured at SCHEDULE time, so a test's monkeypatched fake is
# what a retry would call.
_DEFERRED_RETRY_DELAY_SECONDS = 60.0
_DEFERRED_RETRY_WAIT_SECONDS = 7200.0
_deferred_retry_tasks: set[asyncio.Task] = set()


def _schedule_deferred_retry(run, *, label: str, lock: asyncio.Lock) -> None:
    """Defer a scheduled action behind a busy lock (fire-and-forget).

    The retry does NOT hold the lock across ``run()`` — the actions acquire
    the lock themselves (run_cycle waits bounded; take_snapshot skips), so
    holding it here would deadlock them. It only waits for the lock to be
    FREE, then calls the captured reference.
    """

    async def _retry() -> None:
        waited = 0.0
        while lock.locked():
            if waited >= _DEFERRED_RETRY_WAIT_SECONDS:
                logger.error("%s: the lock was still held %.0fs after the "
                             "scheduled tick — giving up on this deferred retry",
                             label, waited)
                return
            await asyncio.sleep(_DEFERRED_RETRY_DELAY_SECONDS)
            waited += _DEFERRED_RETRY_DELAY_SECONDS
        try:
            result = await run()
            if isinstance(result, dict) and result.get("skipped"):
                logger.info("%s: deferred retry skipped — %s",
                            label, result.get("reason"))
            else:
                logger.info("%s: deferred retry complete", label)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a deferred tick must not kill the loop
            logger.error("%s: deferred retry failed: %s", label, e, exc_info=True)

    task = asyncio.create_task(_retry())
    _deferred_retry_tasks.add(task)
    task.add_done_callback(_deferred_retry_tasks.discard)


# --- Trading-day calendar (US NYSE) ---------------------------------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _observed(d: date) -> date:
    """NYSE 'nearest weekday' rule: Saturday → Friday, Sunday → Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _us_market_holidays(year: int) -> set[date]:
    from dateutil.easter import easter
    days = {
        _observed(date(year, 1, 1)),      # New Year's Day
        _nth_weekday(year, 1, 0, 3),      # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),      # Washington's Birthday
        easter(year) - timedelta(days=2), # Good Friday
        _last_weekday(year, 5, 0),        # Memorial Day
        _observed(date(year, 7, 4)),      # Independence Day
        _nth_weekday(year, 9, 0, 1),      # Labor Day
        _nth_weekday(year, 11, 3, 4),     # Thanksgiving
        _observed(date(year, 12, 25)),    # Christmas
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth (NYSE since 2022)
    return days


_holiday_cache: dict[int, set[date]] = {}


def is_trading_day(dt: datetime) -> bool:
    """True when ``dt``'s date is a US market day (Mon-Fri, not a NYSE holiday).

    The portfolios are almost entirely US listings and the scheduler runs at
    ``sim_run_hour`` UTC, so the NYSE calendar governs. EU holidays (e.g.
    Vienna) are not modelled — those names just show a stale candle and the
    engine no-ops. Guards the prefetch and all scheduled cycles so nothing runs
    on weekends/holidays.
    """
    d = dt.date()
    if d.weekday() >= 5:
        return False
    for y in (d.year - 1, d.year, d.year + 1):
        if y not in _holiday_cache:
            _holiday_cache[y] = _us_market_holidays(y)
        if d in _holiday_cache[y]:
            return False
    return True


async def prefetch_universe() -> dict[str, Any]:
    """Fetch the shared sim universe into the candle cache ONCE.

    All three portfolios share one universe, so fetching it here (before the
    nightly cycles) means they read the DB instead of each re-pulling it. Uses
    the configured provider — with ``auto`` Twelve Data serves the tickers it
    has, paced by the rate limiter and capped by the daily budget, and every
    miss (``IFX.DE`` etc.) falls back to yfinance per ticker. Marks each fetched
    ticker fresh so run_cycle()/daily-core skip it.
    """
    tickers = await _candidate_tickers()
    if not tickers:
        return {"ok": False, "reason": "empty universe"}
    refreshed, errors = await refresh_many(tickers, _SIM_REFRESH_PERIOD, use_provider=True)
    logger.info("Universe prefetch %s: %d/%d fetched, %d errors",
                _universe(), len(refreshed), len(tickers), len(errors))
    return {"ok": True, "universe": _universe(), "total": len(tickers),
            "refreshed": len(refreshed), "errors": errors}


async def _scheduler_tick() -> None:
    """One scheduler wake: monthly rebalance, then the daily sim cycle, then
    the daily-core cycle.

    Factored out of ``_scheduler_loop`` (which only sleeps between ticks) so
    the "each stage runs independently" behaviour is unit-testable. A stage
    that is skipped or fails must not prevent the later stages from running.
    """
    # Nothing trades on weekends or NYSE holidays: skip the whole tick so the
    # portfolios aren't marked/rebalanced on stale candles. The month-end
    # rebalance always lands on a trading day, so it is never lost.
    if not is_trading_day(_utcnow()):
        logger.info("Sim scheduler: %s is not a trading day — skipping all cycles",
                    _utcnow().date())
        return

    # Monthly qv-mom portfolio first: it only acts on the last trading
    # day of the month, and a missed month has no catch-up — it must run
    # even when the daily cycle is skipped below (a manual run holding
    # the lock at wake time), or the whole month is silently missed.
    try:
        from .monthly import run_monthly_cycle
        monthly_result = await run_monthly_cycle()
        if monthly_result.get("skipped"):
            logger.info("Monthly scheduler: skipped — %s", monthly_result.get("reason"))
        else:
            logger.info("Monthly rebalance complete: %d trades", len(monthly_result.get("trades", [])))
    except Exception as e:
        logger.error("Monthly rebalance failed: %s", e, exc_info=True)

    try:
        # Skip if a manual run / DEV backfill is in flight (e.g. the user
        # clicked "Run Bot Now" shortly before the scheduled time). The lock
        # check in run_cycle() also guards against this, but checking here too
        # avoids logging a confusing "skipped — already running" entry every
        # scheduled night a manual run overlaps — and lets this tick return
        # promptly instead of blocking behind a multi-minute DEV backfill.
        # The scheduled cycle itself is DEFERRED, never dropped: the
        # fire-and-forget retry waits for the lock (bounded) and then runs
        # the same run_cycle. NOTE: this must not skip the daily-core stage
        # below — it runs regardless.
        if _run_cycle_lock.locked():
            logger.info("Sim scheduler: a cycle is already in progress — deferring "
                        "the scheduled run (never dropped)")
            _schedule_deferred_retry(run_cycle, label="Sim scheduler cycle",
                                     lock=_run_cycle_lock)
        else:
            result = await run_cycle()
            if result.get("skipped"):
                logger.info("Sim scheduler: run_cycle skipped — %s", result.get("reason"))
            else:
                logger.info("Sim cycle complete: %d trades", len(result["trades"]))
    except Exception as e:
        logger.error("Sim cycle failed: %s", e, exc_info=True)

    # Daily-core portfolio last: it depends on the freshly refreshed
    # candles above (fundamental ranking + daily deployment) and must
    # not race the monthly rebalance. Runs every trading day the
    # scheduler fires — that's the whole point (daily cash deployment).
    try:
        from .daily_core import run_daily_cycle
        dc_result = await run_daily_cycle()
        if dc_result.get("skipped"):
            logger.info("Daily-core scheduler: skipped — %s", dc_result.get("reason"))
        else:
            logger.info("Daily-core cycle complete: %d trades",
                        len(dc_result.get("deployment", {}).get("trades", [])))
    except Exception as e:
        logger.error("Daily-core cycle failed: %s", e, exc_info=True)


async def _scheduler_loop():
    """Background loop that runs the sim cycle daily at sim_run_hour UTC."""
    while True:
        now = _utcnow()
        # Calculate seconds until next sim_run_hour (interpreted as UTC)
        target = now.replace(hour=settings.sim_run_hour, minute=settings.sim_run_minute, second=0, microsecond=0)
        if target <= now:
            # Already past today's run hour — schedule for tomorrow
            target = target + timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.info("Sim scheduler: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)

        await _scheduler_tick()


async def _daily_monthly_snapshot_loop():
    # Imported here (not at module top) to avoid a circular import:
    # monthly imports market/screener lazily at call time.
    from .monthly import (_rebalance_lock, deposit_allowance, refresh_holdings,
                         take_snapshot)

    """Background loop that marks the Monthly qv-mom portfolio once a day.

    Sequence (all idempotent):
      1. Deposit the allowance when a new operator-local month has begun
         (START of month, mirroring the sim portfolio's deposit timing so the
         two "contributed" figures stay in lockstep).
      2. Refresh candles for the current holdings so valuation uses current
         prices.
      3. Write one MonthlySnapshot so the Monthly equity curve moves daily
         instead of only at month-end rebalances.

    The last-trading-day rebalance still runs in _scheduler_loop; on those
    days take_snapshot() skips itself while the rebalance lock is held, and
    the rebalance's own post-trade snapshot is recorded instead.

    Disabled while sim_monthly_enabled is False: the loop keeps sleeping on
    its schedule but performs no deposits, refreshes or snapshots, matching
    the gate in monthly.run_monthly_cycle().
    """
    while True:
        now = _utcnow()
        # Daily mark runs 5 minutes before the main cycle so it never contends
        # with the rebalance that _scheduler_loop fires at sim_run_hour.
        target = now.replace(hour=settings.sim_run_hour, minute=settings.sim_run_minute,
                             second=0, microsecond=0) - timedelta(minutes=5)
        if target <= now:
            # Today's mark already passed — schedule for tomorrow
            target = target + timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.info("Monthly daily snapshot: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)
        if not settings.sim_monthly_enabled:
            continue
        # No new prices on weekends/holidays: a daily mark would just repeat the
        # last close. Deposits/rebalances are idempotent and land on the next
        # trading day or the month-end rebalance.
        if not is_trading_day(_utcnow()):
            logger.info("Monthly daily scheduler: %s is not a trading day — skipping",
                        _utcnow().date())
            continue
        try:
            deposit = await deposit_allowance()
            if deposit.get("deposited"):
                logger.info("Monthly daily scheduler: allowance deposited for %s", deposit.get("month"))
            refresh = await refresh_holdings()
            if refresh.get("errors"):
                logger.warning("Monthly daily scheduler: refresh errors: %s", refresh["errors"])
            snap = await take_snapshot()
            if snap.get("skipped"):
                reason = snap.get("reason", "")
                if "rebalance" in reason:
                    # A DEV backfill or manual rebalance holds the lock (a
                    # mark racing it mid-execution would distort the curve).
                    # The day's mark is DEFERRED, never dropped: a delayed
                    # mark beats a hole in the equity curve.
                    logger.warning("Monthly daily scheduler: mark deferred — %s", reason)
                    _schedule_deferred_retry(take_snapshot,
                                             label="Monthly daily mark",
                                             lock=_rebalance_lock)
                else:
                    logger.info("Monthly daily scheduler: snapshot skipped — %s", reason)
            else:
                logger.info("Monthly daily scheduler: snapshot equity=%.2f", snap.get("total_equity", 0))
        except Exception as e:
            logger.error("Monthly daily snapshot failed: %s", e, exc_info=True)


def start_scheduler():
    """Start the sim scheduler tasks (called from main.py lifespan).

    The universe prefetch is NOT here: it lives in main.py as an app-level task
    so the Dashboard keeps fresh candles / a fresh screener even when the sim
    is disabled."""
    global _scheduler_task, _daily_snapshot_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_scheduler_loop())
    if _daily_snapshot_task is None or _daily_snapshot_task.done():
        _daily_snapshot_task = asyncio.create_task(_daily_monthly_snapshot_loop())


def stop_scheduler():
    """Stop the background scheduler task."""
    global _scheduler_task, _daily_snapshot_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
    _scheduler_task = None
    if _daily_snapshot_task and not _daily_snapshot_task.done():
        _daily_snapshot_task.cancel()
    _daily_snapshot_task = None
    # Deferred scheduled retries belong to the scheduler lifecycle: a
    # shutdown must not leave a task that could fire a cycle afterwards.
    for task in list(_deferred_retry_tasks):
        task.cancel()
    _deferred_retry_tasks.clear()


# ---------------------------------------------------------------------------
# Interactive LLM chat (sim portfolio manager)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pending chat proposals (Phase 3, slice 3D): an [[ACTION]] block only
# PROPOSES trades — they execute solely when the user explicitly confirms the
# proposal (POST /api/sim/chat/confirm → confirm_chat_proposal). The registry
# is in-memory on purpose: an unconfirmed proposal books NOTHING (no stamp,
# no ledger row, no snapshot, no trade) and dies with the process — a reload
# drops it and a confirm then 404s with an honest reason.
# ---------------------------------------------------------------------------

# A pending proposal expires after this long; afterwards confirm/reject
# answer 404 and the UI must ask the LLM again.
_PROPOSAL_TTL_SECONDS = 30 * 60
# Registry cap: beyond this many pending proposals the OLDEST is evicted.
_PROPOSAL_CAP = 20

# proposal_id -> proposal dict (built by _build_chat_proposal).
_pending_proposals: dict[str, dict] = {}
_pending_proposals_lock: asyncio.Lock = asyncio.Lock()


class ProposalNotFound(LookupError):
    """The proposal id is unknown or expired — confirm/reject answer 404."""


class ProposalConfigChanged(Exception):
    """The effective run configuration moved since the proposal — 409."""


def _proposal_expired(proposal: dict) -> bool:
    """TTL check against the proposal's frozen parse-time ``decided_at``."""
    try:
        decided_at = datetime.fromisoformat(proposal["decided_at"])
        age = (_utcnow() - decided_at).total_seconds()
    except Exception:  # noqa: BLE001 — a malformed entry fails safe (stale)
        return True
    return age > _PROPOSAL_TTL_SECONDS


async def _register_proposal(proposal: dict) -> dict:
    """Store one proposal, sweeping expired entries and enforcing the cap
    (oldest first — dict insertion order)."""
    async with _pending_proposals_lock:
        for pid in [pid for pid, p in _pending_proposals.items()
                    if _proposal_expired(p)]:
            _pending_proposals.pop(pid, None)
        _pending_proposals[proposal["proposal_id"]] = proposal
        while len(_pending_proposals) > _PROPOSAL_CAP:
            oldest = next(iter(_pending_proposals))
            logger.info("Chat proposal %s evicted (pending cap %d reached)",
                        oldest, _PROPOSAL_CAP)
            _pending_proposals.pop(oldest, None)
    return proposal


async def _pop_pending_proposal(proposal_id: str) -> dict:
    """Remove and return the pending proposal, or raise ProposalNotFound.

    Unknown AND expired both mean "not found" to the caller (the entry is
    dropped either way, so a stale id can never be confirmed twice).
    """
    async with _pending_proposals_lock:
        proposal = _pending_proposals.pop(proposal_id, None)
    if proposal is None or _proposal_expired(proposal):
        raise ProposalNotFound("proposal not found or expired")
    return proposal


# The preview keys exposed on the wire. The internal plan computed by
# _action_preview additionally carries the exact execution params
# (``target_shares`` / ``budget``) the confirm executor feeds to
# _exec_sell / _exec_buy — same math, never re-derived.
_CHAT_PREVIEW_KEYS = ("feasible", "issue", "price", "est_shares", "est_value",
                      "est_cash_after", "est_position_value_after")


async def _action_preview(action: dict, valuation: dict | None = None,
                          acc: SimAccount | None = None) -> dict:
    """Validate + size ONE chat action against the CURRENT state.

    This is the SINGLE source of the per-action feasibility/size math: the
    proposal builder calls it to estimate what a confirm would do, and the
    confirm executor calls it again with FRESH state to re-validate right
    before executing — the preview can never diverge from the execution rules
    because both run this exact code (prices/cash/positions may move between
    proposal and confirm; the confirm response then shows the fresh numbers).

    Returns a plan dict with the wire preview fields (``feasible``, ``issue``,
    ``price``, ``est_shares``, ``est_value``, ``est_cash_after``,
    ``est_position_value_after``) plus the execution params (``target_shares``
    for SELL, ``budget`` for BUY). ``issue`` is a machine-readable skip reason
    when not feasible: "no price", "no position", "insufficient cash",
    "position at max", "budget < 1".
    """
    ticker = str(action.get("ticker", ""))
    act = str(action.get("action", "")).upper()
    if valuation is None:
        valuation = await valuate()
    if acc is None:
        acc = await _account()
    plan: dict[str, Any] = {
        "feasible": False, "issue": None, "price": None,
        "est_shares": None, "est_value": None, "est_cash_after": None,
        "est_position_value_after": None,
        "target_shares": None, "budget": None,
    }
    price = await _latest_close(ticker)
    plan["price"] = price
    if price is None or price <= 0:
        plan["issue"] = "no price"
        return plan

    total_equity = valuation["total_equity"]
    min_cash = total_equity * (settings.sim_min_cash_pct / 100)
    max_position_value = total_equity * (settings.sim_max_position_pct / 100)

    async with Session() as s:
        pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))

    if act == "SELL":
        if pos is None or pos.shares <= 0:
            plan["issue"] = "no position"
            return plan
        # Optional partial size: "shares" (exact) or "amount" (dollars).
        target: float | None = None
        if "shares" in action:
            target = action["shares"]
        elif "amount" in action:
            target = action["amount"] / price
        est_shares = pos.shares if target is None else min(target, pos.shares)
        if est_shares <= 0:
            plan["issue"] = "budget < 1"
            return plan
        proceeds = est_shares * price
        plan.update(
            feasible=True, target_shares=est_shares,
            est_shares=est_shares, est_value=round(proceeds, 2),
            est_cash_after=round(acc.cash + proceeds, 2),
            est_position_value_after=round((pos.shares - est_shares) * price, 2),
        )
        return plan

    if act == "BUY":
        # Normally the cash reserve is a hard floor. The user can explicitly
        # authorise spending it via "use_reserve": true — then the buy may
        # spend down to zero cash.
        floor = 0.0 if action.get("use_reserve") else min_cash
        if acc.cash < floor:
            plan["issue"] = "insufficient cash"
            return plan
        current_value = (pos.shares * price) if pos else 0.0
        if current_value >= max_position_value:
            plan["issue"] = "position at max"
            return plan
        max_budget = min(acc.cash - floor, max_position_value - current_value)
        # Optional partial size: "shares" or "amount" (dollars), clamped to
        # the risk-limited budget so cash/position limits are never breached.
        if "shares" in action:
            budget = min(max_budget, action["shares"] * price)
        elif "amount" in action:
            budget = min(max_budget, action["amount"])
        else:
            budget = max_budget
        if budget < 1:
            plan["issue"] = "budget < 1"
            return plan
        # Same fractional-share floor as _exec_buy (never round up past budget).
        est_shares = math.floor(budget / price * 10000) / 10000
        if est_shares < 0.0001:
            plan["issue"] = "budget < 1"
            return plan
        cost = est_shares * price
        plan.update(
            feasible=True, budget=budget,
            est_shares=est_shares, est_value=round(cost, 2),
            est_cash_after=round(acc.cash - cost, 2),
            est_position_value_after=round(current_value + cost, 2),
        )
        return plan

    plan["issue"] = "unknown action"
    return plan


async def _build_chat_proposal(actions: list[dict], cfg_at_start: dict,
                               decided_at: datetime) -> dict:
    """Build + register ONE pending proposal for a parsed [[ACTION]] block.

    Shape: {"proposal_id": uuid4 hex, "decided_at": <parse-time freeze, rule 7,
    as an ISO string so the proposal is JSON-serializable end-to-end (SSE
    included)>, "config": <the turn's cfg_at_start, so a confirm can compare>,
    "actions": [<parsed action fields> + {"preview": <_action_preview against
    the CURRENT state>}]}. Estimates only — the confirm executor re-validates
    with fresh state.
    """
    proposal: dict[str, Any] = {
        "proposal_id": uuid.uuid4().hex,
        "decided_at": decided_at.isoformat(),
        "config": cfg_at_start,
        "actions": [],
    }
    valuation = await valuate()
    acc = await _account()
    for action in actions:
        plan = await _action_preview(action, valuation, acc)
        proposal["actions"].append({
            **action,
            "preview": {k: plan[k] for k in _CHAT_PREVIEW_KEYS},
        })
    await _register_proposal(proposal)
    return proposal


async def confirm_chat_proposal(proposal_id: str) -> dict:
    """Execute ONE pending chat proposal (the user explicitly confirmed it).

    Flow: pop the pending entry (ProposalNotFound when unknown/expired) →
    config gate: if the effective configuration changed since the proposal,
    raise ProposalConfigChanged (the entry is already dropped) → else resolve
    the stamp with the DECIDED config (the decision's configuration, not
    confirm-time's), record the run once, execute every action through the
    SAME validation/sizing helper the preview used — re-run against FRESH
    prices/state per action — snapshot if anything traded, and freeze the
    ledger with the proposal's frozen parse-time decided_at (rule 7).

    Returns {"ok": True, "trades": [...], "skipped": [{"ticker", "reason"}],
    "snapshot_taken": bool}.
    """
    proposal = await _pop_pending_proposal(proposal_id)
    cfg = proposal["config"]
    if await _config_changed(cfg):
        raise ProposalConfigChanged(
            "run configuration changed since the proposal — ask again")
    decided_at = datetime.fromisoformat(proposal["decided_at"])
    actions = proposal["actions"]

    # Chat trades and marks are forward records: one paper stamp for the whole
    # confirmed turn, resolved with the DECIDED configuration and recorded
    # once after the config gate (a refused confirm never appears in the
    # execution ledger).
    stamp = await _paper_stamp(cfg, record=False)
    await _record_run(stamp, cfg)

    executed_trades: list[dict] = []
    skipped: list[dict] = []
    for action in actions:
        ticker = action["ticker"]
        act = action["action"]
        reason = action["reason"]
        # Fresh state per action: an earlier action in the same proposal may
        # have moved cash/positions, so the shared math is re-run, never reused.
        plan = await _action_preview(action, await valuate(), await _account())
        if not plan["feasible"]:
            logger.info("Sim chat confirm %s %s skipped: %s", act, ticker,
                        plan["issue"])
            skipped.append({"ticker": ticker, "reason": plan["issue"]})
            continue
        price = plan["price"]
        if act == "SELL":
            t = await _exec_sell(ticker, price, plan["target_shares"],
                                 f"User chat: {reason}", **stamp)
        else:
            override_tag = " [reserve spent]" if action.get("use_reserve") else ""
            t = await _exec_buy(ticker, price, plan["budget"],
                                f"User chat:{override_tag} {reason}", **stamp)
        if t:
            executed_trades.append(t)
        else:
            skipped.append({"ticker": ticker, "reason": "execution refused"})

    snapshot_taken = False
    if executed_trades:
        post_val = await valuate()
        await _write_snapshot(post_val, **stamp)
        snapshot_taken = True

    # Rule 7: freeze this turn's actions with the PROPOSAL's parse-time stamp
    # (best-effort — a skipped action keeps NULL fill columns, a ledger
    # failure never fails the confirm).
    await _freeze_llm_decisions(
        _ledger_chat_rows(actions, executed_trades, decided_at))

    return {"ok": True, "trades": executed_trades, "skipped": skipped,
            "snapshot_taken": snapshot_taken}


async def reject_chat_proposal(proposal_id: str) -> dict:
    """Drop a pending proposal without executing anything (no DB writes)."""
    await _pop_pending_proposal(proposal_id)  # ProposalNotFound when gone
    return {"ok": True}


_SIM_CHAT_SYSTEM_PROMPT = (
    "You are a portfolio manager for a paper-trading simulation bot. "
    "The user can chat with you about the current portfolio and ask you "
    "to take actions. You have access to the current portfolio state, "
    "technical signals, and recent news.\n"
    "\n"
    + _SIM_METHODOLOGY +
    "Rules:\n"
    "1. You can discuss the portfolio, explain your decisions, and answer questions.\n"
    "2. If the user asks you to take an action (e.g. \"sell X to buy Y\"), include "
    "a SPECIAL ACTION block at the end of your response in this format:\n"
    "   [[ACTION]]\n"
    "   {\"actions\": [{\"ticker\": \"...\", \"action\": \"BUY\"|\"SELL\", \"reason\": \"...\"}]}\n"
    "   [[/ACTION]]\n"
    "   The actions in this block are only PROPOSED to the user: they execute "
    "solely after the user explicitly confirms them in the UI. Never say or "
    "imply that the trades were executed — say they are proposed and are "
    "waiting for the user's confirmation.\n"
    "3. You may propose multiple actions in one response (e.g. sell one ticker, buy another).\n"
    "4. You can specify a partial position size per action using optional fields:\n"
    "   - \"shares\": exact number of shares to trade (e.g. 3.5).\n"
    "   - \"amount\": dollar amount to trade (e.g. 67.43). For SELL this is the"
    " value of shares to sell; for BUY it is the dollars to invest.\n"
    "   If neither is given, SELL sells the entire position and BUY invests the"
    " maximum allowed by risk rules.\n"
    "   Example: {\"actions\": [{\"ticker\": \"MDB\", \"action\": \"SELL\", \"amount\": 67.43, \"reason\": \"trim overweight\"}]}\n"
    "5. The same risk management rules apply: respect min cash %, max position %,"
    " and the max open-positions count; the engine will clamp your requested"
    " amounts to stay within them and will refuse a BUY that would exceed the"
    " position count cap.\n"
    "   EXCEPTION: if the user EXPLICITLY asks to spend the cash reserve / dry"
    " powder / remaining cash / \"all available cash\", set \"use_reserve\": true"
    " on that BUY action. This lets the buy spend below the normal min-cash floor"
    " (down to zero cash). Use it only when the user clearly requests it — never"
    " on your own initiative.\n"
    "6. The engine enforces an initial stop loss (a fixed % below the "
    "position's average cost) and an ATR reference stop on every position "
    "(the highest close of the previous 20 completed bars minus 2x ATR of "
    "the previous bar — a reference level recomputed every bar, not a "
    "ratchet). Positions that hit either are auto-sold by the deterministic "
    "layer, so if the user asks why a position vanished it was likely "
    "stopped out — explain that rather than proposing to re-buy it unless "
    "the user explicitly asks.\n"
    "7. The max position % is a buy-time sizing limit, not a ceiling to enforce "
    "on exits. Do NOT sell a position just because its price rose above it — let "
    "winners run. Only trim a position (with an exact \"shares\" or \"amount\", or "
    "the whole position) when it is genuinely overweight and the user asks you to.\n"
    "8. Only propose actions you believe are justified by the signals and portfolio context.\n"
    "9. If you do not agree with the user's request, explain why and omit the ACTION block.\n"
    "10. Do NOT reference specific URLs in your output.\n"
    "11. The \"Last Sim Cycle Decisions\" section in your context lists decisions "
    "that were already executed by the simulation. Treat them as historical — "
    "when asked about them, explain them, but do NOT include them as new actions "
    "unless the user explicitly asks you to take a new trade.\n"
    "12. Be cautious about selling one ticker to buy another (portfolio rotation). "
    "Backtesting showed this reduces returns because tickers with similar "
    "indicator profiles tend to perform similarly — the \"stronger opportunity\" "
    "is rarely actually stronger. If the user asks for a swap, explain the risk "
    "and only proceed if they insist.\n"
)


def _parse_action_block(text: str) -> list[dict] | None:
    """Extract a [[ACTION]]...[[/ACTION]] JSON block from LLM text."""
    if not text:
        return None
    start = text.find("[[ACTION]]")
    end = text.find("[[/ACTION]]")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start + len("[[ACTION]]"):end].strip()
    try:
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None
    actions = parsed.get("actions", [])
    if not isinstance(actions, list):
        return None
    valid = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        ticker = a.get("ticker", "").upper().strip()
        action = str(a.get("action", "")).upper().strip()
        reason = str(a.get("reason", "")).strip()
        if not ticker or action not in ("BUY", "SELL"):
            continue
        entry: dict[str, Any] = {"ticker": ticker, "action": action, "reason": reason}
        # Optional partial-size fields (validated by the caller against the
        # current position / cash).
        for field in ("shares", "amount"):
            val = a.get(field)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                entry[field] = float(val)
        # Optional override: user explicitly authorised spending the cash reserve.
        if a.get("use_reserve") is True:
            entry["use_reserve"] = True
        valid.append(entry)
    return valid if valid else None


async def _gather_sim_chat_news(limit: int = 6) -> dict[str, list[dict]]:
    """Fetch recent SearXNG news for the held tickers + market for the sim chat.

    Returns ``{}`` if SearXNG is disabled or nothing is found. Limited to the
    first ``limit`` held tickers so repeated chat turns stay cheap (results are
    cached in news.py for 4h anyway).
    """
    from .news import gather_news_for_candidates

    valuation = await valuate()
    portfolio_tickers = [p["ticker"] for p in valuation["positions"]][:limit]
    if not portfolio_tickers:
        return {}
    return await gather_news_for_candidates(portfolio_tickers, include_market=True)


async def _build_sim_chat_context() -> str:
    """Build context for the sim chat: portfolio + signals + recent news."""
    valuation = await valuate()
    tickers = await _candidate_tickers()
    signals = await _gather_signals(tickers)

    lines: list[str] = []
    lines.append("## Current Portfolio State")
    lines.append(f"Cash: {valuation['cash']:.2f}")
    lines.append(f"Positions value: {valuation['positions_value']:.2f}")
    lines.append(f"Total equity: {valuation['total_equity']:.2f}")
    lines.append(f"Allowance total: {valuation['allowance_total']:.2f}")
    lines.append(f"Strategy: {settings.sim_strategy}")
    lines.append(f"Max position %: {settings.sim_max_position_pct}")
    lines.append(f"Min cash %: {settings.sim_min_cash_pct}")
    lines.append(f"Max open positions: {settings.sim_max_positions}")
    lines.append(f"Stop loss: {settings.sim_stop_pct:.0f}% below the position's entry price (frozen at entry — a top-up cannot loosen it; monotone ATR ratchet also applies)")
    lines.append(f"Min cash floor (buy-time only, {settings.sim_min_cash_pct}%): {valuation['total_equity'] * settings.sim_min_cash_pct / 100:.2f}")
    lines.append(f"Max position size ({settings.sim_max_position_pct}%): {valuation['total_equity'] * settings.sim_max_position_pct / 100:.2f}")
    lines.append("")

    if valuation["positions"]:
        lines.append("Open positions:")
        for p in valuation["positions"]:
            lines.append(
                f"  - {p['ticker']}: {p['shares']} shares @ avg {p['avg_cost']:.2f} "
                f"| current {p['current_price']:.2f} | value {p['value']:.2f} "
                f"| P&L {p['pnl_pct']:+.2f}%"
            )
    else:
        lines.append("Open positions: none")
    lines.append("")

    # Signals summary (only tickers with non-HOLD or in portfolio)
    portfolio_tickers = {p["ticker"] for p in valuation["positions"]}
    interesting = {
        t: sig for t, sig in signals.items()
        if sig["action"] != "HOLD" or t in portfolio_tickers
    }
    if interesting:
        lines.append("## Signals (interesting tickers)")
        for ticker, sig in sorted(interesting.items()):
            snap = sig.get("snapshot", {})
            wk = "up" if snap.get("weekly_trend_up") else "dn"
            rsi_d = snap.get("rsi_3d_change")
            mh_d = snap.get("macd_hist_3d_change")
            run5 = snap.get("run_5d")
            run20 = snap.get("run_20d")
            run60 = snap.get("run_60d")
            dist52 = snap.get("dist_52w_high")
            rsi_d_s = f" | rsiΔ3 {rsi_d:+.1f}" if rsi_d is not None else ""
            mh_d_s = f" | mhΔ3 {mh_d:+.2f}" if mh_d is not None else ""
            run5_s = f" | run5d {run5:+.1f}%" if run5 is not None else ""
            run20_s = f" | run20d {run20:+.1f}%" if run20 is not None else ""
            run60_s = f" | run60d {run60:+.1f}%" if run60 is not None else ""
            dist52_s = f" | hi52 {dist52:+.1f}%" if dist52 is not None else ""
            lines.append(
                f"  {ticker}: {sig['action']} (strength {sig['strength']}) "
                f"| RSI {snap.get('rsi', 0):.1f}{rsi_d_s} | ADX {snap.get('adx', 0):.0f} "
                f"| wk {wk} | MACD {snap.get('macd', 0):.3f}{mh_d_s}{run5_s}{run20_s}{run60_s}{dist52_s} "
                f"| close {snap.get('close', 0):.2f}"
            )
    lines.append("")

    # Recent news (supplementary, via SearXNG). The system prompt promises the
    # LLM access to recent news — actually deliver it here so it can answer
    # "what's the latest on X" questions from real headlines.
    try:
        news = await _gather_sim_chat_news()
        if news:
            from .news import format_news_for_context, format_market_news_for_context
            lines.append("## Recent News (supplementary context — do not trade on news alone)")
            market_hl = news.get("market", [])
            if market_hl:
                lines.append(format_market_news_for_context(market_hl))
            for ticker in sorted(news.keys()):
                if ticker == "market":
                    continue
                hl = news.get(ticker, [])
                if hl:
                    lines.append(format_news_for_context(ticker, hl))
            lines.append("")
    except Exception as e:
        logger.warning("sim chat news gather failed: %s", e)

    # Last sim cycle decision (the most recent cycle's executed trades).
    # Included so the chat can answer questions about why the bot traded.
    # Marked as already-executed historical decisions — the LLM should not
    # re-propose them as new ACTIONs.
    if _last_llm_decisions or _last_deterministic_trades:
        lines.append("## Last Sim Cycle Decisions (already executed — informational)")
        if _last_deterministic_trades:
            lines.append("Deterministic engine proposed:")
            for t in _last_deterministic_trades:
                lines.append(
                    f"  - {t['ticker']} {t['side']} ×{t.get('shares', '?')} @ "
                    f"{t.get('price', '?'):.2f} — {t.get('reason', '')}"
                )
        if _last_llm_decisions:
            lines.append("LLM decided:")
            for d in _last_llm_decisions:
                lines.append(
                    f"  - {d['ticker']}: {d['action']} — {d['reason']}"
                )
        lines.append("")

    return "\n".join(lines)


async def sim_chat(messages: list[dict]) -> dict[str, Any]:
    """Interactive chat with the sim portfolio manager LLM.

    ``messages`` is the list of NEW messages for this turn (typically just the
    user's latest message). The full prior conversation is loaded from
    persistent storage, merged, sent to the LLM, and the new messages are
    saved so the thread survives page reloads.

    If the LLM includes an [[ACTION]] block, its actions are only PROPOSED:
    they come back as one pending proposal (with a server-computed preview per
    action) and execute solely via confirm_chat_proposal after the user
    explicitly confirms. A turn that only proposes records NOTHING — no
    stamp, no ledger row, no snapshot, no trade.

    Returns {"text": display_text, "proposals": [...], "history": history}.
    The old "trades"/"actions_executed" keys are gone: the envelope is honest
    about the fact that this endpoint never executes.
    """
    # Load persisted history and fold in this turn's incoming messages.
    history = await _chat_history(limit=40)
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        history.append({"role": role, "content": content})
        if role == "user":
            await _append_chat_message("user", content)

    if not history:
        return {"text": "No messages to send.", "proposals": [], "history": []}

    context = await _build_sim_chat_context()
    # The turn's configuration is captured BEFORE the (potentially slow) LLM
    # call; the proposal carries it so a confirm can detect a configuration
    # change that happened while the proposal was pending.
    cfg_at_start = await current_config()

    try:
        out = await llm_mod.chat([
            {"role": "system", "content": _SIM_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ] + history)
    except Exception as e:
        logger.warning("Sim chat LLM call failed: %s", e)
        return {"text": f"LLM unavailable: {e}", "proposals": [], "history": history}

    content = out["text"]
    # Rule 7: frozen when the response is PARSED — a later confirm's ledger
    # write must never re-stamp it.
    decided_at = _utcnow()

    # Check for action block
    actions = _parse_action_block(content)

    # Strip the action block from the visible text so it isn't shown to the
    # user or persisted into history.
    display_text = content
    if actions:
        start = content.find("[[ACTION]]")
        end = content.find("[[/ACTION]]")
        if start != -1 and end != -1:
            display_text = (content[:start] + content[end + len("[[/ACTION]]"):]).strip()

    # Persist the assistant reply so the thread survives reloads and the LLM
    # gets full prior context on the next turn.
    if display_text:
        await _append_chat_message("assistant", display_text)
        history.append({"role": "assistant", "content": display_text})

    # Propose only — never execute here. The pending proposal books nothing;
    # confirm_chat_proposal is the sole execution path.
    proposals: list[dict] = []
    if actions:
        proposals.append(await _build_chat_proposal(actions, cfg_at_start, decided_at))

    return {"text": display_text, "proposals": proposals, "history": history}


async def sim_chat_stream(messages: list[dict]):
    """Streaming variant of sim_chat — yields progressive LLM text deltas.

    Yields dicts:
      - {"type": "delta", "text": "..."}: raw LLM content chunks (may include
        the ``[[ACTION]]`` block as it is generated — the frontend will swap the
        raw text for the stripped display_text when ``done`` arrives).
      - {"type": "done", "text": display_text, "proposals": [...],
         "raw": full_raw_text, "history": [...]} (plus "error" on the error
        path).

    Persistence (user message, assistant display_text) happens AFTER the LLM
    stream completes, so a refresh mid-stream leaves the previous state intact
    instead of an orphan user question.

    If the LLM stream errors out after yielding some text, a partial ``done``
    event is emitted with ``error`` set, so the frontend can show what arrived.

    Like sim_chat, this only PROPOSES: an [[ACTION]] block becomes a pending
    proposal in the done event; execution happens solely via
    confirm_chat_proposal after explicit user confirmation.
    """
    history = await _chat_history(limit=40)
    new_user_contents: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        history.append({"role": role, "content": content})
        if role == "user":
            new_user_contents.append(content)

    if not history:
        yield {
            "type": "done",
            "text": "No messages to send.",
            "proposals": [],
            "raw": "",
            "history": [],
        }
        return

    context = await _build_sim_chat_context()
    # The turn's configuration is captured BEFORE the (potentially slow) LLM
    # stream; the proposal carries it so a confirm can detect a configuration
    # change that happened while the proposal was pending.
    cfg_at_start = await current_config()
    full_parts: list[str] = []
    error_msg: str | None = None

    # Stream the LLM response. We collect the full raw text so we can parse the
    # action block after the stream finishes; the frontend receives raw deltas
    # so the user sees text as it arrives (action block included, stripped later).
    try:
        async for evt in llm_mod.chat_stream([
            {"role": "system", "content": _SIM_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ] + history):
            etype = evt.get("type")
            if etype == "delta":
                chunk = evt.get("text") or ""
                if chunk:
                    full_parts.append(chunk)
                    yield {"type": "delta", "text": chunk}
            elif etype == "thinking":
                # Forward so the frontend can keep the typing indicator alive
                # while a reasoning model (Qwen3) has no visible text yet.
                yield {"type": "thinking"}
            elif etype == "error":
                error_msg = evt.get("text") or "unknown error"
                break
            elif etype == "done":
                # If the streamer collected text we didn't see as deltas
                # (e.g. reasoning-only backend), surface it as a delta.
                seen = "".join(full_parts)
                if evt.get("text") and not seen:
                    full_parts.append(evt["text"])
                    yield {"type": "delta", "text": evt["text"]}
                break
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"

    raw_content = "".join(full_parts)

    # If nothing arrived and we hit an error, emit an assistant-style error msg.
    if not raw_content and error_msg:
        err_text = f"LLM unavailable: {error_msg}"
        # Persist user messages even on failure so the question isn't lost.
        for c in new_user_contents:
            await _append_chat_message("user", c)
        await _append_chat_message("assistant", err_text)
        history.append({"role": "assistant", "content": err_text})
        yield {
            "type": "done",
            "text": err_text,
            "proposals": [],
            "raw": "",
            "history": history,
            "error": error_msg,
        }
        return

    # Parse any action block from the complete raw text.
    actions = _parse_action_block(raw_content)
    # Rule 7: the turn's decision is frozen at PARSE time (the empty-error
    # path above returned before this point — no decision was made).
    decided_at = _utcnow()
    display_text = raw_content
    if actions:
        start = raw_content.find("[[ACTION]]")
        end = raw_content.find("[[/ACTION]]")
        if start != -1 and end != -1:
            display_text = (
                raw_content[:start] + raw_content[end + len("[[/ACTION]]"):]
            ).strip()

    # Persist NOW that the LLM has finished. User messages first, then the
    # assistant reply — so a refresh mid-stream never sees an orphan question.
    for c in new_user_contents:
        await _append_chat_message("user", c)
    if display_text:
        await _append_chat_message("assistant", display_text)
        history.append({"role": "assistant", "content": display_text})

    # Propose only — never execute here (same contract as sim_chat).
    proposals: list[dict] = []
    if actions:
        proposals.append(await _build_chat_proposal(actions, cfg_at_start, decided_at))

    yield {
        "type": "done",
        "text": display_text,
        "proposals": proposals,
        "raw": raw_content,
        "history": history,
        **({"error": error_msg} if error_msg else {}),
    }
