"""Daily-core paper portfolio: the monthly qv-mom strategy as the fundamental
CORE, with daily candle-driven cash deployment on top.

Design (from the §10 backtest A/B, docs/llm-strategy-experiments.md):
  * WHAT to own is decided by the monthly qv-mom ranking — ROE + 12-1
    momentum + 1/P-FCF, top-N with hysteresis, NO stops, NO technical
    entry gates. Fundamentals are the basis for everything.
  * WHEN/HOW is the only thing candles touch: every trading day (at the
    scheduler's run time) free cash is deployed into the highest-ranked
    core names, topping each up toward the equal-weight target. Fresh
    contributions go to work immediately instead of waiting for the
    month-end rebalance — that cash-drag elimination is the measured edge
    (+3.4..+7.5pp IRR over the monthly sim on 2020-2026 windows; positive
    out-of-sample on all four walk-forward windows, see §10).
  * Band releases: same hysteresis as the monthly sim — a held name keeps
    its slot while it stays inside the top ``hold_band`` of the ranking.
    The RANKING is the completed month-end's frame (the month-end band
    policy, session fill contract app/session.py): it changes only when a
    newer month-end completes, so a release fires on the FIRST SESSION
    after that month-end (at that session's strict exact-date open) and is
    structurally empty in between. The backtest showed exit cadence is
    irrelevant; the band is sticky.

Point-in-time discipline mirrors the monthly sim: a decision on month-end
ME only sees fundamental facts with end < ME and filed <= ME, and candles
<= S (the previous completed session) — F's bar never informs it. No
look-ahead.

Scheduler entry: ``run_daily_cycle()`` — deposits the allowance at the start
of a new operator-local month (same timing as sim/monthly so the three
portfolios' "contributed" figures stay comparable), refreshes data, runs one
deployment pass, and writes the daily equity snapshot.
"""
import asyncio
import json
import logging
import math
import os
import threading

import numpy as np
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from . import fundamentals as fundamentals_mod
from . import monthly as monthly_mod
from .config import settings
from .db import (DailyCoreAccount, DailyCoreAllowance, DailyCorePosition,
                 DailyCoreSnapshot, DailyCoreTrade, Session)
from .screener import tickers as universe_tickers, universe_names

logger = logging.getLogger("trade_sentinel.daily_core")

_TZ = monthly_mod._TZ  # same operator-local month anchor as sim/monthly

# ---------------------------------------------------------------------------
# Strategy selection (runtime, persisted — the LLM-backend pattern)
# ---------------------------------------------------------------------------

# The selectable momentum variants. `raw` = classic 12-1 close/close return;
# `residual` = Blitz-Huij-Martens alpha t-stat (§11 walk-forward winner).
STRATEGY_VARIANTS: dict[str, str] = {
    "raw": "Raw 12-1 momentum (classic qv-mom)",
    "residual": "Residual momentum (Blitz-Huij-Martens, §11 winner)",
}

_STATE_FILE = Path("/data/daily_core_state.json")
# Serialises read-modify-write of the state file: the runtime setters and the
# protection decision all read the file, change one key, and write it back, so
# two concurrent writers could drop each other's field. The critical sections
# are tiny, so a plain threading lock is fine (they run in the event loop).
_state_lock = threading.Lock()


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("Could not read daily-core state file: %s", e)
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a concurrent reader never sees a half-written file.
        tmp = _STATE_FILE.with_suffix(_STATE_FILE.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, _STATE_FILE)
    except Exception as e:
        logger.warning("Could not write daily-core state file %s: %s", _STATE_FILE, e)


def _reset_basket_state(state: dict) -> None:
    """Drop the gradient basket chain. It is chained from a specific
    universe/variant's target band, so a switch must not splice the old price
    series into the new one (the protection overlay reads it next cycle)."""
    for k in ("basket_hist", "basket_day", "basket_neg_streak",
              "basket_pos_streak", "basket_out"):
        state.pop(k, None)


def current_variant() -> str:
    """The effective momentum variant: the persisted runtime choice if valid,
    else the env/config default. Reads the file per call (tiny JSON, called
    from the scheduler path once per day and the status endpoint) so a
    hand-edited state file takes effect immediately."""
    state = _load_state()
    v = state.get("mom_variant")
    return v if v in STRATEGY_VARIANTS else settings.sim_daily_core_mom_variant


def set_variant(variant: str) -> None:
    if variant not in STRATEGY_VARIANTS:
        raise ValueError(f"unknown mom variant: {variant!r}")
    with _state_lock:
        state = _load_state()
        state["mom_variant"] = variant
        _reset_basket_state(state)  # the band (thus the basket) changes with variant
        _save_state(state)


# ---------------------------------------------------------------------------
# Strategy universe selection (runtime, persisted, same pattern as the variant)
# ---------------------------------------------------------------------------

def persisted_universe() -> str | None:
    """The runtime-selected universe if it names a real universe file, else
    None. Shared by every sim (daily sim, monthly, daily-core) so the one
    Dashboard control sets the universe everywhere."""
    u = _load_state().get("universe")
    return u if u in universe_names() else None


def current_universe() -> str:
    """The effective qv-mom universe (monthly + daily-core rankings, and thus
    backfills): the persisted runtime choice if valid, else the config default."""
    return persisted_universe() or settings.sim_monthly_universe


def set_universe(name: str) -> None:
    if name not in universe_names():
        raise ValueError(f"unknown universe: {name!r}")
    with _state_lock:
        state = _load_state()
        state["universe"] = name
        _reset_basket_state(state)  # the basket chain belongs to the old universe
        _save_state(state)


# ---------------------------------------------------------------------------
# Protection selection (runtime, persisted, same pattern as the variant)
# ---------------------------------------------------------------------------

# Protection overlays for the live engine: TWO independent switches, mirroring
# the §12-15 backtest parameters (`--exposure-trend` gate + the gradient arm).
# Defaults are OFF (the live engine is unchanged unless the operator picks one).
#   gate     — the deployment gate: park new buys while the market index is
#              below its N-day SMA (off / 200 / 100 / 50).
#   gradient — the basket-slope cash-out. "always" fires ungated (the §13
#              behaviour); an SMA value arms it only in "good times" (market
#              above its N-day SMA, the §14/§15 arm). Re-entry is always
#              unconditional; the gradient's own 10-day/3-close window is fixed.
GATE_MODES: dict[str, str] = {
    "off": "No deployment gate — deploy normally",
    "200": "Park buys while below SMA200",
    "100": "Park buys while below SMA100",
    "50": "Park buys while below SMA50",
}
GRADIENT_MODES: dict[str, str] = {
    "off": "No gradient cash-out",
    "always": "Cash out on a negative basket slope (always armed)",
    "200": "Cash out in good times only (above SMA200)",
    "100": "Cash out in good times only (above SMA100)",
    "50": "Cash out in good times only (above SMA50)",
}

# Legacy single-mode selection -> (gate, gradient). Used to migrate an existing
# state file written before the two switches were split.
_LEGACY_PROTECTION = {
    "none": ("off", "off"),
    "trend200": ("200", "off"),
    "gradient200": ("200", "200"),
    "gradient100": ("100", "100"),
    "gradient50": ("50", "50"),
}

# Gradient window/confirmation for the live signal (the §15 backtest values:
# 10-day slope, 3 consecutive closes).
_GRADIENT_DAYS = 10
_GRADIENT_CONFIRM = 3


def _protection_settings(state: dict) -> tuple[str, str]:
    """(gate, gradient) modes: the persisted pair, else migrated from the
    legacy single `protection` key, else both off."""
    gate = state.get("gate")
    gradient = state.get("gradient_arm")
    if gate in GATE_MODES and gradient in GRADIENT_MODES:
        return gate, gradient
    legacy = state.get("protection")
    return _LEGACY_PROTECTION.get(legacy, ("off", "off")) if isinstance(legacy, str) \
        else ("off", "off")


def current_gate() -> str:
    return _protection_settings(_load_state())[0]


def current_gradient() -> str:
    return _protection_settings(_load_state())[1]


def current_protection_config() -> tuple[int | None, bool, int | None]:
    """Resolved overlays: ``(gate_sma, gradient_active, arm_sma)``.

    ``gate_sma``/``arm_sma`` are None when off; ``arm_sma`` is also None for
    the ungated ("always") gradient. ``gradient_active`` is False when the
    cash-out is disabled."""
    gate, gradient = _protection_settings(_load_state())
    gate_sma = None if gate == "off" else int(gate)
    arm_sma = None if gradient in ("off", "always") else int(gradient)
    return gate_sma, gradient != "off", arm_sma


def set_gate(mode: str) -> None:
    if mode not in GATE_MODES:
        raise ValueError(f"unknown gate mode: {mode!r}")
    with _state_lock:
        state = _load_state()
        _, gradient = _protection_settings(state)  # read under the lock
        state["gate"] = mode
        state["gradient_arm"] = gradient
        state.pop("protection", None)  # drop the legacy key once migrated
        _save_state(state)


def set_gradient(mode: str) -> None:
    if mode not in GRADIENT_MODES:
        raise ValueError(f"unknown gradient mode: {mode!r}")
    with _state_lock:
        state = _load_state()
        gate, _ = _protection_settings(state)  # read under the lock
        state["gate"] = gate
        state["gradient_arm"] = mode
        state.pop("protection", None)  # drop the legacy key once migrated
        _save_state(state)


# ---------------------------------------------------------------------------
# Target-volatility exposure control (runtime, persisted, same pattern)
# ---------------------------------------------------------------------------

# Barroso-Santa-Clara vol management — the broad-universe comparison winner:
# when the portfolio's OWN 21-day realized vol exceeds the target, new
# contributions are parked in cash instead of deployed. It only scales
# deployment DOWN (a calm portfolio stays fully invested); it never sells an
# existing position and never leverages. Measured on the broad S&P 500
# universe: target 0.15 cut max drawdown 33.6% -> 28.0% for ~3pp of IRR and a
# slightly BETTER Sharpe (0.90 -> 0.92) — unlike the binary §12-15 cash-out
# overlays, which destroy value on a broad universe.
TARGET_VOL_MODES: dict[str, str] = {
    "off": "Off (deploy all cash — no vol control)",
    "0.10": "Target 10% volatility",
    "0.15": "Target 15% volatility",
    "0.20": "Target 20% volatility",
    "0.25": "Target 25% volatility",
}

# Realized-vol lookback (trading days), matching the backtest's window.
_VOL_WINDOW = 21

# Candle-history bounds. The strategy only needs a momentum-warmup window
# (>=253 trading days) before its decision date; broad universes were fetched
# with full history, so loading everything made every call slow.
_BACKFILL_WARMUP_DAYS = 600   # ~400 trading days of warmup before the replay start
_LIVE_HISTORY_DAYS = 1460     # ~4y: ample for the 253d momentum + 200d market SMA


def current_target_vol() -> float:
    """The effective annualized target vol (0.0 = off): the persisted runtime
    choice if valid, else the config default."""
    mode = _load_state().get("target_vol")
    if mode == "off":
        return 0.0
    if mode in TARGET_VOL_MODES:
        return float(mode)
    return settings.sim_daily_core_target_vol


def set_target_vol(mode: str) -> None:
    if mode not in TARGET_VOL_MODES:
        raise ValueError(f"unknown target-vol mode: {mode!r}")
    with _state_lock:
        state = _load_state()
        state["target_vol"] = mode
        _save_state(state)


def _vol_deploy_room(equity: float, cash: float, port_rets: list[float],
                     target_vol: float) -> float:
    """How much NEW cash may be deployed before the portfolio's projected vol
    exceeds the target. ``inf`` when vol is calm or not yet measurable (never
    blocks deployment on missing history)."""
    if not target_vol or len(port_rets) < _VOL_WINDOW:
        return float("inf")
    rv = float(np.std(port_rets[-_VOL_WINDOW:]) * math.sqrt(252.0))
    if rv <= target_vol or rv <= 0.0:
        return float("inf")
    return max(equity * (target_vol / rv) - (equity - cash), 0.0)


def _port_return(prev_equity: float, equity: float, contrib_today: float) -> float:
    """Contribution-adjusted log return for the realized-vol tracker (same
    definition as the backtest's per-day ``port_rets``)."""
    base = prev_equity + contrib_today
    return float(math.log(equity / base)) if base > 0 and equity > 0 else 0.0


async def _recent_port_returns(days: int) -> list[float]:
    """Contribution-adjusted daily log returns from stored snapshots — the
    live twin of the backtest's ``port_rets``. Deduped to the last snapshot
    per calendar day (manual cycles can add extras)."""
    async with Session() as s:
        rows = (await s.scalars(select(DailyCoreSnapshot)
                                .order_by(DailyCoreSnapshot.created_at.desc())
                                .limit(days * 3 + 5))).all()
    by_day: dict[str, DailyCoreSnapshot] = {}
    for r in rows:
        # rows are newest-first: keep the first seen per day = the latest
        # snapshot that day (a plain assignment would keep the oldest).
        by_day.setdefault(r.created_at.strftime("%Y-%m-%d"), r)
    seq = [by_day[k] for k in sorted(by_day)][-(days + 1):]
    return [_port_return(p.total_equity, c.total_equity,
                         max(c.allowance_total - p.allowance_total, 0.0))
            for p, c in zip(seq, seq[1:], strict=False)]


def _now() -> datetime:
    """The live decision path's clock (UTC) — injectable for tests.

    Every ``datetime.now(UTC)`` on a decision path routes through this hook
    so the golden parity tests (tests/test_cadence_parity.py) can drive the
    live deployment session by session through a fake multi-session market;
    production returns real UTC now."""
    return datetime.now(UTC)


def _local_now() -> datetime:
    return _now().astimezone(_TZ)


def _current_month() -> str:
    return _local_now().strftime("%Y-%m")


def _live_history_start() -> datetime:
    """Naive-UTC lower bound for the live ranking's candle load (~4y). Candle
    timestamps are stored naive, so the bound must be naive too."""
    return _now().replace(tzinfo=None) - timedelta(days=_LIVE_HISTORY_DAYS)


async def _stored_month_end(now: datetime) -> pd.Timestamp | None:
    """The ranking decision's month-end under the month-end band policy.

    The last calendar month-end <= S (the previous completed session —
    monthly._prev_trading_day/_candidate_month_end, the same session
    calendar as the monthly engine) whose own candle is stored (a data hole
    at the decision day is never papered over — monthly._candle_stored_on).
    One calendar walk + one count query — no frame load, so the per-session
    cycle stays cheap; the expensive frame is the per-month-end cache
    below. None when no completed month-end lies behind us."""
    s = monthly_mod._prev_trading_day(now)
    if s is None:
        return None
    me = monthly_mod._candidate_month_end(s)
    if me is None:
        return None
    if not await monthly_mod._candle_stored_on(
            universe_tickers(current_universe()), me):
        return None
    return pd.Timestamp(me).normalize()


# The month-end ranking frame cache: the fundamentals math is the expensive
# part (~7s full-universe), and under the month-end band policy the SAME
# frame ranks every session of a month — compute it once per (universe,
# variant, month-end) and reuse. Keyed by the resolved decision inputs so a
# universe/variant switch cannot splice the old ranking in; entries for
# older month-ends are pruned (only the newest decision ranks).
_FRAME_CACHE: dict[tuple[str, str, pd.Timestamp],
                   tuple[pd.DataFrame | None, list[str]]] = {}


async def _core_frame(me: pd.Timestamp) -> tuple[pd.DataFrame | None, list[str]]:
    """(eligibility frame, full ranking order) as of the completed month-end
    `me` — cached per (universe, variant, month-end).

    The frame is computed from candles <= me (eligible_frame truncates
    internally), so appending later sessions can never change it: the
    ranking is a month-end constant, which is exactly the deployment's band
    policy. A dead frame (no eligible names) caches as (None, [])."""
    key = (current_universe(), current_variant(), me)
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    tickers = universe_tickers(current_universe())
    fund = await fundamentals_mod.load_fundamentals(tickers)
    if not fund:
        frame, order = None, []
    else:
        close, vol = await monthly_mod.load_frames(
            tickers, pd.Timestamp(me).to_pydatetime(), start=_live_history_start())
        if close.empty:
            frame, order = None, []
        else:
            settings.sim_daily_core_mom_variant = current_variant()
            frame = await asyncio.to_thread(monthly_mod.eligible_frame, me, close, vol, fund)
            if frame is None or not bool(frame["eligible"].any()):
                frame, order = None, []
            else:
                _elig, order = await asyncio.to_thread(monthly_mod._qv_order, frame)
    for k in [k for k in _FRAME_CACHE if k[2] < me]:
        _FRAME_CACHE.pop(k, None)
    _FRAME_CACHE[key] = (frame, order)
    return frame, order


# ---------------------------------------------------------------------------
# Protection overlay: live market gate + target-basket gradient
# ---------------------------------------------------------------------------

def _above_sma(mkt: pd.Series, days: int) -> bool:
    """True when the index's latest close is at/above its `days`-day SMA
    (warmup / non-finite -> True, i.e. don't block)."""
    sma = mkt.rolling(days).mean()
    cur, s = mkt.iloc[-1], sma.iloc[-1]
    return not (np.isfinite(s) and np.isfinite(cur) and float(cur) < float(s))


async def _protection_decision(band: list[str],
                               gate_sma: int | None,
                               gradient_active: bool,
                               arm_sma: int | None,
                               close: pd.DataFrame | None = None,
                               ) -> tuple[bool, bool]:
    """Evaluate the two independent protection overlays: ``(market_ok, basket_out)``.

    Market gate: the equal-weight universe index vs its `gate_sma`-day SMA.
    ``market_ok`` is False below it, so contributions park. ``gate_sma`` None
    (gate off) -> always True.

    Gradient: the target basket (the day's top-N band names) is chained one
    day at a time from stored closes; a confirmed negative N-day slope
    (`_GRADIENT_DAYS`, `_GRADIENT_CONFIRM`) sets ``basket_out``. It is armed
    only in good times when `arm_sma` is set (market above that SMA) and always
    armed when `arm_sma` is None; re-entry is unconditional. Evaluated (and
    persisted) only when ``gradient_active`` so the cash-out is skipped while
    disabled. Streaks and the basket history persist in the daily-core state
    file so the signal is stable across cycles (the scheduler runs once a day).

    ``close`` is the candle frame the overlays advance on — session fill
    contract rule 1: the caller passes it truncated to sessions <= S (the
    previous completed session), so F's bar never informs the overlay. None
    (test convenience) loads the universe's frame unbounded."""
    if close is None:
        close, _vol = await monthly_mod.load_frames(
            universe_tickers(current_universe()), None, start=_live_history_start())
    if close is None or close.empty or not band:
        return True, False

    mkt = close.ffill().mean(axis=1)
    # Deployment gate (independent of the gradient arm).
    market_ok = _above_sma(mkt, gate_sma) if gate_sma and gate_sma > 1 else True

    # Basket chain: day-over-day mean log return of the band's top-N names.
    cv = close.ffill()
    hist = _load_state().get("basket_hist") or []
    if not isinstance(hist, list) or not hist:
        hist = [100.0]
    last_day = _load_state().get("basket_day")
    idx = cv.index
    start_i = 0
    if last_day:
        # First UNprocessed day: `after` holds the days past the last one
        # already folded into `hist`, so the next append must start there.
        # (Starting one earlier re-appended the last stored day every call,
        # roughly doubling the chained basket growth and the slope.)
        after = idx[idx > pd.Timestamp(last_day)]
        start_i = len(idx) - len(after)
    top = band[:settings.sim_monthly_target_n]
    for i in range(max(start_i, 1), len(idx)):
        d_prev, d_cur = idx[i - 1], idx[i]
        rets = []
        for t in top:
            p0, p1 = monthly_mod.px_at(cv, d_prev, t), monthly_mod.px_at(cv, d_cur, t)
            if p0 and p1 and p0 > 0 and p1 > 0:
                rets.append(float(np.log(p1 / p0)))
        if rets:
            hist.append(hist[-1] * float(np.exp(float(np.mean(rets)))))
    hist = hist[-260:]  # bounded history

    out = False
    neg = 0
    pos = 0
    if gradient_active and len(hist) > _GRADIENT_DAYS:
        armed = _above_sma(mkt, arm_sma) if arm_sma and arm_sma > 1 else True
        slope = hist[-1] / hist[-1 - _GRADIENT_DAYS] - 1.0
        state = _load_state()
        neg = int(state.get("basket_neg_streak") or 0)
        pos = int(state.get("basket_pos_streak") or 0)
        was_out = bool(state.get("basket_out") or False)
        if slope < 0:
            neg, pos = neg + 1, 0
        elif slope > 0:
            pos, neg = pos + 1, 0
        if not was_out and armed and neg >= _GRADIENT_CONFIRM:
            was_out = True
        elif was_out and pos >= _GRADIENT_CONFIRM:
            was_out = False
        out = was_out
    with _state_lock:
        state = _load_state()
        state.update({"basket_hist": hist, "basket_day": str(idx[-1].date()),
                      "basket_neg_streak": neg if gradient_active else 0,
                      "basket_pos_streak": pos if gradient_active else 0,
                      "basket_out": out})
        _save_state(state)
    return market_ok, out


# Serialises cycles: the allowance check runs before the (potentially
# multi-minute) data refresh, so two overlapping invocations would both
# pass the gate and double-deposit / duplicate trades. Mirrors
# sim._run_cycle_lock and monthly._rebalance_lock.
_cycle_lock: asyncio.Lock = asyncio.Lock()

# Serialises the (uncached) target computation. The status endpoint calls
# compute_targets() unlocked and the UI fires two status requests on first
# tab load, so without this both would run the ~7s full-universe ranking in
# parallel and race on the stored-ranking write.
_targets_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

async def valuate(now: datetime | None = None) -> dict:
    """Value the daily-core portfolio (same shape as monthly.monthly_valuate).

    ``now`` bounds the marks' clock: prices use stored closes <= now (the
    cycle's own session F — session fill contract rule 3); None keeps the
    historical behavior (the latest stored close)."""
    async with Session() as s:
        acc = await _account(s)
        positions = (await s.scalars(
            select(DailyCorePosition).order_by(DailyCorePosition.ticker))).all()
        allowance_total = (await s.scalar(
            select(func.sum(DailyCoreAllowance.amount)))) or 0.0

    px_map = await monthly_mod._price_usd_map([p.ticker for p in positions],
                                              asof=now)
    positions_value = 0.0
    pos_list = []
    for p in positions:
        price = px_map.get(p.ticker)
        if price is None:
            price = p.avg_cost
        value = p.shares * price
        positions_value += value
        pnl_pct = ((price - p.avg_cost) / p.avg_cost * 100) if p.avg_cost > 0 else 0.0
        pos_list.append({
            "ticker": p.ticker,
            "shares": round(p.shares, 4),
            "avg_cost": round(p.avg_cost, 4),
            "current_price": round(price, 4),
            "value": round(value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "opened_at": p.opened_at.strftime("%Y-%m-%d") if p.opened_at else "",
        })
    return {
        "cash": round(acc.cash, 2),
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(acc.cash + positions_value, 2),
        "allowance_total": round(float(allowance_total), 2),
    }


async def _account(s) -> DailyCoreAccount:
    acc = await s.get(DailyCoreAccount, 1)
    if acc is None:
        acc = DailyCoreAccount(id=1, cash=settings.sim_monthly_start_cash,
                               last_allowance_month=None)
        s.add(acc)
        await s.commit()
    return acc


# ---------------------------------------------------------------------------
# Allowance / execution
# ---------------------------------------------------------------------------

async def deposit_allowance() -> dict:
    """Deposit the monthly allowance if a new operator-local month has begun.

    Deposits at the START of the month (mirroring sim + monthly portfolios)
    so the cumulative "contributed" figures step together. Idempotent per
    month. The money is deployed by the daily cycles that follow.
    """
    month = _current_month()
    async with Session() as s:
        acc = await _account(s)
        if acc.last_allowance_month == month:
            return {"deposited": False, "month": month, "cash": acc.cash}
        acc.cash += settings.sim_monthly_contribution
        acc.last_allowance_month = month
        s.add(DailyCoreAllowance(amount=settings.sim_monthly_contribution, month=month))
        try:
            await s.commit()
        except IntegrityError:
            # The status endpoint calls this unlocked, and the UI fires two
            # concurrent status requests on tab load — at a month rollover
            # both can read the stale marker and insert the same month. The
            # UNIQUE(month) constraint catches the loser here; treat it as
            # "already deposited" instead of returning a 500.
            await s.rollback()
            acc = await _account(s)
            logger.info("Daily-core allowance for %s already deposited (race)", month)
            return {"deposited": False, "month": month, "cash": acc.cash}
        logger.info("Daily-core allowance deposited: %.2f for %s -> cash %.2f",
                    settings.sim_monthly_contribution, month, acc.cash)
        return {"deposited": True, "amount": settings.sim_monthly_contribution,
                "month": month, "cash": acc.cash}


async def _exec_buy(ticker: str, price: float, budget: float, reason: str) -> dict | None:
    if price <= 0 or budget < 1:
        return None
    shares = math.floor(budget / price * 10000) / 10000
    if shares < 0.0001:
        return None
    cost = shares * price
    async with Session() as s:
        acc = await _account(s)
        if acc is None or acc.cash < cost:
            return None
        acc.cash -= cost
        pos = await s.scalar(select(DailyCorePosition).where(DailyCorePosition.ticker == ticker))
        if pos:
            total = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + cost) / total
            pos.shares = total
        else:
            s.add(DailyCorePosition(ticker=ticker, shares=shares, avg_cost=price))
        s.add(DailyCoreTrade(ticker=ticker, side="BUY", shares=shares, price=price,
                             cash_after=acc.cash, reason=reason))
        await s.commit()
        return {"ticker": ticker, "side": "BUY", "shares": shares, "price": price,
                "cost": round(cost, 2), "reason": reason}


async def _exec_sell(ticker: str, price: float, reason: str) -> dict | None:
    if price <= 0:
        # No usable price (e.g. a backfilled avg_cost of 0): booking $0
        # proceeds would delete the position for nothing. Mirror _exec_buy.
        return None
    async with Session() as s:
        pos = await s.scalar(select(DailyCorePosition).where(DailyCorePosition.ticker == ticker))
        if pos is None:
            return None
        proceeds = pos.shares * price
        acc = await _account(s)
        if acc is None:
            return None
        acc.cash += proceeds
        s.add(DailyCoreTrade(ticker=ticker, side="SELL", shares=pos.shares, price=price,
                             cash_after=acc.cash, reason=reason))
        await s.delete(pos)
        await s.commit()
        return {"ticker": ticker, "side": "SELL", "shares": pos.shares,
                "price": price, "proceeds": round(proceeds, 2), "reason": reason}


# ---------------------------------------------------------------------------
# The deployment decision
# ---------------------------------------------------------------------------

def _parse_list(col: str) -> list[str]:
    return [t for t in (col or "").split(",") if t]


async def _load_stored_ranking() -> tuple[datetime, list[str], list[str]] | None:
    """The stored daily ranking row: ``(ranking_date, band, picks)``, or None
    when absent.

    No age gate: under the month-end band policy the stored ranking stays
    authoritative for a whole month — the deployment's ME-match gate decides
    when it is behind the resolved month-end and recomputes."""
    from .db import DailyCoreRanking
    async with Session() as s:
        row = await s.get(DailyCoreRanking, 1)
    if row is None:
        return None
    return row.ranking_date, _parse_list(row.band), _parse_list(row.picks)


async def _store_ranking(band: list[str], picks: list[str], asof: datetime) -> None:
    """Persist the decision: ``ranking_date`` = the decision's month-end (the
    frame's as-of date), so consumers can tell WHICH month-end ranked."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from .db import DailyCoreRanking
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_localize(None)
    stmt = sqlite_insert(DailyCoreRanking).values(
        id=1, ranking_date=asof_ts.to_pydatetime(),
        band=",".join(band), picks=",".join(picks))
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={"ranking_date": stmt.excluded.ranking_date,
              "band": stmt.excluded.band,
              "picks": stmt.excluded.picks,
              "updated_at": stmt.excluded.updated_at})
    async with Session() as s:
        await s.execute(stmt)
        await s.commit()


async def compute_targets(force: bool = False, now: datetime | None = None
                          ) -> tuple[list[str], list[str], pd.DataFrame | None]:
    """Compute (or read the stored) daily-core target portfolio.

    Returns (band, picks, frame): ``band`` is the full hold-band order
    (top ``sim_monthly_hold_band`` ranked tickers — held names stay while
    inside it), ``picks`` the top ``sim_monthly_target_n`` with hysteresis
    applied. ``frame`` is the eligibility frame (None when no data or when
    the stored ranking was used).

    Month-end band policy: the DECISION is the completed month-end's frame
    (the last month-end <= S with its candle stored) — the same ranking
    every session of the month, recomputed only when a NEWER month-end
    completes. ``force=True`` recomputes (and re-stores) at the resolved
    month-end; ``force=False`` reads the stored row when it already matches
    the resolved month-end — no age gate (a whole month is one decision).
    """
    now = now or _now()
    me = await _stored_month_end(now)

    def _matches(stored) -> bool:
        if stored is None or me is None:
            return False
        rd = pd.Timestamp(stored[0])
        if rd.tzinfo is not None:
            rd = rd.tz_localize(None)
        return rd.normalize() == me

    if not force:
        stored = await _load_stored_ranking()
        if _matches(stored):
            return stored[1], stored[2], None
    async with _targets_lock:
        # Double-check inside the lock: two concurrent callers (the UI's
        # duplicate status requests on first load) would otherwise both run
        # the expensive compute. The loser now reads the winner's freshly
        # stored ranking.
        if not force:
            stored = await _load_stored_ranking()
            if _matches(stored):
                return stored[1], stored[2], None
        if me is None:
            # No completed month-end decision behind the cycle: nothing to
            # rank yet (the deployment also skips; no fabricated frame).
            return [], [], None
        frame, order = await _core_frame(me)
        if not order:
            return [], [], frame
        band = order[:settings.sim_monthly_hold_band]
        async with Session() as s:
            held = [p.ticker for p in (await s.scalars(select(DailyCorePosition))).all()]
        # The pick pass always refills (deployment tops up toward the plan);
        # through _deploy_targets so the picks carry the hysteresis-selected
        # membership in RANKING order (deployment funds best rank first;
        # _band_fill's alphabetical sort is for the monthly display path).
        picks = monthly_mod._deploy_targets(
            order, held, settings.sim_monthly_target_n,
            settings.sim_monthly_hold_band, refill=True)
        await _store_ranking(band, picks, me.to_pydatetime())
        return band, picks, frame


async def run_deployment(now: datetime | None = None) -> dict:
    """One daily-core deployment pass under the session fill contract
    (app/session.py), month-end band policy:

      * the DECISION is the completed month-end's qv-mom ranking — the same
        frame every session of the month, recomputed only when a NEWER
        month-end completes. The band therefore switches on the FIRST
        SESSION after that month-end (one session behind the calendar
        month-end, rule 1: data <= S only);
      * releases: held names outside the CURRENT band are evaluated EVERY
        session — structurally empty between month-ends (the band only
        changes at a completed month-end) — and a release leg without an
        executable quote this session retries at the next session;
      * fills execute at the current session's strict exact-date OPEN per
        ticker (rule 2 — never a stale or forward-filled close; no bar, no
        fill);
      * marks value the book at the current session's close (rule 3).

    1. Resolve the month-end ranking + hysteresis picks.
    2. SELL held names that are not in the band (band release).
    3. BUY: deploy all free cash into the hysteresis-selected ``picks``,
       topping each up toward the equal-weight target — best rank first,
       until cash or targets are exhausted. This is the backtest-winning
       "rank deploy, boost=0" rule: fresh cash reinforces the top of the
       ranking, never spreads pro-rata, never waits for month-end.
       Held names the band retained (ranks 11-20) keep their slots; a new
       name is bought only when a slot is actually free, so the book never
       grows past the ten-slot target (audit P0-D).

    Preexisting >target_n state (e.g. holdings carried in from before the
    contract was enforced) is handled deterministically WITHOUT sells: band
    members keep their slots whatever the count, only names that leave the
    band are released, and no new names are bought until a slot frees. The
    excess names stop receiving top-ups (they are dropped from ``picks`` by
    the target_n cap) and exit naturally via the band rule.

    Optional protection overlays (`gate` + `gradient_arm` in the daily-core
    state file, the §12-15 mechanisms), independent, advancing on data <= S
    (rule 1): the deployment gate parks buys while the market index is below
    its SMA; the gradient cash-out sells the whole book on a confirmed
    negative slope of the target basket, armed only in good times (or
    ungated). The engine re-enters on its own rule — no cooldown.
    """
    if now is None:
        now = _now()
    f_ts = pd.Timestamp(now)
    f_date = (f_ts.tz_localize(None) if f_ts.tzinfo is not None else f_ts).normalize()

    me = await _stored_month_end(now)
    if me is None:
        return {"skipped": True, "reason": "no completed month-end ranking yet"}

    async with Session() as s:
        held_rows = (await s.scalars(select(DailyCorePosition))).all()
    held_before = sorted(p.ticker for p in held_rows)

    # Ranking: the stored row when it already matches the resolved month-end
    # (the whole month reuses one decision — no frame load); otherwise
    # compute (cached per month-end) and persist the decision row.
    stored = await _load_stored_ranking()
    rd = pd.Timestamp(stored[0]) if stored else None
    if rd is not None and rd.tzinfo is not None:
        rd = rd.tz_localize(None)
    if stored is not None and rd.normalize() == me:
        band = stored[1]
    else:
        frame, order = await _core_frame(me)
        if not order:
            return {"skipped": True, "reason": "no eligible ranking (fundamentals too thin?)"}
        band = order[:settings.sim_monthly_hold_band]
    band_set = set(band)

    # The hysteresis-selected deployment targets against the CURRENT held
    # set (audit P0-D): held names inside the band keep their slots (ranks
    # 11-20 included); new names fill open slots only when one is actually
    # free. Planned ONCE per session, BEFORE the SELL pass, with the
    # POST-RELEASE membership (``refill=True``: every pending release is
    # assumed to succeed) so a replacement is bought the SAME session its
    # slot frees — planning it with the PRE-release held count instead
    # (refill=len(held_before) < target_n) deferred the replacement to the
    # next session and diverged from the persisted replay, which plans
    # against the post-SELL held set. The executor backstop below (skip
    # new-name buys once the ACTUAL book is at target_n) enforces the
    # ten-slot guarantee when a release fails (no executable quote).
    picks = monthly_mod._deploy_targets(
        band, held_before, settings.sim_monthly_target_n,
        settings.sim_monthly_hold_band, refill=True)
    if stored is None or rd.normalize() != me:
        await _store_ranking(band, picks, me.to_pydatetime())

    trades: list[dict] = []
    # Fill prices: the session's STRICT exact-date OPEN, USD-converted (rule
    # 2) — never a stale or forward-filled close. A ticker without an F bar
    # gets no fill; that leg is retried at the next session.
    fill_map = await monthly_mod._open_usd_map(
        sorted(band_set | set(picks) | set(held_before)), f_date)

    # --- protection overlays: independent deployment gate + gradient
    # (§12-15), advancing on data <= S (rule 1): one load, truncated to S. ---
    gate_sma, gradient_active, arm_sma = current_protection_config()

    block_buys = False
    protection_note = None
    if gate_sma is not None or gradient_active:
        overlay_close = None
        close_full, _vol = await monthly_mod.load_frames(
            universe_tickers(current_universe()), f_date.to_pydatetime(),
            start=_live_history_start())
        if not close_full.empty:
            from .session import prev_session
            s_prev = prev_session(close_full.index, f_date)
            if s_prev is not None:
                overlay_close = close_full.loc[:s_prev]
        if overlay_close is not None and not overlay_close.empty and band:
            gate_ok, basket_out = await _protection_decision(
                band, gate_sma, gradient_active, arm_sma, overlay_close)
        else:
            gate_ok, basket_out = True, False  # no completed data: don't block
        if not gate_ok:
            block_buys = True
            protection_note = "market below SMA — contributions parked"
        # The cash-out is independent of the gate (matching the backtest: the
        # gate only parks buys). Arming is what decides whether it can fire.
        if gradient_active and basket_out:
            # Confirmed negative basket slope (armed): CASH OUT the book at
            # the session's strict opens and park contributions until the
            # signal clears (then the normal deployment re-enters — no
            # cooldown; the signal IS the gate). A leg without an
            # executable quote this session is retried next cycle (the
            # signal persists in the state file).
            for t in held_before:
                price = fill_map.get(t)
                if not price:
                    continue
                r = await _exec_sell(t, price, "daily-core: gradient cash-out")
                if r:
                    trades.append(r)
            held_before = []
            block_buys = True
            protection_note = "gradient cash-out — book moved to cash"

    # SELLs: held names outside the band release their slot — evaluated
    # every session (structurally empty between month-ends, since the band
    # only changes at a completed month-end); a leg without an executable
    # quote this session retries at the next session (no fallback price: a
    # missing quote is not an executable price, rule 2).
    for t in held_before:
        if t not in band_set:
            price = fill_map.get(t)
            if not price:
                continue
            r = await _exec_sell(t, price, "daily-core: out of hold band")
            if r:
                trades.append(r)

    # BUYs: rank-first top-up toward equal weight across the HYSTERESIS
    # picks — NOT the raw band[:target_n]. ``picks`` keeps held names that
    # the band retained (including held ranks 11-20 of the ranking) and adds
    # a new name only when a slot is actually free; buying the raw top-10 of
    # the band instead kept adding names the picks list never chose, growing
    # the book past the ten-slot target (audit P0-D: ten retained positions
    # plus one monthly cash cycle produced an 11th holding). With
    # weight = equity/target_n, exhausting the picks exactly consumes the
    # cash (the backtest-winning "rank deploy, boost=0" rule — fresh cash
    # reinforces the top of the ranking, never spreads pro-rata, never waits
    # for month-end).
    valuation = await valuate(now=now)
    if block_buys:
        return {"deployed": True, "band": band, "picks": picks,
                "held_before": held_before, "trades": trades,
                "gate": current_gate(), "gradient": current_gradient(),
                "blocked": protection_note,
                "valuation": valuation}
    weight = valuation["total_equity"] / max(settings.sim_monthly_target_n, 1)
    # Target-vol control: cap the cash deployed this pass when the portfolio's
    # own 21d realized vol is above target (excess parks in cash; never sells).
    target_vol = current_target_vol()
    room = float("inf")
    if target_vol:
        room = _vol_deploy_room(valuation["total_equity"], valuation["cash"],
                                await _recent_port_returns(_VOL_WINDOW), target_vol)
    held_count = len(valuation["positions"])  # ACTUAL post-SELL count
    for t in picks:
        if room < 1:
            break
        price = fill_map.get(t)
        if not price:
            continue
        held_pos = next((p for p in valuation["positions"] if p["ticker"] == t), None)
        current_value = held_pos["value"] if held_pos else 0.0
        if held_pos is None and held_count >= settings.sim_monthly_target_n:
            # No free slot. The picks planned this replacement BEFORE the
            # SELL pass, assuming every release succeeds; a failed release
            # (no executable quote) leaves the old slot occupied, and
            # filling the replacement anyway created an 11th holding.
            # Top-ups of held picks are unaffected.
            continue
        budget = min(weight - current_value, valuation["cash"], room)
        if budget < 1:
            continue
        r = await _exec_buy(t, price, budget,
                            f"daily-core deploy: rank target {settings.sim_monthly_target_n}")
        if r:
            trades.append(r)
            room -= r["cost"]
            if held_pos is None:
                held_count += 1
            # refresh cash so successive buys see the balance
            valuation = await valuate(now=now)

    return {"deployed": True, "band": band, "picks": picks,
            "held_before": held_before, "trades": trades,
            "gate": current_gate(), "gradient": current_gradient(),
            "target_vol": target_vol,
            "valuation": await valuate(now=now)}


# ---------------------------------------------------------------------------
# Scheduler entry
# ---------------------------------------------------------------------------

async def take_snapshot() -> dict:
    """Record one equity-curve snapshot of the current valuation."""
    post = await valuate()
    async with Session() as s:
        s.add(DailyCoreSnapshot(cash=post["cash"], positions_value=post["positions_value"],
                                total_equity=post["total_equity"],
                                allowance_total=post["allowance_total"]))
        await s.commit()
    logger.info("Daily-core snapshot: equity %.2f (cash %.2f, positions %.2f)",
                post["total_equity"], post["cash"], post["positions_value"])
    return {"snapshotted": True, "total_equity": post["total_equity"]}


async def refresh_data() -> tuple[list[str], list[str]]:
    """Refresh candles (+FX) for the universe and the current holdings.

    Uses the bounded-concurrency ``refresh_many`` batch instead of a
    sequential per-ticker loop. On weekends the market is closed, so the
    full ~176-ticker universe (the heaviest nightly path, and its candles
    cannot change) is skipped and only the holdings are refreshed.
    """
    from .market import refresh_many
    async with Session() as s:
        held = [p.ticker for p in (await s.scalars(select(DailyCorePosition))).all()]
    if _local_now().weekday() >= 5:  # Sat/Sun
        tickers = held
    else:
        tickers = list(dict.fromkeys(universe_tickers(current_universe()) + held))
    # Reuse the nightly universe prefetch: tickers fetched within the freshness
    # window are skipped, so daily-core reads the DB instead of re-pulling.
    return await refresh_many(tickers, "2y",
                              max_age_seconds=settings.market_fresh_seconds)


# ---------------------------------------------------------------------------
# Backfill (synthetic history)
# ---------------------------------------------------------------------------

async def _sync_start_date() -> str | None:
    """Earliest live snapshot date across the other two paper portfolios
    (daily sim + monthly). Used as the daily-core backfill's default start
    so all three equity curves cover the SAME window — the user wants them
    synched, and a 2017 synthetic history next to two 2-week live curves is
    apples-to-oranges on the charts. Returns None when the other tables are
    empty (then the replay falls back to the first eligible month).
    """
    from .db import MonthlySnapshot, Session as DbSession, SimSnapshot

    earliest: datetime | None = None
    async with DbSession() as s:
        d = await s.scalar(select(func.min(SimSnapshot.created_at)))
        if d is not None and (earliest is None or d < earliest):
            earliest = d
    async with DbSession() as s:
        d = await s.scalar(select(func.min(MonthlySnapshot.created_at)))
        if d is not None and (earliest is None or d < earliest):
            earliest = d
    return earliest.strftime("%Y-%m-%d") if earliest else None


async def backfill(start: str | None = None,
                   *, preload: monthly_mod.BackfillPreload | None = None) -> dict:
    """Replay the winning daily-core strategy over historical data and
    REPLACE the portfolio state with the replay's end state.

    Wipes account/positions/trades/allowances/snapshots, then walks every
    stored session from `start` to today as the live engine's cycle stream
    (the session fill contract, app/session.py): deposits the monthly
    allowance on each calendar month's first stored session (unconditional);
    ranks against the newest ELIGIBLE completed month-end <= S (the previous
    stored session — the month-end band policy, so the band switches on the
    first session after a month-end); releases held names outside that band
    (structurally empty between month-ends); fills SELLs and BUYs at the
    CURRENT session's strict exact-date OPEN per ticker (never a stale or
    forward-filled close — a missing quote gives that leg no fill and the
    daily decision re-takes it next session); values the book at that
    session's close. No fees: the replay mirrors the LIVE engine, whose
    `_exec_buy`/`_exec_sell` charge no commission (the paper portfolios are
    fee-free by design — see docs/brokers.md), so a backfill's end state
    matches what the live engine would hold.

    ``start`` semantics:
      - explicit "YYYY-MM-DD": replay from that day (the UI's optional date)
      - None (default): synched with the other sims — the replay starts on
        the earliest snapshot date of the daily sim / monthly portfolios so
        all three equity curves cover the same window
      - "all": the full stored history (2017+, first month with an
        eligible frame) — the long-view replay

    This is a paper-portfolio convenience, not a live track record: the
    trades never happened in real time. The final state converges to
    "what the strategy would hold today", and live cycles continue from
    there. Point-in-time discipline: each month-end ranking only sees
    fundamentals that were public by then (same EDGAR-first fact store as
    the backtest) and candles <= that month-end; the overlays advance on
    candles <= S.
    """
    async with _cycle_lock:
        from . import optimize  # noqa: F401  (ensures strategy modules load)
        from .db import DailyCoreAccount as Acc, DailyCorePosition as Pos, \
            DailyCoreTrade as Tr, DailyCoreAllowance as Al, DailyCoreSnapshot as Sn

        if start is None:
            start = await _sync_start_date()
        elif start == "all":
            start = None  # full stored history
        start_note = start or "first eligible month"

        # `preload` (built once by backfill-all) carries the shared
        # fundamentals + frames; otherwise load them here. Bound the candle
        # load to the replay window + a momentum warmup (`start=None` means
        # full stored history — start=all or before the data begins).
        shared_frames: dict = {}
        if preload is not None:
            tickers = preload.tickers
            fund = preload.fund
            open_px, close, vol = preload.open, preload.close, preload.vol
            shared_frames = preload.frames
        else:
            tickers = universe_tickers(current_universe())
            fund = await fundamentals_mod.load_fundamentals(tickers)
            load_start = ((pd.Timestamp(start) - timedelta(days=_BACKFILL_WARMUP_DAYS)).to_pydatetime()
                          if start else None)
            open_px, close, vol = await monthly_mod.load_ohlcv_frames(
                tickers, None, start=load_start)
        if not fund:
            return {"ok": False, "error": "no fundamentals loaded"}
        if close.empty:
            return {"ok": False, "error": "no candle data"}

        full_idx = pd.DatetimeIndex(close.index).normalize()
        idx = full_idx
        if start:
            idx = idx[idx >= pd.Timestamp(start)]
        if idx.empty:
            return {"ok": False, "error": f"no trading days since {start_note}"}

        # COMPLETED month-ends only, by the US session calendar (the shared
        # monthly.completed_month_ends rule, same as the research backtests
        # and monthly.backfill): the scheduler's last trading day of each
        # calendar month, kept when its candle is stored and it falls inside
        # the replay window. The old "last stored day of each month in the
        # windowed index" rule rebalanced on the replay's last candle when
        # the window ends inside a running month — a partial-month end
        # treated as a completed month-end.
        months = monthly_mod.completed_month_ends(close.index, idx[0], idx[-1])

        # Eligibility frames per month (same caching as the backtest), plus
        # the month-end BEFORE the window: a start like 2026-06-01 should
        # replay June 1-29 against May's month-end ranking (facts public by
        # then), not sit idle until June's month-end. Point-in-time safe:
        # the prior frame only uses facts public by its own date. Keyed off
        # the WINDOW START's month — independent of the rebalance list (a
        # window inside the current partial month still ranks against last
        # month's frame).
        # The frames honor the runtime-selected momentum variant (the UI
        # dropdown) — the replay always models the strategy as configured.
        settings.sim_daily_core_mom_variant = current_variant()
        prior_month_ends: list[pd.Timestamp] = []
        d0 = idx[0]
        prev_y, prev_m = (d0.year - 1, 12) if d0.month == 1 else (d0.year, d0.month - 1)
        pme_sched = pd.Timestamp(monthly_mod.month_last_trading_day(datetime(prev_y, prev_m, 1)))
        prev_idx = full_idx[full_idx <= pme_sched]
        if len(prev_idx):
            prior_month_ends.append(prev_idx[-1])
        frame_cache: dict[pd.Timestamp, pd.DataFrame | None] = {}
        for m in prior_month_ends + months:
            if m in shared_frames:
                frame_cache[m] = shared_frames[m]
                continue
            frame_cache[m] = await asyncio.to_thread(
                monthly_mod.eligible_frame, m, close, vol, fund)
            shared_frames[m] = frame_cache[m]
        def _has_eligible(m: pd.Timestamp) -> bool:
            f = frame_cache[m]
            return f is not None and bool(f["eligible"].any())
        # Dead months are dropped from the REBALANCE list only; the prior
        # month-end frame still ranks, so a window without any completed
        # month-end of its own (e.g. a start inside the current partial
        # month) still rebuilds the book against it — no rebuild days, no
        # releases, daily deployment only.
        if months:
            months = [m for m in months if _has_eligible(m)]
        ranking_months = sorted(set(prior_month_ends) | set(months))
        if not any(_has_eligible(m) for m in ranking_months):
            return {"ok": False, "error": "no month with eligible names"}
        # No idx clamp to months[0]: days before the first eligible
        # month-end in the window replay against the PRIOR month-end's
        # frame (built above), so start=YYYY-06-01 really starts June 1.
        first_day = idx[0]

        # --- wipe + reset ---
        # Preserve the CURRENT month's live allowance (the real deposit the
        # engine already made): the replay re-creates allowance rows for its
        # own month sequence, which may not include the current month, and
        # the status endpoint's deposit_allowance() would then hit the
        # UNIQUE(month) constraint.
        from sqlalchemy import delete as sa_delete
        live_allowance_month: str | None = None
        async with Session() as s:
            acc0 = await s.get(Acc, 1)
            if acc0 is not None:
                live_allowance_month = acc0.last_allowance_month
            for tbl in (Tr, Pos, Al, Sn):
                await s.execute(sa_delete(tbl))
            acc = await _account(s)
            acc.cash = 0.0
            acc.last_allowance_month = None
            await s.commit()

        target_n = settings.sim_monthly_target_n
        hold_band = settings.sim_monthly_hold_band

        # Valuation prices: forward-filled closes so a ticker without a
        # candle on a given day (holiday, partial today, stale feed) is
        # valued at its LAST known close instead of 0. Without this the
        # replay's equity collapses on days when held names lack candles —
        # the snapshot then records near-zero equity and the UI curve
        # cliffs. MARKS ONLY: fills read the STRICT open frame below.
        close_val = close.ffill()

        # --- in-memory replay state ---
        shares: dict[str, float] = {}
        cost_basis: dict[str, float] = {}   # true average cost, as _exec_buy books it
        opened_at: dict[str, str] = {}      # ticker -> actual first-buy date
        cash = 0.0
        contributed = 0.0
        trades: list[tuple[str, str, str, float, float, float]] = []
        snaps: list[tuple[str, float, float]] = []

        def px_of(t: str, d: pd.Timestamp) -> float | None:
            return monthly_mod.px_at(close_val, d, t)

        def fill_px_of(t: str, d: pd.Timestamp) -> float | None:
            # Session fill contract rule 2 (app/session.py): the strict
            # exact-date OPEN per ticker — a stored bar on `d` or NOTHING
            # (a missing quote is not an executable price; the leg retries
            # at the next session, never a stale or forward-filled close).
            return monthly_mod.px_at(open_px, d, t)

        # Protection replay state (same §12-15 mechanisms as the live engine
        # and the optimize backtest): independent deployment gate + gradient.
        gate_sma, gradient_active, arm_sma = current_protection_config()
        mkt_index = close_val.mean(axis=1)
        mkt_gate_map = (mkt_index.rolling(max(gate_sma, 2)).mean()
                        if gate_sma else None)
        mkt_arm_map = (mkt_index.rolling(max(arm_sma, 2)).mean()
                       if (gradient_active and arm_sma) else None)
        grad_hist: list[float] = []
        grad_folded: pd.Timestamp | None = None
        grad_neg = 0
        grad_pos = 0
        grad_out = False
        # Target-vol control replay state: the portfolio's own
        # contribution-adjusted daily returns, exactly as the live engine
        # derives them from the snapshot history.
        target_vol = current_target_vol()
        port_rets: list[float] = []
        prev_equity_close: float | None = None

        def _above_map(sma_map: pd.Series | None, d: pd.Timestamp) -> bool:
            if sma_map is None:
                return True
            sm = sma_map.get(d)
            cu = mkt_index.get(d)
            if sm is None or cu is None or not np.isfinite(sm) or not np.isfinite(cu):
                return True
            return bool(float(cu) >= float(sm))

        def _market_ok_r(d: pd.Timestamp) -> bool:
            return _above_map(mkt_gate_map, d)

        def _market_armed_r(d: pd.Timestamp) -> bool:
            return _above_map(mkt_arm_map, d)

        cur_month: int | None = None
        deposited_months: set[str] = set()  # months the replay actually funded
        locs = full_idx.get_indexer(idx)  # S = the stored session before each walk day
        for i, d in enumerate(idx):
            contrib_today = 0.0
            li = locs[i]
            s_prev = full_idx[li - 1] if li > 0 else None

            # allowance: deposit on the first stored session of a new month
            # (unconditional — the live engine funds the account monthly even
            # when no ranking exists; only the deployment needs a frame).
            # Tracked so the persisted allowance rows match the deposits
            # exactly — the old code derived them from ALL window months
            # (1980+ for start=all), creating ~440 phantom $1k rows that
            # inflated allowance_total to $550k and broke every
            # contributed-normalized UI metric.
            if cur_month is None or d.month != cur_month:
                cur_month = d.month
                cash += settings.sim_monthly_contribution
                contributed += settings.sim_monthly_contribution
                contrib_today = settings.sim_monthly_contribution
                deposited_months.add(d.strftime("%Y-%m"))

            # Point-in-time discipline (month-end band policy): the band is
            # the newest ELIGIBLE month-end <= S (the previous stored
            # session) — the same frame every session of the month, so the
            # band switches on the FIRST SESSION after a month-end. A dead
            # month-end leaves the previous band ranking (the live engine
            # keeps its stored decision row until a live frame returns).
            prior = [m for m in ranking_months
                     if s_prev is not None and m <= s_prev]
            month_td = prior[-1] if prior else None
            frame = frame_cache.get(month_td) if month_td is not None else None
            order: list[str] = []
            if frame is not None and bool(frame["eligible"].any()):
                _elig, order = await asyncio.to_thread(monthly_mod._qv_order, frame)
            band = set(order[:hold_band])

            def do_buy(t: str, budget: float, day=d) -> float:
                """One fee-free leg at the session's strict exact-date OPEN
                (rule 2): 4dp-floored ``share_size`` on the caller's budget,
                no fill when the cash cannot cover it (mirrors the live
                _exec_buy). Returns the booked cost (the cents the live
                executor reports), 0.0 when the leg didn't fill."""
                nonlocal cash
                if t not in shares and len(shares) >= target_n:
                    # Hard slot backstop: `_deploy_targets` plans the
                    # post-release membership (refill=True fills toward
                    # target_n assuming pending releases succeed), so the
                    # executor must check the ACTUAL free capacity before
                    # opening a NEW name — a failed release (no usable
                    # price) leaves its slot occupied.
                    return 0.0
                p = fill_px_of(t, day)
                if p is None or p <= 0:
                    return 0.0  # rule 2: no bar on the fill session → no fill
                sh = monthly_mod.share_size(budget, p)
                if sh < 0.0001:
                    return 0.0
                cost = sh * p
                if cash < cost:
                    return 0.0  # live _exec_buy: insufficient cash → no fill
                if t in shares:
                    new_sh = shares[t] + sh
                    cost_basis[t] = (shares[t] * cost_basis[t] + cost) / new_sh
                    shares[t] = new_sh
                else:
                    shares[t] = sh
                    cost_basis[t] = p
                    opened_at[t] = day.strftime("%Y-%m-%d")
                cash -= cost
                trades.append((day.strftime("%Y-%m-%d"), "BUY", t, cost, sh, p))
                return round(cost, 2)

            def do_sell(t: str, day=d) -> float:
                """One fee-free release leg at the session's strict
                exact-date OPEN (rule 2) — a missing quote gives the leg no
                fill (it retries at the next session; never a stale close).
                Returns the booked proceeds, 0.0 when the leg didn't fill."""
                nonlocal cash
                p = fill_px_of(t, day)
                sh = shares.get(t, 0.0)
                if p is None or p <= 0 or sh <= 0:
                    return 0.0
                notional = sh * p
                cash += notional
                del shares[t]
                cost_basis.pop(t, None)
                opened_at.pop(t, None)
                trades.append((day.strftime("%Y-%m-%d"), "SELL", t, notional, sh, p))
                return round(notional, 2)

            # --- protection overlays: independent gate + gradient arm ---
            # The deployment gate (park buys) and the gradient arm (allow the
            # cash-out only in good times) are separate, matching the
            # backtest. Both advance on data <= S (rule 1): the gate reads
            # the market's SMA at S.
            protect_ok = _market_ok_r(s_prev) if gate_sma else True
            # The chain is built every day regardless of the arm; only the
            # TRIGGER is gated, and re-entry stays unconditional (as §15).
            # Sessions <= S fold into the chained basket; F's close folds in
            # only when F becomes S at the next cycle.
            if gradient_active:
                armed = True if s_prev is None else _market_armed_r(s_prev)
                hi = full_idx.searchsorted(s_prev, side="right") \
                    if s_prev is not None else 0
                lo = 0 if grad_folded is None else \
                    int(full_idx.searchsorted(grad_folded, side="right"))
                for j in range(max(lo, 1), hi):
                    d_prev, d_cur = full_idx[j - 1], full_idx[j]
                    rets = []
                    for t in order[:target_n]:
                        p0, p1 = px_of(t, d_prev), px_of(t, d_cur)
                        if p0 and p1 and p0 > 0 and p1 > 0:
                            rets.append(float(np.log(p1 / p0)))
                    if rets:  # live twin: a day with no priced pair doesn't fold
                        if not grad_hist:
                            grad_hist.append(100.0)
                        grad_hist.append(grad_hist[-1]
                                         * float(np.exp(float(np.mean(rets)))))
                if s_prev is not None:
                    grad_folded = s_prev
                grad_hist = grad_hist[-260:]  # bounded history
                if len(grad_hist) > _GRADIENT_DAYS:
                    slope = grad_hist[-1] / grad_hist[-1 - _GRADIENT_DAYS] - 1.0
                    if slope < 0:
                        grad_neg, grad_pos = grad_neg + 1, 0
                    elif slope > 0:
                        grad_pos, grad_neg = grad_pos + 1, 0
                    if not grad_out and armed and grad_neg >= _GRADIENT_CONFIRM:
                        grad_out = True
                    elif grad_out and grad_pos >= _GRADIENT_CONFIRM:
                        grad_out = False

            # SELL band releases (every session — the band only changes at a
            # completed month-end, so between month-ends this is
            # structurally empty; a leg without a strict-open quote this
            # session retries at the next session)
            for t in list(shares):
                if t not in band:
                    do_sell(t)

            # gradient cash-out (confirmed negative slope in good times)
            if gradient_active and grad_out and shares:
                for t in list(shares):
                    do_sell(t)

            # The hysteresis-selected deployment targets (audit P0-D): the
            # old loop deployed over the raw order[:target_n] — buying the
            # ranking's top-10 even when ten held names already occupied the
            # slots, growing the book past the ten-slot target. Held names
            # inside the band keep their slots; new names fill open slots
            # only when one is actually free — a band-switch session (the
            # releases above freed slots) or an account below capacity.
            # Between month-ends, under the month-end band policy, no new
            # name is bought while an out-of-band release is still pending.
            # A FAILED release (no price) leaves `shares` unchanged while
            # `refill` is True, so the CAPACITY TEST alone does not cap the
            # plan — the executor backstop below (do_buy skips new names
            # once len(shares) >= target_n) enforces the ten-slot guarantee.
            if order:
                deploy_picks = monthly_mod._deploy_targets(
                    order, list(shares), target_n, hold_band,
                    refill=len(shares) < target_n)
            else:
                deploy_picks = []

            # Marks at F's close (rule 3), mirrored from the live valuate():
            # positions summed ticker-ordered, total equity rounded once.
            positions_value = sum((shares.get(t, 0.0) or 0.0) * (px_of(t, d) or 0.0)
                                  for t in sorted(shares))
            total_equity = round(cash + positions_value, 2)
            weight = total_equity / max(target_n, 1)

            # rank-deployment (boost=0): top up the hysteresis picks toward
            # the equal-weight target, best rank first, at the session's
            # strict opens. Blocked while the market gate says "bad times"
            # (contributions park) — the gradient cash-out already emptied
            # the book. Mirrors the live loop: the budget clamps to the
            # ROUNDED cash (valuate's output) and the room decrements by
            # each booked cost.
            if deploy_picks and protect_ok and not grad_out:
                room = (_vol_deploy_room(total_equity, round(cash, 2),
                                         port_rets, target_vol)
                        if target_vol else float("inf"))
                for t in deploy_picks:
                    if room < 1:
                        break
                    p = fill_px_of(t, d)
                    if p is None or p <= 0:
                        continue
                    cur = round(shares.get(t, 0.0) * (px_of(t, d) or 0.0), 2)
                    gap = weight - cur
                    budget = min(gap, round(cash, 2), room)
                    if budget < 1:
                        continue
                    booked = do_buy(t, budget)
                    room -= booked

            # daily snapshot (one point per day). The snapshot carries the
            # contributed total AS OF that day — persisting the final total
            # for every row would make the % curve halve on early points
            # (999/2000 on day 1 instead of 999/1000).
            equity_close = total_equity
            if prev_equity_close is not None:
                port_rets.append(_port_return(prev_equity_close, equity_close, contrib_today))
            prev_equity_close = equity_close
            snaps.append((d.strftime("%Y-%m-%d"), equity_close, contributed))

        # --- persist the end state ---
        # Allowance rows: one per month the replay ACTUALLY deposited (the
        # in-loop tracked set). The old code used every month in the window
        # — for start=all that included ~440 months before the first
        # eligible ranking (1980+), creating phantom $1k rows that inflated
        # allowance_total and broke every contributed-normalized metric.
        replay_months = sorted(deposited_months)
        # The live engine deposits at the START of the month, so it may
        # already have deposited the CURRENT month while the replay window
        # has no candle for it yet (e.g. backfill on the 1st before the
        # market opens). The replay then never re-deposits it, yet the
        # account marker below suppresses the live deposit forever — the
        # account ends permanently one contribution short. Add it back.
        current_month = _current_month()
        if live_allowance_month == current_month and current_month not in replay_months:
            cash += settings.sim_monthly_contribution
            contributed += settings.sim_monthly_contribution
            replay_months = sorted(set(replay_months) | {current_month})
        async with Session() as s:
            acc = await _account(s)
            acc.cash = cash
            for t, sh in shares.items():
                # converge to the live engine's bookkeeping: true average
                # cost and the actual first-buy date (the old code booked
                # the last close into avg_cost and left opened_at defaulting
                # to the write time).
                opened = opened_at.get(t) or idx[-1].strftime("%Y-%m-%d")
                s.add(Pos(ticker=t, shares=sh, avg_cost=cost_basis.get(t, 0.0),
                          opened_at=pd.Timestamp(f"{opened} 16:00:00+00:00").to_pydatetime()))
            for td, side, t, notional, sh, pr in trades:
                # created_at = the replay day (same reason as the snapshots
                # below): the trade log must show when the synthetic trade
                # happened, and shares/price must be real, not 0.
                s.add(Tr(ticker=t, side=side, shares=round(sh, 6), price=round(pr, 6),
                         cash_after=0.0,
                         reason=f"backfill {side.lower()} ${notional:,.0f}",
                         created_at=pd.Timestamp(f"{td} 16:00:00+00:00").to_pydatetime()))
            for m in replay_months:
                s.add(Al(amount=settings.sim_monthly_contribution, month=m))
            # Keep the account marker consistent so deposit_allowance() stays
            # a no-op for the live month. Never let it lag the newest inserted
            # row, or the next live deposit would duplicate that month.
            last_month = max(live_allowance_month or "",
                             replay_months[-1] if replay_months else "")
            acc.last_allowance_month = last_month or None
            for sd, eq, contrib_at_day in snaps:
                # created_at = the replay day: the UI groups snapshots by
                # this column's date, so synthetic history must carry the
                # historical day, not the write time (otherwise all points
                # collapse into "today" and the curve shows one day).
                s.add(Sn(cash=0.0, positions_value=eq, total_equity=eq,
                         allowance_total=round(contrib_at_day, 2),
                         created_at=pd.Timestamp(f"{sd} 16:00:00+00:00").to_pydatetime()))
            await s.commit()

        return {"ok": True, "days": len(idx), "start": str(first_day.date()),
                "end": str(idx[-1].date()), "requested_start": start_note,
                "contributed": round(contributed, 2),
                "final_equity": round(snaps[-1][1], 2), "trades": len(trades),
                "snapshots": len(snaps), "mom_variant": current_variant()}


async def run_daily_cycle(force: bool = False, now: datetime | None = None) -> dict:
    """Scheduler entry: one full daily-core cycle.

    deposit allowance (new month) -> refresh data -> deployment pass ->
    daily equity snapshot. Idempotent per day: the allowance gate is
    monthly and the deployment pass is idempotent by construction
    (top-ups stop when targets are met), so a re-run is harmless.
    ``now`` is injectable for the parity tests; None = now.
    """
    if _cycle_lock.locked() and not force:
        return {"skipped": True, "reason": "already running"}
    async with _cycle_lock:
        if not settings.sim_daily_core_enabled:
            return {"skipped": True, "reason": "daily-core portfolio disabled"}

        allowance = await deposit_allowance()

        refreshed, refresh_errors = await refresh_data()

        deployment = await run_deployment(now=now)

        snap = await take_snapshot()

        return {"allowance": allowance, "refresh_errors": refresh_errors,
                "deployment": deployment, "snapshot": snap}


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------

async def get_trades(limit: int = 100) -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(DailyCoreTrade)
                                .order_by(DailyCoreTrade.created_at.desc()).limit(limit))).all()
    return [{"ticker": r.ticker, "side": r.side, "shares": r.shares, "price": r.price,
             "cash_after": round(r.cash_after, 2), "reason": r.reason,
             "date": r.created_at.isoformat()} for r in rows]


async def get_ranking_date() -> str | None:
    """The stored ranking's date (YYYY-MM-DD), None before the first cycle —
    lets the UI flag a stale band instead of presenting it as today's."""
    from .db import DailyCoreRanking
    async with Session() as s:
        row = await s.get(DailyCoreRanking, 1)
    return row.ranking_date.strftime("%Y-%m-%d") if row else None


async def get_equity_curve(limit: int = 365) -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(DailyCoreSnapshot)
                                .order_by(DailyCoreSnapshot.created_at.desc()).limit(limit))).all()
    return [{"date": r.created_at.strftime("%Y-%m-%d"), "cash": r.cash,
             "positions_value": r.positions_value, "total_equity": r.total_equity,
             "allowance_total": r.allowance_total} for r in reversed(rows)]


async def get_allowances() -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(DailyCoreAllowance)
                                .order_by(DailyCoreAllowance.month.desc()))).all()
    return [{"amount": r.amount, "month": r.month} for r in rows]


async def reset_daily_core() -> dict:
    """Wipe the daily-core portfolio and re-initialize it at start cash.

    Clears trades, positions, allowances and snapshots, and resets the account
    markers so the next deposit/deployment starts clean. The persisted gradient
    basket chain is dropped too (it belongs to the old holdings). Market data,
    fundamentals, the screener, the stored ranking and the universe/variant
    selections are left intact — only portfolio *state* is cleared."""
    from sqlalchemy import delete as sa_delete
    from .db import (DailyCoreAccount as Acc, DailyCorePosition as Pos,
                     DailyCoreTrade as Tr, DailyCoreAllowance as Al,
                     DailyCoreSnapshot as Sn)

    async with _cycle_lock:
        async with Session() as s:
            for tbl in (Tr, Pos, Al, Sn):
                await s.execute(sa_delete(tbl))
            acc = await s.get(Acc, 1)
            if acc is None:
                acc = Acc(id=1, cash=settings.sim_monthly_start_cash,
                          last_allowance_month=None)
                s.add(acc)
            else:
                acc.cash = settings.sim_monthly_start_cash
                acc.last_allowance_month = None
            await s.commit()
        with _state_lock:
            state = _load_state()
            _reset_basket_state(state)
            _save_state(state)
    logger.info("Daily-core reset: cash=%.2f", settings.sim_monthly_start_cash)
    return {"ok": True, "cash": settings.sim_monthly_start_cash}
