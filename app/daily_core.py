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
    Releases are evaluated at the month-end rebuild (the backtest showed
    exit cadence is irrelevant; the band is sticky).

Point-in-time discipline mirrors the monthly sim: a decision on date d only
sees fundamental facts with end < d and filed <= d. No look-ahead.

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


def _local_now() -> datetime:
    return datetime.now(_TZ)


def _current_month() -> str:
    return _local_now().strftime("%Y-%m")


def _live_history_start() -> datetime:
    """Naive-UTC lower bound for the live ranking's candle load (~4y). Candle
    timestamps are stored naive, so the bound must be naive too."""
    return datetime.now(UTC).replace(tzinfo=None) - timedelta(days=_LIVE_HISTORY_DAYS)


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
                               arm_sma: int | None) -> tuple[bool, bool]:
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
    """
    tickers = universe_tickers(current_universe())
    close, _vol = await monthly_mod.load_frames(
        tickers, None, start=_live_history_start())
    if close.empty or not band:
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

async def valuate() -> dict:
    """Value the daily-core portfolio (same shape as monthly.monthly_valuate)."""
    async with Session() as s:
        acc = await _account(s)
        positions = (await s.scalars(
            select(DailyCorePosition).order_by(DailyCorePosition.ticker))).all()
        allowance_total = (await s.scalar(
            select(func.sum(DailyCoreAllowance.amount)))) or 0.0

    px_map = await monthly_mod._price_usd_map([p.ticker for p in positions])
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

def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_list(col: str) -> list[str]:
    return [t for t in (col or "").split(",") if t]


async def _load_stored_ranking(max_age_days: float = 1.0) -> tuple[list[str], list[str]] | None:
    """The stored daily ranking, or None when absent/older than a day.

    The fundamentals math runs once per day in the cycle and persists its
    output to the DailyCoreRanking row; every other consumer reads that.
    """
    from .db import DailyCoreRanking
    async with Session() as s:
        row = await s.get(DailyCoreRanking, 1)
    if row is None:
        return None
    age = _utcnow() - row.ranking_date.replace(tzinfo=UTC)
    if age > timedelta(days=max_age_days):
        return None
    return _parse_list(row.band), _parse_list(row.picks)


async def _store_ranking(band: list[str], picks: list[str]) -> None:
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from .db import DailyCoreRanking
    stmt = sqlite_insert(DailyCoreRanking).values(
        id=1, ranking_date=_utcnow().replace(tzinfo=None),
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


async def compute_targets(force: bool = False) -> tuple[list[str], list[str], pd.DataFrame | None]:
    """Compute (or read the stored) daily-core target portfolio.

    Returns (band, picks, frame): ``band`` is the full hold-band order
    (top ``sim_monthly_hold_band`` ranked tickers — held names stay while
    inside it), ``picks`` the top ``sim_monthly_target_n`` with hysteresis
    applied. ``frame`` is the eligibility frame (None when no data or when
    the stored ranking was used).

    The full-universe fundamentals math is expensive (~7s), so it runs once
    per day during the daily cycle, which persists the result to the
    ``daily_core_ranking`` row. Other consumers (status endpoint / UI) read
    that stored ranking instead of recomputing. ``force=True`` recomputes
    regardless — used by the deployment cycle and the manual refresh path
    so trades always act on today's data.
    """
    if not force:
        stored = await _load_stored_ranking()
        if stored is not None:
            band, picks = stored
            return band, picks, None
    async with _targets_lock:
        # Double-check inside the lock: two concurrent callers (the UI's
        # duplicate status requests on first load) would otherwise both run
        # the expensive compute. The loser now reads the winner's freshly
        # stored ranking.
        if not force:
            stored = await _load_stored_ranking()
            if stored is not None:
                band, picks = stored
                return band, picks, None
        tickers = universe_tickers(current_universe())
        fund = await fundamentals_mod.load_fundamentals(tickers)
        if not fund:
            return [], [], None
        close, vol = await monthly_mod.load_frames(
            tickers, None, start=_live_history_start())
        if close.empty:
            return [], [], None
        iso = datetime.now(UTC).strftime("%Y-%m-%d")
        d: pd.Timestamp = pd.Timestamp(iso)  # type: ignore[assignment]
        # The runtime-selected momentum variant (UI dropdown, persisted)
        # drives the frame math — eligible_frame reads this setting.
        settings.sim_daily_core_mom_variant = current_variant()
        frame = await asyncio.to_thread(monthly_mod.eligible_frame, d, close, vol, fund)
        if frame is None or not bool(frame["eligible"].any()):
            return [], [], frame
        _elig, order = await asyncio.to_thread(monthly_mod._qv_order, frame)
        band = order[:settings.sim_monthly_hold_band]
        async with Session() as s:
            held = [p.ticker for p in (await s.scalars(select(DailyCorePosition))).all()]
        picks = monthly_mod._band_fill(order, held, settings.sim_monthly_target_n,
                                       settings.sim_monthly_hold_band)
        await _store_ranking(band, picks)
        return band, picks, frame


async def run_deployment() -> dict:
    """One daily-core deployment pass.

    1. Compute the qv-mom ranking + hysteresis picks.
    2. SELL held names that fell out of the band (band release, same rule
       as the monthly sim's rebalance).
    3. BUY: deploy all free cash into the top-ranked names, topping each up
       toward the equal-weight target — best rank first, until cash or
       targets are exhausted. This is the backtest-winning "rank deploy,
       boost=0" rule: fresh cash reinforces the top of the ranking, never
       spreads pro-rata, never waits for month-end.

    Optional protection overlays (`gate` + `gradient_arm` in the daily-core
    state file, the §12-15 mechanisms), independent: the deployment gate parks
    buys while the market index is below its SMA; the gradient cash-out sells
    the whole book on a confirmed negative slope of the target basket, armed
    only in good times (or ungated). The engine re-enters on its own rule — no
    cooldown.
    """
    band, picks, frame = await compute_targets(force=True)
    if not band:
        return {"skipped": True, "reason": "no eligible ranking (fundamentals too thin?)"}

    async with Session() as s:
        held_rows = (await s.scalars(select(DailyCorePosition))).all()
    held_before = sorted(p.ticker for p in held_rows)
    trades: list[dict] = []
    px_map = await monthly_mod._price_usd_map(sorted(set(band) | set(held_before)))

    # --- protection overlays: independent deployment gate + gradient (§12-15) ---
    gate_sma, gradient_active, arm_sma = current_protection_config()

    block_buys = False
    protection_note = None
    if (gate_sma is not None or gradient_active) and frame is not None:
        gate_ok, basket_out = await _protection_decision(
            band, gate_sma, gradient_active, arm_sma)
        if not gate_ok:
            block_buys = True
            protection_note = "market below SMA — contributions parked"
        # The cash-out is independent of the gate (matching the backtest: the
        # gate only parks buys). Arming is what decides whether it can fire.
        if gradient_active and basket_out:
            # Confirmed negative basket slope (armed): CASH OUT the book at the
            # latest prices and park contributions until the signal clears (then
            # the normal deployment re-enters — no cooldown; the signal IS the
            # gate).
            for t in held_before:
                price = px_map.get(t)
                if not price:
                    async with Session() as s:
                        pos = await s.scalar(select(DailyCorePosition)
                                             .where(DailyCorePosition.ticker == t))
                    price = pos.avg_cost if pos else 0.0
                r = await _exec_sell(t, price, "daily-core: gradient cash-out")
                if r:
                    trades.append(r)
            held_before = []
            block_buys = True
            protection_note = "gradient cash-out — book moved to cash"

    # SELLs: held names out of the band release their slot.
    for t in held_before:
        if t not in band:
            price = px_map.get(t) or 0.0
            if not price:
                # No candles: fall back to avg_cost like the monthly sim — never sell at 0.
                async with Session() as s:
                    pos = await s.scalar(select(DailyCorePosition).where(DailyCorePosition.ticker == t))
                price = pos.avg_cost if pos else 0.0
            r = await _exec_sell(t, price, "daily-core: out of hold band")
            if r:
                trades.append(r)

    # BUYs: rank-first top-up toward equal weight. The rank loop stops at
    # the top target_n names; with weight = equity/target_n, exhausting
    # those exactly consumes the cash (the backtest-winning "rank deploy,
    # boost=0" rule — fresh cash reinforces the top of the ranking, never
    # spreads pro-rata, never waits for month-end).
    valuation = await valuate()
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
    for t in band[:settings.sim_monthly_target_n]:
        if room < 1:
            break
        price = px_map.get(t)
        if not price:
            continue
        held_pos = next((p for p in valuation["positions"] if p["ticker"] == t), None)
        current_value = held_pos["value"] if held_pos else 0.0
        budget = min(weight - current_value, valuation["cash"], room)
        if budget < 1:
            continue
        r = await _exec_buy(t, price, budget,
                            f"daily-core deploy: rank target {settings.sim_monthly_target_n}")
        if r:
            trades.append(r)
            room -= r["cost"]
            # refresh cash so successive buys see the balance
            valuation = await valuate()

    return {"deployed": True, "band": band, "picks": picks,
            "held_before": held_before, "trades": trades,
            "gate": current_gate(), "gradient": current_gradient(),
            "target_vol": target_vol,
            "valuation": await valuate()}


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
    stored trading day from `start` to today, depositing the monthly
    allowance and applying the rank-deployment rule at each day's close.
    No fees: the replay mirrors the LIVE engine, whose `_exec_buy`/`_exec_sell`
    charge no commission (the paper portfolios are fee-free by design — see
    docs/brokers.md), so a backfill's end state matches what the live engine
    would hold.

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
    there. Point-in-time discipline: each day only sees fundamentals that
    were public by then (same EDGAR-first fact store as the backtest).
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
            close, vol = preload.close, preload.vol
            shared_frames = preload.frames
        else:
            tickers = universe_tickers(current_universe())
            fund = await fundamentals_mod.load_fundamentals(tickers)
            load_start = ((pd.Timestamp(start) - timedelta(days=_BACKFILL_WARMUP_DAYS)).to_pydatetime()
                          if start else None)
            close, vol = await monthly_mod.load_frames(tickers, None, start=load_start)
        if not fund:
            return {"ok": False, "error": "no fundamentals loaded"}
        if close.empty:
            return {"ok": False, "error": "no candle data"}

        idx = pd.DatetimeIndex(close.index)
        if start:
            idx = idx[idx >= pd.Timestamp(start)]
        if idx.empty:
            return {"ok": False, "error": f"no trading days since {start_note}"}

        months: list[pd.Timestamp] = []
        for ym in sorted({(d.year, d.month) for d in idx}):
            sub = idx[(idx.year == ym[0]) & (idx.month == ym[1])]
            if len(sub):
                months.append(sub[-1])

        # Eligibility frames per month (same caching as the backtest), plus
        # the month-end BEFORE the window: a start like 2026-06-01 should
        # replay June 1-29 against May's month-end ranking (facts public by
        # then), not sit idle until June's month-end. Point-in-time safe:
        # the prior frame only uses facts public by its own date.
        # The frames honor the runtime-selected momentum variant (the UI
        # dropdown) — the replay always models the strategy as configured.
        settings.sim_daily_core_mom_variant = current_variant()
        prior_month_ends: list[pd.Timestamp] = []
        if months:
            prev = months[0] - pd.offsets.MonthEnd(1)
            prev_idx = close.index[close.index <= prev]
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
        months = [m for m in months if _has_eligible(m)]
        if not months:
            return {"ok": False, "error": "no month with eligible names"}
        # Ranking sources: window month-ends PLUS the prior month-end (so
        # days between `start` and the first window month-end rank against
        # the last frame before the window instead of sitting idle).
        ranking_months = sorted(set(prior_month_ends) | set(months))
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
        # cliffs. Fills also use the carried price, the same convention as
        # the monthly sim's USD-converted valuation.
        close_val = close.ffill()

        # --- in-memory replay state ---
        shares: dict[str, float] = {}
        cash = 0.0
        contributed = 0.0
        n_contribs = 0
        trades: list[tuple[str, str, str, float, float, float]] = []
        snaps: list[tuple[str, float, float]] = []

        def px_of(t: str, d: pd.Timestamp) -> float | None:
            return monthly_mod.px_at(close_val, d, t)

        # Protection replay state (same §12-15 mechanisms as the live engine
        # and the optimize backtest): independent deployment gate + gradient.
        gate_sma, gradient_active, arm_sma = current_protection_config()
        mkt_index = close_val.mean(axis=1)
        mkt_gate_map = (mkt_index.rolling(max(gate_sma, 2)).mean()
                        if gate_sma else None)
        mkt_arm_map = (mkt_index.rolling(max(arm_sma, 2)).mean()
                       if (gradient_active and arm_sma) else None)
        prev_day_r: pd.Timestamp | None = None
        grad_hist: list[float] = []
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
        for d in idx:
            contrib_today = 0.0
            # Point-in-time discipline: each day uses the ranking of the most
            # recent month-end AT OR BEFORE d (the frame computed from facts
            # public by then). Using a future month-end's frame would leak
            # look-ahead into the replay.
            prior = [m for m in ranking_months if m <= d]
            if not prior:
                continue  # before the first completed month-end — no ranking yet
            month_td = prior[-1]
            frame = frame_cache.get(month_td)
            order: list[str] = []
            if frame is not None and bool(frame["eligible"].any()):
                _elig, order = await asyncio.to_thread(monthly_mod._qv_order, frame)
            band = set(order[:hold_band])

            # allowance: deposit on the first trading day of a new month
            # (same timing as the live deposit_allowance). Tracked so the
            # persisted allowance rows match the deposits exactly — the old
            # code derived them from ALL window months (1980+ for start=all),
            # creating ~440 phantom $1k rows that inflated allowance_total
            # to $550k and broke every contributed-normalized UI metric.
            if cur_month is None or d.month != cur_month:
                cur_month = d.month
                cash += settings.sim_monthly_contribution
                contributed += settings.sim_monthly_contribution
                contrib_today = settings.sim_monthly_contribution
                n_contribs += 1
                deposited_months.add(d.strftime("%Y-%m"))

            def do_buy(t: str, budget: float, day=d) -> None:
                nonlocal cash
                p = px_of(t, day)
                if p is None or p <= 0:
                    return
                # Fee-free (mirrors the live _exec_buy): spend up to the
                # budget from available cash, no commission reserved.
                notional = min(budget, max(cash, 0.0))
                if notional < 1:
                    return
                sh = notional / p
                cash -= notional
                shares[t] = shares.get(t, 0.0) + sh
                trades.append((day.strftime("%Y-%m-%d"), "BUY", t, notional, sh, p))

            def do_sell(t: str, day=d) -> None:
                nonlocal cash
                p = px_of(t, day)
                sh = shares.get(t, 0.0)
                if p is None or p <= 0 or sh <= 0:
                    return
                notional = sh * p
                cash += notional
                del shares[t]
                trades.append((day.strftime("%Y-%m-%d"), "SELL", t, notional, sh, p))

            # --- protection overlays: independent gate + gradient arm ---
            # The deployment gate (park buys) and the gradient arm (allow the
            # cash-out only in good times) are separate, matching the backtest.
            protect_ok = _market_ok_r(d) if gate_sma else True
            # The chain is built every day regardless of the arm; only the
            # TRIGGER is gated, and re-entry stays unconditional (as §15).
            if gradient_active:
                armed = _market_armed_r(d)
                rets = []
                for t in order[:target_n]:
                    p0 = px_of(t, prev_day_r) if prev_day_r is not None else None
                    p1 = px_of(t, d)
                    if p0 and p1 and p0 > 0 and p1 > 0:
                        rets.append(float(np.log(p1 / p0)))
                day_ret = float(np.mean(rets)) if rets else 0.0
                if not grad_hist:
                    grad_hist.append(100.0)
                grad_hist.append(grad_hist[-1] * float(np.exp(day_ret)))
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
                prev_day_r = d

            # SELL band releases (month-end rebuild only, as in the backtest)
            if d in months:
                for t in list(shares):
                    if t not in band:
                        do_sell(t)

            # gradient cash-out (confirmed negative slope in good times)
            if gradient_active and grad_out and shares:
                for t in list(shares):
                    do_sell(t)

            equity = cash + sum((shares.get(t, 0.0) or 0.0) * (px_of(t, d) or 0.0)
                                for t in shares)
            weight = equity / max(target_n, 1)

            # rank-deployment (boost=0): top up top-ranked names toward the
            # equal-weight target, best rank first. On month-end days the
            # full equal-weight rebuild also fills NEW names to weight.
            # Blocked while the market gate says "bad times" (contributions
            # park) — the gradient cash-out already emptied the book.
            if order and protect_ok and not grad_out:
                room = (_vol_deploy_room(equity, cash, port_rets, target_vol)
                        if target_vol else float("inf"))
                for t in order[:target_n]:
                    if room < 1:
                        break
                    p = px_of(t, d)
                    if p is None:
                        continue
                    cur = shares.get(t, 0.0) * p
                    gap = weight - cur
                    budget = min(gap, room)
                    if budget > 1 and cash > 1:
                        cash_before = cash
                        do_buy(t, budget)
                        room -= max(cash_before - cash, 0.0)

            # daily snapshot (one point per day). The snapshot carries the
            # contributed total AS OF that day — persisting the final total
            # for every row would make the % curve halve on early points
            # (999/2000 on day 1 instead of 999/1000).
            equity_close = cash + sum((shares.get(t, 0.0) or 0.0) * (px_of(t, d) or 0.0)
                                      for t in shares)
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
                # avg_cost is not tracked by the replay (irrelevant for the
                # equity curve); book at the last close so valuation works.
                p = px_of(t, idx[-1]) or 0.0
                s.add(Pos(ticker=t, shares=sh, avg_cost=p))
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


async def run_daily_cycle(force: bool = False) -> dict:
    """Scheduler entry: one full daily-core cycle.

    deposit allowance (new month) -> refresh data -> deployment pass ->
    daily equity snapshot. Idempotent per day: the allowance gate is
    monthly and the deployment pass is idempotent by construction
    (top-ups stop when targets are met), so a re-run is harmless.
    """
    if _cycle_lock.locked() and not force:
        return {"skipped": True, "reason": "already running"}
    async with _cycle_lock:
        if not settings.sim_daily_core_enabled:
            return {"skipped": True, "reason": "daily-core portfolio disabled"}

        allowance = await deposit_allowance()

        refreshed, refresh_errors = await refresh_data()

        deployment = await run_deployment()

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
