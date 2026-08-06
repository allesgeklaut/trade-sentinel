"""Autonomous paper-trading simulation engine for Trade Sentinel.

The bot receives an imaginary monthly allowance, decides when to buy/sell
assets from a configurable universe using deterministic signals (with optional
LLM hybrid mode planned), and tracks portfolio performance over time.

All trades are executed at the latest cached daily close price — no real
broker, no intraday, no shorting.  This is a research/education tool.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select

from .analysis import compute
from .config import settings
from .db import (
    SimAccount,
    SimAllowance,
    SimPosition,
    SimSnapshot,
    SimTrade,
    Session,
)
from .market import candles, refresh
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.sim")

# How many candles to refresh for the sim universe (2y is a good balance
# for indicator computation without excessive API load).
_SIM_REFRESH_PERIOD = "2y"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _current_month() -> str:
    return _utcnow().strftime("%Y-%m")


async def _latest_close(ticker: str) -> float | None:
    """Return the most recent cached close price for *ticker*, or None."""
    rows = await candles(ticker)
    if not rows:
        return None
    return float(rows[-1]["close"])


async def _account() -> SimAccount:
    """Return the singleton SimAccount row, creating it if necessary."""
    async with Session() as s:
        acc = await s.get(SimAccount, 1)
        if acc is None:
            acc = SimAccount(id=1, cash=settings.sim_start_cash, last_allowance_month=None)
            s.add(acc)
            await s.commit()
        return acc


# ---------------------------------------------------------------------------
# Allowance
# ---------------------------------------------------------------------------

async def deposit_allowance() -> dict[str, Any]:
    """Deposit the monthly allowance if a new month has begun.

    Returns a dict describing whether a deposit was made and the resulting
    cash balance.
    """
    month = _current_month()
    async with Session() as s:
        acc = await s.get(SimAccount, 1)
        if acc is None:
            acc = SimAccount(id=1, cash=settings.sim_start_cash, last_allowance_month=None)
            s.add(acc)

        if acc.last_allowance_month == month:
            return {"deposited": False, "month": month, "cash": acc.cash}

        acc.cash += settings.sim_monthly_allowance
        acc.last_allowance_month = month
        s.add(SimAllowance(amount=settings.sim_monthly_allowance, month=month))
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
    """
    async with Session() as s:
        acc = await s.get(SimAccount, 1)
        if acc is None:
            acc = SimAccount(id=1, cash=settings.sim_start_cash, last_allowance_month=None)
            s.add(acc)
            await s.commit()

        positions = (await s.scalars(select(SimPosition).order_by(SimPosition.ticker))).all()
        allowance_total = (await s.scalar(select(func.sum(SimAllowance.amount)))) or 0.0

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
        })

    return {
        "cash": round(acc.cash, 2),
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(acc.cash + positions_value, 2),
        "allowance_total": round(float(allowance_total), 2),
    }


# ---------------------------------------------------------------------------
# Deterministic strategy
# ---------------------------------------------------------------------------

async def _candidate_tickers() -> list[str]:
    """Return the list of tickers the sim should consider.

    If ``sim_universe`` is ``watchlist``, use the watchlist table; otherwise
    load from the universe file.
    """
    if settings.sim_universe.lower() == "watchlist":
        from .db import Watchlist
        async with Session() as s:
            rows = (await s.scalars(select(Watchlist).order_by(Watchlist.ticker))).all()
            return [r.ticker for r in rows]
    else:
        return universe_tickers(settings.sim_universe)


async def _exec_buy(ticker: str, price: float, max_budget: float, reason: str) -> dict | None:
    """Execute a paper BUY.  Returns the trade dict or None if skipped."""
    if price <= 0 or max_budget < price:
        return None
    # Buy as many shares as the budget allows (whole shares only)
    shares = int(max_budget // price)
    if shares < 1:
        return None
    cost = shares * price

    async with Session() as s:
        acc = await s.get(SimAccount, 1)
        if acc is None or acc.cash < cost:
            return None
        acc.cash -= cost

        pos = await s.get(SimPosition, ticker)
        if pos:
            # Update weighted average cost
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + cost) / total_shares
            pos.shares = total_shares
        else:
            s.add(SimPosition(ticker=ticker, shares=shares, avg_cost=price))

        trade = SimTrade(
            ticker=ticker, side="BUY", shares=shares, price=price,
            cash_after=acc.cash, reason=reason,
        )
        s.add(trade)
        await s.commit()

        logger.info("Sim BUY %s ×%d @%.2f — %s", ticker, shares, price, reason)
        return {
            "ticker": ticker, "side": "BUY", "shares": shares, "price": price,
            "cash_after": round(acc.cash, 2), "reason": reason,
        }


async def _exec_sell(ticker: str, price: float, shares: float | None, reason: str) -> dict | None:
    """Execute a paper SELL.  *shares*=None means sell entire position.

    Returns the trade dict or None if nothing to sell.
    """
    async with Session() as s:
        pos = await s.get(SimPosition, ticker)
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
            await s.delete(pos)

        trade = SimTrade(
            ticker=ticker, side="SELL", shares=sell_shares, price=price,
            cash_after=acc.cash, reason=reason,
        )
        s.add(trade)
        await s.commit()

        logger.info("Sim SELL %s ×%.4f @%.2f — %s", ticker, sell_shares, price, reason)
        return {
            "ticker": ticker, "side": "SELL", "shares": sell_shares, "price": price,
            "cash_after": round(acc.cash, 2), "reason": reason,
        }


async def _deterministic_decide(valuation: dict[str, Any]) -> list[dict]:
    """Rule-based strategy: use analysis.compute() signals to trade.

    SELL logic:
      - Sell any position whose signal is SELL.
      - Sell any position whose current price drops below ATR stop (from snapshot).

    BUY logic:
      - Among candidates with BUY signal, rank by strength.
      - Buy the top candidate if cash > min_cash_pct of total equity and the
        position wouldn't exceed max_position_pct of total equity.
    """
    trades: list[dict] = []
    tickers = await _candidate_tickers()

    # Gather signals for all candidates
    signals: dict[str, dict] = {}
    for t in tickers:
        try:
            rows = await candles(t)
            r = compute(rows)
            signals[t] = r
        except Exception:
            continue

    total_equity = valuation["total_equity"]
    if total_equity <= 0:
        return trades

    min_cash = total_equity * (settings.sim_min_cash_pct / 100)
    max_position_value = total_equity * (settings.sim_max_position_pct / 100)

    # --- SELL phase ---
    async with Session() as s:
        open_positions = (await s.scalars(select(SimPosition))).all()

    for pos in open_positions:
        sig = signals.get(pos.ticker)
        price = await _latest_close(pos.ticker)
        if price is None:
            continue

        if sig and sig["action"] == "SELL":
            t = await _exec_sell(pos.ticker, price, None, sig["reason"])
            if t:
                trades.append(t)
            continue

        # ATR stop check (from snapshot)
        if sig:
            snap = sig.get("snapshot", {})
            atr_stop = snap.get("atr_stop")
            if atr_stop and price < atr_stop:
                t = await _exec_sell(
                    pos.ticker, price, None,
                    f"ATR stop hit: price {price:.2f} < stop {atr_stop:.2f}",
                )
                if t:
                    trades.append(t)

    # Recompute equity after sells
    if trades:
        valuation = await valuate()
        total_equity = valuation["total_equity"]
        min_cash = total_equity * (settings.sim_min_cash_pct / 100)
        max_position_value = total_equity * (settings.sim_max_position_pct / 100)

    # --- BUY phase ---
    buy_candidates = [
        (t, sig) for t, sig in signals.items()
        if sig["action"] == "BUY"
    ]
    buy_candidates.sort(key=lambda x: x[1]["strength"], reverse=True)

    for ticker, sig in buy_candidates:
        acc = await _account()
        if acc.cash < min_cash:
            break  # not enough cash to keep buffer

        price = await _latest_close(ticker)
        if price is None or price <= 0:
            continue

        # Check existing position size
        async with Session() as s:
            pos = await s.get(SimPosition, ticker)
        current_value = (pos.shares * price) if pos else 0
        if current_value >= max_position_value:
            continue  # position already at max

        budget = min(acc.cash - min_cash, max_position_value - current_value)
        if budget < price:
            continue

        t = await _exec_buy(ticker, price, budget, sig["reason"])
        if t:
            trades.append(t)

    return trades


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

async def run_cycle() -> dict[str, Any]:
    """Run one full sim cycle: deposit allowance → refresh → decide → snapshot.

    This is called by the scheduler or the manual trigger endpoint.
    """
    # 1. Deposit allowance (if new month)
    allowance_result = await deposit_allowance()

    # 2. Refresh candles for the universe
    tickers = await _candidate_tickers()
    refresh_errors: list[str] = []
    for t in tickers:
        try:
            await refresh(t, _SIM_REFRESH_PERIOD)
        except Exception as e:
            refresh_errors.append(f"{t}: {e}")

    # 3. Valuate
    valuation = await valuate()

    # 4. Decide & trade
    strategy = settings.sim_strategy.lower()
    if strategy == "deterministic":
        trades = await _deterministic_decide(valuation)
    elif strategy == "llm":
        # TODO: implement LLM strategy
        logger.warning("LLM strategy not yet implemented, falling back to deterministic")
        trades = await _deterministic_decide(valuation)
    elif strategy == "hybrid":
        # TODO: implement hybrid strategy
        logger.warning("Hybrid strategy not yet implemented, falling back to deterministic")
        trades = await _deterministic_decide(valuation)
    else:
        logger.warning("Unknown strategy '%s', falling back to deterministic", strategy)
        trades = await _deterministic_decide(valuation)

    # 5. Snapshot for equity curve
    post_valuation = await valuate()
    allowance_total = post_valuation["allowance_total"]
    async with Session() as s:
        snap = SimSnapshot(
            cash=post_valuation["cash"],
            positions_value=post_valuation["positions_value"],
            total_equity=post_valuation["total_equity"],
            allowance_total=allowance_total,
        )
        s.add(snap)
        await s.commit()

    return {
        "allowance": allowance_result,
        "refresh_errors": refresh_errors,
        "trades": trades,
        "valuation": post_valuation,
    }


# ---------------------------------------------------------------------------
# Query helpers (for API endpoints)
# ---------------------------------------------------------------------------

async def get_trades(limit: int = 100) -> list[dict]:
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
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SimAllowance).order_by(SimAllowance.month.desc())
            )
        ).all()
        return [
            {"id": r.id, "amount": r.amount, "month": r.month, "at": r.created_at.isoformat()}
            for r in rows
        ]


async def get_equity_curve(limit: int = 365) -> list[dict]:
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SimSnapshot).order_by(SimSnapshot.created_at.desc()).limit(limit)
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
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

async def reset_sim() -> dict[str, Any]:
    """Wipe all sim tables and re-initialize with start cash."""
    async with Session() as s:
        await s.execute(delete(SimTrade))
        await s.execute(delete(SimPosition))
        await s.execute(delete(SimAllowance))
        await s.execute(delete(SimSnapshot))
        await s.execute(delete(SimAccount))

        acc = SimAccount(id=1, cash=settings.sim_start_cash, last_allowance_month=None)
        s.add(acc)
        await s.commit()

    logger.info("Sim reset: cash=%.2f", settings.sim_start_cash)
    return {"ok": True, "cash": settings.sim_start_cash}


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

_scheduler_task: asyncio.Task | None = None


async def _scheduler_loop():
    """Background loop that runs the sim cycle daily at sim_run_hour UTC."""
    while True:
        now = _utcnow()
        # Calculate seconds until next sim_run_hour
        target = now.replace(hour=settings.sim_run_hour, minute=0, second=0, microsecond=0)
        if target <= now:
            # Already past today's run hour — schedule for tomorrow
            from datetime import timedelta
            target = target + timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.info("Sim scheduler: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            result = await run_cycle()
            logger.info("Sim cycle complete: %d trades", len(result["trades"]))
        except Exception as e:
            logger.error("Sim cycle failed: %s", e, exc_info=True)


def start_scheduler():
    """Start the background scheduler task (called from main.py lifespan)."""
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_scheduler_loop())


def stop_scheduler():
    """Stop the background scheduler task."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
    _scheduler_task = None