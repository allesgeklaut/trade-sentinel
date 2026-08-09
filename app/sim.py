"""Autonomous paper-trading simulation engine for Trade Sentinel.

The bot receives an imaginary monthly allowance, decides when to buy/sell
assets from a configurable universe using deterministic signals, an LLM-based
portfolio manager (hybrid mode), or a pure LLM mode, and tracks portfolio
performance over time.

All trades are executed at the latest cached daily close price — no real
broker, no intraday, no shorting.  This is a research/education tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any

import httpx

from sqlalchemy import delete, func, select

from .analysis import compute
from .config import settings
from .db import (
    SimAccount,
    SimAllowance,
    SimBenchmarkAccount,
    SimBenchmarkSnapshot,
    SimPosition,
    SimSnapshot,
    SimTrade,
    Session,
)
from .market import candles, refresh
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.sim")

# Stores the raw LLM reasoning text from the most recent _llm_decide() call.
_last_llm_reasoning: str = ""

# How many candles to refresh for the sim universe (2y is a good balance
# for indicator computation without excessive API load).
_SIM_REFRESH_PERIOD = "2y"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TZ = ZoneInfo("Europe/Vienna")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _now() -> datetime:
    """Current time as naive Vienna local time."""
    return datetime.now(_TZ).replace(tzinfo=None)


def _current_month() -> str:
    return datetime.now(_TZ).strftime("%Y-%m")


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
    """Execute a paper BUY.  Returns the trade dict or None if skipped.

    Uses fractional shares (rounded to 4 decimals) so the bot can always
    deploy capital regardless of share price.
    """
    if price <= 0 or max_budget < 1:
        return None
    # Fractional shares — invest as much of the budget as possible
    shares = round(max_budget / price, 4)
    if shares < 0.0001:
        return None
    cost = shares * price

    async with Session() as s:
        acc = await s.get(SimAccount, 1)
        if acc is None or acc.cash < cost:
            return None
        acc.cash -= cost

        pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
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

        logger.info("Sim BUY %s ×%.4f @%.2f — %s", ticker, shares, price, reason)
        return {
            "ticker": ticker, "side": "BUY", "shares": shares, "price": price,
            "cash_after": round(acc.cash, 2), "reason": reason,
        }


async def _exec_sell(ticker: str, price: float, shares: float | None, reason: str) -> dict | None:
    """Execute a paper SELL.  *shares*=None means sell entire position.

    Returns the trade dict or None if nothing to sell.
    """
    async with Session() as s:
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

    # If no strict BUY signals, use a relaxed fallback: buy the best
    # near-BUY candidates (HOLD with highest strength) so the bot stays
    # active and deploys cash instead of sitting idle.
    if not buy_candidates:
        hold_candidates = [
            (t, sig) for t, sig in signals.items()
            if sig["action"] == "HOLD" and sig["strength"] >= 40
        ]
        hold_candidates.sort(key=lambda x: x[1]["strength"], reverse=True)
        buy_candidates = hold_candidates[:3]  # limit relaxed buys
        if buy_candidates:
            logger.info("No strict BUY signals; using %d relaxed HOLD candidates (strength >= 40)", len(buy_candidates))

    for ticker, sig in buy_candidates:
        acc = await _account()
        if acc.cash < min_cash:
            break  # not enough cash to keep buffer

        price = await _latest_close(ticker)
        if price is None or price <= 0:
            continue

        # Check existing position size
        async with Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
        current_value = (pos.shares * price) if pos else 0
        if current_value >= max_position_value:
            continue  # position already at max

        budget = min(acc.cash - min_cash, max_position_value - current_value)
        if budget < 1:
            continue

        t = await _exec_buy(ticker, price, budget, sig["reason"])
        if t:
            trades.append(t)

    return trades


# ---------------------------------------------------------------------------
# Hybrid / LLM strategy
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You are a disciplined portfolio manager reviewing deterministic technical "
    "signals for a paper-trading simulation.\n"
    "You will receive the current portfolio state and a list of candidate "
    "tickers with their technical indicators (signal action, strength, close "
    "price, RSI, MACD).\n"
    "Your job: review the deterministic trade candidates and the signals, then "
    "return your own decisions.\n"
    "\n"
    "Rules:\n"
    "1. For each candidate ticker you may decide BUY, SELL, or HOLD.\n"
    "2. You may also adjust the deterministic candidates: upgrade a HOLD to a "
    "BUY, downgrade a BUY to HOLD, or reject a SELL.\n"
    "3. Respect risk management: do not buy if cash is too low; do not over-"
    "concentrate in a single ticker.\n"
    "4. Decisions must be grounded in the provided signals and indicators.\n"
    "\n"
    "Return ONLY a JSON array of objects with the fields:\n"
    '  "ticker": string, "action": "BUY"|"SELL"|"HOLD", "reason": string\n'
    "\n"
    "No markdown, no code fences, no prose — just the JSON array.\n"
)


def _build_llm_context(
    valuation: dict[str, Any],
    deterministic_trades: list[dict],
    signals: dict[str, dict],
) -> str:
    """Build the compact context string sent to the LLM."""
    lines: list[str] = []

    # --- Portfolio state ---
    lines.append("## Current Portfolio State")
    lines.append(f"Cash: {valuation['cash']:.2f}")
    lines.append(f"Positions value: {valuation['positions_value']:.2f}")
    lines.append(f"Total equity: {valuation['total_equity']:.2f}")
    lines.append(f"Cumulative allowance deposited: {valuation['allowance_total']:.2f}")
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

    # --- Signals summary ---
    lines.append("## Signals (all candidate tickers)")
    if signals:
        lines.append(
            f"{'ticker':<10} {'action':<6} {'strength':>8} "
            f"{'close':>10} {'rsi':>6} {'macd':>10}"
        )
        for ticker, sig in sorted(signals.items()):
            snap = sig.get("snapshot", {})
            lines.append(
                f"{ticker:<10} {sig['action']:<6} {sig['strength']:>8} "
                f"{snap.get('close', 0):>10.2f} {snap.get('rsi', 0):>6.1f} "
                f"{snap.get('macd', 0):>10.3f}"
            )
    else:
        lines.append("(no signals available)")
    lines.append("")

    # --- Deterministic candidate trades ---
    lines.append("## Deterministic Candidate Trades")
    if deterministic_trades:
        for t in deterministic_trades:
            lines.append(
                f"  - {t['ticker']} {t['side']} ×{t.get('shares', '?')} @ "
                f"{t.get('price', '?'):.2f} — {t.get('reason', '')}"
            )
    else:
        lines.append("(no deterministic trades proposed)")
    lines.append("")

    lines.append("## Your Decisions")
    lines.append(
        "Return ONLY a JSON array of objects: "
        '{"ticker": "...", "action": "BUY|SELL|HOLD", "reason": "..."}. '
        "No markdown, no prose."
    )
    return "\n".join(lines)


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

    # Extract the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end + 1]

    try:
        decisions = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(decisions, list):
        return None

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
        valid.append({"ticker": ticker, "action": action, "reason": reason})
    return valid if valid else None


async def _llm_decide(
    valuation: dict[str, Any],
    deterministic_trades: list[dict],
    signals: dict[str, dict] | None = None,
) -> list[dict]:
    """Hybrid strategy: let an LLM review/adjust deterministic candidates.

    Builds a structured context with the portfolio state, deterministic
    candidates, and signal summaries, asks the LLM for a JSON array of
    decisions, then executes each BUY/SELL through the same exec helpers.

    If the LLM call fails or the response can't be parsed, falls back to the
    deterministic trades.
    """
    # Default fallback
    if signals is None:
        signals = {}

    context = _build_llm_context(valuation, deterministic_trades, signals)
    url = settings.ollama_url.rstrip("/") + "/api/chat"
    timeout = httpx.Timeout(
        connect=10.0,
        read=settings.ollama_timeout_seconds,
        write=30.0,
        pool=10.0,
    )
    payload = {
        "model": settings.ollama_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("LLM decide failed (%s); falling back to deterministic", e)
        return list(deterministic_trades)

    # Ollama chat response: data["message"]["content"]
    content = ""
    try:
        content = data.get("message", {}).get("content", "") or data.get("response", "")
    except (AttributeError, TypeError):
        pass

    # Store raw LLM reasoning for display in the frontend
    global _last_llm_reasoning
    _last_llm_reasoning = content

    decisions = _parse_llm_decisions(content)
    if decisions is None:
        logger.warning("Could not parse LLM decisions; falling back to deterministic. Raw: %s", content[:500])
        return list(deterministic_trades)

    logger.info("LLM returned %d decisions", len(decisions))

    # Recompute equity / budget guards (same logic as deterministic)
    total_equity = valuation["total_equity"]
    if total_equity <= 0:
        return []

    min_cash = total_equity * (settings.sim_min_cash_pct / 100)
    max_position_value = total_equity * (settings.sim_max_position_pct / 100)

    executed: list[dict] = []

    for decision in decisions:
        ticker = decision["ticker"]
        action = decision["action"]
        reason = decision["reason"] or f"LLM {action}"

        price = await _latest_close(ticker)
        if price is None or price <= 0:
            logger.warning("LLM decision for %s skipped: no price", ticker)
            continue

        if action == "BUY":
            acc = await _account()
            if acc.cash < min_cash:
                logger.info("LLM BUY %s skipped: cash %.2f < min_cash %.2f", ticker, acc.cash, min_cash)
                continue

            async with Session() as s:
                pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
            current_value = (pos.shares * price) if pos else 0
            if current_value >= max_position_value:
                logger.info("LLM BUY %s skipped: position at max (%.2f >= %.2f)", ticker, current_value, max_position_value)
                continue

            budget = min(acc.cash - min_cash, max_position_value - current_value)
            if budget < 1:
                logger.info("LLM BUY %s skipped: budget %.2f < $1", ticker, budget)
                continue

            t = await _exec_buy(ticker, price, budget, f"LLM: {reason}")
            if t:
                executed.append(t)
                # Update guards after each buy
                valuation = await valuate()
                total_equity = valuation["total_equity"]
                min_cash = total_equity * (settings.sim_min_cash_pct / 100)
                max_position_value = total_equity * (settings.sim_max_position_pct / 100)

        elif action == "SELL":
            t = await _exec_sell(ticker, price, None, f"LLM: {reason}")
            if t:
                executed.append(t)
                valuation = await valuate()
                total_equity = valuation["total_equity"]
                min_cash = total_equity * (settings.sim_min_cash_pct / 100)
                max_position_value = total_equity * (settings.sim_max_position_pct / 100)

        else:  # HOLD
            logger.info("LLM HOLD %s — %s", ticker, reason)
            continue

    return executed


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

async def _gather_signals(tickers: list[str]) -> dict[str, dict]:
    """Fetch compute() signals for every ticker, skipping failures."""
    signals: dict[str, dict] = {}
    for t in tickers:
        try:
            rows = await candles(t)
            signals[t] = compute(rows)
        except Exception:
            continue
    return signals


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

    # 2b. Refresh benchmark ticker and run benchmark DCA
    benchmark_result = {"deposited": False, "skipped": True}
    if settings.sim_benchmark_enabled:
        try:
            await refresh(settings.sim_benchmark_ticker, _SIM_REFRESH_PERIOD)
        except Exception as e:
            refresh_errors.append(f"{settings.sim_benchmark_ticker}: {e}")
        benchmark_result = await _benchmark_deposit_and_buy()

    # 3. Valuate
    valuation = await valuate()

    # 4. Decide & trade
    strategy = settings.sim_strategy.lower()
    if strategy == "deterministic":
        trades = await _deterministic_decide(valuation)
    elif strategy == "llm":
        # Pure LLM: let the model decide entirely from portfolio context + signals
        signals = await _gather_signals(tickers)
        trades = await _llm_decide(valuation, [], signals)
    elif strategy == "hybrid":
        # Hybrid: run deterministic first, then let LLM review/adjust
        deterministic_trades = await _deterministic_decide(valuation)
        # Recompute valuation after deterministic trades changed the portfolio
        valuation = await valuate()
        signals = await _gather_signals(tickers)
        trades = await _llm_decide(valuation, deterministic_trades, signals)
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

    # 5b. Benchmark snapshot
    await _benchmark_snapshot()

    return {
        "allowance": allowance_result,
        "benchmark": benchmark_result,
        "refresh_errors": refresh_errors,
        "trades": trades,
        "valuation": post_valuation,
        "llm_reasoning": _last_llm_reasoning,
    }


# ---------------------------------------------------------------------------
# Query helpers (for API endpoints)
# ---------------------------------------------------------------------------

def get_last_llm_reasoning() -> str:
    """Return the raw LLM reasoning text from the most recent sim cycle."""
    return _last_llm_reasoning


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

    await reset_benchmark()
    logger.info("Sim reset: cash=%.2f", settings.sim_start_cash)
    return {"ok": True, "cash": settings.sim_start_cash}


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
        "allowance_total": 0.0,  # filled below
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


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

_scheduler_task: asyncio.Task | None = None


async def _scheduler_loop():
    """Background loop that runs the sim cycle daily at sim_run_hour UTC."""
    while True:
        now = _now()
        # Calculate seconds until next sim_run_hour
        target = now.replace(hour=settings.sim_run_hour, minute=settings.sim_run_minute, second=0, microsecond=0)
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