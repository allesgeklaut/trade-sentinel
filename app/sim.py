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
    SimChatMessage,
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
# Stores the deterministic trades from the most recent hybrid cycle (for comparison).
_last_deterministic_trades: list[dict] = []
# Stores the parsed LLM decisions from the most recent cycle.
_last_llm_decisions: list[dict] = []

# How many candles to refresh for the sim universe (2y is a good balance
# for indicator computation without excessive API load).
_SIM_REFRESH_PERIOD = "2y"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Used only for _current_month(): the monthly allowance is a calendar-month
# concept, so we anchor it to the operator's local timezone (Europe/Vienna).
_TZ = ZoneInfo("Europe/Vienna")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    "5. You may receive recent news headlines for supplementary context. News "
    "can explain *why* indicators are moving, but do not make trades based on "
    "news alone — the technical signals and risk rules take priority. Never "
    "reference specific URLs in your output.\n"
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
    news: dict[str, list[dict]] | None = None,
) -> str:
    """Build the compact context string sent to the LLM.

    ``news`` is an optional dict of ``{"market": [...], "TICKER": [...]}``
    headline lists. If empty or None, the news section is omitted.
    """
    from .news import format_news_for_context, format_market_news_for_context

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

    # --- Recent news (optional, supplementary) ---
    if news:
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
    news: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Hybrid strategy: let an LLM review/adjust deterministic candidates.

    Builds a structured context with the portfolio state, deterministic
    candidates, signal summaries, and optional recent news, asks the LLM for
    a JSON array of decisions, then executes each BUY/SELL through the same
    exec helpers.

    If the LLM call fails or the response can't be parsed, falls back to the
    deterministic trades.
    """
    # Default fallback
    if signals is None:
        signals = {}

    global _last_llm_reasoning

    context = _build_llm_context(valuation, deterministic_trades, signals, news)
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
            # Don't use raise_for_status() — Ollama may return non-200 with
            # useful JSON we can still inspect. Match the dashboard chat approach.
            data = resp.json()
    except Exception as e:
        err_detail = f"{type(e).__name__}: {e}"
        if hasattr(e, 'response'):
            try:
                err_detail += f" | status={e.response.status_code} body={e.response.text[:300]}"
            except Exception:
                pass
        logger.warning("LLM decide failed [%s] (url=%s, model=%s); falling back to deterministic",
                       err_detail, url, settings.ollama_model)
        _last_llm_reasoning = (
            f"[LLM UNAVAILABLE — fell back to deterministic]\n"
            f"Error: {err_detail}\n"
            f"Ollama URL: {url}\n"
            f"Model: {settings.ollama_model}\n\n"
            f"Deterministic trades were executed instead:"
        ) + ("\n" + "\n".join(
            f"  - {t['ticker']} {t['side']} ×{t.get('shares', '?')} @ {t.get('price', '?'):.2f} — {t.get('reason', '')}"
            for t in deterministic_trades
        ) if deterministic_trades else "\n  (no deterministic trades either)")
        return list(deterministic_trades)

    # Check for Ollama error response (non-200 or error field in JSON)
    resp_status = resp.status_code if hasattr(resp, 'status_code') else '?'
    if resp_status != 200:
        err_body = json.dumps(data)[:500] if data else getattr(resp, 'text', '')[:500]
        logger.warning("LLM decide got HTTP %s: %s; falling back to deterministic", resp_status, err_body)
        _last_llm_reasoning = (
            f"[LLM ERROR — fell back to deterministic]\n"
            f"HTTP {resp_status} from Ollama\n"
            f"Response: {err_body}\n"
            f"Ollama URL: {url}\n"
            f"Model: {settings.ollama_model}\n\n"
            f"Deterministic trades were executed instead:"
        ) + ("\n" + "\n".join(
            f"  - {t['ticker']} {t['side']} ×{t.get('shares', '?')} @ {t.get('price', '?'):.2f} — {t.get('reason', '')}"
            for t in deterministic_trades
        ) if deterministic_trades else "\n  (no deterministic trades either)")
        return list(deterministic_trades)

    # Ollama chat response: data["message"]["content"]
    content = ""
    try:
        content = data.get("message", {}).get("content", "") or data.get("response", "")
    except (AttributeError, TypeError):
        pass

    # Store raw LLM reasoning for display in the frontend
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


def _top_news_candidates(signals: dict[str, dict], limit: int = 10) -> list[str]:
    """Return the tickers most relevant for news: those with BUY/SELL signals
    or the highest strength HOLDs. Limits network calls per sim cycle."""
    ranked = sorted(
        signals.items(),
        key=lambda x: (x[1]["action"] != "HOLD", x[1]["strength"]),
        reverse=True,
    )
    return [t for t, _ in ranked[:limit]]


async def _gather_news(signals: dict[str, dict]) -> dict[str, list[dict]]:
    """Fetch news for the top candidate tickers + market-wide news.

    Returns ``{}`` if SearXNG is disabled or no news is found.
    """
    from .news import gather_news_for_candidates

    candidates = _top_news_candidates(signals)
    return await gather_news_for_candidates(candidates, include_market=True)


async def run_cycle() -> dict[str, Any]:
    """Run one full sim cycle: deposit allowance → refresh → decide → snapshot.

    This is called by the scheduler or the manual trigger endpoint.
    """
    global _last_deterministic_trades

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
        _last_deterministic_trades = trades
    elif strategy == "llm":
        # Pure LLM: let the model decide entirely from portfolio context + signals
        signals = await _gather_signals(tickers)
        news = await _gather_news(signals)
        trades = await _llm_decide(valuation, [], signals, news)
        _last_deterministic_trades = []
    elif strategy == "hybrid":
        # Hybrid: run deterministic first, then let LLM review/adjust
        deterministic_trades = await _deterministic_decide(valuation)
        # Recompute valuation after deterministic trades changed the portfolio
        valuation = await valuate()
        signals = await _gather_signals(tickers)
        news = await _gather_news(signals)
        trades = await _llm_decide(valuation, deterministic_trades, signals, news)
        _last_deterministic_trades = deterministic_trades
    else:
        logger.warning("Unknown strategy '%s', falling back to deterministic", strategy)
        trades = await _deterministic_decide(valuation)
        _last_deterministic_trades = trades

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


def get_last_llm_summary() -> dict[str, Any]:
    """Return a structured reasoning summary for the frontend.

    Parses the raw LLM JSON and compares it against the deterministic
    trades to highlight where the LLM changed the plan. Returns:
    - raw: the full raw LLM text
    - changes: decisions where the LLM diverged from deterministic
    - confirmations: decisions where the LLM agreed with deterministic
    - holds: HOLD decisions (abbreviated)
    - deterministic_trades: what the deterministic engine proposed
    """
    global _last_llm_decisions

    raw = _last_llm_reasoning
    is_fallback = raw.startswith("[LLM UNAVAILABLE")
    decisions = _parse_llm_decisions(raw) or []
    _last_llm_decisions = decisions

    # Build a lookup of what the deterministic engine proposed per ticker
    det_by_ticker: dict[str, dict] = {}
    for t in _last_deterministic_trades:
        det_by_ticker[t["ticker"]] = t

    changes: list[dict] = []
    confirmations: list[dict] = []
    holds: list[dict] = []

    for d in decisions:
        ticker = d["ticker"]
        action = d["action"]
        reason = d["reason"]
        det = det_by_ticker.get(ticker)

        if action == "HOLD":
            holds.append({"ticker": ticker, "reason": reason})
            continue

        if det:
            det_side = det["side"]
            if action == det_side:
                confirmations.append({
                    "ticker": ticker, "action": action, "reason": reason,
                    "det_reason": det.get("reason", ""),
                })
            else:
                changes.append({
                    "ticker": ticker, "action": action, "reason": reason,
                    "det_action": det_side, "det_reason": det.get("reason", ""),
                    "change_type": "override",
                })
        else:
            # LLM proposed something the deterministic engine didn't
            changes.append({
                "ticker": ticker, "action": action, "reason": reason,
                "det_action": None, "det_reason": None,
                "change_type": "new",
            })

    # Also check tickers where deterministic proposed a trade but LLM didn't mention them
    decision_tickers = {d["ticker"] for d in decisions}
    for ticker, det in det_by_ticker.items():
        if ticker not in decision_tickers:
            changes.append({
                "ticker": ticker, "action": "HOLD",
                "reason": "LLM did not mention this ticker",
                "det_action": det["side"], "det_reason": det.get("reason", ""),
                "change_type": "dropped",
            })

    return {
        "raw": raw,
        "deterministic_trades": _last_deterministic_trades,
        "changes": changes,
        "confirmations": confirmations,
        "holds": holds,
        "fallback": is_fallback,
    }


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
        now = _utcnow()
        # Calculate seconds until next sim_run_hour (interpreted as UTC)
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


# ---------------------------------------------------------------------------
# Interactive LLM chat (sim portfolio manager)
# ---------------------------------------------------------------------------

_SIM_CHAT_SYSTEM_PROMPT = (
    "You are a portfolio manager for a paper-trading simulation bot. "
    "The user can chat with you about the current portfolio and ask you "
    "to take actions. You have access to the current portfolio state, "
    "technical signals, and recent news.\n"
    "\n"
    "Rules:\n"
    "1. You can discuss the portfolio, explain your decisions, and answer questions.\n"
    "2. If the user asks you to take an action (e.g. \"sell X to buy Y\"), include "
    "a SPECIAL ACTION block at the end of your response in this format:\n"
    "   [[ACTION]]\n"
    "   {\"actions\": [{\"ticker\": \"...\", \"action\": \"BUY\"|\"SELL\", \"reason\": \"...\"}]}\n"
    "   [[/ACTION]]\n"
    "3. You may propose multiple actions in one response (e.g. sell one ticker, buy another).\n"
    "4. You can specify a partial position size per action using optional fields:\n"
    "   - \"shares\": exact number of shares to trade (e.g. 3.5).\n"
    "   - \"amount\": dollar amount to trade (e.g. 67.43). For SELL this is the"
    " value of shares to sell; for BUY it is the dollars to invest.\n"
    "   If neither is given, SELL sells the entire position and BUY invests the"
    " maximum allowed by risk rules.\n"
    "   Example: {\"actions\": [{\"ticker\": \"MDB\", \"action\": \"SELL\", \"amount\": 67.43, \"reason\": \"trim overweight\"}]}\n"
    "5. The same risk management rules apply: respect min cash % and max position %;"
    " the engine will clamp your requested amounts to stay within them.\n"
    "6. Only propose actions you believe are justified by the signals and portfolio context.\n"
    "7. If you do not agree with the user's request, explain why and omit the ACTION block.\n"
    "8. Do NOT reference specific URLs in your output.\n"
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
            lines.append(
                f"  {ticker}: {sig['action']} (strength {sig['strength']}) "
                f"| RSI {snap.get('rsi', 0):.1f} | MACD {snap.get('macd', 0):.3f} "
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

    return "\n".join(lines)


async def sim_chat(messages: list[dict]) -> dict[str, Any]:
    """Interactive chat with the sim portfolio manager LLM.

    ``messages`` is the list of NEW messages for this turn (typically just the
    user's latest message). The full prior conversation is loaded from
    persistent storage, merged, sent to the LLM, and the new messages are
    saved so the thread survives page reloads.

    If the LLM includes an [[ACTION]] block, executes the proposed trades.
    Returns the text response, any executed trades, and the full history.
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
        return {"text": "No messages to send.", "trades": [], "actions_executed": False, "history": []}

    context = await _build_sim_chat_context()
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
            {"role": "system", "content": _SIM_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ] + history,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("Sim chat LLM call failed: %s", e)
        return {"text": f"LLM unavailable: {e}", "trades": [], "actions_executed": False, "history": history}

    content = data.get("message", {}).get("content", "") or data.get("response", "")

    # Check for action block
    actions = _parse_action_block(content)
    executed_trades: list[dict] = []

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

    if actions:
        # Execute the proposed actions
        valuation = await valuate()
        total_equity = valuation["total_equity"]
        min_cash = total_equity * (settings.sim_min_cash_pct / 100)
        max_position_value = total_equity * (settings.sim_max_position_pct / 100)

        for action in actions:
            ticker = action["ticker"]
            act = action["action"]
            reason = action["reason"]

            price = await _latest_close(ticker)
            if price is None or price <= 0:
                logger.warning("Sim chat action skipped %s: no price", ticker)
                continue

            if act == "SELL":
                # Optional partial size: "shares" (exact) or "amount" (dollars).
                target_shares: float | None = None
                if "shares" in action:
                    target_shares = action["shares"]
                elif "amount" in action:
                    target_shares = action["amount"] / price
                t = await _exec_sell(ticker, price, target_shares, f"User chat: {reason}")
                if t:
                    executed_trades.append(t)
                    valuation = await valuate()
                    total_equity = valuation["total_equity"]
                    min_cash = total_equity * (settings.sim_min_cash_pct / 100)
                    max_position_value = total_equity * (settings.sim_max_position_pct / 100)

            elif act == "BUY":
                acc = await _account()
                if acc.cash < min_cash:
                    logger.info("Sim chat BUY %s skipped: cash %.2f < min_cash %.2f", ticker, acc.cash, min_cash)
                    continue

                async with Session() as s:
                    pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
                current_value = (pos.shares * price) if pos else 0
                if current_value >= max_position_value:
                    logger.info("Sim chat BUY %s skipped: position at max", ticker)
                    continue

                max_budget = min(acc.cash - min_cash, max_position_value - current_value)

                # Optional partial size: "shares" or "amount" (dollars). Clamp
                # to the risk-limited budget so we never breach cash/position limits.
                if "shares" in action:
                    budget = min(max_budget, action["shares"] * price)
                elif "amount" in action:
                    budget = min(max_budget, action["amount"])
                else:
                    budget = max_budget

                if budget < 1:
                    continue

                t = await _exec_buy(ticker, price, budget, f"User chat: {reason}")
                if t:
                    executed_trades.append(t)
                    valuation = await valuate()
                    total_equity = valuation["total_equity"]
                    min_cash = total_equity * (settings.sim_min_cash_pct / 100)
                    max_position_value = total_equity * (settings.sim_max_position_pct / 100)

        # Take a snapshot after chat-driven trades
        if executed_trades:
            post_val = await valuate()
            async with Session() as s:
                snap = SimSnapshot(
                    cash=post_val["cash"],
                    positions_value=post_val["positions_value"],
                    total_equity=post_val["total_equity"],
                    allowance_total=post_val["allowance_total"],
                )
                s.add(snap)
                await s.commit()

        return {
            "text": display_text,
            "trades": executed_trades,
            "actions_executed": True,
            "history": history,
        }

    return {"text": display_text, "trades": [], "actions_executed": False, "history": history}