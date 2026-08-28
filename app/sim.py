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
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

from sqlalchemy import delete, func, select

from .analysis import compute
from .config import settings
from . import llm as llm_mod
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
from .strategy import (
    StrategyParams,
    llm_buy_budget,
    llm_sell_shares,
    plan_llm_buys,
    propose_trades,
    reconcile_proposals,
    valuate_portfolio,
)

logger = logging.getLogger("trade_sentinel.sim")

# Stores the raw LLM reasoning text from the most recent _llm_decide() call.
_last_llm_reasoning: str = ""
# Stores the prose summary the LLM emitted before its JSON array (new prompt
# format). Empty when the LLM only returned JSON or when it was unavailable.
_last_llm_summary: str = ""
# Stores the deterministic trades from the most recent hybrid cycle (for comparison).
# In the hybrid strategy these are the PROPOSALS (pre-veto); some may have been
# vetoed by the LLM and never executed — see _last_llm_vetoes.
_last_deterministic_trades: list[dict] = []
# Stores the parsed LLM decisions from the most recent cycle.
_last_llm_decisions: list[dict] = []
# Stores the deterministic proposals the LLM vetoed (HOLD) in the most recent
# hybrid cycle. Empty for non-hybrid strategies and for the pure-LLM strategy.
_last_llm_vetoes: list[dict] = []

# In-progress run-cycle state for the frontend status poller. Cleared at the
# start of each run_cycle() and updated at each stage; read by /api/sim/run-status.
# Shape: {"running": bool, "stage": str, "started_at": iso, "updated_at": iso,
#         "detail": str, "error": str|None}
_run_progress: dict[str, Any] = {"running": False, "stage": "", "started_at": "", "updated_at": "", "detail": "", "error": None}

# Serializes run_cycle() calls so the scheduler and a manual "Run Bot Now"
# click can't execute simultaneously. If a cycle is already running, a
# second call returns immediately instead of racing on trades/snapshots.
_run_cycle_lock: asyncio.Lock = asyncio.Lock()
# Holds the asyncio.Task for the currently in-flight run_cycle() (whether
# triggered manually or by the scheduler). Used by the decoupled
# /api/sim/run endpoint so the cycle survives browser disconnects.
_run_cycle_task: asyncio.Task | None = None

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


def _current_week() -> str:
    """ISO calendar week key (YYYY-Www), anchored to the operator's local
    timezone. Used for the weekly LLM portfolio review: the review fires on
    the FIRST cycle of each new week, regardless of manual runs."""
    return datetime.now(_TZ).strftime("%G-W%V")


def _is_stop_out(reason: str) -> bool:
    """True if a SELL was an automatic stop, not a signal-driven decision.

    Mirrors optimize._is_stop_out: the engine stamps these reasons on
    stop-loss exits ("Initial stop: ...", "ATR stop hit: ...",
    "Trailing stop: ...", "Portfolio stop: ..."). Signal-driven SELLs say
    "SELL signal (strength ...)".
    """
    r = reason.lower()
    return any(r.startswith(p) for p in
               ("initial stop", "atr stop", "trailing stop", "portfolio stop"))


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
    """Return the most recent cached close price for *ticker*, or None."""
    rows = await candles(ticker)
    if not rows:
        return None
    return float(rows[-1]["close"])


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
            "thesis": p.thesis or "",
            "buy_date": p.opened_at.strftime("%Y-%m-%d") if p.opened_at else "",
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
    # Fractional shares — floor (not round) so cost never exceeds the
    # budget; rounding up can push a high-priced ticker cents over the
    # available cash and silently drop the buy.
    shares = math.floor(max_budget / price * 10000) / 10000
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
            s.add(SimPosition(ticker=ticker, shares=shares, avg_cost=price,
                              thesis=reason))

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


async def _deterministic_propose(
    valuation: dict[str, Any],
    signals: dict[str, dict] | None = None,
) -> list[dict]:
    """Propose deterministic trades WITHOUT executing them.

    Thin wrapper over ``strategy.propose_trades`` — loads positions from the
    DB, fetches current prices, gathers signals if not pre-supplied, converts
    to plain data, and delegates the SELL/BUY logic to the shared module.
    The hybrid strategy passes the proposals to the LLM for review/veto
    before any DB write happens.

    Each proposal carries everything ``_execute_proposed`` needs:
      - SELL: {"ticker", "side": "SELL", "price", "shares": None, "reason"}
      - BUY:  {"ticker", "side": "BUY",  "price", "budget", "reason", "entry_stop"}
    """
    tickers = await _candidate_tickers()

    if signals is None:
        signals = {}
        for t in tickers:
            try:
                rows = await candles(t)
                signals[t] = compute(rows)
            except Exception:
                continue

    # Load positions from DB and convert to plain data.
    async with Session() as s:
        open_positions = (await s.scalars(select(SimPosition))).all()
    positions = [
        {"ticker": pos.ticker, "shares": pos.shares, "avg_cost": pos.avg_cost,
         "stop_price": None}  # sim recomputes stop from avg_cost (no frozen column)
        for pos in open_positions
    ]

    # Fetch current prices for all held + candidate tickers.
    prices: dict[str, float] = {}
    for p in positions:
        price = await _latest_close(p["ticker"])
        if price is not None:
            prices[p["ticker"]] = price
    for t in signals:
        if t not in prices:
            price = await _latest_close(t)
            if price is not None:
                prices[t] = price

    sp = StrategyParams(
        buy_threshold=40, sell_threshold=-40,
        min_cash_pct=settings.sim_min_cash_pct,
        max_position_pct=settings.sim_max_position_pct,
        max_positions=settings.sim_max_positions,
        stop_type="percent",
        stop_pct=settings.sim_stop_pct,
        use_atr_stop=True,
        max_run_5d=settings.sim_max_run_5d,
        # Entry guards: block falling-knife (5d crash) and parabolic
        # (dist_above SMA200) entries — see strategy.py for the measured
        # rationale. Set the env vars to 0 to disable.
        min_run_5d=settings.sim_min_run_5d,
        max_dist_above=settings.sim_max_dist_above,
        relaxed_hold_strength=40,
        relaxed_hold_limit=3,
    )
    return propose_trades(positions, valuation["cash"], prices, signals, sp)


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
            open_count = await s.scalar(select(func.count()).select_from(SimPosition))
            held = await s.scalar(select(SimPosition).where(SimPosition.ticker == p["ticker"]))
        if open_count >= settings.sim_max_positions and held is None:
            continue  # at cap and this is a new position — skip
        t = await _exec_buy(p["ticker"], p["price"], p["budget"], p["reason"])
        if t:
            executed.append(t)
    return executed


async def _deterministic_decide(valuation: dict[str, Any]) -> list[dict]:
    """Rule-based strategy: propose + execute deterministic trades.

    Thin wrapper over ``_deterministic_propose`` + ``_execute_proposed`` so
    the deterministic-only strategy keeps its original behaviour. The hybrid
    strategy calls the two halves separately so the LLM can review proposals
    before they execute.
    """
    proposals = await _deterministic_propose(valuation)
    return await _execute_proposed(proposals)


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
    "  - **atr_stop**: trailing-volatility stop. Price below it = the trend "
    "broke.\n"
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
    "4. The engine enforces an initial stop loss (a fixed % below the entry "
    "price) and an ATR-based trailing stop: positions that hit either are "
    "auto-sold by the deterministic layer. Do not be surprised if a position "
    "disappears between cycles — that is the stop loss, not a decision you "
    "need to replicate.\n"
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
    "  - **atr_stop**: trailing-volatility stop. Price below it = the trend "
    "broke.\n"
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
    "that hit their initial stop or ATR trailing stop, but you make all buy "
    "and discretionary sell decisions yourself.\n"
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
    "initial stop price (a fixed % below the entry) and the signals include "
    "an ATR trailing stop. Sell any position whose current price is at or "
    "below either stop unless you have a strong indicator-based reason to "
    "override. (The engine also auto-sells stop hits as a safety net, but "
    "do not rely on it — act on your read.)\n"
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
    lines.append(f"Stop loss: {settings.sim_stop_pct:.0f}% (frozen at entry; ATR stop also applies)")
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
    for ticker, sig in signals.items():
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
        lines.append(
            f"{'ticker':<10} {'action':<6} {'strength':>8} "
            f"{'close':>10} {'rsi':>6} {'rsiΔ3':>6} {'adx':>5} {'wk':>3} {'macd':>10} {'mhΔ3':>7} {'run5d':>6} {'run20d':>7} {'run60d':>7} {'hi52d':>6}{atr_col}"
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
            lines.append(
                f"{ticker:<10} {sig['action']:<6} {sig['strength']:>8} "
                f"{(snap.get('close') or 0):>10.2f} {(snap.get('rsi') or 0):>6.1f} "
                f"{rsi_d_s:>6} "
                f"{(snap.get('adx') or 0):>5.0f} {wk:>3} "
                f"{(snap.get('macd') or 0):>10.3f} {mh_d_s:>7} {run5_s:>6} {run20_s:>7} {run60_s:>7} {dist52_s:>6}{atr_s}"
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


def _extract_summary(content: str | None) -> str:
    """Return the prose summary the LLM wrote before its JSON array.

    The new prompt asks for a brief prose summary followed by the JSON array.
    This helper returns everything before the array's opening ``[``. If the
    LLM only returned JSON (no prose), returns an empty string. Fallback
    ``[LLM UNAVAILABLE...]`` strings are detected and returned as-is (the
    caller already treats them specially via ``is_fallback``).
    """
    if not content:
        return ""
    text = content.strip()
    # The fallback message starts with "[" but is not JSON — leave it for
    # is_fallback handling upstream; no prose to extract.
    if text.startswith("[LLM UNAVAILABLE"):
        return ""
    # If the prose is followed by a ```json fence, the opening "[" we want is
    # inside the fence. Strip the fence first so we can locate the array.
    fence_idx = text.find("```")
    if fence_idx > 0:
        # There is prose before a fence. Drop the fence line; the JSON follows.
        first_nl_after_fence = text.find("\n", fence_idx)
        if first_nl_after_fence != -1:
            prose = text[:fence_idx].strip()
            return prose.rstrip(":").strip()
    loc = _find_json_array(text)
    if loc is None or loc[0] <= 0:
        # No JSON array, or it's at the very start — no prose.
        return ""
    prose = text[:loc[0]].strip()
    # Trim a trailing colon or "Then:" / "Decisions:" style lead-ins the LLM
    # might add right before the array.
    prose = prose.rstrip(":").strip()
    return prose


async def _llm_prose_summary(decisions: list[dict]) -> str:
    """Ask the LLM for a brief prose explanation of its decisions.

    Two-shot pattern: the main LLM call (with the big portfolio context)
    tends to emit only a bare JSON array because Qwen3 puts its deliberation
    in the reasoning channel. This second call has a tiny context (just the
    parsed decisions), so the model reliably produces prose in ``content``.

    Returns an empty string if the LLM is unavailable or the call fails.
    """
    if not decisions:
        # No decisions — ask for a summary of why the LLM held everything.
        prompt = (
            "The portfolio was fully reviewed this cycle and no trades were "
            "made (all positions held). Write 2-3 sentences in plain English "
            "explaining why no trades were made."
        )
    else:
        # Truncate to avoid re-sending a huge context — the LLM just needs
        # the ticker/action/reason per decision, not the full portfolio state.
        short = [
            {"ticker": d["ticker"], "action": d["action"], "reason": d.get("reason", "")}
            for d in decisions[:15]
        ]
        prompt = (
            f"Here are my decisions for this cycle:\n"
            f"{json.dumps(short, indent=2)}\n\n"
            f"Write 2-3 sentences in plain English explaining why I made "
            f"these decisions."
        )
    try:
        out = await llm_mod.chat([
            {"role": "system", "content": (
                "You are a portfolio manager. Write a brief 2-3 sentence "
                "summary in plain English. No JSON, no code fences, no "
                "markdown - just prose."
            )},
            {"role": "user", "content": prompt},
        ])
        return (out.get("text") or "").strip()
    except Exception as e:
        logger.warning("LLM prose summary call failed: %s", e)
        return ""


async def _llm_review_proposals(
    valuation: dict[str, Any],
    proposals: list[dict],
    signals: dict[str, dict],
    news: dict[str, list[dict]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Hybrid flow: let the LLM review deterministic proposals before they execute.

    Mirrors the propose → review → execute flow that makes the LLM's HOLD
    actually veto a deterministic trade instead of being a silent no-op:

      1. Build the LLM context with the deterministic *proposals* (not yet
         executed) and the current portfolio state.
      2. Call the LLM; it returns a JSON array of BUY/SELL/HOLD decisions.
      3. Reconcile: a deterministic proposal is vetoed when the LLM returns
         HOLD for its ticker; otherwise it's approved and executes. LLM
         decisions for tickers NOT in the proposals are treated as
         LLM-initiated additions and execute on top.

    Returns ``(executed, vetoed)`` where ``executed`` is the full list of
    trades that ran (deterministic survivors + LLM additions) and ``vetoed``
    is the list of deterministic proposals the LLM blocked (for logging /
    frontend display).

    On LLM call failure or unparseable response, falls back to executing all
    proposals as-is (equivalent to the deterministic-only strategy).
    """
    global _last_llm_reasoning, _last_llm_summary

    context = _build_llm_context(valuation, proposals, signals, news,
                                minimal=settings.sim_llm_minimal_prompt
                                or settings.sim_llm_mode_aware_prompt)
    backend = await llm_mod.current_backend()

    system_prompt = (_LLM_MODE_AWARE_MINIMAL_PROMPT if settings.sim_llm_mode_aware_prompt
                     else _LLM_MINIMAL_SYSTEM_PROMPT if settings.sim_llm_minimal_prompt
                     else _LLM_SYSTEM_PROMPT)

    try:
        out = await llm_mod.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ])
        content = out["text"]
    except Exception as e:
        err_detail = f"{type(e).__name__}: {e}"
        if hasattr(e, 'response'):
            try:
                err_detail += f" | status={e.response.status_code} body={e.response.text[:300]}"
            except Exception:
                pass
        logger.warning("LLM review failed [%s] (backend=%s, model=%s); executing all proposals",
                       err_detail, backend.get("name", "?"), backend.get("model", "?"))
        _last_llm_reasoning = (
            f"[LLM UNAVAILABLE — executing all deterministic proposals]\n"
            f"Error: {err_detail}\n"
            f"Backend: {backend.get('name', '?')}\n"
            f"Model: {backend.get('model', '?')}"
        )
        _last_llm_summary = ""
        executed = await _execute_proposed(proposals)
        return executed, []

    _last_llm_reasoning = content
    decisions = _parse_llm_decisions(content)
    if decisions is None:
        logger.warning("Could not parse LLM decisions; executing all proposals. Raw: %s", content[:500])
        _last_llm_summary = ""
        executed = await _execute_proposed(proposals)
        return executed, []

    # Two-shot: the main call may have put prose in reasoning_content (Qwen3)
    # and only emitted JSON in content. Try extracting prose from content
    # first; if that's empty, make a lightweight second call with just the
    # parsed decisions — the small context reliably produces prose in content.
    _last_llm_summary = _extract_summary(content)
    if not _last_llm_summary:
        _last_llm_summary = await _llm_prose_summary(decisions)

    logger.info("LLM returned %d decisions", len(decisions))

    # --- Reconcile proposals with LLM decisions ---
    approved, vetoed, proposal_tickers = reconcile_proposals(proposals, decisions)
    for p in vetoed:
        logger.info("LLM vetoed %s %s — %s", p["side"], p["ticker"], p.get("llm_reason", ""))

    # Execute approved proposals (SELLs first so cash frees up for BUYs;
    # _execute_proposed preserves order, and strategy.propose_trades already
    # emits SELLs before BUYs).
    executed = await _execute_proposed(approved)

    # --- LLM-initiated additions: decisions for tickers NOT in proposals ---
    # These execute on top of the approved proposals, through the same exec
    # helpers, honouring partial-size fields and risk guards.
    total_equity = valuation["total_equity"]
    if total_equity <= 0:
        return executed, vetoed

    async def _price_of(ticker: str) -> float | None:
        return await _latest_close(ticker)

    async def _value_of(ticker: str) -> float:
        async with Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
        price = await _latest_close(ticker) or 0.0
        return (pos.shares * price) if pos else 0.0

    plan = await plan_llm_buys(
        decisions, valuation["cash"], total_equity,
        StrategyParams(
            min_cash_pct=settings.sim_min_cash_pct,
            max_position_pct=settings.sim_max_position_pct,
        ),
        guarded=True,
        price_of=_price_of,
        value_of=_value_of,
        exclude=proposal_tickers,
    )

    for d in decisions:
        ticker_u = d["ticker"].upper()
        if ticker_u in proposal_tickers:
            continue  # already handled above
        action = d["action"]
        reason = d.get("reason", f"LLM {action}")
        if action == "HOLD":
            continue  # HOLD on a non-proposal ticker = no-op

        price = await _latest_close(d["ticker"])
        if price is None or price <= 0:
            logger.warning("LLM addition for %s skipped: no price", d["ticker"])
            continue

        if action == "BUY":
            budget = plan.get(ticker_u)
            if budget is None:
                continue
            t = await _exec_buy(d["ticker"], price, budget, f"LLM: {reason}")
            if t:
                executed.append(t)
                valuation = await valuate()
                total_equity = valuation["total_equity"]
        elif action == "SELL":
            # Replay evidence (90-day bull window: -8.7% → +4.7%): the LLM's
            # self-initiated SELLs (not proposed by the engine) are churn —
            # it sells winners in bull markets. The engine owns exits via
            # stop-losses and deterministic SELL signals. Skip LLM-initiated
            # SELLs in hybrid mode; only SELLs tied to engine proposals
            # (handled in the reconcile step above) execute.
            logger.info("LLM-initiated SELL %s skipped (engine owns exits)", d["ticker"])

    return executed, vetoed


async def _llm_decide(
    valuation: dict[str, Any],
    deterministic_trades: list[dict],
    signals: dict[str, dict] | None = None,
    news: dict[str, list[dict]] | None = None,
    pure_llm: bool = False,
) -> list[dict]:
    """LLM strategy: let an LLM decide what to buy/sell.

    In hybrid mode (``pure_llm=False``) the LLM reviews deterministic
    proposals and can veto/approve/flip them. In pure-LLM mode
    (``pure_llm=True``) there are no proposals — the LLM picks the names and
    the engine's hard sizing limits (min-cash floor, max-position-%,
    max-positions cap) size every BUY. Both modes use the shared prompt.

    Builds a structured context with the portfolio state, signal summaries,
    and optional recent news, asks the LLM for a JSON array of decisions,
    then executes each BUY/SELL through the same exec helpers.

    If the LLM call fails or the response can't be parsed, falls back to the
    deterministic trades (empty in pure-LLM mode, so effectively no trades).
    """
    # Default fallback
    if signals is None:
        signals = {}

    global _last_llm_reasoning, _last_llm_summary

    system_prompt = (_LLM_MODE_AWARE_MINIMAL_PROMPT if settings.sim_llm_mode_aware_prompt
                     else _LLM_MINIMAL_SYSTEM_PROMPT if settings.sim_llm_minimal_prompt
                     else _LLM_SYSTEM_PROMPT)  # shared prompt: main-branch parity
    # Recent trade history from the DB (newest first) so the LLM sees what it
    # did recently and can avoid round-trips / repeated mistakes.
    recent_trades: list[dict] = []
    async with Session() as s:
        recent_rows = (await s.scalars(
            select(SimTrade).order_by(SimTrade.created_at.desc()).limit(12)
        )).all()
    recent_trades = [
        {"date": tr.created_at.strftime("%Y-%m-%d") if tr.created_at else "",
         "side": tr.side, "ticker": tr.ticker, "shares": tr.shares,
         "price": tr.price, "reason": tr.reason}
        for tr in recent_rows
    ]
    context = _build_llm_context(valuation, deterministic_trades, signals, news,
                                pure_llm=pure_llm, trade_history=recent_trades,
                                minimal=settings.sim_llm_minimal_prompt
                                or settings.sim_llm_mode_aware_prompt)
    backend = await llm_mod.current_backend()

    try:
        out = await llm_mod.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ])
        content = out["text"]
    except Exception as e:
        err_detail = f"{type(e).__name__}: {e}"
        if hasattr(e, 'response'):
            try:
                err_detail += f" | status={e.response.status_code} body={e.response.text[:300]}"
            except Exception:
                pass
        mode = "pure-LLM" if pure_llm else "hybrid"
        logger.warning("LLM decide failed [%s] (backend=%s, model=%s); falling back (%s)",
                       err_detail, backend.get("name", "?"), backend.get("model", "?"), mode)
        if pure_llm:
            _last_llm_reasoning = (
                f"[LLM UNAVAILABLE — no trades executed]\n"
                f"Error: {err_detail}\n"
                f"Backend: {backend.get('name', '?')}\n"
                f"Model: {backend.get('model', '?')}"
            )
        else:
            _last_llm_reasoning = (
                f"[LLM UNAVAILABLE — fell back to deterministic]\n"
                f"Error: {err_detail}\n"
                f"Backend: {backend.get('name', '?')}\n"
                f"Model: {backend.get('model', '?')}\n\n"
                f"Deterministic trades were executed instead:"
            ) + ("\n" + "\n".join(
                f"  - {t['ticker']} {t['side']} ×{t.get('shares', '?')} @ {t.get('price', '?'):.2f} — {t.get('reason', '')}"
                for t in deterministic_trades
            ) if deterministic_trades else "\n  (no deterministic trades either)")
        _last_llm_summary = ""
        return list(deterministic_trades)

    # Store raw LLM reasoning for display in the frontend
    _last_llm_reasoning = content
    decisions = _parse_llm_decisions(content)
    if decisions is None:
        logger.warning("Could not parse LLM decisions; falling back. Raw: %s", content[:500])
        _last_llm_summary = ""
        return list(deterministic_trades)

    logger.info("LLM returned %d decisions", len(decisions))

    # Two-shot prose summary (see _llm_review_proposals for rationale).
    _last_llm_summary = _extract_summary(content)
    if not _last_llm_summary:
        _last_llm_summary = await _llm_prose_summary(decisions)

    # Recompute equity / budget guards (same logic as deterministic)
    total_equity = valuation["total_equity"]
    if total_equity <= 0:
        return []

    executed: list[dict] = []

    # Plan all BUY budgets up front so unsized BUYs split the cash evenly
    # instead of the first one taking everything.
    async def _price_of(ticker: str) -> float | None:
        return await _latest_close(ticker)

    async def _value_of(ticker: str) -> float:
        async with Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
        price = await _latest_close(ticker) or 0.0
        return (pos.shares * price) if pos else 0.0

    plan = await plan_llm_buys(
        decisions, valuation["cash"], total_equity,
        StrategyParams(
            min_cash_pct=settings.sim_min_cash_pct,
            max_position_pct=settings.sim_max_position_pct,
            max_positions=settings.sim_max_positions,
        ),
        guarded=True,
        price_of=_price_of,
        value_of=_value_of,
    )

    # Track held tickers for the max-positions cap on LLM-initiated BUYs.
    # Updated progressively as BUYs execute so the 2nd new BUY sees the 1st.
    held_tickers: set[str] = set()
    async with Session() as s:
        existing = (await s.scalars(select(SimPosition))).all()
    held_tickers = {p.ticker.upper() for p in existing}

    for decision in decisions:
        ticker = decision["ticker"]
        action = decision["action"]
        reason = decision["reason"] or f"LLM {action}"

        price = await _latest_close(ticker)
        if price is None or price <= 0:
            logger.warning("LLM decision for %s skipped: no price", ticker)
            continue

        if action == "BUY":
            # Max-positions cap: block NEW positions when at the cap, but
            # still allow topping up tickers already held (mirrors the
            # deterministic engine's behaviour).
            if (settings.sim_max_positions > 0
                    and len(held_tickers) >= settings.sim_max_positions
                    and ticker.upper() not in held_tickers):
                logger.info("LLM BUY %s skipped (max-positions cap %d)",
                            ticker, settings.sim_max_positions)
                continue
            budget = plan.get(ticker.upper())
            if budget is None:
                logger.info("LLM BUY %s skipped (budget guard)", ticker)
                continue

            t = await _exec_buy(ticker, price, budget, f"LLM: {reason}")
            if t:
                executed.append(t)
                held_tickers.add(ticker.upper())
                # Update guards after each buy
                valuation = await valuate()
                total_equity = valuation["total_equity"]

        elif action == "SELL":
            # Optional partial size: "shares" (exact) or "amount" (dollars).
            target_shares = llm_sell_shares(decision, price)
            t = await _exec_sell(ticker, price, target_shares, f"LLM: {reason}")
            if t:
                executed.append(t)
                valuation = await valuate()
                total_equity = valuation["total_equity"]

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


def _set_progress(stage: str, detail: str = "", *, running: bool = True,
                  started_at: str | None = None, error: str | None = None) -> None:
    """Update the in-progress run-cycle state for the /api/sim/run-status poller.

    Called from run_cycle() at each stage. ``started_at`` is preserved across
    updates so the frontend can show an elapsed timer.
    """
    now = datetime.now(timezone.utc).isoformat()
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

    Concurrency: an asyncio.Lock serialises calls. If a cycle is already
    running, a second call returns immediately with an "already running"
    result instead of racing on trades/snapshots/progress state.
    """
    global _last_deterministic_trades, _last_llm_reasoning, _last_llm_summary
    global _last_llm_decisions, _last_llm_vetoes

    # Non-blocking acquire: if another cycle is in flight, bail out
    # immediately rather than waiting (which would queue a third cycle
    # behind the current one and double-execute when the lock frees).
    if _run_cycle_lock.locked():
        logger.info("run_cycle() skipped — another cycle is already running")
        return {"skipped": True, "reason": "already running"}

    async with _run_cycle_lock:
        # Reset all per-cycle LLM state; the LLM paths set them if they run.
        # The deterministic path leaves them empty so the frontend can show a
        # "deterministic mode — no LLM was called" explanation instead of
        # re-reporting a previous cycle's reasoning as if it belonged here.
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
            allowance_result = await deposit_allowance()

            # 2. Refresh candles for the universe
            _set_progress("refresh", "Refreshing candle data", started_at=started_at)
            tickers = await _candidate_tickers()
            refresh_errors: list[str] = []
            for t in tickers:
                try:
                    await refresh(t, _SIM_REFRESH_PERIOD)
                except Exception as e:
                    refresh_errors.append(f"{t}: {e}")
            _set_progress("refresh", f"Refreshed {len(tickers)} tickers", started_at=started_at)

            # 2b. Refresh benchmark ticker and run benchmark DCA
            benchmark_result = {"deposited": False, "skipped": True}
            if settings.sim_benchmark_enabled:
                try:
                    await refresh(settings.sim_benchmark_ticker, _SIM_REFRESH_PERIOD)
                except Exception as e:
                    refresh_errors.append(f"{settings.sim_benchmark_ticker}: {e}")
                benchmark_result = await _benchmark_deposit_and_buy()

            # 3. Valuate
            _set_progress("valuate", "Valuing portfolio", started_at=started_at)
            valuation = await valuate()

            # 4. Decide & trade
            strategy = settings.sim_strategy.lower()
            if strategy == "deterministic":
                _set_progress("decide", "Deterministic engine deciding", started_at=started_at)
                trades = await _deterministic_decide(valuation)
                _last_deterministic_trades = trades
            elif strategy == "llm":
                _set_progress("signals", "Gathering signals", started_at=started_at)
                signals = await _gather_signals(tickers)
                news = await _gather_news(signals)
                # Pure-LLM mode (main-branch parity): the engine makes NO
                # proposals and NO risk-floor sells — the LLM picks the names,
                # and the engine's hard sizing limits (min-cash floor,
                # max-position-%, max-positions cap) size every BUY.
                _set_progress("decide", "LLM deciding (pure-LLM strategy)", started_at=started_at)
                trades = await _llm_decide(valuation, [], signals, news, pure_llm=True)
                _last_deterministic_trades = []
            elif strategy == "hybrid":
                # Hybrid with a weekly review: the deterministic engine runs
                # every cycle; the LLM reviews the proposals on the FIRST
                # cycle of each calendar week (Europe/Vienna). Manual runs
                # later in the same week do NOT trigger the LLM again — the
                # review is anchored to the week, not to a run counter.
                # On non-review cycles the deterministic proposals execute
                # as-is — the engine manages daily risk, the LLM adds
                # judgment once per week.
                #
                # Failure-marker mode (SIM_LLM_FAILURE_MARKER=true): the
                # weekly cadence is replaced by a failure trigger — the LLM
                # is consulted only when the engine shows signs of failure
                # (2+ stop-out SELLs in the last 5 trading days, or equity
                # >7% below its running peak). No cooldown: a fresh failure
                # triggers a call the same day. This is the validated
                # configuration (60d chop window: -0.8% vs -4.7% baseline).
                review_interval = max(1, settings.sim_llm_review_interval)
                week = _current_week()
                async with Session() as s:
                    acc = await s.get(SimAccount, 1)
                    if acc is None:
                        acc = SimAccount(id=1, cash=settings.sim_start_cash,
                                         last_allowance_month=None)
                        s.add(acc)
                        await s.commit()
                    last_review_week = acc.last_review_week
                if settings.sim_llm_failure_marker:
                    # --- failure marker: stop-out cascade / drawdown ---
                    async with Session() as s:
                        recent_trades = (await s.scalars(
                            select(SimTrade).order_by(SimTrade.created_at.desc()).limit(10)
                        )).all()
                        peak_equity = (await s.scalar(
                            select(func.max(SimSnapshot.total_equity))
                        )) or 0.0
                    # Mirror the replay's 5-trading-day window (optimize.py).
                    cutoff = _utcnow() - timedelta(days=7)
                    stop_outs_5d = sum(
                        1 for t in recent_trades
                        if t.side == "SELL" and _is_stop_out(t.reason)
                        and t.created_at >= cutoff
                    )
                    drawdown = (valuation["total_equity"] / peak_equity - 1) * 100 if peak_equity > 0 else 0.0
                    failure = stop_outs_5d >= 2 or drawdown <= -7.0
                    if failure:
                        logger.info("failure marker: stop_outs_5d=%d drawdown=%.1f%%",
                                    stop_outs_5d, drawdown)
                    is_review_cycle = failure
                else:
                    # Fire the review when the last review is more than
                    # (review_interval - 1) weeks ago (or never happened).
                    is_review_cycle = (
                        last_review_week is None
                        or _week_diff(week, last_review_week) >= review_interval
                    )
                if not is_review_cycle:
                    _set_progress("decide", "Deterministic engine deciding (LLM review off-cycle)", started_at=started_at)
                    trades = await _deterministic_decide(valuation)
                    _last_deterministic_trades = trades
                    _last_llm_vetoes = []
                else:
                    _set_progress("signals", "Gathering signals", started_at=started_at)
                    signals = await _gather_signals(tickers)
                    news = await _gather_news(signals)
                    _set_progress("propose", "Deterministic engine proposing", started_at=started_at)
                    proposals = await _deterministic_propose(valuation, signals)
                    _set_progress("decide", "LLM reviewing proposals (hybrid)", started_at=started_at)
                    trades, vetoed = await _llm_review_proposals(valuation, proposals, signals, news)
                    _last_deterministic_trades = proposals
                    _last_llm_vetoes = vetoed
                    # Mark the week as reviewed — no more LLM calls until the
                    # calendar week changes.
                    async with Session() as s:
                        acc = await s.get(SimAccount, 1)
                        if acc is not None:
                            acc.last_review_week = week
                            await s.commit()
            else:
                logger.warning("Unknown strategy '%s', falling back to deterministic", strategy)
                _set_progress("decide", "Deterministic engine deciding (unknown strategy fallback)", started_at=started_at)
                trades = await _deterministic_decide(valuation)
                _last_deterministic_trades = trades

            # 5. Snapshot for equity curve
            _set_progress("snapshot", "Snapshotting equity curve", started_at=started_at)
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

def get_last_llm_reasoning() -> str:
    """Return the raw LLM reasoning text from the most recent sim cycle."""
    return _last_llm_reasoning


def get_run_progress() -> dict[str, Any]:
    """Return the current/last run-cycle progress for the frontend poller."""
    return dict(_run_progress)


def get_last_llm_summary() -> dict[str, Any]:
    """Return a structured reasoning summary for the frontend.

    Parses the raw LLM JSON and compares it against the deterministic
    trades to highlight where the LLM changed the plan. Returns:
    - raw: the full raw LLM text
    - changes: decisions where the LLM diverged from deterministic
    - confirmations: decisions where the LLM agreed with deterministic
    - holds: HOLD decisions (abbreviated)
    - deterministic_trades: what the deterministic engine proposed
    - vetoes: deterministic proposals the LLM blocked (HOLD) — only populated
      in the hybrid strategy under the propose→review→execute flow
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
    vetoes: list[dict] = []

    # Tickers the LLM explicitly vetoed (HOLD on a proposed trade). These are
    # surfaced as vetoes, not passive holds, because the deterministic trade
    # was blocked from executing.
    vetoed_tickers: set[str] = {v["ticker"] for v in _last_llm_vetoes}

    for d in decisions:
        ticker = d["ticker"]
        action = d["action"]
        reason = d["reason"]
        det = det_by_ticker.get(ticker)

        if action == "HOLD":
            if ticker in vetoed_tickers:
                # HOLD on a proposed trade = veto (blocked execution)
                det_side = det["side"] if det else None
                vetoes.append({
                    "ticker": ticker, "action": "HOLD", "reason": reason,
                    "det_action": det_side, "det_reason": det.get("reason", "") if det else None,
                    "change_type": "veto",
                })
            else:
                # HOLD on a non-proposal ticker = passive hold
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

    # Was the LLM called this cycle? In the deterministic strategy it never is,
    # and both _last_llm_reasoning and _last_llm_summary stay empty (the latter
    # because run_cycle resets it). The frontend uses this flag to show a
    # "deterministic mode — no LLM was called" message instead of an empty panel.
    llm_was_called = bool(_last_llm_reasoning or _last_llm_summary)

    return {
        "raw": raw,
        "summary": _last_llm_summary,
        "deterministic_trades": _last_deterministic_trades,
        "changes": changes,
        "confirmations": confirmations,
        "holds": holds,
        "vetoes": vetoes,
        "fallback": is_fallback,
        "strategy": settings.sim_strategy,
        "failure_marker": settings.sim_llm_failure_marker,
        "mode_aware_prompt": settings.sim_llm_mode_aware_prompt,
        "llm_called": llm_was_called,
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
            target = target + timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.info("Sim scheduler: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)

        try:
            # Skip if a manual run is in flight (e.g. the user clicked "Run
            # Bot Now" shortly before the scheduled time). The lock check in
            # run_cycle() also guards against this, but checking here too
            # avoids logging a confusing "skipped — already running" entry
            # every scheduled night a manual run overlaps.
            if _run_cycle_lock.locked():
                logger.info("Sim scheduler: skipping scheduled run — a cycle is already in progress")
                continue
            result = await run_cycle()
            if result.get("skipped"):
                logger.info("Sim scheduler: run_cycle skipped — %s", result.get("reason"))
            else:
                logger.info("Sim cycle complete: %d trades", len(result["trades"]))
        except Exception as e:
            logger.error("Sim cycle failed: %s", e, exc_info=True)

        # Monthly qv-mom portfolio: independent of the daily cycle, only acts
        # on the last trading day of the month (no-op otherwise).
        try:
            from .monthly import run_monthly_cycle
            monthly_result = await run_monthly_cycle()
            if monthly_result.get("skipped"):
                logger.info("Monthly scheduler: skipped — %s", monthly_result.get("reason"))
            else:
                logger.info("Monthly rebalance complete: %d trades", len(monthly_result.get("trades", [])))
        except Exception as e:
            logger.error("Monthly rebalance failed: %s", e, exc_info=True)


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
    + _SIM_METHODOLOGY +
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
    "5. The same risk management rules apply: respect min cash %, max position %,"
    " and the max open-positions count; the engine will clamp your requested"
    " amounts to stay within them and will refuse a BUY that would exceed the"
    " position count cap.\n"
    "   EXCEPTION: if the user EXPLICITLY asks to spend the cash reserve / dry"
    " powder / remaining cash / \"all available cash\", set \"use_reserve\": true"
    " on that BUY action. This lets the buy spend below the normal min-cash floor"
    " (down to zero cash). Use it only when the user clearly requests it — never"
    " on your own initiative.\n"
    "6. The engine enforces an initial stop loss (a fixed % below the entry price)"
    " and an ATR-based trailing stop on every position. Positions that hit either"
    " are auto-sold by the deterministic layer, so if the user asks why a position"
    " vanished it was likely stopped out — explain that rather than proposing to"
    " re-buy it unless the user explicitly asks.\n"
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
    lines.append(f"Stop loss: {settings.sim_stop_pct:.0f}% (frozen at entry; ATR stop also applies)")
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

    # Last sim cycle decision (from the most recent _llm_decide run).
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

    try:
        out = await llm_mod.chat([
            {"role": "system", "content": _SIM_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ] + history)
    except Exception as e:
        logger.warning("Sim chat LLM call failed: %s", e)
        return {"text": f"LLM unavailable: {e}", "trades": [], "actions_executed": False, "history": history}

    content = out["text"]

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
                # Normally the cash reserve is a hard floor. The user can
                # explicitly authorise spending it via "use_reserve": true in
                # the chat action — then we allow buying down to zero cash.
                floor = 0.0 if action.get("use_reserve") else min_cash
                if acc.cash < floor:
                    logger.info("Sim chat BUY %s skipped: cash %.2f < floor %.2f (reserve override=%s)",
                                ticker, acc.cash, floor, bool(action.get("use_reserve")))
                    continue

                async with Session() as s:
                    pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
                current_value = (pos.shares * price) if pos else 0
                if current_value >= max_position_value:
                    logger.info("Sim chat BUY %s skipped: position at max", ticker)
                    continue

                max_budget = min(acc.cash - floor, max_position_value - current_value)

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

                override_tag = " [reserve spent]" if action.get("use_reserve") else ""
                t = await _exec_buy(ticker, price, budget, f"User chat:{override_tag} {reason}")
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


async def sim_chat_stream(messages: list[dict]):
    """Streaming variant of sim_chat — yields progressive LLM text deltas.

    Yields dicts:
      - {"type": "delta", "text": "..."}: raw LLM content chunks (may include
        the ``[[ACTION]]`` block as it is generated — the frontend will swap the
        raw text for the stripped display_text when ``done`` arrives).
      - {"type": "done", "text": display_text, "trades": [...],
         "actions_executed": bool, "raw": full_raw_text, "history": [...]}

    Persistence (user message, assistant display_text) happens AFTER the LLM
    stream completes, so a refresh mid-stream leaves the previous state intact
    instead of an orphan user question.

    If the LLM stream errors out after yielding some text, a partial ``done``
    event is emitted with ``error`` set, so the frontend can show what arrived.
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
            "trades": [],
            "actions_executed": False,
            "raw": "",
            "history": [],
        }
        return

    context = await _build_sim_chat_context()
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
            "trades": [],
            "actions_executed": False,
            "raw": "",
            "history": history,
            "error": error_msg,
        }
        return

    # Parse any action block from the complete raw text.
    actions = _parse_action_block(raw_content)
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

    executed_trades: list[dict] = []
    actions_executed = False
    if actions:
        # Execute proposed actions (same risk logic as sim_chat).
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
                floor = 0.0 if action.get("use_reserve") else min_cash
                if acc.cash < floor:
                    logger.info("Sim chat BUY %s skipped: cash %.2f < floor %.2f",
                                ticker, acc.cash, floor)
                    continue

                async with Session() as s:
                    pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == ticker))
                current_value = (pos.shares * price) if pos else 0
                if current_value >= max_position_value:
                    logger.info("Sim chat BUY %s skipped: position at max", ticker)
                    continue

                max_budget = min(acc.cash - floor, max_position_value - current_value)
                if "shares" in action:
                    budget = min(max_budget, action["shares"] * price)
                elif "amount" in action:
                    budget = min(max_budget, action["amount"])
                else:
                    budget = max_budget
                if budget < 1:
                    continue

                override_tag = " [reserve spent]" if action.get("use_reserve") else ""
                t = await _exec_buy(ticker, price, budget, f"User chat:{override_tag} {reason}")
                if t:
                    executed_trades.append(t)
                    valuation = await valuate()
                    total_equity = valuation["total_equity"]
                    min_cash = total_equity * (settings.sim_min_cash_pct / 100)
                    max_position_value = total_equity * (settings.sim_max_position_pct / 100)

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
        actions_executed = True

    yield {
        "type": "done",
        "text": display_text,
        "trades": executed_trades,
        "actions_executed": actions_executed,
        "raw": raw_content,
        "history": history,
        **({"error": error_msg} if error_msg else {}),
    }