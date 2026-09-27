"""Savings tracker engine: the operator's REAL Trade Republic account (EUR).

Tracks real money — cash with daily interest accrual (paid out monthly like
Trade Republic), Sparplan buys (virtual, from the savings cash at the cached
close), and a forward mirror that allocates every € invested into positions
into the current daily-core suggestion for an honest "your picks vs
daily-core's picks" comparison.

No connection to Trade Republic exists: the balance is seeded/true-upped
manually, everything else is derived. Deliberately separate from the paper
sims — ``reset-all`` and backfills never touch these tables. The scheduler
pass is idempotent per day/month so downtime catch-up just catches up.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import delete as sa_delete
from sqlalchemy import select

from .config import settings
from .db import (
    DailyCoreRanking,
    SavingsAccount,
    SavingsEvent,
    SavingsMirrorEntry,
    SavingsMirrorPosition,
    SavingsPlan,
    SavingsPosition,
    SavingsSnapshot,
    Session,
)

logger = logging.getLogger("trade_sentinel.savings")

# The paper sims' convention: calendar-month actions anchor to the
# operator's local timezone (shared ALLOWANCE_TZ) so deposits land on the
# local month boundary.
_TZ = ZoneInfo(settings.allowance_tz)

# One cycle lock: the scheduler pass, a manual "Run Now" and an API-triggered
# accrual must not interleave writes to the singleton account row.
_cycle_lock = asyncio.Lock()

# Interest accrues on the balance at end of each day (TR's own convention:
# daily accrual, monthly payout). actual/365 is the savings-account norm.
_ACCRUAL_DIVISOR = 365.0


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _current_month() -> str:
    return datetime.now(_TZ).strftime("%Y-%m")


def _today_local() -> date:
    return datetime.now(_TZ).date()


# ---------------------------------------------------------------------------
# EUR pricing
# ---------------------------------------------------------------------------

_EUR_SUFFIXES = {".DE", ".AS", ".MC", ".PA", ".BR", ".HE", ".VI", ".MI", ".AT"}


def _is_eur_ticker(ticker: str) -> bool:
    """Native-EUR listing? Those price directly in EUR; everything else
    (US names, .SW, .L, .ST …) is converted from its USD/local price."""
    suffix = ticker[ticker.rfind(".") :] if "." in ticker else ""
    return suffix in _EUR_SUFFIXES


async def _eur_usd_rate() -> float | None:
    """Latest EURUSD=X close (USD per EUR), or None when no candle exists."""
    from .db import Candle

    async with Session() as s:
        row = await s.scalar(
            select(Candle.close)
            .where(Candle.ticker == "EURUSD=X")
            .order_by(Candle.timestamp.desc())
            .limit(1)
        )
    return float(row) if row is not None else None


async def _price_eur_map(tickers: list[str]) -> dict[str, float | None]:
    """{ticker: latest EUR price} for the given tickers.

    Native-EUR listings use their own close (1:1). Non-EUR listings are
    converted from their USD price via the latest EURUSD=X close — one
    uniform path for US names and non-€EU listings (.SW/.L/.ST) alike.
    Missing candles or a missing FX rate yield None (caller falls back to
    avg_cost, never 0).
    """
    from .fundamentals import SUFFIX_FX
    from .market import latest_close

    tickers = sorted(set(tickers))
    if not tickers:
        return {}
    fx_tickers = sorted(
        {pm[0] for t in tickers if (pm := SUFFIX_FX.get(t[t.rfind(".") :] if "." in t else ""))}
    )
    rates: dict[str, float | None] = {}
    for p in fx_tickers:
        rates[p] = await latest_close(p)
    eur_usd = await _eur_usd_rate()

    out: dict[str, float | None] = {}
    for t in tickers:
        if _is_eur_ticker(t):
            out[t] = await latest_close(t)
            continue
        usd_px = await latest_close(t)
        if usd_px is None:
            out[t] = None
            continue
        # Local-currency listings: USD-convert via their suffix FX pair
        # (e.g. NESN.SW -> USDCHF=X) then to EUR via EURUSD=X. US tickers
        # have no suffix and are already USD.
        pm = SUFFIX_FX.get(t[t.rfind(".") :] if "." in t else "")
        if pm:
            r = rates.get(pm[0])
            if r is None:
                out[t] = None
                continue
            usd_px = usd_px * r if pm[1] == "mul" else usd_px / r
        out[t] = usd_px / eur_usd if eur_usd else None
    return out


# ---------------------------------------------------------------------------
# Mirror: same € into the daily-core suggestion
# ---------------------------------------------------------------------------


async def _daily_core_picks(max_age_days: float = 4.0) -> list[str] | None:
    """Current daily-core picks (top-10 hysteresis list), or None when the
    stored ranking is absent/older than max_age_days. The mirror defers
    allocation until a fresh ranking exists rather than acting on stale
    picks."""
    async with Session() as s:
        row = await s.get(DailyCoreRanking, 1)
    if row is None:
        return None
    age = _utcnow() - row.ranking_date.replace(tzinfo=UTC)
    if age > timedelta(days=max_age_days):
        return None
    return [t for t in (row.picks or "").split(",") if t]


async def _mirror_allocate(amount: float, source_ticker: str = "") -> dict | None:
    """Spread `amount` € equally over the current daily-core picks, buying
    each at its latest EUR price. Returns a summary dict, or None when
    allocation was deferred (no fresh ranking / no prices)."""
    picks = await _daily_core_picks()
    if not picks or amount <= 0:
        return None
    prices = await _price_eur_map(picks)
    priced = [t for t in picks if prices.get(t)]
    if not priced:
        return None
    per = amount / len(priced)
    bought: list[dict] = []
    async with Session() as s:
        for t in priced:
            price = prices[t] or 0.0
            if price <= 0:
                continue
            shares = per / price
            pos = await s.scalar(
                select(SavingsMirrorPosition).where(SavingsMirrorPosition.ticker == t)
            )
            if pos:
                total = pos.shares + shares
                pos.avg_cost = (pos.shares * pos.avg_cost + per) / total
                pos.shares = total
            else:
                s.add(SavingsMirrorPosition(ticker=t, shares=shares, avg_cost=price))
            bought.append({"ticker": t, "shares": shares, "price": price})
        # Persist the per-pick breakdown so the mirror curve can be replayed
        # as a true time series (shares as of each date × that date's close).
        s.add(SavingsMirrorEntry(
            amount=amount, source_ticker=source_ticker,
            breakdown=json.dumps([[b["ticker"], b["shares"], b["price"]]
                                  for b in bought])))
        await s.commit()
    return {"amount": amount, "pick_count": len(priced), "bought": bought}


async def _mirror_valuate() -> dict:
    """Mirror portfolio valuation in EUR."""
    async with Session() as s:
        rows = (await s.scalars(select(SavingsMirrorPosition))).all()
        entries = (
            await s.scalars(select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))
        ).all()
    prices = await _price_eur_map([r.ticker for r in rows]) if rows else {}
    positions_value = 0.0
    pos_list = []
    for r in rows:
        price = prices.get(r.ticker) or r.avg_cost
        value = r.shares * price
        positions_value += value
        pos_list.append(
            {
                "ticker": r.ticker,
                "shares": r.shares,
                "avg_cost": round(r.avg_cost, 4),
                "current_price": round(price, 4),
                "value": round(value, 2),
                "pnl_pct": round((price - r.avg_cost) / r.avg_cost * 100, 2) if r.avg_cost else 0.0,
            }
        )
    allocated = sum(e.amount for e in entries)
    return {
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(positions_value, 2),
        "allocated": round(allocated, 2),
        "entries": len(entries),
    }


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------


async def _account(s) -> SavingsAccount | None:
    """Fetch the singleton row (no auto-create: an absent row means the
    tracker is uninitialized and every engine action no-ops)."""
    return await s.get(SavingsAccount, 1)


async def initialized() -> bool:
    async with Session() as s:
        return (await _account(s)) is not None


async def held_tickers() -> list[str]:
    """All tickers the tracker needs candles for: positions, plans and the
    mirror book, plus EURUSD=X (EUR pricing of non-EUR names)."""
    async with Session() as s:
        pos = (await s.scalars(select(SavingsPosition))).all()
        plans = (await s.scalars(select(SavingsPlan).where(SavingsPlan.active == 1))).all()
        mirror = (await s.scalars(select(SavingsMirrorPosition))).all()
    out = {p.ticker for p in pos} | {p.ticker for p in plans} | {m.ticker for m in mirror}
    out.add("EURUSD=X")
    return sorted(out)


async def refresh_data() -> dict:
    """Refresh candles for held tickers + EURUSD=X so valuation, Sparplan
    buys and the mirror price on the MOST RECENT data (on a weekend that is
    Friday's close — EOD feeds simply don't move until Monday). Free Yahoo
    bulk path; skips tickers fetched within ``market_fresh_seconds`` so the
    nightly loop and a manual pass don't double-fetch."""
    from .market import refresh_many

    tickers = await held_tickers()
    if not tickers:
        return {"refreshed": [], "errors": []}
    refreshed, errors = await refresh_many(
        tickers, "2y", max_age_seconds=settings.market_fresh_seconds)
    return {"refreshed": refreshed, "errors": errors}


async def _log_event(
    s, kind: str, amount: float = 0.0, ticker: str | None = None, note: str = ""
) -> None:
    s.add(SavingsEvent(kind=kind, amount=amount, ticker=ticker, note=note))


# ---------------------------------------------------------------------------
# Core actions (each returns a summary dict; all idempotent by markers)
# ---------------------------------------------------------------------------


async def initialize(
    cash: float,
    positions: list[dict] | None = None,
    monthly_transfer: float | None = None,
    interest_rate: float | None = None,
) -> dict:
    """(Re)initialize the tracker: set the savings cash and seed current
    holdings. Wipes all previous tracker state — a deliberate, explicit
    operator action (mirrors reset_sim's semantics but for real-money data).

    ``positions``: [{ticker, shares, avg_cost}] in EUR. The total invested
    amount is mirrored into the daily-core suggestion.
    """
    positions = positions or []
    invested = 0.0
    async with Session() as s:
        acc = await _account(s)
        for tbl in (
            SavingsPosition,
            SavingsPlan,
            SavingsEvent,
            SavingsSnapshot,
            SavingsMirrorEntry,
            SavingsMirrorPosition,
        ):
            await s.execute(sa_delete(tbl))
        if acc is None:
            acc = SavingsAccount(id=1)
            s.add(acc)
        acc.cash = max(0.0, float(cash))
        acc.accrued_interest = 0.0
        acc.interest_rate = float(
            interest_rate if interest_rate is not None else settings.savings_interest_rate
        )
        acc.monthly_transfer = float(
            monthly_transfer if monthly_transfer is not None else settings.savings_monthly_transfer
        )
        month = _current_month()
        # Mark the current month as done for both monthly actions: the
        # operator seeded the balance manually, so the tracker must not
        # double-count a transfer or payout for the seed month.
        acc.last_transfer_month = month
        acc.last_interest_month = month
        acc.last_accrual_day = None
        await _log_event(s, "INIT", acc.cash, note="tracker initialized")
        for p in positions:
            ticker = str(p.get("ticker", "")).strip().upper()
            shares = float(p.get("shares", 0) or 0)
            avg_cost = float(p.get("avg_cost", 0) or 0)
            if not ticker or shares <= 0 or avg_cost <= 0:
                continue
            s.add(SavingsPosition(ticker=ticker, shares=shares, avg_cost=avg_cost))
            invested += shares * avg_cost
        await s.commit()
    result = {
        "ok": True,
        "cash": acc.cash,
        "invested": round(invested, 2),
        "last_transfer_month": month,
    }
    if invested > 0:
        mirror = await _mirror_allocate(invested, source_ticker="")
        result["mirror"] = mirror and {"allocated": mirror["amount"], "picks": mirror["pick_count"]}
    return result


async def reset() -> dict:
    """Wipe the tracker back to the uninitialized state (the Savings tab
    shows the setup form again)."""
    async with Session() as s:
        for tbl in (
            SavingsPosition,
            SavingsPlan,
            SavingsEvent,
            SavingsSnapshot,
            SavingsMirrorEntry,
            SavingsMirrorPosition,
            SavingsAccount,
        ):
            await s.execute(sa_delete(tbl))
        await s.commit()
    return {"ok": True}


async def deposit_monthly_transfer() -> dict:
    """Deposit the configured monthly € transfer at the start of each
    operator-local month. Idempotent via last_transfer_month."""
    month = _current_month()
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"deposited": False, "reason": "not initialized", "month": month}
        if acc.last_transfer_month == month:
            return {"deposited": False, "month": month, "amount": acc.monthly_transfer}
        amount = acc.monthly_transfer
        if amount > 0:
            acc.cash += amount
            await _log_event(s, "TRANSFER", amount, note=f"monthly transfer {month}")
        acc.last_transfer_month = month
        await s.commit()
        logger.info("Savings transfer deposited: %.2f for %s", amount, month)
        return {"deposited": amount > 0, "amount": amount, "month": month, "cash": acc.cash}


async def accrue_interest() -> dict:
    """Accrue daily interest on the savings cash for each day since the last
    accrual (catch-up after downtime), up to today (operator-local).

    TR convention: interest accrues on the END-OF-DAY balance each day at
    rate/365; the accumulated amount is paid into the account monthly. The
    tracker books accrual per day (no intra-month compounding) and moves
    the total into cash at the monthly payout — matching TR's statement.
    """
    today = _today_local()
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"accrued": False, "reason": "not initialized"}
        if acc.interest_rate <= 0:
            return {"accrued": False, "reason": "rate <= 0"}
        last = date.fromisoformat(acc.last_accrual_day) if acc.last_accrual_day else None
        if last is None:
            # First accrual after initialization: start from today, not from
            # the account creation date (the INIT seed is a point-in-time
            # balance, and accruing "since creation" would invent interest
            # for days the tracker didn't run).
            acc.last_accrual_day = today.isoformat()
            await s.commit()
            return {
                "accrued": True,
                "days": 0,
                "amount": 0.0,
                "total_accrued": acc.accrued_interest,
            }
        if last >= today:
            return {"accrued": False, "reason": "already accrued today"}
        days = (today - last).days
        daily = acc.cash * (acc.interest_rate / 100.0) / _ACCRUAL_DIVISOR
        acc.accrued_interest += daily * days
        acc.last_accrual_day = today.isoformat()
        await s.commit()
        logger.info("Savings interest accrued: %.2f for %d day(s)", daily * days, days)
        return {
            "accrued": True,
            "days": days,
            "amount": round(daily * days, 4),
            "daily_rate": daily,
            "total_accrued": round(acc.accrued_interest, 4),
        }


async def payout_interest() -> dict:
    """Move the accrued interest into cash at the start of each month
    (TR pays monthly). Idempotent via last_interest_month; skipped when the
    current month was already paid (the seed month at initialization)."""
    month = _current_month()
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"paid": False, "reason": "not initialized", "month": month}
        if acc.last_interest_month == month:
            return {"paid": False, "month": month, "accrued": acc.accrued_interest}
        amount = acc.accrued_interest
        if amount > 0:
            acc.cash += amount
            acc.accrued_interest = 0.0
            await _log_event(s, "INTEREST_PAYOUT", amount, note=f"interest payout {month}")
        acc.last_interest_month = month
        await s.commit()
        logger.info("Savings interest paid out: %.2f for %s", amount, month)
        return {
            "paid": True,
            "amount": round(amount, 2),
            "month": month,
            "cash": round(acc.cash, 2),
        }


async def _exec_sparplan(plan: SavingsPlan, price: float, day: str) -> dict | None:
    """Buy `plan.amount` € of `plan.ticker` from the savings cash at
    `price`. Returns a trade dict, or None when skipped (no price, or the
    cash would go negative — a WARN event points at the true-up)."""
    if price is None or price <= 0:
        return None
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return None
        if acc.cash < plan.amount:
            await _log_event(
                s,
                "WARN",
                0,
                ticker=plan.ticker,
                note=f"insufficient savings cash ({acc.cash:.2f} < {plan.amount:.2f} €) "
                f"for sparplan {day} — true-up the balance",
            )
            await s.commit()
            return None
        shares = plan.amount / price
        acc.cash -= plan.amount
        pos = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == plan.ticker))
        if pos:
            total = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + plan.amount) / total
            pos.shares = total
        else:
            s.add(SavingsPosition(ticker=plan.ticker, shares=shares, avg_cost=price))
        await _log_event(
            s,
            "SPARPLAN_BUY",
            -plan.amount,
            ticker=plan.ticker,
            note=f"sparplan buy {day}: {shares:.6f} sh @ {price:.4f} €",
        )
        await s.commit()
        logger.info("Savings sparplan buy: %s %.2f € @ %.4f", plan.ticker, plan.amount, price)
        mirror = await _mirror_allocate(plan.amount, source_ticker=plan.ticker)
        return {
            "ticker": plan.ticker,
            "amount": plan.amount,
            "shares": shares,
            "price": price,
            "mirror_allocated": bool(mirror),
        }


async def run_sparplans(day: date | None = None) -> dict:
    """Execute every active plan whose day_of_month has come due.

    Each plan runs at most once per calendar month: the event ledger is the
    dedupe (a SPARPLAN_BUY for the same ticker+month means it already ran).
    Catch-up after downtime: ``day`` defaults to today (operator-local), and
    a missed earlier-in-month execution still fires while the month is
    current. Day 29-31 plans run on the last day of shorter months.
    """
    today = day or _today_local()
    month = today.strftime("%Y-%m")
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"executed": 0, "reason": "not initialized"}
        plans = (await s.scalars(select(SavingsPlan).where(SavingsPlan.active == True))).all()  # noqa: E712
    if not plans:
        return {"executed": 0}
    # Already-executed (ticker, month) pairs from the ledger.
    async with Session() as s:
        rows = (
            await s.scalars(select(SavingsEvent).where(SavingsEvent.kind == "SPARPLAN_BUY"))
        ).all()
    done: dict[str, set[str]] = {}
    for r in rows:
        done.setdefault(r.ticker or "", set()).add(r.created_at.strftime("%Y-%m"))
    due: list[SavingsPlan] = []
    for p in plans:
        effective_day = min(p.day_of_month, _days_in_month(today))
        if today.day >= effective_day and month not in done.get(p.ticker, set()):
            due.append(p)
    if not due:
        return {"executed": 0}
    prices = await _price_eur_map([p.ticker for p in due])
    executed: list[dict] = []
    for p in due:
        price = prices.get(p.ticker)
        if price is None or price <= 0:
            continue  # no candle yet — plan stays due and fires next pass
        r = await _exec_sparplan(p, price, today.isoformat())
        if r:
            executed.append(r)
    return {"executed": len(executed), "trades": executed}


def _days_in_month(d: date) -> int:
    if d.month == 12:
        return 31
    return (date(d.year, d.month + 1, 1) - timedelta(days=1)).day


# ---------------------------------------------------------------------------
# Valuation + snapshot
# ---------------------------------------------------------------------------


async def valuate() -> dict:
    """Current tracker valuation in EUR (no side effects)."""
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"initialized": False}
        positions = (await s.scalars(select(SavingsPosition))).all()
        events = (await s.scalars(select(SavingsEvent))).all()
    prices = await _price_eur_map([p.ticker for p in positions]) if positions else {}
    positions_value = 0.0
    pos_list = []
    for p in positions:
        price = prices.get(p.ticker) or p.avg_cost
        value = p.shares * price
        positions_value += value
        pos_list.append(
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_cost": round(p.avg_cost, 4),
                "current_price": round(price, 4),
                "value": round(value, 2),
                "pnl_pct": round((price - p.avg_cost) / p.avg_cost * 100, 2) if p.avg_cost else 0.0,
                "buy_date": p.opened_at.strftime("%Y-%m-%d") if p.opened_at else "",
            }
        )
    # contributed = everything the operator ever put in (seed cash + transfer
    # deposits), excluding interest/saveback (gains, not contributions).
    contributed = sum(e.amount for e in events if e.kind in ("INIT", "TRANSFER") and e.amount > 0)
    # total = cash + positions + unpaid accrued interest (the last is part
    # of the account value but not yet paid into the balance).
    cash_total = acc.cash + acc.accrued_interest
    return {
        "initialized": True,
        "cash": round(acc.cash, 2),
        "accrued_interest": round(acc.accrued_interest, 2),
        "interest_rate": acc.interest_rate,
        "monthly_transfer": acc.monthly_transfer,
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(cash_total + positions_value, 2),
        "contributed": round(contributed, 2),
    }


async def take_snapshot() -> dict:
    """Record one daily equity point (idempotent per day: the UI groups by
    date and the last point of the day wins, mirroring the sims)."""
    val = await valuate()
    if not val.get("initialized"):
        return {"snapshotted": False, "reason": "not initialized"}
    async with Session() as s:
        s.add(
            SavingsSnapshot(
                cash=val["cash"],
                positions_value=val["positions_value"],
                accrued_interest=val["accrued_interest"],
                total_equity=val["total_equity"],
                contributed=val["contributed"],
            )
        )
        await s.commit()
    return {"snapshotted": True, "total_equity": val["total_equity"]}


# ---------------------------------------------------------------------------
# Operator actions
# ---------------------------------------------------------------------------


async def trueup(actual_total: float) -> dict:
    """Reconcile: enter the account total (cash incl. unpaid interest, no
    positions — TR shows them separately) as of NOW; the delta is booked as
    a TRUEUP event. Positions are untouched."""
    val = await valuate()
    if not val.get("initialized"):
        return {"ok": False, "reason": "not initialized"}
    tracked_cash = val["cash"] + val["accrued_interest"]
    delta = float(actual_total) - tracked_cash
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        # Apply the delta proportionally to paid + accrued so the split
        # stays consistent; a fresh month's payout then books cleanly.
        if abs(delta) < 0.005:
            await s.commit()
            return {"ok": True, "delta": 0.0, "cash": acc.cash}
        if acc.cash + delta < 0:
            # A true-up must not make the tracked cash negative — that would
            # mean the operator's real account holds debt TR cash cannot.
            return {"ok": False,
                    "reason": f"true-up would make cash negative "
                              f"({acc.cash:.2f} {delta:+.2f}) — check the entered total"}
        acc.cash += delta
        await _log_event(s, "TRUEUP", delta, note=f"balance true-up to {actual_total:.2f} €")
        await s.commit()
    logger.info("Savings true-up: delta %.2f", delta)
    return {"ok": True, "delta": round(delta, 2), "cash": round(val["cash"] + delta, 2)}


async def add_saveback(amount: float, ticker: str) -> dict:
    """Book a Saveback payout manually (TR invests it into a Sparplan asset
    on the 2nd of the following month; no card-spend math here — the
    operator enters what TR actually invested). Buys the position from
    nothing (Saveback is a bonus, not savings-cash money) and mirrors it."""
    amount = float(amount)
    ticker = ticker.strip().upper()
    if amount <= 0 or not ticker:
        return {"ok": False, "reason": "amount must be > 0 and ticker required"}
    prices = await _price_eur_map([ticker])
    price = prices.get(ticker)
    if price is None or price <= 0:
        return {"ok": False, "reason": f"no price for {ticker} (refresh candles?)"}
    shares = amount / price
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        pos = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == ticker))
        if pos:
            total = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + amount) / total
            pos.shares = total
        else:
            s.add(SavingsPosition(ticker=ticker, shares=shares, avg_cost=price))
        await _log_event(
            s, "SAVEBACK", amount, ticker=ticker, note=f"saveback: {shares:.6f} sh @ {price:.4f} €"
        )
        await s.commit()
    mirror = await _mirror_allocate(amount, source_ticker=ticker)
    return {
        "ok": True,
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "mirror_allocated": bool(mirror),
    }


async def set_config(
    monthly_transfer: float | None = None, interest_rate: float | None = None
) -> dict:
    """Update the persisted configuration (editable at runtime, unlike the
    env seed values)."""
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        if monthly_transfer is not None:
            acc.monthly_transfer = max(0.0, float(monthly_transfer))
        if interest_rate is not None:
            acc.interest_rate = max(0.0, float(interest_rate))
        await s.commit()
        return {
            "ok": True,
            "monthly_transfer": acc.monthly_transfer,
            "interest_rate": acc.interest_rate,
        }


async def add_plan(ticker: str, amount: float, day_of_month: int) -> dict:
    """Create (or reactivate) a Sparplan. One plan per (ticker, day)."""
    ticker = ticker.strip().upper()
    amount, day_of_month = float(amount), int(day_of_month)
    if not ticker or amount <= 0 or not 1 <= day_of_month <= 31:
        return {"ok": False, "reason": "invalid plan (ticker, amount>0, day 1..31)"}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        existing = await s.scalar(
            select(SavingsPlan).where(
                SavingsPlan.ticker == ticker, SavingsPlan.day_of_month == day_of_month
            )
        )
        if existing:
            existing.amount = amount
            existing.active = True
            await s.commit()
            return {
                "ok": True,
                "plan": {
                    "id": existing.id,
                    "ticker": ticker,
                    "amount": amount,
                    "day_of_month": day_of_month,
                    "active": True,
                },
            }
        s.add(SavingsPlan(ticker=ticker, amount=amount, day_of_month=day_of_month))
        await s.commit()
        return {
            "ok": True,
            "plan": {
                "ticker": ticker,
                "amount": amount,
                "day_of_month": day_of_month,
                "active": True,
            },
        }


async def remove_plan(plan_id: int) -> dict:
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        plan = await s.get(SavingsPlan, int(plan_id))
        if plan is None:
            return {"ok": False, "reason": "no such plan"}
        await s.delete(plan)
        await s.commit()
        return {"ok": True}


async def remove_position(ticker: str) -> dict:
    """Remove a position manually (e.g. sold in TR — the tracker does not
    sell). The current value is NOT credited back to the savings cash: the
    operator moves that money in TR, so it arrives via the next true-up."""
    ticker = ticker.strip().upper()
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        pos = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == ticker))
        if pos is None:
            return {"ok": False, "reason": "no such position"}
        await s.delete(pos)
        await _log_event(
            s,
            "WARN",
            0,
            ticker=ticker,
            note="position removed manually — proceeds arrive via true-up",
        )
        await s.commit()
        return {"ok": True}


# ---------------------------------------------------------------------------
# Scheduler entry
# ---------------------------------------------------------------------------


async def run_savings_cycle(force: bool = False) -> dict:
    """One daily savings pass: refresh candles → monthly transfer → monthly
    interest payout → daily accrual (with catch-up) → due Sparplan
    executions → snapshot.

    Idempotent: every stage is gated by a marker or ledger dedupe, so re-runs
    and post-downtime catch-ups are safe. Runs regardless of trading days —
    interest accrues on calendar days, not NYSE days, and EOD data doesn't
    move on weekends anyway (the refresh is a cheap no-op then via the
    freshness window).
    """
    if _cycle_lock.locked() and not force:
        return {"skipped": True, "reason": "already running"}
    async with _cycle_lock:
        refresh = await refresh_data()
        transfer = await deposit_monthly_transfer()
        payout = await payout_interest()
        accrual = await accrue_interest()
        sparplans = await run_sparplans()
        snap = await take_snapshot()
        return {
            "refresh": {"refreshed": len(refresh["refreshed"]),
                        "errors": len(refresh["errors"])},
            "transfer": transfer,
            "payout": payout,
            "accrual": accrual,
            "sparplans": sparplans,
            "snapshot": snap,
        }


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------


async def get_events(limit: int = 200) -> list[dict]:
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SavingsEvent)
                .order_by(SavingsEvent.created_at.desc(), SavingsEvent.id.desc())
                .limit(limit)
            )
        ).all()
    return [
        {
            "kind": r.kind,
            "amount": round(r.amount, 2),
            "ticker": r.ticker,
            "note": r.note,
            "date": r.created_at.isoformat(),
        }
        for r in rows
    ]


async def get_equity_curve(limit: int = 365) -> list[dict]:
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SavingsSnapshot)
                .order_by(SavingsSnapshot.created_at.desc(), SavingsSnapshot.id.desc())
                .limit(limit)
            )
        ).all()
    return [
        {
            "date": r.created_at.strftime("%Y-%m-%d"),
            "cash": r.cash,
            "positions_value": r.positions_value,
            "accrued_interest": r.accrued_interest,
            "total_equity": r.total_equity,
            "contributed": r.contributed,
        }
        for r in reversed(rows)
    ]


async def get_plans() -> list[dict]:
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SavingsPlan).order_by(SavingsPlan.day_of_month, SavingsPlan.ticker)
            )
        ).all()
    return [
        {
            "id": r.id,
            "ticker": r.ticker,
            "amount": r.amount,
            "day_of_month": r.day_of_month,
            "active": bool(r.active),
        }
        for r in rows
    ]


async def get_mirror_curve(limit: int = 365) -> list[dict]:
    """Mirror equity as a TRUE time series: cumulative shares as of each
    snapshot date, valued at that date's EUR close (last close <= date).

    The per-pick breakdown stored on each mirror entry makes the replay
    possible without look-ahead: entries before/at a date contribute their
    shares, priced with historical candles — so the curve shows what the
    mirror would have been worth on that day, not just today's value of
    past buys.
    """
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SavingsSnapshot)
                .order_by(SavingsSnapshot.created_at.desc(), SavingsSnapshot.id.desc())
                .limit(limit)
            )
        ).all()
        entries = (
            await s.scalars(select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))
        ).all()
    if not rows:
        return []
    # Snapshot dates (sorted) and per-entry day keys.
    dates = sorted({r.created_at.strftime("%Y-%m-%d") for r in rows})
    entry_day = {e.id: e.created_at.strftime("%Y-%m-%d") for e in entries}
    # Cumulative shares per ticker and cumulative € allocated as of each
    # date — one sweep over dates with the entries that have landed by then.
    shares_at: dict[str, dict[str, float]] = {}
    alloc_at: dict[str, float] = {}
    running: dict[str, float] = {}
    run_alloc = 0.0
    pending = sorted(
        entries, key=lambda e: (entry_day[e.id], e.id))
    pi = 0
    for d in dates:
        while pi < len(pending) and entry_day[pending[pi].id] <= d:
            e = pending[pi]
            for t, sh, _px in json.loads(e.breakdown or "[]"):
                running[t] = running.get(t, 0.0) + float(sh)
            run_alloc += e.amount
            pi += 1
        shares_at[d] = dict(running)
        alloc_at[d] = round(run_alloc, 2)
    # Historical EUR closes for every ticker in the mirror book.
    tickers = sorted({t for sh in shares_at.values() for t in sh})
    closes = await _eur_close_history(tickers) if tickers else {}
    out = []
    # Carry the last known price forward: on days without a candle for a
    # ticker (weekends/FX gaps) the position is marked at the most recent
    # prior close, not at zero.
    last_px: dict[str, float] = {}
    for d in dates:
        value = 0.0
        for t, sh in shares_at[d].items():
            px = closes.get(t, {}).get(d)
            if px is not None:
                last_px[t] = px
            elif t in last_px:
                px = last_px[t]
            else:
                px = None
            if px is not None:
                value += sh * px
        out.append({"date": d, "allocated": alloc_at[d], "value": round(value, 2)})
    return out


async def _eur_close_history(tickers: list[str]) -> dict[str, dict[str, float]]:
    """{ticker: {YYYY-MM-DD: EUR close}} for the mirror-curve replay.

    Native-EUR tickers use their own closes; others convert their local
    close to EUR using that day's EURUSD=X close (same convention as
    _price_eur_map, but as a per-date series).
    """
    from .db import Candle
    from .fundamentals import SUFFIX_FX

    wanted = set(tickers) | {"EURUSD=X"}
    pairs = sorted(
        {
            pm[0]
            for t in tickers
            if (pm := SUFFIX_FX.get(t[t.rfind("."):] if "." in t else ""))
        }
    )
    wanted |= set(pairs)
    since = datetime.now(UTC) - timedelta(days=365 * 8)
    async with Session() as s:
        rows = (
            (
                await s.scalars(
                    select(Candle)
                    .where(Candle.ticker.in_(wanted), Candle.timestamp >= since)
                    .order_by(Candle.timestamp)
                )
            )
            .all()
        )
    series: dict[str, dict[str, float]] = {}
    for r in rows:
        series.setdefault(r.ticker, {})[r.timestamp.strftime("%Y-%m-%d")] = r.close
    eur_usd = series.get("EURUSD=X", {})
    out: dict[str, dict[str, float]] = {}
    for t in tickers:
        if _is_eur_ticker(t):
            out[t] = dict(series.get(t, {}))
            continue
        pm = SUFFIX_FX.get(t[t.rfind("."):] if "." in t else "")
        local = series.get(t, {})
        conv: dict[str, float] = {}
        for d, px in local.items():
            usd_px = px
            if pm:
                r = series.get(pm[0], {}).get(d)
                if r is None:
                    continue
                usd_px = usd_px * r if pm[1] == "mul" else usd_px / r
            rate = eur_usd.get(d)
            if rate is None:
                # Fallback: the most recent prior EURUSD close (sparse FX
                # candles on weekends/holidays).
                prior = [k for k in eur_usd if k <= d]
                rate = eur_usd[prior[-1]] if prior else None
            if rate:
                conv[d] = usd_px / rate
        out[t] = conv
    return out


async def status() -> dict:
    """Full status payload for the Savings tab."""
    val = await valuate()
    out = {**val, "events": await get_events(100), "plans": await get_plans()}
    if val.get("initialized"):
        mirror = await _mirror_valuate()
        out["mirror"] = mirror
    return out
