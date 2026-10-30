"""Savings tracker engine: the operator's REAL Trade Republic account (EUR).

Tracks real money — cash with daily interest accrual (paid out monthly like
Trade Republic), Sparplan buys (virtual, from the savings cash at the cached
close), one-off manual buys (also from the savings cash), Saveback bonus
investments (new money from nothing), and a forward mirror that allocates
every € invested into positions into the current daily-core suggestion for
an honest "your picks vs daily-core's picks" comparison.

No connection to Trade Republic exists: the balance is seeded/true-upped
manually, everything else is derived. Deliberately separate from the paper
sims — ``reset-all`` and backfills never touch these tables. The scheduler
pass is idempotent per day/month so downtime catch-up just catches up.

External money is a FLOW, not a return: deposits (``TRANSFER``) and dated
withdrawals (``WITHDRAWAL``) move the account's NET contributed capital, so
they never look like investment P&L. ``flow_class_of`` is the one definition
of what each ledger kind means (the persisted ``flow_class`` column is its
write-time snapshot); ``net_contributed`` and the metric series read it, so a
legacy row with a NULL class needs no backfill. Dividends, fees, sale
settlements, splits and corrections are dated ledger events too: each writes
its row in the same transaction as the state it explains, and each carries
the operator-local ``occurred_on`` day.

A reset does not rewrite history: it appends a ``RESET`` boundary row, keeps
every event and snapshot, and clears only the current state. Everything
epoch-shaped (contributed capital, the TWR/XIRR metric series, the mirror
curve) reads from the newest boundary onward.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta
from math import isclose, isfinite
from zoneinfo import ZoneInfo

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .config import settings
from .db import (
    DailyCoreRanking,
    SavingsAccount,
    SavingsEvent,
    SavingsMirrorEntry,
    SavingsMirrorPending,
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

# One cycle lock: the scheduler pass, a manual "Run Now" and the operator
# actions from the API must not interleave writes to the savings state —
# every path does read-modify-write on the singleton account row (cash)
# and on position rows (shares/avg_cost), and two concurrent writers can
# silently lose one update. The cycle SKIPS when the lock is held (its
# stages are all idempotent, so a skipped nightly pass is recovered by the
# next one); operator endpoints instead WAIT (see account_lock).
_cycle_lock = asyncio.Lock()

# The API layer takes this lock around every mutating savings call, so
# operator clicks serialize with the nightly pass instead of racing it —
# waiting a few seconds beats silently losing an interest payout's cash
# bump. Same object as _cycle_lock (the cycle's own reentry check stays
# skip-based).
account_lock = _cycle_lock

# Interest accrues on the balance at end of each day (TR's own convention:
# daily accrual, monthly payout). actual/365 is the savings-account norm.
_ACCRUAL_DIVISOR = 365.0

# Mirror book residue thresholds. A removal reversal cancels a source's
# outstanding allocation by SUMMING negative shares/cost (the mirror entries
# are append-only, never edited), so the rebuilt projection and the replay
# curve see float residue instead of an exact zero. Below these thresholds
# the position is dust, not a holding.
_MIRROR_SHARE_EPS = 1e-12
_MIRROR_COST_EPS = 1e-9


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _current_month() -> str:
    return datetime.now(_TZ).strftime("%Y-%m")


def _today_local() -> date:
    return datetime.now(_TZ).date()


def _naive_utc(value: datetime) -> datetime:
    """The stored value as a naive UTC datetime.

    SQLite's DATETIME drops tzinfo, so a row read back is naive while a
    just-created object still carries the tz-aware datetime the default
    produced. Comparing the two would raise TypeError; this normalises both
    sides to the ONE convention the column stores (naive UTC).
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _local_date(value: datetime) -> date:
    """Operator-local calendar day of a stored timestamp.

    The ledger/curve convention is the operator's calendar (``allowance_tz``),
    not the UTC day the row happened to be written in: a point written at
    23:30 UTC is already the next day in Vienna. Stored values are naive UTC
    (SQLite drops tzinfo), so they are pinned to UTC first — assuming the
    system timezone instead would be wrong on any box not running Europe/Vienna.
    """
    return _naive_utc(value).replace(tzinfo=UTC).astimezone(_TZ).date()


def _local_day_start_utc(day: str) -> datetime:
    """Operator-local midnight of ``day``, as the naive-UTC instant stored.

    Back-dated bookkeeping rows (a split adjustment, a back-dated disposal
    reversal) anchor at the START of their operator-local day, not at
    ``00:00 UTC`` — in Europe/Vienna those differ by one or two hours and
    ``00:00 UTC`` can even fall on the previous local day. Converting the
    local midnight keeps the row's operator-local day exactly the day it
    describes (``_local_date`` maps it back).
    """
    return datetime.fromisoformat(day).replace(tzinfo=_TZ).astimezone(UTC).replace(tzinfo=None)


def _snapshot_day(row: SavingsSnapshot) -> str:
    """Operator-local day an equity point belongs to, "YYYY-MM-DD".

    ``as_of`` (the day the snapshot MEASURES) wins when recorded; legacy rows
    (NULL, written before the column existed) fall back to the created_at
    operator-local day. A malformed ``as_of`` cannot be placed in time, so it
    falls back too rather than crashing a read.
    """
    if row.as_of:
        try:
            return date.fromisoformat(row.as_of).isoformat()
        except ValueError:
            pass
    return _local_date(row.created_at).isoformat()


# ---------------------------------------------------------------------------
# Flow classification — THE one definition of what an event's money IS
# ---------------------------------------------------------------------------


def flow_class_of(kind: str, amount: float) -> str:
    """Classify a ledger event by the money FLOW it represents.

    This is the ONE classification definition. The persisted
    ``SavingsEvent.flow_class`` column is its WRITE-TIME SNAPSHOT (written by
    ``_log_event``/``_log_disposal``); reads that must classify a legacy row —
    written before the column existed, so NULL there — call this function.
    That keeps exactly one definition and needs no backfill migration.

    The table (``amount`` is the SIGNED € amount):

    ==========================  ==========  ================
    kind                        amount      flow_class
    ==========================  ==========  ================
    INIT, TRANSFER              > 0         contribution
    INIT, TRANSFER              <= 0        reconciliation
    WITHDRAWAL                  any         withdrawal
    SPARPLAN_BUY, MANUAL_BUY    any         investment
    SAVEBACK                    any         reward
    INTEREST_PAYOUT             any         interest
    TRUEUP                      any         reconciliation
    SELL                        any         sale
    DIVIDEND                    any         dividend
    FEE                         any         fee
    TAX                         any         tax
    SETTLEMENT                  any         sale_settlement
    SPLIT                       any         corporate_action
    CORRECTION                  any         correction
    WARN, RESET                 any         audit
    anything else               any         unknown
    ==========================  ==========  ================

    Only ``contribution`` and ``withdrawal`` are EXTERNAL capital (they move
    ``net_contributed``). ``dividend`` is investment income earned inside the
    account, ``fee``/``tax``/``investment``/``sale`` move money between the
    account's own forms, ``sale_settlement`` is the proceeds half of a disposal
    pair and
    ``corporate_action``/``correction``/``audit`` are records — none of them is
    fresh capital, so none of them can inflate the contributed figure.

    An ``INIT``/``TRANSFER`` with ``amount <= 0`` is a pre-withdrawal-era
    outflow CORRECTION: withdrawals have their own ``WITHDRAWAL`` kind now, so
    a non-positive INIT/TRANSFER cannot be fresh money — it is reconciliation
    drift. (The old contributed formula summed only positive INIT/TRANSFER,
    and this keeps those legacy rows out of contributed capital exactly as
    before.) An unreadable amount (NaN, non-numeric) compares ``> 0`` false,
    so it lands in reconciliation rather than being booked as fresh capital.

    Unknown kinds return ``"unknown"`` — a guess is never dressed up as
    knowledge. The kind is compared case-insensitively: the ledger stores
    uppercase kinds, and a lowercase variant means the same event, not a
    different one.
    """
    k = (kind or "").strip().upper()
    try:
        a = float(amount)
    except (TypeError, ValueError):
        a = float("nan")
    if k in ("INIT", "TRANSFER"):
        # NaN > 0 is False -> an unreadable amount is never "contribution".
        return "contribution" if a > 0 else "reconciliation"
    if k == "WITHDRAWAL":
        return "withdrawal"
    if k in ("SPARPLAN_BUY", "MANUAL_BUY"):
        return "investment"
    if k == "SAVEBACK":
        return "reward"
    if k == "INTEREST_PAYOUT":
        return "interest"
    if k == "TRUEUP":
        return "reconciliation"
    if k == "SELL":
        return "sale"
    if k == "DIVIDEND":
        return "dividend"
    if k == "FEE":
        return "fee"
    if k == "TAX":
        return "tax"
    if k == "SETTLEMENT":
        return "sale_settlement"
    if k == "SPLIT":
        return "corporate_action"
    if k == "CORRECTION":
        return "correction"
    if k in ("WARN", "RESET"):
        return "audit"
    return "unknown"


def event_flow_class(event) -> str:
    """Flow class of one persisted ledger row.

    The write-time ``flow_class`` snapshot when it was populated, the
    classifier otherwise (legacy rows written before the column existed).
    """
    if event.flow_class:
        return event.flow_class
    return flow_class_of(event.kind, event.amount)


def net_contributed(events) -> float:
    """NET contributed capital (EUR) from the event ledger.

    Contributions (INIT/TRANSFER with amount > 0) ADD; withdrawals
    (``WITHDRAWAL``, stored with the sign already negative) SUBTRACT. Every
    other class — interest, Saveback rewards, buys, sales, WARN audits and
    TRUEUP/outflow corrections (reconciliation) — is not contributed
    capital and never moves this figure.

    A deposit followed by an equal withdrawal therefore leaves contributed
    at zero: taken money out is a flow, never an investment loss. Legacy rows
    (NULL ``flow_class``) are classified by ``flow_class_of`` — no backfill,
    one definition.
    """
    total = 0.0
    for e in events:
        cls = event_flow_class(e)
        if cls == "contribution":
            total += float(e.amount)
        elif cls == "withdrawal":
            total += float(e.amount)  # already signed negative
    return total


# ---------------------------------------------------------------------------
# Epochs — the current epoch begins AFTER the last history-safe reset
# ---------------------------------------------------------------------------


async def _last_reset(s) -> SavingsEvent | None:
    """The newest RESET boundary row (None: the ledger has a single epoch)."""
    return await s.scalar(
        select(SavingsEvent)
        .where(SavingsEvent.kind == "RESET")
        .order_by(SavingsEvent.id.desc())
        .limit(1)
    )


def _epoch_events(events, reset: SavingsEvent | None) -> list[SavingsEvent]:
    """The events of the CURRENT epoch.

    Rows with an id strictly greater than the last RESET row's id. The
    boundary row itself belongs to neither epoch's money: its class is
    ``audit`` and its amount 0, so including it would change nothing — the
    strict comparison just keeps the definition unambiguous.
    """
    if reset is None:
        return list(events)
    return [e for e in events if e.id > reset.id]


def _snapshot_epoch_clause(reset: SavingsEvent | None):
    """SQL condition selecting the snapshots of the CURRENT epoch.

    Membership is CREATION ORDER, nothing else: a point belongs to the new
    epoch when it was written after the RESET row (``created_at >
    reset.created_at``), whatever day it happens to MEASURE. ``as_of`` is a
    valuation day, not an epoch marker — comparing it against the reset's
    ``occurred_on`` hid every fresh same-day point after a reset (a snapshot
    written 10 minutes after the reset measures the same day and was read as
    the old epoch). Legacy rows (NULL ``as_of``) are covered by the same
    rule; they are grouped by their created_at day on read.

    Metrics still group the surviving points by ``as_of`` AFTER this filter —
    membership first, bucketing second — so a catch-up point that measures an
    earlier day stays in the epoch it was written in.
    """
    if reset is None:
        return None
    return SavingsSnapshot.created_at > _naive_utc(reset.created_at)


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

    Quote unit comes FIRST: a London (.L) close is quoted in PENCE, so the
    raw close is scaled by ``instruments.quote_unit_factor`` (0.01) before any
    FX rate. Every other venue's factor is 1.0 and is not multiplied at all,
    so unit-1.0 prices stay bit-for-bit what they were.
    """
    from .fundamentals import SUFFIX_FX
    from .instruments import quote_unit_factor
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
        # Quote unit before FX (pence -> pounds for .L). A 1.0 venue is not
        # multiplied at all: bit-identical to the pre-fix path.
        unit = quote_unit_factor(t)
        if unit != 1.0:
            usd_px = usd_px * unit
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
    allocation is impossible right now (no fresh ranking / no prices) —
    the caller parks the amount as pending instead of dropping it."""
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


async def _pending_total() -> float:
    """Total parked mirror € across all sources."""
    async with Session() as s:
        total = await s.scalar(select(func.sum(SavingsMirrorPending.amount)))
    return float(total or 0.0)


async def _mirror_or_defer(amount: float, source_ticker: str = "") -> dict | None:
    """Allocate `amount` into the mirror, or PARK it as pending when no
    fresh ranking/prices exist. A deferral must never silently drop the €:
    with SIM_ENABLED=false (no daily-core cycles) nothing would ever mirror
    while Sparpläne keep buying, permanently understating the comparison.
    The parked row keeps the SOURCE ticker, so a later position removal
    releases it (and a flush allocates it as a source-tagged entry that
    unwinds with the position like any other)."""
    r = await _mirror_allocate(amount, source_ticker)
    if r is not None:
        return r
    async with Session() as s:
        s.add(SavingsMirrorPending(amount=amount, source_ticker=source_ticker))
        await s.commit()
    logger.info("Savings mirror: %.2f € deferred (no fresh ranking) — source %s",
                amount, source_ticker or "(seed)")
    return None


async def flush_pending_mirror() -> dict:
    """Try to allocate any pending mirror €; returns what remains.

    Flushes PER SOURCE ROW: each parked amount becomes a source-tagged
    mirror entry exactly like a live allocation, so removing a position
    later unwinds its deferred € too — the mirror book keeps per-position
    attribution even for money that arrived late.
    """
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SavingsMirrorPending).order_by(SavingsMirrorPending.id)
            )
        ).all()
    if not rows:
        return {"flushed": 0.0, "pending": 0.0}
    flushed = 0.0
    for row in rows:
        r = await _mirror_allocate(row.amount, row.source_ticker)
        if r is None:
            continue  # still no fresh ranking/prices — stays parked
        async with Session() as s:
            parked = await s.get(SavingsMirrorPending, row.id)
            if parked is not None:
                await s.delete(parked)
                await s.commit()
        flushed += r["amount"]
    remaining = await _pending_total()
    if flushed:
        logger.info("Savings mirror: flushed %.2f € of pending allocations "
                    "(%.2f € still parked)", flushed, remaining)
    return {"flushed": round(flushed, 2), "pending": round(remaining, 2)}


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
    # Mirrored € waiting for a fresh ranking (flushed by the next cycle).
    pending = round(await _pending_total(), 2)
    return {
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(positions_value, 2),
        "allocated": round(allocated, 2),
        "pending": pending,
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
    mirror book, plus the FX pairs their EUR pricing needs (EURUSD=X for
    USD/€ listings, USDCHF=X for .SW, …). Without the pair a .SW/.L/.ST
    holding cannot be priced and silently falls back to avg_cost."""
    from .fundamentals import SUFFIX_FX

    async with Session() as s:
        pos = (await s.scalars(select(SavingsPosition))).all()
        plans = (await s.scalars(select(SavingsPlan).where(SavingsPlan.active == 1))).all()
        mirror = (await s.scalars(select(SavingsMirrorPosition))).all()
    out = {p.ticker for p in pos} | {p.ticker for p in plans} | {m.ticker for m in mirror}
    out.add("EURUSD=X")
    for t in list(out):
        suffix = t[t.rfind("."):] if "." in t else ""
        if (pm := SUFFIX_FX.get(suffix)) and not _is_eur_ticker(t):
            out.add(pm[0])
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
    s,
    kind: str,
    amount: float = 0.0,
    ticker: str | None = None,
    note: str = "",
    flow_class: str | None = None,
    occurred_on: str | None = None,
    shares: float | None = None,
    cost_basis: float | None = None,
) -> None:
    """Append one ledger row.

    ``flow_class`` defaults to ``flow_class_of(kind, amount)`` — every
    existing caller's rows are classified at write time without changing a
    call site. An explicit class is only for callers that know something the
    generic table cannot (``_log_disposal`` writes ``"sale"`` for its SELL
    record).

    ``occurred_on``/``shares``/``cost_basis`` are the dated-record columns:
    the operator-local execution day and, for the bookkeeping events
    (SPLIT/CORRECTION), the NEW share count / per-share average cost. They
    stay NULL when a caller does not pass them, exactly like the rows written
    before those columns existed.
    """
    s.add(
        SavingsEvent(
            kind=kind,
            amount=amount,
            ticker=ticker,
            note=note,
            occurred_on=occurred_on,
            shares=shares,
            cost_basis=cost_basis,
            flow_class=flow_class if flow_class is not None
            else flow_class_of(kind, amount),
        )
    )


async def _log_disposal(
    s,
    ticker: str,
    *,
    shares: float,
    cost_basis: float,
    occurred_on: str,
    price: float | None = None,
    note: str | None = None,
) -> SavingsEvent:
    """Append the DISPOSAL RECORD for (part of) a position removed manually.

    Deliberately NOT settlement accounting: the tracker never sees the sale's
    proceeds (the operator sells in Trade Republic and the money arrives via
    the next true-up), so ``amount`` stays 0.0 and cash, contributed and the
    equity curve are untouched. The row records the disposal FACTS — the
    operator-local date, the DISPOSED shares, the per-share average cost at
    removal — plus the execution ``price`` only when the operator supplied
    one: an unknown price stays NULL rather than a valuation guess.
    Append-only, like the mirror book: a removal never edits or deletes an
    earlier record. The row is returned so a caller can report its id for the
    settlement link (``SavingsEvent.settlement_of``).
    """
    event = SavingsEvent(
        kind="SELL",
        amount=0.0,
        ticker=ticker,
        note=note or "position disposed manually — proceeds arrive via true-up",
        occurred_on=occurred_on,
        shares=float(shares),
        price=None if price is None else float(price),
        cost_basis=float(cost_basis),
        flow_class="sale",
    )
    s.add(event)
    return event


# ---------------------------------------------------------------------------
# Core actions (each returns a summary dict; all idempotent by markers)
# ---------------------------------------------------------------------------


async def initialize(
    cash: float,
    positions: list[dict] | None = None,
    monthly_transfer: float | None = None,
    interest_rate: float | None = None,
) -> dict:
    """Seed the tracker for the FIRST time: savings cash + current holdings.

    NON-DESTRUCTIVE by contract. An already initialized tracker is refused
    with ``{"ok": False, "reason": "already initialized"}`` BEFORE anything is
    read or written: re-initializing a live tracker used to overwrite its cash,
    its positions AND its whole event/snapshot history in one statement (the
    audit's "history rewritten by resets"). The operator's explicit restart
    path is ``reset`` — it closes the current epoch with a RESET boundary row,
    keeps every event and snapshot, clears the current state, and the setup
    form returns so this function can seed a fresh epoch.

    A fresh (uninitialized) DB behaves exactly as before: the current-state
    tables are cleared, the singleton account is created with the seeded
    cash/config, the current month is marked done for both monthly actions
    (the seeded balance must not double-count a transfer or payout) and the
    INIT row is appended. The EVENT LEDGER AND SNAPSHOT CURVE ARE NEVER
    DELETED here — a fresh DB simply has none; after a reset the preserved
    history stays and the new INIT row starts the new epoch.

    ``positions``: [{ticker, shares, avg_cost}] in EUR. The total invested
    amount is mirrored into the daily-core suggestion.
    """
    positions = positions or []
    invested = 0.0
    async with Session() as s:
        if await _account(s) is not None:
            # Refused before ANY mutation: a second initialize leaves no trace.
            return {"ok": False, "reason": "already initialized"}
        for tbl in (
            SavingsPosition,
            SavingsPlan,
            SavingsMirrorEntry,
            SavingsMirrorPending,
            SavingsMirrorPosition,
        ):
            # CURRENT STATE only. SavingsEvent/SavingsSnapshot are history:
            # clearing them here is exactly the audit finding this fixes.
            await s.execute(sa_delete(tbl))
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
        # Aggregate by ticker: two lots of the same ETF at different prices
        # are normal input, and SavingsPosition.ticker is UNIQUE — inserting
        # both would fail the whole init.
        lots: dict[str, list[float]] = {}  # ticker -> [shares, cost_total]
        for p in positions:
            ticker = str(p.get("ticker", "") or "").strip().upper()
            try:
                shares = float(p.get("shares") or 0)
                avg_cost = float(p.get("avg_cost") or 0)
            except (TypeError, ValueError):
                continue
            if not ticker or shares <= 0 or avg_cost <= 0:
                continue
            lot = lots.setdefault(ticker, [0.0, 0.0])
            lot[0] += shares
            lot[1] += shares * avg_cost
        for ticker, (shares, cost_total) in lots.items():
            s.add(SavingsPosition(
                ticker=ticker, shares=shares, avg_cost=cost_total / shares))
            invested += cost_total
        await s.commit()
    result = {
        "ok": True,
        "cash": acc.cash,
        "invested": round(invested, 2),
        "last_transfer_month": month,
    }
    if invested > 0:
        mirror = await _mirror_or_defer(invested, source_ticker="")
        result["mirror"] = mirror and {"allocated": mirror["amount"], "picks": mirror["pick_count"]}
    return result


async def reset() -> dict:
    """Start a NEW EPOCH: clear the tracker's current state, keep its history.

    The audit's acceptance item — "no history rewritten by resets". The old
    implementation deleted every row, including the event ledger and the
    equity curve, so a reset silently rewrote the account's past. Now, in ONE
    transaction:

    1. APPEND a ``RESET`` boundary row (kind RESET, amount 0, flow_class
       ``audit``, operator-local date, note "tracker reset — ledger
       preserved"). It is the epoch marker: contributed capital and the
       metrics series read only the rows AFTER the last RESET row, so the
       preserved history never contaminates the new epoch.
    2. PRESERVE every ``SavingsEvent`` and ``SavingsSnapshot`` row — the money
       ledger and its curve stay queryable bit-for-bit (``get_events`` still
       returns the full history; the raw equity curve keeps both segments).
    3. CLEAR the current state: the singleton account row goes (cash, accrued
       interest and the monthly/daily markers with it, so the tracker is
       uninitialized again and the setup form returns — reset IS the explicit
       restart path, and ``initialize`` can seed the new epoch), positions and
       plans are dropped, and the mirror book (entries, projection, parked €)
       is cleared.

    The asymmetry is deliberate and honest: the MONEY ledger survives; the
    MIRROR comparison restarts with the tracker. Mirror entries are derived
    from the previous epoch's positions and cannot be attributed to the new
    one, so carrying them would fabricate a comparison that never existed.
    """
    day = _today_local().isoformat()
    async with Session() as s:
        # Count the preserved history BEFORE the boundary row is appended, so
        # the number reports what the reset actually preserved.
        events = await s.scalar(select(func.count()).select_from(SavingsEvent))
        snaps = await s.scalar(select(func.count()).select_from(SavingsSnapshot))
        await _log_event(
            s,
            "RESET",
            0.0,
            occurred_on=day,
            note="tracker reset — ledger preserved",
        )
        for tbl in (
            SavingsPosition,
            SavingsPlan,
            SavingsMirrorEntry,
            SavingsMirrorPending,
            SavingsMirrorPosition,
            SavingsAccount,
        ):
            # Current state only: SavingsEvent/SavingsSnapshot are history.
            await s.execute(sa_delete(tbl))
        await s.commit()
    logger.info("Savings tracker reset: new epoch from %s (%s events, %s snapshots preserved)",
                day, events, snaps)
    return {
        "ok": True,
        "occurred_on": day,
        "preserved_events": int(events or 0),
        "preserved_snapshots": int(snaps or 0),
    }


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
    tracker pays out monthly and does not compound inside a month (matching
    TR's statement).

    Catch-up approximation: a multi-day gap is booked as
    ``gap x daily_rate_on_the_current_balance`` rather than a per-day replay
    of each day's end balance. Identical to TR in normal operation (cash
    only moves inside the cycle, which accrues daily); if the balance was
    true-upped mid-gap, the whole gap is re-based onto the new balance. The
    next true-up reconciles the tracked figure against TR's actual payout.
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
    cash would go negative — a WARN event points at the true-up). Marks the
    plan's month so the per-plan dedupe sees it executed."""
    if price is None or price <= 0:
        return None
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return None
        p = await s.get(SavingsPlan, plan.id)
        if p is None:
            return None
        if acc.cash < plan.amount:
            # One WARN per (plan, month, balance): the note embeds the
            # current cash, so a month of daily passes with an unchanged
            # balance logs a single row; any balance change (e.g. a
            # deposit) warns once more — a new situation, not a duplicate.
            await _warn_once(
                plan, day[:7],
                f"insufficient savings cash ({acc.cash:.2f} < {plan.amount:.2f} €) "
                f"for sparplan — true-up the balance",
            )
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
            note=f"sparplan buy {day} (day {plan.day_of_month}): "
                 f"{shares:.6f} sh @ {price:.4f} €",
        )
        p.last_executed_month = day[:7]
        await s.commit()
        logger.info("Savings sparplan buy: %s %.2f € @ %.4f", plan.ticker, plan.amount, price)
        mirror = await _mirror_or_defer(plan.amount, source_ticker=plan.ticker)
        return {
            "ticker": plan.ticker,
            "amount": plan.amount,
            "shares": shares,
            "price": price,
            "mirror_allocated": bool(mirror),
        }


async def run_sparplans(day: date | None = None) -> dict:
    """Execute every active plan whose day_of_month has come due.

    Each PLAN runs at most once per calendar month: the plan row's
    ``last_executed_month`` marker is the dedupe (the ledger cannot
    distinguish two plans for the same ticker on different days). Catch-up
    after downtime: ``day`` defaults to today (operator-local), and a plan
    that MISSED its due day this month (created before it, app down on it)
    still fires while the month is current. A plan created AFTER its due
    day first fires NEXT month — mirroring Trade Republic, where a
    Sparplan set up on the 27th with execution day 2 invests from the 2nd
    of the following month, not retroactively. Day 29-31 plans run on the
    last day of shorter months.
    """
    today = day or _today_local()
    month = today.strftime("%Y-%m")
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"executed": 0, "reason": "not initialized"}
        plans = (await s.scalars(select(SavingsPlan).where(SavingsPlan.active == True))).all()  # noqa: E712
        # created_at is stored tz-aware UTC; the OPERATOR-LOCAL calendar
        # month decides which month a buy belongs to (a buy just after
        # local midnight on the 1st is still the new month locally).
        created_local = {p.id: p.created_at.astimezone(_TZ).date() for p in plans}
    if not plans:
        return {"executed": 0}
    due: list[SavingsPlan] = []
    for p in plans:
        effective_day = min(p.day_of_month, _days_in_month(today))
        if p.last_executed_month == month:
            continue  # this plan already fired this month
        if today.day >= effective_day:
            # Due day reached — but a plan CREATED after the due day this
            # month must not fire retroactively: skip to next month.
            if created_local[p.id] > today.replace(day=effective_day):
                continue
            due.append(p)
    if not due:
        return {"executed": 0}
    prices = await _price_eur_map([p.ticker for p in due])
    executed: list[dict] = []
    deferred: list[str] = []
    for p in due:
        price = prices.get(p.ticker)
        if price is None or price <= 0:
            # No candle (or FX pair) yet — stays due, fires next pass. One
            # WARN per (plan, month) so a permanently unpriceable ticker
            # doesn't flood the ledger with a month of identical entries.
            await _warn_once(p, month,
                             f"no price for {p.ticker} — sparplan deferred; "
                             f"check the ticker or refresh prices")
            deferred.append(p.ticker)
            continue
        r = await _exec_sparplan(p, price, today.isoformat())
        if r:
            executed.append(r)
    out = {"executed": len(executed), "trades": executed}
    if deferred:
        out["deferred"] = deferred
    return out


async def _warn_once(plan: SavingsPlan, month: str, note: str) -> None:
    """Log a WARN only if no IDENTICAL note (same ticker + text) exists yet
    in the given month — a month of daily passes must not flood the ledger
    with the same warning. Callers that embed changing state (the current
    cash) in the note deliberately re-warn once per state change; callers
    with a fixed note get a strict once-per-month gate."""
    async with Session() as s:
        rows = (
            await s.scalars(
                select(SavingsEvent).where(
                    SavingsEvent.kind == "WARN",
                    SavingsEvent.ticker == plan.ticker,
                    SavingsEvent.note == note,
                )
            )
        ).all()
        already = any(r.created_at.astimezone(_TZ).strftime("%Y-%m") == month for r in rows)
        if not already:
            await _log_event(s, "WARN", 0, ticker=plan.ticker, note=note)
            await s.commit()


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
        reset = await _last_reset(s)
        # Contributed capital is an EPOCH figure: after a history-safe reset
        # only the new epoch's flows count, while the preserved pre-reset
        # rows stay in the ledger (get_events / the raw curve still show them).
        events = _epoch_events((await s.scalars(select(SavingsEvent))).all(), reset)
        pending = await s.scalar(select(func.sum(SavingsMirrorPending.amount)))
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
    # contributed = NET contributed capital: everything the operator put in
    # (seed cash + transfer deposits) MINUS dated withdrawals. Interest,
    # Saveback and buys are not contributed capital; a TRUEUP delta is
    # reconciliation (drift), not money in. The withdrawal's signed amount
    # nets out here, so taking money out can never look like a loss.
    contributed = net_contributed(events)
    # total = cash + positions + unpaid accrued interest (the last is part
    # of the account value but not yet paid into the balance).
    cash_total = acc.cash + acc.accrued_interest
    return {
        "initialized": True,
        "cash": round(acc.cash, 2),
        "accrued_interest": round(acc.accrued_interest, 2),
        "mirror_pending": round(float(pending or 0.0), 2),
        "interest_rate": acc.interest_rate,
        "monthly_transfer": acc.monthly_transfer,
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(cash_total + positions_value, 2),
        "contributed": round(contributed, 2),
    }


async def take_snapshot() -> dict:
    """Record one daily equity point (idempotent per day: the UI groups by
    date and the last point of the day wins, mirroring the sims).

    The row is stamped with ``as_of`` = the operator-local day it MEASURES, so
    a pass that runs just after local midnight (or a catch-up) attributes the
    point to the right day instead of the wall-clock UTC date it was written.
    """
    val = await valuate()
    if not val.get("initialized"):
        return {"snapshotted": False, "reason": "not initialized"}
    day = _today_local().isoformat()
    async with Session() as s:
        s.add(
            SavingsSnapshot(
                cash=val["cash"],
                positions_value=val["positions_value"],
                accrued_interest=val["accrued_interest"],
                total_equity=val["total_equity"],
                contributed=val["contributed"],
                as_of=day,
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
        # The whole delta lands in `cash`; `accrued_interest` keeps its
        # tracked value, so cash + accrued equals the entered total and
        # the next monthly payout books the remaining accrued cleanly.
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


async def add_deposit(amount: float) -> dict:
    """Book a one-off cash deposit: fresh money moved INTO the TR cash
    from outside (e.g. a bank transfer). Raises the tracked cash and logs
    a TRANSFER event — a CONTRIBUTION, so contributed/Gain stay honest
    (unlike a true-up delta, which is reconciliation drift). Cash-only:
    nothing is mirrored until a Sparplan or manual buy invests it.
    Money moved OUT is ``withdraw`` (a dated WITHDRAWAL flow); a true-up
    remains the reconciliation tool for drift."""
    amount = float(amount)
    if amount <= 0:
        return {"ok": False, "reason": "amount must be > 0"}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        acc.cash += amount
        await _log_event(s, "TRANSFER", amount, note="manual deposit")
        cash_after = round(acc.cash, 2)
        await s.commit()
    logger.info("Savings deposit: %.2f € (cash now %.2f)", amount, cash_after)
    return {"ok": True, "amount": amount, "cash": cash_after}


def _valid_iso_date(value: str) -> str | None:
    """``"YYYY-MM-DD"`` as-is, or None when it is not exactly that form.

    ``date.fromisoformat`` also accepts the compact ``"YYYYMMDD"`` form, so
    the round-trip check pins the one date marker convention the ledger and
    the disposal record use (``occurred_on``).
    """
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return value if parsed.isoformat() == value else None


def _event_day(occurred_on: str | None) -> str | None:
    """The ledger's operator-local date: ``occurred_on`` or today.

    Returns None when the caller supplied something that is not exactly
    ``"YYYY-MM-DD"`` — the caller refuses before touching any state.
    """
    if occurred_on is None:
        return _today_local().isoformat()
    return _valid_iso_date(occurred_on)


def _positive_amount(value) -> float | None:
    """A finite amount > 0, or None when the value is unusable.

    One guard for every dated event: NaN/inf, zero, negatives and
    non-numeric input all refuse, and no caller reaches a mutation.
    """
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if isfinite(amount) and amount > 0 else None


def _nonfuture_event_day(occurred_on: str | None) -> tuple[str | None, str | None]:
    """Operator-local execution/effective day that may not lie in the FUTURE.

    Returns ``(day, None)`` or ``(None, reason)``. The day must be exactly
    ``"YYYY-MM-DD"`` (``invalid date``) and must not be later than today
    (``future date``). Applies to every operator action that mutates the
    CURRENT book — cash movements (withdraw/dividend/fee/tax/settlement) AND
    position-changing records (SPLIT/CORRECTION/disposal): a future-dated
    action would change today's state while the metrics series cannot contain
    it until a snapshot covers its date, so an equity drop would read as a
    fabricated loss (and a future-dated retroactive split would invent current
    holdings). Same-day dates stay valid: they are execution dates, not
    predictions.
    """
    day = _event_day(occurred_on)
    if day is None:
        return None, "invalid date"
    if date.fromisoformat(day) > _today_local():
        return None, "future date"
    return day, None


# Ledger kinds that CHANGE the real position book. A back-dated split or
# disposal must not rewrite a book whose tracked shares changed after that
# day: without full lot replay, the CURRENT holding is not the lot that
# existed then, so applying the action retroactively would fabricate history.
_POSITION_EVENT_KINDS = ("MANUAL_BUY", "SPARPLAN_BUY", "SELL", "SPLIT", "CORRECTION")


async def _later_position_events(s, ticker: str, day: str) -> bool:
    """Does the CURRENT epoch hold a position-changing row for ``ticker``
    effectively dated after ``day``?

    The effective day is ``occurred_on`` when recorded, else the row's
    operator-local ``created_at`` day (``_event_flow_day``). A back-dated
    split or disposal refuses when one exists — this tracker has no lot
    replay, so it cannot honestly reconstruct which shares existed then; the
    operator's current book is the only truth it holds.
    """
    cutoff = date.fromisoformat(day)
    reset = await _last_reset(s)
    stmt = select(SavingsEvent).where(
        SavingsEvent.kind.in_(_POSITION_EVENT_KINDS),
        SavingsEvent.ticker == ticker,
    )
    if reset is not None:
        stmt = stmt.where(SavingsEvent.id > reset.id)
    rows = (await s.scalars(stmt)).all()
    return any(_event_flow_day(e) > cutoff for e in rows)


async def withdraw(amount: float, occurred_on: str | None = None, note: str = "") -> dict:
    """Book a dated cash withdrawal: fresh money moved OUT of the TR cash
    back to the operator's bank.

    The mirror image of ``add_deposit``: the tracked cash is decremented and
    ONE ledger event is appended — ``kind="WITHDRAWAL"``, the amount stored
    with the money-OUT sign (negative), the operator-local execution date
    (``occurred_on``, default today), and ``flow_class="withdrawal"``. That
    makes the outflow a FLOW in ``net_contributed``: it lowers contributed
    capital instead of showing up as reconciliation drift, so a withdrawal can
    never masquerade as an investment loss. Positions, their average costs,
    the picks mirror and the snapshots are untouched — this is cash leaving
    the account, not a trade.

    Parameters
    ----------
    amount:
        EUR to take out. Must be finite and > 0 (``"invalid amount"``).
    occurred_on:
        ``"YYYY-MM-DD"`` operator-local date of the movement, or None for
        today (``_today_local``). Anything else is ``"invalid date"``; a day
        after today is ``"future date"`` (see ``_nonfuture_event_day``).
    note:
        Free-text ledger note, stored verbatim.

    Refusals (each returns ``{"ok": False, "reason": ...}`` BEFORE touching
    cash or the ledger): invalid amount, invalid/future date, ``not
    initialized``, and ``insufficient cash`` when the amount exceeds the
    tracked cash. The
    cash check is a tracker-consistency guard, not a statement about the real
    account: a larger real-world gap is TRUEUP's reconciliation job (book the
    withdrawal here and let the next true-up absorb the remainder).

    The cash decrement and the event append are ONE transaction. A sub-cent
    tolerance absorbs float residue from summing decimal amounts (the same
    convention ``manual_buy`` uses), so an exactly-cash withdrawal is never
    refused by 1e-15.
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid amount"}
    if not isfinite(amount) or amount <= 0:
        return {"ok": False, "reason": "invalid amount"}
    day, reason = _nonfuture_event_day(occurred_on)
    if reason is not None:
        return {"ok": False, "reason": reason}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        if amount > acc.cash + 0.005:
            return {"ok": False, "reason": "insufficient cash"}
        acc.cash -= amount
        s.add(
            SavingsEvent(
                kind="WITHDRAWAL",
                amount=-amount,          # signed money-out, like SPARPLAN_BUY
                note=note,
                occurred_on=day,
                flow_class="withdrawal",  # == flow_class_of("WITHDRAWAL", -amount)
            )
        )
        cash_after = round(acc.cash, 2)
        await s.commit()
    logger.info("Savings withdrawal: %.2f € on %s (cash now %.2f)",
                amount, day, cash_after)
    return {"ok": True, "amount": amount, "cash": cash_after, "occurred_on": day}


async def add_dividend(
    amount: float, ticker: str, occurred_on: str | None = None, note: str = ""
) -> dict:
    """Book a cash dividend received into the tracked TR cash.

    A dividend is INVESTMENT INCOME, not fresh capital. The cash rises and ONE
    ledger event is appended — ``kind="DIVIDEND"``, ``amount=+amount`` (money
    in), the operator-local ``occurred_on``, the paying ``ticker`` and
    ``flow_class="dividend"``. Because ``net_contributed`` only counts the
    ``contribution``/``withdrawal`` classes, a dividend NEVER raises
    contributed capital: the income shows up in the return where it belongs,
    instead of looking like money the operator put in.

    Parameters
    ----------
    amount:
        EUR received. Must be finite and > 0 (``"invalid amount"``).
    ticker:
        The paying holding. Required (``"ticker required"``); stripped and
        uppercased. It need not be an open position — a dividend can arrive
        for a holding that was already removed.
    occurred_on:
        ``"YYYY-MM-DD"`` operator-local date, or None for today
        (``"invalid date"`` otherwise; a future day is ``"future date"``).
    note:
        Free-text ledger note, stored verbatim.

    Refusals (each returns ``{"ok": False, "reason": ...}`` BEFORE touching
    cash or the ledger): invalid amount, missing ticker, invalid/future date,
    ``not initialized``. The cash increment and the event append are ONE
    transaction.
    """
    amount = _positive_amount(amount)
    if amount is None:
        return {"ok": False, "reason": "invalid amount"}
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "reason": "ticker required"}
    day, reason = _nonfuture_event_day(occurred_on)
    if reason is not None:
        return {"ok": False, "reason": reason}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        acc.cash += amount
        await _log_event(
            s, "DIVIDEND", amount, ticker=ticker, note=note, occurred_on=day)
        cash_after = round(acc.cash, 2)
        await s.commit()
    logger.info("Savings dividend: %.2f € from %s on %s (cash now %.2f)",
                amount, ticker, day, cash_after)
    return {"ok": True, "amount": amount, "ticker": ticker,
            "cash": cash_after, "occurred_on": day}


async def _book_paid_cost(
    kind: str, amount: float, occurred_on: str | None = None, note: str = ""
) -> dict:
    """Shared body of the cash-cost events: a cost PAID from tracked cash.

    ``add_fee`` (TR's FX fee) and ``add_tax`` (a tax paid from the account)
    are the same money shape, so they share this one body instead of two
    drifting copies: validate the amount and the (non-future) day, decrement
    the tracked cash by ``amount`` and append ONE signed-negative ledger row
    of ``kind`` — ``flow_class_of`` classifies it (``fee`` / ``tax``), which
    is deliberately NOT an external flow (only contribution/withdrawal are),
    so the cost hits the RETURN honestly instead of netting against
    contributed capital.

    Refusals (before any mutation): invalid amount, invalid/future date,
    ``not initialized``, and ``insufficient cash`` when the amount exceeds
    the tracked cash + the sub-cent tolerance (the same guard ``withdraw``
    uses). A larger real-world gap is true-up's reconciliation job.
    """
    amount = _positive_amount(amount)
    if amount is None:
        return {"ok": False, "reason": "invalid amount"}
    day, reason = _nonfuture_event_day(occurred_on)
    if reason is not None:
        return {"ok": False, "reason": reason}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        if amount > acc.cash + 0.005:
            return {"ok": False, "reason": "insufficient cash"}
        acc.cash -= amount
        await _log_event(
            s, kind, -amount, note=note, occurred_on=day)
        cash_after = round(acc.cash, 2)
        await s.commit()
    return {"ok": True, "amount": amount, "cash": cash_after, "occurred_on": day}


async def add_fee(amount: float, occurred_on: str | None = None, note: str = "") -> dict:
    """Book a fee the operator PAID from the tracked cash (e.g. TR's FX fee).

    The money never leaves the tracker's accounting — it changes from cash
    into a cost. The cash is decremented by ``amount`` and ONE ledger event is
    appended: ``kind="FEE"``, ``amount=-amount`` (money out of the cash), the
    operator-local ``occurred_on`` and ``flow_class="fee"``.

    That class is deliberately NOT an external flow (only
    ``contribution``/``withdrawal`` are), so a fee is NOT netted against
    contributed capital: it stays inside the account and hits the RETURN
    honestly (a 1000 € account paying a 10 € fee shows −1% for the day, with
    contributed still 1000 €). Treating it as a withdrawal would let a cost
    masquerade as capital the operator took out.

    Refusals (before any mutation): invalid amount, invalid/future date,
    ``not initialized``, and ``insufficient cash`` when the amount exceeds
    the tracked cash + the sub-cent tolerance (the same guard ``withdraw``
    uses). A larger real-world gap is true-up's reconciliation job.
    """
    r = await _book_paid_cost("FEE", amount, occurred_on, note)
    if r.get("ok"):
        logger.info("Savings fee: %.2f € on %s (cash now %.2f)",
                    r["amount"], r["occurred_on"], r["cash"])
    return r


async def add_tax(amount: float, occurred_on: str | None = None, note: str = "") -> dict:
    """Book a tax the operator PAID from the tracked cash.

    The cash-cost sibling of ``add_fee``: the tracked cash drops by
    ``amount`` and ONE ledger event is appended — ``kind="TAX"``,
    ``amount=-amount`` (money out of the cash), the operator-local
    ``occurred_on`` and ``flow_class="tax"``.

    A tax is a REAL cost of the account, not fresh capital leaving: the
    ``tax`` class is not an external flow (only ``contribution``/
    ``withdrawal`` are), so it is never netted against contributed capital and
    hits the RETURN honestly — a 1000 € account paying 100 € tax shows −10%
    with contributed still 1000 €. Treating it as a withdrawal would let a
    cost masquerade as money the operator took out.

    Refusals (before any mutation): invalid amount, invalid/future date,
    ``not initialized``, and ``insufficient cash`` (same guard as
    ``add_fee``).
    """
    r = await _book_paid_cost("TAX", amount, occurred_on, note)
    if r.get("ok"):
        logger.info("Savings tax: %.2f € on %s (cash now %.2f)",
                    r["amount"], r["occurred_on"], r["cash"])
    return r


# The note every settlement row carries: attribution, so a settlement can be
# traced back to the disposal it closes even when the caller supplies no note.
_SETTLEMENT_NOTE = "settlement of the manual disposal — proceeds now in tracked cash"


def _settled_disposal_ids(s, reset: SavingsEvent | None):
    """Subquery: ids of SELL rows that a SETTLEMENT row already references."""
    stmt = select(SavingsEvent.settlement_of).where(
        SavingsEvent.settlement_of.is_not(None))
    if reset is not None:
        stmt = stmt.where(SavingsEvent.id > reset.id)
    return stmt


async def settle_sale(
    ticker: str,
    proceeds: float,
    occurred_on: str | None = None,
    note: str = "",
    *,
    settlement_of: int | None = None,
    legacy: bool = False,
) -> dict:
    """Book the settlement half of a manual disposal: proceeds are now cash.

    ``remove_position``/``_log_disposal`` record WHAT was disposed with
    ``amount=0`` (the tracker never saw the cash at that moment). This is the
    second half of that pair: the sale's proceeds have landed in the tracked
    cash, so ``acc.cash += proceeds`` and ONE ledger event is appended —
    ``kind="SETTLEMENT"``, ``amount=+proceeds``, the operator-local
    ``occurred_on``, the sold ``ticker``, ``flow_class="sale_settlement"`` and
    ``settlement_of`` = the SELL row it closes.

    ONE DISPOSAL, ONE SETTLEMENT. The link is persisted in the SAME
    transaction as the cash increment and it is UNIQUE, so a repeated
    settlement can never credit the proceeds twice (the audit's double-credit
    finding): an explicitly referenced disposal that is already settled is
    refused, and a concurrent duplicate is caught as an ``IntegrityError`` and
    rolled back. The disposal must be a SELL row of the CURRENT epoch (after
    the last RESET), carry the same ticker, and not be dated after the
    settlement day.

    ``settlement_of`` may be omitted: when EXACTLY ONE unsettled SELL row
    exists for the ticker it is linked automatically; zero or several make
    the call refuse with a reason instead of guessing. Some disposals predate
    the SELL record entirely; those may still be settled with the explicit
    ``legacy=True`` opt-in AND an explanatory note — a repeat settlement is
    never silently reclassified as legacy.

    A price is not required to settle. When the SELL row DOES carry the
    operator's execution price, ``proceeds`` may not exceed
    ``shares × price`` by more than the sub-cent tolerance (a settlement
    cannot invent a better outcome than the recorded execution).

    METRICS SEMANTICS: ``sale_settlement`` is NOT an external flow (like
    TRUEUP/reconciliation it never enters ``net_contributed``), which is what
    makes the pair read honestly. While the money is in flight the disposal
    shows as a temporary mark-down (equity drops, no flow explains it); the
    settlement then books the proceeds as a gain inside the account. Over the
    pair the two cancel into the REALIZED outcome — e.g. a 500 € holding sold
    for 600 €: equity 1000 → 500 → 1100 reads −50% then +10% cumulative, which
    is exactly +10% net, not a fabricated external contribution of 600 €.

    ``occurred_on`` is a CASH-moving date: it must be a real day and it may
    not lie in the future (``_nonfuture_event_day``).

    Refusals (before any mutation): invalid proceeds, missing ticker,
    invalid/future date, ``legacy settlement needs a note`` / ``legacy
    settlement cannot link a disposal``, ``not initialized``,
    ``no disposal to settle`` / ``already settled`` / ``no such disposal`` /
    ``not a disposal`` / ``ticker mismatch`` /
    ``ambiguous disposal — specify settlement_of`` / ``date before disposal`` /
    ``proceeds exceed disposal value``.
    """
    proceeds = _positive_amount(proceeds)
    if proceeds is None:
        return {"ok": False, "reason": "invalid proceeds"}
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "reason": "ticker required"}
    day, reason = _nonfuture_event_day(occurred_on)
    if reason is not None:
        return {"ok": False, "reason": reason}
    if legacy and settlement_of is not None:
        return {"ok": False, "reason": "legacy settlement cannot link a disposal"}
    if legacy and not (note or "").strip():
        return {"ok": False, "reason": "legacy settlement needs a note"}
    text = f"{note} — {_SETTLEMENT_NOTE}" if note else _SETTLEMENT_NOTE
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        reset = await _last_reset(s)
        disposal: SavingsEvent | None = None
        if not legacy:
            epoch_sells = (
                await s.scalars(
                    select(SavingsEvent)
                    .where(SavingsEvent.kind == "SELL", SavingsEvent.ticker == ticker)
                    .order_by(SavingsEvent.id)
                )
            ).all()
            if reset is not None:
                epoch_sells = [e for e in epoch_sells if e.id > reset.id]
            settled_ids = {int(v) for v in (
                await s.scalars(_settled_disposal_ids(s, reset))).all() if v is not None}
            if settlement_of is not None:
                try:
                    disposal = await s.get(SavingsEvent, int(settlement_of))
                except (TypeError, ValueError):
                    return {"ok": False, "reason": "no such disposal"}
                if disposal is None:
                    return {"ok": False, "reason": "no such disposal"}
                if disposal.kind != "SELL":
                    return {"ok": False, "reason": "not a disposal"}
                if reset is not None and disposal.id <= reset.id:
                    return {"ok": False, "reason": "no such disposal"}
                if (disposal.ticker or "").upper() != ticker:
                    return {"ok": False, "reason": "ticker mismatch"}
                if disposal.id in settled_ids:
                    return {"ok": False, "reason": "already settled"}
            else:
                unsettled = [e for e in epoch_sells if e.id not in settled_ids]
                if not unsettled:
                    # Zero candidates is either "never recorded" or "already
                    # settled" — the latter is reported, never silently
                    # re-booked (and never downgraded to a legacy settlement).
                    return {"ok": False, "reason": "already settled" if epoch_sells
                            else "no disposal to settle"}
                if len(unsettled) > 1:
                    return {"ok": False,
                            "reason": "ambiguous disposal — specify settlement_of"}
                disposal = unsettled[0]

            disposal_day = _event_flow_day(disposal)
            if date.fromisoformat(day) < disposal_day:
                return {"ok": False, "reason": "date before disposal"}
            if disposal.price is not None and disposal.shares is not None:
                if proceeds > float(disposal.shares) * float(disposal.price) + 0.005:
                    return {"ok": False, "reason": "proceeds exceed disposal value"}
        acc.cash += proceeds
        event = SavingsEvent(
            kind="SETTLEMENT",
            amount=proceeds,
            ticker=ticker,
            note=text,
            occurred_on=day,
            flow_class="sale_settlement",
            settlement_of=None if disposal is None else disposal.id,
        )
        s.add(event)
        try:
            await s.flush()  # UNIQUE(settlement_of) must fire inside the tx
            cash_after = round(acc.cash, 2)
            await s.commit()
        except IntegrityError:
            await s.rollback()
            return {"ok": False, "reason": "already settled"}
    logger.info("Savings sale settlement: %.2f € for %s on %s (cash now %.2f)",
                proceeds, ticker, day, cash_after)
    return {"ok": True, "ticker": ticker, "proceeds": proceeds,
            "cash": cash_after, "occurred_on": day,
            "settlement_of": None if disposal is None else disposal.id}


async def record_split(
    ticker: str, ratio: float, occurred_on: str | None = None, note: str = ""
) -> dict:
    """Record a share split on a REAL holding (a corporate action).

    The real position is adjusted in place — ``shares *= ratio`` and
    ``avg_cost /= ratio`` — which leaves the TOTAL cost basis exactly
    unchanged (the operator's money did not change; only the number of pieces
    did). ONE ledger event is appended: ``kind="SPLIT"``, ``amount=0``, the
    operator-local ``occurred_on``, ``shares`` = the NEW share count,
    ``cost_basis`` = the NEW per-share average cost, ``flow_class=
    "corporate_action"``, and a note documenting the ratio.

    ``ratio`` must be finite and > 0 and must NOT be exactly 1.0: a 1:1
    "split" is a no-op, and the append-only ledger should not fill up with
    rows that changed nothing (``"no-op split"``).

    MIRROR SIDE (the subtle part). The mirror comparison must scale with the
    same corporate action, but it is an append-only book: nothing is edited.
    The basis is the outstanding mirror shares of ``ticker`` AS OF the split's
    ``occurred_on`` day — entries whose operator-local day is at/before the
    split — grouped PER FUNDING SOURCE (an entry's ``source_ticker``, including
    the empty seed source). A corporate action rescales the SECURITY, not the
    cash-flow source: every source that funded the pick must be rescaled with
    it, and the old whole-position aggregate attributed the delta to the split
    ticker alone, which later inflated/released the wrong source.

    ONE adjustment entry is appended per source with non-zero net shares —
    ``amount=0.0`` (no € moved), ``source_ticker`` = that funding source,
    ``created_at`` = the START of the split's operator-local day converted to
    UTC (so the curve's dates BEFORE the split keep replaying pre-split shares
    against pre-split closes), and ``breakdown=[[ticker,
    source_net_shares * (ratio - 1), 0.0]]``. Entries dated AFTER the split
    day are NOT multiplied — a buy after the split is already in post-split
    terms.

    * the 0-€ breakdown adds the split DELTA to the replayed cumulative
      shares — dates after the split value the scaled share count at the
      (halved) close;
    * the 0 price adds no cost, so each source's mirrored cost pool is
      unchanged and its projected average cost per mirror share halves,
      exactly like the real position;
    * because the entry carries the funding source, releasing that source
      later sums it into the reversal (original buy + split delta fully
      reversed — no phantom residue, no over-release of other sources).

    With no outstanding mirror shares for the ticker as of the split day,
    nothing is appended (there is no mirrored quantity to scale).
    ``_release_mirror_source`` and ``_rebuild_mirror_positions`` are used
    unchanged — each adjustment is just one more signed entry; the projection
    is then rebuilt to reflect them.

    ``ratio`` must also keep the new per-share average cost positive: a
    pathological ratio that underflows ``avg_cost /= ratio`` to zero (or a
    denormal that overflows the stored numbers) is refused as
    ``"invalid ratio"`` BEFORE the assignment, leaving no trace.

    CHRONOLOGY. A BACK-DATED split (``occurred_on`` before today) is only
    accepted when the current epoch holds no position-changing ledger row for
    that ticker effectively dated after the split day (MANUAL_BUY,
    SPARPLAN_BUY, SELL, SPLIT, CORRECTION). There is no lot replay here: after
    a later buy/sale the current holding is not the lot that existed on the
    split day, so scaling it retroactively would invent current holdings
    (real 6 → 12 while the mirror book has only 10). The refusal is
    ``"later position changes prevent backdated split"`` and leaves no trace.
    A seeded/untracked holding with no later tracked changes may be adjusted
    (the operator's unadjusted-share assertion), and a same-day split — the
    normal path — is unaffected.

    Dates in the FUTURE are refused (``"future date"``): a split mutates the
    CURRENT real position, so it is not a harmless forward-looking record.

    Refusals (before any mutation): missing ticker, invalid ratio, no-op
    split, invalid/future date, ``not initialized``, ``no such position``,
    ``later position changes prevent backdated split``.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "reason": "ticker required"}
    ratio = _positive_amount(ratio)
    if ratio is None:
        return {"ok": False, "reason": "invalid ratio"}
    if ratio == 1.0:
        return {"ok": False, "reason": "no-op split"}
    day, reason = _nonfuture_event_day(occurred_on)
    if reason is not None:
        return {"ok": False, "reason": reason}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        pos = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == ticker))
        if pos is None:
            return {"ok": False, "reason": "no such position"}
        if await _later_position_events(s, ticker, day):
            return {"ok": False,
                    "reason": "later position changes prevent backdated split"}
        old_shares, old_avg = float(pos.shares), float(pos.avg_cost)
        new_shares = old_shares * ratio
        new_avg = old_avg / ratio
        if not (isfinite(new_shares) and isfinite(new_avg)) or new_avg <= 0.0:
            # A pathological ratio (e.g. a denormal) would overflow the stored
            # numbers or underflow the average cost to zero: refuse BEFORE the
            # assignment, leaving no trace.
            return {"ok": False, "reason": "invalid ratio"}
        pos.shares = new_shares
        pos.avg_cost = new_avg
        detail = (
            f"{ratio:g}-for-1 split: {old_shares:.6f} sh @ {old_avg:.4f} € -> "
            f"{new_shares:.6f} sh @ {new_avg:.4f} €"
        )
        text = f"{note} — {detail}" if note else detail
        # ONE transaction: the real position, its SPLIT record and the mirror
        # adjustments/rebuild commit together (or not at all).
        await _log_event(
            s, "SPLIT", 0.0, ticker=ticker, note=text, occurred_on=day,
            shares=new_shares, cost_basis=new_avg)
        # Mirror side: rescale the SECURITY per funding source, on the shares
        # outstanding AS OF the split day. Later buys are already post-split.
        split_day = date.fromisoformat(day)
        entries = (
            await s.scalars(
                select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))
        ).all()
        net_by_source: dict[str, float] = {}
        for e in entries:
            if _local_date(e.created_at) > split_day:
                continue
            for t, sh, _px in json.loads(e.breakdown or "[]"):
                if t != ticker:
                    continue
                src = e.source_ticker or ""
                net_by_source[src] = net_by_source.get(src, 0.0) + float(sh)
        split_at = _local_day_start_utc(day)
        adjustments = 0
        for src, source_shares in net_by_source.items():
            delta_shares = source_shares * (ratio - 1.0)
            if abs(delta_shares) <= _MIRROR_SHARE_EPS:
                continue
            s.add(SavingsMirrorEntry(
                amount=0.0,  # no € moved: a split changes pieces, not value
                source_ticker=src,  # released with that source, not orphaned
                breakdown=json.dumps([[ticker, delta_shares, 0.0]]),
                # Dated at the split's local-day start so earlier curve dates
                # replay exactly and the entry's local day IS the split day.
                created_at=split_at,
            ))
            adjustments += 1
        mirror_adjusted = adjustments > 0
        if mirror_adjusted:
            await s.flush()  # the rebuild must see the appended entries
            await _rebuild_mirror_positions(s)
        await s.commit()
    logger.info("Savings split: %s %g-for-1 (%s sh @ %.4f €, mirror adjusted=%s)",
                ticker, ratio, new_shares, new_avg, mirror_adjusted)
    return {
        "ok": True,
        "ticker": ticker,
        "ratio": ratio,
        "shares": new_shares,
        "avg_cost": new_avg,
        "occurred_on": day,
        "mirror_adjusted": mirror_adjusted,
    }


async def correct_position(
    ticker: str,
    shares: float | None = None,
    avg_cost: float | None = None,
    occurred_on: str | None = None,
    note: str = "",
) -> dict:
    """Record a DATED bookkeeping correction of one real position.

    The operator's real TR book is the source of truth; when the tracker's
    copy drifted (a missed lot, a wrong cost entered at setup), this fixes the
    supplied fields and appends ONE ledger event — ``kind="CORRECTION"``,
    ``amount=0``, the operator-local ``occurred_on``, ``shares``/``cost_basis``
    = the NEW values (whichever was corrected; the other stays NULL) and
    ``flow_class="correction"``. No cash moves and contributed capital is
    untouched: a correction fixes a number, it is not a flow.

    The note is AUDITABLE: the old values are recorded in the text ("was X sh
    @ Y"), so the change is explainable later even though the position row
    itself is overwritten in place. A caller note is prefixed to the
    synthesized description; with no note the synthesized text is used.

    The MIRROR is deliberately untouched: a correction repairs the real book's
    bookkeeping, it does not describe a corporate action or a trade, so it
    must not silently rescale the comparison. A split is ``record_split``.

    A correction MUTATES the current real position, so a FUTURE
    ``occurred_on`` is refused (``"future date"``) — it is not a record-only
    entry. A same-day date is the normal execution date.

    Refusals (before any mutation): missing ticker, nothing to correct
    (neither field supplied), invalid shares, invalid avg cost, invalid/future
    date, ``not initialized``, ``no such position``.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "reason": "ticker required"}
    if shares is None and avg_cost is None:
        return {"ok": False, "reason": "nothing to correct"}
    new_shares: float | None = None
    if shares is not None:
        new_shares = _positive_amount(shares)
        if new_shares is None:
            return {"ok": False, "reason": "invalid shares"}
    new_avg: float | None = None
    if avg_cost is not None:
        new_avg = _positive_amount(avg_cost)
        if new_avg is None:
            return {"ok": False, "reason": "invalid avg cost"}
    day, reason = _nonfuture_event_day(occurred_on)
    if reason is not None:
        return {"ok": False, "reason": reason}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        pos = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == ticker))
        if pos is None:
            return {"ok": False, "reason": "no such position"}
        old_desc = f"was {pos.shares:.6f} sh @ {pos.avg_cost:.4f} €"
        changes: list[str] = []
        if new_shares is not None:
            pos.shares = new_shares
            changes.append(f"shares to {new_shares:.6f}")
        else:
            new_shares = float(pos.shares)
        if new_avg is not None:
            pos.avg_cost = new_avg
            changes.append(f"avg cost to {new_avg:.4f} €")
        else:
            new_avg = float(pos.avg_cost)
        detail = f"corrected {' and '.join(changes)} ({old_desc})"
        text = f"{note} — {detail}" if note else detail
        await _log_event(
            s, "CORRECTION", 0.0, ticker=ticker, note=text, occurred_on=day,
            shares=new_shares if shares is not None else None,
            cost_basis=new_avg if avg_cost is not None else None)
        await s.commit()
    logger.info("Savings position corrected: %s now %.6f sh @ %.4f € on %s",
                ticker, new_shares, new_avg, day)
    return {
        "ok": True,
        "ticker": ticker,
        "shares": new_shares,
        "avg_cost": new_avg,
        "occurred_on": day,
    }


async def add_saveback(amount: float, ticker: str) -> dict:
    """Book a Saveback payout manually (TR invests it into a Sparplan asset
    on the 2nd of the following month; no card-spend math here — the
    operator enters what TR actually invested). Buys the position from
    nothing — Saveback is a bonus, money TR gives you on top, NOT savings
    cash (a buy paid from your cash is manual_buy). Mirrors it."""
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
    mirror = await _mirror_or_defer(amount, source_ticker=ticker)
    return {
        "ok": True,
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "mirror_allocated": bool(mirror),
    }


async def manual_buy(amount: float, ticker: str) -> dict:
    """Record a one-off asset purchase paid from the tracked savings cash
    (e.g. ETF shares bought directly in TR, outside any Sparplan) — the
    manual counterpart to a Sparplan execution: cash is deducted, a
    MANUAL_BUY ledger event is booked, and the € is mirrored."""
    amount = float(amount)
    ticker = ticker.strip().upper()
    if amount <= 0 or not ticker:
        return {"ok": False, "reason": "amount must be > 0 and ticker required"}
    prices = await _price_eur_map([ticker])
    price = prices.get(ticker)
    if price is None or price <= 0:
        return {"ok": False, "reason": f"no price for {ticker} (refresh candles?)"}
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        if acc.cash + 0.005 < amount:
            return {
                "ok": False,
                "reason": f"insufficient savings cash ({acc.cash:.2f} < {amount:.2f} €) "
                          f"— true-up the balance first",
            }
        shares = amount / price
        acc.cash -= amount
        pos = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == ticker))
        if pos:
            total = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + amount) / total
            pos.shares = total
        else:
            s.add(SavingsPosition(ticker=ticker, shares=shares, avg_cost=price))
        await _log_event(
            s, "MANUAL_BUY", -amount, ticker=ticker,
            note=f"manual buy: {shares:.6f} sh @ {price:.4f} €"
        )
        cash_after = round(acc.cash, 2)
        await s.commit()
    mirror = await _mirror_or_defer(amount, source_ticker=ticker)
    return {
        "ok": True,
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "cash": cash_after,
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


async def _rebuild_mirror_positions(s) -> None:
    """Rebuild the CURRENT mirror holding projection from every signed
    entry breakdown (forward buys plus removal reversals).

    The projection is derived state, so it is safe to rewrite; the entries
    that feed it are append-only. Shares/cost are summed across ALL sources
    in entry order (ids), which reproduces the incrementally-built weighted
    average cost. Fully released picks cancel to float residue and are
    dropped; rows whose remaining cost basis is not positive are dropped too.
    """
    entries = (
        await s.scalars(select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))
    ).all()
    shares: dict[str, float] = {}
    cost: dict[str, float] = {}
    for e in entries:
        for t, sh, px in json.loads(e.breakdown or "[]"):
            shares[t] = shares.get(t, 0.0) + float(sh)
            cost[t] = cost.get(t, 0.0) + float(sh) * float(px)
    for p in (await s.scalars(select(SavingsMirrorPosition))).all():
        await s.delete(p)
    await s.flush()  # frees rows before re-inserting rebuilt ones
    for t, sh in shares.items():
        if abs(sh) <= _MIRROR_SHARE_EPS:
            continue  # released to zero (float residue)
        c = cost.get(t, 0.0)
        if c <= _MIRROR_COST_EPS:
            continue  # no valid cost basis left to report
        s.add(SavingsMirrorPosition(ticker=t, shares=sh, avg_cost=c / sh))


async def _release_mirror_source(
    s,
    source_ticker: str,
    *,
    fraction: float = 1.0,
    created_at: datetime | None = None,
) -> float:
    """Release what `source_ticker` still has OUTSTANDING in the mirror book,
    inside the caller's session.

    The removal is APPEND-ONLY: the net € amount and per-pick shares/cost of
    the source — summed over all its signed entries, so previous reversals
    are included — are cancelled by ONE new negative entry. No historical
    mirror entry is ever updated or deleted, which keeps the mirror curve's
    past dates bit-for-bit identical. A repeated removal sees a net of zero
    and appends nothing; a remove → rebuy → remove cycle only cancels the
    currently outstanding allocation.

    ``fraction`` is the share of the source's outstanding book being released:
    1.0 (the default) is a full removal and is bit-for-bit the legacy path;
    a PARTIAL disposal passes ``disposed / held`` so the reversal — € amount,
    per-pick shares and per-pick cost — is proportional and the remainder
    stays allocated. Still-parked (unallocated) € for the source is scaled by
    the same fraction (or dropped entirely on a full release): it is
    projection/pending state, not a booked entry. Legacy entries without a
    breakdown cancel their amount but contribute no shares (none are
    invented). Seed mirror € (source "") has no ticker link and is never
    reversed. Returns the net € released.

    ``created_at`` dates the reversal entry (a back-dated disposal anchors it
    at that operator-local day's start); None keeps the historical "now".
    """
    if not source_ticker:
        return 0.0
    if not (0.0 < fraction <= 1.0):
        return 0.0
    full = fraction >= 1.0
    parked = (
        await s.scalars(
            select(SavingsMirrorPending).where(
                SavingsMirrorPending.source_ticker == source_ticker
            )
        )
    ).all()
    entries = (
        await s.scalars(select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))
    ).all()
    net_amount = 0.0
    pick_shares: dict[str, float] = {}
    pick_cost: dict[str, float] = {}
    for e in entries:
        if e.source_ticker != source_ticker:
            continue
        net_amount += e.amount
        for t, sh, px in json.loads(e.breakdown or "[]"):
            sh, px = float(sh), float(px)
            pick_shares[t] = pick_shares.get(t, 0.0) + sh
            pick_cost[t] = pick_cost.get(t, 0.0) + sh * px
    reversal = [
        [t, -(sh * fraction), pick_cost[t] / sh]
        for t, sh in pick_shares.items()
        if abs(sh) > _MIRROR_SHARE_EPS
    ]
    released_amount = net_amount * fraction
    for p in parked:
        if full:
            await s.delete(p)
            continue
        remaining = float(p.amount) * (1.0 - fraction)
        if remaining <= _MIRROR_COST_EPS:
            await s.delete(p)
        else:
            p.amount = remaining
    if not reversal and abs(released_amount) <= _MIRROR_COST_EPS:
        # Nothing placed is outstanding: only parked € was released, the
        # current projection is unchanged and no entry is fabricated.
        return 0.0
    s.add(
        SavingsMirrorEntry(
            amount=-released_amount if abs(released_amount) > _MIRROR_COST_EPS else 0.0,
            source_ticker=source_ticker,
            breakdown=json.dumps(reversal),
            created_at=created_at if created_at is not None else _utcnow(),
        )
    )
    await s.flush()  # the rebuild must see the appended reversal
    await _rebuild_mirror_positions(s)
    return released_amount


async def _unwind_mirror(
    source_ticker: str,
    session=None,
    *,
    fraction: float = 1.0,
    created_at: datetime | None = None,
) -> float:
    """Release the mirror € that `source_ticker`'s buys still have
    outstanding (append-only — see _release_mirror_source).

    ``remove_position`` passes its own session so the reversal/projection
    rebuild commits atomically with the real position removal. Called on its
    own it opens and commits its own transaction (kept for compatibility).
    ``fraction``/``created_at`` are forwarded to ``_release_mirror_source``
    (a partial, possibly back-dated disposal). Seed mirror € (source "") is
    never reversed.
    """
    if not source_ticker:
        return 0.0
    if session is not None:
        return await _release_mirror_source(
            session, source_ticker, fraction=fraction, created_at=created_at)
    async with Session() as s:
        released = await _release_mirror_source(
            s, source_ticker, fraction=fraction, created_at=created_at)
        await s.commit()
    if released:
        logger.info("Savings mirror: released %.2f € outstanding (source %s removed)",
                    released, source_ticker)
    return released


async def remove_position(
    ticker: str,
    price: float | None = None,
    *,
    shares: float | None = None,
    occurred_on: str | None = None,
) -> dict:
    """Dispose of (part of) a position manually — a sale in Trade Republic.

    The current value is NOT credited back to the savings cash: the operator
    moves that money in TR, so it arrives via the next true-up (and the
    settlement half is booked explicitly with ``settle_sale``).

    A FULL disposal (``shares`` omitted) behaves exactly as before: the
    position row is dropped, a WARN audit row and a SELL disposal record are
    appended, and the source's whole mirror book is released. A PARTIAL
    disposal (``shares`` supplied) records the EXACT disposed quantity and
    keeps the rest: ``SavingsPosition.shares`` shrinks by the disposed amount
    and its ``avg_cost`` is unchanged (the remaining lot still costs what it
    cost). The SELL row carries the disposed shares, the PRIOR per-share
    average cost, the operator-local ``occurred_on`` (default today) and the
    execution ``price`` only when supplied — amount=0.0, because it is a
    record of facts, not a cash movement, so cash, contributed and the equity
    curve stay exactly as before. The ledger history is never edited.

    The mirror side is released PROPORTIONALLY to the disposed fraction
    (``disposed / held``): the source's outstanding € amount, per-pick shares
    and per-pick cost are cancelled by a negative entry, and its parked
    (unallocated) € is scaled by the same fraction. A full release stays
    exactly the existing path (fraction 1.0). A back-dated disposal dates its
    mirror reversal at the START of that operator-local day converted to UTC;
    a disposal without an explicit date keeps the historical "now" timestamp.

    CHRONOLOGY (back-dated disposals only). A disposal dated at/after today is
    always the operator reporting the current book. A HISTORICAL date is only
    accepted when the current epoch holds no position-changing row for that
    ticker effectively dated after it
    (``"later position changes prevent backdated disposal"``) and no POSITIVE
    mirror allocation for that funding source dated after it
    (``"later mirror allocations prevent backdated disposal"``). Otherwise the
    appended negative reversal would sort BEFORE the original buys in the
    mirror curve and fabricate a short position that never existed; there is
    no lot replay here, so the honest answer is a refusal with no mutation.

    Dates in the FUTURE are refused (``"future date"``): a disposal mutates the
    CURRENT real position. Same-day dates remain valid.

    The position row, the WARN audit event, the SELL disposal record, the
    source's parked mirror €, the append-only mirror reversal and the rebuilt
    mirror projection are ONE transaction: a failure anywhere rolls all of it
    back, so the real positions, the disposal ledger and the comparison book
    can never split apart.

    Returns ``{"ok": True}`` for a legacy full removal (unchanged contract).
    A quantity-specified disposal additionally reports ``shares`` (disposed),
    ``remaining_shares``, ``avg_cost``, ``price``, ``occurred_on``,
    ``disposal_event_id`` (the SELL row, for ``settle_sale(settlement_of=...)``)
    and ``mirror_released``.

    The quantity check is EXACT up to floating-point comparison precision
    (``math.isclose`` rel/abs 1e-12, which only absorbs representation noise):
    an ordinary oversell is ``insufficient shares``, never silently snapped to
    the held amount.

    Refusals (before any mutation): invalid price, invalid shares (NaN/inf/
    non-numeric/<= 0), ``insufficient shares`` (more than held), invalid/future
    date, ``later position changes prevent backdated disposal`` /
    ``later mirror allocations prevent backdated disposal``,
    ``not initialized``, ``no such position``.
    """
    ticker = ticker.strip().upper()
    if price is not None:
        # Validated FIRST: a rejected removal must leave no trace at all.
        try:
            price = float(price)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "invalid price"}
        if not isfinite(price) or price <= 0:
            return {"ok": False, "reason": "invalid price"}
    quantity_disposal = shares is not None
    if quantity_disposal:
        shares = _positive_amount(shares)
        if shares is None:
            return {"ok": False, "reason": "invalid shares"}
    day, reason = _nonfuture_event_day(occurred_on)
    if reason is not None:
        return {"ok": False, "reason": reason}
    disposal_event_id: int | None = None
    async with Session() as s:
        acc = await _account(s)
        if acc is None:
            return {"ok": False, "reason": "not initialized"}
        pos = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == ticker))
        if pos is None:
            return {"ok": False, "reason": "no such position"}
        if occurred_on is not None:
            # A HISTORICAL disposal must not rewrite a book that changed later
            # (no lot replay), nor insert its reversal before the source's own
            # buys. Same-day/undated removals cannot have later dates.
            if await _later_position_events(s, ticker, day):
                return {"ok": False,
                        "reason": "later position changes prevent backdated disposal"}
            later_alloc = (
                await s.scalars(
                    select(SavingsMirrorEntry).where(
                        SavingsMirrorEntry.source_ticker == ticker,
                        SavingsMirrorEntry.amount > 0,
                    )
                )
            ).all()
            cutoff = date.fromisoformat(day)
            if any(_local_date(e.created_at) > cutoff for e in later_alloc):
                return {"ok": False,
                        "reason": "later mirror allocations prevent backdated disposal"}
        held, cost_basis = float(pos.shares), float(pos.avg_cost)
        disposed = held if shares is None else float(shares)
        # EXACT quantities: only floating-point REPRESENTATION noise may snap
        # to the held amount (rel/abs 1e-12). An ordinary oversell refuses —
        # silently clamping it would fabricate a disposal the operator never
        # entered.
        if disposed > held and not isclose(disposed, held, rel_tol=1e-12, abs_tol=1e-12):
            return {"ok": False, "reason": "insufficient shares"}
        if isclose(disposed, held, rel_tol=1e-12, abs_tol=1e-12):
            disposed = held
            full = True
        else:
            full = False
        fraction = 1.0 if full else disposed / held
        if full:
            await s.delete(pos)
        else:
            pos.shares = held - disposed
        await _log_event(
            s,
            "WARN",
            0,
            ticker=ticker,
            note=("position removed manually — proceeds arrive via true-up" if full
                  else "position partially removed manually — proceeds arrive via true-up"),
        )
        disposal = await _log_disposal(
            s,
            ticker,
            shares=disposed,
            cost_basis=cost_basis,
            occurred_on=day,
            price=price,
            note=None if full else
            "position partially disposed manually — proceeds arrive via true-up",
        )
        # A back-dated disposal anchors its mirror reversal at that
        # operator-local day's start; an undated (legacy) removal keeps the
        # historical "now" stamp.
        reversal_at = _local_day_start_utc(day) if occurred_on is not None else None
        released = await _unwind_mirror(
            ticker, s, fraction=fraction, created_at=reversal_at)
        if quantity_disposal:
            await s.flush()  # assign the SELL id before the commit
            disposal_event_id = disposal.id
        await s.commit()
    logger.info("Savings position %s: %s (%.6f sh disposed, mirror €%.2f released)",
                "removed" if full else "partially removed", ticker, disposed, released)
    if not quantity_disposal:
        return {"ok": True}
    return {
        "ok": True,
        "ticker": ticker,
        "shares": disposed,
        "remaining_shares": 0.0 if full else held - disposed,
        "avg_cost": cost_basis,
        "price": price,
        "occurred_on": day,
        "disposal_event_id": disposal_event_id,
        "mirror_released": released,
    }


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
        # Deferred mirror € (no fresh ranking when it was invested) lands in
        # the mirror as soon as a ranking is available again.
        mirror = await flush_pending_mirror()
        snap = await take_snapshot()
        return {
            "refresh": {"refreshed": len(refresh["refreshed"]),
                        "errors": len(refresh["errors"])},
            "transfer": transfer,
            "payout": payout,
            "accrual": accrual,
            "sparplans": sparplans,
            "mirror": mirror,
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
    """Savings equity curve (oldest-first), EUR — the FULL preserved history.

    This is the raw curve, not a metrics input: every snapshot ever written is
    still here, across resets (a history-safe reset never deletes a point, so
    the curve keeps its pre-reset segment). The metrics endpoints read the
    epoch-scoped view (``metric_series``) instead. Each point's date is its
    ``as_of`` day when recorded, else the created_at operator-local day.
    """
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
            "date": _snapshot_day(r),
            "cash": r.cash,
            "positions_value": r.positions_value,
            "accrued_interest": r.accrued_interest,
            "total_equity": r.total_equity,
            "contributed": r.contributed,
        }
        for r in reversed(rows)
    ]


def _event_flow_day(event: SavingsEvent) -> date:
    """Operator-local calendar day of one external flow.

    ``occurred_on`` (the operator's execution date) wins when recorded;
    otherwise the row's ``created_at`` is converted from its stored UTC value
    to the operator's timezone — the same convention the scheduler uses for
    plan months. A malformed ``occurred_on`` marker cannot be placed in time,
    so it falls back to the ``created_at`` day rather than crashing a read.
    """
    if event.occurred_on:
        try:
            return date.fromisoformat(event.occurred_on)
        except ValueError:
            pass
    return _local_date(event.created_at)


async def metric_series(limit: int = 12000) -> dict:
    """Inputs for the savings TWR/XIRR endpoints: equity points + dated flows.

    ``points``
        Per-day snapshot points ``{date, total_equity, contributed}``,
        oldest-first; intraday duplicates collapse to that day's LAST
        snapshot (the chart rule). A point's day is its ``as_of`` when
        recorded (the day it MEASURES — a pass just after local midnight
        attributes correctly), else the created_at operator-local day for
        legacy rows. ``contributed`` here is the CUMULATIVE NET contributed
        capital as of that day, derived from the event ledger's dated
        contributions and withdrawals — the series ``unitized_nav`` consumes,
        where a deposit/withdrawal is a flow and not a return. ``limit``
        truncates the equity window (the newest ``limit`` snapshots), exactly
        like ``/api/savings/equity``.

    ``flows``
        The ledger's dated external flows ``{date, amount}``, signed
        (deposits positive, withdrawals negative), oldest-first. The date is
        ``occurred_on`` when recorded, else the operator-local ``created_at``
        day, so a back-dated withdrawal is discounted on the day it actually
        happened. ``app.performance.dated_external_flows`` derives the
        amounts (zero/float-noise flows omitted, undated rows skipped).
        Flows dated after the last snapshot are excluded: the terminal value
        is that snapshot's equity, so a flow it cannot contain would only
        distort the rate until the next snapshot covers it.

    EPOCH SCOPE. Both sides are the CURRENT EPOCH — the rows after the newest
    ``RESET`` boundary: events by id strictly greater than that row's id,
    snapshots by CREATION ORDER (``created_at`` strictly after the boundary's).
    ``as_of`` is a valuation day, not an epoch marker: a point written after
    the reset belongs to the new epoch even when it measures the reset day
    itself, and the surviving points are grouped by ``as_of`` AFTER that
    membership filter. A history-safe reset preserves the pre-reset ledger
    and curve, and this read must not mix the old epoch's flows/equity into
    the new one (an INIT of 0 after a reset would otherwise read as a total
    loss against the preserved points). The preserved rows are still visible
    through ``get_events`` and ``get_equity_curve``.

    Reads only — this never writes, and it never mutates the snapshot's
    persisted ``contributed`` column (legacy positive-only rows stay as
    they are; the NET series is derived on read, no backfill).
    """
    from .performance import dated_external_flows

    async with Session() as s:
        reset = await _last_reset(s)
        snap_stmt = (
            select(SavingsSnapshot)
            .order_by(SavingsSnapshot.created_at.desc(), SavingsSnapshot.id.desc())
            .limit(limit)
        )
        clause = _snapshot_epoch_clause(reset)
        if clause is not None:
            snap_stmt = snap_stmt.where(clause)
        snaps = (await s.scalars(snap_stmt)).all()
        ev_stmt = select(SavingsEvent).order_by(SavingsEvent.created_at, SavingsEvent.id)
        if reset is not None:
            ev_stmt = ev_stmt.where(SavingsEvent.id > reset.id)
        events = (await s.scalars(ev_stmt)).all()

    by_day: dict[str, SavingsSnapshot] = {}
    for r in reversed(snaps):  # oldest-first: the last row of a day wins
        by_day[_snapshot_day(r)] = r
    points = [
        {
            "date": day,
            "total_equity": by_day[day].total_equity,
            "contributed": 0.0,
        }
        for day in sorted(by_day)
    ]

    # Net external flow per calendar day from the ledger (contributions in,
    # withdrawals out; everything else — interest, rewards, buys, sales,
    # audits, reconciliation — is not external capital).
    per_day: dict[str, float] = {}
    for e in events:
        if event_flow_class(e) not in ("contribution", "withdrawal"):
            continue
        day = _event_flow_day(e).isoformat()
        per_day[day] = per_day.get(day, 0.0) + float(e.amount)

    end = points[-1]["date"] if points else ""
    flow_days = sorted((d, a) for d, a in per_day.items() if end and d <= end)
    cumulative = 0.0
    series: list[tuple[str, float]] = []
    for day, amount in flow_days:
        cumulative += amount
        series.append((day, cumulative))
    flows = dated_external_flows(
        [d for d, _ in series], [c for _, c in series])

    # Cumulative net contributed as of each equity point.
    i = 0
    running = 0.0
    for p in points:
        while i < len(flow_days) and flow_days[i][0] <= p["date"]:
            running += flow_days[i][1]
            i += 1
        p["contributed"] = running

    return {
        "points": points,
        "flows": [{"date": d.isoformat(), "amount": a} for d, a in flows],
    }


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
    past buys. Removal reversals are appended as NEGATIVE entries dated at
    the removal, so every date before it replays exactly as before; a split
    adjustment is a 0-€ entry dated at the split, so pre-split dates keep
    their pre-split shares against their pre-split closes.

    The curve is EPOCH-SCOPED like the metrics: the mirror book is the
    comparison artifact and restarts with the tracker (a reset clears its
    entries), so pre-reset snapshot dates would replay as a flat zero — a
    fabricated comparison. Only the current epoch's points are returned; the
    preserved snapshot rows themselves are untouched.
    """
    async with Session() as s:
        reset = await _last_reset(s)
        snap_stmt = (
            select(SavingsSnapshot)
            .order_by(SavingsSnapshot.created_at.desc(), SavingsSnapshot.id.desc())
            .limit(limit)
        )
        clause = _snapshot_epoch_clause(reset)
        if clause is not None:
            snap_stmt = snap_stmt.where(clause)
        rows = (await s.scalars(snap_stmt)).all()
        entries = (
            await s.scalars(select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))
        ).all()
    if not rows:
        return []
    # Snapshot dates (sorted) and per-entry day keys, both operator-local.
    dates = sorted({_snapshot_day(r) for r in rows})
    entry_day = {e.id: _local_date(e.created_at).isoformat() for e in entries}
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
        shares_at[d] = {
            t: sh for t, sh in running.items() if abs(sh) > _MIRROR_SHARE_EPS
        }
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
    _price_eur_map, but as a per-date series). The listing's quote unit is
    applied before the FX rate (pence -> pounds for .L); a 1.0 venue's series
    is bit-for-bit unchanged. Dates are keyed by the operator-local day —
    the same day convention ``get_mirror_curve`` groups its snapshot rows by,
    so a close and a curve point written in the same session always agree.
    """
    from .db import Candle
    from .fundamentals import SUFFIX_FX
    from .instruments import quote_unit_factor

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
        series.setdefault(r.ticker, {})[_local_date(r.timestamp).isoformat()] = r.close
    eur_usd = series.get("EURUSD=X", {})
    out: dict[str, dict[str, float]] = {}
    for t in tickers:
        if _is_eur_ticker(t):
            out[t] = dict(series.get(t, {}))
            continue
        pm = SUFFIX_FX.get(t[t.rfind("."):] if "." in t else "")
        local = series.get(t, {})
        conv: dict[str, float] = {}
        # Quote unit before FX (pence -> pounds for .L); 1.0 venues are not
        # multiplied at all, so their converted series stay bit-for-bit.
        unit = quote_unit_factor(t)
        for d, px in local.items():
            usd_px = px * unit if unit != 1.0 else px
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
