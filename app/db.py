import asyncio
import logging
import os
import sqlite3
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, UTC
from pathlib import Path

from sqlalchemy import String, Float, Integer, UniqueConstraint, Text, Boolean, event, text
from sqlalchemy import DateTime as _DateTime
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from .config import settings

logger = logging.getLogger("trade_sentinel.db")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DateTime(TypeDecorator):
    """DateTime column that always round-trips as tz-aware UTC.

    SQLite stores naive datetimes; ``timezone=True`` is a no-op for the
    aiosqlite dialect and the tzinfo is stripped on read. This decorator
    re-attaches UTC on load so every consumer sees a tz-aware value, and
    stores a tz-aware value on write. Naive values written by callers
    are assumed to already be UTC and pass through unchanged.
    """

    impl = _DateTime
    cache_ok = True

    def __init__(self, *args, **kwargs):
        kwargs["timezone"] = True
        super().__init__(*args, **kwargs)

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class OptionalDateStr(TypeDecorator):
    """Nullable date-string column stored as '' instead of NULL.

    SQLite treats NULLs as distinct in UNIQUE constraints, so instant facts
    (start=None) stored as real NULLs never matched the (ticker, tag, start,
    end, filed, source) conflict target — every refresh inserted a fresh
    duplicate set. Storing '' makes the unique key total; reads map '' back
    to None so the fact helpers keep their None-means-instant convention."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return "" if value is None else value

    def process_result_value(self, value, dialect):
        return value or None


class Base(DeclarativeBase):
    pass


class Watchlist(Base):
    __tablename__ = "watchlist"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Candle(Base):
    """One cached daily OHLCV bar plus where it came from.

    ``source`` / ``adjustment_basis`` / ``ingested_at`` record the provenance
    of the values CURRENTLY stored in the row (see ``market._store_candles``):
    which provider actually served them, under which adjustment contract, and
    when they were written. All three are nullable on purpose: rows cached
    before provenance tracking keep NULL = unknown. Nothing is backfilled —
    the source, adjustment basis and fetch time of a historical pull were
    never observed, so asserting them would be fabrication.
    """

    __tablename__ = "candles"
    __table_args__ = (UniqueConstraint("ticker", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0)
    # Provenance of the stored OHLCV (request contract, not a verified
    # corporate-action history). NULL = unknown (legacy rows, or a write that
    # did not declare a source). See market.SOURCE_* / ADJUSTMENT_SPLIT_DIVIDEND.
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    adjustment_basis: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    action: Mapped[str] = mapped_column(String(8))
    reason: Mapped[str] = mapped_column(Text)
    snapshot: Mapped[str] = mapped_column(Text)
    strength: Mapped[float] = mapped_column(Float, default=0)


class ScreenerResult(Base):
    __tablename__ = "screener_results"
    __table_args__ = (UniqueConstraint("universe", "ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    universe: Mapped[str] = mapped_column(String(64), index=True)
    ticker: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Float)
    trend: Mapped[str] = mapped_column(String(32))
    return_20d: Mapped[float] = mapped_column(Float)
    return_60d: Mapped[float] = mapped_column(Float)
    rsi: Mapped[float] = mapped_column(Float)
    relative_volume: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    action: Mapped[str | None] = mapped_column(String(8), nullable=True)  # BUY|SELL|HOLD|N/A
    strength: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-100, None if N/A


# =====================================================================
# Autonomous paper-trading simulation tables
# =====================================================================

class SimAccount(Base):
    """Singleton row (id=1) tracking the sim account balance."""

    __tablename__ = "sim_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, default=0)
    last_allowance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)  # YYYY-MM
    last_review_week: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYY-Www (ISO week)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # The REAL seed this account was created/reset with (current state, not a
    # ledger). NULL = unknown: the row predates the column, so the seed is
    # NEVER inferred from the first paper execution (settings can change
    # between account creation and that run — the drift this column closes).
    opening_seed: Mapped[float | None] = mapped_column(Float, nullable=True)


class SimPosition(Base):
    """Current open positions in the sim portfolio."""

    __tablename__ = "sim_positions"
    __table_args__ = (UniqueConstraint("ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    thesis: Mapped[str] = mapped_column(Text, default="")  # BUY reason for context feedback
    # The percent stop FROZEN at entry (fill price * (1 - stop_pct/100)).
    # Persisted so a top-up can never re-anchor the level to the blended
    # average cost (which would loosen it) — the session fill contract's
    # rule 5. NULL on legacy rows created before the column: those keep the
    # documented fallback (level recomputed from avg_cost each cycle).
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    # The LLM sell layer's type-A DEFER state (app/selllayer.py): while a
    # defer is active the position's standing stop is the frozen percent stop
    # plus THIS re-armed ATR level (the defer re-arms the ATR leg only — the
    # monotone ratchet itself is untouched). NULL = no active defer. Cleared
    # when the position closes (the row is deleted on a full sell).
    stop_defer_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_defer_set_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SimTrade(Base):
    """Executed trade log for the sim portfolio."""

    __tablename__ = "sim_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    cash_after: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class SimAllowance(Base):
    """Monthly imaginary deposit log.

    ``UNIQUE (month, epoch)`` replaces the old ``UNIQUE (month)`` so a closed
    epoch's deposit for a month SURVIVES while a later epoch (or a later
    backfill replay) logs that same month again. Legacy rows are migrated to
    epoch 1 — the pre-reset book — so the key stays TOTAL (no NULL epoch, no
    duplicate-month hole) and a second deposit inside ONE epoch is rejected.
    """

    __tablename__ = "sim_allowances"
    __table_args__ = (UniqueConstraint("month", "epoch"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float)
    month: Mapped[str] = mapped_column(String(7))  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class SimSnapshot(Base):
    """Equity-curve snapshot taken after each sim run."""

    __tablename__ = "sim_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_equity: Mapped[float] = mapped_column(Float)
    allowance_total: Mapped[float] = mapped_column(Float, default=0)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class SimBenchmarkAccount(Base):
    """Singleton row (id=1) tracking the DCA benchmark account."""

    __tablename__ = "sim_benchmark_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, default=0)
    shares: Mapped[float] = mapped_column(Float, default=0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0)
    last_allowance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SimBenchmarkSnapshot(Base):
    """Equity-curve snapshot for the DCA benchmark, taken after each sim run."""

    __tablename__ = "sim_benchmark_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    total_equity: Mapped[float] = mapped_column(Float)
    allowance_total: Mapped[float] = mapped_column(Float, default=0)


class SimChatMessage(Base):
    """Persistent conversation history for the sim portfolio-manager chat.

    Stored so the active chat session survives page reloads and the LLM can
    be fed the full prior context on every turn.
    """

    __tablename__ = "sim_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class SimLlmDecision(Base):
    """Append-only ledger of the LLM's own decisions (session fill contract, rule 7).

    One row per LLM decision — hybrid review, pure-LLM cycle, or chat
    ``[[ACTION]]``:

      * ``decided_at`` is FROZEN when the LLM response is parsed. It is
        captured once, before any of the decision's orders can execute, and
        always written from that capture — a later write never re-stamps it.
      * ``relation`` places the decision next to the deterministic engine:
        ``confirmation`` (same side as the proposal), ``change`` (a flip —
        the proposal is vetoed and the flipped order is engine-excluded),
        ``veto`` (HOLD on a proposed ticker), ``hold`` (HOLD on a ticker the
        engine did not propose), ``addition`` (LLM-initiated — the pure-LLM
        path has no proposals) or ``chat_action`` (an operator chat action).
      * ``det_side`` / ``det_reason`` are the deterministic proposal's side
        and reason for that ticker — NULL when there was no proposal (all
        pure-LLM and chat rows).
      * The FILL columns (``filled_at``, ``shares``, ``price``,
        ``cash_after``) are written ONCE, when the decision's trade actually
        executes. ``filled_at`` is stamped when that executed fill is folded
        into the ledger — after the trade committed, so it is always >=
        ``decided_at`` (the decision is parsed before any of its orders run).

    An unfilled decision keeps all four fill columns NULL, and that is the
    honest record, not a gap: there is no pending-order persistence (rule 7),
    so an unfilled decision is simply dropped and re-taken at the next cycle —
    the row says "decided, never filled". Rows are NEVER updated or deleted
    (the pre-existing destructive reset/backfill paths that replace the whole
    sim portfolio are outside this ledger's scope).
    """

    __tablename__ = "sim_llm_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    mode: Mapped[str] = mapped_column(String(8))  # cycle | chat
    # Operator-local "YYYY-MM-DD": the cycle's session F date for cycle rows,
    # the chat turn's local date for chat rows.
    session_date: Mapped[str] = mapped_column(String(10))
    ticker: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(4))  # BUY | SELL | HOLD
    reason: Mapped[str] = mapped_column(Text, default="")
    # The deterministic proposal this decision relates to (NULL = none).
    det_side: Mapped[str | None] = mapped_column(String(4), nullable=True)
    det_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    relation: Mapped[str] = mapped_column(String(12))
    # Frozen fill — all four are NULL together when the decision never filled.
    filled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_after: Mapped[float | None] = mapped_column(Float, nullable=True)


class LlmSellDecision(Base):
    """Append-only ledger of the LLM sell layer's decisions (app/selllayer.py).

    Mirrors ``SimLlmDecision``'s conventions (session fill contract, rule 7):

      * ``decided_at`` is FROZEN at parse — captured once, when the LLM
        response is parsed, before any of the decision's orders can execute,
        and always written from that capture.
      * ``event_type`` is one of ``A`` (sim standing-stop fire), ``B``
        (monthly band-drop at the rebuild) or ``C`` (monthly thesis break);
        ``engine`` is ``sim`` or ``monthly``. ``event_id`` is the stable
        per-event key (``"A:TICKER:DATE"`` / ``"B:TICKER:DATE"`` /
        ``"C:TICKER:FILING"``) — type C dedupes on it: a break is NEW only
        when no CONSULTED (non-fail-safe) row exists for that (ticker, latest
        filing date) — a transient fail-safe row never silences a filing.
      * ``decision`` is CONFIRM|DEFER (A/B) or EXIT|HOLD (C).
      * ``context`` is the full point-in-time context JSON the prompt was
        rendered from; ``engine_branch`` is the JSON of what the ENGINE's own
        rule would do (A: the fired level; B: the release; C: hold-to-rebuild)
        — the counterfactual the scorecard compares against.
      * The FILL columns (``filled_at``, ``shares``, ``price``, ``cash_after``)
        are written ONCE, when the decision's trade actually executes — in
        practice the type-C/D EXIT rows fill (at a later session's strict
        open). Type A/B rows keep all four NULL BY DESIGN: they act
        in-cycle (the stop executes / the retention releases at the
        rebuild) and their executions are traced via SimTrade /
        MonthlyRebalance, not as pending orders. ``filled_at`` is stamped
        at execution, so it is always >= ``decided_at``; an
        unfilled C-EXIT keeps all four NULL — the decided-but-unfilled
        pending record the next cycle re-scans (there is NO pending-order
        persistence beyond this ledger).

    Verdicts are NEVER changed or deleted; only fill stamping and the type-D
    per-book execution lifecycle audit are mutable. ``fail_safe`` marks decisions where the LLM could not answer
    (no backend / transport error / timeout / malformed / refusal) — the
    engine's own rule stood.
    """

    __tablename__ = "llm_sell_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Operator-local "YYYY-MM-DD" — the event's decision session: the
    # stop-fire session S for type A; the decision session F for B and C.
    session_date: Mapped[str] = mapped_column(String(10))
    event_type: Mapped[str] = mapped_column(String(1))  # A | B | C | D
    engine: Mapped[str] = mapped_column(String(8))      # sim | monthly
    event_id: Mapped[str] = mapped_column(String(96), index=True)
    ticker: Mapped[str] = mapped_column(String(32))
    decision: Mapped[str] = mapped_column(String(7))    # CONFIRM | DEFER | EXIT | HOLD
    rationale: Mapped[str] = mapped_column(Text, default="")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    parsed: Mapped[bool] = mapped_column(Boolean, default=False)
    fail_safe: Mapped[bool] = mapped_column(Boolean, default=False)
    fail_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # The full PIT context the prompt rendered + the engine's counterfactual.
    context: Mapped[str] = mapped_column(Text, default="")
    engine_branch: Mapped[str] = mapped_column(Text, default="")
    # Frozen fill — all four are NULL together until the decision executes.
    filled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Type-D execution audit, separate from the immutable verdict/context.
    # JSON {status, books: {book: {status, position_opened_at, epoch,
    # session_date, filled_at, trade_id, shares, price, cash_after, ...}}}.
    # All book sells and this audit commit atomically. Filled/closed legs
    # never run again; legacy fill columns summarize the first filled book.
    # NULL = not processed (or a pre-audit legacy row), not a fabricated fill.
    execution: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Which backend/model answered ("" when the no-backend short-circuit fired).
    backend: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


# =====================================================================
# Monthly qv-mom portfolio (separate paper portfolio, monthly rebalance)
# =====================================================================

class Fundamental(Base):
    """Point-in-time fundamental fact (stockstrat qv-mom input).

    Mirrors the XBRL-style fact format: one row per (ticker, tag, period)
    with the value and the date it became public (``filed``). A rebalance on
    date d only sees facts with end < d and filed <= d.

    Tags follow the SEC XBRL names the scoring math expects:
    NetIncomeLoss, StockholdersEquity, NetCashProvidedByUsedInOperatingActivities,
    PaymentsToAcquirePropertyPlantAndEquipment, CommonStockSharesOutstanding.

    ``source`` records where the fact came from: 'edgar' (SEC companyfacts,
    real filed dates, 2007+) or 'yfinance' (quarterly/annual statements,
    filed = period_end + 45d approximation).
    """

    __tablename__ = "fundamentals"
    # 'filed' is part of the unique key on purpose: EDGAR re-reports the same
    # span in later filings (restatements), and point-in-time correctness needs
    # every filed version kept — a rebalance must see the version that was
    # public then, not the latest restatement. Dedup happens at read time.
    __table_args__ = (UniqueConstraint("ticker", "tag", "start", "end", "filed", "source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    tag: Mapped[str] = mapped_column(String(64), index=True)
    start: Mapped[str | None] = mapped_column(OptionalDateStr(10), nullable=True)  # '' stored; reads as None = instant fact
    end: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD period end
    filed: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD public date
    val: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    source: Mapped[str] = mapped_column(String(16), default="yfinance")  # edgar | yfinance
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SecCik(Base):
    """Ticker -> SEC CIK mapping (SEC company_tickers.json snapshot).

    Cached in the DB so EDGAR fetches don't re-download the ~10k-row file on
    every run. ``cik = 0`` marks tickers checked and confirmed without a CIK
    (ETFs, foreign listings): they fall back to yfinance fundamentals.
    """

    __tablename__ = "sec_ciks"
    __table_args__ = (UniqueConstraint("ticker",),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True)
    cik: Mapped[int] = mapped_column(Integer)  # 0 = known no-CIK
    name: Mapped[str] = mapped_column(String(128), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MonthlyAccount(Base):
    """Singleton row (id=1) tracking the monthly portfolio cash balance."""

    __tablename__ = "monthly_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, default=0)
    last_allowance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)  # YYYY-MM
    last_rebalance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # The REAL seed this account was created/reset with (current state, not a
    # ledger). NULL = unknown legacy row — never inferred or backfilled.
    opening_seed: Mapped[float | None] = mapped_column(Float, nullable=True)


class MonthlyPosition(Base):
    """Current open positions in the monthly portfolio."""

    __tablename__ = "monthly_positions"
    __table_args__ = (UniqueConstraint("ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MonthlyTrade(Base):
    """Executed trade log for the monthly portfolio."""

    __tablename__ = "monthly_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    cash_after: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class MonthlyAllowance(Base):
    """Monthly imaginary deposit log for the monthly portfolio.

    ``UNIQUE (month, epoch)`` replaces the old ``UNIQUE (month)`` so a closed
    epoch's deposit for a month SURVIVES while a later epoch (or a later
    backfill replay) logs that same month again. Legacy rows are migrated to
    epoch 1 — the pre-reset book — so the key stays TOTAL (no NULL epoch, no
    duplicate-month hole) and a second deposit inside ONE epoch is rejected.
    """

    __tablename__ = "monthly_allowances"
    __table_args__ = (UniqueConstraint("month", "epoch"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float)
    month: Mapped[str] = mapped_column(String(7))  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class MonthlySnapshot(Base):
    """Equity-curve point for the monthly portfolio.

    Written daily by the scheduler (one mark per day so the equity curve
    moves every day) and once more by the month-end rebalance after trade
    execution, so rebalance days carry both the daily mark and the
    post-trade state.
    """

    __tablename__ = "monthly_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_equity: Mapped[float] = mapped_column(Float)
    allowance_total: Mapped[float] = mapped_column(Float, default=0)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class MonthlyRebalance(Base):
    """Audit log: one row per monthly rebalance decision.

    ``picked`` / ``held_before`` are comma-joined ticker lists; ``snapshot``
    holds the JSON eligibility frame (factor values per ticker) so every
    decision is explainable after the fact, like stockstrat's holdings.csv.

    ``UNIQUE (rebal_month, epoch)`` replaces the old ``UNIQUE (rebal_month)``
    for the same reason as the allowance tables: a closed epoch's rebalance
    decision for month M must SURVIVE while the fresh epoch rebalances M
    again. Legacy rows are migrated to epoch 1 (the pre-reset book), so the
    key stays total and two decisions for the same month within ONE epoch are
    still rejected.
    """

    __tablename__ = "monthly_rebalances"
    __table_args__ = (UniqueConstraint("rebal_month", "epoch"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rebal_month: Mapped[str] = mapped_column(String(7))  # YYYY-MM (decision month)
    rebal_date: Mapped[datetime] = mapped_column(DateTime)  # decision trading day
    held_before: Mapped[str] = mapped_column(Text, default="")
    picked: Mapped[str] = mapped_column(Text, default="")
    n_new: Mapped[int] = mapped_column(Integer, default=0)
    snapshot: Mapped[str] = mapped_column(Text, default="")  # JSON eligibility frame
    # Additive JSON (app/selllayer.py type-B overlay): {"engine_picks": [...],
    # "executed_picks": [...]} so the LLM's retention difference is auditable;
    # NULL when the sell layer did not run (engine picks == executed picks).
    sell_layer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


# =====================================================================
# Daily-core paper portfolio (qv-mom core + daily cash deployment)
# =====================================================================

class DailyCoreAccount(Base):
    """Singleton row (id=1) tracking the daily-core portfolio cash balance."""

    __tablename__ = "daily_core_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, default=0)
    last_allowance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # The REAL seed this account was created/reset with (current state, not a
    # ledger). NULL = unknown legacy row — never inferred or backfilled.
    opening_seed: Mapped[float | None] = mapped_column(Float, nullable=True)


class DailyCorePosition(Base):
    """Current open positions in the daily-core portfolio."""

    __tablename__ = "daily_core_positions"
    __table_args__ = (UniqueConstraint("ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class DailyCoreTrade(Base):
    """Executed trade log for the daily-core portfolio."""

    __tablename__ = "daily_core_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    cash_after: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class DailyCoreAllowance(Base):
    """Monthly imaginary deposit log for the daily-core portfolio.

    ``UNIQUE (month, epoch)`` replaces the old ``UNIQUE (month)`` so a closed
    epoch's deposit for a month SURVIVES while a later epoch (or a later
    backfill replay) logs that same month again. Legacy rows are migrated to
    epoch 1 — the pre-reset book — so the key stays TOTAL (no NULL epoch, no
    duplicate-month hole) and a second deposit inside ONE epoch is rejected
    (the race the deposit path relies on: both writers stamp the same epoch).
    """

    __tablename__ = "daily_core_allowances"
    __table_args__ = (UniqueConstraint("month", "epoch"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float)
    month: Mapped[str] = mapped_column(String(7))  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class DailyCoreSnapshot(Base):
    """Equity-curve point for the daily-core portfolio (one mark per day)."""

    __tablename__ = "daily_core_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_equity: Mapped[float] = mapped_column(Float)
    allowance_total: Mapped[float] = mapped_column(Float, default=0)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


class DailyCoreRanking(Base):
    """Latest qv-mom ranking for the daily-core portfolio (singleton row id=1).

    The fundamentals math (full-universe TTM net income, momentum, scoring)
    runs ONCE per day during the daily cycle — its output is stored here and
    every other consumer (status endpoint, UI) just reads this row. Before
    the first cycle of a day (or before the app ever ran a cycle) the row is
    absent/stale and the status endpoint may compute it on demand.
    """

    __tablename__ = "daily_core_ranking"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ranking_date: Mapped[datetime] = mapped_column(DateTime)  # as-of date
    band: Mapped[str] = mapped_column(Text, default="")       # comma-joined hold band (top-20)
    picks: Mapped[str] = mapped_column(Text, default="")      # comma-joined top-10 with hysteresis
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # Run provenance (Phase 2, app/run_config.py). mode/config_id are NULL on
    # rows written before stamping existed; epoch is ALWAYS set — legacy rows
    # migrate to 1, the pre-reset epoch (NULL is not a provenance value).
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # paper | backfill
    config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # run_configs.id
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))


# =====================================================================
# Savings tracker (real-money Trade Republic cash account + Sparpläne)
#
# Deliberately separate from the paper sims above: this tracks the
# operator's REAL money (EUR), so no backfill or "reset all sims" may ever
# touch these tables. The engine (app/savings.py) mirrors deposits,
# interest accrual and Sparplan buys; a forward mirror portfolio replays
# the same € into the daily-core suggestion for comparison.
# =====================================================================

class SavingsAccount(Base):
    """Singleton row (id=1) tracking the real TR cash account."""

    __tablename__ = "savings_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, default=0)
    # Unpaid interest accrued since the last monthly payout (TR pays monthly).
    accrued_interest: Mapped[float] = mapped_column(Float, default=0)
    # Annual interest rate in % (TR: 3% for new customers; enter your NET
    # rate after the ~26.375% German withholding if you want the tracker to
    # match the payout TR actually credits).
    interest_rate: Mapped[float] = mapped_column(Float, default=3.0)
    # Recurring transfer into the savings account at each month start (€).
    monthly_transfer: Mapped[float] = mapped_column(Float, default=0)
    # YYYY-MM markers so monthly actions are idempotent (allowance_tz anchor).
    last_transfer_month: Mapped[str | None] = mapped_column(String(7), nullable=True)
    last_interest_month: Mapped[str | None] = mapped_column(String(7), nullable=True)
    # YYYY-MM-DD marker of the last day interest was accrued for (catch-up
    # after downtime accrues each missed day once).
    last_accrual_day: Mapped[str | None] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SavingsPosition(Base):
    """Open positions bought via Sparpläne (or seeded at initialization)."""

    __tablename__ = "savings_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)  # EUR per share
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SavingsPlan(Base):
    """Configured Sparplan: buy `amount` € of `ticker` on `day_of_month`."""

    __tablename__ = "savings_plans"
    __table_args__ = (UniqueConstraint("ticker", "day_of_month"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[float] = mapped_column(Float)      # EUR per execution
    day_of_month: Mapped[int] = mapped_column(Integer)  # 1..31
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    # YYYY-MM of the last execution (operator-local) — the PER-PLAN dedupe:
    # multiple plans for the same ticker on different days each fire once
    # per month (the ledger alone cannot distinguish them by day).
    last_executed_month: Mapped[str | None] = mapped_column(String(7), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SavingsEvent(Base):
    """Immutable ledger of everything that moved money in the tracker.

    kind: INIT | TRANSFER | WITHDRAWAL | INTEREST_PAYOUT | SPARPLAN_BUY |
          MANUAL_BUY | SAVEBACK | TRUEUP | SELL | WARN | DIVIDEND | FEE | TAX |
          SETTLEMENT | SPLIT | CORRECTION | RESET

    A ``SELL`` row is a DISPOSAL RECORD, not settlement: the operator sold a
    position in Trade Republic, but the tracker books no proceeds (they arrive
    via the next true-up), so the row carries ``amount=0`` and instead records
    WHAT was disposed — when (operator-local date), how many shares, at which
    average cost, and the execution price only if the operator supplied one.
    It never invents a valuation, and it never moves cash or contributed
    capital. The four disposal columns are NULL on every other kind, and on
    rows written before disposals were recorded (those were never observed).

    A ``WITHDRAWAL`` row is a dated external flow OUT: ``amount`` is signed
    negative, ``occurred_on`` carries the operator-local date, and it lowers
    NET contributed capital (contributions − withdrawals) — it is never an
    investment loss.

    A ``DIVIDEND`` row is cash income INTO the tracked cash: ``amount`` is
    signed positive and classified ``dividend``. It is investment income, not
    fresh capital, so it never raises NET contributed.

    A ``FEE`` row is a cost the operator PAID from the tracked cash (e.g. TR's
    FX fee): ``amount`` is signed negative and classified ``fee``. The money
    stays inside the tracker (cash → fee), so it is not an external flow and
    hits the return honestly instead of netting against contributed capital.
    A ``TAX`` row is the same kind of cost (cash → tax, classified ``tax``;
    ``savings.add_tax``): real money out, but part of the RETURN, never of
    contributed capital.

    A ``SETTLEMENT`` row is the second half of a disposal PAIR: the proceeds of
    a manually sold position that are now in the tracked cash (``amount``
    signed positive, classified ``sale_settlement``). Like the TRUEUP and
    reconciliation classes it is NOT an external flow, so the pair reads
    honestly — the SELL record showed the temporary mark-down while the money
    was in flight and this row books the proceeds; the net over the pair is the
    realized outcome. A settlement does not require a prior SELL row: legacy
    disposals predate the record, so the note carries the attribution instead.
    When it DOES settle a recorded disposal, ``settlement_of`` carries that
    SELL row's id — and it is UNIQUE, so one disposal can never be settled
    twice (a repeated settle would double-credit the proceeds).

    ``SPLIT`` and ``CORRECTION`` are dated bookkeeping records with
    ``amount=0``: they record a corporate action (a share split) and a
    corrected holding, using the SAME dated-record columns as a disposal but
    holding the NEW values — ``shares`` is the post-event share count and
    ``cost_basis`` the post-event per-share average cost. Neither moves cash or
    contributed capital, and either column stays NULL when that field was not
    part of the event (a correction may fix only one of the two).

    A ``RESET`` row is the EPOCH BOUNDARY of a history-safe reset: ``amount=0``,
    classified ``audit``, dated operator-local. Every event and snapshot before
    it is PRESERVED (the audit's "no history rewritten by resets"); contributed
    capital and the metrics series read only the rows after the LAST ``RESET``
    row, so the preserved pre-reset history never contaminates the new epoch.

    ``flow_class`` is the write-time snapshot of the ONE flow classification
    (``savings.flow_class_of``: contribution | withdrawal | investment |
    reward | interest | reconciliation | sale | dividend | fee | tax |
    sale_settlement | corporate_action | correction | audit). Rows written
    before classification carry NULL and are classified on read through that
    one function — never guessed, never backfilled.
    """

    __tablename__ = "savings_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    amount: Mapped[float] = mapped_column(Float, default=0)  # € (signed)
    ticker: Mapped[str | None] = mapped_column(String(32), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    # --- Dated record columns (SELL/WITHDRAWAL rows; NULL elsewhere) --------
    # Operator-local calendar date of the event, "YYYY-MM-DD" (the
    # String-style marker convention of SavingsPlan.last_executed_month).
    occurred_on: Mapped[str | None] = mapped_column(String(10), nullable=True)
    shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Execution price per share in EUR — operator-supplied. NULL means
    # unknown; a removal must never invent a valuation.
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Per-share average cost in EUR at disposal
    # (SavingsPosition.avg_cost semantics).
    cost_basis: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Write-time snapshot of savings.flow_class_of (see the class docstring).
    flow_class: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # The SELL event row this SETTLEMENT row settles (SETTLEMENT only; NULL on
    # every other kind and on rows written before the link existed — the
    # disposal they settle predates the column, so it was never observed and
    # is NOT backfilled). UNIQUE so the same disposal can never be settled
    # twice (a repeated settle would double-credit the proceeds); SQLite
    # permits any number of NULLs in a unique index, so every unlinked row
    # stays legal.
    settlement_of: Mapped[int | None] = mapped_column(Integer, nullable=True, unique=True)


class SavingsSnapshot(Base):
    """Equity-curve point (one per day): cash + positions, EUR."""

    __tablename__ = "savings_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    accrued_interest: Mapped[float] = mapped_column(Float, default=0)
    total_equity: Mapped[float] = mapped_column(Float)
    contributed: Mapped[float] = mapped_column(Float, default=0)
    # Operator-local "YYYY-MM-DD" this point is AS OF. A snapshot written just
    # after midnight (or by a catch-up pass) belongs to the day it MEASURES,
    # not to the wall-clock day it happened to be written. NULL on rows written
    # before the column existed — those fall back to the created_at
    # operator-local date on read, no backfill (the true day was never
    # recorded).
    as_of: Mapped[str | None] = mapped_column(String(10), nullable=True)


class SavingsMirrorEntry(Base):
    """One mirrored € inflow: the same money that went into positions
    (seed holdings, Sparplan buys) is also allocated to the daily-core
    suggestion as of that date — the forward mirror comparison."""

    __tablename__ = "savings_mirror_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float)  # EUR allocated
    # What the money actually went into ("" for the seed lump into cash).
    source_ticker: Mapped[str] = mapped_column(String(32), default="")
    # JSON [[ticker, shares, price], ...] — the picks bought with this entry,
    # so the mirror curve can be replayed as a true time series (cumulative
    # shares as of each date, valued at that date's close).
    breakdown: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class SavingsMirrorPosition(Base):
    """Mirror portfolio holdings: daily-core picks bought with mirrored €."""

    __tablename__ = "savings_mirror_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)  # EUR per share
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SavingsMirrorPending(Base):
    """Mirrored-€ inflow that could not be allocated yet (no fresh ranking
    or missing prices), parked PER SOURCE: the row keeps its ticker link,
    so a later position removal releases the parked € together with the
    placed mirror entries, and the next pass with a fresh ranking flushes
    it as a source-tagged entry. A deferral is never a silent drop — and
    never an orphan the mirror keeps after the position is gone."""

    __tablename__ = "savings_mirror_pending"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float)  # EUR parked
    source_ticker: Mapped[str] = mapped_column(String(32), default="")  # "" = seed cash
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


# =====================================================================
# Run configuration registry (Phase 2 of the strategy audit)
#
# Every run is identified by an IMMUTABLE configuration: engine, mode and the
# canonical JSON of its settings, deduped by content hash. A reset is not a
# delete — it appends a RunEpoch boundary, so historical rows keep the epoch
# they were written in and a fresh epoch can never be confused with (or
# overwrite) a forward paper record. Nothing writes these tables yet; the
# engine adoption batch (R2) does, through app/run_config.py only.
# =====================================================================

class RunConfig(Base):
    """Append-only registry of IMMUTABLE run configurations.

    A run's configuration is FROZEN when the run starts: ``content`` holds the
    canonical JSON (sorted keys, tight separators) and ``config_hash`` its
    sha256, so the same (engine, mode, content) always resolves to the SAME
    row — the UNIQUE constraint is the dedupe and ``get_or_create`` inserts
    with ON CONFLICT DO NOTHING, which makes it race-safe.

    Rows are NEVER updated or deleted. Counting rows per engine therefore
    reports the number of DISTINCT CONFIGURATIONS REGISTERED — not the number
    of execution runs (a scheduler that runs an unchanged configuration every
    day reuses one row) and not the number of experiments (two configurations
    that never executed are still two rows).

    ``mode`` is the run vocabulary, not the history-row vocabulary:
    ``backfill`` (a synthetic replay) or ``paper`` (a live forward record).
    """

    __tablename__ = "run_configs"
    __table_args__ = (UniqueConstraint("engine", "mode", "config_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    engine: Mapped[str] = mapped_column(String(16))  # daily_core | monthly | sim
    mode: Mapped[str] = mapped_column(String(16))    # paper | backfill
    config_hash: Mapped[str] = mapped_column(String(64))  # sha256 hex of canonical(content)
    content: Mapped[str] = mapped_column(Text)       # the canonical JSON itself (provenance)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class RunEpoch(Base):
    """One reset boundary of one engine's paper book ("history is not rewritten").

    Each row bumps that engine's epoch by one: history rows written before the
    reset keep epoch N, rows written after it are stamped N+1, and every
    epoch-scoped reader (``app.run_config.epoch_clause``) sees only its own
    epoch. CONVENTION: no RunEpoch row for an engine means the current epoch
    is 1; history rows that predate epoch stamping were MIGRATED to 1 at the
    schema upgrade (the logical pre-reset book — no NULL epoch was invented),
    and ``epoch_clause`` still matches NULL for synthetic fixtures.
    Row ids are append-only; ``UNIQUE (engine, epoch)`` makes a duplicate
    boundary impossible, and ``record_boundary`` allocates the next epoch in a
    single statement so concurrent resets cannot both land on the same number.
    """

    __tablename__ = "run_epochs"
    __table_args__ = (UniqueConstraint("engine", "epoch"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    engine: Mapped[str] = mapped_column(String(16))  # daily_core | monthly | sim
    epoch: Mapped[int] = mapped_column(Integer)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class RunExecution(Base):
    """One row per EXECUTION: append-only and deliberately NOT deduped.

    ``RunConfig`` answers "which configuration is this?" (deduped by content
    hash); a ``RunExecution`` answers "when did this run happen, and on which
    data?". Two runs of the identical configuration are therefore TWO rows —
    the honest count of executions — while the configuration registry keeps
    one. Nothing here is inferred: ``provenance`` is the canonical JSON of the
    per-run cache provenance summary the caller passed (sources, adjustment
    bases, fetch/ingest times, staleness notes — whatever the summarizer
    observed), and ``data_asof`` is the market-data cut-off the run saw.

    ``config_id`` references ``run_configs.id`` (no FK: SQLite cannot rebuild
    a table around one, and the columns are written together by R2);
    ``epoch`` is the engine epoch the run executed in (>= 1, default 1 — the
    pre-reset epoch). Rows are never updated or deleted.
    """

    __tablename__ = "run_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    engine: Mapped[str] = mapped_column(String(16))  # daily_core | monthly | sim
    mode: Mapped[str] = mapped_column(String(16))    # paper | backfill
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)  # run_configs.id
    # The market-data cut-off this run saw (NULL = the run did not declare one)
    data_asof: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    provenance: Mapped[str] = mapped_column(Text)  # canonical JSON of the run's provenance
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# History tables that carry the run-provenance stamps (mode/config_id/epoch).
# Current-state tables (accounts, positions) deliberately do NOT: they are not
# history, and R2 replaces them per epoch instead of tagging them.
_RUN_STAMP_MODELS = (
    SimTrade, SimAllowance, SimSnapshot,
    MonthlyTrade, MonthlyAllowance, MonthlySnapshot, MonthlyRebalance,
    DailyCoreTrade, DailyCoreAllowance, DailyCoreSnapshot, DailyCoreRanking,
)
_RUN_STAMP_COLUMNS = (
    ("mode", "VARCHAR(16)"),
    ("config_id", "INTEGER"),
)
# epoch is NOT NULL with a DEFAULT: every row ever stored has a real epoch
# (legacy rows are epoch 1, the pre-reset book). An ALTER can add it directly;
# a partial upgrade that made it nullable must be rebuilt to the model shape.
_EPOCH_ALTER_DDL = "INTEGER NOT NULL DEFAULT 1"

# Tables whose legacy single-column UNIQUE key is too narrow for epoch-safe
# history. Detecting it is the migration gate: any UNIQUE index covering
# exactly this key (and nothing else) means the pre-R1 shape.
_LEGACY_UNIQUE_KEYS = {
    SimAllowance: ("month",),
    MonthlyAllowance: ("month",),
    DailyCoreAllowance: ("month",),
    MonthlyRebalance: ("rebal_month",),
}


def _sqlite_columns(sync_conn, table: str) -> list[str]:
    """Column names of *table*, in declaration order (PRAGMA table_info)."""
    return [row[1] for row in _sqlite_column_info(sync_conn, table)]


def _sqlite_column_info(sync_conn, table: str) -> list[tuple]:
    """Raw PRAGMA table_info rows: (cid, name, type, notnull, default, pk)."""
    return sync_conn.exec_driver_sql(f'PRAGMA table_info("{table}")').fetchall()


def _epoch_default_is_one(raw) -> bool:
    """True when PRAGMA's DEFAULT text is the integer 1 in a known spelling.

    SQLite stores the DEFAULT expression text verbatim, so ``1``, ``'1'`` and
    ``(1)`` are all the same default. Anything else — including NO default at
    all — cannot be trusted as "new rows get epoch 1", so that shape is
    rebuilt to the model's ``DEFAULT 1``.
    """
    if raw is None:
        return False
    value = str(raw).strip()
    while len(value) >= 2 and value[0] == "(" and value[-1] == ")":
        value = value[1:-1].strip()
    value = value.strip("'\"")
    try:
        return float(value) == 1.0
    except ValueError:
        return False


def _epoch_needs_rebuild(sync_conn, table: str) -> bool:
    """True unless ``epoch`` is exactly ``NOT NULL DEFAULT 1``.

    A partially upgraded DB can carry a nullable epoch column (NULLs are the
    pre-reset book) or a NOT NULL column with no default (a writer that omits
    it would fail). SQLite cannot alter a column in place, so the shape is
    rebuilt from the model; NULLs are normalized to 1 first.
    """
    info = {row[1]: row for row in _sqlite_column_info(sync_conn, table)}
    epoch = info.get("epoch")
    if epoch is None:
        return False
    return not epoch[3] or not _epoch_default_is_one(epoch[4])


def _sqlite_unique_index_columns(sync_conn, table: str) -> list[tuple[str, ...]]:
    """Column list of every UNIQUE index on *table* (PRAGMA index_list/info).

    ``unique=True`` and ``UniqueConstraint`` both land here as (auto)indexes,
    so this is the one predicate that works for legacy and current shapes.
    """
    out: list[tuple[str, ...]] = []
    for row in sync_conn.exec_driver_sql(f'PRAGMA index_list("{table}")').fetchall():
        if not row[2]:  # 0 = a plain (non-unique) index
            continue
        name = str(row[1]).replace("'", "''")
        cols = tuple(
            r[2]
            for r in sync_conn.exec_driver_sql(f"PRAGMA index_info('{name}')").fetchall()
        )
        out.append(cols)
    return out


def _rebuild_to_model_shape(sync_conn, model) -> int:
    """Rewrite a legacy table into *model*'s exact shape, preserving every row.

    SQLite cannot drop or widen a UNIQUE constraint in place, so the only
    history-preserving path is CREATE (staging, model shape) -> INSERT SELECT
    -> DROP -> RENAME. Everything destructive happens inside the caller's
    transaction (the INSERT opens it), so any failure leaves the original
    table and every row untouched; the worst residue is an EMPTY staging table
    from an attempt that died before the transaction opened, and the next
    attempt drops it first.

    Every column present in BOTH the legacy table and the model is copied, so
    a partially upgraded DB keeps mode/config_id values it already stamped;
    an ``epoch`` that is present but NULL is the PRE-RESET book and is
    materialized as 1 (the column is NOT NULL in the final shape); columns
    that are new in the model stay at their model default; a legacy column the
    model does not know raises instead of being dropped silently.

    The model's explicit indexes (``index=True`` columns) are recreated after
    the rename — dropping the table drops them with it, and a rebuild must
    leave exactly the schema ``create_all`` would have produced.
    """
    from sqlalchemy import MetaData
    from sqlalchemy.schema import CreateIndex, CreateTable

    name = model.__tablename__
    staging = f"{name}__r1_mig"
    existing = _sqlite_columns(sync_conn, name)
    target = [col.name for col in model.__table__.columns]
    unknown = [col for col in existing if col not in target]
    if unknown:
        raise RuntimeError(
            f"cannot rebuild {name}: legacy column(s) {unknown} are not in the model"
        )
    before = sync_conn.exec_driver_sql(f'SELECT COUNT(*) FROM "{name}"').scalar_one()
    # A staging table can only survive a failed prior attempt.
    sync_conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{staging}"')
    ddl = str(
        CreateTable(model.__table__.to_metadata(MetaData(), name=staging)).compile(
            dialect=sync_conn.dialect
        )
    )
    sync_conn.exec_driver_sql(ddl)
    carried = [col for col in target if col in existing]
    columns = ", ".join(f'"{col}"' for col in carried)
    values = ", ".join(
        'COALESCE("epoch", 1)' if col == "epoch" else f'"{col}"' for col in carried
    )
    sync_conn.exec_driver_sql(
        f'INSERT INTO "{staging}" ({columns}) SELECT {values} FROM "{name}"'
    )
    sync_conn.exec_driver_sql(f'DROP TABLE "{name}"')
    sync_conn.exec_driver_sql(f'ALTER TABLE "{staging}" RENAME TO "{name}"')
    for index in model.__table__.indexes:
        sync_conn.exec_driver_sql(
            str(CreateIndex(index).compile(dialect=sync_conn.dialect))
        )
    after = sync_conn.exec_driver_sql(f'SELECT COUNT(*) FROM "{name}"').scalar_one()
    if after != before:
        raise RuntimeError(f"rebuild of {name} lost rows: {before} -> {after}")
    return int(after)


async def _migrate_phase2_schema(conn) -> None:
    """Converge the run-configuration/epoch schema — idempotent, no backfill.

    ORDER MATTERS: a table whose UNIQUE key or ``epoch`` column needs the
    rebuild runs FIRST because the rebuild creates the full final column set
    itself (stamped values included when a partial upgrade already has them),
    so the generic additive-ALTER loop that follows no-ops there.

    ``mode`` / ``config_id`` are NULL on rows written before stamping existed
    (unknown provenance — never invented). ``epoch`` is always a real value:
    legacy rows become 1 (the logical pre-reset epoch, not invented history),
    and the DEFAULT 1 covers rows a writer forgets to stamp.
    """
    def _sync(sync_conn):
        for model in _RUN_STAMP_MODELS:
            table = model.__tablename__
            legacy_key = _LEGACY_UNIQUE_KEYS.get(model)
            needs_rebuild = _epoch_needs_rebuild(sync_conn, table) or (
                legacy_key is not None
                and legacy_key in _sqlite_unique_index_columns(sync_conn, table)
            )
            if needs_rebuild:
                # Normalize before the copy: a NULL epoch is the pre-reset
                # book (1), and the final column is NOT NULL.
                if "epoch" in _sqlite_columns(sync_conn, table):
                    sync_conn.exec_driver_sql(
                        f'UPDATE "{table}" SET epoch = 1 WHERE epoch IS NULL'
                    )
                _rebuild_to_model_shape(sync_conn, model)
            existing = set(_sqlite_columns(sync_conn, table))
            # Stamp order matters only for shape parity with create_all:
            # mode/config_id first, epoch last, exactly like the model.
            for col, coltype in _RUN_STAMP_COLUMNS:
                if col not in existing:
                    sync_conn.exec_driver_sql(
                        f'ALTER TABLE "{table}" ADD COLUMN {col} {coltype}'
                    )
            if "epoch" not in existing:
                sync_conn.exec_driver_sql(
                    f'ALTER TABLE "{table}" ADD COLUMN epoch {_EPOCH_ALTER_DDL}'
                )
        # Settlement link (savings): the SELL row a SETTLEMENT settles. NULL
        # on legacy rows and every other kind — never backfilled, because a
        # pre-column settlement's disposal was never observed. UNIQUE so one
        # disposal cannot be settled twice (SQLite allows many NULLs).
        savings_events = SavingsEvent.__tablename__
        if "settlement_of" not in set(_sqlite_columns(sync_conn, savings_events)):
            sync_conn.exec_driver_sql(
                f"ALTER TABLE {savings_events} ADD COLUMN settlement_of INTEGER"
            )
        # create_all already enforces the uniqueness inline on a fresh DB, so
        # the NAMED index is added only when no unique index covers the column
        # yet (a legacy table) — otherwise both paths would carry two
        # identical unique indexes.
        if ("settlement_of",) not in _sqlite_unique_index_columns(
            sync_conn, savings_events
        ):
            sync_conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_savings_events_settlement_of "
                "ON savings_events (settlement_of)"
            )
        # Account opening seed (capital contract): the REAL seed an account was
        # created/reset with. CURRENT STATE, so the reset path overwrites it in
        # the same transaction (it is deliberately NOT a history stamp). Purely
        # additive and nullable: a legacy account row keeps NULL = unknown —
        # the seed is never inferred from the first paper execution (settings
        # can change between account creation and that run) and never
        # backfilled from cash.
        for model in (SimAccount, MonthlyAccount, DailyCoreAccount):
            table = model.__tablename__
            if "opening_seed" not in set(_sqlite_columns(sync_conn, table)):
                sync_conn.exec_driver_sql(
                    f'ALTER TABLE "{table}" ADD COLUMN opening_seed FLOAT'
                )

    await conn.run_sync(_sync)


def _configure_sqlite_pragmas(dbapi_conn, _record) -> None:
    """Set durability/concurrency pragmas on every new SQLite connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def _should_apply_sqlite_pragmas(url: str) -> bool:
    return url.startswith("sqlite")


def create_db_engine(database_url: str):
    """Create the app's async engine with SQLite tuning applied.

    WAL + synchronous=NORMAL: durable-enough commits without fsync-per-write.
    busy_timeout: writers queue briefly instead of failing with 'database is
    locked' when the nightly scheduler and a manual API call overlap.
    foreign_keys: SQLite defaults to OFF; SQLAlchemy never enables it for you.
    pool_pre_ping: drop connections silently invalidated by container restarts.
    """
    engine = create_async_engine(database_url, connect_args={"timeout": 5}, pool_pre_ping=True)
    if _should_apply_sqlite_pragmas(database_url):
        event.listen(engine.sync_engine, "connect", _configure_sqlite_pragmas)
    return engine


# Note: WAL mode persists in the DB file itself, so it only needs setting
# once per database — but re-applying it per connection is harmless.
engine = create_db_engine(settings.database_url)


# =====================================================================
# PROD/DEV database switch
#
# PROD is the live book. DEV is a disposable sandbox that may be cloned from
# PROD at any time (see ``clone_prod_to_dev``). Selection is PER REQUEST via
# a ContextVar set by the ``ts_db`` cookie middleware; everything that runs
# outside a request (the schedulers) keeps the default "prod".
# =====================================================================

def _sqlite_path(eng) -> Path:
    """Filesystem path of a SQLite engine's database (from its URL).

    Raises for anything that is not a real SQLite file: the DEV engine must
    be a clone target a backup can land on, and never ``:memory:``.
    """
    url = eng.sync_engine.url
    if not url.drivername.startswith("sqlite"):
        raise RuntimeError(f"expected a SQLite engine, got {url.drivername!r}")
    database = url.database
    if not database or database == ":memory:":
        raise RuntimeError("expected a file-backed SQLite engine")
    return Path(database)


def _create_dev_engine(prod_engine, dev_url: str):
    """Create the DEV engine, or return None (error-logged) when DEV is unsafe.

    Two misconfigurations are refused instead of obeyed, because
    ``clone_prod_to_dev`` would otherwise let ``os.replace`` land a backup
    copy on the wrong file:

    * a non-file URL (``:memory:``, a non-SQLite backend) — a clone target
      must be a real file;
    * a path that resolves to the PROD database — a copy-pasted
      ``DATABASE_URL_DEV`` would overwrite the live book with its own clone.

    An empty URL is the documented "DEV disabled" switch and returns None
    silently: the feature is off by default.
    """
    if not dev_url:
        return None
    try:
        dev_engine = create_db_engine(dev_url)
        dev_path = _sqlite_path(dev_engine)
        prod_path = _sqlite_path(prod_engine)
    except Exception as e:  # noqa: BLE001 — a misconfigured DEV must not kill startup
        logger.error("DATABASE_URL_DEV %r is not usable (%s) — DEV database disabled",
                     dev_url, e)
        return None
    if dev_path.resolve() == prod_path.resolve():
        logger.error("DATABASE_URL_DEV resolves to the PROD database (%s) — DEV "
                     "disabled to protect PROD", prod_path)
        return None
    return dev_engine


engine_dev = _create_dev_engine(engine, settings.database_url_dev)
DB_NAMES = ("prod", "dev")

# Per-request DB selection. ContextVars are task-local: the cookie middleware
# wraps a request in ``use_db``, while background scheduler tasks keep the
# "prod" default and can never be redirected by a browser.
_active_db: ContextVar[str] = ContextVar("trade_sentinel_active_db", default="prod")

# name -> (engine it was built from, sessionmaker). The engine identity is
# remembered so a swapped engine (tests, maintenance) transparently gets a
# fresh sessionmaker instead of a factory bound to a disposed engine.
_sessionmakers: dict[str, tuple[object, async_sessionmaker]] = {}


def current_db() -> str:
    """The database serving the current context: "prod" (default) or "dev"."""
    return _active_db.get()


@contextmanager
def use_db(name: str):
    """Pin the active DB for a block (request middleware, tests, seeding).

    Resets to the previous value on exit — including on an exception — so a
    failed request can never leak its selection into the next one.
    """
    if name not in DB_NAMES:
        raise ValueError(f"unknown db name {name!r} (choose from {DB_NAMES})")
    token = _active_db.set(name)
    try:
        yield
    finally:
        _active_db.reset(token)


def active_engine():
    """The SQLAlchemy engine behind the active DB.

    Use this (or, preferably, just ``Session()``) anywhere code needs the
    engine object rather than a session: a request routed to DEV must never
    accidentally read PROD. Raises when DEV is selected but not configured.
    """
    name = current_db()
    eng = engine if name == "prod" else engine_dev
    if eng is None:
        raise RuntimeError(
            f"the {name.upper()} database is not configured — DATABASE_URL_DEV is unset")
    return eng


def dev_enabled() -> bool:
    """True when a usable DEV engine exists (ie. the switch is on)."""
    return engine_dev is not None


def _sessionmaker_for(name: str, eng) -> async_sessionmaker:
    cached = _sessionmakers.get(name)
    if cached is not None and cached[0] is eng:
        return cached[1]
    maker = async_sessionmaker(eng, expire_on_commit=False)
    _sessionmakers[name] = (eng, maker)
    return maker


def Session():
    """Session factory bound to the ACTIVE database (PROD by default).

    Modules keep their ``from .db import Session`` + ``Session()`` contract —
    this function resolves the ContextVar at CALL time, so a call made while
    the request middleware's ``use_db("dev")`` block is active talks to the
    sandbox, and a background task (no context, default "prod") always talks
    to the live book.
    """
    name = current_db()
    return _sessionmaker_for(name, active_engine())()


def _disable_dev(reason) -> None:
    """Switch the feature off for this process (error-logged).

    A DEV database that cannot even be opened must degrade to "feature
    disabled" — never take the app (and therefore PROD) down with it.
    """
    global engine_dev
    if engine_dev is not None:
        logger.error("DEV database unusable (%s) — PROD/DEV switch disabled for "
                     "this run; every request is served by PROD", reason)
    engine_dev = None
    _sessionmakers.pop("dev", None)


def dev_db_exists() -> bool:
    return engine_dev is not None and _sqlite_path(engine_dev).exists()


def prod_db_exists() -> bool:
    return _sqlite_path(engine).exists()


async def clone_prod_to_dev() -> dict:
    """Overwrite DEV with a fresh copy of PROD (SQLite online-backup API).

    Safe while PROD is live: the backup API snapshots committed pages under
    WAL. The copy lands in a temp file next to the DEV file and is swapped in
    with ``os.replace`` (atomic on one filesystem); the DEV engine is
    disposed around the swap so no pooled connection keeps pointing at the
    replaced inode, and stale ``-wal``/``-shm`` sidecars of the OLD file are
    removed — they must never pair with the new file.
    """
    if engine is None or engine_dev is None:
        raise RuntimeError("DATABASE_URL_DEV is not configured — no DEV engine to clone into")
    prod_path, dev_path = _sqlite_path(engine), _sqlite_path(engine_dev)
    if not prod_path.exists():
        raise RuntimeError(f"PROD database {prod_path} does not exist — nothing to clone")
    # Defense in depth: _create_dev_engine already refuses a DEV URL that
    # resolves to PROD, but a programmatic engine swap must never let
    # os.replace overwrite PROD with the backup copy either.
    if prod_path.resolve() == dev_path.resolve():
        raise RuntimeError("DEV database path resolves to the PROD database — refusing to clone over PROD")
    tmp_path = dev_path.with_suffix(dev_path.suffix + ".tmp")
    dev_path.parent.mkdir(parents=True, exist_ok=True)

    def _backup() -> None:
        # closing(): `with sqlite3.connect(...)` commits but does NOT close.
        # A leftover tmp from a crashed clone must never poison this backup
        # (sqlite3.connect would open it as-is and the backup would write
        # into it).
        tmp_path.unlink(missing_ok=True)
        with closing(sqlite3.connect(str(prod_path), timeout=30)) as src, \
                closing(sqlite3.connect(str(tmp_path))) as dst:
            src.backup(dst)

    try:
        await asyncio.to_thread(_backup)
        await engine_dev.dispose()  # no pooled connection may hold the old inode
        os.replace(tmp_path, dev_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)  # never leave a half-written copy behind
        except OSError:
            pass
        raise
    for sidecar in (f"{dev_path}-wal", f"{dev_path}-shm"):
        Path(sidecar).unlink(missing_ok=True)
    return {"ok": True, "path": str(dev_path), "size_bytes": dev_path.stat().st_size,
            "cloned_at": datetime.now(UTC).isoformat()}


async def init_db(db_engine=None):
    """Create the schema and converge existing databases (idempotent).

    Without arguments it migrates BOTH configured databases: PROD and, when
    the switch is on, DEV (so the sandbox always has a current schema, even
    before the auto-seed clone replaces it). ``db_engine`` exists for tests
    and maintenance scripts that run the migrations against one specific
    engine; the app always calls it bare.

    A DEV database that cannot be opened is disabled with an error log
    instead of failing startup — PROD must never depend on the sandbox.
    """
    if db_engine is not None:
        await _migrate_engine(db_engine)
        return
    await _migrate_engine(engine)
    if engine_dev is not None:
        try:
            await _migrate_engine(engine_dev)
        except Exception as e:  # noqa: BLE001 — PROD stays up whatever DEV does
            _disable_dev(e)


async def _migrate_engine(db_engine):
    """Bring ONE engine's schema up to date (create_all + additive ALTERs)."""
    target = db_engine
    async with target.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Phase 2 run provenance: the run_configs/run_epochs registry and the
        # mode/config_id/epoch stamps on every history table, plus the
        # UNIQUE-key reshaping that lets a reset-starting epoch log a month /
        # rebalance month a closed epoch already logged. Runs before the
        # additive ALTER loops below and is a complete no-op on a current DB.
        await _migrate_phase2_schema(conn)
        # D's per-book lifecycle/fill audit. Never infer historical fills.
        columns = await conn.run_sync(
            lambda c: _sqlite_columns(c, "llm_sell_decisions"))
        if "execution" not in columns:
            await conn.execute(text(
                "ALTER TABLE llm_sell_decisions ADD COLUMN execution TEXT"))
        # Manual migration for screener_results: add action/strength columns
        # for existing DBs created before these columns existed. SQLAlchemy's
        # create_all does not alter existing tables. Idempotent: sqlite raises
        # OperationalError if the column already exists.
        for col, coltype in [("action", "TEXT"), ("strength", "REAL")]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE screener_results ADD COLUMN {col} {coltype}")
                )
            except Exception:
                pass  # column already exists — expected on subsequent starts
        # Migration for candles: provenance metadata. Purely additive and
        # nullable — rows cached before this tracking keep NULL (unknown
        # source/basis/fetch time). Deliberately NO backfill: nobody observed
        # those historical pulls, so any value would be fabricated.
        for col, coltype in [
            ("source", "VARCHAR(32)"),
            ("adjustment_basis", "VARCHAR(32)"),
            ("ingested_at", "DATETIME"),
        ]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE candles ADD COLUMN {col} {coltype}")
                )
            except Exception:
                pass  # column already exists — expected on subsequent starts
        # Migration for sim_positions: add thesis column for LLM context feedback.
        try:
            await conn.execute(
                text("ALTER TABLE sim_positions ADD COLUMN thesis TEXT DEFAULT ''")
            )
        except Exception:
            pass  # column already exists
        # Migration for sim_positions: the percent stop FROZEN at entry
        # (session fill contract, rule 5) — persisted on the position so a
        # top-up can never loosen it. Nullable: legacy rows keep the
        # avg_cost-based fallback, no data migration.
        try:
            await conn.execute(
                text("ALTER TABLE sim_positions ADD COLUMN stop_price FLOAT")
            )
        except Exception:
            pass  # column already exists
        # Migration for sim_positions: the LLM sell layer's type-A DEFER state
        # (stop_defer_level / stop_defer_set_at, app/selllayer.py). Additive and
        # nullable — legacy rows keep NULL (no active defer), no data migration.
        for col, coltype in [
            ("stop_defer_level", "FLOAT"),
            ("stop_defer_set_at", "DATETIME"),
        ]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE sim_positions ADD COLUMN {col} {coltype}")
                )
            except Exception:
                pass  # column already exists — expected on subsequent starts
        # Migration for monthly_rebalances: the sell-layer audit JSON (engine
        # picks vs executed picks). Additive and nullable — NULL when the
        # overlay did not run.
        try:
            await conn.execute(
                text("ALTER TABLE monthly_rebalances ADD COLUMN sell_layer TEXT")
            )
        except Exception:
            pass  # column already exists
        # Migration for sim_account: add last_review_week for calendar-based
        # weekly LLM portfolio reviews.
        try:
            await conn.execute(
                text("ALTER TABLE sim_account ADD COLUMN last_review_week VARCHAR(8)")
            )
        except Exception:
            pass  # column already exists
        # Migration for savings_mirror_entries: the per-pick breakdown JSON
        # (added with the true time-series mirror curve). Pre-release dev
        # builds of the savings tracker created the table without it.
        try:
            await conn.execute(
                text("ALTER TABLE savings_mirror_entries ADD COLUMN breakdown TEXT DEFAULT ''")
            )
        except Exception:
            pass  # column (or table) already exists
        # Migration for savings_plans: per-plan execution marker (the dedupe
        # used to be ledger-derived and ticker-keyed, which silently skipped
        # a second plan for the same ticker within a month).
        try:
            await conn.execute(
                text("ALTER TABLE savings_plans ADD COLUMN last_executed_month VARCHAR(7)")
            )
        except Exception:
            pass  # column (or table) already exists
        # Migration for savings_events: the append-only disposal record (the
        # SELL row written when a position is removed manually) needs the
        # disposal facts. Purely additive and nullable, deliberately NO
        # backfill: removals that happened before this tracking were never
        # observed — what/when/at which price would be fabricated — so those
        # rows keep NULL in all five columns.
        for col, coltype in [
            ("occurred_on", "VARCHAR(10)"),
            ("shares", "FLOAT"),
            ("price", "FLOAT"),
            ("cost_basis", "FLOAT"),
            ("flow_class", "VARCHAR(16)"),
        ]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE savings_events ADD COLUMN {col} {coltype}")
                )
            except Exception:
                pass  # column (or table) already exists — expected on subsequent starts
        # Migration for savings_snapshots: the operator-local "as of" day of an
        # equity point (a snapshot written just after midnight attributes to the
        # day it measures, not to the wall clock). Purely additive and nullable,
        # deliberately NO backfill: the day a legacy point measured was never
        # recorded, so it keeps NULL and readers fall back to created_at.
        try:
            await conn.execute(
                text("ALTER TABLE savings_snapshots ADD COLUMN as_of VARCHAR(10)")
            )
        except Exception:
            pass  # column (or table) already exists — expected on subsequent starts
        # Migration for deferred mirror €: pre-release dev builds parked it
        # in a scalar savings_account.mirror_pending column with no source
        # attribution (a removed position could not release its parked €).
        # The per-source savings_mirror_pending table replaced the column;
        # move any legacy amount into the table once. Fresh installs never
        # had the column — the SELECT errors and is swallowed, same
        # convention as the ALTER migrations above.
        try:
            legacy_pending = (await conn.execute(
                text("SELECT mirror_pending FROM savings_account WHERE id = 1")
            )).scalar()
        except Exception:
            legacy_pending = None
        if legacy_pending is not None and float(legacy_pending) > 0:
            # Source is unknown for legacy € — "" (the seed convention)
            # keeps it alive and allocated on the next flush.
            await conn.execute(
                text("INSERT INTO savings_mirror_pending "
                     "(amount, source_ticker, created_at) "
                     "VALUES (:amount, '', :created)"),
                {
                    "amount": float(legacy_pending),
                    "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
                },
            )
            await conn.execute(
                text("UPDATE savings_account SET mirror_pending = 0 WHERE id = 1")
            )
        # Migration for fundamentals: add the `source` column (edgar|yfinance)
        # and rebuild the unique constraint to (ticker, tag, start, end, filed,
        # source) — EDGAR re-reports spans in later filings and point-in-time
        # correctness needs every filed version kept. SQLite can't ALTER a
        # UNIQUE constraint — rebuild the table the SQL-safe way.
        # NULL starts are normalized to '' (see OptionalDateStr): NULLs are
        # distinct in SQLite UNIQUE constraints, so instant facts never hit
        # the upsert's conflict target and duplicated on every refresh —
        # the rebuild dedupes those while it's at it.
        from sqlalchemy import inspect as sa_inspect
        insp = await conn.run_sync(lambda sync_conn: sa_inspect(sync_conn))
        has_fund = await conn.run_sync(lambda sc: insp.has_table("fundamentals"))
        cols = {c["name"] for c in (await conn.run_sync(lambda sc: insp.get_columns("fundamentals")))} if has_fund else set()
        uq_cols: set[str] = set()
        if has_fund:
            uqs = await conn.run_sync(lambda sc: insp.get_unique_constraints("fundamentals"))
            for uq in uqs:
                uq_cols.update(uq.get("column_names", []))
        need_rebuild = has_fund and ("source" not in cols or "filed" not in uq_cols)
        if has_fund and not need_rebuild:
            # Schema is current — but pre-OptionalDateStr DBs may still hold
            # start=NULL rows (NULLs escaped the UNIQUE constraint and the
            # upsert duplicated them). Normalize + dedupe in place if so.
            null_starts = (await conn.execute(
                text("SELECT COUNT(*) FROM fundamentals WHERE start IS NULL")
            )).scalar()
            if null_starts:
                await conn.execute(text("""
                    CREATE TABLE fundamentals_new (
                        id INTEGER NOT NULL PRIMARY KEY,
                        ticker VARCHAR(32) NOT NULL,
                        tag VARCHAR(64) NOT NULL,
                        start VARCHAR(10),
                        "end" VARCHAR(10) NOT NULL,
                        filed VARCHAR(10) NOT NULL,
                        val FLOAT NOT NULL,
                        currency VARCHAR(8) NOT NULL DEFAULT 'USD',
                        source VARCHAR(16) NOT NULL DEFAULT 'yfinance',
                        updated_at DATETIME NOT NULL,
                        UNIQUE (ticker, tag, start, "end", filed, source)
                    )
                """))
                await conn.execute(text("""
                    INSERT INTO fundamentals_new
                        (id, ticker, tag, start, "end", filed, val, currency, source, updated_at)
                    SELECT id, ticker, tag, COALESCE(start, ''), "end", filed, val, currency,
                           source, updated_at
                    FROM fundamentals f
                    WHERE NOT EXISTS (
                        SELECT 1 FROM fundamentals g
                        WHERE g.ticker = f.ticker AND g.tag = f.tag
                          AND COALESCE(g.start, '') = COALESCE(f.start, '')
                          AND g."end" = f."end" AND g.filed = f.filed
                          AND g.source = f.source
                          AND (g.updated_at > f.updated_at
                               OR (g.updated_at = f.updated_at AND g.id > f.id))
                    )
                """))
                await conn.execute(text("DROP TABLE fundamentals"))
                await conn.execute(text("ALTER TABLE fundamentals_new RENAME TO fundamentals"))
                await conn.execute(text(
                    "CREATE INDEX ix_fundamentals_ticker ON fundamentals (ticker)"))
                await conn.execute(text(
                    "CREATE INDEX ix_fundamentals_tag ON fundamentals (tag)"))
        if need_rebuild:
            await conn.execute(text("""
                CREATE TABLE fundamentals_new (
                    id INTEGER NOT NULL PRIMARY KEY,
                    ticker VARCHAR(32) NOT NULL,
                    tag VARCHAR(64) NOT NULL,
                    start VARCHAR(10),
                    "end" VARCHAR(10) NOT NULL,
                    filed VARCHAR(10) NOT NULL,
                    val FLOAT NOT NULL,
                    currency VARCHAR(8) NOT NULL DEFAULT 'USD',
                    source VARCHAR(16) NOT NULL DEFAULT 'yfinance',
                    updated_at DATETIME NOT NULL,
                    UNIQUE (ticker, tag, start, "end", filed, source)
                )
            """))
            # Old schemas (created before the EDGAR work on this branch) may
            # lack the `source` column — SELECT would fail on it, so detect
            # and substitute the literal default for those rows. When the
            # column is absent the source is identical for every row, so
            # that join condition drops out entirely.
            source_expr = "source" if "source" in cols else "'yfinance'"
            source_cond = "AND g.source = f.source" if "source" in cols else ""
            # start=NULL rows collide once normalized to '' (they were
            # duplicating silently) — keep the newest by updated_at.
            await conn.execute(text(f"""
                INSERT INTO fundamentals_new
                    (id, ticker, tag, start, "end", filed, val, currency, source, updated_at)
                SELECT id, ticker, tag, COALESCE(start, ''), "end", filed, val, currency,
                       {source_expr}, updated_at
                FROM fundamentals f
                WHERE NOT EXISTS (
                    SELECT 1 FROM fundamentals g
                    WHERE g.ticker = f.ticker AND g.tag = f.tag
                      AND COALESCE(g.start, '') = COALESCE(f.start, '')
                      AND g."end" = f."end" AND g.filed = f.filed
                      {source_cond}
                      AND (g.updated_at > f.updated_at
                           OR (g.updated_at = f.updated_at AND g.id > f.id))
                )
            """))
            await conn.execute(text("DROP TABLE fundamentals"))
            await conn.execute(text("ALTER TABLE fundamentals_new RENAME TO fundamentals"))
            await conn.execute(text(
                "CREATE INDEX ix_fundamentals_ticker ON fundamentals (ticker)"))
            await conn.execute(text(
                "CREATE INDEX ix_fundamentals_tag ON fundamentals (tag)"))
