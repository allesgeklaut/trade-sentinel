from datetime import datetime, timezone

from sqlalchemy import String, Float, Integer, UniqueConstraint, Text
from sqlalchemy import DateTime as _DateTime
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from .config import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
            return value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    pass


class Watchlist(Base):
    __tablename__ = "watchlist"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Candle(Base):
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


class SimAllowance(Base):
    """Monthly imaginary deposit log."""

    __tablename__ = "sim_allowances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float)
    month: Mapped[str] = mapped_column(String(7), unique=True)  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SimSnapshot(Base):
    """Equity-curve snapshot taken after each sim run."""

    __tablename__ = "sim_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_equity: Mapped[float] = mapped_column(Float)
    allowance_total: Mapped[float] = mapped_column(Float, default=0)


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
    """

    __tablename__ = "fundamentals"
    __table_args__ = (UniqueConstraint("ticker", "tag", "start", "end"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    tag: Mapped[str] = mapped_column(String(64), index=True)
    start: Mapped[str | None] = mapped_column(String(10), nullable=True)  # YYYY-MM-DD, None = instant fact
    end: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD period end
    filed: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD public date
    val: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MonthlyAccount(Base):
    """Singleton row (id=1) tracking the monthly portfolio cash balance."""

    __tablename__ = "monthly_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, default=0)
    last_allowance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)  # YYYY-MM
    last_rebalance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


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


class MonthlyAllowance(Base):
    """Monthly imaginary deposit log for the monthly portfolio."""

    __tablename__ = "monthly_allowances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    amount: Mapped[float] = mapped_column(Float)
    month: Mapped[str] = mapped_column(String(7), unique=True)  # YYYY-MM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MonthlySnapshot(Base):
    """Equity-curve snapshot taken after each monthly rebalance."""

    __tablename__ = "monthly_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_equity: Mapped[float] = mapped_column(Float)
    allowance_total: Mapped[float] = mapped_column(Float, default=0)


class MonthlyRebalance(Base):
    """Audit log: one row per monthly rebalance decision.

    ``picked`` / ``held_before`` are comma-joined ticker lists; ``snapshot``
    holds the JSON eligibility frame (factor values per ticker) so every
    decision is explainable after the fact, like stockstrat's holdings.csv.
    """

    __tablename__ = "monthly_rebalances"
    __table_args__ = (UniqueConstraint("rebal_month",),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rebal_month: Mapped[str] = mapped_column(String(7))  # YYYY-MM (decision month)
    rebal_date: Mapped[datetime] = mapped_column(DateTime)  # decision trading day
    held_before: Mapped[str] = mapped_column(Text, default="")
    picked: Mapped[str] = mapped_column(Text, default="")
    n_new: Mapped[int] = mapped_column(Integer, default=0)
    snapshot: Mapped[str] = mapped_column(Text, default="")  # JSON eligibility frame
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)



engine = create_async_engine(settings.database_url)
Session = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Manual migration for screener_results: add action/strength columns
        # for existing DBs created before these columns existed. SQLAlchemy's
        # create_all does not alter existing tables. Idempotent: sqlite raises
        # OperationalError if the column already exists.
        from sqlalchemy import text
        for col, coltype in [("action", "TEXT"), ("strength", "REAL")]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE screener_results ADD COLUMN {col} {coltype}")
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
        # Migration for sim_account: add last_review_week for calendar-based
        # weekly LLM portfolio reviews.
        try:
            await conn.execute(
                text("ALTER TABLE sim_account ADD COLUMN last_review_week VARCHAR(8)")
            )
        except Exception:
            pass  # column already exists