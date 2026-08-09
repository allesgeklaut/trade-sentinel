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


# =====================================================================
# Autonomous paper-trading simulation tables
# =====================================================================

class SimAccount(Base):
    """Singleton row (id=1) tracking the sim account balance."""

    __tablename__ = "sim_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, default=0)
    last_allowance_month: Mapped[str | None] = mapped_column(String(7), nullable=True)  # YYYY-MM
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



engine = create_async_engine(settings.database_url)
Session = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)