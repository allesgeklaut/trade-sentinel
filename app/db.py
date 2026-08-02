from datetime import datetime
from sqlalchemy import String, Float, DateTime, Integer, UniqueConstraint, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from .config import settings
class Base(DeclarativeBase): pass
class Watchlist(Base):
    __tablename__="watchlist"; ticker: Mapped[str]=mapped_column(String(16), primary_key=True); created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
class Candle(Base):
    __tablename__="candles"; __table_args__=(UniqueConstraint("ticker","timestamp"),)
    id: Mapped[int]=mapped_column(Integer, primary_key=True); ticker: Mapped[str]=mapped_column(String(16), index=True); timestamp: Mapped[datetime]=mapped_column(DateTime); open: Mapped[float]=mapped_column(Float); high: Mapped[float]=mapped_column(Float); low: Mapped[float]=mapped_column(Float); close: Mapped[float]=mapped_column(Float); volume: Mapped[float]=mapped_column(Float, default=0)
class Signal(Base):
    __tablename__="signals"; id: Mapped[int]=mapped_column(Integer, primary_key=True); ticker: Mapped[str]=mapped_column(String(16), index=True); created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow); action: Mapped[str]=mapped_column(String(8)); reason: Mapped[str]=mapped_column(Text); snapshot: Mapped[str]=mapped_column(Text)
engine=create_async_engine(settings.database_url); Session=async_sessionmaker(engine, expire_on_commit=False)
async def init_db():
    async with engine.begin() as conn: await conn.run_sync(Base.metadata.create_all)
