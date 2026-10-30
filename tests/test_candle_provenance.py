"""Tests for candle provenance: source, adjustment basis, ingestion time.

Cached rows used to lose where their OHLCV actually came from on every upsert.
These tests cover the Phase 2 provenance foundation:

* ``market._store_candles`` persists the metadata atomically with the OHLCV
  (insert AND conflict update), copies rows without mutating the caller's
  payload, and assigns the UTC ingestion time itself;
* direct legacy calls without metadata stay supported and honestly unknown
  (NULL source/basis) — provenance always describes the last write;
* ``market.refresh`` records the provider that ACTUALLY served the bars:
  forced Yahoo, Twelve Data success, Twelve Data failure -> Yahoo fallback,
  quota exhausted -> Yahoo fallback; ``refresh_yfinance`` is always Yahoo even
  when the configured provider is Twelve Data;
* rows cached before provenance tracking keep NULL metadata, and ``init_db``'s
  additive candle migration is idempotent and preserves legacy rows.

Every DB here is an in-memory SQLite built from the ORM metadata (or a
hand-shaped legacy table); the real project database is never touched, and
all providers are mocked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import StaticPool, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
from app import market
from app.db import Base, Candle

BASIS = market.ADJUSTMENT_SPLIT_DIVIDEND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(day: int, close: float = 1.0) -> dict:
    """One provider-shaped daily bar (plain dict, no provenance keys)."""
    return {
        "timestamp": datetime(2026, 1, day),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 100.0,
    }


def _many_bars(count: int, *, close: float = 1.0) -> list[dict]:
    """``count`` distinct daily bars — large enough to cross chunk boundaries."""
    base = datetime(2010, 1, 1)
    return [
        {
            "timestamp": base + timedelta(days=i),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 100.0 + i,
        }
        for i in range(count)
    ]


def _engine():
    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


async def _candles(session_factory, ticker: str = "AAPL") -> list[Candle]:
    async with session_factory() as s:
        return list(
            (await s.scalars(
                select(Candle).where(Candle.ticker == ticker).order_by(Candle.timestamp)
            )).all()
        )


@pytest.fixture
async def candle_db(monkeypatch):
    """In-memory candle DB + mocked provider plumbing (never /data)."""
    engine = _engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(market, "Session", session_factory)
    # Never sleep on a rate-limit slot / hit the day budget in refresh tests.
    monkeypatch.setattr(market.settings, "twelve_data_max_per_min", 100000)
    monkeypatch.setattr(market.settings, "twelve_data_daily_budget", 100000)
    monkeypatch.setattr(market, "_twelve_limiter", market._TwelveLimiter())
    market._fresh_at.clear()
    yield session_factory
    market._fresh_at.clear()
    await engine.dispose()


def _forbid_yahoo(*_a, **_k):
    raise AssertionError("yahoo_history must not be called on this path")


def _forbid_twelve(*_a, **_k):
    raise AssertionError("Twelve Data must not be called on this path")


# ---------------------------------------------------------------------------
# _store_candles: metadata travels with the OHLCV
# ---------------------------------------------------------------------------

class TestStoreCandlesProvenance:
    async def test_store_persists_ohlcv_and_metadata(self, candle_db):
        ingested = datetime(2026, 1, 5, 12, 30, 45, tzinfo=UTC)

        await market._store_candles(
            "AAPL",
            [_bar(2, close=101.5), _bar(3, close=102.5)],
            source=market.SOURCE_TWELVEDATA,
            adjustment_basis=BASIS,
            ingested_at=ingested,
        )

        rows = await _candles(candle_db)
        assert [r.close for r in rows] == [101.5, 102.5]
        assert all(r.source == market.SOURCE_TWELVEDATA for r in rows)
        assert all(r.adjustment_basis == BASIS for r in rows)
        assert all(r.ingested_at == ingested for r in rows)
        assert all(r.ingested_at.tzinfo is not None for r in rows)

    async def test_store_does_not_mutate_caller_payload(self, candle_db):
        payload = [_bar(2), _bar(3)]
        snapshot = [dict(row) for row in payload]

        await market._store_candles(
            "AAPL", payload,
            source=market.SOURCE_YFINANCE,
            adjustment_basis=BASIS,
        )

        # The provider rows are copied, never annotated in place.
        assert payload == snapshot
        assert all("source" not in row and "ingested_at" not in row for row in payload)

    async def test_row_without_volume_keeps_the_column_default(self, candle_db):
        """Regression: a normalized direct-call row may omit ``volume``.

        The old ``sqlite_insert(Candle).values([{'ticker', **x}])`` let the
        ``Candle.volume`` column default (0) apply for streams that don't
        carry volume; the explicit field copy must preserve that instead of
        raising KeyError.
        """
        row = {
            "timestamp": datetime(2026, 1, 2),
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
        }

        await market._store_candles(
            "AAPL", [row],
            source=market.SOURCE_YFINANCE, adjustment_basis=BASIS,
        )

        stored = (await _candles(candle_db))[0]
        assert (stored.open, stored.high, stored.low, stored.close) == (1.0, 2.0, 0.5, 1.5)
        assert stored.volume == 0  # column default, not fabricated data
        assert stored.source == market.SOURCE_YFINANCE
        assert stored.adjustment_basis == BASIS
        assert "volume" not in row  # the caller's row is still not mutated

    async def test_explicit_invalid_volume_is_not_silently_coerced(self, candle_db):
        """Only a MISSING volume key defaults; an explicit ``None`` must still
        fail the NOT NULL constraint exactly as the old insert did."""
        row = {**_bar(2), "volume": None}

        with pytest.raises(IntegrityError):
            await market._store_candles(
                "AAPL", [row], source=market.SOURCE_YFINANCE,
                adjustment_basis=BASIS,
            )

        assert await _candles(candle_db) == []

    async def test_storage_assigns_utc_ingestion_time(self, candle_db):
        before = datetime.now(UTC)
        await market._store_candles(
            "AAPL", [_bar(2)], source=market.SOURCE_YFINANCE,
            adjustment_basis=BASIS,
        )
        after = datetime.now(UTC)

        row = (await _candles(candle_db))[0]
        assert row.ingested_at is not None
        assert row.ingested_at.tzinfo is not None
        assert row.ingested_at.utcoffset() == timedelta(0)
        assert before - timedelta(seconds=1) <= row.ingested_at <= after + timedelta(seconds=1)

    async def test_repeated_store_updates_prices_and_provenance_together(self, candle_db):
        first = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        second = first + timedelta(hours=6)
        await market._store_candles(
            "AAPL", [_bar(2, close=100.0)],
            source=market.SOURCE_TWELVEDATA, adjustment_basis=BASIS, ingested_at=first,
        )
        await market._store_candles(
            "AAPL", [_bar(2, close=110.0)],
            source=market.SOURCE_YFINANCE, adjustment_basis=BASIS, ingested_at=second,
        )

        rows = await _candles(candle_db)
        assert len(rows) == 1  # conflict target upserted, no duplicate bar
        row = rows[0]
        # Price and provenance moved together — never new OHLC with stale tags.
        assert row.close == 110.0
        assert row.source == market.SOURCE_YFINANCE
        assert row.adjustment_basis == BASIS
        assert row.ingested_at == second

    async def test_empty_payload_is_a_noop(self, candle_db):
        await market._store_candles(
            "AAPL", [], source=market.SOURCE_TWELVEDATA, adjustment_basis=BASIS
        )
        assert await _candles(candle_db) == []


# ---------------------------------------------------------------------------
# Legacy direct calls: supported, honestly unknown
# ---------------------------------------------------------------------------

class TestUnknownProvenanceIsHonest:
    async def test_call_without_metadata_stores_unknown_source_and_basis(self, candle_db):
        await market._store_candles("AAPL", [_bar(2, close=7.0)])

        row = (await _candles(candle_db))[0]
        assert row.close == 7.0
        assert row.source is None
        assert row.adjustment_basis is None
        # The write itself was observed even though its origin was not.
        assert row.ingested_at is not None

    async def test_row_rewritten_without_metadata_does_not_claim_a_stale_source(self, candle_db):
        tagged_at = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        await market._store_candles(
            "AAPL", [_bar(2, close=100.0)],
            source=market.SOURCE_TWELVEDATA, adjustment_basis=BASIS,
            ingested_at=tagged_at,
        )
        await market._store_candles("AAPL", [_bar(2, close=200.0)])

        row = (await _candles(candle_db))[0]
        assert row.close == 200.0
        # The current content's origin is unknown; keeping 'twelvedata' would
        # assert provenance nobody observed for this write.
        assert row.source is None
        assert row.adjustment_basis is None
        assert row.ingested_at > tagged_at


# ---------------------------------------------------------------------------
# Large payloads: chunking under SQLite's per-statement parameter cap
# ---------------------------------------------------------------------------

class TestLargeBatchChunking:
    async def test_store_3650_bars_crosses_the_parameter_cap(self, candle_db):
        """A 10y Twelve Data payload (3,650 bars) has 36,500 host params.

        That exceeded SQLite's 32,766-parameter statement cap and made the
        whole refresh fail with zero rows stored; chunking must store every
        bar with its provenance.
        """
        ingested = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)

        await market._store_candles(
            "AAPL", _many_bars(3650, close=10.0),
            source=market.SOURCE_TWELVEDATA, adjustment_basis=BASIS,
            ingested_at=ingested,
        )

        rows = await _candles(candle_db)
        assert len(rows) == 3650
        assert len({r.timestamp for r in rows}) == 3650  # no duplicate bars
        assert all(r.close == 10.0 for r in rows)
        assert all(r.source == market.SOURCE_TWELVEDATA for r in rows)
        assert all(r.adjustment_basis == BASIS for r in rows)
        assert all(r.ingested_at is not None for r in rows)
        assert all(r.ingested_at == ingested for r in rows)

    async def test_restore_updates_prices_and_provenance_across_chunks(self, candle_db):
        first = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)
        second = first + timedelta(hours=12)
        await market._store_candles(
            "AAPL", _many_bars(3650, close=10.0),
            source=market.SOURCE_TWELVEDATA, adjustment_basis=BASIS,
            ingested_at=first,
        )
        await market._store_candles(
            "AAPL", _many_bars(3650, close=20.0),
            source=market.SOURCE_YFINANCE, adjustment_basis=BASIS,
            ingested_at=second,
        )

        rows = await _candles(candle_db)
        assert len(rows) == 3650  # ON CONFLICT upsert, not a second batch
        assert {r.close for r in rows} == {20.0}
        assert all(r.source == market.SOURCE_YFINANCE for r in rows)
        assert all(r.adjustment_basis == BASIS for r in rows)
        assert all(r.ingested_at == second for r in rows)

    async def test_mid_chunk_failure_rolls_back_the_whole_batch(self, candle_db):
        """Chunks share ONE transaction/commit: a failing later chunk must
        discard the rows an earlier chunk already wrote."""
        bars = _many_bars(market.CANDLE_UPSERT_MAX_ROWS + 100)
        bars[market.CANDLE_UPSERT_MAX_ROWS + 50]["volume"] = None  # chunk 2

        with pytest.raises(IntegrityError):
            await market._store_candles(
                "AAPL", bars,
                source=market.SOURCE_YFINANCE, adjustment_basis=BASIS,
            )

        assert await _candles(candle_db) == []


def test_chunk_constant_stays_under_the_sqlite_parameter_cap():
    """Derive the invariant, not just the number.

    Every Candle column except the autoincrement id is bound in the upsert.
    If a column is added, this fails and CANDLE_UPSERT_MAX_ROWS (and its
    arithmetic comment) must be re-derived — SQLite rejects statements with
    more than 32,766 host parameters.
    """
    bound_columns = len(Candle.__table__.columns) - 1  # id is never bound
    assert bound_columns == 10
    assert market.CANDLE_UPSERT_MAX_ROWS <= 2000
    assert market.CANDLE_UPSERT_MAX_ROWS * bound_columns <= 32766


# ---------------------------------------------------------------------------
# refresh / refresh_yfinance: tag the provider that actually served the bars
# ---------------------------------------------------------------------------

class TestRefreshProvenance:
    async def test_yfinance_provider_tags_yahoo(self, candle_db, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "yfinance")
        monkeypatch.setattr(market, "yahoo_history", lambda t, p: [_bar(2, close=11.0)])
        monkeypatch.setattr(market, "_twelve_history", _forbid_twelve)

        await market.refresh("AAPL", "2y")

        row = (await _candles(candle_db))[0]
        assert row.close == 11.0
        assert row.source == market.SOURCE_YFINANCE
        assert row.adjustment_basis == BASIS

    async def test_twelve_success_tags_twelvedata(self, candle_db, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")

        async def fake_twelve(ticker, outputsize):
            return [_bar(2, close=12.0)]

        monkeypatch.setattr(market, "_twelve_history", fake_twelve)
        monkeypatch.setattr(market, "yahoo_history", _forbid_yahoo)

        await market.refresh("AAPL", "2y")

        row = (await _candles(candle_db))[0]
        assert row.close == 12.0
        assert row.source == market.SOURCE_TWELVEDATA
        assert row.adjustment_basis == BASIS

    async def test_twelve_failure_tags_the_yahoo_fallback(self, candle_db, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")

        async def boom(*_a, **_k):
            raise ValueError("symbol not found")

        calls: list[tuple[str, str]] = []

        def fake_yahoo(ticker, period):
            calls.append((ticker, period))
            return [_bar(2, close=13.0)]

        monkeypatch.setattr(market, "_twelve_history", boom)
        monkeypatch.setattr(market, "yahoo_history", fake_yahoo)

        await market.refresh("NEIN", "2y")

        assert calls == [("NEIN", "2y")]
        row = (await _candles(candle_db, "NEIN"))[0]
        assert row.close == 13.0
        assert row.source == market.SOURCE_YFINANCE
        assert row.adjustment_basis == BASIS

    async def test_exhausted_quota_tags_the_yahoo_fallback(self, candle_db, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")
        monkeypatch.setattr(market.settings, "twelve_data_daily_budget", 1)
        limiter = market._TwelveLimiter()
        assert await limiter.acquire() is True  # spend the only credit
        monkeypatch.setattr(market, "_twelve_limiter", limiter)
        monkeypatch.setattr(market, "_twelve_history", _forbid_twelve)
        monkeypatch.setattr(market, "yahoo_history", lambda t, p: [_bar(2, close=14.0)])

        await market.refresh("AAPL", "2y")

        row = (await _candles(candle_db))[0]
        assert row.close == 14.0
        assert row.source == market.SOURCE_YFINANCE
        assert row.adjustment_basis == BASIS

    async def test_refresh_yfinance_is_always_yahoo_even_with_twelvedata_provider(
        self, candle_db, monkeypatch
    ):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")
        monkeypatch.setattr(market, "_twelve_history", _forbid_twelve)
        monkeypatch.setattr(market, "yahoo_history", lambda t, p: [_bar(2, close=15.0)])

        await market.refresh_yfinance("AAPL", "10y")

        row = (await _candles(candle_db))[0]
        assert row.close == 15.0
        assert row.source == market.SOURCE_YFINANCE
        assert row.adjustment_basis == BASIS

    async def test_chart_json_shape_is_unchanged(self, candle_db, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "yfinance")
        monkeypatch.setattr(market, "yahoo_history", lambda t, p: [_bar(2, close=5.0)])

        await market.refresh("AAPL", "2y")
        out = await market.candles("AAPL", "6m")

        assert out == [{
            "time": "2026-01-02",
            "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 100.0,
        }]


# ---------------------------------------------------------------------------
# Legacy schema migration: additive, idempotent, no fabricated provenance
# ---------------------------------------------------------------------------

_LEGACY_CANDLES_DDL = """
CREATE TABLE candles (
    id INTEGER NOT NULL PRIMARY KEY,
    ticker VARCHAR(32) NOT NULL,
    timestamp DATETIME NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    UNIQUE (ticker, timestamp)
)
"""


async def _legacy_db(monkeypatch):
    """Temp DB shaped like a pre-provenance deployment, with two cached rows."""
    engine = _engine()
    async with engine.begin() as conn:
        await conn.execute(text(_LEGACY_CANDLES_DDL))
        await conn.execute(text(
            "INSERT INTO candles (id, ticker, timestamp, open, high, low, close, volume) "
            "VALUES (1, 'AAPL', '2026-01-02 00:00:00.000000', 10, 11, 9, 10.5, 1000), "
            "       (2, 'AAPL', '2026-01-03 00:00:00.000000', 11, 12, 10, 11.5, 2000)"
        ))
    monkeypatch.setattr(db_mod, "engine", engine)
    return engine


class TestLegacyCandleMigration:
    async def test_init_db_adds_columns_and_preserves_legacy_rows(self, monkeypatch):
        engine = await _legacy_db(monkeypatch)
        try:
            await db_mod.init_db()

            async with engine.connect() as conn:
                cols = {
                    row[1]
                    for row in (await conn.execute(text("PRAGMA table_info(candles)"))).all()
                }
                raw = (await conn.execute(text(
                    "SELECT id, ticker, close, volume, source, adjustment_basis, ingested_at "
                    "FROM candles ORDER BY id"))).all()

            assert {"source", "adjustment_basis", "ingested_at"} <= cols
            assert [row[:4] for row in raw] == [
                (1, "AAPL", 10.5, 1000.0),
                (2, "AAPL", 11.5, 2000.0),
            ]
            # Never backfilled: the historical pulls were never observed.
            assert [row[4:] for row in raw] == [(None, None, None), (None, None, None)]

            # The ORM can read the migrated table and sees unknown provenance.
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            rows = await _candles(session_factory)
            assert [r.close for r in rows] == [10.5, 11.5]
            assert all(r.source is None and r.adjustment_basis is None for r in rows)
            assert all(r.ingested_at is None for r in rows)
        finally:
            await engine.dispose()

    async def test_init_db_is_idempotent_on_second_startup(self, monkeypatch):
        engine = await _legacy_db(monkeypatch)
        try:
            await db_mod.init_db()
            await db_mod.init_db()  # ALTERs must no-op, not fail or duplicate data

            async with engine.connect() as conn:
                count = (await conn.execute(text("SELECT COUNT(*) FROM candles"))).scalar()
                close = (await conn.execute(
                    text("SELECT close FROM candles WHERE id = 2"))).scalar()
            assert count == 2
            assert close == 11.5
        finally:
            await engine.dispose()

    async def test_migration_tags_nothing_but_a_real_refresh_tags_its_row(self, monkeypatch):
        engine = await _legacy_db(monkeypatch)
        try:
            await db_mod.init_db()
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            monkeypatch.setattr(market, "Session", session_factory)
            ingested = datetime(2026, 1, 6, 8, 0, tzinfo=UTC)

            # Refresh the 2026-01-03 bar only.
            await market._store_candles(
                "AAPL", [_bar(3, close=42.0)],
                source=market.SOURCE_TWELVEDATA, adjustment_basis=BASIS,
                ingested_at=ingested,
            )

            rows = await _candles(session_factory)
            untouched, refreshed = rows
            assert (untouched.close, untouched.source, untouched.ingested_at) == (10.5, None, None)
            assert refreshed.close == 42.0
            assert refreshed.source == market.SOURCE_TWELVEDATA
            assert refreshed.adjustment_basis == BASIS
            assert refreshed.ingested_at == ingested
        finally:
            await engine.dispose()

    async def test_init_db_on_fresh_db_creates_columns(self, monkeypatch):
        """A brand-new DB gets the columns from create_all — the ALTER guard
        must not fail on the table it just created (and re-running is safe)."""
        engine = _engine()
        monkeypatch.setattr(db_mod, "engine", engine)
        try:
            await db_mod.init_db()
            await db_mod.init_db()

            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as s:
                s.add(Candle(ticker="AAPL", timestamp=datetime(2026, 1, 2),
                             open=1, high=1, low=1, close=1, volume=1))
                await s.commit()

            rows = await _candles(session_factory)
            assert rows[0].source is None  # ORM inserts without metadata: unknown
            assert rows[0].ingested_at is None
        finally:
            await engine.dispose()
