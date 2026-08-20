"""Tests for app.screener: universe parsing, dedup, and scorer thresholds.

Uses an in-memory SQLite DB (same pattern as test_sim.py) so screener
result persistence can be exercised without touching the real volume.
"""

from __future__ import annotations

import pytest
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app import screener
from app.db import Base, ScreenerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """Replace the screener's Session with an in-memory SQLite DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(screener, "Session", session_factory)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)

    yield session_factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Universe parsing
# ---------------------------------------------------------------------------

class TestUniverseParsing:
    def test_tickers_strips_whitespace_and_uppercases(self, tmp_path, monkeypatch):
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "test.txt").write_text(
            "  aapl  \n# comment\nMSFT\n\n  nvda \n"
        )
        result = screener.tickers("test")
        assert result == ["AAPL", "MSFT", "NVDA"]

    def test_tickers_skips_comments_and_blanks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "u.txt").write_text("# header\n\n   \n# another\nTSLA\n")
        assert screener.tickers("u") == ["TSLA"]

    def test_tickers_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        with pytest.raises(ValueError, match="Unknown universe"):
            screener.tickers("../etc/passwd")

    def test_tickers_rejects_slash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        with pytest.raises(ValueError, match="Unknown universe"):
            screener.tickers("sub/dir")

    def test_tickers_rejects_missing_universe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        with pytest.raises(ValueError, match="Unknown universe"):
            screener.tickers("nonexistent")

    def test_universe_names_lists_stems(self, tmp_path, monkeypatch):
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "alpha.txt").write_text("A\n")
        (tmp_path / "beta.txt").write_text("B\n")
        (tmp_path / "ignore.md").write_text("nope")
        names = screener.universe_names()
        assert names == ["alpha", "beta"]


class TestDedup:
    async def test_run_deduplicates_symbols_preserving_order(self, tmp_path, monkeypatch, mem_db):
        """A universe with duplicate symbols should be deduped before processing."""
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "dup.txt").write_text("AAPL\nMSFT\nAAPL\nNVDA\nMSFT\n")

        # Mock refresh + candles + score to avoid network
        async def mock_refresh(ticker, period="2y"):
            pass

        async def mock_candles(ticker, period=None):
            return [{"close": 100.0, "volume": 1.0}] * 70

        def mock_score(rows):
            return {"score": 50.0, "trend": "NEUTRAL", "return_20d": 0.0,
                    "return_60d": 0.0, "rsi": 50.0, "relative_volume": 1.0,
                    "close": 100.0}

        monkeypatch.setattr(screener, "refresh", mock_refresh)
        monkeypatch.setattr(screener, "candles", mock_candles)
        monkeypatch.setattr(screener, "score", mock_score)

        result = await screener.run("dup")
        assert result["processed"] == 3  # AAPL, MSFT, NVDA — deduped
        assert result["ranked"] == 3

        # Confirm no (universe, ticker) uniqueness violation in the DB
        async with mem_db() as s:
            rows = (await s.scalars(
                __import__("sqlalchemy").select(ScreenerResult).where(
                    ScreenerResult.universe == "dup"
                ).order_by(ScreenerResult.ticker)
            )).all()
            assert len(rows) == 3
            assert sorted(r.ticker for r in rows) == ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# Scorer thresholds
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_returns_none_for_short_history(self):
        rows = [{"close": 100.0, "volume": 1.0} for _ in range(64)]
        assert screener.score(rows) is None

    def test_score_returns_dict_at_threshold(self):
        """65 rows is the minimum; score should return a dict, not None."""
        rows = [{"close": 100.0 + 0.1 * i, "volume": 1_000_000.0} for i in range(65)]
        out = screener.score(rows)
        assert out is not None
        assert "score" in out and "trend" in out and "rsi" in out

    def test_score_keys_shape(self):
        rows = [{"close": 100.0 + 0.1 * i, "volume": 1_000_000.0} for i in range(250)]
        out = screener.score(rows)
        assert set(out.keys()) == {
            "score", "trend", "return_20d", "return_60d",
            "rsi", "relative_volume", "close",
        }


# ---------------------------------------------------------------------------
# run() error handling
# ---------------------------------------------------------------------------

class TestRunErrorHandling:
    async def test_run_continues_on_ticker_failure(self, tmp_path, monkeypatch, mem_db):
        """A failing ticker should be skipped, not abort the whole screen run."""
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "mix.txt").write_text("GOOD\nBAD\n")

        call_count = {"n": 0}

        async def mock_refresh(ticker, period="2y"):
            pass

        async def mock_candles(ticker, period=None):
            call_count["n"] += 1
            if ticker == "BAD":
                raise ValueError("simulated fetch failure")
            from datetime import datetime, timedelta
            base = datetime(2022, 1, 3)
            return [
                {
                    "timestamp": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": 100.0 + 0.1 * i,
                    "high": 100.5 + 0.1 * i,
                    "low": 99.5 + 0.1 * i,
                    "close": 100.0 + 0.1 * i,
                    "volume": 1_000_000.0,
                }
                for i in range(250)
            ]

        monkeypatch.setattr(screener, "refresh", mock_refresh)
        monkeypatch.setattr(screener, "candles", mock_candles)

        result = await screener.run("mix")
        assert result["processed"] == 2
        assert result["ranked"] == 1  # only GOOD produced a score
        assert call_count["n"] == 2  # both tickers were attempted

        async with mem_db() as s:
            from sqlalchemy import select as sa_select
            rows = (await s.scalars(
                sa_select(ScreenerResult).where(ScreenerResult.universe == "mix")
            )).all()
            assert len(rows) == 1
            assert rows[0].ticker == "GOOD"

    async def test_run_clears_running_flag_on_failure(self, tmp_path, monkeypatch, mem_db):
        """If the run aborts (e.g. commit failure), the progress state must be
        cleared with an error so the frontend poller doesn't freeze at
        "Updating N/M" forever."""
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "boom.txt").write_text("AAPL\n")

        async def mock_refresh(ticker, period="2y"):
            pass

        async def mock_candles(ticker, period=None):
            from datetime import datetime, timedelta
            base = datetime(2022, 1, 3)
            return [
                {
                    "timestamp": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": 100.0, "high": 100.5, "low": 99.5,
                    "close": 100.0, "volume": 1_000_000.0,
                }
                for i in range(250)
            ]

        monkeypatch.setattr(screener, "refresh", mock_refresh)
        monkeypatch.setattr(screener, "candles", mock_candles)

        # Break the DB session so the final commit raises.
        class BoomSession:
            def __init__(self, *a, **k):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def execute(self, *a, **k):
                raise RuntimeError("simulated commit failure")
            async def add(self, *a, **k):
                pass
            async def commit(self, *a, **k):
                raise RuntimeError("simulated commit failure")

        monkeypatch.setattr(screener, "Session", BoomSession)

        with pytest.raises(RuntimeError, match="simulated commit failure"):
            await screener.run("boom")

        progress = screener.get_screener_progress()
        assert progress["running"] is False
        assert progress["error"] == "simulated commit failure"
        assert progress["op"] == "update"


# ---------------------------------------------------------------------------
# Signal column (action / strength from analysis.compute)
# ---------------------------------------------------------------------------

class TestSignalColumn:
    async def test_run_populates_action_for_full_history(self, tmp_path, monkeypatch, mem_db):
        """A ticker with >=206 candles should get a BUY/SELL/HOLD action + strength."""
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "full.txt").write_text("AAPL\n")

        async def mock_refresh(ticker, period="2y"):
            pass

        async def mock_candles(ticker, period=None):
            # 250 rows is enough for compute() (needs >=206). Include timestamps
            # because the weekly trend filter parses the time column.
            from datetime import datetime, timedelta
            base = datetime(2022, 1, 3)
            return [
                {
                    "timestamp": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": 100.0 + 0.1 * i,
                    "high": 100.5 + 0.1 * i,
                    "low": 99.5 + 0.1 * i,
                    "close": 100.0 + 0.1 * i,
                    "volume": 1_000_000.0,
                }
                for i in range(250)
            ]

        monkeypatch.setattr(screener, "refresh", mock_refresh)
        monkeypatch.setattr(screener, "candles", mock_candles)

        result = await screener.run("full")
        assert result["ranked"] == 1

        async with mem_db() as s:
            from sqlalchemy import select as sa_select
            row = await s.scalar(sa_select(ScreenerResult).where(ScreenerResult.universe == "full"))
            assert row is not None
            assert row.action in ("BUY", "SELL", "HOLD")
            assert row.strength is not None
            assert 0 <= row.strength <= 100

    async def test_run_marks_na_for_short_history(self, tmp_path, monkeypatch, mem_db):
        """A ticker with <206 candles (e.g. recent IPO) should get action='N/A'."""
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "ipo.txt").write_text("NEWIPO\n")

        async def mock_refresh(ticker, period="2y"):
            pass

        async def mock_candles(ticker, period=None):
            # 100 rows is enough for score() (>=65) but not compute() (>=206)
            return [{"close": 100.0 + 0.1 * i, "volume": 1_000_000.0} for i in range(100)]

        monkeypatch.setattr(screener, "refresh", mock_refresh)
        monkeypatch.setattr(screener, "candles", mock_candles)

        result = await screener.run("ipo")
        assert result["ranked"] == 1

        async with mem_db() as s:
            from sqlalchemy import select as sa_select
            row = await s.scalar(sa_select(ScreenerResult).where(ScreenerResult.universe == "ipo"))
            assert row is not None
            assert row.action == "N/A"
            assert row.strength is None

    async def test_results_api_includes_action_and_strength(self, tmp_path, monkeypatch, mem_db):
        """The results() API response must include action and strength fields."""
        monkeypatch.setattr(screener, "_UNIVERSES_DIR", tmp_path)
        (tmp_path / "api.txt").write_text("AAPL\n")

        async def mock_refresh(ticker, period="2y"):
            pass

        async def mock_candles(ticker, period=None):
            from datetime import datetime, timedelta
            base = datetime(2022, 1, 3)
            return [
                {
                    "timestamp": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": 100.0 + 0.1 * i,
                    "high": 100.5 + 0.1 * i,
                    "low": 99.5 + 0.1 * i,
                    "close": 100.0 + 0.1 * i,
                    "volume": 1_000_000.0,
                }
                for i in range(250)
            ]

        monkeypatch.setattr(screener, "refresh", mock_refresh)
        monkeypatch.setattr(screener, "candles", mock_candles)

        await screener.run("api")
        out = await screener.results("api")
        assert len(out) == 1
        assert "action" in out[0]
        assert "strength" in out[0]
        assert out[0]["action"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# init_db() ALTER TABLE migration guard (idempotent)
# ---------------------------------------------------------------------------

class TestMigrationGuard:
    async def test_init_db_idempotent_with_new_columns(self, monkeypatch):
        """init_db() must be safe to call twice without error, even with the
        new action/strength columns. The ALTER TABLE guard should no-op on
        the second call when the columns already exist."""
        from sqlalchemy import StaticPool
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        import app.db as db_mod

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(db_mod, "engine", engine)
        monkeypatch.setattr(db_mod, "Session", session_factory)

        # First call creates tables + adds columns
        await db_mod.init_db()
        # Second call should not raise (columns already exist)
        await db_mod.init_db()

        # Verify the columns exist by inserting a row with action/strength
        async with session_factory() as s:
            from app.db import ScreenerResult
            from datetime import datetime, timezone
            s.add(ScreenerResult(
                universe="test", ticker="X", score=50.0, trend="NEUTRAL",
                return_20d=0.0, return_60d=0.0, rsi=50.0, relative_volume=1.0,
                close=100.0, updated_at=datetime.now(timezone.utc),
                action="HOLD", strength=42.0,
            ))
            await s.commit()

        await engine.dispose()