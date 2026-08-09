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
            return [{"close": 100.0 + 0.1 * i, "volume": 1_000_000.0} for i in range(250)]

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