"""Tests for app.market: refresh_many batching (no network).

All tests monkeypatch app.market.refresh so no provider is contacted.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app import market


@pytest.fixture(autouse=True)
def _reset_market_state(monkeypatch):
    """Fresh limiter + fresh map per test; no per-minute sleeps in tests."""
    monkeypatch.setattr(market.settings, "twelve_data_max_per_min", 100000)
    monkeypatch.setattr(market.settings, "twelve_data_daily_budget", 100000)
    monkeypatch.setattr(market, "_twelve_limiter", market._TwelveLimiter())
    market._fresh_at.clear()
    yield
    market._fresh_at.clear()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRefresh:
    """Recording stand-in for market.refresh with controllable behavior."""

    def __init__(self, fail: set[str] | None = None, stagger: bool = False):
        self.calls: list[str] = []
        self.fail = fail or set()
        self.stagger = stagger

    async def __call__(self, ticker, period="2y"):
        if self.stagger:
            # Later tickers finish first (reverse completion order) to prove
            # refresh_many reports results in INPUT order, not completion order.
            delay = max(0.0, 0.03 * (10 - self.calls.__len__()))
            await asyncio.sleep(delay)
        self.calls.append(ticker)
        if ticker in self.fail:
            raise ValueError("boom")


# ---------------------------------------------------------------------------
# refresh_many
# ---------------------------------------------------------------------------

class TestRefreshMany:
    async def test_success_returns_input_order(self, monkeypatch):
        fake = FakeRefresh(stagger=True)
        monkeypatch.setattr(market, "refresh_yfinance", fake)

        refreshed, errors = await market.refresh_many(["A", "B", "C"])

        assert refreshed == ["A", "B", "C"]
        assert errors == []
        assert sorted(fake.calls) == ["A", "B", "C"]

    async def test_isolates_failures(self, monkeypatch):
        fake = FakeRefresh(fail={"BAD"})
        monkeypatch.setattr(market, "refresh_yfinance", fake)

        refreshed, errors = await market.refresh_many(["GOOD1", "BAD", "GOOD2"])

        assert refreshed == ["GOOD1", "GOOD2"]
        assert errors == ["BAD: boom"]

    async def test_dedupes_input_preserving_order(self, monkeypatch):
        fake = FakeRefresh()
        monkeypatch.setattr(market, "refresh_yfinance", fake)

        refreshed, errors = await market.refresh_many(["A", "A", "B", "A"])

        assert refreshed == ["A", "B"]
        assert fake.calls == ["A", "B"]
        assert errors == []

    async def test_clamps_invalid_concurrency(self, monkeypatch):
        fake = FakeRefresh()
        monkeypatch.setattr(market, "refresh_yfinance", fake)

        for bad in (0, -3):
            refreshed, errors = await market.refresh_many(["A", "B"], concurrency=bad)
            assert refreshed == ["A", "B"]
        # Two invocations × two tickers — clamping must not drop any.
        assert sorted(fake.calls) == ["A", "A", "B", "B"]

    async def test_empty_input(self, monkeypatch):
        fake = FakeRefresh()
        monkeypatch.setattr(market, "refresh_yfinance", fake)

        refreshed, errors = await market.refresh_many([])
        assert refreshed == []
        assert errors == []
        assert fake.calls == []

    async def test_on_result_fires_once_per_ticker(self, monkeypatch):
        fake = FakeRefresh(fail={"BAD"})
        monkeypatch.setattr(market, "refresh_yfinance", fake)

        events: list[tuple[str, bool, str | None]] = []

        def on_result(ticker, ok, err):
            events.append((ticker, ok, err))

        await market.refresh_many(["OK1", "BAD"], on_result=on_result)

        assert sorted(events) == [("BAD", False, "boom"), ("OK1", True, None)]

    async def test_passes_period_through(self, monkeypatch):
        seen_periods: list[str] = []

        async def fake_refresh(ticker, period="2y"):
            seen_periods.append(period)

        monkeypatch.setattr(market, "refresh_yfinance", fake_refresh)

        await market.refresh_many(["A", "B"], period="10y")
        assert seen_periods == ["10y", "10y"]

    async def test_concurrency_bounds_parallelism(self, monkeypatch):
        """Never more than `concurrency` refreshes in flight at once."""
        running = 0
        max_running = 0

        async def fake_refresh(ticker, period="2y"):
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.02)
            running -= 1

        monkeypatch.setattr(market, "refresh_yfinance", fake_refresh)

        await market.refresh_many([f"T{i}" for i in range(10)], concurrency=2)
        assert max_running <= 2


# ---------------------------------------------------------------------------
# PERIOD_COUNTS constant
# ---------------------------------------------------------------------------

def test_period_counts_matches_previous_inline_dict():
    assert market.PERIOD_COUNTS == {"6m": 126, "2y": 504, "5y": 1260,
                                    "10y": 2520, "max": 5000}


# ---------------------------------------------------------------------------
# Provider selection (auto = Twelve Data when a key is set, else yfinance)
# ---------------------------------------------------------------------------

class TestProviderSelection:
    def _set(self, monkeypatch, provider: str, key: str) -> None:
        monkeypatch.setattr(market.settings, "market_data_provider", provider)
        monkeypatch.setattr(market.settings, "twelve_data_api_key", key)

    def test_auto_without_key_is_yfinance(self, monkeypatch):
        self._set(monkeypatch, "auto", "")
        assert market.provider() == "yfinance"

    def test_auto_with_key_is_twelvedata(self, monkeypatch):
        self._set(monkeypatch, "auto", "secret")
        assert market.provider() == "twelvedata"

    def test_yfinance_forces_yfinance_even_with_key(self, monkeypatch):
        self._set(monkeypatch, "yfinance", "secret")
        assert market.provider() == "yfinance"

    def test_twelvedata_requires_key(self, monkeypatch):
        self._set(monkeypatch, "twelvedata", "")
        with pytest.raises(ValueError):
            market.provider()

    def test_unknown_value_rejected(self, monkeypatch):
        self._set(monkeypatch, "nonsense", "")
        with pytest.raises(ValueError):
            market.provider()


# ---------------------------------------------------------------------------
# Twelve Data key resolution (inline env var vs secret file)
# ---------------------------------------------------------------------------

class TestTwelveKeyFile:
    def _set(self, monkeypatch, key: str, key_file: str) -> None:
        monkeypatch.setattr(market.settings, "twelve_data_api_key", key)
        monkeypatch.setattr(market.settings, "twelve_data_api_key_file", key_file)

    def test_env_key_wins(self, monkeypatch, tmp_path):
        f = tmp_path / "k"
        f.write_text("from-file\n")
        self._set(monkeypatch, "from-env", str(f))
        assert market._twelve_key() == "from-env"

    def test_file_used_when_env_empty(self, monkeypatch, tmp_path):
        f = tmp_path / "k"
        f.write_text("from-file\n")
        self._set(monkeypatch, "", str(f))
        assert market._twelve_key() == "from-file"

    def test_missing_file_is_empty(self, monkeypatch, tmp_path):
        self._set(monkeypatch, "", str(tmp_path / "nope"))
        assert market._twelve_key() == ""

    def test_provider_uses_file_key(self, monkeypatch, tmp_path):
        f = tmp_path / "k"
        f.write_text("from-file\n")
        monkeypatch.setattr(market.settings, "market_data_provider", "auto")
        self._set(monkeypatch, "", str(f))
        assert market.provider() == "twelvedata"


# ---------------------------------------------------------------------------
# search: always Yahoo (canonical symbols), regardless of candle provider
# ---------------------------------------------------------------------------

class TestSearchUsesYahoo:
    async def test_search_uses_yahoo_even_with_twelvedata(self, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")

        yahoo_calls: list[str] = []

        def fake_yahoo(q):
            yahoo_calls.append(q)
            return [{"symbol": "IFX.DE", "name": "Infineon", "exchange": "XETRA",
                     "country": "Germany", "type": "Equity"}]

        monkeypatch.setattr(market, "yahoo_search", fake_yahoo)

        out = await market.search("ifx.de")
        assert yahoo_calls == ["ifx.de"]
        assert out[0]["symbol"] == "IFX.DE"


# ---------------------------------------------------------------------------
# refresh: Twelve Data primary, per-ticker yfinance fallback
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal async context manager standing in for the SQLAlchemy Session."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *args, **kwargs):
        return None

    async def commit(self):
        return None


def _one_bar(ticker):
    return [{"timestamp": datetime(2026, 1, 2), "open": 1.0, "high": 1.0,
             "low": 1.0, "close": 1.0, "volume": 0.0}]


class TestRefreshFallback:
    async def test_twelve_success_skips_yfinance(self, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")
        monkeypatch.setattr(market, "Session", lambda: _FakeSession())

        async def fake_twelve(ticker, outputsize):
            return _one_bar(ticker)

        def boom(*a, **k):
            raise AssertionError("yfinance must not be called on Twelve success")

        monkeypatch.setattr(market, "_twelve_history", fake_twelve)
        monkeypatch.setattr(market, "yahoo_history", boom)

        await market.refresh("AAPL", "2y")  # must not raise

    async def test_twelve_failure_falls_back_to_yfinance(self, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")
        monkeypatch.setattr(market, "Session", lambda: _FakeSession())

        async def boom(ticker, outputsize):
            raise ValueError("symbol not found")

        yahoo_calls: list[tuple[str, str]] = []

        def fake_yahoo(ticker, period):
            yahoo_calls.append((ticker, period))
            return _one_bar(ticker)

        monkeypatch.setattr(market, "_twelve_history", boom)
        monkeypatch.setattr(market, "yahoo_history", fake_yahoo)

        await market.refresh("NEIN", "2y")
        assert yahoo_calls == [("NEIN", "2y")]

    async def test_yfinance_provider_never_calls_twelve(self, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "yfinance")
        monkeypatch.setattr(market, "Session", lambda: _FakeSession())

        async def boom(*a, **k):
            raise AssertionError("Twelve Data must not be called when provider=yfinance")

        monkeypatch.setattr(market, "_twelve_history", boom)
        monkeypatch.setattr(market, "yahoo_history", lambda t, p: _one_bar(t))

        await market.refresh("AAPL", "2y")  # must not raise


class TestRefreshYfinance:
    async def test_forces_yahoo_even_with_twelvedata_provider(self, monkeypatch):
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")
        monkeypatch.setattr(market, "Session", lambda: _FakeSession())

        async def boom(*a, **k):
            raise AssertionError("bulk refresh must not call the metered provider")

        periods: list[str] = []

        def fake_yahoo(ticker, period):
            periods.append(period)
            return _one_bar(ticker)

        monkeypatch.setattr(market, "_twelve_history", boom)
        monkeypatch.setattr(market, "yahoo_history", fake_yahoo)

        await market.refresh_yfinance("AAPL", "10y")
        assert periods == ["10y"]

    async def test_rejects_unknown_period(self, monkeypatch):
        with pytest.raises(ValueError):
            await market.refresh_yfinance("AAPL", "bogus")


# ---------------------------------------------------------------------------
# Twelve Data limiter + budget
# ---------------------------------------------------------------------------

class TestTwelveLimiter:
    async def test_budget_is_a_hard_cap(self, monkeypatch):
        monkeypatch.setattr(market.settings, "twelve_data_daily_budget", 2)
        monkeypatch.setattr(market.settings, "twelve_data_max_per_min", 100000)
        lim = market._TwelveLimiter()
        assert await lim.acquire() is True
        assert await lim.acquire() is True
        assert await lim.acquire() is False


class TestRefreshBudget:
    async def _setup(self, monkeypatch, lim):
        monkeypatch.setattr(market, "_twelve_limiter", lim)
        monkeypatch.setattr(market, "provider", lambda: "twelvedata")
        monkeypatch.setattr(market, "Session", lambda: _FakeSession())

    async def test_uses_twelve_when_budget_available(self, monkeypatch):
        monkeypatch.setattr(market.settings, "twelve_data_daily_budget", 100)
        monkeypatch.setattr(market.settings, "twelve_data_max_per_min", 100000)
        await self._setup(monkeypatch, market._TwelveLimiter())

        async def fake_twelve(ticker, outputsize):
            return _one_bar(ticker)

        def boom(*a, **k):
            raise AssertionError("yfinance must not be used while budget remains")

        monkeypatch.setattr(market, "_twelve_history", fake_twelve)
        monkeypatch.setattr(market, "yahoo_history", boom)

        await market.refresh("AAPL", "2y")  # must not raise

    async def test_spent_budget_falls_back_to_yfinance(self, monkeypatch):
        monkeypatch.setattr(market.settings, "twelve_data_daily_budget", 1)
        monkeypatch.setattr(market.settings, "twelve_data_max_per_min", 100000)
        lim = market._TwelveLimiter()
        assert await lim.acquire() is True  # spend the only credit
        await self._setup(monkeypatch, lim)

        async def boom(*a, **k):
            raise AssertionError("Twelve Data must not be called once the budget is spent")

        yahoo_calls: list[str] = []
        monkeypatch.setattr(market, "_twelve_history", boom)
        monkeypatch.setattr(market, "yahoo_history",
                            lambda t, p: yahoo_calls.append(t) or _one_bar(t))

        await market.refresh("AAPL", "2y")
        assert yahoo_calls == ["AAPL"]


# ---------------------------------------------------------------------------
# refresh_many: freshness skip (reuse the nightly prefetch)
# ---------------------------------------------------------------------------

class TestRefreshManyFreshSkip:
    async def test_skips_recently_fetched_tickers(self, monkeypatch):
        fetched: list[str] = []

        async def fake_yahoo(ticker, period):
            fetched.append(ticker)

        monkeypatch.setattr(market, "refresh_yfinance", fake_yahoo)
        market._mark_fresh("A")  # A was already fetched this window

        refreshed, errors = await market.refresh_many(["A", "B"], max_age_seconds=3600)

        assert fetched == ["B"]          # A skipped
        assert refreshed == ["A", "B"]   # both usable
        assert errors == []

    async def test_no_skip_without_max_age(self, monkeypatch):
        fetched: list[str] = []

        async def fake_yahoo(ticker, period):
            fetched.append(ticker)

        monkeypatch.setattr(market, "refresh_yfinance", fake_yahoo)
        market._mark_fresh("A")

        await market.refresh_many(["A", "B"])
        assert fetched == ["A", "B"]

    async def test_use_provider_routes_to_configured_refresh(self, monkeypatch):
        seen: list[str] = []

        async def fake_refresh(ticker, period):
            seen.append(ticker)

        monkeypatch.setattr(market, "refresh", fake_refresh)
        await market.refresh_many(["A", "B"], use_provider=True)
        assert seen == ["A", "B"]


