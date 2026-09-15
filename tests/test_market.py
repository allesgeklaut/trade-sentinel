"""Tests for app.market: refresh_many batching (no network).

All tests monkeypatch app.market.refresh so no provider is contacted.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app import market

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
        monkeypatch.setattr(market, "refresh", fake)

        refreshed, errors = await market.refresh_many(["A", "B", "C"])

        assert refreshed == ["A", "B", "C"]
        assert errors == []
        assert sorted(fake.calls) == ["A", "B", "C"]

    async def test_isolates_failures(self, monkeypatch):
        fake = FakeRefresh(fail={"BAD"})
        monkeypatch.setattr(market, "refresh", fake)

        refreshed, errors = await market.refresh_many(["GOOD1", "BAD", "GOOD2"])

        assert refreshed == ["GOOD1", "GOOD2"]
        assert errors == ["BAD: boom"]

    async def test_dedupes_input_preserving_order(self, monkeypatch):
        fake = FakeRefresh()
        monkeypatch.setattr(market, "refresh", fake)

        refreshed, errors = await market.refresh_many(["A", "A", "B", "A"])

        assert refreshed == ["A", "B"]
        assert fake.calls == ["A", "B"]
        assert errors == []

    async def test_clamps_invalid_concurrency(self, monkeypatch):
        fake = FakeRefresh()
        monkeypatch.setattr(market, "refresh", fake)

        for bad in (0, -3):
            refreshed, errors = await market.refresh_many(["A", "B"], concurrency=bad)
            assert refreshed == ["A", "B"]
        # Two invocations × two tickers — clamping must not drop any.
        assert sorted(fake.calls) == ["A", "A", "B", "B"]

    async def test_empty_input(self, monkeypatch):
        fake = FakeRefresh()
        monkeypatch.setattr(market, "refresh", fake)

        refreshed, errors = await market.refresh_many([])
        assert refreshed == []
        assert errors == []
        assert fake.calls == []

    async def test_on_result_fires_once_per_ticker(self, monkeypatch):
        fake = FakeRefresh(fail={"BAD"})
        monkeypatch.setattr(market, "refresh", fake)

        events: list[tuple[str, bool, str | None]] = []

        def on_result(ticker, ok, err):
            events.append((ticker, ok, err))

        await market.refresh_many(["OK1", "BAD"], on_result=on_result)

        assert sorted(events) == [("BAD", False, "boom"), ("OK1", True, None)]

    async def test_passes_period_through(self, monkeypatch):
        seen_periods: list[str] = []

        async def fake_refresh(ticker, period="2y"):
            seen_periods.append(period)

        monkeypatch.setattr(market, "refresh", fake_refresh)

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

        monkeypatch.setattr(market, "refresh", fake_refresh)

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

