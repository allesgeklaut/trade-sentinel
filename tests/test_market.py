"""Tests for app.market: refresh_many batching (no network).

All tests monkeypatch app.market.refresh so no provider is contacted.
"""

from __future__ import annotations

import asyncio

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
