"""Tests for app.news: SearXNG news search, caching, and graceful degradation.

All tests mock httpx.AsyncClient so no network is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import news

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the in-memory cache before each test."""
    news._cache.clear()
    yield
    news._cache.clear()


@pytest.fixture
def searxng_enabled(monkeypatch):
    monkeypatch.setattr(news.settings, "searxng_url", "http://localhost:8081")
    yield


def _mock_response(results=None):
    """Build a mock httpx.Response with the given SearXNG results."""
    resp = MagicMock()
    resp.json.return_value = {"results": results or []}
    resp.raise_for_status = MagicMock()
    return resp


def _sample_results():
    return [
        {"title": "Apple beats Q3 estimates", "content": "Apple reported strong earnings, beating analyst expectations.", "url": "https://example.com/1"},
        {"title": "AAPL hits new high", "content": "Shares reached a new 52-week high on strong demand.", "url": "https://example.com/2"},
        {"title": "Tim Cook announces new product", "content": "Apple CEO Tim Cook unveiled a new device line.", "url": "https://example.com/3"},
    ]


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------

class TestDisabled:
    async def test_returns_empty_when_disabled(self, monkeypatch):
        monkeypatch.setattr(news.settings, "searxng_url", "")
        result = await news.search_news("AAPL")
        assert result == []

    async def test_search_market_news_disabled(self, monkeypatch):
        monkeypatch.setattr(news.settings, "searxng_url", "")
        result = await news.search_market_news()
        assert result == []

    async def test_search_ticker_news_disabled(self, monkeypatch):
        monkeypatch.setattr(news.settings, "searxng_url", "")
        result = await news.search_ticker_news("AAPL")
        assert result == []

    async def test_gather_news_disabled(self, monkeypatch):
        monkeypatch.setattr(news.settings, "searxng_url", "")
        result = await news.gather_news_for_candidates(["AAPL", "NVDA"])
        assert result == {}


# ---------------------------------------------------------------------------
# Search and parsing
# ---------------------------------------------------------------------------

class TestSearch:
    async def test_search_returns_parsed_results(self, searxng_enabled):
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=_mock_response(_sample_results()))
            mock_client_cls.return_value = mock_client

            result = await news.search_news("AAPL stock news")
            assert len(result) == 3
            assert result[0]["title"] == "Apple beats Q3 estimates"
            assert "content" in result[0]
            assert "url" in result[0]

    async def test_search_truncates_content(self, searxng_enabled):
        long_content = "x" * 500
        raw = [{"title": "T", "content": long_content, "url": "u"}]
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=_mock_response(raw))
            mock_client_cls.return_value = mock_client

            result = await news.search_news("test")
            assert len(result[0]["content"]) <= news._MAX_SNIPPET

    async def test_search_skips_empty_titles(self, searxng_enabled):
        raw = [
            {"title": "", "content": "no title", "url": "u"},
            {"title": "Good", "content": "has title", "url": "u"},
        ]
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=_mock_response(raw))
            mock_client_cls.return_value = mock_client

            result = await news.search_news("test")
            assert len(result) == 1
            assert result[0]["title"] == "Good"

    async def test_search_respects_limit(self, searxng_enabled):
        raw = [{"title": f"T{i}", "content": "c", "url": "u"} for i in range(10)]
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=_mock_response(raw))
            mock_client_cls.return_value = mock_client

            result = await news.search_news("test", limit=3)
            assert len(result) == 3


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    async def test_second_call_uses_cache(self, searxng_enabled):
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=_mock_response(_sample_results()))
            mock_client_cls.return_value = mock_client

            # First call hits the network
            result1 = await news.search_news("AAPL stock news")
            assert len(result1) == 3
            assert mock_client.get.call_count == 1

            # Second call should use cache — no new network call
            result2 = await news.search_news("AAPL stock news")
            assert result2 == result1
            assert mock_client.get.call_count == 1  # still 1


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    async def test_network_error_returns_empty(self, searxng_enabled):
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
            mock_client_cls.return_value = mock_client

            result = await news.search_news("AAPL")
            assert result == []

    async def test_timeout_returns_empty(self, searxng_enabled):
        import httpx as _httpx
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(side_effect=_httpx.TimeoutException("timed out"))
            mock_client_cls.return_value = mock_client

            result = await news.search_news("AAPL")
            assert result == []

    async def test_empty_results_returns_empty(self, searxng_enabled):
        with patch("app.news.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=_mock_response([]))
            mock_client_cls.return_value = mock_client

            result = await news.search_news("OBSCURE")
            assert result == []


# ---------------------------------------------------------------------------
# gather_news_for_candidates
# ---------------------------------------------------------------------------

class TestGatherNews:
    async def test_gather_returns_dict_of_results(self, searxng_enabled):
        with patch("app.news.search_news", new_callable=AsyncMock) as mock_search:
            async def fake_search(query, limit=5):
                if "AAPL" in query:
                    return [{"title": "Apple news", "content": "c", "url": "u"}]
                if "market" in query.lower():
                    return [{"title": "Market news", "content": "c", "url": "u"}]
                return []
            mock_search.side_effect = fake_search

            result = await news.gather_news_for_candidates(["AAPL", "NVDA"], include_market=True)
            assert "market" in result
            assert "AAPL" in result
            assert "NVDA" not in result  # no news returned
            assert len(result["AAPL"]) == 1
            assert len(result["market"]) == 1

    async def test_gather_skips_failed_tickers(self, searxng_enabled):
        with patch("app.news.search_news", new_callable=AsyncMock) as mock_search:
            async def fake_search(query, limit=5):
                if "AAPL" in query:
                    return [{"title": "Apple", "content": "c", "url": "u"}]
                raise Exception("network error")
            mock_search.side_effect = fake_search

            result = await news.gather_news_for_candidates(["AAPL", "NVDA"])
            assert "AAPL" in result
            assert "NVDA" not in result  # failed, skipped


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_news_for_context(self):
        headlines = [
            {"title": "Apple beats estimates", "content": "", "url": ""},
            {"title": "New product launched", "content": "", "url": ""},
        ]
        out = news.format_news_for_context("AAPL", headlines)
        assert "AAPL" in out
        assert "Apple beats estimates" in out
        assert "New product launched" in out

    def test_format_news_empty(self):
        assert news.format_news_for_context("AAPL", []) == ""

    def test_format_market_news(self):
        headlines = [{"title": "Markets rally on Fed cut", "content": "", "url": ""}]
        out = news.format_market_news_for_context(headlines)
        assert "Market" in out
        assert "Markets rally" in out

    def test_format_market_news_empty(self):
        assert news.format_market_news_for_context([]) == ""
