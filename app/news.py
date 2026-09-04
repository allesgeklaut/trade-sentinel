"""News search via SearXNG for the sim bot and chat endpoint.

Queries a SearXNG instance's REST API for recent news headlines about a
ticker or the broader market. Results are cached in-memory for 4 hours
to avoid repeated calls within a sim cycle.

All functions return ``[]`` on any error — news is a supplementary context
that degrades gracefully. If ``settings.searxng_url`` is empty, the module
is disabled and all calls return ``[]`` immediately.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger("trade_sentinel.news")

# In-memory cache: query → (timestamp, results)
_cache: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 4 * 3600  # 4 hours
_MAX_SNIPPET = 200  # chars per news snippet in the LLM context
_MAX_RESULTS = 5  # results per ticker search


def _is_enabled() -> bool:
    return bool(settings.searxng_url)


def _cache_get(query: str) -> list[dict] | None:
    entry = _cache.get(query)
    if entry is None:
        return None
    ts, results = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _cache[query]
        return None
    return results


def _cache_put(query: str, results: list[dict]) -> None:
    _cache[query] = (time.monotonic(), results)


async def search_news(query: str, limit: int = _MAX_RESULTS) -> list[dict]:
    """Search for news about *query* via SearXNG.

    Returns a list of ``{"title": str, "content": str, "url": str}`` dicts
    with ``content`` truncated to ~200 chars. Returns ``[]`` if disabled,
    on network error, or if no results are found.
    """
    if not _is_enabled():
        return []

    cached = _cache_get(query)
    if cached is not None:
        return cached

    url = settings.searxng_url.rstrip("/") + "/search"
    params = {"q": query, "categories": "news", "format": "json"}

    try:
        async with httpx.AsyncClient(timeout=settings.searxng_timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("news search failed for '%s': %s", query, e)
        _cache_put(query, [])
        return []

    results: list[dict] = []
    for r in data.get("results", [])[:limit]:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        if not title:
            continue
        results.append({
            "title": title[:_MAX_SNIPPET],
            "content": content[:_MAX_SNIPPET],
            "url": r.get("url", ""),
        })

    _cache_put(query, results)
    return results


async def search_market_news(limit: int = _MAX_RESULTS) -> list[dict]:
    """Fetch broad market news for macro context ('stock market news today')."""
    return await search_news("stock market news today", limit=limit)


async def search_ticker_news(ticker: str, limit: int = _MAX_RESULTS) -> list[dict]:
    """Fetch news for a single ticker (e.g. 'AAPL stock news')."""
    return await search_news(f"{ticker} stock news", limit=limit)


def format_news_for_context(ticker: str, headlines: list[dict]) -> str:
    """Format headlines as a compact one-line summary for LLM context.

    Example: ``NVDA: Nvidia beats Q3 estimates | AI chip demand surges``
    """
    if not headlines:
        return ""
    parts = [h["title"] for h in headlines[:3]]
    return f"{ticker}: " + " | ".join(parts)


def format_market_news_for_context(headlines: list[dict]) -> str:
    """Format market-wide headlines for LLM context."""
    if not headlines:
        return ""
    parts = [h["title"] for h in headlines[:3]]
    return "Market: " + " | ".join(parts)


async def gather_news_for_candidates(
    tickers: list[str],
    include_market: bool = True,
) -> dict[str, list[dict]]:
    """Fetch news for multiple tickers concurrently + optional market news.

    Returns ``{"market": [...], "NVDA": [...], ...}`` — tickers with no
    news are omitted from the dict. The "market" key is only present if
    ``include_market`` is True.
    """
    if not _is_enabled():
        return {}

    tasks: dict[str, Any] = {}
    if include_market:
        tasks["market"] = search_market_news()
    for t in tickers:
        tasks[t] = search_ticker_news(t)

    keys = list(tasks.keys())
    values = await asyncio.gather(*tasks.values(), return_exceptions=True)

    result: dict[str, list[dict]] = {}
    for key, val in zip(keys, values, strict=False):
        if isinstance(val, Exception):
            logger.warning("news gather failed for '%s': %s", key, val)
            continue
        if val:
            result[key] = val
    return result
