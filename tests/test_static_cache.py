"""Static-asset cache headers: a deploy must be atomic for the whole page.

2026-11-03 phone report: the Savings equity chart went blank after a deploy.
The page HTML was fresh (its route sets ``Cache-Control: no-cache``) but the
phone kept the PREVIOUS ``portfolio.js`` from its heuristic cache — so the new
page called ``PaperPortfolioUI.buildSavingsSeries`` on a module that did not
have it, the TypeError killed the chart, and the rest of the page rendered
normally. Reproduced by serving the old module to the new page (the exact
TypeError above).

The fix has two halves, both pinned here:
1. every static subresource is served ``Cache-Control: no-cache`` (the same
   rule the index route already had) so browsers revalidate — the ETag still
   makes that a cheap 304;
2. the asset tags carry a ``?v=`` token, which is what unsticks a device that
   ALREADY cached the old URL under the old headers (a new HTML references a
   URL that device has never seen).
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from app import main

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "static" / "index.html"


async def test_static_subresources_are_served_no_cache():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ("/portfolio.js", "/workspace.css", "/site.webmanifest", "/favicon.svg"):
            response = await client.get(path)
            assert response.status_code == 200, path
            assert response.headers.get("cache-control") == "no-cache", (
                f"{path} must be revalidated on every load — without Cache-Control a browser "
                "heuristically caches it and can run the previous module against a new page"
            )
            # The revalidation path stays cheap: ETag/Last-Modified are intact.
            assert response.headers.get("etag"), path

        # The 304 must carry the header too — otherwise a browser can fall
        # back to heuristic freshness after the first revalidation.
        response = await client.get("/portfolio.js")
        etag = response.headers["etag"]
        revalidated = await client.get("/portfolio.js", headers={"If-None-Match": etag})
        assert revalidated.status_code == 304
        assert revalidated.headers.get("cache-control") == "no-cache"


async def test_index_route_is_still_no_cache():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.headers.get("cache-control") == "no-cache"


def test_asset_tags_carry_a_version_token():
    """A new HTML must reference a URL the device has never cached.

    The token is bumped by hand when either asset changes; this test only
    enforces that the mechanism is present (both tags, same token), so a
    future edit cannot silently drop it.
    """
    source = INDEX.read_text(encoding="utf-8")
    css = re.search(r'<link rel="stylesheet" href="workspace\.css\?v=([^"]+)"', source)
    js = re.search(r'<script src="portfolio\.js\?v=([^"]+)"', source)
    assert css, "workspace.css must carry a ?v= cache-buster"
    assert js, "portfolio.js must carry a ?v= cache-buster"
    assert css.group(1) == js.group(1), (
        "the two asset tokens must move together — index.html is served no-cache, "
        "so a mismatched pair can mix an old stylesheet with a new module"
    )
