#!/usr/bin/env python3
"""Bounded OFFLINE browser fixture harness for the Trade Sentinel UI.

WHY
---
Browser/UI verification needs a deterministic, populated Trade Sentinel page
without the real FastAPI app, scheduler, database, market-data provider or LLM.
This module serves the repository's *actual* ``static/index.html`` (plus its
static assets) and synthetic, shape-compatible ``/api`` payloads from
``tests/fixtures/ui_phase3.json``. It is a testing aid only.

GUARANTEES
----------
* Standard library only. The module imports no ``app`` module, opens no
  database, and never makes an outbound HTTP/socket connection.
* Static files are resolved beneath the real ``static/`` directory, with
  traversal/encoded-traversal/nul/backslash requests refused. There is no
  directory listing and no repository path is ever returned (or logged).
* Only the fixture API routes implemented below are served. Any other
  ``/api/...`` path fails closed with HTTP 501 JSON.
* POST/DELETE routes are acknowledgements only: no provider call, no trade,
  no DB write, no in-memory portfolio mutation. Mutation acks carry
  ``"fixture_ack": true`` and ``"applied": false``.
* PROD/DEV database-switch simulation (testing only): ``GET /api/db/status``
  reports the synthetic active database, and ``POST /api/db/clone`` is an
  ack-only clone (409 while a simulated clone runs or dev is disabled).
  The active database is selected by the ``ts_db=prod|dev`` cookie (or a
  ``?db=prod|dev`` query override, which also sets the cookie on ``/``);
  missing/unknown values mean prod, and dev requires ``dev_enabled``.
  Guard parity mirrors the real app: dev-only destructive POSTs answer 403
  unless the active database is dev, and prod-only savings writes (18 POSTs
  plus the two savings DELETEs) answer 403 while the active database is dev.
 * The type-D news-agent routes mirror app/main.py's ``/api/news/*``:
  ``POST /api/news/consult`` is a FAIL-CLOSED synthetic ack (no real agent
  consult, no LLM, no search call — the payload mirrors the live response
  shape with ``fixture_ack``/``applied: false``), and
  ``GET /api/news/decisions/{event_id}`` serves one synthetic D row for the
  manual 4-part ``D:{ticker}:{date}:m`` ids of the monthly book's held
  tickers; every other news path fails closed (404/501).
* Every response carries ``X-Trade-Sentinel-Fixture: testing-only`` plus
  ``X-Fixture-Mode`` / ``X-Fixture-Label`` headers and a content-security
  policy that keeps all traffic same-origin. External CDN script tags in
  ``index.html`` are therefore blocked by the browser; inject the deterministic
  chart/markdown stubs from ``tests/ui_phase3_smoke.js`` for visual runs.
* Requests are recorded in memory and exposed at ``/__fixture__/requests``.
  ``/__fixture__/health`` reports mode, counters and the fixture label.

USAGE
-----
    python tests/ui_fixture_server.py                    # populated, 127.0.0.1:8765
    python tests/ui_fixture_server.py --empty            # empty-state fixture
    python tests/ui_fixture_server.py --host 127.0.0.1 --port 8765

Empty mode can also be selected per browser session by opening:

    http://127.0.0.1:8765/?fixture=empty

That request sets the ``ts_fixture_mode=empty`` cookie, so every later
same-origin ``/api`` call from that browser context uses the empty fixture.
Use ``/?fixture=populated`` (or clear the cookie) to switch back. Direct API
clients may pass ``Cookie: ts_fixture_mode=empty`` or a ``fixture=empty`` query
parameter on any request. CLI ``--empty`` sets the server-wide default mode.

The synthetic PROD/DEV database switch works the same way: ``/?db=dev`` sets
the ``ts_db=dev`` cookie on the index document, and any request may use the
``db=dev`` query parameter or the ``ts_db`` cookie. Missing/unknown values
select prod; dev additionally requires ``dev_enabled`` in the mode's ``db``
fixture section.

Cookies are emitted as SEPARATE ``Set-Cookie`` header lines (one cookie per
header, per RFC 6265) — never comma-joined into a single value, which real
browsers would parse as one wrong cookie. ``FixtureResponse.cookies`` carries
the list; only the index document sets cookies.

The harness deliberately does NOT serve jsDelivr/CDN assets. Tests must inject
the stubs from ``tests/ui_phase3_smoke.js`` with Playwright
``page.addInitScript``; that script labels the run as navigation/layout only
and does not pretend the stubbed canvas is a real LightweightCharts render.
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ui_phase3.json"
MODE_COOKIE = "ts_fixture_mode"
DB_COOKIE = "ts_db"
DB_NAMES = frozenset({"prod", "dev"})
# Pinned guard parity (mirrors the real app's PROD/DEV switch):
# dev-only destructive POSTs -> 403 unless active == "dev".
DEV_ONLY_POSTS = frozenset({
    ("sim", "backfill"),
    ("sim", "backfill-benchmark"),
    ("backfill-all",),
    ("monthly", "backfill"),
    ("dailycore", "backfill"),
    ("sim", "reset"),
    ("sim", "reset-all"),
})
# Prod-only savings writes -> 403 while active == "dev" (18 POST routes).
SAVINGS_PROD_ONLY_POSTS = frozenset({
    ("savings", name) for name in (
        "initialize", "reset", "trueup", "config", "deposit", "withdraw",
        "dividend", "fee", "tax", "settle", "split", "correct", "saveback",
        "buy", "plans", "sell", "refresh", "run",
    )
})
MAX_BODY_BYTES = 1_000_000
MAX_STATIC_BYTES = 10_000_000
ALLOWED_STATIC_SUFFIXES = {
    ".css", ".gif", ".html", ".ico", ".jpeg", ".jpg", ".js", ".json", ".map",
    ".mjs", ".png", ".svg", ".ttf", ".txt", ".webmanifest", ".webp", ".woff",
    ".woff2",
}
MIME_OVERRIDES = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}
CSP = (
    "default-src 'self'; "
    "base-uri 'none'; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "img-src 'self' data:; "
    "object-src 'none'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'"
)
BASE_FIXTURE_HEADERS = {
    "X-Trade-Sentinel-Fixture": "testing-only",
    "Cache-Control": "no-store",
    "Content-Security-Policy": CSP,
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
COUNTER_KEYS = (
    "total",
    "static",
    "fixture_api",
    "post_ack",
    "introspection",
    "unknown_api",
    "not_found",
    "rejected",
    "method_not_allowed",
    "server_error",
    "db_guard",
    "db_conflict",
)
HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"})


class FixtureError(Exception):
    """A controlled fixture failure that becomes a JSON error response."""

    def __init__(self, status: int, detail: str, kind: str = "rejected", headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.kind = kind
        self.headers = headers or {}


@dataclass(frozen=True)
class FixtureResponse:
    """A fully materialised fixture response, shared by HTTP and unit tests."""

    status: int
    body: bytes
    content_type: str
    kind: str
    mode: str
    headers: dict[str, str] = field(default_factory=dict)
    # One entry per Set-Cookie header. RFC 6265 allows exactly ONE cookie per
    # Set-Cookie header, so a comma-joined value would be parsed by real
    # browsers as a single (wrong) cookie; the HTTP layer emits each entry as
    # its own header line instead.
    cookies: list[str] = field(default_factory=list)

    def json(self):
        """Decode a JSON response body (tests only; HTML bodies return bytes)."""
        return json.loads(self.body.decode("utf-8"))


def load_fixture(path: Path | str = DEFAULT_FIXTURE) -> dict:
    """Load and minimally validate the synthetic fixture document."""
    fixture_path = Path(path)
    try:
        raw = fixture_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read fixture: {fixture_path.name}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fixture is not valid JSON: {fixture_path.name}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("fixture root must be an object")
    if not isinstance(data.get("label"), str) or not data["label"]:
        raise ValueError("fixture must have a non-empty label")
    modes = data.get("modes")
    if not isinstance(modes, dict):
        raise ValueError("fixture must have a modes object")
    for name in ("populated", "empty"):
        if not isinstance(modes.get(name), dict):
            raise ValueError(f"fixture mode {name!r} must be an object")
    return data


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _as_list(value, name: str) -> list:
    if not isinstance(value, list):
        raise FixtureError(500, f"fixture data {name} must be a list", "server_error")
    return value


def _as_dict(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise FixtureError(500, f"fixture data {name} must be an object", "server_error")
    return value


def _parse_limit(query: dict[str, list[str]], default: int, maximum: int = 12000) -> int:
    raw = query.get("limit", [str(default)])[0]
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise FixtureError(422, "limit must be an integer", "rejected") from exc
    if not 1 <= value <= maximum:
        raise FixtureError(422, f"limit must be between 1 and {maximum}", "rejected")
    return value


def _parse_scope(query: dict[str, list[str]]) -> str:
    scope = query.get("scope", ["daily_core"])[0]
    if scope not in {"daily_core", "monthly", "sim"}:
        raise FixtureError(422, "scope must be one of daily_core, monthly, sim", "rejected")
    return scope


def _parse_fixture_number(data: dict, key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


class FixtureBackend:
    """Mode-aware router, recorder and payload factory for the fixture server."""

    def __init__(
        self,
        fixture_path: Path | str = DEFAULT_FIXTURE,
        mode: str = "populated",
        static_dir: Path | str = STATIC_DIR,
    ):
        self.fixture = load_fixture(fixture_path)
        self.modes = self.fixture["modes"]
        if mode not in self.modes:
            raise ValueError(f"unknown fixture mode {mode!r}")
        self.default_mode = mode
        self.label = self.fixture["label"]
        self.static_dir = Path(static_dir).resolve()
        self._lock = threading.Lock()
        self._requests: list[dict] = []
        self._counters = {key: 0 for key in COUNTER_KEYS}
        self._seq = 0

    # -- public surface -------------------------------------------------

    @property
    def available_modes(self) -> list[str]:
        return sorted(self.modes)

    def dispatch(
        self,
        method: str,
        raw_path: str,
        headers: dict | None = None,
        body: bytes = b"",
    ) -> FixtureResponse:
        """Resolve one request. No sockets are touched here (unit-testable)."""
        method = method.upper()
        headers = headers or {}
        mode = self.default_mode
        path = raw_path
        query: dict[str, list[str]] = {}
        try:
            if raw_path.startswith("//"):
                raise FixtureError(400, "absolute-form request targets are refused", "rejected")
            parsed = urlsplit(raw_path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            if parsed.scheme or parsed.netloc:
                raise FixtureError(400, "absolute-form request targets are refused", "rejected")
            if not path.startswith("/"):
                raise FixtureError(400, "request path must be absolute", "rejected")
            if path.startswith("//"):
                raise FixtureError(400, "absolute-form request targets are refused", "rejected")
            if "\x00" in path or "\\" in path:
                raise FixtureError(400, "path traversal refused", "rejected")
            if any(segment == ".." for segment in path.split("/")):
                raise FixtureError(400, "path traversal refused", "rejected")
            if method not in HTTP_METHODS:
                raise FixtureError(405, "method not allowed", "method_not_allowed")
            mode, cookie_mode = self._select_mode(query, headers)
            db, db_requested = self._select_db(query, headers, mode)
            response = self._route(
                method, path, query, mode, bool(cookie_mode), body,
                db=db, db_requested=db_requested,
            )
        except FixtureError as exc:
            response = self._error(mode, exc.detail, kind=exc.kind, status=exc.status, extra=exc.headers)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback or a path
            response = self._error(
                mode,
                "fixture backend error",
                kind="server_error",
                status=500,
                extra={"error_type": type(exc).__name__},
            )
        self._record(method, path, query, response)
        return response

    # -- mode selection -------------------------------------------------

    def _select_mode(self, query: dict[str, list[str]], headers) -> tuple[str, bool]:
        if "fixture" in query:
            requested = query["fixture"][-1]
            if requested not in self.modes:
                raise FixtureError(
                    400,
                    "unknown fixture mode; use populated or empty",
                    "rejected",
                )
            return requested, False
        cookie_header = headers.get("Cookie", "") if hasattr(headers, "get") else ""
        if cookie_header:
            cookie = SimpleCookie()
            try:
                cookie.load(cookie_header)
            except Exception:  # noqa: BLE001 - a malformed cookie must not 500
                cookie = None
            if cookie is not None and MODE_COOKIE in cookie:
                requested = cookie[MODE_COOKIE].value
                if requested in self.modes:
                    return requested, True
        return self.default_mode, False

    def _select_db(self, query: dict[str, list[str]], headers, mode: str) -> tuple[str, str | None]:
        """Resolve the synthetic active database: query > cookie > prod.

        Returns ``(effective_db, requested_query_db)``. Unknown cookie values
        mean prod; dev additionally requires ``dev_enabled`` in the mode's
        ``db`` section. The second element is the raw ``?db=`` value (or None)
        so the index document can set the ``ts_db`` cookie, mirroring how
        ``?fixture=`` sets ``ts_fixture_mode``.
        """
        dev_enabled = self._db_config(mode).get("dev_enabled", True) is True

        def effective(requested: str) -> str:
            return "dev" if requested == "dev" and dev_enabled else "prod"

        if "db" in query:
            requested = query["db"][-1]
            if requested not in DB_NAMES:
                raise FixtureError(400, "unknown db selection; use prod or dev", "rejected")
            return effective(requested), requested
        cookie_header = headers.get("Cookie", "") if hasattr(headers, "get") else ""
        if cookie_header:
            cookie = SimpleCookie()
            try:
                cookie.load(cookie_header)
            except Exception:  # noqa: BLE001 - a malformed cookie must not 500
                cookie = None
            if cookie is not None and DB_COOKIE in cookie:
                requested = cookie[DB_COOKIE].value
                if requested in DB_NAMES:
                    return effective(requested), None
        return "prod", None

    # -- dispatch -------------------------------------------------------

    def _route(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        mode: str,
        cookie_mode: bool,
        body: bytes,
        *,
        db: str = "prod",
        db_requested: str | None = None,
    ) -> FixtureResponse:
        if path == "/__fixture__/health":
            if method not in {"GET", "HEAD"}:
                raise FixtureError(405, "method not allowed", "method_not_allowed")
            return self._json(200, self._health_payload(mode), mode, "introspection")
        if path == "/__fixture__/requests":
            if method == "DELETE":
                return self._json(200, self._clear_requests(mode), mode, "introspection")
            if method in {"GET", "HEAD"}:
                return self._json(200, self._requests_payload(query, mode), mode, "introspection")
            raise FixtureError(405, "method not allowed", "method_not_allowed")
        if path.startswith("/__fixture__/"):
            raise FixtureError(404, "unknown fixture introspection route", "not_found")

        if path == "/api" or path.startswith("/api/"):
            if method in {"GET", "HEAD"}:
                return self._api_get(path, query, mode, db)
            if method == "POST":
                return self._api_post(path, query, mode, body, db)
            if method == "DELETE":
                return self._api_delete(path, query, mode, db)
            raise FixtureError(405, "method not allowed", "method_not_allowed")

        if method not in {"GET", "HEAD"}:
            raise FixtureError(405, "method not allowed", "method_not_allowed")
        cookies: list[str] | None = None
        if path in {"/", "/index.html"}:
            cookies = []
            if not cookie_mode:
                cookies.append(f"{MODE_COOKIE}={mode}; Path=/; SameSite=Lax")
            if db_requested is not None:
                cookies.append(f"{DB_COOKIE}={db_requested}; Path=/; SameSite=Lax")
        return self._static(path, mode, cookies=cookies)

    # -- static assets --------------------------------------------------

    def _static(self, path: str, mode: str, cookies: list[str] | None) -> FixtureResponse:
        candidate = self._static_candidate(path)
        if candidate is None:
            raise FixtureError(404, "not found", "not_found")
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            raise FixtureError(404, "not found", "not_found") from exc
        if len(data) > MAX_STATIC_BYTES:
            raise FixtureError(404, "not found", "not_found")
        suffix = candidate.suffix.lower()
        content_type = MIME_OVERRIDES.get(suffix) or mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        headers = dict(BASE_FIXTURE_HEADERS)
        headers["X-Fixture-Mode"] = mode
        headers["X-Fixture-Label"] = self.label
        headers["Vary"] = "Cookie"
        if path in {"/", "/index.html"}:
            headers["Cache-Control"] = "no-cache"
        return FixtureResponse(200, data, content_type, "static", mode, headers, list(cookies or []))

    def _static_candidate(self, path: str) -> Path | None:
        if path in {"/", "/index.html"}:
            candidate = (self.static_dir / "index.html").resolve()
            return candidate if candidate.is_file() else None
        relative = path.lstrip("/")
        if not relative or relative.endswith("/"):
            return None
        parts = relative.split("/")
        if any(part in {"", ".", ".."} or part.startswith(".") for part in parts):
            return None
        candidate = (self.static_dir / Path(*parts)).resolve()
        try:
            candidate.relative_to(self.static_dir)
        except ValueError:
            return None
        if candidate.suffix.lower() not in ALLOWED_STATIC_SUFFIXES:
            return None
        if not candidate.is_file():
            return None
        return candidate

    # -- fixture introspection -----------------------------------------

    def _health_payload(self, mode: str) -> dict:
        with self._lock:
            counters = dict(self._counters)
            recorded = self._seq
        return {
            "ok": True,
            "testing_only": True,
            "fixture_label": self.label,
            "fixture_version": self.fixture.get("version"),
            "fixture_as_of": self.fixture.get("as_of"),
            "notice": self.fixture.get("notice", "Synthetic UI fixture - testing only."),
            "mode": mode,
            "default_mode": self.default_mode,
            "available_modes": self.available_modes,
            "counters": counters,
            "recorded_requests": recorded,
            "outbound_network_requests": 0,
            "cdn_dependencies_served": False,
            "app_modules_imported": False,
            "mutation_route_policy": "acknowledgement-only; no provider, trade or DB write",
        }

    def _requests_payload(self, query: dict[str, list[str]], mode: str) -> dict:
        limit = _parse_limit(query, default=200, maximum=5000)
        with self._lock:
            entries = copy.deepcopy(self._requests)
            counters = dict(self._counters)
        return {
            "ok": True,
            "testing_only": True,
            "fixture_label": self.label,
            "mode": mode,
            "default_mode": self.default_mode,
            "count": len(entries),
            "returned": len(entries[-limit:]),
            "requests": entries[-limit:],
            "counters": counters,
        }

    def _clear_requests(self, mode: str) -> dict:
        with self._lock:
            cleared = len(self._requests)
            self._requests.clear()
        return {
            "ok": True,
            "testing_only": True,
            "fixture_label": self.label,
            "mode": mode,
            "cleared": cleared,
        }

    # -- API GET --------------------------------------------------------

    def _api_get(self, path: str, query: dict[str, list[str]], mode: str, db: str = "prod") -> FixtureResponse:
        api = [part for part in path.split("/") if part][1:]
        data = self.modes[mode]

        if api == ["db", "status"]:
            return self._json(200, self._db_status(mode, db), mode, "fixture_api")
        if api == ["watchlist"]:
            return self._json(200, _as_list(data.get("watchlist"), "watchlist"), mode, "fixture_api")
        if len(api) == 2 and api[0] == "info":
            ticker = api[1].upper()
            names = _as_dict(data.get("names", {}), "names")
            return self._json(200, {"ticker": ticker, "name": names.get(ticker, "")}, mode, "fixture_api")
        if api == ["symbols"]:
            q = query.get("q", [""])[0].strip()
            if len(q) < 2:
                raise FixtureError(422, "q must be at least 2 characters", "rejected")
            symbols = _as_list(data.get("symbols"), "symbols")
            needle = q.lower()
            matches = [
                row for row in symbols
                if needle in str(row.get("symbol", "")).lower() or needle in str(row.get("name", "")).lower()
            ]
            return self._json(200, matches, mode, "fixture_api")
        if len(api) == 2 and api[0] == "dashboard":
            ticker = api[1].upper()
            dashboards = _as_dict(data.get("dashboards"), "dashboards")
            dashboard = dashboards.get(ticker)
            if not isinstance(dashboard, dict):
                raise FixtureError(404, f"no fixture dashboard for {ticker}", "not_found")
            candles = _as_list(dashboard.get("candles"), "dashboard.candles")
            period = query.get("period", [None])[0]
            counts = {"6m": 126, "2y": 504, "5y": 1260, "10y": 2520, "max": 5000}
            if period in counts:
                candles = candles[-counts[period]:]
            payload = {**dashboard, "candles": copy.deepcopy(candles)}
            return self._json(200, payload, mode, "fixture_api")

        if api == ["screener", "universes"]:
            return self._json(200, _as_list(data.get("universes"), "universes"), mode, "fixture_api")
        if api == ["screener", "status"]:
            return self._json(200, _as_dict(data["screener"]["status"], "screener.status"), mode, "fixture_api")
        if len(api) == 2 and api[0] == "screener":
            universe = api[1]
            universes = _as_list(data.get("universes"), "universes")
            if universe not in universes:
                raise FixtureError(404, "unknown universe", "not_found")
            results = _as_dict(data["screener"].get("results", {}), "screener.results")
            return self._json(200, _as_list(results.get(universe, []), f"screener.results.{universe}"), mode, "fixture_api")
        if api == ["llm", "status"]:
            return self._json(200, _as_dict(data.get("llm"), "llm"), mode, "fixture_api")
        if api and api[0] == "news":
            return self._news_get(api, mode)
        if api == ["run-config"]:
            # Live CURRENT (next-run) configuration of every engine — canned
            # from the fixture document (sim/monthly/daily_core, each with
            # engine + epoch + a couple of current_config fields).
            return self._json(200, _as_dict(data.get("run_config"), "run_config"), mode, "fixture_api")

        if api and api[0] in {"sim", "monthly", "daily_core", "dailycore"}:
            return self._portfolio_get(api, query, mode)

        if len(api) == 2 and api[0] == "metrics":
            metric = api[1]
            if metric not in {"xirr", "twr"}:
                raise FixtureError(501, "unknown fixture API route", "unknown_api")
            scope = _parse_scope(query)
            metrics = _as_dict(data.get("metrics"), "metrics")
            payload = _as_dict(metrics.get(metric, {}), f"metrics.{metric}").get(scope)
            if not isinstance(payload, dict):
                raise FixtureError(500, f"fixture metrics.{metric}.{scope} is missing", "server_error")
            return self._json(200, payload, mode, "fixture_api")

        if api and api[0] == "savings":
            return self._savings_get(api, query, mode)

        raise FixtureError(501, "unknown fixture API route", "unknown_api")

    def _portfolio_get(self, api: list[str], query: dict[str, list[str]], mode: str) -> FixtureResponse:
        account = api[0]
        account_key = "daily_core" if account in {"daily_core", "dailycore"} else account
        portfolio = _as_dict(self.modes[mode]["portfolios"].get(account_key), f"portfolios.{account_key}")
        tail = api[1:]
        if tail == ["status"]:
            return self._json(200, _as_dict(portfolio.get("status"), f"{account_key}.status"), mode, "fixture_api")
        if tail == ["trades"]:
            limit = _parse_limit(query, default=100, maximum=500)
            trades = _as_list(portfolio.get("trades"), f"{account_key}.trades")
            return self._json(200, copy.deepcopy(trades[:limit]), mode, "fixture_api")
        if tail == ["equity"]:
            limit = _parse_limit(query, default=365, maximum=12000)
            equity = _as_list(portfolio.get("equity"), f"{account_key}.equity")
            return self._json(200, copy.deepcopy(equity[-limit:]), mode, "fixture_api")
        if account == "sim":
            if tail == ["allowances"]:
                return self._json(200, copy.deepcopy(_as_list(portfolio.get("allowances"), "sim.allowances")), mode, "fixture_api")
            if tail == ["benchmark"]:
                limit = _parse_limit(query, default=365, maximum=12000)
                benchmark = copy.deepcopy(_as_dict(portfolio.get("benchmark"), "sim.benchmark"))
                curve = _as_list(portfolio.get("benchmark_equity"), "sim.benchmark_equity")
                benchmark["equity_curve"] = copy.deepcopy(curve[-limit:])
                return self._json(200, benchmark, mode, "fixture_api")
            if tail == ["run-status"]:
                return self._json(200, _as_dict(portfolio.get("run_status"), "sim.run_status"), mode, "fixture_api")
            if tail == ["reasoning"]:
                return self._json(200, _as_dict(portfolio.get("reasoning"), "sim.reasoning"), mode, "fixture_api")
            if tail == ["chat", "history"]:
                return self._json(200, copy.deepcopy(_as_list(portfolio.get("chat_history"), "sim.chat_history")), mode, "fixture_api")
        if account == "monthly" and tail == ["rebalances"]:
            limit = _parse_limit(query, default=24, maximum=120)
            rebalances = _as_list(portfolio.get("rebalances"), "monthly.rebalances")
            return self._json(200, copy.deepcopy(rebalances[:limit]), mode, "fixture_api")
        raise FixtureError(501, "unknown fixture API route", "unknown_api")

    def _news_get(self, api: list[str], mode: str) -> FixtureResponse:
        """GET /api/news/decisions/{event_id} — the synthetic D row.

        Mirrors app/main.py's news-agent view: only the manual 4-part
        ``D:{ticker}:{date}:m`` event ids of the monthly book's held tickers
        answer 200 (one hard-coded synthetic record, built here — NOT in the
        pinned ui_phase3.json document); every other news path fails closed
        exactly like the live app's honest 404 / unknown-route 501.
        """
        if len(api) == 3 and api[1] == "decisions":
            event_id = api[2]
            parts = event_id.split(":")
            held = self._held_tickers(mode, "monthly")
            if (len(parts) == 4 and parts[0] == "D" and parts[3] == "m"
                    and parts[1] in held):
                return self._json(200, self._synthetic_d_row(mode, event_id),
                                  mode, "fixture_api")
            raise FixtureError(
                404, f"no type-D decision for event_id {event_id!r}", "not_found")
        raise FixtureError(501, "unknown fixture API route", "unknown_api")

    def _synthetic_d_row(self, mode: str, event_id: str) -> dict:
        """The canned type-D record (built in the harness, not in the pinned
        fixture document): a HOLD with one citation and one search trace."""
        as_of = str(self.fixture.get("as_of", "2026-10-30"))[:10]
        parts = event_id.split(":")
        ticker = parts[1] if len(parts) > 1 else ""
        session = parts[2] if len(parts) > 2 else as_of
        return {
            "event_id": event_id,
            "event_type": "D",
            "engine": "monthly",
            "ticker": ticker,
            "session_date": session,
            "decision": "HOLD",
            "rationale": "Synthetic fixture row — no real agent consult ran (testing only).",
            "parsed": True,
            "fail_safe": False,
            "fail_reason": None,
            "prompt_version": "sellshadow/1-d",
            "context": {
                "position": {
                    "ticker": ticker,
                    "shares_held": 8.0,
                    "entry": {"date": "2026-07-31", "price": 401.2},
                    "current": {"date": as_of, "close": 438.75},
                    "pnl_pct": 9.36,
                    "months_held": 3.0,
                    "stated_thesis_condition": "ROE > 0 and operating cash flow positive",
                    "latest_fundamentals": {},
                },
                "citations": [{"title": f"Fixture headline — {ticker}",
                               "source": "fixture-news", "date": as_of}],
                "trace": [{"name": "searxng_web_search",
                           "arguments": {"query": f"{ticker} news"},
                           "ok": True, "kind": "ok", "result_chars": 42,
                           "result_head": "synthetic fixture search result"}],
                "temperature": 0.6,
                "reasoning_effort": "",
                "tool_calls": 1,
                "turns": 2,
                "news_source": "searxng-mcp",
            },
            "engine_branch": {
                "rule": "hold to next month-end rebuild",
                "note": "shadow: no execution authority; the fill columns stay NULL",
            },
            "backend": "fixture-offline",
            "model": "fixture-news-model",
            "latency_ms": 0.0,
            "decided_at": f"{as_of}T12:00:00+00:00",
        }

    def _savings_get(self, api: list[str], query: dict[str, list[str]], mode: str) -> FixtureResponse:
        savings = _as_dict(self.modes[mode].get("savings"), "savings")
        tail = api[1:]
        if tail == ["status"]:
            return self._json(200, _as_dict(savings.get("status"), "savings.status"), mode, "fixture_api")
        if tail == ["equity"]:
            limit = _parse_limit(query, default=365, maximum=12000)
            equity = _as_list(savings.get("equity"), "savings.equity")
            return self._json(200, copy.deepcopy(equity[-limit:]), mode, "fixture_api")
        if tail == ["plans"]:
            status = _as_dict(savings.get("status"), "savings.status")
            return self._json(200, copy.deepcopy(_as_list(status.get("plans", []), "savings.status.plans")), mode, "fixture_api")
        if tail == ["mirror"]:
            limit = _parse_limit(query, default=365, maximum=12000)
            mirror = copy.deepcopy(_as_dict(savings.get("mirror"), "savings.mirror"))
            curve = _as_list(mirror.get("curve", []), "savings.mirror.curve")
            mirror["curve"] = copy.deepcopy(curve[-limit:])
            return self._json(200, mirror, mode, "fixture_api")
        if tail == ["metrics", "xirr"]:
            metrics = _as_dict(savings.get("metrics"), "savings.metrics")
            return self._json(200, _as_dict(metrics.get("xirr"), "savings.metrics.xirr"), mode, "fixture_api")
        if tail == ["metrics", "twr"]:
            metrics = _as_dict(savings.get("metrics"), "savings.metrics")
            return self._json(200, _as_dict(metrics.get("twr"), "savings.metrics.twr"), mode, "fixture_api")
        raise FixtureError(501, "unknown fixture API route", "unknown_api")

    # -- API mutations (acknowledgements only) --------------------------

    def _api_post(self, path: str, query: dict[str, list[str]], mode: str, body: bytes, db: str = "prod") -> FixtureResponse:
        api = [part for part in path.split("/") if part][1:]
        data = self.modes[mode]
        if not api:
            raise FixtureError(501, "unknown fixture API route", "unknown_api")

        if api == ["db", "clone"]:
            return self._db_clone(mode)
        self._check_db_guard(api, db, method="POST")

        if len(api) == 2 and api[0] == "watchlist":
            return self._ack(mode, {"ticker": api[1].upper(), "fetched": False})
        if len(api) == 2 and api[0] == "refresh":
            return self._ack(mode, {"ok": True})
        if api == ["refresh-watchlist"]:
            return self._ack(mode, {"refreshed": list(data.get("watchlist", [])), "errors": [], "total": len(data.get("watchlist", []))})
        if api == ["sim", "refresh"]:
            tickers = self._held_tickers(mode, "sim")
            return self._ack(mode, {"refreshed": tickers, "errors": [], "total": len(tickers)})
        if api == ["sim", "run"]:
            return self._ack(mode, {"started": True})
        if api == ["sim", "run-status"]:
            raise FixtureError(501, "unknown fixture API route", "unknown_api")
        if api == ["sim", "backfill"]:
            status = self._portfolio_status(mode, "sim")
            return self._ack(mode, {
                "ok": True,
                "days": 45,
                "start": "2026-09-01",
                "end": self.fixture.get("as_of", "2026-10-30"),
                "requested_start": "all",
                "snapshots": 45,
                "trades": 4,
                "final_equity": status.get("total_equity", 0.0),
            })
        if api == ["sim", "backfill-benchmark"]:
            benchmark = self._portfolio_status(mode, "sim")
            bench = _as_dict(self._portfolio(mode, "sim").get("benchmark"), "sim.benchmark")
            return self._ack(mode, {
                "ok": True,
                "days": 45,
                "start": "2026-09-01",
                "end": self.fixture.get("as_of", "2026-10-30"),
                "n_buys": 2,
                "contributed": bench.get("allowance_total", 0.0),
                "final_equity": bench.get("total_equity", benchmark.get("total_equity", 0.0)),
            })
        if api == ["backfill-all"]:
            final = {
                "sim": self._account_equity(mode, "sim"),
                "monthly": self._account_equity(mode, "monthly"),
                "daily_core": self._account_equity(mode, "daily_core"),
            }
            bench = _as_dict(self._portfolio(mode, "sim").get("benchmark"), "sim.benchmark")
            results = {
                "daily_sim": {"ok": True, "final_equity": final["sim"]},
                "monthly": {"ok": True, "final_equity": final["monthly"]},
                "daily_core": {"ok": True, "final_equity": final["daily_core"]},
                "benchmark": {"ok": True, "final_equity": bench.get("total_equity", 0.0)},
            }
            return self._ack(mode, {"ok": True, "start": "2026-09-01", "requested_start": "all", "results": results, "errors": {}})
        if api == ["sim", "reset"] or api == ["sim", "reset-all"]:
            return self._ack(mode, {"ok": True})
        if api == ["sim", "universe"]:
            payload = self._body_json(body)
            return self._ack(mode, {"universe": payload.get("universe", "diversified-plus")})
        if api == ["sim", "chat"]:
            # New envelope: {"text", "proposals", "history"} — the chat never
            # executes trades; proposals carry per-action previews and only run
            # via POST /api/sim/chat/confirm. No trades/actions_executed keys.
            return self._ack(mode, {
                "text": self._chat_text(),
                "proposals": [copy.deepcopy(self._chat_proposal(mode))],
                "history": self._chat_history(mode),
            })
        if api == ["sim", "chat", "stream"]:
            return self._sse(mode, sim=True)
        if api == ["sim", "chat", "confirm"]:
            # Ack-only confirm: canned executed trades + skipped + snapshot flag
            # from the fixture document. (The real app 404s unknown/expired
            # proposal ids and 409s on a config change; this harness acks every
            # well-formed confirm — unknown POST paths still fail closed 501.)
            confirm = _as_dict(data.get("chat_confirm"), "chat_confirm")
            payload = {
                "ok": confirm.get("ok", True) is True,
                "trades": copy.deepcopy(_as_list(confirm.get("trades", []), "chat_confirm.trades")),
                "skipped": copy.deepcopy(_as_list(confirm.get("skipped", []), "chat_confirm.skipped")),
                "snapshot_taken": confirm.get("snapshot_taken", False) is True,
            }
            return self._ack(mode, payload)
        if api == ["sim", "chat", "reject"]:
            return self._ack(mode, {"ok": True})
        if len(api) == 3 and api[0] == "chat" and api[2] == "stream":
            return self._sse(mode)

        if api == ["monthly", "run"]:
            status = self._portfolio_status(mode, "monthly")
            picks = [p.get("ticker") for p in status.get("positions", []) if isinstance(p, dict)]
            return self._ack(mode, {"skipped": False, "month": "2026-10", "picks": picks, "trades": []})
        if api == ["monthly", "refresh"]:
            tickers = self._held_tickers(mode, "monthly")
            return self._ack(mode, {"refreshed": tickers, "errors": [], "total": len(tickers)})
        if api == ["monthly", "backfill"]:
            status = self._portfolio_status(mode, "monthly")
            return self._ack(mode, {
                "ok": True,
                "months": 2,
                "start": "2026-09-01",
                "end": self.fixture.get("as_of", "2026-10-30"),
                "requested_start": "all",
                "snapshots": 45,
                "trades": 3,
                "rebalances": 2,
                "final_equity": status.get("total_equity", 0.0),
            })

        if api == ["dailycore", "run"]:
            status = self._portfolio_status(mode, "daily_core")
            return self._ack(mode, {"skipped": False, "deployment": {"trades": []}, "ranking_date": status.get("ranking_date")})
        if api == ["dailycore", "refresh"]:
            tickers = self._held_tickers(mode, "daily_core")
            return self._ack(mode, {"refreshed": tickers, "errors": [], "total": len(tickers)})
        if api == ["dailycore", "backfill"]:
            status = self._portfolio_status(mode, "daily_core")
            valuation = _as_dict(status.get("valuation", {}), "daily_core.status.valuation")
            return self._ack(mode, {
                "ok": True,
                "days": 45,
                "start": "2026-09-01",
                "end": self.fixture.get("as_of", "2026-10-30"),
                "requested_start": "all",
                "mom_variant": status.get("mom_variant", "residual"),
                "snapshots": 45,
                "trades": 4,
                "final_equity": valuation.get("total_equity", 0.0),
            })
        if api == ["dailycore", "strategy"]:
            payload = self._body_json(body)
            variant = payload.get("variant", "residual")
            labels = _as_dict(self._portfolio_status(mode, "daily_core").get("mom_variants", {}), "mom_variants")
            return self._ack(mode, {"variant": variant, "label": labels.get(variant, variant)})
        if api == ["dailycore", "gate"]:
            payload = self._body_json(body)
            gate = payload.get("mode", "off")
            labels = _as_dict(self._portfolio_status(mode, "daily_core").get("gates", {}), "gates")
            return self._ack(mode, {"mode": gate, "label": labels.get(gate, gate)})
        if api == ["dailycore", "gradient"]:
            payload = self._body_json(body)
            gradient = payload.get("mode", "off")
            labels = _as_dict(self._portfolio_status(mode, "daily_core").get("gradients", {}), "gradients")
            return self._ack(mode, {"mode": gradient, "label": labels.get(gradient, gradient)})
        if api == ["dailycore", "targetvol"]:
            payload = self._body_json(body)
            target = payload.get("mode", "off")
            labels = _as_dict(self._portfolio_status(mode, "daily_core").get("target_vols", {}), "target_vols")
            return self._ack(mode, {"mode": target, "label": labels.get(target, target)})

        if len(api) == 3 and api[0] == "screener" and api[1] in {"run", "refresh", "load_deep"}:
            universe = api[2]
            results = self._screener_results(mode, universe)
            total = len(results)
            if api[1] == "run":
                return self._ack(mode, {"universe": universe, "processed": total, "ranked": total})
            if api[1] == "refresh":
                return self._ack(mode, {"universe": universe, "refreshed": total, "skipped": 0, "errors": [], "total": total})
            return self._ack(mode, {"universe": universe, "period": "10y", "loaded": total, "errors": [], "total": total})
        if api == ["screener", "sync"]:
            sync = _as_dict(data["screener"].get("sync"), "screener.sync")
            return self._ack(mode, copy.deepcopy(sync))

        if api == ["llm", "select"]:
            payload = self._body_json(body)
            return self._ack(mode, {
                "backend": payload.get("backend", ""),
                "model": payload.get("model", ""),
            })

        if api == ["news", "consult"]:
            # Fail-closed synthetic ack mirroring POST /api/news/consult in
            # app/main.py: NO real consult runs here (no LLM, no search MCP).
            # The payload carries the live response's shape so the UI panel
            # renders identically, plus fixture_ack/applied: false.
            payload = self._body_json(body)
            ticker = str(payload.get("ticker", "")).strip().upper()
            held = self._held_tickers(mode, "monthly")
            if not ticker:
                return self._ack(mode, {"ok": False,
                                        "reason": "ticker must not be empty"})
            if ticker not in held:
                return self._ack(mode, {
                    "ok": False,
                    "reason": f"{ticker} is not in the monthly book (fixture)",
                })
            as_of = str(self.fixture.get("as_of", "2026-10-30"))[:10]
            row = self._synthetic_d_row(mode, f"D:{ticker}:{as_of}:m")
            return self._ack(mode, {
                "ok": True,
                "decision": row["decision"],
                "rationale": row["rationale"],
                "citations": row["context"]["citations"],
                "turns": row["context"]["turns"],
                "tool_calls": row["context"]["tool_calls"],
                "latency_ms": 0.0,
                "fail_safe": False,
                "fail_reason": None,
                "event_id": row["event_id"],
                "model": row["model"],
                "backend": row["backend"],
            })

        if api and api[0] == "savings":
            return self._savings_post(api, mode, body)

        raise FixtureError(501, "unknown fixture API route", "unknown_api")

    def _savings_post(self, api: list[str], mode: str, body: bytes) -> FixtureResponse:
        tail = api[1:]
        savings = _as_dict(self.modes[mode].get("savings"), "savings")
        status = _as_dict(savings.get("status"), "savings.status")
        cash = status.get("cash", 0.0) if isinstance(status.get("cash"), (int, float)) else 0.0
        if tail == ["initialize"]:
            payload = self._body_json(body)
            positions = payload.get("positions", [])
            invested = 0.0
            if isinstance(positions, list):
                invested = sum(
                    float(p.get("shares", 0)) * float(p.get("avg_cost", 0))
                    for p in positions
                    if isinstance(p, dict)
                )
            return self._ack(mode, {
                "cash": _parse_fixture_number(payload, "cash", 0.0),
                "invested": invested,
                "mirror": {"picks": 0},
            })
        if tail == ["reset"]:
            return self._ack(mode, {"ok": True})
        if tail == ["trueup"]:
            payload = self._body_json(body)
            actual = _parse_fixture_number(payload, "actual_cash", cash)
            return self._ack(mode, {"delta": 0.0, "cash": actual})
        if tail == ["config"]:
            return self._ack(mode, {"ok": True})
        if tail == ["deposit"]:
            payload = self._body_json(body)
            amount = _parse_fixture_number(payload, "amount", 0.0)
            return self._ack(mode, {"amount": amount, "cash": cash})
        if tail == ["withdraw"]:
            payload = self._body_json(body)
            amount = _parse_fixture_number(payload, "amount", 0.0)
            occurred = payload.get("occurred_on") or self.fixture.get("as_of", "2026-10-30")
            return self._ack(mode, {"amount": amount, "cash": cash, "occurred_on": occurred})
        if tail == ["buy"]:
            payload = self._body_json(body)
            amount = _parse_fixture_number(payload, "amount", 0.0)
            ticker = str(payload.get("ticker", "FIXT")).upper()
            price = 100.0
            return self._ack(mode, {"amount": amount, "ticker": ticker, "price": price, "shares": round(amount / price, 4), "cash": cash})
        if tail == ["saveback"]:
            payload = self._body_json(body)
            amount = _parse_fixture_number(payload, "amount", 0.0)
            ticker = str(payload.get("ticker", "FIXT")).upper()
            price = 100.0
            return self._ack(mode, {"amount": amount, "ticker": ticker, "price": price, "shares": round(amount / price, 4)})
        if tail == ["plans"]:
            payload = self._body_json(body)
            return self._ack(mode, {
                "id": 900,
                "ticker": str(payload.get("ticker", "FIXT")).upper(),
                "amount": _parse_fixture_number(payload, "amount", 0.0),
                "day_of_month": int(payload.get("day_of_month", 1)) if isinstance(payload.get("day_of_month"), int) else 1,
                "active": True,
            })
        if tail == ["refresh"]:
            tickers = self._savings_tickers(mode)
            return self._ack(mode, {"refreshed": tickers, "errors": [], "total": len(tickers)})
        if tail == ["run"]:
            return self._ack(mode, {
                "skipped": False,
                "refresh": {"refreshed": len(self._savings_tickers(mode))},
                "transfer": {"deposited": True, "amount": status.get("monthly_transfer", 0.0)},
                "payout": {"paid": True, "amount": status.get("accrued_interest", 0.0)},
                "accrual": {"accrued": True, "days": 1, "amount": 0.09},
                "sparplans": {"executed": len(status.get("plans", []))},
            })
        if tail and tail[0] in {"sell", "dividend", "fee", "tax", "settle", "split", "correct"}:
            return self._ack(mode, {"ok": True})
        raise FixtureError(501, "unknown fixture API route", "unknown_api")

    def _api_delete(self, path: str, query: dict[str, list[str]], mode: str, db: str = "prod") -> FixtureResponse:
        api = [part for part in path.split("/") if part][1:]
        self._check_db_guard(api, db, method="DELETE")
        if len(api) == 2 and api[0] == "watchlist":
            return self._ack(mode, {"ok": True, "ticker": api[1].upper()})
        if api == ["sim", "chat", "history"]:
            return self._ack(mode, {"ok": True})
        if len(api) == 3 and api[0] == "savings" and api[1] == "plans":
            return self._ack(mode, {"ok": True, "id": api[2]})
        if len(api) == 3 and api[0] == "savings" and api[1] == "positions":
            return self._ack(mode, {"ok": True, "ticker": api[2].upper()})
        raise FixtureError(501, "unknown fixture API route", "unknown_api")

    # -- PROD/DEV database switch (synthetic, testing only) -------------

    def _db_config(self, mode: str) -> dict:
        cfg = self.modes[mode].get("db")
        return cfg if isinstance(cfg, dict) else {}

    def _db_status(self, mode: str, db: str) -> dict:
        cfg = self._db_config(mode)
        return {
            "active": db,
            "dev_enabled": cfg.get("dev_enabled", True) is True,
            "dev_seeded": cfg.get("dev_seeded", False) is True,
            "cloning": cfg.get("cloning", False) is True,
        }

    def _db_clone(self, mode: str) -> FixtureResponse:
        cfg = self._db_config(mode)
        if cfg.get("dev_enabled", True) is not True:
            raise FixtureError(409, "dev database is disabled", "db_conflict")
        if cfg.get("cloning", False) is True:
            raise FixtureError(409, "a simulated dev clone is already running", "db_conflict")
        size = cfg.get("clone_size_bytes", 0)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            size = 0
        return self._ack(mode, {"ok": True, "size_bytes": size})

    def _check_db_guard(self, api: list[str], db: str, method: str) -> None:
        """Mirror the real app's PROD/DEV guards exactly (403 fail-closed)."""
        route = tuple(api)
        if method == "POST":
            if route in DEV_ONLY_POSTS and db != "dev":
                raise FixtureError(
                    403,
                    "dev-only destructive action refused while the prod database is active",
                    "db_guard",
                )
            if route in SAVINGS_PROD_ONLY_POSTS and db == "dev":
                raise FixtureError(
                    403,
                    "savings writes are prod-only while the dev database is active",
                    "db_guard",
                )
        elif method == "DELETE":
            if api and api[0] == "savings" and len(api) == 3 and api[1] in {"plans", "positions"} and db == "dev":
                raise FixtureError(
                    403,
                    "savings writes are prod-only while the dev database is active",
                    "db_guard",
                )

    # -- payload helpers ------------------------------------------------

    def _portfolio(self, mode: str, name: str) -> dict:
        portfolios = _as_dict(self.modes[mode].get("portfolios"), "portfolios")
        return _as_dict(portfolios.get(name), f"portfolios.{name}")

    def _portfolio_status(self, mode: str, name: str) -> dict:
        return _as_dict(self._portfolio(mode, name).get("status"), f"{name}.status")

    def _account_equity(self, mode: str, name: str) -> float:
        """Total equity from a status payload, including daily-core's nesting."""
        status = self._portfolio_status(mode, name)
        if name == "daily_core":
            valuation = _as_dict(status.get("valuation", {}), "daily_core.status.valuation")
            value = valuation.get("total_equity", 0.0)
        else:
            value = status.get("total_equity", 0.0)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    def _held_tickers(self, mode: str, account: str) -> list[str]:
        status = self._portfolio_status(mode, account)
        positions = status.get("positions", [])
        if account == "daily_core":
            valuation = _as_dict(status.get("valuation", {}), "daily_core.valuation")
            positions = valuation.get("positions", [])
        return sorted({str(p.get("ticker")) for p in positions if isinstance(p, dict) and p.get("ticker")})

    def _savings_tickers(self, mode: str) -> list[str]:
        status = _as_dict(self.modes[mode]["savings"].get("status"), "savings.status")
        positions = status.get("positions", [])
        return sorted({str(p.get("ticker")) for p in positions if isinstance(p, dict) and p.get("ticker")})

    def _screener_results(self, mode: str, universe: str) -> list:
        data = self.modes[mode]
        universes = _as_list(data.get("universes"), "universes")
        if universe not in universes:
            raise FixtureError(404, "unknown universe", "not_found")
        results = _as_dict(data["screener"].get("results", {}), "screener.results")
        return _as_list(results.get(universe, []), f"screener.results.{universe}")

    def _body_json(self, body: bytes) -> dict:
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FixtureError(400, "request body must be valid JSON", "rejected") from exc
        if not isinstance(payload, dict):
            raise FixtureError(400, "request body must be a JSON object", "rejected")
        return payload

    def _chat_text(self) -> str:
        return "Fixture chat stub - navigation/layout testing only; no LLM or provider call was made."

    def _chat_proposal(self, mode: str) -> dict:
        """The canned pending chat proposal (one feasible + one infeasible action)."""
        data = self.modes[mode]
        proposal = _as_dict(data.get("chat_proposal"), "chat_proposal")
        return copy.deepcopy(proposal)

    def _chat_history(self, mode: str) -> list:
        portfolio = _as_dict(self.modes[mode]["portfolios"].get("sim"), "portfolios.sim")
        return copy.deepcopy(_as_list(portfolio.get("chat_history", []), "sim.chat_history"))

    # -- response construction -----------------------------------------

    def _ack(self, mode: str, payload: dict) -> FixtureResponse:
        body = copy.deepcopy(payload)
        body.setdefault("ok", True)
        body["fixture_ack"] = True
        body["applied"] = False
        return self._json(200, body, mode, "post_ack")

    def _sse(self, mode: str, sim: bool = False) -> FixtureResponse:
        text = self._chat_text()
        if sim:
            # Sim manager chat: the done envelope carries proposals + history,
            # never trades/actions_executed.
            done = {
                "type": "done",
                "text": text,
                "proposals": [copy.deepcopy(self._chat_proposal(mode))],
                "history": self._chat_history(mode),
            }
        else:
            # Per-ticker research chat: plain {"text", "model"} envelope.
            done = {"type": "done", "text": text, "model": "fixture-chat-model"}
        events = [
            {"type": "delta", "text": text},
            done,
        ]
        body = "".join(f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n" for event in events)
        headers = dict(BASE_FIXTURE_HEADERS)
        headers["X-Fixture-Mode"] = mode
        headers["X-Fixture-Label"] = self.label
        headers["X-Accel-Buffering"] = "no"
        return FixtureResponse(
            200,
            body.encode("utf-8"),
            "text/event-stream; charset=utf-8",
            "post_ack",
            mode,
            headers,
        )

    def _json(self, status: int, payload, mode: str, kind: str) -> FixtureResponse:
        headers = dict(BASE_FIXTURE_HEADERS)
        headers["X-Fixture-Mode"] = mode
        headers["X-Fixture-Label"] = self.label
        return FixtureResponse(status, _json_bytes(payload), "application/json; charset=utf-8", kind, mode, headers)

    def _error(
        self,
        mode: str,
        detail: str,
        *,
        kind: str,
        status: int,
        extra: dict | None = None,
    ) -> FixtureResponse:
        payload = {
            "detail": detail,
            "testing_only": True,
            "fixture_label": self.label,
            "fixture_mode": mode,
        }
        if extra:
            payload.update(extra)
        return self._json(status, payload, mode, kind)

    def _record(self, method: str, path: str, query: dict[str, list[str]], response: FixtureResponse) -> None:
        with self._lock:
            self._seq += 1
            self._counters["total"] += 1
            if response.kind in self._counters:
                self._counters[response.kind] += 1
            self._requests.append({
                "seq": self._seq,
                "method": method,
                "path": path,
                "query": {key: list(values) for key, values in query.items()},
                "status": response.status,
                "kind": response.kind,
                "mode": response.mode,
                "at": datetime.now(UTC).isoformat(),
            })


def _parse_content_length(value: str | None) -> int:
    """Validate an HTTP Content-Length before any body read.

    Empty/absent is 0. Malformed or negative lengths are rejected instead of
    being coerced to 0 (which would silently ignore a body) or passed to
    ``rfile.read`` (where ``-1`` waits for EOF and can hang a client).
    """
    if value is None or value.strip() == "":
        return 0
    try:
        length = int(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Content-Length") from exc
    if length < 0:
        raise ValueError("negative Content-Length")
    return length


class FixtureRequestHandler(BaseHTTPRequestHandler):
    """Thin HTTP adapter around :class:`FixtureBackend`."""

    server_version = "TradeSentinelFixture/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def _reject_length(self, status: int, detail: str) -> None:
        self.close_connection = True
        backend = self.server.backend
        body = _json_bytes({
            "detail": detail,
            "testing_only": True,
            "fixture_label": backend.label,
            "fixture_mode": backend.default_mode,
        })
        self.send_response(status)
        self.send_header("X-Trade-Sentinel-Fixture", "testing-only")
        self.send_header("X-Fixture-Label", backend.label)
        self.send_header("X-Fixture-Mode", backend.default_mode)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _handle(self, method: str) -> None:
        try:
            length = _parse_content_length(self.headers.get("Content-Length"))
        except ValueError as exc:
            self._reject_length(400, str(exc))
            return
        if length > MAX_BODY_BYTES:
            self._reject_length(413, "request body too large for the fixture harness")
            return
        body = self.rfile.read(length) if length else b""
        response = self.server.backend.dispatch(method, self.path, self.headers, body)
        self.send_response(response.status)
        for name, value in response.headers.items():
            self.send_header(name, value)
        # One Set-Cookie header LINE per cookie (never comma-joined): real
        # browsers parse a single Set-Cookie value as exactly one cookie.
        for cookie in response.cookies:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(response.body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._handle("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle("DELETE")

    def do_PUT(self) -> None:  # noqa: N802
        self._handle("PUT")

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle("PATCH")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._handle("OPTIONS")

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)


class FixtureServer(ThreadingHTTPServer):
    """Threaded local HTTP server carrying the fixture backend."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, address: tuple[str, int], backend: FixtureBackend, verbose: bool = False):
        self.backend = backend
        self.verbose = verbose
        super().__init__(address, FixtureRequestHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline Trade Sentinel UI fixture server (testing only).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765; 0 = ephemeral)")
    parser.add_argument("--empty", action="store_true", help="serve the empty-state fixture instead of populated")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="fixture JSON path")
    parser.add_argument("--verbose", action="store_true", help="log each request (paths only, no headers/bodies)")
    args = parser.parse_args(argv)

    mode = "empty" if args.empty else "populated"
    try:
        backend = FixtureBackend(args.fixture, mode=mode)
    except ValueError as exc:
        print(f"fixture error: {exc}", file=sys.stderr)
        return 2

    server = FixtureServer((args.host, args.port), backend, verbose=args.verbose)
    host, port = server.server_address[:2]
    if host not in {"127.0.0.1", "::1", "localhost"}:
        print("WARNING: fixture harness binds a non-loopback address and is unauthenticated; testing only.", file=sys.stderr)
    print("Trade Sentinel OFFLINE UI fixture - TESTING ONLY", flush=True)
    print(f"  fixture : {backend.label} (mode={mode}, as_of={backend.fixture.get('as_of')})", flush=True)
    print(f"  url     : http://{host}:{port}/", flush=True)
    print(f"  empty   : http://{host}:{port}/?fixture=empty", flush=True)
    print("  db      : ?db=dev sets the ts_db cookie; /api/db/status, /api/db/clone (synthetic, testing only)", flush=True)
    print("  health  : /__fixture__/health   requests: /__fixture__/requests", flush=True)
    print("  CDN deps: NOT served; inject tests/ui_phase3_smoke.js stubs for visual runs", flush=True)
    print("  no app imports, no database, no provider/LLM calls, no outbound requests", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nfixture server stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
