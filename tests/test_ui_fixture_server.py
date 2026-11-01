"""Offline tests for the bounded Phase 3 browser fixture harness.

Every check runs against ``tests/ui_fixture_server.py`` and the synthetic
``tests/fixtures/ui_phase3.json``. No real app, database, provider, LLM or
external network is used; the only socket is a loopback ThreadingHTTPServer on
an ephemeral port.
"""

from __future__ import annotations

import ast
import http.client
import json
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from tests.ui_fixture_server import (
    DEFAULT_FIXTURE,
    REPO_ROOT,
    FixtureBackend,
    _parse_content_length,
)

SERVER_SOURCE = Path(__file__).with_name("ui_fixture_server.py")
PAPER_ACCOUNTS = ("sim", "monthly", "daily_core")
DEV_DB = {"Cookie": "ts_db=dev"}
PROD_DB = {"Cookie": "ts_db=prod"}
# Pinned guard-parity tables (must mirror the fixture server and the real app).
DEV_ONLY_POSTS = (
    "/api/sim/backfill",
    "/api/sim/backfill-benchmark",
    "/api/backfill-all",
    "/api/monthly/backfill",
    "/api/dailycore/backfill",
    "/api/sim/reset",
    "/api/sim/reset-all",
)
SAVINGS_PROD_ONLY_POSTS = tuple(
    f"/api/savings/{name}" for name in (
        "initialize", "reset", "trueup", "config", "deposit", "withdraw",
        "dividend", "fee", "tax", "settle", "split", "correct", "saveback",
        "buy", "plans", "sell", "refresh", "run",
    )
)
SAVINGS_PROD_ONLY_DELETES = (
    "/api/savings/plans/7",
    "/api/savings/positions/VWCE.DE",
)


@contextmanager
def running_server(backend: FixtureBackend):
    from tests.ui_fixture_server import FixtureServer

    server = FixtureServer(("127.0.0.1", 0), backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def http_request(server, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        return response.status, response.headers, data
    finally:
        conn.close()


def get_json(backend: FixtureBackend, path: str, headers: dict | None = None):
    response = backend.dispatch("GET", path, headers or {})
    assert response.status == 200, (path, response.status, response.body)
    return response.json()


def walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from walk_strings(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from walk_strings(value)


# ---------------------------------------------------------------------------
# Fixture document
# ---------------------------------------------------------------------------


def test_fixture_document_is_self_contained_and_populated():
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["label"] == "phase3-ui-fixture"
    assert fixture["version"] == 1
    assert set(fixture["modes"]) == {"populated", "empty"}
    assert "testing only" in fixture["notice"].lower()

    for text in walk_strings(fixture):
        assert "/opt/" not in text
        assert "http://" not in text
        assert "https://" not in text
        assert "../" not in text

    populated = fixture["modes"]["populated"]
    assert populated["watchlist"] == ["AAPL", "MSFT", "NVDA", "JPM"]
    assert populated["names"]["AAPL"] == "Apple Inc."
    assert len(populated["universes"]) >= 1
    assert set(populated["screener"]["results"]) == set(populated["universes"])

    for account, expected_label in zip(PAPER_ACCOUNTS, ("USD", "USD", "USD"), strict=True):
        assert expected_label == "USD"
        portfolio = populated["portfolios"][account]
        assert portfolio["trades"], f"{account} needs nonzero trades"
        assert portfolio["allowances"], f"{account} needs nonzero allowances"
        assert portfolio["equity"], f"{account} needs a dated equity curve"
        for row in portfolio["allowances"]:
            assert row["mode"] == "paper"
            assert row["config_id"] == 42
            assert row["epoch"] == 1
        for row in portfolio["equity"]:
            assert row["mode"] == "paper"
            assert row["config_id"] == 42
            assert row["epoch"] == 1
        assert portfolio["status"]

    savings = populated["savings"]
    assert savings["status"]["initialized"] is True
    assert savings["status"]["positions"]
    assert savings["status"]["plans"]
    assert savings["status"]["events"]
    assert savings["equity"]
    assert savings["mirror"]["curve"]
    assert savings["metrics"]["xirr"]["xirr"] is not None

    assert fixture["modes"]["empty"]["watchlist"] == []
    assert fixture["modes"]["empty"]["savings"]["status"] == {
        "initialized": False, "events": [], "plans": []
    }


# ---------------------------------------------------------------------------
# Route payloads (populated)
# ---------------------------------------------------------------------------


def test_populated_route_payloads_match_frontend_renderers():
    backend = FixtureBackend(mode="populated")

    watchlist = get_json(backend, "/api/watchlist")
    assert watchlist == ["AAPL", "MSFT", "NVDA", "JPM"]
    info = get_json(backend, "/api/info/AAPL")
    assert info["ticker"] == "AAPL"
    assert info["name"] == "Apple Inc."
    symbols = get_json(backend, "/api/symbols?q=app")
    assert symbols and {"symbol", "name", "exchange"} <= set(symbols[0])

    dashboard = get_json(backend, "/api/dashboard/AAPL?period=6m")
    assert {"action", "reason", "snapshot", "strength", "candles"} <= set(dashboard)
    assert dashboard["action"] in {"BUY", "SELL", "HOLD"}
    assert dashboard["snapshot"]["close"] == 214.5
    assert dashboard["candles"]
    assert set(dashboard["candles"][0]) == {"time", "open", "high", "low", "close", "volume"}
    assert dashboard["candles"][0]["time"] < dashboard["candles"][-1]["time"]

    universes = get_json(backend, "/api/screener/universes")
    assert "diversified-plus" in universes
    screen = get_json(backend, "/api/screener/diversified-plus")
    assert screen and {"ticker", "action", "score", "trend", "updated_at"} <= set(screen[0])
    assert get_json(backend, "/api/screener/status")["running"] is False

    sim = get_json(backend, "/api/sim/status")
    assert {"cash", "positions", "positions_value", "total_equity", "allowance_total"} <= set(sim)
    assert sim["positions"] and sim["total_equity"] > sim["allowance_total"]
    assert {"ticker", "shares", "avg_cost", "current_price", "value", "pnl_pct"} <= set(sim["positions"][0])
    sim_trades = get_json(backend, "/api/sim/trades?limit=50")
    assert sim_trades and {"ticker", "side", "shares", "price", "reason", "at"} <= set(sim_trades[0])
    allowances = get_json(backend, "/api/sim/allowances")
    assert allowances and {"amount", "month", "mode", "config_id", "epoch"} <= set(allowances[0])
    equity = get_json(backend, "/api/sim/equity?limit=12000")
    assert len(equity) == 45
    assert equity == sorted(equity, key=lambda row: row["at"])
    benchmark = get_json(backend, "/api/sim/benchmark?limit=12000")
    assert benchmark["ticker"] == "URTH"
    assert benchmark["equity_curve"] and "total_equity" in benchmark["equity_curve"][0]
    assert get_json(backend, "/api/sim/run-status")["running"] is False
    assert get_json(backend, "/api/sim/reasoning")["llm_called"] is False
    assert get_json(backend, "/api/sim/chat/history")[0]["role"] == "user"

    monthly = get_json(backend, "/api/monthly/status")
    assert {"cash", "positions", "total_equity", "allowance_total", "sim_monthly_universe"} <= set(monthly)
    assert monthly["positions"] and "stale" in monthly["positions"][0]
    monthly_trades = get_json(backend, "/api/monthly/trades?limit=50")
    assert monthly_trades and "date" in monthly_trades[0]
    monthly_equity = get_json(backend, "/api/monthly/equity?limit=12000")
    assert monthly_equity == sorted(monthly_equity, key=lambda row: row["date"])
    assert monthly_equity[0]["mode"] == "paper"
    assert get_json(backend, "/api/monthly/rebalances")

    dailycore = get_json(backend, "/api/dailycore/status")
    assert {"valuation", "band", "picks", "rank_map", "mom_variant", "gates", "gradients", "target_vols"} <= set(dailycore)
    assert dailycore["band"] and dailycore["valuation"]["positions"]
    assert dailycore["rank_map"]["AAPL"] == 1
    assert dailycore["target_vol"] == 0.15
    dailycore_equity = get_json(backend, "/api/dailycore/equity?limit=12000")
    assert dailycore_equity == sorted(dailycore_equity, key=lambda row: row["date"])
    assert dailycore_equity[-1]["epoch"] == 1
    assert get_json(backend, "/api/dailycore/trades?limit=50")

    xirr = get_json(backend, "/api/metrics/xirr?scope=sim")
    assert xirr["xirr"] is not None and xirr["window_days"] >= 30
    twr = get_json(backend, "/api/metrics/twr?scope=monthly")
    assert twr["twr_pct"] is not None and twr["points"]

    savings = get_json(backend, "/api/savings/status")
    assert savings["initialized"] is True
    assert {"cash", "positions", "plans", "events", "mirror", "contributed"} <= set(savings)
    assert savings["positions"] and "pnl_pct" in savings["positions"][0]
    assert get_json(backend, "/api/savings/equity?limit=12000") == sorted(
        get_json(backend, "/api/savings/equity?limit=12000"), key=lambda row: row["date"]
    )
    mirror = get_json(backend, "/api/savings/mirror?limit=12000")
    assert mirror["curve"] and {"date", "allocated", "value"} <= set(mirror["curve"][0])
    assert get_json(backend, "/api/savings/metrics/xirr")["scope"] == "savings"
    assert get_json(backend, "/api/savings/metrics/twr")["max_drawdown_pct"] is not None

    llm = get_json(backend, "/api/llm/status")
    assert llm["active_backend"] == "fixture-offline"
    assert llm["backends"][0]["models"]


def test_query_slicing_and_validation():
    backend = FixtureBackend(mode="populated")
    assert len(get_json(backend, "/api/sim/equity?limit=3")) == 3
    assert len(get_json(backend, "/api/monthly/equity?limit=2")) == 2
    assert len(get_json(backend, "/api/dailycore/equity?limit=4")) == 4
    assert len(get_json(backend, "/api/savings/equity?limit=2")) == 2
    assert get_json(backend, "/api/dashboard/AAPL?period=6m")["candles"] == get_json(
        backend, "/api/dashboard/AAPL"
    )["candles"]
    assert len(get_json(backend, "/api/screener/diversified-plus")) == 6
    assert get_json(backend, "/api/metrics/xirr?scope=daily_core")["scope"] == "daily_core"

    for path, expected in (
        ("/api/sim/equity?limit=0", 422),
        ("/api/sim/equity?limit=abc", 422),
        ("/api/metrics/xirr?scope=bogus", 422),
        ("/api/symbols?q=a", 422),
        ("/api/screener/not-a-universe", 404),
        ("/api/dashboard/NOPE", 404),
    ):
        response = backend.dispatch("GET", path)
        assert response.status == expected, (path, response.status)
        assert response.json()["testing_only"] is True


# ---------------------------------------------------------------------------
# Empty mode and mode selection
# ---------------------------------------------------------------------------


def test_empty_mode_renders_empty_states_without_shape_drift():
    backend = FixtureBackend(mode="empty")
    assert get_json(backend, "/api/watchlist") == []
    assert get_json(backend, "/api/sim/status")["positions"] == []
    assert get_json(backend, "/api/sim/trades?limit=50") == []
    assert get_json(backend, "/api/sim/allowances") == []
    assert get_json(backend, "/api/sim/equity?limit=50") == []
    assert get_json(backend, "/api/sim/benchmark")["equity_curve"] == []
    assert get_json(backend, "/api/monthly/status")["total_equity"] == 0.0
    assert get_json(backend, "/api/monthly/equity") == []
    assert get_json(backend, "/api/dailycore/status")["band"] == []
    assert get_json(backend, "/api/dailycore/status")["ranking_date"] is None
    assert get_json(backend, "/api/dailycore/equity") == []
    assert get_json(backend, "/api/screener/diversified-plus") == []
    assert get_json(backend, "/api/savings/status") == {
        "initialized": False, "events": [], "plans": []
    }
    assert get_json(backend, "/api/savings/equity") == []
    assert get_json(backend, "/api/savings/mirror")["curve"] == []
    assert get_json(backend, "/api/metrics/xirr?scope=sim")["points"] == 0
    assert get_json(backend, "/api/metrics/twr?scope=sim")["points"] == []
    assert get_json(backend, "/api/savings/metrics/xirr")["xirr"] is None
    assert get_json(backend, "/api/llm/status")["backends"] == []


def test_cookie_and_query_mode_selection():
    backend = FixtureBackend(mode="populated")
    response = backend.dispatch("GET", "/?fixture=empty")
    assert response.status == 200
    assert any(c.startswith("ts_fixture_mode=empty") for c in response.cookies)
    assert "Set-Cookie" not in response.headers  # cookies live in response.cookies now

    empty_status = backend.dispatch("GET", "/api/savings/status", {"Cookie": "ts_fixture_mode=empty"}).json()
    assert empty_status["initialized"] is False
    populated_status = backend.dispatch("GET", "/api/savings/status").json()
    assert populated_status["initialized"] is True

    assert backend.dispatch("GET", "/api/watchlist?fixture=empty").json() == []
    assert backend.dispatch("GET", "/api/watchlist?fixture=populated").json() == ["AAPL", "MSFT", "NVDA", "JPM"]
    invalid = backend.dispatch("GET", "/?fixture=nope")
    assert invalid.status == 400
    assert invalid.json()["detail"].startswith("unknown fixture mode")


# ---------------------------------------------------------------------------
# Fail-closed routing / static safety
# ---------------------------------------------------------------------------


def test_unknown_api_routes_fail_closed_501():
    backend = FixtureBackend(mode="populated")
    for method, path in (
        ("GET", "/api/not-a-real-route"),
        ("GET", "/api/dailycore/allowances"),
        ("POST", "/api/not-a-real-route"),
        ("DELETE", "/api/not-a-real-route"),
    ):
        response = backend.dispatch(method, path)
        assert response.status == 501, (method, path, response.status)
        payload = response.json()
        assert payload["detail"] == "unknown fixture API route"
        assert payload["testing_only"] is True
        assert b"/opt/" not in response.body
        assert response.headers["X-Trade-Sentinel-Fixture"] == "testing-only"


def test_static_serving_and_path_traversal_refusal():
    backend = FixtureBackend(mode="populated")
    index = backend.dispatch("GET", "/")
    assert index.status == 200
    assert b"<title>Trade Sentinel</title>" in index.body
    assert index.headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert backend.dispatch("GET", "/index.html").status == 200
    assert backend.dispatch("GET", "/favicon.svg").status == 200
    assert backend.dispatch("GET", "/site.webmanifest").status == 200

    for path in (
        "/../AGENTS.md",
        "/%2e%2e/AGENTS.md",
        "/..%2fAGENTS.md",
        "/%2e%2e%2fAGENTS.md",
        "/app/main.py",
        "/tests/ui_fixture_server.py",
        "/static/../AGENTS.md",
        "/%00/x",
        "/..\\AGENTS.md",
    ):
        response = backend.dispatch("GET", path)
        assert response.status in {400, 404}, (path, response.status)
        assert b"Global agent rules" not in response.body
        assert b"FastAPI" not in response.body
        assert b"/opt/stacks" not in response.body
        assert response.headers["X-Trade-Sentinel-Fixture"] == "testing-only"

    unknown = backend.dispatch("GET", "/no-such-asset.txt")
    assert unknown.status == 404
    assert unknown.json()["testing_only"] is True


def test_unsupported_methods_and_oversized_paths_are_refused():
    backend = FixtureBackend(mode="populated")
    assert backend.dispatch("PUT", "/api/sim/status").status == 405
    assert backend.dispatch("POST", "/api/sim/status").status == 501
    assert backend.dispatch("GET", "///example.com/x").status == 400
    assert backend.dispatch("GET", "http://example.com/api/watchlist").status == 400


# ---------------------------------------------------------------------------
# Recording, ack-only mutations, headers
# ---------------------------------------------------------------------------


def test_request_recording_exposes_scopes_and_kinds():
    backend = FixtureBackend(mode="populated")
    backend.dispatch("GET", "/api/metrics/xirr?scope=sim")
    backend.dispatch("GET", "/api/metrics/twr?scope=monthly&limit=5")
    backend.dispatch("POST", "/api/savings/deposit", body=b'{"amount": 25}')

    payload = get_json(backend, "/__fixture__/requests")
    entries = payload["requests"]
    assert payload["count"] == 3
    assert payload["testing_only"] is True
    assert entries[0]["path"] == "/api/metrics/xirr"
    assert entries[0]["query"] == {"scope": ["sim"]}
    assert entries[0]["kind"] == "fixture_api"
    assert entries[1]["query"] == {"scope": ["monthly"], "limit": ["5"]}
    assert entries[2]["method"] == "POST"
    assert entries[2]["kind"] == "post_ack"

    clear = backend.dispatch("DELETE", "/__fixture__/requests")
    assert clear.status == 200
    assert clear.json()["cleared"] == 4  # the three probes + the recording read itself
    assert get_json(backend, "/__fixture__/requests")["count"] == 1  # the DELETE itself


def test_post_routes_are_acknowledgements_only():
    backend = FixtureBackend(mode="populated")
    before_status = get_json(backend, "/api/sim/status")
    before_trades = get_json(backend, "/api/sim/trades?limit=50")
    before_savings = get_json(backend, "/api/savings/status")

    checks = [
        ("POST", "/api/sim/run", b"", None),
        ("POST", "/api/sim/reset", b"", DEV_DB),  # dev-only: 403 on prod by design
        ("POST", "/api/savings/deposit", b'{"amount": 25}', None),
        ("POST", "/api/savings/buy", b'{"amount": 100, "ticker": "VWCE.DE"}', None),
        ("POST", "/api/dailycore/gate", b'{"mode": "100"}', None),
        ("POST", "/api/dailycore/strategy", b'{"variant": "raw"}', None),
        ("POST", "/api/screener/run/diversified-plus", b"", None),
        ("DELETE", "/api/watchlist/AAPL", b"", None),
        ("DELETE", "/api/savings/positions/SAP.DE", b"", None),
    ]
    for method, path, body, headers in checks:
        response = backend.dispatch(method, path, headers=headers, body=body)
        assert response.status == 200, (method, path, response.status, response.body)
        payload = response.json()
        assert payload["fixture_ack"] is True
        assert payload["applied"] is False
        assert b"http://" not in response.body
        assert b"https://" not in response.body

    assert get_json(backend, "/api/sim/status") == before_status
    assert get_json(backend, "/api/sim/trades?limit=50") == before_trades
    assert get_json(backend, "/api/savings/status") == before_savings


def test_health_reports_label_mode_and_zero_external_claims():
    backend = FixtureBackend(mode="populated")
    health = get_json(backend, "/__fixture__/health")
    assert health["fixture_label"] == "phase3-ui-fixture"
    assert health["mode"] == "populated"
    assert health["default_mode"] == "populated"
    assert health["available_modes"] == ["empty", "populated"]
    assert health["outbound_network_requests"] == 0
    assert health["cdn_dependencies_served"] is False
    assert health["app_modules_imported"] is False
    assert health["mutation_route_policy"].startswith("acknowledgement-only")
    assert set(health["counters"]) >= {
        "total", "static", "fixture_api", "post_ack", "unknown_api", "rejected"
    }
    assert health["counters"]["total"] == 0  # health itself is recorded after the snapshot


# ---------------------------------------------------------------------------
# Loopback HTTP round trip (still fully offline)
# ---------------------------------------------------------------------------


def test_threading_http_server_round_trip():
    backend = FixtureBackend(mode="populated")
    with running_server(backend) as server:
        status, headers, body = http_request(server, "GET", "/")
        assert status == 200
        assert b"<title>Trade Sentinel</title>" in body
        assert headers["X-Trade-Sentinel-Fixture"] == "testing-only"
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert headers["Access-Control-Allow-Origin"] is None

        status, headers, body = http_request(server, "GET", "/api/sim/status")
        assert status == 200
        assert headers["X-Fixture-Mode"] == "populated"
        assert json.loads(body)["total_equity"] == 11149.77

        status, headers, body = http_request(
            server,
            "POST",
            "/api/sim/run",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert json.loads(body)["fixture_ack"] is True

        status, headers, body = http_request(server, "HEAD", "/")
        assert status == 200
        assert body == b""
        assert int(headers["Content-Length"]) > 1000

        status, headers, body = http_request(server, "GET", "/api/not-a-route")
        assert status == 501
        assert json.loads(body)["testing_only"] is True


def test_stream_route_is_deterministic_and_local():
    backend = FixtureBackend(mode="populated")
    response = backend.dispatch("POST", "/api/sim/chat/stream", body=b"{}")
    assert response.status == 200
    assert response.content_type.startswith("text/event-stream")
    text = response.body.decode("utf-8")
    assert text.startswith("data: ")
    assert '"type":"done"' in text
    assert "Fixture chat stub" in text
    assert "http://" not in text
    assert "https://" not in text


def test_content_length_is_validated_before_body_read():
    assert _parse_content_length(None) == 0
    assert _parse_content_length("") == 0
    assert _parse_content_length(" 12 ") == 12
    for value in ("-1", "abc", "1e3"):
        try:
            _parse_content_length(value)
        except ValueError:
            pass
        else:  # pragma: no cover - the failure message is the assertion
            raise AssertionError(f"{value!r} should be rejected")

    backend = FixtureBackend(mode="populated")
    with running_server(backend) as server:
        port = server.server_address[1]
        for header in ("-1", "nope"):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                request = (
                    "POST /api/sim/run HTTP/1.1\r\n"
                    "Host: fixture\r\n"
                    f"Content-Length: {header}\r\n"
                    "Connection: close\r\n\r\n"
                )
                sock.sendall(request.encode("ascii"))
                chunks = []
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
            assert data.startswith(b"HTTP/1.1 400"), (header, data[:80])
            assert b"testing_only" in data

        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            request = (
                "POST /api/sim/run HTTP/1.1\r\n"
                "Host: fixture\r\n"
                "Content-Length: 1000001\r\n"
                "Connection: close\r\n\r\n"
            )
            sock.sendall(request.encode("ascii"))
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            data = b"".join(chunks)
        assert data.startswith(b"HTTP/1.1 413"), data[:80]
        assert b"testing_only" in data


# ---------------------------------------------------------------------------
# No app/network imports
# ---------------------------------------------------------------------------


def test_server_module_imports_no_app_or_network_client_modules():
    tree = ast.parse(SERVER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {
        "app", "fastapi", "starlette", "uvicorn", "httpx", "requests",
        "sqlalchemy", "aiosqlite", "yfinance", "urllib3", "aiohttp",
    }
    assert not (imported & forbidden), imported & forbidden
    assert "http" in imported  # stdlib server, not a client library
    assert "urllib" in imported  # stdlib URL parsing only


def test_dispatch_and_mutations_import_no_app_modules_in_fresh_process():
    code = """
import json
import sys
sys.path.insert(0, '.')
from tests.ui_fixture_server import FixtureBackend
backend = FixtureBackend(mode='populated')
for method, path, body in (
    ('GET', '/api/sim/status', b''),
    ('GET', '/api/dailycore/status', b''),
    ('POST', '/api/sim/run', b''),
    ('POST', '/api/savings/deposit', b'{"amount": 1}'),
    ('POST', '/api/screener/run/diversified-plus', b''),
):
    response = backend.dispatch(method, path, body=body)
    assert response.status in (200, 501), (method, path, response.status)
leaked = sorted(name for name in sys.modules if name == 'app' or name.startswith('app.'))
print(json.dumps({'leaked': leaked}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"leaked": []}


def test_lab_global_actions_are_single_acks_and_dailycore_equity_is_nested():
    backend = FixtureBackend(mode="populated")
    backend.dispatch("DELETE", "/__fixture__/requests")
    backfill = backend.dispatch("POST", "/api/backfill-all", headers=DEV_DB)  # dev-only route
    assert backfill.status == 200
    payload = backfill.json()
    assert payload["fixture_ack"] is True
    assert payload["applied"] is False
    assert payload["results"]["daily_sim"]["final_equity"] == 11149.77
    assert payload["results"]["monthly"]["final_equity"] == 11106.39
    assert payload["results"]["daily_core"]["final_equity"] == 10371.29
    assert payload["results"]["benchmark"]["final_equity"] == 4019.40

    reset = backend.dispatch("POST", "/api/sim/reset-all", headers=DEV_DB)  # dev-only route
    assert reset.status == 200
    assert reset.json()["fixture_ack"] is True
    assert reset.json()["applied"] is False

    requests = backend.dispatch("GET", "/__fixture__/requests").json()["requests"]
    mutating = [
        entry for entry in requests
        if not entry["path"].startswith("/__fixture__/") and entry["method"] in {"POST", "DELETE", "PUT", "PATCH"}
    ]
    assert [(entry["method"], entry["path"]) for entry in mutating] == [
        ("POST", "/api/backfill-all"),
        ("POST", "/api/sim/reset-all"),
    ]
    # The two Lab actions must not trigger any equity read before navigation.
    assert not [entry for entry in requests if entry["path"].endswith("/equity")]


def test_optional_playwright_smoke_is_offline_and_honestly_labelled():
    smoke = SERVER_SOURCE.with_name("ui_phase3_smoke.js")
    text = smoke.read_text(encoding="utf-8")
    assert '"/opt/stacks/.playwright-out"' in text
    assert "page.route(\"**/*\"" in text
    assert "page.addInitScript" in text
    assert "page.on(\"pageerror\"" in text
    assert "provider_api_components" in text
    assert "provider_api_all_zero" in text
    assert "all_view_checks_ok" in text
    assert "viewResults.every((entry) => entry.ok)" in text
    assert "doc_scroll_width" in text
    assert "overflow:" in text
    assert "readiness_checks" in text
    assert "FIXTURE CHART STUB - layout/navigation only (not real canvas)" in text
    assert "limitation: \"DOM placeholder only" in text
    assert "marked.setOptions" not in text  # definitions use object shorthand, not the app call
    assert "setOptions() { }" in text
    assert "Renderer: function Renderer()" in text
    assert "parse(text)" in text
    assert "integrity" in text
    assert "removeAttribute(\"integrity\")" not in text


# ---------------------------------------------------------------------------
# PROD/DEV database-switch simulation (synthetic, testing only)
# ---------------------------------------------------------------------------


def test_db_status_route_reflects_cookie_query_and_defaults():
    backend = FixtureBackend(mode="populated")
    assert get_json(backend, "/api/db/status") == {
        "active": "prod", "dev_enabled": True, "dev_seeded": True, "cloning": False,
    }
    assert get_json(backend, "/api/db/status?db=dev")["active"] == "dev"
    assert get_json(backend, "/api/db/status", PROD_DB)["active"] == "prod"
    assert get_json(backend, "/api/db/status", DEV_DB)["active"] == "dev"
    # Missing/unknown cookie values fall back to prod.
    assert get_json(backend, "/api/db/status", {"Cookie": "ts_db=bogus"})["active"] == "prod"
    assert get_json(backend, "/api/db/status", {"Cookie": "other=1"})["active"] == "prod"
    # Unknown ?db= query values are rejected like unknown fixture modes.
    invalid = backend.dispatch("GET", "/api/db/status?db=nope")
    assert invalid.status == 400
    assert invalid.json()["detail"].startswith("unknown db selection")

    empty = FixtureBackend(mode="empty")
    assert get_json(empty, "/api/db/status") == {
        "active": "prod", "dev_enabled": True, "dev_seeded": False, "cloning": False,
    }
    assert get_json(empty, "/api/db/status?db=dev") == {
        "active": "dev", "dev_enabled": True, "dev_seeded": False, "cloning": False,
    }


def test_db_status_falls_back_to_prod_when_dev_disabled():
    backend = FixtureBackend(mode="populated")
    backend.modes["populated"]["db"]["dev_enabled"] = False
    assert get_json(backend, "/api/db/status?db=dev") == {
        "active": "prod", "dev_enabled": False, "dev_seeded": True, "cloning": False,
    }
    assert get_json(backend, "/api/db/status", DEV_DB)["active"] == "prod"


def test_db_query_on_index_sets_ts_db_cookie_next_to_mode_cookie():
    backend = FixtureBackend(mode="populated")
    response = backend.dispatch("GET", "/?fixture=empty&db=dev")
    assert response.status == 200
    # One entry per cookie; the HTTP layer emits each as its own Set-Cookie line.
    assert any(c.startswith("ts_fixture_mode=empty") for c in response.cookies)
    assert any(c.startswith("ts_db=dev") for c in response.cookies)
    assert "Set-Cookie" not in response.headers

    plain = backend.dispatch("GET", "/")
    assert any(c.startswith("ts_fixture_mode=populated") for c in plain.cookies)
    assert not any(c.startswith("ts_db=") for c in plain.cookies)

    cookie_mode = backend.dispatch("GET", "/", {"Cookie": "ts_fixture_mode=empty"})
    assert cookie_mode.cookies == []


def test_index_cookies_are_emitted_as_separate_set_cookie_header_lines():
    """Real-browser contract: one cookie per Set-Cookie header (RFC 6265).

    A comma-joined single Set-Cookie value is parsed by browsers as ONE cookie,
    silently dropping ts_db; http.client's get_all() proves each cookie gets
    its own header line on the wire.
    """
    backend = FixtureBackend(mode="populated")
    with running_server(backend) as server:
        status, headers, _body = http_request(server, "GET", "/?fixture=empty&db=dev")
        assert status == 200
        set_cookies = headers.get_all("Set-Cookie")
        assert set_cookies is not None and len(set_cookies) == 2, set_cookies
        assert any(c.startswith("ts_fixture_mode=empty") for c in set_cookies)
        assert any(c.startswith("ts_db=dev") for c in set_cookies)

        status, headers, _body = http_request(server, "GET", "/")
        assert status == 200
        set_cookies = headers.get_all("Set-Cookie")
        assert set_cookies is not None and len(set_cookies) == 1, set_cookies
        assert set_cookies[0].startswith("ts_fixture_mode=populated")


def test_db_clone_acks_with_size_and_both_409_paths():
    backend = FixtureBackend(mode="populated")
    response = backend.dispatch("POST", "/api/db/clone", headers=DEV_DB, body=b"{}")
    assert response.status == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["fixture_ack"] is True
    assert payload["applied"] is False
    assert isinstance(payload["size_bytes"], int) and payload["size_bytes"] > 0

    backend.modes["populated"]["db"]["cloning"] = True
    cloning = backend.dispatch("POST", "/api/db/clone", body=b"{}")
    assert cloning.status == 409
    assert cloning.json()["testing_only"] is True
    assert "detail" in cloning.json()

    backend.modes["populated"]["db"]["cloning"] = False
    backend.modes["populated"]["db"]["dev_enabled"] = False
    disabled = backend.dispatch("POST", "/api/db/clone", body=b"{}")
    assert disabled.status == 409
    assert disabled.json()["testing_only"] is True


def test_db_guard_parity_dev_only_destructive_posts():
    backend = FixtureBackend(mode="populated")
    for path in DEV_ONLY_POSTS:
        blocked = backend.dispatch("POST", path, headers=PROD_DB, body=b"{}")
        assert blocked.status == 403, (path, blocked.status)
        assert blocked.json()["detail"]
        assert blocked.json()["testing_only"] is True
        default_blocked = backend.dispatch("POST", path, body=b"{}")
        assert default_blocked.status == 403, (path, default_blocked.status)
        allowed = backend.dispatch("POST", path, headers=DEV_DB, body=b"{}")
        assert allowed.status == 200, (path, allowed.status)
        assert allowed.json()["fixture_ack"] is True
        assert allowed.json()["applied"] is False
        via_query = backend.dispatch("POST", path + "?db=dev", body=b"{}")
        assert via_query.status == 200, (path, via_query.status)


def test_db_guard_parity_prod_only_savings_writes():
    backend = FixtureBackend(mode="populated")
    for path in SAVINGS_PROD_ONLY_POSTS:
        allowed = backend.dispatch("POST", path, headers=PROD_DB, body=b"{}")
        assert allowed.status == 200, (path, allowed.status)
        assert allowed.json()["fixture_ack"] is True
        blocked = backend.dispatch("POST", path, headers=DEV_DB, body=b"{}")
        assert blocked.status == 403, (path, blocked.status)
        via_query = backend.dispatch("POST", path + "?db=dev", body=b"{}")
        assert via_query.status == 403, (path, via_query.status)
    for path in SAVINGS_PROD_ONLY_DELETES:
        assert backend.dispatch("DELETE", path, headers=PROD_DB).status == 200, path
        blocked = backend.dispatch("DELETE", path, headers=DEV_DB)
        assert blocked.status == 403, (path, blocked.status)
        assert blocked.json()["testing_only"] is True


def test_db_guard_parity_leaves_unguarded_routes_alone():
    backend = FixtureBackend(mode="populated")
    unguarded = (
        ("POST", "/api/sim/run"),
        ("POST", "/api/sim/refresh"),
        ("POST", "/api/monthly/run"),
        ("POST", "/api/monthly/refresh"),
        ("POST", "/api/dailycore/run"),
        ("POST", "/api/dailycore/refresh"),
        ("POST", "/api/dailycore/strategy"),
        ("POST", "/api/dailycore/gate"),
        ("POST", "/api/dailycore/gradient"),
        ("POST", "/api/dailycore/targetvol"),
        ("POST", "/api/sim/universe"),
        ("POST", "/api/screener/sync"),
        ("POST", "/api/screener/run/diversified-plus"),
        ("POST", "/api/chat/AAPL/stream"),
        ("POST", "/api/llm/select"),
        # A non-empty ticker: an empty one is a 422 now (live parity), so the
        # unguarded-ness probe uses an unknown-but-valid ticker (200 ok:false).
        ("POST", "/api/news/consult", b'{"ticker": "ZZZ"}'),
        ("POST", "/api/news/scan", b"{}"),
        ("GET", "/api/news/scan/status"),
        ("DELETE", "/api/sim/chat/history"),
        ("DELETE", "/api/watchlist/AAPL"),
        ("POST", "/api/refresh/AAPL"),
    )
    for entry in unguarded:
        method, path = entry[0], entry[1]
        body = entry[2] if len(entry) > 2 else b"{}"
        for headers in (None, DEV_DB, PROD_DB):
            response = backend.dispatch(method, path, headers=headers, body=body)
            assert response.status == 200, (method, path, headers, response.status)
    # Savings GETs (incl. GET savings/plans) stay unguarded on dev.
    assert get_json(backend, "/api/savings/plans", DEV_DB) is not None
    assert get_json(backend, "/api/savings/status", DEV_DB)["initialized"] is True
    assert get_json(backend, "/api/db/status", DEV_DB)["active"] == "dev"


def test_unknown_db_routes_fail_closed_501():
    backend = FixtureBackend(mode="populated")
    for method, path in (
        ("GET", "/api/db"),
        ("GET", "/api/db/nope"),
        ("POST", "/api/db/nope"),
        ("POST", "/api/db/status"),
        ("DELETE", "/api/db/status"),
        ("DELETE", "/api/db/clone"),
    ):
        response = backend.dispatch(method, path, body=b"{}")
        assert response.status == 501, (method, path, response.status)
        payload = response.json()
        assert payload["detail"] == "unknown fixture API route"
        assert payload["testing_only"] is True


# ---------------------------------------------------------------------------
# Phase 3D/3F fixture routes: new sim-chat envelope, confirm/reject acks,
# GET /api/run-config
# ---------------------------------------------------------------------------

PREVIEW_KEYS = {"feasible", "issue", "price", "est_shares", "est_value",
                "est_cash_after", "est_position_value_after"}


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item)


def test_sim_chat_envelope_has_proposals_and_no_legacy_keys():
    for mode in ("populated", "empty"):
        backend = FixtureBackend(mode=mode)
        response = backend.dispatch("POST", "/api/sim/chat", body=b'{"messages": [{"role": "user", "content": "hi"}]}')
        assert response.status == 200, (mode, response.status)
        assert response.kind == "post_ack"
        payload = response.json()
        assert payload["fixture_ack"] is True and payload["applied"] is False
        assert set(payload) >= {"text", "proposals", "history"}
        # The legacy envelope keys are gone everywhere in the payload.
        assert "trades" not in set(_walk_keys(payload))
        assert "actions_executed" not in set(_walk_keys(payload))
        assert isinstance(payload["history"], list)

        proposals = payload["proposals"]
        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal["proposal_id"] == "fixture-proposal-3d-0001"
        assert proposal["decided_at"] == "2026-10-30T12:00:00+00:00"
        assert isinstance(proposal["config"], dict)
        assert len(proposal["actions"]) == 2
        feasible = next(a for a in proposal["actions"] if a["preview"]["feasible"] is True)
        infeasible = next(a for a in proposal["actions"] if a["preview"]["feasible"] is False)
        assert feasible["action"] == "BUY" and feasible["ticker"]
        assert PREVIEW_KEYS <= set(feasible["preview"])
        assert feasible["preview"]["price"] > 0
        assert feasible["preview"]["est_shares"] > 0
        assert feasible["preview"]["est_cash_after"] is not None
        assert infeasible["preview"]["issue"] == "no price"
        assert infeasible["preview"]["est_value"] is None


def test_sim_chat_stream_done_carries_proposals_not_trades():
    backend = FixtureBackend(mode="populated")
    response = backend.dispatch("POST", "/api/sim/chat/stream", body=b"{}")
    assert response.status == 200
    assert response.content_type.startswith("text/event-stream")
    text = response.body.decode("utf-8")
    assert '"type":"done"' in text
    assert '"proposals"' in text
    assert '"history"' in text
    assert '"trades"' not in text
    assert '"actions_executed"' not in text
    assert "http://" not in text and "https://" not in text


def test_sim_chat_confirm_and_reject_are_fixture_acks():
    backend = FixtureBackend(mode="populated")
    before = backend.dispatch("GET", "/__fixture__/health").json()["counters"]["post_ack"]

    confirm = backend.dispatch("POST", "/api/sim/chat/confirm", body=b'{"proposal_id": "fixture-proposal-3d-0001"}')
    assert confirm.status == 200
    payload = confirm.json()
    assert payload["fixture_ack"] is True and payload["applied"] is False
    assert payload["ok"] is True
    assert payload["snapshot_taken"] is True
    assert payload["skipped"] == []
    assert payload["trades"] and {"ticker", "side", "shares", "price"} <= set(payload["trades"][0])

    reject = backend.dispatch("POST", "/api/sim/chat/reject", body=b'{"proposal_id": "fixture-proposal-3d-0001"}')
    assert reject.status == 200
    assert reject.json()["ok"] is True
    assert reject.json()["fixture_ack"] is True

    after = backend.dispatch("GET", "/__fixture__/health").json()["counters"]["post_ack"]
    assert after == before + 2

    # confirm/reject are NOT savings writes: unguarded on both databases,
    # exactly like the existing sim/chat POSTs.
    for headers in (None, PROD_DB, DEV_DB):
        for path in ("/api/sim/chat/confirm", "/api/sim/chat/reject", "/api/sim/chat", "/api/sim/chat/stream"):
            response = backend.dispatch("POST", path, headers=headers, body=b'{"proposal_id": "x"}')
            assert response.status == 200, (path, headers, response.status)

    # Unknown POST paths (incl. near-miss proposal routes) still fail closed.
    for path in ("/api/sim/chat/confir", "/api/sim/chat/confirm/extra", "/api/sim/chat/unknown"):
        response = backend.dispatch("POST", path, body=b"{}")
        assert response.status == 501, (path, response.status)
        assert response.json()["detail"] == "unknown fixture API route"


def test_run_config_route_serves_canned_live_config_both_modes():
    for mode in ("populated", "empty"):
        backend = FixtureBackend(mode=mode)
        response = backend.dispatch("GET", "/api/run-config")
        assert response.status == 200, (mode, response.status)
        assert response.kind == "fixture_api"
        payload = response.json()
        assert set(payload) == {"sim", "monthly", "daily_core"}
        for engine in ("sim", "monthly", "daily_core"):
            entry = payload[engine]
            assert entry["engine"] == engine
            assert isinstance(entry["epoch"], int) and entry["epoch"] >= 1
            # a couple of real current_config fields beside engine/epoch
            assert len([k for k in entry if k not in {"engine", "epoch"}]) >= 2
        assert payload["sim"]["epoch"] == 1
        # read-only: unguarded on dev too, and not an ack
        dev = backend.dispatch("GET", "/api/run-config", DEV_DB)
        assert dev.status == 200 and dev.kind == "fixture_api"
        assert "fixture_ack" not in dev.json()


def test_ticker_chat_stream_envelope_drops_legacy_trades_key():
    backend = FixtureBackend(mode="populated")
    sse = backend.dispatch("POST", "/api/chat/AAPL/stream", body=b"{}")
    text = sse.body.decode("utf-8")
    assert '"trades"' not in text and '"actions_executed"' not in text


# ---------------------------------------------------------------------------
# Type-D news-agent routes: fixture parity with app/main.py's /api/news/*
# (POST /api/news/consult = fail-closed synthetic ack, NO real consult;
#  POST /api/news/scan = fail-closed ack of the background full scan;
#  GET /api/news/scan/status = one synthetic COMPLETED scan state;
#  GET /api/news/decisions/{event_id} = one synthetic D row)
# ---------------------------------------------------------------------------

# The live route's response shape (app/main.py news_consult) — the fixture
# ack must carry exactly these keys (plus fixture_ack/applied).
NEWS_CONSULT_RESPONSE_KEYS = {
    "ok", "decision", "rationale", "citations", "turns", "tool_calls",
    "latency_ms", "fail_safe", "fail_reason", "event_id", "model", "backend",
}
# The live decisions route's keys (app/main.py news_decision).
NEWS_DECISION_RESPONSE_KEYS = {
    "event_id", "event_type", "engine", "ticker", "session_date", "decision",
    "rationale", "parsed", "fail_safe", "fail_reason", "prompt_version",
    "context", "engine_branch", "backend", "model", "latency_ms", "decided_at",
}
# The live scan routes' shapes (app/main.py news_scan / news_scan_status):
# the start ack and the scan-state keys.
NEWS_SCAN_ACK_KEYS = {"ok", "started", "total"}
NEWS_SCAN_STATE_KEYS = {
    "running", "started_at", "finished_at", "as_of", "total", "done",
    "current", "results", "errors",
}
NEWS_SCAN_RESULT_KEYS = {"ticker", "decision", "fail_safe", "event_id"}
NEWS_ROUTES = (
    ("POST", "/api/news/consult"),
    ("POST", "/api/news/scan"),
    ("GET", "/api/news/scan/status"),
    ("GET", "/api/news/decisions/{event_id}"),
)


def test_news_consult_route_is_a_fail_closed_synthetic_ack():
    for mode in ("populated", "empty"):
        backend = FixtureBackend(mode=mode)
        as_of = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))["as_of"][:10]
        monthly = get_json(backend, "/api/monthly/status")
        tickers = [p["ticker"] for p in monthly["positions"]]
        for ticker in (tickers[:1] or ["MSFT"]):
            response = backend.dispatch(
                "POST", "/api/news/consult", body=json.dumps({"ticker": ticker}).encode())
            assert response.status == 200, (mode, response.status)
            assert response.kind == "post_ack"
            payload = response.json()
            assert payload["fixture_ack"] is True
            assert payload["applied"] is False       # NO real consult ran
            assert b"http://" not in response.body and b"https://" not in response.body
            if mode == "populated" and ticker in tickers:
                assert payload["ok"] is True
                assert NEWS_CONSULT_RESPONSE_KEYS <= set(payload)
                assert payload["event_id"] == f"D:{ticker}:{as_of}:m"
                assert payload["decision"] in {"EXIT", "HOLD"}
            else:
                assert payload["ok"] is False
                assert {"ok", "reason"} <= set(payload)
        # unknown ticker: honest ok:false, still an ack, still 200 (live parity)
        response = backend.dispatch("POST", "/api/news/consult", body=b'{"ticker": "ZZZ"}')
        assert response.status == 200, (mode, response.status)
        payload = response.json()
        assert payload["ok"] is False and payload["reason"]
        assert payload["fixture_ack"] is True and payload["applied"] is False
        # empty ticker: 422 with the live detail string (app/main.py:935)
        for body in (b'{"ticker": "  "}', b"{}"):
            response = backend.dispatch("POST", "/api/news/consult", body=body)
            assert response.status == 422, (mode, body, response.status)
            assert response.json()["detail"] == "ticker must not be empty"
            assert b"http://" not in response.body and b"https://" not in response.body


def test_news_decisions_route_serves_one_synthetic_d_row():
    backend = FixtureBackend(mode="populated")
    as_of = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))["as_of"][:10]
    ticker = get_json(backend, "/api/monthly/status")["positions"][0]["ticker"]
    event_id = f"D:{ticker}:{as_of}:m"
    response = backend.dispatch("GET", f"/api/news/decisions/{event_id}")
    assert response.status == 200, response.status
    assert response.kind == "fixture_api"
    payload = response.json()
    assert NEWS_DECISION_RESPONSE_KEYS <= set(payload)
    assert payload["event_id"] == event_id and payload["event_type"] == "D"
    assert payload["engine"] == "monthly" and payload["ticker"] == ticker
    assert payload["session_date"] == as_of and payload["decided_at"]
    ctx = payload["context"]
    assert ctx["position"]["ticker"] == ticker
    assert isinstance(ctx["citations"], list) and ctx["trace"]
    assert ctx["news_source"] == "searxng-mcp"
    # fail-closed: everything else 404s/501s like the live app
    for path in (
        f"/api/news/decisions/D:{ticker}:{as_of}",       # 3-part nightly id: not stored
        "/api/news/decisions/A:AAA:2026-03-06",          # non-D event id
        "/api/news/decisions/D:ZZZ:2026-10-30:m",        # ticker not in the book
        "/api/news/decisions/D:MSFT:2026-10-30",         # no :m marker
        "/api/news/decisions/nope",
        "/api/news/other",
        "/api/news",
    ):
        r = backend.dispatch("GET", path)
        assert r.status in (404, 501), (path, r.status)
    assert backend.dispatch("POST", f"/api/news/decisions/{event_id}", body=b"{}").status == 501
    assert backend.dispatch("GET", "/api/news/consult").status == 501
    # empty mode: nothing held -> the consult acks ok:false, decisions 404
    empty = FixtureBackend(mode="empty")
    r = empty.dispatch("POST", "/api/news/consult", body=b'{"ticker": "MSFT"}')
    assert r.status == 200 and r.json()["ok"] is False
    assert empty.dispatch("GET", f"/api/news/decisions/{event_id}").status == 404


def test_news_routes_are_unguarded_like_the_live_app():
    """app/main.py guards /api/news/consult only with _refuse_during_clone
    (never dev-only/prod-only), so the fixture must answer 200 on BOTH
    synthetic databases — parity with test_db_guard_parity_leaves_unguarded."""
    backend = FixtureBackend(mode="populated")
    as_of = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))["as_of"][:10]
    ticker = get_json(backend, "/api/monthly/status")["positions"][0]["ticker"]
    event_id = f"D:{ticker}:{as_of}:m"
    for headers in (None, PROD_DB, DEV_DB):
        consult = backend.dispatch("POST", "/api/news/consult", headers=headers,
                                   body=json.dumps({"ticker": ticker}).encode())
        assert consult.status == 200, (headers, consult.status)
        decision = backend.dispatch("GET", f"/api/news/decisions/{event_id}", headers)
        assert decision.status == 200, (headers, decision.status)
        scan = backend.dispatch("POST", "/api/news/scan", headers=headers, body=b"{}")
        assert scan.status == 200, (headers, scan.status)
        status = backend.dispatch("GET", "/api/news/scan/status", headers)
        assert status.status == 200, (headers, status.status)


def _union_size(backend: FixtureBackend, mode: str) -> int:
    monthly = [p["ticker"] for p in get_json(backend, "/api/monthly/status")["positions"]]
    dc_status = get_json(backend, "/api/dailycore/status")
    daily = [p["ticker"] for p in dc_status["valuation"]["positions"]]
    return len(set(monthly) | set(daily))


def test_news_scan_route_is_a_fail_closed_synthetic_ack():
    """POST /api/news/scan mirrors app/main.py's news_scan: a fail-closed
    synthetic ack ({ok, started, total}) with total = the UNION of both paper
    books' held tickers — NO scan is ever started (no agent/LLM/search)."""
    for mode in ("populated", "empty"):
        backend = FixtureBackend(mode=mode)
        expected_total = _union_size(backend, mode)
        response = backend.dispatch("POST", "/api/news/scan", body=b"{}")
        assert response.status == 200, (mode, response.status)
        assert response.kind == "post_ack"
        payload = response.json()
        assert payload["fixture_ack"] is True
        assert payload["applied"] is False          # NO real scan ran
        assert NEWS_SCAN_ACK_KEYS <= set(payload)
        assert payload["ok"] is True and payload["started"] is True
        assert payload["total"] == expected_total
        assert b"http://" not in response.body and b"https://" not in response.body


def test_news_scan_status_serves_one_synthetic_completed_state():
    """GET /api/news/scan/status mirrors app/main.py's news_scan_status: one
    synthetic COMPLETED state (running false) with 2 results on the
    populated fixture; the no-http(s)-in-bodies guarantee holds."""
    backend = FixtureBackend(mode="populated")
    as_of = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))["as_of"][:10]
    response = backend.dispatch("GET", "/api/news/scan/status")
    assert response.status == 200, response.status
    assert response.kind == "fixture_api"
    payload = response.json()
    assert set(payload) == NEWS_SCAN_STATE_KEYS
    assert payload["running"] is False
    assert payload["as_of"] == as_of
    assert payload["total"] == payload["done"] == _union_size(backend, "populated")
    assert payload["current"] is None and payload["errors"] == []
    assert len(payload["results"]) == 2
    for result in payload["results"]:
        assert NEWS_SCAN_RESULT_KEYS <= set(result)
        assert result["decision"] in {"EXIT", "HOLD"}
        assert result["fail_safe"] is False
        assert result["event_id"] == f"D:{result['ticker']}:{as_of}:m"
    assert b"http://" not in response.body and b"https://" not in response.body
    # empty mode: no held names -> total 0, no results
    empty = FixtureBackend(mode="empty")
    payload = empty.dispatch("GET", "/api/news/scan/status").json()
    assert payload["running"] is False and payload["total"] == 0
    assert payload["results"] == []
    # fail-closed: every other news-scan path 404s/501s like the live app
    for method, path in (
        ("GET", "/api/news/scan"),
        ("POST", "/api/news/scan/status"),
        ("DELETE", "/api/news/scan"),
        ("GET", "/api/news/scan/extra"),
        ("POST", "/api/news/scan/extra", ),
    ):
        r = backend.dispatch(method, path, body=b"{}" if method == "POST" else b"")
        assert r.status in (404, 501), (method, path, r.status)
        assert r.json()["testing_only"] is True
