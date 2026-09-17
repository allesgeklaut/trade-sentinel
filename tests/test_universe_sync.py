"""Tests for app.universe_sync: holdings-CSV parsing, atomic write, and change
detection. Network is monkeypatched and the generated-universe dir is a tmp dir
so neither a live fetch nor a developer's /data can affect the run."""

from __future__ import annotations

import csv
import io

import pytest

from app import screener, universe_sync


def _csv(n_symbols: int = 0, as_of: str = "Sep 15, 2026",
         extra_equities: list[str] | None = None,
         include_non_equity: bool = True) -> bytes:
    """Build an iShares-style holdings CSV (preamble + header + rows)."""
    rows: list[list[str]] = [
        ["iShares Core S&P 500 ETF"],
        ["Fund Holdings as of", as_of],
        ["Stock", "-"],
        [],
        ["Ticker", "Name", "Sector", "Asset Class", "Weight (%)"],
    ]
    if include_non_equity:
        rows.append(["USD", "US DOLLAR", "Cash", "Cash", "0.10"])
        rows.append(["BZX", "MONEY MARKET", "Money Market", "Money Market", "0.05"])
        rows.append(["ES", "S&P FUTURE", "Futures", "Futures", "0.01"])
    equities = list(extra_equities or []) + [f"T{i:03d}" for i in range(n_symbols)]
    for t in equities:
        rows.append([t, t + " INC", "Information Technology", "Equity", "0.10"])
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def universes_tmp(tmp_path, monkeypatch):
    """Isolated repo + extra universe dirs; returns the extra (generated) dir."""
    repo = tmp_path / "repo"
    repo.mkdir()
    extra = tmp_path / "extra"
    monkeypatch.setattr(screener, "_UNIVERSES_DIR", repo)
    monkeypatch.setattr(screener, "_EXTRA_UNIVERSES_DIR", extra)
    return extra


def _serve(payload: bytes, monkeypatch) -> None:
    monkeypatch.setattr(universe_sync, "_http_get", lambda url, **kw: payload)


class TestParse:
    def test_filters_non_equity_and_normalizes_class_symbols(self):
        payload = _csv(extra_equities=["AAPL", "BRK B", "BF B"])
        symbols, as_of = universe_sync.parse_constituents(payload)
        assert symbols == ["AAPL", "BF-B", "BRK-B"]  # cash/futures rows dropped
        assert as_of == "Sep 15, 2026"

    def test_missing_header_raises(self):
        with pytest.raises(ValueError, match="Ticker"):
            universe_sync.parse_constituents(b"not,a,holdings,file\n1,2,3,4\n")

    def test_no_equities_raises(self):
        with pytest.raises(ValueError, match="no equity"):
            universe_sync.parse_constituents(_csv(0))

    def test_tolerates_missing_as_of(self):
        payload = _csv(1, as_of="")
        symbols, as_of = universe_sync.parse_constituents(payload)
        assert symbols == ["T000"]
        assert as_of is None


class TestSync:
    async def test_writes_file_and_reports_added(self, universes_tmp, monkeypatch):
        _serve(_csv(450, extra_equities=["AAPL", "BRK B"]), monkeypatch)
        r = await universe_sync.sync_sp500()

        assert r["count"] == 452
        assert r["changed"] is True
        assert "AAPL" in r["added"] and "BRK-B" in r["added"]
        assert r["removed"] == []

        path = universes_tmp / "sp500.txt"
        assert path.exists()
        body = path.read_text()
        assert "AUTO-GENERATED" in body and r["url"] in body
        assert not list(universes_tmp.glob("*.tmp"))  # atomic replace cleans up
        assert screener.tickers("sp500") == sorted(
            ["AAPL", "BRK-B"] + [f"T{i:03d}" for i in range(450)])

    async def test_second_sync_is_idempotent(self, universes_tmp, monkeypatch):
        _serve(_csv(450, extra_equities=["AAPL"]), monkeypatch)
        await universe_sync.sync_sp500()
        r2 = await universe_sync.sync_sp500()
        assert r2["changed"] is False
        assert r2["added"] == [] and r2["removed"] == []

    async def test_detects_removed_symbols(self, universes_tmp, monkeypatch):
        _serve(_csv(450, extra_equities=["AAPL", "MSFT"]), monkeypatch)
        await universe_sync.sync_sp500()
        _serve(_csv(451, extra_equities=["AAPL"]), monkeypatch)
        r = await universe_sync.sync_sp500()
        assert r["changed"] is True
        assert r["removed"] == ["MSFT"]

    async def test_refuses_suspiciously_small_list(self, universes_tmp, monkeypatch):
        _serve(_csv(450, extra_equities=["AAPL"]), monkeypatch)
        await universe_sync.sync_sp500()
        _serve(_csv(10), monkeypatch)
        with pytest.raises(ValueError, match="refusing to write"):
            await universe_sync.sync_sp500()
        assert len(screener.tickers("sp500")) == 451  # previous list intact

    async def test_fetch_failure_preserves_previous_file(self, universes_tmp, monkeypatch):
        _serve(_csv(450), monkeypatch)
        await universe_sync.sync_sp500()
        before = (universes_tmp / "sp500.txt").read_text()

        def _boom(url, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(universe_sync, "_http_get", _boom)
        with pytest.raises(RuntimeError):
            await universe_sync.sync_sp500()
        assert (universes_tmp / "sp500.txt").read_text() == before
