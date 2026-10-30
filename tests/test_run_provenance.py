"""Per-run DATA provenance summary: honest cache aggregation, offline only.

Covers the Phase 2 contract of ``app.run_provenance.summarize``:

* the membership comes from the run configuration's single members key
  (``universe_members``, either inside ``universe`` or at the top level, with
  the legacy ``members`` spelling accepted);
* only candles for the members + the FX pairs those members require are
  counted, and only fundamentals for the members; symbols outside the scope
  are never included;
* candles are bounded by ``timestamp <= asof`` and fundamentals by
  ``filed <= asof``'s UTC calendar day; ``ingested_at`` is an observation
  time and is NOT bounded;
* NULL source / adjustment basis / ingestion time is reported as
  ``"unknown"`` and counted — never assumed;
* missing / unresolved / empty membership produces an explicit unknown scope
  and an empty summary without touching the database;
* the result is deterministic and JSON-safe, never copies config secrets;
* a caller-owned session is never committed (and its uncommitted rows are
  what the summary reads).

Every DB is a temp-file SQLite built from the ORM metadata; no provider, LLM,
HTTP or production-database access happens. An autouse fixture blocks
``socket.socket.connect`` for AF_INET/AF_INET6 and counts every attempt.
"""

from __future__ import annotations

import hashlib
import json
import socket
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
from app import run_provenance
from app.db import Base, Candle, Fundamental

SOURCE_YF = "yfinance"
SOURCE_TD = "twelvedata"
BASIS = "split_dividend_adjusted"

# Every outbound INET connect attempt observed by the guard below. The suite
# asserts it stays empty; the final report states the exact count.
_INET_CONNECT_ATTEMPTS: list[tuple[int, object]] = []


@pytest.fixture(autouse=True)
def _block_inet_connect(monkeypatch):
    """Fail any AF_INET/AF_INET6 connect and record the attempt."""
    real_connect = socket.socket.connect
    start = len(_INET_CONNECT_ATTEMPTS)

    def guarded_connect(sock, address, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            _INET_CONNECT_ATTEMPTS.append((sock.family, address))
            raise AssertionError(f"outbound INET connect blocked: {address!r}")
        return real_connect(sock, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield
    attempts = _INET_CONNECT_ATTEMPTS[start:]
    assert attempts == [], f"network connect attempt(s) in this test: {attempts!r}"


@pytest.fixture
async def store(tmp_path):
    """A temp-file SQLite DB with the ORM schema and its own session factory."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'run-provenance.db'}",
        connect_args={"timeout": 5},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield engine, factory
    finally:
        await engine.dispose()


def _members_digest(members: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(members)).encode("utf-8")).hexdigest()


_AUTO = object()


def _engine_config(
    members,
    *,
    name: str = "sp500",
    members_hash=_AUTO,
    config_hash: str | None = None,
) -> dict:
    """The daily_core/monthly shape: membership inside the universe block."""
    universe = {"name": name}
    if members_hash is _AUTO:
        universe["members_hash"] = _members_digest(members) if members else None
    else:
        universe["members_hash"] = members_hash
    universe["universe_members"] = members
    config = {
        "engine": "daily_core",
        "universe": universe,
        "settings": {"sim_monthly_target_n": 20},
    }
    if config_hash is not None:
        config["hash"] = config_hash
    return config


def _sim_config(members, *, name: str = "watchlist") -> dict:
    """The sim shape: membership at the config's top level."""
    return {
        "engine": "sim",
        "universe": {
            "name": name,
            "source": "file",
            "members_hash": _members_digest(members) if members else None,
        },
        "universe_members": members,
        "settings": {"sim_strategy": "momentum"},
    }


def _candle(
    ticker: str,
    day: datetime,
    *,
    source: str | None = SOURCE_YF,
    basis: str | None = BASIS,
    ingested: datetime | None = None,
    close: float = 1.0,
) -> Candle:
    return Candle(
        ticker=ticker,
        timestamp=day,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        source=source,
        adjustment_basis=basis,
        ingested_at=ingested,
    )


def _fact(
    ticker: str,
    filed: str,
    *,
    source: str = "edgar",
    tag: str = "NetIncomeLoss",
) -> Fundamental:
    return Fundamental(
        ticker=ticker,
        tag=tag,
        start=None,
        end="2025-12-31",
        filed=filed,
        val=1.0,
        currency="USD",
        source=source,
    )


async def _seed(factory, *rows) -> None:
    async with factory() as s:
        s.add_all(rows)
        await s.commit()


def _utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Scope honesty: no members => no query, explicit unknown scope
# ---------------------------------------------------------------------------


class TestScopeHonesty:
    async def test_missing_members_never_scans_the_cache(self):
        """An absent membership returns an empty summary and does NOT query."""

        def forbidden_factory(*_a, **_k):
            raise AssertionError("summarize must not open a session without members")

        summary = await run_provenance.summarize(
            {"engine": "daily_core", "settings": {"x": 1}},
            asof=_utc(2026, 1, 10),
            session_factory=forbidden_factory,
        )
        assert summary["scope"]["known"] is False
        assert summary["scope"]["reason"] == "members_missing"
        assert summary["members_hash"] is None
        assert summary["candles"]["rows"] == 0
        assert summary["candles"]["by_source"] == []
        assert summary["fundamentals"]["rows"] == 0
        json.dumps(summary, allow_nan=False)

    async def test_unresolved_engine_universe_is_explicitly_unknown(self):
        """The engines' None membership is 'unresolved', not an empty scan."""
        config = {
            "engine": "monthly",
            "universe": {"name": "custom", "members_hash": None, "universe_members": None},
        }
        summary = await run_provenance.summarize(
            config,
            asof=_utc(2026, 1, 10),
            session_factory=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("no session for an unknown scope")
            ),
        )
        assert summary["scope"]["known"] is False
        assert summary["scope"]["reason"] == "members_unresolved"
        assert summary["scope"]["universe"] == "custom"
        assert summary["candles"]["rows"] == 0
        assert summary["fundamentals"]["rows"] == 0

    @pytest.mark.parametrize(
        "members, reason",
        [
            ([], "members_empty"),
            ("AAPL", "members_invalid"),
            (["AAPL", None], "members_invalid"),
            (["AAPL", "  "], "members_invalid"),
        ],
    )
    async def test_empty_or_malformed_membership_is_not_widened(self, members, reason):
        config = {"universe_members": members}
        summary = await run_provenance.summarize(
            config,
            asof=_utc(2026, 1, 10),
            session_factory=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("no session for an unknown scope")
            ),
        )
        assert summary["scope"]["known"] is False
        assert summary["scope"]["reason"] == reason
        assert summary["scope"]["member_count"] == 0
        assert summary["candles"]["rows"] == 0
        assert summary["fundamentals"]["rows"] == 0


# ---------------------------------------------------------------------------
# Config shapes, membership hash, FX pair derivation
# ---------------------------------------------------------------------------


class TestScoping:
    async def test_nested_engine_shape_uses_provided_members_hash(self, store):
        _engine, factory = store
        await _seed(factory, _candle("AAPL", _utc(2026, 1, 2)))
        config = _engine_config(
            ["AAPL", "VOD.L"],
            members_hash="config-supplied-hash",
            config_hash="cfg-hash",
        )
        summary = await run_provenance.summarize(
            config, asof=_utc(2026, 1, 10), session_factory=factory
        )
        assert summary["scope"]["known"] is True
        assert summary["scope"]["member_count"] == 2
        assert summary["members_hash"] == "config-supplied-hash"
        assert summary["scope"]["members_hash_source"] == "config"
        assert summary["config_hash"] == "cfg-hash"
        assert summary["scope"]["fx_pairs"] == ["GBPUSD=X"]

    async def test_sim_top_level_members_and_computed_hash(self, store):
        _engine, factory = store
        await _seed(factory, _candle("AAPL", _utc(2026, 1, 2)))
        config = _sim_config(["AAPL"])
        config["universe"]["members_hash"] = None  # force the computed fallback
        summary = await run_provenance.summarize(
            config, asof=_utc(2026, 1, 10), session_factory=factory
        )
        assert summary["scope"]["known"] is True
        assert summary["scope"]["universe"] == "watchlist"
        assert summary["members_hash"] == _members_digest(["AAPL"])
        assert summary["scope"]["members_hash_source"] == "computed"
        assert summary["scope"]["fx_pairs"] == []

    async def test_legacy_nested_members_key_is_accepted(self, store):
        _engine, factory = store
        await _seed(factory, _candle("MSFT", _utc(2026, 1, 2)))
        config = {
            "universe": {
                "name": "legacy",
                "members": ["MSFT"],
                "members_hash": _members_digest(["MSFT"]),
            }
        }
        summary = await run_provenance.summarize(
            config, asof=_utc(2026, 1, 10), session_factory=factory
        )
        assert summary["scope"]["known"] is True
        assert summary["scope"]["member_count"] == 1
        assert summary["candles"]["rows"] == 1

    async def test_flat_members_key_and_top_level_hash(self, store):
        _engine, factory = store
        await _seed(factory, _candle("AAPL", _utc(2026, 1, 2)))
        summary = await run_provenance.summarize(
            {"members": ["AAPL"], "members_hash": "flat-hash"},
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        assert summary["members_hash"] == "flat-hash"
        assert summary["scope"]["members_hash_source"] == "config"

    async def test_required_fx_pairs_are_explicit_and_sorted(self, store):
        _engine, factory = store
        config = _engine_config(["SAP.DE", "VOD.L", "AAPL", "RY.TO", "NESN.SW"])
        summary = await run_provenance.summarize(
            config, asof=_utc(2026, 1, 10), session_factory=factory
        )
        # EUR (SAP.DE) and GBP (VOD.L) pairs; SNB/CHF comes from NESN.SW.
        assert summary["scope"]["fx_pairs"] == ["EURUSD=X", "GBPUSD=X", "USDCHF=X"]
        # .TO has no explicit mapping: recorded, never assumed USD or EUR.
        assert summary["scope"]["unresolved_fx_members"] == ["RY.TO"]
        assert summary["scope"]["ticker_count"] == 8

    async def test_usd_native_members_require_no_eurusd_pair(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2)),
            _candle("BRK.B", _utc(2026, 1, 2)),
            _candle("EURUSD=X", _utc(2026, 1, 2), source=SOURCE_TD),
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL", "BRK.B"]),
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        assert summary["scope"]["fx_pairs"] == []
        assert summary["scope"]["ticker_count"] == 2
        # The unused EURUSD pair in the cache is outside the scope.
        assert summary["candles"]["rows"] == 2

    async def test_no_config_settings_or_secrets_are_copied(self, store):
        _engine, factory = store
        await _seed(factory, _candle("AAPL", _utc(2026, 1, 2)))
        config = _engine_config(["AAPL"])
        config["settings"] = {
            "api_key": "SECRET-TOKEN",
            "bank_pin": "PIN-VALUE-987654",
        }
        summary = await run_provenance.summarize(
            config, asof=_utc(2026, 1, 10), session_factory=factory
        )
        dumped = json.dumps(summary, sort_keys=True, allow_nan=False)
        assert "SECRET-TOKEN" not in dumped
        assert "PIN-VALUE-987654" not in dumped
        assert "settings" not in summary
        assert "bank_pin" not in dumped


# ---------------------------------------------------------------------------
# Candle aggregation: grouped source/basis, unknown provenance, fetch times
# ---------------------------------------------------------------------------


class TestCandleAggregation:
    async def test_null_source_basis_and_ingest_are_unknown_not_assumed(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2), source=None, basis=None, ingested=None),
            _candle("AAPL", _utc(2026, 1, 3), source=None, basis=BASIS, ingested=None),
            _candle(
                "AAPL",
                _utc(2026, 1, 4),
                source=None,
                basis=BASIS,
                ingested=_utc(2026, 1, 4, 12, 30),
            ),
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]),
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        candles = summary["candles"]
        assert candles["rows"] == 3
        assert candles["by_source"] == [{"source": "unknown", "count": 3}]
        assert candles["source_known"] == 0
        assert candles["source_unknown"] == 3
        assert candles["by_adjustment_basis"] == [
            {"basis": BASIS, "count": 2},
            {"basis": "unknown", "count": 1},
        ]
        assert candles["basis_known"] == 2
        assert candles["basis_unknown"] == 1
        assert candles["by_source_basis"] == [
            {"source": "unknown", "basis": BASIS, "count": 2},
            {"source": "unknown", "basis": "unknown", "count": 1},
        ]
        assert candles["ingested_at_known"] == 1
        assert candles["ingested_at_unknown"] == 2
        assert candles["min_ingested_at"] == "2026-01-04T12:30:00+00:00"
        assert candles["max_ingested_at"] == "2026-01-04T12:30:00+00:00"
        assert candles["min_timestamp"] == "2026-01-02T00:00:00+00:00"
        assert candles["max_timestamp"] == "2026-01-04T00:00:00+00:00"
        json.dumps(summary, sort_keys=True, allow_nan=False)

    async def test_mixed_known_and_unknown_source_counts(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2), source=None, basis=None),
            _candle("AAPL", _utc(2026, 1, 3), source=SOURCE_YF, basis=BASIS),
        )
        candles = (
            await run_provenance.summarize(
                _engine_config(["AAPL"]),
                asof=_utc(2026, 1, 10),
                session_factory=factory,
            )
        )["candles"]
        assert candles["rows"] == 2
        assert candles["by_source"] == [
            {"source": "unknown", "count": 1},
            {"source": SOURCE_YF, "count": 1},
        ]
        assert candles["source_known"] == 1
        assert candles["source_unknown"] == 1

    async def test_two_known_sources_are_reported_separately(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2), source=SOURCE_YF),
            _candle("AAPL", _utc(2026, 1, 3), source=SOURCE_YF),
            _candle("AAPL", _utc(2026, 1, 4), source=SOURCE_TD, basis="raw"),
            _candle("AAPL", _utc(2026, 1, 5), source=SOURCE_TD, basis="raw"),
            _candle("AAPL", _utc(2026, 1, 6), source=SOURCE_TD, basis="raw"),
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]),
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        candles = summary["candles"]
        assert candles["by_source"] == [
            {"source": SOURCE_TD, "count": 3},
            {"source": SOURCE_YF, "count": 2},
        ]
        assert candles["source_known"] == 5
        assert candles["source_unknown"] == 0
        assert candles["by_adjustment_basis"] == [
            {"basis": "raw", "count": 3},
            {"basis": BASIS, "count": 2},
        ]

    async def test_ingested_at_is_not_bounded_by_asof(self, store):
        """ingested_at is when we observed the row, not the bar's event time."""
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2), ingested=_utc(2026, 12, 31, 23, 59)),
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]),
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        assert summary["candles"]["rows"] == 1
        assert summary["candles"]["max_ingested_at"] == "2026-12-31T23:59:00+00:00"

    async def test_known_scope_with_empty_cache_is_zero_not_error(self, store):
        _engine, factory = store
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]),
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        assert summary["scope"]["known"] is True
        assert summary["candles"]["rows"] == 0
        assert summary["candles"]["by_source"] == []
        assert summary["candles"]["min_timestamp"] is None
        assert summary["candles"]["max_timestamp"] is None
        assert summary["candles"]["ingested_at_unknown"] == 0
        assert summary["fundamentals"]["rows"] == 0
        assert summary["fundamentals"]["max_filed"] is None

    async def test_fundamental_rows_group_by_source_and_report_filed_span(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _fact("AAPL", "2026-01-01", source="edgar"),
            _fact("AAPL", "2026-01-05", source="edgar", tag="StockholdersEquity"),
            _fact("AAPL", "2026-01-03", source="yfinance"),
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]),
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        fundamentals = summary["fundamentals"]
        assert fundamentals["rows"] == 3
        assert fundamentals["by_source"] == [
            {"source": "edgar", "count": 2},
            {"source": "yfinance", "count": 1},
        ]
        assert fundamentals["source_known"] == 3
        assert fundamentals["source_unknown"] == 0
        assert fundamentals["min_filed"] == "2026-01-01"
        assert fundamentals["max_filed"] == "2026-01-05"

    async def test_asof_defaults_to_now_utc(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2020, 1, 2)),
            _candle("AAPL", _utc(2999, 1, 2)),  # far future: never seen "now"
        )
        before = datetime.now(UTC)
        summary = await run_provenance.summarize(_engine_config(["AAPL"]), session_factory=factory)
        after = datetime.now(UTC)
        parsed = datetime.fromisoformat(summary["asof"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert before <= parsed <= after
        assert summary["candles"]["rows"] == 1
        assert summary["candles"]["max_timestamp"] == "2020-01-02T00:00:00+00:00"

    async def test_output_is_deterministic(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2), source=None, basis=None),
            _fact("AAPL", "2026-01-01"),
            _candle("VOD.L", _utc(2026, 1, 2), source=SOURCE_TD),
            _candle("GBPUSD=X", _utc(2026, 1, 2), source=SOURCE_TD),
        )
        config = _engine_config(["VOD.L", "AAPL"])
        first = await run_provenance.summarize(
            config, asof=_utc(2026, 1, 10), session_factory=factory
        )
        second = await run_provenance.summarize(
            config, asof=_utc(2026, 1, 10), session_factory=factory
        )
        assert first == second
        json.dumps(first, sort_keys=True, allow_nan=False)


# ---------------------------------------------------------------------------
# Time bounds and universe containment
# ---------------------------------------------------------------------------


class TestTimeBoundsAndContainment:
    async def test_future_candles_and_facts_are_excluded(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 10, 11, 59, 59, 999999)),
            _candle("AAPL", _utc(2026, 1, 10, 12, 0, 0)),  # == asof: seen
            _candle("AAPL", _utc(2026, 1, 10, 12, 0, 0, 1)),  # after: excluded
            _candle("AAPL", _utc(2026, 1, 11)),
            _fact("AAPL", "2025-12-31"),
            _fact("AAPL", "2026-01-10"),  # == asof's day: seen
            _fact("AAPL", "2026-01-11"),  # next day: excluded
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]),
            asof=_utc(2026, 1, 10, 12),
            session_factory=factory,
        )
        candles = summary["candles"]
        assert candles["rows"] == 2
        assert candles["min_timestamp"] == "2026-01-10T11:59:59.999999+00:00"
        assert candles["max_timestamp"] == "2026-01-10T12:00:00+00:00"
        fundamentals = summary["fundamentals"]
        assert fundamentals["rows"] == 2
        assert fundamentals["min_filed"] == "2025-12-31"
        assert fundamentals["max_filed"] == "2026-01-10"

    async def test_offset_aware_asof_is_converted_to_utc(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 1, 22, 59)),
            _candle("AAPL", _utc(2026, 1, 1, 23, 0, 0)),
            _candle("AAPL", _utc(2026, 1, 1, 23, 0, 0, 1)),
            _fact("AAPL", "2026-01-01"),
            _fact("AAPL", "2026-01-02"),
        )
        asof = datetime(
            2026, 1, 2, 1, 0, tzinfo=timezone(timedelta(hours=2))
        )  # == 2026-01-01T23:00Z
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]), asof=asof, session_factory=factory
        )
        assert summary["asof"] == "2026-01-01T23:00:00+00:00"
        assert summary["candles"]["rows"] == 2
        assert summary["candles"]["max_timestamp"] == "2026-01-01T23:00:00+00:00"
        # Fundamental filed dates use the UTC calendar day (Jan 1), not +02:00.
        assert summary["fundamentals"]["rows"] == 1
        assert summary["fundamentals"]["max_filed"] == "2026-01-01"

    async def test_naive_asof_is_read_as_utc(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2)),
            _candle("AAPL", _utc(2026, 1, 2, 0, 0, 0, 1)),
            _fact("AAPL", "2026-01-02"),
            _fact("AAPL", "2026-01-03"),
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL"]),
            asof=datetime(2026, 1, 2),
            session_factory=factory,
        )
        assert summary["asof"] == "2026-01-02T00:00:00+00:00"
        assert summary["candles"]["rows"] == 1
        assert summary["fundamentals"]["rows"] == 1

    async def test_symbols_outside_the_scope_are_excluded(self, store):
        _engine, factory = store
        await _seed(
            factory,
            _candle("AAPL", _utc(2026, 1, 2)),
            _candle("MSFT", _utc(2026, 1, 2)),  # not in the universe
            _candle("EURUSD=X", _utc(2026, 1, 2)),  # AAPL is USD-native
            _candle("GBPUSD=X", _utc(2026, 1, 2)),  # no .L member
            _candle("VOD.L", _utc(2026, 1, 2)),  # member
            _fact("AAPL", "2026-01-02"),
            _fact("MSFT", "2026-01-02"),  # outside the universe
            _fact("VOD.L", "2026-01-02"),  # member
        )
        summary = await run_provenance.summarize(
            _engine_config(["AAPL", "VOD.L"]),
            asof=_utc(2026, 1, 10),
            session_factory=factory,
        )
        assert summary["scope"]["fx_pairs"] == ["GBPUSD=X"]
        # AAPL + VOD.L + GBPUSD=X candles only (EURUSD comes with a later test).
        assert summary["candles"]["rows"] == 3
        assert summary["fundamentals"]["rows"] == 2
        # The unused EURUSD pair above is still outside VOD.L's scope.
        assert summary["scope"]["ticker_count"] == 3


# ---------------------------------------------------------------------------
# Transactions: caller-owned sessions are never committed
# ---------------------------------------------------------------------------


class _CommitCounter:
    """Count SQLAlchemy connection commits on an engine."""

    def __init__(self, engine):
        self.engine = engine
        self.count = 0
        self._listener = self._on_commit
        event.listen(engine.sync_engine, "commit", self._listener)

    def _on_commit(self, _conn):
        self.count += 1

    def close(self):
        event.remove(self.engine.sync_engine, "commit", self._listener)


class TestTransactionOwnership:
    async def test_caller_session_is_used_and_never_committed(self, store):
        engine, factory = store
        config = _engine_config(["AAPL"])
        counter = _CommitCounter(engine)
        try:
            async with factory() as session:
                session.add(_candle("AAPL", _utc(2026, 1, 2)))
                await session.flush()
                summary = await run_provenance.summarize(
                    config,
                    asof=_utc(2026, 1, 10),
                    session=session,
                    session_factory=lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("the caller-owned session must win")
                    ),
                )
                # The uncommitted row is visible: the caller's transaction was
                # actually used, not a fresh connection.
                assert summary["candles"]["rows"] == 1
                assert session.in_transaction() is True
                assert counter.count == 0
                await session.rollback()
        finally:
            counter.close()
        # Nothing was committed: the row is gone after the caller's rollback.
        async with factory() as check:
            assert await check.scalar(select(func.count()).select_from(Candle)) == 0

    async def test_owned_session_path_is_read_only(self, store, monkeypatch):
        engine, factory = store
        await _seed(factory, _candle("AAPL", _utc(2026, 1, 2)))
        monkeypatch.setattr(db_mod, "Session", factory)
        counter = _CommitCounter(engine)
        try:
            summary = await run_provenance.summarize(
                _engine_config(["AAPL"]), asof=_utc(2026, 1, 10)
            )
        finally:
            counter.close()
        assert summary["candles"]["rows"] == 1
        assert counter.count == 0


class TestApiContract:
    async def test_rejects_non_dict_config_and_non_datetime_asof(self):
        with pytest.raises(TypeError, match="config must be a dict"):
            await run_provenance.summarize([])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="asof must be a datetime"):
            await run_provenance.summarize(
                {"members": []},
                asof="2026-01-01",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Network guard evidence
# ---------------------------------------------------------------------------


async def test_network_guard_counter_is_zero(store):
    """No test may have attempted an AF_INET/AF_INET6 connect."""

    def forbidden_factory(*_a, **_k):
        raise AssertionError("this probe needs no DB")

    await run_provenance.summarize(
        {"members": []}, asof=_utc(2026, 1, 10), session_factory=forbidden_factory
    )
    assert _INET_CONNECT_ATTEMPTS == []
