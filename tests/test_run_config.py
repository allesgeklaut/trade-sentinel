"""R1: run-configuration registry, run epochs and their schema migration.

The audit items under test (purely additive — no engine reads this yet):

* every run is identified by an IMMUTABLE configuration: (engine, mode,
  canonical JSON) deduped by content hash, so counting rows reports the number
  of DISTINCT CONFIGURATIONS registered, not repeated execution runs;
* historical/forward provenance: ``mode`` (paper | backfill), ``config_id``
  and ``epoch`` are stamped on every history table. ``mode``/``config_id``
  are NULL on rows written before stamping existed (unknown, never invented);
  ``epoch`` is always a real value — legacy rows migrate to 1, the pre-reset
  book — so the epoch-aware UNIQUE keys stay total (no duplicate-month hole);
* a reset appends a ``RunEpoch`` boundary instead of rewriting history, so a
  closed epoch's month/rebalance-month rows SURVIVE while the fresh epoch logs
  the same month again — that is why the old single-column UNIQUE keys of the
  allowance tables (and ``MonthlyRebalance.rebal_month``) are reshaped to
  ``(month, epoch)`` / ``(rebal_month, epoch)``;
* helpers accept a caller-owned session and never commit it, so a reset can
  write its boundary and its state in ONE transaction;
* ``SavingsEvent.settlement_of`` (added for the disposal-pair fix) is
  nullable, never backfilled, and UNIQUE so a disposal cannot be settled
  twice.

Everything runs OFFLINE against temp-file SQLite databases: no provider call,
no network, and never the production database.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
from app import run_config
from app.db import (
    DailyCoreAccount,
    MonthlyAccount,
    MonthlyRebalance,
    RunConfig,
    RunEpoch,
    RunExecution,
    SimAccount,
    SimAllowance,
    SimTrade,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Tables rebuilt for epoch-safe history: their PRE-R1 shape (old single-column
# UNIQUE key, no stamps). ``savings_events`` is rebuilt too so the settlement
# migration starts from a genuinely pre-column table.
_LEGACY_DDL = {
    "sim_allowances": """
        CREATE TABLE sim_allowances (
            id INTEGER NOT NULL PRIMARY KEY,
            amount FLOAT NOT NULL,
            month VARCHAR(7) NOT NULL UNIQUE,
            created_at DATETIME NOT NULL
        )
    """,
    "monthly_allowances": """
        CREATE TABLE monthly_allowances (
            id INTEGER NOT NULL PRIMARY KEY,
            amount FLOAT NOT NULL,
            month VARCHAR(7) NOT NULL UNIQUE,
            created_at DATETIME NOT NULL
        )
    """,
    "daily_core_allowances": """
        CREATE TABLE daily_core_allowances (
            id INTEGER NOT NULL PRIMARY KEY,
            amount FLOAT NOT NULL,
            month VARCHAR(7) NOT NULL UNIQUE,
            created_at DATETIME NOT NULL
        )
    """,
    "monthly_rebalances": """
        CREATE TABLE monthly_rebalances (
            id INTEGER NOT NULL PRIMARY KEY,
            rebal_month VARCHAR(7) NOT NULL,
            rebal_date DATETIME NOT NULL,
            held_before TEXT NOT NULL,
            picked TEXT NOT NULL,
            n_new INTEGER NOT NULL,
            snapshot TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            UNIQUE (rebal_month)
        )
    """,
    "savings_events": """
        CREATE TABLE savings_events (
            id INTEGER NOT NULL PRIMARY KEY,
            kind VARCHAR(16) NOT NULL,
            amount FLOAT NOT NULL,
            ticker VARCHAR(32),
            note TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            occurred_on VARCHAR(10),
            shares FLOAT,
            price FLOAT,
            cost_basis FLOAT,
            flow_class VARCHAR(16)
        )
    """,
}

# Current-state account rows as they looked BEFORE the opening-seed column:
# nullable markers, no opening_seed (the seed was never recorded then).
_LEGACY_ACCOUNT_DDL = {
    "sim_account": """
        CREATE TABLE sim_account (
            id INTEGER NOT NULL PRIMARY KEY,
            cash FLOAT NOT NULL,
            last_allowance_month VARCHAR(7),
            last_review_week VARCHAR(8),
            created_at DATETIME NOT NULL
        )
    """,
    "monthly_account": """
        CREATE TABLE monthly_account (
            id INTEGER NOT NULL PRIMARY KEY,
            cash FLOAT NOT NULL,
            last_allowance_month VARCHAR(7),
            last_rebalance_month VARCHAR(7),
            created_at DATETIME NOT NULL
        )
    """,
    "daily_core_account": """
        CREATE TABLE daily_core_account (
            id INTEGER NOT NULL PRIMARY KEY,
            cash FLOAT NOT NULL,
            last_allowance_month VARCHAR(7),
            created_at DATETIME NOT NULL
        )
    """,
}

# History tables that only gain the three nullable stamps (no rebuild).
_STAMPED_NOT_REBUILT = (
    "sim_trades", "sim_snapshots",
    "monthly_trades", "monthly_snapshots",
    "daily_core_trades", "daily_core_snapshots", "daily_core_ranking",
)

_STAMP_COLUMNS = ("mode", "config_id", "epoch")

# The 11 history tables the brief lists (current-state tables stay unstamped).
_HISTORY_TABLES = (
    "sim_trades", "sim_allowances", "sim_snapshots",
    "monthly_trades", "monthly_allowances", "monthly_snapshots",
    "monthly_rebalances",
    "daily_core_trades", "daily_core_allowances", "daily_core_snapshots",
    "daily_core_ranking",
)

_SEEDED_ALLOWANCES = [
    (1, 1000.0, "2026-01", "2026-01-02 10:00:00.000000"),
    (9, 1100.0, "2026-02", "2026-02-02 10:00:00.000000"),
]
_SEEDED_REBALANCE = (
    7, "2026-01", "2026-01-30 00:00:00.000000", "AAA,BBB", "CCC,DDD", 2,
    '{"frame": 1}', "2026-01-30 12:00:00.000000",
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_engine(tmp_path, name: str):
    return create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / name}",
        connect_args={"timeout": 5},
    )


@pytest.fixture
async def temp_db(tmp_path):
    """A temp-file DB already converged to the current schema."""
    engine = _make_engine(tmp_path, "fresh.db")
    try:
        await db_mod.init_db(engine)
        yield engine, async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.fixture
async def blank_db(tmp_path):
    """A temp-file ENGINE with no schema at all (migration tests build it)."""
    engine = _make_engine(tmp_path, "legacy.db")
    try:
        yield engine, async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _columns(engine, table: str) -> list[tuple]:
    """(name, type, notnull, default, pk) per column, in declaration order."""
    async with engine.connect() as conn:
        rows = (await conn.execute(text(f'PRAGMA table_info("{table}")'))).all()
    return [(row[1], row[2], row[3], row[4], row[5]) for row in rows]


async def _unique_indexes(engine, table: str) -> list[tuple[str, ...]]:
    async with engine.connect() as conn:
        listed = (await conn.execute(text(f'PRAGMA index_list("{table}")'))).all()
        out = []
        for row in listed:
            if not row[2]:
                continue
            cols = tuple(
                r[2]
                for r in (await conn.execute(text(f"PRAGMA index_info('{row[1]}')"))).all()
            )
            out.append(cols)
        return out


async def _scalar(engine, sql: str):
    async with engine.connect() as conn:
        return (await conn.execute(text(sql))).scalar_one()


async def _index_names(engine, table: str) -> set[str]:
    async with engine.connect() as conn:
        rows = (await conn.execute(text(f'PRAGMA index_list("{table}")'))).all()
    return {row[1] for row in rows}


def _legacy_unique_key(table: str) -> tuple[str, ...]:
    return ("rebal_month",) if table == "monthly_rebalances" else ("month",)


async def _build_legacy_db(engine) -> None:
    """Build the PRE-R1 schema and seed it (offline, raw SQL, no ORM writes)."""
    async with engine.begin() as conn:
        await conn.run_sync(db_mod.Base.metadata.create_all)
        # The four rebuilt tables: recreate them in their historical shape
        # (``unique=True`` / ``UniqueConstraint(old key)``, no stamp columns).
        for table, ddl in _LEGACY_DDL.items():
            await conn.execute(text(f'DROP TABLE "{table}"'))
            await conn.execute(text(ddl))
        # The remaining history tables never had the stamps: drop the columns
        # create_all just made (SQLite 3.35+), then add them back via init_db.
        for table in _STAMPED_NOT_REBUILT:
            for col in _STAMP_COLUMNS:
                await conn.execute(text(f'ALTER TABLE "{table}" DROP COLUMN {col}'))
        # The registry did not exist before R1.
        await conn.execute(text("DROP TABLE run_configs"))
        await conn.execute(text("DROP TABLE run_epochs"))
        # Current-state accounts predate the opening-seed column (and are not
        # history: they must gain the column WITHOUT any rebuild).
        for table, ddl in _LEGACY_ACCOUNT_DDL.items():
            await conn.execute(text(f'DROP TABLE "{table}"'))
            await conn.execute(text(ddl))

        await conn.execute(text(
            "INSERT INTO sim_allowances (id, amount, month, created_at) VALUES "
            "(1, 1000.0, '2026-01', '2026-01-02 10:00:00.000000'), "
            "(9, 1100.0, '2026-02', '2026-02-02 10:00:00.000000')"
        ))
        await conn.execute(text(
            "INSERT INTO monthly_allowances (id, amount, month, created_at) VALUES "
            "(4, 250.0, '2026-03', '2026-03-02 10:00:00.000000')"
        ))
        await conn.execute(text(
            "INSERT INTO daily_core_allowances (id, amount, month, created_at) VALUES "
            "(2, 250.0, '2026-04', '2026-04-02 10:00:00.000000')"
        ))
        await conn.execute(
            text(
                "INSERT INTO monthly_rebalances (id, rebal_month, rebal_date, "
                "held_before, picked, n_new, snapshot, created_at) VALUES "
                "(:id, :month, :date, :held, :picked, :n_new, :snapshot, :created)"
            ),
            dict(zip(
                ("id", "month", "date", "held", "picked", "n_new", "snapshot", "created"),
                _SEEDED_REBALANCE,
                strict=True,
            )),
        )
        await conn.execute(text(
            "INSERT INTO sim_trades (id, ticker, side, shares, price, cash_after, "
            "reason, created_at) VALUES (3, 'AAA', 'BUY', 1.5, 10.0, 985.0, 'seed', "
            "'2026-01-02 10:00:00.000000')"
        ))
        await conn.execute(text(
            "INSERT INTO monthly_snapshots (id, created_at, cash, positions_value, "
            "total_equity, allowance_total) VALUES "
            "(5, '2026-01-03 10:00:00.000000', 10.0, 20.0, 30.0, 0.0)"
        ))
        await conn.execute(text(
            "INSERT INTO daily_core_ranking (id, ranking_date, band, picks, updated_at) "
            "VALUES (1, '2026-01-04 00:00:00.000000', 'AAA', 'AAA', "
            "'2026-01-04 10:00:00.000000')"
        ))
        await conn.execute(text(
            "INSERT INTO savings_events (id, kind, amount, note, created_at) VALUES "
            "(1, 'SELL', 0, 'sold', '2026-01-02 10:00:00.000000'), "
            "(2, 'SETTLEMENT', 100, 'proceeds', '2026-01-03 10:00:00.000000')"
        ))


@pytest.fixture
async def migrated_db(blank_db):
    """A legacy DB already migrated by init_db (run twice: it must no-op)."""
    engine, factory = blank_db
    await _build_legacy_db(engine)
    await db_mod.init_db(engine)
    await db_mod.init_db(engine)
    return engine, factory


# ---------------------------------------------------------------------------
# canonical / config_hash
# ---------------------------------------------------------------------------


def test_canonical_is_deterministic_and_key_order_insensitive():
    a = {"momentum": 12, "top": 10, "nested": {"b": [3, 1], "a": None}}
    b = {"nested": {"a": None, "b": [3, 1]}, "top": 10, "momentum": 12}
    assert run_config.canonical(a) == run_config.canonical(b)
    assert run_config.canonical(a) == (
        '{"momentum":12,"nested":{"a":null,"b":[3,1]},"top":10}'
    )
    assert run_config.canonical({}) == "{}"


def test_config_hash_depends_on_content_only():
    base = {"top": 10, "band": 20}
    reordered = {"band": 20, "top": 10}
    assert run_config.config_hash(base) == run_config.config_hash(reordered)
    assert len(run_config.config_hash(base)) == 64
    assert run_config.config_hash(base) != run_config.config_hash({"top": 11, "band": 20})
    # the hash is content-only on purpose: engine/mode are separate columns
    assert run_config.config_hash({"a": "ü"}) == run_config.config_hash({"a": "ü"})


def test_canonical_rejects_non_finite_and_non_dict_values():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            run_config.canonical({"threshold": bad})
    with pytest.raises(ValueError):
        run_config.canonical({"nested": {"values": [1.0, float("nan")]}})
    with pytest.raises(TypeError):
        run_config.canonical([("not", "a dict")])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_or_create
# ---------------------------------------------------------------------------


async def test_get_or_create_dedupes_identical_configs(temp_db):
    engine, factory = temp_db
    config = {"top": 10, "band": 20}
    first = await run_config.get_or_create("sim", "paper", config, session_factory=factory)
    async with factory() as s:
        created_at = await s.scalar(select(RunConfig.created_at).where(RunConfig.id == first))
        content = await s.scalar(select(RunConfig.content).where(RunConfig.id == first))

    second = await run_config.get_or_create(
        "sim", "paper", {"band": 20, "top": 10}, session_factory=factory
    )

    assert second == first
    assert content == '{"band":20,"top":10}'
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_configs") == 1
    async with factory() as s:
        again = await s.scalar(select(RunConfig.created_at).where(RunConfig.id == first))
    assert again == created_at  # append-only: the FIRST sighting survives


async def test_get_or_create_separates_content_mode_and_engine(temp_db):
    _, factory = temp_db
    config = {"top": 10}
    ids = {
        "sim/paper": await run_config.get_or_create(
            "sim", "paper", config, session_factory=factory),
        "sim/backfill": await run_config.get_or_create(
            "sim", "backfill", config, session_factory=factory),
        "monthly/paper": await run_config.get_or_create(
            "monthly", "paper", config, session_factory=factory),
        "sim/paper/other": await run_config.get_or_create(
            "sim", "paper", {"top": 11}, session_factory=factory),
    }
    assert len(set(ids.values())) == 4


async def test_get_or_create_rejects_unknown_engine_and_mode(temp_db):
    _, factory = temp_db
    with pytest.raises(ValueError, match="unknown engine"):
        await run_config.get_or_create("crypto", "paper", {}, session_factory=factory)
    with pytest.raises(ValueError, match="unknown mode"):
        await run_config.get_or_create("sim", "live", {}, session_factory=factory)
    assert await run_config.current_epoch("sim", session_factory=factory) == 1


async def test_get_or_create_is_race_safe(temp_db):
    engine, factory = temp_db
    config = {"top": 10, "workers": 12}
    ids = await asyncio.gather(*[
        run_config.get_or_create("daily_core", "paper", config, session_factory=factory)
        for _ in range(12)
    ])
    assert len(set(ids)) == 1
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_configs") == 1


async def test_get_or_create_does_not_commit_a_caller_owned_session(temp_db):
    engine, factory = temp_db
    async with factory() as s:
        row_id = await run_config.get_or_create(
            "sim", "paper", {"x": 1}, session=s
        )
        assert row_id is not None
        await s.rollback()
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_configs") == 0


async def test_duplicate_config_row_is_rejected_by_the_unique_constraint(temp_db):
    _, factory = temp_db
    async with factory() as s:
        s.add(RunConfig(engine="sim", mode="paper", config_hash="h" * 64, content="{}"))
        await s.commit()
    async with factory() as s:
        s.add(RunConfig(engine="sim", mode="paper", config_hash="h" * 64, content="{}"))
        with pytest.raises(IntegrityError):
            await s.commit()


# ---------------------------------------------------------------------------
# record_boundary / current_epoch
# ---------------------------------------------------------------------------


async def test_current_epoch_starts_at_one_and_boundaries_bump_it(temp_db):
    _, factory = temp_db
    assert await run_config.current_epoch("sim", session_factory=factory) == 1
    assert await run_config.record_boundary("sim", "first reset", session_factory=factory) == 2
    assert await run_config.current_epoch("sim", session_factory=factory) == 2
    assert await run_config.record_boundary("sim", session_factory=factory) == 3
    assert await run_config.record_boundary("sim", "third", session_factory=factory) == 4
    async with factory() as s:
        notes = (await s.scalars(
            select(RunEpoch.note).where(RunEpoch.engine == "sim").order_by(RunEpoch.epoch)
        )).all()
    assert list(notes) == ["first reset", "", "third"]


async def test_record_boundary_is_per_engine(temp_db):
    _, factory = temp_db
    assert await run_config.record_boundary("sim", session_factory=factory) == 2
    assert await run_config.current_epoch("monthly", session_factory=factory) == 1
    assert await run_config.record_boundary("monthly", session_factory=factory) == 2
    assert await run_config.record_boundary("sim", session_factory=factory) == 3
    assert await run_config.current_epoch("daily_core", session_factory=factory) == 1


async def test_record_boundary_is_race_safe(temp_db):
    engine, factory = temp_db
    epochs = await asyncio.gather(*[
        run_config.record_boundary("monthly", f"reset {i}", session_factory=factory)
        for i in range(8)
    ])
    assert sorted(epochs) == list(range(2, 10))
    assert await _scalar(engine, "SELECT COUNT(DISTINCT epoch) FROM run_epochs") == 8
    assert await run_config.current_epoch("monthly", session_factory=factory) == 9


async def test_record_boundary_rejects_a_duplicate_epoch(temp_db):
    _, factory = temp_db
    await run_config.record_boundary("sim", session_factory=factory)
    async with factory() as s:
        s.add(RunEpoch(engine="sim", epoch=2, note="duplicate"))
        with pytest.raises(IntegrityError):
            await s.commit()


async def test_record_boundary_does_not_commit_a_caller_owned_session(temp_db):
    engine, factory = temp_db
    async with factory() as s:
        epoch = await run_config.record_boundary("sim", "in caller tx", session=s)
        assert epoch == 2
        await s.rollback()
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_epochs") == 0
    assert await run_config.current_epoch("sim", session_factory=factory) == 1


async def test_boundary_and_stamped_rows_share_one_caller_transaction(temp_db):
    """The R2 reset shape: boundary + state written atomically, or not at all."""
    engine, factory = temp_db
    async with factory() as s:
        await run_config.get_or_create("sim", "paper", {"top": 10}, session=s)
        epoch = await run_config.record_boundary("sim", "reset", session=s)
        s.add(SimTrade(
            ticker="AAA", side="BUY", shares=1.0, price=10.0, cash_after=990.0,
            reason="new epoch", mode="paper", epoch=epoch,
        ))
        await s.commit()
    assert await run_config.current_epoch("sim", session_factory=factory) == 2
    assert await run_config.has_paper_rows("sim", 2, session_factory=factory) is True
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_configs") == 1

    # A rolled-back attempt leaves neither a boundary nor a stamped row.
    async with factory() as s:
        await run_config.record_boundary("sim", "aborted reset", session=s)
        s.add(SimTrade(
            ticker="BBB", side="BUY", shares=1.0, price=10.0, cash_after=980.0,
            reason="aborted", mode="paper", epoch=3,
        ))
        await s.rollback()
    assert await run_config.current_epoch("sim", session_factory=factory) == 2
    async with factory() as s:
        assert await s.scalar(select(func.count()).select_from(SimTrade)) == 1


# ---------------------------------------------------------------------------
# epoch_clause / has_paper_rows
# ---------------------------------------------------------------------------


async def test_epoch_clause_truth_table(temp_db):
    """NULL ≡ 1 stays supported for synthetic/pre-migration fixtures.

    The current model stores a real epoch on every row (NOT NULL DEFAULT 1), so
    this probe table deliberately keeps a nullable column — it is the only way
    a pre-migration fixture can still look.
    """
    engine, _ = temp_db
    probe = sa.Table(
        "epoch_clause_probe",
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("epoch", sa.Integer, nullable=True),
    )
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE epoch_clause_probe (id INTEGER PRIMARY KEY, epoch INTEGER)"
        ))
        for row_id, epoch in ((1, None), (2, 1), (3, 2), (4, 3)):
            await conn.execute(
                text("INSERT INTO epoch_clause_probe (id, epoch) VALUES (:id, :epoch)"),
                {"id": row_id, "epoch": epoch},
            )

    async with engine.connect() as conn:
        async def count(cur: int) -> int:
            return await conn.scalar(
                select(func.count()).select_from(probe).where(
                    run_config.epoch_clause(probe, cur)
                )
            )

        assert await count(0) == 2   # <= 1: NULL (legacy) + epoch 1
        assert await count(1) == 2
        assert await count(2) == 1
        assert await count(3) == 1
        assert await count(4) == 0


async def test_epoch_is_stamped_on_every_new_row(temp_db):
    """New rows always carry a real epoch (1 by default) — no NULL hole."""
    engine, factory = temp_db
    async with factory() as s:
        s.add(SimTrade(
            ticker="default", side="BUY", shares=1.0, price=1.0, cash_after=0.0,
            reason="",
        ))
        await s.commit()
    async with engine.begin() as conn:
        await conn.execute(text(
            "INSERT INTO sim_trades (ticker, side, shares, price, cash_after, reason, "
            "created_at) VALUES ('server_default', 'BUY', 1.0, 1.0, 0.0, '', "
            "'2026-01-02 10:00:00.000000')"
        ))
    async with engine.connect() as conn:
        epochs = (await conn.execute(text(
            "SELECT ticker, epoch FROM sim_trades ORDER BY id"))).all()
    assert [tuple(row) for row in epochs] == [("default", 1), ("server_default", 1)]

    # The DB itself refuses a NULL epoch, so no writer can re-open the hole.
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO sim_trades (ticker, side, shares, price, cash_after, "
                "reason, created_at, epoch) VALUES ('nulled', 'BUY', 1.0, 1.0, 0.0, "
                "'', '2026-01-02 10:00:00.000000', NULL)"
            ))


async def test_has_paper_rows_truth_table(temp_db):
    """FAIL CLOSED: only a known ``backfill`` row is safe to replace."""
    _, factory = temp_db
    async with factory() as s:
        rows = [
            ("paper_e2", "paper", 2),
            ("backfill_e3", "backfill", 3),
            ("unknown_e4", None, 4),
            ("weird_e5", "forward-ish", 5),
            ("paper_e1", "paper", 1),
        ]
        for ticker, mode, epoch in rows:
            s.add(SimTrade(
                ticker=ticker, side="BUY", shares=1.0, price=1.0, cash_after=0.0,
                reason="", mode=mode, epoch=epoch,
            ))
        await s.commit()

    assert await run_config.has_paper_rows("sim", 2, session_factory=factory) is True
    assert await run_config.has_paper_rows("sim", 1, session_factory=factory) is True
    assert await run_config.has_paper_rows("sim", 0, session_factory=factory) is True
    # a known backfill is the ONLY replaceable shape
    assert await run_config.has_paper_rows("sim", 3, session_factory=factory) is False
    # an unstamped legacy row (mode NULL) is PROTECTED, not disposable
    assert await run_config.has_paper_rows("sim", 4, session_factory=factory) is True
    # any unexpected mode fails closed too
    assert await run_config.has_paper_rows("sim", 5, session_factory=factory) is True
    assert await run_config.has_paper_rows("sim", 6, session_factory=factory) is False


async def test_has_paper_rows_is_engine_scoped(temp_db):
    _, factory = temp_db
    async with factory() as s:
        s.add(MonthlyRebalance(
            rebal_month="2026-01", rebal_date=datetime(2026, 1, 30, tzinfo=UTC),
            mode="paper", epoch=2,
        ))
        s.add(SimAllowance(amount=100.0, month="2026-01", mode="backfill", epoch=2))
        await s.commit()

    # The monthly paper row must not make the sim book look protected.
    assert await run_config.has_paper_rows("sim", 2, session_factory=factory) is False
    assert await run_config.has_paper_rows("monthly", 2, session_factory=factory) is True
    assert await run_config.has_paper_rows("monthly", 1, session_factory=factory) is False


# ---------------------------------------------------------------------------
# init_db migration
# ---------------------------------------------------------------------------


async def test_init_db_migrates_a_legacy_database(migrated_db):
    engine, _ = migrated_db

    # 1. Every history table now carries the stamps; current-state tables do not.
    for table in _HISTORY_TABLES:
        cols = [row[0] for row in await _columns(engine, table)]
        assert set(_STAMP_COLUMNS) <= set(cols), table
    for table in ("sim_account", "sim_positions", "monthly_account",
                  "monthly_positions", "daily_core_account",
                  "daily_core_positions", "savings_account"):
        cols = [row[0] for row in await _columns(engine, table)]
        assert set(_STAMP_COLUMNS).isdisjoint(cols), table

    # 2. The allowance / rebalance UNIQUE keys are epoch-aware now.
    for table in ("sim_allowances", "monthly_allowances", "daily_core_allowances"):
        uniques = await _unique_indexes(engine, table)
        assert ("month", "epoch") in uniques
        assert ("month",) not in uniques
    uniques = await _unique_indexes(engine, "monthly_rebalances")
    assert ("rebal_month", "epoch") in uniques
    assert ("rebal_month",) not in uniques

    # 3. Every legacy row survived bit-for-bit, ids included; mode/config_id
    #    unknown (NULL) and epoch materialized as 1 — the pre-reset book.
    async with engine.connect() as conn:
        allowances = (await conn.execute(text(
            "SELECT id, amount, month, created_at, mode, config_id, epoch "
            "FROM sim_allowances ORDER BY id"))).all()
        rebalance = (await conn.execute(text(
            "SELECT id, rebal_month, rebal_date, held_before, picked, n_new, "
            "snapshot, created_at, mode, config_id, epoch "
            "FROM monthly_rebalances"))).one()
        trade = (await conn.execute(text(
            "SELECT id, ticker, epoch FROM sim_trades"))).one()
        events = (await conn.execute(text(
            "SELECT id, kind, settlement_of FROM savings_events ORDER BY id"))).all()
        allowance_total = (await conn.execute(text(
            "SELECT COUNT(*) FROM monthly_allowances"))).scalar_one()
        ranking = (await conn.execute(text(
            "SELECT id, band FROM daily_core_ranking"))).one()
    assert [tuple(row[:4]) for row in allowances] == _SEEDED_ALLOWANCES
    assert all(tuple(row[4:]) == (None, None, 1) for row in allowances)
    assert tuple(rebalance[:8]) == _SEEDED_REBALANCE
    assert tuple(rebalance[8:]) == (None, None, 1)
    assert tuple(trade) == (3, "AAA", 1)
    assert [tuple(row) for row in events] == [(1, "SELL", None), (2, "SETTLEMENT", None)]
    assert allowance_total == 1
    assert tuple(ranking) == (1, "AAA")

    # 3b. epoch is a REAL value everywhere: NOT NULL with DEFAULT 1, so no
    #     writer (or migrator) can leave the duplicate-month hole open.
    for table in _HISTORY_TABLES:
        epoch = [row for row in await _columns(engine, table) if row[0] == "epoch"][0]
        assert epoch[2] == 1, table          # NOT NULL
        assert epoch[3] == "1", table        # DEFAULT 1

    # 4. The savings settlement link: column added, legacy rows NULL, UNIQUE.
    cols = [row[0] for row in await _columns(engine, "savings_events")]
    assert "settlement_of" in cols
    assert ("settlement_of",) in await _unique_indexes(engine, "savings_events")

    # 5. The registry tables exist with their dedupe/boundary constraints.
    assert ("engine", "mode", "config_hash") in await _unique_indexes(
        engine, "run_configs")
    assert ("engine", "epoch") in await _unique_indexes(engine, "run_epochs")


async def test_init_db_is_a_noop_on_an_already_migrated_database(migrated_db):
    """init_db ran twice in the fixture; the second run changed nothing."""
    engine, _ = migrated_db
    before = {
        table: (await _columns(engine, table), await _unique_indexes(engine, table))
        for table in _HISTORY_TABLES
    }
    counts = {
        table: await _scalar(engine, f'SELECT COUNT(*) FROM "{table}"')
        for table in _HISTORY_TABLES
    }
    await db_mod.init_db(engine)
    for table in _HISTORY_TABLES:
        assert (await _columns(engine, table)) == before[table][0], table
        assert (await _unique_indexes(engine, table)) == before[table][1], table
        assert await _scalar(engine, f'SELECT COUNT(*) FROM "{table}"') == counts[table]
    assert await _scalar(engine, "SELECT COUNT(*) FROM sim_allowances") == 2


async def test_fresh_and_migrated_table_shapes_match(tmp_path):
    """create_all's shape IS the migration target — no drift on either path."""
    fresh = _make_engine(tmp_path, "shape_fresh.db")
    legacy = _make_engine(tmp_path, "shape_legacy.db")
    try:
        async with fresh.begin() as conn:
            await conn.run_sync(db_mod.Base.metadata.create_all)
        await _build_legacy_db(legacy)
        await db_mod.init_db(legacy)
        for table in _HISTORY_TABLES:
            assert sorted(await _columns(legacy, table)) == sorted(
                await _columns(fresh, table)
            ), table
            assert sorted(await _unique_indexes(legacy, table)) == sorted(
                await _unique_indexes(fresh, table)
            ), table
        for table in ("run_configs", "run_epochs", "run_executions",
                      "sim_account", "monthly_account", "daily_core_account"):
            assert sorted(await _columns(legacy, table)) == sorted(
                await _columns(fresh, table)
            ), table
    finally:
        await fresh.dispose()
        await legacy.dispose()


async def test_rebuild_preserves_partially_upgraded_stamps(blank_db):
    """A DB that already has the columns keeps the values it stamped.

    A partially upgraded DB may carry a NULLABLE epoch column; the rebuild must
    normalize its NULLs to 1 (the pre-reset book) and leave the final
    NOT NULL DEFAULT 1 shape."""
    engine, _ = blank_db
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE sim_allowances (
                id INTEGER NOT NULL PRIMARY KEY,
                amount FLOAT NOT NULL,
                month VARCHAR(7) NOT NULL,
                created_at DATETIME NOT NULL,
                mode VARCHAR(16),
                config_id INTEGER,
                epoch INTEGER,
                UNIQUE (month)
            )
        """))
        await conn.execute(text(
            "INSERT INTO sim_allowances VALUES "
            "(1, 100.0, '2026-01', '2026-01-02 10:00:00.000000', 'paper', 42, 3)"))
        await conn.execute(text(
            "INSERT INTO sim_allowances VALUES "
            "(2, 200.0, '2026-02', '2026-02-02 10:00:00.000000', NULL, NULL, NULL)"))
        await conn.execute(text("""
            CREATE TABLE monthly_rebalances (
                id INTEGER NOT NULL PRIMARY KEY,
                rebal_month VARCHAR(7) NOT NULL,
                rebal_date DATETIME NOT NULL,
                held_before TEXT NOT NULL,
                picked TEXT NOT NULL,
                n_new INTEGER NOT NULL,
                snapshot TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                mode VARCHAR(16),
                config_id INTEGER,
                epoch INTEGER,
                UNIQUE (rebal_month)
            )
        """))
        await conn.execute(text(
            "INSERT INTO monthly_rebalances VALUES "
            "(1, '2026-01', '2026-01-30 00:00:00.000000', '', '', 0, '', "
            "'2026-01-30 12:00:00.000000', 'backfill', 7, 2)"))

    await db_mod.init_db(engine)

    async with engine.connect() as conn:
        allowances = (await conn.execute(text(
            "SELECT id, amount, month, mode, config_id, epoch "
            "FROM sim_allowances ORDER BY id"
        ))).all()
        rebalance = (await conn.execute(text(
            "SELECT rebal_month, mode, config_id, epoch FROM monthly_rebalances"
        ))).one()
    assert [tuple(row) for row in allowances] == [
        (1, 100.0, "2026-01", "paper", 42, 3),   # stamped values survive
        (2, 200.0, "2026-02", None, None, 1),    # a NULL epoch is the pre-reset book
    ]
    assert tuple(rebalance) == ("2026-01", "backfill", 7, 2)
    epoch = [row for row in await _columns(engine, "sim_allowances") if row[0] == "epoch"][0]
    assert epoch[2] == 1 and epoch[3] == "1"
    assert ("month", "epoch") in await _unique_indexes(engine, "sim_allowances")
    assert ("rebal_month", "epoch") in await _unique_indexes(engine, "monthly_rebalances")


async def test_partial_epoch_upgrade_rebuilds_a_plain_history_table(blank_db):
    """A nullable epoch on a non-constrained history table is rebuilt too.

    ``sim_trades`` has no legacy UNIQUE key, so it is normally a pure ALTER
    target — but an intermediate build left ``epoch`` nullable, which SQLite
    cannot tighten in place. The rebuild must keep rows, normalize NULL to 1,
    and recreate the table's explicit index."""
    engine, _ = blank_db
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE sim_trades (
                id INTEGER NOT NULL PRIMARY KEY,
                ticker VARCHAR(32) NOT NULL,
                side VARCHAR(4) NOT NULL,
                shares FLOAT NOT NULL,
                price FLOAT NOT NULL,
                cash_after FLOAT NOT NULL,
                reason TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                mode VARCHAR(16),
                config_id INTEGER,
                epoch INTEGER
            )
        """))
        await conn.execute(text(
            "CREATE INDEX ix_sim_trades_ticker ON sim_trades (ticker)"))
        await conn.execute(text(
            "INSERT INTO sim_trades VALUES (1, 'AAA', 'BUY', 1.0, 10.0, 990.0, 'r', "
            "'2026-01-02 10:00:00.000000', 'paper', 5, NULL)"))

    await db_mod.init_db(engine)

    columns = {row[0]: row for row in await _columns(engine, "sim_trades")}
    assert columns["epoch"][2] == 1 and columns["epoch"][3] == "1"
    async with engine.connect() as conn:
        row = (await conn.execute(text(
            "SELECT ticker, mode, config_id, epoch FROM sim_trades"))).one()
    assert tuple(row) == ("AAA", "paper", 5, 1)
    assert "ix_sim_trades_ticker" in await _index_names(engine, "sim_trades")
    async with engine.connect() as conn:
        indexed = [r[2] for r in (await conn.execute(
            text("PRAGMA index_info('ix_sim_trades_ticker')"))).all()]
    assert indexed == ["ticker"]


async def test_partial_epoch_without_default_is_normalized_to_default_one(blank_db):
    """A NOT NULL epoch WITHOUT a server default must gain ``DEFAULT 1``.

    The migration gate is "is it exactly NOT NULL DEFAULT 1?", not merely
    "does the column exist": a writer that omits the epoch would fail on such
    a column, and stale values must survive while the default is materialized.
    """
    engine, _ = blank_db
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE sim_trades (
                id INTEGER NOT NULL PRIMARY KEY,
                ticker VARCHAR(32) NOT NULL,
                side VARCHAR(4) NOT NULL,
                shares FLOAT NOT NULL,
                price FLOAT NOT NULL,
                cash_after FLOAT NOT NULL,
                reason TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                mode VARCHAR(16),
                config_id INTEGER,
                epoch INTEGER NOT NULL
            )
        """))
        await conn.execute(text(
            "INSERT INTO sim_trades VALUES (1, 'AAA', 'BUY', 1.0, 10.0, 990.0, 'r', "
            "'2026-01-02 10:00:00.000000', 'paper', 5, 7)"))

    # Pre-migration the shape cannot default the epoch at all.
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO sim_trades (ticker, side, shares, price, cash_after, "
                "reason, created_at) VALUES ('BBB', 'BUY', 1.0, 1.0, 0.0, '', "
                "'2026-01-03 10:00:00.000000')"))

    await db_mod.init_db(engine)

    epoch = [row for row in await _columns(engine, "sim_trades") if row[0] == "epoch"][0]
    assert epoch[2] == 1 and epoch[3] == "1"
    # The stamped row survives and a raw INSERT omitting epoch now lands in 1.
    async with engine.begin() as conn:
        await conn.execute(text(
            "INSERT INTO sim_trades (ticker, side, shares, price, cash_after, "
            "reason, created_at) VALUES ('BBB', 'BUY', 1.0, 1.0, 0.0, '', "
            "'2026-01-03 10:00:00.000000')"))
    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT ticker, epoch FROM sim_trades ORDER BY id"))).all()
    assert [tuple(row) for row in rows] == [("AAA", 7), ("BBB", 1)]


async def test_settlement_of_has_exactly_one_unique_index_on_both_paths(tmp_path):
    """Fresh (inline UNIQUE) and migrated (named index) agree: one index only."""
    fresh = _make_engine(tmp_path, "settle_fresh.db")
    legacy = _make_engine(tmp_path, "settle_legacy.db")
    try:
        await db_mod.init_db(fresh)
        await _build_legacy_db(legacy)
        await db_mod.init_db(legacy)
        for engine, label in ((fresh, "fresh"), (legacy, "migrated")):
            covering = [
                cols for cols in await _unique_indexes(engine, "savings_events")
                if cols == ("settlement_of",)
            ]
            assert covering == [("settlement_of",)], label
    finally:
        await fresh.dispose()
        await legacy.dispose()


async def test_account_opening_seed_migration_is_additive(blank_db):
    """Legacy current-state accounts gain a NULLABLE seed — and nothing else.

    The seed is NOT inferred from cash or from the first paper execution: a
    legacy row keeps NULL, because the settings in force when it was created
    were never recorded. Repeated startups are no-ops.
    """
    engine, _ = blank_db
    async with engine.begin() as conn:
        await conn.run_sync(db_mod.Base.metadata.create_all)
        for table, ddl in _LEGACY_ACCOUNT_DDL.items():
            await conn.execute(text(f'DROP TABLE "{table}"'))
            await conn.execute(text(ddl))
            await conn.execute(text(
                f"INSERT INTO {table} (id, cash, created_at) VALUES "
                "(1, 123.45, '2026-01-02 10:00:00.000000')"
            ))
    before = {table: await _columns(engine, table) for table in _LEGACY_ACCOUNT_DDL}

    await db_mod.init_db(engine)
    after_first = {table: await _columns(engine, table) for table in _LEGACY_ACCOUNT_DDL}
    await db_mod.init_db(engine)   # repeated startup must not re-ALTER
    after_second = {table: await _columns(engine, table) for table in _LEGACY_ACCOUNT_DDL}

    for table in _LEGACY_ACCOUNT_DDL:
        # opening_seed is the ONLY schema change: FLOAT, nullable, no default.
        assert after_first[table] == before[table] + [
            ("opening_seed", "FLOAT", 0, None, 0)
        ], table
        assert after_second[table] == after_first[table], table
        async with engine.connect() as conn:
            row = (await conn.execute(text(
                f"SELECT id, cash, opening_seed FROM {table}"))).one()
        assert tuple(row) == (1, 123.45, None), table   # cash kept, seed unknown


async def test_account_opening_seed_roundtrips_on_a_fresh_db(temp_db):
    """A fresh account stores the REAL seed it was created with (or NULL)."""
    engine, factory = temp_db
    columns = {row[0]: row for row in await _columns(engine, "sim_account")}
    assert columns["opening_seed"] == ("opening_seed", "FLOAT", 0, None, 0)

    async with factory() as s:
        s.add(SimAccount(id=1, cash=0.0, opening_seed=100.0))
        s.add(MonthlyAccount(id=1, cash=0.0, opening_seed=100.25))
        s.add(DailyCoreAccount(id=1, cash=0.0))   # seed unknown, never guessed
        await s.commit()
    async with factory() as s:
        assert (await s.get(SimAccount, 1)).opening_seed == 100.0
        assert (await s.get(MonthlyAccount, 1)).opening_seed == 100.25
        assert (await s.get(DailyCoreAccount, 1)).opening_seed is None


async def test_rebuild_aborts_on_an_unknown_legacy_column(blank_db):
    """An unknown column means data would be lost: fail loudly, keep the table."""
    engine, _ = blank_db
    await _build_legacy_db(engine)
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE sim_allowances ADD COLUMN mystery TEXT"))
        await conn.execute(text(
            "UPDATE sim_allowances SET mystery = 'keep me' WHERE id = 1"))

    with pytest.raises(RuntimeError, match="mystery"):
        await db_mod.init_db(engine)

    rows = await _scalar(engine, "SELECT COUNT(*) FROM sim_allowances")
    value = await _scalar(engine, "SELECT mystery FROM sim_allowances WHERE id = 1")
    assert rows == 2
    assert value == "keep me"


async def test_failed_rebuild_rolls_back_and_preserves_the_legacy_rows(blank_db):
    """The destructive half is transactional: a failure loses no history."""
    engine, _ = blank_db
    async with engine.begin() as conn:
        # Synthetic legacy table whose ``amount`` is nullable — the final shape
        # is NOT NULL, so the copy fails after the staging table exists.
        await conn.execute(text("""
            CREATE TABLE sim_allowances (
                id INTEGER NOT NULL PRIMARY KEY,
                amount FLOAT,
                month VARCHAR(7) NOT NULL UNIQUE,
                created_at DATETIME NOT NULL
            )
        """))
        await conn.execute(text(
            "INSERT INTO sim_allowances (id, amount, month, created_at) VALUES "
            "(1, 100.0, '2026-01', '2026-01-02 10:00:00.000000'), "
            "(2, NULL, '2026-02', '2026-02-02 10:00:00.000000')"))

    with pytest.raises(IntegrityError):
        await db_mod.init_db(engine)

    cols = [row[0] for row in await _columns(engine, "sim_allowances")]
    assert "mode" not in cols  # the migration did not half-apply
    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT id, amount, month FROM sim_allowances ORDER BY id"))).all()
    assert [tuple(row) for row in rows] == [
        (1, 100.0, "2026-01"), (2, None, "2026-02")
    ]


# ---------------------------------------------------------------------------
# Constraint semantics after the rebuild
# ---------------------------------------------------------------------------


async def test_allowance_unique_constraint_is_epoch_aware(migrated_db):
    """Same month across epochs coexists; same (month, epoch) is refused.

    The migrated legacy row ('2026-01') is epoch 1, so there is NO NULL hole:
    a writer cannot slip a duplicate of it in with a NULL epoch."""
    engine, factory = migrated_db
    async with factory() as s:
        s.add(SimAllowance(amount=1.0, month="2026-01", mode="paper", epoch=2))
        await s.commit()
    async with factory() as s:
        s.add(SimAllowance(amount=2.0, month="2026-01", mode="paper", epoch=3))
        await s.commit()
    assert await _scalar(
        engine, "SELECT COUNT(*) FROM sim_allowances WHERE month = '2026-01'") == 3

    # same (month, epoch) -> rejected
    async with factory() as s:
        s.add(SimAllowance(amount=3.0, month="2026-01", mode="paper", epoch=2))
        with pytest.raises(IntegrityError):
            await s.commit()

    # the legacy epoch-1 row is protected too (this is the hole NULL created)
    async with factory() as s:
        s.add(SimAllowance(amount=4.0, month="2026-01", mode="paper", epoch=1))
        with pytest.raises(IntegrityError):
            await s.commit()

    # a fresh month may be logged by the new epoch and later epochs alike
    async with factory() as s:
        s.add(SimAllowance(amount=5.0, month="2025-12", mode="paper", epoch=1))
        await s.commit()
    async with factory() as s:
        s.add(SimAllowance(amount=6.0, month="2025-12", mode="paper", epoch=2))
        await s.commit()
    assert await _scalar(
        engine, "SELECT COUNT(*) FROM sim_allowances WHERE month = '2025-12'") == 2


async def test_rebalance_unique_constraint_is_epoch_aware(migrated_db):
    """Same rebal_month across epochs coexists; same (month, epoch) is refused."""
    engine, factory = migrated_db
    when = datetime(2026, 2, 27, tzinfo=UTC)
    async with factory() as s:
        s.add(MonthlyRebalance(rebal_month="2026-01", rebal_date=when, epoch=2))
        await s.commit()
    async with factory() as s:
        s.add(MonthlyRebalance(rebal_month="2026-01", rebal_date=when, epoch=3))
        await s.commit()
    assert await _scalar(
        engine, "SELECT COUNT(*) FROM monthly_rebalances WHERE rebal_month = '2026-01'"
    ) == 3  # the legacy seeded row (epoch 1) + the two new epochs

    async with factory() as s:
        s.add(MonthlyRebalance(rebal_month="2026-01", rebal_date=when, epoch=2))
        with pytest.raises(IntegrityError):
            await s.commit()
    async with factory() as s:
        s.add(MonthlyRebalance(rebal_month="2026-01", rebal_date=when, epoch=1))
        with pytest.raises(IntegrityError):
            await s.commit()


async def test_savings_settlement_of_is_unique_and_nullable(migrated_db):
    engine, factory = migrated_db
    async with factory() as s:
        await s.execute(text(
            "INSERT INTO savings_events (kind, amount, note, created_at, settlement_of) "
            "VALUES ('SETTLEMENT', 100, 'linked', '2026-02-01 10:00:00.000000', 1)"))
        await s.execute(text(
            "INSERT INTO savings_events (kind, amount, note, created_at, settlement_of) "
            "VALUES ('SETTLEMENT', 50, 'legacy a', '2026-03-01 10:00:00.000000', NULL)"))
        await s.execute(text(
            "INSERT INTO savings_events (kind, amount, note, created_at, settlement_of) "
            "VALUES ('SETTLEMENT', 60, 'legacy b', '2026-04-01 10:00:00.000000', NULL)"))
        await s.commit()
    assert await _scalar(
        engine, "SELECT COUNT(*) FROM savings_events WHERE settlement_of IS NULL") == 4

    async with factory() as s:
        s.add(db_mod.SavingsEvent(
            kind="SETTLEMENT", amount=70.0, note="double credit", settlement_of=1))
        with pytest.raises(IntegrityError):
            await s.commit()


# ---------------------------------------------------------------------------
# record_execution: the append-only, NOT deduped execution log
# ---------------------------------------------------------------------------


async def test_run_executions_are_append_only_and_not_deduped(temp_db):
    """Two identical executions are two rows; the CONFIG stays deduped."""
    engine, factory = temp_db
    config_id = await run_config.get_or_create(
        "sim", "paper", {"top": 10}, session_factory=factory)
    provenance = {"candles": {"source": "yfinance", "rows": 120}}
    first = await run_config.record_execution(
        "sim", "paper", config_id, 1, provenance, session_factory=factory)
    second = await run_config.record_execution(
        "sim", "paper", config_id, 1, provenance, session_factory=factory)
    assert first != second
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_executions") == 2
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_configs") == 1


async def test_record_execution_stores_canonical_provenance_and_asof(temp_db):
    engine, factory = temp_db
    config_id = await run_config.get_or_create(
        "monthly", "backfill", {"top": 20}, session_factory=factory)
    asof = datetime(2026, 1, 30, 21, 0, tzinfo=UTC)
    execution_id = await run_config.record_execution(
        "monthly", "backfill", config_id, 2,
        {"sources": ["edgar", "yfinance"], "stale_days": 0},
        data_asof=asof, session_factory=factory)
    async with factory() as s:
        row = await s.get(RunExecution, execution_id)
    assert (row.engine, row.mode, row.config_id, row.epoch) == (
        "monthly", "backfill", config_id, 2)
    assert row.data_asof == asof
    assert row.provenance == '{"sources":["edgar","yfinance"],"stale_days":0}'
    assert row.created_at.tzinfo is not None


async def test_record_execution_does_not_commit_a_caller_owned_session(temp_db):
    engine, factory = temp_db
    config_id = await run_config.get_or_create("sim", "paper", {}, session_factory=factory)
    async with factory() as s:
        await run_config.record_execution("sim", "paper", config_id, 1, {"a": 1}, session=s)
        await s.rollback()
    assert await _scalar(engine, "SELECT COUNT(*) FROM run_executions") == 0


async def test_record_execution_validates_its_inputs(temp_db):
    _, factory = temp_db
    config_id = await run_config.get_or_create("sim", "paper", {}, session_factory=factory)
    with pytest.raises(ValueError, match="unknown engine"):
        await run_config.record_execution(
            "nope", "paper", config_id, 1, {}, session_factory=factory)
    with pytest.raises(ValueError, match="unknown mode"):
        await run_config.record_execution(
            "sim", "live", config_id, 1, {}, session_factory=factory)
    with pytest.raises(ValueError, match="config_id"):
        await run_config.record_execution("sim", "paper", 0, 1, {}, session_factory=factory)
    with pytest.raises(ValueError, match="epoch"):
        await run_config.record_execution("sim", "paper", config_id, 0, {}, session_factory=factory)
    with pytest.raises(TypeError, match="data_asof"):
        await run_config.record_execution(
            "sim", "paper", config_id, 1, {}, data_asof="2026-01-01",
            session_factory=factory)
    with pytest.raises(ValueError, match="timezone-aware"):
        await run_config.record_execution(
            "sim", "paper", config_id, 1, {}, data_asof=datetime(2026, 1, 1),
            session_factory=factory)
    with pytest.raises(ValueError):
        await run_config.record_execution(
            "sim", "paper", config_id, 1, {"x": float("nan")}, session_factory=factory)


async def test_run_execution_schema_defaults_epoch_and_requires_provenance(temp_db):
    engine, _ = temp_db
    columns = {row[0]: row for row in await _columns(engine, "run_executions")}
    assert set(columns) == {
        "id", "engine", "mode", "epoch", "config_id", "data_asof", "provenance",
        "created_at",
    }
    assert columns["epoch"][2] == 1 and columns["epoch"][3] == "1"
    assert columns["config_id"][2] == 1   # NOT NULL
    assert columns["provenance"][2] == 1  # NOT NULL
    assert columns["data_asof"][2] == 0   # nullable
    assert columns["engine"][2] == 1 and columns["mode"][2] == 1


# ---------------------------------------------------------------------------
# Import contract
# ---------------------------------------------------------------------------


def test_schema_and_module_do_not_import_engines(tmp_path):
    """db + run_config stay engine-import-free (engines import THEM)."""
    code = (
        "import sys\n"
        "import app.db, app.run_config\n"
        "engines = {'app.daily_core', 'app.monthly', 'app.sim', 'app.savings',"
        " 'app.main', 'app.optimize', 'app.analysis', 'app.screener'}\n"
        "print(sorted(engines & set(sys.modules)))\n"
    )
    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path / 'imports.db'}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == "[]", result.stdout + result.stderr
