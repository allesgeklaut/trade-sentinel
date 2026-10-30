"""Run configuration + epoch registry: the ONE API for run provenance.

POLICY (Phase 2 of the strategy audit — "give each run an immutable
configuration/version, data provenance and mode: historical experiment versus
forward paper. Do not overwrite a forward record." plus "record the number of
configurations tried"):

* **Immutable configurations, honest execution records.** A run's settings are
  FROZEN when the run starts. ``get_or_create`` registers (engine, mode,
  canonical JSON) and returns the SAME ``RunConfig`` id for identical content,
  so a scheduler that runs an unchanged configuration every day keeps one row.
  Config rows are append-only: never updated, never deleted. Counting rows for
  an engine reports the number of DISTINCT CONFIGURATIONS registered — NOT a
  count of execution runs, and not by itself a count of experiments performed.
  ``record_execution`` is the separate, deliberately NON-deduped log of actual
  runs (one row each), carrying the run's per-execution data provenance; that
  is what makes "how many times did this run, on which data?" answerable
  without inflating the configuration count.
* **Modes.** ``paper`` marks a live forward record, ``backfill`` a synthetic
  replay. A backfill writes its own rows under its own configuration/mode and
  must never overwrite a paper row — the mode stamp is what makes the two
  tellable apart afterwards. A history row with mode NULL was written before
  stamping existed: it cannot be PROVEN forward, so the fail-closed guard
  (``has_paper_rows``) treats it as PROTECTED, never as disposable.
* **Epochs, the reset-without-rewriting-history contract.** ``record_boundary``
  does not delete anything: it appends one ``RunEpoch`` row and bumps that
  engine's epoch. History rows keep the epoch they were written in, and
  ``epoch_clause`` scopes a reader (or a writer's duplicate check) to one
  epoch. CONVENTION: no ``RunEpoch`` row for an engine => the current epoch is
  1, and legacy rows are MIGRATED to epoch 1 (the logical pre-reset book —
  storing NULL would invent an unknown and re-open the duplicate-month hole).
  ``epoch_clause`` still matches NULL for synthetic/pre-migration fixtures,
  but every row the current model stores has a real epoch >= 1.
  ``UNIQUE (engine, epoch)`` plus a single-statement allocation make two
  concurrent resets land on two different epoch numbers.

TRANSACTIONS: every helper takes an optional caller-owned ``AsyncSession``
(``session=``) and an optional ``session_factory=``. A CALLER-OWNED SESSION IS
NEVER COMMITTED here — the caller owns the transaction, so a reset can write
the boundary row and the account/state rows in ONE atomic transaction and a
cycle can register its configuration together with the rows it explains. When
no session is passed, the helper opens one from the factory (default:
``db.Session``, resolved at call time so an engine fixture's patched factory
is honoured) and commits its own writes. Read-only helpers never commit.

IMPORT CONTRACT: this module imports ``app.db``, the SQLAlchemy core and the
stdlib ONLY. It must never import an engine module (daily_core/monthly/sim) —
those import THIS module, and that cycle is fatal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import func, insert, or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from . import db as _db
from .db import (
    DailyCoreAllowance,
    DailyCoreRanking,
    DailyCoreSnapshot,
    DailyCoreTrade,
    MonthlyAllowance,
    MonthlyRebalance,
    MonthlySnapshot,
    MonthlyTrade,
    RunConfig,
    RunEpoch,
    RunExecution,
    SimAllowance,
    SimSnapshot,
    SimTrade,
)

# The history tables that belong to each engine — the ONE place this mapping
# lives. Test/snapshot tables are history; current-state tables (accounts,
# positions) are not stamped and are absent on purpose.
ENGINES: dict[str, tuple[type, ...]] = {
    "daily_core": (
        DailyCoreTrade, DailyCoreAllowance, DailyCoreSnapshot, DailyCoreRanking,
    ),
    "monthly": (
        MonthlyTrade, MonthlyAllowance, MonthlySnapshot, MonthlyRebalance,
    ),
    "sim": (
        SimTrade, SimAllowance, SimSnapshot,
    ),
}

# The run/history mode vocabulary. Paper is a forward record; backfill is a
# synthetic replay. Anything else is a bug, not an extension point.
MODES: frozenset[str] = frozenset({"paper", "backfill"})

# How often a boundary allocation reopens its transaction on a concurrent
# writer. SQLite serializes writers, so one retry is normally enough.
_BOUNDARY_ATTEMPTS = 6


def _validate_engine(engine: str) -> str:
    """Reject an unknown engine instead of registering an unusable config."""
    if engine not in ENGINES:
        raise ValueError(
            f"unknown engine {engine!r}; expected one of {sorted(ENGINES)}"
        )
    return engine


def _validate_mode(mode: str) -> str:
    """Reject an unknown mode: the vocabulary is closed (paper | backfill)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    return mode


def canonical(config: dict[str, Any]) -> str:
    """Deterministic JSON for *config*: sorted keys, tight separators.

    Two callers that build the same settings map in a different key order get
    the identical string, hence the identical hash. ``allow_nan=False`` keeps
    the registry honest: NaN/Infinity have no canonical JSON spelling and
    would make the stored text (the provenance record) lie about what was
    loaded. A value that cannot be serialized raises — it is never silently
    dropped.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_content(content: str) -> str:
    """sha256 hex digest of canonical content (the ONE hash definition)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """sha256 hex digest of ``canonical(config)`` encoded as UTF-8."""
    return _hash_content(canonical(config))


def _positive_int(value, name: str) -> int:
    """Coerce *value* to an int >= 1, or raise (ids/epochs are never 0)."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {number}")
    return number


@asynccontextmanager
async def _session_scope(
    session: AsyncSession | None,
    session_factory,
) -> AsyncIterator[tuple[AsyncSession, bool]]:
    """Yield ``(session, owned)``; ``owned`` means this helper commits.

    A caller-owned session is used exactly as passed in: no commit, no close,
    no rollback — transaction boundaries belong to the caller.
    """
    if session is not None:
        yield session, False
        return
    factory = session_factory if session_factory is not None else _db.Session
    async with factory() as owned_session:
        yield owned_session, True


async def get_or_create(
    engine: str,
    mode: str,
    config: dict[str, Any],
    *,
    session: AsyncSession | None = None,
    session_factory=None,
) -> int:
    """Return the ``RunConfig`` id for (engine, mode, content), creating it once.

    Content-hash dedupe: identical settings return the SAME id and leave the
    one row untouched (its ``created_at`` is the FIRST sighting — append-only).
    The insert is ``ON CONFLICT DO NOTHING`` against
    ``UNIQUE (engine, mode, config_hash)``, so two concurrent first runs
    cannot create two rows; the follow-up SELECT returns whichever row won.
    A caller-owned session is not committed here, so config registration can
    share a transaction with the rows it explains.
    """
    _validate_engine(engine)
    _validate_mode(mode)
    content = canonical(config)
    digest = _hash_content(content)
    match = (
        RunConfig.engine == engine,
        RunConfig.mode == mode,
        RunConfig.config_hash == digest,
    )
    async with _session_scope(session, session_factory) as (s, owned):
        row_id = await s.scalar(select(RunConfig.id).where(*match))
        if row_id is None:
            await s.execute(
                sqlite_insert(RunConfig)
                .values(engine=engine, mode=mode, config_hash=digest, content=content)
                .on_conflict_do_nothing(
                    index_elements=["engine", "mode", "config_hash"]
                )
            )
            row_id = await s.scalar(select(RunConfig.id).where(*match))
            if row_id is None:  # pragma: no cover - the row must exist by now
                raise RuntimeError(
                    f"run_configs row for {engine}/{mode}/{digest} vanished"
                )
        if owned:
            await s.commit()
        return int(row_id)


def _next_epoch_select(engine: str):
    """``COALESCE(MAX(epoch), 1) + 1`` for *engine* as a scalar subquery.

    Evaluated INSIDE the INSERT statement, i.e. while SQLite holds the write
    lock, so a concurrent boundary cannot read the same MAX and allocate the
    same number twice. ``UNIQUE (engine, epoch)`` remains the backstop.
    """
    return (
        select(func.coalesce(func.max(RunEpoch.epoch), 1) + 1)
        .where(RunEpoch.engine == engine)
        .scalar_subquery()
    )


async def _record_boundary_in(s: AsyncSession, engine: str, note: str) -> int:
    """Append the boundary row in *s* (no commit) and return the new epoch."""
    await s.execute(
        insert(RunEpoch).values(
            engine=engine, epoch=_next_epoch_select(engine), note=note
        )
    )
    epoch = await s.scalar(
        select(func.max(RunEpoch.epoch)).where(RunEpoch.engine == engine)
    )
    if epoch is None:  # pragma: no cover - the insert above guarantees a row
        raise RuntimeError(f"run_epochs row for {engine} vanished after insert")
    return int(epoch)


async def record_boundary(
    engine: str,
    note: str = "",
    *,
    session: AsyncSession | None = None,
    session_factory=None,
) -> int:
    """Append ONE reset boundary for *engine*; return the new epoch number.

    Nothing is deleted or re-tagged: the new boundary is the FIRST epoch number
    after the current one (no rows => current epoch 1 => the first boundary is
    2), and every history row keeps the epoch it was written in. Callers then
    stamp their new rows with the returned epoch.

    Inside a caller-owned session this is a single statement-batch with no
    commit, so a reset can append the boundary and rewrite current state
    atomically. When this helper owns the transaction it retries the whole
    allocation on a concurrent writer (``UNIQUE (engine, epoch)`` collision or
    SQLite BUSY), because a failed transaction must be reopened to re-read the
    now-committed MAX.
    """
    _validate_engine(engine)
    note = "" if note is None else str(note)
    if session is not None:
        return await _record_boundary_in(session, engine, note)
    factory = session_factory if session_factory is not None else _db.Session
    for attempt in range(1, _BOUNDARY_ATTEMPTS + 1):
        async with factory() as s:
            try:
                epoch = await _record_boundary_in(s, engine, note)
                await s.commit()
                return epoch
            except IntegrityError:
                # RunEpoch's only constraint is UNIQUE (engine, epoch): another
                # reset allocated this number first.
                await s.rollback()
                if attempt == _BOUNDARY_ATTEMPTS:
                    raise
            except OperationalError as exc:
                message = str(exc).lower()
                await s.rollback()
                if attempt == _BOUNDARY_ATTEMPTS or not (
                    "locked" in message or "busy" in message
                ):
                    raise
    raise RuntimeError("unreachable: boundary retry loop exhausted")  # pragma: no cover


async def current_epoch(
    engine: str,
    *,
    session: AsyncSession | None = None,
    session_factory=None,
) -> int:
    """The engine's current epoch: ``COALESCE(MAX(epoch), 1)``.

    Read-only (never commits). No boundary rows yet means epoch 1 — the
    implicit first epoch, also the value ``epoch_clause`` treats as the
    pre-epoch book (it additionally matches NULL for synthetic fixtures).
    """
    _validate_engine(engine)
    async with _session_scope(session, session_factory) as (s, _owned):
        value = await s.scalar(
            select(func.coalesce(func.max(RunEpoch.epoch), 1)).where(
                RunEpoch.engine == engine
            )
        )
    return int(value) if value is not None else 1


def epoch_clause(tbl, cur: int):
    """The epoch filter for history table *tbl* at epoch *cur* (NULL ≡ 1).

    ``tbl`` is normally a history-table class from ``ENGINES``; a Core
    ``Table`` works too, so synthetic/pre-migration fixtures can use it.
    Epoch 1 (or anything <= 1, the "pre-epoch" book) must also match legacy
    rows whose epoch is NULL; a later epoch matches its stamped rows only.
    """
    column = getattr(tbl, "epoch", None)
    if column is None:
        column = tbl.c.epoch
    epoch = int(cur)
    if epoch <= 1:
        return or_(column.is_(None), column == 1)
    return column == epoch


async def has_paper_rows(
    engine: str,
    cur: int,
    *,
    session: AsyncSession | None = None,
    session_factory=None,
) -> bool:
    """True iff epoch *cur* of *engine* holds a record that is NOT provably synthetic.

    The "do not overwrite a forward record" guard, FAIL CLOSED: only
    ``mode == "backfill"`` is knowingly synthetic. A row whose mode is NULL
    (written before stamping existed) or any other/unexpected value is treated
    as PROTECTED — otherwise the first post-upgrade backfill would wipe a real
    legacy forward book whose rows simply predate the stamp. Callers must
    refuse the destructive path when this returns True (the explicit reset /
    new-epoch path remains available).

    Rows outside ``epoch_clause(cur)`` are ignored: another epoch's history is
    not this epoch's to replace.
    """
    _validate_engine(engine)
    async with _session_scope(session, session_factory) as (s, _owned):
        for tbl in ENGINES[engine]:
            found = await s.scalar(
                select(tbl.id)
                .where(
                    or_(tbl.mode.is_(None), tbl.mode != "backfill"),
                    epoch_clause(tbl, cur),
                )
                .limit(1)
            )
            if found is not None:
                return True
    return False


async def record_execution(
    engine: str,
    mode: str,
    config_id: int,
    epoch: int,
    provenance: dict[str, Any],
    *,
    data_asof: datetime | None = None,
    session: AsyncSession | None = None,
    session_factory=None,
) -> int:
    """Append ONE execution row (append-only, NEVER deduped); return its id.

    ``get_or_create`` counts distinct CONFIGURATIONS; this records EXECUTIONS:
    two runs of the identical configuration are two rows, which is what makes
    "how many times did it run, and on which data?" answerable without
    inflating the configuration count. ``provenance`` is the caller's per-run
    cache/instrument provenance summary (provider sources, adjustment bases,
    ingest times, staleness notes) and is stored as canonical JSON —
    DELIBERATELY separate from the config hash, because two runs of one
    configuration can see different data. ``data_asof`` is the market-data
    cut-off the run actually saw (NULL = not declared by this run).

    ``config_id`` must be the id returned by ``get_or_create`` and ``epoch``
    the engine epoch the run executed in (>= 1; 1 is the pre-reset book).
    A caller-owned session is not committed here, so the execution record can
    be written in the same transaction as the rows it explains.
    """
    _validate_engine(engine)
    _validate_mode(mode)
    config_id = _positive_int(config_id, "config_id")
    epoch = _positive_int(epoch, "epoch")
    if data_asof is not None and not isinstance(data_asof, datetime):
        raise TypeError("data_asof must be a datetime or None")
    if data_asof is not None and data_asof.tzinfo is None:
        # Naive would be stored as-is and read back as UTC — a silent
        # mis-statement of when the data was current.
        raise ValueError("data_asof must be timezone-aware")
    content = canonical(provenance)
    async with _session_scope(session, session_factory) as (s, owned):
        execution = RunExecution(
            engine=engine,
            mode=mode,
            epoch=epoch,
            config_id=config_id,
            data_asof=data_asof,
            provenance=content,
        )
        s.add(execution)
        await s.flush()
        new_id = execution.id
        if owned:
            await s.commit()
        return int(new_id)
