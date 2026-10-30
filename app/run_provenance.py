"""Per-run DATA provenance: what the cache actually held for a run's scope.

This is the DATA half of Phase 2's run provenance and is deliberately separate
from the immutable configuration half (``app.run_config``):

* the configuration records what a run was TOLD to do (settings, universe
  membership, mode), frozen before it starts;
* :func:`summarize` records what the cached market data ACTUALLY looked like
  when the run observed it — which provider sources and adjustment bases the
  rows carry, when they were ingested, and the time span they cover.

It is an observation, NOT an immutable data snapshot and NOT a content-replay
guarantee. ``Candle.source`` / ``adjustment_basis`` / ``ingested_at`` describe
the rows CURRENTLY stored: a later provider pull, correction, restatement or
corporate-action re-adjustment can revise the same bars/facts, and a provider
can revise history without changing our row identity. The summary says
"at ``asof`` the cache looked like this", nothing more. There is no content
hash of the OHLCV or of a fact value here, by design — that guarantee would
need data we do not persist.

HONESTY RULES (no provenance is ever assumed):

* NULL source / adjustment basis / ingestion time is reported as
  ``"unknown"`` (and counted in ``*_unknown``), never as a provider default.
  Rows cached before provenance tracking are therefore visibly unknown.
* Unknown members mean an UNKNOWN SCOPE: no query is made and the summary is
  empty. Missing membership is never "helpfully" widened to the whole cache,
  because a full-cache scan would report data the run never saw.
* Required FX pairs are derived from the membership through
  :mod:`app.instruments` (the single suffix metadata owner). A USD-native
  listing needs no pair, so USD-only universes contribute no ``EURUSD=X``
  guess; unresolved foreign suffixes are listed explicitly so the gap is
  visible instead of silently assumed.

SCOPE AND TIME:

* Only candles for ``members + required FX pairs`` are counted.
* Only fundamentals for the members themselves are counted (FX pairs carry no
  statements).
* ``Candle.timestamp <= asof``; ``Fundamental.filed <= asof``'s UTC calendar
  day. ``asof`` defaults to ``datetime.now(UTC)``; a naive value is read as
  UTC and an offset-aware value is converted to UTC. ``ingested_at`` is the
  time WE observed the row, not an event time, so it is intentionally NOT
  upper-bounded by ``asof`` (a run executed today may read bars ingested
  today for a historical date).

CONFIG SHAPES (the membership content has exactly ONE members key):

1. The engines' shape — ``config["universe"]`` is a mapping carrying
   ``"universe_members"`` (daily_core / monthly) or the membership sits at the
   top level as ``config["universe_members"]`` (sim). ``"members"`` is also
   accepted as the legacy spelling. Lookup order is:
   ``universe.universe_members``, ``universe.members``,
   ``universe_members``, ``members`` — the first key PRESENT wins.
2. A flat config with ``config["members"]`` or
   ``config["universe_members"]`` plus optional ``"members_hash"``.

``members_hash`` is taken from the config when it supplies one (nested
``universe.members_hash`` or top-level) and reported with
``members_hash_source="config"``; otherwise it is computed as the sha256 of
``",".join(sorted unique members)`` — the same convention every engine's
universe snapshot uses — with ``members_hash_source="computed"``. An optional
``"config_hash"`` / ``"hash"`` on the config is passed through untouched; no
settings value is copied, so no secret can leak into the stored summary.

A scope that cannot be resolved is reported, never guessed: ``known=False``
plus ``reason`` one of ``"members_missing"`` (no members key),
``"members_unresolved"`` (the engines' explicit None), ``"members_empty"``
(no members), or ``"members_invalid"`` (not a list of non-blank strings). The
summary then has zero rows and makes no query. A KNOWN scope stores only the
membership hash / count / FX pairs — not the member list itself — because the
immutable configuration (referenced by the execution row) already owns it.

TRANSACTIONS: like ``app.run_config``, every call accepts a caller-owned
``AsyncSession`` (``session=``) or a ``session_factory=``. A caller-owned
session is used as-is: this module issues SELECTs only and never commits,
closes or rolls back it. When no session is passed, one is opened from the
factory (default ``db.Session``, resolved at call time) for read-only
aggregates and never committed.

IMPORT CONTRACT: ``app.db``, the pure ``app.instruments`` metadata module, the
SQLAlchemy core and the stdlib ONLY. Never import an engine module
(daily_core / monthly / sim / optimize): those import THIS module lazily, and
a module-level cycle would be fatal.
"""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import db as _db
from . import instruments
from .db import Candle, Fundamental

# One spelling for "the cache did not record this provenance".
_UNKNOWN = "unknown"

# Membership lookups, in contract precedence order (see the module docstring).
_MEMBERS_KEYS: tuple[str, ...] = ("universe_members", "members")


def _utc(value: datetime) -> datetime:
    """*value* as tz-aware UTC; a naive value is read as already UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _text(value: Any) -> str | None:
    """A non-empty stripped string, else None (config values are untrusted)."""
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _members_digest(members: list[str]) -> str:
    """sha256 of the sorted-join membership — the engines' hash convention."""
    return hashlib.sha256(",".join(members).encode("utf-8")).hexdigest()


def _first_present(mapping: dict, keys: tuple[str, ...]) -> tuple[bool, Any]:
    """``(present, value)`` for the first key in *keys* present in *mapping*."""
    for key in keys:
        if key in mapping:
            return True, mapping[key]
    return False, None


def _resolve_scope(config: dict) -> tuple[dict, list[str]]:
    """Resolve the query scope from *config*; return ``(scope, members)``.

    Absent / None / empty / malformed membership yields ``known=False`` with
    an explicit reason and NO members; the caller must then return an empty
    summary without querying the cache.
    """
    block = config.get("universe")
    if not isinstance(block, dict):
        block = {}

    name = (
        _text(block.get("name"))
        or _text(config.get("universe_name"))
        or (_text(config.get("universe")) if not isinstance(config.get("universe"), dict) else None)
    )
    provided_hash = _text(block.get("members_hash")) or _text(config.get("members_hash"))
    config_hash = _text(config.get("config_hash")) or _text(config.get("hash"))

    found, raw = _first_present(block, _MEMBERS_KEYS)
    if not found:
        found, raw = _first_present(config, _MEMBERS_KEYS)

    def unknown(reason: str) -> tuple[dict, list[str]]:
        scope = {
            "known": False,
            "reason": reason,
            "universe": name,
            "member_count": 0,
            "members_hash": provided_hash,
            "members_hash_source": "config" if provided_hash else None,
            "fx_pairs": [],
            "unresolved_fx_members": [],
            "ticker_count": 0,
            "config_hash": config_hash,
        }
        return scope, []

    if not found:
        return unknown("members_missing")
    if raw is None:
        return unknown("members_unresolved")
    if not isinstance(raw, (list, tuple)):
        return unknown("members_invalid")

    cleaned: list[str] = []
    for item in raw:
        ticker = _text(item)
        if ticker is None:
            return unknown("members_invalid")
        cleaned.append(ticker)
    members = sorted(set(cleaned))
    if not members:
        return unknown("members_empty")

    fx_pairs: set[str] = set()
    unresolved: list[str] = []
    for ticker in members:
        pair_mode = instruments.suffix_fx(ticker)
        if pair_mode is not None:
            fx_pairs.add(pair_mode[0])
        elif not instruments.is_usd_native(ticker):
            # A foreign-looking dot suffix with no explicit FX mapping: the
            # listing currency is unresolved. Record it; never guess USD/EUR.
            unresolved.append(ticker)

    computed_hash = _members_digest(members)
    scope = {
        "known": True,
        "reason": None,
        "universe": name,
        "member_count": len(members),
        "members_hash": provided_hash or computed_hash,
        "members_hash_source": "config" if provided_hash else "computed",
        "fx_pairs": sorted(fx_pairs),
        "unresolved_fx_members": sorted(unresolved),
        "ticker_count": len(set(members) | fx_pairs),
        "config_hash": config_hash,
    }
    return scope, members


def _iso(value: Any) -> str | None:
    """ISO-8601 UTC string for a stored datetime; None stays None.

    SQLite returns datetimes through ``db.DateTime`` (tz-aware UTC), but the
    fallback to strings keeps an aggregate result JSON-safe if a dialect ever
    hands one back raw.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    return str(value)


def _min_datetime(current: datetime | None, candidate: Any) -> datetime | None:
    """Smaller of two UTC datetimes; non-datetime candidates are ignored."""
    if not isinstance(candidate, datetime):
        return current
    candidate = _utc(candidate)
    if current is None or candidate < current:
        return candidate
    return current


def _max_datetime(current: datetime | None, candidate: Any) -> datetime | None:
    """Larger of two UTC datetimes; non-datetime candidates are ignored."""
    if not isinstance(candidate, datetime):
        return current
    candidate = _utc(candidate)
    if current is None or candidate > current:
        return candidate
    return current


def _empty_candle_summary() -> dict:
    """The honest empty candle summary (unknown scope / no cached rows)."""
    return {
        "rows": 0,
        "by_source": [],
        "by_adjustment_basis": [],
        "by_source_basis": [],
        "source_known": 0,
        "source_unknown": 0,
        "basis_known": 0,
        "basis_unknown": 0,
        "min_timestamp": None,
        "max_timestamp": None,
        "ingested_at_known": 0,
        "ingested_at_unknown": 0,
        "min_ingested_at": None,
        "max_ingested_at": None,
    }


def _empty_fundamental_summary() -> dict:
    """The honest empty fundamental summary (unknown scope / no cached rows)."""
    return {
        "rows": 0,
        "by_source": [],
        "source_known": 0,
        "source_unknown": 0,
        "min_filed": None,
        "max_filed": None,
    }


def _candle_summary(rows) -> dict:
    """Aggregate ``(source, basis, count, min_ts, max_ts, min_ing, max_ing, null_ing)``."""
    summary = _empty_candle_summary()
    by_source: dict[str, int] = {}
    by_basis: dict[str, int] = {}
    by_pair: dict[tuple[str, str], int] = {}
    total = 0
    ingested_unknown = 0
    min_ts = max_ts = None
    min_ing = max_ing = None

    for source, basis, count, row_min_ts, row_max_ts, row_min_ing, row_max_ing, null_ing in rows:
        count = int(count or 0)
        total += count
        source_label = _UNKNOWN if source is None else str(source)
        basis_label = _UNKNOWN if basis is None else str(basis)
        by_source[source_label] = by_source.get(source_label, 0) + count
        by_basis[basis_label] = by_basis.get(basis_label, 0) + count
        by_pair[(source_label, basis_label)] = by_pair.get((source_label, basis_label), 0) + count
        ingested_unknown += int(null_ing or 0)
        min_ts = _min_datetime(min_ts, row_min_ts)
        max_ts = _max_datetime(max_ts, row_max_ts)
        min_ing = _min_datetime(min_ing, row_min_ing)
        max_ing = _max_datetime(max_ing, row_max_ing)

    source_unknown = by_source.get(_UNKNOWN, 0)
    basis_unknown = by_basis.get(_UNKNOWN, 0)
    summary.update(
        {
            "rows": total,
            "by_source": [
                {"source": label, "count": count} for label, count in sorted(by_source.items())
            ],
            "by_adjustment_basis": [
                {"basis": label, "count": count} for label, count in sorted(by_basis.items())
            ],
            "by_source_basis": [
                {"source": source, "basis": basis, "count": count}
                for (source, basis), count in sorted(by_pair.items())
            ],
            "source_known": total - source_unknown,
            "source_unknown": source_unknown,
            "basis_known": total - basis_unknown,
            "basis_unknown": basis_unknown,
            "min_timestamp": _iso(min_ts),
            "max_timestamp": _iso(max_ts),
            "ingested_at_known": total - ingested_unknown,
            "ingested_at_unknown": ingested_unknown,
            "min_ingested_at": _iso(min_ing),
            "max_ingested_at": _iso(max_ing),
        }
    )
    return summary


def _fundamental_summary(rows) -> dict:
    """Aggregate ``(source, count, min_filed, max_filed)`` per source."""
    summary = _empty_fundamental_summary()
    by_source: dict[str, int] = {}
    total = 0
    min_filed: str | None = None
    max_filed: str | None = None

    for source, count, row_min_filed, row_max_filed in rows:
        count = int(count or 0)
        total += count
        source_label = _UNKNOWN if source is None else str(source)
        by_source[source_label] = by_source.get(source_label, 0) + count
        if isinstance(row_min_filed, str) and (min_filed is None or row_min_filed < min_filed):
            min_filed = row_min_filed
        if isinstance(row_max_filed, str) and (max_filed is None or row_max_filed > max_filed):
            max_filed = row_max_filed

    source_unknown = by_source.get(_UNKNOWN, 0)
    summary.update(
        {
            "rows": total,
            "by_source": [
                {"source": label, "count": count} for label, count in sorted(by_source.items())
            ],
            "source_known": total - source_unknown,
            "source_unknown": source_unknown,
            "min_filed": min_filed,
            "max_filed": max_filed,
        }
    )
    return summary


@asynccontextmanager
async def _session_scope(session: AsyncSession | None, session_factory):
    """Yield a session; a caller-owned one is never committed or closed."""
    if session is not None:
        yield session
        return
    factory = session_factory if session_factory is not None else _db.Session
    async with factory() as owned:
        yield owned


async def _candle_rows(session: AsyncSession, tickers: list[str], upper: datetime):
    """Grouped candle provenance for *tickers* with ``timestamp <= upper``."""
    stmt = (
        select(
            Candle.source,
            Candle.adjustment_basis,
            func.count(),
            func.min(Candle.timestamp),
            func.max(Candle.timestamp),
            func.min(Candle.ingested_at),
            func.max(Candle.ingested_at),
            func.sum(case((Candle.ingested_at.is_(None), 1), else_=0)),
        )
        .where(Candle.ticker.in_(tickers), Candle.timestamp <= upper)
        .group_by(Candle.source, Candle.adjustment_basis)
    )
    return list((await session.execute(stmt)).all())


async def _fundamental_rows(session: AsyncSession, members: list[str], filed_by: str):
    """Grouped fundamental provenance for *members* filed on/before *filed_by*."""
    stmt = (
        select(
            Fundamental.source,
            func.count(),
            func.min(Fundamental.filed),
            func.max(Fundamental.filed),
        )
        .where(Fundamental.ticker.in_(members), Fundamental.filed <= filed_by)
        .group_by(Fundamental.source)
    )
    return list((await session.execute(stmt)).all())


async def summarize(
    config: dict,
    *,
    asof: datetime | None = None,
    session: AsyncSession | None = None,
    session_factory=None,
) -> dict:
    """Summarize the cached data provenance a run's scope could read.

    ``config`` is the run's immutable configuration (the engines' shape; see
    the module docstring). ``asof`` is the run's clock — the upper bound of
    observable candle timestamps and the UTC calendar day for fundamental
    filed dates — defaulting to ``datetime.now(UTC)``. ``session`` is a
    caller-owned session (never committed); otherwise ``session_factory``
    (default ``db.Session``) opens a read-only one.

    Returns a JSON-safe dict: ``asof`` (ISO), ``config_hash``,
    ``members_hash``, a ``scope`` block, and per-table ``candles`` /
    ``fundamentals`` summaries. Counts are grouped by source and adjustment
    basis with NULL reported as ``"unknown"``; min/max stored timestamps and
    fetch times are included, and unknown provenance is counted, never
    assumed. An unknown / missing / empty membership returns
    ``scope.known == False`` and an empty summary WITHOUT touching the cache.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    if asof is not None and not isinstance(asof, datetime):
        raise TypeError("asof must be a datetime or None")

    clock = _utc(asof) if asof is not None else datetime.now(UTC)
    scope, members = _resolve_scope(config)

    result = {
        "asof": clock.isoformat(),
        "config_hash": scope["config_hash"],
        "members_hash": scope["members_hash"],
        "scope": scope,
        "candles": _empty_candle_summary(),
        "fundamentals": _empty_fundamental_summary(),
    }
    if not scope["known"]:
        return result

    tickers = sorted(set(members) | set(scope["fx_pairs"]))
    # SQLite stores naive datetimes; the TypeDecorator re-attaches UTC on read.
    # Normalizing the bound to naive UTC keeps the comparison unambiguous (an
    # aware offset would otherwise be formatted with its LOCAL fields).
    upper = clock.replace(tzinfo=None)
    filed_by = clock.date().isoformat()

    async with _session_scope(session, session_factory) as s:
        result["candles"] = _candle_summary(await _candle_rows(s, tickers, upper))
        result["fundamentals"] = _fundamental_summary(await _fundamental_rows(s, members, filed_by))
    return result
