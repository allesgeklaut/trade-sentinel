"""Atomic, position-lifecycle-bound type-D exits on the two paper books.

The immutable verdict is name-level; its execution audit is per book. All
eligible legs of one verdict commit together with that audit and the frozen
fill summary. This module has no research calls and no Savings authority.
"""
from __future__ import annotations

import json
import logging

import pandas as pd
from sqlalchemy import select, text

from . import db
from .db import (DailyCoreAccount, DailyCorePosition, LlmSellDecision,
                 MonthlyAccount, MonthlyPosition, RunEpoch)

logger = logging.getLogger("trade_sentinel.newsexits")


def timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def audit_json(value: str | None) -> dict:
    obj = json.loads(value or "{}")
    if not isinstance(obj, dict):
        raise ValueError("news execution/context must be a JSON object")
    return obj


async def fill_pending(now=None) -> dict:
    from . import daily_core, monthly

    clock = timestamp(now or monthly._now())
    session = clock.tz_localize(None).normalize()
    day = session.strftime("%Y-%m-%d")
    async with db.Session() as s:
        rows = (await s.scalars(select(LlmSellDecision).where(
            LlmSellDecision.engine == "monthly", LlmSellDecision.event_type == "D",
            LlmSellDecision.decision == "EXIT", LlmSellDecision.filled_at.is_(None))
            .order_by(LlmSellDecision.decided_at, LlmSellDecision.id))).all()
    rows = [r for r in rows if audit_json(r.execution).get("status", "pending") == "pending"]
    result = {"fills": [], "as_of": day, "pending": 0, "closed": 0}
    if not rows:
        return result
    opens = await monthly._open_usd_map(sorted({r.ticker for r in rows}), session)
    engines = {"monthly": (monthly, MonthlyPosition, MonthlyAccount),
               "daily_core": (daily_core, DailyCorePosition, DailyCoreAccount)}
    # Fixed lock order, shared with the public book writers. Never call the
    # lock-taking executor wrappers recursively from inside these locks.
    async with monthly._state_write_lock, daily_core._state_write_lock:
        stamps = {b: await mod._paper_stamp(record=False)
                  for b, (mod, _, _) in engines.items()}
        for candidate in rows:
            fills = []
            async with db.Session() as s:
                # Serialize distinct processes too, BEFORE reading positions
                # or decisions. A later ledger/provenance failure rolls back
                # ALL book trades, cash changes and position deletions.
                await s.execute(text("BEGIN IMMEDIATE"))
                row = await s.get(LlmSellDecision, candidate.id)
                audit = audit_json(row.execution)
                if row.filled_at is not None or audit.get("status", "pending") != "pending":
                    continue
                envelope = audit_json(row.context)
                # The real writers wrap the research context in "position".
                # Direct contexts remain readable for older/imported records.
                context = envelope.get("position", envelope)
                scope = envelope.get("execution_scope")
                scoped_books = scope["books"] if scope is not None else None
                decided = timestamp(row.decided_at)
                prior = (await s.scalars(select(LlmSellDecision).where(
                    LlmSellDecision.event_type == "D", LlmSellDecision.ticker == row.ticker,
                    LlmSellDecision.filled_at.is_not(None),
                    LlmSellDecision.decided_at <= row.decided_at,
                    LlmSellDecision.filled_at >= row.decided_at))).all()
                positions = {b: await s.scalar(select(model).where(model.ticker == row.ticker))
                             for b, (_, model, _) in engines.items()}
                books = scoped_books if scoped_books is not None else context.get("books")
                if books is None:
                    # Legacy context-free rows retain only conservative
                    # pre-decision-day authority, never a same-day acquisition.
                    books = [b for b, p in positions.items() if p is not None]
                    books += [b for p in prior for b in audit_json(p.execution).get("books", {})]
                allowed = {b for b in books if b in engines}
                targets = audit.get("books", {})
                for book in sorted(allowed, key=lambda b: b != "monthly"):
                    targets.setdefault(book, {"status": "pending"})
                for book in sorted(targets, key=lambda b: (b != "monthly", b)):
                    leg = targets[book]
                    if leg["status"] != "pending":
                        continue
                    # Old pending audits may include an unconsulted book from
                    # the former context-path bug. They cannot widen scope.
                    if book not in allowed:
                        leg.update(status="cancelled", reason="book was not in the consulted scope")
                        continue
                    mod, _, account_model = engines[book]
                    stamp = stamps[book]
                    expected = scoped_books[book] if scoped_books is not None else {}
                    pos = positions[book]
                    previous = next((p for p in prior if
                        audit_json(p.execution).get("books", {}).get(book, {}).get("status") == "filled"
                        and audit_json(p.execution)["books"][book].get("epoch") == stamp["epoch"]), None)
                    if previous is not None:
                        leg.update(status="superseded", reason="same position already exited",
                                   by_event_id=previous.event_id)
                        continue
                    reset = await s.scalar(select(RunEpoch.id).where(
                        RunEpoch.engine == mod._ENGINE, RunEpoch.created_at > row.decided_at).limit(1))
                    if reset is not None or (leg.get("epoch") is not None
                                             and leg["epoch"] != stamp["epoch"]) or (
                            scoped_books is not None and expected["epoch"] != stamp["epoch"]):
                        leg.update(status="cancelled", reason="book epoch changed since decision")
                        continue
                    if pos is None:
                        leg.update(status="not_held", reason="consulted position is no longer held")
                        continue
                    opened = timestamp(pos.opened_at)
                    entry_day = context.get("positions", {}).get(book, {}).get("entry", {}).get("date")
                    if (opened > decided
                            or (scoped_books is not None
                                and timestamp(expected["position_opened_at"]) != opened)
                            or (scoped_books is None and opened.date() == decided.date())
                            or (leg.get("position_opened_at") is not None
                                and leg["position_opened_at"] != opened.isoformat())
                            or (entry_day is not None and entry_day != opened.date().isoformat())):
                        leg.update(status="cancelled", reason="position replaced since decision")
                        continue
                    leg.update(position_opened_at=opened.isoformat(), epoch=stamp["epoch"])
                    if decided.date() >= clock.date():
                        leg["reason"] = "waiting for a session after the decision date"
                        continue
                    px = opens.get(row.ticker)
                    if not px:
                        leg["reason"] = "no executable exact-date open this session"
                        continue
                    reason = ("material-news exit" if book == "monthly"
                              else "daily-core: material-news exit")
                    trade = await mod._exec_sell_in_session(s, row.ticker, px, reason, stamp)
                    if trade is None:
                        raise RuntimeError(f"D {row.event_id}: {book} sell refused; rolled back")
                    acc = await s.get(account_model, 1)
                    leg.update(status="filled", session_date=day, filled_at=clock.isoformat(),
                               trade_id=trade["trade_id"], shares=trade["shares"], price=trade["price"],
                               cash_after=float(acc.cash))
                    leg.pop("reason", None)
                    # The name-level scorer counts the verdict ONCE. Preserve
                    # the primary fill summary; every book is in the audit.
                    if row.filled_at is None:
                        row.filled_at = clock.to_pydatetime()
                        row.shares = float(trade["shares"])
                        row.price = float(trade["price"])
                        row.cash_after = float(acc.cash)
                    await mod._record_run(stamp, mod.current_config(), session=s,
                                          asof=clock.to_pydatetime())
                    fills.append({"ticker": row.ticker, "price": float(px),
                                  "book": book.replace("_", "-"), "event_id": row.event_id})
                states = {leg["status"] for leg in targets.values()}
                state = ("pending" if "pending" in states else "filled" if "filled" in states
                         else "superseded" if "superseded" in states else "not_executed")
                row.execution = json.dumps({"status": state, "books": targets}, sort_keys=True,
                                           allow_nan=False)
                await s.commit()
            # Do not report a fill until its complete transaction committed.
            result["fills"].extend(fills)
            result["pending"] += state == "pending"
            result["closed"] += state not in ("pending", "filled")
            for fill in fills:
                logger.info("D EXIT %s filled at %s open %.2f (%s)",
                            fill["ticker"], day, fill["price"], fill["book"])
    return result
