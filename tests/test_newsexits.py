"""Regression coverage for the D-promotion review: execution != scoring."""
import asyncio
import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app import daily_core, db, monthly, newsagent, selllayer, sellscore
from app.config import settings
from app.db import (
    DailyCoreAccount,
    DailyCorePosition,
    DailyCoreTrade,
    LlmSellDecision,
    MonthlyAccount,
    MonthlyPosition,
    MonthlyTrade,
    RunEpoch,
)

from .test_selllayer import TestTypeDFill as _Seeds
from .test_selllayer import _add_open_candle, _pending_d_row
from .test_selllayer import mem_db as mem_db

DECIDED = datetime(2026, 2, 2, 5, tzinfo=UTC)
FILL = datetime(2026, 2, 3, 23, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def execution_enabled(monkeypatch):
    monkeypatch.setattr(settings, "sim_news_execute", True)


async def _state(factory):
    async with factory() as s:
        return {
            "monthly": list((await s.scalars(select(MonthlyPosition.ticker))).all()),
            "daily_core": list((await s.scalars(select(DailyCorePosition.ticker))).all()),
            "monthly_trades": list((await s.scalars(select(MonthlyTrade))).all()),
            "core_trades": list((await s.scalars(select(DailyCoreTrade))).all()),
            "rows": list((await s.scalars(select(LlmSellDecision)
                                         .order_by(LlmSellDecision.decided_at))).all()),
            "cash": [(await s.get(model, 1)).cash for model in
                     (MonthlyAccount, DailyCoreAccount) if await s.get(model, 1)],
        }


async def _seed_both():
    await _Seeds()._seed_monthly_position()
    await _Seeds()._seed_core_position(shares=25)
    await _pending_d_row(DECIDED)
    await _add_open_candle("AAA", "2026-02-03", 42)


async def test_both_books_fill_once_with_distinct_trade_ids_and_cash(mem_db):
    await _seed_both()
    before = await _state(mem_db)
    result = await selllayer.run_type_d_fill_step(now=FILL)
    after = await _state(mem_db)
    assert len(result["fills"]) == 2
    assert after["monthly"] == after["daily_core"] == []
    assert len(after["monthly_trades"]) == len(after["core_trades"]) == 1
    assert after["cash"] == pytest.approx([before["cash"][0] + 100 * 42,
                                          before["cash"][1] + 25 * 42])
    row = after["rows"][0]
    audit = json.loads(row.execution)
    assert audit["status"] == "filled"
    for book, trades in (("monthly", after["monthly_trades"]),
                         ("daily_core", after["core_trades"])):
        leg = audit["books"][book]
        assert leg["status"] == "filled" and leg["trade_id"] == trades[0].id
        assert leg["session_date"] == "2026-02-03" and leg["price"] == 42
        assert leg["epoch"] == trades[0].epoch
    assert row.filled_at == FILL  # injectable cycle clock, not wall-clock tests
    assert await selllayer.news_exit_excluded(pd.Timestamp("2026-01-30")) == {"AAA"}
    repeat = await selllayer.run_type_d_fill_step(now=FILL)
    assert repeat["fills"] == []
    assert len((await _state(mem_db))["monthly_trades"]) == 1


async def test_duplicate_is_superseded_and_cannot_sell_month_end_reentry(mem_db):
    await _Seeds()._seed_monthly_position()
    await _pending_d_row(DECIDED, session_date="2026-02-01")
    # Even a same-day newer verdict is retired when the older one fills.
    await _pending_d_row(datetime(2026, 2, 3, 5, tzinfo=UTC), session_date="2026-02-02")
    await _add_open_candle("AAA", "2026-02-03", 42)
    result = await selllayer.run_type_d_fill_step(now=FILL)
    assert len(result["fills"]) == 1 and result["closed"] == 1
    state = await _state(mem_db)
    duplicate = state["rows"][1]
    audit = json.loads(duplicate.execution)
    assert audit["status"] == "superseded"
    assert audit["books"]["monthly"]["by_event_id"] == state["rows"][0].event_id
    async with mem_db() as s:
        s.add(MonthlyPosition(ticker="AAA", shares=10, avg_cost=50,
                              opened_at=datetime(2026, 3, 2, tzinfo=UTC)))
        await s.commit()
    await _add_open_candle("AAA", "2026-03-03", 51)
    assert (await selllayer.run_type_d_fill_step(
        now=datetime(2026, 3, 3, 23, 30, tzinfo=UTC)))["fills"] == []
    assert (await _state(mem_db))["monthly"] == ["AAA"]
    report = await sellscore.score_sell_decisions()
    history = {r["event_id"]: r for r in report["decisions"]}
    assert history[duplicate.event_id]["status"] == "skipped"
    assert report["types"]["D"]["n_closed"] == 1


async def test_pending_verdict_cannot_sell_position_acquired_later(mem_db):
    await _pending_d_row(DECIDED)
    async with mem_db() as s:
        s.add(MonthlyPosition(ticker="AAA", shares=10, avg_cost=50,
                              opened_at=datetime(2026, 2, 3, tzinfo=UTC)))
        await s.commit()
    await _add_open_candle("AAA", "2026-02-03", 51)
    result = await selllayer.run_type_d_fill_step(now=FILL)
    state = await _state(mem_db)
    assert result["fills"] == [] and result["closed"] == 1
    assert state["monthly"] == ["AAA"]
    assert json.loads(state["rows"][0].execution)["books"]["monthly"]["status"] == "cancelled"


async def test_missing_open_freezes_lifecycle_and_replacement_is_not_sold(mem_db):
    await _Seeds()._seed_monthly_position()
    await _pending_d_row(DECIDED)
    assert (await selllayer.run_type_d_fill_step(now=FILL))["pending"] == 1
    async with mem_db() as s:
        position = await s.scalar(select(MonthlyPosition))
        position.opened_at = datetime(2026, 2, 4, tzinfo=UTC)
        await s.commit()
    await _add_open_candle("AAA", "2026-02-04", 51)
    result = await selllayer.run_type_d_fill_step(
        now=datetime(2026, 2, 4, 23, 30, tzinfo=UTC))
    assert result["fills"] == [] and result["closed"] == 1
    assert (await _state(mem_db))["monthly"] == ["AAA"]


async def test_reset_since_decision_cancels_old_verdict(mem_db):
    await _Seeds()._seed_monthly_position()
    await _pending_d_row(DECIDED)
    async with mem_db() as s:
        s.add(RunEpoch(engine="monthly", epoch=2, note="synthetic reset",
                       created_at=datetime(2026, 2, 2, 12, tzinfo=UTC)))
        await s.commit()
    await _add_open_candle("AAA", "2026-02-03", 42)
    result = await selllayer.run_type_d_fill_step(now=FILL)
    assert result["fills"] == [] and result["closed"] == 1
    assert (await _state(mem_db))["monthly"] == ["AAA"]


async def test_consulted_books_bound_execution_scope(mem_db):
    await _seed_both()
    async with mem_db() as s:
        row = await s.scalar(select(LlmSellDecision))
        row.context = json.dumps({"position": {"books": ["daily_core"], "positions": {
            "daily_core": {"entry": {"date": "2026-01-05"}}}}})
        # A pending audit written by the previous context-path bug must not
        # widen the authoritative consulted scope on retry.
        row.execution = json.dumps({"status": "pending", "books": {
            "monthly": {"status": "pending"}, "daily_core": {"status": "pending"}}})
        await s.commit()
    result = await selllayer.run_type_d_fill_step(now=FILL)
    assert [f["book"] for f in result["fills"]] == ["daily-core"]
    state = await _state(mem_db)
    assert state["monthly"] == ["AAA"] and state["daily_core"] == []
    audit = json.loads(state["rows"][0].execution)
    assert audit["books"]["monthly"]["status"] == "cancelled"


@pytest.mark.parametrize("driver", ["manual", "nightly"])
@pytest.mark.parametrize("change", ["none", "add_book", "replace", "reset"])
async def test_real_news_writers_freeze_scope_before_agent_call(
        mem_db, monkeypatch, tmp_path, driver, change):
    """Use the actual context loader and ledger writer, not an invented shape.

    Agent execution is stubbed, but its concurrent book changes occur AFTER
    snapshot capture and BEFORE the verdict's frozen timestamp.
    """
    monkeypatch.setattr(newsagent, "Session", mem_db)
    monkeypatch.setattr(settings, "searxng_mcp_url", "http://synthetic.invalid/mcp")
    opened = datetime(2026, 2, 2, 17, 58, tzinfo=UTC)
    replaced = datetime(2026, 2, 2, 17, 59, tzinfo=UTC)
    decided = datetime(2026, 2, 2, 18, tzinfo=UTC)
    if change == "add_book":
        await _Seeds()._seed_core_position()
    else:
        await _Seeds()._seed_monthly_position()
        async with mem_db() as s:
            pos = await s.scalar(select(MonthlyPosition))
            pos.opened_at = opened
            await s.commit()
    writer = newsagent._record_ledger_row

    async def timed_writer(**kwargs):
        return await writer(**kwargs, decided_at=decided)

    async def agent(context, *, event_id, **kwargs):
        # The lifecycle snapshot is NOT sent to the frozen research prompt.
        assert "execution_scope" not in context
        assert "position_opened_at" not in newsagent.render_prompt(context)[1]["content"]
        async with mem_db() as s:
            if change == "add_book":
                assert context["books"] == ["daily_core"]
                s.add(MonthlyPosition(ticker="AAA", shares=10, avg_cost=40,
                                      opened_at=replaced))
            elif change == "replace":
                pos = await s.scalar(select(MonthlyPosition))
                await s.delete(pos)
                await s.flush()
                s.add(MonthlyPosition(ticker="AAA", shares=100, avg_cost=40,
                                      opened_at=replaced))
            elif change == "reset":
                s.add(RunEpoch(engine="monthly", epoch=2,
                               created_at=replaced, note="synthetic mid-consult reset"))
            await s.commit()
        return newsagent._build_record(context, event_id, {
            "decision": "EXIT", "parsed": True, "fail_safe": False,
            "rationale": "synthetic integration verdict", "citations": []},
            trace=[], turns=1, tool_calls=0, backend="test", model="test",
            error=None, error_class=None, latency_ms=1)

    monkeypatch.setattr(newsagent, "_record_ledger_row", timed_writer)
    monkeypatch.setattr(newsagent, "run_agent", agent)
    if driver == "manual":
        result = await newsagent.run_manual_consult(
            "AAA", as_of="2026-02-02", out_dir=str(tmp_path))
        assert result["ok"] is True
    else:
        result = await newsagent.run_news_scan(as_of="2026-02-02", out_dir=str(tmp_path))
        assert result["n_ledger_rows"] == 1 and result["n_errors"] == 0
    before = await _state(mem_db)
    context = json.loads(before["rows"][0].context)
    assert "position" in context and "execution_scope" in context
    scope = context["execution_scope"]["books"]
    if change != "add_book":
        assert scope["monthly"] == {"epoch": 1, "position_opened_at": opened.isoformat()}
    await _add_open_candle("AAA", "2026-02-03", 42)
    filled = await selllayer.run_type_d_fill_step(now=FILL)
    state = await _state(mem_db)
    if change in ("replace", "reset"):
        assert filled["fills"] == [] and filled["closed"] == 1
        assert state["monthly"] == ["AAA"]
    else:
        expected = "daily-core" if change == "add_book" else "monthly"
        assert [f["book"] for f in filled["fills"]] == [expected]
        assert state["monthly"] == (["AAA"] if change == "add_book" else [])


async def test_date_only_legacy_same_day_position_is_not_authorized(mem_db):
    await _Seeds()._seed_monthly_position()
    await _pending_d_row(datetime(2026, 2, 2, 18, tzinfo=UTC))
    async with mem_db() as s:
        row = await s.scalar(select(LlmSellDecision))
        row.context = json.dumps({"position": {"books": ["monthly"], "positions": {
            "monthly": {"entry": {"date": "2026-02-02"}}}}})
        pos = await s.scalar(select(MonthlyPosition))
        pos.opened_at = datetime(2026, 2, 2, 17, 59, tzinfo=UTC)
        await s.commit()
    await _add_open_candle("AAA", "2026-02-03", 42)
    result = await selllayer.run_type_d_fill_step(now=FILL)
    assert result["fills"] == [] and result["closed"] == 1
    assert (await _state(mem_db))["monthly"] == ["AAA"]


async def test_legacy_wrapped_entry_day_mismatch_closes_without_trade(mem_db):
    await _Seeds()._seed_monthly_position()
    await _pending_d_row(DECIDED)
    async with mem_db() as s:
        row = await s.scalar(select(LlmSellDecision))
        row.context = json.dumps({"position": {"books": ["monthly"], "positions": {
            "monthly": {"entry": {"date": "2025-12-01"}}}}})
        await s.commit()
    await _add_open_candle("AAA", "2026-02-03", 42)
    result = await selllayer.run_type_d_fill_step(now=FILL)
    assert result["fills"] == [] and result["closed"] == 1


@pytest.mark.parametrize("failure", ["ledger_commit", "second_book_refused", "provenance"])
async def test_trade_and_audit_rollback_together_then_retry(mem_db, monkeypatch, failure):
    await _seed_both()
    before = await _state(mem_db)
    with monkeypatch.context() as patch:
        if failure == "ledger_commit":
            commit = AsyncSession.commit

            async def fail_commit(s):
                if any(isinstance(obj, LlmSellDecision) and obj.execution for obj in s.dirty):
                    raise RuntimeError("synthetic audit commit failure")
                return await commit(s)
            patch.setattr(AsyncSession, "commit", fail_commit)
        elif failure == "second_book_refused":
            async def refuse(*args, **kwargs):
                return None
            patch.setattr(daily_core, "_exec_sell_in_session", refuse)
        else:
            async def fail_provenance(*args, **kwargs):
                raise RuntimeError("synthetic provenance failure")
            patch.setattr(monthly, "_record_run", fail_provenance)
        result = await selllayer.run_type_d_fill_step(now=FILL)
    assert "error" in result
    state = await _state(mem_db)
    assert state["monthly"] == state["daily_core"] == ["AAA"]
    assert state["monthly_trades"] == state["core_trades"] == []
    assert state["cash"] == before["cash"]
    assert state["rows"][0].execution is None and state["rows"][0].filled_at is None
    assert await selllayer.news_exit_excluded(pd.Timestamp("2026-01-30")) == set()
    assert len((await selllayer.run_type_d_fill_step(now=FILL))["fills"]) == 2


async def test_concurrent_fill_calls_are_idempotent(mem_db):
    await _seed_both()
    results = await asyncio.gather(selllayer.run_type_d_fill_step(now=FILL),
                                   selllayer.run_type_d_fill_step(now=FILL))
    assert sum(len(r["fills"]) for r in results) == 2
    state = await _state(mem_db)
    assert len(state["monthly_trades"]) == len(state["core_trades"]) == 1


async def test_fill_visible_immediately_then_scores_once_after_window(mem_db):
    await _seed_both()
    # A complete Jan month-end allows the scorer to resolve the governing ME.
    await _add_open_candle("AAA", "2026-01-30", 40)
    await selllayer.run_type_d_fill_step(now=FILL)
    # A delayed bookkeeping timestamp must not shift the economic session.
    async with mem_db() as s:
        ledger = await s.scalar(select(LlmSellDecision))
        ledger.filled_at = datetime(2026, 2, 4, 0, 5, tzinfo=UTC)
        await s.commit()
    report = await sellscore.score_sell_decisions()
    row = report["decisions"][0]
    assert row["status"] == "pending" and row["execution"]["status"] == "filled"
    assert row["window"]["start"] == "2026-02-03"
    assert report["types"]["D"]["n_filled"] == 1  # not twice for two books
    assert {b: leg["price"] for b, leg in row["execution"]["books"].items()} == {
        "monthly": 42, "daily_core": 42}
    await _add_open_candle("AAA", "2026-02-27", 44)
    await _add_open_candle("AAA", "2026-03-02", 45)
    report = await sellscore.score_sell_decisions()
    assert report["types"]["D"]["n_decided"] == 1
    assert report["types"]["D"]["n_pending"] == 0
    event = report["types"]["D"]["events"][0]
    assert event["window"]["end"] == "2026-03-02"
    assert event["delta"] < 0  # holding rose; early exit underperformed


async def test_legacy_filled_verdict_is_never_reexecuted(mem_db):
    await _seed_both()
    async with mem_db() as s:
        row = await s.scalar(select(LlmSellDecision))
        row.filled_at, row.price, row.shares, row.cash_after = FILL, 42, 100, 4200
        await s.commit()
    assert (await selllayer.run_type_d_fill_step(now=FILL))["fills"] == []
    report = await sellscore.score_sell_decisions()
    assert report["decisions"][0]["execution"]["status"] == "filled"
    assert "legacy" in report["decisions"][0]["execution"]["note"]


async def test_nightly_fills_even_when_scan_and_rebuild_are_skipped(mem_db, monkeypatch):
    await _seed_both()
    monkeypatch.setattr(settings, "sim_monthly_enabled", True)
    monkeypatch.setattr(settings, "sim_monthly_llm_sells", False)
    monkeypatch.setattr(settings, "sim_news_scan_weekday", 0)  # Tuesday fill, Monday scan
    async def no_scan(**kwargs):
        pytest.fail("the cadence gate should skip research, not fills")
    async def no_rebuild(**kwargs):
        return {"skipped": True, "reason": "no pending month-end rebuild"}
    monkeypatch.setattr(newsagent, "run_news_scan", no_scan)
    monkeypatch.setattr(monthly, "run_rebalance", no_rebuild)
    result = await monthly.run_monthly_cycle(now=FILL)
    assert result["skipped"] is True
    state = await _state(mem_db)
    assert state["monthly"] == state["daily_core"] == []
    assert len(state["monthly_trades"]) == len(state["core_trades"]) == 1


async def test_additive_audit_migration_is_idempotent_and_preserves_verdict(mem_db):
    await _pending_d_row(DECIDED)
    async with mem_db() as s:
        await s.execute(text("ALTER TABLE llm_sell_decisions DROP COLUMN execution"))
        await s.commit()
    engine = mem_db.kw["bind"]
    await db._migrate_engine(engine)
    await db._migrate_engine(engine)
    async with mem_db() as s:
        row = await s.scalar(select(LlmSellDecision))
        assert row.event_id == "D:AAA:2026-02-01:m"
        assert row.decision == "EXIT" and row.decided_at == DECIDED
        assert row.execution is None and row.filled_at is None
