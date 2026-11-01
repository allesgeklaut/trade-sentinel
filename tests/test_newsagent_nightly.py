"""Offline behavior tests for the type-D material-news agent's NIGHTLY
wiring: ``run_news_scan`` (the scan core extracted from the ``news`` CLI
command, called by ``monthly.run_monthly_cycle``) and the sellscore's
shadow handling of D rows.

Zero network: the MCP client and the LLM turn are the fakes from
tests/test_newsagent.py, injected through run_news_scan's pass-through of
run_agent's injection points; the DB is in-memory SQLite (the temp-DB
fixture pattern from test_selllayer.py / test_monthly.py).

Covered:
  (a) armed happy path: exactly ONE D ledger row — event_type "D", engine
      "monthly", parsed, not fail-safe, ALL fill columns NULL (the flagship
      shadow property), news_source "searxng-mcp", prompt_version
      "sellshadow/1-d", temperature 0.6, non-empty trace;
  (b) scan-time dedupe: a rerun for the same night adds no row
      (n_deduped == 1);
  (c) a fail-safe row never silences: a rerun after a fail-safe consults
      again (a new row is appended);
  (d) knob-off: searxng_mcp_url="" -> status "disabled", zero rows, no
      exception; no stored sessions -> status "no-sessions";
  (e) per-name error isolation: one name's failure is collected
      (n_errors == 1) while the other name still gets its row — nothing
      escapes the scan;
  (f) sellscore: with one D row + one A row, the A stats and the overall
      block are byte-identical to the A-only run, n_d_shadow == 1, the D
      entry appears in the history with status "shadow" and is not in A's
      rows/scored/overall;
  (g) the wired nightly: run_monthly_cycle calls the C step and then the D
      step (declared order C-then-D), a raising D step does not break the
      cycle (the rebuild still runs), and D is NOT gated on
      sim_monthly_llm_sells.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import monthly, newsagent, selllayer, sellscore, sim
from app.config import settings
from app.db import (
    Base,
    Candle,
    LlmSellDecision,
    MonthlyAccount,
    MonthlyPosition,
    SimAccount,
    SimTrade,
)
from tests.test_newsagent import _HOLD, FakeMcp, _tool_call, _turn, transport_turns

AS_OF = "2026-03-06"
NOW = datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures / helpers (the temp-DB pattern from test_selllayer.py/test_monthly.py
# + the fake patterns from test_newsagent.py)
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite for every engine the scan touches. newsagent binds
    ``Session`` at import time, so patch it on the newsagent module too
    (app.db / monthly / sim / fundamentals / edgar resolve it at call time,
    which covers selllayer's append-only recorder)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod
    import app.edgar as edgar_mod
    import app.fundamentals as fund_mod
    monkeypatch.setattr(newsagent, "Session", session_factory)
    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(monthly, "Session", session_factory)
    monkeypatch.setattr(sim, "Session", session_factory)
    monkeypatch.setattr(fund_mod, "Session", session_factory)
    monkeypatch.setattr(edgar_mod, "Session", session_factory)

    async with session_factory() as s:
        s.add(SimAccount(id=1, cash=100_000.0))
        s.add(MonthlyAccount(id=1, cash=100_000.0))
        await s.commit()
    yield session_factory
    await engine.dispose()


@pytest.fixture
def armed(monkeypatch):
    """The D knob ON (an MCP URL is configured)."""
    monkeypatch.setattr(settings, "searxng_mcp_url", "http://searxng-mcp:8808/mcp")


async def _seed_book(mem_db, tickers=("AAA",)):
    """One monthly-book position per ticker + one stored candle session
    (AS_OF) so the as-of default and the close lookup resolve."""
    async with mem_db() as s:
        for t in tickers:
            s.add(MonthlyPosition(ticker=t, shares=100.0, avg_cost=100.0,
                                  opened_at=datetime(2026, 1, 5)))
            s.add(Candle(ticker=t, timestamp=datetime(2026, 3, 6), open=100.0,
                         high=101.0, low=99.0, close=100.5, volume=1e6))
        await s.commit()


async def _ledger_rows(mem_db) -> list:
    async with mem_db() as s:
        return (await s.scalars(
            select(LlmSellDecision).order_by(LlmSellDecision.id))).all()


def _hold_transport():
    """A scripted transport: one search, then the HOLD verdict."""
    return transport_turns([_turn(calls=[_tool_call()]), _turn(text=_HOLD)])


def _transport_raising_for(fail_ticker: str):
    """A transport that raises for one name's turns and scripts
    search-then-HOLD for the others."""
    state = {"n": 0}

    async def _t(messages, tools):
        if fail_ticker in str(messages):
            raise RuntimeError(f"proxy down for {fail_ticker}")
        state["n"] += 1
        if state["n"] % 2 == 1:
            return _turn(calls=[_tool_call()])
        return _turn(text=_HOLD)
    return _t


def _d_row(**over) -> LlmSellDecision:
    base = dict(decided_at=datetime(2026, 3, 6, 23, 1, tzinfo=UTC),
                session_date=AS_OF, event_type="D", engine="monthly",
                event_id=f"D:AAA:{AS_OF}", ticker="AAA", decision="HOLD",
                rationale="nothing material", prompt_version="sellshadow/1-d",
                parsed=True, fail_safe=False, fail_reason=None,
                context="{}", engine_branch="{}", backend="test", model="m1",
                latency_ms=1.0)
    base.update(over)
    return LlmSellDecision(**base)


# ---------------------------------------------------------------------------
# (a)-(e) run_news_scan — the scan core
# ---------------------------------------------------------------------------

class TestRunNewsScan:
    async def test_armed_happy_path_one_shadow_row(self, mem_db, armed,
                                                   tmp_path):
        await _seed_book(mem_db)
        out = str(tmp_path / "news-out")
        res = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out,
                                            mcp=FakeMcp(),
                                            transport=_hold_transport())
        assert res["status"] == "ok"
        assert res["as_of"] == AS_OF
        assert res["n_contexts"] == 1 and res["n_ledger_rows"] == 1
        assert res["n_deduped"] == 0 and res["n_errors"] == 0 and res["errors"] == []
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1                       # exactly ONE row
        r = rows[0]
        assert r.event_type == "D" and r.engine == "monthly"
        assert r.ticker == "AAA" and r.event_id == f"D:AAA:{AS_OF}"
        assert r.session_date == AS_OF
        assert r.decision == "HOLD" and r.parsed is True and r.fail_safe is False
        # the flagship shadow property: EVERY fill column is NULL
        assert r.filled_at is None and r.shares is None
        assert r.price is None and r.cash_after is None
        assert r.prompt_version == "sellshadow/1-d"
        ctx = json.loads(r.context)
        assert ctx["news_source"] == "searxng-mcp"
        assert ctx["temperature"] == 0.6
        assert len(ctx["trace"]) == 1               # non-empty trace
        # the artifacts landed (best-effort, but they do land)
        assert (tmp_path / "news-out" / f"news-consult-{AS_OF}.jsonl").exists()
        assert (tmp_path / "news-out" / f"news-run-{AS_OF}.json").exists()
        assert res["log_path"] == str(tmp_path / "news-out"
                                     / f"news-consult-{AS_OF}.jsonl")
        assert res["artifact_path"] == str(tmp_path / "news-out"
                                           / f"news-run-{AS_OF}.json")

    async def test_as_of_defaults_to_last_session(self, mem_db, armed, tmp_path):
        await _seed_book(mem_db)
        res = await newsagent.run_news_scan(out_dir=str(tmp_path / "n"),
                                            mcp=FakeMcp(),
                                            transport=_hold_transport())
        assert res["status"] == "ok" and res["as_of"] == AS_OF
        assert (await _ledger_rows(mem_db))[0].event_id == f"D:AAA:{AS_OF}"

    async def test_rerun_same_night_dedupes(self, mem_db, armed, tmp_path):
        await _seed_book(mem_db)
        out = str(tmp_path / "news-out")
        res1 = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out,
                                             mcp=FakeMcp(),
                                             transport=_hold_transport())
        assert res1["n_ledger_rows"] == 1
        res2 = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out,
                                             mcp=FakeMcp(),
                                             transport=_hold_transport())
        assert res2["status"] == "ok"
        assert res2["n_ledger_rows"] == 0 and res2["n_deduped"] == 1
        assert len(await _ledger_rows(mem_db)) == 1   # append-only: no new row

    async def test_fail_safe_row_does_not_silence(self, mem_db, armed, tmp_path):
        """A fail-safe consult for tonight (MCP down) must NOT dedupe: the
        rerun consults again and appends a new row."""
        await _seed_book(mem_db)
        async with mem_db() as s:
            s.add(_d_row(parsed=False, fail_safe=True, fail_reason="mcp failure",
                         rationale="fail-safe"))
            await s.commit()
        res = await newsagent.run_news_scan(as_of=AS_OF,
                                            out_dir=str(tmp_path / "news-out"),
                                            mcp=FakeMcp(),
                                            transport=_hold_transport())
        assert res["n_deduped"] == 0 and res["n_ledger_rows"] == 1
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 2                       # append-only: re-consulted
        assert rows[1].parsed is True and rows[1].fail_safe is False

    async def test_knob_off_disabled_zero_rows(self, mem_db, monkeypatch,
                                               tmp_path):
        monkeypatch.setattr(settings, "searxng_mcp_url", "")
        await _seed_book(mem_db)
        res = await newsagent.run_news_scan(as_of=AS_OF,
                                            out_dir=str(tmp_path / "n"),
                                            mcp=FakeMcp(),
                                            transport=_hold_transport())
        assert res["status"] == "disabled"
        assert res["n_ledger_rows"] == 0 and res["n_contexts"] == 0
        assert res["n_errors"] == 0                 # no exception escaped
        assert await _ledger_rows(mem_db) == []     # zero DB writes

    async def test_no_sessions(self, mem_db, armed, tmp_path):
        res = await newsagent.run_news_scan(out_dir=str(tmp_path / "n"))
        assert res["status"] == "no-sessions" and res["as_of"] is None
        assert res["n_ledger_rows"] == 0

    async def test_per_name_error_isolation(self, mem_db, armed, tmp_path,
                                             monkeypatch):
        """Two names; one name's pipeline fails — the other still gets its
        row, the failure is collected (n_errors == 1), nothing escapes.

        The transport raises for BBB (run_agent fail-safes it internally —
        it never raises by contract), and the scan-level seam (the ledger
        write) raises for BBB too, which is what the scan's per-name
        try/except collects."""
        await _seed_book(mem_db, tickers=("AAA", "BBB"))
        out = str(tmp_path / "news-out")
        real_record = newsagent._record_ledger_row

        async def flaky_record(*, rec, **kwargs):
            if kwargs.get("ticker") == "BBB":
                raise RuntimeError("ledger write boom")
            return await real_record(rec=rec, **kwargs)

        monkeypatch.setattr(newsagent, "_record_ledger_row", flaky_record)
        res = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out,
                                            mcp=FakeMcp(),
                                            transport=_transport_raising_for("BBB"))
        assert res["status"] == "ok"                # nothing escaped
        assert res["n_errors"] == 1 and res["n_ledger_rows"] == 1
        assert res["errors"][0]["ticker"] == "BBB"
        assert "ledger write boom" in res["errors"][0]["error"]
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1 and rows[0].ticker == "AAA"
        assert rows[0].parsed is True and rows[0].fail_safe is False


# ---------------------------------------------------------------------------
# (f) sellscore — D rows are visible but never scored/bucketed
# ---------------------------------------------------------------------------

class TestSellscoreShadow:
    async def _seed_tape_and_a_row(self, mem_db):
        """The canned AAA tape + one scorable A-DEFER row (the recipe from
        test_sellscore.py's hand-computed A test)."""
        sessions = [d.strftime("%Y-%m-%d")
                    for d in pd.bdate_range("2026-01-02", "2026-07-31")]
        async with mem_db() as s:
            for i, d in enumerate(sessions):
                op = 100.0 + 0.10 * i
                s.add(Candle(ticker="AAA",
                             timestamp=datetime.strptime(d, "%Y-%m-%d"),
                             open=op, high=op + 1.0, low=op - 1.0,
                             close=op + 0.5, volume=1e6))
            s.add(SimTrade(ticker="AAA", side="SELL", shares=10.0, price=97.5,
                           cash_after=0.0, reason="re-armed stop",
                           created_at=datetime(2026, 3, 20)))
            s.add(LlmSellDecision(
                decided_at=datetime(2026, 3, 6, 23, 0, tzinfo=UTC),
                session_date="2026-03-06", event_type="A", engine="sim",
                event_id="A:AAA:2026-03-06", ticker="AAA", decision="DEFER",
                rationale="x", prompt_version="sellshadow/1", parsed=True,
                fail_safe=False, fail_reason=None,
                context=json.dumps({"fill_price": 102.0}),
                engine_branch=json.dumps({"action": "defer", "level": 95.0}),
                backend="test", model="m1", latency_ms=1.0))
            await s.commit()

    async def test_d_row_shadow_not_in_a_bucket(self, mem_db):
        await self._seed_tape_and_a_row(mem_db)
        rep_a_only = await sellscore.score_sell_decisions(now=NOW)
        assert rep_a_only["n_d_shadow"] == 0
        assert rep_a_only["types"]["A"]["n_rows"] == 1
        assert rep_a_only["types"]["A"]["n_decided"] == 1

        async with mem_db() as s:
            s.add(_d_row())
            s.add(_d_row(event_id="X:AAA:2026-03-06", event_type="X"))
            await s.commit()
        rep = await sellscore.score_sell_decisions(now=NOW)

        # the D row is counted, visible in the history, but scored nowhere
        assert rep["n_d_shadow"] == 1
        assert "D" not in rep["types"]
        d_entry = next(d for d in rep["decisions"] if d["event_type"] == "D")
        assert d_entry["event_id"] == f"D:AAA:{AS_OF}"
        assert d_entry["status"] == "shadow"
        assert d_entry["skip_reason"] == \
            "type-D shadow row — scoring pending the D scorer"
        assert d_entry["delta"] is None and d_entry["window"] is None
        # NOT bucketed into A/B/C, NOT in the scored events, NOT in overall
        assert rep["types"]["A"]["n_rows"] == 1
        assert all(e["event_id"] != f"D:AAA:{AS_OF}"
                   for e in rep["types"]["A"]["events"])
        assert rep["overall"]["n_rows"] == 1
        # the A stats and the overall block are UNAFFECTED by the D row
        assert rep["types"]["A"] == rep_a_only["types"]["A"]
        assert rep["overall"] == rep_a_only["overall"]
        # any OTHER unknown event_type is skipped-with-reason the same way
        x_entry = next(d for d in rep["decisions"] if d["event_type"] == "X")
        assert x_entry["status"] == "skipped"
        assert "unknown event_type" in x_entry["skip_reason"]
        assert rep["n_d_shadow"] == 1               # X is not a D row
        assert rep["overall"]["n_rows"] == 1       # X counted nowhere either


# ---------------------------------------------------------------------------
# (g) the wired nightly — run_monthly_cycle calls C then D, D never breaks it
# ---------------------------------------------------------------------------

class TestNightlyWiring:
    def _patch_cycle(self, monkeypatch, tmp_path, d_impl):
        monkeypatch.setattr(settings, "sim_monthly_enabled", True)
        monkeypatch.setattr(settings, "sim_news_out_dir", str(tmp_path / "news-out"))
        order: list[str] = []

        async def fake_c(now=None, **kwargs):
            order.append("C")
            return {"fills": [], "scans": [], "as_of": "2026-03-06"}

        async def fake_rebalance(force=False, now=None):
            order.append("rebalance")
            return {"rebalanced": True}

        async def fake_d(*, out_dir, dry_run):
            order.append("D")
            assert out_dir == str(tmp_path / "news-out")
            assert dry_run is False
            return await d_impl()

        monkeypatch.setattr(selllayer, "run_type_c_step", fake_c)
        monkeypatch.setattr(newsagent, "run_news_scan", fake_d)
        monkeypatch.setattr(monthly, "run_rebalance", fake_rebalance)
        return order

    async def test_cycle_calls_c_then_d_then_rebuild(self, mem_db, monkeypatch,
                                                    tmp_path):
        async def ok_d():
            return {"status": "ok", "as_of": AS_OF, "n_contexts": 1,
                    "n_ledger_rows": 1, "n_deduped": 0, "n_errors": 0,
                    "errors": [], "log_path": None, "artifact_path": None}
        order = self._patch_cycle(monkeypatch, tmp_path, ok_d)
        monkeypatch.setattr(settings, "sim_monthly_llm_sells", True)
        out = await monthly.run_monthly_cycle(
            now=datetime(2026, 3, 6, 23, 30, tzinfo=UTC))
        assert out == {"rebalanced": True}
        assert order == ["C", "D", "rebalance"]    # declared order C-then-D

    async def test_raising_d_does_not_break_the_cycle(self, mem_db, monkeypatch,
                                                      tmp_path):
        async def boom_d():
            raise RuntimeError("news scan exploded")
        order = self._patch_cycle(monkeypatch, tmp_path, boom_d)
        monkeypatch.setattr(settings, "sim_monthly_llm_sells", True)
        out = await monthly.run_monthly_cycle(
            now=datetime(2026, 3, 6, 23, 30, tzinfo=UTC))
        assert out == {"rebalanced": True}          # the rebuild still ran
        assert order == ["C", "D", "rebalance"]

    async def test_d_not_gated_on_sim_monthly_llm_sells(self, mem_db, monkeypatch,
                                                        tmp_path):
        """sim_monthly_llm_sells owns A/B/C only — the D step runs (and is
        disabled by its OWN knob inside run_news_scan) regardless."""
        async def ok_d():
            return {"status": "disabled", "as_of": AS_OF, "n_contexts": 0,
                    "n_ledger_rows": 0, "n_deduped": 0, "n_errors": 0,
                    "errors": [], "log_path": None, "artifact_path": None}
        order = self._patch_cycle(monkeypatch, tmp_path, ok_d)
        monkeypatch.setattr(settings, "sim_monthly_llm_sells", False)
        out = await monthly.run_monthly_cycle(
            now=datetime(2026, 3, 6, 23, 30, tzinfo=UTC))
        assert out == {"rebalanced": True}
        assert order == ["D", "rebalance"]          # C skipped, D still runs


# ---------------------------------------------------------------------------
# (h) the MANUAL consult — run_manual_consult + the scan/CLI --fresh mode
# ---------------------------------------------------------------------------

_EXIT_URL = ('{"decision": "EXIT", "citations": [{"title": "AAA fraud probe", '
             '"source": "reuters", "date": "2026-10-31", '
             '"url": "https://reuters.com/aaa-probe"}], '
             '"rationale": "fraud accusation verified"}')


def _cyclic_transport(verdict_text):
    """A scripted transport that cycles search -> verdict -> search -> ...
    so EVERY agent run gets its own two turns (repeated CLI invocations in
    one test share the counter)."""
    state = {"n": 0}

    async def _t(messages, tools):
        state["n"] += 1
        if state["n"] % 2 == 1:
            return _turn(calls=[_tool_call()])
        return _turn(text=verdict_text)
    return _t


class TestManualConsult:
    async def test_happy_path_records_m_row(self, mem_db, armed, tmp_path):
        await _seed_book(mem_db)
        res = await newsagent.run_manual_consult(
            "AAA", as_of=AS_OF, out_dir=str(tmp_path / "n"),
            mcp=FakeMcp(), transport=_hold_transport())
        assert res["ok"] is True
        rec = res["record"]
        assert rec["event_id"] == f"D:AAA:{AS_OF}:m"
        assert rec["decision"] == "HOLD" and rec["parsed"] is True
        assert rec["fail_safe"] is False
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1
        r = rows[0]
        assert r.event_type == "D" and r.engine == "monthly"
        assert r.event_id == f"D:AAA:{AS_OF}:m" and r.ticker == "AAA"
        assert r.session_date == AS_OF and r.decision == "HOLD"
        # the flagship shadow property: EVERY fill column is NULL
        assert r.filled_at is None and r.shares is None
        assert r.price is None and r.cash_after is None
        ctx = json.loads(r.context)
        assert ctx["position"]["ticker"] == "AAA"
        assert ctx["news_source"] == "searxng-mcp"
        assert len(ctx["trace"]) == 1
        # the best-effort artifact landed
        assert (tmp_path / "n" / f"news-consult-{AS_OF}.jsonl").exists()

    async def test_as_of_defaults_to_last_session(self, mem_db, armed, tmp_path):
        await _seed_book(mem_db)
        res = await newsagent.run_manual_consult(
            "AAA", out_dir=str(tmp_path / "n"), mcp=FakeMcp(),
            transport=_hold_transport())
        assert res["ok"] is True
        assert res["record"]["event_id"] == f"D:AAA:{AS_OF}:m"

    async def test_repeated_consults_never_dedupe(self, mem_db, armed, tmp_path):
        """Fresh semantics: every manual consult runs again — no self-dedupe,
        no dedupe against nightly rows."""
        await _seed_book(mem_db)
        out = str(tmp_path / "n")
        r1 = await newsagent.run_manual_consult(
            "AAA", as_of=AS_OF, out_dir=out, mcp=FakeMcp(),
            transport=_hold_transport())
        r2 = await newsagent.run_manual_consult(
            "AAA", as_of=AS_OF, out_dir=out, mcp=FakeMcp(),
            transport=_hold_transport())
        assert r1["ok"] is True and r2["ok"] is True
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 2                       # append-only: both stand
        assert all(r.event_id == f"D:AAA:{AS_OF}:m" for r in rows)

    async def test_unknown_ticker_is_honest_error(self, mem_db, armed, tmp_path):
        await _seed_book(mem_db)
        res = await newsagent.run_manual_consult(
            "ZZZ", as_of=AS_OF, out_dir=str(tmp_path / "n"), mcp=FakeMcp(),
            transport=_hold_transport())
        assert res["ok"] is False and "record" not in res
        assert "not in the monthly book" in res["reason"] and "ZZZ" in res["reason"]
        assert await _ledger_rows(mem_db) == []    # zero DB writes

    async def test_knob_off_is_disabled_not_broken(self, mem_db, monkeypatch,
                                                    tmp_path):
        monkeypatch.setattr(settings, "searxng_mcp_url", "")
        await _seed_book(mem_db)
        res = await newsagent.run_manual_consult(
            "AAA", as_of=AS_OF, out_dir=str(tmp_path / "n"), mcp=FakeMcp(),
            transport=_hold_transport())
        assert res["ok"] is False
        assert res["reason"] == \
            "the D step is disabled — SEARXNG_MCP_URL is not set"
        assert await _ledger_rows(mem_db) == []    # zero DB writes, no error

    async def test_never_raises_on_recorder_failure(self, mem_db, armed,
                                                    tmp_path, monkeypatch):
        await _seed_book(mem_db)

        async def boom(*, rec, **kwargs):
            raise RuntimeError("ledger write boom")

        monkeypatch.setattr(newsagent, "_record_ledger_row", boom)
        res = await newsagent.run_manual_consult(
            "AAA", as_of=AS_OF, out_dir=str(tmp_path / "n"), mcp=FakeMcp(),
            transport=_hold_transport())
        assert res["ok"] is False                   # collected, not raised
        assert "ledger write boom" in res["reason"]

    async def test_fail_safe_record_is_a_valid_result(self, mem_db, armed,
                                                      tmp_path):
        """An agent failure (transport down) is a fail-safe RECORD, not an
        error: ok:True with the fail-safe record, and the row is appended."""
        await _seed_book(mem_db)

        async def boom(messages, tools):
            raise RuntimeError("proxy down")

        res = await newsagent.run_manual_consult(
            "AAA", as_of=AS_OF, out_dir=str(tmp_path / "n"), mcp=FakeMcp(),
            transport=boom)
        assert res["ok"] is True
        assert res["record"]["fail_safe"] is True
        assert res["record"]["decision"] == "HOLD"
        assert "proxy down" in res["record"]["fail_reason"]
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1 and rows[0].fail_safe is True
        assert rows[0].event_id == f"D:AAA:{AS_OF}:m"


class TestFreshScanAndCli:
    async def test_scan_fresh_uses_m_marker_and_never_dedupes(
            self, mem_db, armed, tmp_path):
        """--fresh: 4-part :m event ids, no dedupe at all (against nightly
        rows or itself); without --fresh the nightly 3-part id and its
        dedupe are unchanged."""
        await _seed_book(mem_db)
        out = str(tmp_path / "n")
        r1 = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out, fresh=True,
                                           mcp=FakeMcp(),
                                           transport=_hold_transport())
        r2 = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out, fresh=True,
                                           mcp=FakeMcp(),
                                           transport=_hold_transport())
        assert r1["n_ledger_rows"] == 1 and r2["n_ledger_rows"] == 1
        assert r1["n_deduped"] == 0 and r2["n_deduped"] == 0
        rows = await _ledger_rows(mem_db)
        assert [r.event_id for r in rows] == [f"D:AAA:{AS_OF}:m"] * 2
        # the nightly 3-part path is unchanged: consults, then dedupes
        r3 = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out,
                                           mcp=FakeMcp(),
                                           transport=_hold_transport())
        r4 = await newsagent.run_news_scan(as_of=AS_OF, out_dir=out,
                                           mcp=FakeMcp(),
                                           transport=_hold_transport())
        assert r3["n_ledger_rows"] == 1 and r4["n_deduped"] == 1
        assert r4["n_ledger_rows"] == 0
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 3
        assert rows[-1].event_id == f"D:AAA:{AS_OF}"    # exact 3-part form

    async def test_cli_fresh_prints_the_response(self, mem_db, armed,
                                                 monkeypatch, tmp_path, capsys):
        """The --fresh CLI run: :m event ids, repeated runs consult again,
        and the printed output carries the decision, the rationale and each
        citation (title + URL) after the summary line."""
        await _seed_book(mem_db)
        monkeypatch.setattr(newsagent, "McpClient", lambda url: FakeMcp())
        monkeypatch.setattr(newsagent, "_default_transport",
                            _cyclic_transport(_EXIT_URL))
        base = ["news", "--no-dry-run", "--fresh", "--as-of", AS_OF,
                "--out", str(tmp_path / "n")]
        assert await newsagent._cmd_news(
            newsagent._build_parser().parse_args(base)) == 0
        out = capsys.readouterr().out
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1 and rows[0].event_id == f"D:AAA:{AS_OF}:m"
        assert rows[0].decision == "EXIT"
        assert "EXIT" in out
        assert "rationale: fraud accusation verified" in out
        assert "AAA fraud probe" in out
        assert "https://reuters.com/aaa-probe" in out
        # a second --fresh run consults AGAIN (no self-dedupe)
        assert await newsagent._cmd_news(
            newsagent._build_parser().parse_args(base)) == 0
        capsys.readouterr()
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 2
        assert all(r.event_id == f"D:AAA:{AS_OF}:m" for r in rows)

    async def test_cli_without_fresh_keeps_nightly_3_part(
            self, mem_db, armed, monkeypatch, tmp_path, capsys):
        await _seed_book(mem_db)
        monkeypatch.setattr(newsagent, "McpClient", lambda url: FakeMcp())
        monkeypatch.setattr(newsagent, "_default_transport",
                            _cyclic_transport(_HOLD))
        base = ["news", "--no-dry-run", "--as-of", AS_OF,
                "--out", str(tmp_path / "n")]
        assert await newsagent._cmd_news(
            newsagent._build_parser().parse_args(base)) == 0
        assert await newsagent._cmd_news(        # deduped: consults once
            newsagent._build_parser().parse_args(base)) == 0
        out = capsys.readouterr().out
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1
        assert rows[0].event_id == f"D:AAA:{AS_OF}"      # exact 3-part form
        assert "rationale: nothing material" in out     # response still printed

    def test_parser_fresh_flag_and_help_documents_marker(self, capsys):
        p = newsagent._build_parser()
        assert p.parse_args(["news"]).fresh is False
        assert p.parse_args(["news", "--fresh"]).fresh is True
        with pytest.raises(SystemExit):
            p.parse_args(["news", "--help"])
        help_out = capsys.readouterr().out
        assert "D:{ticker}:{as_of}:m" in help_out
        assert ":m" in help_out and "deduped" in help_out
