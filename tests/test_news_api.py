"""Offline API tests for the type-D manual news-consult routes in app/main.py:
``POST /api/news/consult`` and ``GET /api/news/decisions/{event_id}``.

Zero network: the agent's MCP client and LLM turn are the fakes from
tests/test_newsagent.py, injected through the same seam the CLI tests use
(``newsagent.McpClient`` / ``newsagent._default_transport``); the DB is
in-memory SQLite (the temp-DB pattern from test_newsagent_nightly.py) —
main.py and newsagent.py bind ``Session`` at import time, so both are
patched, plus ``app.db`` for selllayer's append-only recorder.

Covered:
  (a) POST /api/news/consult happy path: 200, the full response shape, the
      4-part ``:m`` event_id, exactly one appended shadow ledger row (all
      fill columns NULL); ``as_of: null`` defaults to the last session;
  (b) knob-off -> the honest disabled response (200, ok:false, the exact
      reason, zero rows); unknown ticker -> 200 ok:false with the reason;
      malformed as_of / empty ticker -> 422;
  (c) a fail-safe agent record IS a valid 200 response — the route never
      500s on an agent failure;
  (d) GET /api/news/decisions/{event_id}: 200 with the context JSON parsed
      (position/citations/trace) plus decision/rationale/parsed/fail_safe/
      decided_at; an honest 404 for unknown ids AND for non-D event_ids
      (this is the news-agent view, not the A/B/C ledger).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import main, newsagent
from app.config import settings
from app.db import (
    Base,
    Candle,
    LlmSellDecision,
    MonthlyAccount,
    MonthlyPosition,
    SimAccount,
)
from tests.test_newsagent import _HOLD, FakeMcp, _tool_call, _turn

AS_OF = "2026-03-06"


# ---------------------------------------------------------------------------
# Fixtures / helpers (the temp-DB pattern from test_newsagent_nightly.py)
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite for every module the routes touch. main.py and
    newsagent.py bind ``Session`` at import time, so patch both (app.db /
    selllayer resolve it at call time, which covers the recorder)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod
    monkeypatch.setattr(newsagent, "Session", session_factory)
    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(main, "Session", session_factory)

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


@pytest.fixture
def fake_agent(monkeypatch):
    """The offline agent seam: a FakeMcp plus a scripted LLM transport that
    cycles search -> HOLD, so every consult gets its own two turns."""
    monkeypatch.setattr(newsagent, "McpClient", lambda url: FakeMcp())
    state = {"n": 0}

    async def _t(messages, tools):
        state["n"] += 1
        if state["n"] % 2 == 1:
            return _turn(calls=[_tool_call()])
        return _turn(text=_HOLD)
    monkeypatch.setattr(newsagent, "_default_transport", _t)
    return state


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://test") as c:
        yield c


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


# ---------------------------------------------------------------------------
# POST /api/news/consult
# ---------------------------------------------------------------------------

class TestNewsConsultRoute:
    async def test_happy_path(self, mem_db, armed, fake_agent, client):
        await _seed_book(mem_db)
        r = await client.post("/api/news/consult",
                              json={"ticker": "AAA", "as_of": AS_OF})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert {"ok", "decision", "rationale", "citations", "turns",
                "tool_calls", "latency_ms", "fail_safe", "fail_reason",
                "event_id", "model", "backend"} <= set(body)
        assert body["event_id"] == f"D:AAA:{AS_OF}:m"   # the 4-part :m marker
        assert body["decision"] == "HOLD" and body["fail_safe"] is False
        assert body["fail_reason"] is None
        assert body["tool_calls"] == 1 and body["turns"] == 2
        assert body["rationale"] == "nothing material"
        assert isinstance(body["citations"], list)
        assert body["model"] and body["backend"]
        # exactly ONE appended shadow ledger row, fills NULL
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1
        row = rows[0]
        assert row.event_id == f"D:AAA:{AS_OF}:m"
        assert row.event_type == "D" and row.engine == "monthly"
        assert row.ticker == "AAA" and row.session_date == AS_OF
        assert row.filled_at is None and row.shares is None
        assert row.price is None and row.cash_after is None

    async def test_as_of_null_defaults_to_last_session(self, mem_db, armed,
                                                       fake_agent, client):
        await _seed_book(mem_db)
        r = await client.post("/api/news/consult", json={"ticker": "AAA"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["event_id"] == f"D:AAA:{AS_OF}:m"

    async def test_ticker_case_insensitive(self, mem_db, armed, fake_agent,
                                           client):
        await _seed_book(mem_db)
        r = await client.post("/api/news/consult",
                              json={"ticker": " aaa ", "as_of": AS_OF})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert r.json()["event_id"] == f"D:AAA:{AS_OF}:m"

    async def test_knob_off_disabled_response(self, mem_db, monkeypatch,
                                               client):
        monkeypatch.setattr(settings, "searxng_mcp_url", "")
        await _seed_book(mem_db)
        r = await client.post("/api/news/consult",
                              json={"ticker": "AAA", "as_of": AS_OF})
        assert r.status_code == 200                     # honest, not an error
        assert r.json() == {
            "ok": False,
            "reason": "the D step is disabled — SEARXNG_MCP_URL is not set",
        }
        assert await _ledger_rows(mem_db) == []         # zero DB writes

    async def test_unknown_ticker(self, mem_db, armed, fake_agent, client):
        await _seed_book(mem_db)
        r = await client.post("/api/news/consult",
                              json={"ticker": "ZZZ", "as_of": AS_OF})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "not in the monthly book" in body["reason"]
        assert "ZZZ" in body["reason"]
        assert await _ledger_rows(mem_db) == []

    async def test_fail_safe_record_is_a_valid_response(self, mem_db, armed,
                                                         monkeypatch, client):
        """The route NEVER 500s on an agent failure — a fail-safe record is
        a valid 200 response (and lands in the ledger as fail-safe)."""
        await _seed_book(mem_db)
        monkeypatch.setattr(newsagent, "McpClient", lambda url: FakeMcp())

        async def boom(messages, tools):
            raise RuntimeError("proxy down")

        monkeypatch.setattr(newsagent, "_default_transport", boom)
        r = await client.post("/api/news/consult",
                              json={"ticker": "AAA", "as_of": AS_OF})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["fail_safe"] is True
        assert body["decision"] == "HOLD"
        assert body["fail_reason"] and "proxy down" in body["fail_reason"]
        assert body["event_id"] == f"D:AAA:{AS_OF}:m"
        rows = await _ledger_rows(mem_db)
        assert len(rows) == 1 and rows[0].fail_safe is True
        assert rows[0].event_id == f"D:AAA:{AS_OF}:m"

    async def test_bad_as_of_and_empty_ticker_are_422(self, mem_db, armed,
                                                      fake_agent, client):
        await _seed_book(mem_db)
        r = await client.post("/api/news/consult",
                              json={"ticker": "AAA", "as_of": "03/06/2026"})
        assert r.status_code == 422
        r = await client.post("/api/news/consult", json={"ticker": "   "})
        assert r.status_code == 422
        assert await _ledger_rows(mem_db) == []


# ---------------------------------------------------------------------------
# GET /api/news/decisions/{event_id}
# ---------------------------------------------------------------------------

class TestNewsDecisionRoute:
    async def _consult(self, client):
        r = await client.post("/api/news/consult",
                              json={"ticker": "AAA", "as_of": AS_OF})
        assert r.status_code == 200 and r.json()["ok"] is True
        return r.json()["event_id"]

    async def test_200_with_parsed_context(self, mem_db, armed, fake_agent,
                                           client):
        await _seed_book(mem_db)
        event_id = await self._consult(client)
        assert event_id == f"D:AAA:{AS_OF}:m"
        r = await client.get(f"/api/news/decisions/{event_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["event_id"] == event_id
        assert body["event_type"] == "D" and body["engine"] == "monthly"
        assert body["ticker"] == "AAA" and body["session_date"] == AS_OF
        assert body["decision"] == "HOLD" and body["parsed"] is True
        assert body["fail_safe"] is False and body["fail_reason"] is None
        assert body["rationale"] == "nothing material"
        assert body["decided_at"]                  # ISO string
        ctx = body["context"]                      # parsed, not raw JSON
        assert isinstance(ctx, dict)
        assert ctx["position"]["ticker"] == "AAA"
        assert isinstance(ctx["citations"], list)
        assert len(ctx["trace"]) == 1
        assert ctx["news_source"] == "searxng-mcp"
        assert ctx["position"]["entry"]["price"] == 100.0

    async def test_honest_404_for_unknown_id(self, mem_db, client):
        r = await client.get("/api/news/decisions/D:NOPE:2026-03-06:m")
        assert r.status_code == 404
        assert "no type-D decision" in r.json()["detail"]

    async def test_refuses_non_d_event_ids(self, mem_db, armed, fake_agent,
                                           client):
        """Only event_type "D" rows answer — this is the news-agent view,
        not the A/B/C ledger."""
        await _seed_book(mem_db)
        async with mem_db() as s:
            s.add(LlmSellDecision(
                decided_at=datetime(2026, 3, 6, 23, 0, tzinfo=UTC),
                session_date=AS_OF, event_type="A", engine="sim",
                event_id="A:AAA:2026-03-06", ticker="AAA", decision="DEFER",
                rationale="x", prompt_version="sellshadow/1", parsed=True,
                fail_safe=False, fail_reason=None, context="{}",
                engine_branch="{}", backend="test", model="m1", latency_ms=1.0))
            await s.commit()
        r = await client.get("/api/news/decisions/A:AAA:2026-03-06")
        assert r.status_code == 404
        # an un-stored nightly 3-part D id is just an unknown id here
        r = await client.get(f"/api/news/decisions/D:AAA:{AS_OF}")
        assert r.status_code == 404

    async def test_stored_nightly_d_row_answers(self, mem_db, armed, client):
        """A stored NIGHTLY 3-part D row is also a type-D row and answers
        (the view serves the whole D track, nightly and manual)."""
        await _seed_book(mem_db)
        async with mem_db() as s:
            s.add(LlmSellDecision(
                decided_at=datetime(2026, 3, 6, 23, 1, tzinfo=UTC),
                session_date=AS_OF, event_type="D", engine="monthly",
                event_id=f"D:AAA:{AS_OF}", ticker="AAA", decision="HOLD",
                rationale="nightly row", prompt_version="sellshadow/1-d",
                parsed=True, fail_safe=False, fail_reason=None,
                context=json.dumps({"position": {"ticker": "AAA"},
                                    "citations": [], "trace": []}),
                engine_branch="{}", backend="test", model="m1",
                latency_ms=1.0))
            await s.commit()
        r = await client.get(f"/api/news/decisions/D:AAA:{AS_OF}")
        assert r.status_code == 200
        body = r.json()
        assert body["event_id"] == f"D:AAA:{AS_OF}"
        assert body["context"]["position"]["ticker"] == "AAA"
        assert body["rationale"] == "nightly row"
