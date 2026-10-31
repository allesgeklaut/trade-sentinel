"""Offline tests for the Phase 3 chat proposal/confirm split (slice 3D) and
the live run-config endpoint (slice 3F).

The LLM is ALWAYS stubbed (fake ``httpx.AsyncClient`` / monkeypatched
``chat_stream`` — never a network call), the DB is in-memory SQLite, never
the production DB.

Covered:

1. a chat turn with an ``[[ACTION]]`` block returns proposals and writes
   NOTHING (no trades, no snapshot, no ledger rows, no RunExecution);
2. the envelope is honest — the ``trades`` / ``actions_executed`` keys are
   gone entirely;
3. display text is stripped of the action block and persisted; the raw stream
   keeps its deltas (block included); the done event carries the proposals;
4. the proposal preview math matches the execution math (feasible and each
   infeasible case, shares/amount clamping mirrored);
5. confirm happy path: fresh stamp, trades + snapshot_taken, ledger rows
   whose decided_at equals the proposal's frozen parse-time stamp (rule 7),
   and the pending entry removed (a second confirm 404s);
6. confirm of an expired proposal → 404;
7. confirm after a config change → 409 with nothing written, pending dropped;
8. reject → ok, then confirm → 404;
9. registry cap eviction: 21 proposals → the oldest is gone;
10. guard parity: confirm 409s while ``main._db_cloning`` is set on a
    dev-routed request, and proceeds when prod-routed;
11. ``GET /api/run-config``: sim/monthly/daily_core keys, each with epoch >= 1
    and its engine's config fields; a dev-routed request follows the cookie's
    universe resolution;
12. the per-action skip reasons in the confirm response.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import StaticPool, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import db as db_mod
from app import main, sim
from app.config import settings
from app.db import (
    Base,
    RunExecution,
    SimAccount,
    SimChatMessage,
    SimLlmDecision,
    SimPosition,
    SimSnapshot,
    SimTrade,
)

# A mid-day, DST-safe parse time for the frozen-clock test.
_T0 = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)

_ACTION_BUY_500 = (
    'Buying.\n[[ACTION]]\n'
    '{"actions": [{"ticker": "AAPL", "action": "BUY", "amount": 500.0, '
    '"reason": "go"}]}\n[[/ACTION]]'
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_pending_proposals():
    """The pending-proposal registry is process-global — keep tests isolated."""
    sim._pending_proposals.clear()
    yield
    sim._pending_proposals.clear()


@pytest.fixture
async def mem_db(monkeypatch, tmp_path):
    """In-memory SQLite wired into every engine's Session factory (and the
    per-DB state files pointed at tmp), so nothing touches /data."""
    import app.daily_core as daily_core_mod
    import app.monthly as monthly_mod

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(sim, "Session", factory)
    monkeypatch.setattr(db_mod, "Session", factory)
    monkeypatch.setattr(daily_core_mod, "Session", factory)
    monkeypatch.setattr(monthly_mod, "Session", factory)
    monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "sim_state.json")
    monkeypatch.setattr(sim, "_STATE_FILE_DEV", tmp_path / "sim_state_dev.json")
    monkeypatch.setattr(daily_core_mod, "_STATE_FILE", tmp_path / "daily_core_state.json")
    monkeypatch.setattr(daily_core_mod, "_STATE_FILE_DEV",
                        tmp_path / "daily_core_state_dev.json")

    async with factory() as s:
        s.add(SimAccount(id=1, cash=10000.0, last_allowance_month=None))
        await s.commit()

    yield factory
    await engine.dispose()


@pytest.fixture
async def dev_engine(mem_db, monkeypatch, tmp_path):
    """A second file-backed engine wired in as DEV (the two_file_dbs pattern
    from tests/test_db_switch.py) so the ts_db cookie routes for real."""
    dev_path = tmp_path / "dev.sqlite"
    eng_dev = db_mod.create_db_engine(f"sqlite+aiosqlite:///{dev_path}")
    async with eng_dev.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(db_mod, "engine_dev", eng_dev)
    db_mod._sessionmakers.clear()
    yield eng_dev
    db_mod._sessionmakers.clear()
    await eng_dev.dispose()


@pytest.fixture
async def client(mem_db):
    """httpx ASGI client against the real app (no server, no network)."""
    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://test") as c:
        yield c


@pytest.fixture
def llm_backend(monkeypatch, tmp_path):
    """Configure a fake backend so the LLM paths reach the stubbed HTTP client."""
    monkeypatch.setattr(sim.llm_mod.settings, "llm_backends", json.dumps([
        {"name": "test", "type": "ollama", "url": "http://test:11434", "model": "m1"},
    ]))
    monkeypatch.setattr(sim.llm_mod, "_DEFAULT_STATE_FILE", tmp_path / "llm_state.json")


def _stub_llm(monkeypatch, *contents: str) -> None:
    """Stub ``llm_mod.chat`` by faking ``httpx.AsyncClient`` (no network)."""
    queue = list(contents)

    class FakeResponse:
        status_code = 200

        def json(self):
            text = queue[0] if len(queue) == 1 else queue.pop(0)
            return {"message": {"content": text}}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)


def _mock_context(monkeypatch) -> None:
    async def mock_context():
        return "mock context"

    monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)


def _close(prices: dict):
    async def fake_close(ticker):
        return prices.get(ticker)

    return fake_close


async def _counts(factory) -> dict[str, int]:
    """Row counts of every table a proposing/confirming turn could touch."""
    async with factory() as s:
        out = {}
        for name, model in (
            ("trades", SimTrade),
            ("snapshots", SimSnapshot),
            ("ledger", SimLlmDecision),
            ("executions", RunExecution),
            ("chat_messages", SimChatMessage),
            ("positions", SimPosition),
        ):
            out[name] = await s.scalar(select(func.count()).select_from(model)) or 0
        return out


async def _http_chat(client, monkeypatch, text: str, prices: dict | None = None) -> dict:
    """One full chat turn over HTTP with a stubbed LLM; returns the envelope."""
    _mock_context(monkeypatch)
    if prices is not None:
        monkeypatch.setattr(sim, "_latest_close", _close(prices))
    _stub_llm(monkeypatch, text)
    r = await client.post("/api/sim/chat",
                          json={"messages": [{"role": "user", "content": "go"}]})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1+2. A proposing turn books nothing; the envelope is honest
# ---------------------------------------------------------------------------

class TestChatTurnProposesOnly:
    async def test_action_turn_returns_proposals_and_writes_nothing(
            self, mem_db, monkeypatch, llm_backend):
        monkeypatch.setattr(sim, "_latest_close", _close({"AAPL": 100.0}))
        _mock_context(monkeypatch)
        _stub_llm(monkeypatch, _ACTION_BUY_500)

        before = await _counts(mem_db)
        out = await sim.sim_chat([{"role": "user", "content": "buy AAPL"}])
        after = await _counts(mem_db)

        # 1: proposals returned; the DB changed ONLY by the two persisted
        # chat messages (user + stripped assistant reply).
        assert len(out["proposals"]) == 1
        assert after == {**before, "chat_messages": before["chat_messages"] + 2}

        # 2: the envelope has no execution keys at all.
        assert set(out.keys()) == {"text", "proposals", "history"}
        assert "trades" not in out and "actions_executed" not in out

        # Proposal shape: id, frozen parse-time decided_at, the turn's config,
        # one entry per parsed action with the pinned preview fields.
        proposal = out["proposals"][0]
        assert set(proposal.keys()) == {"proposal_id", "decided_at", "config", "actions"}
        assert len(proposal["proposal_id"]) == 32
        datetime.fromisoformat(proposal["decided_at"])  # valid ISO timestamp
        assert proposal["config"]["engine"] == "sim"
        action = proposal["actions"][0]
        assert action["ticker"] == "AAPL" and action["action"] == "BUY"
        assert action["amount"] == 500.0 and action["reason"] == "go"
        assert set(action["preview"].keys()) == {
            "feasible", "issue", "price", "est_shares", "est_value",
            "est_cash_after", "est_position_value_after",
        }
        assert action["preview"]["feasible"] is True
        # The action block is stripped from the display text but the raw
        # stream keeps it (checked in the stream tests below).
        assert "[[ACTION]]" not in out["text"]
        assert out["history"][-1]["role"] == "assistant"
        assert "[[ACTION]]" not in out["history"][-1]["content"]

    async def test_http_envelope_has_no_execution_keys(
            self, client, mem_db, monkeypatch, llm_backend):
        body = await _http_chat(client, monkeypatch, _ACTION_BUY_500,
                                prices={"AAPL": 100.0})
        assert set(body.keys()) == {"text", "proposals", "history"}
        assert "trades" not in body and "actions_executed" not in body
        assert body["proposals"][0]["proposal_id"]
        counts = await _counts(mem_db)
        assert counts["trades"] == 0
        assert counts["snapshots"] == 0
        assert counts["ledger"] == 0
        assert counts["executions"] == 0


# ---------------------------------------------------------------------------
# 3. Streaming: raw deltas keep the block, done carries the proposal
# ---------------------------------------------------------------------------

class TestStreamProposals:
    async def test_stream_keeps_raw_deltas_and_done_carries_proposals(
            self, mem_db, monkeypatch, llm_backend):
        monkeypatch.setattr(sim, "_latest_close", _close({"AAPL": 100.0}))
        _mock_context(monkeypatch)
        raw = _ACTION_BUY_500

        async def fake_chat_stream(messages):
            yield {"type": "delta", "text": raw[:20]}
            yield {"type": "delta", "text": raw[20:]}
            yield {"type": "done", "text": raw, "reasoning": "",
                   "backend": "a", "model": "m1"}

        monkeypatch.setattr(sim.llm_mod, "chat_stream", fake_chat_stream)

        events = [e async for e in sim.sim_chat_stream(
            [{"role": "user", "content": "buy AAPL"}])]
        deltas = [e for e in events if e["type"] == "delta"]
        assert "".join(d["text"] for d in deltas) == raw  # raw keeps the block
        done = next(e for e in events if e["type"] == "done")
        assert done["raw"] == raw
        assert "[[ACTION]]" not in done["text"]
        assert "Buying." in done["text"]
        assert len(done["proposals"]) == 1
        assert done["proposals"][0]["actions"][0]["preview"]["feasible"] is True
        assert "trades" not in done and "actions_executed" not in done

        # Persisted history is the stripped display text.
        history = await sim.get_chat_history()
        assert history[0] == {"role": "user", "content": "buy AAPL"}
        assert history[-1]["role"] == "assistant"
        assert "[[ACTION]]" not in history[-1]["content"]

        # Still nothing executed by the turn alone.
        counts = await _counts(mem_db)
        assert counts["trades"] == 0
        assert counts["ledger"] == 0
        assert counts["snapshots"] == 0
        assert counts["executions"] == 0

    async def test_http_stream_done_event_carries_proposals(
            self, client, mem_db, monkeypatch, llm_backend):
        monkeypatch.setattr(sim, "_latest_close", _close({"AAPL": 100.0}))
        _mock_context(monkeypatch)
        raw = _ACTION_BUY_500

        async def fake_chat_stream(messages):
            yield {"type": "delta", "text": raw}
            yield {"type": "done", "text": raw, "reasoning": "",
                   "backend": "a", "model": "m1"}

        monkeypatch.setattr(sim.llm_mod, "chat_stream", fake_chat_stream)

        r = await client.post("/api/sim/chat/stream",
                              json={"messages": [{"role": "user", "content": "buy"}]})
        assert r.status_code == 200, r.text
        events = [json.loads(line[len("data: "):])
                  for line in r.text.splitlines() if line.startswith("data: ")]
        deltas = [e for e in events if e["type"] == "delta"]
        assert any("[[ACTION]]" in d["text"] for d in deltas)  # raw deltas keep it
        done = next(e for e in events if e["type"] == "done")
        assert "[[ACTION]]" in done["raw"]
        assert "[[ACTION]]" not in done["text"]
        assert len(done["proposals"]) == 1
        assert "trades" not in done and "actions_executed" not in done


# ---------------------------------------------------------------------------
# 4. Preview math == execution math
# ---------------------------------------------------------------------------

class TestPreviewMath:
    async def test_action_preview_cases(self, mem_db, monkeypatch):
        """Every feasibility branch of the shared _action_preview helper."""
        await sim._exec_buy("AAPL", 100.0, 9000.0, "seed")  # 90 shares, cash 1000
        monkeypatch.setattr(sim, "_latest_close", _close({"AAPL": 100.0, "ZZZ": 50.0}))
        valuation = await sim.valuate()
        acc = await sim._account()

        # BUY AAPL: position at max (9000 >= 10% of 10000 = 1000).
        plan = await sim._action_preview(
            {"ticker": "AAPL", "action": "BUY", "reason": "x"}, valuation, acc)
        assert plan["feasible"] is False
        assert plan["issue"] == "position at max"
        assert plan["price"] == 100.0

        # BUY ZZZ: feasible — max_budget = min(1000-500, 1000-0) = 500 → 10 sh.
        plan = await sim._action_preview(
            {"ticker": "ZZZ", "action": "BUY", "reason": "x"}, valuation, acc)
        assert plan["feasible"] is True
        assert plan["budget"] == pytest.approx(500.0)
        assert plan["est_shares"] == pytest.approx(10.0)
        assert plan["est_value"] == pytest.approx(500.0)
        assert plan["est_cash_after"] == pytest.approx(500.0)
        assert plan["est_position_value_after"] == pytest.approx(500.0)

        # shares/amount clamping is mirrored: 100 shares @50 = $5000 → $500.
        plan = await sim._action_preview(
            {"ticker": "ZZZ", "action": "BUY", "shares": 100.0, "reason": "x"},
            valuation, acc)
        assert plan["feasible"] is True
        assert plan["est_shares"] == pytest.approx(10.0)
        assert plan["est_value"] == pytest.approx(500.0)
        plan = await sim._action_preview(
            {"ticker": "ZZZ", "action": "BUY", "amount": 50.0, "reason": "x"},
            valuation, acc)
        assert plan["est_shares"] == pytest.approx(1.0)
        assert plan["est_value"] == pytest.approx(50.0)

        # SELL without size liquidates: 90 shares → $9000 proceeds.
        plan = await sim._action_preview(
            {"ticker": "AAPL", "action": "SELL", "reason": "x"}, valuation, acc)
        assert plan["feasible"] is True
        assert plan["est_shares"] == pytest.approx(90.0)
        assert plan["est_value"] == pytest.approx(9000.0)
        assert plan["est_cash_after"] == pytest.approx(10000.0)
        assert plan["est_position_value_after"] == pytest.approx(0.0)
        # SELL amount clamps to the position: $2500 → 25 shares.
        plan = await sim._action_preview(
            {"ticker": "AAPL", "action": "SELL", "amount": 2500.0, "reason": "x"},
            valuation, acc)
        assert plan["est_shares"] == pytest.approx(25.0)

        # SELL ZZZ: no position.
        plan = await sim._action_preview(
            {"ticker": "ZZZ", "action": "SELL", "reason": "x"}, valuation, acc)
        assert plan["feasible"] is False
        assert plan["issue"] == "no position"

        # AAA has no close → no price.
        plan = await sim._action_preview(
            {"ticker": "AAA", "action": "BUY", "reason": "x"}, valuation, acc)
        assert plan["feasible"] is False
        assert plan["issue"] == "no price"
        assert plan["price"] is None

        # Drain cash below the floor: cash 100 < 5% of (100 + 9000) = 455.
        async with mem_db() as s:
            acc_row = await s.get(SimAccount, 1)
            acc_row.cash = 100.0
            await s.commit()
        valuation = await sim.valuate()
        acc = await sim._account()
        plan = await sim._action_preview(
            {"ticker": "ZZZ", "action": "BUY", "reason": "x"}, valuation, acc)
        assert plan["feasible"] is False
        assert plan["issue"] == "insufficient cash"

        # budget < 1: a tiny max-position % leaves no room.
        monkeypatch.setattr(settings, "sim_max_position_pct", 0.001)
        async with mem_db() as s:
            acc_row = await s.get(SimAccount, 1)
            acc_row.cash = 10000.0
            await s.commit()
        valuation = await sim.valuate()
        acc = await sim._account()
        plan = await sim._action_preview(
            {"ticker": "ZZZ", "action": "BUY", "reason": "x"}, valuation, acc)
        assert plan["feasible"] is False
        assert plan["issue"] == "budget < 1"

    async def test_preview_matches_confirm_execution(
            self, mem_db, monkeypatch, llm_backend):
        """A feasible preview's estimates are exactly what confirm executes."""
        monkeypatch.setattr(sim, "_latest_close", _close({"AAPL": 100.0}))
        _mock_context(monkeypatch)
        # amount 5000 with default 10% max position → clamped to $1000.
        _stub_llm(monkeypatch, (
            '[[ACTION]]\n{"actions": [{"ticker": "AAPL", "action": "BUY", '
            '"amount": 5000.0, "reason": "x"}]}\n[[/ACTION]]'))
        out = await sim.sim_chat([{"role": "user", "content": "buy"}])
        preview = out["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_shares"] == pytest.approx(10.0)
        assert preview["est_value"] == pytest.approx(1000.0)
        assert preview["est_cash_after"] == pytest.approx(9000.0)
        assert preview["est_position_value_after"] == pytest.approx(1000.0)

        conf = await sim.confirm_chat_proposal(out["proposals"][0]["proposal_id"])
        assert len(conf["trades"]) == 1
        trade = conf["trades"][0]
        assert trade["shares"] == pytest.approx(preview["est_shares"])
        assert trade["price"] == preview["price"]
        assert trade["cash_after"] == pytest.approx(preview["est_cash_after"], abs=0.01)


# ---------------------------------------------------------------------------
# 5. Confirm happy path (rule 7 freeze, pending removal)
# ---------------------------------------------------------------------------

class TestConfirmHappyPath:
    async def test_confirm_executes_with_frozen_decided_at(
            self, mem_db, monkeypatch, llm_backend):
        # Freeze the FIRST _utcnow() call (the parse-time decided_at) at _T0.
        state = {"calls": 0}

        def _now():
            state["calls"] += 1
            if state["calls"] == 1:
                return _T0
            return _T0 + timedelta(seconds=30 * state["calls"])

        monkeypatch.setattr(sim, "_utcnow", _now)
        monkeypatch.setattr(sim, "_latest_close", _close({"AAPL": 100.0}))
        _mock_context(monkeypatch)
        _stub_llm(monkeypatch, _ACTION_BUY_500)

        out = await sim.sim_chat([{"role": "user", "content": "buy"}])
        proposal = out["proposals"][0]
        pid = proposal["proposal_id"]
        assert proposal["decided_at"] == _T0.isoformat()

        conf = await sim.confirm_chat_proposal(pid)
        assert conf["ok"] is True
        assert conf["snapshot_taken"] is True
        assert conf["skipped"] == []
        assert [(t["ticker"], t["side"]) for t in conf["trades"]] == [("AAPL", "BUY")]
        assert conf["trades"][0]["shares"] == pytest.approx(5.0)
        assert conf["trades"][0]["reason"].startswith("User chat:")

        async with mem_db() as s:
            trades = (await s.scalars(select(SimTrade))).all()
            snaps = (await s.scalars(select(SimSnapshot))).all()
            execs = (await s.scalars(select(RunExecution))).all()
            rows = (await s.scalars(
                select(SimLlmDecision).order_by(SimLlmDecision.id))).all()
        # Forward paper records: trade + snapshot stamped, one RunExecution.
        assert len(trades) == 1 and trades[0].mode == "paper" and trades[0].epoch == 1
        assert len(snaps) == 1 and snaps[0].mode == "paper" and snaps[0].epoch == 1
        assert any(e.engine == "sim" and e.mode == "paper" for e in execs)
        # Rule 7: the ledger row's decided_at is the PROPOSAL's frozen stamp,
        # never re-stamped at confirm time; the fill came later.
        assert len(rows) == 1
        assert rows[0].mode == "chat" and rows[0].relation == "chat_action"
        assert rows[0].decided_at == _T0
        assert rows[0].filled_at is not None and rows[0].filled_at > rows[0].decided_at

        # The pending entry is consumed: a second confirm raises not-found.
        with pytest.raises(sim.ProposalNotFound):
            await sim.confirm_chat_proposal(pid)


# ---------------------------------------------------------------------------
# 6+7+8+12. HTTP-level confirm/reject behavior
# ---------------------------------------------------------------------------

class TestConfirmRejectHttp:
    async def test_expired_or_unknown_proposal_confirm_404(
            self, client, mem_db, monkeypatch, llm_backend):
        body = await _http_chat(client, monkeypatch, _ACTION_BUY_500,
                                prices={"AAPL": 100.0})
        pid = body["proposals"][0]["proposal_id"]

        # Unknown id.
        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": "0" * 32})
        assert r.status_code == 404
        assert r.json()["detail"] == "proposal not found or expired"

        # Expired: shrink the TTL below the proposal's age.
        monkeypatch.setattr(sim, "_PROPOSAL_TTL_SECONDS", -1)
        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": pid})
        assert r.status_code == 404
        assert r.json()["detail"] == "proposal not found or expired"
        counts = await _counts(mem_db)
        assert counts["trades"] == 0
        assert counts["ledger"] == 0
        assert counts["snapshots"] == 0
        assert counts["executions"] == 0

    async def test_config_change_between_proposal_and_confirm_409(
            self, client, mem_db, monkeypatch, llm_backend):
        body = await _http_chat(client, monkeypatch, _ACTION_BUY_500,
                                prices={"AAPL": 100.0})
        pid = body["proposals"][0]["proposal_id"]
        before = await _counts(mem_db)

        # The effective configuration moves while the proposal is pending.
        monkeypatch.setattr(settings, "sim_max_position_pct", 47.0)
        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": pid})
        assert r.status_code == 409
        assert r.json()["detail"] == (
            "run configuration changed since the proposal — ask again")

        # Nothing was written by the refused confirm...
        after = await _counts(mem_db)
        assert after == before
        # ...and the pending entry was dropped: retrying now 404s (not 409).
        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": pid})
        assert r.status_code == 404

    async def test_reject_then_confirm_404(
            self, client, mem_db, monkeypatch, llm_backend):
        body = await _http_chat(client, monkeypatch, _ACTION_BUY_500,
                                prices={"AAPL": 100.0})
        pid = body["proposals"][0]["proposal_id"]

        r = await client.post("/api/sim/chat/reject", json={"proposal_id": pid})
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        counts = await _counts(mem_db)
        assert counts["trades"] == 0 and counts["ledger"] == 0
        assert counts["snapshots"] == 0 and counts["executions"] == 0

        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": pid})
        assert r.status_code == 404
        r = await client.post("/api/sim/chat/reject", json={"proposal_id": pid})
        assert r.status_code == 404

    async def test_confirm_skip_reasons_are_machine_readable(
            self, client, mem_db, monkeypatch, llm_backend):
        # ZZZ has no price, AAPL has no position for the SELL, the BUY is fine.
        _stub_llm_text = (
            '[[ACTION]]\n{"actions": [\n'
            '  {"ticker": "ZZZ", "action": "BUY", "amount": 100.0, "reason": "no quote"},\n'
            '  {"ticker": "AAPL", "action": "SELL", "reason": "no position"},\n'
            '  {"ticker": "AAPL", "action": "BUY", "amount": 100.0, "reason": "ok"}\n'
            ']}\n[[/ACTION]]')
        body = await _http_chat(client, monkeypatch, _stub_llm_text,
                                prices={"AAPL": 100.0})
        proposal = body["proposals"][0]
        assert len(proposal["actions"]) == 3
        previews = {a["ticker"] + a["action"]: a["preview"] for a in proposal["actions"]}
        assert previews["ZZZBUY"]["feasible"] is False
        assert previews["ZZZBUY"]["issue"] == "no price"
        assert previews["AAPLSELL"]["feasible"] is False
        assert previews["AAPLSELL"]["issue"] == "no position"
        assert previews["AAPLBUY"]["feasible"] is True

        r = await client.post("/api/sim/chat/confirm",
                              json={"proposal_id": proposal["proposal_id"]})
        assert r.status_code == 200, r.text
        conf = r.json()
        assert conf["ok"] is True
        assert conf["skipped"] == [
            {"ticker": "ZZZ", "reason": "no price"},
            {"ticker": "AAPL", "reason": "no position"},
        ]
        assert len(conf["trades"]) == 1
        assert conf["trades"][0]["ticker"] == "AAPL"
        assert conf["trades"][0]["side"] == "BUY"
        assert conf["snapshot_taken"] is True

        # All three actions are frozen in the ledger; only the BUY filled.
        async with mem_db() as s:
            rows = (await s.scalars(
                select(SimLlmDecision).order_by(SimLlmDecision.id))).all()
        assert [row.ticker for row in rows] == ["ZZZ", "AAPL", "AAPL"]
        assert rows[0].filled_at is None and rows[0].shares is None
        assert rows[1].filled_at is None and rows[1].shares is None
        assert rows[2].filled_at is not None
        assert rows[2].shares == pytest.approx(conf["trades"][0]["shares"])


# ---------------------------------------------------------------------------
# 9. Registry cap eviction
# ---------------------------------------------------------------------------

class TestProposalCap:
    async def test_cap_evicts_oldest(self, mem_db):
        ids = []
        for _ in range(21):
            pid = uuid.uuid4().hex
            ids.append(pid)
            await sim._register_proposal({
                "proposal_id": pid,
                "decided_at": sim._utcnow().isoformat(),
                "config": {},
                "actions": [],
            })
        assert len(sim._pending_proposals) == 20  # capped
        assert ids[0] not in sim._pending_proposals  # oldest evicted
        assert ids[1] in sim._pending_proposals
        assert ids[-1] in sim._pending_proposals
        with pytest.raises(sim.ProposalNotFound):
            await sim.confirm_chat_proposal(ids[0])


# ---------------------------------------------------------------------------
# 10. Clone-guard parity (mirrors TestCloneMutualExclusion in test_db_switch)
# ---------------------------------------------------------------------------

class TestCloneGuardParity:
    async def test_confirm_409_while_clone_runs_dev_routed(
            self, client, mem_db, dev_engine, monkeypatch, llm_backend):
        body = await _http_chat(client, monkeypatch, _ACTION_BUY_500,
                                prices={"AAPL": 100.0})
        pid = body["proposals"][0]["proposal_id"]

        monkeypatch.setattr(main, "_db_cloning", True)
        client.cookies.set("ts_db", "dev")
        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": pid})
        assert r.status_code == 409, r.text
        assert "clone is running" in r.json()["detail"]

        # The guard fired before the endpoint body: the proposal is still
        # pending, and a PROD-routed confirm proceeds.
        client.cookies.delete("ts_db")
        monkeypatch.setattr(main, "_db_cloning", False)
        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": pid})
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True
        assert r.json()["trades"]

    async def test_confirm_proceeds_prod_routed_while_clone_runs(
            self, client, mem_db, dev_engine, monkeypatch, llm_backend):
        body = await _http_chat(client, monkeypatch, _ACTION_BUY_500,
                                prices={"AAPL": 100.0})
        pid = body["proposals"][0]["proposal_id"]

        # A clone swaps only the DEV file: PROD-routed confirms never see the
        # guard, even while it is set.
        monkeypatch.setattr(main, "_db_cloning", True)
        r = await client.post("/api/sim/chat/confirm", json={"proposal_id": pid})
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True
        assert len(r.json()["trades"]) == 1


# ---------------------------------------------------------------------------
# 11. GET /api/run-config (slice 3F)
# ---------------------------------------------------------------------------

class TestRunConfigEndpoint:
    async def test_run_config_shape(self, client, mem_db):
        r = await client.get("/api/run-config")
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body.keys()) == {"sim", "monthly", "daily_core"}
        for key in ("sim", "monthly", "daily_core"):
            assert body[key]["engine"] == key
            assert body[key]["epoch"] >= 1
        # Each value is the engine's own current_config() plus epoch/engine.
        assert body["sim"]["sim_strategy"] == settings.sim_strategy
        assert body["sim"]["universe"]["name"] == settings.sim_universe
        assert body["sim"]["universe"]["members_hash"]
        assert body["sim"]["universe_members"]
        assert "variant" in body["monthly"]
        assert "mom_variant" in body["daily_core"]
        assert "gate" in body["daily_core"]

    async def test_run_config_dev_routed_follows_cookie(
            self, client, mem_db, dev_engine, monkeypatch, tmp_path):
        """sim.current_config() resolves the universe from the DB-routed state
        file, so a dev-routed request reports the DEV universe selection."""
        import app.daily_core as daily_core_mod

        dev_state = tmp_path / "daily_core_state_dev.json"
        monkeypatch.setattr(daily_core_mod, "_STATE_FILE_DEV", dev_state)
        dev_state.write_text(json.dumps({"universe": "sp500"}))

        prod = (await client.get("/api/run-config")).json()
        assert prod["sim"]["universe"]["name"] == settings.sim_universe
        assert prod["sim"]["universe"]["source"] == "file"

        client.cookies.set("ts_db", "dev")
        try:
            dev = (await client.get("/api/run-config")).json()
            assert dev["sim"]["universe"]["name"] == "sp500"
            assert dev["sim"]["universe"]["source"] == "file"
            assert dev["sim"]["universe_members"]  # resolved from sp500.txt
            # The shared daily-core selection moves monthly/daily-core too.
            assert dev["monthly"]["universe"]["name"] == "sp500"
            assert dev["daily_core"]["universe"]["name"] == "sp500"
            for key in ("sim", "monthly", "daily_core"):
                assert dev[key]["engine"] == key
                assert dev[key]["epoch"] >= 1
        finally:
            client.cookies.delete("ts_db")
