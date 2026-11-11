"""Sizing clock (session fill contract rule 1): a session's order target and
per-name gaps must read known closes <= S, never F's own close.

The regression: ``run_deployment`` valued the book with ``valuate(now=F)``
(and re-valued it with F after every BUY), and ``backfill`` sized with
``px_of(t, d)`` — F's own close. Mutating ONLY F's close, with the <= S
history and F's opens untouched, therefore changed the F-day trade sizes
(e.g. a $1500 book holding 5 shares at S-close 100, target 2, F-open 100
bought 5/10 shares at F-close 100 but 4/11 at F-close 140).

These tests pin the repair:

* live top-up: a funded, nonempty book with several picks — the F-day trade
  tuples are identical under the F-close mutation;
* live replacement: a same-session band release + replacement — identical
  F-day trade tuples too;
* rule 3 stays REPORTING: the returned valuation is the F-close book in both
  the blocked and the successful payload (distinct from the S sizing marks);
* persisted replay: an F-day top-up AND an F-day replacement have identical
  trades but a different end-of-day snapshot, and the mutated close is
  consumed as S by the NEXT session (no observation freeze);
* the live clock uses the SAME session the ranking resolved (``_decision_session``
  is ``monthly._prev_trading_day``), and the target-vol return history is
  bounded to snapshots created before the session's start.

Bars are VALID OHLC in the live tests (high >= both the base and the mutated
close, low <= both; the open is unchanged by the mutation) so the regression
does not depend on malformed candles.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, monthly, run_config
from app.config import settings
from app.db import (
    Base,
    Candle,
    DailyCoreAccount,
    DailyCoreAllowance,
    DailyCorePosition,
    DailyCoreRanking,
    DailyCoreSnapshot,
    DailyCoreTrade,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite for BOTH engine modules (the daily-core engine and
    ``monthly``'s price-map/FX helpers share one cache)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(daily_core, "Session", session_factory)
    monkeypatch.setattr(monthly, "Session", session_factory)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)
    # Isolation safety: if any path ever computes the ranking frame, its
    # fundamentals/EDGAR reads must stay in the in-memory DB too — never the
    # process default engine (/data). A valid cache below keeps this inert.
    import app.edgar as edgar_mod
    import app.fundamentals as fund_mod
    monkeypatch.setattr(fund_mod, "Session", session_factory)
    monkeypatch.setattr(edgar_mod, "Session", session_factory)

    yield session_factory
    await engine.dispose()


@pytest.fixture
def live_env(mem_db, monkeypatch, tmp_path):
    """Patch the live decision path's environment: universe, state file and
    the ``monthly`` mark clock (``_price_usd_map`` bounds its "latest close"
    lookback with ``monthly._now()``, so it is pinned to the synthetic tape)."""
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: list(LIVE_TICKERS))
    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: list(LIVE_TICKERS))
    monkeypatch.setattr(monthly, "_now", lambda: F)
    return mem_db


# ---------------------------------------------------------------------------
# Synthetic live tape (valid OHLC, F's close optionally mutated)
# ---------------------------------------------------------------------------

LIVE_TICKERS = ["A01", "B02", "C03"]
TAPE_DAYS = pd.DatetimeIndex(["2025-01-31", "2025-02-03", "2025-02-04",
                              "2025-02-05", "2025-02-06"])
F_DAY = pd.Timestamp("2025-02-05")
F = datetime(2025, 2, 5, 21, 0, tzinfo=UTC)     # the cycle's own session
F2 = datetime(2025, 2, 6, 21, 0, tzinfo=UTC)    # the next cycle's session
S = pd.Timestamp("2025-02-04")                  # previous completed session


def _frames(per_ticker: dict[str, dict[str, float]],
            f_closes: dict[str, float] | None = None,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A flat 5-session tape from per-ticker values.

    ``per_ticker[t]`` carries ``s`` (the mark on every session <= S, i.e. the
    S close), ``open`` (F's strict open — unchanged by the mutation) and ``f``
    (the BASE F close). ``f_closes`` overrides selected tickers' F close: that
    is the ONLY difference between the base and mutated runs.

    The high/low columns are computed from every possible value of the
    ticker, so each bar stays VALID OHLC in BOTH runs (high >= max(open,
    every close), low <= min(open, every close)).
    """
    cols = list(per_ticker)
    opens = pd.DataFrame({t: per_ticker[t]["open"] for t in cols}, index=TAPE_DAYS)
    close = pd.DataFrame({t: per_ticker[t]["s"] for t in cols}, index=TAPE_DAYS)
    for t in cols:
        close.loc[F_DAY, t] = per_ticker[t]["f"]
    for t, px in (f_closes or {}).items():
        close.loc[F_DAY, t] = px
    vol = pd.DataFrame(1e6, index=TAPE_DAYS, columns=cols)
    return opens, close, vol


def _candles(per_ticker: dict[str, dict[str, float]],
             opens: pd.DataFrame, close: pd.DataFrame,
             f_closes: dict[str, float] | None = None) -> list[Candle]:
    rows: list[Candle] = []
    for d in TAPE_DAYS:
        for t, spec in per_ticker.items():
            values = [spec["s"], spec["open"], spec["f"],
                      (f_closes or {}).get(t, spec["f"])]
            rows.append(Candle(
                ticker=t, timestamp=d.to_pydatetime(),
                open=float(opens.at[d, t]), close=float(close.at[d, t]),
                high=float(max(values)), low=float(min(values)), volume=1e6))
    return rows


async def _wipe(session_factory) -> None:
    async with session_factory() as s:
        for model in (Candle, DailyCoreTrade, DailyCorePosition,
                      DailyCoreSnapshot, DailyCoreAllowance, DailyCoreAccount,
                      DailyCoreRanking):
            await s.execute(sa_delete(model))
        await s.commit()


async def _seed_live(session_factory, per_ticker, f_closes, *, cash,
                     positions, band, ranking_date="2025-01-31") -> None:
    """Seed the live book with a KNOWN-MATCHING ranking cache.

    The cache row must carry the SAME paper configuration the cycle resolves:
    the ME-match gate (``daily_core._ranking_matches``) correctly invalidates
    an UNKNOWN provenance (legacy NULL ``config_id``), and that recompute
    would leave this fixture's synthetic tape. The effective config is
    captured HERE, i.e. after the scenario applied its state/universe/settings
    patches (the same point ``compute_targets`` captures it).
    """
    opens, close, _vol = _frames(per_ticker, f_closes)
    async with session_factory() as s:
        # Caller-owned session: the config row commits with the seed below.
        stamp = {
            "config_id": await run_config.get_or_create(
                "daily_core", "paper", daily_core.current_config(), session=s),
            "epoch": await run_config.current_epoch("daily_core", session=s),
        }
        s.add_all(_candles(per_ticker, opens, close, f_closes))
        s.add(DailyCoreRanking(
            id=1,
            ranking_date=pd.Timestamp(ranking_date).to_pydatetime(),
            band=",".join(band), picks="",
            mode="paper", config_id=stamp["config_id"],
            epoch=stamp["epoch"]))
        acc = await daily_core._account(s)
        acc.cash = cash
        acc.last_allowance_month = "2025-02"
        for t, shares, avg in positions:
            s.add(DailyCorePosition(ticker=t, shares=shares, avg_cost=avg,
                                    opened_at=datetime(2025, 1, 15)))
        await s.commit()


def _legs(trades: list[dict]) -> list[tuple[str, str, float, float]]:
    return [(t["side"], t["ticker"], round(float(t["shares"]), 6),
             round(float(t["price"]), 6)) for t in trades]


# ---------------------------------------------------------------------------
# Live: the F-day order sizes never read F's close
# ---------------------------------------------------------------------------


def test_live_topup_sizes_ignore_f_close(mem_db, live_env, monkeypatch):
    """Funded book + multiple picks: F-close 100 -> 140 (same <= S history and
    F opens) must NOT change the F-day trade tuples. Sizing is sensitive on
    the old code: F marks 700/500 put the weight at 900, buying 2/4/9 instead
    of the S-mark 3.3333/3.3333/8.3333."""
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 3)
    per_ticker = {
        "A01": {"s": 100.0, "open": 100.0, "f": 100.0},
        "B02": {"s": 100.0, "open": 100.0, "f": 100.0},
        "C03": {"s": 100.0, "open": 100.0, "f": 100.0},
    }

    async def scenario(f_closes):
        await _wipe(mem_db)
        await _seed_live(mem_db, per_ticker, f_closes, cash=1500.0,
                         positions=[("A01", 5.0, 100.0), ("B02", 5.0, 100.0)],
                         band=["A01", "B02", "C03"])
        r = await daily_core.run_deployment(now=F)
        return r, _legs(r["trades"])

    base, base_legs = asyncio.run(scenario(None))
    mut, mut_legs = asyncio.run(scenario({"A01": 140.0}))

    # THE REPAIR: identical F-day fills under the F-close mutation.
    assert base_legs == mut_legs
    assert base_legs == [
        ("BUY", "A01", 3.3333, 100.0),
        ("BUY", "B02", 3.3333, 100.0),
        ("BUY", "C03", 8.3333, 100.0),
    ]
    # All three picks are topped up toward weight = S equity 2500 / 3.
    assert base["valuation"]["cash"] == pytest.approx(0.01, abs=0.02)

    # Rule 3 is REPORTING: the returned valuation (and the snapshot clock)
    # stays the F-close book — it differs exactly by the mutated close.
    assert base["valuation"]["total_equity"] == pytest.approx(2500.0, abs=0.02)
    assert mut["valuation"]["total_equity"] == pytest.approx(2833.33, abs=0.02)

    async def size_marks():
        return await daily_core._sizing_valuation(S)
    sizing = asyncio.run(size_marks())
    assert sizing["total_equity"] == pytest.approx(2500.0, abs=0.02)


def test_live_replacement_sizes_ignore_f_close(mem_db, live_env, monkeypatch):
    """A successful band release + same-session replacement: mutating only the
    retained name's F close must not change the session's SELL/BUY sizes. The
    old F-close weight (2000/2 = 1000) bought 3/25 shares; the S-mark weight
    (1800/2 = 900) buys 4/22.5."""
    monkeypatch.setattr(settings, "sim_monthly_target_n", 2)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 3)
    per_ticker = {
        "A01": {"s": 100.0, "open": 100.0, "f": 100.0},
        "B02": {"s": 60.0, "open": 60.0, "f": 60.0},
        "C03": {"s": 40.0, "open": 40.0, "f": 40.0},
    }
    # The month-end band switch: A01 retained, B02 out of the band, C03 the
    # best-ranked free name. The band is CURRENT here, so the release and the
    # replacement both execute this session.
    band = ["A01", "C03", "DDD"]

    async def scenario(f_closes):
        await _wipe(mem_db)
        await _seed_live(mem_db, per_ticker, f_closes, cash=1000.0,
                         positions=[("A01", 5.0, 100.0), ("B02", 5.0, 60.0)],
                         band=band)
        return await daily_core.run_deployment(now=F)

    base = asyncio.run(scenario(None))
    mut = asyncio.run(scenario({"A01": 140.0}))

    assert _legs(base["trades"]) == _legs(mut["trades"])
    assert _legs(base["trades"]) == [
        ("SELL", "B02", 5.0, 60.0),
        ("BUY", "A01", 4.0, 100.0),
        ("BUY", "C03", 22.5, 40.0),
    ]
    # F-close reporting differs (held A01 marked at 140 after the fills).
    assert base["valuation"]["total_equity"] == pytest.approx(1800.0, abs=0.02)
    assert mut["valuation"]["total_equity"] == pytest.approx(2160.0, abs=0.02)


def test_live_blocked_payload_reports_f_close_not_sizing_marks(
        mem_db, live_env, monkeypatch):
    """The BLOCKED branch returns before the BUY loop, but rule 3 still owns
    the payload: it must report the F-close book (the S sizing marks would say
    1500 + 5 * 100 = 2000; F's close says 1500 + 5 * 140 = 2200)."""
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 3)
    per_ticker = {
        "A01": {"s": 100.0, "open": 100.0, "f": 100.0},
        "B02": {"s": 100.0, "open": 100.0, "f": 100.0},
        "C03": {"s": 100.0, "open": 100.0, "f": 100.0},
    }
    opens, close, vol = _frames(per_ticker, {"A01": 140.0})

    async def fake_load_frames(_tickers, _asof, start=None):
        return close, vol

    async def blocked(*_args, **_kwargs):
        return False, False

    monkeypatch.setattr(monthly, "load_frames", fake_load_frames)
    monkeypatch.setattr(daily_core, "_protection_decision", blocked)
    daily_core.set_gate("200")   # arms the overlay block on the state file

    async def scenario():
        await _wipe(mem_db)
        await _seed_live(mem_db, per_ticker, {"A01": 140.0}, cash=1500.0,
                         positions=[("A01", 5.0, 100.0)],
                         band=["A01", "B02", "C03"])
        return await daily_core.run_deployment(now=F)

    r = asyncio.run(scenario())
    assert r["blocked"]
    assert r["trades"] == []
    # F-close reporting: 1500 cash + 5 * 140 (never the S mark 100).
    assert r["valuation"]["total_equity"] == pytest.approx(2200.0, abs=0.02)
    a01 = next(p for p in r["valuation"]["positions"] if p["ticker"] == "A01")
    assert a01["current_price"] == pytest.approx(140.0)


def test_live_next_session_consumes_the_close_as_s(mem_db, live_env, monkeypatch):
    """No observation freeze: F's close is forbidden for F's own sizing, but
    the NEXT session's S IS F, so the mutated close sizes that session's
    top-up (and the earlier session's trades stay identical)."""
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 2)
    per_ticker = {
        "A01": {"s": 100.0, "open": 100.0, "f": 100.0},
        "B02": {"s": 50.0, "open": 50.0, "f": 50.0},
    }

    async def scenario(f_closes):
        await _wipe(mem_db)
        await _seed_live(mem_db, per_ticker, f_closes, cash=1000.0,
                         positions=[], band=["A01", "B02"])
        r1 = await daily_core.run_deployment(now=F)
        r2 = await daily_core.run_deployment(now=F2)
        return _legs(r1["trades"]), _legs(r2["trades"])

    base_f, base_next = asyncio.run(scenario(None))
    mut_f, mut_next = asyncio.run(scenario({"A01": 140.0}))

    # session F: identical (its S is untouched by the mutation)
    assert base_f == mut_f
    assert len(base_f) == 2
    # session F2: its S is F's close, so the mutated run tops up B02 further
    assert base_next == []
    assert [leg[:2] for leg in mut_next] == [("BUY", "B02")]
    assert mut_next[0][2] > 0.0


def test_live_decision_session_is_the_ranking_session(mem_db, live_env):
    """The sizing clock is the SAME ``S`` the ranking resolved — the shared
    session calendar (``monthly._prev_trading_day``), not a candle-frame
    fallback."""
    assert daily_core._decision_session(F) == S
    assert monthly._prev_trading_day(F) == S.to_pydatetime()


def test_target_vol_history_excludes_the_same_sessions_snapshot(mem_db):
    """The target-vol return history is an S input. A snapshot the SAME
    session already wrote is F-close REPORTING (a forced second cycle) and
    must never be fed back into the F budget as an S return."""
    async def seed():
        async with mem_db() as s:
            for day, eq in (("2025-02-03", 1000.0), ("2025-02-04", 1010.0),
                            ("2025-02-05", 1050.0)):
                s.add(DailyCoreSnapshot(
                    cash=0.0, positions_value=eq, total_equity=eq,
                    allowance_total=0.0,
                    created_at=datetime.fromisoformat(
                        f"{day}T23:30:00+00:00")))
            await s.commit()

    asyncio.run(seed())
    session_start = datetime(2025, 2, 5, tzinfo=UTC)   # F's session start

    async def run():
        return (await daily_core._recent_port_returns(21, before=session_start),
                await daily_core._recent_port_returns(21))

    bounded, unbounded = asyncio.run(run())
    assert len(bounded) == 1     # 02-03 -> 02-04 only (sessions <= S)
    assert len(unbounded) == 2   # ...plus F's own 02-04 -> 02-05 point
    assert bounded != unbounded


# ---------------------------------------------------------------------------
# Persisted replay: identical F-day trades, different EOD snapshot
# ---------------------------------------------------------------------------

REPLAY_TICKERS = ["AAA", "BBB", "CCC"]
REPLAY_START = "2025-12-01"
REPLAY_END = "2026-02-06"

# Month-end decisions: December holds AAA/BBB, January replaces BBB with CCC.
REPLAY_ORDERS = {
    pd.Timestamp("2025-12-31"): ["AAA", "BBB"],
    pd.Timestamp("2026-01-30"): ["AAA", "CCC"],
}


def _replay_frames(f_close: dict[str, dict[str, float]] | None = None):
    """Synthetic replay tape (TRUE NYSE sessions): flat opens, flat closes with
    one deliberate drift (BBB 50 -> 30 on 2026-01-02, so a top-up is due on
    2026-01-05) and optional single-day F-close overrides
    (``{day: {ticker: close}}``)."""
    from app import sim as sim_mod

    days = pd.DatetimeIndex([
        d for d in pd.bdate_range(REPLAY_START, REPLAY_END)
        if sim_mod.is_trading_day(d.to_pydatetime())])
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0, "CCC": 40.0}, index=days)
    close.loc[pd.Timestamp("2026-01-02"), "BBB"] = 30.0
    for day, overrides in (f_close or {}).items():
        for ticker, px in overrides.items():
            close.loc[pd.Timestamp(day), ticker] = px
    open_px = pd.DataFrame({"AAA": 100.0, "BBB": 50.0, "CCC": 40.0},
                           index=days)
    vol = pd.DataFrame(1e6, index=days, columns=REPLAY_TICKERS)
    return open_px, close, vol


def _fake_frame(m, _close, _vol, _fund):
    order = REPLAY_ORDERS.get(pd.Timestamp(m).normalize())
    if order is None:
        return None
    frame = pd.DataFrame({"eligible": [True] * len(order)},
                         index=pd.Index(order, name="ticker"))
    frame.attrs["order"] = order
    return frame


def _fake_order(frame):
    return frame["eligible"], list(frame.attrs["order"])


@pytest.fixture
def replay_env(mem_db, monkeypatch, tmp_path):
    """Patch the backfill's loaders/ranking so the replay walks the synthetic
    tape; the state file is per-test (never a shared literal path)."""
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: list(REPLAY_TICKERS))
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-02")
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 2)

    async def fake_fund(_tickers):
        return {t: {"x": []} for t in REPLAY_TICKERS}
    monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals", fake_fund)
    monkeypatch.setattr(monthly, "eligible_frame", _fake_frame)
    monkeypatch.setattr(monthly, "_qv_order", _fake_order)
    return mem_db


def _run_replay(mem_db, monkeypatch, f_close=None):
    """Replay the tape and return (trades by day, snapshot equity by day)."""
    open_px, close, vol = _replay_frames(f_close)

    async def fake_load_ohlcv(_tickers, _asof, start=None):
        return open_px, close, vol
    monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv)

    async def run():
        r = await daily_core.backfill(start=REPLAY_START)
        assert r["ok"] is True, r
        async with mem_db() as s:
            rows = (await s.scalars(
                select(DailyCoreTrade).order_by(DailyCoreTrade.id))).all()
            trades: dict[str, list[tuple]] = {}
            for tr in rows:
                trades.setdefault(tr.created_at.strftime("%Y-%m-%d"), []).append(
                    (tr.side, tr.ticker, round(float(tr.shares), 6),
                     round(float(tr.price), 6)))
            snaps = (await s.scalars(
                select(DailyCoreSnapshot)
                .order_by(DailyCoreSnapshot.created_at))).all()
            equity = {sn.created_at.strftime("%Y-%m-%d"):
                      round(float(sn.total_equity), 2) for sn in snaps}
        return trades, equity

    return asyncio.run(run())


def test_replay_topup_and_eod_are_different_clocks(replay_env, monkeypatch):
    """2026-01-05 is a top-up day (BBB drifted to 30 at the S close). Mutating
    ONLY that session's AAA close keeps the F-day trade tuples identical while
    the EOD snapshot changes — and the mutation never leaks backwards."""
    base_trades, base_eq = _run_replay(replay_env, monkeypatch)
    mut_trades, mut_eq = _run_replay(
        replay_env, monkeypatch, {"2026-01-05": {"AAA": 140.0}})

    topup_day = "2026-01-05"
    assert base_trades[topup_day] == [
        ("BUY", "BBB", 3.5555, 50.0),
    ]
    # THE REPAIR: the F-close mutation changes neither the leg nor its size.
    assert base_trades[topup_day] == mut_trades[topup_day]
    # Rule 3 reporting: the EOD equity DOES see the mutated close (AAA held
    # 6.6666 sh: +266.66), because reporting is valued at F's close.
    assert base_eq[topup_day] == pytest.approx(2000.0, abs=0.02)
    assert mut_eq[topup_day] == pytest.approx(2266.66, abs=0.02)


def test_replay_replacement_trades_identical_eod_differs(replay_env, monkeypatch):
    """2026-02-02 is a band-switch session: BBB is released and CCC bought the
    SAME session. Mutating only that session's retained AAA close keeps the
    SELL+BUY tuples identical but changes the EOD snapshot; the next session
    consumes the mutated close as its S (a CCC top-up appears)."""
    base_trades, base_eq = _run_replay(replay_env, monkeypatch)
    mut_trades, mut_eq = _run_replay(
        replay_env, monkeypatch, {"2026-02-02": {"AAA": 140.0}})

    day = "2026-02-02"
    # The release, the retained-name top-up and the replacement all execute at
    # this session's strict opens (this session is also February's first, so
    # the month's contribution is in the sizing cash).
    assert base_trades[day] == [
        ("SELL", "BBB", 16.8888, 50.0),
        ("BUY", "AAA", 3.3334, 100.0),
        ("BUY", "CCC", 25.0, 40.0),
    ]
    assert base_trades[day] == mut_trades[day]
    # Rule 3 reporting: 10 AAA shares marked at F's mutated close (140) rather
    # than the S mark (100) — +400 on the session's snapshot.
    assert base_eq[day] == pytest.approx(3000.0, abs=0.02)
    assert mut_eq[day] == pytest.approx(3400.0, abs=0.02)

    # No observation freeze: at 2026-02-03 the mutated close is history (S =
    # 2026-02-02), so the sizing valuation is larger and CCC is topped up;
    # the base run has nothing left to top up.
    next_day = "2026-02-03"
    assert base_trades.get(next_day, []) == []
    assert [leg[:2] for leg in mut_trades[next_day]] == [("BUY", "CCC")]
    assert mut_trades[next_day][0][2] > 0.0
