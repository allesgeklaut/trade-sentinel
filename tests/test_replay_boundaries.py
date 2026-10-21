"""Replay boundary contract tests (audit P0-D/P0-E + review rounds A-D).

The daily-core replay and its research backtest must behave like the live
engine at calendar boundaries:

- every day ranks against the most recent COMPLETED month-end at or before
  it — the PRE-window month-end frame for days before the window's first
  month-end, never a future frame (audit P0-E);
- a month rebalances only on its US session calendar month-end, and only
  when that candle is stored — never on a partial month's last candle;
  appending future candles cannot change any decision inside the window;
- a window with no in-window month-end (short partial window) still runs
  and deploys against the pre-window frame;
- deployment funds the hysteresis-selected picks (ten-slot target),
  rank-first, never the raw ``band[:target_n]`` (audit P0-D), and gated
  entries respect capacity + selected membership.

The tests use flat synthetic prices and canned eligibility frames (roe
distinct per rank, everything else equal) so ``_qv_order`` is fully
determined and every printed trade is exact.
"""

from __future__ import annotations

import asyncio
import re

import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, optimize
from app import monthly as monthly_mod
from app.config import settings
from app.db import Base

# ------------------------------------------------------------------
# Deterministic market + canned eligibility frames
# ------------------------------------------------------------------

TICKERS = ["A01", "B02", "C03", "D04", "E05", "F06", "G07", "H08"]


def _canned_frame(order: list[str]) -> pd.DataFrame:
    """A frame whose ``_qv_order`` equals `order`: roe decays with rank
    (score-dominant), mom/p_fcf/vol equal for all so roe_rank decides and
    roe is the first tie-break."""
    roe = {t: 0.5 - 0.01 * i for i, t in enumerate(order)}
    rows = {t: {"roe": roe.get(t, 0.001), "p_fcf": 20.0, "mcap": 1e10,
                "dollar_vol": 1e8, "mom": 0.1, "price": 100.0,
                "vol": 0.2, "eligible": True}
            for t in TICKERS}
    return pd.DataFrame(rows).T


def _flat_market(last_day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # History starts well before the replay windows so the 253-day momentum
    # warmup passes; all prices flat at 100.
    dates = pd.bdate_range("2023-06-01", last_day)
    close = pd.DataFrame(100.0, index=dates, columns=TICKERS)
    vol = pd.DataFrame(1e6, index=dates, columns=TICKERS)
    return close, vol


@pytest.fixture
def small_slots(monkeypatch):
    """Shrink the slot knobs so the 8-name market exercises band + capacity
    logic."""
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 6)


@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory daily-core DB (same isolation as test_daily_core's mem_db)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(daily_core, "Session", session_factory)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)

    yield session_factory
    await engine.dispose()


@pytest.fixture
def jumped_market(monkeypatch):
    """close with per-name jumps: A01 jumps on 2025-02-10, B02 on 2025-03-10,
    C03 on 2025-03-14 (each opens the above-sma50 entry gate for that name
    on that day)."""
    close, vol = _flat_market("2025-03-14")
    close.loc["2025-02-10":, "A01"] = 130.0
    close.loc["2025-03-10":, "B02"] = 130.0
    close.loc["2025-03-14":, "C03"] = 130.0
    return close, vol


def _run_optimizer(monkeypatch, close, vol, frames, *, start, end=None,
                   dca="rank", entry_gate="none", exit_cadence="monthly",
                   verbose=True, frame_cache=None):
    """Run the deterministic daily-core backtest on canned frames."""
    monkeypatch.setattr(
        monthly_mod, "eligible_frame",
        lambda d, c, v, f: frames.get(pd.Timestamp(d).normalize()))
    return asyncio.run(optimize._daily_core_backtest(
        start, end, "diversified-plus", 1000.0, verbose,
        entry_gate, exit_cadence, False, dca,
        frame_cache=frame_cache, close_preloaded=(close, vol),
        fund_preloaded={}, with_baseline=False))


_TRADE_RE = re.compile(r"^  (\d{4}-\d{2}-\d{2})\s+(BUY|SELL)\s+(\S+)\s+\$")


def _printed_trades(capsys) -> list[tuple[str, str, str]]:
    """Parse the verbose trade log: ``  DATE SIDE TICKER $notional``."""
    out: list[tuple[str, str, str]] = []
    for line in capsys.readouterr().out.splitlines():
        m = _TRADE_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


STABLE_ORDER = TICKERS  # A01..H08


# ------------------------------------------------------------------
# Prior-window ranking (audit P0-E)
# ------------------------------------------------------------------

class TestPriorWindowRanking:
    def test_mid_month_window_ranks_against_prior_frame(
            self, small_slots, monkeypatch, capsys):
        """A window starting mid-month ranks its first days against the
        PRE-window (Dec-31) month-end frame. The old ``month_for`` fallback
        returned months[0] — the January month-end, a FUTURE frame — so a
        January run traded on January's own ranking."""
        close, vol = _flat_market("2025-02-28")
        rotated = ["D04", "E05", "F06", "G07", "H08", "A01", "B02", "C03"]
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): _canned_frame(rotated),
            pd.Timestamp("2025-02-28"): _canned_frame(rotated),
        }
        _run_optimizer(monkeypatch, close, vol, frames, start="2025-01-15")
        trades = _printed_trades(capsys)

        first_day_buys = [t for (d, side, t) in trades
                          if d == "2025-01-15" and side == "BUY"]
        # the prior frame's top-3, deployed immediately on the window's
        # first day — rank-first
        assert first_day_buys == ["A01", "B02", "C03"]
        # and NOT the January frame's names (the future-frame leak)
        assert not any(t in ("D04", "E05", "F06") for t in first_day_buys)

    def test_month_for_never_returns_the_first_future_month(
            self, small_slots, monkeypatch, capsys):
        """Data starting exactly at the window (no pre-window month-end at
        all): no ranking -> no trades before the first month-end, instead of
        falling back to a future frame."""
        dates = pd.bdate_range("2024-01-02", "2025-02-28")
        close = pd.DataFrame(100.0, index=dates, columns=TICKERS)
        vol = pd.DataFrame(1e6, index=dates, columns=TICKERS)
        # only 2024-01-31 carries a frame — the later months run frame-less
        frames = {pd.Timestamp("2024-01-31"): _canned_frame(TICKERS),
                  pd.Timestamp("2025-01-31"): _canned_frame(TICKERS),
                  pd.Timestamp("2025-02-28"): _canned_frame(TICKERS)}
        res = _run_optimizer(monkeypatch, close, vol, frames,
                             start="2024-01-02")
        trades = _printed_trades(capsys)
        # Jan 2-30 has NO prior month-end frame: no trades at all —
        # months[0] (Jan-31) is never leaked backwards
        assert not any(d < "2024-01-31" for (d, _, _) in trades)
        # from the January month-end onward the ranking exists and deploys
        assert any(d == "2024-01-31" for (d, side, _) in trades
                   if side == "BUY")
        assert res is not None


# ------------------------------------------------------------------
# Future-data invariance incl. the actual month-end endpoint
# ------------------------------------------------------------------

class TestFutureDataInvariance:
    def _frames(self, months: list[str], order: list[str]) -> dict:
        return {pd.Timestamp(m): _canned_frame(order) for m in months}

    def test_truncation_at_month_end_endpoint_keeps_decisions(
            self, small_slots, monkeypatch, capsys):
        """Candles truncated EXACTLY at the January month-end: the endpoint
        still rebalances (the old later-candle rule LOST the truncated
        run's own month-end) and every decision through it is identical."""
        close, vol = _flat_market("2025-02-28")
        rotated = ["D04", "E05", "F06", "G07", "H08", "A01", "B02", "C03"]
        frames = self._frames(["2024-12-31", "2025-01-31", "2025-02-28"],
                              STABLE_ORDER)
        frames[pd.Timestamp("2025-01-31")] = _canned_frame(rotated)
        _run_optimizer(monkeypatch, close, vol, frames, start="2025-01-15")
        full_trades = _printed_trades(capsys)

        close_t = close.loc[:"2025-01-31"]
        vol_t = vol.loc[:"2025-01-31"]
        frames_t = {pd.Timestamp("2024-12-31"): frames[pd.Timestamp("2024-12-31")],
                    pd.Timestamp("2025-01-31"): frames[pd.Timestamp("2025-01-31")]}
        _run_optimizer(monkeypatch, close_t, vol_t, frames_t, start="2025-01-15")
        trunc_trades = _printed_trades(capsys)

        cut = "2025-01-31"
        assert [t for t in trunc_trades if t[0] <= cut] == \
            [t for t in full_trades if t[0] <= cut]
        # the endpoint itself rebalances in the truncated run: July's picks
        # B02/C03 fell out of January's band and are released there
        assert any(d == cut and side == "SELL" and t in ("B02", "C03")
                   for (d, side, t) in trunc_trades)

    def test_appended_future_candles_change_nothing_before_the_cut(
            self, small_slots, monkeypatch, capsys):
        """The mirror direction: appending February candles to a frame that
        ends at the January month-end cannot change the January decisions
        (review A: the later-candle completion test made January's
        membership depend on February's existence)."""
        close, vol = _flat_market("2025-02-28")
        frames = self._frames(["2024-12-31", "2025-01-31"], STABLE_ORDER)
        # window ends at the January month-end in BOTH runs
        _run_optimizer(monkeypatch, close, vol, frames,
                       start="2025-01-15", end="2025-01-31")
        with_future = _printed_trades(capsys)

        close_t = close.loc[:"2025-01-31"]
        vol_t = vol.loc[:"2025-01-31"]
        _run_optimizer(monkeypatch, close_t, vol_t, frames,
                       start="2025-01-15", end="2025-01-31")
        without_future = _printed_trades(capsys)

        assert with_future == without_future


# ------------------------------------------------------------------
# No partial-month rebalance; short windows on the prior frame
# ------------------------------------------------------------------

class TestPartialMonthBoundaries:
    def test_window_ending_mid_month_never_rebalances_the_partial_month(
            self, small_slots, monkeypatch, capsys):
        """Data ends mid-March: March has no month-end — no release, no new
        name after the last true month-end (Feb-28); the stretch is only
        mark-to-marketed."""
        close, vol = _flat_market("2025-03-14")
        rotated = ["D04", "E05", "F06", "G07", "H08", "A01", "B02", "C03"]
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): _canned_frame(rotated),
            pd.Timestamp("2025-02-28"): _canned_frame(rotated),
        }
        res = _run_optimizer(monkeypatch, close, vol, frames,
                             start="2025-01-15")
        trades = _printed_trades(capsys)

        # releases happen only on true month-ends
        sells = [d for (d, side, _) in trades if side == "SELL"]
        assert all(d <= "2025-02-28" for d in sells)
        # no NEW name opened after Feb-28 either (capacity: the book is at
        # target and the partial month never refills it)
        held_after_feb = {"A01", "D04", "E05"}  # Jan-31 rebuild outcome
        late_buys = [t for (d, side, t) in trades
                     if d > "2025-02-28" and side == "BUY"]
        assert set(late_buys) <= held_after_feb
        # the replay still covers the partial stretch
        assert res.window[1] == "2025-03-14"

    def test_short_no_rebalance_window_deploys_on_prior_frame(
            self, small_slots, monkeypatch, capsys):
        """A Jan-02..Jan-15 window has no month-end of its own but still
        runs: it deposits on its first day and deploys against Dec-31's
        frame (review B — the old code returned early instead)."""
        close, vol = _flat_market("2025-02-28")
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-02-28"): _canned_frame(STABLE_ORDER),
        }
        res = _run_optimizer(monkeypatch, close, vol, frames,
                             start="2025-01-02", end="2025-01-15")
        trades = _printed_trades(capsys)

        buys = [(d, t) for (d, side, t) in trades if side == "BUY"]
        assert ("2025-01-02", "A01") in buys          # deployed immediately
        assert {t for (_, t) in buys} == {"A01", "B02", "C03"}
        assert res.contributed == pytest.approx(1000.0)  # one contribution
        assert not any(side == "SELL" for (_, side, _) in trades)


# ------------------------------------------------------------------
# Ten-slot contract inside the research replay
# ------------------------------------------------------------------

class TestReplayTenSlotContract:
    def test_rebuild_releases_and_fills_in_rank_order(
            self, small_slots, monkeypatch, capsys):
        """At the January month-end the ranking rotates: the held band
        members keep their slots, the out-of-band releases free exactly
        their slots, and the fills are bought rank-first. The book never
        exceeds the target between month-ends."""
        close, vol = _flat_market("2025-02-28")
        rotated = ["D04", "E05", "F06", "G07", "H08", "A01", "B02", "C03"]
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): _canned_frame(rotated),
            pd.Timestamp("2025-02-28"): _canned_frame(rotated),
        }
        _run_optimizer(monkeypatch, close, vol, frames, start="2025-01-15")
        trades = _printed_trades(capsys)

        jan31_sells = [t for (d, side, t) in trades
                       if d == "2025-01-31" and side == "SELL"]
        jan31_buys = [t for (d, side, t) in trades
                      if d == "2025-01-31" and side == "BUY"]
        # releases: B02/C03 fell out of the 6-band; fills: D04/E05 (the top
        # of the ranking) — NOT the alphabetical-first names
        assert set(jan31_sells) == {"B02", "C03"}
        assert set(jan31_buys) == {"D04", "E05"}
        # between month-ends: no new name while the book is at capacity
        held_after_jan31 = {"A01", "D04", "E05"}
        mid_buys = [t for (d, side, t) in trades
                    if "2025-01-31" < d < "2025-02-28" and side == "BUY"]
        assert set(mid_buys) <= held_after_jan31


# ------------------------------------------------------------------
# Failed release: executors must guard the ACTUAL free capacity (round 3)
# ------------------------------------------------------------------

class TestPendingReleaseExecutionGuard:
    """``_deploy_targets(refill=True)`` plans the POST-release membership —
    its fill count assumes the pending releases succeed (9 held, 8 in-band,
    1 pending -> the plan queues 2 replacements, not 1). The executors must
    therefore hard-guard every new-name buy with the ACTUAL free capacity
    (``len(shares) < target_n``): a release that FAILS leaves its slot
    occupied, and buying the full plan would create an over-capacity book.
    px_of forward-fills, so the reachable failed-release lever in a replay
    is a corrupt 0.0 close (trade() SELL guards ``p <= 0``)."""

    def test_failed_release_occupies_slot_and_only_one_fill_buys(
            self, small_slots, monkeypatch, capsys):
        close, vol = _flat_market("2025-02-28")
        rotated = ["A01", "D04", "E05", "F06", "G07", "H08", "B02", "C03"]
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-02-28"): _canned_frame(rotated),
        }
        # Corrupt feed: B02's closes go to 0.0 late in February, so its
        # Feb-28 release FAILS (SELL guards p <= 0) and B02 stays held
        # out-of-band — exactly the pending-release state the plan is
        # optimistic about.
        close.loc["2025-02-21":, "B02"] = 0.0

        _run_optimizer(monkeypatch, close, vol, frames,
                       start="2025-01-02", end="2025-02-28", dca="monthly")
        trades = _printed_trades(capsys)

        # book simulation: the held set never exceeds the target
        held: set[str] = set()
        for (_, side, t) in trades:
            if side == "BUY":
                held.add(t)
            else:
                held.discard(t)
            assert len(held) <= 3
        # Feb-28: C03 released successfully, B02's release FAILED (stays);
        # the plan queued TWO fills (D04, E05) but only ONE slot was free —
        # the capacity guard skipped E05 instead of growing the book to 4
        feb28 = [(side, t) for (d, side, t) in trades if d == "2025-02-28"]
        assert ("SELL", "C03") in feb28
        assert ("SELL", "B02") not in feb28
        assert ("BUY", "D04") in feb28          # the one affordable fill
        assert ("BUY", "A01") in feb28          # held top-up (never capped)
        assert not any(side == "BUY" and t == "E05"
                       for (_, side, t) in trades)
        assert held == {"A01", "B02", "D04"}
        # the parked February contribution deploys only at the rebuild
        assert not any("2025-02-03" <= d < "2025-02-28"
                       for (d, _, _) in trades)


# ------------------------------------------------------------------
# Dead month: no future-based trim, causal simulation (round 3)
# ------------------------------------------------------------------

class TestDeadMonthCausality:
    def _dead_frame(self) -> pd.DataFrame:
        # a frame whose eligibility is all-False (fundamentals coverage too
        # thin): current_order yields no ranking for its days
        return _canned_frame(STABLE_ORDER).assign(eligible=False)

    def test_dead_january_still_deploys_on_prior_frame(
            self, small_slots, monkeypatch, capsys):
        """Valid Dec-31 prior + dead Jan-31 + valid Feb-28: January must
        still fund and deploy against the PRIOR frame. The old future-based
        trim cut January's days entirely when Feb's frame was alive, so the
        prior frame never funded Jan-2 and January's contribution vanished
        — the replay's START depended on FUTURE eligibility."""
        close, vol = _flat_market("2025-02-28")
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            # Jan-31's frame missing from the dict -> dead anchor
            pd.Timestamp("2025-02-28"): _canned_frame(STABLE_ORDER),
        }
        res = _run_optimizer(monkeypatch, close, vol, frames,
                             start="2025-01-02", end="2025-02-28")
        trades = _printed_trades(capsys)

        jan2_buys = [t for (d, side, t) in trades
                     if side == "BUY" and d == "2025-01-02"]
        assert jan2_buys == ["A01", "B02", "C03"]  # the prior frame's picks
        assert res is not None
        assert res.contributed == pytest.approx(2000.0)  # Jan NOT defunded
        # the dead January anchor yields no ranking: cash sits in February
        # until Feb-28's valid frame becomes current
        assert not any("2025-01-31" < d < "2025-02-28"
                       for (d, _, _) in trades)
        # Feb-28: the valid ranking resumes — top-ups of the held names
        feb28_buys = [t for (d, side, t) in trades
                      if d == "2025-02-28" and side == "BUY"]
        assert set(feb28_buys) == {"A01", "B02", "C03"}

    def test_future_frames_cannot_change_the_january_prefix(
            self, small_slots, monkeypatch, capsys):
        """Mutating FEBRUARY's frame (or truncating the tape at Jan-31)
        cannot change any January decision: the month ranks against the
        prior frame only."""
        close, vol = _flat_market("2025-02-28")
        rotated = ["D04", "E05", "F06", "G07", "H08", "A01", "B02", "C03"]
        base = {pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER)}
        # run A: dead Jan + stable Feb
        _run_optimizer(monkeypatch, close, vol,
                       {**base, pd.Timestamp("2025-02-28"):
                        _canned_frame(STABLE_ORDER)},
                       start="2025-01-02", end="2025-02-28")
        prefix_a = [t for t in _printed_trades(capsys) if t[0] <= "2025-01-31"]
        # run B: dead Jan + MUTATED Feb frame
        _run_optimizer(monkeypatch, close, vol,
                       {**base, pd.Timestamp("2025-02-28"):
                        _canned_frame(rotated)},
                       start="2025-01-02", end="2025-02-28")
        prefix_b = [t for t in _printed_trades(capsys) if t[0] <= "2025-01-31"]
        # run C: tape truncated AT the (dead) January month-end
        _run_optimizer(monkeypatch, close.loc[:"2025-01-31"],
                       vol.loc[:"2025-01-31"], dict(base),
                       start="2025-01-02", end="2025-01-31")
        prefix_c = [t for t in _printed_trades(capsys) if t[0] <= "2025-01-31"]

        assert prefix_a == prefix_b == prefix_c
        assert [t for t in prefix_a if t[1] == "BUY"] == \
            [("2025-01-02", "BUY", "A01"), ("2025-01-02", "BUY", "B02"),
             ("2025-01-02", "BUY", "C03")]

    def test_all_window_frames_dead_still_runs_on_prior_frame(
            self, small_slots, monkeypatch, capsys):
        """ALL in-window frames dead + a valid prior frame: no early return —
        January deploys on the prior frame, later days hold cash, and the
        run still reports its contributions."""
        close, vol = _flat_market("2025-02-28")
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): self._dead_frame(),
            pd.Timestamp("2025-02-28"): self._dead_frame(),
        }
        res = _run_optimizer(monkeypatch, close, vol, frames,
                             start="2025-01-02", end="2025-02-28")
        trades = _printed_trades(capsys)

        assert res is not None
        assert res.contributed == pytest.approx(2000.0)
        assert set(t for (d, side, t) in trades
                   if side == "BUY" and d <= "2025-01-31") == \
            {"A01", "B02", "C03"}
        assert not any(d > "2025-01-31" for (d, _, _) in trades)


# ------------------------------------------------------------------
# Entry gate: capacity + selected membership on pending re-entry
# ------------------------------------------------------------------

class TestGatedEntryBoundaries:
    def test_gate_parks_until_it_opens_and_never_exceeds_capacity(
            self, small_slots, monkeypatch, capsys, jumped_market):
        """dca=daily + above-sma50: contributions park while the gate is
        closed; each fill buys exactly when ITS gate opens, one at a time,
        and the book never exceeds the target."""
        close, vol = jumped_market
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-02-28"): _canned_frame(STABLE_ORDER),
        }
        _run_optimizer(monkeypatch, close, vol, frames, start="2025-01-15",
                       dca="daily", entry_gate="above-sma50")
        trades = _printed_trades(capsys)

        buys = [(d, t) for (d, side, t) in trades if side == "BUY"]
        # nothing is bought while the gate is closed (the parked
        # contributions are NOT force-deployed)
        assert not any(d < "2025-02-10" for (d, _) in buys)
        # each gated fill opens in its own turn: A01 (02-10), B02 (03-10),
        # C03 (03-14)
        assert {t for (_, t) in buys} == {"A01", "B02", "C03"}
        assert not any(side == "SELL" for (_, side, _) in trades)

    def test_rank_mode_respects_the_entry_gate(
            self, small_slots, monkeypatch, capsys, jumped_market):
        """entry-gate is NOT inert under dca=rank: new names are gated the
        same way (previously the rank deployment ignored the gate and the
        rank+gate sweep arms silently measured gate=none)."""
        close, vol = jumped_market
        frames = {
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-01-31"): _canned_frame(STABLE_ORDER),
            pd.Timestamp("2025-02-28"): _canned_frame(STABLE_ORDER),
        }
        _run_optimizer(monkeypatch, close, vol, frames, start="2025-01-15",
                       dca="rank", entry_gate="above-sma50")
        trades = _printed_trades(capsys)

        buys = [(d, t) for (d, side, t) in trades if side == "BUY"]
        assert not any(d < "2025-02-10" for (d, _) in buys)
        assert {t for (_, t) in buys} == {"A01", "B02", "C03"}


# ------------------------------------------------------------------
# Sweep cache calendar parity
# ------------------------------------------------------------------

class TestSweepCacheParity:
    def test_sweep_months_are_calendar_month_ends(self, small_slots):
        close, _ = _flat_market("2025-03-14")  # mid-March last candle
        got = optimize._sweep_months(close.index, "2016-06-01")
        # March's partial candle is not a month-end; the sweep keys match
        # what the research backtest selects
        assert got[-1] == pd.Timestamp("2025-02-28")
        assert pd.Timestamp("2025-03-14") not in got

    def test_backtest_consumes_sweep_cache_missing_prior_key(
            self, small_slots, monkeypatch, capsys):
        """A caller-provided frame cache (built for the sweep's keys) is
        consumed by the research backtest without KeyError; the PRE-window
        month-end key it lacks is built on demand (safe cache fill)."""
        close, vol = _flat_market("2025-02-28")
        sweep_months = optimize._sweep_months(close.index, "2016-06-01")
        frames = {d: _canned_frame(STABLE_ORDER) for d in sweep_months}
        # the cache omits the pre-window month-end (a stale/foreign cache);
        # the dispatcher still serves it so the on-demand fill finds data
        stale = {d: f for d, f in frames.items() if d != pd.Timestamp("2024-12-31")}
        res = _run_optimizer(monkeypatch, close, vol, frames,
                             start="2025-01-15", frame_cache=dict(stale))
        trades = _printed_trades(capsys)
        assert res is not None
        # the missing prior frame was built on demand: the mid-month days
        # still ranked and deployed
        first_day_buys = [t for (d, side, t) in trades
                          if d == "2025-01-15" and side == "BUY"]
        assert first_day_buys == ["A01", "B02", "C03"]


# ------------------------------------------------------------------
# daily_core.backfill boundaries (persisted replay)
# ------------------------------------------------------------------

@pytest.fixture
def rotating_backfill_market(monkeypatch):
    """8-name market with a ranking that ROTATES on the August month-end:
    July's picks B02/C03 fall out of the August band. The fixture returns
    an ``install_market(last_day, orders)`` hook — `orders` maps
    "YYYY-MM-DD" month-end -> ranking order; canned frames are built from
    them (frames and `_qv_order` lookups aligned by construction)."""
    DEFAULT_ORDERS = {"2026-07-31": TICKERS,
                      "2026-08-31": ["G07", "H08", "D04", "E05", "F06",
                                     "A01", "B02", "C03"]}

    def install_market(last_day: str, orders: dict[str, list[str]]) -> None:
        merged = dict(DEFAULT_ORDERS)
        merged.update(orders)
        frames = {pd.Timestamp(k): _canned_frame(v) for k, v in merged.items()}
        frame_orders = {id(f): o
                        for f, o in zip(frames.values(), merged.values(),
                                        strict=True)}

        dates = pd.bdate_range("2026-07-01", last_day)
        close = pd.DataFrame(100.0, index=dates, columns=TICKERS)
        vol = pd.DataFrame(1e6, index=dates, columns=TICKERS)

        monkeypatch.setattr(daily_core, "universe_tickers",
                            lambda _u: list(TICKERS))

        async def fake_load_fundamentals(_t):
            return {"x": []}
        monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals",
                            fake_load_fundamentals)

        async def fake_load_frames(_tickers, _asof, start=None):
            return close, vol
        monkeypatch.setattr(monthly_mod, "load_frames", fake_load_frames)
        monkeypatch.setattr(
            monthly_mod, "eligible_frame",
            lambda d, c, v, f: frames.get(pd.Timestamp(d).normalize()))
        monkeypatch.setattr(
            monthly_mod, "_qv_order",
            lambda frame, lowvol_tilt=None: (
                frame, frame_orders.get(id(frame), [])))

    return install_market


def _backfill_trades(side: str | None = None) -> list[tuple[str, str, str]]:
    async def query():
        async with daily_core.Session() as s:
            rows = (await s.scalars(select(daily_core.DailyCoreTrade))).all()
            return [(r.created_at.strftime("%Y-%m-%d"), r.side, r.ticker)
                    for r in rows]
    rows = asyncio.run(query())
    if side is None:
        return rows
    return [(d, s_, t) for (d, s_, t) in rows if s_ == side]


def _backfill_positions() -> list[str]:
    async def query():
        async with daily_core.Session() as s:
            rows = (await s.scalars(select(daily_core.DailyCorePosition))).all()
            return sorted(p.ticker for p in rows)
    return asyncio.run(query())


class TestDailyCoreBackfillBoundaries:
    def test_no_release_on_partial_last_candle(
            self, mem_db, small_slots, rotating_backfill_market, monkeypatch):
        """Data ends mid-August (2026-08-21): August has no month-end, so
        July's picks that August's ranking dropped stay HELD — the persisted
        replay no longer releases/rebalances on the replay's last candle."""
        rotating_backfill_market("2026-08-21", {"2026-07-31": TICKERS})
        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-08")

        r = asyncio.run(daily_core.backfill(start="2026-07-01"))
        assert r["ok"] is True

        trades = _backfill_trades()
        sells = [(d, t) for (d, side, t) in trades if side == "SELL"]
        # July's rebuild bought A01/B02/C03; nothing out-of-band is ever
        # released because August's month-end never arrives
        assert sells == []
        assert _backfill_positions() == ["A01", "B02", "C03"]

    def test_release_at_the_true_month_end_when_candle_stored(
            self, mem_db, small_slots, rotating_backfill_market, monkeypatch):
        """Same replay with candles through 2026-08-31 (the true US session
        month-end): the August month-end releases the out-of-band names and
        refills rank-first."""
        rotating_backfill_market("2026-08-31", {
            "2026-07-31": TICKERS,
            "2026-08-31": ["G07", "H08", "D04", "E05", "F06", "A01", "B02",
                           "C03"]})
        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-08")

        r = asyncio.run(daily_core.backfill(start="2026-07-01"))
        assert r["ok"] is True

        sells = _backfill_trades("SELL")
        assert sorted(t for (_, _, t) in sells) == ["B02", "C03"]
        aug31_buys = sorted(t for (d, side, t) in _backfill_trades("BUY")
                            if d == "2026-08-31")
        assert aug31_buys == ["G07", "H08"]
        assert _backfill_positions() == ["A01", "G07", "H08"]
