"""Tests for the shared synthetic market gate index (``app/market_index``).

Covered here:

* the economic construction — equal weight over daily simple RETURNS, so two
  names at very different nominal prices with opposing returns average -22.5%,
  not the 0% a nominal-price mean reports (audit P1-A);
* invariance to independent constant per-column rescaling, including the
  derived gate outcome;
* staggered inception, temporary missing quotes, and non-finite/non-positive
  cells (treated as unavailable, never as returns);
* prefix causality: history never changes when future rows are appended;
* empty / all-invalid inputs fail safe;
* wiring evidence that the LIVE gate+arm (``daily_core._protection_decision``),
  the persisted replay (``daily_core.backfill``) and the research replay
  (``optimize._daily_core_backtest``) all read this one helper.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, market_index, monthly, optimize
from app.db import Base
from app.market_index import equal_weight_return_index


@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite for the backfill wiring test (same pattern as the
    daily-core suite): no real DB volume is touched."""
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


# ---------------------------------------------------------------------------
# Construction / economics
# ---------------------------------------------------------------------------


class TestEqualWeightConstruction:
    def test_two_names_at_very_different_prices_are_equal_weighted(self):
        """Audit P1-A: AAA 100->105 (+5%), BBB 10->5 (-50%).

        Equal-weight return = -22.5% -> index 77.5. The old nominal mean
        (105 + 5) / 2 reports a 0% move, i.e. the $100 stock drowns the $10
        stock's -50% and the gate's economic signal disappears.
        """
        dates = pd.bdate_range("2026-01-05", periods=2)
        close = pd.DataFrame({"AAA": [100.0, 105.0], "BBB": [10.0, 5.0]}, index=dates)

        idx = equal_weight_return_index(close)

        assert idx.iloc[0] == pytest.approx(100.0)
        assert idx.iloc[1] == pytest.approx(77.5)
        # Pin the contrast to the nominal-price mean explicitly.
        nominal = close.mean(axis=1)
        assert nominal.iloc[1] == pytest.approx(nominal.iloc[0])

    def test_returns_are_chained_multiplicatively(self):
        dates = pd.bdate_range("2026-01-05", periods=3)
        close = pd.DataFrame({"AAA": [100.0, 110.0, 99.0], "BBB": [50.0, 45.0, 54.0]}, index=dates)

        idx = equal_weight_return_index(close)

        # d1: mean(+10%, -10%) = 0%; d2: mean(-10%, +20%) = +5%.
        assert idx.tolist() == pytest.approx([100.0, 100.0, 105.0])


class TestRescaleInvariance:
    @staticmethod
    def _trend_frame(rising: bool) -> pd.DataFrame:
        dates = pd.bdate_range("2026-01-01", periods=40)
        path = np.linspace(100.0, 140.0, 40) if rising else np.linspace(140.0, 100.0, 40)
        # Very different nominal units, identical percentage paths.
        return pd.DataFrame({"HIGH": path / 97.0, "LOW": path * 13.0}, index=dates)

    @pytest.mark.parametrize("rising", [True, False])
    def test_independent_constant_rescaling_is_invariant(self, rising):
        close = self._trend_frame(rising)
        scaled = pd.DataFrame(
            {"HIGH": close["HIGH"] * 0.013, "LOW": close["LOW"] * 1234.5}, index=close.index
        )

        base_idx = equal_weight_return_index(close)
        scaled_idx = equal_weight_return_index(scaled)

        assert np.allclose(base_idx.to_numpy(), scaled_idx.to_numpy(), rtol=1e-12, atol=0.0)
        # The gate outcome is derived from the same series and must match.
        gate = daily_core._above_sma(base_idx, 5)
        scaled_gate = daily_core._above_sma(scaled_idx, 5)
        assert scaled_gate is gate
        assert gate is rising

    def test_single_column_rescale_does_not_move_the_index(self):
        dates = pd.bdate_range("2026-01-01", periods=6)
        close = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 99.5, 103.0, 104.0, 103.0],
                "BBB": [10.0, 9.8, 10.1, 10.4, 10.2, 10.6],
                "CCC": [500.0, 505.0, 500.0, 512.0, 515.0, 510.0],
            },
            index=dates,
        )

        base_idx = equal_weight_return_index(close)
        only_bbb_scaled = close.copy()
        only_bbb_scaled["BBB"] = close["BBB"] * 137.0
        scaled_idx = equal_weight_return_index(only_bbb_scaled)

        assert np.allclose(base_idx.to_numpy(), scaled_idx.to_numpy(), rtol=1e-12, atol=0.0)
        assert daily_core._above_sma(base_idx, 5) is daily_core._above_sma(scaled_idx, 5)

    def test_live_gate_outcome_is_rescale_invariant(self, tmp_path, monkeypatch):
        """End-to-end on the LIVE path: the same economic tape in different
        nominal units must produce the same gate decision."""
        close = self._trend_frame(rising=False)
        scaled = close.mul(pd.Series({"HIGH": 0.013, "LOW": 1234.5}))
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")

        base_gate, base_out = asyncio.run(
            daily_core._protection_decision(["HIGH"], 5, False, None, close)
        )
        scaled_gate, scaled_out = asyncio.run(
            daily_core._protection_decision(["HIGH"], 5, False, None, scaled)
        )

        assert base_gate is False
        assert scaled_gate is base_gate
        assert scaled_out is base_out is False


# ---------------------------------------------------------------------------
# Missing / invalid quotes, staggered inception
# ---------------------------------------------------------------------------


class TestUnavailableQuotes:
    def test_missing_quote_contributes_zero_and_stays_in_the_average(self):
        """A temporarily missing quote is carried flat (return 0) rather than
        dropped, so it keeps its equal-weight share of the cross-section."""
        dates = pd.bdate_range("2026-01-05", periods=4)
        close = pd.DataFrame(
            {"AAA": [100.0, np.nan, np.nan, 110.0], "BBB": [50.0, 40.0, 36.0, 40.0]}, index=dates
        )

        idx = equal_weight_return_index(close)

        # d1: mean(AAA 0, BBB -20%) = -10%; d2: mean(0, -10%) = -5%;
        # d3: AAA's gap move (+10%) is attributed on reappearance:
        #     mean(+10%, +11.11%) -> 100*0.9*0.95*1.105555 = 94.525.
        assert idx.tolist() == pytest.approx([100.0, 90.0, 85.5, 94.525])
        # Dropping AAA instead would compound to 100*0.8*0.9*1.1111 = 80.0.
        assert idx.iloc[-1] != pytest.approx(80.0)

    def test_first_observation_contributes_no_return(self):
        dates = pd.bdate_range("2026-01-05", periods=4)
        close = pd.DataFrame(
            {"AAA": [100.0, 100.0, 110.0, 110.0], "BBB": [np.nan, np.nan, 20.0, 22.0]}, index=dates
        )

        idx = equal_weight_return_index(close)

        # BBB's first price (20) has no prior; only AAA's +10% counts on d2.
        # From d3 on, BBB has a prior close and joins the average (+10%).
        assert idx.tolist() == pytest.approx([100.0, 100.0, 110.0, 115.5])

    def test_nonfinite_and_nonpositive_cells_are_unavailable(self):
        dates = pd.bdate_range("2026-01-05", periods=5)
        dirty = pd.DataFrame(
            {
                "AAA": [100.0, 0.0, 100.0, np.inf, 110.0],
                "BBB": [50.0, 55.0, -1.0, 60.0, 66.0],
                "BAD": [np.nan, -5.0, 0.0, -np.inf, np.nan],
            },
            index=dates,
        )
        clean = pd.DataFrame(
            {
                "AAA": [100.0, np.nan, 100.0, np.nan, 110.0],
                "BBB": [50.0, 55.0, np.nan, 60.0, 66.0],
                "BAD": [np.nan, np.nan, np.nan, np.nan, np.nan],
            },
            index=dates,
        )

        dirty_idx = equal_weight_return_index(dirty)
        clean_idx = equal_weight_return_index(clean)
        assert np.array_equal(dirty_idx.to_numpy(), clean_idx.to_numpy())

        # d1: mean(AAA 0, BBB +10%) = +5%; d2: mean(0, 0) = 0;
        # d3: mean(0, +1/11) = +1/22; d4: mean(+10%, +10%) = +10%.
        assert dirty_idx.tolist() == pytest.approx([100.0, 105.0, 105.0, 105.0 * 23 / 22, 120.75])

    def test_empty_and_all_invalid_inputs_are_safe(self):
        assert equal_weight_return_index(None).empty

        dates = pd.bdate_range("2026-01-05", periods=3)
        assert equal_weight_return_index(pd.DataFrame(index=dates)).empty is False
        # No columns: no constituents -> no returns ever (flat, never NaN).
        assert equal_weight_return_index(pd.DataFrame(index=dates)).tolist() == pytest.approx(
            [100.0, 100.0, 100.0]
        )

        assert equal_weight_return_index(pd.DataFrame(index=dates)).index.equals(dates)

        all_bad = pd.DataFrame(
            {"A": [np.nan, 0.0, np.inf], "B": [-1.0, -np.inf, np.nan]}, index=dates
        )
        assert equal_weight_return_index(all_bad).tolist() == pytest.approx([100.0, 100.0, 100.0])

        single = pd.DataFrame({"A": [42.0]}, index=dates[:1])
        assert equal_weight_return_index(single).tolist() == pytest.approx([100.0])

    def test_empty_row_index_returns_empty_series(self):
        out = equal_weight_return_index(pd.DataFrame(columns=["AAA", "BBB"], dtype=float))
        assert out.empty
        assert out.index.equals(pd.Index([]))


# ---------------------------------------------------------------------------
# Causality / no future-data influence
# ---------------------------------------------------------------------------


def _synthetic_close(periods: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(20261030)
    dates = pd.bdate_range("2024-01-02", periods=periods)
    cols = {}
    for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        rets = rng.normal(0.0004 * (i - 1), 0.012, periods)
        cols[ticker] = (10.0 + 90.0 * i) * np.exp(np.cumsum(rets))
    close = pd.DataFrame(cols, index=dates)
    close.iloc[17, 1] = np.nan  # a temporary hole
    return close


class TestCausality:
    def test_prefix_values_match_the_full_history(self):
        close = _synthetic_close()
        full = equal_weight_return_index(close)

        for k in (1, 2, 3, 10, 60, len(close) - 1):
            prefix = equal_weight_return_index(close.iloc[:k])
            assert np.allclose(prefix.to_numpy(), full.iloc[:k].to_numpy(), rtol=1e-12, atol=0.0), k

    def test_appending_future_rows_does_not_change_history(self):
        close = _synthetic_close()
        base_idx = equal_weight_return_index(close)

        future_dates = pd.bdate_range(close.index[-1] + pd.offsets.BDay(1), periods=25)
        future = pd.DataFrame(
            {
                "AAA": np.linspace(1e3, 1e-3, 25),
                "BBB": np.linspace(1e-3, 1e6, 25),
                "CCC": np.linspace(500.0, 1.0, 25),
                "DDD": np.linspace(2.0, 900.0, 25),
                # A "future member" must not create history either.
                "NEW": np.linspace(1e5, 2e5, 25),
            },
            index=future_dates,
        )
        full = equal_weight_return_index(pd.concat([close, future]))

        assert np.allclose(
            base_idx.to_numpy(), full.loc[close.index].to_numpy(), rtol=1e-12, atol=0.0
        )
        # Future membership cannot leak backwards: NEW has no prior close
        # before the append point, so the base prefix is untouched.
        assert full.loc[close.index[0]] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Shared wiring: live, persisted replay, research
# ---------------------------------------------------------------------------


def _sentinel_index(frame: pd.DataFrame, *, last: float) -> pd.Series:
    """A recognizable index: flat 100 except the final value."""
    values = np.full(len(frame.index), 100.0)
    values[-1] = last
    return pd.Series(values, index=frame.index, name="sentinel")


def _market_close() -> pd.DataFrame:
    """32 sessions: 20 up, then 12 down — a negative last-10-day slope."""
    dates = pd.bdate_range("2026-01-01", periods=32)
    px = [100.0 + i * (10.0 / 19) for i in range(20)] + [
        109.0 - i * (43.0 / 11) for i in range(1, 13)
    ]
    return pd.DataFrame({"AAA": px}, index=dates)


class TestSharedWiring:
    @pytest.mark.parametrize(("last", "expected"), [(90.0, False), (110.0, True)])
    def test_live_gate_reads_the_shared_index(self, tmp_path, monkeypatch, last, expected):
        close = _market_close()
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        seen: list[pd.DataFrame] = []

        def spy(frame, **kwargs):
            seen.append(frame)
            return _sentinel_index(frame, last=last)

        monkeypatch.setattr(market_index, "equal_weight_return_index", spy)

        gate_ok, basket_out = asyncio.run(
            daily_core._protection_decision(["AAA"], 5, False, None, close)
        )

        assert seen and seen[0] is close
        assert gate_ok is expected
        assert basket_out is False  # gradient disabled

    @pytest.mark.parametrize(
        ("last", "armed", "expected_out"), [(90.0, False, False), (110.0, True, True)]
    )
    def test_live_gradient_arm_reads_the_same_shared_index(
        self, tmp_path, monkeypatch, last, armed, expected_out
    ):
        """The arm is the same series as the gate, so a synthetic index that
        is below/above its SMA must disarm/arm the cash-out accordingly."""
        close = _market_close()
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        daily_core._save_state({"basket_neg_streak": 2, "basket_out": False})
        monkeypatch.setattr(
            market_index,
            "equal_weight_return_index",
            lambda frame, **kw: _sentinel_index(frame, last=last),
        )

        gate_ok, basket_out = asyncio.run(
            daily_core._protection_decision(["AAA"], None, True, 20, close)
        )

        assert gate_ok is True  # gate is off; only the arm is configured
        assert daily_core._above_sma(_sentinel_index(close, last=last), 20) is armed
        assert basket_out is expected_out

    async def test_backfill_replay_reads_the_shared_index(self, mem_db, monkeypatch, tmp_path):
        """The persisted replay builds its gate/arm series through the helper,
        from the RAW close frame (the helper owns the validity/carry rules)."""
        dates = pd.bdate_range("2026-07-01", "2026-08-28")
        close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=dates)
        open_px = close.copy()
        vol = pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"])

        def fake_frame(m, c, v, f):
            return pd.DataFrame(
                {"eligible": [True, True]}, index=pd.Index(["AAA", "BBB"], name="ticker")
            )

        def fake_order(frame):
            return frame["eligible"], ["AAA", "BBB"]

        async def fake_load_ohlcv(_tickers, _asof, start=None):
            return open_px, close, vol

        async def fake_load_fundamentals(_tickers):
            return {"AAA": {"x": []}, "BBB": {"x": []}}

        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-08")
        monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: ["AAA", "BBB"])
        monkeypatch.setattr(
            daily_core.fundamentals_mod, "load_fundamentals", fake_load_fundamentals
        )
        monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv)
        monkeypatch.setattr(monthly, "eligible_frame", fake_frame)
        monkeypatch.setattr(monthly, "_qv_order", fake_order)

        seen: list[pd.DataFrame] = []
        real = equal_weight_return_index

        def spy(frame, **kwargs):
            seen.append(frame)
            return real(frame, **kwargs)

        monkeypatch.setattr(market_index, "equal_weight_return_index", spy)

        result = await daily_core.backfill(start="2026-08-01")

        assert result["ok"] is True
        assert len(seen) == 1
        assert seen[0] is close  # raw frame, not close.ffill()
        # Flat fixture prices -> a flat index at the base (no invented return).
        assert real(close).nunique() == 1

    async def test_research_backtest_reads_the_shared_index(self, monkeypatch):
        dates = pd.bdate_range("2023-01-03", "2025-12-31")
        rng = np.random.default_rng(7)
        close = pd.DataFrame(
            {
                t: 40.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates))))
                for t in ("AAA", "BBB")
            },
            index=dates,
        )
        vol = pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"])
        dead = pd.DataFrame(
            {"eligible": [False, False]}, index=pd.Index(["AAA", "BBB"], name="ticker")
        )

        monkeypatch.setattr(monthly, "eligible_frame", lambda d, c, v, f: dead)

        seen: list[pd.DataFrame] = []
        real = equal_weight_return_index

        def spy(frame, **kwargs):
            seen.append(frame)
            return real(frame, **kwargs)

        monkeypatch.setattr(market_index, "equal_weight_return_index", spy)

        summary = await optimize._daily_core_backtest(
            "2024-01-02",
            None,
            "diversified-plus",
            1000.0,
            False,
            "none",
            "monthly",
            False,
            "rank",
            close_preloaded=(close, vol),
            fund_preloaded={},
            open_preloaded=close,
            with_baseline=False,
        )

        assert summary is not None
        assert seen and seen[0] is close
