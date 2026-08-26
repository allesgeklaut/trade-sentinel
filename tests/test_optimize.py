"""Tests for the walk-forward optimization / backtest tool (app.optimize).

These test the pure, deterministic parts of the tool (signal series, replay,
parameter sweep) using synthetic candles — no DB, no network.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from app import optimize
from app.analysis import compute
from app.optimize import (
    ReplayParams,
    _is_stop_out,
    _replay,
    _row_action,
    _row_strength,
    _select_cases,
    _signal_series,
    _snapshot_for_day,
    _split_windows,
    _trade_outcomes,
    _reconstruct_portfolio_states,
    TradeCase,
)


def _gen_candles(start_price: float, daily_drift: float,
                 n: int = 500, seed: int = 42) -> list[dict]:
    """Generate *n* daily OHLCV candles (same shape as market.candles())."""
    rng = random.Random(seed)
    rows: list[dict] = []
    price = start_price
    base = datetime(2025, 1, 1)
    for i in range(n):
        noise = rng.gauss(0, 0.015)
        close = price * (1 + daily_drift + noise)
        open_ = price
        high = max(open_, close) * (1 + abs(rng.gauss(0, 0.004)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, 0.004)))
        vol = 1_000_000 * (1 + abs(rng.gauss(0, 0.2)))
        rows.append({
            "time": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "open": round(open_, 2), "high": round(high, 2),
            "low": round(low, 2), "close": round(close, 2),
            "volume": round(vol, 0),
        })
        price = close
    return rows


class TestSignalSeries:
    def test_returns_expected_columns(self):
        df = _signal_series(_gen_candles(100.0, 0.001))
        for col in ("time", "close", "net", "bullish", "bearish",
                    "trend_up", "trend_down", "dist_above", "dist_below",
                    "atr_stop", "weekly_trend_up",
                    "rsi_3d_change", "macd_hist_3d_change", "run_5d"):
            assert col in df.columns
        assert len(df) == 500

    def test_uptrend_has_positive_net(self):
        df = _signal_series(_gen_candles(100.0, 0.01, seed=1))
        # Late in a strong uptrend the net score should be clearly positive.
        assert df["net"].iloc[-1] > 0

    def test_downtrend_has_negative_net(self):
        df = _signal_series(_gen_candles(100.0, -0.01, seed=2))
        assert df["net"].iloc[-1] < 0


class TestRowAction:
    def test_buy_requires_trend_and_distance(self):
        row = {"net": 50, "trend_up": True, "dist_above": 5,
               "trend_down": False, "dist_below": 0, "weekly_trend_up": True}
        assert _row_action(row, ReplayParams(buy_threshold=40, sell_threshold=-40)) == "BUY"

    def test_buy_blocked_without_trend(self):
        row = {"net": 50, "trend_up": False, "dist_above": 5,
               "trend_down": False, "dist_below": 0, "weekly_trend_up": True}
        assert _row_action(row, ReplayParams()) == "HOLD"

    def test_buy_blocked_by_weekly_trend(self):
        row = {"net": 50, "trend_up": True, "dist_above": 5,
               "trend_down": False, "dist_below": 0, "weekly_trend_up": False}
        assert _row_action(row, ReplayParams(buy_threshold=40, sell_threshold=-40)) == "HOLD"

    def test_sell(self):
        row = {"net": -50, "trend_up": False, "dist_above": 0,
               "trend_down": True, "dist_below": 5, "weekly_trend_up": True}
        assert _row_action(row, ReplayParams()) == "SELL"

    def test_sell_not_blocked_by_weekly_trend(self):
        row = {"net": -50, "trend_up": False, "dist_above": 0,
               "trend_down": True, "dist_below": 5, "weekly_trend_up": False}
        assert _row_action(row, ReplayParams()) == "SELL"

    def test_threshold_respected(self):
        row = {"net": 30, "trend_up": True, "dist_above": 5,
               "trend_down": False, "dist_below": 0, "weekly_trend_up": True}
        # buy_threshold=40 → 30 is not enough → HOLD
        assert _row_action(row, ReplayParams(buy_threshold=40)) == "HOLD"
        # buy_threshold=20 → 30 qualifies → BUY
        assert _row_action(row, ReplayParams(buy_threshold=20)) == "BUY"


class TestRowStrength:
    def test_buy_uses_bullish(self):
        row = {"bullish": 70.0, "bearish": 10.0}
        assert _row_strength(row, "BUY") == 70

    def test_sell_uses_bearish(self):
        row = {"bullish": 10.0, "bearish": 60.0}
        assert _row_strength(row, "SELL") == 60

    def test_hold_uses_max(self):
        row = {"bullish": 30.0, "bearish": 55.0}
        assert _row_strength(row, "HOLD") == 55


class TestSplitWindows:
    def test_non_overlapping(self):
        days = [f"2025-01-{i:02d}" for i in range(1, 11)]  # 10 days
        wins = list(_split_windows(days, 4, 2))
        # i=0 (train 0-3, test 4-5), i=2 (train 2-5, test 6-7),
        # i=4 (train 4-7, test 8-9) → 3 windows
        assert len(wins) == 3
        # Each test window is strictly after its train window.
        for train_s, train_e, test_s, test_e in wins:
            assert train_e < test_s


class TestReplay:
    def test_replay_buys_in_uptrend(self):
        # A single strongly-uptrending ticker should generate BUY trades.
        series = {"UP": _signal_series(_gen_candles(100.0, 0.01, seed=1))}
        res = _replay(series, ReplayParams(start_cash=10000.0, monthly_allowance=0.0))
        assert res.n_trades > 0
        assert any(t["side"] == "BUY" for t in res.trades)
        assert res.final_equity > 10000.0

    def test_replay_respects_window(self):
        series = {"UP": _signal_series(_gen_candles(100.0, 0.01, seed=1))}
        res = _replay(series, ReplayParams(start_cash=10000.0, monthly_allowance=0.0),
                      start="2025-01-01", end="2025-06-30")
        # Equity curve only spans the requested window.
        assert res.equity_curve
        assert res.equity_curve[0]["time"] >= "2025-01-01"
        assert res.equity_curve[-1]["time"] <= "2025-06-30"

    def test_replay_no_trades_when_no_signal(self):
        # A flat/choppy series with high thresholds should produce few trades.
        series = {"FLAT": _signal_series(_gen_candles(100.0, 0.0, seed=3))}
        res = _replay(series, ReplayParams(start_cash=10000.0, monthly_allowance=0.0,
                                           buy_threshold=90, sell_threshold=-90,
                                           relaxed_hold_strength=100))
        # High thresholds + no relaxed fallback → nothing qualifies.
        assert res.n_trades == 0


class TestReplayTradeDates:
    """The replay stamps a `date` on every trade so the benchmark can locate
    the decision day without price-matching."""

    def test_trades_carry_a_date(self):
        series = {"UP": _signal_series(_gen_candles(100.0, 0.01, seed=1))}
        res = _replay(series, ReplayParams(start_cash=10000.0, monthly_allowance=0.0))
        assert res.n_trades > 0
        for t in res.trades:
            assert "date" in t and t["date"]
            # Date is a YYYY-MM-DD string within the generated range.
            assert t["date"] >= "2025-01-01"


class TestPaperPortfolioFloor:
    """PaperPortfolio.buy must floor (not round) fractional shares so the
    cost never exceeds the budget — rounding up can silently drop a buy."""

    def test_high_price_ticker_budget_fits(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        # ASML.AS-like price: round(250/1641.60, 4) = 0.1523 → $250.02,
        # which exceeded the $250 budget and dropped the buy.
        pf.buy("ASML.AS", 1641.60, 250.0, "test")
        assert "ASML.AS" in pf.positions
        shares = pf.positions["ASML.AS"]
        assert shares * 1641.60 <= 250.0
        assert pf.cash >= 1000.0 - 250.0

    def test_multiple_buys_never_exceed_cash(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("A", 334.07, 250.0, "test")
        pf.buy("B", 1191.74, 250.0, "test")
        pf.buy("C", 426.54, 250.0, "test")
        pf.buy("D", 1641.60, 250.0, "test")
        assert set(pf.positions) == {"A", "B", "C", "D"}
        assert pf.cash >= 0.0


class TestPaperPortfolioThesis:
    """PaperPortfolio must track the entry thesis (BUY reason) and buy date
    for each position, clear them on full sell, and pass them through to the
    LLM context so the LLM can judge 'is the thesis still valid?'."""

    def test_thesis_set_on_new_buy(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("NVDA", 100.0, 500.0, "strong uptrend + high ADX", date="2025-06-01")
        assert pf.thesis["NVDA"] == "strong uptrend + high ADX"
        assert pf.buy_date["NVDA"] == "2025-06-01"

    def test_thesis_not_overwritten_on_top_up(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("NVDA", 100.0, 300.0, "original thesis", date="2025-06-01")
        pf.buy("NVDA", 110.0, 300.0, "top-up reason", date="2025-06-15")
        # The thesis from the first BUY is preserved — the LLM sees why it
        # originally opened the position, not the top-up reason.
        assert pf.thesis["NVDA"] == "original thesis"
        assert pf.buy_date["NVDA"] == "2025-06-01"

    def test_thesis_cleared_on_full_sell(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("AMD", 100.0, 500.0, "momentum play", date="2025-06-01")
        pf.sell("AMD", 110.0, None, "take profit", date="2025-06-15")
        assert "AMD" not in pf.thesis
        assert "AMD" not in pf.buy_date

    def test_thesis_preserved_on_partial_sell(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("AMD", 100.0, 500.0, "momentum play", date="2025-06-01")
        pf.sell("AMD", 110.0, 2.0, "partial take profit", date="2025-06-15")
        # Partial sell keeps the position — thesis must survive.
        assert pf.thesis["AMD"] == "momentum play"
        assert pf.buy_date["AMD"] == "2025-06-01"

    def test_pf_to_positions_includes_thesis(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("NVDA", 100.0, 500.0, "AI infrastructure leader", date="2025-06-01")
        positions = optimize._pf_to_positions(pf)
        assert positions[0]["thesis"] == "AI infrastructure leader"
        assert positions[0]["buy_date"] == "2025-06-01"

    def test_full_sell_clears_position_metadata(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("AMD", 100.0, 500.0, "momentum play", date="2025-06-01")
        pf.sell("AMD", 110.0, None, "take profit", date="2025-06-15")
        # Full sell clears the thesis/buy-date bookkeeping.
        assert "AMD" not in pf.thesis
        assert "AMD" not in pf.buy_date

    def test_rebuy_opens_fresh_position(self):
        pf = optimize.PaperPortfolio(cash=1000.0)
        pf.buy("AMD", 100.0, 500.0, "momentum play", date="2025-06-01")
        pf.sell("AMD", 110.0, None, "take profit", date="2025-06-15")
        # Re-buying opens a fresh position with a new thesis.
        pf.buy("AMD", 105.0, 500.0, "new thesis", date="2025-07-01")
        assert pf.thesis["AMD"] == "new thesis"
        assert pf.buy_date["AMD"] == "2025-07-01"


class TestReplayTopUpAtCap:
    """The max-positions cap must block NEW positions but still allow topping
    up tickers already held. Previously the BUY loop used 'break' when at the
    cap, which stopped all buying — including top-ups — leaving ~50% of equity
    idle as cash for the entire replay. The fix changed 'break' to 'continue'
    with a 'ticker not in positions' guard.
    """

    def _uptrending_series(self, n_tickers: int = 12, n: int = 500) -> dict:
        """Generate n_tickers uptrending tickers. BUY signals fire from ~day
        200 onward (once SMA200 is available and trend_up is confirmed)."""
        return {
            f"UP{i}": _signal_series(_gen_candles(100.0, 0.01, seed=i, n=n))
            for i in range(n_tickers)
        }

    def test_top_ups_happen_when_at_cap(self):
        # 12 uptrending tickers, max_positions=10: the cap is hit once BUYs
        # start firing (~day 200), but the monthly allowance keeps depositing
        # cash. Held tickers with continuing BUY signals should be topped up
        # despite the cap.
        series = self._uptrending_series()
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=1000.0,
            max_positions=10,
            max_position_pct=10.0,
            min_cash_pct=5.0,
            stop_type="none",
            use_atr_stop=False,
        )
        # Start after day 200 so BUY signals are active immediately.
        res = _replay(series, params, start="2025-07-15", end="2026-05-15")

        # The cap was hit (10 positions) — verify via the position-count curve.
        assert res.position_count_curve
        assert max(res.position_count_curve) <= 10, "exceeded max_positions cap"
        assert any(n >= 10 for n in res.position_count_curve), "never hit the cap"

        # Top-ups: a BUY on a ticker that already had an earlier BUY. Count
        # distinct tickers bought, then count how many BUYs are repeats.
        bought_tickers: set[str] = set()
        top_ups = 0
        for t in res.trades:
            if t["side"] == "BUY":
                if t["ticker"] in bought_tickers:
                    top_ups += 1
                else:
                    bought_tickers.add(t["ticker"])
        assert top_ups > 0, (
            f"no top-up BUYs happened ({top_ups}); the cap blocked all buying. "
            f"distinct tickers bought: {len(bought_tickers)}, total trades: {res.n_trades}"
        )

    def test_cash_deploys_when_at_cap(self):
        # Same scenario, but assert that cash actually gets deployed rather
        # than sitting idle. Before the fix, avg cash was ~50% of equity
        # because top-ups were blocked. After the fix it should be well below
        # 30% (the max_position_pct ceiling means cash draws down gradually as
        # positions grow into their 10% caps and the monthly allowance deploys
        # via top-ups).
        series = self._uptrending_series()
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=1000.0,
            max_positions=10,
            max_position_pct=10.0,
            min_cash_pct=5.0,
            stop_type="none",
            use_atr_stop=False,
        )
        res = _replay(series, params, start="2025-07-15", end="2026-05-15")

        assert res.cash_curve and res.equity_curve
        cash_pcts = [
            (c / e["equity"]) * 100 if e["equity"] > 0 else 100.0
            for c, e in zip(res.cash_curve, res.equity_curve)
        ]
        # Skip the first ~20 days while positions are being opened; the test
        # is about the steady state where the cap is hit and cash should
        # still deploy via top-ups.
        steady_state = cash_pcts[20:]
        avg_cash = sum(steady_state) / len(steady_state) if steady_state else 100.0
        assert avg_cash < 30.0, (
            f"avg cash in steady state was {avg_cash:.1f}% of equity — cash is "
            f"sitting idle because top-ups are blocked by the cap. Before the "
            f"fix this was ~50%; after it should be well under 30%."
        )

    def test_no_new_positions_above_cap(self):
        # The cap must still block NEW tickers — only top-ups of held tickers
        # should pass through. Verify the position count never exceeds the cap.
        series = self._uptrending_series(n_tickers=15)
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=1000.0,
            max_positions=10,
            max_position_pct=10.0,
            min_cash_pct=5.0,
            stop_type="none",
            use_atr_stop=False,
        )
        res = _replay(series, params, start="2025-07-15", end="2026-05-15")
        assert max(res.position_count_curve) <= 10, (
            f"position count {max(res.position_count_curve)} exceeded cap 10"
        )


class TestSnapshotForDay:
    """_snapshot_for_day must produce the same snapshot as analysis.compute()
    so the benchmark feeds the LLM identical fields the live sim does."""

    def test_matches_compute_on_last_row(self):
        candles = _gen_candles(100.0, 0.01, seed=1)
        df = _signal_series(candles)
        last_date = df["time"].iloc[-1]

        probe = _snapshot_for_day(df, last_date)
        live = compute(candles)

        assert set(probe["snapshot"].keys()) == set(live["snapshot"].keys())
        for k in live["snapshot"]:
            assert probe["snapshot"][k] == live["snapshot"][k], f"field {k} differs"
        assert probe["action"] == live["action"]
        assert probe["strength"] == live["strength"]

    def test_works_on_arbitrary_day(self):
        df = _signal_series(_gen_candles(100.0, 0.01, seed=1))
        mid_date = df["time"].iloc[250]
        result = _snapshot_for_day(df, mid_date)
        assert "snapshot" in result
        snap = result["snapshot"]
        assert snap["close"] is not None
        assert snap["rsi"] is not None
        assert snap["rsi_3d_change"] is not None
        assert snap["macd_hist_3d_change"] is not None

    def test_raises_on_missing_date(self):
        df = _signal_series(_gen_candles(100.0, 0.01, seed=1))
        with pytest.raises(KeyError):
            _snapshot_for_day(df, "1999-01-01")


class TestTradeOutcomes:
    def test_buy_outcome_is_forward_return(self):
        # A BUY at 100 that closes at 120 twenty days later → +20% outcome,
        # badness -20 (a *good* buy).
        df = _signal_series(_gen_candles(100.0, 0.01, seed=1))
        trades = [{"ticker": "UP", "side": "BUY", "shares": 1.0, "price": df["close"].iloc[100],
                   "reason": "BUY", "date": df["time"].iloc[100]}]
        cases = _trade_outcomes({"UP": df}, trades, forward_days=20)
        assert len(cases) == 1
        c = cases[0]
        assert c.det_side == "BUY"
        assert c.outcome_pct > 0          # price rose over the window
        assert c.badness == -c.outcome_pct  # badness is negative for a good buy

    def test_sell_outcome_badness_signs_flipped(self):
        # A SELL before a rally is *bad*: outcome positive, badness positive.
        df = _signal_series(_gen_candles(100.0, 0.01, seed=1))
        trades = [{"ticker": "UP", "side": "SELL", "shares": 1.0, "price": df["close"].iloc[100],
                   "reason": "SELL", "date": df["time"].iloc[100]}]
        cases = _trade_outcomes({"UP": df}, trades, forward_days=20)
        c = cases[0]
        assert c.outcome_pct > 0
        assert c.badness == c.outcome_pct   # positive badness = bad sell

    def test_missing_date_is_skipped(self):
        df = _signal_series(_gen_candles(100.0, 0.01, seed=1))
        trades = [{"ticker": "UP", "side": "BUY", "shares": 1.0, "price": 100.0,
                   "reason": "BUY", "date": "1999-01-01"}]  # not in the series
        assert _trade_outcomes({"UP": df}, trades) == []

    def test_position_before_is_attached(self):
        # A SELL that closes a prior BUY should carry position_before.
        df = _signal_series(_gen_candles(100.0, 0.01, seed=1))
        buy_date = df["time"].iloc[100]
        sell_date = df["time"].iloc[130]
        trades = [
            {"ticker": "UP", "side": "BUY", "shares": 2.0, "price": df["close"].iloc[100],
             "reason": "BUY", "date": buy_date},
            {"ticker": "UP", "side": "SELL", "shares": 2.0, "price": df["close"].iloc[130],
             "reason": "SELL", "date": sell_date},
        ]
        cases = _trade_outcomes({"UP": df}, trades, forward_days=20)
        # BUY first → no prior position; SELL → prior position present.
        assert cases[0].position_before is None
        assert cases[1].position_before is not None
        assert cases[1].position_before["shares"] == 2.0


class TestReconstructPortfolioStates:
    def test_buy_decreases_cash_and_opens_position(self):
        states = _reconstruct_portfolio_states(
            [{"ticker": "A", "side": "BUY", "shares": 1.0, "price": 50.0,
              "reason": "", "date": "2025-01-15"}],
            start_cash=1000.0, monthly_allowance=0.0,
        )
        assert states[0]["cash"] == 1000.0          # before the trade
        assert states[0]["positions"] == []         # nothing held yet
        assert states[0]["total_equity"] == 1000.0

    def test_sell_closing_position_shows_it_before(self):
        states = _reconstruct_portfolio_states(
            [
                {"ticker": "A", "side": "BUY", "shares": 2.0, "price": 50.0,
                 "reason": "", "date": "2025-01-15"},
                {"ticker": "A", "side": "SELL", "shares": 2.0, "price": 60.0,
                 "reason": "", "date": "2025-02-15"},
            ],
            start_cash=1000.0, monthly_allowance=0.0,
        )
        # Before the SELL, A was held (2 shares @ 50), valued at the SELL price.
        assert states[1]["positions"][0]["ticker"] == "A"
        assert states[1]["positions"][0]["shares"] == 2.0

    def test_monthly_allowance_deposited_once_per_month(self):
        states = _reconstruct_portfolio_states(
            [
                {"ticker": "A", "side": "BUY", "shares": 1.0, "price": 10.0,
                 "reason": "", "date": "2025-01-10"},
                {"ticker": "A", "side": "BUY", "shares": 1.0, "price": 10.0,
                 "reason": "", "date": "2025-01-20"},
                {"ticker": "A", "side": "BUY", "shares": 1.0, "price": 10.0,
                 "reason": "", "date": "2025-02-05"},
            ],
            start_cash=0.0, monthly_allowance=1000.0,
        )
        # Jan trades see one 1000 deposit; the Feb trade sees a second one.
        assert states[0]["cash"] == 1000.0
        assert states[1]["cash"] == 990.0   # 1000 - 10 (first buy applied)
        assert states[2]["cash"] == 1980.0  # 990 + 1000 (Feb) - 10 (third buy)

    def test_allowance_catches_up_across_skip_months(self):
        # Sparse trades: one in Jan, the next in Apr. The replay deposits on
        # the first trading day of every month, so the Apr trade must have
        # accumulated Jan+Feb+Mar+Apr = 4 allowances (3 catch-up + its own).
        states = _reconstruct_portfolio_states(
            [
                {"ticker": "A", "side": "BUY", "shares": 1.0, "price": 10.0,
                 "reason": "", "date": "2025-01-10"},
                {"ticker": "A", "side": "BUY", "shares": 1.0, "price": 10.0,
                 "reason": "", "date": "2025-04-10"},
            ],
            start_cash=0.0, monthly_allowance=1000.0,
        )
        assert states[0]["cash"] == 1000.0          # Jan: first deposit
        # Before the Apr buy: 990 (after Jan buy) + 3000 (Feb,Mar,Apr) = 3990
        assert states[1]["cash"] == 3990.0


class TestSelectCases:
    def _case(self, ticker, side, outcome, date="2025-01-01"):
        badness = -outcome if side == "BUY" else outcome
        return TradeCase(date=date, ticker=ticker, det_side=side, det_reason="",
                         entry_price=100.0, forward_close=100.0 * (1 + outcome / 100),
                         outcome_pct=outcome, badness=badness)

    def test_worst_cases_first(self):
        cases = [
            self._case("A", "BUY", -10),    # badness +10
            self._case("B", "SELL", 30),    # badness +30
            self._case("C", "BUY", -50),    # badness +50 (worst)
        ]
        picked = _select_cases(cases, n_worst=3, n_control=0)
        assert [c.ticker for c in picked] == ["C", "B", "A"]

    def test_controls_flagged_and_from_low_badness(self):
        cases = [
            self._case("A", "BUY", -10),    # badness +10
            self._case("B", "SELL", 30),    # badness +30
            self._case("C", "BUY", 50, date="2025-02-01"),   # badness -50 (best → control)
        ]
        picked = _select_cases(cases, n_worst=2, n_control=1)
        controls = [c for c in picked if c.is_control]
        assert len(controls) == 1
        assert controls[0].ticker == "C"

    def test_dedupes_by_ticker(self):
        # Two bad trades on A — only A's worst should be picked once.
        cases = [
            self._case("A", "BUY", -20, date="2025-01-01"),
            self._case("A", "BUY", -40, date="2025-02-01"),
            self._case("B", "SELL", 30),
        ]
        picked = _select_cases(cases, n_worst=3, n_control=0)
        # A appears only once (its -40 trade), B once.
        tickers = [c.ticker for c in picked]
        assert tickers.count("A") == 1
        assert "B" in tickers


class TestIsStopOut:
    def test_initial_stop(self):
        assert _is_stop_out("Initial stop: 71.88 <= 72.87")

    def test_atr_stop(self):
        assert _is_stop_out("ATR stop hit: 24.61 < 25.00")

    def test_trailing_stop(self):
        assert _is_stop_out("Trailing stop: 30.00 <= 32.00 (peak 40.00, -20%)")

    def test_portfolio_stop(self):
        assert _is_stop_out("Portfolio stop: equity 100 is -20% from peak 125")

    def test_signal_sell_is_not_a_stop(self):
        assert not _is_stop_out("SELL signal (strength 57)")

    def test_buy_is_not_a_stop(self):
        assert not _is_stop_out("BUY signal (strength 67)")


class TestHybridReplayLLMAdditionGuards:
    """LLM-initiated BUY additions must respect the max-positions cap: block
    NEW positions when at the cap, but allow topping up tickers already held.
    Previously (before Step 1b) LLM additions were uncapped — the 60-day
    replay sprawled to 19 positions. Now both hybrid and pure-LLM modes enforce
    the cap on LLM BUYs the same way the deterministic engine does."""

    async def test_llm_additions_respect_position_cap(self, monkeypatch):
        import json as _json

        from app import llm as llm_mod

        # 12 uptrending tickers: the deterministic engine proposes at most
        # max_positions=10 of them; the LLM approves those and tries to add
        # the remaining 2 — which must now be blocked by the cap.
        series = {
            f"UP{i}": _signal_series(_gen_candles(100.0, 0.01, seed=i, n=500))
            for i in range(12)
        }
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=10,
            max_position_pct=1.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        buy_all = _json.dumps([
            {"ticker": f"UP{i}", "action": "BUY", "reason": "add"} for i in range(12)
        ])

        async def fake_chat(messages):
            return {"text": buy_all}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        res = await optimize._hybrid_replay(series, params, start="2025-07-15", end="2026-05-15")

        # The cap must hold: at most max_positions distinct tickers bought.
        bought = {t["ticker"] for t in res.trades if t["side"] == "BUY"}
        assert len(bought) <= params.max_positions, (
            f"LLM additions exceeded the cap: bought {len(bought)} distinct "
            f"tickers (cap {params.max_positions}). The max-positions cap must "
            f"block new LLM BUYs beyond the limit."
        )

    async def test_llm_top_ups_allowed_at_cap(self, monkeypatch):
        """At the position cap, the LLM must still be able to top up a ticker
        it already holds — the cap blocks NEW positions only."""
        import json as _json

        from app import llm as llm_mod

        # 12 tickers, but only buy 3 of them via the LLM. max_positions=3 so
        # the cap is hit immediately; the LLM's later BUYs are top-ups.
        # UP0 drifts DOWN so its position value falls below the 10% position
        # cap between weekly reviews, giving the top-up room to execute.
        series = {
            f"UP{i}": _signal_series(_gen_candles(100.0, 0.02, seed=i, n=300))
            for i in range(1, 12)
        }
        series["UP0"] = _signal_series(_gen_candles(100.0, -0.01, seed=0, n=300))
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=3,
            max_position_pct=10.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        # LLM always tries to buy UP0, UP1, UP2 (the same 3 every cycle).
        buy_3 = _json.dumps([
            {"ticker": f"UP{i}", "action": "BUY", "reason": "add"} for i in range(3)
        ])

        async def fake_chat(messages):
            return {"text": buy_3}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        res = await optimize._hybrid_replay(series, params,
                                            start="2025-08-01", end="2025-10-31",
                                            pure_llm=True)

        # Distinct tickers bought = 3 (the cap). But there should be more than
        # 3 BUY trades — the top-ups go through because the tickers are held.
        bought_tickers = {t["ticker"] for t in res.trades if t["side"] == "BUY"}
        assert len(bought_tickers) <= 3, "cap exceeded"
        # Top-ups: more BUY trades than distinct tickers means some were repeats.
        n_buys = sum(1 for t in res.trades if t["side"] == "BUY")
        assert n_buys > len(bought_tickers), (
            f"no top-up BUYs happened ({n_buys} buys, {len(bought_tickers)} distinct) — "
            f"the cap blocked top-ups of held tickers"
        )


class TestPureLLMAutoStops:
    """In pure-LLM mode the engine must auto-sell positions that hit their
    initial stop or ATR trailing stop — the LLM is told to focus on
    discretionary exits, not replicate the stop logic. Before the fix the
    pure-LLM mode set proposals=[] and skipped the entire SELL phase, so
    broken positions bled indefinitely while the LLM held."""

    def _spike_then_crash_candles(self) -> list[dict]:
        """A series that uptrends for ~220 days (enough to confirm SMA200 +
        trigger BUY signals), then drops 25%+ to blow through the 15% stop."""
        # Uptrend phase: price rises from 100 to ~130 over 220 days.
        up = _gen_candles(100.0, 0.0012, seed=1, n=220)
        # Crash phase: price drops from ~130 to ~90 over 60 days (-0.6%/day).
        # The 15% stop from the ~130 entry is ~110.5; the price crosses below
        # it within ~20 days of the crash start.
        crash_start = up[-1]["close"]
        crash = _gen_candles(crash_start, -0.006, seed=99, n=60)
        # Shift crash dates to continue after the uptrend.
        up_last_date = datetime.strptime(up[-1]["time"], "%Y-%m-%d")
        for i, row in enumerate(crash, 1):
            row["time"] = (up_last_date + timedelta(days=i)).strftime("%Y-%m-%d")
            # Keep the close continuous: chain from the previous close.
            if i == 1:
                row["open"] = crash_start
        # Re-chain closes so the crash starts from the uptrend's last close.
        price = crash_start
        for row in crash:
            row["close"] = round(price * (1 + (-0.006 + random.Random(99).gauss(0, 0.01))), 2)
            row["open"] = round(price, 2)
            row["high"] = round(max(row["open"], row["close"]) * 1.002, 2)
            row["low"] = round(min(row["open"], row["close"]) * 0.998, 2)
            price = row["close"]
        return up + crash

    async def test_pure_llm_no_engine_stops(self, monkeypatch):
        """Parity with main: the engine makes NO risk-floor sells in pure-LLM
        mode — if the LLM only returns HOLDs, positions are held (no stop
        flushing, no trailing stops)."""
        import json as _json

        from app import llm as llm_mod

        series = {"CRASH": _signal_series(self._spike_then_crash_candles())}
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=0,
            max_position_pct=100.0,
            min_cash_pct=0.0,
            stop_type="percent",
            stop_pct=15.0,
            use_atr_stop=True,
        )

        # LLM: BUY on first day, HOLD everything afterward. The engine must
        # NOT sell — no risk floor in pure-LLM mode.
        call_count = [0]

        async def fake_chat(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"text": _json.dumps([
                    {"ticker": "CRASH", "action": "BUY", "reason": "uptrend"}
                ])}
            return {"text": _json.dumps([
                {"ticker": "CRASH", "action": "HOLD", "reason": "hold"}
            ])}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        res = await optimize._hybrid_replay(series, params,
                                            start="2025-08-01", end="2025-12-31",
                                            pure_llm=True)

        buys = [t for t in res.trades if t["side"] == "BUY"]
        sells = [t for t in res.trades if t["side"] == "SELL"]
        assert buys, "expected the LLM BUY to execute"
        assert not sells, (
            f"engine risk floor sold positions in pure-LLM mode: {len(sells)} "
            f"SELLs. Parity with main: the engine makes no risk-floor sells; "
            f"the LLM owns all exits."
        )

    async def test_pure_llm_llm_still_decides_buys(self, monkeypatch):
        """The auto-stop layer must not take over BUY decisions — the LLM
        keeps full buy discretion."""
        import json as _json

        from app import llm as llm_mod

        series = {"UP": _signal_series(_gen_candles(100.0, 0.01, seed=1, n=400))}
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        # LLM returns HOLD for everything — no BUYs should happen.
        async def fake_chat(messages):
            return {"text": "[]"}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        res = await optimize._hybrid_replay(series, params,
                                            start="2025-07-01", end="2025-12-31",
                                            pure_llm=True)
        buys = [t for t in res.trades if t["side"] == "BUY"]
        assert not buys, (
            "pure-LLM mode bought tickers without the LLM issuing a BUY — "
            "the auto-stop layer must not take over BUY decisions"
        )


class TestPureLLMAntiChurn:
    """Parity with main: the engine makes NO exits in pure-LLM mode — the LLM
    owns all sell decisions (no risk floor, no holding floor, no cooldown).
    The engine's hard sizing limits (min-cash, max-pos-%, max-positions cap)
    are the only constraints."""

    async def test_llm_sell_discretion_preserved(self, monkeypatch):
        """The LLM may sell a position any time — there is no holding-period
        floor (it was removed: it fought the LLM and caused churn)."""
        import json as _json

        from app import llm as llm_mod

        series = {"UP": _signal_series(_gen_candles(100.0, 0.005, seed=1, n=300))}
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=0,
            max_position_pct=100.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        call_count = [0]

        async def fake_chat(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                # Day 1: buy UP.
                return {"text": _json.dumps([
                    {"ticker": "UP", "action": "BUY", "reason": "uptrend"}
                ])}
            # Day 2: sell immediately — the LLM owns discretionary exits.
            return {"text": _json.dumps([
                {"ticker": "UP", "action": "SELL", "reason": "rotate"}
            ])}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        res = await optimize._hybrid_replay(series, params,
                                            start="2025-08-01", end="2025-08-05",
                                            pure_llm=True)

        buys = [t for t in res.trades if t["side"] == "BUY"]
        sells = [t for t in res.trades if t["side"] == "SELL"]
        assert buys, "expected the LLM BUY to execute on day 1"
        assert sells, (
            "the LLM's early SELL was blocked — the LLM must own discretionary "
            "exits; the engine makes no exits in pure-LLM mode"
        )

    async def test_sell_after_5_days_allowed(self, monkeypatch):
        """After the holding period has passed, LLM SELLs are allowed."""
        import json as _json

        from app import llm as llm_mod

        series = {"UP": _signal_series(_gen_candles(100.0, 0.005, seed=1, n=300))}
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=0,
            max_position_pct=100.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        call_count = [0]

        async def fake_chat(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"text": _json.dumps([
                    {"ticker": "UP", "action": "BUY", "reason": "uptrend"}
                ])}
            return {"text": _json.dumps([
                {"ticker": "UP", "action": "SELL", "reason": "take profit"}
            ])}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        # Run 15 days so the LLM buys on day 1 and sells on day 2 (LLM owns
        # exits — no holding floor).
        res = await optimize._hybrid_replay(series, params,
                                            start="2025-08-01", end="2025-08-21",
                                            pure_llm=True)

        sells = [t for t in res.trades if t["side"] == "SELL"]
        # The LLM's SELL (day 2) goes through — no holding floor.
        assert sells, (
            "expected the LLM SELL to execute — the LLM owns discretionary exits"
        )


class TestPureLLMBuyGuards:
    """Pure-LLM BUY behavior (parity with main): the LLM keeps full buy
    discretion on any ticker — no candidate filter, no cooldown. The engine
    only enforces hard sizing limits (min-cash, max-pos-%, max-positions cap)
    on each BUY."""

    async def test_buy_discretion_preserved_on_weak_signal(self, monkeypatch):
        """The LLM may buy any ticker it chooses — the engine does not filter
        candidates by signal strength. (The candidate filter was removed.)"""
        import json as _json

        from app import llm as llm_mod

        # A downtrending ticker with weak signals. The LLM decides to buy it —
        # that is its call to make. No engine filter blocks the entry.
        series = {"DOWN": _signal_series(_gen_candles(100.0, -0.008, seed=2, n=300))}
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=0,
            max_position_pct=100.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        async def fake_chat(messages):
            return {"text": _json.dumps([
                {"ticker": "DOWN", "action": "BUY", "reason": "contrarian"}
            ])}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        res = await optimize._hybrid_replay(series, params,
                                            start="2025-09-01", end="2025-10-31",
                                            pure_llm=True)

        buys = [t for t in res.trades if t["side"] == "BUY"]
        assert buys, (
            "the LLM's BUY on a weak-signal ticker was blocked — the candidate "
            "filter must not constrain the LLM's buy discretion"
        )

    async def test_rebuy_allowed_after_sell(self, monkeypatch):
        """Parity with main: there is no re-buy cooldown — the LLM may re-buy
        a ticker it sold, even the next day. The engine does not restrict
        LLM BUY choices."""
        import json as _json

        from app import llm as llm_mod

        series = {"UP": _signal_series(_gen_candles(100.0, 0.005, seed=1, n=300))}
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=0,
            max_position_pct=100.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        call_count = [0]

        async def fake_chat(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"text": _json.dumps([
                    {"ticker": "UP", "action": "BUY", "reason": "entry"}
                ])}
            if call_count[0] == 2:
                # Day 2: sell (LLM owns exits — allowed immediately).
                return {"text": _json.dumps([
                    {"ticker": "UP", "action": "SELL", "reason": "exit"}
                ])}
            # After the sell (day 3+), re-buy — no cooldown blocks it.
            return {"text": _json.dumps([
                {"ticker": "UP", "action": "BUY", "reason": "looks strong again"}
            ])}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        res = await optimize._hybrid_replay(series, params,
                                            start="2025-08-01", end="2025-08-31",
                                            pure_llm=True)

        sells = [t for t in res.trades if t["side"] == "SELL"]
        assert sells, "expected the LLM SELL to execute (LLM owns exits)"
        # The re-buy goes through — no cooldown in parity mode.
        rebuys = [t for t in res.trades if t["side"] == "BUY"]
        assert len(rebuys) >= 2, (
            f"expected the entry BUY + re-buy (no cooldown), got {len(rebuys)}"
        )


class TestPlanLlmBuys:
    """plan_llm_buys must plan budgets against a running cash total so the
    sum never exceeds available cash and later BUYs aren't silently dropped
    at execution."""

    async def test_sized_buys_are_deducted_sequentially(self):
        from app.strategy import StrategyParams, plan_llm_buys

        decisions = [
            {"ticker": "AAA", "action": "BUY", "amount": 600.0},
            {"ticker": "BBB", "action": "BUY", "amount": 600.0},
            {"ticker": "CCC", "action": "BUY"},
        ]
        prices = {"AAA": 50.0, "BBB": 50.0, "CCC": 50.0}

        async def price_of(t):
            return prices.get(t)

        async def value_of(t):
            return 0.0

        plan = await plan_llm_buys(
            decisions, cash=1000.0, equity=1000.0, params=StrategyParams(),
            guarded=False, price_of=price_of, value_of=value_of,
        )

        # AAA takes 600, BBB clamps to the remaining 400, CCC gets nothing.
        assert plan == {"AAA": 600.0, "BBB": 400.0}
        assert "CCC" not in plan
        assert sum(plan.values()) <= 1000.0

    async def test_unsized_split_after_sized_deduction(self):
        from app.strategy import StrategyParams, plan_llm_buys

        decisions = [
            {"ticker": "AAA", "action": "BUY", "amount": 600.0},
            {"ticker": "BBB", "action": "BUY"},
            {"ticker": "CCC", "action": "BUY"},
        ]
        prices = {"AAA": 50.0, "BBB": 50.0, "CCC": 50.0}

        async def price_of(t):
            return prices.get(t)

        async def value_of(t):
            return 0.0

        plan = await plan_llm_buys(
            decisions, cash=1000.0, equity=1000.0, params=StrategyParams(),
            guarded=False, price_of=price_of, value_of=value_of,
        )

        # AAA takes 600; the two unsized BUYs split the remaining 400.
        assert plan["AAA"] == 600.0
        assert plan.get("BBB") == plan.get("CCC") == 200.0
        assert sum(plan.values()) <= 1000.0


class TestWindowResult:
    """_WindowResult computes per-window deltas (LLM minus deterministic)."""

    def _wr(self, det_ret=1.0, llm_ret=-2.0, det_dd=12.0, llm_dd=15.0,
            det_sharpe=2.8, llm_sharpe=2.6, det_trades=52, llm_trades=94):
        from app.optimize import ReplayResult, _WindowResult
        det = ReplayResult(params=ReplayParams(),
                           total_return_pct=det_ret, max_drawdown_pct=det_dd,
                           sharpe=det_sharpe, n_trades=det_trades)
        llm = ReplayResult(params=ReplayParams(),
                           total_return_pct=llm_ret, max_drawdown_pct=llm_dd,
                           sharpe=llm_sharpe, n_trades=llm_trades)
        return _WindowResult(start="2026-06-01", end="2026-08-21", det=det, llm=llm)

    def test_return_delta(self):
        wr = self._wr(det_ret=1.6, llm_ret=-2.1)
        assert wr.return_delta == pytest.approx(-3.7)

    def test_drawdown_delta(self):
        wr = self._wr(det_dd=12.18, llm_dd=15.04)
        assert wr.drawdown_delta == pytest.approx(2.86)

    def test_sharpe_delta(self):
        wr = self._wr(det_sharpe=2.84, llm_sharpe=2.66)
        assert wr.sharpe_delta == pytest.approx(-0.18)

    def test_trades_delta(self):
        wr = self._wr(det_trades=52, llm_trades=94)
        assert wr.trades_delta == 42


class TestWalkforwardSummary:
    """_print_walkforward_summary must report mean + worst + best deltas."""

    def test_summary_prints_mean_worst_best(self, capsys):
        from app.optimize import ReplayResult, _WindowResult, _print_walkforward_summary
        wr1 = _WindowResult(
            start="d1", end="d2",
            det=ReplayResult(params=ReplayParams(), total_return_pct=1.0,
                             max_drawdown_pct=5.0, sharpe=2.0, n_trades=10),
            llm=ReplayResult(params=ReplayParams(), total_return_pct=2.0,
                             max_drawdown_pct=6.0, sharpe=1.8, n_trades=15),
        )
        wr2 = _WindowResult(
            start="d3", end="d4",
            det=ReplayResult(params=ReplayParams(), total_return_pct=1.0,
                             max_drawdown_pct=5.0, sharpe=2.0, n_trades=10),
            llm=ReplayResult(params=ReplayParams(), total_return_pct=-1.0,
                             max_drawdown_pct=8.0, sharpe=1.5, n_trades=20),
        )
        _print_walkforward_summary([wr1, wr2])
        out = capsys.readouterr().out
        # Mean return delta = (1.0 + -2.0) / 2 = -0.5
        assert "-0.50%" in out
        # Worst return delta = -2.0
        assert "-2.00%" in out
        # Best return delta = +1.0
        assert "1.00%" in out
        # Positive windows count
        assert "Positive-return windows: 1/2" in out


class TestLlmWalkforwardWindows:
    """_llm_walkforward must cut non-overlapping windows from the tail when
    there is enough history, and overlap from the front when there isn't."""

    def test_non_overlapping_when_enough_days(self, monkeypatch):
        from app.optimize import _llm_walkforward, ReplayResult, ReplayParams
        # 120 days, 4 windows of 30 → non-overlapping.
        all_days = [f"2025-01-{i:02d}" for i in range(1, 121)]

        calls = []

        def fake_det(series, params, start=None, end=None):
            calls.append(("det", start, end))
            return ReplayResult(params=params)

        async def fake_llm(series, params, start=None, end=None, pure_llm=False, news=None, review_interval=1, veto_only=False, no_llm_sells=False, minimal_prompt=False, marker_gated=False, failure_marker=False, failure_stop_outs=2, failure_drawdown=7.0):
            calls.append(("llm", start, end))
            return ReplayResult(params=params)

        monkeypatch.setattr(optimize, "_replay", fake_det)
        monkeypatch.setattr(optimize, "_hybrid_replay", fake_llm)
        monkeypatch.setattr(optimize, "_print_window_row", lambda wr: None)
        async def fake_backend():
            return {"name": "x", "model": "y"}
        monkeypatch.setattr(optimize.llm_mod, "current_backend", fake_backend)

        import asyncio
        results = asyncio.run(_llm_walkforward(
            {}, ReplayParams(), all_days,
            n_windows=4, days_per_window=30,
            pure_llm=True, start=None, end=None,
        ))
        assert len(results) == 4
        # The 4 windows should be non-overlapping and cover the tail 120 days.
        llm_calls = [(s, e) for tag, s, e in calls if tag == "llm"]
        # Each window is 30 days; windows are ordered chronologically.
        for i in range(3):
            assert llm_calls[i][1] < llm_calls[i + 1][0], "windows overlap"

    def test_overlapping_when_not_enough_days(self, monkeypatch):
        from app.optimize import _llm_walkforward, ReplayParams
        # 90 days, 4 windows of 30 → need 120, only 90. The 4th window
        # overlaps the 3rd at the front. We get 4 windows total.
        all_days = [f"2025-01-{i:02d}" for i in range(1, 91)]

        def fake_det(series, params, start=None, end=None):
            from app.optimize import ReplayResult
            return ReplayResult(params=params)

        async def fake_llm(series, params, start=None, end=None, pure_llm=False, news=None, review_interval=1, veto_only=False, no_llm_sells=False, minimal_prompt=False, marker_gated=False, failure_marker=False, failure_stop_outs=2, failure_drawdown=7.0):
            from app.optimize import ReplayResult
            return ReplayResult(params=params)

        monkeypatch.setattr(optimize, "_replay", fake_det)
        monkeypatch.setattr(optimize, "_hybrid_replay", fake_llm)
        monkeypatch.setattr(optimize, "_print_window_row", lambda wr: None)
        async def fake_backend():
            return {"name": "x", "model": "y"}
        monkeypatch.setattr(optimize.llm_mod, "current_backend", fake_backend)

        import asyncio
        results = asyncio.run(_llm_walkforward(
            {}, ReplayParams(), all_days,
            n_windows=4, days_per_window=30,
            pure_llm=True, start=None, end=None,
        ))
        # Still get 4 windows even though the 4th overlaps the 3rd at front.
        assert len(results) == 4

    def test_too_few_days_raises(self, monkeypatch):
        from app.optimize import _llm_walkforward, ReplayParams
        all_days = [f"2025-01-{i:02d}" for i in range(1, 11)]  # 10 days

        async def fake_backend():
            return {"name": "x", "model": "y"}
        monkeypatch.setattr(optimize.llm_mod, "current_backend", fake_backend)

        import asyncio
        with pytest.raises(ValueError, match="at least 30"):
            asyncio.run(_llm_walkforward(
                {}, ReplayParams(), all_days,
                n_windows=4, days_per_window=30,
                pure_llm=True, start=None, end=None,
            ))


class TestReviewInterval:
    """review_interval controls how often the LLM is consulted: on non-review
    weeks the deterministic proposals execute as-is with no LLM call.

    The review is anchored to the calendar week (mirrors the live sim): the
    LLM fires on the first trading day of each ISO week, at most once per
    week, and review_interval>1 skips that many weeks between reviews."""

    async def test_llm_called_only_on_review_days(self, monkeypatch):
        import json as _json

        from app import llm as llm_mod

        # 12 uptrending tickers so the deterministic engine proposes trades
        # most days.
        series = {
            f"UP{i}": _signal_series(_gen_candles(100.0, 0.02, seed=i, n=300))
            for i in range(12)
        }
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=10,
            max_position_pct=10.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        call_count = [0]

        async def fake_chat(messages):
            call_count[0] += 1
            # Approve everything the engine proposed.
            return {"text": "[]"}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        # 10 trading days (Aug 1-14) spanning ISO weeks 30, 31, 32; review
        # every 5 weeks → LLM called only on the first day (week 30, 1 call).
        res = await optimize._hybrid_replay(series, params,
                                            start="2025-08-01", end="2025-08-14",
                                            pure_llm=False,
                                            review_interval=5)
        assert call_count[0] == 1, (
            f"expected 1 LLM call (first day of week 30) with review_interval=5 "
            f"over 3 calendar weeks, got {call_count[0]}"
        )
        # Trades still happened on non-review days (deterministic executed).
        assert res.n_trades > 0

    async def test_review_interval_1_calls_once_per_week(self, monkeypatch):
        import json as _json

        from app import llm as llm_mod

        series = {
            f"UP{i}": _signal_series(_gen_candles(100.0, 0.02, seed=i, n=300))
            for i in range(12)
        }
        params = ReplayParams(
            start_cash=10000.0,
            monthly_allowance=0.0,
            max_positions=10,
            max_position_pct=10.0,
            min_cash_pct=0.0,
            stop_type="none",
            use_atr_stop=False,
        )

        call_count = [0]

        async def fake_chat(messages):
            call_count[0] += 1
            return {"text": "[]"}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        monkeypatch.setattr(llm_mod, "current_backend", lambda: {"name": "x", "model": "y"})
        monkeypatch.setattr(optimize.settings, "llm_backends", '[{"name":"x"}]')

        # 14 days in the synthetic series (no weekends, Aug 1-14 = weeks
        # 30, 31, 32), review every 1 → LLM called once per week (3 calls).
        res = await optimize._hybrid_replay(series, params,
                                            start="2025-08-01", end="2025-08-14",
                                            pure_llm=False,
                                            review_interval=1)
        assert call_count[0] == 3, (
            f"expected 3 LLM calls (once per calendar week) with "
            f"review_interval=1, got {call_count[0]}"
        )
