"""Tests for the walk-forward optimization / backtest tool (app.optimize).

These test the pure, deterministic parts of the tool (signal series, replay,
parameter sweep) using synthetic candles — no DB, no network.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from app import optimize
from app.optimize import (
    ReplayParams,
    _replay,
    _row_action,
    _row_strength,
    _select_cases,
    _signal_series,
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
                    "atr_stop", "weekly_trend_up"):
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
