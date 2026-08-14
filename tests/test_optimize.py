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
    _signal_series,
    _split_windows,
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
