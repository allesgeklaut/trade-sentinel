"""Tests for app.analysis.compute using synthetic OHLCV series.

Each test builds a deterministic price series designed to trigger a specific
signal path (BULLISH, BEARISH, early-exit SELL, HOLD/choppy), then asserts
that compute() returns the expected action and that key snapshot fields are
populated correctly.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from app.analysis import (
    ATR_STOP_LOOKBACK,
    ATR_STOP_MULT,
    MIN_CANDLES,
    compute,
    decide_action,
    entry_eligible,
    signal_series,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_candles(
    n: int,
    close_fn,
    *,
    base_volume: float = 1_000_000,
    volume_fn=None,
    start_price: float = 100.0,
) -> list[dict]:
    """Build *n* daily candles.

    ``close_fn(i, prev_close)`` returns the close for day *i*.
    High/low/open are derived from the close with a small synthetic spread.
    """
    rows: list[dict] = []
    prev_close = start_price
    base = datetime(2024, 1, 1)
    for i in range(n):
        close = close_fn(i, prev_close)
        spread = close * 0.005
        op = prev_close  # open = previous close (simple)
        hi = max(op, close) + spread
        lo = min(op, close) - spread
        vol = volume_fn(i) if volume_fn else base_volume
        rows.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": round(op, 4),
                "high": round(hi, 4),
                "low": round(lo, 4),
                "close": round(close, 4),
                "volume": float(vol),
            }
        )
        prev_close = close
    return rows


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_raises_on_short_series(self):
        rows = _build_candles(50, lambda i, p: p)
        with pytest.raises(ValueError, match=str(MIN_CANDLES)):
            compute(rows)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            compute([])


class TestBullish:
    def test_buy_signal(self):
        """A realistic uptrend with periodic pullbacks that should trigger BUY.

        We build a series that:
        - starts flat for 100 days so SMA-200 is well below current price,
        - then alternates up (+0.6) and down (-0.4) days, creating a net
          uptrend while keeping RSI in the 58-62 range (not overbought),
        - the last bar is an up day so MACD > signal and RSI is rising,
        - and a volume surge on the last 5 bars.

        A constant linear climb (no down days) would push RSI → ~100 and
        trigger the overbought filter, so periodic pullbacks are essential.
        """

        def close_fn(i, prev):
            if i < 100:
                return 100.0  # flat base
            day = i - 100
            # Alternating up/down with net positive drift keeps RSI ~60
            if day % 2 == 0:
                return prev + 0.6  # up day
            return prev - 0.4  # down day

        def vol_fn(i):
            # Surge on the last 5 bars (last index = 252, day=152, even → up)
            if i >= 248:
                return 2_000_000  # 2× base → surge
            return 1_000_000

        # n=253 → last index 252, day=152 (even → up day)
        rows = _build_candles(253, close_fn, volume_fn=vol_fn)
        result = compute(rows)

        assert result["action"] == "BUY"
        assert result["strength"] > 50
        assert "BUY" in result["reason"]
        snap = result["snapshot"]
        assert snap["close"] > snap["sma50"]
        assert snap["sma50"] > snap["sma200"]
        assert snap["rsi"] is not None and 50 < snap["rsi"] < 75
        assert snap["vol_surge"] is True
        assert snap["atr_stop"] is not None
        assert snap["atr_pct"] is not None
        assert snap["strength"] == result["strength"]


class TestBearish:
    def test_sell_signal_death_cross(self):
        """A steady downtrend triggering the full bearish SELL path."""

        def close_fn(i, prev):
            if i < 100:
                return 200.0 - 0.01 * i  # nearly flat high
            return 200.0 - 0.35 * (i - 100)  # steady decline

        rows = _build_candles(250, close_fn)
        result = compute(rows)

        assert result["action"] == "SELL"
        assert "SELL" in result["reason"]
        snap = result["snapshot"]
        assert snap["close"] < snap["sma50"]
        assert snap["sma50"] < snap["sma200"]
        assert result["strength"] >= 0  # inverted score


class TestEarlyExit:
    def test_single_dip_in_uptrend_is_hold(self):
        """A single sharp drop in a strong uptrend should NOT trigger SELL.

        The old model sold on any day the close dipped below SMA-50 with a
        bearish MACD. The scoring model is more conservative: a one-day dip
        in a genuine uptrend is noise, not a trend reversal, so it stays HOLD.
        """
        def close_fn(i, prev):
            if i < 200:
                return 100.0 + 0.3 * i  # strong uptrend to ~160
            if i < 300:
                offset = i - 200
                return prev + (0.01 if offset % 2 == 1 else -0.01)
            return prev - 1.0  # single sharp drop

        rows = _build_candles(301, close_fn)
        result = compute(rows)
        assert result["action"] == "HOLD"

    def test_sell_signal_death_cross(self):
        """A steady downtrend should produce SELL under the scoring model."""
        def close_fn(i, prev):
            if i < 100:
                return 200.0 - 0.01 * i  # nearly flat high
            return 200.0 - 0.35 * (i - 100)  # steady decline

        rows = _build_candles(250, close_fn)
        result = compute(rows)
        assert result["action"] == "SELL"
        assert "SELL" in result["reason"]


class TestHold:
    def test_hold_on_sideways(self):
        """A flat / sideways market should produce HOLD."""

        def close_fn(i, prev):
            # Oscillate in a tight band around 100
            return 100.0 + 0.5 * math.sin(i * 0.15)

        rows = _build_candles(250, close_fn)
        result = compute(rows)

        assert result["action"] == "HOLD"
        assert "HOLD" in result["reason"]
        # In a sideways market some minor conditions (macd_bull, sma50_rising,
        # fresh_cross) may be marginally true, pushing strength to ~50.
        # HOLD is correct as long as strength doesn't exceed 50.
        assert result["strength"] <= 50


class TestSnapshotShape:
    def test_snapshot_keys(self):
        rows = _build_candles(250, lambda i, p: 100.0 + 0.1 * i)
        result = compute(rows)
        snap = result["snapshot"]
        expected_keys = {
            "close", "sma20", "sma50", "sma200",
            "rsi", "macd", "macd_signal", "macd_hist", "atr14", "adx",
            "atr_stop", "atr_pct", "vol_surge", "weekly_trend_up", "net_score", "strength",
            "rsi_3d_change", "macd_hist_3d_change", "run_5d",
            "run_20d", "run_60d", "dist_52w_high", "dist_above",
            "trend_up", "trend_down",
        }
        assert set(snap.keys()) == expected_keys

    def test_candles_truncated(self):
        rows = _build_candles(300, lambda i, p: 100.0 + 0.1 * i)
        result = compute(rows)
        assert len(result["candles"]) == 300  # full dataset returned, not truncated

    def test_candles_not_truncated_when_short(self):
        rows = _build_candles(206, lambda i, p: 100.0 + 0.1 * i)
        result = compute(rows)
        assert len(result["candles"]) == 206  # full dataset returned, not truncated


class TestVolume:
    def test_no_vol_surge_when_flat(self):
        rows = _build_candles(250, lambda i, p: 100.0 + 0.2 * i, base_volume=1_000_000)
        result = compute(rows)
        assert result["snapshot"]["vol_surge"] is False

    def test_vol_surge_when_spiked(self):
        rows = _build_candles(
            250,
            lambda i, p: 100.0 + 0.2 * i,
            volume_fn=lambda i: 3_000_000 if i >= 248 else 1_000_000,
        )
        result = compute(rows)
        assert result["snapshot"]["vol_surge"] is True


# ---------------------------------------------------------------------------
# ATR stop: prior-known, reachable level (Phase 1 financial correctness)
# ---------------------------------------------------------------------------

def _uptrend_rows(n: int = 260) -> list[dict]:
    """A steady uptrend with a small daily spread (ATR > 0)."""
    return _build_candles(n, lambda i, p: 100.0 + 0.3 * i)


class TestAtrStop:
    """The ATR stop used to be ``close - 2 * ATR`` on the *same* bar that was
    then compared against that close — unreachable by construction, so the
    stop could never fire. It is now a prior-known level built from the
    previous completed bar only."""

    def test_stop_is_below_the_close_so_it_can_trigger(self):
        rows = _uptrend_rows()
        snap = compute(rows)["snapshot"]
        # A reachable stop sits below the current close in an uptrend.
        assert snap["atr_stop"] is not None
        assert snap["atr_stop"] < snap["close"], (
            "the ATR stop must be reachable — it has to sit below the close "
            "for the close to be able to break it"
        )

    def test_stop_matches_the_prior_bar_formula(self):
        rows = _uptrend_rows()
        df = signal_series(rows)
        expected = (
            df["close"].rolling(ATR_STOP_LOOKBACK).max().shift(1)
            - ATR_STOP_MULT * df["atr14"].shift(1)
        )
        assert df["atr_stop"].iloc[-1] == pytest.approx(expected.iloc[-1])
        # ... and the snapshot publishes exactly that level.
        snap = compute(rows)["snapshot"]
        assert snap["atr_stop"] == pytest.approx(round(expected.iloc[-1], 2))

    def test_stop_ignores_the_current_bar(self):
        """The level must not move using the bar it is checked against:
        crashing the last close leaves the published stop unchanged."""
        rows = _uptrend_rows()
        before = compute(rows)["snapshot"]["atr_stop"]

        crashed = [dict(r) for r in rows]
        last = crashed[-1]
        last_close = last["close"] * 0.5
        last["close"] = last_close
        last["low"] = min(last["low"], last_close)
        last["high"] = max(last["high"], last["close"])
        after = compute(crashed)["snapshot"]["atr_stop"]

        assert after == before, (
            "the ATR stop changed when only the current bar changed — the "
            "level must be known before that bar closes"
        )

    def test_old_same_bar_formula_was_unreachable(self):
        """Guards the regression: the old level (close - 2*ATR) can never be
        broken by the same close, so it must not be what we publish."""
        rows = _uptrend_rows()
        snap = compute(rows)["snapshot"]
        old_level = snap["close"] - ATR_STOP_MULT * snap["atr14"]
        assert snap["close"] >= old_level          # unreachable by construction
        assert snap["atr_stop"] != pytest.approx(old_level)

    def test_percent_stop_is_unaffected(self):
        """The frozen percent stop lives in strategy.py and is independent of
        the ATR candidate level."""
        from app.strategy import StrategyParams, propose_trades

        params = StrategyParams(stop_type="percent", stop_pct=10.0,
                                use_atr_stop=False)
        positions = [{"ticker": "AAA", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(
            positions, 0.0, {"AAA": 89.0},
            {"AAA": {"action": "HOLD", "strength": 10, "reason": "hold",
                     "snapshot": {"atr_stop": 1.0, "close": 89.0}}},
            params,
        )
        assert [p["side"] for p in out] == ["SELL"]
        assert out[0]["reason"].startswith("Initial stop")


# ---------------------------------------------------------------------------
# Shared entry gate
# ---------------------------------------------------------------------------

class TestEntryEligible:
    """One definition of "entry eligible", shared by the strict BUY and the
    relaxed HOLD fallback."""

    def test_uptrend_above_2pct_and_weekly_up(self):
        assert entry_eligible(True, 5.0, True) is True

    def test_requires_daily_uptrend(self):
        assert entry_eligible(False, 5.0, True) is False

    def test_requires_more_than_2pct_above_sma200(self):
        assert entry_eligible(True, 2.0, True) is False
        assert entry_eligible(True, -3.0, True) is False

    def test_requires_weekly_uptrend(self):
        assert entry_eligible(True, 5.0, False) is False

    def test_missing_distance_is_not_eligible(self):
        assert entry_eligible(True, None, True) is False

    def test_decide_action_uses_the_same_gate(self):
        # Trend up but only 1% above the SMA200 → HOLD, not BUY.
        assert decide_action(60, True, False, 1.0, 0.0, 40, -40, True) == "HOLD"
        # Same score, now above the SMA200 → BUY.
        assert decide_action(60, True, False, 5.0, 0.0, 40, -40, True) == "BUY"
        # Weekly trend down → HOLD.
        assert decide_action(60, True, False, 5.0, 0.0, 40, -40, False) == "HOLD"
