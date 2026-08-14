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

from app.analysis import compute, MIN_CANDLES


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
            "atr_stop", "atr_pct", "vol_surge", "net_score", "strength",
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