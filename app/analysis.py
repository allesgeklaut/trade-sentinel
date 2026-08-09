"""Deterministic technical analysis engine for Trade Sentinel.

Computes SMA-20/50/200, RSI(14), MACD(12/26/9), ATR(14), relative volume,
and derives a BUY / SELL / HOLD signal with a 0-100 strength score and a
data-grounded human-readable reason.

The returned dict shape (consumed by ``main.py`` and the chat system prompt):

    {
        "action":   "BUY" | "SELL" | "HOLD",
        "reason":   str,            # dynamic, includes actual indicator values
        "snapshot": dict,           # rounded indicator values + atr_stop / atr_pct / vol_surge / strength
        "strength": int,            # 0-100
        "candles":  list[dict],     # last 120 OHLCV rows (for charting)
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import desc, select

from .db import Signal, Session

# Minimum rows required for SMA-200 to be valid *and* for the 6-day SMA-50
# lookback comparison to be non-NaN.  200 + 6 = 206.
MIN_CANDLES = 206


def compute(rows: list[dict]) -> dict:
    """Compute indicators and derive a BUY/SELL/HOLD signal from OHLCV rows.

    ``rows`` must be a list of dicts with keys: timestamp, open, high, low,
    close, volume — ordered oldest-first.
    """
    if len(rows) < MIN_CANDLES:
        raise ValueError(f"Need at least {MIN_CANDLES} daily candles, got {len(rows)}")

    d = pd.DataFrame(rows)
    c = d.close
    vol = d.volume

    # --- moving averages -------------------------------------------------
    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()

    # --- RSI(14) — Wilder smoothing -------------------------------------
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down)

    # --- MACD(12, 26, 9) -------------------------------------------------
    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()

    # --- ATR(14) ---------------------------------------------------------
    prev = c.shift()
    tr = pd.concat(
        [d.high - d.low, (d.high - prev).abs(), (d.low - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()

    x = d.iloc[-1]

    # --- volume ----------------------------------------------------------
    avg_vol_20 = vol.tail(20).mean()
    vol_surge = bool(x.volume > 1.25 * avg_vol_20) if avg_vol_20 > 0 else False

    # --- trend -----------------------------------------------------------
    trend_up = bool(x.close > x.sma50 > x.sma200)
    trend_down = bool(x.close < x.sma50 < x.sma200)
    sma50_rising = bool(x.sma50 > d.sma50.iloc[-6])
    sma50_falling = bool(x.sma50 < d.sma50.iloc[-6])

    # --- momentum --------------------------------------------------------
    rsi_now = float(x.rsi)
    rsi_prev = float(d.rsi.iloc[-2])
    rsi_rising = rsi_now > rsi_prev
    fresh_cross = rsi_now > 50 and rsi_prev <= 50
    breakdown = rsi_now < 50 and rsi_prev >= 50
    not_overbought = rsi_now < 75
    not_oversold = rsi_now > 25
    macd_bull = bool(x.macd > x.macd_signal)
    macd_bear = bool(x.macd < x.macd_signal)

    # --- decisions -------------------------------------------------------
    bullish = (
        trend_up
        and sma50_rising
        and (fresh_cross or (50 < rsi_now <= 70 and rsi_rising))
        and macd_bull
        and not_overbought
        and vol_surge
    )
    early_exit = bool(x.close < x.sma50) and macd_bear and breakdown
    bearish = trend_down and sma50_falling
    sell = early_exit or bearish

    if bullish and not sell:
        action = "BUY"
    elif sell:
        action = "SELL"
    else:
        action = "HOLD"

    # --- strength score (0-100) -----------------------------------------
    strength = 0
    if trend_up:
        strength += 40
    if sma50_rising:
        strength += 15
    if macd_bull:
        strength += 15
    if fresh_cross:
        strength += 20
    if vol_surge:
        strength += 10
    strength = min(strength, 100)
    if action == "SELL":
        strength = 100 - strength  # invert: a clean bearish setup scores high

    # --- dynamic reason --------------------------------------------------
    atr_stop = float(x.close) - 2 * float(x.atr14)
    if action == "BUY":
        reason = (
            f"BUY: close {x.close:.2f} > SMA50 {x.sma50:.2f} > SMA200 {x.sma200:.2f}; "
            f"RSI {rsi_now:.1f} ({'cross' if fresh_cross else 'rising'}); "
            f"MACD {x.macd:.3f} > signal {x.macd_signal:.3f}; "
            f"vol {'surge' if vol_surge else 'normal'}; "
            f"ATR stop ~{atr_stop:.2f}."
        )
    elif action == "SELL":
        reason = (
            f"SELL: {'early exit — ' if early_exit else ''}"
            f"close {x.close:.2f} {'< SMA50' if trend_down else '< SMA50 (early)'} "
            f"{x.sma50:.2f}; RSI {rsi_now:.1f}; "
            f"MACD {x.macd:.3f} < signal {x.macd_signal:.3f}."
        )
    else:
        reason = (
            f"HOLD: no complete setup. close {x.close:.2f}, "
            f"SMA50 {x.sma50:.2f}, SMA200 {x.sma200:.2f}, "
            f"RSI {rsi_now:.1f}, MACD {x.macd:.3f}/{x.macd_signal:.3f}."
        )

    # --- snapshot --------------------------------------------------------
    def norm(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(v) else round(v, 2)

    snap = {k: norm(x.get(k)) for k in ["close", "sma20", "sma50", "sma200", "rsi", "macd", "macd_signal", "atr14"]}
    snap["atr_stop"] = norm(atr_stop)
    snap["atr_pct"] = norm(100 * float(x.atr14) / float(x.close))
    snap["vol_surge"] = vol_surge
    snap["strength"] = int(strength)

    return {
        "action": action,
        "reason": reason,
        "snapshot": snap,
        "strength": int(strength),
        "candles": rows,  # full period — market.py already sliced by range
    }


async def persist(ticker: str, result: dict) -> bool:
    """Persist a signal, skipping if identical to the most recent one.

    Returns ``True`` if a new row was inserted, ``False`` if deduplicated.
    """
    async with Session() as s:
        last = await s.scalar(
            select(Signal)
            .where(Signal.ticker == ticker)
            .order_by(desc(Signal.created_at))
            .limit(1)
        )
        if (
            last
            and last.action == result["action"]
            and last.reason == result["reason"]
        ):
            return False
        s.add(
            Signal(
                ticker=ticker,
                action=result["action"],
                reason=result["reason"],
                snapshot=json.dumps(result["snapshot"]),
                strength=result["strength"],
            )
        )
        await s.commit()
        return True


async def history(ticker: str) -> list[dict]:
    async with Session() as s:
        r = (
            await s.scalars(
                select(Signal)
                .where(Signal.ticker == ticker)
                .order_by(desc(Signal.created_at))
                .limit(20)
            )
        ).all()
        return [
            {
                "at": x.created_at.isoformat(),
                "action": x.action,
                "reason": x.reason,
                "strength": x.strength,
                "snapshot": json.loads(x.snapshot),
            }
            for x in r
        ]