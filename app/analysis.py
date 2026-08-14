"""Deterministic technical analysis engine for Trade Sentinel.

Computes SMA-20/50/200, RSI(14), MACD(12/26/9), ATR(14), ADX(14), relative
volume, and derives a BUY / SELL / HOLD signal from a weighted technical score.

Instead of requiring every indicator to agree (a boolean AND gate that
almost never fires), each indicator contributes points to a bullish score
and a bearish score (each 0-100). The net score drives the action:

    net = bullish - bearish
    BUY  when net >= +40 and price is in a genuine uptrend
          (close > SMA-50 > SMA-200, >2% above SMA-200)
    SELL when net <= -40 and price is in a genuine downtrend
          (close < SMA-50 < SMA-200, >2% below SMA-200)
    HOLD otherwise

Industry-standard confirmation rules (see Wilder's ADX, pullback entries):

  - **ADX(14) > 25** gates the trend weight: a choppy-but-rising market
    (low ADX) earns less trend score than a strong, clean trend.
  - **RSI pullback entry**: bullish points are awarded when RSI is *rising
    from* the 40-55 zone (a pullback turning up), not when RSI is already
    stretched above 70 (overbought, which is penalized).
  - **MACD histogram rising**: momentum confirmation uses the *histogram*
    (macd - signal) turning up, not just the lagging boolean crossover.
  - **Volume surge** acts as a confirmation bonus, not a primary driver.

The returned dict shape (consumed by ``main.py`` and the chat system prompt):

    {
        "action":   "BUY" | "SELL" | "HOLD",
        "reason":   str,            # dynamic, includes the net score + key values
        "snapshot": dict,           # rounded indicator values + atr_stop / atr_pct / vol_surge / net_score / strength
        "strength": int,            # 0-100 confidence in the action
        "candles":  list[dict],     # last 120 OHLCV rows (for charting)
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import desc, select

from .db import Signal, Session

# Minimum rows required for SMA-200 to be valid *and* for the 6-day SMA-50
# lookback comparison to be non-NaN.  200 + 6 = 206.
MIN_CANDLES = 206

# Net-score thresholds for a decisive signal.
BUY_THRESHOLD = 40
SELL_THRESHOLD = -40


def _np_where(cond, a, b):
    """Element-wise where that tolerates NaN conditions (treats NaN as False)."""
    return pd.Series(np.where(cond.fillna(False), a, b), index=cond.index)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (ADX).

    Returns a single ADX series. ADX > 25 indicates a strong trend
    (regardless of direction); below 20 is a weak/choppy market.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=close.index)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def signal_series(rows: list[dict]) -> pd.DataFrame:
    """Compute the per-day indicator + scoring series for a ticker's candles.

    Returns a DataFrame with one row per candle (oldest-first) carrying the
    raw indicator values and scoring components (net, bullish, bearish, trend
    flags, distance from SMA200, ATR stop, close, time) so callers can derive
    action/strength per-parameter-set without recomputing.

    This is the single source of truth for the scoring logic: ``compute``
    reads its last row and the optimize walk-forward replay iterates every
    row, so both share the exact same weights.
    """
    d = pd.DataFrame(rows)
    c = d.close
    vol = d.volume

    # Normalize a time column: accept "time", "timestamp", or synthesize a
    # positional index when neither is present (compute() never needed one).
    if "time" not in d.columns:
        d["time"] = d["timestamp"] if "timestamp" in d.columns else d.index.astype(str)

    # --- moving averages -------------------------------------------------
    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()

    # --- RSI(14) — Wilder smoothing -------------------------------------
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down)

    # --- MACD(12, 26, 9) + histogram -------------------------------------
    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d.macd - d.macd_signal

    # --- ATR(14) ---------------------------------------------------------
    prev = c.shift()
    tr = pd.concat(
        [d.high - d.low, (d.high - prev).abs(), (d.low - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()

    # --- ADX(14) — trend strength ---------------------------------------
    d["adx"] = _adx(d.high, d.low, c, 14)

    avg_vol_20 = vol.rolling(20).mean()
    vol_surge = (vol > 1.25 * avg_vol_20) & (avg_vol_20 > 0)

    trend_up = (c > d.sma50) & (d.sma50 > d.sma200)
    trend_down = (c < d.sma50) & (d.sma50 < d.sma200)
    sma50_rising = d.sma50 > d.sma50.shift(5)
    sma50_falling = d.sma50 < d.sma50.shift(5)

    rsi_now = d.rsi
    rsi_rising = rsi_now > rsi_now.shift(1)
    rsi_falling = rsi_now < rsi_now.shift(1)
    macd_hist = d.macd_hist
    macd_hist_rising = macd_hist > macd_hist.shift(1)
    macd_hist_falling = macd_hist < macd_hist.shift(1)
    macd_bull = d.macd > d.macd_signal
    macd_bear = d.macd < d.macd_signal

    dist_above = (c / d.sma200 - 1) * 100
    dist_below = (1 - c / d.sma200) * 100

    # ADX > 25 = strong trend; gates the trend weight so choppy-but-rising
    # markets don't earn the full trend score.
    strong_trend = d.adx > 25

    # --- weighted bullish score (0-100) ----------------------------------
    # Trend is the foundation but gated by ADX strength.
    bullish = pd.Series(0.0, index=d.index)
    bullish += _np_where(trend_up & strong_trend, 25, _np_where(trend_up, 12, _np_where(c > d.sma50, 5, 0)))
    bullish += _np_where(sma50_rising, 10, 0)
    # MACD histogram rising = momentum confirmation (not just boolean crossover)
    bullish += _np_where(macd_hist_rising & macd_bull, 15, _np_where(macd_bull, 5, 0))
    # RSI pullback entry: reward RSI rising from 40-55 (pullback turning up),
    # small reward for 55-65, neutral for 65-70, penalize > 70 (overbought).
    bullish += _np_where(
        rsi_rising & (rsi_now >= 40) & (rsi_now <= 55), 20,
        _np_where((rsi_now > 55) & (rsi_now <= 65), 10,
        _np_where((rsi_now > 65) & (rsi_now <= 70), 0,
        _np_where(rsi_now > 70, -10, 0))))
    bullish += _np_where(vol_surge & (c > d.sma50), 10, 0)
    bullish += _np_where(dist_above > 5, 10, _np_where(dist_above > 2, 5, 0))
    bullish = bullish.clip(upper=100, lower=0)

    # --- weighted bearish score (0-100) -----------------------------------
    bearish = pd.Series(0.0, index=d.index)
    bearish += _np_where(trend_down & strong_trend, 25, _np_where(trend_down, 12, _np_where(c < d.sma50, 5, 0)))
    bearish += _np_where(sma50_falling, 10, 0)
    bearish += _np_where(macd_hist_falling & macd_bear, 15, _np_where(macd_bear, 5, 0))
    # RSI rally-then-fall: reward RSI falling from 45-60 (rally turning down),
    # small reward for 35-45, neutral for 30-35, penalize < 30 (oversold).
    bearish += _np_where(
        rsi_falling & (rsi_now <= 60) & (rsi_now >= 45), 20,
        _np_where((rsi_now < 45) & (rsi_now >= 35), 10,
        _np_where((rsi_now < 35) & (rsi_now >= 30), 0,
        _np_where(rsi_now < 30, -10, 0))))
    bearish += _np_where(vol_surge & (c < d.sma50), 10, 0)
    bearish += _np_where(dist_below > 5, 10, _np_where(dist_below > 2, 5, 0))
    bearish = bearish.clip(upper=100, lower=0)

    net = bullish - bearish
    atr_stop = c - 2 * d.atr14

    return pd.DataFrame({
        "time": d["time"],
        "close": c,
        "sma20": d.sma20,
        "sma50": d.sma50,
        "sma200": d.sma200,
        "rsi": d.rsi,
        "macd": d.macd,
        "macd_signal": d.macd_signal,
        "macd_hist": d.macd_hist,
        "atr14": d.atr14,
        "adx": d.adx,
        "vol_surge": vol_surge,
        "net": net,
        "bullish": bullish,
        "bearish": bearish,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "dist_above": dist_above,
        "dist_below": dist_below,
        "atr_stop": atr_stop,
    })


def decide_action(net, trend_up, trend_down, dist_above, dist_below,
                  buy_threshold, sell_threshold) -> str:
    """Derive BUY/SELL/HOLD from the net score and trend guards.

    Requires a genuine trend (and >2% distance from SMA200) so sideways chop
    stays HOLD. Shared by ``compute`` and the optimize replay.
    """
    if net >= buy_threshold and trend_up and dist_above > 2:
        return "BUY"
    if net <= sell_threshold and trend_down and dist_below > 2:
        return "SELL"
    return "HOLD"


def strength_for(action: str, bullish: float, bearish: float) -> int:
    """Derive the 0-100 confidence for an action. Shared scoring helper."""
    if action == "BUY":
        return int(round(bullish))
    if action == "SELL":
        return int(round(bearish))
    return int(round(max(bullish, bearish)))


def compute(rows: list[dict]) -> dict:
    """Compute indicators and derive a BUY/SELL/HOLD signal from OHLCV rows.

    ``rows`` must be a list of dicts with keys: timestamp, open, high, low,
    close, volume — ordered oldest-first.
    """
    if len(rows) < MIN_CANDLES:
        raise ValueError(f"Need at least {MIN_CANDLES} daily candles, got {len(rows)}")

    df = signal_series(rows)
    x = df.iloc[-1]

    trend_up = bool(x.trend_up)
    trend_down = bool(x.trend_down)
    rsi_now = float(x.rsi)
    adx_now = float(x.adx)
    macd_bull = bool(x.macd > x.macd_signal)
    macd_bear = bool(x.macd < x.macd_signal)
    vol_surge = bool(x.vol_surge)

    net = float(x.net)
    bullish = float(x.bullish)
    bearish = float(x.bearish)
    dist_above = float(x.dist_above)
    dist_below = float(x.dist_below)

    action = decide_action(net, trend_up, trend_down, dist_above, dist_below,
                           BUY_THRESHOLD, SELL_THRESHOLD)
    strength = strength_for(action, bullish, bearish)

    atr_stop = float(x.atr_stop)

    # --- dynamic reason --------------------------------------------------
    if action == "BUY":
        reason = (
            f"BUY (score {net:+.0f}): close {x.close:.2f} > SMA50 {x.sma50:.2f} > SMA200 {x.sma200:.2f}; "
            f"RSI {rsi_now:.1f}; ADX {adx_now:.0f}; MACD {'bullish' if macd_bull else 'mixed'}; "
            f"vol {'surge' if vol_surge else 'normal'}; ATR stop ~{atr_stop:.2f}."
        )
    elif action == "SELL":
        reason = (
            f"SELL (score {net:+.0f}): close {x.close:.2f} "
            f"{'< SMA50' if x.close < x.sma50 else 'vs SMA50'} {x.sma50:.2f}; "
            f"RSI {rsi_now:.1f}; ADX {adx_now:.0f}; MACD {'bearish' if macd_bear else 'mixed'}."
        )
    else:
        reason = (
            f"HOLD (score {net:+.0f}): close {x.close:.2f}, SMA50 {x.sma50:.2f}, "
            f"SMA200 {x.sma200:.2f}, RSI {rsi_now:.1f}, ADX {adx_now:.0f}, "
            f"MACD {x.macd:.3f}/{x.macd_signal:.3f}."
        )

    # --- snapshot --------------------------------------------------------
    def norm(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(v) else round(v, 2)

    snap = {k: norm(x[k]) for k in ["close", "sma20", "sma50", "sma200", "rsi", "macd", "macd_signal", "macd_hist", "atr14", "adx"]}
    snap["atr_stop"] = norm(atr_stop)
    snap["atr_pct"] = norm(100 * float(x.atr14) / float(x.close))
    snap["vol_surge"] = vol_surge
    snap["net_score"] = int(net)
    snap["strength"] = strength

    return {
        "action": action,
        "reason": reason,
        "snapshot": snap,
        "strength": strength,
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