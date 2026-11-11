"""Walk-forward optimization / backtest tool for the deterministic strategy.

This is a *separate, read-only* analysis tool. It never touches the live sim
account, positions, or trades. It replays the deterministic decision rules
against stored historical candles and reports how well a given parameter set
would have performed.

ONE EXPLICIT EXCEPTION: the ``prune-candles`` subcommand is a maintenance
command, not an analysis — it deletes ``Candle`` rows from the SHARED live
database (the same candle cache every engine reads), so it is the one
command here that is not read-only. Its 8-year guard below exists because
the savings mirror reads 8 years of that same history back.

Run inside the container (the DB lives in the Docker volume):

    docker compose exec trade-sentinel python -m app.optimize --help

Stages:
  1. Backtest replay  — score the *current* rules on stored candle history.
  2. Parameter sweep  — grid-search thresholds, report best in-sample params.
  3. Walk-forward     — fit on train windows, score on following test windows
                        to detect overfitting / regime change.

The indicator series are precomputed once per ticker (vectorized) so the
per-day replay is cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete as sa_delete, func, select

from .analysis import (
    MIN_CANDLES,
    decide_action,
    signal_series as _signal_series,
    snapshot_from_row,
    strength_for,
)
from .config import settings
from . import factor_fills
from . import llm as llm_mod
from . import market_index as market_index_mod
from . import price_freshness
from .db import Candle, Session
from .screener import tickers as universe_tickers
from .strategy import (
    StrategyParams,
    llm_sell_shares,
    plan_llm_buys,
    propose_trades,
    reconcile_proposals,
    valuate_portfolio,
)

logger = logging.getLogger("trade_sentinel.optimize")


async def _aval(value: float | None) -> float | None:
    """Wrap a sync value as an awaitable for plan_llm_buys' async callbacks."""
    return value


async def _val_of(pf: PaperPortfolio, t: str, prices: dict[str, float]) -> float:
    """Value currently held in `t` (0.0 when unheld) as an awaitable."""
    return float(pf.positions.get(t, 0) * prices.get(t, 0))


# ---------------------------------------------------------------------------
# Paper portfolio
# ---------------------------------------------------------------------------

@dataclass
class PaperPortfolio:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # ticker -> shares
    avg_cost: dict[str, float] = field(default_factory=dict)
    peak_price: dict[str, float] = field(default_factory=dict)  # ticker -> highest close since buy
    stop_price: dict[str, float] = field(default_factory=dict)  # ticker -> frozen initial stop
    thesis: dict[str, str] = field(default_factory=dict)  # ticker -> BUY reason (entry thesis)
    buy_date: dict[str, str] = field(default_factory=dict)  # ticker -> date of first BUY
    trades: list[dict] = field(default_factory=list)

    def buy(self, ticker: str, price: float, budget: float, reason: str,
            stop: float | None = None, date: str = "") -> None:
        if budget < 1 or price <= 0:
            return
        # Floor (not round) the fractional share count so cost never exceeds
        # the budget — rounding up can push a high-priced ticker cents over
        # the available cash and silently drop the buy.
        shares = math.floor(budget / price * 10000) / 10000
        if shares < 0.0001:
            return
        cost = shares * price
        if cost > self.cash:
            return
        self.cash -= cost
        if ticker in self.positions:
            total = self.positions[ticker] + shares
            self.avg_cost[ticker] = (self.positions[ticker] * self.avg_cost[ticker] + cost) / total
            self.positions[ticker] = total
        else:
            self.positions[ticker] = shares
            self.avg_cost[ticker] = price
            self.peak_price[ticker] = price
            if stop is not None:
                self.stop_price[ticker] = stop
            # Record the entry thesis and buy date for the context feedback
            # loop — the LLM sees why it bought each position and for how long.
            self.thesis[ticker] = reason
            self.buy_date[ticker] = date
        self.trades.append({"ticker": ticker, "side": "BUY", "shares": shares,
                            "price": price, "reason": reason, "date": date})

    def sell(self, ticker: str, price: float, shares: float | None, reason: str,
             date: str = "") -> None:
        if ticker not in self.positions or self.positions[ticker] <= 0:
            return
        sell_shares = self.positions[ticker] if shares is None else min(shares, self.positions[ticker])
        if sell_shares <= 0:
            return
        self.cash += sell_shares * price
        self.positions[ticker] -= sell_shares
        if self.positions[ticker] <= 0.0001:
            del self.positions[ticker]
            del self.avg_cost[ticker]
            self.peak_price.pop(ticker, None)
            self.stop_price.pop(ticker, None)
            self.thesis.pop(ticker, None)
            self.buy_date.pop(ticker, None)
        self.trades.append({"ticker": ticker, "side": "SELL", "shares": sell_shares,
                            "price": price, "reason": reason, "date": date})

    def update_peaks(self, prices: dict[str, float]) -> None:
        """Update the peak-price tracker for held positions."""
        for ticker in self.positions:
            price = prices.get(ticker)
            if price is not None and price > self.peak_price.get(ticker, 0):
                self.peak_price[ticker] = price

    def equity(self, prices: dict[str, float]) -> float:
        pos_value = sum(shares * prices.get(t, 0) for t, shares in self.positions.items())
        return self.cash + pos_value


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

@dataclass
class ReplayParams:
    buy_threshold: int = 40
    sell_threshold: int = -40
    min_cash_pct: float = settings.sim_min_cash_pct
    max_position_pct: float = settings.sim_max_position_pct
    relaxed_hold_strength: int = 40
    relaxed_hold_limit: int = 3
    use_atr_stop: bool = True
    trailing_stop_pct: float = 0.0  # 0 = disabled; e.g. 15 = sell if price drops 15% from peak
    monthly_allowance: float = settings.sim_monthly_allowance
    start_cash: float = settings.sim_start_cash
    # Regime filter: when enabled, new BUYs are blocked while the market
    # benchmark is below its 200-day SMA (broad-market downtrend).
    regime_filter: bool = False
    regime_ticker: str = "URTH"
    # --- Risk management (industry-standard trend-following controls) ---
    # Initial stop loss: frozen at entry, not updated daily. Prevents
    # catastrophic losses before the SELL signal (death cross) fires.
    #   "none"   = disabled
    #   "percent" = entry_price * (1 - stop_pct/100)
    #   "atr"    = entry_price - stop_atr_mult * ATR_at_entry
    stop_type: str = "none"
    stop_pct: float = 15.0         # percentage drop from entry (stop_type="percent")
    stop_atr_mult: float = 2.0     # ATR multiple (stop_type="atr")
    # Risk-based position sizing: risk this % of equity per trade, sized
    # by stop distance. 0 = use flat max_position_pct.
    risk_pct: float = 0.0
    # Sector diversification cap: max % of equity in any one sector. 0 = disabled.
    max_sector_pct: float = 0.0
    # Portfolio-level circuit breaker: if total equity drops this % from its
    # peak, sell all positions and block new BUYs for the rest of the window.
    # 0 = disabled. e.g. 20 = deleverage when portfolio is down 20% from peak.
    portfolio_stop_pct: float = 0.0
    # Max simultaneous open positions. 0 = unlimited. Forces diversification
    # so no single crash can sink the portfolio.
    max_positions: int = 0
    # Max 5-day run-up allowed before blocking a BUY. 0 = disabled.
    # e.g. 15 = don't buy if price has risen more than 15% in the last 5 days
    # (chasing a short-term spike that's prone to reversion).
    max_run_5d: float = 0.0
    # Mirror on the downside: block BUYs into a 5-day crash (falling-knife
    # guard). Only active when negative; 0 disables.
    min_run_5d: float = 0.0
    # Block BUYs into parabolic extensions: dist_above SMA200 > this %.
    # 0 = disabled.
    max_dist_above: float = 0.0
    # Quality guard: block BUY entries for names with KNOWN non-positive ROE.
    # Missing fundamentals stay neutral (ETFs, thin coverage). Requires the
    # caller to pass a quality lookup (see _QualityLookup). Off by default
    # until the replay A/B verdict.
    block_negative_roe: bool = False


# Sector groupings for the global-large-cap universe. Used by the sector
# diversification cap to prevent correlated positions from concentrating risk.
# Covers both global-large-cap and diversified-plus (the + extensions added
# 2026-08: semis, networking, defence primes, electrification, cyber, AI apps).
SECTORS: dict[str, set[str]] = {
    "semiconductors": {"NVDA", "AMD", "AVGO", "TSM", "ASML.AS", "ASML", "MU", "ARM",
                       "MRVL", "QCOM", "ANET", "SOXX", "SMH", "IFX.DE",
                       "BESI.AS", "NEM.DE", "NOKIA.HE", "ENR.DE", "TER",
                       "AMAT", "LRCX", "KLAC", "SMCI"},
    "hyperscalers": {"MSFT", "GOOGL", "AMZN", "META", "ORCL", "PLTR", "NOW",
                     "CRM", "ADBE", "SNOW", "DDOG", "MDB", "AI", "SOUN",
                     "PATH", "UPST", "TEM", "RGTI", "IONQ", "RKLB", "CRWV",
                     "FIG", "CRCL", "APP"},
    "european_tech": {"SAP.DE", "SIE.DE", "AMS.MC", "DSY.PA", "AI.PA",
                     "HO.PA", "SAAB-B.ST", "SOF.BR"},
    "space_defense": {"SPCX", "ASTS", "LUNR", "RDW", "KTOS", "PL", "IRDM",
                      "LMT", "RTX"},
    "data_center_energy": {"ETN", "GEV", "CEG", "VST", "VRT", "PWR", "HUBB"},
    "industrial": {"ROK", "ISRG", "CGNX", "DELL"},
    "networking_legacy_it": {"IBM", "CSCO", "NET", "ZS"},
    "cybersecurity": {"CRWD", "PANW", "ZS", "NET"},
    "medical": {"ISRG", "SYK", "MDT", "SHL.DE", "CRSP", "VEEV", "GH", "BNTX"},
    "pharma": {"LLY", "JNJ", "UNH", "PFE", "TMO", "XLV"},
    "financials": {"JPM", "GS", "V", "BLK", "XLF"},
    "consumer": {"WMT", "PG", "COST", "KO", "HD", "XLP"},
    "energy": {"XOM", "CVX", "XLE"},
    "utilities_bonds": {"NEE", "TLT", "XLU", "GLD"},
    "broad_etf": {"QQQ", "SPY", "VOO", "VT", "URTH"},
}

_TICKER_SECTOR: dict[str, str] = {}
for _sector, _tickers in SECTORS.items():
    for _t in _tickers:
        _TICKER_SECTOR[_t] = _sector


def _sector_of(ticker: str) -> str:
    """Return the sector group for a ticker, or 'other' if unknown."""
    return _TICKER_SECTOR.get(ticker, "other")


@dataclass
class ReplayResult:
    params: ReplayParams
    equity_curve: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    n_trades: int = 0
    cash_curve: list[float] = field(default_factory=list)  # cash per day, aligned with equity_curve
    position_count_curve: list[int] = field(default_factory=list)  # open positions per day


def _sharpe(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    if var <= 0:
        return 0.0
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(252)


def _max_drawdown(equity: list[float], invested: list[float] | None = None) -> float:
    """Max peak-to-trough drawdown, in PERCENT — the shared cash-flow-neutral
    definition from ``app.performance`` (audit P0-B).

    With ``invested`` (cumulative contributions per point) the drawdown is
    measured on the contribution-adjusted unitized NAV, so a deposit that
    leaves the portfolio value unchanged can no longer read as a loss (the
    old equity/invested-ratio arithmetic reported an 8.33% drawdown for the
    audit's deposit-only reproduction). Without ``invested`` the plain
    equity drawdown is used. One definition for the backend and the UI.
    """
    from .performance import max_drawdown
    return max_drawdown(equity, invested)


# Warmup kept ahead of a replay start when bounding the series load. Signals
# look backward (SMA200 needs 206 candles), so dropping rows older than this
# cannot change any value from the replay start onward.
_SERIES_WARMUP_DAYS = 500


async def _load_series(tickers: list[str],
                       start: datetime | None = None) -> dict[str, pd.DataFrame]:
    """Load and precompute the signal series for each ticker.

    ``start`` optionally bounds the load to a warmup window before the replay
    start (``None`` = full history, for full-range replays). The old version
    ran one query PER ticker and hydrated full ORM rows, which is what made
    the broad-universe daily-sim backfill slow."""
    series: dict[str, pd.DataFrame] = {}
    since = ((pd.Timestamp(start) - pd.Timedelta(days=_SERIES_WARMUP_DAYS)).to_pydatetime()
             if start else None)
    async with Session() as s:
        q = (select(Candle.ticker, Candle.timestamp, Candle.open, Candle.high,
                    Candle.low, Candle.close, Candle.volume)
             .where(Candle.ticker.in_(list(tickers))))
        if since is not None:
            q = q.where(Candle.timestamp >= since)
        rows = (await s.execute(q.order_by(Candle.ticker, Candle.timestamp))).all()
    by_ticker: dict[str, list] = {}
    for r in rows:
        by_ticker.setdefault(r.ticker, []).append(r)
    for t, rs in by_ticker.items():
        if len(rs) < MIN_CANDLES:
            continue
        data = [{
            "time": r.timestamp.strftime("%Y-%m-%d"),
            "open": r.open, "high": r.high, "low": r.low,
            "close": r.close, "volume": r.volume,
        } for r in rs]
        try:
            ser = _signal_series(data)
        except Exception as e:
            logger.warning("Could not compute series for %s: %s", t, e)
            continue
        # Carry F's strict session bar alongside the indicators: the session
        # fill contract needs the OPEN (fill price) and the LOW (stop trigger)
        # per session. ``_signal_series`` returns closes/indicators only, so
        # the three raw bar columns are attached here — minimally, no ffill
        # (a missing open stays NaN = no fill that session).
        ser = ser.copy()
        ser["open"] = [r["open"] for r in data]
        ser["high"] = [r["high"] for r in data]
        ser["low"] = [r["low"] for r in data]
        series[t] = ser
    return series


async def _load_regime(ticker: str) -> dict[str, bool]:
    """Load the market regime state per day.

    Returns a dict mapping ``"YYYY-MM-DD"`` → ``True`` if the market is in an
    uptrend (close > SMA200) on that day, ``False`` otherwise. Used by the
    regime filter to block new BUYs during broad-market downtrends.
    """
    async with Session() as s:
        rows = (
            await s.scalars(
                select(Candle).where(Candle.ticker == ticker).order_by(Candle.timestamp)
            )
        ).all()
    if len(rows) < MIN_CANDLES:
        logger.warning("Regime ticker %s has only %d candles; regime filter disabled", ticker, len(rows))
        return {}
    df = pd.DataFrame([{
        "time": r.timestamp.strftime("%Y-%m-%d"),
        "close": r.close,
    } for r in rows])
    df["sma200"] = df["close"].rolling(200).mean()
    df["uptrend"] = df["close"] > df["sma200"]
    valid = df["sma200"].notna()
    return {t: bool(u) for t, u in
            zip(df.loc[valid, "time"], df.loc[valid, "uptrend"], strict=False)}


def _col_f(v) -> float:
    """pandas scalar -> float via numpy (pyright-stub friendly)."""
    return float(np.asarray(v).reshape(-1)[0])


def _col_opt(v) -> float | None:
    return None if pd.isna(v) else _col_f(v)


def _col_b(v) -> bool:
    return bool(v)


def px_of_col(frame: pd.DataFrame, d: pd.Timestamp, col: str) -> float | None:
    """frame.at[d, col] as float; None when missing/NaN (pyright-friendly).

    This is the research engines' STRICT exact-date lookup — the same rule as
    ``session.open_at``: no forward fill, ever, so a missing quote is "no
    fill" rather than a stale price.
    """
    try:
        v = frame.at[d, col]
    except KeyError:
        return None
    if pd.isna(v):
        return None
    return float(np.asarray(v).reshape(-1)[0])


def _exec_price(v) -> float | None:
    """An executable price: finite and > 0, else None (no fill).

    Mirrors ``strategy._bar_price`` for the research engines: a missing or
    non-positive quote is not an executable price, so the leg simply has no
    fill this session and the decision is re-taken next session.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def _series_carry_bars(series: dict[str, pd.DataFrame]) -> bool:
    """True when the precomputed series carry F's strict session bars.

    ``_load_series`` attaches ``open``/``high``/``low`` to every series, so the
    backtest replays built from it run in CONTRACT mode (decision on data <=
    S, fills at F's strict open, standing-order stops). A caller that hands
    ``_replay`` its own ``_signal_series`` frame (close-only) runs in the
    documented LEGACY mode instead."""
    return any("open" in df.columns for df in series.values())


def _by_time_index(series: dict[str, pd.DataFrame],
                   ) -> tuple[dict[str, dict[str, dict]], bool]:
    """``{ticker: {time: row-dict}}`` for the hot replay loop, plus the
    contract-mode flag.

    One dict per row (not a pandas Series) to avoid per-access overhead; the
    raw scoring components are kept so action/strength can be derived
    per-parameter-set at replay time (the sweep varies the thresholds). When
    the series carry the session bars, each row additionally carries ``open``
    and ``low`` — the contract's fill price and stop-trigger price.
    """
    by_time: dict[str, dict[str, dict]] = {}
    cols = ("time close net bullish bearish trend_up trend_down dist_above "
            "dist_below atr_stop atr14 weekly_trend_up run_5d run_20d "
            "run_60d dist_52w_high").split()
    carries = _series_carry_bars(series)

    for t, df in series.items():
        by_time[t] = {
            str(time): {
                "close": _col_f(close),
                "net": _col_f(net),
                "bullish": _col_f(bullish),
                "bearish": _col_f(bearish),
                "trend_up": _col_b(trend_up),
                "trend_down": _col_b(trend_down),
                "dist_above": _col_f(dist_above),
                "dist_below": _col_f(dist_below),
                "atr_stop": _col_opt(atr_stop),
                "atr14": _col_opt(atr14),
                "weekly_trend_up": True if pd.isna(weekly_trend_up) else _col_b(weekly_trend_up),
                "run_5d": _col_opt(run_5d),
                "run_20d": _col_opt(run_20d),
                "run_60d": _col_opt(run_60d),
                "dist_52w_high": _col_opt(dist_52w_high),
            }
            for time, close, net, bullish, bearish, trend_up, trend_down,
                dist_above, dist_below, atr_stop, atr14, weekly_trend_up,
                run_5d, run_20d, run_60d, dist_52w_high
            in zip(*(df[c] for c in cols), strict=True)
        }
        if carries and "open" in df.columns and "low" in df.columns:
            for time, op, lo in zip(df["time"], df["open"], df["low"],
                                    strict=True):
                row = by_time[t][str(time)]
                row["open"] = _col_opt(op)
                row["low"] = _col_opt(lo)
    return by_time, carries


def _session_bars_for(by_time: dict[str, dict[str, dict]], day: str | None,
                      ) -> dict[str, dict]:
    """F's strict session bars ``{ticker: {"open","low","close"}}`` for `day`.

    Only tickers that actually have a bar on `day` with an executable open
    appear — a missing quote is not an executable price (``app/session.py``
    rule 2), so an absent ticker simply has no fill this session. This is the
    ``session_bars`` payload ``strategy.propose_trades`` consumes in contract
    mode (open fills + standing-order stops)."""
    bars: dict[str, dict] = {}
    if day is None:
        return bars
    for t, idx in by_time.items():
        row = idx.get(day)
        if row is None:
            continue
        open_px = _exec_price(row.get("open"))
        if open_px is None:
            continue
        bars[t] = {"open": open_px,
                   "low": _exec_price(row.get("low")),
                   "close": _exec_price(row.get("close"))}
    return bars


def _standing_stop_fill(bar: dict | None, level: float,
                        fallback_close: float | None) -> float | None:
    """Fill price for a stop standing order (contract rule 4).

    A level known at S's close stands for session F: fill at F's open when
    the open is already through the level, else AT the level when F's low
    breaches it, else no stop fill this session. ``bar`` None means the
    ticker has no executable F bar — no fill (retried next session); the
    caller then falls back to the legacy close comparison by passing
    ``fallback_close`` (None = contract mode, no fill).
    """
    if bar is not None:
        open_px = _exec_price(bar.get("open"))
        if open_px is not None:
            if open_px <= level:
                return open_px
            low = _exec_price(bar.get("low"))
            if low is not None and low <= level:
                return level
        return None
    # Legacy (close-only series): the original same-close comparison.
    return fallback_close if fallback_close is not None and fallback_close <= level else None


def _candidate_tickers() -> list[str]:
    if settings.sim_universe.lower() == "watchlist":
        # Watchlist lives in the DB; fall back to universe file for the tool.
        logger.warning("sim_universe=watchlist; optimize uses universe file instead")
    return universe_tickers(settings.sim_universe)


def _row_action(row: dict, params: ReplayParams) -> str:
    """Derive the BUY/SELL/HOLD action for a row under the given thresholds."""
    return decide_action(
        row["net"], row["trend_up"], row["trend_down"],
        row["dist_above"], row["dist_below"],
        params.buy_threshold, params.sell_threshold,
        row.get("weekly_trend_up", True),
    )


def _row_strength(row: dict, action: str) -> int:
    """Derive the 0-100 strength for a row under the given action."""
    return strength_for(action, row["bullish"], row["bearish"])


def _replay(series: dict[str, pd.DataFrame], params: ReplayParams,
            start: str | None = None, end: str | None = None,
            regime: dict[str, bool] | None = None,
            quality: _QualityLookup | None = None,
            cost_oneway: float | Callable[[str], float] = 0.0) -> ReplayResult:
    """Run the deterministic strategy over the precomputed series.

    ``start``/``end`` are inclusive date strings (YYYY-MM-DD) used to bound
    the replay window (e.g. a walk-forward test window). ``regime`` is an
    optional day→uptrend map; when provided and ``params.regime_filter`` is
    set, new BUYs are blocked on days where the market is not in an uptrend.
    ``quality`` is the optional monthly fundamentals provider; when given
    together with ``params.block_negative_roe``, BUY entries into names with
    known non-positive ROE are blocked (missing data stays neutral).

    ``cost_oneway`` is the research-only one-way cost hook (default ``0.0`` =
    fee-free, exactly today's behaviour, bit-identical): either a flat rate
    or a ``ticker -> rate`` callable. It is applied exactly like the
    ``_daily_core_backtest`` ``cost_oneway`` wrapper around ``_fill_buy``:
    a BUY's budget is capped at ``cash / (1 + rate)`` so the fee is reserved
    INSIDE the spend, and every executed leg (proposal or standing stop)
    then pays ``shares * price * rate`` out of cash on top of the notional,
    so a full-cash buy can never drive cash negative. It is NOT the live
    sim's policy — the paper book stays fee-free by design.

    Session fill contract (app/session.py):

      * CONTRACT mode — the series carry the strict session bars (``open``/
        ``low``, attached by ``_load_series``). Every session F decides on
        data <= S (S = the previous session of the timeline, so the decision
        row is shifted one session back); the deterministic proposals fill at
        F's strict exact-date OPEN; a stop level known at S's close (initial
        stop, ATR ratchet, the replay's own trailing stop) is a standing
        order for F — fill at the open when open <= level, else at the level
        when F's low breaches it, else no stop fill. A ticker without an F
        bar gets no fill and the decision is re-taken next session. The
        portfolio circuit breaker and the regime/BUY gates read the S close
        (marks at F's close only advance the trackers for the NEXT session).
      * LEGACY mode — a caller that hands ``_replay`` its own close-only
        ``_signal_series`` frame keeps the pre-contract behaviour: the
        decision shift still applies, but fills and stops compare against the
        decision-day close (there is no open series to fill at).
    """
    # Build a global timeline of all trading days across tickers.
    all_days: set[str] = set()
    for df in series.values():
        all_days.update(df["time"].tolist())
    days_full = sorted(all_days)
    lo, hi = 0, len(days_full)
    if start:
        lo = next((i for i, d in enumerate(days_full) if d >= start), len(days_full))
    if end:
        hi = next((i for i, d in enumerate(days_full) if d > end), len(days_full))
    if lo >= hi:
        return ReplayResult(params=params)

    # Index each series by time for O(1) lookup. Convert each row to a plain
    # dict (not a pandas Series) to avoid per-access overhead in the hot loop.
    # The raw scoring components are kept so action/strength can be derived
    # per-parameter-set at replay time (the sweep varies the thresholds).
    by_time, carries_bars = _by_time_index(series)

    pf = PaperPortfolio(cash=params.start_cash)
    equity_curve: list[dict] = []
    invested_curve: list[float] = []  # cumulative capital deposited per day
    cash_curve: list[float] = []      # cash held per day (aligned with equity_curve)
    position_count_curve: list[int] = []  # open positions per day
    daily_returns: list[float] = []
    prev_equity: float | None = None
    last_deposit_month: str | None = None
    portfolio_peak: float = 0.0     # highest equity seen (for portfolio stop)
    risk_off: bool = False          # circuit breaker active (no new BUYs)
    cumulative_invested = params.start_cash

    last_known_prices: dict[str, float] = {}  # carry forward for missing-data days
    last_decision_prices: dict[str, float] = {}  # decision closes (<= S), carried
    sp = _replay_params_to_strategy(params)

    def _cost_rate(t: str) -> float:
        return float(cost_oneway(t)) if callable(cost_oneway) else float(cost_oneway)

    def _charge_fee(trade: dict) -> None:
        """Debit the one-way fee of one executed trade from cash (0.0 = no-op)."""
        fee = trade["shares"] * trade["price"] * _cost_rate(trade["ticker"])
        if fee > 0:
            pf.cash -= fee
            if -1e-9 < pf.cash < 0:  # ULP guard mirroring _fill_buy's exact-cash check
                pf.cash = 0.0

    def _sell_with_cost(ticker: str, price: float, reason: str, date: str,
                        shares: float | None = None) -> None:
        n0 = len(pf.trades)
        pf.sell(ticker, price, shares, reason, date=date)
        if len(pf.trades) > n0:
            _charge_fee(pf.trades[-1])

    def _execute_with_cost(p: dict) -> None:
        p = dict(p)
        rate = _cost_rate(p["ticker"])
        if p["side"] == "BUY" and rate > 0:
            p["budget"] = min(float(p.get("budget", 0.0)), pf.cash / (1.0 + rate))
        n0 = len(pf.trades)
        _execute_proposal(pf, p)
        if len(pf.trades) > n0:
            _charge_fee(pf.trades[-1])

    for i in range(lo, hi):
        day = days_full[i]
        # S = the previous session of the (unbounded) timeline — the decision
        # row. F's own bar is never a decision input (contract rule 1).
        s_day = days_full[i - 1] if i > 0 else None

        # Monthly allowance deposit (first trading day of a new month).
        month = day[:7]
        if month != last_deposit_month:
            pf.cash += params.monthly_allowance
            cumulative_invested += params.monthly_allowance
            last_deposit_month = month

        # MARKS: F's close, carried forward when a ticker has no data on this
        # day (e.g. US holidays where European markets are open). Without this,
        # positions are priced at 0 on sparse-data days, producing fake 95%
        # drawdowns.
        prices: dict[str, float] = {}
        for t, idx in by_time.items():
            row = idx.get(day)
            if row is not None:
                p = row["close"]
                prices[t] = p
                last_known_prices[t] = p
            elif t in last_known_prices:
                prices[t] = last_known_prices[t]

        # DECISION closes: the S row (data <= S), carried forward so a held
        # name without an S bar is valued at its last known close instead of
        # zero (only the BUDGETS/valuations read these; fills read F's bar).
        decision_prices: dict[str, float] = {}
        for t, idx in by_time.items():
            row = idx.get(s_day) if s_day is not None else None
            if row is not None:
                p = row["close"]
                decision_prices[t] = p
                last_decision_prices[t] = p
            elif t in last_decision_prices:
                decision_prices[t] = last_decision_prices[t]

        # F's strict session bars — the ONLY place F's data enters a fill
        # (`None` = legacy close-only series).
        session_bars = _session_bars_for(by_time, day) if carries_bars else None
        fill_prices_source = decision_prices if carries_bars else prices

        total_equity = pf.equity(prices)
        if total_equity <= 0:
            equity_curve.append({"time": day, "equity": 0.0})
            invested_curve.append(cumulative_invested)
            cash_curve.append(pf.cash)
            position_count_curve.append(len(pf.positions))
            continue

        # Portfolio-level circuit breaker: if equity has dropped
        # portfolio_stop_pct from its peak, sell everything and go risk-off.
        # Contract mode judges the drawdown on the S close (the last completed
        # session's equity) and stands the cash-out order for F at F's opens;
        # legacy mode keeps the old same-day close comparison.
        eq_for_stop = (prev_equity if (carries_bars and prev_equity is not None)
                       else total_equity)
        if eq_for_stop > portfolio_peak:
            portfolio_peak = eq_for_stop
        if params.portfolio_stop_pct > 0 and portfolio_peak > 0:
            dd = (portfolio_peak - eq_for_stop) / portfolio_peak * 100
            if dd >= params.portfolio_stop_pct:
                if not risk_off:
                    # Sell all positions at F's strict open (no bar = no fill;
                    # risk_off keeps the order standing and retries it).
                    for ticker in list(pf.positions.keys()):
                        price = (_exec_price((session_bars.get(ticker) or {}).get("open"))
                                 if session_bars is not None
                                 else prices.get(ticker, 0))
                        if price:
                            _sell_with_cost(ticker, price,
                                    f"Portfolio stop: equity {eq_for_stop:.0f} "
                                    f"is -{dd:.1f}% from peak {portfolio_peak:.0f}",
                                    day)
                    risk_off = True
            elif risk_off and dd < params.portfolio_stop_pct / 2:
                # Re-engage when drawdown recovers to half the stop level.
                risk_off = False

        # Legacy mode advances the peak-price tracker on today's close BEFORE
        # the stop check (the pre-contract same-close convention). Contract
        # mode advances it at the END of the session, so the level that
        # governs F is the one known at S's close.
        if not carries_bars:
            pf.update_peaks(prices)

        # --- Pre-filter: trailing stop (a standing order in contract mode) ---
        # Sell if price has dropped trailing_stop_pct from its peak since entry.
        # This is a backtest-only SELL that strategy.propose_trades doesn't know
        # about (it doesn't track per-position peaks), so we apply it before
        # calling propose_trades and let the core logic handle the rest.
        if params.trailing_stop_pct > 0:
            for ticker in list(pf.positions.keys()):
                peak = pf.peak_price.get(ticker, 0)
                if peak <= 0:
                    continue
                stop_level = peak * (1 - params.trailing_stop_pct / 100)
                bar = session_bars.get(ticker) if session_bars is not None else None
                legacy_close = prices.get(ticker) if session_bars is None else None
                fill = _standing_stop_fill(bar, stop_level, legacy_close)
                if fill is not None:
                    _sell_with_cost(ticker, fill,
                            f"Trailing stop: {fill:.2f} <= {stop_level:.2f} "
                            f"(peak {peak:.2f}, -{params.trailing_stop_pct}%)",
                            day)

        # --- Core: delegate SELL to strategy.propose_trades, execute, then BUY ---
        # propose_trades works on a snapshot; SELLs must execute first so the
        # BUY phase sees freed slots + cash (matching the old inline behaviour
        # where SELLs mutated pf before BUYs ran). In contract mode the shared
        # engine resolves F's open from ``session_bars`` and applies the
        # standing-order stop rule; ``prices`` stay decision-side (S closes).
        signals = _signals_for_day_from_bytime(by_time, day, params, s_day=s_day)

        # SELL phase: get proposals, execute them immediately.
        proposals = propose_trades(
            positions=_pf_to_positions(pf),
            cash=pf.cash,
            prices=fill_prices_source,
            signals=signals,
            params=sp,
            sector_of=_sector_of if params.max_sector_pct > 0 else None,
            quality_of=quality.of(day) if quality else None,
            session_bars=session_bars,
        )
        for p in proposals:
            if p["side"] == "SELL":
                _execute_with_cost({**p, "date": day})

        # BUY phase: re-derive post-SELL state, re-propose for BUYs only.
        # Regime filter: block new BUYs when the broad market is below its
        # 200-day SMA, or when the circuit breaker is active. The regime state
        # is a decision input, so it is read at S (data <= S).
        market_ok = not risk_off
        if params.regime_filter and regime is not None:
            market_ok = market_ok and regime.get(s_day, True)  # type: ignore[arg-type]

        if market_ok:
            proposals2 = propose_trades(
                positions=_pf_to_positions(pf),
                cash=pf.cash,
                prices=fill_prices_source,
                signals=signals,
                params=sp,
                sector_of=_sector_of if params.max_sector_pct > 0 else None,
                quality_of=quality.of(day) if quality else None,
                session_bars=session_bars,
            )
            for p in proposals2:
                if p["side"] == "BUY":
                    _execute_with_cost({**p, "date": day})

        # Record equity.
        total_equity = pf.equity(prices)
        equity_curve.append({"time": day, "equity": round(total_equity, 2)})
        invested_curve.append(cumulative_invested)
        cash_curve.append(pf.cash)
        position_count_curve.append(len(pf.positions))
        if prev_equity is not None and prev_equity > 0:
            daily_returns.append(total_equity / prev_equity - 1)
        prev_equity = total_equity
        if carries_bars:
            # Contract: F's close only advances the trackers for the NEXT
            # session (the peak/equity levels that govern F are S's).
            pf.update_peaks(prices)
            portfolio_peak = max(portfolio_peak, total_equity)

    final_equity = equity_curve[-1]["equity"] if equity_curve else 0.0
    walk_days = days_full[lo:hi]
    total_invested = params.start_cash + params.monthly_allowance * len(set(d[:7] for d in walk_days))
    total_return_pct = (final_equity / total_invested - 1) * 100 if total_invested > 0 else 0.0

    return ReplayResult(
        params=params,
        equity_curve=equity_curve,
        trades=pf.trades,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        sharpe=_sharpe(daily_returns),
        max_drawdown_pct=_max_drawdown([e["equity"] for e in equity_curve], invested_curve),
        n_trades=len(pf.trades),
        cash_curve=cash_curve,
        position_count_curve=position_count_curve,
    )


# ---------------------------------------------------------------------------
# Hybrid replay: deterministic skeleton + LLM per-day review (mirrors live sim)
# ---------------------------------------------------------------------------

def _signals_for_day(
    series: dict[str, pd.DataFrame],
    by_time: dict[str, dict[str, dict]],
    day: str,
    params: ReplayParams,
    s_day: str | None = None,
) -> dict[str, dict]:
    """Build the ``{ticker: compute()-shaped result}`` dict for one session.

    Mirrors what ``sim._gather_signals`` produces for the live cycle: each
    ticker maps to ``{"action", "reason", "snapshot", "strength"}``. The
    snapshot is built via ``snapshot_from_row`` (the single source of truth
    shared with ``analysis.compute``) so the LLM sees exactly the fields the
    live engine shows it. Tickers with no row for the DECISION session are
    skipped (the live sim skips them too when they did not trade on S).

    ``s_day`` is the decision session S (the previous completed session): the
    indicators are read from that row, so F's bar is never a decision input
    (session fill contract rule 1). None keeps the legacy callers' behaviour
    (``day`` itself is the decision row).
    """
    dec_day = s_day if s_day is not None else day
    out: dict[str, dict] = {}
    for t, idx in by_time.items():
        row = idx.get(dec_day)
        if row is None:
            continue
        df = series[t]
        x = df.iloc[df.index[df["time"] == dec_day].tolist()[0]]
        net = float(x.net)
        bullish = float(x.bullish)
        bearish = float(x.bearish)
        trend_up = bool(x.trend_up)
        trend_down = bool(x.trend_down)
        weekly_trend_up = bool(x.weekly_trend_up) if not pd.isna(x.weekly_trend_up) else True
        vol_surge = bool(x.vol_surge)
        dist_above = float(x.dist_above)
        dist_below = float(x.dist_below)
        atr_stop = None if pd.isna(x.atr_stop) else float(x.atr_stop)
        action = decide_action(net, trend_up, trend_down, dist_above, dist_below,
                                params.buy_threshold, params.sell_threshold,
                                weekly_trend_up)
        strength = strength_for(action, bullish, bearish)
        snap = snapshot_from_row(x, net, bullish, bearish, strength,
                                  weekly_trend_up, vol_surge, atr_stop)
        reason = f"{action} (strength {strength})"
        out[t] = {"action": action, "reason": reason, "snapshot": snap,
                  "strength": strength}
    return out


def _pf_to_positions(pf: PaperPortfolio) -> list[dict]:
    """Convert PaperPortfolio.positions to the plain-data shape strategy.py expects.

    Includes ``thesis`` and ``buy_date`` (set by PaperPortfolio.buy) so the
    LLM context builder can show why each position was bought and for how long.
    """
    return [
        {"ticker": t, "shares": s, "avg_cost": pf.avg_cost.get(t, 0),
         "stop_price": pf.stop_price.get(t),
         "thesis": pf.thesis.get(t, ""), "buy_date": pf.buy_date.get(t, "")}
        for t, s in pf.positions.items()
    ]


def _replay_params_to_strategy(p: ReplayParams) -> StrategyParams:
    """Convert ReplayParams to the shared StrategyParams."""
    return StrategyParams(
        buy_threshold=p.buy_threshold,
        sell_threshold=p.sell_threshold,
        min_cash_pct=p.min_cash_pct,
        max_position_pct=p.max_position_pct,
        max_positions=p.max_positions,
        stop_type=p.stop_type,
        stop_pct=p.stop_pct,
        stop_atr_mult=p.stop_atr_mult,
        use_atr_stop=p.use_atr_stop,
        max_run_5d=p.max_run_5d,
        min_run_5d=p.min_run_5d,
        max_dist_above=p.max_dist_above,
        block_negative_roe=p.block_negative_roe,
        relaxed_hold_strength=p.relaxed_hold_strength,
        relaxed_hold_limit=p.relaxed_hold_limit,
        max_sector_pct=p.max_sector_pct,
        risk_pct=p.risk_pct,
    )


class _QualityLookup:
    """Monthly-granularity quality provider for the replay engines.

    Wraps ``{ticker: {"roe", "p_fcf"}}`` maps cached per calendar month: the
    fundamentals move quarterly, so a fresh quality snapshot per month (keyed
    by the month of the replay day) matches the monthly fetch cadence and
    avoids re-running the TTM math for every trading day. ``of(day)`` returns
    the ``quality_of(ticker)`` callable ``strategy.propose_trades`` expects
    for that day, or None when no quality data exists at all (feature off /
    empty fact store).
    """

    def __init__(self, fund: dict[str, dict[str, list[dict]]],
                 closes: dict[str, dict[str, float]] | None = None):
        self._fund = fund
        # {ticker: {date: close}} — USD-converted replay closes for P/FCF.
        self._closes = closes or {}
        self._cache: dict[str, dict[str, dict[str, float | None]]] = {}

    def month_map(self, day: str) -> dict[str, dict[str, float | None]]:
        """Quality map {ticker: {"roe","p_fcf"}} for `day`'s month (cached)."""
        month = day[:7]
        cached = self._cache.get(month)
        if cached is not None:
            return cached
        from .monthly import quality_snapshot

        asof: pd.Timestamp = pd.Timestamp(pd.to_datetime(day).date())  # type: ignore[assignment]
        out: dict[str, dict[str, float | None]] = {}
        px_of = self._closes.get(month, {})
        for t in self._fund:
            q = quality_snapshot(self._fund, t, asof, px_of.get(t))
            if q is not None:
                out[t] = q
        self._cache[month] = out
        return out

    def of(self, day: str) -> Callable[[str], dict[str, float | None] | None] | None:
        """``quality_of(ticker)`` callable for `day`, or None (feature off)."""
        mmap = self.month_map(day)
        if not mmap:
            return None

        def _lookup(ticker: str) -> dict[str, float | None] | None:
            return mmap.get(ticker.upper())

        return _lookup


async def _load_quality_lookup(tickers: list[str]) -> _QualityLookup | None:
    """Build a _QualityLookup from the fundamentals store + candle closes.

    Loads the EDGAR-first fact store once and the monthly frames (for
    USD-converted period-end closes feeding P/FCF). Returns None when the
    fact store is empty (feature effectively off).
    """
    from . import monthly as monthly_mod
    from .fundamentals import load_fundamentals

    fund = await load_fundamentals(tickers)
    if not fund:
        return None
    close, _ = await monthly_mod.load_frames(tickers, None)
    # {month: {ticker: first USD close of the month}} — one price per ticker
    # per month is enough for P/FCF (the quality snapshot only needs a scale).
    # First (not last) close keeps the monthly cache point-in-time.
    return _QualityLookup(fund, _month_price_map(close))


def _deterministic_propose_replay(
    pf: PaperPortfolio,
    by_time: dict[str, dict[str, dict]],
    prices: dict[str, float],
    day: str,
    params: ReplayParams,
    quality_of: Callable[[str], dict[str, float | None] | None] | None = None,
    s_day: str | None = None,
    session_bars: dict[str, dict] | None = None,
) -> list[dict]:
    """Propose deterministic trades for one session WITHOUT mutating ``pf``.

    Thin wrapper over ``strategy.propose_trades`` — converts PaperPortfolio
    to plain data, builds signals from the precomputed by_time rows, calls
    the shared propose logic, and stamps each proposal with the session + any
    entry_stop the shared logic computed.

    ``prices`` are DECISION prices (S closes) — budgets and valuations are
    decision-side. ``s_day`` is the decision session (F's bar never informs a
    decision, contract rule 1), and ``session_bars`` carries F's strict
    ``{open, low, close}`` per ticker so the shared engine fills at F's open
    and applies the standing-order stop rule. Both default to the legacy
    single-session behaviour (``s_day = day``, ``session_bars = None``).
    """
    signals = _signals_for_day_from_bytime(by_time, day, params, s_day=s_day)
    sp = _replay_params_to_strategy(params)
    proposals = propose_trades(
        positions=_pf_to_positions(pf),
        cash=pf.cash,
        prices=prices,
        signals=signals,
        params=sp,
        quality_of=quality_of,
        session_bars=session_bars,
    )
    # Stamp the day on each proposal (strategy.propose_trades is day-agnostic).
    for p in proposals:
        p["date"] = day
    return proposals


def _signals_for_day_from_bytime(
    by_time: dict[str, dict[str, dict]],
    day: str,
    params: ReplayParams,
    s_day: str | None = None,
) -> dict[str, dict]:
    """Build the ``{ticker: signal}`` dict for one session from by_time rows.

    Converts the raw scoring components in by_time (net, bullish, bearish,
    trend_up, etc.) into the same {action, strength, reason, snapshot} shape
    that ``analysis.compute()`` and ``_signals_for_day`` produce, so the
    shared ``strategy.propose_trades`` can consume them uniformly.

    ``s_day`` is the DECISION session S — the indicators are read from that
    row, one session before F (contract rule 1: data <= S only). None keeps
    the legacy behaviour where ``day`` itself is the decision row (direct
    callers/tests that hand in a single-session view).
    """
    dec_day = s_day if s_day is not None else day
    signals: dict[str, dict] = {}
    for t, idx in by_time.items():
        row = idx.get(dec_day)
        if row is None:
            continue
        action = _row_action(row, params)
        strength = _row_strength(row, action)
        # Canonical snapshot conversions (analysis.snapshot_from_row): the
        # entry gate and the tightened HOLD fallback read net_score /
        # trend_up / trend_down / weekly_trend_up — the adapter omitted
        # them, so the fallback failed closed in EVERY replay while the
        # live engine kept using it (review D2).
        net = row.get("net")
        net_score: int | None = None
        try:
            net_f = float(net)
        except (TypeError, ValueError):
            net_f = float("nan")
        # math.isfinite, not just isna: int(+inf)/int(-inf) raise
        # OverflowError — a non-finite score fails closed to None instead.
        if math.isfinite(net_f):
            net_score = int(net_f)
        signals[t] = {
            "action": action,
            "strength": strength,
            "reason": f"{action} (strength {strength})",
            "snapshot": {
                "atr_stop": row.get("atr_stop"),
                "run_5d": row.get("run_5d"),
                "atr14": row.get("atr14"),
                "dist_above": row.get("dist_above"),
                # by_time rows carry weekly_trend_up already normalized
                # (NaN -> True, matching the canonical conversion).
                "net_score": net_score,
                "trend_up": bool(row.get("trend_up", False)),
                "trend_down": bool(row.get("trend_down", False)),
                "weekly_trend_up": bool(row.get("weekly_trend_up", True)),
            },
        }
    return signals


def _current_week_from(day: str) -> str:
    """ISO calendar week key (YYYY-Www) for a replay day.

    Mirror of the live sim's :func:`sim._current_week`, but anchored to the
    replayed date instead of ``datetime.now`` so historical backtests get the
    same weekly-review cadence as production.
    """
    from datetime import datetime
    return datetime.strptime(day, "%Y-%m-%d").strftime("%G-W%V")


def _execute_proposal(pf: PaperPortfolio, p: dict) -> dict | None:
    """Execute a single proposal against ``pf``. Returns the trade dict or None."""
    if p["side"] == "SELL":
        before = pf.positions.get(p["ticker"], 0)
        pf.sell(p["ticker"], p["price"], p.get("shares"), p["reason"],
                date=p.get("date", ""))
        if pf.positions.get(p["ticker"], 0) < before or p["ticker"] not in pf.positions:
            return {"ticker": p["ticker"], "side": "SELL", "price": p["price"],
                    "shares": before - pf.positions.get(p["ticker"], 0),
                    "reason": p["reason"], "date": p.get("date", "")}
        return None
    # BUY
    before = pf.positions.get(p["ticker"], 0)
    pf.buy(p["ticker"], p["price"], p["budget"], p["reason"],
           stop=p.get("entry_stop"), date=p.get("date", ""))
    if pf.positions.get(p["ticker"], 0) > before:
        return {"ticker": p["ticker"], "side": "BUY", "price": p["price"],
                "shares": pf.positions[p["ticker"]] - before,
                "reason": p["reason"], "date": p.get("date", "")}
    return None


async def _hybrid_replay(
    series: dict[str, pd.DataFrame],
    params: ReplayParams,
    start: str | None = None,
    end: str | None = None,
    news: dict[str, list[dict]] | None = None,
    pure_llm: bool = False,
    review_interval: int = 1,
    veto_only: bool = False,
    no_llm_sells: bool = True,
    minimal_prompt: bool = False,
    mode_aware: bool = False,
    llm_max_positions: int = 0,
    marker_gated: bool = False,
    failure_marker: bool = False,
    failure_stop_outs: int = 2,
    failure_drawdown: float = 7.0,
    quality: _QualityLookup | None = None,
    fundamentals_context: bool = False,
) -> ReplayResult:
    """Replay the hybrid strategy (deterministic + LLM review).

    For each trading day:
      1. Run the deterministic SELL + BUY phases against the paper portfolio
         (mirrors ``sim._deterministic_decide``).
      2. Build the LLM context with the post-deterministic portfolio state,
         all candidate signals, and the deterministic trades (mirrors
         ``sim._build_llm_context``).
      3. Call the LLM once with ``_LLM_SYSTEM_PROMPT`` and execute its BUY/SELL
         decisions through ``PaperPortfolio`` (mirrors ``sim._llm_decide``).

    SELLs the LLM adds (not proposed by deterministic) are honoured, as in the
    live hybrid sim. The result is comparable to a pure-deterministic
    ``_replay`` over the same window.

    ``review_interval`` controls how often the LLM is consulted: 1 (default)
    reviews the first trading day of every calendar week; larger values skip
    that many weeks. Between reviews the deterministic proposals execute
    as-is with no LLM call — the engine runs daily, the LLM reviews weekly
    (the "weekly portfolio review" strategy: deterministic risk management +
    LLM judgment on selection). This mirrors the live sim's calendar-week
    anchoring exactly.

    ``news`` is optional and currently unused (the live sim gathers news via
    SearXNG; a historical replay has no dated news archive). When None the
    news section is omitted from the context, same as a live cycle with
    SEARXNG_URL unset.

    Session fill contract: the deterministic skeleton runs in CONTRACT mode
    when the series carry session bars (``_load_series`` does): signals come
    from data <= S (the previous session) and the skeleton's fills execute at
    F's strict open. The LLM legs use the same decision-side prices/budgets
    and, when F's bar carries an executable open, fill there too; a series
    without opens (a caller-supplied close-only ``_signal_series`` frame)
    keeps the legacy close-fill behaviour end to end.
    """
    from .sim import _LLM_SYSTEM_PROMPT, _LLM_MINIMAL_SYSTEM_PROMPT, _LLM_MODE_AWARE_MINIMAL_PROMPT, _PURE_LLM_SYSTEM_PROMPT, _build_llm_context, _parse_llm_decisions, _week_diff

    # Build the global timeline and per-ticker day index (same as _replay).
    all_days: set[str] = set()
    for df in series.values():
        all_days.update(df["time"].tolist())
    days_full = sorted(all_days)
    lo, hi = 0, len(days_full)
    if start:
        lo = next((i for i, d in enumerate(days_full) if d >= start), len(days_full))
    if end:
        hi = next((i for i, d in enumerate(days_full) if d > end), len(days_full))
    if lo >= hi:
        return ReplayResult(params=params)
    days = days_full[lo:hi]

    by_time, carries_bars = _by_time_index(series)

    pf = PaperPortfolio(cash=params.start_cash)
    equity_curve: list[dict] = []
    invested_curve: list[float] = []
    daily_returns: list[float] = []
    prev_equity: float | None = None
    last_deposit_month: str | None = None
    cumulative_invested = params.start_cash
    last_known_prices: dict[str, float] = {}
    last_decision_prices: dict[str, float] = {}
    last_review_week: str | None = None
    peak_equity: float = params.start_cash

    for day_idx, day in enumerate(days, 1):
            # Monthly allowance deposit
            month = day[:7]
            if month != last_deposit_month:
                pf.cash += params.monthly_allowance
                cumulative_invested += params.monthly_allowance
                last_deposit_month = month

            # Carry-forward prices for missing-data days
            prices: dict[str, float] = {}
            for t, idx in by_time.items():
                row = idx.get(day)
                if row is not None:
                    p = row["close"]
                    prices[t] = p
                    last_known_prices[t] = p
                elif t in last_known_prices:
                    prices[t] = last_known_prices[t]

            # Decision session S + decision closes (data <= S) and F's strict
            # session bars — the same split `_replay` uses, so the hybrid
            # skeleton shares the contract semantics.
            g_i = lo + day_idx - 1
            s_day = days_full[g_i - 1] if g_i > 0 else None
            decision_prices: dict[str, float] = {}
            for t, idx in by_time.items():
                row = idx.get(s_day) if s_day is not None else None
                if row is not None:
                    p = row["close"]
                    decision_prices[t] = p
                    last_decision_prices[t] = p
                elif t in last_decision_prices:
                    decision_prices[t] = last_decision_prices[t]
            session_bars = (_session_bars_for(by_time, day)
                            if carries_bars else None)
            decision_source = decision_prices if carries_bars else prices

            logger.info("hybrid day %d/%d %s — equity %.2f, cash %.2f, %d positions",
                        day_idx, len(days), day, pf.equity(prices),
                        pf.cash, len(pf.positions))

            if pf.equity(prices) <= 0:
                equity_curve.append({"time": day, "equity": 0.0})
                invested_curve.append(cumulative_invested)
                continue

            # Track per-position peaks (mirrors _replay) so the ATR trailing
            # stop and any future trailing logic see the same state in both
            # engines.
            pf.update_peaks(prices)

            # --- 1. Deterministic PROPOSE phase (no mutation) ---
            signals = _signals_for_day(series, by_time, day, params, s_day=s_day)
            if pure_llm:
                # Pure-LLM mode (main-branch parity): the engine makes NO
                # proposals and NO risk-floor sells — the LLM picks the names,
                # and the engine's hard sizing limits (min-cash floor,
                # max-position-%, max-positions cap) size every BUY. This is
                # the design that measured stable and near-deterministic
                # (mean -0.82% vs det, stddev 0.79% over 3 runs) — free
                # sizing and engine-driven exits added variance + losses.
                proposals = []
            else:
                proposals = _deterministic_propose_replay(
                    pf, by_time, decision_source, day, params,
                    quality_of=quality.of(day) if quality else None,
                    s_day=s_day, session_bars=session_bars)
                if proposals:
                    logger.info("  det proposed: %d — %s",
                                len(proposals),
                                ", ".join(f"{p['side']} {p['ticker']}" for p in proposals))
                else:
                    logger.info("  det proposed: no trades")

            # --- 2 + 3. LLM review → reconcile → execute ---
            total_equity = pf.equity(prices)
            # Pure-LLM mode: the LLM is the SOLE decision-maker — it must be
            # consulted every trading day (the engine makes no proposals, so
            # there is no weekly-review concept). The weekly cadence below
            # only applies to hybrid mode.
            week = _current_week_from(day)
            if pure_llm:
                is_review_day = True
            else:
                # Default cadence: fire the review on the first trading day of
                # each calendar week (mirrors the live sim's weekly review) —
                # the LLM is consulted at most once per ISO week, anchored to
                # the week, not a day counter. review_interval=1 still reviews
                # every week; larger values skip intermediate weeks.
                week: str = _current_week_from(day)
                if marker_gated:
                    # Marker-gated mode: the deterministic engine's proposals ARE
                    # the marker — an LLM review is only worthwhile when there is
                    # something to veto/approve. The marker can fire on ANY day
                    # (not just the scheduled cadence); review_interval then acts
                    # only as a cooldown so proposals on consecutive days don't
                    # force a call every day.
                    is_review_day = bool(proposals) and (
                        last_review_week is None
                        or _week_diff(week, last_review_week) >= review_interval
                    )
                elif failure_marker:
                    # Failure-marker mode: the deterministic engine usually works
                    # well — only consult the LLM when the engine shows signs of
                    # failure. Two failure signatures from the data:
                    #   1. Stop-out cascade: N+ stop-loss SELLs in the last 5
                    #      trading days (the engine bought into a falling tape).
                    #   2. Equity drawdown: equity > X% below its running peak
                    #      (the engine is bleeding without stopping).
                    # The marker fires on ANY day with NO cooldown — a fresh
                    # cascade triggers a call the same day (validated: no-cooldown
                    # beat the weekly-cooldown variant on the 60d window).
                    stop_outs_5d = sum(
                        1 for t in pf.trades[-10:]
                        if t["side"] == "SELL" and _is_stop_out(t["reason"])
                        and t["date"] >= days[max(0, day_idx - 6)]
                    )
                    peak_equity = max(peak_equity, total_equity)
                    drawdown = (total_equity / peak_equity - 1) * 100 if peak_equity > 0 else 0.0
                    failure = stop_outs_5d >= failure_stop_outs or drawdown <= -failure_drawdown
                    is_review_day = failure
                    if failure:
                        logger.info("  failure marker: stop_outs_5d=%d drawdown=%.1f%%",
                                    stop_outs_5d, drawdown)
                else:
                    is_review_day = (
                        last_review_week is None
                        or _week_diff(week, last_review_week) >= review_interval
                    )
            if total_equity > 0 and (settings.llm_backends or settings.ollama_model) and is_review_day:
                allowance_total = cumulative_invested
                valuation = valuate_portfolio(_pf_to_positions(pf), pf.cash, prices, allowance_total)
                signals = _signals_for_day(series, by_time, day, params, s_day=s_day)
                context = _build_llm_context(valuation, proposals, signals, news,
                                            pure_llm=pure_llm,
                                            trade_history=list(reversed(pf.trades[-15:]))[:12],
                                            minimal=minimal_prompt or mode_aware,
                                            max_positions=llm_max_positions,
                                            quality=quality.month_map(day) if (fundamentals_context and quality) else None)
                system_prompt = (
                    _LLM_MODE_AWARE_MINIMAL_PROMPT if mode_aware
                    else _LLM_MINIMAL_SYSTEM_PROMPT if minimal_prompt
                    else _PURE_LLM_SYSTEM_PROMPT if pure_llm
                    else _LLM_SYSTEM_PROMPT)  # shared prompt: main-branch parity
                logger.info("  llm phase: calling LLM (%d signals, %d proposals)...",
                            len(signals), len(proposals))
                try:
                    out = await llm_mod.chat([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ])
                    content = out["text"]
                except Exception as e:
                    logger.warning("LLM call failed on %s: %s — executing all proposals", day, e)
                    content = ""
                decisions = _parse_llm_decisions(content) or []

                if not decisions:
                    # LLM unavailable/parse-failed: execute all proposals (fallback)
                    for p in proposals:
                        _execute_proposal(pf, p)
                    if proposals:
                        logger.info("  (fallback) executed all %d proposals", len(proposals))
                else:
                    # Reconcile proposals with LLM decisions (veto vs approve)
                    approved, vetoed, proposal_tickers = reconcile_proposals(proposals, decisions)
                    for p in vetoed:
                        logger.info("  llm VETOED %s %s — %s", p["side"], p["ticker"], p.get("llm_reason", ""))

                    # Execute approved proposals (SELLs first, already ordered)
                    for p in approved:
                        _execute_proposal(pf, p)

                    # Execute LLM additions (decisions for tickers NOT in proposals)
                    # Main-branch parity: hard sizing on every LLM BUY — min-cash
                    # floor, max-position-% ceiling, max-positions cap. This is
                    # the design that measured stable (stddev 0.79% over 3 runs);
                    # free sizing (guarded=False) made results ~3x more volatile.
                    # In veto-only mode the LLM's job is confined to blocking
                    # bad deterministic proposals; it never sizes new entries.
                    if not veto_only:
                        plan = await plan_llm_buys(
                            decisions, pf.cash, pf.equity(prices),
                            _replay_params_to_strategy(params),
                            guarded=True,
                            price_of=lambda t, _prices=prices: _aval(_prices.get(t)),
                            value_of=lambda t, _prices=prices: _val_of(pf, t, _prices),
                            exclude=proposal_tickers,
                        )
                        held_tickers = set(pf.positions.keys())
                        for d in decisions:
                            tu = d["ticker"].upper()
                            if tu in proposal_tickers:
                                continue
                            action = d["action"]
                            reason = d.get("reason", f"LLM {action}")
                            price = prices.get(d["ticker"])
                            # Contract mode: the LLM's legs fill at F's strict
                            # open when the ticker has an F bar (a missing bar
                            # = no fill, exactly like the deterministic legs);
                            # the budget was still sized decision-side.
                            if session_bars is not None:
                                price = _exec_price(
                                    (session_bars.get(d["ticker"]) or {}).get("open"))
                            if price is None or price <= 0 or action == "HOLD":
                                continue
                            if action == "BUY":
                                # Max-positions cap: block NEW positions when
                                # at the cap, but still allow topping up
                                # tickers already held. ``llm_max_positions``
                                # raises the cap for LLM BUYs only — the
                                # deterministic engine still caps at
                                # ``params.max_positions``; the extra slots
                                # belong to the LLM's picks.
                                cap = (llm_max_positions
                                       if llm_max_positions > 0
                                       else params.max_positions)
                                if (cap > 0
                                        and len(held_tickers) >= cap
                                        and tu not in held_tickers):
                                    logger.info("  llm BUY %s skipped (max-positions cap %d)",
                                                tu, cap)
                                    continue
                                budget = plan.get(tu)
                                if budget is None:
                                    continue
                                entry_stop = (price * (1 - params.stop_pct / 100)
                                              if params.stop_type == "percent" else None)
                                pf.buy(d["ticker"], price, budget, f"LLM: {reason}",
                                       stop=entry_stop, date=day)
                                held_tickers.add(tu)
                            elif action == "SELL" and not (no_llm_sells and not pure_llm):
                                target_shares = llm_sell_shares(d, price)
                                pf.sell(d["ticker"], price, target_shares,
                                        f"LLM: {reason}", date=day)

                    summary = ", ".join(f"{d['ticker']}={d['action']}" for d in decisions)
                    logger.info("  llm returned %d decisions: %s", len(decisions), summary)
                    if vetoed:
                        logger.info("  vetoes: %d — %s",
                                    len(vetoed),
                                    ", ".join(f"{p['side']} {p['ticker']}" for p in vetoed))
                # Mark the week as reviewed — no more LLM calls until the
                # calendar week changes (same anchoring as the live sim).
                # Pure-LLM mode consults the LLM every day, so no week marker.
                if not pure_llm:
                    last_review_week = week
            else:
                # No LLM configured, or not a review day: execute all
                # proposals as-is (deterministic). On non-review days the
                # engine runs alone — the LLM only reviews on review days.
                #
                # Two-phase propose/execute, mirroring _replay exactly: SELLs
                # execute first, then BUYs are RE-PROPOSED against the
                # post-SELL state so a freed slot can be refilled the same
                # day. Without this, a stop-out SELL and a same-day re-entry
                # BUY (both proposed in one batch) would be blocked by the
                # max-positions cap — a structural divergence from the det
                # benchmark that masqueraded as "LLM alpha" (+5.48% on the
                # 90d bull window with ZERO LLM calls). On review days we
                # keep the single-phase batch because the LLM reviews the
                # pre-SELL proposal list (that's what the live sim shows it).
                for p in proposals:
                    if p["side"] == "SELL":
                        _execute_proposal(pf, p)
                proposals2 = _deterministic_propose_replay(
                    pf, by_time, decision_source, day, params,
                    quality_of=quality.of(day) if quality else None,
                    s_day=s_day, session_bars=session_bars)
                for p in proposals2:
                    if p["side"] == "BUY":
                        _execute_proposal(pf, p)

            # Record equity
            total_equity = pf.equity(prices)
            equity_curve.append({"time": day, "equity": round(total_equity, 2)})
            invested_curve.append(cumulative_invested)
            if prev_equity is not None and prev_equity > 0:
                daily_returns.append(total_equity / prev_equity - 1)
            prev_equity = total_equity

    final_equity = equity_curve[-1]["equity"] if equity_curve else 0.0
    total_invested = params.start_cash + params.monthly_allowance * len(set(d[:7] for d in days))
    total_return_pct = (final_equity / total_invested - 1) * 100 if total_invested > 0 else 0.0

    return ReplayResult(
            params=params,
            equity_curve=equity_curve,
            trades=pf.trades,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            sharpe=_sharpe(daily_returns),
            max_drawdown_pct=_max_drawdown([e["equity"] for e in equity_curve], invested_curve),
            n_trades=len(pf.trades),
        )


# ---------------------------------------------------------------------------
# Parameter sweep + walk-forward
# ---------------------------------------------------------------------------

def _param_grid(regime_filter: bool = False) -> list[ReplayParams]:
    """Grid over the tunable thresholds and risk parameters.

    Sweeps the highest-impact levers: buy/sell thresholds, trailing stop,
    max position, initial stop type, risk sizing, and sector cap. The
    risk management params (stop_type, risk_pct, max_sector_pct) are swept
    as a small set of proven configurations to keep the grid manageable.
    """
    # Risk management configurations (proven configs from A/B testing).
    # Each tuple: (stop_type, stop_pct, stop_atr_mult, risk_pct, max_sector_pct,
    #              portfolio_stop_pct, max_positions)
    risk_configs = [
        # No risk management (baseline — lets optimizer compare)
        ("none", 0, 0, 0, 0, 0, 0),
        # 15% stop + 1% risk + 20% sector cap + max 10 positions (diversified)
        ("percent", 15, 0, 1.0, 20, 0, 10),
        # 15% stop + 1% risk + 20% sector cap + max 10 + 20% portfolio stop
        ("percent", 15, 0, 1.0, 20, 20, 10),
        # 15% stop + 1% risk + 15% sector cap + max 20 positions (less concentrated)
        ("percent", 15, 0, 1.0, 15, 0, 20),
        # 2x ATR stop + 1% risk + 20% sector cap + max 10 positions
        ("atr", 0, 2.0, 1.0, 20, 0, 10),
        # 2x ATR stop + 1% risk + 20% sector cap + max 10 + 20% portfolio stop
        ("atr", 0, 2.0, 1.0, 20, 20, 10),
    ]
    grid = []
    for buy in (40, 50):
        for sell in (-40, -30):
            for max_pos in (10,):
                    for stop_type, stop_pct, stop_atr, risk_pct, sector_pct, pf_stop, max_n in risk_configs:
                        for max_run_5d in (0, 12, 15, 20):
                            grid.append(ReplayParams(
                                buy_threshold=buy, sell_threshold=sell,
                                relaxed_hold_strength=40,
                                max_position_pct=max_pos,
                                use_atr_stop=False,
                                trailing_stop_pct=0,
                                regime_filter=regime_filter,
                                stop_type=stop_type,
                                stop_pct=stop_pct,
                                stop_atr_mult=stop_atr,
                                risk_pct=risk_pct,
                                max_sector_pct=sector_pct,
                                portfolio_stop_pct=pf_stop,
                                max_positions=max_n,
                                max_run_5d=max_run_5d,
                            ))
    return grid


def _split_windows(days: list[str], train_days: int, test_days: int):
    """Yield (train_start, train_end, test_start, test_end) sliding windows."""
    i = 0
    while i + train_days + test_days <= len(days):
        train = days[i:i + train_days]
        test = days[i + train_days:i + train_days + test_days]
        yield train[0], train[-1], test[0], test[-1]
        i += test_days  # non-overlapping test windows


def _score(result: ReplayResult) -> float:
    """Composite score used to pick the 'best' params on a train window.

    Heavily penalizes drawdown (2x) so the optimizer prefers risk-controlled
    configs over high-return/high-drawdown ones. A config returning +50% with
    95% drawdown scores -140, while +30% return with 20% drawdown scores -10.
    """
    return result.total_return_pct - 2.0 * result.max_drawdown_pct


def _run_sweep(series: dict[str, pd.DataFrame], start: str, end: str,
               regime: dict[str, bool] | None = None,
               regime_filter: bool = False) -> list[tuple[ReplayParams, ReplayResult]]:
    grid = _param_grid(regime_filter=regime_filter)
    results = []
    for idx, params in enumerate(grid):
        if idx and idx % 20 == 0:
            logger.info("  sweep %s..%s: %d/%d params", start, end, idx, len(grid))
        res = _replay(series, params, start=start, end=end, regime=regime)
        results.append((params, res))
    results.sort(key=lambda x: _score(x[1]), reverse=True)
    return results


def _walk_forward(series: dict[str, pd.DataFrame], days: list[str],
                  train_days: int, test_days: int,
                  regime: dict[str, bool] | None = None,
                  regime_filter: bool = False) -> list[dict]:
    """Run walk-forward: fit best params on each train window, score on test."""
    import time

    splits = list(_split_windows(days, train_days, test_days))
    total = len(splits)
    logger.info("Walk-forward: %d windows (train %dd / test %dd)", total, train_days, test_days)
    windows = []
    for w_idx, (train_s, train_e, test_s, test_e) in enumerate(splits, 1):
        t0 = time.time()
        logger.info("Window %d/%d: train %s..%s → test %s..%s",
                    w_idx, total, train_s, train_e, test_s, test_e)
        sweep = _run_sweep(series, train_s, train_e, regime=regime, regime_filter=regime_filter)
        best_params, best_train = sweep[0]
        test_res = _replay(series, best_params, start=test_s, end=test_e, regime=regime)
        elapsed = time.time() - t0
        logger.info("Window %d/%d done in %.0fs: buy=%d sell=%d trail=%.0f maxpos=%d "
                    "stop=%s risk=%.1f%% sector=%.0f%% pf_stop=%.0f%% maxn=%d run5d=%.0f | "
                    "train %+.1f%% → test %+.1f%% (sharpe %.2f, dd %.1f%%, %d trades)",
                    w_idx, total, elapsed,
                    best_params.buy_threshold, best_params.sell_threshold,
                    best_params.trailing_stop_pct, best_params.max_position_pct,
                    best_params.stop_type, best_params.risk_pct, best_params.max_sector_pct,
                    best_params.portfolio_stop_pct, best_params.max_positions,
                    best_params.max_run_5d,
                    best_train.total_return_pct, test_res.total_return_pct,
                    test_res.sharpe, test_res.max_drawdown_pct, test_res.n_trades)
        windows.append({
            "train": f"{train_s}..{train_e}",
            "test": f"{test_s}..{test_e}",
            "best_params": {
                "buy_threshold": best_params.buy_threshold,
                "sell_threshold": best_params.sell_threshold,
                "relaxed_hold_strength": best_params.relaxed_hold_strength,
                "max_position_pct": best_params.max_position_pct,
                "use_atr_stop": best_params.use_atr_stop,
                "stop_type": best_params.stop_type,
                "stop_pct": best_params.stop_pct,
                "risk_pct": best_params.risk_pct,
                "max_sector_pct": best_params.max_sector_pct,
                "portfolio_stop_pct": best_params.portfolio_stop_pct,
                "max_positions": best_params.max_positions,
                "max_run_5d": best_params.max_run_5d,
            },
            "train_score": round(_score(sweep[0][1]), 2),
            "train_return_pct": round(sweep[0][1].total_return_pct, 2),
            "test_return_pct": round(test_res.total_return_pct, 2),
            "test_sharpe": round(test_res.sharpe, 2),
            "test_max_dd_pct": round(test_res.max_drawdown_pct, 2),
            "test_n_trades": test_res.n_trades,
        })
    return windows


# ---------------------------------------------------------------------------
# LLM benchmark: would the LLM have turned the deterministic model's bad calls?
# ---------------------------------------------------------------------------

# Forward window (trading days) used to judge whether a deterministic trade was
# "bad" in hindsight. A BUY is bad when the price fell over this window; a SELL
# / stop-out is bad when the price recovered over it.
_BENCH_FORWARD_DAYS = 20
# How many worst-mistake cases to probe, and how many control cases (deterministic
# was right) to include so we can see whether the LLM blindly overrides good calls.
_BENCH_N_WORST = 8
_BENCH_N_CONTROL = 2


@dataclass
class TradeCase:
    """A single deterministic decision probed against the LLM.

    ``outcome_pct`` is the forward return after the decision (negative is bad
    for a BUY, positive is bad for a SELL). ``badness`` is a single number that
    is large when the decision was clearly wrong (signed so the worst mistakes
    sort first): for a BUY it's ``-forward_return``; for a SELL it's
    ``+forward_return``. A control case is simply a trade whose ``badness`` is
    close to zero or the opposite sign.

    ``position_before`` is the position held in this ticker just before the
    trade (shares + avg cost), reconstructed from the trade stream, so the LLM
    probe can be shown truthful portfolio context (e.g. a SELL really was
    closing a position, not acting on a bare signal).
    """
    date: str
    ticker: str
    det_side: str              # "BUY" | "SELL" (SELL covers signal + stop exits)
    det_reason: str
    entry_price: float
    forward_close: float | None  # close N trading days later (None if window truncated)
    outcome_pct: float          # forward return over the window
    badness: float              # signed: higher = worse deterministic call
    position_before: dict[str, float] | None = None  # {"shares":..,"avg_cost":..} or None
    portfolio_state: dict[str, Any] | None = None     # full ledger just before the trade
    is_control: bool = False


def _reconstruct_portfolio_states(
    trades: list[dict],
    start_cash: float = settings.sim_start_cash,
    monthly_allowance: float = settings.sim_monthly_allowance,
) -> list[dict[str, Any]]:
    """Reconstruct the full portfolio state just before each chronological trade.

    Walks the trade stream maintaining cash + a {ticker: {shares, avg_cost}}
    ledger, depositing the monthly allowance on the first trade of each month,
    so the LLM probe sees the same portfolio context the live engine had.
    Returns one state dict per trade (aligned 1:1 with ``trades``):
        {"cash": float, "positions": [{"ticker","shares","avg_cost","value"}],
         "total_equity": float, "positions_value": float}
    Equity is valued at each trade's price (close on the decision day) — an
    approximation for other tickers, but accurate for the decision ticker.
    """
    cash = start_cash
    ledger: dict[str, dict[str, float]] = {}
    last_month: str | None = None
    states: list[dict[str, Any]] = []
    for tr in trades:
        date = tr.get("date") or ""
        month = date[:7]
        # Deposit the allowance for every month between the last seen month
        # and this trade's month. The replay deposits on the first trading
        # day of *every* month, so with sparse trades we must catch up the
        # skipped months — otherwise cash drifts negative and the LLM probe
        # sees an unrealistic portfolio.
        if last_month is not None and month > last_month:
            y0, m0 = int(last_month[:4]), int(last_month[5:7])
            y1, m1 = int(month[:4]), int(month[5:7])
            months_between = (y1 - y0) * 12 + (m1 - m0)
            cash += monthly_allowance * months_between
        elif last_month is None:
            # First trade: deposit one allowance for its month.
            cash += monthly_allowance
        last_month = month

        # Value the decision ticker at the trade price; other held tickers are
        # approximated at their last known trade price (good enough for context).
        prices: dict[str, float] = {}
        for t, pos in ledger.items():
            prices[t] = pos.get("_last_price", pos["avg_cost"])
        prices[tr["ticker"]] = tr["price"]

        positions = []
        positions_value = 0.0
        for t, pos in ledger.items():
            price = prices.get(t, pos["avg_cost"])
            value = pos["shares"] * price
            positions_value += value
            positions.append({"ticker": t, "shares": pos["shares"],
                              "avg_cost": pos["avg_cost"], "value": value})
        states.append({
            "cash": cash, "positions": positions,
            "positions_value": positions_value,
            "total_equity": cash + positions_value,
        })

        # Apply the trade to the ledger.
        side = tr["side"]
        shares = tr["shares"]
        price = tr["price"]
        ticker = tr["ticker"]
        if side == "BUY":
            cost = shares * price
            if ticker in ledger:
                old = ledger[ticker]
                total = old["shares"] + shares
                ledger[ticker] = {
                    "shares": total,
                    "avg_cost": (old["shares"] * old["avg_cost"] + cost) / total,
                    "_last_price": price,
                }
            else:
                ledger[ticker] = {"shares": shares, "avg_cost": price, "_last_price": price}
            cash -= cost
        else:  # SELL
            if ticker in ledger:
                cash += shares * price
                remaining = ledger[ticker]["shares"] - shares
                if remaining <= 0.0001:
                    ledger.pop(ticker, None)
                else:
                    ledger[ticker]["shares"] = remaining
                    ledger[ticker]["_last_price"] = price
    return states


def _trade_outcomes(
    series: dict[str, pd.DataFrame],
    trades: list[dict],
    forward_days: int = _BENCH_FORWARD_DAYS,
) -> list[TradeCase]:
    """Score each replay trade by its forward price action.

    For a BUY, ``outcome_pct`` is the N-day forward return (bad if negative).
    For a SELL, ``outcome_pct`` is also the N-day forward return (bad if
    positive — we sold right before a rally). ``badness`` is signed so the
    worst mistakes sort first: BUY badness = -outcome, SELL badness = +outcome.

    Trades carry a ``date`` field (stamped by the replay loop) used to locate
    the decision day in the ticker's series. ``position_before`` is filled from
    ``_reconstruct_positions`` so the LLM probe gets truthful context.
    """
    closes: dict[str, dict[str, float]] = {}
    for t, df in series.items():
        closes[t] = {str(time): _col_f(c)
                     for time, c in zip(df["time"], df["close"], strict=False)}

    portfolio_states = _reconstruct_portfolio_states(trades)
    cases: list[TradeCase] = []
    for tr, state in zip(trades, portfolio_states, strict=False):
        ticker = tr["ticker"]
        side = tr["side"]
        price = tr["price"]
        date = tr.get("date") or ""
        cmap = closes.get(ticker)
        if cmap is None or date not in cmap:
            continue

        # Forward N trading days within THIS ticker's calendar.
        dates = sorted(cmap.keys())
        i = dates.index(date)
        j = min(i + forward_days, len(dates) - 1)
        fwd = cmap[dates[j]]
        outcome = (fwd / price - 1) * 100 if price > 0 else 0.0
        badness = -outcome if side == "BUY" else outcome
        # Position in this ticker just before the trade.
        pos_before = next(
            ({"shares": p["shares"], "avg_cost": p["avg_cost"]}
             for p in state["positions"] if p["ticker"] == ticker),
            None,
        )
        cases.append(TradeCase(
            date=date, ticker=ticker, det_side=side,
            det_reason=tr.get("reason", ""),
            entry_price=price, forward_close=fwd,
            outcome_pct=outcome, badness=badness,
            position_before=pos_before,
            portfolio_state=state,
        ))
    return cases


def _select_cases(
    cases: list[TradeCase],
    n_worst: int = _BENCH_N_WORST,
    n_control: int = _BENCH_N_CONTROL,
) -> list[TradeCase]:
    """Pick the worst deterministic mistakes plus a few control cases.

    Worst cases = highest ``badness``. Controls = deterministic calls that were
    clearly right (lowest ``badness``, i.e. the model's call aligned with what
    happened next). Both groups are de-duplicated by ticker so the probe covers
    breadth rather than hammering one name. Controls are flagged so the report
    can distinguish them.
    """
    by_ticker: dict[str, list[TradeCase]] = {}
    for c in cases:
        by_ticker.setdefault(c.ticker, []).append(c)

    worst: list[TradeCase] = []
    for group in by_ticker.values():
        group.sort(key=lambda c: c.badness, reverse=True)
        worst.append(group[0])  # each ticker's worst single decision
    worst.sort(key=lambda c: c.badness, reverse=True)

    controls: list[TradeCase] = []
    for group in by_ticker.values():
        group.sort(key=lambda c: c.badness)  # lowest badness = decision was right
        controls.append(group[0])
    controls.sort(key=lambda c: c.badness)

    picked: list[TradeCase] = []
    seen: set[str] = set()
    for c in worst:
        if len(picked) >= n_worst:
            break
        if c.ticker in seen:
            continue
        seen.add(c.ticker)
        picked.append(c)
    n_before_ctl = len(picked)
    for c in controls:
        if len(picked) - n_before_ctl >= n_control:
            break
        if c.ticker in seen:
            continue
        seen.add(c.ticker)
        c.is_control = True
        picked.append(c)
    return picked


def _snapshot_for_day(df: pd.DataFrame, date: str) -> dict[str, Any]:
    """Reconstruct the analysis.compute() snapshot for a given trading day.

    Uses ``snapshot_from_row`` (the single source of truth for the snapshot
    shape shared with ``analysis.compute``) so the benchmark always shows the
    LLM the same fields the live hybrid engine does — no drift.
    """
    i = df.index[df["time"] == date].tolist()
    if not i:
        raise KeyError(f"{date} not in series")
    x = df.iloc[i[0]]

    net = float(x.net)
    bullish = float(x.bullish)
    bearish = float(x.bearish)
    trend_up = bool(x.trend_up)
    trend_down = bool(x.trend_down)
    weekly_trend_up = bool(x.weekly_trend_up) if not pd.isna(x.weekly_trend_up) else True
    vol_surge = bool(x.vol_surge)
    dist_above = float(x.dist_above)
    dist_below = float(x.dist_below)
    atr_stop = float(x.atr_stop)

    action = decide_action(
        net, trend_up, trend_down, dist_above, dist_below,
        ReplayParams().buy_threshold, ReplayParams().sell_threshold,
        weekly_trend_up,
    )
    strength = strength_for(action, bullish, bearish)
    snap = snapshot_from_row(x, net, bullish, bearish, strength,
                             weekly_trend_up, vol_surge, atr_stop)
    return {"action": action, "reason": "", "snapshot": snap, "strength": strength}


def _live_sim_params() -> ReplayParams:
    """ReplayParams mirroring the live sim's risk configuration.

    The benchmark must replay with the same risk rules the live
    ``_deterministic_decide`` applies (initial stop, ATR stop, max-positions
    cap, position-size and cash-floor limits), otherwise the reconstructed
    portfolio state won't match what the engine would actually have held — and
    the LLM probe would be judged against context the engine never produced
    (e.g. a 29-position portfolio judged against a 10-position cap).
    """
    return ReplayParams(
        max_positions=settings.sim_max_positions,
        max_position_pct=settings.sim_max_position_pct,
        min_cash_pct=settings.sim_min_cash_pct,
        # The live sim applies a frozen initial stop (sim_stop_pct) plus the
        # ATR trailing stop on every cycle; mirror that here so the trade list
        # and portfolio state reflect production behaviour.
        stop_type="percent",
        stop_pct=settings.sim_stop_pct,
        use_atr_stop=True,
        # The live sim also blocks BUYs after a 5-day run-up beyond
        # sim_max_run_5d (prevents chasing short-term spikes). Mirror it so
        # the replay's buy candidates match what the engine would actually
        # have considered.
        max_run_5d=settings.sim_max_run_5d,
        # Entry guards mirror the live sim's settings — they are OPT-IN
        # (defaults 0 = disabled) because the A/B showed a return/dd
        # trade-off, not a free lunch. See config.py + strategy.py.
        min_run_5d=settings.sim_min_run_5d,
        max_dist_above=settings.sim_max_dist_above,
    )


def _is_stop_out(reason: str) -> bool:
    """True if a deterministic SELL was an automatic stop, not a signal-driven decision.

    The replay stamps these reasons (see the SELL phase in ``_replay``):
      - "Initial stop: ..."   (frozen % stop)
      - "ATR stop hit: ..."    (trailing-volatility stop)
      - "Trailing stop: ..."  (% from peak)
      - "Portfolio stop: ..." (circuit breaker)
    Signal-driven SELLs say "SELL signal (strength ...)".
    """
    r = reason.lower()
    return any(r.startswith(p) for p in
               ("initial stop", "atr stop", "trailing stop", "portfolio stop"))


def _build_llm_probe_context(case: TradeCase, snapshot: dict[str, Any],
                             params: ReplayParams | None = None) -> str:
    """Build the user-message context for a single decision probe.

    Mirrors the compact signal summary from sim._build_llm_context but for one
    ticker / one day, with the reconstructed portfolio state (cash, equity,
    open positions) and the deterministic decision stated plainly so the LLM
    can confirm or override it. The portfolio state is reconstructed from the
    trade stream so it reflects what the engine actually held at the time.

    Stop-out SELLs are framed as "the stop is about to fire — hold through it
    or let it sell?" rather than as a completed decision, so the LLM can
    exercise judgement on whether the shakeout is justified (e.g. RSI oversold
    + weekly trend still up) instead of deferring to the auto-sell rule.
    """
    snap = snapshot["snapshot"]
    det = case.det_side
    p = params or _live_sim_params()
    state = case.portfolio_state or {
        "cash": 0.0, "positions": [], "positions_value": 0.0, "total_equity": 0.0,
    }
    equity = state["total_equity"]
    cash = state["cash"]
    max_pos_str = "unlimited" if p.max_positions <= 0 else str(p.max_positions)
    stop_out = det == "SELL" and _is_stop_out(case.det_reason)
    lines = [
        "## Decision Probe (single ticker, one historical day)",
    ]
    if stop_out:
        lines.append(
            f"The deterministic engine's stop-loss is about to fire on {case.ticker}: "
            f"{case.det_reason}"
        )
        lines.append(
            "The stop is an automatic rule — your job is to decide whether to "
            "OVERRIDE it (HOLD) or let it execute (SELL). Consider whether the "
            "indicators suggest the position is about to recover (e.g. RSI "
            "oversold and turning up, weekly trend still up, MACD histogram "
            "rising) or whether the trend really is broken."
        )
    else:
        lines.append(f"Deterministic engine proposed: {det} — {case.det_reason}")
    lines += [
        "",
        "## Portfolio context (reconstructed at the decision time)",
        f"Cash: {cash:.2f}",
        f"Positions value: {state['positions_value']:.2f}",
        f"Total equity: {equity:.2f}",
        f"Open positions: {len(state['positions'])}/{max_pos_str}",
        f"Min cash ({p.min_cash_pct:g}%): {equity * p.min_cash_pct / 100:.2f} | "
        f"Max position ({p.max_position_pct:g}%): {equity * p.max_position_pct / 100:.2f}",
        f"Stop loss: {p.stop_pct:g}% (frozen at entry; ATR stop also applies)",
        "",
    ]
    if state["positions"]:
        lines.append("Open positions:")
        for p in state["positions"]:
            lines.append(
                f"  - {p['ticker']}: {p['shares']:.4f} shares @ avg {p['avg_cost']:.2f} "
                f"| value {p['value']:.2f}"
            )
    else:
        lines.append("Open positions: none")
    # Be explicit about the decision ticker's existing position.
    if case.position_before:
        pb = case.position_before
        lines.append(
            f"Existing position in {case.ticker}: {pb['shares']:.4f} shares @ avg {pb['avg_cost']:.2f}"
        )
    else:
        lines.append(f"Existing position in {case.ticker}: none")
    lines += [
        "",
        "## Signals",
        f"{'ticker':<10} {'action':<6} {'strength':>8} "
        f"{'close':>10} {'rsi':>6} {'rsiΔ3':>6} {'adx':>5} {'wk':>3} "
        f"{'macd':>10} {'mhΔ3':>7}",
    ]
    rsi_d = snap.get("rsi_3d_change")
    mh_d = snap.get("macd_hist_3d_change")
    rsi_d_s = f"{rsi_d:+.1f}" if rsi_d is not None else "  -  "
    mh_d_s = f"{mh_d:+.2f}" if mh_d is not None else "  -  "
    lines.append(
        f"{case.ticker:<10} {snapshot['action']:<6} {snapshot['strength']:>8} "
        f"{snap.get('close', 0):>10.2f} {snap.get('rsi', 0):>6.1f} "
        f"{rsi_d_s:>6} "
        f"{snap.get('adx', 0):>5.0f} "
        f"{'up' if snap.get('weekly_trend_up') else 'dn':>3} "
        f"{snap.get('macd', 0):>10.3f} {mh_d_s:>7}"
    )
    lines += [
        "",
        "## Deterministic Candidate Trades",
    ]
    if stop_out:
        lines.append(
            f"  - {case.ticker} SELL (stop about to fire) — {case.det_reason}"
        )
    else:
        lines.append(
            f"  - {case.ticker} {det} (strength {snapshot['strength']}) — {case.det_reason}"
        )
    lines += [
        "",
        "## Your Decision",
        "Return ONLY a JSON array of objects: "
        '{"ticker": "...", "action": "BUY|SELL|HOLD", "reason": "..."}. '
        "No markdown, no prose.",
    ]
    return "\n".join(lines)


async def _llm_probe(case: TradeCase, snapshot: dict[str, Any],
                     params: ReplayParams | None = None) -> dict[str, Any]:
    """Send one decision probe to the configured LLM backend.

    Reuses the live hybrid sim's system prompt and JSON-array format so the
    benchmark reflects production behaviour. Returns the raw content, the
    parsed decision (if any), and a status flag. ``params`` is forwarded to
    the context builder so the risk limits shown match the replay that
    produced the trade.

    For stop-out cases, a short addendum is appended to the system prompt
    allowing the LLM to override the auto-sell (which the live prompt's rule 4
    otherwise forbids), since the whole point of probing a stop shakeout is to
    ask whether holding through it would have been better.
    """
    # Imported here to avoid a circular import at module load (sim imports
    # analysis; optimize imports analysis). sim only needs to be present at
    # call time.
    from .sim import _LLM_SYSTEM_PROMPT, _parse_llm_decisions

    system_prompt = _LLM_SYSTEM_PROMPT
    if _is_stop_out(case.det_reason):
        system_prompt += (
            "\n\nBENCHMARK OVERRIDE: rule 4 is relaxed for this probe. The "
            "stop is about to fire but has NOT executed yet — you may answer "
            "HOLD to override it and keep the position, or SELL to let it "
            "execute. Base your choice on the indicators: hold through the "
            "stop when the trend is merely pausing (RSI oversold and turning "
            "up, weekly trend still up, MACD histogram rising), let it sell "
            "when the trend really is broken."
        )

    context = _build_llm_probe_context(case, snapshot, params=params)
    try:
        out = await llm_mod.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ])
    except Exception as e:
        return {"raw": "", "decision": None, "status": f"error: {type(e).__name__}: {e}"}

    content = out["text"]
    decisions = _parse_llm_decisions(content)
    picked = None
    if decisions:
        # Take the decision matching this ticker (case-insensitive); fall back
        # to the first if the model returned only one.
        for d in decisions:
            if d["ticker"].upper() == case.ticker.upper():
                picked = d
                break
        if picked is None and len(decisions) == 1:
            picked = decisions[0]
    return {"raw": content, "decision": picked, "status": "ok"}


def _llm_turned(det_side: str, llm_action: str | None) -> str:
    """Classify the LLM's verdict vs the deterministic decision."""
    if llm_action is None:
        return "no-decision"
    if llm_action == det_side:
        return "agreed"
    if llm_action == "HOLD":
        return "turned-to-HOLD"
    # BUY vs SELL flip
    return f"turned-to-{llm_action}"


async def _run_llm_benchmark(
    series: dict[str, pd.DataFrame],
    start: str | None,
    end: str | None,
    n_worst: int = _BENCH_N_WORST,
    n_control: int = _BENCH_N_CONTROL,
    forward_days: int = _BENCH_FORWARD_DAYS,
    skip_llm: bool = False,
) -> None:
    """Run the full LLM-vs-deterministic benchmark and print a report.

    ``skip_llm`` runs only the deterministic replay + case selection and prints
    the candidates without calling the LLM — useful for previewing which cases
    would be probed before spending tokens.

    The replay uses ``_live_sim_params`` (the live sim's risk configuration:
    max-positions cap, initial + ATR stops, position/cash limits) so the
    reconstructed portfolio state matches what the engine would actually have
    held, and the LLM is judged against truthful context.
    """
    params = _live_sim_params()
    res = _replay(series, params, start=start, end=end)
    print(f"\n=== Deterministic backtest ({start or 'start'}..{end or 'end'}) ===")
    print(f"  Trades: {res.n_trades} | Return: {res.total_return_pct:+.2f}% | "
          f"Max DD: {res.max_drawdown_pct:.2f}% | Sharpe: {res.sharpe:.2f}")
    print(f"  Risk config: max_positions={params.max_positions}, "
          f"stop={params.stop_type} {params.stop_pct:g}%, "
          f"atr_stop={params.use_atr_stop}, "
          f"max_pos_pct={params.max_position_pct:g}%, "
          f"min_cash_pct={params.min_cash_pct:g}%")

    if not res.trades:
        print("No trades to benchmark.")
        return

    cases = _trade_outcomes(series, res.trades, forward_days=forward_days)
    if not cases:
        print("Could not score any trades (no forward price data).")
        return
    picked = _select_cases(cases, n_worst=n_worst, n_control=n_control)
    print(f"\n=== Selected {len(picked)} cases "
          f"({sum(1 for c in picked if not c.is_control)} worst + "
          f"{sum(1 for c in picked if c.is_control)} control) "
          f"| forward window = {forward_days}d ===")
    for c in picked:
        tag = "CONTROL" if c.is_control else "WORST"
        print(f"  [{tag}] {c.date} {c.ticker:<8} {c.det_side:<5} @ {c.entry_price:>9.2f} "
              f"→ {c.forward_close:>9.2f} ({c.outcome_pct:+6.2f}% over {forward_days}d) "
              f"| badness {c.badness:+6.2f} | {c.det_reason}")

    if skip_llm:
        print("\n--skip-llm set; not calling the LLM. Re-run without it to probe.")
        return

    if not (settings.llm_backends or settings.ollama_model):
        print("\nNo LLM backend configured (set LLM_BACKENDS or OLLAMA_MODEL); cannot probe the LLM.")
        return

    active = await llm_mod.current_backend()
    print(f"\n=== Probing LLM ({active.get('name', '?')} · {active.get('model', '?')}) ... ===")
    rows: list[dict[str, Any]] = []
    for idx, c in enumerate(picked, 1):
        df = series.get(c.ticker)
        if df is None:
            print(f"  [{idx}/{len(picked)}] {c.ticker} {c.date}: no series, skipped")
            continue
        try:
            snapshot = _snapshot_for_day(df, c.date)
        except KeyError as e:
            print(f"  [{idx}/{len(picked)}] {c.ticker} {c.date}: {e}")
            continue
        print(f"  [{idx}/{len(picked)}] {c.ticker} {c.date} "
              f"(det={c.det_side}, fwd {c.outcome_pct:+.2f}%) ...", end=" ", flush=True)
        probe = await _llm_probe(c, snapshot, params=params)
        llm_action = (probe["decision"] or {}).get("action") if probe["decision"] else None
        llm_reason = (probe["decision"] or {}).get("reason", "") if probe["decision"] else ""
        verdict = _llm_turned(c.det_side, llm_action)
        print(verdict)
        rows.append({
            "case": c, "snapshot": snapshot, "probe": probe,
            "llm_action": llm_action, "llm_reason": llm_reason, "verdict": verdict,
        })

    # --- Final report ----------------------------------------------------
    print(f"\n=== LLM benchmark report ({len(rows)}/{len(picked)} probed) ===")
    header = (f"{'date':<12} {'ticker':<9} {'tag':<8} {'det':<5} {'fwd%':>7} "
              f"{'llm':<7} {'verdict':<18} reason")
    print(header)
    print("-" * len(header))
    turned = 0
    saved = 0      # LLM turned a call the deterministic model got wrong (badness > 0)
    harmed = 0     # LLM turned a call the deterministic model got right (badness < 0)
    for r in rows:
        c = r["case"]
        tag = "CONTROL" if c.is_control else "WORST"
        llm = r["llm_action"] or "-"
        reason = (r["llm_reason"][:60] + "…") if len(r["llm_reason"]) > 61 else r["llm_reason"]
        print(f"{c.date:<12} {c.ticker:<9} {tag:<8} {c.det_side:<5} "
              f"{c.outcome_pct:>+7.2f}% {llm:<7} {r['verdict']:<18} {reason}")
        if r["verdict"] not in ("agreed", "no-decision"):
            turned += 1
            # A turn helps when the deterministic call was wrong (badness > 0:
            # the forward outcome went against the decision). It harms when the
            # deterministic call was right (badness < 0: the outcome confirmed
            # it). This is based on the actual outcome, not the tag, so it's
            # correct even when --n-worst extends into cases the model got right.
            if c.badness > 0:
                saved += 1
            else:
                harmed += 1
    print(f"\n  LLM turned {turned}/{len(rows)} decisions "
          f"(saved {saved}, harmed {harmed}).")
    # Token-budget note: each probe is a single small user message + the fixed
    # system prompt, so the total cost is ~ len(rows) round-trips.

    # Dump raw LLM reasoning to a JSON file for later inspection.
    out_path = "/tmp/llm_benchmark_reasoning.json"
    try:
        with open(out_path, "w") as f:
            json.dump([{
                "date": r["case"].date, "ticker": r["case"].ticker,
                "det_side": r["case"].det_side, "outcome_pct": r["case"].outcome_pct,
                "is_control": r["case"].is_control,
                "llm_action": r["llm_action"], "verdict": r["verdict"],
                "raw": r["probe"]["raw"],
            } for r in rows], f, indent=2)
        print(f"  Raw LLM reasoning written to {out_path}")
    except OSError as e:
        print(f"  (could not write reasoning file: {e})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(res: ReplayResult, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"  Final equity:      ${res.final_equity:,.2f}")
    print(f"  Total return:      {res.total_return_pct:+.2f}%")
    print(f"  Sharpe (annual):   {res.sharpe:.2f}")
    print(f"  Max drawdown:      {res.max_drawdown_pct:.2f}%")
    print(f"  Trades:            {res.n_trades}")


def _print_cash_summary(res: ReplayResult, params: ReplayParams) -> None:
    """Print cash-utilization stats from the replay's cash/position curves.

    Shows how much equity sat idle as cash, how often the max-positions cap
    blocked deployment, and a monthly breakdown so the drag is visible.
    """
    if not res.cash_curve or not res.equity_curve:
        print("  (no cash curve data)")
        return
    n_days = len(res.equity_curve)
    cash_pcts = [
        (c / e["equity"]) * 100 if e["equity"] > 0 else 0.0
        for c, e in zip(res.cash_curve, res.equity_curve, strict=False)
    ]
    avg_cash_pct = sum(cash_pcts) / n_days
    at_cap = sum(1 for n in res.position_count_curve
                 if params.max_positions > 0 and n >= params.max_positions)
    pct_at_cap = (at_cap / n_days) * 100 if n_days else 0
    idle_days = sum(1 for p in cash_pcts if p > 30)

    # Longest stretch at cap
    longest_cap = 0; current_cap = 0
    for n in res.position_count_curve:
        if params.max_positions > 0 and n >= params.max_positions:
            current_cap += 1
            longest_cap = max(longest_cap, current_cap)
        else:
            current_cap = 0

    print(f"\n=== Cash utilization ({n_days} days) ===")
    print(f"  Avg cash as % of equity:    {avg_cash_pct:.1f}%")
    print(f"  Days at max-positions cap:  {at_cap}/{n_days} ({pct_at_cap:.0f}%)")
    print(f"  Longest stretch at cap:     {longest_cap} days")
    print(f"  Days with >30% cash:        {idle_days} ({idle_days/n_days*100:.0f}%)")

    # Monthly breakdown
    from collections import defaultdict
    monthly = defaultdict(list)
    for e, c, n in zip(res.equity_curve, res.cash_curve, res.position_count_curve, strict=False):
        monthly[e["time"][:7]].append((c, n, e["equity"]))
    print(f"\n  {'month':<8} {'avg_cash_%':>10} {'avg_pos':>8} {'end_cash':>11}")
    for m in sorted(monthly):
        vals = monthly[m]
        avg_c = sum(v[0] / v[2] * 100 for v in vals if v[2] > 0) / max(1, len(vals))
        avg_p = sum(v[1] for v in vals) / len(vals)
        end_c = vals[-1][0]
        print(f"  {m:<8} {avg_c:>9.1f}% {avg_p:>8.1f} {end_c:>11.2f}")


# ---------------------------------------------------------------------------
# Multi-window LLM vs deterministic benchmark (reliable scoreboard)
# ---------------------------------------------------------------------------

@dataclass
class _WindowResult:
    """One window's deterministic vs LLM comparison."""
    start: str
    end: str
    det: ReplayResult
    llm: ReplayResult

    @property
    def return_delta(self) -> float:
        return self.llm.total_return_pct - self.det.total_return_pct

    @property
    def drawdown_delta(self) -> float:
        return self.llm.max_drawdown_pct - self.det.max_drawdown_pct

    @property
    def sharpe_delta(self) -> float:
        return self.llm.sharpe - self.det.sharpe

    @property
    def trades_delta(self) -> int:
        return self.llm.n_trades - self.det.n_trades


async def _llm_walkforward(
    series: dict[str, pd.DataFrame],
    params: ReplayParams,
    all_days: list[str],
    n_windows: int,
    days_per_window: int,
    pure_llm: bool,
    start: str | None,
    end: str | None,
    review_interval: int = 1,
    veto_only: bool = False,
    no_llm_sells: bool = True,
    news: dict[str, list[dict]] | None = None,
    minimal_prompt: bool = False,
    mode_aware: bool = False,
    llm_max_positions: int = 0,
    marker_gated: bool = False,
    failure_marker: bool = False,
    failure_stop_outs: int = 2,
    failure_drawdown: float = 7.0,
    quality: _QualityLookup | None = None,
    fundamentals_context: bool = False,
) -> list[_WindowResult]:
    """Run N non-overlapping windows, each deterministic vs LLM.

    Windows are cut from the tail of the bounded ``all_days`` list so the most
    recent history is always covered. If the bounded range is shorter than
    ``n_windows * days_per_window``, windows are allowed to overlap from the
    front (the tail windows stay non-overlapping) — this keeps the most recent
    windows intact at the cost of older ones sharing data, which is the
    conservative direction for a backtest.
    """
    days = list(all_days)
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    if len(days) < days_per_window:
        raise ValueError(
            f"need at least {days_per_window} trading days in range, have {len(days)}"
        )

    # Cut n_windows windows from the tail.
    windows: list[tuple[str, str]] = []
    total_needed = n_windows * days_per_window
    if len(days) >= total_needed:
        # Non-overlapping: take the last total_needed days, split into n_windows.
        base = days[-total_needed:]
        for i in range(n_windows):
            chunk = base[i * days_per_window:(i + 1) * days_per_window]
            windows.append((chunk[0], chunk[-1]))
    else:
        # Not enough for full non-overlap: every window is still full-size
        # (days_per_window days), but their start indices are evenly spaced
        # across the available range so they overlap. This keeps every window
        # the same length (comparable metrics) at the cost of shared data.
        last_start = len(days) - days_per_window  # start of the tail window
        if last_start < 0:
            raise ValueError(
                f"need at least {days_per_window} trading days in range, have {len(days)}"
            )
        step = last_start / max(1, n_windows - 1) if n_windows > 1 else 0
        starts = [round(i * step) for i in range(n_windows)]
        for s in starts:
            chunk = days[s:s + days_per_window]
            windows.append((chunk[0], chunk[-1]))

    mode_label = "pure-LLM" if pure_llm else "hybrid"
    active = await llm_mod.current_backend()
    print(f"\n=== {mode_label} walk-forward ({len(windows)} windows × {days_per_window} days) ===")
    print(f"  Backend: {active.get('name', '?')} · {active.get('model', '?')}")
    print(f"  Risk config: max_positions={params.max_positions}, "
          f"stop={params.stop_type} {params.stop_pct:g}%, "
          f"atr_stop={params.use_atr_stop}, max_run_5d={params.max_run_5d:g}%")

    results: list[_WindowResult] = []
    for i, (w_start, w_end) in enumerate(windows, 1):
        print(f"\n--- Window {i}/{len(windows)}: {w_start}..{w_end} ---")
        det = _replay(series, params, start=w_start, end=w_end, quality=quality)
        logger.info("window %d/%d %s..%s deterministic: return %.2f%%, dd %.2f%%, %d trades",
                    i, len(windows), w_start, w_end,
                    det.total_return_pct, det.max_drawdown_pct, det.n_trades)
        llm = await _hybrid_replay(series, params, start=w_start, end=w_end,
                                   pure_llm=pure_llm,
                                   review_interval=review_interval,
                                   veto_only=veto_only,
                                   no_llm_sells=no_llm_sells,
                                   minimal_prompt=minimal_prompt,
                                   mode_aware=mode_aware,
                                   llm_max_positions=llm_max_positions,
                                   marker_gated=marker_gated,
                                   failure_marker=failure_marker,
                                   failure_stop_outs=failure_stop_outs,
                                   failure_drawdown=failure_drawdown,
                                   quality=quality,
                                   fundamentals_context=fundamentals_context)
        logger.info("window %d/%d %s..%s %s: return %.2f%%, dd %.2f%%, %d trades",
                    i, len(windows), w_start, w_end, mode_label,
                    llm.total_return_pct, llm.max_drawdown_pct, llm.n_trades)
        wr = _WindowResult(start=w_start, end=w_end, det=det, llm=llm)
        results.append(wr)
        _print_window_row(wr)

    return results


def _print_window_row(wr: _WindowResult) -> None:
    """Print one window's compact comparison row."""
    print(f"  {wr.start}..{wr.end}")
    print(f"    {'':>14} {'det':>10} {'llm':>10} {'delta':>9}")
    print(f"    {'Return':>14} {wr.det.total_return_pct:>+9.2f}% {wr.llm.total_return_pct:>+9.2f}% "
          f"{wr.return_delta:>+8.2f}%")
    print(f"    {'Max drawdown':>14} {wr.det.max_drawdown_pct:>9.2f}% {wr.llm.max_drawdown_pct:>9.2f}% "
          f"{wr.drawdown_delta:>+8.2f}%")
    print(f"    {'Sharpe':>14} {wr.det.sharpe:>10.2f} {wr.llm.sharpe:>10.2f} "
          f"{wr.sharpe_delta:>+9.2f}")
    print(f"    {'Trades':>14} {wr.det.n_trades:>10} {wr.llm.n_trades:>10} "
          f"{wr.trades_delta:>+9}")


def _print_walkforward_summary(results: list[_WindowResult]) -> None:
    """Print the aggregate mean + worst-window delta table."""
    if not results:
        return
    n = len(results)
    mean_ret = sum(r.return_delta for r in results) / n
    mean_dd = sum(r.drawdown_delta for r in results) / n
    mean_sharpe = sum(r.sharpe_delta for r in results) / n
    mean_trades = sum(r.trades_delta for r in results) / n

    worst_ret = min(r.return_delta for r in results)
    worst_dd = max(r.drawdown_delta for r in results)  # worst = largest extra drawdown
    worst_sharpe = min(r.sharpe_delta for r in results)
    worst_trades = max(r.trades_delta for r in results)  # worst = most extra churn

    best_ret = max(r.return_delta for r in results)
    pos_windows = sum(1 for r in results if r.return_delta > 0)

    mode = "pure-LLM" if results else "hybrid"
    print(f"\n=== Aggregate ({n} windows) — {mode} vs deterministic ===")
    print(f"  {'':>14} {'mean':>10} {'worst':>10} {'best':>10}")
    print(f"  {'Return delta':>14} {mean_ret:>+9.2f}% {worst_ret:>+9.2f}% {best_ret:>+9.2f}%")
    print(f"  {'Drawdown delta':>14} {mean_dd:>+9.2f}% {worst_dd:>+9.2f}% "
          f"{min(r.drawdown_delta for r in results):>+9.2f}%")
    print(f"  {'Sharpe delta':>14} {mean_sharpe:>+9.2f} {worst_sharpe:>+9.2f} "
          f"{max(r.sharpe_delta for r in results):>+9.2f}")
    print(f"  {'Trades delta':>14} {mean_trades:>+10} {worst_trades:>+10} "
          f"{min(r.trades_delta for r in results):>+10}")
    print(f"\n  Positive-return windows: {pos_windows}/{n}")


# The savings mirror reads 8 years of candle history back (app/savings.py
# ``_eur_close_history``, ``timedelta(days=365 * 8)``), so a prune cutoff
# inside that window would silently truncate the mirror's history. Keep the
# two windows in sync if either changes.
_PRUNE_MIN_HISTORY_YEARS = 8


async def _prune_candles(before: str, dry_run: bool, allow_degrade: bool = False) -> None:
    """Delete candle rows strictly before `before` (YYYY-MM-DD).

    Rationale: the monthly backtest cannot start earlier than the first month
    with eligible fundamentals (~2022 for the yfinance feed), and the daily
    engine only needs ~2y of history — old rows are dead weight. The FX
    pseudo-tickers and every ticker keep everything after the cutoff.

    FAIL-CLOSED GUARD: a cutoff WITHIN 8 years of today is refused unless
    ``allow_degrade`` is passed — the savings mirror reads 8 years of candle
    history back from the same shared DB (app/savings.py
    ``_eur_close_history``), so pruning rows newer than that window would
    degrade its mirror curve. The guard applies to ``--dry-run`` too: a dry
    run reports exactly the deletion that would happen, so it is refused on
    the same terms. Nothing is deleted on refusal."""
    from .db import Candle
    cutoff = datetime.fromisoformat(before).replace(tzinfo=UTC)
    now = datetime.now(UTC)
    oldest_safe_cutoff = now - timedelta(days=365 * _PRUNE_MIN_HISTORY_YEARS)
    if cutoff > oldest_safe_cutoff and not allow_degrade:
        print(f"refused: --before {cutoff.date()} is within {_PRUNE_MIN_HISTORY_YEARS} years "
              f"of today ({now.date()}) — the savings mirror reads 8 years of candle history "
              "back from the same DB (app/savings.py _eur_close_history), so pruning that "
              "range would degrade its history; pass --allow-degrade to delete anyway")
        return
    async with Session() as s:
        n = (await s.scalar(select(func.count()).select_from(Candle)
                            .where(Candle.timestamp < cutoff))) or 0
        total = (await s.scalar(select(func.count()).select_from(Candle))) or 0
        if dry_run:
            print(f"dry run: would delete {n} of {total} candles before {cutoff.date()}")
            return
        await s.execute(sa_delete(Candle).where(Candle.timestamp < cutoff))
        await s.commit()
        print(f"deleted {n} of {total} candles before {cutoff.date()}")


async def _monthly_backtest(start: str | None, end: str | None, universe: str,
                            contribution: float, verbose: bool,
                            frame_cache: dict[pd.Timestamp, pd.DataFrame | None] | None = None,
                            close_preloaded: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                            fund_preloaded: dict[str, dict[str, list[dict]]] | None = None,
                            open_preloaded: pd.DataFrame | None = None,
                            fixed_capital: float = 0.0,
                            ) -> BacktestSummary | None:
    """DCA backtest of the monthly qv-mom-v1 strategy on stored candles.

    Cadence — the SAME session fill contract (app/session.py) the live
    monthly engine and ``monthly.backfill`` implement: the DECISION is the
    completed month-end's qv-mom frame (fundamentals public by ME, candles
    <= ME), the rebalance FILLS on the FIRST stored session after that
    month-end at that session's strict exact-date OPEN, and the book is
    marked at closes. One session of cash drag, never a month-end-close fill.

    It is an AGGREGATE-VALUE model (one portfolio value per rebalance), not a
    share ledger, so it cannot claim live parity. The approximations that
    remain are disclosed rather than hidden:

      * per-name legs cannot be deferred individually: a name with no stored
        bar on the fill session drops out of that period's equal-weight
        return (the live engine would keep the old position and retry the
        leg at the following session);
      * no intra-period marks: only the fill sessions and the window's last
        close are priced, so intra-month drawdowns do not enter the value (a
        research number can therefore look smoother than the live curve);
      * contributions are calendar-causal but settle lumpy at the fill marks
        (a month whose contribution day arrives between two fills is parked
        at 0% until the next fill deploys it), and the money-weighted IRR
        assumes uniformly spaced contributions;
      * one equal-weight basket rebalance per fill, with the one-way cost on
        swapped notional (``settings.sim_monthly_cost_oneway``) — the same
        research-only cost model as the daily-core backtest.

    ``fixed_capital`` (default ``0.0`` — bit-identical to the old behaviour)
    is the single-shot funding hook for fixed-capital studies (Phase 4 slice
    4B): when > 0 the monthly contributions are suppressed and one lump sum
    of ``fixed_capital`` is funded at the window's first session. The lump
    sits in ``parked`` until the first live rebalance deploys it, exactly
    like a contribution, and the money-weighted IRR degenerates to the
    single-flow annualized return over the window's calendar months.
    """
    from . import monthly as monthly_mod
    from . import fundamentals as fundamentals_mod

    tickers = universe_tickers(universe)
    logger.info("Monthly backtest universe: %d tickers", len(tickers))

    if close_preloaded is not None and fund_preloaded is not None:
        close, vol = close_preloaded
        fund = fund_preloaded
        if open_preloaded is None:
            # Pre-contract callers only carry closes. Falling back to the
            # close frame keeps them runnable, but fills then use the OLD
            # close convention — say so loudly rather than silently.
            logger.warning(
                "Monthly backtest: no open frame supplied with the preloaded "
                "closes — fills fall back to the CLOSE (pre-contract "
                "convention); pass open_preloaded for the session-fill rule.")
            open_px = close
        else:
            open_px = open_preloaded
    else:
        if open_preloaded is not None:
            close, vol = await monthly_mod.load_frames(tickers, None)
            open_px = open_preloaded
        else:
            open_px, close, vol = await monthly_mod.load_ohlcv_frames(tickers, None)
        fund = await fundamentals_mod.load_fundamentals(tickers)
    if close.empty:
        print("No candle data found. Run the monthly rebalance (or a data refresh) first.")
        return
    logger.info("Frames: %d days x %d tickers; fundamentals for %d tickers",
                len(close), close.shape[1], len(fund))

    # The strict fill series, aligned to the close calendar WITHOUT any
    # forward fill — a missing open is not an executable price.
    open_px = open_px.reindex(index=close.index, columns=close.columns)

    idx = pd.DatetimeIndex(close.index)
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        print("No trading days in the requested window.")
        return

    # COMPLETED month-ends only, by the US session calendar (same rule as
    # _daily_core_backtest — see monthly.completed_month_ends): a window
    # ending inside a still-running month keeps that partial stretch —
    # mark-to-marketed below without a synthetic rebalance.
    months = monthly_mod.completed_month_ends(close.index, idx[0], idx[-1])

    # Momentum warmup uses closes *before* the window (--start filters only
    # the rebalance months, the full frame is loaded). Early months whose
    # warmup is thin simply have no eligible names and get trimmed below —
    # only refuse when even the LAST window day can't reach 253 days of
    # history.
    n_days_to_first = int((close.index <= idx[-1]).sum())
    if n_days_to_first < settings.sim_monthly_min_history_days:
        print(f"Not enough history ({n_days_to_first} trading days up to "
              f"{idx[-1].date()}; need {settings.sim_monthly_min_history_days} "
              f"for momentum warmup).")
        return

    if not months:
        print("No completed month-end inside the requested window.")
        return

    # eligibility frame per rebalance date is the expensive part — cache by
    # month (a prebuilt cache can be passed in by the parameter sweep);
    # missing keys are built here so a foreign/partial cache never
    # KeyErrors on a month the window's calendar rule selects.
    if frame_cache is None:
        frame_cache = {}
    for i, d in enumerate(months):
        if d not in frame_cache:
            frame_cache[d] = await asyncio.to_thread(
                monthly_mod.eligible_frame, d, close, vol, fund)
            if (i + 1) % 12 == 0:
                logger.info("eligibility %d/%d", i + 1, len(months))

    # The FILL session of each month-end: the FIRST stored session strictly
    # after it (the deferred convention). A month-end whose next session is
    # outside the window is a PENDING rebalance — the previous picks keep
    # marking and nothing is fabricated. months are increasing, so the first
    # pending one ends the sequence.
    fill_days: list[pd.Timestamp] = []
    for me in months:
        later = idx[idx > pd.Timestamp(me)]
        if not len(later):
            break
        fill_days.append(later[0])

    # NO dead-month trim on eligibility: dropping months whose frame turned
    # out dead erased their funding and made the aggregate depend on FUTURE
    # eligibility. A dead month stays in the sequence and simply holds the
    # previous picks through it (pick_portfolio returns the holdings for a
    # dead frame), while its contribution is still settled — the live
    # engine deposits monthly regardless of eligibility. (Fee policy is
    # unchanged: this aggregate keeps its one-way cost model.)
    cost = settings.sim_monthly_cost_oneway
    contributed = 0.0
    holdings: list[str] = []
    churns: list[float] = []
    picks_hist: list[tuple[str, list[str]]] = []

    def period_ret(d0: pd.Timestamp, d1: pd.Timestamp, names: list[str],
                   *, end_at_close: bool = False) -> float:
        """Equal-weight return of `names` from the fill at d0 to d1.

        Both ends read the STRICT open (fill-to-fill), except the window's
        final mark (``end_at_close``), which reads the closes so the trailing
        stretch is marked at the last session's close. A name without a price
        at either end drops out of the mean (documented approximation: the
        live engine keeps the old position and retries the leg).
        """
        if not names:
            return 0.0
        p0 = open_px.loc[d0].reindex(names)
        p1 = (close.loc[d1] if end_at_close else open_px.loc[d1]).reindex(names)
        r = (p1 / p0 - 1.0).dropna()
        return _col_f(r.mean()) if len(r) else 0.0

    # Contributions are counted per CALENDAR month present in the window and
    # settled at the marks: each month-end settles every not-yet-booked
    # month whose contribution day has arrived by then, and the trailing
    # stretch settles the rest. The live engine deposits on the month's
    # first trading day regardless of eligibility or of whether the
    # month-end candle is stored, so a LEADING hole (start Jan-2, Jan-31's
    # bar missing, first completed end in February) funds BOTH January and
    # February, and a trailing stretch after an interior hole can span
    # several funding months — keying the count on month-ends alone dropped
    # those deposits. Parked-vs-invested separation: only the deployed
    # portfolio earns the month-gap returns — contributions sit as parked
    # cash (0%) until a live-frame rebalance deploys them, so no
    # contribution earns a return for a period before it is invested (a
    # dead frame leaves money parked; no synthetic rebalance is fabricated).
    # Approximations kept and disclosed (no live-parity claim): settlement
    # is lumpy at the marks (a hole month's deposit books at the next
    # month-end) and the money-weighted IRR assumes uniformly spaced
    # contributions.
    first_days = sorted(_contribution_days(
        idx, {(d.year, d.month) for d in idx}))
    invested = 0.0   # deployed portfolio value (earns the gap returns)
    parked = 0.0     # contributions received, not yet deployed (cash: 0%)
    booked: set[tuple[int, int]] = set()
    n_contrib = 0
    fixed = float(fixed_capital or 0.0)
    if fixed > 0:
        # Single-shot funding (fixed-capital studies): suppress the monthly
        # contributions and fund one lump sum at the window's first session.
        # The lump sits in `parked` until the first live rebalance deploys
        # it, exactly like a contribution. At fixed == 0 nothing below
        # changes (bit-identical: contribution passes through unchanged).
        contribution = 0.0
        parked += fixed
        contributed += fixed

    def _settle_arrived(mark: pd.Timestamp) -> None:
        """Book every calendar month whose contribution day has arrived by
        ``mark`` — never a future key, never earlier than the mark."""
        nonlocal parked, contributed, n_contrib
        for day in first_days:
            key = (int(day.year), int(day.month))
            if key not in booked and day <= mark:
                parked += contribution
                contributed += contribution
                n_contrib += 1
                booked.add(key)

    prev_fill: pd.Timestamp | None = None
    for me, fd in zip(months, fill_days, strict=False):
        # Mark the deployed book from the PREVIOUS fill to this one (strict
        # opens at both ends — the fill-to-fill holding period).
        if prev_fill is not None:
            invested *= 1.0 + period_ret(prev_fill, fd, holdings)
        # Deposits that arrived by the fill session join the rebalance: the
        # live engine deposits at the month's first session and deploys in
        # the same cycle, so a fill on the month's first session sees that
        # month's contribution.
        _settle_arrived(fd)
        frame = frame_cache[me]
        live = frame is not None and bool(frame["eligible"].any())
        picks, _ = monthly_mod.pick_portfolio(me, close, vol, fund, holdings, frame) if frame is not None else (holdings, None)
        # turnover = fraction of the portfolio's slots swapped at this
        # rebalance: buys of new names plus sells of dropped names, each leg
        # one-way.
        adds = len(set(picks) - set(holdings))
        drops = len(set(holdings) - set(picks))
        turn = (adds + drops) / max(max(len(picks), len(holdings)), 1)
        # A live frame rebalances (one pass for every month it has a ranking,
        # no-op months included): parked cash joins the book. The fresh
        # contribution is deployed by buying new/rebalanced names — charging
        # it separately would double-count the swap leg on entry months.
        # One-way cost on swapped notional only; a dead frame keeps money
        # parked (turn is 0 there, so no cost either).
        if live:
            invested += parked
            parked = 0.0
        invested *= 1.0 - turn * cost
        churns.append(turn)
        if picks != holdings:
            picks_hist.append((me.strftime("%Y-%m"), list(picks)))
        holdings = list(picks)
        prev_fill = fd

    # Trailing stretch: mark the deployed book from the LAST fill's open to
    # the window's last close (the last session's mark), and settle every
    # remaining funding month — parked, so none of them earns the stretch's
    # return. When no rebalance ever filled (a window that ends before the
    # first fill session) everything stays parked and nothing is marked.
    if prev_fill is not None:
        invested *= 1.0 + period_ret(prev_fill, idx[-1], holdings,
                                     end_at_close=True)
    _settle_arrived(idx[-1])

    value = invested + parked

    if fixed > 0:
        # Single-flow annualized return over the window's calendar months
        # (the money-weighted formula below assumes n equal monthly flows,
        # which a lump sum is not).
        n_months = len({(d.year, d.month) for d in idx})
        irr = ((value / fixed) ** (12.0 / max(n_months, 1)) - 1.0) if value > 0 else -1.0
    else:
        # money-weighted IRR: contribution * sum((1+r)^k) = final value
        n = n_contrib

        def f(r: float) -> float:
            return contribution * sum((1.0 + r) ** k for k in range(n)) - value

        lo, hi = -0.90, 1.0
        if f(lo) > 0:
            irr = -1.0
        else:
            for _ in range(200):
                mid = (lo + hi) / 2.0
                if f(mid) < 0:
                    lo = mid
                else:
                    hi = mid
            irr = (1.0 + (lo + hi) / 2.0) ** 12 - 1.0

    print(f"\n=== Monthly qv-mom backtest [{universe}] {months[0].date()}..{months[-1].date()} ===")
    if fixed > 0:
        print(f"  fixed capital: one lump sum ${fixed:,.0f} at the window's first session "
              f"(monthly contributions suppressed) = ${contributed:,.0f}")
    else:
        print(f"  contributions: {n_contrib} x ${contribution:,.0f} = ${contributed:,.0f}")
    print(f"  final value:   ${value:,.0f}  (multiple {value / max(contributed, 1e-9):.2f}x)")
    print(f"  money-weighted IRR: {irr:.2%}/yr   avg monthly turnover: {sum(churns) / len(churns):.1%}"
          if churns else f"  money-weighted IRR: {irr:.2%}/yr   avg monthly turnover: n/a")
    print(f"  (research-only cost model: {cost * 1e4:.0f} bps one-way on swapped notional "
          f"at the fill; fundamentals: SEC EDGAR point-in-time "
          f"for US filers, yfinance approximation for CIK-less listings)")
    if verbose:
        print("\nLast 12 rebalances:")
        for m, picks in picks_hist[-12:]:
            print(f"  {m}: {', '.join(picks)}")

    return BacktestSummary(
        irr=irr,
        final_value=value,
        contributed=contributed,
        avg_turnover=sum(churns) / len(churns) if churns else 0.0,
        n_trades=0,
        window=(pd.Timestamp(months[0]).strftime("%Y-%m-%d"),
                pd.Timestamp(months[-1]).strftime("%Y-%m-%d")),
    )


@dataclass
class BacktestSummary:
    """Result summary for a daily-core or monthly backtest run. Returned by
    the backtests so tests and the parameter sweep can consume them without
    parsing stdout."""
    irr: float
    final_value: float
    contributed: float
    avg_turnover: float
    n_trades: int
    window: tuple[str, str]
    sharpe: float = 0.0       # annualized, contribution-adjusted daily returns
    max_drawdown: float = 0.0  # peak-to-trough, fraction (positive number)
    brake_events: int = 0      # portfolio-stop cash-outs that fired
    basket_events: int = 0     # gradient-filter cash-outs that fired
    # Per-fill ledger for the daily-core replay: one dict per leg with
    # ``date``, ``side``, ``ticker``, ``notional``, ``shares`` and ``price``
    # (the strict-open fill price). tests/test_replay_fills.py compares it
    # against daily_core.backfill's trade rows.
    trades: list[dict] = field(default_factory=list)
    n_thesis_break: int = 0     # thesis-break (M5) exits that fired
    # Per thesis-break exit: {"date" (the fill), "ticker", "decision_date"
    # (S, the session whose public fundamentals broke the thesis)}. The trade
    # ledger carries no reason string, so the hook records its exits here for
    # sellshadow's post-hoc classifier.
    thesis_break_exits: list[dict] = field(default_factory=list)


def _thesis_break_fundamentals(rec: dict | None, asof: pd.Timestamp) -> bool:
    """The PREDECLARED M5 thesis-break condition, evaluated on the fundamentals
    PUBLIC as of ``asof`` (filing-date PIT: an entry is public only when
    ``filed <= asof``; restatements replace originals — ``monthly._asof_entries``):

    the fundamentals record is missing, OR equity <= 0, OR TTM NI missing,
    OR shares missing, OR **ROE = TTM NI / equity <= 0** (the quality leg
    collapses). F/P-FCF changes alone do NOT trigger (the value leg is a
    rank, not a gate). PRICE-side gates (mcap, dollar_vol, momentum,
    staleness) are explicitly NOT re-checked here — price events are not
    thesis breaks.

    Uses the engine's OWN helpers (``monthly._latest_as_of`` /
    ``monthly._ttm_as_of``), so asof=S gives exactly "the fundamentals public
    as of session S" — the same hard gates ``monthly.eligible_frame`` applies.
    """
    from . import monthly as monthly_mod
    if not rec:
        return True
    eq = monthly_mod._latest_as_of(rec.get("StockholdersEquity", []), asof)
    ni = monthly_mod._ttm_as_of(rec.get("NetIncomeLoss", []), asof)
    sh = monthly_mod._latest_as_of(
        rec.get("CommonStockSharesOutstanding", [])
        + rec.get("EntityCommonStockSharesOutstanding", []),
        asof,
    )
    return (not eq or eq[1] <= 0 or ni is None or sh is None
            or ni / eq[1] <= 0)


def _inv_vol_weights(names: list[str], frame: pd.DataFrame | None,
                     target_n: int, default_w: float) -> dict[str, float]:
    """Inverse-vol target weights for the portfolio's candidate names.

    The names with measurable positive vol split the pool proportionally
    to 1/vol; names with missing/non-finite vol keep ``default_w`` (never
    silently zeroed over missing data). The weights sum to
    ``len(names) * default_w`` — i.e. the same total equity the equal-
    weight deployment would spread — so the two modes deploy cash at the
    same overall rate."""
    out = {t: default_w for t in names}
    if frame is None or "vol" not in frame.columns or not names:
        return out
    ivals: dict[str, float] = {}
    for t in names:
        v = frame["vol"].get(t)
        if v is not None and np.isfinite(v) and float(v) > 0:
            ivals[t] = 1.0 / float(v)
    if len(ivals) <= 1:
        return out
    total = sum(ivals.values())
    # measurable names split the pool REMAINING after the unmeasurable
    # names take their equal weight — the total sums to len(names)*default_w.
    pool = len(ivals) * default_w
    for t, iv in ivals.items():
        out[t] = (iv / total) * pool
    return out


def _fill_buy(cash: float, price: float, delta_value: float,
              cost: float) -> tuple[float, float, float] | None:
    """One BUY fill: ``(notional, shares, cash_after)`` or None when the
    spend would be under $1 or the booked cost exceeds the exact cash.

    The zero-fee fill itself — 4dp-floored sizing, the $1 / ``0.0001``-share
    minimums and the EXACT-cash guard — is ``factor_fills.buy_fill``
    (app/factor_fills.py); this wrapper adds ONLY the research replay's
    one-way ``cost`` around it, because a fee-aware adaptation belongs outside
    the primitives and never as mutated sizes there. ``shares =
    floor4(budget / price)`` and ``notional = shares * price`` are therefore
    identical to the live executor (``daily_core.backfill.do_buy`` /
    ``daily_core._exec_buy``, and the monthly engines' ``share_size`` call),
    and the leg is DROPPED when the account cash cannot cover that booked cost
    — the live engine skips it and re-takes the decision next session. That
    guard matters at ``cost=0`` in the razor-thin case where the caller's
    budget is the ROUNDED cash (``round(cash, 2)``) and the 4dp share floor
    lands a fraction of a cent above the exact cash: shrinking the budget to
    the exact cash instead would book a fill the live engine never makes
    (tests/test_replay_fills.py pins the golden parity).

    With a non-zero ``cost`` the one-way fee is reserved INSIDE the spend, so
    a full-cash research buy can never drive cash negative — without this the
    fee was charged on top and every full-cash buy left the account at
    ``-cash * cost``, which also blocked follow-on buys until cash recovered.
    """
    budget = delta_value
    if cost > 0:
        budget = min(budget, max(cash, 0.0) / (1.0 + cost))
    if budget < 1.0:
        return None
    fill = factor_fills.buy_fill(cash, budget, price)
    if fill is None:
        return None
    if cash < fill.notional * (1.0 + cost):
        # The FEE-aware head-room check on top of the primitive's zero-fee
        # EXACT-cash guard. Even with the budget capped above, the float
        # rounding of ``notional * (1.0 + cost)`` can exceed the cash by an
        # ULP; only this guard stops a negative-cash booking at cost > 0.
        # At cost=0 it is the primitive's own guard (already satisfied).
        # A missing fill is NOT forced: the research replay retries the leg
        # at the next session, exactly like the live cycle.
        return None
    # The fee comes out of CASH only — the size is never re-floored. At
    # cost=0 ``notional * (1.0 + 0.0)`` is exactly ``notional``, so this is
    # bit-identical to the old expression and to ``fill.cash_after``.
    return fill.notional, fill.shares, cash - fill.notional * (1.0 + cost)


def _contribution_days(days: pd.DatetimeIndex,
                       month_keys: set[tuple[int, int]]) -> set[pd.Timestamp]:
    """The days on which a monthly contribution is received.

    One per calendar month key in ``month_keys``, deposited on that month's
    FIRST trading day in the window (mirroring the live daily-core engine
    and ``daily_core.backfill``, which deposit at the start of the month)
    rather than parking the cash until the month-end rebuild. Callers pass
    the calendar months that should fund — currently every calendar month
    present in the replay window, so an interior month with normal daily
    bars but a missing month-end candle still receives its deposit."""
    out: set[pd.Timestamp] = set()
    seen: set[tuple[int, int]] = set()
    for d in days:
        k = (d.year, d.month)
        if k in month_keys and k not in seen:
            seen.add(k)
            out.add(d)
    return out


def _month_price_map(close: pd.DataFrame) -> dict[str, dict[str, float]]:
    """``{month: {ticker: FIRST close of that month}}`` for the P/FCF scale.

    The quality lookup caches one map per month; using the month's first
    close (known at the month's start) instead of its last removes the
    look-ahead where an early-month decision was priced with a later
    close."""
    out: dict[str, dict[str, float]] = {}
    if close.empty:
        return out
    for t in close.columns:
        for ts, px in close[t].dropna().items():
            m = pd.Timestamp(ts).strftime("%Y-%m")  # type: ignore[arg-type]
            out.setdefault(m, {}).setdefault(t, float(px))
    return out


async def _daily_core_backtest(
    start: str | None,
    end: str | None,
    universe: str,
    contribution: float,
    verbose: bool,
    entry_gate: str,
    exit_cadence: str,
    crash_pause: bool,
    dca: str,
    frame_cache: dict[pd.Timestamp, pd.DataFrame | None] | None = None,
    close_preloaded: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    fund_preloaded: dict[str, dict[str, list[dict]]] | None = None,
    open_preloaded: pd.DataFrame | None = None,
    cost_oneway: float | None = None,
    with_baseline: bool = True,
    vol_weight: bool | None = None,
    target_vol: float | None = None,
    portfolio_stop_pct: float | None = None,
    exposure_trend_days: int | None = None,
    trailing_stop_pct: float | None = None,
    basket_trend_days: int | None = None,
    basket_confirm_days: int | None = None,
    basket_threshold: float | None = None,
    basket_drawdown: float | None = None,
    basket_er_min: float | None = None,
    basket_good_times: bool | None = None,
    basket_arm_sma: int | None = None,
    fixed_capital: float = 0.0,
    thesis_break_exit: bool | None = None,
) -> BacktestSummary | None:
    """Deterministic "daily-core" backtest: the monthly qv-mom portfolio as the
    FUNDAMENTAL core, with minor candle-driven daily adjustments on top.

    Philosophy (per the owner's design): fundamentals decide WHAT to own —
    the monthly qv-mom ranking (ROE + 12-1 momentum + 1/P-FCF, top-N with
    hysteresis, no stops) is the portfolio. Candles only adjust WHEN:

      - ``dca`` deployment: "monthly" parks the contribution until month-end
        (the monthly sim's behaviour), "daily" deploys it immediately into
        the most underweight holding (kills the average ~2-week cash drag).
      - ``entry-gate``: candles time the ENTRY of a NEW core name —
        "above-sma50" waits for the price to be above its SMA50, "not-crash"
        merely blocks entries into a 5-day crash. "none" buys at once.
      - ``exit-cadence``: how fast a held name that dropped out of the
        hysteresis band is released — "monthly" (the sim) waits for the
        month-end rebalance, "daily" releases it the next day.
      - ``crash-pause``: pause NEW entries when the universe is in a
        pullback regime (breadth + median 5d run deeply negative).
      - ``thesis_break_exit`` (default OFF — the no-arg path is
        bit-identical): the M5 fundamental thesis-break exit. At each
        session S (the decision pass, on data <= S) each holding's
        fundamentals are re-derived with the engine's OWN helpers
        (``monthly._latest_as_of`` / ``_ttm_as_of``, filing-date PIT). The
        PREDECLARED thesis-break condition (fundamental-only): the
        fundamentals record is missing, OR equity <= 0, OR TTM NI missing,
        OR shares missing, OR **ROE = TTM NI / equity <= 0**. F/P-FCF
        changes alone do NOT trigger (the value leg is a rank, not a gate);
        PRICE-side gates (mcap, dollar_vol, momentum, staleness) are
        explicitly NOT re-checked intraday. A break discovered at S schedules
        an exit filled at F's open (the standing-order convention, in the F
        loop before the BUY phase, same cost path as every other sell).
        Re-entry is UNCHANGED (the ranking can bring the name back at a
        later rebuild). Exits are counted (``n_thesis_break``) and recorded
        per fill (``thesis_break_exits``) in the summary.

    With all knobs at their "monthly/none" defaults the replay reduces to the
    monthly qv-mom backtest (same eligibility, same hysteresis, same cost
    model) — that equivalence is the sanity check. The report compares the
    money-weighted IRR against the monthly baseline over the same window.

    SESSION FILL CONTRACT (app/session.py) — this replay walks F = each stored
    session in order and treats ``S`` = the previous stored session:

      * DECISION on data <= S: the ranking frame is the newest COMPLETED
        month-end <= S (so the band switches on the first session AFTER a
        month-end, and the month-end rebuild/release pass runs on that first
        session — ``rebuild``), and every overlay (crash-pause breadth,
        exposure gate, portfolio-stop drawdown, entry gate, vol management)
        reads S's close. F's own bar never changes the decision.
      * SIZING on data <= S: the deployment budgets (target weight, the
        current held value behind every gap, the vol-cap's deployment room)
        are marked at S's close with the causal prior-close carry
        (``sizing_px_of``). The order is placed at F's open, so F's own
        close — known only after the fill — can never size it; a mutated
        F-day close leaves that F-day's legs untouched and becomes visible
        only when it IS the next session's S. F's close still marks the
        EOD/reporting equity and the vol/drawdown observations (after the
        fills), and advances the trailing-stop peaks for the next session.
      * FILLS at F's strict exact-date OPEN per ticker (``session.open_at``):
        never forward-filled, never a stale close. A ticker without a bar on
        F has NO fill and the decision is re-taken next session (the band
        release / deployment retries). Marks/valuation use the
        forward-filled CLOSE, but only while the print behind the cell is
        younger than ``settings.factor_price_max_age_days``
        (``app/price_freshness.py``); a stale print marks at the position's
        cost basis — the same fallback ``daily_core.backfill`` and the live
        ``valuate()`` use.
      * The trailing stop is a standing order: the level known at S's close
        fills at F's open when the open is already through it. There is no
        low/high frame in this loader, so an intrabar breach that closes back
        above the level waits for the next session's open (documented, not
        modelled) — the live engine has F's low and would fill at the level.
      * ``cost_oneway`` is the research-only cost model (default
        ``settings.sim_monthly_cost_oneway`` = 10 bps one-way). It is NOT the
        live engine's policy: the live paper portfolios are fee-free, which is
        why tests/test_replay_fills.py runs this replay at
        ``cost_oneway=0.0`` to compare fills with ``daily_core.backfill``.

    ``fixed_capital`` (default ``0.0`` — bit-identical to the old behaviour)
    is the single-shot funding hook for fixed-capital studies (Phase 4 slice
    4B): when > 0 the monthly contributions are suppressed and one lump sum
    of ``fixed_capital`` is deposited at the window's first session (into
    ``cash``, or ``pending_cash`` under dca="monthly"), the turnover
    bookkeeping treats that first-month deployment as a fresh contribution
    (not churn), and the money-weighted IRR degenerates to the single-flow
    annualized return over the window's calendar months.
    """
    from . import monthly as monthly_mod
    from . import fundamentals as fundamentals_mod

    tickers = universe_tickers(universe)
    logger.info("Daily-core backtest universe: %d tickers", len(tickers))

    if close_preloaded is not None and fund_preloaded is not None:
        close, vol = close_preloaded
        fund = fund_preloaded
        if open_preloaded is None:
            # Pre-contract callers only carry closes (the parameter sweep did
            # until opens were threaded through). Falling back to the close
            # frame keeps them runnable, but fills then use the OLD close
            # convention — say so loudly rather than silently.
            logger.warning(
                "Daily-core backtest: no open frame supplied with the "
                "preloaded closes — fills fall back to the CLOSE "
                "(pre-contract convention); pass open_preloaded for the "
                "session-fill rule.")
            open_px = close
        else:
            open_px = open_preloaded
    else:
        if open_preloaded is not None:
            close, vol = await monthly_mod.load_frames(tickers, None)
            open_px = open_preloaded
        else:
            open_px, close, vol = await monthly_mod.load_ohlcv_frames(tickers, None)
        fund = await fundamentals_mod.load_fundamentals(tickers)
    if close.empty:
        print("No candle data found. Run the monthly rebalance (or a data refresh) first.")
        return
    logger.info("Frames: %d days x %d tickers; fundamentals for %d tickers",
                len(close), close.shape[1], len(fund))

    # The strict fill series, aligned to the close calendar WITHOUT any
    # forward fill — a missing open is not an executable price.
    open_px = open_px.reindex(index=close.index, columns=close.columns)

    idx = pd.DatetimeIndex(close.index)
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        print("No trading days in the requested window.")
        return

    days_all = idx  # every trading day in the window
    last_day = days_all[-1]

    # COMPLETED month-ends only, by the US session calendar (the same
    # scheduler rule the live engine uses) — see
    # monthly.completed_month_ends. A frame ending inside a still-running
    # month has no month-end, and an ``end`` cut mid-month must not
    # rebalance at an arbitrary partial-month date.
    full_idx = pd.DatetimeIndex(close.index)
    months = monthly_mod.completed_month_ends(close.index, days_all[0], last_day)
    month_set = set(months)
    # Position of every walk day in the FULL stored index: S = the stored
    # session before it (contract: the decision is made on S's completed
    # bar, so the first window day is allowed to decide on the session
    # before the window — the ranking frame and the overlays both use it).
    locs = full_idx.get_indexer(pd.DatetimeIndex(days_all))

    # Most recent PRE-window month-end, keyed off the WINDOW START's month —
    # independent of the rebalance list above (a short Jan-02..Jan-15 window
    # has no month-end of its own but must still rank against Dec-31's
    # frame). The days between the window start and the first window
    # month-end rank against the last frame public BEFORE the window (facts
    # public by its own date — point-in-time safe). The old ``month_for``
    # fallback returned months[0] instead, a FUTURE frame, so a January run
    # traded on January's own month-end ranking (audit P0-E).
    prior_month_end: pd.Timestamp | None = None
    d0 = days_all[0]
    prev_y, prev_m = (d0.year - 1, 12) if d0.month == 1 else (d0.year, d0.month - 1)
    pme_sched = pd.Timestamp(
        monthly_mod.month_last_trading_day(datetime(prev_y, prev_m, 1)))
    prev_idx = full_idx[full_idx <= pme_sched]
    if len(prev_idx):
        # Last stored candle at/below the scheduled prior month-end: a
        # ranking-only frame (no rebalance trigger), point-in-time safe.
        prior_month_end = prev_idx[-1]

    # Momentum warmup: same rule as _monthly_backtest.
    n_days_to_first = int((close.index <= last_day).sum())
    if n_days_to_first < settings.sim_monthly_min_history_days:
        print(f"Not enough history ({n_days_to_first} trading days up to "
              f"{last_day.date()}; need {settings.sim_monthly_min_history_days} "
              f"for momentum warmup).")
        return

    # NOTE: no "months empty -> return" guard here. A window with NO
    # in-window month-end (e.g. Jan-02..Jan-15) still runs — every day ranks
    # against the pre-window frame and the daily deployment builds the book;
    # it simply never hits a rebuild day.

    # Eligibility frames per MONTH (fundamentals move quarterly; daily
    # overlays reuse the current month's frame — no per-day TTM recompute).
    # A prebuilt cache can be passed in (parameter sweep: frames built once,
    # reused across every config); any month key it lacks is built here so a
    # cache assembled by a different helper or window never KeyErrors or
    # silently drops a month's ranking. NOTE: the missing keys are built
    # with the AMBIENT mom variant — the sweep pins the variant BEFORE
    # building its cache and passes complete caches for its own keys.
    if frame_cache is None:
        frame_cache = {}
    n_built = 0
    for d in months:
        if d not in frame_cache:
            frame_cache[d] = await asyncio.to_thread(
                monthly_mod.eligible_frame, d, close, vol, fund)
            n_built += 1
            if n_built % 12 == 0:
                logger.info("eligibility %d/%d (cache fill)", n_built, len(months))

    # The pre-window ranking frame: build it into the cache when a caller's
    # prebuilt cache doesn't carry it (the sweep builds from its own start —
    # its windows normally already include the prior month-end, so this is a
    # single extra frame at most).
    if prior_month_end is not None and prior_month_end not in frame_cache:
        frame_cache[prior_month_end] = await asyncio.to_thread(
            monthly_mod.eligible_frame, prior_month_end, close, vol, fund)

    # NO dead-month trim on frame coverage: cutting days/months whose frame
    # turned out ineligible made the replay's START depend on FUTURE
    # eligibility (a valid Dec-31 prior + empty Jan-31 + valid Feb-28 frame
    # trimmed January entirely, so the prior frame never funded Jan-2, and
    # January's contribution vanished). The simulation is causal instead:
    # every window day is walked in order; a day ranks against its most
    # recent completed month-end (the pre-window frame before the first
    # one) and a dead/missing frame simply yields no ranking — cash sits
    # and no entries happen until a valid frame becomes current again.
    # Contributions are booked per calendar month present in the window
    # (see contrib_keys below), independent of rebalance bars.

    cost = (settings.sim_monthly_cost_oneway if cost_oneway is None
            else float(cost_oneway))
    target_n = settings.sim_monthly_target_n
    hold_band = settings.sim_monthly_hold_band
    fixed = float(fixed_capital or 0.0)
    if fixed > 0:
        # Single-shot funding hook (fixed-capital studies): suppress the
        # monthly contributions; one lump sum funds at the window's first
        # session (booked in the day loop below). At fixed == 0 every branch
        # below is bit-identical to the old behaviour (contribution passes
        # through unchanged).
        contribution = 0.0

    # --- Crash-pause regime series (breadth + median 5d run) ---
    # Daily-frame variant of the LLM context's regime heuristic: PULLBACK =
    # <60% of eligible names above their SMA50 AND median 5d run < -2.5%.
    sma50_frame = close.rolling(50).mean()
    run5_frame = close.pct_change(5)
    above50 = (close > sma50_frame)

    def pullback_day(d: pd.Timestamp | None) -> bool:
        if d is None or d not in above50.index:
            return False
        row = above50.loc[d].dropna()
        if row.empty:
            return False
        breadth = row.mean()
        r5 = run5_frame.loc[d].dropna()
        med5 = _col_f(r5.median()) if len(r5) else 0.0
        return bool(breadth < 0.60 and med5 < -0.025)

    def entry_ok(t: str, d_ref: pd.Timestamp | None) -> bool:
        """The entry gate as a DECISION input — evaluated on data <= S.

        The gated fill happens at F's open, so reading the SMA50/5-day run at
        F's close would be look-ahead: the gate would "know" the session's
        outcome before the fill. ``d_ref`` is S; during warmup/missing data
        the fail-closed branches below match the pre-contract behaviour.
        """
        if entry_gate == "none" or d_ref is None:
            return True
        if entry_gate == "above-sma50":
            s = px_of_col(sma50_frame, d_ref, t)
            c = px_of(t, d_ref)
            return s is not None and c is not None and c > s
        if entry_gate == "not-crash":
            r5 = px_of_col(run5_frame, d_ref, t)
            return r5 is None or r5 > -0.10
        return True

    # --- Portfolio state ---
    shares: dict[str, float] = {}     # ticker -> shares held
    peak_px: dict[str, float] = {}    # ticker -> highest close since entry (trailing stop)
    # True average cost per holding, tracked exactly like daily_core's
    # do_buy/do_sell bookkeeping: the fallback mark when a position has no
    # USABLE market print (price-age policy — a stale print is not a mark).
    cost_basis: dict[str, float] = {}
    cash = 0.0
    contributed = 0.0
    n_contribs = 0
    pending_pick: dict[str, pd.Timestamp] = {}   # ticker -> first day seen in-band (entry-gate)
    churns: list[float] = []
    trades_log: list[tuple[str, str, str, float]] = []
    # M5 thesis-break exits: one record per FILLED exit — the ledger carries
    # no reason string, so the hook records {date, ticker, decision_date} here
    # for the summary (and sellshadow's post-hoc classifier).
    thesis_break_log: list[dict] = []
    # Monthly turnover bookkeeping: notional traded per calendar month,
    # relative to the equity at that month's start. Fresh-contribution
    # deployment is excluded (same convention as the monthly baseline,
    # where deploying the contribution is not counted as churn).
    month_buys = 0.0
    month_sells = 0.0
    equity_month_start = 0.0
    cur_month: int | None = None
    first_month_closed = False  # fixed-capital lump excluded from the first month's churn

    # Valuation closes: each ticker valued at its last valid close <= day
    # (same rule as eligible_frame). A raw close.at[d, t] treats a ticker
    # with no candle that day (market not closed yet, holiday) as worthless,
    # which valued whole positions at $0 when the window ended on a
    # partial/intraday day — final equity collapsed to leftover cash and
    # IRR to -100%.
    close_val = close.ffill()

    # Price-age policy (app/price_freshness.py): a forward-filled cell is a
    # mark only while the print BEHIND it is younger than
    # settings.factor_price_max_age_days. Computed ONCE from the RAW close
    # frame (never from close_val — ffill erases the print dates) and read
    # per day; a stale cell behaves like a missing close: the position is
    # marked at its cost basis (the same fallback daily_core.backfill and the
    # live valuate() use), never at the ancient print.
    fresh_marks = price_freshness.freshness_mask(
        close, None, settings.factor_price_max_age_days)

    def _mark_is_fresh(t: str, d: pd.Timestamp) -> bool:
        try:
            return bool(fresh_marks.at[d, t])
        except KeyError:
            return False

    def px_of(t: str, d: pd.Timestamp) -> float | None:
        return monthly_mod.px_at(close_val, d, t)

    def fill_px_of(t: str, d: pd.Timestamp) -> float | None:
        # Session fill contract rule 2 (app/session.py): the strict
        # exact-date OPEN — a stored bar on `d`, or NOTHING. A missing quote
        # is not an executable price, so the leg retries at the next session
        # (never a stale or forward-filled close).
        return monthly_mod.px_at(open_px, d, t)

    def sizing_px_of(t: str, s_ref: pd.Timestamp | None) -> float | None:
        """The SIZING mark: ticker t's last valid close at/before S.

        Target weights, cash gaps and current held values are all struck at
        F's OPEN, so they may only read prices already known when the order
        is placed — S's completed close. Reading F's own close instead made
        the budget depend on the session's OUTCOME (a name that rallied into
        the close sized its own top-up differently, and the whole book's
        weight moved with it), which is the look-ahead this helper removes.
        The mark is carried forward from the last stored close <= S
        (``close_val`` is the point-in-time ffill the eligibility frame and
        every overlay use), never from F's bar. ``None`` when the window has
        no previous stored session: the book is empty then, so nothing is
        marked.

        PRICE-AGE POLICY (``app/price_freshness.py``): the carried-forward
        cell is a mark only while the print behind it is fresh. A stale
        print falls to the SAME cost-basis fallback the persisted replay and
        the live valuate() use (``None`` when even that is unknown, which the
        callers read as 0.0 for an unheld name).
        """
        if s_ref is None:
            return None
        if not _mark_is_fresh(t, s_ref):
            return cost_basis.get(t)
        return px_of(t, s_ref)

    def eod_px_of(t: str, d: pd.Timestamp) -> float | None:
        """The EOD/reporting mark: F's close while its print is fresh; a
        stale print is not a mark, so the cost basis reports instead (same
        rule as the persisted replay's ``mark_f``)."""
        if not _mark_is_fresh(t, d):
            return cost_basis.get(t)
        return px_of(t, d)

    def trade(d: pd.Timestamp, t: str, side: str, delta_value: float) -> float:
        """Execute one fill at d's strict exact-date OPEN.

        Returns the BOOKED notional rounded to cents (0.0 when the leg did
        not fill) so deployment callers can decrement their room exactly as
        ``daily_core.backfill`` does. ``delta_value`` caps the spend; shares
        are 4dp-floored through the shared ``factor_fills`` primitives on
        the open, so at ``cost=0`` the booked ticker/side/shares/price rows
        are identical to the live executor's (tests/test_replay_fills.py
        pins that parity).
        """
        nonlocal cash, month_buys, month_sells
        p = fill_px_of(t, d)
        if p is None or p <= 0:
            return 0.0
        if side == "BUY":
            if t not in shares and len(shares) >= target_n:
                # Hard ten-slot backstop: `_deploy_targets` plans the
                # post-release membership (refill=True fills toward
                # target_n assuming the pending releases succeed), so an
                # executor must check the ACTUAL free capacity before
                # opening a NEW name — a failed release (no usable price)
                # leaves its slot occupied and the book must not grow.
                return 0.0
            fill = _fill_buy(cash, p, min(delta_value, round(cash, 2)), cost)
            if fill is None:
                return 0.0
            notional, sh, cash = fill
            prev_sh = shares.get(t, 0.0)
            shares[t] = prev_sh + sh
            # True average cost, booked exactly like daily_core's do_buy
            # (shares-weighted at the booked notional). It is only the
            # fallback MARK for a position whose print went stale — never a
            # decision input.
            cost_basis[t] = ((prev_sh * cost_basis[t] + notional) / shares[t]
                             if prev_sh > 0 else p)
            # trailing-stop peak: a fresh position starts at its entry price;
            # an add to an existing one keeps the higher peak.
            peak_px[t] = max(peak_px.get(t, p), p)
            month_buys += notional
            trades_log.append((d.strftime("%Y-%m-%d"), side, t, notional, sh, p))
            return round(notional, 2)
        sh = shares.get(t, 0.0)
        if sh <= 0:
            return 0.0
        # The zero-fee sell liquidates EVERY positive supplied share as-is
        # (no re-flooring, no rounding — the position owns the quantity) and
        # is ``factor_fills.sell_fill``; the one-way ``cost`` is applied here,
        # around the primitive, exactly as before. At cost=0
        # ``notional * 1.0`` is exactly ``notional``, so the cash update is
        # bit-identical to the old expression and to ``fill.cash_after``.
        fill = factor_fills.sell_fill(cash, sh, p)
        if fill is None:
            return 0.0
        notional = fill.notional
        cash = cash + fill.notional * (1.0 - cost)
        del shares[t]
        peak_px.pop(t, None)
        cost_basis.pop(t, None)
        month_sells += notional
        trades_log.append((d.strftime("%Y-%m-%d"), side, t, notional, sh, p))
        return round(notional, 2)

    # Ranking memo: the eligibility frame is per-month, so the qv-mom
    # order is identical for every trading day within that month —
    # recomputing it per day (a ~4ms _ttm_as_of walk per eligible
    # ticker) dominated the profile. Keyed by the frame object id since
    # distinct months have distinct frames.
    order_memo: dict[int, tuple[list[str], pd.DataFrame | None]] = {}

    def current_order(d: pd.Timestamp, month_td: pd.Timestamp | None) -> tuple[list[str], pd.DataFrame | None]:
        frame = frame_cache.get(month_td) if month_td is not None else None
        if frame is None or not frame["eligible"].any():
            return [], frame
        key = id(frame)
        hit = order_memo.get(key)
        if hit is not None:
            return hit
        _elig, order = monthly_mod._qv_order(frame)
        result = (order, frame)
        order_memo[key] = result
        return result

    def month_for(s_prev: pd.Timestamp | None) -> pd.Timestamp | None:
        # The most recent COMPLETED month-end at or before S (the previous
        # stored session). Before the first window month-end this is the
        # PRE-WINDOW month-end frame (preloaded above); when even that does
        # not exist (the window starts at the first stored candles) it is
        # None — no ranking, no trades that day. It is NEVER months[0]:
        # substituting the first window month-end for days before it leaked
        # that future frame's ranking backwards (audit P0-E: a synthetic run
        # bought on 2025-01-01 on a 2025-01-31 ranking). Deferring by one
        # session is the contract's decision rule (data <= S), and it makes
        # the band switch — and therefore the month-end rebuild — happen on
        # the FIRST session after the month-end.
        if s_prev is None:
            return prior_month_end
        m = [m for m in months if m <= s_prev]
        return m[-1] if m else prior_month_end

    pending_cash = 0.0  # contributions not yet deployed (dca="monthly")
    # One contribution per CALENDAR month present in the replay window —
    # the live engine deposits monthly regardless of eligibility or of
    # whether the month's scheduled month-end candle is stored. Keying on
    # rebalance month-ends alone made an INTERIOR month with normal daily
    # bars but a missing month-end candle lose its funding entirely. The
    # monthly baseline settles the same calendar-month sequence, so the
    # two backtests' contributed counts stay equal.
    contrib_keys = {(d.year, d.month) for d in days_all}
    contrib_days = _contribution_days(days_all, contrib_keys)

    # --- Risk overlays (opt-in; measured via the daily-core sweep) ---
    # vol_weight: per-name target weights proportional to 1/realized vol
    # (the frame's `vol` column, annualized) instead of 1/N. Held names not
    # in the current frame keep equal weight (can't compute what we don't
    # measure — never silently zero a position).
    if vol_weight is None:
        vol_weight = settings.sim_daily_core_vol_weight
    # target_vol: Barroso-Santa-Clara vol management. Track the portfolio's
    # own daily returns; when its 21d realized vol exceeds the target, cap
    # deployment so (realized/target-1) of equity stays in cash. Only caps
    # DOWN — a calm portfolio never leverages up past fully invested.
    if target_vol is None:
        target_vol = settings.sim_daily_core_target_vol
    # portfolio_stop_pct (0 = off): peak-to-trough circuit breaker — when the
    # strategy's own equity is this % below its running peak, CASH OUT (sell
    # every holding) and park contributions until the market trend recovers.
    # This is the "don't give the win back" brake: the qv-mom design has no
    # exits except the monthly hysteresis band, so a momentum crash round-trips
    # the run-up (e.g. Jun→Jul 2026: peak 137% → 112%).
    if portfolio_stop_pct is None:
        portfolio_stop_pct = settings.sim_daily_core_portfolio_stop
    # exposure_trend_days (0 = off): deploy cash only while the equal-weight
    # universe index trades above its N-day SMA. Park contributions in cash in
    # a downtrend (Faber-style time-series momentum on the deployment stream);
    # also serves as the RE-ENTRY gate after a portfolio-stop trigger, since a
    # fully-cashed portfolio's own drawdown is frozen and can't recover.
    if exposure_trend_days is None:
        exposure_trend_days = settings.sim_daily_core_exposure_trend
    # trailing_stop_pct (0 = off): per-name trailing stop — exit a holding
    # when its price falls this % below its own peak since entry. This is the
    # factor-level protection the portfolio/market brakes can't provide: a
    # momentum-SLEEVE crash while the broad market holds (Jul 2026). The name
    # can be re-bought later if the ranking brings it back.
    if trailing_stop_pct is None:
        trailing_stop_pct = settings.sim_daily_core_trailing_stop
    # thesis_break_exit (M5): the fundamental thesis-break exit — see the
    # docstring for the predeclared condition. Default OFF via
    # settings.sim_daily_core_thesis_break; the no-arg path is bit-identical.
    if thesis_break_exit is None:
        thesis_break_exit = settings.sim_daily_core_thesis_break
    # basket_trend_days (0 = off): the "gradient filter" — the N-day
    # rate-of-change (the slope) of the strategy's OWN target basket
    # (equal-weight top-N candidates). When it is negative for
    # `basket_confirm_days` consecutive days, cash out and park
    # contributions; re-enter when it is positive for the same streak.
    # Unlike the market-trend gate this sees a momentum-SLEEVE reversal
    # (the basket's own trend), and unlike a frozen equity curve the
    # basket index keeps moving while in cash so re-entry can trigger.
    if basket_trend_days is None:
        basket_trend_days = settings.sim_daily_core_basket_trend
    if basket_confirm_days is None:
        basket_confirm_days = settings.sim_daily_core_basket_confirm
    if basket_threshold is None:
        basket_threshold = settings.sim_daily_core_basket_threshold
    # basket_drawdown (0 = off): cash out when the SIGNAL BASKET is this far
    # below its running peak; re-enter when the drawdown halves. Unlike the
    # portfolio-equity brake, the basket keeps moving while in cash, so the
    # re-entry rule can actually fire; the halving gives hysteresis so a
    # single real drawdown is one event, not a slope-driven whipsaw.
    if basket_drawdown is None:
        basket_drawdown = settings.sim_daily_core_basket_drawdown
    # basket_er_min (0 = off): Kaufman efficiency ratio gate. ER = |net move|
    # over N days / sum of |daily moves| (1 = perfect trend, ~0 = pure chop).
    # When set, the cash-out triggers are armed ONLY while ER >= the minimum:
    # trend-following signals pay in efficient moves and whipsaw in chop.
    if basket_er_min is None:
        basket_er_min = settings.sim_daily_core_basket_er_min
    # basket_good_times: arm the gradient cash-out only while the market is
    # above its 200-day SMA ("gradient filter only in good times"). The
    # market-trend gate itself handles bad times; this is the bull-market
    # sleeve-crash catcher.
    if basket_good_times is None:
        basket_good_times = settings.sim_daily_core_basket_good_times
    if basket_arm_sma is None:
        basket_arm_sma = settings.sim_daily_core_basket_arm_sma

    # Shared synthetic equal-weight return index (app/market_index) and its
    # SMA. NOT close.ffill().mean(axis=1): a nominal-price average weights a
    # $1,000 share 100x a $10 share, so units/splits moved the gate without an
    # economic return (audit P1-A). The same series arms the gradient filter,
    # the deployment trend gate and the portfolio-stop re-entry test.
    # Point-in-time safe: the chain at d uses rows <= d only.
    mkt_index = market_index_mod.equal_weight_return_index(close)
    mkt_sma = (mkt_index.rolling(exposure_trend_days).mean()
               if exposure_trend_days and exposure_trend_days > 0 else None)
    # Market SMA used to ARM the gradient filter in "good times"
    # (independent of the deployment gate's window; window configurable).
    mkt_sma_arm = mkt_index.rolling(max(basket_arm_sma, 2)).mean()

    def market_above_arm_sma(d: pd.Timestamp | None) -> bool:
        """True while the market index is at/above its 200-day SMA (the
        good-times arming condition). Warmup/missing -> True (armed)."""
        if d is None:
            return True
        sma = mkt_sma_arm.get(d)
        idx = mkt_index.get(d)
        if sma is None or idx is None or not np.isfinite(sma) or not np.isfinite(idx):
            return True
        return bool(float(idx) >= float(sma))

    def market_uptrend(d: pd.Timestamp | None) -> bool:
        """True when the universe index is at/above its SMA (always True when
        the filter is disabled)."""
        if d is None or mkt_sma is None:
            return True
        sma = mkt_sma.get(d)
        idx = mkt_index.get(d)
        if sma is None or idx is None or not np.isfinite(sma) or not np.isfinite(idx):
            return True  # warmup / missing -> don't block
        return bool(float(idx) >= float(sma))

    # Distinct re-entry signal for the portfolio-stop brake. It cannot reuse
    # `market_uptrend`: with the exposure gate disabled that returns True every
    # day, so the brake would cash out and re-buy on alternating days (measured:
    # 12 whipsaw events vs 1 with a trend window). Fall back to SMA200 (the
    # daily-core market gate) when no exposure window is configured.
    _reentry_days = (exposure_trend_days
                     if exposure_trend_days and exposure_trend_days > 0 else 200)
    _mkt_sma_reentry = mkt_index.rolling(_reentry_days).mean()

    def market_recovered(d: pd.Timestamp | None) -> bool:
        """True once the market index is back at/above its re-entry SMA."""
        if d is None:
            return True
        sma = _mkt_sma_reentry.get(d)
        idx = mkt_index.get(d)
        if sma is None or idx is None or not np.isfinite(sma) or not np.isfinite(idx):
            return True  # warmup / missing -> allow re-entry
        return bool(float(idx) >= float(sma))

    port_days: list[pd.Timestamp] = []
    port_eq: list[float] = []
    port_invested: list[float] = []  # cumulative contributions per day (for normalized DD)
    port_rets: list[float] = []
    peak_eq = 0.0        # running peak of the portfolio's own equity
    risk_off = False     # portfolio-stop brake is active (all cash)
    brake_events = 0     # how many times the brake fired (for the report)
    # gradient filter ("basket trend") state: chained equal-weight index of
    # the current month's top-N target basket, its N-day ROC, and the
    # consecutive-day streaks that arm the cash-out / re-entry.
    basket_hist: list[float] = []
    basket_out = False
    basket_events = 0
    neg_streak = 0
    pos_streak = 0
    basket_peak = 0.0
    prev_day: pd.Timestamp | None = None

    def _record_equity(d: pd.Timestamp, equity: float, contrib_today: float) -> None:
        """Append one day of equity; maintain the matching log-return list
        (contribution-adjusted) that `realized_vol` reads, plus the running
        cumulative-contribution series that the unitized-NAV drawdown
        (app.performance) consumes."""
        nonlocal peak_eq
        port_days.append(d)
        port_eq.append(equity)
        prev_inv = port_invested[-1] if port_invested else 0.0
        port_invested.append(prev_inv + contrib_today)
        peak_eq = max(peak_eq, equity)
        if len(port_eq) < 2:
            port_rets.append(0.0)
            return
        base = port_eq[-2] + contrib_today
        port_rets.append(float(np.log(equity / base)) if base > 0 and equity > 0 else 0.0)

    def realized_vol(days: int = 21) -> float | None:
        if len(port_rets) < days:
            return None
        recent = port_rets[-days:]
        return float(np.std(recent) * np.sqrt(252.0))

    # Per-day inverse-vol weight map, built from the day's order — see
    # _inv_vol_weights. Empty dict when vol_weight is off.
    vw_map: dict[str, float] = {}

    def weight_of(t: str, default_w: float, frame: pd.DataFrame | None) -> float:
        """Target weight for ticker t: the day's inverse-vol weight when
        vol_weight is on (the map covers the deployment candidates), else
        the equal weight."""
        if not vol_weight or not vw_map:
            return default_w
        return vw_map.get(t, default_w)

    for i, d in enumerate(days_all):
        li = locs[i]
        # S = the stored session BEFORE F (contract rule 1): the decision
        # row. Every overlay below reads S, never F's own bar.
        s_prev = full_idx[li - 1] if li > 0 else None
        # The first session AFTER a completed month-end: the DEFERRED rebuild
        # day. The band (and the parked "monthly" pool) switch on S's
        # completed month-end, exactly like the live engine's stored decision
        # row and daily_core.backfill's `S` lookup.
        rebuild = s_prev is not None and s_prev in month_set
        month_td = month_for(s_prev)
        order, frame = current_order(d, month_td)

        # --- monthly turnover bookkeeping ---
        if cur_month is None or d.month != cur_month:
            if cur_month is not None and equity_month_start > 0:
                # One-way turnover = ((buys - contribution) + sells) / 2 /
                # equity at month start. Deploying the fresh contribution is
                # not churn (same convention as the monthly baseline, where
                # slot swaps exclude it); buys beyond the contribution are.
                # Buys can exceed the contribution within a month (sell
                # proceeds recycled into rebalancing) — the excess is churn.
                # A fixed-capital lump deployed in the window's first month is
                # a fresh contribution there, not churn (same exclusion rule).
                excl = fixed if (fixed > 0 and not first_month_closed) else contribution
                churns.append(((max(month_buys - excl, 0.0)
                                + month_sells) / 2.0) / equity_month_start)
                first_month_closed = True
            month_buys, month_sells = 0.0, 0.0
            cur_month = d.month
            equity_month_start = cash + pending_cash + sum(
                (shares.get(t, 0.0) or 0.0) * (px_of(t, d) or 0.0) for t in shares)

        # --- contribution ---
        # Received on the first trading day of each month (same timing as the
        # live engine and daily_core.backfill). "monthly" parks it in
        # pending_cash until the month-end rebuild; "daily"/"rank" leave it in
        # cash so the daily deployment puts it to work immediately.
        if fixed > 0:
            # Single-shot funding: one lump sum at the window's first session
            # (dca="monthly" parks it until the first rebuild, like every
            # other contribution); monthly contributions stay suppressed.
            if d == days_all[0]:
                contributed += fixed
                n_contribs += 1
                if dca == "monthly":
                    pending_cash += fixed
                else:
                    cash += fixed
        elif d in contrib_days:
            contributed += contribution
            n_contribs += 1
            if dca == "monthly":
                pending_cash += contribution
            else:
                cash += contribution
        # month-end rebuild releases the parked "monthly" pool (deferred to
        # the first session after the month-end, like every other rebalance
        # leg here)
        if rebuild and pending_cash:
            cash += pending_cash
            pending_cash = 0.0

        # --- Portfolio-stop brake (peak-to-trough cash-out) ---
        # Measure the drawdown on S's marks (the stop is a decision known at
        # S's close) and stand the cash-out order for F at F's strict opens.
        # Re-engage only on a market uptrend: once fully cashed the
        # portfolio's own drawdown is frozen, so a drawdown-recovery rule
        # could never fire.
        if portfolio_stop_pct > 0 and s_prev is not None:
            eq_mark = cash + pending_cash + sum(
                (shares.get(t, 0.0) or 0.0) * (px_of(t, s_prev) or 0.0)
                for t in shares)
            dd = (peak_eq - eq_mark) / peak_eq if peak_eq > 0 else 0.0
            if dd >= portfolio_stop_pct:
                if not risk_off:
                    risk_off = True
                    brake_events += 1
                # Retried every session (a leg without an F bar keeps the
                # standing order) until the book is cash.
                for t in list(shares):
                    trade(d, t, "SELL", 0.0)
                    pending_pick.pop(t, None)
            elif risk_off and market_recovered(s_prev):
                risk_off = False

        # --- Gradient filter cash-out: when the basket trend confirmed
        # negative, exit the book (the flip was set at S's close, i.e. the
        # previous iteration's end-of-day pass).
        if (basket_trend_days > 0 or basket_drawdown > 0) and basket_out and shares:
            for t in list(shares):
                trade(d, t, "SELL", 0.0)
                pending_pick.pop(t, None)

        # --- SELL phase: releases from the hysteresis band ---
        if order:
            band = set(order[:hold_band])
            if exit_cadence == "daily" or rebuild:
                for t in list(shares):
                    if t not in band:
                        trade(d, t, "SELL", 0.0)
                        pending_pick.pop(t, None)

        # --- Per-name trailing stop: a STANDING ORDER struck at S's close
        # (level = peak since entry - X%) and filled at F's open when the
        # open is already through it. The loader carries no low/high frame,
        # so an intrabar breach that closes back above the level is not
        # modelled — it waits for the next session's open (documented in the
        # docstring). The peak advances at the END of the session on F's
        # close, so the level that governs F is the one known at S.
        if trailing_stop_pct > 0:
            for t in list(shares):
                level = peak_px.get(t, 0.0) * (1.0 - trailing_stop_pct)
                if level <= 0:
                    continue
                op = fill_px_of(t, d)
                if op is not None and op <= level:
                    trade(d, t, "SELL", 0.0)
                    pending_pick.pop(t, None)

        # --- Thesis-break exit (M5): a STANDING ORDER struck at S — the
        # fundamentals PUBLIC as of S (filing-date PIT, the engine's own
        # _latest_as_of/_ttm_as_of helpers) — filled at F's open, before the
        # BUY phase, same cost path as every other sell. The predeclared
        # condition lives in ``_thesis_break_fundamentals`` (fundamental-only:
        # record missing / equity <= 0 / TTM NI missing / shares missing /
        # ROE <= 0; F/P-FCF changes alone do NOT trigger; price-side gates are
        # not re-checked intraday). A failed fill (no F bar) keeps the standing
        # order: the name stays held and the break re-fires next session.
        # Re-entry is unchanged — the ranking can bring the name back.
        if thesis_break_exit and s_prev is not None:
            for t in list(shares):
                if _thesis_break_fundamentals(fund.get(t), s_prev):
                    if trade(d, t, "SELL", 0.0) > 0:
                        thesis_break_log.append({
                            "date": d.strftime("%Y-%m-%d"), "ticker": t,
                            "decision_date": (s_prev.strftime("%Y-%m-%d")
                                              if s_prev is not None else None),
                        })
                    pending_pick.pop(t, None)

        # --- BUY phase: keep the portfolio at target weights ---
        # SIZING equity: marked at S's close (session fill contract rule 1 —
        # the budget is struck at F's open, so F's own close cannot size it),
        # summed ticker-ordered and rounded once. This is deliberately NOT
        # the end-of-session reporting equity below (``equity_close``), which
        # stays on F's closes after the fills: one number decides the order,
        # the other reports the day.
        positions_value = sum(
            (shares.get(t, 0.0) or 0.0) * (sizing_px_of(t, s_prev) or 0.0)
            for t in sorted(shares))
        sizing_equity = round(cash + pending_cash + positions_value, 2)
        weight = sizing_equity / max(target_n, 1)

        # The hysteresis-selected deployment targets (audit P0-D). On a
        # rebuild day the releases above freed slots and the picks refill
        # them from the top of the ranking; below target capacity the same
        # fill applies. Otherwise — a non-rebuild day under the deferred
        # "monthly" exit cadence — held names keep their slots and NO new
        # name is bought while an out-of-band release is still pending.
        # Deploying the raw order[:target_n] instead bought names the picks
        # never chose and grew the book past the ten-slot target. The
        # capacity test decides whether fills are PLANNED; when a FAILED
        # release leaves the book at/over capacity while `refill` is True,
        # the executor backstops (this replay's trade() and BUY blocks)
        # enforce the ACTUAL free capacity — the ten-slot guarantee
        # survives incomplete state.
        if order:
            picks: list[str] = monthly_mod._deploy_targets(
                order, list(shares), target_n, hold_band,
                refill=len(shares) < target_n)
        else:
            picks = []

        # Inverse-vol weights among the day's deployment candidates (the
        # hysteresis picks). Sums to the same total as equal weight.
        vw_map = (_inv_vol_weights(picks, frame, target_n, weight)
                  if vol_weight and picks else {})

        paused = ((crash_pause and pullback_day(s_prev)) or risk_off
                  or basket_out
                  or (exposure_trend_days > 0 and not market_uptrend(s_prev)))

        # Barroso-Santa-Clara vol management: cap deployment when the
        # portfolio's own recent vol exceeds the target. Only scales DOWN
        # (a calm portfolio stays fully invested); the excess sits in cash
        # until vol normalises. Both sides of the cap use the SIZING equity
        # (S's marks) and `port_rets` through S only — `_record_equity` for
        # F runs after the fills, so F's just-closed return can never size
        # F's own order.
        room = float("inf")
        if target_vol and len(port_rets) >= 21:
            rv = realized_vol(21)
            if rv is not None and rv > target_vol:
                deploy_cap = sizing_equity * (target_vol / rv)
                deployed_now = sizing_equity - cash - pending_cash
                room = max(deploy_cap - deployed_now, 0.0)

        # --- Rank-weighted cash deployment (dca="rank") ---
        # Fresh cash goes where the qv-mom ranking says it belongs: each day,
        # deploy (a slice of) available cash into the HYSTERESIS picks with
        # target weights ABOVE their current value — i.e. new cash reinforces
        # the strongest core names instead of equal-weight rebalancing into
        # laggards. A held name at/above target gets nothing; a not-yet-held
        # pick (a slot that freed) is bought to its target. This is the
        # momentum-preserving variant of "deploy daily": the monthly sim's
        # edge is concentrating fresh capital into the top of the ranking —
        # rank deployment does that every day without the ~2-week cash drag.
        if dca == "rank" and picks and not paused:
            rank_of = {t: i for i, t in enumerate(order)}
            for t in picks:
                rank_i = rank_of.get(t, target_n - 1)
                if room < 1:
                    break
                # rule 2: no strict-open bar this session -> no fill; the
                # daily pass simply re-takes the decision next session.
                if fill_px_of(t, d) is None:
                    continue
                # score-weighted target: rank 1 gets up to `weight` * boost,
                # linearly decaying to the equal weight at the band edge.
                boost = 1.0 + settings.sim_monthly_rank_boost * (
                    1.0 - rank_i / max(target_n - 1, 1))
                tgt = weight_of(t, weight, frame) * boost
                # `cur` is marked at S's close and rounded to cents: the
                # budget is placed at F's open, so F's own close must not
                # move the gap. (daily_core.backfill sizes with the same S
                # mark since the sizing-clock fix — the golden parity in
                # tests/test_replay_fills.py compares the two.)
                cur = round((shares.get(t, 0.0) or 0.0)
                            * (sizing_px_of(t, s_prev) or 0.0), 2)
                gap = tgt - cur
                if t not in shares and len(shares) >= target_n:
                    continue  # no free slot: a failed release occupies it
                # NEW names respect the entry gate (it used to be inert under
                # rank deploy — the gate knob silently did nothing in
                # rank-mode sweeps). A gated fill waits in pending_pick for
                # the daily re-check below; held-name top-ups are never gated.
                if t not in shares and entry_gate != "none" \
                        and not entry_ok(t, s_prev):
                    pending_pick[t] = d
                    continue
                # Budget mirrors backfill: the gap, the ROUNDED cash and the
                # vol-cap room, all in dollars; < $1 is not worth a fill.
                budget = min(gap, round(cash, 2), room)
                if budget < 1:
                    continue
                room -= trade(d, t, "BUY", budget)

        # Month-end equal-weight rebuild (dca="monthly" only): the baseline
        # semantics — spread the FULL portfolio (incl. fresh contribution)
        # 1/N across the picks. Implemented as: buy each underweight pick up
        # to weight (releasing cash pro-rata), sell nothing here (releases
        # happened in the SELL phase). This makes all-defaults ≈ baseline.
        if dca == "monthly" and rebuild and order and not paused:
            for t in picks:
                if t not in shares and len(shares) >= target_n:
                    continue  # no free slot: a failed release occupies it
                if fill_px_of(t, d) is None:
                    continue
                # pro-rata gap: current value at S's close (the sizing clock),
                # never F's own close — the fills happen at F's open.
                cur = round((shares.get(t, 0.0) or 0.0)
                            * (sizing_px_of(t, s_prev) or 0.0), 2)
                gap = weight_of(t, weight, frame) - cur
                budget = min(gap, round(cash, 2), room)
                if budget < 1:
                    continue
                room -= trade(d, t, "BUY", budget)

        # daily top-up of underweight holdings (dca="daily" only; rank mode
        # and month-end days already allocated). NOT pro-rata: the fresh
        # cash goes to the single most underweight holding — equal-weight
        # rebalancing into laggards dilutes momentum and was measured worse
        # than the monthly baseline on 2020-2026. Candidates are the
        # hysteresis picks (held names the band retained): an out-of-band
        # release waiting for the month-end is not topped up, and no new
        # name is opened here.
        if dca == "daily" and picks and not paused and not rebuild:
            worst_t, worst_gap = None, 0.0
            for t in picks:
                if t not in shares:
                    continue
                if fill_px_of(t, d) is None:
                    continue
                # current value at S's close (the sizing clock): the top-up is
                # spent at F's open, so F's close cannot move the gap.
                gap = weight_of(t, weight, frame) - round(
                    (shares.get(t, 0.0) or 0.0)
                    * (sizing_px_of(t, s_prev) or 0.0), 2)
                if gap > worst_gap:
                    worst_t, worst_gap = t, gap
            if worst_t is not None and worst_gap > 1.0:
                trade(d, worst_t, "BUY", min(worst_gap, round(cash, 2), room))

        # --- NEW entries (core names not yet held) ---
        # Month-end cadences only: rank deployment opens its new names in the
        # daily pass above (it targets every hysteresis pick, held or not), so
        # re-running this block would double-deploy on the rebuild session.
        if rebuild and dca != "rank" and order:
            for t in picks:
                if t in shares or paused:
                    continue
                if len(shares) >= target_n:
                    continue  # no free slot: a failed release occupies it
                if fill_px_of(t, d) is None:
                    continue
                if entry_gate != "none" and not entry_ok(t, s_prev):
                    pending_pick[t] = d  # wait for the gate to open
                    continue
                budget = min(weight_of(t, weight, frame), round(cash, 2), room)
                if budget > 1:
                    trade(d, t, "BUY", budget)
                    pending_pick.pop(t, None)
                    room -= budget
        elif entry_gate != "none" and order:
            # daily re-check of gated entries once the gate opens. The fill
            # must be in the day's hysteresis-selected deployment targets
            # (``picks`` — capacity-aware by construction: a refill only
            # happens while the book is below target) and the book must
            # actually be below capacity: buying a pending name outside the
            # selected set, or while a failed release still occupies the
            # slot, grew the book past the ten-slot target. The gate itself
            # is a decision input, so it is re-read at S (data <= S).
            for t in list(pending_pick):
                if t in shares or paused:
                    continue
                if t not in band:
                    pending_pick.pop(t, None)
                    continue
                if t not in picks or len(shares) >= target_n:
                    continue
                if fill_px_of(t, d) is None:
                    continue
                if entry_ok(t, s_prev):
                    budget = min(weight_of(t, weight, frame), round(cash, 2), room)
                    if budget > 1:
                        room -= trade(d, t, "BUY", budget)
                    pending_pick.pop(t, None)

        # Track the portfolio's own daily equity for the vol gate. The 21d
        # realized vol is computed from log returns of *invested* equity;
        # contribution days subtract the deposit from the base so a fresh
        # $1000 doesn't read as a +x% return.
        equity_close = round(cash + pending_cash + sum(
            (shares.get(t, 0.0) or 0.0) * (eod_px_of(t, d) or 0.0)
            for t in sorted(shares)), 2)
        contrib_today = (fixed if (fixed > 0 and d == days_all[0])
                         else (contribution if d in contrib_days else 0.0))
        _record_equity(d, equity_close, contrib_today)

        # Contract rule 1/4: F's close only advances the trailing-stop peaks
        # for the NEXT session — the level that governs F is the one known
        # at S's close.
        if trailing_stop_pct > 0:
            for t in list(shares):
                p = px_of(t, d)
                if p is not None and p > 0:
                    peak_px[t] = max(peak_px.get(t, 0.0), p)

        # --- Gradient filter: the N-day slope of the target basket ---
        # Chain the equal-weight top-N basket from yesterday to today (one
        # day's cross-sectional mean return). The day-over-day log return is
        # the gradient's building block; the N-day ROC over `basket_hist` is
        # the slope. Cash out when the slope is below -`basket_threshold` (a
        # real drawdown, not noise) for the confirm streak; re-enter on any
        # positive slope for the same streak. The threshold is the "on
        # demand" switch: calm-market dips no longer trigger.
        if basket_trend_days > 0 or basket_drawdown > 0:
            top = order[:target_n]
            rets = []
            for t in top:
                p0 = px_of(t, prev_day) if prev_day is not None else None
                p1 = px_of(t, d)
                if p0 and p1 and p0 > 0 and p1 > 0:
                    rets.append(float(np.log(p1 / p0)))
            day_ret = float(np.mean(rets)) if rets else 0.0
            if not basket_hist:
                basket_hist.append(100.0)
            basket_hist.append(basket_hist[-1] * float(np.exp(day_ret)))
            basket_peak = max(basket_peak, basket_hist[-1])
            # Efficiency ratio over the last 20 basket prints: |net| / path.
            er = 1.0
            er_win = 20
            if basket_er_min > 0 and len(basket_hist) > er_win:
                seg = basket_hist[-er_win:]
                path = sum(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg)))
                er = (abs(seg[-1] - seg[0]) / path) if path > 0 else 1.0
            armed = (basket_er_min <= 0) or (er >= basket_er_min)
            if basket_good_times:
                armed = armed and market_above_arm_sma(d)
            # (a) slope trigger (gradient filter). The ER gate arms the EXIT
            # only: re-entry stays unconditional so chop can never lock the
            # portfolio in cash forever.
            if basket_trend_days > 0 and len(basket_hist) > basket_trend_days:
                slope = basket_hist[-1] / basket_hist[-1 - basket_trend_days] - 1.0
                if slope < -basket_threshold:
                    neg_streak += 1
                    pos_streak = 0
                elif slope > 0:
                    pos_streak += 1
                    neg_streak = 0
                if not basket_out and armed and neg_streak >= basket_confirm_days:
                    basket_out = True
                    basket_events += 1
                elif basket_out and pos_streak >= basket_confirm_days:
                    basket_out = False
            # (b) drawdown trigger (on-demand, hysteresis via halving)
            if basket_drawdown > 0 and basket_peak > 0:
                bdd = (basket_peak - basket_hist[-1]) / basket_peak
                if not basket_out and armed and bdd >= basket_drawdown:
                    basket_out = True
                    basket_events += 1
                elif basket_out and bdd <= basket_drawdown / 2.0:
                    basket_out = False
        prev_day = d

    # final valuation (EOD mark policy: a stale print is not a mark — the
    # position reports at its cost basis, like every other EOD mark)
    value = cash + pending_cash + sum(
        (shares.get(t, 0.0) or 0.0) * (eod_px_of(t, last_day) or 0.0) for t in shares)

    def irr_of(v: float, n: int) -> float:
        def f(r: float) -> float:
            return contribution * sum((1.0 + r) ** k for k in range(n)) - v
        lo, hi = -0.90, 2.0
        if f(lo) > 0:
            return -1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return (1.0 + (lo + hi) / 2.0) ** 12 - 1.0

    if fixed > 0:
        # Single-flow annualized return over the window's calendar months
        # (irr_of assumes n_contribs equal monthly flows, which a lump is not).
        n_months = len(contrib_keys)
        irr = ((value / fixed) ** (12.0 / max(n_months, 1)) - 1.0) if value > 0 else -1.0
    else:
        irr = irr_of(value, n_contribs)
    cfg = (f"dca={dca} entry-gate={entry_gate} exit-cadence={exit_cadence} "
           f"crash-pause={'on' if crash_pause else 'off'} "
           f"cost={cost * 1e4:.0f}bps-oneway(research-only)"
           + (f" rank-boost={settings.sim_monthly_rank_boost:g}"
              if dca == "rank" else "")
           + (f" mom={settings.sim_daily_core_mom_variant}"
              if settings.sim_daily_core_mom_variant != "raw" else "")
           + (" vol-weight" if vol_weight else "")
           + (f" target-vol={target_vol:g}" if target_vol else "")
           + (f" portfolio-stop={portfolio_stop_pct:g}" if portfolio_stop_pct else "")
           + (f" exposure-trend={exposure_trend_days}" if exposure_trend_days else "")
           + (f" trailing-stop={trailing_stop_pct:g}" if trailing_stop_pct else "")
           + (" thesis-break=on" if thesis_break_exit else "")
           + (f" basket-trend={basket_trend_days}/{basket_confirm_days}"
              + (f" threshold={basket_threshold:g}" if basket_threshold else "")
              if basket_trend_days else "")
           + (f" basket-drawdown={basket_drawdown:g}" if basket_drawdown else "")
           + (f" basket-er-min={basket_er_min:g}"
              if basket_er_min else "")
           + (f" basket-good-times(sma{basket_arm_sma})"
              if basket_good_times else "")
           + (" lowvol-tilt" if settings.sim_daily_core_lowvol_tilt else ""))
    ann_sharpe = 0.0
    if len(port_rets) >= 21:
        mu = float(np.mean(port_rets))
        sd = float(np.std(port_rets))
        ann_sharpe = (mu / sd) * np.sqrt(252.0) if sd > 0 else 0.0
    # Max drawdown on the contribution-adjusted unitized NAV
    # (performance.max_drawdown's cash-flow-neutral definition, shared with
    # the live UI): a deposit that leaves the portfolio value unchanged is
    # not a loss. Raw-equity DD reads deposits as gains and quietly
    # understates the real peak-to-trough loss of a DCA portfolio.
    mdd = _max_drawdown(port_eq, port_invested) / 100.0
    print(f"\n=== Daily-core backtest [{universe}] {days_all[0].date()}..{last_day.date()} ===")
    print(f"  config: {cfg}")
    if fixed > 0:
        print(f"  fixed capital: one lump sum ${fixed:,.0f} at the window's first session "
              f"(monthly contributions suppressed) = ${contributed:,.0f}")
    else:
        print(f"  contributions: {n_contribs} x ${contribution:,.0f} = ${contributed:,.0f}")
    print(f"  final value:   ${value:,.0f}  (multiple {value / max(contributed, 1e-9):.2f}x)")
    print(f"  money-weighted IRR: {irr:.2%}/yr   Sharpe (ann): {ann_sharpe:.2f}   "
          f"max DD: {mdd:.1%}   avg monthly turnover: "
          f"{(sum(churns) / len(churns)) if churns else 0:.1%}  trades: {len(trades_log)}"
          + (f"  brake events: {brake_events}" if portfolio_stop_pct else "")
          + (f"  basket events: {basket_events}"
             if (basket_trend_days or basket_drawdown) else "")
          + (f"  thesis-break exits: {len(thesis_break_log)}"
             if thesis_break_exit else ""))
    if verbose:
        print("\nLast 15 trades:")
        for td, side, t, notional, _sh, _px in trades_log[-15:]:
            print(f"  {td} {side:<4} {t:<10} ${notional:,.0f}")

    # --- Monthly baseline over the same window (for comparison) ---
    # Skippable: the sweep computes the (config-independent) baseline once
    # per window instead of once per config.
    if with_baseline and months:
        # A short no-rebalance window has no monthly-baseline counterpart
        # (the baseline needs its own completed month-ends).
        print("\n--- Monthly qv-mom baseline over the same window ---")
        await _monthly_backtest(
            start or (days_all[0].strftime("%Y-%m-%d")),
            end or (last_day.strftime("%Y-%m-%d")),
            universe, contribution, False,
            frame_cache=frame_cache, close_preloaded=close_preloaded,
            fund_preloaded=fund_preloaded,
            open_preloaded=None if close_preloaded is None else open_preloaded,
            fixed_capital=fixed,
        )

    return BacktestSummary(
        irr=irr,
        final_value=value,
        contributed=contributed,
        avg_turnover=(sum(churns) / len(churns)) if churns else 0.0,
        n_trades=len(trades_log),
        window=(pd.Timestamp(days_all[0]).strftime("%Y-%m-%d"),
                pd.Timestamp(last_day).strftime("%Y-%m-%d")),
        sharpe=ann_sharpe,
        max_drawdown=mdd,
        brake_events=brake_events,
        basket_events=basket_events,
        trades=[{"date": td, "side": side, "ticker": t,
                 "notional": notional, "shares": sh, "price": px}
                for td, side, t, notional, sh, px in trades_log],
        n_thesis_break=len(thesis_break_log),
        thesis_break_exits=thesis_break_log,
    )


async def _main(args: argparse.Namespace) -> None:
    if args.command == "prune-candles":
        await _prune_candles(args.before, args.dry_run, args.allow_degrade)
        return

    if args.command == "monthly-backtest":
        await _monthly_backtest(
            args.start, args.end,
            args.universe or settings.sim_monthly_universe,
            args.contribution if args.contribution is not None else settings.sim_monthly_contribution,
            args.picks,
        )
        return

    if args.command == "daily-core":
        if getattr(args, "mom", None):
            settings.sim_daily_core_mom_variant = args.mom
        if getattr(args, "lowvol_tilt", False):
            settings.sim_daily_core_lowvol_tilt = True
        await _daily_core_backtest(
            args.start, args.end,
            args.universe or settings.sim_monthly_universe,
            args.contribution if args.contribution is not None else settings.sim_monthly_contribution,
            args.verbose,
            args.entry_gate, args.exit_cadence, args.crash_pause, args.dca,
            vol_weight=getattr(args, "vol_weight", False) or None,
            target_vol=getattr(args, "target_vol", None) or None,
            portfolio_stop_pct=getattr(args, "portfolio_stop", None) or None,
            exposure_trend_days=getattr(args, "exposure_trend", None) or None,
            trailing_stop_pct=getattr(args, "trailing_stop", None) or None,
            basket_trend_days=getattr(args, "basket_trend", None) or None,
            basket_confirm_days=getattr(args, "basket_confirm", None) or None,
            basket_threshold=getattr(args, "basket_threshold", None) or None,
            basket_drawdown=getattr(args, "basket_drawdown", None) or None,
            basket_er_min=getattr(args, "basket_er_min", None) or None,
            basket_good_times=getattr(args, "basket_good_times", False) or None,
            basket_arm_sma=getattr(args, "basket_arm_sma", None) or None,
            thesis_break_exit=getattr(args, "thesis_break", False) or None,
        )
        return

    if args.command == "daily-core-sweep":
        await _sweep_daily_core(args)
        return

    tickers = _candidate_tickers()
    logger.info("Loading series for %d tickers...", len(tickers))
    series = await _load_series(tickers)
    logger.info("Loaded series for %d tickers", len(series))

    if not series:
        print("No tickers with enough candle history found. Nothing to do.")
        return

    all_days = sorted(set().union(*(set(df["time"]) for df in series.values())))

    # Load the market regime series if the regime filter is enabled.
    regime: dict[str, bool] | None = None
    if getattr(args, "regime", False):
        regime_ticker = getattr(args, "regime_ticker", "URTH")
        logger.info("Loading regime series for %s...", regime_ticker)
        regime = await _load_regime(regime_ticker)
        logger.info("Loaded regime for %d days", len(regime))

    if args.command == "backtest":
        # Replay with the live sim's risk configuration (max-positions cap,
        # initial + ATR stops, run-up block, position/cash floors) so the
        # backtest reflects what the engine would actually have held — not a
        # vanilla unlimited-position replay. ``regime_filter`` is the one knob
        # the live sim doesn't set, so it stays opt-in via --regime.
        params = _live_sim_params()
        params.regime_filter = getattr(args, "regime", False)
        if getattr(args, "no_entry_guards", False):
            params.min_run_5d = 0.0
            params.max_dist_above = 0.0
        quality = None
        if getattr(args, "block_negative_roe", False):
            params.block_negative_roe = True
            quality = await _load_quality_lookup(universe_tickers(settings.sim_monthly_universe))
            n_with_fund = len(quality._fund) if quality else 0
            print(f"Quality guard ON: {n_with_fund} tickers with fundamentals")
        res = _replay(series, params, start=args.start, end=args.end, regime=regime, quality=quality)
        label = "Backtest (live sim risk config)"
        if params.regime_filter:
            label += " + regime filter"
        if getattr(args, "no_entry_guards", False):
            label += " [guards OFF]"
        if getattr(args, "block_negative_roe", False):
            label += " [ROE guard ON]"
        _print_result(res, label)
        print(f"  Risk config: max_positions={params.max_positions}, "
              f"stop={params.stop_type} {params.stop_pct:g}%, "
              f"atr_stop={params.use_atr_stop}, max_run_5d={params.max_run_5d:g}%, "
              f"min_run_5d={params.min_run_5d:g}%, max_dist_above={params.max_dist_above:g}%, "
              f"max_pos_pct={params.max_position_pct:g}%, "
              f"min_cash_pct={params.min_cash_pct:g}%")
        if args.trades:
            for t in res.trades:
                print(f"  {t['side']:<4} {t['ticker']:<8} {t['shares']:>10.4f} @ {t['price']:>10.2f} — {t['reason']}")
        if getattr(args, "cash", False):
            _print_cash_summary(res, params)

    elif args.command == "sweep":
        regime_on = getattr(args, "regime", False)
        sweep = _run_sweep(series, args.start, args.end, regime=regime, regime_filter=regime_on)
        print(f"\n=== Parameter sweep ({args.start}..{args.end}) — top 10 ===")
        for params, res in sweep[:10]:
            print(f"  buy={params.buy_threshold:<3} sell={params.sell_threshold:<4} "
                  f"trail={params.trailing_stop_pct:>4.0f}% maxpos={params.max_position_pct:<3}% "
                  f"stop={params.stop_type:<7} risk={params.risk_pct:.1f}% sec={params.max_sector_pct:>2.0f}% "
                  f"run5d={params.max_run_5d:>2.0f}% "
                  f"→ ret {res.total_return_pct:+7.2f}%  dd {res.max_drawdown_pct:5.2f}%  "
                  f"sharpe {res.sharpe:5.2f}  trades {res.n_trades}")

    elif args.command == "walkforward":
        # Bound the walk to the requested window (defaults to full history).
        walk_days = all_days
        if args.start:
            walk_days = [d for d in walk_days if d >= args.start]
        if args.end:
            walk_days = [d for d in walk_days if d <= args.end]
        windows = _walk_forward(series, walk_days, args.train_days, args.test_days,
                                regime=regime, regime_filter=getattr(args, "regime", False))
        print(f"\n=== Walk-forward (train {args.train_days}d / test {args.test_days}d) ===")
        for w in windows:
            print(f"  train {w['train']} → test {w['test']}: "
                  f"params {w['best_params']} | train {w['train_return_pct']:+6.2f}% (score {w['train_score']:+.1f}) | "
                  f"test {w['test_return_pct']:+6.2f}% (sharpe {w['test_sharpe']:.2f}, "
                  f"dd {w['test_max_dd_pct']:.2f}%, {w['test_n_trades']} trades)")
        if windows:
            oos = [w["test_return_pct"] for w in windows]
            print(f"\n  Out-of-sample test returns: {oos}")
            print(f"  Mean OOS return: {sum(oos)/len(oos):+.2f}%  "
                  f"Positive windows: {sum(1 for r in oos if r > 0)}/{len(oos)}")

    elif args.command == "llm-benchmark":
        await _run_llm_benchmark(
            series, start=args.start, end=args.end,
            n_worst=args.n_worst, n_control=args.n_control,
            forward_days=args.forward_days, skip_llm=args.skip_llm,
        )

    elif args.command == "hybrid-replay":
        # Replay the hybrid strategy (deterministic + per-day LLM review) over
        # the window, then run a pure-deterministic replay over the same
        # window for a side-by-side comparison. Both use _live_sim_params() so
        # the only difference is the LLM review phase.
        if not (settings.llm_backends or settings.ollama_model):
            print("No LLM backend configured (set LLM_BACKENDS or OLLAMA_MODEL); "
                  "hybrid-replay needs the LLM. Aborting.")
            return
        params = _live_sim_params()
        start, end = args.start, args.end
        if args.days:
            # Simulate only the last N trading days of the (possibly bounded)
            # window — a quick, natural way to run short pure-LLM comparisons.
            window_days = all_days
            if start:
                window_days = [d for d in window_days if d >= start]
            if end:
                window_days = [d for d in window_days if d <= end]
            if len(window_days) > args.days:
                start = window_days[-args.days]
                end = window_days[-1]
        pure_llm = getattr(args, "pure_llm", False)
        mode_label = "pure-LLM" if pure_llm else "hybrid"
        print(f"\n=== Pure-deterministic replay ({start}..{end}) ===")
        det = _replay(series, params, start=start, end=end)
        _print_result(det, "Deterministic baseline")
        print(f"  Risk config: max_positions={params.max_positions}, "
              f"stop={params.stop_type} {params.stop_pct:g}%, "
              f"atr_stop={params.use_atr_stop}, max_run_5d={params.max_run_5d:g}%, "
              f"max_pos_pct={params.max_position_pct:g}%, "
              f"min_cash_pct={params.min_cash_pct:g}%")

        active = await llm_mod.current_backend()
        review_interval = getattr(args, "review_interval", 1)
        veto_only = getattr(args, "veto_only", False)
        minimal_prompt = getattr(args, "minimal_prompt", False)
        mode_aware = getattr(args, "mode_aware_prompt", False)
        marker_gated = False
        failure_marker = getattr(args, "failure_marker", False)
        failure_stop_outs = getattr(args, "failure_stop_outs", 2)
        failure_drawdown = getattr(args, "failure_drawdown", 7.0)
        fundamentals_context = getattr(args, "fundamentals_context", False)
        block_negative_roe = getattr(args, "block_negative_roe", False)
        quality = None
        if fundamentals_context or block_negative_roe:
            params.block_negative_roe = block_negative_roe
            quality = await _load_quality_lookup(universe_tickers(settings.sim_monthly_universe))
        print(f"\n=== {mode_label} replay ({start}..{end}) — probing LLM "
              f"once per {review_interval} calendar week(s) ===")
        print(f"  Backend: {active.get('name', '?')} · {active.get('model', '?')}")
        hyb = await _hybrid_replay(series, params, start=start, end=end,
                                   pure_llm=pure_llm,
                                   review_interval=review_interval,
                                   veto_only=veto_only,
                                   minimal_prompt=minimal_prompt,
                                   mode_aware=mode_aware,
                                   marker_gated=marker_gated,
                                   failure_marker=failure_marker,
                                   failure_stop_outs=failure_stop_outs,
                                   failure_drawdown=failure_drawdown,
                                   quality=quality,
                                   fundamentals_context=fundamentals_context)
        _print_result(hyb, f"{mode_label} (deterministic + LLM review)" if not pure_llm else "Pure LLM")

        # Side-by-side comparison
        print(f"\n=== Comparison ({start}..{end}) ===")
        print(f"  {'':>20} {'deterministic':>14} {mode_label:>14} {'delta':>10}")
        print(f"  {'Return':>20} {det.total_return_pct:>+13.2f}% {hyb.total_return_pct:>+13.2f}% "
              f"{hyb.total_return_pct - det.total_return_pct:>+9.2f}%")
        print(f"  {'Max drawdown':>20} {det.max_drawdown_pct:>13.2f}% {hyb.max_drawdown_pct:>13.2f}% "
              f"{hyb.max_drawdown_pct - det.max_drawdown_pct:>+9.2f}%")
        print(f"  {'Sharpe':>20} {det.sharpe:>14.2f} {hyb.sharpe:>14.2f} "
              f"{hyb.sharpe - det.sharpe:>+10.2f}")
        print(f"  {'Trades':>20} {det.n_trades:>14} {hyb.n_trades:>14} "
              f"{hyb.n_trades - det.n_trades:>+10}")
        print(f"  {'Final equity':>20} {det.final_equity:>14.2f} {hyb.final_equity:>14.2f} "
              f"{hyb.final_equity - det.final_equity:>+10.2f}")

        if args.trades:
            print(f"\n  {mode_label} trades:")
            for t in hyb.trades:
                print(f"    {t['date']} {t['side']:<4} {t['ticker']:<8} "
                      f"{t['shares']:>9.4f} @ {t['price']:>10.2f} — {t['reason']}")

    elif args.command == "llm-walkforward":
        # Multi-window LLM vs deterministic benchmark. Runs N non-overlapping
        # windows, each with a deterministic _replay + an LLM _hybrid_replay,
        # then reports per-window + mean + worst-window delta. This is the
        # reliable scoreboard for evaluating pure-LLM / hybrid strategy
        # changes — a single window is noise.
        if not (settings.llm_backends or settings.ollama_model):
            print("No LLM backend configured (set LLM_BACKENDS or OLLAMA_MODEL); "
                  "llm-walkforward needs the LLM. Aborting.")
            return
        params = _live_sim_params()
        if getattr(args, "no_entry_guards", False):
            params.min_run_5d = 0.0
            params.max_dist_above = 0.0
        pure_llm = getattr(args, "pure_llm", False)
        n_windows = args.windows
        days_per_window = args.days_per_window
        review_interval = getattr(args, "review_interval", 1)
        veto_only = getattr(args, "veto_only", False)
        no_llm_sells = not getattr(args, "allow_llm_sells", False)
        minimal_prompt = getattr(args, "minimal_prompt", False)
        mode_aware = getattr(args, "mode_aware_prompt", False)
        llm_max_positions = getattr(args, "llm_max_positions", 0)
        marker_gated = getattr(args, "marker_gated", False)
        failure_marker = getattr(args, "failure_marker", False)
        failure_stop_outs = getattr(args, "failure_stop_outs", 2)
        failure_drawdown = getattr(args, "failure_drawdown", 7.0)
        # Fundamentals features (A/B flags): quality guard in the deterministic
        # engine, ROE/P-FCF columns in the LLM context. Both share the
        # monthly-granularity quality provider loaded once from the fact store.
        fundamentals_context = getattr(args, "fundamentals_context", False)
        block_negative_roe = getattr(args, "block_negative_roe", False)
        quality = None
        if fundamentals_context or block_negative_roe:
            params.block_negative_roe = block_negative_roe
            quality = await _load_quality_lookup(universe_tickers(settings.sim_monthly_universe))
            print(f"Quality data: {len(quality.month_map(all_days[-1]) if all_days else {})} tickers"
                  if quality else "Quality data: none (empty fundamentals store)")
        if pure_llm and (marker_gated or failure_marker):
            print("Cannot combine --pure-llm with --marker-gated / --failure-marker: "
                  "the markers key off the deterministic engine's proposals and "
                  "stop-outs, which do not exist in pure-LLM mode (the LLM would "
                  "never be consulted). Run pure-LLM without markers.")
            return
        try:
            results = await _llm_walkforward(
                series, params, all_days,
                n_windows=n_windows, days_per_window=days_per_window,
                pure_llm=pure_llm,
                start=args.start, end=args.end,
                review_interval=review_interval,
                veto_only=veto_only,
                no_llm_sells=no_llm_sells,
                minimal_prompt=minimal_prompt,
                mode_aware=mode_aware,
                llm_max_positions=llm_max_positions,
                marker_gated=marker_gated,
                failure_marker=failure_marker,
                failure_stop_outs=failure_stop_outs,
                failure_drawdown=failure_drawdown,
                quality=quality,
                fundamentals_context=fundamentals_context,
            )
        except ValueError as e:
            print(f"Cannot run walk-forward: {e}")
            return
        _print_walkforward_summary(results)

        if getattr(args, "trades", False):
            mode_label = "pure-LLM" if pure_llm else "hybrid"
            for wr in results:
                print(f"\n  {mode_label} trades ({wr.start}..{wr.end}):")
                for t in wr.llm.trades:
                    print(f"    {t['date']} {t['side']:<4} {t['ticker']:<8} "
                          f"{t['shares']:>9.4f} @ {t['price']:>10.2f} — {t['reason']}")


# ---------------------------------------------------------------------------
# Daily-core parameter sweep + walk-forward
# ---------------------------------------------------------------------------

_SWEEP_WINDOWS = [("2017-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"),
                  ("2022-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")]


def _sweep_months(idx, start: str) -> list[pd.Timestamp]:
    """Completed calendar month-ends at/after `start` (the shared
    ``monthly.completed_month_ends`` rule — the sweep's frame-cache keys
    must match what the backtests look up, otherwise a caller-provided
    cache misses the month-end keys: silent ranking misses, or a data hole
    at the true month-end leaving a stale last-stored-candle key)."""
    from . import monthly as monthly_mod
    return monthly_mod.completed_month_ends(idx, pd.Timestamp(start), None)


async def _sweep_load_market(universe: str):
    from . import monthly as monthly_mod
    from . import fundamentals as fundamentals_mod

    tickers = universe_tickers(universe)
    fund = await fundamentals_mod.load_fundamentals(tickers)
    # Opens are the CONTRACT fill series (strict exact-date, never ffilled);
    # the sweep reuses them for every config instead of refilling at closes.
    open_px, close, vol = await monthly_mod.load_ohlcv_frames(tickers, None)
    return tickers, open_px, close, vol, fund


def _sweep_grid() -> list[tuple]:
    """The knob grid: dca x entry-gate x exit-cadence x crash-pause x boost.
    The risk overlays (mom variant / vol-weight / target-vol / lowvol-tilt)
    are swept separately in stage-3 (they interact but a full cross would be
    10x the configs — stage-3 fixes the stage-1 winner and varies one risk
    knob at a time, then the best 2-3 combinations)."""
    grid = []
    for dca, gate, exit_c, pause in itertools.product(
            ["monthly", "daily", "rank"], ["none", "not-crash", "above-sma50"],
            ["monthly", "daily"], [False, True]):
        for boost in ([0.0, 0.25, 0.5] if dca == "rank" else [0.0]):
            grid.append((dca, gate, exit_c, pause, boost))
    return grid


def _sweep_risk_grid() -> list[dict]:
    """Risk-overlay grid, evaluated on top of the stage-1 winner. Each dict
    overrides settings for one config; {} = the stage-1 winner unchanged
    (control arm)."""
    return [
        {},
        {"mom": "residual"},
        {"vol_weight": True},
        {"target_vol": 0.20},
        {"target_vol": 0.25},
        {"target_vol": 0.30},
        {"lowvol_tilt": True},
        {"mom": "residual", "vol_weight": True},
        {"mom": "residual", "target_vol": 0.25},
        {"vol_weight": True, "target_vol": 0.25},
        {"mom": "residual", "vol_weight": True, "target_vol": 0.25},
    ]


# Per-worker state, set once by the pool initializer. NOTE: on Python 3.12+
# the default multiprocessing start method is "forkserver" (the app venv is
# 3.14), so `initargs` are pickled to each worker rather than inherited via
# fork copy-on-write — one copy per worker, not per task. Passing the frames
# through the initializer still avoids re-pickling them for every grid item.
_SWEEP_WORKER_STATE: dict = {}


def _sweep_worker_init(open_px: pd.DataFrame, close: pd.DataFrame,
                       vol: pd.DataFrame,
                       fund: dict[str, dict[str, list[dict]]],
                       fc: dict[pd.Timestamp, pd.DataFrame | None]) -> None:
    _SWEEP_WORKER_STATE["open"] = open_px
    _SWEEP_WORKER_STATE["close"] = close
    _SWEEP_WORKER_STATE["vol"] = vol
    _SWEEP_WORKER_STATE["fund"] = fund
    _SWEEP_WORKER_STATE["fc"] = fc


def _sweep_run_config(task: tuple[str, str, str, bool, float]) -> dict:
    """One config in one worker process. Pure compute: the backtest with
    preloaded frames never touches the DB; boost is process-local state."""
    import asyncio

    dca, gate, exit_c, pause, boost = task
    settings.sim_monthly_rank_boost = boost

    async def go() -> BacktestSummary | None:
        return await _daily_core_backtest(
            "2017-01-01", None, settings.sim_monthly_universe,
            settings.sim_monthly_contribution, False,
            gate, exit_c, pause, dca,
            frame_cache=_SWEEP_WORKER_STATE["fc"],
            close_preloaded=(_SWEEP_WORKER_STATE["close"], _SWEEP_WORKER_STATE["vol"]),
            fund_preloaded=_SWEEP_WORKER_STATE["fund"],
            open_preloaded=_SWEEP_WORKER_STATE["open"],
            with_baseline=False)

    r = asyncio.run(go())
    return {
        "dca": dca, "gate": gate, "exit": exit_c, "pause": pause, "boost": boost,
        "irr": r.irr if r else None, "value": r.final_value if r else 0.0,
        "turnover": r.avg_turnover if r else 0.0, "trades": r.n_trades if r else 0,
        "sharpe": r.sharpe if r else 0.0, "max_dd": r.max_drawdown if r else 0.0,
    }


async def _sweep_stage1(fc, open_px, close, vol, fund, baseline_irr: float,
                        out_path: str, workers: int) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    grid = _sweep_grid()
    print(f"stage1: {len(grid)} configs on {workers} workers", flush=True)
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_sweep_worker_init,
                             initargs=(open_px, close, vol, fund, fc)) as ex:
        for res in ex.map(_sweep_run_config, grid, chunksize=1):
            if res["irr"] is not None:
                results.append(res)
            n = len(results)
            if n % 10 == 0:
                print(f"  {n} done ({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda r: -(r["irr"] or -1))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n=== stage1 top 15 ({time.time() - t0:.0f}s) — baseline {baseline_irr:.2%} ===")
    for r in results[:15]:
        print(f"  dca={r['dca']:<7} gate={r['gate']:<11} exit={r['exit']:<7} "
              f"pause={int(r['pause'])} boost={r['boost']:<4} "
              f"IRR {r['irr']:7.2%}  turn {r['turnover']:.1%}  trades {r['trades']}")
    return results


async def _sweep_stage2(fc, open_px, close, vol, fund, results: list[dict],
                        top_n: int, out_path: str) -> None:
    seen: set = set()
    configs = []
    for r in results:
        key = (r["dca"], r["gate"], r["exit"], r["pause"], r["boost"])
        if key in seen:
            continue
        seen.add(key)
        configs.append(r)
        if len(configs) >= top_n:
            break

    print(f"\nstage2: top {len(configs)} configs x {len(_SWEEP_WINDOWS)} "
          f"walk-forward windows + monthly baseline", flush=True)
    report = []
    for w_start, w_end in _SWEEP_WINDOWS:
        base = await _monthly_backtest(w_start, w_end, settings.sim_monthly_universe,
                                       settings.sim_monthly_contribution, False,
                                       frame_cache=fc, close_preloaded=(close, vol),
                                       fund_preloaded=fund, open_preloaded=open_px)
        assert base is not None, "monthly backtest returned no summary"
        row = {"window": f"{w_start[:4]}-{w_end[:4]}", "baseline_irr": base.irr,
               "configs": []}
        for c in configs:
            settings.sim_monthly_rank_boost = c["boost"]
            r = await _daily_core_backtest(
                w_start, w_end, settings.sim_monthly_universe,
                settings.sim_monthly_contribution, False,
                c["gate"], c["exit"], c["pause"], c["dca"],
                frame_cache=fc, close_preloaded=(close, vol), fund_preloaded=fund,
                open_preloaded=open_px, with_baseline=False)
            row["configs"].append({
                "cfg": f"{c['dca']}/{c['gate']}/{c['exit']}/p{int(c['pause'])}/b{c['boost']}",
                "irr": r.irr if r else None})
        report.append(row)
        wins = sum(1 for x in row["configs"] if x["irr"] is not None and x["irr"] > row["baseline_irr"])
        print(f"\n=== {row['window']}: baseline {row['baseline_irr']:.2%} | "
              f"configs beating it: {wins}/{len(configs)}")
        for x in sorted(row["configs"], key=lambda x: -(x["irr"] or -99)):
            if x["irr"] is not None:
                print(f"  {x['cfg']:<40} {x['irr']:7.2%} "
                      f"{'>' if x['irr'] > row['baseline_irr'] else '<'} baseline")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)


async def _sweep_daily_core(args: argparse.Namespace) -> None:
    """`daily-core-sweep` CLI: grid-search the daily-core knobs in parallel,
    then walk-forward the top configs."""
    universe = args.universe or settings.sim_monthly_universe
    workers = args.workers
    out_dir = args.out or "/data/sweeps"
    os.makedirs(out_dir, exist_ok=True)
    stage1_path = os.path.join(out_dir, "daily_core_stage1.json")
    stage2_path = os.path.join(out_dir, "daily_core_walkforward.json")

    from . import monthly as monthly_mod

    print("loading market frames...", flush=True)
    _tickers, open_px, close, vol, fund = await _sweep_load_market(universe)
    print(f"frames: {close.shape[0]} days x {close.shape[1]} tickers", flush=True)

    fc: dict[pd.Timestamp, pd.DataFrame | None] = {}
    t0 = time.time()
    # `fc` is the RAW momentum cache and the default for every non-residual
    # arm. Pin the variant explicitly: the ambient setting (e.g. .env's
    # SIM_DAILY_CORE_MOM_VARIANT=residual) would otherwise make the raw /
    # control arms run residual frames while reporting `mom=raw`.
    settings.sim_daily_core_mom_variant = "raw"
    for d in _sweep_months(close.index, "2016-06-01"):
        fc[d] = await asyncio.to_thread(monthly_mod.eligible_frame, d, close, vol, fund)
    print(f"frame cache: {len(fc)} months ({time.time() - t0:.0f}s)", flush=True)

    base = await _monthly_backtest("2017-01-01", None, universe,
                                   settings.sim_monthly_contribution, False,
                                   frame_cache=fc, close_preloaded=(close, vol),
                                   fund_preloaded=fund, open_preloaded=open_px)
    assert base is not None, "monthly backtest returned no summary"
    print(f"baseline IRR full window: {base.irr:.2%}", flush=True)

    results = await _sweep_stage1(fc, open_px, close, vol, fund, base.irr,
                                  stage1_path, workers)
    if args.stage in ("2", "all"):
        await _sweep_stage2(fc, open_px, close, vol, fund, results,
                            top_n=args.top, out_path=stage2_path)
    if args.stage in ("3", "all"):
        await _sweep_stage3(fc, open_px, close, vol, fund, results,
                            out_path=os.path.join(out_dir, "daily_core_risk_overlays.json"))
    if args.stage in ("4", "all"):
        await _sweep_stage4(fc, open_px, close, vol, fund,
                            out_path=os.path.join(out_dir, "daily_core_risk_walkforward.json"))


async def _sweep_stage4(fc, open_px, close, vol, fund, out_path: str) -> None:
    """Walk-forward A/B of the risk overlays on the LIVE daily-core config
    (rank deploy, boost=0, no gates) across the non-overlapping OOS windows.
    In-sample stage-3 numbers are necessary but not sufficient — the 2026-09
    doc methodology treats walk-forward as the decision grade. The frame
    cache depends on the mom variant (raw vs residual), so both are built
    once up front and reused across windows."""
    from . import monthly as monthly_mod
    # Live config: rank deploy, boost 0, entry-gate none, exit monthly,
    # no crash pause — the overlays sit on top.
    overlays: list[dict] = [
        {"label": "control (live=residual)", "mom": "residual"},
        {"label": "raw (reference)", "mom": "raw"},
        {"label": "residual+tv0.25", "mom": "residual", "target_vol": 0.25},
        {"label": "tv0.25", "mom": "raw", "target_vol": 0.25},
        {"label": "lowvol-tilt", "mom": "raw", "lowvol_tilt": True},
        {"label": "vol-weight", "mom": "raw", "vol_weight": True},
        # Protection overlays ("don't give the win back"): peak-to-trough
        # cash-out brake + exposure trend filter, on the residual core.
        {"label": "res+stop10", "mom": "residual", "portfolio_stop": 0.10},
        {"label": "res+stop15", "mom": "residual", "portfolio_stop": 0.15},
        {"label": "res+stop10+t200", "mom": "residual", "portfolio_stop": 0.10,
         "exposure_trend": 200},
        {"label": "res+trend200", "mom": "residual", "exposure_trend": 200},
        {"label": "res+trend100", "mom": "residual", "exposure_trend": 100},
        {"label": "res+trail15", "mom": "residual", "trailing_stop": 0.15},
        {"label": "res+trail20", "mom": "residual", "trailing_stop": 0.20},
        # Gradient filter ("basket trend"): cash out when the N-day slope of
        # the strategy's own target basket stays negative for the confirm
        # streak. The candidate that beat everything on the Jul-2026 episode.
        {"label": "res+basket10c3", "mom": "residual",
         "basket_trend": 10, "basket_confirm": 3},
        {"label": "res+basket15c3", "mom": "residual",
         "basket_trend": 15, "basket_confirm": 3},
        {"label": "res+basket20c3", "mom": "residual",
         "basket_trend": 20, "basket_confirm": 3},
        {"label": "res+basket30c5", "mom": "residual",
         "basket_trend": 30, "basket_confirm": 5},
        {"label": "res+g10t5", "mom": "residual", "basket_trend": 10,
         "basket_confirm": 3, "basket_threshold": 0.05},
        {"label": "res+bdd8", "mom": "residual", "basket_drawdown": 0.08},
        # Owner's insight (§15): arm the gradient cash-out ONLY in good times
        # (market > SMA200) so bad-market chop can't whipsaw it.
        {"label": "res+g10c3gt", "mom": "residual", "basket_trend": 10,
         "basket_confirm": 3, "basket_good_times": True},
        {"label": "res+g15c3gt", "mom": "residual", "basket_trend": 15,
         "basket_confirm": 3, "basket_good_times": True},
        {"label": "res+g20c3gt", "mom": "residual", "basket_trend": 20,
         "basket_confirm": 3, "basket_good_times": True},
        {"label": "res+g10c3gt+t200", "mom": "residual", "basket_trend": 10,
         "basket_confirm": 3, "basket_good_times": True, "exposure_trend": 200},
        # Arming-window study (§15): SMA50 = strong uptrends only.
        {"label": "res+g10c3gt50+t200", "mom": "residual", "basket_trend": 10,
         "basket_confirm": 3, "basket_good_times": True, "basket_arm_sma": 50,
         "exposure_trend": 200},
        {"label": "res+g10c3gt100+t200", "mom": "residual", "basket_trend": 10,
         "basket_confirm": 3, "basket_good_times": True, "basket_arm_sma": 100,
         "exposure_trend": 200},
        {"label": "res+g10c3gt50", "mom": "residual", "basket_trend": 10,
         "basket_confirm": 3, "basket_good_times": True, "basket_arm_sma": 50},
        {"label": "res+bdd12", "mom": "residual", "basket_drawdown": 0.12},
    ]
    # one frame cache per mom variant
    print("stage4: building residual frame cache...", flush=True)
    t0 = time.time()
    settings.sim_daily_core_mom_variant = "residual"
    fc_res: dict[pd.Timestamp, pd.DataFrame | None] = {}
    for d in _sweep_months(close.index, "2016-06-01"):
        fc_res[d] = await asyncio.to_thread(monthly_mod.eligible_frame, d, close, vol, fund)
    settings.sim_daily_core_mom_variant = "raw"
    print(f"stage4: residual cache done ({time.time() - t0:.0f}s)", flush=True)

    report = []
    for w_start, w_end in _SWEEP_WINDOWS:
        row = {"window": f"{w_start[:4]}-{w_end[:4]}", "arms": []}
        for ov in overlays:
            settings.sim_daily_core_mom_variant = ov.get("mom", "raw")
            settings.sim_daily_core_lowvol_tilt = ov.get("lowvol_tilt", False)
            use_fc = fc_res if ov.get("mom") == "residual" else fc
            r = await _daily_core_backtest(
                w_start, w_end, settings.sim_monthly_universe,
                settings.sim_monthly_contribution, False,
                "none", "monthly", False, "rank",
                frame_cache=use_fc, close_preloaded=(close, vol), fund_preloaded=fund,
                open_preloaded=open_px, with_baseline=False,
                vol_weight=ov.get("vol_weight") or None,
                target_vol=ov.get("target_vol") or None,
                portfolio_stop_pct=ov.get("portfolio_stop") or None,
                exposure_trend_days=ov.get("exposure_trend") or None,
                trailing_stop_pct=ov.get("trailing_stop") or None,
                basket_trend_days=ov.get("basket_trend") or None,
                basket_confirm_days=ov.get("basket_confirm") or None,
                basket_threshold=ov.get("basket_threshold") or None,
                basket_drawdown=ov.get("basket_drawdown") or None,
                basket_good_times=ov.get("basket_good_times") or None,
                basket_arm_sma=ov.get("basket_arm_sma") or None)
            if r is not None:
                row["arms"].append({"label": ov["label"], "irr": r.irr,
                                    "sharpe": r.sharpe, "max_dd": r.max_drawdown,
                                    "turnover": r.avg_turnover,
                                    "brake_events": r.brake_events})
            settings.sim_daily_core_mom_variant = "raw"
            settings.sim_daily_core_lowvol_tilt = False
        report.append(row)
        print(f"\n=== {row['window']} (OOS walk-forward, rank deploy b0) ===")
        for a in sorted(row["arms"], key=lambda x: -x["sharpe"]):
            print(f"  {a['label']:<22} IRR {a['irr']:7.2%}  Sharpe {a['sharpe']:5.2f}  "
                  f"maxDD {a['max_dd']:6.1%}  turn {a['turnover']:.1%}", flush=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)


async def _sweep_stage3(fc, open_px, close, vol, fund, results: list[dict],
                        out_path: str) -> None:
    """Risk-overlay sweep on top of the stage-1 winner: fix the best
    (dca/gate/exit/pause/boost) config, then vary the risk knobs one at a
    time (plus the 2-3 natural combinations). Reports IRR + Sharpe + max-DD
    for each so the winner is chosen on risk-adjusted return, not IRR alone.
    Mirrors the 2026-09-14 sweep verdict: boost=0 rank deploy was already
    optimal on the deployment knobs, so the remaining edge is risk control.
    """
    if not results:
        print("stage3: no stage-1 results — run stage 1 first")
        return
    from . import monthly as monthly_mod
    best = results[0]
    # Apply the winner's boost: stage-1 set it inside worker processes, so
    # the parent's settings still hold the default — without this the
    # "control" arm runs the wrong config (found 2026-09-14: winner b=0.25,
    # control ran b=0).
    settings.sim_monthly_rank_boost = best["boost"]
    print(f"\nstage3: risk overlays on top of stage-1 winner "
          f"{best['dca']}/{best['gate']}/{best['exit']}/p{int(best['pause'])}/b{best['boost']}",
          flush=True)
    report = []
    for ov in _sweep_risk_grid():
        settings.sim_daily_core_mom_variant = ov.get("mom", "raw")
        settings.sim_daily_core_lowvol_tilt = ov.get("lowvol_tilt", False)
        # frames depend on mom variant + tilt => rebuild when they change
        local_fc = fc
        if ov.get("mom") == "residual" or ov.get("lowvol_tilt"):
            local_fc = {}
            for d in _sweep_months(close.index, "2016-06-01"):
                local_fc[d] = await asyncio.to_thread(
                    monthly_mod.eligible_frame, d, close, vol, fund)
        r = await _daily_core_backtest(
            "2017-01-01", None, settings.sim_monthly_universe,
            settings.sim_monthly_contribution, False,
            best["gate"], best["exit"], best["pause"], best["dca"],
            frame_cache=local_fc, close_preloaded=(close, vol), fund_preloaded=fund,
            open_preloaded=open_px, with_baseline=False,
            vol_weight=ov.get("vol_weight") or None,
            target_vol=ov.get("target_vol") or None)
        label = "+".join(f"{k}={v}" for k, v in ov.items()) or "control"
        if r is not None:
            report.append({"overlay": ov, "label": label, "irr": r.irr,
                           "sharpe": r.sharpe, "max_dd": r.max_drawdown,
                           "turnover": r.avg_turnover, "trades": r.n_trades})
            print(f"  {label:<55} IRR {r.irr:7.2%}  Sharpe {r.sharpe:5.2f}  "
                  f"maxDD {r.max_drawdown:6.1%}  turn {r.avg_turnover:.1%}", flush=True)
        # restore defaults before the next overlay
        settings.sim_daily_core_mom_variant = "raw"
        settings.sim_daily_core_lowvol_tilt = False
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward optimization for the deterministic strategy")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backtest", help="Score the current rules on stored history")
    b.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start")
    b.add_argument("--end", default=None, help="YYYY-MM-DD inclusive end")
    b.add_argument("--trades", action="store_true", help="Print every trade")
    b.add_argument("--cash", action="store_true", help="Print cash-utilization summary (idle cash, max-positions cap)")
    b.add_argument("--regime", action="store_true", help="Enable market regime filter (block BUYs when market < SMA200)")
    b.add_argument("--regime-ticker", default="URTH", help="Benchmark ticker for the regime filter")
    b.add_argument("--no-entry-guards", action="store_true",
                   help="Disable the falling-knife (min_run_5d) and parabolic-extension "
                        "(max_dist_above) BUY blocks — A/B escape hatch for benchmarking")
    b.add_argument("--block-negative-roe", action="store_true",
                   help="Deterministic quality guard: block BUY entries for names "
                        "with KNOWN non-positive ROE (missing stays neutral)")

    s = sub.add_parser("sweep", help="Grid-search thresholds")
    s.add_argument("--start", default=None)
    s.add_argument("--end", default=None)
    s.add_argument("--regime", action="store_true", help="Enable market regime filter")
    s.add_argument("--regime-ticker", default="URTH", help="Benchmark ticker for the regime filter")

    w = sub.add_parser("walkforward", help="Walk-forward fit/test")
    w.add_argument("--train-days", type=int, default=504)
    w.add_argument("--test-days", type=int, default=126)
    w.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start of the walk")
    w.add_argument("--end", default=None, help="YYYY-MM-DD inclusive end of the walk")
    w.add_argument("--regime", action="store_true", help="Enable market regime filter")
    w.add_argument("--regime-ticker", default="URTH", help="Benchmark ticker for the regime filter")

    lb = sub.add_parser("llm-benchmark",
                        help="Probe whether the LLM would turn the deterministic model's worst calls")
    lb.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start of the replay window")
    lb.add_argument("--end", default="2026-08-11",
                    help="YYYY-MM-DD inclusive end (default 2026-08-11 to leave a 20d forward buffer)")
    lb.add_argument("--n-worst", type=int, default=_BENCH_N_WORST,
                    help=f"Number of worst deterministic mistakes to probe (default {_BENCH_N_WORST})")
    lb.add_argument("--n-control", type=int, default=_BENCH_N_CONTROL,
                    help=f"Number of control (deterministic-was-right) cases (default {_BENCH_N_CONTROL})")
    lb.add_argument("--forward-days", type=int, default=_BENCH_FORWARD_DAYS,
                    help=f"Trading days of forward price action used to judge a trade (default {_BENCH_FORWARD_DAYS})")
    lb.add_argument("--skip-llm", action="store_true",
                    help="Select and print the candidate cases without calling the LLM (token-free preview)")

    hr = sub.add_parser("hybrid-replay",
                        help="Replay the hybrid strategy (deterministic + per-day LLM review) on history")
    hr.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start of the replay window")
    hr.add_argument("--end", default=None, help="YYYY-MM-DD inclusive end of the replay window")
    hr.add_argument("--days", type=int, default=None,
                    help="Simulate only the last N trading days (overrides --start; default: full history)")
    hr.add_argument("--trades", action="store_true", help="Print every hybrid trade")
    hr.add_argument("--pure-llm", action="store_true",
                    help="Pure LLM mode: skip deterministic proposals, let the LLM decide from scratch")
    hr.add_argument("--review-interval", type=int, default=1,
                    help="Consult the LLM once per N calendar weeks (1=weekly); "
                         "deterministic proposals execute as-is in between")
    hr.add_argument("--veto-only", action="store_true",
                    help="Hybrid: LLM may only veto/approve deterministic proposals "
                         "(no LLM-initiated BUY/SELL additions)")
    hr.add_argument("--minimal-prompt", action="store_true",
                    help="Use the minimal system prompt (no methodology / regime / veto rules)")
    hr.add_argument("--mode-aware-prompt", action="store_true",
                    help="Use the mode-aware minimal prompt (engine owns exits; "
                         "LLM adds BUYs only). Implies the minimal context")
    hr.add_argument("--failure-marker", action="store_true",
                    help="Consult the LLM only when the engine shows failure "
                         "(2+ stop-out SELLs in 5 trading days or equity >7%% "
                         "below its running peak); no cooldown")
    hr.add_argument("--failure-stop-outs", type=int, default=2,
                    help="With --failure-marker: stop-out SELLs in 5 days that "
                         "count as failure (default 2)")
    hr.add_argument("--failure-drawdown", type=float, default=7.0,
                    help="With --failure-marker: equity drawdown %% below peak "
                         "that counts as failure (default 7.0)")
    hr.add_argument("--fundamentals-context", action="store_true",
                    help="Add point-in-time ROE%% / P-FCF columns to the LLM's "
                         "signal table (monthly qv-mom quality data; missing = '-')")
    hr.add_argument("--block-negative-roe", action="store_true",
                    help="Deterministic quality guard: block BUY entries for "
                         "names with KNOWN non-positive ROE (missing stays neutral)")

    lwf = sub.add_parser("llm-walkforward",
                         help="Multi-window LLM vs deterministic benchmark (reliable scoreboard)")
    lwf.add_argument("--start", default=None,
                     help="YYYY-MM-DD inclusive start of the full walk-forward range")
    lwf.add_argument("--end", default=None,
                     help="YYYY-MM-DD inclusive end of the full walk-forward range")
    lwf.add_argument("--windows", type=int, default=4,
                     help="Number of non-overlapping windows to score (default 4)")
    lwf.add_argument("--days-per-window", type=int, default=30,
                     help="Trading days per window (default 30)")
    lwf.add_argument("--pure-llm", action=argparse.BooleanOptionalAction, default=True,
                     help="Pure LLM mode (default); use --no-pure-llm for hybrid (deterministic + LLM review)")
    lwf.add_argument("--review-interval", type=int, default=1,
                     help="Consult the LLM once per N calendar weeks (1=weekly); "
                          "deterministic proposals execute as-is in between")
    lwf.add_argument("--veto-only", action="store_true",
                     help="Hybrid: LLM may only veto/approve deterministic proposals "
                          "(no LLM-initiated BUY/SELL additions)")
    lwf.add_argument("--allow-llm-sells", action="store_true",
                     help="Hybrid: allow LLM-initiated SELLs (default: engine owns "
                          "exits via stops and SELL signals)")
    lwf.add_argument("--minimal-prompt", action="store_true",
                      help="Use the minimal system prompt: no methodology, no regime "
                           "guidance, no veto rules — just portfolio + signals and "
                           "BUY/SELL/HOLD instructions")
    lwf.add_argument("--mode-aware-prompt", action="store_true",
                      help="Use the mode-aware minimal prompt: tells the LLM the engine "
                           "owns exits (its SELLs are ignored, proposals auto-execute) "
                           "and its only job is to add high-quality BUYs, with a compact "
                           "entry-quality rubric. Implies the minimal context.")
    lwf.add_argument("--llm-max-positions", type=int, default=0,
                      help="Raise the max-positions cap for LLM BUYs only "
                           "(default 0 = use the engine's max_positions). The "
                           "deterministic engine still caps at max_positions; "
                           "the extra slots above it belong to the LLM's picks.")
    lwf.add_argument("--no-entry-guards", action="store_true",
                      help="Disable the falling-knife (min_run_5d) and "
                           "parabolic-extension (max_dist_above) BUY blocks in "
                           "both the det and hybrid replays — A/B escape hatch")
    lwf.add_argument("--marker-gated", action="store_true",
                     help="Deterministic marker: only consult the LLM on review "
                          "days when the engine actually PROPOSES trades (no "
                          "wasted calls on zero-proposal days)")
    lwf.add_argument("--failure-marker", action="store_true",
                     help="Failure marker: consult the LLM only when the engine "
                          "shows failure — stop-out SELLs in 5 days or equity "
                          "below its running peak. No cooldown: a fresh failure "
                          "triggers a call the same day")
    lwf.add_argument("--failure-stop-outs", type=int, default=2,
                     help="With --failure-marker: stop-out SELLs in 5 days that "
                          "count as failure (default 2)")
    lwf.add_argument("--failure-drawdown", type=float, default=7.0,
                     help="With --failure-marker: equity drawdown %% below peak "
                          "that counts as failure (default 7.0)")
    lwf.add_argument("--fundamentals-context", action="store_true",
                     help="Add point-in-time ROE%% / P-FCF columns to the LLM's "
                          "signal table (monthly qv-mom quality data; missing = '-')")
    lwf.add_argument("--block-negative-roe", action="store_true",
                     help="Deterministic quality guard: block BUY entries for "
                          "names with KNOWN non-positive ROE (missing stays neutral)")
    lwf.add_argument("--trades", action="store_true", help="Print every LLM trade per window")

    mb = sub.add_parser("monthly-backtest",
                        help="DCA backtest of the monthly qv-mom strategy on stored candles")
    mb.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start")
    mb.add_argument("--end", default=None, help="YYYY-MM-DD inclusive end")
    mb.add_argument("--universe", default=None,
                    help="Universe name (default: sim_monthly_universe)")
    mb.add_argument("--contribution", type=float, default=None,
                    help="Monthly contribution in USD (default: sim_monthly_contribution)")
    mb.add_argument("--picks", action="store_true", help="Print the last 12 rebalance picks")

    dc = sub.add_parser("daily-core",
                        help="Daily-core backtest: monthly qv-mom fundamentals as the "
                             "core portfolio, minor candle-driven daily adjustments on top. "
                             "Compares IRR against the monthly baseline on the same window")
    dc.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start")
    dc.add_argument("--end", default=None, help="YYYY-MM-DD inclusive end")
    dc.add_argument("--universe", default=None,
                    help="Universe name (default: sim_monthly_universe)")
    dc.add_argument("--contribution", type=float, default=None,
                    help="Monthly contribution in USD (default: sim_monthly_contribution)")
    dc.add_argument("--dca", choices=["monthly", "daily", "rank"], default="monthly",
                    help="Deploy contributions at month-end (monthly sim), daily "
                         "into the most underweight holding, or daily rank-weighted "
                         "(fresh cash reinforces the top-ranked core names)")
    dc.add_argument("--entry-gate", choices=["none", "above-sma50", "not-crash"],
                    default="none",
                    help="Candle-driven timing for NEW core-name entries "
                         "(none = buy at once, like the monthly sim)")
    dc.add_argument("--exit-cadence", choices=["monthly", "daily"], default="monthly",
                    help="Release held names that left the hysteresis band at "
                         "month-end (monthly sim) or the next day")
    dc.add_argument("--crash-pause", action="store_true",
                    help="Pause new entries in a pullback regime")
    dc.add_argument("--mom", choices=["raw", "residual"], default=None,
                    help="Momentum variant: raw 12-1 close/close (default) or "
                         "Blitz-Huij-Martens residual momentum")
    dc.add_argument("--vol-weight", action="store_true",
                    help="Weight positions ∝ 1/realized-vol instead of equal weight")
    dc.add_argument("--target-vol", type=float, default=None,
                    help="Cap deployment when portfolio 21d realized vol exceeds this "
                         "(annualized, e.g. 0.25); 0/unset = off")
    dc.add_argument("--lowvol-tilt", action="store_true",
                    help="Add pct_rank(-vol) as a 4th score term (defensive tilt)")
    dc.add_argument("--portfolio-stop", type=float, default=None,
                    help="Peak-to-trough cash-out brake: sell everything to cash when the "
                         "portfolio's equity is this %% below its running peak (0 = off)")
    dc.add_argument("--exposure-trend", type=int, default=None,
                    help="Deploy cash only while the equal-weight universe index is above "
                         "its N-day SMA (0 = off); also the re-entry gate after a stop")
    dc.add_argument("--trailing-stop", type=float, default=None,
                    help="Per-name trailing stop: exit a holding when its price falls this "
                         "fraction (e.g. 0.15) below its own peak since entry (0 = off)")
    dc.add_argument("--thesis-break", action="store_true",
                    help="M5 fundamental thesis-break exit: sell a holding at the next "
                         "session's open when its newly filed fundamentals (filing-date "
                         "PIT) would knock it out of the eligible frame's hard gates — "
                         "record missing, equity <= 0, TTM NI missing, shares missing, or "
                         "ROE = TTM NI / equity <= 0 (F/P-FCF changes alone do not trigger)")
    dc.add_argument("--basket-trend", type=int, default=None,
                    help="Gradient filter: N-day rate-of-change of the target basket; "
                         "cash out when negative for --basket-confirm days (0 = off)")
    dc.add_argument("--basket-confirm", type=int, default=None,
                    help="Consecutive days the basket gradient must stay negative "
                         "(and positive to re-enter) — default 3")
    dc.add_argument("--basket-threshold", type=float, default=None,
                    help="Minimum |slope| to count as a negative (e.g. 0.05 = only a "
                         "real 5%% N-day drop triggers; 0 = any negative slope)")
    dc.add_argument("--basket-drawdown", type=float, default=None,
                    help="Cash out when the target basket is this far below its peak "
                         "(e.g. 0.08); re-enter when the drawdown halves (0 = off)")
    dc.add_argument("--basket-er-min", type=float, default=None,
                    help="Kaufman efficiency-ratio gate: only arm the trigger while the "
                         "basket path is efficient (e.g. 0.3; 0 = always armed)")
    dc.add_argument("--basket-good-times", action="store_true",
                    help="Arm the gradient cash-out only while the market is above its "
                         "SMA (bull-market sleeve-crash catcher)")
    dc.add_argument("--basket-arm-sma", type=int, default=None,
                    help="Window of the good-times arming SMA (default 200; e.g. 50)")
    dc.add_argument("--verbose", action="store_true", help="Print the last trades")

    sw = sub.add_parser("daily-core-sweep",
                        help="Grid-search the daily-core knobs in parallel, then "
                             "walk-forward the top configs against the monthly baseline")
    sw.add_argument("--stage", default="all", choices=["1", "2", "3", "4", "all"],
                    help="1 = in-sample grid, 2 = walk-forward of the top configs, "
                         "3 = risk-overlay sweep on the stage-1 winner, "
                         "4 = walk-forward A/B of the risk overlays on the live config")
    sw.add_argument("--workers", type=int, default=6,
                    help="Worker processes for stage1 (default 6)")
    sw.add_argument("--universe", default=None,
                    help="Universe name (default: sim_monthly_universe)")
    sw.add_argument("--top", type=int, default=8,
                    help="Configs carried into the walk-forward (default 8)")
    sw.add_argument("--out", default="/data/sweeps",
                    help="Directory for the results JSONs (default /data/sweeps)")

    pc = sub.add_parser("prune-candles",
                        help="Delete candle rows before a cutoff date (frees DB space, "
                             "does not affect the daily engine which only needs ~2y). "
                             "WARNING: this is the one subcommand that WRITES to the shared "
                             "live candle DB, and a cutoff within 8 years of today is refused "
                             "unless --allow-degrade (the savings mirror reads 8 years back)")
    pc.add_argument("--before", default="2016-01-01",
                    help="YYYY-MM-DD: delete candles strictly before this date (default 2016-01-01; "
                         "a cutoff within 8 years of today needs --allow-degrade)")
    pc.add_argument("--dry-run", action="store_true", help="Only report what would be deleted")
    pc.add_argument("--allow-degrade", action="store_true",
                    help="Allow a --before cutoff within 8 years of today, deleting candle rows "
                         "the savings mirror's 8-year history reads (app/savings.py "
                         "_eur_close_history)")

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
