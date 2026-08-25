"""Walk-forward optimization / backtest tool for the deterministic strategy.

This is a *separate, read-only* analysis tool. It never touches the live sim
account, positions, or trades. It replays the deterministic decision rules
against stored historical candles and reports how well a given parameter set
would have performed.

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
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import select

from .analysis import (
    MIN_CANDLES,
    decide_action,
    signal_series as _signal_series,
    snapshot_from_row,
    strength_for,
)
from .config import settings
from . import llm as llm_mod
from .db import Candle, Session
from .screener import tickers as universe_tickers
from .strategy import (
    StrategyParams,
    llm_buy_budget,
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


# Sector groupings for the global-large-cap universe. Used by the sector
# diversification cap to prevent correlated positions from concentrating risk.
SECTORS: dict[str, set[str]] = {
    "semiconductors": {"NVDA", "AMD", "AVGO", "TSM", "ASML.AS", "MU", "ARM",
                       "MRVL", "QCOM", "ANET", "SOXX", "SMH", "IFX.DE",
                       "BESI.AS", "NEM.DE", "NOKIA.HE", "ENR.DE", "TER"},
    "hyperscalers": {"MSFT", "GOOGL", "AMZN", "META", "ORCL", "PLTR", "NOW",
                     "CRM", "ADBE", "SNOW", "DDOG", "MDB", "AI", "SOUN",
                     "PATH", "UPST", "TEM", "RGTI", "IONQ", "RKLB", "CRWV",
                     "FIG", "CRCL"},
    "european_tech": {"SAP.DE", "SIE.DE", "AMS.MC", "DSY.PA", "AI.PA",
                     "HO.PA", "SAAB-B.ST", "SOF.BR"},
    "space_defense": {"SPCX", "ASTS", "LUNR", "RDW", "KTOS", "PL", "IRDM"},
    "data_center_energy": {"ETN", "GEV", "CEG", "VST", "VRT"},
    "industrial": {"ROK"},
    "cybersecurity": {"CRWD", "PANW"},
    "medical": {"ISRG", "SYK", "MDT", "SHL.DE", "CRSP", "VEEV", "GH", "BNTX"},
    "pharma": {"LLY", "JNJ", "UNH", "PFE", "TMO", "XLV"},
    "financials": {"JPM", "GS", "V", "BLK", "XLF"},
    "consumer": {"WMT", "PG", "COST", "KO", "HD", "XLP"},
    "energy": {"XOM", "CVX", "XLE"},
    "utilities_bonds": {"NEE", "TLT", "XLU"},
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
    """Max peak-to-trough drawdown as a percentage.

    If ``invested`` (cumulative capital deposited so far per day) is provided,
    the drawdown is measured on the **return ratio** (equity / invested) rather
    than raw equity. This neutralises the growing capital base from monthly
    DCA deposits — a 10% drawdown means the portfolio lost 10% of *its current
    capital base*, not that the equity dropped from a late peak to an early
    low point. Without this, a DCA backtest reports absurd drawdowns like 95%
    because the equity naturally grows over time from deposits alone.
    """
    if invested is None:
        # Fixed-capital backtest: drawdown on raw equity.
        peak = -math.inf
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            if peak > 0:
                dd = (peak - e) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        return max_dd
    # DCA backtest: drawdown on the return ratio (equity / invested).
    # This is a time-weighted measure: if equity is 105% of invested and
    # drops to 95% of invested, that's a ~9.5% drawdown regardless of how
    # much capital has been deposited.
    peak_ratio = -math.inf
    max_dd = 0.0
    for e, inv in zip(equity, invested):
        if inv <= 0:
            continue
        ratio = e / inv
        if ratio > peak_ratio:
            peak_ratio = ratio
        if peak_ratio > 0:
            dd = (peak_ratio - ratio) / peak_ratio * 100
            if dd > max_dd:
                max_dd = dd
    return max_dd


async def _load_series(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load and precompute the signal series for each ticker."""
    series: dict[str, pd.DataFrame] = {}
    async with Session() as s:
        for idx, t in enumerate(tickers):
            if idx and idx % 10 == 0:
                logger.info("Loading series... %d/%d", idx, len(tickers))
            rows = (
                await s.scalars(
                    select(Candle).where(Candle.ticker == t).order_by(Candle.timestamp)
                )
            ).all()
            if len(rows) < MIN_CANDLES:
                continue
            data = [{
                "time": r.timestamp.strftime("%Y-%m-%d"),
                "open": r.open, "high": r.high, "low": r.low,
                "close": r.close, "volume": r.volume,
            } for r in rows]
            try:
                series[t] = _signal_series(data)
            except Exception as e:
                logger.warning("Could not compute series for %s: %s", t, e)
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
    return {row.time: bool(row.uptrend) for row in df.itertuples(index=False) if pd.notna(row.sma200)}


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
            regime: dict[str, bool] | None = None) -> ReplayResult:
    """Run the deterministic strategy over the precomputed series.

    ``start``/``end`` are inclusive date strings (YYYY-MM-DD) used to bound
    the replay window (e.g. a walk-forward test window). ``regime`` is an
    optional day→uptrend map; when provided and ``params.regime_filter`` is
    set, new BUYs are blocked on days where the market is not in an uptrend.
    """
    # Build a global timeline of all trading days across tickers.
    all_days: set[str] = set()
    for df in series.values():
        all_days.update(df["time"].tolist())
    days = sorted(all_days)
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    if not days:
        return ReplayResult(params=params)

    # Index each series by time for O(1) lookup. Convert each row to a plain
    # dict (not a pandas Series) to avoid per-access overhead in the hot loop.
    # The raw scoring components are kept so action/strength can be derived
    # per-parameter-set at replay time (the sweep varies the thresholds).
    by_time: dict[str, dict[str, dict]] = {}
    for t, df in series.items():
        by_time[t] = {
            row.time: {
                "close": float(row.close),
                "net": float(row.net),
                "bullish": float(row.bullish),
                "bearish": float(row.bearish),
                "trend_up": bool(row.trend_up),
                "trend_down": bool(row.trend_down),
                "dist_above": float(row.dist_above),
                "dist_below": float(row.dist_below),
                "atr_stop": None if pd.isna(row.atr_stop) else float(row.atr_stop),
                "atr14": float(row.atr14) if not pd.isna(row.atr14) else None,
                "weekly_trend_up": bool(row.weekly_trend_up) if not pd.isna(row.weekly_trend_up) else True,
                "run_5d": None if pd.isna(row.run_5d) else float(row.run_5d),
                "run_20d": None if pd.isna(row.run_20d) else float(row.run_20d),
                "run_60d": None if pd.isna(row.run_60d) else float(row.run_60d),
                "dist_52w_high": None if pd.isna(row.dist_52w_high) else float(row.dist_52w_high),
            }
            for row in df.itertuples(index=False)
        }

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
    sp = _replay_params_to_strategy(params)

    for day in days:
        # Monthly allowance deposit (first trading day of a new month).
        month = day[:7]
        if month != last_deposit_month:
            pf.cash += params.monthly_allowance
            cumulative_invested += params.monthly_allowance
            last_deposit_month = month

        # Prices for this day across all tickers. Carry forward the last known
        # close when a ticker has no data on this day (e.g. US holidays where
        # European markets are open). Without this, positions are priced at 0
        # on sparse-data days, producing fake 95% drawdowns.
        prices: dict[str, float] = {}
        for t, idx in by_time.items():
            row = idx.get(day)
            if row is not None:
                p = row["close"]
                prices[t] = p
                last_known_prices[t] = p
            elif t in last_known_prices:
                prices[t] = last_known_prices[t]

        total_equity = pf.equity(prices)
        if total_equity <= 0:
            equity_curve.append({"time": day, "equity": 0.0})
            invested_curve.append(cumulative_invested)
            cash_curve.append(pf.cash)
            position_count_curve.append(len(pf.positions))
            continue

        # Track portfolio peak for circuit breaker.
        if total_equity > portfolio_peak:
            portfolio_peak = total_equity

        # Portfolio-level circuit breaker: if equity has dropped
        # portfolio_stop_pct from its peak, sell everything and go risk-off.
        if params.portfolio_stop_pct > 0 and portfolio_peak > 0:
            dd = (portfolio_peak - total_equity) / portfolio_peak * 100
            if dd >= params.portfolio_stop_pct:
                if not risk_off:
                    # Sell all positions immediately.
                    for ticker in list(pf.positions.keys()):
                        price = prices.get(ticker, 0)
                        if price > 0:
                            pf.sell(ticker, price, None,
                                    f"Portfolio stop: equity {total_equity:.0f} "
                                    f"is -{dd:.1f}% from peak {portfolio_peak:.0f}",
                                    date=day)
                    risk_off = True
            elif risk_off and dd < params.portfolio_stop_pct / 2:
                # Re-engage when drawdown recovers to half the stop level.
                risk_off = False

        min_cash = total_equity * (params.min_cash_pct / 100)
        max_position_value = total_equity * (params.max_position_pct / 100)

        # Update peak prices for trailing stop tracking.
        pf.update_peaks(prices)

        # --- Pre-filter: trailing stop ---
        # Sell if price has dropped trailing_stop_pct from its peak since entry.
        # This is a backtest-only SELL that strategy.propose_trades doesn't know
        # about (it doesn't track per-position peaks), so we apply it before
        # calling propose_trades and let the core logic handle the rest.
        if params.trailing_stop_pct > 0:
            for ticker in list(pf.positions.keys()):
                peak = pf.peak_price.get(ticker, 0)
                price = prices.get(ticker, 0)
                if peak > 0 and price > 0:
                    stop_level = peak * (1 - params.trailing_stop_pct / 100)
                    if price <= stop_level:
                        pf.sell(ticker, price, None,
                                f"Trailing stop: {price:.2f} <= {stop_level:.2f} "
                                f"(peak {peak:.2f}, -{params.trailing_stop_pct}%)",
                                date=day)

        # --- Core: delegate SELL to strategy.propose_trades, execute, then BUY ---
        # propose_trades works on a snapshot; SELLs must execute first so the
        # BUY phase sees freed slots + cash (matching the old inline behaviour
        # where SELLs mutated pf before BUYs ran).
        signals = _signals_for_day_from_bytime(by_time, day, params)

        # SELL phase: get proposals, execute them immediately.
        proposals = propose_trades(
            positions=_pf_to_positions(pf),
            cash=pf.cash,
            prices=prices,
            signals=signals,
            params=sp,
            sector_of=_sector_of if params.max_sector_pct > 0 else None,
        )
        for p in proposals:
            if p["side"] == "SELL":
                _execute_proposal(pf, {**p, "date": day})

        # BUY phase: re-derive post-SELL state, re-propose for BUYs only.
        # Regime filter: block new BUYs when the broad market is below its
        # 200-day SMA, or when the circuit breaker is active.
        market_ok = not risk_off
        if params.regime_filter and regime is not None:
            market_ok = market_ok and regime.get(day, True)

        if market_ok:
            proposals2 = propose_trades(
                positions=_pf_to_positions(pf),
                cash=pf.cash,
                prices=prices,
                signals=signals,
                params=sp,
                sector_of=_sector_of if params.max_sector_pct > 0 else None,
            )
            for p in proposals2:
                if p["side"] == "BUY":
                    _execute_proposal(pf, {**p, "date": day})

        # Record equity.
        total_equity = pf.equity(prices)
        equity_curve.append({"time": day, "equity": round(total_equity, 2)})
        invested_curve.append(cumulative_invested)
        cash_curve.append(pf.cash)
        position_count_curve.append(len(pf.positions))
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
) -> dict[str, dict]:
    """Build the ``{ticker: compute()-shaped result}`` dict for one day.

    Mirrors what ``sim._gather_signals`` produces for the live cycle: each
    ticker maps to ``{"action", "reason", "snapshot", "strength"}``. The
    snapshot is built via ``snapshot_from_row`` (the single source of truth
    shared with ``analysis.compute``) so the LLM sees exactly the fields the
    live engine shows it. Tickers with no row for this day are skipped (the
    live sim skips them too when ``_latest_close`` returns None).
    """
    out: dict[str, dict] = {}
    for t, idx in by_time.items():
        row = idx.get(day)
        if row is None:
            continue
        df = series[t]
        x = df.iloc[df.index[df["time"] == day].tolist()[0]]
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
        relaxed_hold_strength=p.relaxed_hold_strength,
        relaxed_hold_limit=p.relaxed_hold_limit,
        max_sector_pct=p.max_sector_pct,
        risk_pct=p.risk_pct,
    )


def _deterministic_propose_replay(
    pf: PaperPortfolio,
    by_time: dict[str, dict[str, dict]],
    prices: dict[str, float],
    day: str,
    params: ReplayParams,
) -> list[dict]:
    """Propose deterministic trades for one day WITHOUT mutating ``pf``.

    Thin wrapper over ``strategy.propose_trades`` — converts PaperPortfolio
    to plain data, builds signals from the precomputed by_time rows, calls
    the shared propose logic, and stamps each proposal with the day + any
    entry_stop the shared logic computed.
    """
    signals = _signals_for_day_from_bytime(by_time, day, params)
    sp = _replay_params_to_strategy(params)
    proposals = propose_trades(
        positions=_pf_to_positions(pf),
        cash=pf.cash,
        prices=prices,
        signals=signals,
        params=sp,
    )
    # Stamp the day on each proposal (strategy.propose_trades is day-agnostic).
    for p in proposals:
        p["date"] = day
    return proposals


def _signals_for_day_from_bytime(
    by_time: dict[str, dict[str, dict]],
    day: str,
    params: ReplayParams,
) -> dict[str, dict]:
    """Build the ``{ticker: signal}`` dict for one day from precomputed by_time rows.

    Converts the raw scoring components in by_time (net, bullish, bearish,
    trend_up, etc.) into the same {action, strength, reason, snapshot} shape
    that ``analysis.compute()`` and ``_signals_for_day`` produce, so the
    shared ``strategy.propose_trades`` can consume them uniformly.
    """
    signals: dict[str, dict] = {}
    for t, idx in by_time.items():
        row = idx.get(day)
        if row is None:
            continue
        action = _row_action(row, params)
        strength = _row_strength(row, action)
        signals[t] = {
            "action": action,
            "strength": strength,
            "reason": f"{action} (strength {strength})",
            "snapshot": {
                "atr_stop": row.get("atr_stop"),
                "run_5d": row.get("run_5d"),
                "atr14": row.get("atr14"),
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
    return datetime.strptime(day, "%Y-%m-%d").strftime("%Y-W%W")


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
    """
    from .sim import _LLM_SYSTEM_PROMPT, _LLM_MINIMAL_SYSTEM_PROMPT, _PURE_LLM_SYSTEM_PROMPT, _build_llm_context, _parse_llm_decisions, _week_diff

    # Build the global timeline and per-ticker day index (same as _replay).
    all_days: set[str] = set()
    for df in series.values():
        all_days.update(df["time"].tolist())
    days = sorted(all_days)
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    if not days:
        return ReplayResult(params=params)

    by_time: dict[str, dict[str, dict]] = {}
    for t, df in series.items():
        by_time[t] = {
            row.time: {
                "close": float(row.close),
                "net": float(row.net),
                "bullish": float(row.bullish),
                "bearish": float(row.bearish),
                "trend_up": bool(row.trend_up),
                "trend_down": bool(row.trend_down),
                "dist_above": float(row.dist_above),
                "dist_below": float(row.dist_below),
                "atr_stop": None if pd.isna(row.atr_stop) else float(row.atr_stop),
                "atr14": float(row.atr14) if not pd.isna(row.atr14) else None,
                "weekly_trend_up": bool(row.weekly_trend_up) if not pd.isna(row.weekly_trend_up) else True,
                "run_5d": None if pd.isna(row.run_5d) else float(row.run_5d),
                "run_20d": None if pd.isna(row.run_20d) else float(row.run_20d),
                "run_60d": None if pd.isna(row.run_60d) else float(row.run_60d),
                "dist_52w_high": None if pd.isna(row.dist_52w_high) else float(row.dist_52w_high),
            }
            for row in df.itertuples(index=False)
        }

    pf = PaperPortfolio(cash=params.start_cash)
    equity_curve: list[dict] = []
    invested_curve: list[float] = []
    daily_returns: list[float] = []
    prev_equity: float | None = None
    last_deposit_month: str | None = None
    cumulative_invested = params.start_cash
    last_known_prices: dict[str, float] = {}
    llm_trades: list[dict] = list(pf.trades)  # full trade log (det + LLM)
    last_review_week: str | None = None

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

            logger.info("hybrid day %d/%d %s — equity %.2f, cash %.2f, %d positions",
                        day_idx, len(days), day, pf.equity(prices),
                        pf.cash, len(pf.positions))

            if pf.equity(prices) <= 0:
                equity_curve.append({"time": day, "equity": 0.0})
                invested_curve.append(cumulative_invested)
                continue

            # --- 1. Deterministic PROPOSE phase (no mutation) ---
            signals = _signals_for_day(series, by_time, day, params)
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
                proposals = _deterministic_propose_replay(pf, by_time, prices, day, params)
                if proposals:
                    logger.info("  det proposed: %d — %s",
                                len(proposals),
                                ", ".join(f"{p['side']} {p['ticker']}" for p in proposals))
                else:
                    logger.info("  det proposed: no trades")

            # --- 2 + 3. LLM review → reconcile → execute ---
            total_equity = pf.equity(prices)
            # Fire the review on the first trading day of each calendar week
            # (mirrors the live sim's weekly review): the LLM is consulted at
            # most once per ISO week, anchored to the week, not a day counter.
            # review_interval=1 still reviews every week; larger values skip
            # intermediate weeks.
            week = _current_week_from(day)
            is_review_day = (
                last_review_week is None
                or _week_diff(week, last_review_week) >= review_interval
            )
            if total_equity > 0 and (settings.llm_backends or settings.ollama_model) and is_review_day:
                allowance_total = cumulative_invested
                valuation = valuate_portfolio(_pf_to_positions(pf), pf.cash, prices, allowance_total)
                signals = _signals_for_day(series, by_time, day, params)
                context = _build_llm_context(valuation, proposals, signals, news,
                                            pure_llm=pure_llm,
                                            trade_history=list(reversed(pf.trades[-15:]))[:12],
                                            minimal=minimal_prompt)
                system_prompt = (_LLM_MINIMAL_SYSTEM_PROMPT if minimal_prompt
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
                            price_of=lambda t: _aval(prices.get(t)),
                            value_of=lambda t: _aval(pf.positions.get(t, 0) * prices.get(t, 0)),
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
                            if price is None or price <= 0 or action == "HOLD":
                                continue
                            if action == "BUY":
                                # Max-positions cap: block NEW positions when at
                                # the cap, but still allow topping up tickers
                                # already held.
                                if (params.max_positions > 0
                                        and len(held_tickers) >= params.max_positions
                                        and tu not in held_tickers):
                                    logger.info("  llm BUY %s skipped (max-positions cap %d)",
                                                tu, params.max_positions)
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
                last_review_week = week
            else:
                # No LLM configured, or not a review week: execute all
                # proposals as-is (deterministic). On non-review weeks the
                # engine runs alone — the LLM only reviews once per week.
                for p in proposals:
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
        closes[t] = {row.time: float(row.close) for row in df.itertuples(index=False)}

    portfolio_states = _reconstruct_portfolio_states(trades)
    cases: list[TradeCase] = []
    for tr, state in zip(trades, portfolio_states):
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
    for ticker, group in by_ticker.items():
        group.sort(key=lambda c: c.badness, reverse=True)
        worst.append(group[0])  # each ticker's worst single decision
    worst.sort(key=lambda c: c.badness, reverse=True)

    controls: list[TradeCase] = []
    for ticker, group in by_ticker.items():
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
        for c, e in zip(res.cash_curve, res.equity_curve)
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
    for e, c, n in zip(res.equity_curve, res.cash_curve, res.position_count_curve):
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
    no_llm_sells: bool = False,
    news: dict[str, list[dict]] | None = None,
    minimal_prompt: bool = False,
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
        det = _replay(series, params, start=w_start, end=w_end)
        logger.info("window %d/%d %s..%s deterministic: return %.2f%%, dd %.2f%%, %d trades",
                    i, len(windows), w_start, w_end,
                    det.total_return_pct, det.max_drawdown_pct, det.n_trades)
        llm = await _hybrid_replay(series, params, start=w_start, end=w_end,
                                   pure_llm=pure_llm,
                                   review_interval=review_interval,
                                   veto_only=veto_only,
                                   no_llm_sells=no_llm_sells,
                                   minimal_prompt=minimal_prompt)
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


async def _main(args: argparse.Namespace) -> None:
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
        res = _replay(series, params, start=args.start, end=args.end, regime=regime)
        label = "Backtest (live sim risk config)"
        if params.regime_filter:
            label += " + regime filter"
        _print_result(res, label)
        print(f"  Risk config: max_positions={params.max_positions}, "
              f"stop={params.stop_type} {params.stop_pct:g}%, "
              f"atr_stop={params.use_atr_stop}, max_run_5d={params.max_run_5d:g}%, "
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
        print(f"\n=== {mode_label} replay ({start}..{end}) — probing LLM "
              f"once per {review_interval} calendar week(s) ===")
        print(f"  Backend: {active.get('name', '?')} · {active.get('model', '?')}")
        hyb = await _hybrid_replay(series, params, start=start, end=end,
                                   pure_llm=pure_llm,
                                   review_interval=review_interval,
                                   veto_only=veto_only,
                                   minimal_prompt=minimal_prompt)
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
        pure_llm = getattr(args, "pure_llm", False)
        n_windows = args.windows
        days_per_window = args.days_per_window
        review_interval = getattr(args, "review_interval", 1)
        veto_only = getattr(args, "veto_only", False)
        no_llm_sells = getattr(args, "no_llm_sells", False)
        minimal_prompt = getattr(args, "minimal_prompt", False)
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
    lwf.add_argument("--no-llm-sells", action="store_true",
                     help="Hybrid: block LLM-initiated SELLs (engine owns exits via "
                          "stops and SELL signals; LLM owns entries)")
    lwf.add_argument("--minimal-prompt", action="store_true",
                     help="Use the minimal system prompt: no methodology, no regime "
                          "guidance, no veto rules — just portfolio + signals and "
                          "BUY/SELL/HOLD instructions")
    lwf.add_argument("--trades", action="store_true", help="Print every LLM trade per window")

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
