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
import logging
import math
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import select

from .analysis import (
    MIN_CANDLES,
    decide_action,
    signal_series as _signal_series,
    strength_for,
)
from .config import settings
from .db import Candle, Session
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.optimize")


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
    trades: list[dict] = field(default_factory=list)

    def buy(self, ticker: str, price: float, budget: float, reason: str,
            stop: float | None = None) -> None:
        if budget < 1 or price <= 0:
            return
        shares = round(budget / price, 4)
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
        self.trades.append({"ticker": ticker, "side": "BUY", "shares": shares,
                            "price": price, "reason": reason})

    def sell(self, ticker: str, price: float, shares: float | None, reason: str) -> None:
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
        self.trades.append({"ticker": ticker, "side": "SELL", "shares": sell_shares,
                            "price": price, "reason": reason})

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


def _max_drawdown(equity: list[float]) -> float:
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
            }
            for row in df.itertuples(index=False)
        }

    pf = PaperPortfolio(cash=params.start_cash)
    equity_curve: list[dict] = []
    daily_returns: list[float] = []
    prev_equity: float | None = None
    last_deposit_month: str | None = None

    for day in days:
        # Monthly allowance deposit (first trading day of a new month).
        month = day[:7]
        if month != last_deposit_month:
            pf.cash += params.monthly_allowance
            last_deposit_month = month

        # Prices for this day across all tickers.
        prices: dict[str, float] = {}
        for t, idx in by_time.items():
            row = idx.get(day)
            if row is not None:
                prices[t] = row["close"]

        total_equity = pf.equity(prices)
        if total_equity <= 0:
            equity_curve.append({"time": day, "equity": 0.0})
            continue

        min_cash = total_equity * (params.min_cash_pct / 100)
        max_position_value = total_equity * (params.max_position_pct / 100)

        # Update peak prices for trailing stop tracking.
        pf.update_peaks(prices)

        # --- SELL phase ---
        for ticker in list(pf.positions.keys()):
            row = by_time.get(ticker, {}).get(day)
            if row is None:
                continue
            price = row["close"]
            action = _row_action(row, params)
            if action == "SELL":
                pf.sell(ticker, price, None, f"SELL signal (strength {_row_strength(row, action)})")
                continue
            # Initial stop loss (frozen at entry): exits before the slow
            # death-cross SELL signal catches up, limiting catastrophic losses.
            if params.stop_type != "none" and ticker in pf.stop_price:
                sp = pf.stop_price[ticker]
                if price <= sp:
                    pf.sell(ticker, price, None,
                            f"Initial stop: {price:.2f} <= {sp:.2f}")
                    continue
            # Trailing stop: sell if price has dropped trailing_stop_pct from
            # its peak since the position was opened.
            if params.trailing_stop_pct > 0:
                peak = pf.peak_price.get(ticker, price)
                stop_level = peak * (1 - params.trailing_stop_pct / 100)
                if price <= stop_level:
                    pf.sell(ticker, price, None,
                            f"Trailing stop: {price:.2f} <= {stop_level:.2f} (peak {peak:.2f}, -{params.trailing_stop_pct}%)")
                    continue
            if params.use_atr_stop:
                atr_stop = row["atr_stop"]
                if atr_stop is not None and price < atr_stop:
                    pf.sell(ticker, price, None, f"ATR stop hit: {price:.2f} < {atr_stop:.2f}")

        # Recompute equity after sells.
        total_equity = pf.equity(prices)
        min_cash = total_equity * (params.min_cash_pct / 100)
        max_position_value = total_equity * (params.max_position_pct / 100)

        # --- BUY phase ---
        # Regime filter: block new BUYs when the broad market is below its
        # 200-day SMA. SELLs and ATR stops still execute (we manage risk on
        # the way down, we just don't add new exposure).
        market_ok = True
        if params.regime_filter and regime is not None:
            market_ok = regime.get(day, True)

        buy_candidates = []
        if market_ok:
            buy_candidates = [
                (t, by_time[t][day]) for t in by_time
                if day in by_time[t] and _row_action(by_time[t][day], params) == "BUY"
            ]
            buy_candidates.sort(key=lambda x: _row_strength(x[1], "BUY"), reverse=True)

        if not buy_candidates and market_ok:
            hold_candidates = [
                (t, by_time[t][day]) for t in by_time
                if day in by_time[t]
                and _row_action(by_time[t][day], params) == "HOLD"
                and _row_strength(by_time[t][day], "HOLD") >= params.relaxed_hold_strength
            ]
            hold_candidates.sort(key=lambda x: _row_strength(x[1], "HOLD"), reverse=True)
            buy_candidates = hold_candidates[:params.relaxed_hold_limit]

        for ticker, row in buy_candidates:
            if pf.cash < min_cash:
                break
            price = row["close"]
            current_value = pf.positions.get(ticker, 0) * price
            if current_value >= max_position_value:
                continue

            # Sector diversification cap: limit total exposure per sector
            # to prevent correlated positions from concentrating risk.
            if params.max_sector_pct > 0:
                sector = _sector_of(ticker)
                sector_value = sum(
                    pf.positions.get(t, 0) * prices.get(t, 0)
                    for t in pf.positions
                    if _sector_of(t) == sector
                )
                sector_cap = total_equity * (params.max_sector_pct / 100)
                if sector_value >= sector_cap:
                    continue

            budget = min(pf.cash - min_cash, max_position_value - current_value)

            # Risk-based position sizing: size by stop distance so each
            # trade risks a fixed % of equity (industry-standard 1% rule).
            if params.risk_pct > 0 and params.stop_type != "none":
                atr = row.get("atr14")
                if params.stop_type == "atr" and atr and atr > 0:
                    stop = price - params.stop_atr_mult * atr
                elif params.stop_type == "percent":
                    stop = price * (1 - params.stop_pct / 100)
                else:
                    stop = 0
                if stop > 0 and price > stop:
                    risk_amount = total_equity * (params.risk_pct / 100)
                    risk_per_share = price - stop
                    if risk_per_share > 0:
                        budget_by_risk = (risk_amount / risk_per_share) * price
                        budget = min(budget, budget_by_risk)

            if budget < 1:
                continue

            # Compute frozen stop for this entry
            entry_stop = None
            if params.stop_type == "percent":
                entry_stop = price * (1 - params.stop_pct / 100)
            elif params.stop_type == "atr":
                atr = row.get("atr14")
                if atr and atr > 0:
                    entry_stop = price - params.stop_atr_mult * atr

            pf.buy(ticker, price, budget,
                   f"BUY signal (strength {_row_strength(row, 'BUY')})",
                   stop=entry_stop)

        # Record equity.
        total_equity = pf.equity(prices)
        equity_curve.append({"time": day, "equity": round(total_equity, 2)})
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
        max_drawdown_pct=_max_drawdown([e["equity"] for e in equity_curve]),
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
    # Each tuple: (stop_type, stop_pct, stop_atr_mult, risk_pct, max_sector_pct)
    risk_configs = [
        # No risk management (baseline)
        ("none", 0, 0, 0, 0),
        # 15% initial stop only
        ("percent", 15, 0, 0, 0),
        # 15% stop + 1% risk sizing
        ("percent", 15, 0, 1.0, 0),
        # 15% stop + 1% risk + 25% sector cap (best overall)
        ("percent", 15, 0, 1.0, 25),
        # 2x ATR stop + 1% risk
        ("atr", 0, 2.0, 1.0, 0),
        # 2x ATR stop + 1% risk + 25% sector cap
        ("atr", 0, 2.0, 1.0, 25),
    ]
    grid = []
    for buy in (40, 50):
        for sell in (-40, -30):
            for trail in (0.0, 15.0):
                for max_pos in (15, 20):
                    for stop_type, stop_pct, stop_atr, risk_pct, sector_pct in risk_configs:
                        grid.append(ReplayParams(
                            buy_threshold=buy, sell_threshold=sell,
                            relaxed_hold_strength=40,
                            max_position_pct=max_pos,
                            use_atr_stop=False,
                            trailing_stop_pct=trail,
                            regime_filter=regime_filter,
                            stop_type=stop_type,
                            stop_pct=stop_pct,
                            stop_atr_mult=stop_atr,
                            risk_pct=risk_pct,
                            max_sector_pct=sector_pct,
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
    """Composite score used to pick the 'best' params on a train window."""
    # Prefer higher return, penalize drawdown. Sharpe is secondary.
    return result.total_return_pct - 0.5 * result.max_drawdown_pct


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
                    "stop=%s risk=%.1f%% sector=%.0f%% | "
                    "train %+.1f%% → test %+.1f%% (sharpe %.2f, dd %.1f%%, %d trades)",
                    w_idx, total, elapsed,
                    best_params.buy_threshold, best_params.sell_threshold,
                    best_params.trailing_stop_pct, best_params.max_position_pct,
                    best_params.stop_type, best_params.risk_pct, best_params.max_sector_pct,
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
# CLI
# ---------------------------------------------------------------------------

def _print_result(res: ReplayResult, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"  Final equity:      ${res.final_equity:,.2f}")
    print(f"  Total return:      {res.total_return_pct:+.2f}%")
    print(f"  Sharpe (annual):   {res.sharpe:.2f}")
    print(f"  Max drawdown:      {res.max_drawdown_pct:.2f}%")
    print(f"  Trades:            {res.n_trades}")


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
        params = ReplayParams(regime_filter=getattr(args, "regime", False))
        res = _replay(series, params, start=args.start, end=args.end, regime=regime)
        label = "Backtest (current rules)"
        if params.regime_filter:
            label += " + regime filter"
        _print_result(res, label)
        if args.trades:
            for t in res.trades:
                print(f"  {t['side']:<4} {t['ticker']:<8} {t['shares']:>10.4f} @ {t['price']:>10.2f} — {t['reason']}")

    elif args.command == "sweep":
        regime_on = getattr(args, "regime", False)
        sweep = _run_sweep(series, args.start, args.end, regime=regime, regime_filter=regime_on)
        print(f"\n=== Parameter sweep ({args.start}..{args.end}) — top 10 ===")
        for params, res in sweep[:10]:
            print(f"  buy={params.buy_threshold:<3} sell={params.sell_threshold:<4} "
                  f"trail={params.trailing_stop_pct:>4.0f}% maxpos={params.max_position_pct:<3}% "
                  f"stop={params.stop_type:<7} risk={params.risk_pct:.1f}% sec={params.max_sector_pct:>2.0f}% "
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward optimization for the deterministic strategy")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backtest", help="Score the current rules on stored history")
    b.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start")
    b.add_argument("--end", default=None, help="YYYY-MM-DD inclusive end")
    b.add_argument("--trades", action="store_true", help="Print every trade")
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

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
