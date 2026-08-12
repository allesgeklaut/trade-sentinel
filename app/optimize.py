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
import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select

from .analysis import MIN_CANDLES
from .config import settings
from .db import Candle, Session
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.optimize")


# ---------------------------------------------------------------------------
# Vectorized indicator / signal series
# ---------------------------------------------------------------------------

def _signal_series(rows: list[dict]) -> pd.DataFrame:
    """Compute the per-day signal series for a ticker's candle history.

    Mirrors ``analysis.compute`` but returns a DataFrame with one row per
    candle (oldest-first) carrying the fields the deterministic strategy needs:
    action, strength, atr_stop, close. This lets the replay index into the
    series per-day instead of recomputing indicators every day.
    """
    d = pd.DataFrame(rows)
    c = d.close
    vol = d.volume

    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / down)

    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()

    prev = c.shift()
    tr = pd.concat(
        [d.high - d.low, (d.high - prev).abs(), (d.low - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()

    avg_vol_20 = vol.rolling(20).mean()
    vol_surge = (vol > 1.25 * avg_vol_20) & (avg_vol_20 > 0)

    trend_up = (c > d.sma50) & (d.sma50 > d.sma200)
    trend_down = (c < d.sma50) & (d.sma50 < d.sma200)
    sma50_rising = d.sma50 > d.sma50.shift(5)
    sma50_falling = d.sma50 < d.sma50.shift(5)

    rsi_now = d.rsi
    macd_bull = d.macd > d.macd_signal
    macd_bear = d.macd < d.macd_signal

    dist_above = (c / d.sma200 - 1) * 100
    dist_below = (1 - c / d.sma200) * 100

    bullish = pd.Series(0.0, index=d.index)
    bullish += np_where(trend_up, 30, np_where(c > d.sma50, 15, 0))
    bullish += np_where(sma50_rising, 15, 0)
    bullish += np_where(macd_bull, 15, 0)
    bullish += np_where((rsi_now >= 50) & (rsi_now <= 70), 20, np_where((rsi_now > 70) & (rsi_now <= 80), 10, 0))
    bullish += np_where(vol_surge & (c > d.sma50), 10, 0)
    bullish += np_where(dist_above > 5, 10, np_where(dist_above > 2, 5, 0))
    bullish = bullish.clip(upper=100)

    bearish = pd.Series(0.0, index=d.index)
    bearish += np_where(trend_down, 30, np_where(c < d.sma50, 15, 0))
    bearish += np_where(sma50_falling, 15, 0)
    bearish += np_where(macd_bear, 15, 0)
    bearish += np_where(rsi_now < 30, 20, np_where(rsi_now < 50, 10, 0))
    bearish += np_where(vol_surge & (c < d.sma50), 10, 0)
    bearish += np_where(dist_below > 5, 10, np_where(dist_below > 2, 5, 0))
    bearish = bearish.clip(upper=100)

    net = bullish - bearish

    atr_stop = c - 2 * d.atr14

    out = pd.DataFrame({
        "time": d["time"],
        "close": c,
        "net": net,
        "bullish": bullish,
        "bearish": bearish,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "dist_above": dist_above,
        "dist_below": dist_below,
        "atr_stop": atr_stop,
    })
    return out


def np_where(cond, a, b):
    """Element-wise where that tolerates NaN conditions (treats NaN as False)."""
    import numpy as np
    return pd.Series(np.where(cond.fillna(False), a, b), index=cond.index)


# ---------------------------------------------------------------------------
# Paper portfolio
# ---------------------------------------------------------------------------

@dataclass
class PaperPortfolio:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # ticker -> shares
    avg_cost: dict[str, float] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)

    def buy(self, ticker: str, price: float, budget: float, reason: str) -> None:
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
        self.trades.append({"ticker": ticker, "side": "SELL", "shares": sell_shares,
                            "price": price, "reason": reason})

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
    monthly_allowance: float = settings.sim_monthly_allowance
    start_cash: float = settings.sim_start_cash


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
        for t in tickers:
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


def _candidate_tickers() -> list[str]:
    if settings.sim_universe.lower() == "watchlist":
        # Watchlist lives in the DB; fall back to universe file for the tool.
        logger.warning("sim_universe=watchlist; optimize uses universe file instead")
    return universe_tickers(settings.sim_universe)


def _row_action(row: dict, params: ReplayParams) -> str:
    """Derive the BUY/SELL/HOLD action for a row under the given thresholds."""
    if row["net"] >= params.buy_threshold and row["trend_up"] and row["dist_above"] > 2:
        return "BUY"
    if row["net"] <= params.sell_threshold and row["trend_down"] and row["dist_below"] > 2:
        return "SELL"
    return "HOLD"


def _row_strength(row: dict, action: str) -> int:
    """Derive the 0-100 strength for a row under the given action."""
    if action == "BUY":
        return int(round(row["bullish"]))
    if action == "SELL":
        return int(round(row["bearish"]))
    return int(round(max(row["bullish"], row["bearish"])))


def _replay(series: dict[str, pd.DataFrame], params: ReplayParams,
            start: str | None = None, end: str | None = None) -> ReplayResult:
    """Run the deterministic strategy over the precomputed series.

    ``start``/``end`` are inclusive date strings (YYYY-MM-DD) used to bound
    the replay window (e.g. a walk-forward test window).
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

        # --- SELL phase ---
        for ticker in list(pf.positions.keys()):
            row = by_time.get(ticker, {}).get(day)
            if row is None:
                continue
            price = row["close"]
            action = _row_action(row, params)
            if action == "SELL":
                pf.sell(ticker, price, None, f"SELL signal (strength {_row_strength(row, action)})")
            elif params.use_atr_stop:
                atr_stop = row["atr_stop"]
                if atr_stop is not None and price < atr_stop:
                    pf.sell(ticker, price, None, f"ATR stop hit: {price:.2f} < {atr_stop:.2f}")

        # Recompute equity after sells.
        total_equity = pf.equity(prices)
        min_cash = total_equity * (params.min_cash_pct / 100)
        max_position_value = total_equity * (params.max_position_pct / 100)

        # --- BUY phase ---
        buy_candidates = [
            (t, by_time[t][day]) for t in by_time
            if day in by_time[t] and _row_action(by_time[t][day], params) == "BUY"
        ]
        buy_candidates.sort(key=lambda x: _row_strength(x[1], "BUY"), reverse=True)

        if not buy_candidates:
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
            budget = min(pf.cash - min_cash, max_position_value - current_value)
            if budget < 1:
                continue
            pf.buy(ticker, price, budget, f"BUY signal (strength {_row_strength(row, 'BUY')})")

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

def _param_grid() -> list[ReplayParams]:
    """A small grid over the tunable thresholds."""
    grid = []
    for buy in (30, 40, 50):
        for sell in (-50, -40, -30):
            for relaxed in (35, 40, 45):
                grid.append(ReplayParams(
                    buy_threshold=buy, sell_threshold=sell,
                    relaxed_hold_strength=relaxed,
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


def _run_sweep(series: dict[str, pd.DataFrame], start: str, end: str) -> list[tuple[ReplayParams, ReplayResult]]:
    results = []
    for params in _param_grid():
        res = _replay(series, params, start=start, end=end)
        results.append((params, res))
    results.sort(key=lambda x: _score(x[1]), reverse=True)
    return results


def _walk_forward(series: dict[str, pd.DataFrame], days: list[str],
                  train_days: int, test_days: int) -> list[dict]:
    """Run walk-forward: fit best params on each train window, score on test."""
    windows = []
    for train_s, train_e, test_s, test_e in _split_windows(days, train_days, test_days):
        sweep = _run_sweep(series, train_s, train_e)
        best_params, _ = sweep[0]
        test_res = _replay(series, best_params, start=test_s, end=test_e)
        windows.append({
            "train": f"{train_s}..{train_e}",
            "test": f"{test_s}..{test_e}",
            "best_params": {
                "buy_threshold": best_params.buy_threshold,
                "sell_threshold": best_params.sell_threshold,
                "relaxed_hold_strength": best_params.relaxed_hold_strength,
            },
            "train_return_pct": round(_score(sweep[0][1]), 2),
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

    if args.command == "backtest":
        res = _replay(series, ReplayParams(), start=args.start, end=args.end)
        _print_result(res, "Backtest (current rules)")
        if args.trades:
            for t in res.trades:
                print(f"  {t['side']:<4} {t['ticker']:<8} {t['shares']:>10.4f} @ {t['price']:>10.2f} — {t['reason']}")

    elif args.command == "sweep":
        sweep = _run_sweep(series, args.start, args.end)
        print(f"\n=== Parameter sweep ({args.start}..{args.end}) — top 10 ===")
        for params, res in sweep[:10]:
            print(f"  buy={params.buy_threshold:<3} sell={params.sell_threshold:<4} "
                  f"relaxed={params.relaxed_hold_strength:<3} "
                  f"→ ret {res.total_return_pct:+7.2f}%  dd {res.max_drawdown_pct:5.2f}%  "
                  f"sharpe {res.sharpe:5.2f}  trades {res.n_trades}")

    elif args.command == "walkforward":
        # Bound the walk to the requested window (defaults to full history).
        walk_days = all_days
        if args.start:
            walk_days = [d for d in walk_days if d >= args.start]
        if args.end:
            walk_days = [d for d in walk_days if d <= args.end]
        windows = _walk_forward(series, walk_days, args.train_days, args.test_days)
        print(f"\n=== Walk-forward (train {args.train_days}d / test {args.test_days}d) ===")
        for w in windows:
            print(f"  train {w['train']} → test {w['test']}: "
                  f"params {w['best_params']} | train {w['train_return_pct']:+6.2f}% | "
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

    s = sub.add_parser("sweep", help="Grid-search thresholds")
    s.add_argument("--start", default=None)
    s.add_argument("--end", default=None)

    w = sub.add_parser("walkforward", help="Walk-forward fit/test")
    w.add_argument("--train-days", type=int, default=504)
    w.add_argument("--test-days", type=int, default=126)
    w.add_argument("--start", default=None, help="YYYY-MM-DD inclusive start of the walk")
    w.add_argument("--end", default=None, help="YYYY-MM-DD inclusive end of the walk")

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
