"""Sweep the LLM sell-gating over a replay window to measure equity impact.

Runs the deterministic replay, probes the LLM on every SELL-signal trade,
then re-runs the replay with LLM-gated SELLs suppressed where the LLM said
HOLD. Compares return, drawdown, and Sharpe between pure-deterministic and
LLM-gated modes.

Run inside the container:

    docker compose exec trade-sentinel /app/.venv/bin/python -m app.llm_sweep \
        --start 2025-01-01 --end 2026-07-15
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import json
from typing import Any

import pandas as pd

from .config import settings
from . import llm as llm_mod
from .optimize import (
    ReplayParams,
    _candidate_tickers,
    _is_stop_out,
    _live_sim_params,
    _load_series,
    _max_drawdown,
    _reconstruct_portfolio_states,
    _replay,
    _score,
    _sharpe,
    _snapshot_for_day,
    TradeCase,
    _build_llm_probe_context,
)
from .sim import _LLM_SYSTEM_PROMPT, _parse_llm_decisions

logger = logging.getLogger("trade_sentinel.llm_sweep")


async def _probe_sell(
    ticker: str,
    date: str,
    reason: str,
    series: dict[str, pd.DataFrame],
    portfolio_state: dict[str, Any],
    params: ReplayParams,
) -> str:
    """Probe the LLM on a single SELL-signal decision. Returns 'SELL' or 'HOLD'."""
    from .optimize import TradeCase

    df = series.get(ticker)
    if df is None:
        return "SELL"  # can't probe, let the deterministic decision stand

    try:
        snapshot = _snapshot_for_day(df, date)
    except KeyError:
        return "SELL"

    # Find the position in this ticker from the portfolio state
    pos = next(
        ({"shares": p["shares"], "avg_cost": p["avg_cost"]}
         for p in portfolio_state["positions"] if p["ticker"] == ticker),
        None,
    )

    case = TradeCase(
        date=date, ticker=ticker, det_side="SELL",
        det_reason=reason, entry_price=snapshot["snapshot"].get("close", 0),
        forward_close=None, outcome_pct=0, badness=0,
        position_before=pos, portfolio_state=portfolio_state,
    )

    context = _build_llm_probe_context(case, snapshot, params=params)

    system_prompt = _LLM_SYSTEM_PROMPT
    if _is_stop_out(reason):
        system_prompt += (
            "\n\nBENCHMARK OVERRIDE: rule 4 is relaxed for this probe. The "
            "stop is about to fire but has NOT executed yet — you may answer "
            "HOLD to override it and keep the position, or SELL to let it "
            "execute."
        )

    try:
        out = await llm_mod.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ])
    except Exception as e:
        logger.warning("LLM probe failed for %s %s: %s", ticker, date, e)
        return "SELL"

    content = out["text"]
    decisions = _parse_llm_decisions(content)
    if decisions:
        for d in decisions:
            if d["ticker"].upper() == ticker.upper():
                return d["action"]
        if len(decisions) == 1:
            return decisions[0]["action"]
    return "SELL"


async def run_llm_sweep(start: str, end: str) -> None:
    """Run the LLM sell-gating sweep and print a comparison report."""
    tickers = _candidate_tickers()
    logger.info("Loading series for %d tickers...", len(tickers))
    series = await _load_series(tickers)
    logger.info("Loaded series for %d tickers", len(series))

    params = _live_sim_params()
    # Use the live max_run_5d setting
    params.max_run_5d = settings.sim_max_run_5d

    # 1. Run the pure deterministic replay
    logger.info("Running deterministic replay (%s..%s)...", start, end)
    det_res = _replay(series, params, start=start, end=end)
    print(f"\n=== Deterministic replay ({start}..{end}) ===")
    print(f"  Return: {det_res.total_return_pct:+.2f}%  DD: {det_res.max_drawdown_pct:.2f}%  "
          f"Sharpe: {det_res.sharpe:.2f}  Trades: {det_res.n_trades}")
    print(f"  Risk config: max_run_5d={params.max_run_5d}%, stop={params.stop_type} {params.stop_pct}%")

    # 2. Find all SELL-signal trades (not stop-outs — those stay automatic)
    sell_trades = [
        (i, t) for i, t in enumerate(det_res.trades)
        if t["side"] == "SELL" and not _is_stop_out(t["reason"])
    ]
    print(f"\n  SELL-signal trades to gate: {len(sell_trades)}")
    stop_trades = [
        t for t in det_res.trades
        if t["side"] == "SELL" and _is_stop_out(t["reason"])
    ]
    print(f"  Stop-out trades (auto, not gated): {len(stop_trades)}")

    if not sell_trades:
        print("  No SELL-signal trades to gate. Done.")
        return

    if not (settings.llm_backends or settings.ollama_model):
        print("\nNo LLM backend configured (set LLM_BACKENDS or OLLAMA_MODEL); cannot probe the LLM.")
        return

    # 3. Reconstruct portfolio states at each trade
    portfolio_states = _reconstruct_portfolio_states(det_res.trades)

    # 4. Probe the LLM on each SELL-signal trade
    active = await llm_mod.current_backend()
    print(f"\n=== Probing LLM ({active.get('name', '?')} · {active.get('model', '?')}) on {len(sell_trades)} SELLs ... ===")
    suppress: set[int] = set()  # trade indices to suppress (LLM said HOLD)
    for idx, (trade_idx, trade) in enumerate(sell_trades, 1):
        ticker = trade["ticker"]
        date = trade.get("date", "")
        reason = trade["reason"]
        state = portfolio_states[trade_idx]
        print(f"  [{idx}/{len(sell_trades)}] {ticker} {date} (strength in reason) ...", end=" ", flush=True)
        llm_action = await _probe_sell(ticker, date, reason, series, state, params)
        print(llm_action)
        if llm_action == "HOLD":
            suppress.add(trade_idx)

    print(f"\n  LLM suppressed {len(suppress)}/{len(sell_trades)} SELLs")

    # 5. Re-run the replay with suppressed SELLs removed from the trade list.
    # We do this by running a modified replay where we skip the SELL phase
    # for suppressed trades. The simplest approach: re-run the replay and
    # post-filter the trades, then recompute the equity curve from the
    # remaining trades.
    gated_trades = [
        t for i, t in enumerate(det_res.trades)
        if i not in suppress
    ]

    # Re-simulate: walk the gated trades chronologically, building a new
    # equity curve. We reuse PaperPortfolio for this.
    from .optimize import PaperPortfolio

    pf = PaperPortfolio(cash=params.start_cash)
    equity_curve: list[dict] = []
    invested_curve: list[float] = []
    daily_returns: list[float] = []
    prev_equity: float | None = None
    last_deposit_month: str | None = None
    cumulative_invested = params.start_cash

    # Build by_time for price lookups
    by_time: dict[str, dict[str, float]] = {}
    for t, df in series.items():
        by_time[t] = {row.time: float(row.close) for row in df.itertuples(index=False)}

    all_days = sorted(set().union(*(set(df["time"]) for df in series.values())))
    all_days = [d for d in all_days if start <= d <= end]

    # Build a set of (ticker, date) for gated trades to execute
    gated_buy = {(t["ticker"], t["date"]) for t in gated_trades if t["side"] == "BUY"}
    gated_sell = {(t["ticker"], t["date"]) for t in gated_trades if t["side"] == "SELL"}

    last_known_prices: dict[str, float] = {}

    for day in all_days:
        month = day[:7]
        if month != last_deposit_month:
            pf.cash += params.monthly_allowance
            cumulative_invested += params.monthly_allowance
            last_deposit_month = month

        # Carry-forward prices
        prices: dict[str, float] = {}
        for t, idx in by_time.items():
            if day in idx:
                p = idx[day]
                prices[t] = p
                last_known_prices[t] = p
            elif t in last_known_prices:
                prices[t] = last_known_prices[t]

        # Execute gated trades for this day
        for tr in gated_trades:
            if tr.get("date") != day:
                continue
            if tr["side"] == "BUY":
                # Recompute budget from current state
                total_equity = pf.equity(prices)
                if total_equity <= 0:
                    continue
                min_cash = total_equity * (params.min_cash_pct / 100)
                max_position_value = total_equity * (params.max_position_pct / 100)
                current_value = pf.positions.get(tr["ticker"], 0) * prices.get(tr["ticker"], 0)
                if current_value >= max_position_value:
                    continue
                if len(pf.positions) >= params.max_positions and tr["ticker"] not in pf.positions:
                    continue
                budget = min(pf.cash - min_cash, max_position_value - current_value)
                if budget < 1:
                    continue
                entry_stop = None
                if params.stop_type == "percent":
                    entry_stop = tr["price"] * (1 - params.stop_pct / 100)
                pf.buy(tr["ticker"], tr["price"], budget, tr["reason"],
                       stop=entry_stop, date=day)
            else:  # SELL
                pf.sell(tr["ticker"], tr["price"], None, tr["reason"], date=day)

        total_equity = pf.equity(prices)
        equity_curve.append({"time": day, "equity": round(total_equity, 2)})
        invested_curve.append(cumulative_invested)
        if prev_equity is not None and prev_equity > 0:
            daily_returns.append(total_equity / prev_equity - 1)
        prev_equity = total_equity

    final_equity = equity_curve[-1]["equity"] if equity_curve else 0.0
    total_invested = cumulative_invested
    gated_return = (final_equity / total_invested - 1) * 100 if total_invested > 0 else 0.0
    gated_dd = _max_drawdown([e["equity"] for e in equity_curve], invested_curve)
    gated_sharpe = _sharpe(daily_returns)
    gated_n_trades = len(gated_trades)

    # 6. Report
    print(f"\n=== LLM sell-gated replay ({start}..{end}) ===")
    print(f"  Return: {gated_return:+.2f}%  DD: {gated_dd:.2f}%  "
          f"Sharpe: {gated_sharpe:.2f}  Trades: {gated_n_trades}")

    gated_score = gated_return - 2.0 * gated_dd
    det_score = _score(det_res)

    print(f"\n=== Comparison ===")
    print(f"  {'':>20} {'deterministic':>14} {'LLM-gated':>14} {'delta':>10}")
    print(f"  {'Return':>20} {det_res.total_return_pct:>+13.2f}% {gated_return:>+13.2f}% {gated_return - det_res.total_return_pct:>+9.2f}%")
    print(f"  {'Max drawdown':>20} {det_res.max_drawdown_pct:>13.2f}% {gated_dd:>13.2f}% {gated_dd - det_res.max_drawdown_pct:>+9.2f}%")
    print(f"  {'Sharpe':>20} {det_res.sharpe:>14.2f} {gated_sharpe:>14.2f} {gated_sharpe - det_res.sharpe:>+10.2f}")
    print(f"  {'Trades':>20} {det_res.n_trades:>14} {gated_n_trades:>14} {gated_n_trades - det_res.n_trades:>+10}")
    print(f"  {'Score (ret-2dd)':>20} {det_score:>+14.1f} {gated_score:>+14.1f} {gated_score - det_score:>+10.1f}")

    # Show which SELLs were suppressed
    if suppress:
        print(f"\n  Suppressed SELLs (LLM said HOLD):")
        for i, t in enumerate(det_res.trades):
            if i in suppress:
                print(f"    {t['date']} {t['ticker']:<10} @ {t['price']:.2f} — {t['reason'][:50]}")

    # Save reasoning
    out_path = "/tmp/llm_sweep_result.json"
    try:
        with open(out_path, "w") as f:
            json.dump({
                "window": f"{start}..{end}",
                "deterministic": {
                    "return_pct": det_res.total_return_pct,
                    "max_dd_pct": det_res.max_drawdown_pct,
                    "sharpe": det_res.sharpe,
                    "n_trades": det_res.n_trades,
                },
                "llm_gated": {
                    "return_pct": gated_return,
                    "max_dd_pct": gated_dd,
                    "sharpe": gated_sharpe,
                    "n_trades": gated_n_trades,
                },
                "suppressed_sells": [
                    {"ticker": det_res.trades[i]["ticker"],
                     "date": det_res.trades[i]["date"],
                     "reason": det_res.trades[i]["reason"]}
                    for i in suppress
                ],
            }, f, indent=2)
        print(f"\n  Results saved to {out_path}")
    except OSError as e:
        print(f"  (could not save: {e})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="LLM sell-gating sweep: measure equity impact")
    parser.add_argument("--start", default="2025-01-01", help="YYYY-MM-DD inclusive start")
    parser.add_argument("--end", default="2026-07-15", help="YYYY-MM-DD inclusive end")
    args = parser.parse_args()
    asyncio.run(run_llm_sweep(args.start, args.end))


if __name__ == "__main__":
    main()