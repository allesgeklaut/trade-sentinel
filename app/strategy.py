"""Shared strategy logic for the live sim and the backtest replay.

Both ``app.sim`` (live paper-trading engine, DB-backed) and ``app.optimize``
(backtest / hybrid replay, in-memory) need to:

  - propose deterministic trades (SELL signals + stops, ranked BUYs)
  - reconcile deterministic proposals with LLM decisions (veto / approve / flip)
  - valuate a portfolio (cash + positions at current prices)

Previously these were duplicated across the two modules with subtle drift
(e.g. the max-positions cap blocked top-ups in both copies, fixed in two
places). This module is the single source of truth for that logic. Both
callers pass in plain data (dicts / lists) — no DB or PaperPortfolio types
leak across the boundary.

The input shapes:

  - ``positions``: list of ``{"ticker": str, "shares": float, "avg_cost": float,
    "stop_price": float | None}`` (``stop_price`` is the frozen initial stop,
    or None to recompute from avg_cost each cycle — the sim's old behaviour).
  - ``cash``: float
  - ``prices``: ``{ticker: close_price}`` for the current day
  - ``signals``: ``{ticker: {"action": "BUY"|"SELL"|"HOLD", "strength": int,
    "reason": str, "snapshot": {"atr_stop": float|None, "run_5d": float|None,
    "atr14": float|None, ...}}}`` — same shape as ``analysis.compute()`` /
    ``optimize._signals_for_day`` return.
  - ``params``: a ``StrategyParams`` instance ( thresholds + risk config ).

The output shape (proposals):

  - SELL: ``{"ticker", "side": "SELL", "price", "shares": None, "reason"}``
  - BUY:  ``{"ticker", "side": "BUY",  "price", "budget", "reason",
    "entry_stop": float|None}``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StrategyParams:
    """Thresholds + risk config for the deterministic strategy.

    Both the live sim (constructed from ``settings.*``) and the backtest
    replay (constructed from ``ReplayParams``) produce this. Keeping it a
    plain dataclass means neither module's types leak across the boundary.
    """
    buy_threshold: int = 40
    sell_threshold: int = -40
    min_cash_pct: float = 5.0
    max_position_pct: float = 10.0
    max_positions: int = 10
    stop_type: str = "percent"   # "none" | "percent" | "atr"
    stop_pct: float = 15.0       # percentage drop from entry (stop_type="percent")
    stop_atr_mult: float = 2.0   # ATR multiple (stop_type="atr")
    use_atr_stop: bool = True
    max_run_5d: float = 12.0     # block BUYs after a 5-day run-up > this % (0 = disabled)
    relaxed_hold_strength: int = 40
    relaxed_hold_limit: int = 3
    # Sector diversification cap: max % of equity in any one sector. 0 = disabled.
    # When enabled, the caller must pass a ``sector_of`` callable to propose_trades.
    max_sector_pct: float = 0.0
    # Risk-based position sizing: risk this % of equity per trade, sized by
    # stop distance. 0 = use flat max_position_pct.
    risk_pct: float = 0.0
    relaxed_hold_strength: int = 40
    relaxed_hold_limit: int = 3


def valuate_portfolio(
    positions: list[dict],
    cash: float,
    prices: dict[str, float],
    allowance_total: float,
) -> dict[str, Any]:
    """Build the ``valuation`` dict the LLM context builder expects.

    Mirrors ``sim.valuate()`` but operates on plain data instead of reading
    the DB. Returns ``{cash, positions, positions_value, total_equity,
    allowance_total}`` where each position carries ``current_price``,
    ``value``, and ``pnl_pct``.
    """
    pos_list: list[dict] = []
    positions_value = 0.0
    for p in positions:
        price = prices.get(p["ticker"], 0.0)
        value = p["shares"] * price
        positions_value += value
        avg_cost = p.get("avg_cost", 0.0)
        pnl_pct = ((price / avg_cost - 1) * 100) if avg_cost > 0 else 0.0
        pos_list.append({
            "ticker": p["ticker"], "shares": p["shares"],
            "avg_cost": avg_cost, "current_price": price,
            "value": value, "pnl_pct": pnl_pct,
        })
    return {
        "cash": cash,
        "positions": pos_list,
        "positions_value": positions_value,
        "total_equity": cash + positions_value,
        "allowance_total": allowance_total,
    }


def propose_trades(
    positions: list[dict],
    cash: float,
    prices: dict[str, float],
    signals: dict[str, dict],
    params: StrategyParams,
    sector_of: callable | None = None,
) -> list[dict]:
    """Propose deterministic trades WITHOUT executing them.

    Single source of truth for the SELL + BUY logic. Both the live sim
    (DB-backed) and the backtest replay (in-memory) call this with plain
    data; the caller is responsible for executing the returned proposals.

    SELL phase (per held position, in order):
      1. Signal SELL → propose SELL.
      2. Initial stop (frozen at entry if ``stop_price`` is set on the
         position, else recomputed from ``avg_cost * (1 - stop_pct/100)``).
      3. ATR trailing stop (from the signal's snapshot).

    BUY phase (ranked by strength, strict BUYs first, else relaxed HOLD
    fallback):
      - ``max_run_5d`` blocks BUYs after a short-term spike.
      - ``max_positions`` cap blocks NEW positions but allows topping up
        tickers already held.
      - ``max_position_pct`` ceiling prevents over-concentration.
      - ``min_cash_pct`` floor preserves a cash buffer.
      - Local cash reservation so successive BUY proposals in the same cycle
        see reduced cash.
    """
    proposals: list[dict] = []
    total_equity = cash + sum(p["shares"] * prices.get(p["ticker"], 0) for p in positions)
    if total_equity <= 0:
        return proposals

    min_cash = total_equity * (params.min_cash_pct / 100)
    max_position_value = total_equity * (params.max_position_pct / 100)
    held_tickers = {p["ticker"] for p in positions}
    open_count = len(held_tickers)
    sim_cash = cash  # local copy; decremented as BUYs are proposed

    # Track sector exposure progressively (for the sector cap).
    sector_values: dict[str, float] = {}
    if params.max_sector_pct > 0 and sector_of:
        for p in positions:
            sec = sector_of(p["ticker"])
            sector_values[sec] = sector_values.get(sec, 0) + p["shares"] * prices.get(p["ticker"], 0)

    # --- SELL phase ---
    for pos in positions:
        ticker = pos["ticker"]
        sig = signals.get(ticker)
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue

        if sig and sig["action"] == "SELL":
            proposals.append({"ticker": ticker, "side": "SELL",
                              "price": price, "shares": None,
                              "reason": sig.get("reason", "SELL signal")})
            continue

        # Initial stop loss: use frozen stop_price if available, else
        # recompute from avg_cost (the sim's old behaviour).
        if params.stop_type != "none":
            if pos.get("stop_price") is not None:
                sp = pos["stop_price"]
            elif params.stop_type == "percent":
                sp = pos.get("avg_cost", 0) * (1 - params.stop_pct / 100)
            else:
                sp = None
            if sp is not None and price <= sp:
                proposals.append({"ticker": ticker, "side": "SELL",
                                  "price": price, "shares": None,
                                  "reason": f"Initial stop: {price:.2f} <= {sp:.2f}"})
                continue

        # ATR trailing stop
        if params.use_atr_stop and sig:
            snap = sig.get("snapshot", {})
            atr_stop = snap.get("atr_stop")
            if atr_stop and price < atr_stop:
                proposals.append({"ticker": ticker, "side": "SELL",
                                  "price": price, "shares": None,
                                  "reason": f"ATR stop hit: {price:.2f} < {atr_stop:.2f}"})

    # --- BUY phase ---
    max_run_5d = params.max_run_5d
    buy_candidates = [
        (t, sig) for t, sig in signals.items()
        if sig["action"] == "BUY"
        and not (max_run_5d > 0
                 and sig.get("snapshot", {}).get("run_5d") is not None
                 and sig["snapshot"]["run_5d"] > max_run_5d)
    ]
    buy_candidates.sort(key=lambda x: x[1]["strength"], reverse=True)

    # Relaxed fallback: buy the best near-BUY (HOLD with high strength) so
    # the bot stays active and deploys cash instead of sitting idle.
    if not buy_candidates:
        hold_candidates = [
            (t, sig) for t, sig in signals.items()
            if sig["action"] == "HOLD" and sig["strength"] >= params.relaxed_hold_strength
            and not (max_run_5d > 0
                     and sig.get("snapshot", {}).get("run_5d") is not None
                     and sig["snapshot"]["run_5d"] > max_run_5d)
        ]
        hold_candidates.sort(key=lambda x: x[1]["strength"], reverse=True)
        buy_candidates = hold_candidates[:params.relaxed_hold_limit]

    for ticker, sig in buy_candidates:
        if sim_cash < min_cash:
            break  # not enough cash to keep buffer

        # Max open positions: block NEW positions when at the cap, but still
        # allow topping up tickers already held (a top-up doesn't open a new
        # position, and the max_position_pct ceiling below prevents
        # over-concentration in a single name).
        if (params.max_positions > 0
                and open_count >= params.max_positions
                and ticker not in held_tickers):
            continue

        price = prices.get(ticker)
        if price is None or price <= 0:
            continue

        current_shares = next((p["shares"] for p in positions if p["ticker"] == ticker), 0)
        current_value = current_shares * price
        if current_value >= max_position_value:
            continue

        # Sector diversification cap: limit total exposure per sector.
        # Tracked progressively so the 2nd same-sector BUY sees the 1st one.
        if params.max_sector_pct > 0 and sector_of:
            sec = sector_of(ticker)
            sector_cap = total_equity * (params.max_sector_pct / 100)
            if sector_values.get(sec, 0) >= sector_cap:
                continue

        budget = min(sim_cash - min_cash, max_position_value - current_value)

        # Risk-based position sizing: size by stop distance so each
        # trade risks a fixed % of equity (industry-standard 1% rule).
        if params.risk_pct > 0 and params.stop_type != "none":
            atr = sig.get("snapshot", {}).get("atr14")
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

        # Frozen entry stop for the new proposal
        entry_stop = None
        if params.stop_type == "percent":
            entry_stop = price * (1 - params.stop_pct / 100)
        elif params.stop_type == "atr":
            atr = sig.get("snapshot", {}).get("atr14")
            if atr and atr > 0:
                entry_stop = price - params.stop_atr_mult * atr

        proposals.append({"ticker": ticker, "side": "BUY",
                          "price": price, "budget": budget,
                          "reason": sig.get("reason", "BUY signal"),
                          "entry_stop": entry_stop})
        # Reserve the budget locally so the next BUY proposal sees reduced
        # cash. Only bump open_count for NEW positions — a top-up doesn't
        # change the position count. Track sector exposure progressively.
        sim_cash -= budget
        if ticker not in held_tickers:
            open_count += 1
        if params.max_sector_pct > 0 and sector_of:
            sec = sector_of(ticker)
            sector_values[sec] = sector_values.get(sec, 0) + budget

    return proposals


def llm_buy_budget(
    decision: dict,
    cash: float,
    equity: float,
    price: float,
    current_value: float,
    params: StrategyParams,
    guarded: bool,
) -> float | None:
    """Budget for an LLM BUY decision, or None to skip.

    ``price`` is the execution price, ``current_value`` the value already
    held in this ticker. ``guarded=True`` applies the hard risk limits (min
    cash floor and max position %) — used in hybrid mode, matching the
    deterministic engine. ``guarded=False`` lets the LLM decide sizing and
    cash reserve; the only hard bounds are the actual cash balance and the
    $1 minimum. In both modes the optional partial-size fields (``shares`` /
    ``amount``) clamp the budget.
    """
    min_cash = equity * (params.min_cash_pct / 100)
    max_position_value = equity * (params.max_position_pct / 100)

    if guarded:
        if cash < min_cash:
            return None
        if current_value >= max_position_value:
            return None
        budget = min(cash - min_cash, max_position_value - current_value)
    else:
        budget = cash

    if "shares" in decision:
        budget = min(budget, decision["shares"] * price)
    elif "amount" in decision:
        budget = min(budget, decision["amount"])

    if budget < 1:
        return None
    return budget


def llm_sell_shares(decision: dict, price: float) -> float | None:
    """Shares to sell for an LLM SELL decision (None = entire position)."""
    if "shares" in decision:
        return float(decision["shares"])
    if "amount" in decision:
        return float(decision["amount"]) / price if price > 0 else None
    return None


def reconcile_proposals(
    proposals: list[dict],
    decisions: list[dict],
) -> tuple[list[dict], list[dict], set[str]]:
    """Reconcile deterministic proposals with LLM decisions.

    Returns ``(approved, vetoed, proposal_tickers)`` where:
      - ``approved``: proposals the LLM agreed with (or didn't mention —
        silence = consent). These should be executed.
      - ``vetoed``: proposals the LLM blocked (HOLD) or flipped (e.g. BUY→SELL).
        These should NOT be executed. Each carries ``llm_reason``.
      - ``proposal_tickers``: the set of tickers (uppercase) that were in the
        proposals. The caller uses this to identify LLM additions (decisions
        for tickers NOT in this set).

    A flip (LLM returns a different action than the proposal) is treated as a
    veto of the proposal; the flipped action is handled by the caller as an
    LLM addition (it's in ``decisions`` but not in ``proposal_tickers``).
    """
    llm_by_ticker = {d["ticker"].upper(): d for d in decisions}
    proposal_tickers: set[str] = set()
    approved: list[dict] = []
    vetoed: list[dict] = []

    for p in proposals:
        tu = p["ticker"].upper()
        proposal_tickers.add(tu)
        llm_d = llm_by_ticker.get(tu)

        if llm_d is None:
            # LLM didn't comment on this proposal — silence = consent.
            approved.append(p)
        elif llm_d["action"] == "HOLD":
            vetoed.append({**p, "llm_reason": llm_d.get("reason", "")})
        elif llm_d["action"] == p["side"]:
            approved.append(p)
        else:
            # Flip: veto the proposal; the flipped action is an LLM addition.
            vetoed.append({**p, "llm_reason": llm_d.get("reason", "")})

    return approved, vetoed, proposal_tickers