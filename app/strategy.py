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
    "atr14": float|None, "net_score": int|None, "trend_up": bool,
    "dist_above": float|None, "weekly_trend_up": bool, ...}}}`` — same shape
    as ``analysis.compute()`` / ``optimize._signals_for_day`` return.
  - ``params``: a ``StrategyParams`` instance ( thresholds + risk config ).

The output shape (proposals):

  - SELL: ``{"ticker", "side": "SELL", "price", "shares": None, "reason"}``
  - BUY:  ``{"ticker", "side": "BUY",  "price", "budget", "reason",
    "entry_stop": float|None}``

Stop semantics:

  - ``stop_type="percent"`` — the entry-anchored control. In the backtest
    replay the level is frozen at entry (``stop_price`` on the position);
    the live sim stores no entry stop and recomputes it from the position's
    blended average cost on every cycle.
  - ``atr_stop`` (from the signal snapshot) — a **prior-known indicator
    candidate**: the highest close of the previous 20 completed bars minus
    2x ATR of the previous completed bar (see ``analysis.signal_series``).
    It is recomputed per bar and can move down as well as up, so it is
    NOT a persistent high-water-mark ratchet and NOT frozen at entry. It is
    nevertheless a real, reachable exit level: today's close is compared
    against a level that was already known before today's bar closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any

from .analysis import entry_eligible


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
    # Mirror of max_run_5d on the downside: block BUYs into a 5-day crash
    # (bearish-bounce scoring reads falling knives as high-strength entries;
    # pooled 21d forward returns of crash entries: -30% vs +7% for the
    # rising tier). 0 = disabled.
    min_run_5d: float = -15.0
    # Block BUYs into parabolic extensions: dist_above SMA200 > this %.
    # Entries above +100% averaged -13.2% forward vs +10% in the 20-50%
    # sweet spot; 4 of 10 one-year stop-outs entered above +50%. The scoring
    # *rewards* extension (+10 for dist_above > 5), so this guard closes the
    # hole the score can't see. 0 = disabled.
    max_dist_above: float = 80.0
    relaxed_hold_strength: int = 40
    relaxed_hold_limit: int = 3
    # Sector diversification cap: max % of equity in any one sector. 0 = disabled.
    # When enabled, the caller must pass a ``sector_of`` callable to propose_trades.
    max_sector_pct: float = 0.0
    # Risk-based position sizing: risk this % of equity per trade, sized by
    # stop distance. 0 = use flat max_position_pct.
    risk_pct: float = 0.0
    # Block BUY entries for names with KNOWN non-positive ROE (quality guard).
    # Missing fundamentals are neutral (ETFs like GLD, thin coverage) — only
    # known-bad data blocks. 0 = disabled (default until the replay A/B).
    block_negative_roe: bool = False


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
            "thesis": p.get("thesis", ""), "buy_date": p.get("buy_date", ""),
        })
    return {
        "cash": cash,
        "positions": pos_list,
        "positions_value": positions_value,
        "total_equity": cash + positions_value,
        "allowance_total": allowance_total,
    }


def _fallback_eligible(sig: dict, params: StrategyParams) -> tuple[bool, str]:
    """Can this HOLD be entered through the relaxed fallback?

    Returns ``(eligible, rationale)``. The rationale explains *why* the
    candidate qualifies (or, when it does not, why it is blocked).

    The relaxed fallback exists to capture a **near-threshold** BUY: the
    signal is a HOLD only because the net score did not reach the BUY
    threshold, while every trend gate already passed. Two conditions
    therefore have to hold:

      1. the net score is present, finite and positive (bullish >
         bearish) — a bearish HOLD can never be entered, and
      2. the same entry gate a strict BUY must clear
         (``analysis.entry_eligible``: daily uptrend, more than 2% above the
         SMA-200, weekly trend up).

    Anything unknown fails closed: a snapshot without a net score, without
    the trend flags or without an explicit ``weekly_trend_up`` flag is not
    eligible for the fallback. A missing weekly flag is NOT treated as
    "no weekly data → no gate": the weekly filter is a hard
    multi-timeframe confirmation, not an optional extra.
    """
    snap = sig.get("snapshot", {}) or {}
    if "weekly_trend_up" not in snap:
        return False, "no weekly trend flag in the snapshot"
    weekly_trend_up = bool(snap.get("weekly_trend_up"))

    net = snap.get("net_score")
    if net is None:
        return False, "no net score in the snapshot"
    try:
        net_val = float(net)
    except (TypeError, ValueError):
        return False, f"unreadable net score {net!r}"
    if not math.isfinite(net_val):
        return False, f"net score {net!r} is not a finite number"
    if net_val <= 0:
        return False, f"net score {net_val:+.0f} is not bullish"

    trend_up = bool(snap.get("trend_up"))
    dist_above = snap.get("dist_above")
    if not entry_eligible(trend_up, dist_above, weekly_trend_up):
        return False, "the strict BUY entry gate did not pass"

    try:
        dist_txt = f"{float(dist_above):+.1f}% above SMA200"
    except (TypeError, ValueError):
        dist_txt = "above SMA200"
    near = "near-threshold " if net_val < params.buy_threshold else ""
    rationale = (
        f"relaxed HOLD entry ({near}net {net_val:+.0f} vs BUY threshold "
        f"{params.buy_threshold}): daily uptrend, {dist_txt}, weekly trend up"
    )
    return True, rationale


def propose_trades(
    positions: list[dict],
    cash: float,
    prices: dict[str, float],
    signals: dict[str, dict],
    params: StrategyParams,
    sector_of: Callable[[str], str] | None = None,
    quality_of: Callable[[str], dict[str, float | None] | None] | None = None,
) -> list[dict]:
    """Propose deterministic trades WITHOUT executing them.

    Single source of truth for the SELL + BUY logic. Both the live sim
    (DB-backed) and the backtest replay (in-memory) call this with plain
    data; the caller is responsible for executing the returned proposals.

    SELL phase (per held position, in order):
      1. Signal SELL → propose SELL.
      2. Initial stop (frozen at entry if ``stop_price`` is set on the
         position, else recomputed from ``avg_cost * (1 - stop_pct/100)``).
      3. ATR reference stop (from the signal's snapshot) — a prior-known
         per-bar candidate level, not a ratchet and not frozen at entry.

    BUY phase (ranked by strength, strict BUYs first, else relaxed HOLD
    fallback):
      - ``max_run_5d`` blocks BUYs after a short-term spike.
      - ``min_run_5d`` blocks BUYs into a 5-day crash.
      - ``max_dist_above`` blocks BUYs into parabolic extensions.
      - ``block_negative_roe`` + ``quality_of`` block entries for names with
        KNOWN non-positive ROE (missing fundamentals stay neutral — ETFs,
        thin coverage).
      - ``max_position_pct`` ceiling prevents over-concentration.
      - ``min_cash_pct`` floor preserves a cash buffer.
      - ``max_sector_pct`` ceiling limits per-sector exposure (the budget is
        clamped to the remaining sector room, not just pre-checked).
      - With ``use_atr_stop`` armed, no entry at or below the published ATR
        reference level (the level is anchored to the market's 20-bar high,
        so a pullback entry could otherwise be stopped out immediately).
      - A ticker sold in this cycle is never bought again in the same cycle.
      - Local cash reservation so successive BUY proposals in the same cycle
        see reduced cash.

    The relaxed HOLD fallback is a *near-threshold* entry, not a free pass:
    a HOLD candidate must clear the same entry gate as a strict BUY
    (``analysis.entry_eligible``: daily uptrend, >2% above the SMA-200,
    weekly trend up) and must have a positive net score. A bearish HOLD
    (bullish <= bearish) can therefore never enter.
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
    # Tickers proposed for sale in this cycle. They must not be bought again
    # in the same cycle: the caller executes the SELL first, so a BUY for the
    # same ticker would sell and immediately re-buy the same position at the
    # same price (pure churn, twice the costs).
    sell_tickers: set[str] = set()

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
            sell_tickers.add(ticker)
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
                sell_tickers.add(ticker)
                continue

        # ATR reference stop: the level in the snapshot was computed from the
        # PREVIOUS completed bar (prior 20-bar high close - 2x ATR), so it is
        # already known before today's bar — today's close can break it.
        # This is an indicator candidate level: it is recomputed per bar and
        # is not a persistent ratchet, nor frozen at entry.
        if params.use_atr_stop and sig:
            snap = sig.get("snapshot", {})
            atr_stop = snap.get("atr_stop")
            if atr_stop and price <= atr_stop:
                proposals.append({"ticker": ticker, "side": "SELL",
                                  "price": price, "shares": None,
                                  "reason": f"ATR stop hit: {price:.2f} <= {atr_stop:.2f}"})
                sell_tickers.add(ticker)

    # --- BUY phase ---
    max_run_5d = params.max_run_5d
    min_run_5d = params.min_run_5d
    max_dist_above = params.max_dist_above

    def _entry_ok(ticker: str, sig: dict) -> bool:
        """Entry-time guards shared by strict BUYs and the relaxed fallback.

        Blocks the entry shapes the data condemns: chasing a 5-day spike
        (max_run_5d), catching a 5-day crash (min_run_5d — bearish-bounce
        scoring reads falling knives as strong), buying parabolic
        extensions (max_dist_above — the score rewards being far above the
        SMA200, but entries > +100% average -13% forward), and entries into
        names with known non-positive ROE (block_negative_roe).
        """
        snap = sig.get("snapshot", {})
        run5 = snap.get("run_5d")
        if run5 is not None:
            if max_run_5d > 0 and run5 > max_run_5d:
                return False
            if min_run_5d < 0 and run5 < min_run_5d:
                return False
        dist = snap.get("dist_above")
        if max_dist_above > 0 and dist is not None and dist > max_dist_above:
            return False
        # Quality guard: block entries for names with KNOWN non-positive ROE.
        # quality_of returns None (or a dict without "roe") for tickers with
        # no fundamentals — those stay neutral (ETFs, thin coverage).
        if params.block_negative_roe and quality_of is not None:
            q = quality_of(ticker)
            roe = (q or {}).get("roe")
            if roe is not None and roe <= 0:
                return False
        return True

    buy_candidates = [
        (t, sig) for t, sig in signals.items()
        if sig["action"] == "BUY"
        and _entry_ok(t, sig)
    ]
    buy_candidates.sort(key=lambda x: x[1]["strength"], reverse=True)

    # Relaxed fallback: buy the best near-BUY (HOLD with high strength) so
    # the bot stays active and deploys cash instead of sitting idle.
    #
    # The fallback is NOT a loophole. A HOLD is only a near-threshold BUY
    # when the *trend* gates already passed — the score was the only thing
    # missing. So the candidate must:
    #   * have a positive net score (bullish > bearish). A bearish HOLD
    #     (e.g. bullish 0 / bearish 65) can never enter, whatever its
    #     "strength" says — the HOLD strength is max(bullish, bearish),
    #     which is exactly what made bearish HOLDs look attractive.
    #   * clear the SAME entry gate as a strict BUY (analysis.entry_eligible:
    #     daily uptrend, >2% above the SMA-200, weekly trend up).
    # Unknown/missing snapshot fields fail closed.
    fallback_reason: dict[str, str] = {}
    if not buy_candidates:
        hold_candidates = []
        for t, sig in signals.items():
            if sig["action"] != "HOLD":
                continue
            if sig["strength"] < params.relaxed_hold_strength:
                continue
            if not _entry_ok(t, sig):
                continue
            ok, why = _fallback_eligible(sig, params)
            if not ok:
                continue
            hold_candidates.append((t, sig, why))
        hold_candidates.sort(key=lambda x: x[1]["strength"], reverse=True)
        selected = hold_candidates[:params.relaxed_hold_limit]
        buy_candidates = [(t, sig) for t, sig, _ in selected]
        fallback_reason = {t: why for t, _sig, why in selected}

    for ticker, sig in buy_candidates:
        if sim_cash < min_cash:
            break  # not enough cash to keep buffer

        # Never buy a ticker that this cycle already sells: the caller
        # executes the SELL first, so the BUY would immediately re-buy the
        # position that was just stopped out (churn at the same price).
        if ticker in sell_tickers:
            continue

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

        # The ATR reference level is anchored to the market's 20-bar closing
        # high, not to this position's entry. A pullback entry can therefore
        # publish a level ABOVE the entry price — the new position would be
        # stopped out on the very next cycle (buy -> sell -> buy churn at the
        # same price). So: when the ATR stop is armed, only enter above the
        # published level. The level itself is prior-known (computed from the
        # previous completed bar), so this check uses no future information.
        if params.use_atr_stop:
            atr_level = sig.get("snapshot", {}).get("atr_stop")
            if (atr_level is not None
                    and math.isfinite(atr_level)
                    and atr_level > 0
                    and price <= atr_level):
                continue

        current_shares = next((p["shares"] for p in positions if p["ticker"] == ticker), 0)
        current_value = current_shares * price
        if current_value >= max_position_value:
            continue

        # Sector diversification cap: limit total exposure per sector.
        # Tracked progressively so the 2nd same-sector BUY sees the 1st one.
        # The pre-check only blocks a BUY that would *start* above the cap;
        # the budget is clamped to the remaining sector room below, so a
        # large BUY can never push the sector past the cap.
        sector_room: float | None = None
        if params.max_sector_pct > 0 and sector_of:
            sec = sector_of(ticker)
            sector_cap = total_equity * (params.max_sector_pct / 100)
            sector_room = sector_cap - sector_values.get(sec, 0.0)
            if sector_room <= 0:
                continue

        budget = min(sim_cash - min_cash, max_position_value - current_value)
        if sector_room is not None:
            # Never allocate more than the sector still has room for.
            budget = min(budget, sector_room)

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

        # Frozen entry stop for the new proposal. The percent stop is the
        # entry-anchored control (frozen at entry in the replay; the live sim
        # recomputes it from the blended average cost every cycle). The ATR
        # entry stop is frozen at entry too (entry price - N x ATR) and is
        # distinct from the per-bar ATR reference candidate checked in the
        # SELL phase.
        entry_stop = None
        if params.stop_type == "percent":
            entry_stop = price * (1 - params.stop_pct / 100)
        elif params.stop_type == "atr":
            atr = sig.get("snapshot", {}).get("atr14")
            if atr and atr > 0:
                entry_stop = price - params.stop_atr_mult * atr

        # A fallback BUY explains why a HOLD was entered (near-threshold
        # rationale); a strict BUY keeps the signal's own reason.
        reason = fallback_reason.get(ticker) or sig.get("reason", "BUY signal")

        proposals.append({"ticker": ticker, "side": "BUY",
                          "price": price, "budget": budget,
                          "reason": reason,
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
    unsized_remaining: int = 1,
) -> float | None:
    """Budget for an LLM BUY decision, or None to skip.

    ``price`` is the execution price, ``current_value`` the value already
    held in this ticker. ``guarded=True`` applies the hard risk limits (min
    cash floor and max position %) — used in hybrid mode, matching the
    deterministic engine. ``guarded=False`` lets the LLM decide sizing and
    cash reserve; the only hard bounds are the actual cash balance and the
    $1 minimum. In both modes the optional partial-size fields (``shares`` /
    ``amount``) clamp the budget.

    ``unsized_remaining``: number of pending BUY decisions (including this
    one) that carry no ``shares``/``amount``. When the LLM issues several
    unsized BUYs in one cycle, each gets an equal slice of the cash
    (``cash / unsized_remaining``) instead of the first one taking
    everything — mirroring the per-proposal cash reservation in
    ``propose_trades``.
    """
    min_cash = equity * (params.min_cash_pct / 100)
    max_position_value = equity * (params.max_position_pct / 100)

    sized = "shares" in decision or "amount" in decision
    if guarded:
        if cash < min_cash:
            return None
        if current_value >= max_position_value:
            return None
        budget = min(cash - min_cash, max_position_value - current_value)
    elif sized:
        budget = cash
    else:
        budget = cash / max(1, unsized_remaining)

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


async def plan_llm_buys(
    decisions: list[dict],
    cash: float,
    equity: float,
    params: StrategyParams,
    guarded: bool,
    price_of: Callable[[str], Awaitable[float | None]],
    value_of: Callable[[str], Awaitable[float]],
    exclude: set[str] | None = None,
) -> dict[str, float]:
    """Plan the execution budget for every LLM BUY decision in a batch.

    Single source of truth for the budget math shared by the live sim and
    the backtest replay. ``price_of(ticker)`` and ``value_of(ticker)`` are
    async callables returning the execution price and the value already held
    in the ticker (None price skips the ticker). ``exclude`` lists tickers
    handled elsewhere (e.g. approved proposals in hybrid mode).

    Returns ``{ticker: budget}`` in decision order. Unsized BUYs (no
    ``shares`` / ``amount``) split the available cash evenly so the first
    BUY does not consume everything; sized BUYs get their requested size
    clamped to what is allowed. Each planned budget is deducted from a
    running cash total so the sum never exceeds the available cash — a BUY
    that no longer fits is skipped rather than silently dropped at
    execution.
    """
    excluded = {t.upper() for t in (exclude or set())}
    buys = [
        d for d in decisions
        if d.get("action") == "BUY" and d["ticker"].upper() not in excluded
    ]
    unsized = sum(1 for d in buys if "shares" not in d and "amount" not in d)
    remaining_cash = cash
    plan: dict[str, float] = {}
    for d in buys:
        ticker = d["ticker"].upper()
        price = await price_of(ticker)
        if price is None or price <= 0:
            continue
        is_unsized = "shares" not in d and "amount" not in d
        budget = llm_buy_budget(
            d, remaining_cash, equity, price, await value_of(ticker), params,
            guarded=guarded,
            unsized_remaining=max(1, unsized) if is_unsized else 1,
        )
        if is_unsized:
            unsized -= 1
        if budget is None:
            continue
        plan[ticker] = budget
        remaining_cash -= budget
    return plan


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
