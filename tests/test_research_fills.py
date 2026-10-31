"""Fee-aware RESEARCH fills = thin wrappers over ``app.factor_fills``.

``app/factor_fills.py`` owns the ONE zero-fee fill definition for the factor
portfolios, and its contract is explicit: a fee-aware adaptation "MUST be
added as a wrapper around these primitives — never by mutating the sizes
here". The research daily-core replay (``optimize._daily_core_backtest``) is
that wrapper: it models a one-way ``cost`` on top of the zero-fee buy/sell,
so the live paper book and the replay share every size and every guard.

These tests pin the wrapper, not the primitives (tests/test_factor_fills.py
pins those):

  * ``cost=0``: ``optimize._fill_buy`` returns exactly the primitive's
    ``(notional, shares, cash_after)`` — BIT-identically, on a grid that
    includes the razor case where the caller's rounded budget
    (``round(cash, 2)``) floors a hair above the exact cash, so the leg is
    dropped and never resized down to the cash;
  * ``cost>0``: the same tuple the pre-wrapper formula produced (budget cap,
    4dp floor, notional, fee-inclusive guard, cash_after) — bit-identically,
    with the fee coming out of CASH only (sizes are never re-floored);
  * the wrapper KEEPS the fee-inclusive guard the zero-fee primitive cannot
    have: real floating-point cases where ``notional <= cash`` but
    ``notional * (1 + cost) > cash`` are rejected instead of booking the ULP
    of negative cash;
  * the replay's SELL goes through ``factor_fills.sell_fill`` and then applies
    the old ``notional * (1 - cost)`` expression, so a full cash-out's cash is
    bit-identical to the pre-wrapper ledger at every cost — and it liquidates
    the position's shares AS-IS (no 4dp re-flooring);
  * cash never goes negative: buys are fee-inclusively guarded, deposits only
    add, so ``sell_fill``'s negative-cash rejection is unreachable;
  * non-finite inputs are no-fill (the primitive's hardened contract), which
    is the only place the wrapper differs from the pre-wrapper code — and that
    state (non-finite cash) is not reachable in the replay.

Self-contained: synthetic frames only, no network, no database.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app import factor_fills, optimize
from app import monthly as monthly_mod
from app.config import settings

CTICKERS = ["A01", "B02", "C03", "D04"]

# The tape's crash: closes drop on 2025-01-20, so that session's completed
# close (S) stands the portfolio-stop cash-out at the NEXT session's open.
CRASH = pd.Timestamp("2025-01-20")
TRIGGER = pd.Timestamp("2025-01-21")
START = "2024-12-16"
CONTRIBUTION = 1000.0


# ---------------------------------------------------------------------------
# The pre-wrapper fill formulas, transcribed (the golden reference)
# ---------------------------------------------------------------------------

def _legacy_formula_fill(cash: float, price: float, delta_value: float,
                         cost: float) -> tuple[float, float, float] | None:
    """``_fill_buy`` BEFORE the factor_fills adoption, with the 4dp floor
    written out literally: budget cap, ``floor(budget / price * 10000) /
    10000`` shares, ``notional = shares * price``, the fee-inclusive guard and
    ``cash - notional * (1 + cost)``.

    Only valid for finite inputs and a positive price (the pre-wrapper code
    routed unusable sizes through ``share_size``, which the literal floor
    cannot express — see ``_legacy_share_size_fill``).
    """
    budget = delta_value
    if cost > 0:
        budget = min(budget, max(cash, 0.0) / (1.0 + cost))
    if budget < 1.0:
        return None
    shares = math.floor(budget / price * 10000) / 10000
    if shares < 0.0001:
        return None
    notional = shares * price
    if cash < notional * (1.0 + cost):
        return None
    return notional, shares, cash - notional * (1.0 + cost)


def _legacy_share_size_fill(cash: float, price: float, delta_value: float,
                            cost: float) -> tuple[float, float, float] | None:
    """The pre-wrapper ``_fill_buy`` EXACTLY: identical arithmetic, but sizing
    through the shared ``factor_fills.share_size`` (the engine-to-engine
    import the factor_fills adoption removed) instead of the literal floor."""
    budget = delta_value
    if cost > 0:
        budget = min(budget, max(cash, 0.0) / (1.0 + cost))
    if budget < 1.0:
        return None
    shares = factor_fills.share_size(budget, price)
    if shares < 0.0001:
        return None
    notional = shares * price
    if cash < notional * (1.0 + cost):
        return None
    return notional, shares, cash - notional * (1.0 + cost)


def _floor4(shares: float) -> float:
    """The 4dp share quantum, written out — what a re-flooring would do."""
    return math.floor(shares * 10000.0) / 10000.0


# ---------------------------------------------------------------------------
# BUY: cost=0 is exactly the shared zero-fee primitive
# ---------------------------------------------------------------------------

_CASH = [1000.0, 1000.01, 333.33, 100.0, 99.99, 2.5, 1.0, 1e6, 12345.6789]
_PRICE = [3.0, 0.0001, 7.77, 99.0, 137.31, 1e6, 2.0]
_BUDGET = [1000.0, 500.0, 49.99, 1.0, 1e9, 0.5]
_COST = [0.0005, 0.001, 0.0025, 0.01, 0.05, 0.25]


def test_zero_cost_fill_is_the_primitive_bit_for_bit():
    """At cost=0 ``_fill_buy`` must BE ``factor_fills.buy_fill``: same size,
    same notional, same cash_after — bit-for-bit, including the no-fill legs
    (the primitive's exact-cash guard and minimums are the whole decision).

    The old formula is compared too, so this pins the primitive adoption as a
    pure refactor rather than a re-derivation.
    """
    fills = no_fills = 0
    for cash in _CASH:
        for price in _PRICE:
            for budget in _BUDGET:
                prim = factor_fills.buy_fill(cash, budget, price)
                got = optimize._fill_buy(cash, price, budget, 0.0)
                old = _legacy_formula_fill(cash, price, budget, 0.0)
                if prim is None:
                    no_fills += 1
                    assert got is None, (cash, price, budget)
                    assert old is None, (cash, price, budget)
                    continue
                fills += 1
                assert got == (prim.notional, prim.shares, cash - prim.notional)
                assert got == (prim.notional, prim.shares, prim.cash_after)
                assert got == old, (cash, price, budget)
                notional, shares, cash_after = got
                assert shares == prim.shares
                assert notional == shares * price
                assert cash_after == cash - notional
                assert cash_after >= 0.0
    # the grid is not vacuous in either direction
    assert fills > 100 and no_fills > 10


def test_zero_cost_razor_budget_above_exact_cash_books_nothing():
    """The razor edge the primitive's docstring warns about: the caller sizes
    with ``round(cash, 2)``, and when that rounds UP the 4dp share floor can
    land a hair above the EXACT cash. The leg must be DROPPED (the live
    executor skips it), never resized down to the cash.

    cash=100.006 -> round(cash, 2)=100.01 -> 100.01 shares at $1 = $100.01,
    which is $0.004 more than the account holds.
    """
    cash = 100.006
    budget = round(cash, 2)              # what trade() passes: min(delta, round)
    assert budget > cash                 # the razor: the budget rounded UP
    assert optimize._fill_buy(cash, 1.0, budget, 0.0) is None
    assert factor_fills.buy_fill(cash, budget, 1.0) is None
    assert _legacy_formula_fill(cash, 1.0, budget, 0.0) is None
    # shrinking the budget to the exact cash would have booked $100.006 of
    # shares — that is exactly the resize the primitive refuses to make.
    assert factor_fills.share_size(cash, 1.0) == 100.006

    # control: an exactly-affordable full-cash buy books and leaves $0.
    assert optimize._fill_buy(100.01, 1.0, 100.01, 0.0) == (100.01, 100.01, 0.0)


# ---------------------------------------------------------------------------
# BUY: cost>0 keeps the old fee arithmetic bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cost", _COST)
def test_fee_cost_fill_matches_the_old_formula_bit_for_bit(cost):
    """For any one-way cost the wrapper must return EXACTLY what the
    pre-wrapper code returned: the budget cap mirrors the same expression, the
    size is the primitive's 4dp floor, and the fee comes out of cash only
    (``cash - notional * (1 + cost)``) — no re-flooring, no clamping.
    """
    fills = no_fills = 0
    for cash in _CASH:
        for price in _PRICE:
            for budget in _BUDGET:
                got = optimize._fill_buy(cash, price, budget, cost)
                old = _legacy_formula_fill(cash, price, budget, cost)
                assert got == old, (cash, price, budget, cost)
                if got is None:
                    no_fills += 1
                    continue
                fills += 1
                notional, shares, cash_after = got
                # sizes are the zero-fee primitive's — the fee never touches
                # the share count
                assert shares == factor_fills.share_size(
                    min(budget, max(cash, 0.0) / (1.0 + cost)), price)
                assert notional == shares * price
                assert cash_after == cash - notional * (1.0 + cost)
                assert cash_after >= 0.0
                # when the fee cap does not bind, the SAME leg as cost=0
                free = optimize._fill_buy(cash, price, budget, 0.0)
                if budget <= max(cash, 0.0) / (1.0 + cost) and free is not None:
                    assert (free[0], free[1]) == (notional, shares)
    assert fills > 100 and no_fills > 10


@pytest.mark.parametrize("cash,price,cost", [
    (180.63, 100.0, 0.0035),
    (930.93, 100.0, 0.001),
    (3.03, 1.0, 0.01),
])
def test_fee_inclusive_guard_rejects_a_leg_the_zero_fee_primitive_accepts(
        cash, price, cost):
    """The wrapper's fee-inclusive guard is NOT redundant.

    Each case is a real float pair where the capped budget floors to a
    ``notional`` that is affordable fee-free (``notional <= cash``, so the
    primitive books it) but whose fee-inclusive total
    ``notional * (1 + cost)`` exceeds the cash by a few ULPs — booking it
    would leave the account at -1e-13..-3e-14. Only the wrapper's guard
    prevents that.
    """
    budget = min(cash, cash / (1.0 + cost))       # full-cash buy, cap binds
    prim = factor_fills.buy_fill(cash, budget, price)
    assert prim is not None                         # fee-free: affordable
    assert prim.notional <= cash
    assert prim.cash_after >= 0.0
    assert cash - prim.notional * (1.0 + cost) < 0.0  # fee-inclusive: NOT
    # ... and the wrapper refuses it, so cash can never go negative.
    assert optimize._fill_buy(cash, price, cash, cost) is None
    # the pre-wrapper code had the same guard (this is the preserved contract)
    assert _legacy_formula_fill(cash, price, cash, cost) is None


def test_full_cash_buy_never_books_negative_cash():
    """Sweep full-cash buys at several costs: every booked leg leaves
    ``cash_after >= 0`` and books exactly ``cash - notional*(1+cost)``. This is
    the invariant the SELL's negative-cash rejection relies on."""
    booked = 0
    for cost in _COST + [0.0]:
        for cash in _CASH:
            for price in _PRICE:
                for budget in (cash, round(cash, 2), cash / 2.0):
                    got = optimize._fill_buy(cash, price, budget, cost)
                    if got is None:
                        continue
                    booked += 1
                    notional, _shares, cash_after = got
                    assert notional <= cash
                    assert cash_after == cash - notional * (1.0 + cost)
                    assert cash_after >= 0.0
    assert booked > 50


# ---------------------------------------------------------------------------
# BUY: invalid inputs are no-fill (the primitive's hardened contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cash,price,budget", [
    (1000.0, float("nan"), 1000.0),     # NaN price
    (1000.0, float("inf"), 1000.0),     # inf price
    (1000.0, 0.0, 1000.0),              # non-positive price
    (1000.0, -3.0, 1000.0),
    (1000.0, 3.0, float("nan")),        # NaN budget
    (1000.0, 3.0, float("-inf")),       # negative-infinity budget
    (1000.0, 3.0, 0.5),                 # sub-$1 budget
    (1000.0, 1e6, 1.0),                 # sub-quantum size
    (1e308, 5e-324, 1e308),             # sizing overflow
])
def test_unusable_inputs_are_no_fill(cash, price, budget):
    """The old path already routed sizing through the shared ``share_size``,
    so invalid prices/budgets are no-fill in BOTH implementations."""
    assert optimize._fill_buy(cash, price, budget, 0.0) is None
    assert optimize._fill_buy(cash, price, budget, 0.001) is None
    assert factor_fills.buy_fill(cash, budget, price) is None
    assert _legacy_share_size_fill(cash, price, budget, 0.0) is None


def test_infinite_budget_spends_everything_at_cost_but_is_a_no_fill_at_zero():
    """An unbounded budget follows the historical rule: at cost=0 the size
    overflows to no-fill, while at cost>0 the fee cap first bounds the spend
    to ``cash / (1 + cost)`` and the leg is executable. Both are what the
    pre-wrapper code did (same expressions, same order)."""
    assert optimize._fill_buy(1000.0, 3.0, float("inf"), 0.0) is None
    for cost in (0.001, 0.05):
        assert optimize._fill_buy(1000.0, 3.0, float("inf"), cost) == (
            optimize._fill_buy(1000.0, 3.0, 1000.0, cost))
        assert optimize._fill_buy(1000.0, 3.0, float("inf"), cost) == (
            _legacy_formula_fill(1000.0, 3.0, float("inf"), cost))


def test_non_finite_cash_is_now_rejected_and_was_a_garbage_fill_before():
    """The ONE behavioral delta of the adoption, and it is unreachable.

    The pre-wrapper code had no cash-validity check: ``cash < notional *
    (1 + cost)`` is False for NaN, so it returned a fill with a NaN
    ``cash_after`` (and +inf cash returned an infinite one). The primitive
    rejects non-finite cash, so the wrapper now returns None. The replay
    cannot reach this state — cash starts at 0.0 and only moves by
    fee-inclusively guarded buys, sells and non-negative deposits — so no
    reachable behavior changes.
    """
    for cash in (float("nan"), float("inf"), float("-inf")):
        assert factor_fills.buy_fill(cash, 1000.0, 3.0) is None
        assert optimize._fill_buy(cash, 3.0, 1000.0, 0.0) is None
        assert optimize._fill_buy(cash, 3.0, 1000.0, 0.01) is None
    # what the pre-wrapper arithmetic would have done with NaN cash
    leaked = _legacy_formula_fill(float("nan"), 3.0, 1000.0, 0.0)
    assert leaked is not None and math.isnan(leaked[2])


# ---------------------------------------------------------------------------
# SELL: the replay books the old expression bit-for-bit through sell_fill
# ---------------------------------------------------------------------------

def _canned_frame(order: list[str]) -> pd.DataFrame:
    """A frame whose ``_qv_order`` equals ``order`` (roe decays with rank)."""
    roe = {t: 0.5 - 0.01 * i for i, t in enumerate(order)}
    return pd.DataFrame({t: {"roe": roe.get(t, 0.001), "p_fcf": 20.0,
                             "mcap": 1e10, "dollar_vol": 1e8, "mom": 0.1,
                             "price": 100.0, "vol": 0.2, "eligible": True}
                         for t in CTICKERS}).T


def _frames() -> dict[pd.Timestamp, pd.DataFrame]:
    """Canned month-end frames: the pre-window frame that funds the December
    build, then Dec-31, Jan-31 and Feb-28-2025."""
    return {pd.Timestamp("2024-11-29"): _canned_frame(CTICKERS),
            pd.Timestamp("2024-12-31"): _canned_frame(CTICKERS),
            pd.Timestamp("2025-01-31"): _canned_frame(CTICKERS),
            pd.Timestamp("2025-02-28"): _canned_frame(CTICKERS)}


def _crash_tape(open_px: float = 99.0) -> tuple[pd.DataFrame, pd.DataFrame,
                                                pd.DataFrame]:
    """(close, vol, open) whose closes fall 10% on CRASH, so the
    portfolio-stop brake liquidates the whole book at TRIGGER's open.

    ``open_px`` is deliberately independent of the closes: a tiny open makes
    the 4dp floor leave a sub-quantum residue in an accumulated position, and
    a large one keeps the prices realistic.
    """
    dates = pd.bdate_range("2023-06-01", "2025-02-28")
    close = pd.DataFrame(100.0, index=dates, columns=CTICKERS)
    close.loc[close.index >= CRASH] = 100.0 * 0.90
    vol = pd.DataFrame(1e6, index=dates, columns=CTICKERS)
    open_frame = pd.DataFrame(float(open_px), index=dates, columns=CTICKERS)
    return close, vol, open_frame


@pytest.fixture
def small_book(monkeypatch):
    """Shrink the slot knobs so the four-name tape exercises a 3-slot book."""
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 6)
    monkeypatch.setattr(settings, "sim_daily_core_mom_variant", "raw")
    monkeypatch.setattr(settings, "sim_monthly_rank_boost", 0.0)


async def _run(monkeypatch, close, vol, open_frame, *, cost: float,
               **kw) -> optimize.BacktestSummary:
    """Run the research daily-core replay on a canned tape (dca="rank")."""
    monkeypatch.setattr(
        monthly_mod, "eligible_frame",
        lambda d, c, v, f: _frames().get(pd.Timestamp(d).normalize()))
    res = await optimize._daily_core_backtest(
        START, None, "diversified-plus", CONTRIBUTION, False,
        "none", "daily", False, "rank",
        frame_cache=None, close_preloaded=(close, vol), fund_preloaded={},
        open_preloaded=open_frame, cost_oneway=cost, with_baseline=False, **kw)
    assert res is not None
    return res


def _ledger(res: optimize.BacktestSummary, index: pd.DatetimeIndex, *,
            cost: float, end: str | None = None,
            contribution: float = CONTRIBUTION,
            ) -> tuple[float, dict[str, float], float]:
    """Replay the booked trades through the PRE-WRAPPER cash expressions.

    Contributions land on the first session of each calendar month in the
    window (the replay's own cadence), then every trade in booked order:
    ``cash - notional*(1+cost)`` for a BUY and ``cash + notional*(1-cost)`` for
    a SELL — the exact legacy formulas. Returns ``(final_cash, holdings,
    min_cash)``; ``min_cash`` is the lowest cash balance at ANY step, so a
    negative-cash booking anywhere is visible to the caller.
    """
    idx = pd.DatetimeIndex(index)
    idx = idx[idx >= pd.Timestamp(START)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    events: list[tuple[pd.Timestamp, int, dict | None]] = []
    seen: set[tuple[int, int]] = set()
    for d in idx:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            events.append((d, 0, None))
    events += [(pd.Timestamp(t["date"]), i + 1, t)
               for i, t in enumerate(res.trades)]
    events.sort(key=lambda e: (e[0], e[1]))

    cash = 0.0
    min_cash = 0.0
    held: dict[str, float] = {}
    n_contribs = 0
    for _d, kind, trade in events:
        if kind == 0:
            cash += contribution
            n_contribs += 1
        elif trade["side"] == "BUY":
            cash = cash - trade["notional"] * (1.0 + cost)
            held[trade["ticker"]] = held.get(trade["ticker"], 0.0) + trade["shares"]
        else:
            cash = cash + trade["notional"] * (1.0 - cost)
            held[trade["ticker"]] = held.get(trade["ticker"], 0.0) - trade["shares"]
        min_cash = min(min_cash, cash)
    # the reconstructed deposit cadence must be the engine's, or the exact
    # cash comparison below would be testing the wrong ledger
    assert n_contribs * contribution == res.contributed
    return cash, held, min_cash


@pytest.mark.parametrize("cost", [0.0, 0.001, 0.002, 0.25])
async def test_full_liquidation_cash_is_the_old_ledger_bit_for_bit(
        small_book, monkeypatch, cost):
    """A portfolio-stop cash-out liquidates every name at TRIGGER's open and
    leaves the account in cash, so ``final_value`` IS the cash balance.

    Replaying the booked trades through the legacy expressions must reproduce
    that balance EXACTLY (bit-for-bit) — which pins the SELL branch's
    ``cash = cash + notional*(1-cost)``, the BUY branch's fee arithmetic, and
    the fact that the fee is taken from cash only. The same replay also proves
    cash never goes negative at any step for every plausible one-way cost
    (0..100%; sells only ADD cash below 100%).
    """
    close, vol, open_frame = _crash_tape()
    res = await _run(monkeypatch, close, vol, open_frame, cost=cost,
                     portfolio_stop_pct=0.001)

    sells = [t for t in res.trades if t["side"] == "SELL"]
    buys = [t for t in res.trades if t["side"] == "BUY"]
    assert buys and len(sells) >= 3, "the brake must liquidate a built book"
    assert all(pd.Timestamp(t["date"]) == TRIGGER for t in sells)

    cash, held, min_cash = _ledger(res, close.index, cost=cost)
    # every sold leg books notional == shares * price exactly (no fee inside
    # the notional, no re-sizing) and the sell prices are the strict opens
    for leg in sells:
        assert leg["notional"] == leg["shares"] * leg["price"]
        assert leg["price"] == pytest.approx(99.0)
    # fully cashed out: nothing is held at the end, so final_value == cash
    assert all(shares == 0.0 for shares in held.values())
    assert res.final_value == cash
    # no negative cash at ANY step of the legacy ledger
    assert min_cash >= 0.0


async def test_sell_liquidates_the_position_as_is_without_reflooring(
        small_book, monkeypatch):
    """The SELL uses ``shares[t]`` verbatim — no 4dp re-flooring.

    The $3.30 opens make the accumulated position a float whose value is NOT
    its own 4dp floor (e.g. 202.00900000000001 vs 202.009), so a re-flooring
    sell would book a visibly different share count. The primitive's contract
    is that the caller owns the position quantity.
    """
    close, vol, open_frame = _crash_tape(open_px=3.30)
    res = await _run(monkeypatch, close, vol, open_frame, cost=0.0,
                     portfolio_stop_pct=0.001)
    sells = [t for t in res.trades if t["side"] == "SELL"]
    assert len(sells) >= 3

    held: dict[str, float] = {}
    residue_checked = 0
    for t in res.trades:
        if t["side"] == "BUY":
            held[t["ticker"]] = held.get(t["ticker"], 0.0) + t["shares"]
            continue
        position = held[t["ticker"]]
        assert t["shares"] == position        # as-is, bit-for-bit
        if position != _floor4(position):
            residue_checked += 1
            assert t["shares"] != _floor4(position)
        held[t["ticker"]] = position - t["shares"]
    assert residue_checked >= 1, "tape must produce a sub-quantum residue"
    assert all(shares == 0.0 for shares in held.values())


async def test_sell_without_a_usable_open_keeps_the_position(small_book,
                                                             monkeypatch):
    """A leg with NO executable price books nothing and keeps the position.

    The brake is a standing order: a name without an exact-date open on the
    trigger session is retried the next session and sells there, with the
    position's unchanged share count. That is the only failure mode the
    primitive contract asks a caller to handle.
    """
    close, vol, open_frame = _crash_tape()
    open_frame = open_frame.copy()
    open_frame.loc[TRIGGER, "A01"] = np.nan        # no executable open
    res = await _run(monkeypatch, close, vol, open_frame, cost=0.0,
                     portfolio_stop_pct=0.001)

    on_trigger = [t for t in res.trades if pd.Timestamp(t["date"]) == TRIGGER]
    assert "A01" not in [t["ticker"] for t in on_trigger]
    assert {"B02", "C03"} <= {t["ticker"] for t in on_trigger
                              if t["side"] == "SELL"}
    a01 = [t for t in res.trades if t["ticker"] == "A01" and t["side"] == "SELL"]
    assert len(a01) == 1
    assert pd.Timestamp(a01[0]["date"]) > TRIGGER
    # the retried sell still liquidates the whole position
    _, held, min_cash = _ledger(res, close.index, cost=0.0)
    assert held.get("A01", 0.0) == 0.0
    assert min_cash >= 0.0

    # no-bar and unusable-price legs are non-executable at the primitive too
    assert factor_fills.sell_fill(100.0, 3.0, float("nan")) is None
    assert factor_fills.sell_fill(100.0, 3.0, 0.0) is None
    assert factor_fills.sell_fill(100.0, 3.0, -1.0) is None
    assert factor_fills.sell_fill(100.0, 0.0, 10.0) is None
    assert factor_fills.sell_fill(100.0, -3.0, 10.0) is None
    assert factor_fills.sell_fill(-1.0, 3.0, 10.0) is None


# ---------------------------------------------------------------------------
# The replay delegates to the shared primitives (no local arithmetic left)
# ---------------------------------------------------------------------------

class _Spy:
    """Records the primitive calls the replay makes, then delegates."""

    def __init__(self, deny_sells: bool = False) -> None:
        self.buy_calls: list[tuple[float, float, float]] = []
        self.sell_calls: list[tuple[float, float, float]] = []
        self.deny_sells = deny_sells

    def buy_fill(self, cash: float, budget: float, price: float):
        self.buy_calls.append((cash, budget, price))
        return factor_fills.buy_fill(cash, budget, price)

    def sell_fill(self, cash: float, shares: float, price: float):
        self.sell_calls.append((cash, shares, price))
        if self.deny_sells:
            return None
        return factor_fills.sell_fill(cash, shares, price)


@pytest.mark.parametrize("cost", [0.0, 0.001])
async def test_replay_routes_both_sides_through_factor_fills(
        small_book, monkeypatch, cost):
    """BUY and SELL both go through ``app.factor_fills`` — the wrapper holds
    no fill arithmetic of its own. The spy delegates, so the booked trade
    stream must be identical to an unspied run. The recorded SELL arguments
    also prove ``sell_fill`` never SEES negative cash (its rejection is
    unreachable): buys are fee-inclusively guarded and deposits only add."""
    close, vol, open_frame = _crash_tape()
    spy = _Spy()
    with monkeypatch.context() as mp:
        mp.setattr(optimize, "factor_fills", spy)
        spied = await _run(monkeypatch, close, vol, open_frame, cost=cost,
                           portfolio_stop_pct=0.001)
    plain = await _run(monkeypatch, close, vol, open_frame, cost=cost,
                       portfolio_stop_pct=0.001)

    assert plain.trades == spied.trades
    assert len(spy.buy_calls) == len(
        [t for t in spied.trades if t["side"] == "BUY"])
    assert len(spy.sell_calls) >= 3
    assert {price for _c, _b, price in spy.buy_calls} == {99.0}
    assert {price for _c, _s, price in spy.sell_calls} == {99.0}
    assert all(cash >= 0.0 for cash, _s, _p in spy.sell_calls)


async def test_a_sell_the_primitive_refuses_books_nothing(small_book,
                                                          monkeypatch):
    """``sell_fill`` returning None is the SELL arm's only failure mode: no
    notional, no cash movement, no bookkeeping mutation — the position (and
    its trailing-stop peak) survive and the brake retries next session."""
    close, vol, open_frame = _crash_tape()
    spy = _Spy(deny_sells=True)
    monkeypatch.setattr(optimize, "factor_fills", spy)
    res = await _run(monkeypatch, close, vol, open_frame, cost=0.0,
                     portfolio_stop_pct=0.001)

    assert spy.sell_calls, "the brake must have tried to sell"
    assert [t for t in res.trades if t["side"] == "SELL"] == []
    _, held, _min_cash = _ledger(res, close.index, cost=0.0)
    assert sum(held.values()) > 0.0, "a refused sell keeps the position"
    # repeated retries each session while the brake is armed
    assert len(spy.sell_calls) > len(held)
