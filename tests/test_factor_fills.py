"""Pure execution primitives shared by the factor portfolios (app.factor_fills).

Phase 2's bounded shared-accounting foundation: the zero-fee fill arithmetic
must be identical in the Monthly live executor and its persisted backfill
(tests/test_cadence_parity.py pins the two trade streams against each other).
These tests pin the primitive itself, independently of any engine:

  * the frozen Phase 1 sizing formula (4dp floor, $1 minimum budget, 0.0001
    minimum shares) with the historical float operation order;
  * the EXACT-cash guard — an unaffordable leg is dropped, never resized down
    to the cash (the razor edge the rounded-budget research replay hits);
  * zero fees: the notional is exactly ``shares * price``, nothing else;
  * invalid / non-finite / negative inputs and overflow are non-executable —
    ``0.0`` sizing or ``None`` fill, never an exception or a partial trade;
  * sells liquidate every positive supplied share with no re-flooring;
  * a buy→sell round trip conserves cash exactly (same price, no fee);
  * ``Fill`` is immutable and carries exactly the four documented fields;
  * the module stays pure (no DB/strategy imports) and ``monthly.share_size``
    remains a thin wrapper over it for existing callers.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from app import factor_fills, monthly
from app.factor_fills import Fill, buy_fill, sell_fill, share_size

# Inputs whose arithmetic overflows: (cash, budget, price) for BUY and
# (cash, shares, price) for SELL. A 1e308 budget over a subnormal 5e-324 price
# makes budget/price infinite before any share can be booked.
_OVERFLOWING_BUY = [
    (1e308, 1e308, 5e-324),   # raw = budget/price -> inf
    (1e308, 1e308, 1e-308),   # raw -> 1e616 -> inf
]
_OVERFLOWING_SELL = [
    (1e308, 1e308, 1e308),    # shares * price -> inf
    (1e308, 1.0, 1e308),      # cash + notional -> inf (2e308)
]


# ---------------------------------------------------------------------------
# Sizing: the frozen Phase 1 formula
# ---------------------------------------------------------------------------

def test_share_size_floors_to_four_decimals():
    # literals from the legacy formula: floor(budget / price * 10000) / 10000
    assert share_size(1000.0, 3.0) == 333.3333
    assert share_size(100.0, 3.0) == 33.3333
    assert share_size(123.45, 5.0) == 24.69
    assert share_size(250.0, 10.0) == 25.0
    assert share_size(1000.0, 2.0) == 500.0
    assert share_size(1.0, 3.0) == 0.3333


def test_share_size_matches_the_phase1_formula_bit_for_bit():
    """The primitive must reproduce the historical expression EXACTLY (same
    operands, same order) — live and replay sizes have to stay bit-identical
    or the golden cadence parity breaks."""
    grid = [
        (1000.0, 3.0), (1000.0, 7.0), (123.45, 5.0), (999.99, 137.31),
        (1.0, 1.0), (1.0, 10000.0), (99.9999, 1e6), (100.0, 1e6),
        (123456.789, 0.0001), (29000.0, 97.38527268275928),
        (1e6, 1e6), (1.0, 2.5), (5000.0, 2.5),
    ]
    for budget, price in grid:
        expected = math.floor(budget / price * 10000) / 10000
        assert share_size(budget, price) == expected, (budget, price)


def test_share_size_minimum_budget_and_minimum_shares():
    assert share_size(0.99, 1.0) == 0.0          # sub-$1 budget: no leg
    assert share_size(1.0, 3.0) == 0.3333        # $1 budget is executable
    # exactly one share quantum is executable; a hair under it is not
    assert share_size(100.0, 1e6) == 0.0001
    assert share_size(99.9999, 1e6) == 0.0
    assert share_size(1.0, 1e6) == 0.0


def test_share_size_invalid_inputs_return_zero_without_raising():
    bad = [float("nan"), float("inf"), float("-inf"), -1.0, -1e-9, 0.0,
           None, "100", True, 10 ** 400]
    for value in bad:
        assert share_size(value, 10.0) == 0.0, value   # bad budget
        assert share_size(1000.0, value) == 0.0, value  # bad price


def test_share_size_overflow_returns_zero():
    assert share_size(1e308, 5e-324) == 0.0
    assert share_size(1e308, 1e-308) == 0.0


# ---------------------------------------------------------------------------
# BUY fills
# ---------------------------------------------------------------------------

def test_buy_fill_value_and_zero_fee():
    fill = buy_fill(1000.0, 250.0, 10.0)
    assert fill == Fill(shares=25.0, price=10.0, notional=250.0,
                        cash_after=750.0)
    # zero fees: the cash outlay is exactly shares * price, nothing else
    assert fill.notional == fill.shares * fill.price
    assert fill.cash_after == 1000.0 - fill.notional


def test_buy_fill_preserves_the_float_operation_order():
    cash, budget, price = 1000.0, 1000.0, 3.0
    fill = buy_fill(cash, budget, price)
    assert fill is not None
    assert fill.shares == 333.3333
    assert fill.notional == 333.3333 * 3.0       # == 999.9999 exactly
    assert fill.cash_after == cash - 333.3333 * 3.0


def test_buy_fill_exact_cash_is_allowed():
    """A leg whose booked notional equals the EXACT cash executes (and zeroes
    the cash); the guard rejects only a notional strictly above the cash."""
    fill = buy_fill(1000.0, 1000.0, 2.0)
    assert fill is not None
    assert fill.notional == 1000.0
    assert fill.cash_after == 0.0


def test_buy_fill_razor_edge_is_rejected_not_resized():
    """The rounded-budget razor edge: a budget a hair above the exact cash
    sizes a 4dp-floored cost above it. The leg must be DROPPED — never
    shrunk (which would book a trade the live executor refuses)."""
    assert buy_fill(999.99, 1000.0, 2.0) is None      # cost 1000 > cash
    assert buy_fill(999.995, 1000.0, 3.0) is None     # cost 999.9999 > cash
    # ...and the budget itself is never clamped down to the cash:
    assert buy_fill(1.0, 1000.0, 2.0) is None
    assert share_size(1000.0, 2.0) == 500.0           # sizing is untouched


def test_buy_fill_rejects_sub_minimum_sizing():
    assert buy_fill(1000.0, 0.99, 1.0) is None        # sub-$1 budget
    assert buy_fill(1000.0, 1.0, 1e6) is None         # < 0.0001 shares
    assert buy_fill(1000.0, 100.0, 1e6) is not None   # exactly 0.0001 shares


@pytest.mark.parametrize("cash,budget,price", [
    (float("nan"), 100.0, 10.0),
    (float("inf"), 100.0, 10.0),
    (-1.0, 100.0, 10.0),
    (None, 100.0, 10.0),
    ("1000", 100.0, 10.0),
    (1000.0, float("nan"), 10.0),
    (1000.0, float("inf"), 10.0),
    (1000.0, -10.0, 10.0),
    (1000.0, None, 10.0),
    (1000.0, 100.0, float("nan")),
    (1000.0, 100.0, float("inf")),
    (1000.0, 100.0, 0.0),
    (1000.0, 100.0, -10.0),
    (1000.0, 100.0, None),
    (1000.0, 100.0, "10"),
    (1000.0, 100.0, True),
])
def test_buy_fill_invalid_inputs_are_not_executable(cash, budget, price):
    assert buy_fill(cash, budget, price) is None


@pytest.mark.parametrize("cash,budget,price", _OVERFLOWING_BUY)
def test_buy_fill_overflow_is_not_executable(cash, budget, price):
    assert buy_fill(cash, budget, price) is None


# ---------------------------------------------------------------------------
# SELL fills
# ---------------------------------------------------------------------------

def test_sell_fill_liquidates_all_supplied_shares_without_reflooring():
    fill = sell_fill(100.0, 0.123456789, 10.0)   # NOT floored to 0.1234
    assert fill is not None
    assert fill.shares == 0.123456789
    assert fill.notional == 0.123456789 * 10.0
    assert fill.cash_after == 100.0 + fill.notional


def test_sell_fill_value_and_zero_fee():
    fill = sell_fill(250.0, 25.0, 10.0)
    assert fill == Fill(shares=25.0, price=10.0, notional=250.0,
                        cash_after=500.0)
    assert fill.notional == fill.shares * fill.price  # no proceeds haircut


@pytest.mark.parametrize("cash,shares,price", [
    (100.0, 0.0, 10.0),          # nothing to sell -> no fill
    (100.0, -1.0, 10.0),
    (100.0, float("nan"), 10.0),
    (100.0, float("inf"), 10.0),
    (100.0, None, 10.0),
    (100.0, 1.0, 0.0),
    (100.0, 1.0, -10.0),
    (100.0, 1.0, float("nan")),
    (100.0, 1.0, float("inf")),
    (float("nan"), 1.0, 10.0),
    (-1.0, 1.0, 10.0),
    (None, 1.0, 10.0),
])
def test_sell_fill_invalid_inputs_are_not_executable(cash, shares, price):
    assert sell_fill(cash, shares, price) is None


@pytest.mark.parametrize("cash,shares,price", _OVERFLOWING_SELL)
def test_sell_fill_overflow_is_not_executable(cash, shares, price):
    assert sell_fill(cash, shares, price) is None


# ---------------------------------------------------------------------------
# Round trip + Fill shape
# ---------------------------------------------------------------------------

def test_round_trip_conserves_cash_exactly():
    start = 10_000.0
    buy = buy_fill(start, 5_000.0, 2.5)
    assert buy is not None
    sell = sell_fill(buy.cash_after, buy.shares, buy.price)
    assert sell is not None
    # same operands, same product -> bit-identical notional both legs
    assert sell.notional == buy.notional
    # no fee is inserted anywhere: cash comes back exactly
    assert sell.cash_after == start
    assert sell.shares == buy.shares


def test_round_trip_conserves_cash_at_a_non_round_price():
    start = 1_000.0
    buy = buy_fill(start, 999.99, 137.31)
    assert buy is not None
    sell = sell_fill(buy.cash_after, buy.shares, buy.price)
    assert sell is not None
    # only the float subtraction/addition residue can differ (no fee term)
    assert sell.cash_after == pytest.approx(start, abs=1e-9)
    assert sell.notional == buy.notional


def test_fill_is_immutable_with_the_documented_fields():
    fill = buy_fill(1000.0, 250.0, 10.0)
    with pytest.raises(FrozenInstanceError):
        fill.shares = 1.0  # type: ignore[misc]
    assert [f.name for f in fields(Fill)] == \
        ["shares", "price", "notional", "cash_after"]
    assert fill == Fill(shares=fill.shares, price=fill.price,
                        notional=fill.notional, cash_after=fill.cash_after)
    assert hash(fill) == hash(Fill(shares=fill.shares, price=fill.price,
                                   notional=fill.notional,
                                   cash_after=fill.cash_after))


# ---------------------------------------------------------------------------
# Adoption / purity guards
# ---------------------------------------------------------------------------

def test_monthly_share_size_is_a_thin_compatibility_wrapper():
    """Existing callers (optimize, daily-core backfill) import
    ``monthly.share_size``: it must delegate to the shared helper for every
    input, including the invalid ones the old inline formula raised on."""
    grid = [(1000.0, 3.0), (1.0, 10000.0), (99.9999, 1e6), (123.45, 5.0),
            (4000.0, 137.31)]
    for budget, price in grid:
        assert monthly.share_size(budget, price) == share_size(budget, price)
    for bad in (float("nan"), float("inf"), -1.0, 0.0, None):
        assert monthly.share_size(1000.0, bad) == 0.0
        assert monthly.share_size(bad, 10.0) == 0.0


def test_module_is_pure_stdlib_only():
    """No DB/strategy imports: the primitive must be importable (and
    inspectable) without pulling the engines' dependency graph."""
    code = (
        "import sys; import app.factor_fills as f;"
        "assert 'sqlalchemy' not in sys.modules, 'sqlalchemy imported';"
        "assert 'pandas' not in sys.modules, 'pandas imported';"
        "assert 'app.db' not in sys.modules, 'app.db imported';"
        "assert 'app.monthly' not in sys.modules, 'app.monthly imported';"
        "print(sorted(m for m in sys.modules if m.startswith('app')))"
    )
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", code], cwd=root,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "['app', 'app.factor_fills']"
    assert factor_fills.__all__ == ["Fill", "buy_fill", "sell_fill", "share_size"]
