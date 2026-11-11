"""Zero-fee execution primitives shared by the paper factor portfolios.

This module is the ONE numeric definition of an execution fill for the
factor engines (Monthly today; Daily-Core and the research replays adopt it
through their own workers). It is deliberately pure: no database, no
settings, no strategy imports — only the standard library — so a live
executor, a persisted backfill and a research replay can all call the same
arithmetic without importing the engine that owns it.

FEE MODEL — ZERO FEES, EXPLICITLY
    Every fill here is **zero-fee**: the paper portfolios are commission-free
    by design (docs/brokers.md), so ``notional`` is exactly
    ``shares * price`` and ``cash_after`` moves by exactly that notional. No
    commission, spread, slippage or tax term exists in these formulas. A
    fee-aware adaptation belongs to the research replays (which already model
    a one-way cost) and MUST be added as a wrapper around these primitives —
    never by mutating the sizes here — or the paper book and its replay
    diverge.

SIZING (frozen Phase 1 contract)
    * ``shares = floor(budget / price * 10000) / 10000`` — fractional shares
      floored to 4 decimal places (``_SHARE_QUANTUM``).
    * a leg under $1 of budget is not executable (``_MIN_BUDGET``);
    * a leg under ``0.0001`` shares is not executable (``_MIN_SHARES``);
    * a BUY whose booked notional exceeds the account's EXACT cash is
      rejected — it is never re-sized down to the cash. That razor edge is
      real: when a caller passes a rounded budget (``round(cash, 2)``) the
      4dp share floor can land a hair above the exact cash, and the live
      executor skips the leg rather than booking a trade it would refuse;
    * a SELL liquidates EVERY POSITIVE share supplied — no re-flooring, no
      rounding. The caller owns the position's share count.

INVALID INPUT CONTRACT
    Non-numeric, non-finite (NaN/inf) or non-positive prices, non-numeric /
    non-finite / negative cash, budget, shares (and arithmetic overflow) are
    all rejected as non-executable: ``share_size`` returns ``0.0`` and the
    fill builders return ``None``. They never raise and never return a
    partial/shrunken trade, so a caller's "no fill" branch is the only
    failure mode it has to handle.

FLOAT ORDER
    On valid inputs the operations are performed in the historical order
    (``budget / price * 10000``, then ``shares * price``, then
    ``cash - notional`` / ``cash + notional``) so live and replay fill
    values stay bit-identical to the Phase 1 trade sets
    (tests/test_cadence_parity.py, tests/test_factor_fills.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

__all__ = ["Fill", "buy_fill", "sell_fill", "share_size"]

# Share quantum: fractional shares are floored to 1e-4 (4 decimal places).
_SHARE_QUANTUM = 1e-4
# floor(budget / price * _SHARE_SCALE) / _SHARE_SCALE == floor4 sizing.
_SHARE_SCALE = 10000.0
# A leg below one dollar of budget is not worth executing.
_MIN_BUDGET = 1.0
# A leg below one share quantum is not executable.
_MIN_SHARES = _SHARE_QUANTUM


@dataclass(frozen=True, slots=True)
class Fill:
    """One executed zero-fee fill (immutable).

    ``shares`` and ``price`` are the executed quantities, ``notional`` is
    ``shares * price`` (the cash value traded, fees excluded by contract)
    and ``cash_after`` is the account cash once the fill has settled. Only
    the primitives in this module construct a ``Fill``; they guarantee every
    field is finite, ``shares``/``price`` are positive and ``notional`` /
    ``cash_after`` are non-negative.
    """

    shares: float
    price: float
    notional: float
    cash_after: float


def _finite(value: object) -> float | None:
    """Coerce a numeric scalar to a finite ``float``, or ``None``.

    Non-numeric objects (strings, ``None``, ``Decimal``), booleans (not a
    price or a cash amount) and non-finite values (NaN/inf)
    are unusable; an integer too large for a float overflows and is unusable
    too. Callers turn ``None`` into "no fill".
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def share_size(budget: float, price: float) -> float:
    """Fractional-share size toward ``budget`` at ``price``, floored to 4dp.

    Returns ``0.0`` when the leg is not executable: unusable (non-numeric /
    non-finite / non-positive) price, unusable or negative budget, a budget
    below $1, or an arithmetic overflow in the size itself. Never raises.
    """
    b = _finite(budget)
    p = _finite(price)
    if b is None or p is None or p <= 0 or b < _MIN_BUDGET:
        return 0.0
    raw = b / p  # may overflow to +inf for a subnormal price
    if not math.isfinite(raw):
        return 0.0
    scaled = raw * _SHARE_SCALE
    if not math.isfinite(scaled):
        return 0.0
    return math.floor(scaled) / _SHARE_SCALE


def buy_fill(cash: float, budget: float, price: float) -> Fill | None:
    """A zero-fee BUY of ``share_size(budget, price)`` shares at ``price``.

    ``None`` (no fill, no mutation) when the leg is not executable: invalid
    inputs (see module docstring), sub-minimum sizing (``< 0.0001`` shares),
    a booked ``notional`` above the EXACT ``cash`` — the budget is never
    clamped down to the cash — or an overflow. On success the cash outlay is
    exactly ``shares * price``.
    """
    c = _finite(cash)
    b = _finite(budget)
    p = _finite(price)
    if c is None or c < 0 or b is None or b < 0 or p is None or p <= 0:
        return None
    shares = share_size(b, p)
    if shares < _MIN_SHARES:
        return None
    notional = shares * p
    if not math.isfinite(notional):
        return None
    if notional > c:  # EXACT cash guard — a shrunken trade is never booked
        return None
    cash_after = c - notional
    if not math.isfinite(cash_after) or cash_after < 0:
        return None
    return Fill(shares=shares, price=p, notional=notional, cash_after=cash_after)


def sell_fill(cash: float, shares: float, price: float) -> Fill | None:
    """A zero-fee SELL liquidating ALL positive ``shares`` at ``price``.

    The supplied share count is used as-is (no re-flooring — the position's
    own quantity is what gets sold). ``None`` (no fill, no mutation) when the
    leg is not executable: invalid inputs (see module docstring), a
    non-positive share count, or an overflow. On success the cash proceeds
    are exactly ``shares * price``.
    """
    c = _finite(cash)
    q = _finite(shares)
    p = _finite(price)
    if c is None or c < 0 or q is None or q <= 0 or p is None or p <= 0:
        return None
    notional = q * p
    if not math.isfinite(notional):
        return None
    cash_after = c + notional
    if not math.isfinite(cash_after):
        return None
    return Fill(shares=q, price=p, notional=notional, cash_after=cash_after)
