"""Shared cash-flow-neutral (time-weighted) performance metrics.

Phase 1 "financial correctness" — one definition of drawdown for the backend
and the UI.  See ``docs/ui-strategy-audit-2026-10-21.md`` (finding P0-B).

Why the previous convention was wrong
-------------------------------------
``app/optimize.py`` and ``static/index.html`` measured the drawdown of
``equity / cumulative_contributions``.  That is not a drawdown of the
portfolio: the denominator steps up on every deposit, so the "drawdown"
mostly measures the deposit schedule.  The audit's reproduction: invest
1000, grow to 1200, then deposit 1000 with no price move -> equity 2200,
contributions 2000, the ratio falls from 1.20 to 1.10 and the app reported
an **8.33% drawdown** although not a single unit of capital was lost.

The convention used here (Portfolio Performance / time-weighted return)
-----------------------------------------------------------------------
External money in or out is not a return.  With ``invested`` = the
CUMULATIVE external contribution per point:

    flow_t = invested_t - invested_{t-1}     external net flow on day t
    base_t = equity_{t-1} + flow_t           value at the start of day t
    nav_t  = nav_{t-1} * equity_t / base_t   chained unitized NAV

The chain starts at ``nav = 1.0`` on the first funded point, so the max
drawdown of the NAV is the real peak-to-trough loss: a deposit or a
withdrawal that leaves the portfolio value unchanged leaves the NAV
unchanged (and therefore the drawdown unchanged).

This is the same contribution-adjusted daily return the optimizer already
uses for its Sharpe (``base = port_eq[-2] + contrib_today``), so the
drawdown and the Sharpe now measure the same thing.

Conventions (also mirrored by ``mddOf`` in ``static/index.html``)
----------------------------------------------------------------
* ``invested`` is the cumulative contribution per point — the same shape as
  ``optimize.py``'s ``invested_curve`` and the UI's ``contributed`` field.
* Points before the account is funded (value <= 0) are skipped: the chain
  starts at the first point with a positive value.
* A day whose beginning-of-day base is not positive (e.g. a withdrawal
  larger than the portfolio) has no computable return — the NAV is carried
  forward unchanged.  Withdrawing *everything* is not a loss.
* A total loss is NOT skipped: a value of 0 on a positive base is a real
  -100% return, the NAV goes to 0 and the drawdown is 100%.
* After a total loss the chain is broken; the next funded point restarts the
  NAV at 1.0 (the 100% drawdown has already been recorded).
* An INVALID observation (None, a non-numeric value, NaN, +/-Infinity — in
  either the equity or the contribution series) is skipped: the NAV is
  carried forward unchanged and the last VALID equity/contribution
  baseline is kept, so a bad data point can never look like a loss.  The JS
  counterpart (``mddOf``) applies exactly the same policy.
* Length mismatches are an error, never a silent truncation.
"""

from __future__ import annotations

import math

__all__ = ["max_drawdown", "unitized_nav", "funded_start_index"]


def _floats(values) -> list[float]:
    """Coerce a sequence to floats; invalid entries become NaN.

    Invalid means: ``None``, a non-numeric value, NaN or +/-Infinity.  NaN
    marks an invalid observation: such points are skipped (the NAV is
    carried forward, the last valid baseline is kept) instead of poisoning
    the whole chain or masquerading as a total loss.

    The JS counterpart (``mddOf`` in ``static/index.html``) uses the same
    policy: null / undefined / non-numeric / NaN / Infinity are invalid.
    """
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        out.append(f if math.isfinite(f) else float("nan"))
    return out


def funded_start_index(equity: list[float]) -> int:
    """Index of the first point with a positive value — where the NAV chain
    starts.  ``-1`` when the curve never gets funded (nothing to measure)."""
    for i, v in enumerate(_floats(equity)):
        if math.isfinite(v) and v > 0:
            return i
    return -1


def unitized_nav(equity: list[float], invested: list[float] | None = None) -> list[float]:
    """Cash-flow-adjusted (time-weighted) NAV series, aligned with ``equity``.

    Parameters
    ----------
    equity:
        Portfolio value per point (end of day).
    invested:
        CUMULATIVE external contributions per point.  ``None`` means "no
        external flows known" — the chain then behaves like a plain
        equity curve normalized to its first funded point.

    Returns
    -------
    list[float]
        One NAV value per input point.  Points before the chain starts are
        ``0.0`` (the account was not funded yet).  Afterwards the values are
        the chained NAV; a ``0.0`` means the portfolio was wiped out (chain
        broken) and the next funded point restarts at ``1.0``.  An invalid
        observation (None / non-numeric / NaN / Infinity) carries the
        current NAV forward — it is never a ``0.0``.

    Raises
    ------
    ValueError
        If ``invested`` has a different length than ``equity``.
    """
    eq = _floats(equity)
    n = len(eq)
    if invested is None:
        inv = [0.0] * n
    else:
        if len(invested) != n:
            raise ValueError(
                f"invested has {len(invested)} points but equity has {n} "
                "(length mismatch is never truncated silently)")
        inv = _floats(invested)

    nav: list[float] = [0.0] * n
    started = False
    cur = 0.0          # current NAV (0.0 = chain not started / broken)
    prev_equity = 0.0
    prev_invested = 0.0

    for i in range(n):
        e = eq[i]
        c = inv[i]
        if not (math.isfinite(e) and math.isfinite(c)):
            # Invalid observation (None / non-numeric / NaN / Infinity):
            # carry the NAV forward and keep the last VALID equity and
            # contribution baseline.  It must NOT become a 0.0 NAV value -
            # that would look like a total loss and fabricate a 100%
            # drawdown out of a data glitch.
            if started:
                nav[i] = cur
            continue
        if not started:
            if e <= 0:
                continue  # not funded yet -> skip the startup
            started = True
            cur = 1.0
            nav[i] = cur
            prev_equity = e
            prev_invested = c
            continue
        flow = c - prev_invested          # external net flow (may be negative)
        base = prev_equity + flow         # beginning-of-day value
        if base > 0:
            if cur <= 0 and e > 0:
                cur = 1.0                 # chain broken by a total loss -> restart
            else:
                cur = cur * (e / base)    # chain the cash-flow-adjusted return
        # base <= 0: no computable return (withdrawal >= the portfolio value)
        # -> the NAV is carried forward unchanged.  A full withdrawal is not
        # a loss.
        nav[i] = cur
        prev_equity = e
        prev_invested = c

    return nav


def _peak_to_trough(values: list[float]) -> float:
    """Max peak-to-trough decline of a series, in PERCENT (0.0 = no decline)."""
    peak = 0.0
    mdd = 0.0
    for v in values:
        if not math.isfinite(v):
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100.0
            if dd > mdd:
                mdd = dd
    return mdd


def max_drawdown(equity: list[float], invested: list[float] | None = None) -> float:
    """Maximum peak-to-trough drawdown, in PERCENT (0.0 = no drawdown).

    With ``invested`` (the cumulative external contributions per point) the
    drawdown is measured on the cash-flow-adjusted, unitized NAV, so
    deposits and withdrawals never look like losses.  Without ``invested``
    the plain equity drawdown is used.

    >>> max_drawdown([1000, 1200, 2200], [1000, 1000, 2000])   # deposit, no loss
    0.0
    >>> round(max_drawdown([1000, 1200, 900], [1000, 1000, 1000]), 6)  # 25% loss
    25.0
    """
    if invested is None:
        return _peak_to_trough(_floats(equity))
    if len(invested) != len(equity):
        raise ValueError(
            f"invested has {len(invested)} points but equity has {len(equity)} "
            "(length mismatch is never truncated silently)")
    return _peak_to_trough(unitized_nav(equity, invested))
