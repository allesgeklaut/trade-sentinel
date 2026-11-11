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

Dated money-weighted return (XIRR)
----------------------------------
``xirr`` answers the other question the time-weighted NAV cannot: what
annualized rate did THIS cash-flow schedule actually earn?  It discounts the
dated external flows (the same ``invested`` series, day-over-day deltas) and
the terminal portfolio value to NPV = 0 with actual/365 day counts — the
Excel XIRR convention.  TWR and XIRR measure different things and both are
reported: TWR is the manager's return per unit of capital, XIRR is the
investor's money-weighted return.  See ``docs/ui-strategy-audit-2026-10-21.md``
("TWR charts and dated XIRR are not delivered").
"""

from __future__ import annotations

import math
from datetime import date, datetime

__all__ = [
    "max_drawdown",
    "unitized_nav",
    "funded_start_index",
    "dated_external_flows",
    "xirr",
]


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


# ---------------------------------------------------------------------------
# Dated money-weighted return (XIRR)
# ---------------------------------------------------------------------------

_DAYS_PER_YEAR = 365.0
# Below a tenth of a cent a "flow" is float noise from re-summing the same
# deposit history, not money moving in or out.
_FLOW_EPSILON = 1e-9
# NPV(0) this close to zero RELATIVE to the gross flows means the schedule
# earned exactly 0%: return 0.0 instead of asking the solver for a root that
# floating point cannot resolve.
_NPV_ZERO_RELATIVE = 1e-12
_MAX_ANNUAL_RATE = 1e6          # +100,000,000%/yr — absurd but finite bound
_MIN_BRACKET_RATE = -0.9999     # NAV factor 1e-4: just short of a total loss


def _as_date(value) -> date | None:
    """Coerce a ``date``, ``datetime`` or ISO-8601 string to a date.

    ``None``/unparseable means "this point cannot be placed in time" — the
    caller skips it (an undated flow cannot be discounted).
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def dated_external_flows(dates, contributions) -> list[tuple[date, float]]:
    """External cash flows derived from a CUMULATIVE contribution series.

    ``flow_t = contributed_t - contributed_{t-1}``: a deposit is a positive
    flow, a withdrawal a negative one.  The first VALID observation is
    measured from a 0 baseline, because that first contribution is how the
    window got funded — the capital contributed by the window's first day is
    an inflow on that day.  (Pre-window market gains are not visible in a
    contribution series; see the ``/api/metrics/xirr`` window caveat.)

    Assumptions / edge cases
    ------------------------
    * ``contributions`` is the same cumulative series ``unitized_nav``
      consumes (the UI's ``contributed`` / the DB's ``allowance_total``).
    * An undated point is skipped: a flow without a date cannot be
      discounted.
    * An invalid contribution (None, non-numeric, NaN, +/-Infinity) is
      skipped and the last VALID baseline is kept — exactly the
      ``unitized_nav`` policy, so a data glitch is never a deposit.
    * Zero (and sub-cent float-noise) deltas are omitted: a day without a
      deposit or withdrawal is not a cash flow.
    * Length mismatches raise — never a silent truncation.

    Returns
    -------
    list[tuple[date, float]]
        ``(date, signed_flow)`` in input order, zero flows omitted.
    """
    if len(dates) != len(contributions):
        raise ValueError(
            f"dates has {len(dates)} points but contributions has "
            f"{len(contributions)} (length mismatch is never truncated silently)")

    flows: list[tuple[date, float]] = []
    previous = 0.0
    for d, c in zip(dates, contributions, strict=True):
        day = _as_date(d)
        if day is None:
            continue
        try:
            value = float(c)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        flow = value - previous
        previous = value
        if abs(flow) > _FLOW_EPSILON:
            flows.append((day, flow))
    return flows


def xirr(dates, flows, final_value, final_date=None) -> float | None:
    """Dated money-weighted annualized return (XIRR), as a decimal fraction.

    Solves ``NPV(r) = 0`` for the annualized rate ``r`` with ACTUAL/365 day
    counts — the Excel ``XIRR`` convention::

        NPV(r) = sum_i  -flow_i / (1 + r) ** ((d_i - d_0) / 365)
                 + final_value / (1 + r) ** ((final_date - d_0) / 365)

    ``flows`` are the investor's external cash flows, positive INTO the
    portfolio (a deposit) and negative OUT (a withdrawal); the sign is
    flipped because paying money in is an outflow to the investor.
    ``final_value`` is the portfolio value on ``final_date`` and is treated as
    a terminal inflow (as if the position were liquidated then).

    Parameters
    ----------
    dates:
        One date per flow (``date``/``datetime``/ISO string).  Undated or
        unparseable entries are skipped; unorderable runs are sorted by date,
        so callers do not have to pre-sort.
    flows:
        Signed external flows, same length as ``dates``.
    final_value:
        Terminal portfolio value.  ``None``/non-finite means "not known" and
        the result is ``None``.
    final_date:
        Date of ``final_value``; defaults to the latest flow date.

    Returns
    -------
    float | None
        The annualized rate as a FRACTION (``0.10`` = +10%/yr), or ``None``
        when the money-weighted return is undefined.  Undefined means: no
        dated flows, no positive terminal value (e.g. a total loss cannot be
        annualized — the limit is -100%, not a solved rate), every flow on
        one date with a non-zero net, or no sign change of NPV over the
        bracket.  Note that alternating signs admit MULTIPLE IRRs; the first
        sign change found is returned (Excel has the same limitation).

    Raises
    ------
    ValueError
        If ``dates`` and ``flows`` differ in length.
    """
    if len(dates) != len(flows):
        raise ValueError(
            f"dates has {len(dates)} points but flows has {len(flows)} "
            "(length mismatch is never truncated silently)")

    pairs: list[tuple[date, float]] = []
    for d, f in zip(dates, flows, strict=True):
        day = _as_date(d)
        if day is None:
            continue
        try:
            value = float(f)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or abs(value) <= _FLOW_EPSILON:
            continue
        pairs.append((day, value))

    try:
        terminal = float(final_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(terminal):
        return None

    if final_date is None:
        if not pairs:
            return None
        end = max(d for d, _ in pairs)
    else:
        end = _as_date(final_date)
        if end is None:
            return None

    # Investor-perspective cash flows: money paid in is negative, the
    # terminal liquidation is positive.
    cashflows: list[tuple[date, float]] = [(d, -v) for d, v in pairs]
    if terminal != 0.0:
        cashflows.append((end, terminal))
    if not cashflows:
        return None

    start = min(d for d, _ in cashflows)
    times = [(d - start).days / _DAYS_PER_YEAR for d, _ in cashflows]
    amounts = [v for _, v in cashflows]

    # Everything lands on one date: NPV does not depend on r, so there is no
    # time dimension to annualize.  A net zero schedule is exactly 0%.
    if max(times) == 0.0:
        return 0.0 if abs(math.fsum(amounts)) <= _FLOW_EPSILON else None

    net_at_zero = math.fsum(amounts)
    gross = math.fsum(abs(a) for a in amounts)
    if abs(net_at_zero) <= _NPV_ZERO_RELATIVE * max(gross, 1.0):
        return 0.0

    def npv(rate: float) -> float:
        if rate <= -1.0:
            return math.inf
        total = 0.0
        for cf, t in zip(amounts, times, strict=True):
            try:
                factor = (1.0 + rate) ** t
            except OverflowError:      # discount factor beyond float range -> term ~ 0
                continue
            if factor == 0.0:          # rate ~ -100% over a long horizon: unbounded
                return math.inf if cf > 0.0 else -math.inf
            total += cf / factor
        return total

    # Bracket the root.  NPV is monotone in r for a single sign change
    # (deposits, then one terminal value), so bisection is sufficient and
    # deterministic; the expanding bound keeps a short, extreme gain inside
    # the bracket instead of failing.
    lo = _MIN_BRACKET_RATE
    hi = 1.0
    f_lo = npv(lo)
    f_hi = npv(hi)
    while f_lo * f_hi > 0 and hi < _MAX_ANNUAL_RATE:
        hi = min(hi * 2.0, _MAX_ANNUAL_RATE)
        f_hi = npv(hi)
    if f_lo * f_hi > 0:
        return None

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo <= 1e-12:
            return mid
        f_mid = npv(mid)
        if f_mid == 0.0:
            return mid
        if (f_mid < 0.0) == (f_lo < 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
