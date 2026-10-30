"""Sizing-clock causality in the RESEARCH daily-core replay.

``optimize._daily_core_backtest`` places every fill at F's strict exact-date
OPEN (session fill contract, app/session.py). Everything that SIZES those
fills must therefore be knowable when the order is placed:

  * the target weight / portfolio equity,
  * the current held value behind every deployment gap,
  * the vol-management deployment room,

are all marked at **S's close** — S = the previous stored session, the same S
the ranking frame and every overlay read — carrying the last stored close
forward when S has no bar for a name (causal prior-close carry,
``sizing_px_of``). F's own close is a REPORTING mark: it values the
end-of-session equity, the log-return/vol series and the drawdown after the
fills are booked, and it becomes a decision/sizing input only when it IS the
next session's S.

The regression this file pins (the confirmed look-ahead): the BUY phase used
F's close for ``positions_value``/``equity_now``/``weight`` and for each
gap's ``cur``, so a name that moved into its own close sized its own F-day
top-up — an F-close-only mutation changed F's OPEN fill sizes. The tests
below mutate NOTHING but one held ticker's close on F (opens identical, every
close at/before S identical) and require F's legs — (date, side, ticker,
shares, fill price) — to be bit-identical, while the same mutation must be
visible in the end-of-session mark and, once F becomes S, in the NEXT
session's sizing.

The tapes keep ``open != close`` on every bar (opens 99, closes 100) so a
close-priced fill or a close-priced gap can never pass by accident.

Covered sizing variants of THIS engine: rank deployment (and its
inverse-vol weighting), the month-end pro-rata rebuild (``dca="monthly"``),
the daily most-underweight top-up (``dca="daily"``), the release+refill
replacement path, and the target-vol deployment cap. Self-contained: no test
module is imported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import monthly as monthly_mod
from app import optimize
from app.config import settings

CTICKERS = ["A01", "B02", "C03", "D04", "E05", "F06", "G07", "H08"]

# A session the replay trades on: it receives the February contribution and is
# the FIRST session after the January month-end (the deferred rebuild day).
X = pd.Timestamp("2025-02-03")
# The next session: a non-rebuild day, where the daily top-up spends the cash
# the rebuild day left in the account (``dca="daily"``).
X1 = pd.Timestamp("2025-02-04")

# The month-end frames the replay ranks against. The pre-window frame keeps
# the December build running; STABLE is the unchanged ranking.
STABLE = CTICKERS
ROTATED = ["D04", "E05", "F06", "G07", "A01", "B02", "C03", "H08"]


def _canned_frame(order: list[str]) -> pd.DataFrame:
    """A frame whose ``_qv_order`` equals ``order`` (roe decays with rank)."""
    roe = {t: 0.5 - 0.01 * i for i, t in enumerate(order)}
    return pd.DataFrame({t: {"roe": roe.get(t, 0.001), "p_fcf": 20.0,
                             "mcap": 1e10, "dollar_vol": 1e8, "mom": 0.1,
                             "price": 100.0, "vol": 0.2, "eligible": True}
                         for t in CTICKERS}).T


def _frames(month_end_order: list[str] | None = None,
            ) -> dict[pd.Timestamp, pd.DataFrame]:
    """Canned month-end frames: Nov-29-2024 (the pre-window frame that funds
    the December build), Dec-31, Jan-31 and Feb-28-2025."""
    order = month_end_order or STABLE
    return {pd.Timestamp("2024-11-29"): _canned_frame(STABLE),
            pd.Timestamp("2024-12-31"): _canned_frame(STABLE),
            pd.Timestamp("2025-01-31"): _canned_frame(order),
            pd.Timestamp("2025-02-28"): _canned_frame(STABLE)}


def _flat_tape() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(close, vol, open) with a REAL open/close gap: closes flat 100, opens
    flat 99 on every bar. A close fill (or a close-priced gap) would change
    the booked shares/price, which the flat closes make immediately visible."""
    dates = pd.bdate_range("2023-06-01", "2025-02-28")   # 253-day warmup
    close = pd.DataFrame(100.0, index=dates, columns=CTICKERS)
    vol = pd.DataFrame(1e6, index=dates, columns=CTICKERS)
    open_px = pd.DataFrame(99.0, index=dates, columns=CTICKERS)
    return close, vol, open_px


def _volatile_tape() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A ±1.5% alternating tape (realized vol well above a 20% target) with
    ``open = 0.99 * close`` so the vol-management cap is live and a close
    fill is distinguishable."""
    dates = pd.bdate_range("2023-06-01", "2025-02-28")
    steps = np.where(np.arange(len(dates)) % 2 == 0, 0.015, -0.015)
    path = np.cumprod(1.0 + steps)
    close = pd.DataFrame(np.outer(path, np.ones(len(CTICKERS))),
                         index=dates, columns=CTICKERS) * 100.0
    vol = pd.DataFrame(1e6, index=dates, columns=CTICKERS)
    return close, vol, close * 0.99


@pytest.fixture
def small_slots(monkeypatch):
    """Shrink the slot knobs so an 8-name market exercises the 3-slot book."""
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 6)
    monkeypatch.setattr(settings, "sim_daily_core_mom_variant", "raw")
    monkeypatch.setattr(settings, "sim_monthly_rank_boost", 0.0)


async def _run(monkeypatch, close, vol, open_px, frames, *, dca="rank",
               end=None, **kw):
    """Run the research daily-core replay on a canned tape."""
    monkeypatch.setattr(
        monthly_mod, "eligible_frame",
        lambda d, c, v, f: frames.get(pd.Timestamp(d).normalize()))
    return await optimize._daily_core_backtest(
        "2024-12-16", end, "diversified-plus", 1000.0, False,
        "none", "daily", False, dca,
        frame_cache=None, close_preloaded=(close, vol), fund_preloaded={},
        open_preloaded=open_px, cost_oneway=0.0, with_baseline=False, **kw)


def _legs(res, lo=None, hi=None) -> list[tuple[str, str, str, float, float]]:
    """The booked legs as (date, side, ticker, shares, fill price)."""
    return [(t["date"], t["side"], t["ticker"], t["shares"], t["price"])
            for t in res.trades
            if (lo is None or t["date"] >= lo) and (hi is None or t["date"] <= hi)]


def _shares_held_through(res, ticker: str, day: str) -> float:
    """Net shares of ``ticker`` after the fills on ``day``."""
    held = 0.0
    for t in res.trades:
        if t["date"] <= day and t["ticker"] == ticker:
            held += t["shares"] if t["side"] == "BUY" else -t["shares"]
    return held


def _mutated_close(close: pd.DataFrame, day: pd.Timestamp, ticker: str,
                   factor: float) -> pd.DataFrame:
    """The tape with ONLY ``ticker``'s close on ``day`` multiplied — opens and
    every other close (including all closes <= S = the previous session) are
    untouched."""
    out = close.copy()
    out.loc[day, ticker] = float(close.at[day, ticker]) * factor
    return out


def _assert_mutation_is_confined(close, close_m, open_px, day, ticker):
    """Guard: the mutated tape differs from the base tape in exactly one cell
    (a CLOSE), so any change on/after ``day`` cannot come from a fill price
    or from a decision bar at/before S."""
    diff = (close_m != close) & ~(close_m.isna() & close.isna())
    assert not diff.drop(columns=[ticker]).to_numpy().any()
    assert list(diff.index[diff[ticker]]) == [day]
    assert float(close_m.at[day, ticker]) != float(close.at[day, ticker])
    # the fill price is unchanged everywhere (strict opens are not mutated)
    assert open_px.at[day, ticker] != float(close.at[day, ticker])


# ---------------------------------------------------------------------------
# Fresh contribution + nonempty book: rank and pro-rata sizing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dca,vol_weight", [
    ("rank", False),      # rank-weighted deployment, equal weights
    ("rank", True),       # ... with inverse-vol target weights
    ("monthly", False),   # month-end pro-rata equal-weight rebuild
    ("monthly", True),    # ... with inverse-vol target weights
])
async def test_fresh_contribution_sizing_ignores_fs_own_close(
        small_slots, monkeypatch, dca, vol_weight):
    """X receives the February contribution with a nonempty book, so the
    deployment gap is sized on real cash + held value. Mutating ONE held
    name's close on X must not move a single X-day leg: the budget is struck
    at X's OPEN, so X's close cannot size it.

    Factor 1.4 and 0.615 (±~40%) both leave X's legs identical to the
    unmutated run — the invariance is not about the specific mark.
    """
    close, vol, open_px = _flat_tape()
    kw = {"dca": dca, "vol_weight": vol_weight}
    base = await _run(monkeypatch, close, vol, open_px, _frames(), **kw)

    # the tape exercises the path: a book was built BEFORE X (every month-end
    # contribution deployed), and X trades a top-up.
    assert len(_legs(base, None, "2025-02-02")) >= 3
    legs_x = _legs(base, X.strftime("%Y-%m-%d"), X.strftime("%Y-%m-%d"))
    assert legs_x, f"{dca} booked no leg on X"
    assert all(side == "BUY" for _d, side, _t, _sh, _p in legs_x)

    for factor in (1.4, 0.615):
        close_m = _mutated_close(close, X, "A01", factor)
        _assert_mutation_is_confined(close, close_m, open_px, X, "A01")
        mut = await _run(monkeypatch, close_m, vol, open_px, _frames(), **kw)
        # Nothing at/before F moves; from the NEXT session on the mutated
        # close is legitimately S (and there is no cash left to deploy on
        # 2025-02-04 in this variant, so no later leg changes either).
        assert _legs(mut, None, X.strftime("%Y-%m-%d")) == \
            _legs(base, None, X.strftime("%Y-%m-%d"))
        # every X-day fill prices at the strict OPEN (99), never at the close
        for _d, _side, _t, _sh, price in _legs(mut, X.strftime("%Y-%m-%d"),
                                               X.strftime("%Y-%m-%d")):
            assert price == pytest.approx(99.0)
            assert price != pytest.approx(100.0)

        # the mutation IS material — it marks the END-OF-SESSION equity
        # (after the fills) at the mutated close. Run the same window cut at
        # X: identical legs, and the final value moves by exactly the mark
        # delta on the shares held.
        eod_base = await _run(monkeypatch, close, vol, open_px, _frames(),
                              end=X.strftime("%Y-%m-%d"), **kw)
        eod_mut = await _run(monkeypatch, close_m, vol, open_px, _frames(),
                             end=X.strftime("%Y-%m-%d"), **kw)
        assert _legs(eod_mut) == _legs(eod_base)
        held = _shares_held_through(eod_base, "A01", X.strftime("%Y-%m-%d"))
        assert held > 0
        delta = ((float(close_m.at[X, "A01"]) - float(close.at[X, "A01"]))
                 * held)
        assert eod_mut.final_value - eod_base.final_value == pytest.approx(
            delta, rel=1e-9, abs=1e-6)


# ---------------------------------------------------------------------------
# The mark is S's close specifically — not F's, not an earlier candle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dca", ["rank", "monthly"])
async def test_sizing_mark_is_s_close_not_f_nor_an_earlier_candle(
        small_slots, monkeypatch, dca):
    """Complement to the invariance tests: raising A01's close ON S
    (2025-01-31) — F's bar and every earlier close stay at 100 — must shrink
    A01's X-day top-up, because the gap's current value is marked at S's
    completed close.

    On the flat tape every top-up is equal (~$333 each); after the S-close
    raise A01 is already worth more, so it gets a materially smaller buy while
    the two untouched names split the rest — and every fill still prices at
    X's strict OPEN. A mark taken at F's close (100 on X) or at an earlier
    close (100) would leave all three equal, so this pins WHICH known price
    sizes the order, not merely that F's is excluded.
    """
    close, vol, open_px = _flat_tape()
    prev = pd.Timestamp("2025-01-31")
    close_s = _mutated_close(close, prev, "A01", 1.4)
    day = X.strftime("%Y-%m-%d")

    base = await _run(monkeypatch, close, vol, open_px, _frames(), dca=dca)
    base_by_ticker = {leg[2]: leg[3] * leg[4] for leg in _legs(base, day, day)}
    assert set(base_by_ticker) == {"A01", "B02", "C03"}
    assert max(base_by_ticker.values()) - min(base_by_ticker.values()) <= 0.02, (
        "the flat tape should size the three top-ups equally (to the cent)")

    srun = await _run(monkeypatch, close_s, vol, open_px, _frames(), dca=dca)
    got = {leg[2]: leg[3] * leg[4] for leg in _legs(srun, day, day)}
    assert set(got) == {"A01", "B02", "C03"}
    assert got["A01"] < 0.6 * got["B02"], (
        "A01 is marked at S's elevated close, so its gap must be smaller")
    assert got["A01"] < base_by_ticker["A01"]
    assert got["B02"] > base_by_ticker["B02"]      # the freed budget is deployed
    assert got["B02"] == pytest.approx(got["C03"], abs=0.02)
    for leg in _legs(srun, day, day):
        assert leg[4] == pytest.approx(99.0)       # strict OPEN, not the close


# ---------------------------------------------------------------------------
# Daily most-underweight top-up: X's close is invisible on X, visible at X+1
# ---------------------------------------------------------------------------

async def test_daily_topup_ignores_fs_own_close_then_reads_it_as_s(
        small_slots, monkeypatch):
    """``dca="daily"`` tops up the single most underweight holding. On X1
    (2025-02-04) the book is nonempty and the February cash is still in the
    account (the rebuild day left it there), so the top-up is sized on real
    cash + held value.

    * mutating X1's own close leaves X1's OPEN fill untouched;
    * the SAME mutation, now known because X1 is S for the next session,
      changes the 2025-02-05 top-up — so the changed close is genuinely an
      input, just one session later.
    """
    close, vol, open_px = _flat_tape()
    day, nxt = X1.strftime("%Y-%m-%d"), "2025-02-05"
    base = await _run(monkeypatch, close, vol, open_px, _frames(), dca="daily")
    legs_day = _legs(base, day, day)
    assert legs_day and all(side == "BUY" for _d, side, _t, _s, _p in legs_day)

    close_m = _mutated_close(close, X1, "A01", 1.4)
    _assert_mutation_is_confined(close, close_m, open_px, X1, "A01")
    mut = await _run(monkeypatch, close_m, vol, open_px, _frames(), dca="daily")

    assert _legs(mut, day, day) == legs_day
    assert {p for _d, _s, _t, _sh, p in _legs(mut, day, day)} == {99.0}

    # ... and one session later the mutated close IS an input: A01 now marks
    # above its target, so the top-up (weight and gap both move) books a
    # different size at the still-strict open.
    assert _legs(mut, nxt, nxt) != _legs(base, nxt, nxt)
    assert _legs(mut, nxt, nxt)
    for _d, _s, _t, _sh, price in _legs(mut, nxt, nxt):
        assert price == pytest.approx(99.0)

    # EOD mark: the cut-at-X1 window books identical legs and marks the
    # mutated close after the fills.
    eod_base = await _run(monkeypatch, close, vol, open_px, _frames(),
                          dca="daily", end=day)
    eod_mut = await _run(monkeypatch, close_m, vol, open_px, _frames(),
                         dca="daily", end=day)
    assert _legs(eod_mut) == _legs(eod_base)
    held = _shares_held_through(eod_base, "A01", day)
    assert held > 0
    assert eod_mut.final_value - eod_base.final_value == pytest.approx(
        (float(close_m.at[X1, "A01"]) - float(close.at[X1, "A01"])) * held,
        rel=1e-9, abs=1e-6)


# ---------------------------------------------------------------------------
# Replacement (release + refill) on the deferred rebuild session
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dca", ["rank", "monthly"])
async def test_replacement_sizing_ignores_fs_own_close(
        small_slots, monkeypatch, dca):
    """The January ranking rotates, so the deferred rebuild session X
    releases C03 and opens D04 while topping up the kept names — a
    replacement whose sizes are struck on the post-release cash and the held
    book (rank deployment and the month-end pro-rata both do this on the
    rebuild session). Mutating the held A01's X close must not move the
    release, the refill or the top-ups."""
    close, vol, open_px = _flat_tape()
    frames = _frames(month_end_order=ROTATED)
    base = await _run(monkeypatch, close, vol, open_px, frames, dca=dca)
    legs_x = _legs(base, X.strftime("%Y-%m-%d"), X.strftime("%Y-%m-%d"))
    assert [leg[1] for leg in legs_x].count("SELL") == 1
    assert ("SELL", "C03") in {(leg[1], leg[2]) for leg in legs_x}
    assert ("BUY", "D04") in {(leg[1], leg[2]) for leg in legs_x}
    assert {"A01", "B02"} <= {leg[2] for leg in legs_x if leg[1] == "BUY"}

    close_m = _mutated_close(close, X, "A01", 1.4)
    _assert_mutation_is_confined(close, close_m, open_px, X, "A01")
    mut = await _run(monkeypatch, close_m, vol, open_px, frames, dca=dca)
    assert _legs(mut, None, X.strftime("%Y-%m-%d")) == \
        _legs(base, None, X.strftime("%Y-%m-%d")), (
        "the replacement session sized on F's own close")

    # capacity survives: never more than the 3 slots
    held: set[str] = set()
    for _d, side, ticker, _sh, price in _legs(mut):
        if side == "BUY":
            held.add(ticker)
        else:
            held.discard(ticker)
        assert len(held) <= 3
        assert price == pytest.approx(99.0)

    # the mutated close is still a REPORTING mark
    eod_base = await _run(monkeypatch, close, vol, open_px, frames,
                          dca=dca, end=X.strftime("%Y-%m-%d"))
    eod_mut = await _run(monkeypatch, close_m, vol, open_px, frames,
                         dca=dca, end=X.strftime("%Y-%m-%d"))
    assert _legs(eod_mut) == _legs(eod_base)
    held_a01 = _shares_held_through(eod_base, "A01", X.strftime("%Y-%m-%d"))
    assert held_a01 > 0
    assert eod_mut.final_value - eod_base.final_value == pytest.approx(
        (float(close_m.at[X, "A01"]) - float(close.at[X, "A01"])) * held_a01,
        rel=1e-9, abs=1e-6)


# ---------------------------------------------------------------------------
# Target-vol deployment room: S's closes and returns, never F's own
# ---------------------------------------------------------------------------

async def test_target_vol_deployment_room_ignores_fs_own_close(
        small_slots, monkeypatch):
    """Barroso-Santa-Clara vol management caps deployment at
    ``equity * target_vol / realized_vol``. Both the equity side and the
    realized-vol series must be known at F's open: the equity is marked at
    S's close (sizing equity) and the returns come from the EOD series that
    ends at S (F's own just-closed return is recorded after the fills).

    The volatile tape makes the cap bind on X (fewer/ smaller legs than the
    uncapped run). Mutating A01's X close by +40% must leave X's legs
    untouched — a cap or a vol input that read F's own close would move them.
    """
    close, vol, open_px = _volatile_tape()
    uncapped = await _run(monkeypatch, close, vol, open_px, _frames(),
                          dca="rank")
    base = await _run(monkeypatch, close, vol, open_px, _frames(),
                      dca="rank", target_vol=0.20)
    day = X.strftime("%Y-%m-%d")
    assert _legs(uncapped, day, day)            # the tape trades on X ...
    assert len(_legs(base, day, day)) < len(_legs(uncapped, day, day)), (
        "the target-vol cap did not bind — the test would be vacuous")

    close_m = _mutated_close(close, X, "A01", 1.4)
    _assert_mutation_is_confined(close, close_m, open_px, X, "A01")
    mut = await _run(monkeypatch, close_m, vol, open_px, _frames(),
                     dca="rank", target_vol=0.20)
    assert _legs(mut, day, day) == _legs(base, day, day), (
        "the deployment cap read F's own close (equity or just-closed return)")
    for _d, _s, _t, _sh, price in _legs(mut, day, day):
        assert price == pytest.approx(float(open_px.at[X, _t]))
        assert price != pytest.approx(float(close.at[X, _t]))

    eod_base = await _run(monkeypatch, close, vol, open_px, _frames(),
                          dca="rank", target_vol=0.20, end=day)
    eod_mut = await _run(monkeypatch, close_m, vol, open_px, _frames(),
                         dca="rank", target_vol=0.20, end=day)
    assert _legs(eod_mut) == _legs(eod_base)
    held = _shares_held_through(base, "A01", day)
    assert held > 0
    delta = ((float(close_m.at[X, "A01"]) - float(close.at[X, "A01"])) * held)
    assert eod_mut.final_value - eod_base.final_value == pytest.approx(
        delta, rel=1e-9, abs=1e-6)
