"""Daily-core paper portfolio: the monthly qv-mom strategy as the fundamental
CORE, with daily candle-driven cash deployment on top.

Design (from the §10 backtest A/B, docs/llm-strategy-experiments.md):
  * WHAT to own is decided by the monthly qv-mom ranking — ROE + 12-1
    momentum + 1/P-FCF, top-N with hysteresis, NO stops, NO technical
    entry gates. Fundamentals are the basis for everything.
  * WHEN/HOW is the only thing candles touch: every trading day (at the
    scheduler's run time) free cash is deployed into the highest-ranked
    core names, topping each up toward the equal-weight target. Fresh
    contributions go to work immediately instead of waiting for the
    month-end rebalance — that cash-drag elimination is the measured edge
    (+3..+5.5pp IRR over the monthly sim on 2020-2026 windows).
  * Band releases: same hysteresis as the monthly sim — a held name keeps
    its slot while it stays inside the top ``hold_band`` of the ranking.
    Releases are evaluated at the month-end rebuild (the backtest showed
    exit cadence is irrelevant; the band is sticky).

Point-in-time discipline mirrors the monthly sim: a decision on date d only
sees fundamental facts with end < d and filed <= d. No look-ahead.

Scheduler entry: ``run_daily_cycle()`` — deposits the allowance at the start
of a new operator-local month (same timing as sim/monthly so the three
portfolios' "contributed" figures stay comparable), refreshes data, runs one
deployment pass, and writes the daily equity snapshot.
"""
import asyncio
import logging
import math
from datetime import datetime, timedelta, UTC

import pandas as pd
from sqlalchemy import func, select

from . import fundamentals as fundamentals_mod
from . import monthly as monthly_mod
from .config import settings
from .db import (DailyCoreAccount, DailyCoreAllowance, DailyCorePosition,
                 DailyCoreSnapshot, DailyCoreTrade, Session)
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.daily_core")

_TZ = monthly_mod._TZ  # same operator-local month anchor as sim/monthly


def _current_month() -> str:
    return datetime.now(_TZ).strftime("%Y-%m")


# Serialises cycles: the allowance check runs before the (potentially
# multi-minute) data refresh, so two overlapping invocations would both
# pass the gate and double-deposit / duplicate trades. Mirrors
# sim._run_cycle_lock and monthly._rebalance_lock.
_cycle_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

async def valuate() -> dict:
    """Value the daily-core portfolio (same shape as monthly.monthly_valuate)."""
    async with Session() as s:
        acc = await _account(s)
        positions = (await s.scalars(
            select(DailyCorePosition).order_by(DailyCorePosition.ticker))).all()
        allowance_total = (await s.scalar(
            select(func.sum(DailyCoreAllowance.amount)))) or 0.0

    px_map = await monthly_mod._price_usd_map([p.ticker for p in positions])
    positions_value = 0.0
    pos_list = []
    for p in positions:
        price = px_map.get(p.ticker)
        if price is None:
            price = p.avg_cost
        value = p.shares * price
        positions_value += value
        pnl_pct = ((price - p.avg_cost) / p.avg_cost * 100) if p.avg_cost > 0 else 0.0
        pos_list.append({
            "ticker": p.ticker,
            "shares": round(p.shares, 4),
            "avg_cost": round(p.avg_cost, 4),
            "current_price": round(price, 4),
            "value": round(value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "opened_at": p.opened_at.strftime("%Y-%m-%d") if p.opened_at else "",
        })
    return {
        "cash": round(acc.cash, 2),
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(acc.cash + positions_value, 2),
        "allowance_total": round(float(allowance_total), 2),
    }


async def _account(s) -> DailyCoreAccount:
    acc = await s.get(DailyCoreAccount, 1)
    if acc is None:
        acc = DailyCoreAccount(id=1, cash=settings.sim_monthly_start_cash,
                               last_allowance_month=None)
        s.add(acc)
        await s.commit()
    return acc


# ---------------------------------------------------------------------------
# Allowance / execution
# ---------------------------------------------------------------------------

async def deposit_allowance() -> dict:
    """Deposit the monthly allowance if a new operator-local month has begun.

    Deposits at the START of the month (mirroring sim + monthly portfolios)
    so the cumulative "contributed" figures step together. Idempotent per
    month. The money is deployed by the daily cycles that follow.
    """
    month = _current_month()
    async with Session() as s:
        acc = await _account(s)
        if acc.last_allowance_month == month:
            return {"deposited": False, "month": month, "cash": acc.cash}
        acc.cash += settings.sim_monthly_contribution
        acc.last_allowance_month = month
        s.add(DailyCoreAllowance(amount=settings.sim_monthly_contribution, month=month))
        await s.commit()
        logger.info("Daily-core allowance deposited: %.2f for %s -> cash %.2f",
                    settings.sim_monthly_contribution, month, acc.cash)
        return {"deposited": True, "amount": settings.sim_monthly_contribution,
                "month": month, "cash": acc.cash}


async def _exec_buy(ticker: str, price: float, budget: float, reason: str) -> dict | None:
    if price <= 0 or budget < 1:
        return None
    shares = math.floor(budget / price * 10000) / 10000
    if shares < 0.0001:
        return None
    cost = shares * price
    async with Session() as s:
        acc = await _account(s)
        if acc is None or acc.cash < cost:
            return None
        acc.cash -= cost
        pos = await s.scalar(select(DailyCorePosition).where(DailyCorePosition.ticker == ticker))
        if pos:
            total = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + cost) / total
            pos.shares = total
        else:
            s.add(DailyCorePosition(ticker=ticker, shares=shares, avg_cost=price))
        s.add(DailyCoreTrade(ticker=ticker, side="BUY", shares=shares, price=price,
                             cash_after=acc.cash, reason=reason))
        await s.commit()
        return {"ticker": ticker, "side": "BUY", "shares": shares, "price": price,
                "cost": round(cost, 2), "reason": reason}


async def _exec_sell(ticker: str, price: float, reason: str) -> dict | None:
    async with Session() as s:
        pos = await s.scalar(select(DailyCorePosition).where(DailyCorePosition.ticker == ticker))
        if pos is None:
            return None
        proceeds = pos.shares * price
        acc = await _account(s)
        if acc is None:
            return None
        acc.cash += proceeds
        s.add(DailyCoreTrade(ticker=ticker, side="SELL", shares=pos.shares, price=price,
                             cash_after=acc.cash, reason=reason))
        await s.delete(pos)
        await s.commit()
        return {"ticker": ticker, "side": "SELL", "shares": pos.shares,
                "price": price, "proceeds": round(proceeds, 2), "reason": reason}


# ---------------------------------------------------------------------------
# The deployment decision
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_list(col: str) -> list[str]:
    return [t for t in (col or "").split(",") if t]


async def _load_stored_ranking(max_age_days: float = 1.0) -> tuple[list[str], list[str]] | None:
    """The stored daily ranking, or None when absent/older than a day.

    The fundamentals math runs once per day in the cycle and persists its
    output to the DailyCoreRanking row; every other consumer reads that.
    """
    from .db import DailyCoreRanking
    async with Session() as s:
        row = await s.get(DailyCoreRanking, 1)
    if row is None:
        return None
    age = _utcnow() - row.ranking_date.replace(tzinfo=UTC)
    if age > timedelta(days=max_age_days):
        return None
    return _parse_list(row.band), _parse_list(row.picks)


async def _store_ranking(band: list[str], picks: list[str]) -> None:
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from .db import DailyCoreRanking
    stmt = sqlite_insert(DailyCoreRanking).values(
        id=1, ranking_date=_utcnow().replace(tzinfo=None),
        band=",".join(band), picks=",".join(picks))
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={"ranking_date": stmt.excluded.ranking_date,
              "band": stmt.excluded.band,
              "picks": stmt.excluded.picks,
              "updated_at": stmt.excluded.updated_at})
    async with Session() as s:
        await s.execute(stmt)
        await s.commit()


async def compute_targets(force: bool = False) -> tuple[list[str], list[str], pd.DataFrame | None]:
    """Compute (or read the stored) daily-core target portfolio.

    Returns (band, picks, frame): ``band`` is the full hold-band order
    (top ``sim_monthly_hold_band`` ranked tickers — held names stay while
    inside it), ``picks`` the top ``sim_monthly_target_n`` with hysteresis
    applied. ``frame`` is the eligibility frame (None when no data or when
    the stored ranking was used).

    The full-universe fundamentals math is expensive (~7s), so it runs once
    per day during the daily cycle, which persists the result to the
    ``daily_core_ranking`` row. Other consumers (status endpoint / UI) read
    that stored ranking instead of recomputing. ``force=True`` recomputes
    regardless — used by the deployment cycle and the manual refresh path
    so trades always act on today's data.
    """
    if not force:
        stored = await _load_stored_ranking()
        if stored is not None:
            band, picks = stored
            return band, picks, None
    tickers = universe_tickers(settings.sim_monthly_universe)
    fund = await fundamentals_mod.load_fundamentals(tickers)
    if not fund:
        return [], [], None
    close, vol = await monthly_mod.load_frames(tickers, None)
    if close.empty:
        return [], [], None
    iso = datetime.now(UTC).strftime("%Y-%m-%d")
    d: pd.Timestamp = pd.Timestamp(iso)  # type: ignore[assignment]
    frame = await asyncio.to_thread(monthly_mod.eligible_frame, d, close, vol, fund)
    if frame is None or not bool(frame["eligible"].any()):
        return [], [], frame
    _elig, order = await asyncio.to_thread(monthly_mod._qv_order, frame)
    band = order[:settings.sim_monthly_hold_band]
    async with Session() as s:
        held = [p.ticker for p in (await s.scalars(select(DailyCorePosition))).all()]
    picks = monthly_mod._band_fill(order, held, settings.sim_monthly_target_n,
                                   settings.sim_monthly_hold_band)
    await _store_ranking(band, picks)
    return band, picks, frame


async def run_deployment() -> dict:
    """One daily-core deployment pass.

    1. Compute the qv-mom ranking + hysteresis picks.
    2. SELL held names that fell out of the band (band release, same rule
       as the monthly sim's rebalance).
    3. BUY: deploy all free cash into the top-ranked names, topping each up
       toward the equal-weight target — best rank first, until cash or
       targets are exhausted. This is the backtest-winning "rank deploy,
       boost=0" rule: fresh cash reinforces the top of the ranking, never
       spreads pro-rata, never waits for month-end.
    """
    band, picks, _frame = await compute_targets(force=True)
    if not band:
        return {"skipped": True, "reason": "no eligible ranking (fundamentals too thin?)"}

    async with Session() as s:
        held_rows = (await s.scalars(select(DailyCorePosition))).all()
    held_before = sorted(p.ticker for p in held_rows)

    # SELLs first: held names out of the band release their slot.
    trades: list[dict] = []
    px_map = await monthly_mod._price_usd_map(sorted(set(band) | set(held_before)))
    for t in held_before:
        if t not in band:
            price = px_map.get(t) or 0.0
            if not price:
                # No candles: fall back to avg_cost like the monthly sim — never sell at 0.
                async with Session() as s:
                    pos = await s.scalar(select(DailyCorePosition).where(DailyCorePosition.ticker == t))
                price = pos.avg_cost if pos else 0.0
            r = await _exec_sell(t, price, "daily-core: out of hold band")
            if r:
                trades.append(r)

    # BUYs: rank-first top-up toward equal weight. The rank loop stops at
    # the top target_n names; with weight = equity/target_n, exhausting
    # those exactly consumes the cash (the backtest-winning "rank deploy,
    # boost=0" rule — fresh cash reinforces the top of the ranking, never
    # spreads pro-rata, never waits for month-end).
    valuation = await valuate()
    weight = valuation["total_equity"] / max(settings.sim_monthly_target_n, 1)
    for t in band[:settings.sim_monthly_target_n]:
        price = px_map.get(t)
        if not price:
            continue
        held_pos = next((p for p in valuation["positions"] if p["ticker"] == t), None)
        current_value = held_pos["value"] if held_pos else 0.0
        budget = min(weight - current_value, valuation["cash"])
        if budget < 1:
            continue
        r = await _exec_buy(t, price, budget,
                            f"daily-core deploy: rank target {settings.sim_monthly_target_n}")
        if r:
            trades.append(r)
            # refresh cash so successive buys see the balance
            valuation = await valuate()

    return {"deployed": True, "band": band, "picks": picks,
            "held_before": held_before, "trades": trades,
            "valuation": await valuate()}


# ---------------------------------------------------------------------------
# Scheduler entry
# ---------------------------------------------------------------------------

async def take_snapshot() -> dict:
    """Record one equity-curve snapshot of the current valuation."""
    post = await valuate()
    async with Session() as s:
        s.add(DailyCoreSnapshot(cash=post["cash"], positions_value=post["positions_value"],
                                total_equity=post["total_equity"],
                                allowance_total=post["allowance_total"]))
        await s.commit()
    logger.info("Daily-core snapshot: equity %.2f (cash %.2f, positions %.2f)",
                post["total_equity"], post["cash"], post["positions_value"])
    return {"snapshotted": True, "total_equity": post["total_equity"]}


async def refresh_data() -> tuple[list[str], list[str]]:
    """Refresh candles (+FX) for the universe and the current holdings."""
    from .market import refresh
    async with Session() as s:
        held = [p.ticker for p in (await s.scalars(select(DailyCorePosition))).all()]
    tickers = universe_tickers(settings.sim_monthly_universe)
    errors: list[str] = []
    refreshed: list[str] = []
    for t in list(dict.fromkeys(tickers + held)):
        try:
            await refresh(t, "2y")
            refreshed.append(t)
        except Exception as e:
            errors.append(f"{t}: {e}")
    return refreshed, errors


# ---------------------------------------------------------------------------
# Backfill (synthetic history)
# ---------------------------------------------------------------------------

async def _sync_start_date() -> str | None:
    """Earliest live snapshot date across the other two paper portfolios
    (daily sim + monthly). Used as the daily-core backfill's default start
    so all three equity curves cover the SAME window — the user wants them
    synched, and a 2017 synthetic history next to two 2-week live curves is
    apples-to-oranges on the charts. Returns None when the other tables are
    empty (then the replay falls back to the first eligible month).
    """
    from .db import MonthlySnapshot, Session as DbSession, SimSnapshot

    earliest: datetime | None = None
    async with DbSession() as s:
        d = await s.scalar(select(func.min(SimSnapshot.created_at)))
        if d is not None and (earliest is None or d < earliest):
            earliest = d
    async with DbSession() as s:
        d = await s.scalar(select(func.min(MonthlySnapshot.created_at)))
        if d is not None and (earliest is None or d < earliest):
            earliest = d
    return earliest.strftime("%Y-%m-%d") if earliest else None


async def backfill(start: str | None = None) -> dict:
    """Replay the winning daily-core strategy over historical data and
    REPLACE the portfolio state with the replay's end state.

    Wipes account/positions/trades/allowances/snapshots, then walks every
    stored trading day from `start` to today, depositing the monthly
    allowance and applying the rank-deployment rule at each day's close
    with 10 bps one-way paper costs — the same simulation as
    `optimize daily-core --dca rank` with boost=0.

    ``start`` semantics:
      - explicit "YYYY-MM-DD": replay from that day (the UI's optional date)
      - None (default): synched with the other sims — the replay starts on
        the earliest snapshot date of the daily sim / monthly portfolios so
        all three equity curves cover the same window
      - "all": the full stored history (2017+, first month with an
        eligible frame) — the long-view replay

    This is a paper-portfolio convenience, not a live track record: the
    trades never happened in real time. The final state converges to
    "what the strategy would hold today", and live cycles continue from
    there. Point-in-time discipline: each day only sees fundamentals that
    were public by then (same EDGAR-first fact store as the backtest).
    """
    async with _cycle_lock:
        from . import optimize  # noqa: F401  (ensures strategy modules load)
        from .db import DailyCoreAccount as Acc, DailyCorePosition as Pos, \
            DailyCoreTrade as Tr, DailyCoreAllowance as Al, DailyCoreSnapshot as Sn

        if start is None:
            start = await _sync_start_date()
        elif start == "all":
            start = None  # full stored history
        start_note = start or "first eligible month"

        tickers = universe_tickers(settings.sim_monthly_universe)
        fund = await fundamentals_mod.load_fundamentals(tickers)
        if not fund:
            return {"ok": False, "error": "no fundamentals loaded"}
        close, vol = await monthly_mod.load_frames(tickers, None)
        if close.empty:
            return {"ok": False, "error": "no candle data"}

        idx = pd.DatetimeIndex(close.index)
        if start:
            idx = idx[idx >= pd.Timestamp(start)]
        if idx.empty:
            return {"ok": False, "error": f"no trading days since {start_note}"}

        months: list[pd.Timestamp] = []
        for ym in sorted({(d.year, d.month) for d in idx}):
            sub = idx[(idx.year == ym[0]) & (idx.month == ym[1])]
            if len(sub):
                months.append(sub[-1])

        # Eligibility frames per month (same caching as the backtest), plus
        # the month-end BEFORE the window: a start like 2026-06-01 should
        # replay June 1-29 against May's month-end ranking (facts public by
        # then), not sit idle until June's month-end. Point-in-time safe:
        # the prior frame only uses facts public by its own date.
        prior_month_ends: list[pd.Timestamp] = []
        if months:
            prev = months[0] - pd.offsets.MonthEnd(1)
            prev_idx = close.index[close.index <= prev]
            if len(prev_idx):
                prior_month_ends.append(prev_idx[-1])
        frame_cache: dict[pd.Timestamp, pd.DataFrame | None] = {}
        for m in prior_month_ends + months:
            frame_cache[m] = await asyncio.to_thread(
                monthly_mod.eligible_frame, m, close, vol, fund)
        def _has_eligible(m: pd.Timestamp) -> bool:
            f = frame_cache[m]
            return f is not None and bool(f["eligible"].any())
        months = [m for m in months if _has_eligible(m)]
        if not months:
            return {"ok": False, "error": "no month with eligible names"}
        # No idx clamp to months[0]: days before the first eligible
        # month-end in the window replay against the PRIOR month-end's
        # frame (built above), so start=YYYY-06-01 really starts June 1.
        first_day = idx[0]

        # --- wipe + reset ---
        # Preserve the CURRENT month's live allowance (the real deposit the
        # engine already made): the replay re-creates allowance rows for its
        # own month sequence, which may not include the current month, and
        # the status endpoint's deposit_allowance() would then hit the
        # UNIQUE(month) constraint.
        from sqlalchemy import delete as sa_delete
        live_allowance_month: str | None = None
        async with Session() as s:
            acc0 = await s.get(Acc, 1)
            if acc0 is not None:
                live_allowance_month = acc0.last_allowance_month
            for tbl in (Tr, Pos, Al, Sn):
                await s.execute(sa_delete(tbl))
            acc = await _account(s)
            acc.cash = 0.0
            acc.last_allowance_month = None
            await s.commit()

        cost = settings.sim_monthly_cost_oneway
        target_n = settings.sim_monthly_target_n
        hold_band = settings.sim_monthly_hold_band

        # Valuation prices: forward-filled closes so a ticker without a
        # candle on a given day (holiday, partial today, stale feed) is
        # valued at its LAST known close instead of 0. Without this the
        # replay's equity collapses on days when held names lack candles —
        # the snapshot then records near-zero equity and the UI curve
        # cliffs. Fills also use the carried price, the same convention as
        # the monthly sim's USD-converted valuation.
        close_val = close.ffill()

        # --- in-memory replay state ---
        shares: dict[str, float] = {}
        cash = 0.0
        contributed = 0.0
        n_contribs = 0
        trades: list[tuple[str, str, str, float]] = []
        snaps: list[tuple[str, float, float]] = []

        def px_of(t: str, d: pd.Timestamp) -> float | None:
            return monthly_mod.px_at(close_val, d, t)

        cur_month: int | None = None
        for d in idx:
            # Point-in-time discipline: each day uses the ranking of the most
            # recent month-end AT OR BEFORE d (the frame computed from facts
            # public by then). Using a future month-end's frame would leak
            # look-ahead into the replay.
            prior = [m for m in months if m <= d]
            if not prior:
                continue  # before the first completed month-end — no ranking yet
            month_td = prior[-1]
            frame = frame_cache.get(month_td)
            order: list[str] = []
            if frame is not None and bool(frame["eligible"].any()):
                _elig, order = await asyncio.to_thread(monthly_mod._qv_order, frame)
            band = set(order[:hold_band])

            # allowance: deposit on the first trading day of a new month
            # (same timing as the live deposit_allowance)
            if cur_month is None or d.month != cur_month:
                cur_month = d.month
                cash += settings.sim_monthly_contribution
                contributed += settings.sim_monthly_contribution
                n_contribs += 1

            def do_buy(t: str, budget: float, day=d) -> None:
                nonlocal cash
                p = px_of(t, day)
                if p is None or p <= 0:
                    return
                # Reserve the one-way fee inside the spend so cash can never
                # go negative (a full-cash spend plus fee would).
                notional = min(budget, max(cash, 0.0) / (1.0 + cost))
                if notional < 1:
                    return
                sh = notional / p
                cash -= notional
                cash -= notional * cost
                shares[t] = shares.get(t, 0.0) + sh
                trades.append((day.strftime("%Y-%m-%d"), "BUY", t, notional))

            def do_sell(t: str, day=d) -> None:
                nonlocal cash
                p = px_of(t, day)
                sh = shares.get(t, 0.0)
                if p is None or p <= 0 or sh <= 0:
                    return
                notional = sh * p
                cash += notional * (1.0 - cost)
                del shares[t]
                trades.append((day.strftime("%Y-%m-%d"), "SELL", t, notional))

            # SELL band releases (month-end rebuild only, as in the backtest)
            if d in months:
                for t in list(shares):
                    if t not in band:
                        do_sell(t)

            equity = cash + sum((shares.get(t, 0.0) or 0.0) * (px_of(t, d) or 0.0)
                                for t in shares)
            weight = equity / max(target_n, 1)

            # rank-deployment (boost=0): top up top-ranked names toward the
            # equal-weight target, best rank first. On month-end days the
            # full equal-weight rebuild also fills NEW names to weight.
            if order:
                for t in order[:target_n]:
                    p = px_of(t, d)
                    if p is None:
                        continue
                    cur = shares.get(t, 0.0) * p
                    gap = weight - cur
                    if gap > 1 and cash > 1:
                        do_buy(t, gap)

            # daily snapshot (one point per day). The snapshot carries the
            # contributed total AS OF that day — persisting the final total
            # for every row would make the % curve halve on early points
            # (999/2000 on day 1 instead of 999/1000).
            equity_close = cash + sum((shares.get(t, 0.0) or 0.0) * (px_of(t, d) or 0.0)
                                      for t in shares)
            snaps.append((d.strftime("%Y-%m-%d"), equity_close, contributed))

        # --- persist the end state ---
        # Allowance rows: one per month the replay actually deposited,
        # derived from the replay's own month sequence (safer than a
        # DateOffset sweep, which can drift across the trimmed start).
        replay_months = sorted({d.strftime("%Y-%m") for d in idx})
        async with Session() as s:
            acc = await _account(s)
            acc.cash = cash
            for t, sh in shares.items():
                # avg_cost is not tracked by the replay (irrelevant for the
                # equity curve); book at the last close so valuation works.
                p = px_of(t, idx[-1]) or 0.0
                s.add(Pos(ticker=t, shares=sh, avg_cost=p))
            for _td, side, t, notional in trades:
                s.add(Tr(ticker=t, side=side, shares=0.0, price=0.0,
                         cash_after=0.0, reason=f"backfill {side.lower()} ${notional:,.0f}"))
            for m in replay_months:
                s.add(Al(amount=settings.sim_monthly_contribution, month=m))
            # The current month's LIVE deposit happened before the wipe and
            # is part of the replay's own sequence only if that month had
            # trading days here — it does (idx ends today), but keep the
            # account marker consistent so deposit_allowance() stays a no-op
            # for the live month.
            acc.last_allowance_month = live_allowance_month or replay_months[-1]
            for sd, eq, contrib_at_day in snaps:
                # created_at = the replay day: the UI groups snapshots by
                # this column's date, so synthetic history must carry the
                # historical day, not the write time (otherwise all points
                # collapse into "today" and the curve shows one day).
                s.add(Sn(cash=0.0, positions_value=eq, total_equity=eq,
                         allowance_total=round(contrib_at_day, 2),
                         created_at=pd.Timestamp(f"{sd} 16:00:00+00:00").to_pydatetime()))
            await s.commit()

        return {"ok": True, "days": len(idx), "start": str(first_day.date()),
                "end": str(idx[-1].date()), "requested_start": start_note,
                "contributed": round(contributed, 2),
                "final_equity": round(snaps[-1][1], 2), "trades": len(trades),
                "snapshots": len(snaps)}


async def run_daily_cycle(force: bool = False) -> dict:
    """Scheduler entry: one full daily-core cycle.

    deposit allowance (new month) -> refresh data -> deployment pass ->
    daily equity snapshot. Idempotent per day: the allowance gate is
    monthly and the deployment pass is idempotent by construction
    (top-ups stop when targets are met), so a re-run is harmless.
    """
    if _cycle_lock.locked() and not force:
        return {"skipped": True, "reason": "already running"}
    async with _cycle_lock:
        if not settings.sim_daily_core_enabled:
            return {"skipped": True, "reason": "daily-core portfolio disabled"}

        allowance = await deposit_allowance()

        refreshed, refresh_errors = await refresh_data()

        deployment = await run_deployment()

        snap = await take_snapshot()

        return {"allowance": allowance, "refresh_errors": refresh_errors,
                "deployment": deployment, "snapshot": snap}


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------

async def get_trades(limit: int = 100) -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(DailyCoreTrade)
                                .order_by(DailyCoreTrade.created_at.desc()).limit(limit))).all()
    return [{"ticker": r.ticker, "side": r.side, "shares": r.shares, "price": r.price,
             "cash_after": round(r.cash_after, 2), "reason": r.reason,
             "date": r.created_at.isoformat()} for r in rows]


async def get_equity_curve(limit: int = 365) -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(DailyCoreSnapshot)
                                .order_by(DailyCoreSnapshot.created_at.desc()).limit(limit))).all()
    return [{"date": r.created_at.strftime("%Y-%m-%d"), "cash": r.cash,
             "positions_value": r.positions_value, "total_equity": r.total_equity,
             "allowance_total": r.allowance_total} for r in reversed(rows)]


async def get_allowances() -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(DailyCoreAllowance)
                                .order_by(DailyCoreAllowance.month.desc()))).all()
    return [{"amount": r.amount, "month": r.month} for r in rows]
