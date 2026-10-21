"""Safety regressions for the shared strategy engine (``app/strategy.py``).

Phase 1 financial correctness — these tests pin the behaviour the UI/strategy
audit (docs/ui-strategy-audit-2026-10-21.md, P0-C) found broken:

  1. The relaxed HOLD fallback used to buy **bearish** HOLDs (its strength is
     ``max(bullish, bearish)``) and it bypassed the daily/weekly trend checks
     that had blocked the strict BUY. It now requires a positive net score and
     the *same* entry gate as a strict BUY.
  2. The ATR stop used to be ``close - 2 * ATR`` on the same bar that was
     then compared against that close — unreachable, so it never fired. It is
     now a prior-known, reachable level (see ``tests/test_analysis.py`` for
     the indicator-level tests).
  3. The sector cap was only pre-checked: a BUY could still be sized past the
     cap. The budget is now clamped to the remaining sector room.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from app.strategy import StrategyParams, propose_trades

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sig(
    action: str = "HOLD",
    strength: int = 60,
    net: int | None = 25,
    trend_up: bool = True,
    trend_down: bool = False,
    dist_above: float | None = 5.0,
    weekly: bool = True,
    run_5d: float | None = 0.0,
    atr_stop: float | None = None,
    close: float = 100.0,
    atr14: float | None = 2.0,
) -> dict:
    """A signal dict in the shape ``analysis.compute()`` produces."""
    snap = {
        "close": close,
        "atr14": atr14,
        "atr_stop": atr_stop,
        "run_5d": run_5d,
        "dist_above": dist_above,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "weekly_trend_up": weekly,
    }
    if net is not None:
        snap["net_score"] = net
    return {
        "action": action,
        "strength": strength,
        "reason": f"{action} (strength {strength})",
        "snapshot": snap,
    }


def _params(**kw) -> StrategyParams:
    base = dict(
        buy_threshold=40,
        sell_threshold=-40,
        min_cash_pct=0.0,
        max_position_pct=100.0,
        max_positions=10,
        stop_type="none",
        use_atr_stop=False,
        max_run_5d=0.0,
        min_run_5d=0.0,
        max_dist_above=0.0,
        relaxed_hold_strength=40,
        relaxed_hold_limit=3,
    )
    base.update(kw)
    return StrategyParams(**base)


def _buys(out: list[dict]) -> list[dict]:
    return [p for p in out if p["side"] == "BUY"]


# ---------------------------------------------------------------------------
# 1. The relaxed HOLD fallback
# ---------------------------------------------------------------------------

class TestFallbackEntryGate:
    """A HOLD may only be entered when it is a genuine near-threshold BUY."""

    def test_bearish_hold_cannot_enter(self):
        """The audit reproduction: bullish 0 / bearish 65 (strength 65) with a
        weekly downtrend must not produce a BUY."""
        sig = _sig(action="HOLD", strength=65, net=-65, trend_up=False,
                   trend_down=True, dist_above=-8.0, weekly=False)
        out = propose_trades([], 10_000.0, {"BEAR": 100.0}, {"BEAR": sig},
                             _params())
        assert out == [], (
            "a bearish HOLD entered through the relaxed fallback: "
            f"{[(p['ticker'], p['side']) for p in out]}"
        )

    def test_bearish_hold_with_strong_score_cannot_enter(self):
        """A HOLD whose strength comes from the *bearish* score never enters."""
        sig = _sig(action="HOLD", strength=80, net=-80, trend_up=False,
                   trend_down=True, dist_above=-10.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_positive_net_but_no_daily_uptrend_cannot_enter(self):
        sig = _sig(action="HOLD", strength=60, net=25, trend_up=False,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_positive_net_but_weekly_down_cannot_enter(self):
        sig = _sig(action="HOLD", strength=60, net=25, trend_up=True,
                   dist_above=5.0, weekly=False)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_positive_net_but_too_close_to_sma200_cannot_enter(self):
        """Same gate as a strict BUY: more than 2% above the SMA-200."""
        sig = _sig(action="HOLD", strength=60, net=25, trend_up=True,
                   dist_above=1.5, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_near_threshold_hold_enters_when_the_gate_passes(self):
        """A HOLD that only misses the score threshold still enters — that is
        the point of the fallback."""
        sig = _sig(action="HOLD", strength=45, net=25, trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert len(_buys(out)) == 1
        assert out[0]["ticker"] == "X"

    def test_fallback_reason_keeps_the_near_threshold_rationale(self):
        sig = _sig(action="HOLD", strength=45, net=25, trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        reason = _buys(out)[0]["reason"]
        assert "relaxed HOLD entry" in reason
        assert "+25" in reason          # the net score
        assert "weekly trend up" in reason

    def test_missing_snapshot_fields_fail_closed(self):
        """A snapshot without a net score / trend flags cannot be verified →
        the fallback must not fire."""
        bare = {"action": "HOLD", "strength": 60, "reason": "hold",
                "snapshot": {"close": 100.0, "run_5d": 0.0}}
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": bare}, _params())
        assert out == []

    def test_unknown_net_score_fails_closed(self):
        sig = _sig(action="HOLD", strength=60, net=None, trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_nan_net_score_fails_closed(self):
        """A NaN net score is not "positive" — the comparison ``nan <= 0`` is
        False, so it must be rejected explicitly."""
        sig = _sig(action="HOLD", strength=60, net=float("nan"), trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_infinite_net_score_fails_closed(self):
        sig = _sig(action="HOLD", strength=60, net=float("inf"), trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_missing_weekly_flag_fails_closed(self):
        """No weekly flag is NOT "no weekly data → no gate": the weekly
        filter is a hard confirmation, so an absent flag must block."""
        sig = _sig(action="HOLD", strength=60, net=20, trend_up=True,
                   dist_above=3.0, weekly=True)
        del sig["snapshot"]["weekly_trend_up"]
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_nan_distance_fails_closed(self):
        sig = _sig(action="HOLD", strength=60, net=20, trend_up=True,
                   dist_above=float("nan"), weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_infinite_distance_fails_closed(self):
        sig = _sig(action="HOLD", strength=60, net=20, trend_up=True,
                   dist_above=float("inf"), weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert out == []

    def test_strict_buy_is_unaffected(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig}, _params())
        assert len(_buys(out)) == 1
        assert _buys(out)[0]["reason"] == sig["reason"]

    def test_strict_buy_wins_over_the_fallback(self):
        """When a strict BUY exists, the fallback is not used at all."""
        buy = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        hold = _sig(action="HOLD", strength=70, net=30, trend_up=True,
                    dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"B": 100.0, "H": 100.0},
                             {"B": buy, "H": hold}, _params())
        assert [p["ticker"] for p in _buys(out)] == ["B"]

    def test_fallback_respects_the_limit(self):
        sigs = {f"T{i}": _sig(action="HOLD", strength=50 + i, net=25,
                              trend_up=True, dist_above=5.0, weekly=True)
                for i in range(5)}
        out = propose_trades([], 100_000.0,
                             {t: 100.0 for t in sigs}, sigs,
                             _params(relaxed_hold_limit=2, max_position_pct=10.0))
        # Only the two strongest HOLDs are taken (10% position cap each).
        assert len(_buys(out)) == 2
        assert [p["ticker"] for p in _buys(out)] == ["T4", "T3"]

    def test_fallback_respects_the_strength_threshold(self):
        sig = _sig(action="HOLD", strength=30, net=25, trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig},
                             _params(relaxed_hold_strength=40))
        assert out == []

    def test_fallback_still_obeys_the_entry_guards(self):
        """The fallback routes through the same entry guards (run-up, crash,
        parabolic extension, ROE) as a strict BUY."""
        spike = _sig(action="HOLD", strength=60, net=25, trend_up=True,
                     dist_above=5.0, weekly=True, run_5d=30.0)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": spike},
                             _params(max_run_5d=12.0))
        assert out == []

        crash = _sig(action="HOLD", strength=60, net=25, trend_up=True,
                     dist_above=5.0, weekly=True, run_5d=-30.0)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": crash},
                             _params(min_run_5d=-15.0))
        assert out == []

        parabolic = _sig(action="HOLD", strength=60, net=25, trend_up=True,
                         dist_above=95.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": parabolic},
                             _params(max_dist_above=80.0))
        assert out == []

    def test_fallback_respects_the_roe_guard(self):
        sig = _sig(action="HOLD", strength=60, net=25, trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades(
            [], 10_000.0, {"X": 100.0}, {"X": sig},
            _params(block_negative_roe=True),
            quality_of=lambda t: {"roe": -0.2},
        )
        assert out == []


# ---------------------------------------------------------------------------
# 2. The ATR stop
# ---------------------------------------------------------------------------

class TestAtrStop:
    """The ATR stop must be a level today's close can actually break."""

    def test_stop_triggers_a_sell(self):
        sig = _sig(action="HOLD", strength=10, net=5, atr_stop=95.0, close=90.0)
        positions = [{"ticker": "X", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 0.0, {"X": 90.0}, {"X": sig},
                             _params(use_atr_stop=True))
        assert [p["side"] for p in out] == ["SELL"]
        assert out[0]["reason"].startswith("ATR stop hit")

    def test_stop_triggers_on_an_exact_touch(self):
        sig = _sig(action="HOLD", strength=10, net=5, atr_stop=90.0, close=90.0)
        positions = [{"ticker": "X", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 0.0, {"X": 90.0}, {"X": sig},
                             _params(use_atr_stop=True))
        assert [p["side"] for p in out] == ["SELL"]

    def test_no_sell_above_the_stop(self):
        sig = _sig(action="HOLD", strength=10, net=5, atr_stop=80.0, close=90.0)
        positions = [{"ticker": "X", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 0.0, {"X": 90.0}, {"X": sig},
                             _params(use_atr_stop=True))
        assert out == []

    def test_stop_can_be_switched_off(self):
        sig = _sig(action="HOLD", strength=10, net=5, atr_stop=95.0, close=90.0)
        positions = [{"ticker": "X", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 0.0, {"X": 90.0}, {"X": sig},
                             _params(use_atr_stop=False))
        assert out == []

    def test_percent_stop_is_independent_of_the_atr_level(self):
        """The frozen percent stop still fires on its own."""
        sig = _sig(action="HOLD", strength=10, net=5, atr_stop=1.0, close=89.0)
        positions = [{"ticker": "X", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 0.0, {"X": 89.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=10.0,
                                     use_atr_stop=False))
        assert [p["side"] for p in out] == ["SELL"]
        assert out[0]["reason"].startswith("Initial stop")


# ---------------------------------------------------------------------------
# 2b. The ATR level is market-anchored: no entry at/below it, no churn
# ---------------------------------------------------------------------------

class TestAtrEntryGuard:
    """The ATR reference level is anchored to the market's 20-bar closing high,
    not to the position's entry. A pullback entry can therefore publish a
    level ABOVE the entry price — the position would be stopped out on the
    next cycle (buy -> sell -> buy churn at the same price)."""

    # The reviewer's reproduction: close 128.46, published level 128.97.
    PRICE = 128.46
    LEVEL = 128.97

    def _buy_signal(self) -> dict:
        return _sig(action="BUY", strength=80, net=60, trend_up=True,
                    dist_above=8.0, weekly=True, atr_stop=self.LEVEL,
                    close=self.PRICE)

    def test_no_buy_at_or_below_the_published_level(self):
        out = propose_trades([], 10_000.0, {"X": self.PRICE},
                             {"X": self._buy_signal()},
                             _params(use_atr_stop=True))
        assert _buys(out) == [], (
            "entered below the published ATR level — the position would be "
            "stopped out on the next cycle (churn at the same price)"
        )

    def test_no_topup_at_or_below_the_published_level(self):
        positions = [{"ticker": "X", "shares": 1, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 10_000.0, {"X": self.PRICE},
                             {"X": self._buy_signal()},
                             _params(use_atr_stop=True))
        assert _buys(out) == []

    def test_buy_above_the_published_level_is_allowed(self):
        out = propose_trades([], 10_000.0, {"X": 130.0},
                             {"X": self._buy_signal()},
                             _params(use_atr_stop=True))
        assert len(_buys(out)) == 1

    def test_guard_is_off_when_the_atr_stop_is_disabled(self):
        """Without an armed ATR stop there is nothing to be stopped out by,
        so the entry must not be filtered."""
        out = propose_trades([], 10_000.0, {"X": self.PRICE},
                             {"X": self._buy_signal()},
                             _params(use_atr_stop=False))
        assert len(_buys(out)) == 1

    def test_non_finite_or_non_positive_levels_do_not_block(self):
        for level in (float("nan"), float("inf"), 0.0, -5.0, None):
            sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                       dist_above=8.0, weekly=True, atr_stop=level,
                       close=self.PRICE)
            out = propose_trades([], 10_000.0, {"X": self.PRICE}, {"X": sig},
                                 _params(use_atr_stop=True))
            assert len(_buys(out)) == 1, f"level {level!r} must not block a BUY"

    def test_two_cycle_loop_does_not_churn(self):
        """Cycle 1: no entry below the level. Cycle 2: the position cannot be
        sold and re-bought at the same price."""
        sig = self._buy_signal()
        params = _params(use_atr_stop=True)

        cycle1 = propose_trades([], 10_000.0, {"X": self.PRICE}, {"X": sig},
                                params)
        assert _buys(cycle1) == []

        # If the position is somehow held and the price is below the level,
        # it is stopped out — and the same cycle must not buy it back.
        positions = [{"ticker": "X", "shares": 10, "avg_cost": 128.46,
                      "stop_price": None}]
        cycle2 = propose_trades(positions, 10_000.0, {"X": self.PRICE},
                                {"X": sig}, params)
        sides = [(p["ticker"], p["side"]) for p in cycle2]
        assert ("X", "SELL") in sides
        assert ("X", "BUY") not in sides, (
            f"the ticker was sold and bought again in the same cycle: {sides}"
        )


class TestNoSellThenBuySameCycle:
    """A ticker sold in a cycle must not be bought again in the same cycle."""

    def _held(self, ticker="X", shares=10, avg_cost=100.0):
        return [{"ticker": ticker, "shares": shares, "avg_cost": avg_cost,
                 "stop_price": None}]

    def test_signal_sell_blocks_a_same_cycle_buy(self):
        sell_sig = _sig(action="SELL", strength=80, net=-60, trend_up=False,
                        trend_down=True, dist_above=-5.0, weekly=True)
        out = propose_trades(self._held(), 10_000.0, {"X": 100.0},
                             {"X": sell_sig}, _params())
        sides = [p["side"] for p in out]
        assert sides == ["SELL"], sides

    def test_initial_stop_blocks_a_same_cycle_buy(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=8.0, weekly=True)
        out = propose_trades(self._held(), 10_000.0, {"X": 80.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=10.0))
        sides = [p["side"] for p in out]
        assert sides == ["SELL"], sides

    def test_atr_stop_blocks_a_same_cycle_buy(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=8.0, weekly=True, atr_stop=90.0, close=85.0)
        out = propose_trades(self._held(), 10_000.0, {"X": 85.0}, {"X": sig},
                             _params(use_atr_stop=True))
        sides = [p["side"] for p in out]
        assert sides == ["SELL"], sides

    def test_other_tickers_are_unaffected(self):
        sell_sig = _sig(action="SELL", strength=80, net=-60, trend_up=False,
                        trend_down=True, dist_above=-5.0, weekly=True)
        buy_sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                       dist_above=8.0, weekly=True)
        out = propose_trades(self._held(), 10_000.0,
                             {"X": 100.0, "Y": 100.0},
                             {"X": sell_sig, "Y": buy_sig}, _params())
        sides = sorted((p["ticker"], p["side"]) for p in out)
        assert sides == [("X", "SELL"), ("Y", "BUY")]


# ---------------------------------------------------------------------------
# 3. The sector cap
# ---------------------------------------------------------------------------

class TestSectorCap:
    """The sector cap was only pre-checked — the budget could still overshoot
    it. The budget is now clamped to the remaining sector room."""

    def _sector_of(self, ticker: str) -> str:
        return ticker[0]  # "A*" -> sector A

    def test_budget_is_clamped_to_the_remaining_sector_room(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        params = _params(max_position_pct=100.0, max_sector_pct=10.0)
        # Equity 10 000 → sector cap 1 000.
        out = propose_trades([], 10_000.0, {"A1": 100.0}, {"A1": sig},
                             params, sector_of=self._sector_of)
        assert len(_buys(out)) == 1
        assert _buys(out)[0]["budget"] == 1000.0

    def test_second_buy_in_the_same_sector_sees_the_first(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        params = _params(max_position_pct=100.0, max_sector_pct=15.0)
        out = propose_trades([], 10_000.0, {"A1": 100.0, "A2": 100.0},
                             {"A1": sig, "A2": sig}, params,
                             sector_of=self._sector_of)
        budgets = [p["budget"] for p in _buys(out)]
        assert sum(budgets) == 1500.0
        assert budgets == [1500.0, 0.0] or budgets[0] == 1500.0

    def test_sector_exposure_never_exceeds_the_cap(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        params = _params(max_position_pct=100.0, max_sector_pct=20.0)
        prices = {f"A{i}": 100.0 for i in range(4)}
        out = propose_trades([], 10_000.0, prices, {t: sig for t in prices},
                             params, sector_of=self._sector_of)
        spent = sum(p["budget"] for p in _buys(out))
        assert spent <= 2000.0 + 1e-9, (
            f"sector exposure {spent} exceeded the 20% cap (2 000)"
        )

    def test_a_full_sector_blocks_new_buys(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        params = _params(max_position_pct=100.0, max_sector_pct=10.0)
        # Sector A already holds 1 000; equity is 11 000 → cap 1 100.
        positions = [{"ticker": "A1", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 10_000.0, {"A1": 100.0, "A2": 100.0},
                             {"A1": sig, "A2": sig}, params,
                             sector_of=self._sector_of)
        buys = _buys(out)
        # The remaining 100 of sector room can be used, but never exceeded —
        # the second ticker in the sector gets nothing.
        assert sum(p["budget"] for p in buys) <= 100.0 + 1e-9
        assert [p["ticker"] for p in buys] == ["A1"]

    def test_other_sectors_are_unaffected(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        params = _params(max_position_pct=100.0, max_sector_pct=10.0)
        positions = [{"ticker": "A1", "shares": 10, "avg_cost": 100.0,
                      "stop_price": None}]
        out = propose_trades(positions, 10_000.0, {"A1": 100.0, "B1": 100.0},
                             {"A1": sig, "B1": sig}, params,
                             sector_of=self._sector_of)
        buys = {p["ticker"]: p["budget"] for p in _buys(out)}
        # Sector B has its own full cap (10% of 11 000 = 1 100).
        assert buys["B1"] == 1100.0
        # Sector A only had 100 of room left.
        assert buys.get("A1", 0.0) <= 100.0 + 1e-9

    def test_no_sector_cap_means_no_clamp(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        params = _params(max_position_pct=100.0, max_sector_pct=0.0)
        out = propose_trades([], 10_000.0, {"A1": 100.0}, {"A1": sig},
                             params, sector_of=self._sector_of)
        assert _buys(out)[0]["budget"] == 10_000.0


# ---------------------------------------------------------------------------
# 4. StrategyParams hygiene
# ---------------------------------------------------------------------------

class TestStrategyParams:
    def test_relaxed_fields_are_declared_once(self):
        names = [f.name for f in fields(StrategyParams)]
        assert names.count("relaxed_hold_strength") == 1
        assert names.count("relaxed_hold_limit") == 1
        assert len(names) == len(set(names)), f"duplicate fields: {names}"

    def test_relaxed_defaults(self):
        p = StrategyParams()
        assert p.relaxed_hold_strength == 40
        assert p.relaxed_hold_limit == 3


# ---------------------------------------------------------------------------
# 5. Session fill contract: fills at the next session's open
# ---------------------------------------------------------------------------
#
# The audit's "executable, fresh next-session fills" item: a decision taken
# on a completed bar may not fill at that same close. When the caller
# supplies ``session_bars`` (the NEXT session's strict exact-date bar, see
# ``app.session``), SELLs/BUYs fill at F's open, stop levels known at the
# previous close act as standing orders (open-gap → fill at open, intrabar
# breach → fill at the level), and a ticker without an F bar produces no
# proposal at all.

def _held(ticker: str, shares: float = 10.0, avg_cost: float = 100.0,
          stop_price: float | None = None,
          atr_stop_level: float | None = None) -> dict:
    return {"ticker": ticker, "shares": shares, "avg_cost": avg_cost,
            "stop_price": stop_price, "atr_stop_level": atr_stop_level}


class TestSessionFillPrices:
    """Contract fills: price fields carry the EXECUTION price (F's open)."""

    def test_buy_fills_at_open_not_close(self):
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        bars = {"X": {"open": 98.0, "low": 97.0, "close": 99.0}}
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=15.0),
                             session_bars=bars)
        buys = _buys(out)
        assert len(buys) == 1
        assert buys[0]["price"] == 98.0, "the BUY must fill at F's open"
        assert buys[0]["price"] != 100.0, "same-close fill is not executable"
        # The entry stop anchors to the price actually paid (the open).
        assert buys[0]["entry_stop"] == pytest.approx(98.0 * 0.85)

    def test_sell_fills_at_open_not_close(self):
        sig = _sig(action="SELL", strength=70, net=-60, trend_up=False,
                   trend_down=True, dist_above=-8.0)
        bars = {"X": {"open": 102.0, "low": 101.0, "close": 100.0}}
        out = propose_trades([_held("X")], 1_000.0, {"X": 100.0},
                             {"X": sig}, _params(), session_bars=bars)
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 102.0, "the SELL must fill at F's open"

    def test_missing_f_bar_means_no_fill(self):
        """No F bar → no proposal at all, even for a hard stop breach at the
        decision close: the decision is re-taken next cycle, nothing stale
        fills."""
        # Massive stop breach known at S's close (close 50 <= stop 90).
        pos = _held("X", stop_price=90.0)
        sig = _sig(action="HOLD", strength=60, net=25, close=50.0)
        out = propose_trades([pos], 1_000.0, {"X": 50.0}, {"X": sig},
                             _params(use_atr_stop=True),
                             session_bars={})  # F has no bar for X
        assert out == []
        # Same for a fresh BUY signal without an F bar.
        buy = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": buy},
                             _params(), session_bars={})
        assert out == []

    def test_missing_open_is_not_executable(self):
        """A bar without an open is not an executable price."""
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        bars = {"X": {"open": None, "low": 90.0, "close": 99.0}}
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig},
                             _params(), session_bars=bars)
        assert out == []

    def test_stop_gap_down_fills_at_open(self):
        """Open <= level: the standing order fills at the OPEN (market order
        at the open after the gap), not at the level, not at the close."""
        pos = _held("X", stop_price=90.0)
        # strength 30 keeps the relaxed top-up fallback out of the way: this
        # test isolates the standing stop, not the near-BUY entry logic.
        sig = _sig(action="HOLD", strength=30, net=25, close=95.0)
        bars = {"X": {"open": 85.0, "low": 84.0, "close": 86.0}}
        out = propose_trades([pos], 1_000.0, {"X": 95.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=15.0,
                                     use_atr_stop=True),
                             session_bars=bars)
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 85.0
        assert sells[0]["reason"].lower().startswith("initial stop"), \
            "stop-out reasons keep the stop-out marker"
        assert "filled at the open" in sells[0]["reason"]

    def test_stop_intrabar_breach_fills_at_level(self):
        """Open above the level but low <= level: the order fills AT the
        level (the price the resting order attains), never at the close."""
        pos = _held("X", stop_price=90.0)
        sig = _sig(action="HOLD", strength=30, net=25, close=95.0)
        bars = {"X": {"open": 92.0, "low": 89.0, "close": 94.0}}
        out = propose_trades([pos], 1_000.0, {"X": 95.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=15.0,
                                     use_atr_stop=True),
                             session_bars=bars)
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 90.0, "intrabar breach fills at the level"
        assert "filled at the level" in sells[0]["reason"]

    def test_stop_untriggered_means_no_fill(self):
        pos = _held("X", stop_price=90.0)
        sig = _sig(action="HOLD", strength=30, net=25, close=95.0)
        bars = {"X": {"open": 95.0, "low": 94.0, "close": 96.0}}
        out = propose_trades([pos], 1_000.0, {"X": 95.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=15.0,
                                     use_atr_stop=True),
                             session_bars=bars)
        assert out == []

    def test_tightest_stop_level_governs(self):
        """Percent stop + monotone ratchet: the highest (tightest) level is
        the standing order."""
        pos = _held("X", stop_price=85.0, atr_stop_level=92.0)
        sig = _sig(action="HOLD", strength=60, net=25, close=95.0)
        # Open above 92 but low breaches it → fill AT the ratchet level.
        bars = {"X": {"open": 93.0, "low": 91.0, "close": 94.0}}
        out = propose_trades([pos], 1_000.0, {"X": 95.0}, {"X": sig},
                             _params(use_atr_stop=True), session_bars=bars)
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 92.0
        assert sells[0]["reason"].lower().startswith("atr stop")

    def test_legacy_percent_stop_fallback_uses_avg_cost(self):
        """A position without a persisted stop (legacy rows) recomputes the
        level from the blended average cost — documented fallback."""
        pos = _held("X", avg_cost=100.0)  # stop_price=None
        sig = _sig(action="HOLD", strength=60, net=25, close=84.0)
        bars = {"X": {"open": 84.0, "low": 83.0, "close": 85.0}}
        out = propose_trades([pos], 1_000.0, {"X": 84.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=15.0,
                                     use_atr_stop=True),
                             session_bars=bars)
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 84.0  # open <= 85.0 → filled at the open
        assert sells[0]["reason"].lower().startswith("initial stop")

    def test_mutating_f_bar_does_not_change_the_decision(self):
        """The decision is frozen to data <= S: mutating F's close/low (away
        from any stop trigger) must produce identical proposals."""
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True)
        cash, prices, signals, params = (
            10_000.0, {"X": 100.0}, {"X": sig},
            _params(stop_type="percent", stop_pct=15.0))
        bars_a = {"X": {"open": 99.0, "low": 98.0, "close": 99.0}}
        bars_b = {"X": {"open": 99.0, "low": 95.0, "close": 110.0}}
        out_a = propose_trades([], cash, prices, signals, params,
                               session_bars=bars_a)
        out_b = propose_trades([], cash, prices, signals, params,
                               session_bars=bars_b)
        assert out_a == out_b, "F's bar must not inform the decision"

    def test_entry_gate_judges_the_price_actually_paid(self):
        """No entry at or below the published ATR reference level — judged
        against F's open (the fill), not the decision close."""
        sig = _sig(action="BUY", strength=80, net=60, trend_up=True,
                   dist_above=5.0, weekly=True, atr_stop=99.0)
        # Decision close 100 > 99 (gate would pass at the close), but the
        # open 98 <= 99 → no fill (entering would churn the stop out).
        bars = {"X": {"open": 98.0, "low": 97.0, "close": 101.0}}
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig},
                             _params(use_atr_stop=True), session_bars=bars)
        assert out == []
        # Open above the level → the entry fills at the open.
        bars = {"X": {"open": 99.5, "low": 99.0, "close": 101.0}}
        out = propose_trades([], 10_000.0, {"X": 100.0}, {"X": sig},
                             _params(use_atr_stop=True), session_bars=bars)
        buys = _buys(out)
        assert len(buys) == 1 and buys[0]["price"] == 99.5

    def test_legacy_path_fills_at_decision_close(self):
        """session_bars=None keeps the legacy convention for unwired
        callers: fills and stop comparisons at the decision close."""
        pos = _held("X", stop_price=90.0)
        sig = _sig(action="HOLD", strength=60, net=25, close=89.0)
        out = propose_trades([pos], 1_000.0, {"X": 89.0}, {"X": sig},
                             _params(stop_type="percent", stop_pct=15.0,
                                     use_atr_stop=True))
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 89.0
        assert sells[0]["reason"] == "Initial stop: 89.00 <= 90.00"
