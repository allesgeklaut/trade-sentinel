"""Tests for app.main's multi-portfolio backfill coordinator (no DB/network).

Regression guard for the "Backfill All 4" full-history button: the coordinator
must pass the literal "all" through to every backfill, because None means
"synced window" to each of them (not full history). Passing None made the
full-history button silently replay the short synced range.
"""

from __future__ import annotations

import pytest

from app import daily_core, main, monthly, sim


@pytest.fixture
def recorded(monkeypatch):
    """Replace the four backfills with recorders; capture the start they got."""
    calls: dict[str, list] = {}

    async def rec_backfill(start=None):
        calls.setdefault("sim", []).append(start)
        return {"ok": True, "final_equity": 1.0}

    async def rec_bench(start=None):
        calls.setdefault("bench", []).append(start)
        return {"ok": True, "final_equity": 1.0}

    async def rec_monthly_preload(start=None):
        calls.setdefault("monthly_preload", []).append(start)
        return object()

    async def rec_monthly(start=None, *, preload=None):
        calls.setdefault("monthly", []).append(start)
        return {"ok": True, "final_equity": 1.0}

    async def rec_daily_core(start=None, *, preload=None):
        calls.setdefault("daily_core", []).append(start)
        return {"ok": True, "final_equity": 1.0}

    monkeypatch.setattr(sim, "backfill", rec_backfill)
    monkeypatch.setattr(sim, "backfill_benchmark", rec_bench)
    monkeypatch.setattr(monthly, "preload_backfill", rec_monthly_preload)
    monkeypatch.setattr(monthly, "backfill", rec_monthly)
    monkeypatch.setattr(daily_core, "backfill", rec_daily_core)
    return calls


class TestBackfillAllWindow:
    async def test_all_passes_full_history_to_every_backfill(self, recorded):
        r = await main.backfill_all(start="all")
        assert r["ok"] is True
        assert r["requested_start"] == "all"
        # Every replay must see the "all" sentinel (full history), NOT None
        # (which each maps to the short synced window).
        assert recorded["sim"] == ["all"]
        assert recorded["bench"] == ["all"]
        assert recorded["monthly"] == ["all"]
        assert recorded["daily_core"] == ["all"]
        # The shared preload takes a date or None (= load everything), never
        # the "all" sentinel (pd.Timestamp("all") would raise).
        assert recorded["monthly_preload"] == [None]

    async def test_explicit_date_pins_every_backfill(self, recorded):
        await main.backfill_all(start="2026-01-02")
        assert recorded["sim"] == ["2026-01-02"]
        assert recorded["bench"] == ["2026-01-02"]
        assert recorded["monthly"] == ["2026-01-02"]
        assert recorded["daily_core"] == ["2026-01-02"]
        assert recorded["monthly_preload"] == ["2026-01-02"]

    async def test_default_uses_shared_synced_window(self, recorded, monkeypatch):
        async def fake_sync():
            return "2026-03-04"

        monkeypatch.setattr(daily_core, "_sync_start_date", fake_sync)
        r = await main.backfill_all(start=None)
        assert recorded["sim"] == ["2026-03-04"]
        assert recorded["bench"] == ["2026-03-04"]
        assert recorded["monthly"] == ["2026-03-04"]
        assert recorded["daily_core"] == ["2026-03-04"]
        assert r["requested_start"] == "synced (earliest of the other sims)"
