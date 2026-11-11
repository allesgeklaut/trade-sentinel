"""Pytest configuration for the tests/ package.

The repo-root conftest.py makes the project importable and pins
DATABASE_URL_DEV via the process env. This file adds the SETTINGS
hermeticity fixture: pydantic-settings reads ``.env`` from the CWD, so a
host-side pytest run otherwise inherits the operator's ambient deployment
values from the gitignored .env on this machine (SIM_DAILY_CORE_MOM_VARIANT=
residual, LLM_BACKENDS=[...], LLM_REASONING_EFFORT=low, SEARXNG_URL=...,
SEARXNG_MCP_URL=..., TWELVE_DATA_API_KEY=...). Those leaks DO change test
outcomes — most visibly the mom variant: the monthly/daily-core engines
resolve it through ``settings`` (``daily_core.current_variant()`` falls back
to the settings value when no state file exists, and /data does not exist
under pytest), so the ambient ``residual`` turned the qv-mom frame all-NaN on
flat fixtures ("no eligible frame") and made nine tests red until they were
re-run with the code default ``raw``.
"""

from __future__ import annotations

import pytest

# Settings fields whose AMBIENT .env value differs from the code default on
# this host AND whose leak can alter a test outcome. Each is reset to its
# CODE default (read from pydantic's model_fields, not hardcoded) at every
# test's setup. A test's own monkeypatch.setattr(settings, ...) runs AFTER
# this autouse fixture and therefore always wins — the fixture only
# neutralizes ambient leaks, it never fights per-test overrides.
#
# (DATABASE_URL_DEV also differs ambiently but is already pinned to "" by the
# root conftest.py via the process env, which beats .env in pydantic-settings,
# so it is deliberately not repeated here.)
_SETTINGS_RESET_FIELDS = (
    # Ambient .env: "residual" (code default "raw"). VERIFIED leak: the
    # monthly + daily-core qv-mom frame math reads this field (via
    # daily_core.current_variant()'s settings fallback), and residual
    # momentum on flat fixtures is all-NaN -> no eligible frame -> 9 tests
    # red (test_monthly, TestDeferredFillCadence, test_run_config_adoption,
    # test_selllayer TestTypeBRebalanceIntegration, ...).
    "sim_daily_core_mom_variant",
    # Ambient .env: two LAN backends (code default ""). llm.current_backend()
    # flips from a real backend to None, which changes the sell layer's
    # no-backend short-circuit, consult records and chat paths for any test
    # that does not configure its own backends (tests that need one set it
    # themselves — e.g. tests/test_sim.py's llm_backend fixture).
    "llm_backends",
    # Ambient .env: "low" (code default ""). It rides verbatim into the LLM
    # wire payload on thinking-budget backends, so payload-shape assertions
    # would otherwise depend on the host's value.
    "llm_reasoning_effort",
    # Ambient .env: "http://searxng:8080" (code default ""). Empty disables
    # the news search; the ambient URL only resolves on the docker network,
    # so an unpatched call would depend on the host network.
    "searxng_url",
    # Ambient .env: "http://searxng-mcp:8808/mcp" (code default ""). Same
    # leak class for the type-D news agent's MCP enable check
    # (app/newsagent.py); tests set their own value when they exercise it.
    "searxng_mcp_url",
    # Ambient .env: a real secret key (code default ""). Under
    # MARKET_DATA_PROVIDER=auto a set key silently selects twelvedata in
    # market.provider(); tests that exercise the provider selection set
    # their own key (tests/test_market.py).
    "twelve_data_api_key",
)


@pytest.fixture(autouse=True)
def _settings_code_defaults(monkeypatch):
    """Reset the decision-input Settings fields to their CODE DEFAULTS.

    Reads each field's default from ``Settings.model_fields`` (the single
    source of truth for "code default") and monkeypatches the live
    ``settings`` singleton — only when the ambient value actually differs,
    so the fixture's intent stays explicit: it neutralizes .env LEAKS into
    pytest, not values a deployment deliberately shares with the code
    default. Because monkeypatch unwinds in reverse order, a test's own
    ``monkeypatch.setattr(settings, ...)`` applied after this autouse
    fixture wins for the duration of that test (pinned by
    tests/test_sim_guards.py::TestSettingsHermeticity).
    """
    from app.config import Settings, settings

    for name in _SETTINGS_RESET_FIELDS:
        default = Settings.model_fields[name].default
        if getattr(settings, name) != default:
            monkeypatch.setattr(settings, name, default)
