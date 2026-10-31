"""Pytest configuration: make the project root importable so ``from app...`` works."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The PROD/DEV database switch must stay OFF in tests regardless of any live
# .env: pydantic-settings loads .env from the CWD, so a host-side pytest run
# would otherwise build a real DEV engine (e.g. pointing at the container-only
# /data volume) that init_db() would try to connect/migrate. A real env var
# takes precedence over .env in pydantic-settings, so pinning it here wins both
# ways. Tests that need a DEV database inject their own engines.
os.environ["DATABASE_URL_DEV"] = ""


@pytest.fixture(autouse=True)
def _isolate_generated_universes(tmp_path, monkeypatch):
    """Keep the real generated-universe dir (``/data/universes``) out of every
    test. Once a sync has run on the host/container, tests that call
    ``screener.universe_names()`` (e.g. the daily-core suite) would otherwise
    see the real generated files."""
    from app import screener
    monkeypatch.setattr(screener, "_EXTRA_UNIVERSES_DIR", tmp_path / "_no_extra_universes")
