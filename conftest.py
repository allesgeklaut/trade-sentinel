"""Pytest configuration: make the project root importable so ``from app...`` works."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _isolate_generated_universes(tmp_path, monkeypatch):
    """Keep the real generated-universe dir (``/data/universes``) out of every
    test. Once a sync has run on the host/container, tests that call
    ``screener.universe_names()`` (e.g. the daily-core suite) would otherwise
    see the real generated files."""
    from app import screener
    monkeypatch.setattr(screener, "_EXTRA_UNIVERSES_DIR", tmp_path / "_no_extra_universes")
