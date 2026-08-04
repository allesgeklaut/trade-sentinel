"""Pytest configuration: make the project root importable so ``from app...`` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))