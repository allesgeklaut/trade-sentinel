#!/usr/bin/env bash
# Multi-window LLM vs deterministic benchmark for Trade Sentinel.
#
# Runs N non-overlapping windows, each with a deterministic _replay + an LLM
# _hybrid_replay, then reports per-window + mean + worst-window delta. This is
# the reliable scoreboard for evaluating pure-LLM / hybrid strategy changes —
# a single window is noise.
#
# Usage:  ./llm-walkforward.sh [extra args...]
# Examples:
#   ./llm-walkforward.sh                              # 4 windows × 30 days, pure-LLM
#   ./llm-walkforward.sh --windows 6                  # 6 windows × 30 days
#   ./llm-walkforward.sh --days-per-window 60         # 4 windows × 60 days
#   ./llm-walkforward.sh --start 2026-01-01 --end 2026-08-21
#   ./llm-walkforward.sh --no-pure-llm                # hybrid (deterministic + LLM review)
#   ./llm-walkforward.sh --trades                     # print every LLM trade per window
#
# All args are passed through to `app.optimize llm-walkforward`.
# Output is logged to stderr (visible on the terminal) and also written to
# /tmp/llm-walkforward-result.txt.

set -euo pipefail

OUT="/tmp/llm-walkforward-result.txt"

echo "Running llm-walkforward (multi-window LLM vs deterministic benchmark)..."
echo "Output will also be saved to $OUT"
echo

docker exec trade-sentinel /app/.venv/bin/python -m app.optimize llm-walkforward "$@" 2>&1 | tee "$OUT"