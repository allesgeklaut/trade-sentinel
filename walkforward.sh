#!/usr/bin/env bash
# Walk-forward optimization for the Trade Sentinel strategy.
#
# Usage:  ./walkforward.sh [extra args...]
# Examples:
#   ./walkforward.sh                          # full 10y walk-forward
#   ./walkforward.sh --start 2020-01-01       # start from 2020
#   ./walkforward.sh --start 2017-01-01 --end 2024-12-31
#
# Progress is logged to stderr (visible on the terminal) and the full
# output is also written to walkforward-result.txt.

set -euo pipefail

OUT="/tmp/walkforward-result.txt"

echo "Running walk-forward optimization (inside the trade-sentinel container)..."
echo "Output will also be saved to $OUT"
echo

docker exec trade-sentinel /app/.venv/bin/python -m app.optimize walkforward "$@" 2>&1 | tee "$OUT"