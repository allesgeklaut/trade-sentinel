#!/usr/bin/env bash
# Walk-forward optimization for the Trade Sentinel strategy.
#
# Usage:  ./walkforward.sh [extra args...]
# Examples:
#   ./walkforward.sh                          # full walk-forward (2016+)
#   ./walkforward.sh --start 2020-01-01        # start from 2020
#   ./walkforward.sh --start 2017-01-01 --end 2024-12-31
#
# Default bounds: 2016-01-01 .. 2026-07-15 (all tickers with sufficient data
# start in 2016; only IFX.DE goes back to 2000).
#
# The sweep grid now includes max_run_5d (0/12/15/20%) as a dimension, so the
# walk-forward will select the best run-up block per train window and test
# it out-of-sample. Grid size: 96 configs per window.
#
# Progress is logged to stderr (visible on the terminal) and the full
# output is also written to walkforward-result.txt.

set -euo pipefail

OUT="/tmp/walkforward-result.txt"

# Default bounds: 2016-01-01 .. 2026-07-15. All usable tickers start in 2016;
# end leaves a buffer for forward-outcome scoring.
DEFAULT_ARGS="--start 2016-01-01 --end 2026-07-15"

echo "Running walk-forward optimization (inside the trade-sentinel container)..."
echo "Output will also be saved to $OUT"
echo

docker exec trade-sentinel /app/.venv/bin/python -m app.optimize walkforward $DEFAULT_ARGS "$@" 2>&1 | tee "$OUT"