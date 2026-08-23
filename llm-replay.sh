#!/usr/bin/env bash
# Pure-LLM / hybrid vs deterministic replay comparison for Trade Sentinel.
#
# Usage:  ./llm-replay.sh [N trading days] [extra args...]
# Examples:
#   ./llm-replay.sh                       # full history, pure-LLM (default)
#   ./llm-replay.sh 60                    # last 60 trading days, pure-LLM
#   ./llm-replay.sh 120 --pure-llm        # last 120 trading days, pure-LLM
#   ./llm-replay.sh 90 --start 2025-01-01 # last 90 days from 2025-01-01 on
#   ./llm-replay.sh 60 --no-pure-llm      # last 60 trading days, hybrid mode
#
# The first positional argument is the number of trading days to simulate
# (trailing window); it becomes --days for the optimize tool. Any further
# args are passed through to `app.optimize hybrid-replay` (e.g. --start,
# --end, --trades). Pure-LLM mode is the default; pass --no-pure-llm to
# switch to the hybrid (deterministic + LLM review) mode.
#
# Output is logged to stderr (visible on the terminal) and also written to
# llm-replay-result.txt.

set -euo pipefail

OUT="/tmp/llm-replay-result.txt"

DAYS=""
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    DAYS="$1"
    shift
fi

ARGS=()
if [[ -n "$DAYS" ]]; then
    ARGS+=(--days "$DAYS")
fi
# Default to pure-LLM mode; --no-pure-llm switches to the hybrid replay.
NO_PURE_LLM=0
PASS=()
for a in "$@"; do
    if [[ "$a" == "--no-pure-llm" ]]; then
        NO_PURE_LLM=1
    else
        PASS+=("$a")
    fi
done
if [[ "$NO_PURE_LLM" -eq 0 ]]; then
    ARGS+=(--pure-llm)
fi
ARGS+=("${PASS[@]}")

echo "Running hybrid-replay (${DAYS:-full history} trading days)..."
echo "Output will also be saved to $OUT"
echo

docker exec trade-sentinel /app/.venv/bin/python -m app.optimize hybrid-replay "${ARGS[@]}" 2>&1 | tee "$OUT"
