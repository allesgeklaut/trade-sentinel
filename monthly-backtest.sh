#!/usr/bin/env bash
# DCA backtest of the monthly qv-mom portfolio (stockstrat strategy).
#
# Usage:  ./monthly-backtest.sh [extra args...]
# Examples:
#   ./monthly-backtest.sh                                # full eligible window
#   ./monthly-backtest.sh --start 2023-01-01             # from 2023
#   ./monthly-backtest.sh --start 2022-01-01 --end 2025-12-31
#   ./monthly-backtest.sh --contribution 500 --picks     # $500/mo + show picks
#   ./monthly-backtest.sh --universe diversified-plus    # explicit universe
#
# Replays the strategy on the candles + fundamentals stored in the container
# DB: every month-end it deposits the contribution, picks the top-10
# (quality-value-momentum with hysteresis) and trades at that close (10 bps
# one-way). Reports the money-weighted IRR.
#
# Notes:
#  - Months before the first month with eligible fundamentals are trimmed
#    automatically (the yfinance feed only reaches back ~4-5 years).
#  - Data must exist first: run one manual rebalance (UI "Rebalance Now" or
#    curl -X POST localhost:8002/api/monthly/run) to fetch candles+fundamentals,
#    or the screener's "Load 10y" for the daily universe.

set -euo pipefail

echo "Running the monthly qv-mom DCA backtest (inside the trade-sentinel container)..."
echo

docker exec trade-sentinel /app/.venv/bin/python -m app.optimize monthly-backtest "$@"