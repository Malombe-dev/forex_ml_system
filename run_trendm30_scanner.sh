#!/bin/bash
# TrendM30 High-WinRate ML Scanner Launcher
# Usage:
#   ./run_trendm30_scanner.sh            -> Run single scan on live bars
#   ./run_trendm30_scanner.sh --loop     -> Continuous daemon refreshing after every M30 candle close
#   ./run_trendm30_scanner.sh --all-pairs --loop -> Continuous daemon scanning all 10 FX pairs + Gold

cd "$(dirname "$0")" || exit
source venv/bin/activate
python3 trendm30_ml_filter.py "$@"
