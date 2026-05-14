#!/bin/bash
# Forex Predictor Launcher
cd "$(dirname "$0")" || exit
source venv/bin/activate
python production_predictor.py
read -p "Press ENTER to close..."
