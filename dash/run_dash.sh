#!/bin/bash
# Forex Predictor Launcher
cd "$(dirname "$0")/.." || exit
source venv/bin/activate
cd dash || exit
python server.py
read -p "Press ENTER to close..."