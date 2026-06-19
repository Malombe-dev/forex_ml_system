#!/bin/bash
echo "Starting forex ML system..."

mkdir -p signals models

echo "Starting dashboard (port 5050)..."
python dash/server.py &

echo "Starting Telegram monitor..."
python telegram_signal_monitor.py &

echo "Starting production predictor..."
python production_predictor.py
