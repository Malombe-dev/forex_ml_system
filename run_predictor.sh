#!/bin/bash
# Forex Predictor Launcher

# 1. Jump to the exact folder where THIS script is saved
cd "$(dirname "$0")" || exit

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Handle SSL certificates dynamically 
export REQUESTS_CA_BUNDLE=$(python -c "import certifi; print(certifi.where())")
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

# 4. Gating and execution parity (Unified Live Book + Two-Stage confirmation)
export ADVISOR_GATE=1
export TWO_STAGE_LIVE=1
export HTF_BIAS_GATE=0   # Kept 0: HTF bias was OFF in the +99R replay book
export CLOSED_BARS_ONLY=1

# 5. Run the predictor
python production_predictor.py

# 6. Keep window open on exit/crash
read -p "Press ENTER to close..."