#!/bin/bash
# Forex Predictor Launcher

# 1. Jump to the exact folder where THIS script is saved
cd "$(dirname "$0")" || exit

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Handle SSL certificates dynamically 
export REQUESTS_CA_BUNDLE=$(python -c "import certifi; print(certifi.where())")
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

# 4. Run the predictor
python production_predictor.py

# 5. Keep window open on exit/crash
read -p "Press ENTER to close..."