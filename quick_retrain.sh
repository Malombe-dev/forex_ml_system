#!/bin/bash
# QUICK START: Retrain after P0/P1 fixes
# ============================================

set -e  # Exit on error

echo "🚀 FOREX ML RETRAIN — QUICK START"
echo "=================================="
echo ""

# Step 1: Install schedule dependency
echo "📦 Step 1: Update dependencies..."
pip install schedule>=1.2.0 -q && echo "  ✅ schedule installed" || echo "  ⚠️  schedule install had issues"

# Step 2: Clean up stale files (optional)
echo ""
echo "🧹 Step 2: Cleaning up stale preprocessing artifacts..."
bash cleanup_stale_models.sh

# Step 3: Run retrain orchestrator
echo ""
echo "🎯 Step 3: Running complete retrain pipeline..."
python retrain_all.py

# Step 4: Verify output
echo ""
echo "✅ RETRAIN COMPLETE!"
echo ""
echo "🔍 Verify models were created:"
ls -lh models/model_v5_*.pkl models/scaler_v5_*.pkl 2>/dev/null | tail -4

echo ""
echo "📊 View metrics:"
echo "  • Reversion: cat models/metrics_reversion/metrics_test.json | jq '.models.Ensemble'"
echo "  • Continuation: cat models/metrics_continuation/metrics_test.json | jq '.models.Ensemble'"

echo ""
echo "🚀 Ready to deploy! Run:"
echo "  python run_live_predictor.py"
