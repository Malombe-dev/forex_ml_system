#!/bin/bash
# FULL RETRAIN — correct end-to-end pipeline
# ============================================
# Runs everything retrain_all.py / quick_retrain.sh do NOT do:
# zone regeneration + label generation, before validation/preprocessing/training.
#
# set -e: stop immediately on any failure. Do NOT train on a step that
# silently failed or used stale data — that's exactly how the last several
# stale-cache issues went unnoticed for multiple runs.
set -e

echo "======================================================================"
echo "  FULL RETRAIN PIPELINE"
echo "  Started: $(date)"
echo "======================================================================"

# ── Step 1: Regenerate zones (all 4 symbols, correct pip_size each) ─────────
echo ""
echo "▶ STEP 1/5: Regenerating zones (smc_detector.py)"
echo "----------------------------------------------------------------------"
python smc_detector.py --symbol EURUSD --pip_size 0.0001
python smc_detector.py --symbol GBPUSD --pip_size 0.0001
python smc_detector.py --symbol USDJPY --pip_size 0.01
python smc_detector.py --symbol XAUUSD --pip_size 1.0

echo ""
echo "  Verifying zone cache freshness:"
ls -la data/features/merged_zones_*.parquet
echo ""
echo "  ⚠️  Confirm all 4 files above show TODAY's date before continuing."
echo "     (This script will proceed regardless — check manually if unsure.)"

# ── Step 2: Regenerate labels from the fresh zones ───────────────────────────
echo ""
echo "▶ STEP 2/5: Generating labels (label_generator.py)"
echo "----------------------------------------------------------------------"
python label_generator.py --symbol ALL

# ── Step 3: Validate the fresh label data ────────────────────────────────────
echo ""
echo "▶ STEP 3/5: Validating labels (data_validation.py)"
echo "----------------------------------------------------------------------"
python data_validation.py

# ── Step 4: Preprocess (clip/encode/scale, writes training_labels_clean.parquet) ──
echo ""
echo "▶ STEP 4/5: Preprocessing (data_preprocessing.py)"
echo "----------------------------------------------------------------------"
python data_preprocessing.py

# ── Step 5: Train ─────────────────────────────────────────────────────────────
echo ""
echo "▶ STEP 5/5: Training (model_trainer.py)"
echo "----------------------------------------------------------------------"
python model_trainer.py

echo ""
echo "======================================================================"
echo "  ✅ FULL RETRAIN PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo "======================================================================"
echo ""
echo "Next:"
echo "  1. Check calibration curves:"
echo "       models/metrics_reversion/calibration_curves.png"
echo "       models/metrics_continuation/calibration_curves.png"
echo "  2. Re-run the feature audit against the fresh data:"
echo "       python diagnose_all_features.py"
echo "  3. Check reversion ROC-AUC against the model_trainer.py's own"
echo "     printed 'realistic expectations' block — treat that as a"
echo "     rough guide, not a hard target (h4_fvg_inside taught us some"
echo "     features carry more real signal than a generic prior assumes)."
echo "  4. Confirm which model version (v5/v6) this run actually saved"
echo "     under before deploying — that inconsistency is still"
echo "     unreconciled across model_trainer.py / retrain_all.py /"
echo "     quick_retrain.sh."