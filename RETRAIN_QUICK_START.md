# RETRAIN QUICK REFERENCE
## What to do & in what order

---

## **BEFORE YOU START**

Verify you've applied all P0 fixes:
```bash
# Check ensemble weights are correct
grep -n "raw_prob = (p_xgb \* w\['xgb'\])" production_predictor.py
# Should find: production_predictor.py:1413

# Check htf_alignment uses D1 not H1
grep -n "_check_d1_alignment" production_predictor.py
# Should find: production_predictor.py:2164

# Check threshold override is gone
grep -n "self.prob_threshold.*0\.60" production_predictor.py
# Should find: 0 results (it's removed)
```

---

## **STEP 1: Fix Missing Dependencies** (2 min)

```bash
# Add schedule to requirements.txt
pip install schedule>=1.2.0

# Or reinstall all:
pip install -r requirements.txt
```

---

## **STEP 2: Clean Up Stale Files** (1 min)

```bash
bash cleanup_stale_models.sh
```

Expected output:
```
Removing stale: models/scaler_preprocessing.pkl
Removing stale: models/encoders_preprocessing.pkl
✅ Cleanup complete
```

---

## **STEP 3: Run Complete Retrain** (15-45 min, depends on data size)

```bash
python retrain_all.py
```

This will:
1. ✅ Verify environment (Python, dependencies, data files)
2. ✅ Validate data schema & freshness
3. ✅ Train both tasks (reversion + continuation) with correct logic
4. ✅ Generate metrics, ROC curves, calibration plots
5. ✅ Verify models load and features match
6. ✅ Show final readiness report

Expected output:
```
════════════════════════════════════════════════════════════════════════════
  STEP 1: PRE-FLIGHT CHECKS
════════════════════════════════════════════════════════════════════════════
...
✅ Reversion: AUC 0.7156 ✓ (good edge)
✅ Continuation: AUC 0.7089 ✓ (good edge)

════════════════════════════════════════════════════════════════════════════
  STEP 6: READINESS REPORT
════════════════════════════════════════════════════════════════════════════

🎯 RETRAIN COMPLETE — LIVE READY

✅ STATUS: Ready for live deployment
```

---

## **STEP 4: (OPTIONAL) Apply zone_touch_count Fix** (5 min)

This improves signal confidence on re-touched zones (+1-2% win rate).

```bash
# Review the fix:
cat FIX_zone_touch_count.py

# Apply manually:
# 1. Open production_predictor.py
# 2. Find ActiveZoneTracker class (line ~283)
# 3. Replace add_zone() method with the patched version from FIX_zone_touch_count.py
# 4. Replace is_duplicate() method with the patched version
# 5. Save
```

Or apply later — this is optional.

---

## **STEP 5: Deploy to Live** (1 min)

```bash
# Start live predictor with new models
python run_live_predictor.py

# Monitor signals
tail -f signals/all_signals.json | jq '.[-1]'

# Watch performance
# Expected first 5 signals:
#   • Win rate ≥55%: ✓ Working as trained
#   • 40–55%: Acceptable variance
#   • <40%: Debug feature logs
```

---

## **WHAT YOU'LL GET**

New trained models (v5):
```
models/
├── model_v5_reversion.pkl        ← XGBoost (fixed ensemble weights)
├── model_v5_continuation.pkl
├── rf_v5_reversion.pkl           ← Random Forest
├── rf_v5_continuation.pkl
├── lr_v5_reversion.pkl           ← Logistic Regression
├── lr_v5_continuation.pkl
├── scaler_v5_reversion.pkl       ← StandardScaler (27 features)
├── scaler_v5_continuation.pkl
├── encoders_v5_reversion.pkl     ← Label encoders
├── encoders_v5_continuation.pkl
├── metadata_v5_reversion.json    ← Schema & weights (0.50/0.35/0.15)
├── metadata_v5_continuation.json
└── metrics_{reversion,continuation}/
    ├── roc_pr_curves.png         ← ROC & Precision-Recall curves
    ├── calibration_curves.png    ← Probability calibration
    ├── feature_importances.png   ← Top 15 features by importance
    ├── confusion_matrix_65.png   ← Win/Loss distribution
    └── metrics_test.json         ← Full metrics on test set
```

---

## **EXPECTED METRICS (from train/test split)**

**Reversion model (test set — never seen during training):**
- ROC-AUC: 0.715–0.720 ← Baseline for comparison
- Precision @0.65: 55–60% ← Usable edge
- Signals @0.65: ~100–120 per 5K rows

**Continuation model (test set):**
- ROC-AUC: 0.708–0.715
- Precision @0.65: 65–70%
- Signals @0.65: ~10–20 per 5K rows

If you see:
- **AUC <0.55:** Data problem — check label_generator.py
- **AUC >0.85:** Leakage — check zone_time cutoff in model_trainer.py
- **AUC 0.55–0.72:** ✓ Normal; real edge exists

---

## **AFTER DEPLOYMENT: MONITORING**

### Weekly Check
```bash
# Check live win rate
cat signals/all_signals.json | jq '[.[] | select(.status=="hit_tp" or .status=="hit_sl")] | length as $total | [.[] | select(.status=="hit_tp")] | length as $wins | {total_trades: $total, wins: $wins, win_rate: ($wins/$total*100|round/1)}'

# Expected:
# ≥55%: All good, don't retrain for 30 days
# 40–55%: Normal variance, monitor
# <40%: Debug or retrain
```

### Monthly Retrain Criteria
```
If win_rate < 52% for last 20 signals:
  → python retrain_all.py  (use 60-month rolling window)

If win_rate ≥ 55% consistently:
  → No retrain needed; wait 30 days
```

### Quarterly Full Retrain (Optional)
```bash
# Every 90 days, do a full fresh train to recalibrate ensemble weights:
python retrain_all.py  (uses 300-month window by default)
```

---

## **TROUBLESHOOTING**

### "ModuleNotFoundError: No module named 'schedule'"
```bash
pip install schedule>=1.2.0
```

### "FileNotFoundError: data/labels/training_labels_clean.parquet"
```bash
# Regenerate labels:
python label_generator.py
python data_validation.py
python data_preprocessing.py
# Then retry retrain_all.py
```

### "ROC-AUC 0.50 (random baseline)"
```bash
# Check:
# 1. Is label data correct? (python data_validation.py)
# 2. Are features populated? (grep "zone_touch_count\|htf_trend" data/labels/training_labels_clean.parquet)
# 3. Is date-based train/test split working? (check split_by_date() in model_trainer.py)
```

### "Win rate <40% on live signals"
```bash
# Debug:
# 1. Check feature logs: tail -f signals/feature_logs/features_*.jsonl | jq '.'
# 2. Verify ensemble weights: cat models/metadata_v5_reversion.json | jq '.ensemble_weights'
# 3. Check live HTF trend: grep "h1_trend_alignment\|htf_trend_alignment" signals/feature_logs/*.jsonl | jq '.'
```

---

## **ONE-LINER QUICK START**

```bash
pip install schedule>=1.2.0 && bash cleanup_stale_models.sh && python retrain_all.py
```

Done! ✅

---

## **FILES CREATED FOR YOU**

- `retrain_all.py` — Main orchestrator (use this)
- `quick_retrain.sh` — One-liner wrapper
- `cleanup_stale_models.sh` — Remove old artifacts
- `FIX_zone_touch_count.py` — Optional enhancement
- `AUDIT_VERIFICATION_RESULTS.md` — Full verification report

---

**Questions?** Check the full audit: `AUDIT_VERIFICATION_RESULTS.md`
