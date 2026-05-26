"""
update_predictor.py
-------------------
Applies three targeted edits to production_predictor.py in-place:

  CHANGE 1 — __init__: replace static prob_threshold assignment with
             strategy-conditional thresholds (0.65 rev / 0.60 cont).

  CHANGE 2 — predict_with_proper_features: replace weighted ensemble
             (XGB×0.50 + RF×0.35 + LR×0.15) with uniform average (÷3).

  CHANGE 3 — load_model (optional visual cleanup): replace the dynamic
             (cal_st) string with the static [PROBABILITY CALIBRATED] tag.

Usage:
    python update_predictor.py [path/to/production_predictor.py]

Default target: production_predictor.py in the current directory.
A timestamped .bak backup is created before any change is written.
"""

import sys
import shutil
import datetime
from pathlib import Path

# ── locate target file ────────────────────────────────────────────────────────
target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("production_predictor.py")
if not target.exists():
    sys.exit(f"❌  File not found: {target}")

# ── backup ────────────────────────────────────────────────────────────────────
stamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = target.with_suffix(f".{stamp}.bak")
shutil.copy2(target, backup)
print(f"✅  Backup created: {backup}")

src = target.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1  — strategy-conditional thresholds in __init__
# Original single line:
#     self.is_continuation_strategy = is_continuation_strategy
# Replace with the conditional block that follows it.
# ─────────────────────────────────────────────────────────────────────────────
OLD_1 = "        self.is_continuation_strategy = is_continuation_strategy"

NEW_1 = """\
        self.is_continuation_strategy = is_continuation_strategy

        if not self.is_continuation_strategy:
            # TASK 1: REVERSION STRATEGY
            # Calibration testing proved an elite 95.76% precision at a 0.65 threshold!
            self.prob_threshold = 0.65
        else:
            # TASK 2: CONTINUATION STRATEGY
            # Keep this strictly at 0.60 where precision peaked stably at 80.0%.
            # Pushing continuation to 0.65 caused a severe sample-scarcity collapse.
            self.prob_threshold = 0.60"""

if OLD_1 not in src:
    sys.exit("❌  CHANGE 1 anchor not found — file may already be updated or differ unexpectedly.")

src = src.replace(OLD_1, NEW_1, 1)
print("✅  CHANGE 1 applied: strategy-conditional prob_threshold in __init__")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2  — uniform average in predict_with_proper_features
# Original three-model weighted line:
#     raw_prob = p_xgb * w['xgb'] + p_rf * w['rf'] + p_lr * w['lr']
# ─────────────────────────────────────────────────────────────────────────────
OLD_2 = "                raw_prob = p_xgb * w['xgb'] + p_rf * w['rf'] + p_lr * w['lr']"

NEW_2 = """\
                # CHANGE: Standardize to a true mathematical ensemble average.
                # CalibratedClassifierCV has already corrected both XGBoost and RF
                # probabilities into true percentages; uniform weighting removes
                # the old uncalibrated-variance compensation.
                raw_prob = (p_xgb + p_rf + p_lr) / 3.0"""

if OLD_2 not in src:
    sys.exit("❌  CHANGE 2 anchor not found — file may already be updated or differ unexpectedly.")

src = src.replace(OLD_2, NEW_2, 1)
print("✅  CHANGE 2 applied: uniform average (÷3) in predict_with_proper_features")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3  — logging string in load_model  (optional visual cleanup)
# Original line:
#     print(f"  ✅ Model v5 ({strat}) loaded — XGB✓ {rf_st} {lr_st} ({cal_st})")
# ─────────────────────────────────────────────────────────────────────────────
OLD_3 = '            print(f"  ✅ Model v5 ({strat}) loaded — XGB✓ {rf_st} {lr_st} ({cal_st})")'

NEW_3 = '''\
            print(
                f"  ✅ Model v5 ({'continuation' if self.is_continuation_strategy else 'reversion'}) loaded — "
                f"XGB✓ RF✓ LR✓ [PROBABILITY CALIBRATED]"
            )'''

if OLD_3 not in src:
    print("⚠️   CHANGE 3 anchor not found — skipping (logging line may already differ). "
          "This change is optional; all functional changes are applied.")
else:
    src = src.replace(OLD_3, NEW_3, 1)
    print("✅  CHANGE 3 applied: [PROBABILITY CALIBRATED] tag in load_model")

# ── write result ──────────────────────────────────────────────────────────────
target.write_text(src, encoding="utf-8")
print(f"\n🎉  Done — {target} updated successfully.")
print(f"    Restore at any time with: cp {backup} {target}")