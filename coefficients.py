import pickle, json
import numpy as np

with open("models/metadata_v6_reversion.json") as f:
    meta = json.load(f)
feature_names = meta["feature_names"]

with open("models/lr_v6_reversion.pkl", "rb") as f:
    lr_calibrated = pickle.load(f)

with open("models/scaler_v6_reversion.pkl", "rb") as f:
    scaler = pickle.load(f)

# CalibratedClassifierCV stores one fitted calibrator per CV fold in
# .calibrated_classifiers_; each wraps the underlying LogisticRegression
# (attribute name varies by sklearn version: estimator or base_estimator).
inner_models = []
for cc in lr_calibrated.calibrated_classifiers_:
    est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
    if est is not None:
        inner_models.append(est)

print(f"Found {len(inner_models)} inner LogisticRegression fold(s)\n")

# Average coefficients across CV folds
coefs = np.array([m.coef_[0] for m in inner_models])
mean_coef = coefs.mean(axis=0)
std_coef  = coefs.std(axis=0)

print(f"{'feature':30s} {'mean_coef':>10s} {'std_coef':>10s} {'scaler_mean':>12s} {'scaler_scale':>13s}")
order = np.argsort(-np.abs(mean_coef))
for i in order:
    name = feature_names[i]
    print(f"{name:30s} {mean_coef[i]:10.3f} {std_coef[i]:10.3f} "
          f"{scaler.mean_[i]:12.3f} {scaler.scale_[i]:13.3f}")

# Highlight touch count specifically
if "zone_touch_count" in feature_names:
    idx = feature_names.index("zone_touch_count")
    print(f"\nzone_touch_count -> coef={mean_coef[idx]:.3f}  "
          f"(scaler mean={scaler.mean_[idx]:.2f}, scale={scaler.scale_[idx]:.2f})")
    print(f"  logit change per +1 raw touch_count unit ≈ {mean_coef[idx]/scaler.scale_[idx]:.4f}")
    print(f"  logit change from touch=2 (near-training-mean) to touch=18 (your live zones) ≈ "
          f"{(mean_coef[idx]/scaler.scale_[idx]) * (18-2):.3f}")