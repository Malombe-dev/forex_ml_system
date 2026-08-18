import pickle

print("="*90)
print("ensemble_calibrator_v6_reversion.pkl  (the ONE that actually loaded as a real object)")
print("="*90)
with open("models/ensemble_calibrator_v6_reversion.pkl", "rb") as f:
    obj = pickle.load(f)
print(f"type: {type(obj)}")
print(f"method: {getattr(obj, 'method', '?')}")
print(f"classes_: {getattr(obj, 'classes_', '?')}")
print(f"has predict_proba: {hasattr(obj, 'predict_proba')}")
print(f"has predict: {hasattr(obj, 'predict')}")
try:
    print(f"n_features_in_: {getattr(obj, 'n_features_in_', '?')}")
except Exception:
    pass

print()
print("="*90)
print("calibrator_v6_reversion.pkl  (has xgb_calibrated / rf_calibrated / lr_calibrated keys)")
print("="*90)
with open("models/calibrator_v6_reversion.pkl", "rb") as f:
    obj2 = pickle.load(f)
for k, v in obj2.items():
    print(f"  {k:20s} -> {type(v).__name__}"
          + (f"  method={getattr(v,'method','')}" if hasattr(v, 'method') else ""))

print()
print("="*90)
print("calibration_ensemble_v6_reversion.pkl  (has xgb_model/rf_model/lr_model/weights keys)")
print("="*90)
with open("models/calibration_ensemble_v6_reversion.pkl", "rb") as f:
    obj3 = pickle.load(f)
for k, v in obj3.items():
    print(f"  {k:20s} -> {type(v).__name__}: {v if not hasattr(v,'shape') and not hasattr(v,'predict') else '(model/array)'}")