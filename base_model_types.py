import pickle

for name, path in [
    ("XGB (model_v6_reversion.pkl)", "models/model_v6_reversion.pkl"),
    ("RF  (rf_v6_reversion.pkl)",    "models/rf_v6_reversion.pkl"),
    ("LR  (lr_v6_reversion.pkl)",    "models/lr_v6_reversion.pkl"),
]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    is_calibrated = type(obj).__name__ == "CalibratedClassifierCV"
    print(f"{name:35s} -> {type(obj).__name__:30s} calibrated={is_calibrated}")
    if is_calibrated:
        # peek at what's inside the calibration wrapper
        inner = getattr(obj, 'estimator', getattr(obj, 'base_estimator', None))
        print(f"    wraps: {type(inner).__name__}   method={getattr(obj,'method','?')}")