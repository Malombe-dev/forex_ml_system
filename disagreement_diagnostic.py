"""
disagreement_diagnostic_v2.py
---------------------------------------------------------------
Tests whether the 0.4-0.8 miscalibrated band in the REVERSION ensemble
is disproportionately populated by rows where XGB / RF / LR disagree.

Built directly from your uploaded model_trainer.py + features.py —
no guessed column names or feature logic this time:

  - Label: df['label'] is a continuous score (0.5, 0.6, 0.75, 1.0, ...).
    Binary win/loss = (label >= 1.0).astype(int)   [model_trainer.py:755]
  - Features: NUMERIC_FEATURES + <cat>_encoded for zone_type, session,
    approach_atr_regime, in the exact order from features.py
  - Test split: zone_time >= VAL_END_DATE (2024-02-01)   [model_trainer.py:75]
  - Ensemble = 0.50*p_xgb + 0.35*p_rf + 0.15*p_lr (NOT a simple mean)
    [model_trainer.py:731-735]
  - No ensemble calibrator is currently active (your last run logged
    "No ensemble calibrator available" for reversion), so this script
    uses the same raw weighted blend your metrics report used.

Run from ~/Desktop/all/forex_ml_system/:
    python disagreement_diagnostic_v2.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # your real schema

# ------------------------------------------------------------------
DATA_PATH = Path("data/labels/training_labels_clean.parquet")
MODELS_DIR = Path("models")

XGB_PATH = MODELS_DIR / "model_v6_reversion.pkl"
RF_PATH = MODELS_DIR / "rf_v6_reversion.pkl"
LR_PATH = MODELS_DIR / "lr_v6_reversion.pkl"
SCALER_PATH = MODELS_DIR / "scaler_v6_reversion.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders_v6_reversion.pkl"

VAL_END_DATE = pd.Timestamp("2024-02-01")   # test split start, per model_trainer.py
ZONE_TIME_COL = "zone_time"
LABEL_COL = "label"                          # continuous; binarized below at >= 1.0

ENSEMBLE_WEIGHTS = {'xgb': 0.50, 'rf': 0.35, 'ridge': 0.15}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_feature_matrix(df, encoders):
    """Mirror DualTaskTrainer.prepare_features() inference-time branch exactly."""
    df = df.copy()

    # Same sentinel/outlier cleanup as prepare_features
    if 'time_since_last_touch' in df.columns:
        df['time_since_last_touch'] = df['time_since_last_touch'].clip(upper=500)
    if 'zone_strength' in df.columns:
        cap = df['zone_strength'].quantile(0.99)
        df['zone_strength'] = df['zone_strength'].clip(upper=cap)

    numeric_features = list(NUMERIC_FEATURES)
    categorical_features = list(CATEGORICAL_FEATURES)

    missing_numeric = [f for f in numeric_features if f not in df.columns]
    missing_cats = [f for f in categorical_features if f not in df.columns]
    if missing_numeric or missing_cats:
        raise ValueError(
            f"Missing numeric: {missing_numeric}\nMissing categorical: {missing_cats}"
        )

    X = df[numeric_features].copy()
    for cat_col in categorical_features:
        if cat_col not in encoders:
            raise ValueError(f"No saved encoder for '{cat_col}' in encoders_v6_reversion.pkl")
        le = encoders[cat_col]
        encoded = []
        for val in df[cat_col].astype(str):
            encoded.append(le.transform([val])[0] if val in le.classes_ else 0)
        X[f'{cat_col}_encoded'] = encoded

    return X


def main():
    print("=" * 70)
    print("DISAGREEMENT DIAGNOSTIC v2 — REVERSION ensemble, TEST set")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    df[ZONE_TIME_COL] = pd.to_datetime(df[ZONE_TIME_COL])
    test_df = df[df[ZONE_TIME_COL] >= VAL_END_DATE].reset_index(drop=True)
    print(f"Test rows: {len(test_df)}  (zone_time >= {VAL_END_DATE.date()})")

    y_raw = test_df[LABEL_COL].astype(np.float64)
    y_true = (y_raw >= 1.0).astype(int).values
    print(f"Win rate (label>=1.0): {y_true.mean():.4f}  "
          f"(expect ≈0.419 per your training logs)")

    xgb_model = load_pickle(XGB_PATH)
    rf_model = load_pickle(RF_PATH)
    lr_model = load_pickle(LR_PATH)
    scaler = load_pickle(SCALER_PATH)
    encoders = load_pickle(ENCODERS_PATH)

    X_test = build_feature_matrix(test_df, encoders)
    X_test_scaled = scaler.transform(X_test)

    p_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
    p_rf = rf_model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
    p_lr = lr_model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)

    p_ensemble = (
        p_xgb * ENSEMBLE_WEIGHTS['xgb'] +
        p_rf * ENSEMBLE_WEIGHTS['rf'] +
        p_lr * ENSEMBLE_WEIGHTS['ridge']
    )
    p_ensemble = np.clip(p_ensemble, 1e-7, 1 - 1e-7).astype(np.float32)

    print(f"Ensemble ROC-AUC sanity check: ", end="")
    from sklearn.metrics import roc_auc_score
    print(f"{roc_auc_score(y_true, p_ensemble):.4f}  (expect ≈0.845 per your logs)")

    # ---- disagreement metrics (equal-weight spread across the 3 base models) ----
    stacked = np.vstack([p_xgb, p_rf, p_lr])
    disagreement_std = stacked.std(axis=0)
    disagreement_range = stacked.max(axis=0) - stacked.min(axis=0)

    out = pd.DataFrame({
        "y_true": y_true,
        "p_xgb": p_xgb,
        "p_rf": p_rf,
        "p_lr": p_lr,
        "p_ensemble": p_ensemble,
        "disagreement_std": disagreement_std,
        "disagreement_range": disagreement_range,
    })

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out["bucket"] = pd.cut(out["p_ensemble"], bins=bins, include_lowest=True)

    print("\n" + "-" * 70)
    print("STEP 1 — Disagreement by probability bucket")
    print("-" * 70)
    summary = out.groupby("bucket", observed=True).agg(
        n=("y_true", "size"),
        predicted_prob=("p_ensemble", "mean"),
        actual_win_rate=("y_true", "mean"),
        mean_disagreement_std=("disagreement_std", "mean"),
        mean_disagreement_range=("disagreement_range", "mean"),
    )
    summary["calib_error"] = (summary["predicted_prob"] - summary["actual_win_rate"]).abs()
    print(summary.round(4).to_string())

    print("\n" + "-" * 70)
    print("STEP 2 — Correlation: bucket calibration error vs mean disagreement")
    print("-" * 70)
    bucket_corr = np.corrcoef(
        summary["calib_error"].fillna(0),
        summary["mean_disagreement_std"].fillna(0),
    )[0, 1]
    print(f"Pearson r (calib_error, mean disagreement) across buckets = {bucket_corr:.3f}")
    print("r > ~0.4  → supports the hypothesis (miscalibrated buckets = more disagreement)")
    print("r ≈ 0 or negative → hypothesis not supported by this data")

    print("\n" + "-" * 70)
    print("STEP 3 — Within-bucket split: low vs high disagreement")
    print("-" * 70)
    rows = []
    for b, g in out.groupby("bucket", observed=True):
        if len(g) < 10:
            continue
        median_dis = g["disagreement_std"].median()
        low = g[g["disagreement_std"] <= median_dis]
        high = g[g["disagreement_std"] > median_dis]
        rows.append({
            "bucket": str(b),
            "predicted_prob": g["p_ensemble"].mean(),
            "n_low": len(low),
            "win_rate_low_disagreement": low["y_true"].mean(),
            "n_high": len(high),
            "win_rate_high_disagreement": high["y_true"].mean(),
        })
    split_df = pd.DataFrame(rows).set_index("bucket")
    print(split_df.round(4).to_string())

    print("\n" + "-" * 70)
    print("STEP 4 — Agreement-gated signal set")
    print("-" * 70)
    print("Reject signals where models disagree a lot, even if blended prob clears threshold.")
    for thresh in (0.65, 0.70, 0.75, 0.80):
        for max_disagreement in (None, 0.15, 0.10):
            mask = out["p_ensemble"] >= thresh
            label = "no gate"
            if max_disagreement is not None:
                mask = mask & (out["disagreement_std"] <= max_disagreement)
                label = f"disagreement_std <= {max_disagreement}"
            n = mask.sum()
            if n == 0:
                print(f"  threshold={thresh:.2f}  {label:28s}  n=0 (no signals)")
                continue
            precision = out.loc[mask, "y_true"].mean()
            print(f"  threshold={thresh:.2f}  {label:28s}  n={n:4d}  "
                  f"precision(win rate)={precision:.4f}")

    out.to_csv("disagreement_diagnostic_output.csv", index=False)
    print("\nSaved full row-level output → disagreement_diagnostic_output.csv")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()