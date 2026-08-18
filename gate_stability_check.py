"""
gate_stability_check.py
---------------------------------------------------------------
Checks whether the disagreement gate (disagreement_std <= 0.10) at the
PROB_THRESHOLD you actually trade at (0.65) holds up consistently across
time, or whether it only "worked" because of one lucky stretch inside the
2024-02-01 -> 2026-08-06 test window.

SCOPE / HONESTY NOTE
    This is NOT full walk-forward retraining. It uses your existing trained
    reversion models (model_v6_reversion.pkl etc.) and slices the already
    held-out test set into N sequential time chunks, checking gate
    performance in each chunk separately. True walk-forward would retrain
    on rolling windows ending just before each chunk — ask if you want that
    built next; it's a bigger job (needs to call into model_trainer.py's
    training loop repeatedly).

Built on the same verified pipeline as disagreement_diagnostic_v2.py:
  - features.py NUMERIC_FEATURES / CATEGORICAL_FEATURES (exact schema)
  - label = df['label'] >= 1.0   [model_trainer.py:755]
  - ensemble = 0.50*xgb + 0.35*rf + 0.15*lr   [model_trainer.py:731-735]
  - reproduces test-set AUC 0.8454 exactly, so trust in this pipeline
    is inherited from that earlier verification.

Run from ~/Desktop/all/forex_ml_system/:
    python gate_stability_check.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ------------------------------------------------------------------
DATA_PATH = Path("data/labels/training_labels_clean.parquet")
MODELS_DIR = Path("models")

XGB_PATH = MODELS_DIR / "model_v6_reversion.pkl"
RF_PATH = MODELS_DIR / "rf_v6_reversion.pkl"
LR_PATH = MODELS_DIR / "lr_v6_reversion.pkl"
SCALER_PATH = MODELS_DIR / "scaler_v6_reversion.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders_v6_reversion.pkl"

VAL_END_DATE = pd.Timestamp("2024-02-01")
ZONE_TIME_COL = "zone_time"
LABEL_COL = "label"

ENSEMBLE_WEIGHTS = {'xgb': 0.50, 'rf': 0.35, 'ridge': 0.15}
PROB_THRESHOLD = 0.65          # <- the threshold you actually trade at
MAX_DISAGREEMENT = 0.10        # <- the gate validated earlier
N_CHUNKS = 5                   # split the test window into this many sequential slices


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_feature_matrix(df, encoders):
    df = df.copy()
    if 'time_since_last_touch' in df.columns:
        df['time_since_last_touch'] = df['time_since_last_touch'].clip(upper=500)
    if 'zone_strength' in df.columns:
        cap = df['zone_strength'].quantile(0.99)
        df['zone_strength'] = df['zone_strength'].clip(upper=cap)

    numeric_features = list(NUMERIC_FEATURES)
    categorical_features = list(CATEGORICAL_FEATURES)
    X = df[numeric_features].copy()
    for cat_col in categorical_features:
        le = encoders[cat_col]
        encoded = []
        for val in df[cat_col].astype(str):
            encoded.append(le.transform([val])[0] if val in le.classes_ else 0)
        X[f'{cat_col}_encoded'] = encoded
    return X


def score_precision(mask, y_true):
    n = int(mask.sum())
    if n == 0:
        return n, float('nan')
    return n, float(y_true[mask].mean())


def main():
    print("=" * 78)
    print(f"GATE STABILITY CHECK — threshold={PROB_THRESHOLD}, "
          f"max_disagreement={MAX_DISAGREEMENT}, {N_CHUNKS} time chunks")
    print("=" * 78)

    df = pd.read_parquet(DATA_PATH)
    df[ZONE_TIME_COL] = pd.to_datetime(df[ZONE_TIME_COL])
    test_df = df[df[ZONE_TIME_COL] >= VAL_END_DATE].sort_values(ZONE_TIME_COL).reset_index(drop=True)
    print(f"Full test set: {len(test_df)} rows, "
          f"{test_df[ZONE_TIME_COL].min().date()} -> {test_df[ZONE_TIME_COL].max().date()}")

    y_true = (test_df[LABEL_COL].astype(np.float64) >= 1.0).astype(int).values

    xgb_model = load_pickle(XGB_PATH)
    rf_model = load_pickle(RF_PATH)
    lr_model = load_pickle(LR_PATH)
    scaler = load_pickle(SCALER_PATH)
    encoders = load_pickle(ENCODERS_PATH)

    X = build_feature_matrix(test_df, encoders)
    X_scaled = scaler.transform(X)

    p_xgb = xgb_model.predict_proba(X_scaled)[:, 1].astype(np.float32)
    p_rf = rf_model.predict_proba(X_scaled)[:, 1].astype(np.float32)
    p_lr = lr_model.predict_proba(X_scaled)[:, 1].astype(np.float32)
    p_ensemble = np.clip(
        p_xgb * ENSEMBLE_WEIGHTS['xgb'] + p_rf * ENSEMBLE_WEIGHTS['rf'] + p_lr * ENSEMBLE_WEIGHTS['ridge'],
        1e-7, 1 - 1e-7
    ).astype(np.float32)

    print(f"Full-set ensemble AUC sanity check: {roc_auc_score(y_true, p_ensemble):.4f}  "
          f"(expect ~0.845)")

    disagreement_std = np.vstack([p_xgb, p_rf, p_lr]).std(axis=0)

    test_df = test_df.copy()
    test_df["y_true"] = y_true
    test_df["p_ensemble"] = p_ensemble
    test_df["disagreement_std"] = disagreement_std

    # ---- split into N sequential time chunks of roughly equal row count ----
    chunk_ids = pd.qcut(test_df.index, N_CHUNKS, labels=False)
    test_df["chunk"] = chunk_ids

    print("\n" + "-" * 78)
    print(f"Per-chunk: no-gate vs gated precision @ threshold={PROB_THRESHOLD}")
    print("-" * 78)
    header = (f"{'Chunk':<8}{'Date range':<26}{'NoGate n':>9}{'NoGate prec':>13}"
              f"{'Gated n':>9}{'Gated prec':>12}{'Delta':>9}")
    print(header)

    deltas = []
    for c in sorted(test_df["chunk"].unique()):
        g = test_df[test_df["chunk"] == c]
        date_range = f"{g[ZONE_TIME_COL].min().date()} -> {g[ZONE_TIME_COL].max().date()}"

        no_gate_mask = g["p_ensemble"] >= PROB_THRESHOLD
        gated_mask = no_gate_mask & (g["disagreement_std"] <= MAX_DISAGREEMENT)

        n_ng, prec_ng = score_precision(no_gate_mask.values, g["y_true"].values)
        n_g, prec_g = score_precision(gated_mask.values, g["y_true"].values)

        delta = (prec_g - prec_ng) if (not np.isnan(prec_g) and not np.isnan(prec_ng)) else float('nan')
        if not np.isnan(delta):
            deltas.append(delta)

        prec_ng_str = f"{prec_ng:.1%}" if not np.isnan(prec_ng) else "n/a"
        prec_g_str = f"{prec_g:.1%}" if not np.isnan(prec_g) else "n/a"
        delta_str = f"{delta:+.1%}" if not np.isnan(delta) else "n/a"

        print(f"{c:<8}{date_range:<26}{n_ng:>9}{prec_ng_str:>13}{n_g:>9}{prec_g_str:>12}{delta_str:>9}")

    print("-" * 78)
    if deltas:
        print(f"Chunks where gate helped (delta > 0):  {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
        print(f"Chunks where gate hurt   (delta < 0):  {sum(1 for d in deltas if d < 0)}/{len(deltas)}")
        print(f"Mean delta across chunks: {np.mean(deltas):+.1%}")
        print(f"Std of delta across chunks: {np.std(deltas):.1%}  (lower = more consistent)")
    else:
        print("Not enough signals in any chunk to compare — try fewer chunks (lower N_CHUNKS).")

    print("\n" + "-" * 78)
    print("Interpretation")
    print("-" * 78)
    print("If the gate helped (delta > 0) in most/all chunks, with a small std,")
    print("that's real evidence of a stable effect over time, not a one-window fluke.")
    print("If it flips sign across chunks or the std is large relative to the mean,")
    print("treat the earlier 83.4%->92.4% result as fragile — don't trust it blindly live.")

    test_df.to_csv("gate_stability_output.csv", index=False)
    print("\nSaved full row-level output -> gate_stability_output.csv")
    print("=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()