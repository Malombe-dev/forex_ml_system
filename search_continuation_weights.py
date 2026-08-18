"""
search_continuation_weights.py
---------------------------------------------------------------
Tests whether LR is pulling its weight in the CONTINUATION ensemble
(current: {'xgb': 0.50, 'rf': 0.35, 'ridge': 0.15}), by grid-searching
alternative weight splits on the VAL set only, then confirming the
winner ONCE on the TEST set.

Because continuation has very few signals at high thresholds (n=2 at
0.65 in your last run), precision is too noisy to select on. Selection
here is by log_loss / Brier on val (proper scoring rules over the full
distribution, not just the top bucket) — precision is reported for
context only, not used to pick the winner.

Usage:
    python search_continuation_weights.py

Run from ~/Desktop/all/forex_ml_system/
"""

import pickle
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, precision_score

from model_trainer import DualTaskTrainer, TRAIN_END_DATE, VAL_END_DATE

DATA_PATH = Path("data/labels/training_labels_clean.parquet")
MODELS_DIR = Path("models")
TASK = "continuation"

XGB_PATH = MODELS_DIR / f"model_v6_{TASK}.pkl"
RF_PATH = MODELS_DIR / f"rf_v6_{TASK}.pkl"
LR_PATH = MODELS_DIR / f"lr_v6_{TASK}.pkl"
SCALER_PATH = MODELS_DIR / f"scaler_v6_{TASK}.pkl"
ENCODERS_PATH = MODELS_DIR / f"encoders_v6_{TASK}.pkl"

BASELINE = {'xgb': 0.50, 'rf': 0.35, 'ridge': 0.15}
PRECISION_THRESHOLD = 0.55  # continuation's only threshold with a usable n in your last run


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["zone_time"] = pd.to_datetime(df["zone_time"])
    if split == "val":
        out = df[(df["zone_time"] >= TRAIN_END_DATE) & (df["zone_time"] < VAL_END_DATE)]
    elif split == "test":
        out = df[df["zone_time"] >= VAL_END_DATE]
    else:
        raise ValueError(split)
    return out.reset_index(drop=True)


def get_base_predictions(df: pd.DataFrame):
    trainer = DualTaskTrainer()
    trainer.label_encoders = load_pickle(ENCODERS_PATH)
    X, y = trainer.prepare_features(df, task=TASK, train_indices=None)

    scaler = load_pickle(SCALER_PATH)
    X_scaled = scaler.transform(X)

    xgb_model = load_pickle(XGB_PATH)
    rf_model = load_pickle(RF_PATH)
    lr_model = load_pickle(LR_PATH)

    p_xgb = xgb_model.predict_proba(X_scaled)[:, 1].astype(np.float64)
    p_rf = rf_model.predict_proba(X_scaled)[:, 1].astype(np.float64)
    p_lr = lr_model.predict_proba(X_scaled)[:, 1].astype(np.float64)
    y_true = (y.astype(np.float64) >= 1.0).astype(int).values
    return y_true, p_xgb, p_rf, p_lr


def blend(p_xgb, p_rf, p_lr, weights):
    p = p_xgb * weights['xgb'] + p_rf * weights['rf'] + p_lr * weights['ridge']
    return np.clip(p, 1e-7, 1 - 1e-7)


def candidate_weights():
    """Baseline + a grid of alternatives, all summing to 1.0."""
    candidates = {"baseline (current)": BASELINE}
    # Drop LR entirely, keep xgb:rf ratio close to baseline's 0.50:0.35
    candidates["drop_lr"] = {"xgb": 0.59, "rf": 0.41, "ridge": 0.0}
    # Shrink LR's weight
    for ridge_w in (0.05, 0.10):
        remaining = 1.0 - ridge_w
        xgb_w = round(remaining * (0.50 / 0.85), 3)
        rf_w = round(remaining - xgb_w, 3)
        candidates[f"shrink_lr_{ridge_w:.2f}"] = {"xgb": xgb_w, "rf": rf_w, "ridge": ridge_w}
    # Equal weight, for reference
    candidates["equal_weight"] = {"xgb": 1 / 3, "rf": 1 / 3, "ridge": 1 / 3}
    return candidates


def evaluate(y_true, p_xgb, p_rf, p_lr, weights):
    p = blend(p_xgb, p_rf, p_lr, weights)
    pred = (p >= PRECISION_THRESHOLD).astype(int)
    n_signals = int(pred.sum())
    precision = precision_score(y_true, pred, zero_division=0) if n_signals > 0 else float("nan")
    return {
        "log_loss": log_loss(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "roc_auc": roc_auc_score(y_true, p),
        f"precision@{PRECISION_THRESHOLD}": precision,
        f"n_signals@{PRECISION_THRESHOLD}": n_signals,
    }


def main():
    print("=" * 70)
    print(f"WEIGHT SEARCH — {TASK} ensemble, selecting on VAL set")
    print("=" * 70)
    print("Loading val split...")
    y_val, pxv, prv, plv = get_base_predictions(load_split("val"))
    print(f"  n={len(y_val)}  win_rate={y_val.mean():.4f}")

    candidates = candidate_weights()
    results = {}
    print(f"\n{'candidate':>20}  {'weights':>28}  {'log_loss':>9}  {'brier':>7}  {'roc_auc':>8}  "
          f"{'prec@'+str(PRECISION_THRESHOLD):>10}  {'n_sig':>6}")
    for name, w in candidates.items():
        r = evaluate(y_val, pxv, prv, plv, w)
        results[name] = r
        w_str = f"x={w['xgb']:.2f} r={w['rf']:.2f} l={w['ridge']:.2f}"
        prec = r[f"precision@{PRECISION_THRESHOLD}"]
        prec_str = f"{prec:.1%}" if not np.isnan(prec) else "n/a"
        print(f"{name:>20}  {w_str:>28}  {r['log_loss']:>9.4f}  {r['brier']:>7.4f}  "
              f"{r['roc_auc']:>8.4f}  {prec_str:>10}  {r[f'n_signals@{PRECISION_THRESHOLD}']:>6}")

    # Select winner by val log_loss (a proper scoring rule over the full
    # distribution — precision on a handful of signals is too noisy to trust).
    winner_name = min(results, key=lambda k: results[k]["log_loss"])
    winner_weights = candidates[winner_name]
    print(f"\n✅ Best on VAL by log_loss: '{winner_name}'  {winner_weights}")

    if winner_name == "baseline (current)":
        print("Baseline already wins on val — no change recommended. Stopping before touching test.")
        return

    print(f"\nConfirming '{winner_name}' vs baseline ONCE on TEST set...")
    y_test, pxt, prt, plt = get_base_predictions(load_split("test"))
    print(f"  n={len(y_test)}  win_rate={y_test.mean():.4f}\n")

    for name in ("baseline (current)", winner_name):
        r = evaluate(y_test, pxt, prt, plt, candidates[name])
        prec = r[f"precision@{PRECISION_THRESHOLD}"]
        prec_str = f"{prec:.1%}" if not np.isnan(prec) else "n/a"
        print(f"  TEST  {name:>20}  log_loss={r['log_loss']:.4f}  brier={r['brier']:.4f}  "
              f"roc_auc={r['roc_auc']:.4f}  precision@{PRECISION_THRESHOLD}={prec_str}  "
              f"n={r[f'n_signals@{PRECISION_THRESHOLD}']}")

    print("\nThis TEST comparison is the honest estimate. Do not re-run search against test.")


if __name__ == "__main__":
    main()