"""
tune_disagreement_gate.py
---------------------------------------------------------------
Tunes per-bucket disagreement_std gates WITHOUT touching the test set
for selection — only for a single, final confirmation.

Two modes:

  python tune_disagreement_gate.py search
      Runs on the VAL split (TRAIN_END_DATE <= zone_time < VAL_END_DATE).
      Grid-searches a disagreement_std cutoff per probability bucket,
      subject to a minimum-coverage floor so we don't pick a threshold
      that survives on 3 signals. Writes the winning per-bucket config to
      gate_config.json and prints the full grid so you can sanity check
      picks by eye, not just trust the argmax.

  python tune_disagreement_gate.py apply
      Loads gate_config.json (produced by `search`) and applies it ONCE
      to the TEST split. Prints gated vs ungated precision/coverage side
      by side. This is the only number that should be trusted as a real
      estimate — do not re-run `search` against test and iterate.

Reuses DualTaskTrainer.prepare_features() directly from model_trainer.py
so feature construction and the label definition are guaranteed to match
what was actually trained, instead of a hand-rolled copy.

Run from ~/Desktop/all/forex_ml_system/:
    python tune_disagreement_gate.py search
    python tune_disagreement_gate.py apply
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

from model_trainer import DualTaskTrainer, TRAIN_END_DATE, VAL_END_DATE

DATA_PATH = Path("data/labels/training_labels_clean.parquet")
MODELS_DIR = Path("models")
TASK = "reversion"  # the gate only applies here — continuation has too few signals to gate meaningfully

XGB_PATH = MODELS_DIR / f"model_v6_{TASK}.pkl"
RF_PATH = MODELS_DIR / f"rf_v6_{TASK}.pkl"
LR_PATH = MODELS_DIR / f"lr_v6_{TASK}.pkl"
SCALER_PATH = MODELS_DIR / f"scaler_v6_{TASK}.pkl"
ENCODERS_PATH = MODELS_DIR / f"encoders_v6_{TASK}.pkl"

ENSEMBLE_WEIGHTS = {'xgb': 0.50, 'rf': 0.35, 'ridge': 0.15}

BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Only these buckets showed calibration error correlated with disagreement
# in the original diagnostic — no point gating buckets that were already fine.
GATED_BUCKETS = ["(0.4, 0.5]", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]"]

# Candidate cutoffs to grid-search per bucket.
CANDIDATE_CUTOFFS = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, None]  # None = no gate
MIN_SIGNALS_FLOOR = 15  # don't accept a threshold that leaves fewer than this many signals

CONFIG_PATH = Path("gate_config.json")


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


def score_split(df: pd.DataFrame):
    """Returns a DataFrame with y_true, p_xgb, p_rf, p_lr, p_ensemble, disagreement_std, bucket."""
    trainer = DualTaskTrainer()
    trainer.label_encoders = load_pickle(ENCODERS_PATH)
    X, y = trainer.prepare_features(df, task=TASK, train_indices=None)

    scaler = load_pickle(SCALER_PATH)
    X_scaled = scaler.transform(X)

    xgb_model = load_pickle(XGB_PATH)
    rf_model = load_pickle(RF_PATH)
    lr_model = load_pickle(LR_PATH)

    p_xgb = xgb_model.predict_proba(X_scaled)[:, 1].astype(np.float32)
    p_rf = rf_model.predict_proba(X_scaled)[:, 1].astype(np.float32)
    p_lr = lr_model.predict_proba(X_scaled)[:, 1].astype(np.float32)

    p_ensemble = (
        p_xgb * ENSEMBLE_WEIGHTS['xgb'] +
        p_rf * ENSEMBLE_WEIGHTS['rf'] +
        p_lr * ENSEMBLE_WEIGHTS['ridge']
    )
    p_ensemble = np.clip(p_ensemble, 1e-7, 1 - 1e-7).astype(np.float32)

    y_true = (y.astype(np.float64) >= 1.0).astype(int).values
    stacked = np.vstack([p_xgb, p_rf, p_lr])
    disagreement_std = stacked.std(axis=0)

    out = pd.DataFrame({
        "y_true": y_true,
        "p_xgb": p_xgb, "p_rf": p_rf, "p_lr": p_lr,
        "p_ensemble": p_ensemble,
        "disagreement_std": disagreement_std,
    })
    out["bucket"] = pd.cut(out["p_ensemble"], bins=BINS, include_lowest=True).astype(str)
    print(f"  n={len(out)}  ROC-AUC={roc_auc_score(y_true, p_ensemble):.4f}  win_rate={y_true.mean():.4f}")
    return out


def search():
    print("=" * 70)
    print(f"SEARCH — tuning per-bucket disagreement gate on VAL split ({TASK})")
    print("=" * 70)
    val = score_split(load_split("val"))

    config = {}
    print("\nPer-bucket grid (val set — this is what selection is based on):")
    for bucket in GATED_BUCKETS:
        g = val[val["bucket"] == bucket]
        if len(g) == 0:
            print(f"\n  {bucket}: no val rows in this bucket — skipping, leaving ungated")
            continue

        print(f"\n  {bucket}  (n={len(g)}, ungated win_rate={g['y_true'].mean():.4f})")
        print(f"    {'cutoff':>8}  {'n':>6}  {'precision':>10}  {'coverage':>9}")
        rows = []
        for cutoff in CANDIDATE_CUTOFFS:
            if cutoff is None:
                mask = pd.Series(True, index=g.index)
                label = "none"
            else:
                mask = g["disagreement_std"] <= cutoff
                label = f"{cutoff:.2f}"
            n = int(mask.sum())
            if n == 0:
                print(f"    {label:>8}  {n:>6}  {'n/a':>10}  {0.0:>8.1%}")
                continue
            precision = g.loc[mask, "y_true"].mean()
            coverage = n / len(g)
            rows.append((cutoff, n, precision, coverage))
            print(f"    {label:>8}  {n:>6}  {precision:>9.1%}  {coverage:>8.1%}")

        # Pick the tightest cutoff (highest precision) that still clears the
        # minimum-signal floor. Ties broken toward the looser (larger n) option.
        qualifying = [r for r in rows if r[1] >= MIN_SIGNALS_FLOOR]
        if not qualifying:
            print(f"    ⚠️  nothing clears the {MIN_SIGNALS_FLOOR}-signal floor — leaving bucket ungated")
            continue
        best = max(qualifying, key=lambda r: (r[2], r[1]))
        config[bucket] = best[0]
        print(f"    → selected cutoff={best[0]}  (n={best[1]}, precision={best[2]:.1%})")

    with open(CONFIG_PATH, "w") as f:
        json.dump({str(k): v for k, v in config.items()}, f, indent=2)
    print(f"\n✅ Wrote {CONFIG_PATH}")
    print("Next: python tune_disagreement_gate.py apply   (checks this config ONCE on test)")


def apply_to_test():
    if not CONFIG_PATH.exists():
        print(f"No {CONFIG_PATH} found — run `search` first.")
        return
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("=" * 70)
    print(f"APPLY — confirming frozen gate config on TEST split ({TASK})")
    print("=" * 70)
    print(f"Config: {config}\n")

    test = score_split(load_split("test"))

    print(f"\n  {'bucket':>14}  {'n_ungated':>10}  {'prec_ungated':>13}  "
          f"{'n_gated':>8}  {'prec_gated':>11}")
    total_ungated_n = total_gated_n = 0
    total_ungated_wins = total_gated_wins = 0
    for bucket in GATED_BUCKETS:
        g = test[test["bucket"] == bucket]
        if len(g) == 0:
            continue
        n_ungated = len(g)
        prec_ungated = g["y_true"].mean()
        cutoff = config.get(bucket)
        if cutoff is None:
            gated = g
        else:
            gated = g[g["disagreement_std"] <= cutoff]
        n_gated = len(gated)
        prec_gated = gated["y_true"].mean() if n_gated > 0 else float("nan")
        print(f"  {bucket:>14}  {n_ungated:>10}  {prec_ungated:>12.1%}  "
              f"{n_gated:>8}  {prec_gated:>10.1%}" if n_gated > 0 else
              f"  {bucket:>14}  {n_ungated:>10}  {prec_ungated:>12.1%}  {n_gated:>8}  {'n/a':>10}")
        total_ungated_n += n_ungated
        total_ungated_wins += g["y_true"].sum()
        total_gated_n += n_gated
        total_gated_wins += gated["y_true"].sum()

    print(f"\n  TOTAL (gated buckets only): ungated n={total_ungated_n} "
          f"precision={total_ungated_wins/total_ungated_n:.1%}" if total_ungated_n else "")
    if total_gated_n:
        print(f"                              gated   n={total_gated_n} "
              f"precision={total_gated_wins/total_gated_n:.1%}")
    print("\nThis is the honest held-out estimate. Do not tune further against test.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "search"
    if mode == "search":
        search()
    elif mode == "apply":
        apply_to_test()
    else:
        print("Usage: python tune_disagreement_gate.py [search|apply]")