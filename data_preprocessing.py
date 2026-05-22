"""
DATA PREPROCESSING
==================
Cleans training_labels_dual.parquet and outputs training_labels_clean.parquet.
Run this as Phase 3.6 in main_pipeline.py — AFTER data_validation.py.

Operations (in order)
---------------------
1. Clip outliers
     risk_pips        → clamp to [0.5, 30.0]
     zone_strength    → clamp at 99th percentile
     distance_from_ema20 → clamp at 99th percentile
2. Drop duplicate features (already removed in features.py v4, but guard anyway)
     approach_rsi, approach_ema_dist — if somehow still present, drop them
3. Filter stale zones
     zone_touch_count > MAX_TOUCH_COUNT (default 5) → drop row
4. Encode categoricals consistently
     LabelEncoder fitted on TRAINING rows only (zone_time chronological split)
     Saves encoders to models/encoders_preprocessing.pkl for inference use
5. Fit and save StandardScaler on training rows
     Saves to models/scaler_preprocessing.pkl
6. Save clean parquet
     data/labels/training_labels_clean.parquet

The clean parquet keeps ALL original columns — it is a strict superset of
the raw parquet. model_trainer.py reads it directly via load_with_outcomes().

Usage
-----
  python data_preprocessing.py                           # default paths
  python data_preprocessing.py --input  data/labels/training_labels_dual.parquet
                                --output data/labels/training_labels_clean.parquet
"""

import sys
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder

from features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    N_FEATURES,
    assert_feature_columns_exact,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
RISK_PIPS_MIN   = 0.5
RISK_PIPS_MAX   = 500.0   # 4H XAUUSD zones can have 200-500 pip SL; old 30.0 clipped everything
PERCENTILE_CLIP = 99          # clip zone_strength and distance_from_ema20 here
MAX_TOUCH_COUNT = 20           # drop zones visited more than this many times
# Train boundary — must match model_trainer.py TRAIN_END_DATE
# Import it directly so the two files always agree
try:
    from model_trainer import TRAIN_END_DATE as _TRAIN_END_DATE
except ImportError:
    import pandas as _pd
    _TRAIN_END_DATE = _pd.Timestamp("2022-08-04")  # fallback matches model_trainer default

# Features to drop if somehow present (removed in features.py v4 as r=1.000 duplicates)
DUPLICATE_FEATURES = ["approach_ema_dist", "approach_rsi"]

# Paths
DEFAULT_INPUT  = "data/labels/training_labels_dual.parquet"
DEFAULT_OUTPUT = "data/labels/training_labels_clean.parquet"
MODELS_DIR     = Path("models")


# ── Step 1: clip outliers ──────────────────────────────────────────────────────

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 1 — Clipping outliers")
    df = df.copy()

    # risk_pips: hard clip to [RISK_PIPS_MIN, RISK_PIPS_MAX]
    before_rp = df["risk_pips"].describe()
    df["risk_pips"] = df["risk_pips"].clip(lower=RISK_PIPS_MIN, upper=RISK_PIPS_MAX)
    after_rp  = df["risk_pips"].describe()
    logger.info(
        f"  risk_pips     : max {before_rp['max']:.1f}p → {after_rp['max']:.1f}p "
        f"(clipped to [{RISK_PIPS_MIN}, {RISK_PIPS_MAX}])"
    )

    # zone_strength: clip at 99th percentile
    if "zone_strength" in df.columns:
        cap = float(np.percentile(df["zone_strength"].dropna(), PERCENTILE_CLIP))
        n_clipped = (df["zone_strength"] > cap).sum()
        df["zone_strength"] = df["zone_strength"].clip(upper=cap)
        logger.info(
            f"  zone_strength : 99th pct cap = {cap:.2f} "
            f"({n_clipped} rows clipped)"
        )

    # distance_from_ema20: clip at 99th percentile
    if "distance_from_ema20" in df.columns:
        cap = float(np.percentile(df["distance_from_ema20"].dropna(), PERCENTILE_CLIP))
        n_clipped = (df["distance_from_ema20"] > cap).sum()
        df["distance_from_ema20"] = df["distance_from_ema20"].clip(upper=cap)
        logger.info(
            f"  dist_ema20    : 99th pct cap = {cap:.4f} "
            f"({n_clipped} rows clipped)"
        )

    return df


# ── Step 2: drop known duplicate features ─────────────────────────────────────

def drop_duplicate_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 2 — Dropping r=1.000 duplicate features (if present)")
    dropped = []
    for col in DUPLICATE_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)
    if dropped:
        logger.warning(
            f"  Dropped {dropped} — these should have been removed in features.py v4. "
            "Update features.py so they never appear."
        )
    else:
        logger.info("  OK — duplicate features not present (features.py v4+ schema)")
    return df


# ── Step 3: filter stale zones ────────────────────────────────────────────────

def filter_stale_zones(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 3 — Filtering stale zones (zone_touch_count > MAX_TOUCH_COUNT)")
    n_before  = len(df)
    stale     = df["zone_touch_count"] > MAX_TOUCH_COUNT
    n_dropped = stale.sum()
    df        = df[~stale].reset_index(drop=True)
    logger.info(
        f"  Dropped {n_dropped}/{n_before} rows with touch_count > {MAX_TOUCH_COUNT} "
        f"({n_dropped/max(n_before,1):.1%})"
    )
    logger.info(f"  Remaining: {len(df):,} rows")
    return df


# ── Step 4: encode categoricals (fit on train split only) ─────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    train_end: int,
    save_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode each categorical feature. Encoders are fitted on the
    first `train_end` rows (chronological order) to avoid data leakage.

    Unknown values at inference time are mapped to the first class (index 0)
    rather than raising. This mirrors production_predictor.py behaviour.
    """
    logger.info("Step 4 — Encoding categoricals (fit on training rows only)")
    df       = df.copy()
    encoders = {}

    train_df = df.iloc[:train_end]

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            logger.warning(f"  ⚠️  {col} not found in parquet — skipping encoder")
            continue

        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        encoders[col] = le

        encoded = []
        for val in df[col].astype(str):
            if val in le.classes_:
                encoded.append(int(le.transform([val])[0]))
            else:
                logger.warning(
                    f"  ⚠️  Unknown value '{val}' in {col} — mapped to class 0"
                )
                encoded.append(0)

        df[f"{col}_encoded"] = encoded
        logger.info(
            f"  {col:25s} classes={list(le.classes_)}  "
            f"(fitted on {len(train_df)} train rows)"
        )

    # Save encoders for use at inference time
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(encoders, f, protocol=4)
    logger.info(f"  Saved encoders → {save_path}")

    return df, encoders


# ── Step 5: fit and save StandardScaler ───────────────────────────────────────

def fit_scaler(
    df: pd.DataFrame,
    train_end: int,
    save_path: Path,
) -> StandardScaler:
    """
    Fit StandardScaler on training rows only. Validates that the final
    FEATURE_COLUMNS are all present before fitting.
    """
    logger.info("Step 5 — Fitting StandardScaler on training rows")

    # Check all encoded feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot fit scaler — {len(missing)} FEATURE_COLUMNS missing: {missing}\n"
            "Make sure encode_categoricals() ran first."
        )

    assert_feature_columns_exact(
        [c for c in FEATURE_COLUMNS if c in df.columns]
    )

    X_train = df.iloc[:train_end][list(FEATURE_COLUMNS)]
    scaler  = StandardScaler()
    scaler.fit(X_train)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f, protocol=4)

    logger.info(
        f"  Scaler fitted on {len(X_train)} training rows "
        f"({N_FEATURES} features) → saved to {save_path}"
    )
    return scaler


# ── Step 6: save clean parquet ────────────────────────────────────────────────

def save_clean_parquet(df: pd.DataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy", index=False)
    logger.info(f"  Saved {len(df):,} rows → {out}")


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, train_end: int) -> None:
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"  Total rows  : {len(df):,}")
    print(f"  Train rows  : {train_end:,}  ({train_end/len(df):.0%})")
    print(f"  Val+Test    : {len(df) - train_end:,}")
    print(f"  Features    : {N_FEATURES} (canonical order)")

    print("\n📊 Label Statistics (post-clean):")
    rev_win  = (df["label"] == 1.0).mean()
    cont_win = (df["label_cont"] >= 0.9).mean()
    print(f"  Reversion win rate    : {rev_win:.1%}")
    print(f"  Continuation win rate : {cont_win:.1%}")

    print("\n📊 Feature ranges (post-clip):")
    for col in ["risk_pips", "zone_strength", "distance_from_ema20",
                "rsi", "zone_touch_count"]:
        if col in df.columns:
            s = df[col]
            print(f"  {col:25s} min={s.min():.2f}  mean={s.mean():.2f}  max={s.max():.2f}")

    print("=" * 70 + "\n")


# ── Main orchestrator ──────────────────────────────────────────────────────────

def preprocess(
    input_path:  str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Run all preprocessing steps in order. Returns the cleaned DataFrame.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {src}\n"
            "Run label_generator.py (and data_validation.py) first."
        )

    logger.info("=" * 70)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 70)
    logger.info(f"  Input  : {src}")
    logger.info(f"  Output : {output_path}")
    logger.info("")

    # Sort chronologically so train/val split is time-correct
    df = pd.read_parquet(src)
    if "zone_time" in df.columns:
        df["zone_time"] = pd.to_datetime(df["zone_time"], errors="coerce")
        df = df.sort_values("zone_time").reset_index(drop=True)
        logger.info(f"  Sorted by zone_time. Range: {df['zone_time'].min().date()} → "
                    f"{df['zone_time'].max().date()}")

    logger.info(f"  Rows loaded: {len(df):,}")
    logger.info("")

    df        = clip_outliers(df)
    logger.info("")
    df        = drop_duplicate_features(df)
    logger.info("")
    df        = filter_stale_zones(df)
    logger.info("")

    # Use the same date boundary as model_trainer.py — row-fraction splits
    # produce different boundaries each time data grows, causing scaler/encoder
    # mismatch between preprocessing and training.
    train_end = int((df["zone_time"] < _TRAIN_END_DATE).sum()) if "zone_time" in df.columns                 else int(len(df) * 0.70)
    logger.info(
        f"  Training split boundary: row {train_end} "
        f"(zone_time < {_TRAIN_END_DATE.date()}  — matches model_trainer.py)"
    )
    logger.info("")

    df, _ = encode_categoricals(
        df,
        train_end  = train_end,
        save_path  = MODELS_DIR / "encoders_preprocessing.pkl",
    )
    logger.info("")

    fit_scaler(
        df,
        train_end  = train_end,
        save_path  = MODELS_DIR / "scaler_preprocessing.pkl",
    )
    logger.info("")

    logger.info("Step 6 — Saving clean parquet")
    save_clean_parquet(df, output_path)

    print_summary(df, train_end)

    logger.info("✅ Preprocessing complete")
    logger.info(f"   Next: python model_trainer.py")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocessing for Forex ML pipeline")
    parser.add_argument(
        "--input",  default=DEFAULT_INPUT,
        help=f"Input parquet (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output parquet (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    try:
        preprocess(input_path=args.input, output_path=args.output)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ PREPROCESSING FAILED")
        logger.error("=" * 70)
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()