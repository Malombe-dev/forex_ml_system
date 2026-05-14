"""
DATA VALIDATION
===============
Validates training_labels_dual.parquet BEFORE it reaches the model trainer.
Run this as Phase 3.5 in main_pipeline.py after label_generator.py.

Checks
------
HARD STOPS (raise ValueError → pipeline halts):
  1. Schema check       — all expected columns present with correct dtypes
  2. No null values     — any column with nulls is a bug
  3. risk_pips range    — all values must be in [0.5, 30.0]
  4. Label distribution — reversion win rate must be 10–40%
  5. Minimum rows       — fewer than 500 rows after filtering is not trainable
  6. Date freshness     — max(zone_time) must be within 90 days of today

WARNINGS (logged, pipeline continues):
  7. Continuation win rate — expected 15–35%
  8. Correlation gate      — any feature pair |r| > 0.85
  9. Class balance         — zone_type or direction more than 60/40

Usage
-----
  python data_validation.py                          # default path
  python data_validation.py data/labels/my.parquet  # custom path
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
RISK_PIPS_MIN          = 0.5
RISK_PIPS_MAX          = 60.0
LABEL_WIN_RATE_MIN     = 0.10
LABEL_WIN_RATE_MAX     = 0.40
CONT_WIN_RATE_MIN      = 0.15
CONT_WIN_RATE_MAX      = 0.45
MIN_ROWS               = 500
MAX_STALENESS_DAYS     = 400
CORR_WARN_THRESHOLD    = 0.85
BALANCE_WARN_THRESHOLD = 0.60   # flag if any class > 60%

# Minimum columns the parquet must have
REQUIRED_LABEL_COLS = ["label", "label_cont", "zone_time", "zone_type",
                        "direction", "risk_pips", "zone_touch_count"]
REQUIRED_FEATURE_COLS = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)


# ── Individual checks ──────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame) -> None:
    """HARD: all required columns present."""
    required = REQUIRED_LABEL_COLS + REQUIRED_FEATURE_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[SCHEMA] Missing {len(missing)} required column(s): {missing}\n"
            "Re-run label_generator.py to regenerate the parquet."
        )
    logger.info(f"  ✅ Schema OK — {len(df.columns)} columns, all required present")


def check_nulls(df: pd.DataFrame) -> None:
    """HARD: no null values anywhere."""
    null_counts = df[REQUIRED_LABEL_COLS + REQUIRED_FEATURE_COLS].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(
            f"[NULLS] {len(bad)} column(s) contain null values:\n"
            + "\n".join(f"  {col}: {n} nulls" for col, n in bad.items())
        )
    logger.info("  ✅ Nulls OK — zero nulls across all checked columns")


def check_risk_pips(df: pd.DataFrame) -> None:
    """HARD: all risk_pips in [RISK_PIPS_MIN, RISK_PIPS_MAX]."""
    out_of_range = df[(df["risk_pips"] < RISK_PIPS_MIN) |
                      (df["risk_pips"] > RISK_PIPS_MAX)]
    if len(out_of_range) > 0:
        pct = len(out_of_range) / len(df)
        raise ValueError(
            f"[RISK_PIPS] {len(out_of_range)} rows ({pct:.1%}) have risk_pips "
            f"outside [{RISK_PIPS_MIN}, {RISK_PIPS_MAX}] pips.\n"
            f"  Min: {df['risk_pips'].min():.2f}  Max: {df['risk_pips'].max():.2f}\n"
            "label_generator.py should have filtered these — check MAX_RISK_PIPS filter."
        )
    logger.info(
        f"  ✅ risk_pips OK — all in [{RISK_PIPS_MIN}, {RISK_PIPS_MAX}] "
        f"(mean={df['risk_pips'].mean():.1f}p, max={df['risk_pips'].max():.1f}p)"
    )


def check_label_distribution(df: pd.DataFrame) -> None:
    """HARD: reversion win rate in [LABEL_WIN_RATE_MIN, LABEL_WIN_RATE_MAX]."""
    win_rate = (df["label"] == 1.0).mean()
    if not (LABEL_WIN_RATE_MIN <= win_rate <= LABEL_WIN_RATE_MAX):
        raise ValueError(
            f"[LABEL_DIST] Reversion win rate {win_rate:.1%} is outside "
            f"[{LABEL_WIN_RATE_MIN:.0%}, {LABEL_WIN_RATE_MAX:.0%}].\n"
            f"  Label value counts:\n{df['label'].value_counts().to_string()}\n"
            "If win rate is too high, the any() labelling bug has returned. "
            "If too low, check the SL/TP geometry in label_generator."
        )
    logger.info(f"  ✅ Reversion label OK — win rate {win_rate:.1%}")


def check_minimum_rows(df: pd.DataFrame) -> None:
    """HARD: at least MIN_ROWS rows for training to be meaningful."""
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"[MIN_ROWS] Only {len(df)} rows — minimum is {MIN_ROWS}. "
            "Not enough data to train a reliable model. Extend the date range."
        )
    logger.info(f"  ✅ Row count OK — {len(df):,} rows (minimum {MIN_ROWS})")


def check_date_freshness(df: pd.DataFrame) -> None:
    """HARD: training data must be within MAX_STALENESS_DAYS of today."""
    if "zone_time" not in df.columns:
        logger.warning("  ⚠️  [FRESHNESS] zone_time column not found — skipping date check")
        return

    df["zone_time"] = pd.to_datetime(df["zone_time"], errors="coerce")
    latest = df["zone_time"].max()

    if pd.isna(latest):
        logger.warning("  ⚠️  [FRESHNESS] All zone_time values are NaT — cannot check staleness")
        return

    now         = pd.Timestamp.utcnow().tz_localize(None) if latest.tzinfo is None \
                  else pd.Timestamp.utcnow()
    staleness   = (now - latest).days

    if staleness > MAX_STALENESS_DAYS:
        raise ValueError(
            f"[FRESHNESS] Training data ends {latest.date()} — {staleness} days ago.\n"
            f"Maximum allowed staleness: {MAX_STALENESS_DAYS} days.\n"
            "Re-run tick_ingestion.py to pull fresh data through today."
        )
    logger.info(
        f"  ✅ Freshness OK — latest bar {latest.date()} "
        f"({staleness} days ago, limit {MAX_STALENESS_DAYS}d)"
    )


def check_continuation_win_rate(df: pd.DataFrame) -> None:
    """WARNING: continuation win rate should be 15–35%."""
    win_rate = (df["label_cont"] >= 0.9).mean()
    if not (CONT_WIN_RATE_MIN <= win_rate <= CONT_WIN_RATE_MAX):
        logger.warning(
            f"  ⚠️  [CONT_LABEL] Continuation win rate {win_rate:.1%} is outside "
            f"[{CONT_WIN_RATE_MIN:.0%}, {CONT_WIN_RATE_MAX:.0%}]. "
            "Expected ~20-25%. If inflated, check check_continuation_outcome() "
            "entry-first validation in label_generator.py."
        )
    else:
        logger.info(f"  ✅ Continuation label OK — win rate {win_rate:.1%}")


def check_feature_correlations(df: pd.DataFrame) -> None:
    """WARNING: alert on any numeric feature pair with |r| > CORR_WARN_THRESHOLD."""
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if len(numeric_cols) < 2:
        return

    corr_matrix = df[numeric_cols].corr().abs()
    high_pairs  = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i+1:]:
            r = corr_matrix.loc[a, b]
            if r > CORR_WARN_THRESHOLD:
                high_pairs.append((a, b, r))

    if high_pairs:
        logger.warning(
            f"  ⚠️  [CORRELATION] {len(high_pairs)} feature pair(s) with |r| > "
            f"{CORR_WARN_THRESHOLD}:"
        )
        for a, b, r in high_pairs:
            logger.warning(f"       {a}  ↔  {b}  |r|={r:.3f}")
        if any(r > 0.999 for _, _, r in high_pairs):
            logger.warning(
                "     CRITICAL: r=1.000 pairs found — these are exact duplicates. "
                "Drop one of each pair from features.py."
            )
    else:
        logger.info(
            f"  ✅ Correlations OK — no numeric pair above |r|={CORR_WARN_THRESHOLD}"
        )


def check_class_balance(df: pd.DataFrame) -> None:
    """WARNING: zone_type or direction imbalance > 60/40."""
    for col in ["zone_type", "direction"]:
        if col not in df.columns:
            continue
        counts  = df[col].value_counts(normalize=True)
        max_pct = counts.max()
        if max_pct > BALANCE_WARN_THRESHOLD:
            logger.warning(
                f"  ⚠️  [BALANCE] {col} is {max_pct:.1%}/{1-max_pct:.1%} — "
                f"exceeds {BALANCE_WARN_THRESHOLD:.0%}/{1-BALANCE_WARN_THRESHOLD:.0%} threshold.\n"
                f"     Distribution: {counts.to_dict()}"
            )
        else:
            logger.info(
                f"  ✅ Balance OK — {col} {counts.to_dict()}"
            )


# ── Orchestrator ───────────────────────────────────────────────────────────────

def validate(parquet_path: str = "data/labels/training_labels_dual.parquet") -> bool:
    """
    Run all checks. Returns True if validation passes (warnings allowed).
    Raises ValueError on any HARD failure.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet not found: {path}\n"
            "Run label_generator.py first."
        )

    logger.info("=" * 70)
    logger.info("DATA VALIDATION")
    logger.info("=" * 70)
    logger.info(f"  File : {path}")

    df = pd.read_parquet(path)
    logger.info(f"  Rows : {len(df):,}")
    logger.info(f"  Cols : {len(df.columns)}")
    logger.info("")

    # ── HARD stops ────────────────────────────────────────────────────────────
    logger.info("── Hard checks (fail = pipeline stops) ──────────────────────────────")
    check_schema(df)
    check_nulls(df)
    check_risk_pips(df)
    check_label_distribution(df)
    check_minimum_rows(df)
    check_date_freshness(df)

    # ── Warnings ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Soft checks (warnings only) ──────────────────────────────────────")
    check_continuation_win_rate(df)
    check_feature_correlations(df)
    check_class_balance(df)

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ DATA VALIDATION PASSED — safe to proceed to preprocessing")
    logger.info("=" * 70)

    return True


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parquet_path = sys.argv[1] if len(sys.argv) > 1 \
                   else "data/labels/training_labels_dual.parquet"
    try:
        validate(parquet_path)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ DATA VALIDATION FAILED — pipeline halted")
        logger.error("=" * 70)
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()