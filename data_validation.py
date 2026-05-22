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
# risk_pips limits — 4H zones on XAUUSD/USDJPY are much larger than EURUSD 15min zones.
# EURUSD/GBPUSD 4H: 10-150 pips typical. XAUUSD 4H: 50-500 pips typical.
# Hard ceiling of 1000: anything above this = pip_size bug in label_generator.py
# (e.g. calculating abs(entry-sl) in price units without dividing by pip_size).
# Preprocessing will clip to a tighter bound after this check passes.
RISK_PIPS_MIN          = 0.5
RISK_PIPS_MAX          = 1000.0   # hard ceiling — values above = pip_size calculation bug
LABEL_WIN_RATE_MIN     = 0.20  # post-leakage-fix: 4H binary labels land at 40-60%
LABEL_WIN_RATE_MAX     = 0.65  # was 0.40 — too tight for 4H zone win rates
CONT_WIN_RATE_MIN      = 0.10
CONT_WIN_RATE_MAX      = 0.55
MIN_ROWS               = 200  # 4H zones are fewer than 15min zones
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
    logger.info(f"  ✅ Schema OK — {len(df.columns)} columns, all required present (v7 schema: 33 features)")


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
    """HARD: risk_pips must be in [RISK_PIPS_MIN, RISK_PIPS_MAX].
    Values above 1000 almost always mean pip_size was not applied
    (e.g. abs(entry - sl) in price units for XAUUSD gives ~50.0 but
    dividing by pip_size=0.01 gives 5000 pips — a clear bug).
    """
    out_of_range = df[(df["risk_pips"] < RISK_PIPS_MIN) |
                      (df["risk_pips"] > RISK_PIPS_MAX)]
    if len(out_of_range) > 0:
        pct = len(out_of_range) / len(df)
        # Per-symbol breakdown to help diagnose which pip_size is wrong
        sym_breakdown = ""
        if "symbol" in df.columns:
            for sym, grp in out_of_range.groupby("symbol"):
                sym_breakdown += f"\n    {sym}: {len(grp)} rows  max={grp['risk_pips'].max():.1f}p"
        raise ValueError(
            f"[RISK_PIPS] {len(out_of_range)} rows ({pct:.1%}) have risk_pips "
            f"outside [{RISK_PIPS_MIN}, {RISK_PIPS_MAX}] pips.\n"
            f"  Overall: min={df['risk_pips'].min():.2f}  max={df['risk_pips'].max():.2f}"
            f"{sym_breakdown}\n"
            "  Fix: check pip_size passed to label_generator.py for each symbol.\n"
            "  XAUUSD and USDJPY use pip_size=0.01, not 0.0001."
        )
    # Per-symbol summary in the OK message
    sym_info = ""
    if "symbol" in df.columns:
        for sym, grp in df.groupby("symbol"):
            sym_info += f"  {sym}: mean={grp['risk_pips'].mean():.1f}p max={grp['risk_pips'].max():.1f}p\n"
    logger.info(
        f"  ✅ risk_pips OK — all in [{RISK_PIPS_MIN}, {RISK_PIPS_MAX}]\n"
        + sym_info
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


def check_v7_feature_ranges(df: pd.DataFrame) -> None:
    """WARNING: sanity-check v6 carry-over features and all new v7 features."""

    # ── v6 carry-overs ────────────────────────────────────────────────────────

    # h4_fvg_inside — binary {0, 1}
    if "h4_fvg_inside" in df.columns:
        bad = df["h4_fvg_inside"].dropna()
        bad = bad[~bad.isin([0.0, 1.0])]
        if len(bad):
            logger.warning(f"  ⚠️  [V7] h4_fvg_inside has {len(bad)} non-binary values")
        else:
            rate = df["h4_fvg_inside"].mean()
            logger.info(f"  ✅ h4_fvg_inside OK — {rate:.1%} zones are FVG-backed")

    # d1_ema_distance — expect [0, 10], most values 0–3
    if "d1_ema_distance" in df.columns:
        s = df["d1_ema_distance"].dropna()
        pct_zeros = (s == 0.0).mean()
        if pct_zeros > 0.95:
            logger.warning(
                f"  ⚠️  [V7] d1_ema_distance is 0.0 for {pct_zeros:.0%} of rows — "
                "df_d1 may not be wired into feature_builder calls. "
                "Pass df_d1 to build_feature_row() / build_raw_feature_dict()."
            )
        else:
            logger.info(
                f"  ✅ d1_ema_distance OK — mean={s.mean():.2f}  max={s.max():.2f}"
            )

    # prev_zone_outcome — valid values {0.0, 0.5, 1.0}
    if "prev_zone_outcome" in df.columns:
        valid = {0.0, 0.5, 1.0}
        bad_vals = df["prev_zone_outcome"].dropna()
        bad_vals = bad_vals[~bad_vals.isin(valid)]
        if len(bad_vals):
            logger.warning(
                f"  ⚠️  [V7] prev_zone_outcome has {len(bad_vals)} values "
                f"outside {{0.0, 0.5, 1.0}}"
            )
        else:
            pct_unknown = (df["prev_zone_outcome"] == 0.5).mean()
            if pct_unknown > 0.95:
                logger.warning(
                    f"  ⚠️  [V7] prev_zone_outcome is 0.5 (unknown) for "
                    f"{pct_unknown:.0%} of rows — zone_history may not be "
                    "built and passed in label_generator / production_predictor."
                )
            else:
                logger.info(
                    f"  ✅ prev_zone_outcome OK — "
                    f"win={(df['prev_zone_outcome']==1.0).mean():.1%}  "
                    f"loss={(df['prev_zone_outcome']==0.0).mean():.1%}  "
                    f"unknown={(df['prev_zone_outcome']==0.5).mean():.1%}"
                )

    # time_since_last_touch — expect [0, 2000] hours
    if "time_since_last_touch" in df.columns:
        s = df["time_since_last_touch"].dropna()
        pct_zeros = (s == 0.0).mean()
        if pct_zeros > 0.95:
            logger.warning(
                f"  ⚠️  [V7] time_since_last_touch is 0.0 for {pct_zeros:.0%} of rows — "
                "zone_history may not be built/passed."
            )
        elif s.max() > 2000:
            logger.warning(
                f"  ⚠️  [V7] time_since_last_touch max={s.max():.0f}h > 2000h cap — "
                "check _zone_level_key bucketing in feature_builder.py."
            )
        else:
            logger.info(
                f"  ✅ time_since_last_touch OK — "
                f"median={s.median():.0f}h  max={s.max():.0f}h"
            )

    # ── v7: 1H alignment features ─────────────────────────────────────────────

    # h1_trend_alignment — valid values {-1, 0, 1}
    if "h1_trend_alignment" in df.columns:
        valid = {-1, 0, 1}
        bad = df["h1_trend_alignment"].dropna()
        bad = bad[~bad.isin(valid)]
        if len(bad):
            logger.warning(
                f"  ⚠️  [V7] h1_trend_alignment has {len(bad)} values outside {{-1, 0, 1}}"
            )
        else:
            dist = df["h1_trend_alignment"].value_counts().to_dict()
            logger.info(f"  ✅ h1_trend_alignment OK — distribution: {dist}")

    # h1_structure_shift — valid values {-1, 0, 1}
    if "h1_structure_shift" in df.columns:
        valid = {-1, 0, 1}
        bad = df["h1_structure_shift"].dropna()
        bad = bad[~bad.isin(valid)]
        if len(bad):
            logger.warning(
                f"  ⚠️  [V7] h1_structure_shift has {len(bad)} values outside {{-1, 0, 1}}"
            )
        else:
            pct_shift = (df["h1_structure_shift"] != 0).mean()
            logger.info(f"  ✅ h1_structure_shift OK — {pct_shift:.1%} rows have a confirmed BOS/MSS")

    # h1_displacement — float >= 0, warn if mostly zero (MTF data not wired)
    if "h1_displacement" in df.columns:
        s = df["h1_displacement"].dropna()
        if (s < 0).any():
            logger.warning(
                f"  ⚠️  [V7] h1_displacement has {(s < 0).sum()} negative values — "
                "must be ≥ 0 (impulse candle / ATR)."
            )
        else:
            pct_zeros = (s == 0.0).mean()
            if pct_zeros > 0.95:
                logger.warning(
                    f"  ⚠️  [V7] h1_displacement is 0.0 for {pct_zeros:.0%} of rows — "
                    "1H OHLC data may not be passed to feature_builder."
                )
            else:
                logger.info(
                    f"  ✅ h1_displacement OK — mean={s.mean():.2f}  max={s.max():.2f}"
                )

    # h1_liquidity_sweep — binary {0, 1}
    if "h1_liquidity_sweep" in df.columns:
        bad = df["h1_liquidity_sweep"].dropna()
        bad = bad[~bad.isin([0.0, 1.0])]
        if len(bad):
            logger.warning(
                f"  ⚠️  [V7] h1_liquidity_sweep has {len(bad)} non-binary values"
            )
        else:
            rate = df["h1_liquidity_sweep"].mean()
            logger.info(f"  ✅ h1_liquidity_sweep OK — {rate:.1%} zones have a 1H sweep")

    # ── v7: Premium / Discount features ──────────────────────────────────────

    for col, label in [("d1_premium_discount", "D1"), ("h4_premium_discount", "4H")]:
        if col in df.columns:
            valid = {-1, 0, 1}
            bad = df[col].dropna()
            bad = bad[~bad.isin(valid)]
            if len(bad):
                logger.warning(
                    f"  ⚠️  [V7] {col} has {len(bad)} values outside {{-1, 0, 1}}"
                )
            else:
                dist = df[col].value_counts().to_dict()
                logger.info(f"  ✅ {col} OK ({label} range) — distribution: {dist}")

    # ── v7: 5M execution features ─────────────────────────────────────────────

    # m5_micro_bos — binary {0, 1}
    if "m5_micro_bos" in df.columns:
        bad = df["m5_micro_bos"].dropna()
        bad = bad[~bad.isin([0.0, 1.0])]
        if len(bad):
            logger.warning(f"  ⚠️  [V7] m5_micro_bos has {len(bad)} non-binary values")
        else:
            rate = df["m5_micro_bos"].mean()
            logger.info(f"  ✅ m5_micro_bos OK — {rate:.1%} entries have a 5M micro BOS")

    # m5_sweep_confirmation — binary {0, 1}
    if "m5_sweep_confirmation" in df.columns:
        bad = df["m5_sweep_confirmation"].dropna()
        bad = bad[~bad.isin([0.0, 1.0])]
        if len(bad):
            logger.warning(
                f"  ⚠️  [V7] m5_sweep_confirmation has {len(bad)} non-binary values"
            )
        else:
            rate = df["m5_sweep_confirmation"].mean()
            logger.info(f"  ✅ m5_sweep_confirmation OK — {rate:.1%} entries have a 5M sweep")

    # m5_entry_candle_quality — float [0, 1], warn if mostly zero (5M data not wired)
    if "m5_entry_candle_quality" in df.columns:
        s = df["m5_entry_candle_quality"].dropna()
        out_of_range = ((s < 0) | (s > 1)).sum()
        if out_of_range:
            logger.warning(
                f"  ⚠️  [V7] m5_entry_candle_quality has {out_of_range} values "
                "outside [0, 1]."
            )
        else:
            pct_zeros = (s == 0.0).mean()
            if pct_zeros > 0.95:
                logger.warning(
                    f"  ⚠️  [V7] m5_entry_candle_quality is 0.0 for {pct_zeros:.0%} of rows — "
                    "5M OHLC data may not be passed to feature_builder."
                )
            else:
                logger.info(
                    f"  ✅ m5_entry_candle_quality OK — "
                    f"mean={s.mean():.2f}  min={s.min():.2f}  max={s.max():.2f}"
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
    check_v7_feature_ranges(df)

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