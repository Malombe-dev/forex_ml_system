"""
CANONICAL FEATURE SCHEMA
========================
Single source of truth for ALL feature columns used in training AND inference.

v5 (current) — added 7 new high-signal features:
  approach_rsi         : RSI at the bar price first touches the zone (momentum)
  approach_distance    : pips from zone when first touch occurs (closeness)
  candle_body_ratio    : body/range of the approach bar (pin bar signal)
  volume_ratio         : approach bar volume vs 20-bar avg (confirm interest)
  zone_age_bars        : bars since zone was formed (freshness)
  higher_tf_rsi        : RSI on the 1h timeframe at touch (HTF momentum)
  approach_momentum    : close[touch] - close[touch-3] / ATR (speed of approach)

v4:
  - Removed approach_rsi and approach_ema_dist (r=1.000 duplicates)
  - N_FEATURES: 14 → 12
"""

from typing import Tuple

NUMERIC_FEATURES: Tuple[str, ...] = (
    # ── Zone formation features ───────────────────────────────────────────
    "zone_strength",
    "zone_confluence",
    "risk_pips",

    # ── Formation-time market context ─────────────────────────────────────
    "distance_from_ema20",
    "rsi",
    "hour",
    "ema_trend",

    # ── Zone quality predictors ───────────────────────────────────────────
    "zone_touch_count",
    "htf_trend_alignment",
    "zone_source",

    # ── NEW v5: Approach-time features (most predictive) ─────────────────
    "approach_distance",    # pips away from zone edge at first touch — precision
    "candle_body_ratio",    # body/range at approach bar — pin bar = 1.0, doji = 0.0
    "zone_age_bars",        # bars since zone formed — fresh zones outperform
    "approach_momentum",    # (close - close[3 bars ago]) / ATR — slowing = reversal
)

CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "zone_type",
    "session",
    "approach_atr_regime",
)

FEATURE_COLUMNS: Tuple[str, ...] = NUMERIC_FEATURES + tuple(
    f"{c}_encoded" for c in CATEGORICAL_FEATURES
)

N_FEATURES: int = len(FEATURE_COLUMNS)   # 17


def assert_feature_columns_exact(actual: list, expected: Tuple[str, ...] = FEATURE_COLUMNS) -> None:
    actual_tuple = tuple(actual)
    if actual_tuple != expected:
        missing     = [c for c in expected if c not in actual_tuple]
        extra       = [c for c in actual_tuple if c not in expected]
        wrong_order = actual_tuple != expected and not missing and not extra
        raise ValueError(
            f"Feature column mismatch!\n"
            f"  Expected ({len(expected)}): {list(expected)}\n"
            f"  Actual   ({len(actual)}):   {actual}\n"
            f"  Missing : {missing}\n"
            f"  Extra   : {extra}\n"
            f"  Order wrong: {wrong_order}"
        )


def get_session(hour: int) -> str:
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "ny"
    if 0 <= hour < 8:
        return "asia"
    return "sydney"


def get_atr_regime(atr_val: float, atr_ma: float) -> str:
    if atr_ma <= 0:
        return "normal"
    if atr_val > atr_ma * 1.2:
        return "high"
    if atr_val < atr_ma * 0.8:
        return "low"
    return "normal"