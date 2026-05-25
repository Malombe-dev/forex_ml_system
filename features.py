"""
CANONICAL FEATURE SCHEMA
========================
Single source of truth for ALL feature columns used in training AND inference.

v8 (current):
  REMOVED (always constant in live prediction — miscalibrates model):
    prev_zone_outcome     → always 0.5 live (no persistent zone history DB)
                            XGB importance 11%, RF 14% → significant miscalibration
    time_since_last_touch → always 0.0 live
                            XGB importance 6%, RF 2%

v7:
  REMOVED (replaced by richer 1H-specific features):
    h1_confluence        → replaced by h1_trend_alignment, h1_structure_shift,
                           h1_displacement, h1_liquidity_sweep
  REMOVED (r=1.000 duplicate of h4_fvg_inside):
    zone_source          → identical FVG-origin flag, redundant column dropped

  ADDED 1H alignment features:
    h1_trend_alignment   : int {-1, 0, 1} — 1H EMA20/50 trend direction
                           (-1=bearish, 0=neutral, 1=bullish)
    h1_structure_shift   : int {-1, 0, 1} — 1H BOS/MSS confirmed direction
                           (-1=bearish MSS, 0=none, 1=bullish MSS)
    h1_displacement      : float ≥ 0 — size of last strong 1H institutional
                           move in ATR units (impulse candle / ATR)
    h1_liquidity_sweep   : int {0, 1} — 1H swept swing liquidity before reversal
                           (1 = sweep detected within look-back window)

  ADDED premium/discount features:
    d1_premium_discount  : int {-1, 0, 1} — price position in D1 swing range
                           (-1=discount, 0=equilibrium ±10%, 1=premium)
    h4_premium_discount  : int {-1, 0, 1} — price position in 4H swing range

  ADDED 5M execution features:
    m5_micro_bos         : int {0, 1} — micro break of structure on 5M before entry
    m5_sweep_confirmation: int {0, 1} — micro liquidity grab on 5M before entry
    m5_entry_candle_quality : float [0, 1] — composite score:
                           engulfing + displacement + strong close + rejection wick

v6:
  h1_confluence, h4_fvg_inside, d1_ema_distance,
  prev_zone_outcome, time_since_last_touch added.

v5:
  approach_distance, candle_body_ratio, zone_age_bars, approach_momentum added.

v4:
  Removed approach_rsi and approach_ema_dist (r=1.000 duplicates).
"""

from typing import Tuple

NUMERIC_FEATURES: Tuple[str, ...] = (
    # ── Zone formation features ───────────────────────────────────────────────
    "zone_strength",
    "zone_confluence",
    "risk_pips",

    # ── Formation-time market context ─────────────────────────────────────────
    "distance_from_ema20",
    "rsi",
    "hour",
    "ema_trend",

    # ── Zone quality predictors ───────────────────────────────────────────────
    "zone_touch_count",
    "htf_trend_alignment",

    # ── v5: Approach-time features ────────────────────────────────────────────
    "approach_distance",
    "candle_body_ratio",
    "zone_age_bars",
    "approach_momentum",

    # ── v6: Zone-memory + MTF-edge features ───────────────────────────────────
    # NOTE: h1_confluence REMOVED — replaced by 4 granular 1H features below
    # NOTE: prev_zone_outcome REMOVED v8 — always 0.5 in live (no persistent
    #       zone history DB); XGB importance 11%, RF 14% → miscalibrates live scores
    # NOTE: time_since_last_touch REMOVED v8 — always 0.0 in live; XGB 6%, RF 2%
    "h4_fvg_inside",
    "d1_ema_distance",

    # ── v7: 1H alignment features (replaces generic h1_confluence) ────────────
    "h1_trend_alignment",       # -1=bearish, 0=neutral, 1=bullish  (1H EMA trend)
    "h1_structure_shift",       # -1=bearish MSS, 0=none, 1=bullish MSS
    "h1_displacement",          # impulse candle size / ATR on 1H (≥0)
    "h1_liquidity_sweep",       # 1=1H swept swing liquidity before reversal

    # ── v7: Premium / Discount features ──────────────────────────────────────
    "d1_premium_discount",      # -1=discount, 0=equilibrium, 1=premium  (D1 range)
    "h4_premium_discount",      # -1=discount, 0=equilibrium, 1=premium  (4H range)

    # ── v7: 5M execution features ────────────────────────────────────────────
    "m5_micro_bos",             # 1=micro BOS present on 5M
    "m5_sweep_confirmation",    # 1=micro sweep before entry on 5M
    "m5_entry_candle_quality",  # composite 0–1 score
)

CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "zone_type",
    "session",
    "approach_atr_regime",
)

FEATURE_COLUMNS: Tuple[str, ...] = NUMERIC_FEATURES + tuple(
    f"{c}_encoded" for c in CATEGORICAL_FEATURES
)

N_FEATURES: int = len(FEATURE_COLUMNS)   # 27 numeric + 3 categorical-encoded = 27 total


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