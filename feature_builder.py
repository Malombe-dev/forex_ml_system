"""
CANONICAL FEATURE BUILDER  v5
==============================
build_feature_row() is the SINGLE function that produces a feature DataFrame
for one zone candidate. Called identically from label_generator.py,
production_predictor.py, and model_trainer.py.

v5: added 5 new approach-time features — approach_rsi, approach_distance,
    candle_body_ratio, zone_age_bars, approach_momentum.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    N_FEATURES,
    assert_feature_columns_exact,
    get_session,
    get_atr_regime,
)

logger = logging.getLogger(__name__)

RISK_PIPS_FALLBACK_COUNT: int = 0


def build_feature_row(
    zone: Dict,
    df: pd.DataFrame,
    entry: float,
    sl: float,
    encoders: Optional[Dict] = None,
    pip_size: float = 0.0001,
    touch_index: Optional[int] = None,
) -> pd.DataFrame:
    raw = _compute_raw_features(zone, df, entry, sl, pip_size, touch_index)

    encoded_cats: Dict[str, float] = {}
    for col in CATEGORICAL_FEATURES:
        enc_col = f"{col}_encoded"
        val = str(raw[col])
        if encoders is not None:
            if col not in encoders:
                raise ValueError(
                    f"build_feature_row: encoder for '{col}' not found. "
                    f"Available keys: {list(encoders.keys())}"
                )
            le = encoders[col]
            if val not in le.classes_:
                val = le.classes_[0]
            encoded_cats[enc_col] = float(le.transform([val])[0])
        else:
            encoded_cats[enc_col] = 0.0

    row: Dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        row[col] = float(raw[col])
    for col in CATEGORICAL_FEATURES:
        row[f"{col}_encoded"] = encoded_cats[f"{col}_encoded"]

    row_df = pd.DataFrame([row], columns=list(FEATURE_COLUMNS))
    assert_feature_columns_exact(row_df.columns.tolist())
    return row_df


def build_raw_feature_dict(
    zone: Dict,
    df: pd.DataFrame,
    entry: float,
    sl: float,
    pip_size: float = 0.0001,
    touch_index: Optional[int] = None,
) -> Dict:
    return _compute_raw_features(zone, df, entry, sl, pip_size, touch_index)


def _compute_raw_features(
    zone: Dict,
    df: pd.DataFrame,
    entry: float,
    sl: float,
    pip_size: float,
    touch_index: Optional[int],
) -> Dict:
    global RISK_PIPS_FALLBACK_COUNT

    # ── Formation bar ─────────────────────────────────────────────────────────
    zone_index = int(zone.get("index", len(df) - 1))
    zone_index = min(zone_index, len(df) - 1)
    form_bar   = df.iloc[zone_index]

    # ── ATR / EMA / RSI at formation ─────────────────────────────────────────
    atr_val = _safe_atr(form_bar, df)
    ema_20  = _safe_ema(form_bar, df, "ema_20", zone_index)
    ema_50  = _safe_ema(form_bar, df, "ema_50", zone_index)
    rsi_val = _safe_rsi(form_bar)

    # ── Hour / session at formation ───────────────────────────────────────────
    hour    = _extract_hour(form_bar, df, zone_index)
    session = get_session(hour)

    # ── Zone features ─────────────────────────────────────────────────────────
    zone_strength = float(zone.get("strength", 1.0))
    raw_conf = zone.get("confluence", 1)
    if isinstance(raw_conf, dict):
        zone_confluence = int(raw_conf.get("count", raw_conf.get("score", 1)))
    else:
        zone_confluence = int(raw_conf)

    # ── risk_pips ─────────────────────────────────────────────────────────────
    risk_pips_valid = (
        entry and sl
        and entry != sl
        and np.isfinite(entry)
        and np.isfinite(sl)
    )
    if risk_pips_valid:
        risk_pips = abs(entry - sl) / pip_size
    else:
        risk_pips = 0.0

    if not np.isfinite(risk_pips) or risk_pips <= 0:
        fallback_val = min(atr_val / pip_size, 30.0)
        RISK_PIPS_FALLBACK_COUNT += 1
        logger.warning(
            f"risk_pips ATR fallback fired (zone_index={zone_index}) → "
            f"fallback={fallback_val:.1f}p. Total: {RISK_PIPS_FALLBACK_COUNT}"
        )
        risk_pips = fallback_val

    # ── Formation-time distance and trend ─────────────────────────────────────
    close_form          = float(form_bar.get("close", entry))
    distance_from_ema20 = abs(close_form - ema_20) / atr_val if atr_val > 0 else 0.0
    ema_trend           = 1 if ema_20 > ema_50 else 0

    # ── approach_atr_regime ───────────────────────────────────────────────────
    t_idx = min(int(touch_index), len(df) - 1) if touch_index is not None else zone_index
    t_bar = df.iloc[t_idx]
    t_atr = _safe_atr(t_bar, df)
    t_atr_ma = _rolling_atr_mean(df, t_idx, window=50)
    approach_atr_reg = get_atr_regime(t_atr, t_atr_ma)

    # ── zone_touch_count and htf_trend_alignment ──────────────────────────────
    zone_touch_count    = int(zone.get("touch_count", 1))
    htf_trend_alignment = int(zone.get("htf_alignment", 0))

    # ── NEW v5: Approach-time features ───────────────────────────────────────
    # 1. approach_rsi — RSI at the touch bar (oversold BUY / overbought SELL)
    approach_rsi = _safe_rsi(t_bar)

    # 2. approach_distance — how many pips away from zone edge at touch bar
    zone_edge = zone.get("price_high") if zone.get("type") == "demand" else zone.get("price_low")
    if zone_edge is not None and atr_val > 0:
        approach_distance = abs(float(t_bar.get("close", zone_edge)) - zone_edge) / pip_size
        approach_distance = min(approach_distance, 50.0)  # cap at 50 pips
    else:
        approach_distance = 0.0

    # 3. candle_body_ratio — body/range of approach bar (1.0=strong trend, 0=doji)
    t_high = float(t_bar.get("high", 0))
    t_low  = float(t_bar.get("low",  0))
    t_open = float(t_bar.get("open", 0))
    t_close_val = float(t_bar.get("close", 0))
    candle_range = t_high - t_low
    candle_body  = abs(t_close_val - t_open)
    candle_body_ratio = (candle_body / candle_range) if candle_range > 0 else 0.5

    # 4. zone_age_bars — how many bars since zone formed (fresh = lower = better)
    zone_age_bars = float(t_idx - zone_index)
    zone_age_bars = max(0.0, min(zone_age_bars, 500.0))  # cap at 500 bars

    # 5. approach_momentum — (close - close[3 bars ago]) / ATR
    #    Positive = price moving toward zone fast, negative = slowing/reversing
    if t_idx >= 3 and atr_val > 0:
        close_3ago = float(df.iloc[t_idx - 3].get("close", t_close_val))
        approach_momentum = (t_close_val - close_3ago) / atr_val
        approach_momentum = max(-5.0, min(approach_momentum, 5.0))  # cap at ±5 ATR
    else:
        approach_momentum = 0.0

    return {
        # ── Numeric (14) ───────────────────────────────────────────────────
        "zone_strength":        zone_strength,
        "zone_confluence":      float(zone_confluence),
        "risk_pips":            risk_pips,
        "distance_from_ema20":  distance_from_ema20,
        "rsi":                  rsi_val,
        "hour":                 float(hour),
        "ema_trend":            float(ema_trend),
        "zone_touch_count":     float(zone_touch_count),
        "htf_trend_alignment":  float(htf_trend_alignment),
        "zone_source":          1.0 if (
            any(s == "fvg" for s in zone.get("sources", [])) or
            zone.get("source", "") == "fvg" or
            zone.get("type_detail", "") == "fvg"
        ) else 0.0,
        # NEW v5
        "approach_distance":    float(approach_distance),
        "candle_body_ratio":    float(candle_body_ratio),
        "zone_age_bars":        float(zone_age_bars),
        "approach_momentum":    float(approach_momentum),
        # ── Categorical (3) ────────────────────────────────────────────────
        "zone_type":            str(zone.get("type", "demand")),
        "session":              session,
        "approach_atr_regime":  approach_atr_reg,
    }


def _safe_atr(bar: pd.Series, df: pd.DataFrame) -> float:
    val = float(bar.get("atr", np.nan))
    if not np.isfinite(val) or val <= 0:
        val = float(df["atr"].median()) if "atr" in df.columns else 0.0001
    return val


def _safe_ema(bar: pd.Series, df: pd.DataFrame, col: str, idx: int) -> float:
    val = float(bar.get(col, np.nan))
    if not np.isfinite(val) and col in df.columns:
        val = float(df[col].ffill().iloc[idx])
    return val if np.isfinite(val) else float(bar.get("close", 0.0))


def _safe_rsi(bar: pd.Series) -> float:
    val = float(bar.get("rsi", np.nan))
    return val if np.isfinite(val) else 50.0


def _extract_hour(bar: pd.Series, df: pd.DataFrame, zone_index: int) -> int:
    ts = bar.get("time", None)
    if ts is None and "time" in df.columns:
        ts = df["time"].iloc[zone_index]
    if ts is None:
        ts = bar.name
    if ts is not None and pd.notna(ts):
        try:
            return int(pd.to_datetime(ts).hour)
        except Exception:
            pass
    return 12


def _rolling_atr_mean(df: pd.DataFrame, zone_index: int, window: int = 50) -> float:
    if "atr" not in df.columns:
        return 0.0
    series = df["atr"].iloc[: zone_index + 1]
    if len(series) < 2:
        return 0.0
    val = series.rolling(window).mean().iloc[-1]
    return float(val) if np.isfinite(val) else float(series.mean())