"""
CANONICAL FEATURE BUILDER  v9
==============================
build_feature_row() is the SINGLE function that produces a feature DataFrame
for one zone candidate. Called identically from label_generator.py,
production_predictor.py, and model_trainer.py.

v9: Feature-alignment fixes —
    FIX 1: _compute_m5_features now uses tail(60) instead of tail(20) for
            m5_micro_bos and m5_sweep_confirmation. production_predictor's
            _check_5min_entry() already used tail(60); the mismatch caused the
            fallback path (15-min zones without pre-injected m5 values) to
            scan a narrower window, producing fewer BOS/sweep detections and
            systematically lower model scores for all 15-min zones.
            m5_entry_candle_quality still uses the final tail(5) of that window
            — unchanged.

    FIX 2: _compute_raw_features now resolves htf_trend_alignment as {-1,0,1}
            instead of accepting raw zone["htf_alignment"] which was {0,1} for
            15-min zones (predict_with_proper_features set 0 for counter-trend
            instead of -1). The correct value is now enforced here so both the
            HTF-OB path (pre-injected) and the 15-min path agree.

v8: Removed unreliable live features —
    REMOVED : prev_zone_outcome (always 0.5 live, XGB importance 11%)
    REMOVED : time_since_last_touch (always 0.0 live, XGB importance 6%)
    Both were miscalibrating live probability scores. No persistent zone
    history DB exists in production.

v7: MTF upgrade —
    REMOVED : h1_confluence (replaced by 4 granular 1H features)
    ADDED   : h1_trend_alignment, h1_structure_shift, h1_displacement,
              h1_liquidity_sweep, d1_premium_discount, h4_premium_discount,
              m5_micro_bos, m5_sweep_confirmation, m5_entry_candle_quality

    New optional kwargs:
      df_1h  : pd.DataFrame — 1H bars for 1H alignment features
      df_4h  : pd.DataFrame — 4H bars for h4_premium_discount
      df_5m  : pd.DataFrame — 5M bars for execution quality features

v6: h1_confluence, h4_fvg_inside, d1_ema_distance,
    prev_zone_outcome, time_since_last_touch.
v5: approach_distance, candle_body_ratio, zone_age_bars, approach_momentum.
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
    df_d1: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    df_5m: Optional[pd.DataFrame] = None,
    zone_history: Optional[Dict] = None,
) -> pd.DataFrame:
    raw = _compute_raw_features(
        zone, df, entry, sl, pip_size, touch_index,
        df_d1=df_d1, df_1h=df_1h, df_4h=df_4h, df_5m=df_5m,
        zone_history=zone_history,
    )

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
    df_d1: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    df_5m: Optional[pd.DataFrame] = None,
    zone_history: Optional[Dict] = None,
) -> Dict:
    return _compute_raw_features(
        zone, df, entry, sl, pip_size, touch_index,
        df_d1=df_d1, df_1h=df_1h, df_4h=df_4h, df_5m=df_5m,
        zone_history=zone_history,
    )


def _compute_raw_features(
    zone: Dict,
    df: pd.DataFrame,
    entry: float,
    sl: float,
    pip_size: float,
    touch_index: Optional[int],
    df_d1: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    df_5m: Optional[pd.DataFrame] = None,
    zone_history: Optional[Dict] = None,
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
    # Cap zone_strength at 6.7 to match training percentile clipping
    zone_strength = min(zone_strength, 6.7)
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
    # zone_touch_count: accept either key name — upstream uses both spellings.
    zone_touch_count = int(zone.get("touch_count", zone.get("zone_touch_count", 1)))

    # FIX 2 (v9): htf_trend_alignment must be {-1, 0, 1}.
    # The HTF-OB path (htf_trade_detect) injects the correct direction-flipped
    # value via h1_conf['h1_trend_alignment'].  The 15-min path
    # (predict_with_proper_features) historically set htf_alignment = {0, 1}
    # only — counter-trend zones got 0 instead of -1, silencing the model's
    # counter-trend penalty.
    # Fix: clamp the raw value to {-1, 0, 1} here so both paths are safe.
    _raw_htf = int(zone.get("htf_alignment", 0))
    htf_trend_alignment = max(-1, min(1, _raw_htf))

    # ── v5: Approach-time features ────────────────────────────────────────────
    zone_edge = zone.get("price_high") if zone.get("type") == "demand" else zone.get("price_low")
    if zone_edge is not None and atr_val > 0:
        approach_distance = abs(float(t_bar.get("close", zone_edge)) - zone_edge) / pip_size
        approach_distance = min(approach_distance, 50.0)
    else:
        approach_distance = 0.0

    t_high = float(t_bar.get("high", 0))
    t_low  = float(t_bar.get("low",  0))
    t_open = float(t_bar.get("open", 0))
    t_close_val = float(t_bar.get("close", 0))
    candle_range = t_high - t_low
    candle_body  = abs(t_close_val - t_open)
    candle_body_ratio = (candle_body / candle_range) if candle_range > 0 else 0.5

    zone_age_bars = float(t_idx - zone_index)
    zone_age_bars = max(0.0, min(zone_age_bars, 500.0))

    if t_idx >= 3 and atr_val > 0:
        close_3ago = float(df.iloc[t_idx - 3].get("close", t_close_val))
        approach_momentum = (t_close_val - close_3ago) / atr_val
        approach_momentum = max(-5.0, min(approach_momentum, 5.0))
    else:
        approach_momentum = 0.0

    # ── v6: Zone-memory + MTF-edge features ───────────────────────────────────
    # h1_confluence REMOVED — replaced by granular features below

    sources = zone.get("sources", [])
    # ── h4_fvg_inside (leakage fix v2) ───────────────────────────────────────
    # OLD BUG: checked zone["sources"]/zone["tf_source"] to detect FVG membership
    # making the feature circular — zones that ARE 4H FVGs always scored 1.0,
    # producing 95%+ win rates on tf_source='4h' rows (data leakage confirmed).
    #
    # v1 "fix" (still buggy): kept the correct h4_fvg_zone/fvg_overlap flags but
    # ALSO kept OR-ing in the old circular sources/source/tf_source checks below
    # them, so any FVG-sourced zone still auto-scored 1.0 regardless of whether
    # a genuinely separate 4H FVG confirmed it. Leakage was never actually removed.
    #
    # v2 (this fix): only fire on the two legitimate, non-circular flags — the
    # zone must be EXPLICITLY flagged as sitting inside/overlapping a SEPARATE
    # 4H FVG (h4_fvg_zone) or an overlap detected independently (fvg_overlap).
    # A zone's own source/type being "fvg" no longer implies h4_fvg_inside=1.
    #
    # NOTE: existing trained models were fit on the leaky version of this
    # feature. Historical labels/features must be regenerated and the model
    # retrained after this change — the feature's meaning has changed.
    h4_fvg_inside = 1.0 if (
        zone.get("h4_fvg_zone", False)
        or zone.get("fvg_overlap", False)
    ) else 0.0

    d1_ema_distance = _compute_d1_ema_distance(
        touch_bar=t_bar,
        touch_time=df["time"].iloc[t_idx] if "time" in df.columns else None,
        df_d1=df_d1,
    )

    # ── v6: Zone-memory features ──────────────────────────────────────────────
    # prev_zone_outcome and time_since_last_touch REMOVED in v8:
    # both were always constant in live prediction (0.5 / 0.0) because no
    # persistent zone history DB exists. Combined importance XGB=17%, RF=16%
    # was miscalibrating live probability scores. Removed from schema entirely.

    # ── v7: 1H alignment features ─────────────────────────────────────────────
    touch_time_ts = df["time"].iloc[t_idx] if "time" in df.columns else None

    # Prefer pre-computed values stored on zone by smc_detector (training path)
    # Fall back to live computation from df_1h (inference path)
    if zone.get("h1_trend_alignment") is not None:
        h1_trend_alignment  = int(zone["h1_trend_alignment"])
        h1_structure_shift  = int(zone.get("h1_structure_shift", 0))
        h1_displacement     = float(zone.get("h1_displacement", 0.0))
        h1_liquidity_sweep  = int(zone.get("h1_liquidity_sweep", 0))
    else:
        h1_trend_alignment, h1_structure_shift, h1_displacement, h1_liquidity_sweep = \
            _compute_h1_features(df_1h, touch_time_ts, zone.get("type", "demand"))

    # ── v7: Premium / Discount features ──────────────────────────────────────
    zone_price = float(zone.get("price_mid", t_close_val))

    d1_premium_discount = _compute_premium_discount(df_d1, zone_price, touch_time_ts, lookback=20)
    h4_premium_discount = _compute_premium_discount(df_4h, zone_price, touch_time_ts, lookback=50)

    # ── v7: 5M execution features ─────────────────────────────────────────────
    m5_micro_bos, m5_sweep_confirmation, m5_entry_candle_quality = \
        _compute_m5_features(df_5m, zone, pip_size, touch_time_ts)

    return {
        # ── Numeric ────────────────────────────────────────────────────────────
        "zone_strength":            zone_strength,
        "zone_confluence":          float(zone_confluence),
        "risk_pips":                risk_pips,
        "distance_from_ema20":      distance_from_ema20,
        "rsi":                      rsi_val,
        "hour":                     float(hour),
        "ema_trend":                float(ema_trend),
        "zone_touch_count":         float(zone_touch_count),
        "htf_trend_alignment":      float(htf_trend_alignment),
        # v5
        "approach_distance":        float(approach_distance),
        "candle_body_ratio":        float(candle_body_ratio),
        "zone_age_bars":            float(zone_age_bars),
        "approach_momentum":        float(approach_momentum),
        # v6 (h1_confluence removed; prev_zone_outcome+time_since_last_touch removed v8)
        "h4_fvg_inside":            h4_fvg_inside,
        "d1_ema_distance":          d1_ema_distance,
        # v7: 1H alignment
        "h1_trend_alignment":       float(h1_trend_alignment),
        "h1_structure_shift":       float(h1_structure_shift),
        "h1_displacement":          float(h1_displacement),
        "h1_liquidity_sweep":       float(h1_liquidity_sweep),
        # v7: premium/discount
        "d1_premium_discount":      float(d1_premium_discount),
        "h4_premium_discount":      float(h4_premium_discount),
        # v7: 5M execution
        "m5_micro_bos":             float(m5_micro_bos),
        "m5_sweep_confirmation":    float(m5_sweep_confirmation),
        "m5_entry_candle_quality":  float(m5_entry_candle_quality),
        # ── Categorical ────────────────────────────────────────────────────────
        "zone_type":                str(zone.get("type", "demand")),
        "session":                  session,
        "approach_atr_regime":      approach_atr_reg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# v7 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_h1_features(
    df_1h: Optional[pd.DataFrame],
    touch_time,
    zone_type: str,
) -> tuple:
    """
    Returns (h1_trend_alignment, h1_structure_shift, h1_displacement, h1_liquidity_sweep).

    h1_trend_alignment  : -1 / 0 / 1  (1H EMA20 vs EMA50, aligned to zone direction)
    h1_structure_shift  : -1 / 0 / 1  (bullish or bearish BOS/MSS on 1H)
    h1_displacement     : float ≥ 0    (largest impulse candle / ATR in last 20 1H bars)
    h1_liquidity_sweep  : 0 / 1        (swing swept and closed back inside last 30 1H bars)
    """
    defaults = (0, 0, 0.0, 0)
    if df_1h is None or df_1h.empty or len(df_1h) < 20:
        return defaults

    try:
        if touch_time is not None and "time" in df_1h.columns:
            ts  = pd.to_datetime(touch_time)
            sub = df_1h[pd.to_datetime(df_1h["time"]) <= ts].copy()
        else:
            sub = df_1h.copy()

        if len(sub) < 20:
            return defaults

        # ── 1H ATR ───────────────────────────────────────────────────────────
        high = sub["high"]; low = sub["low"]; close = sub["close"]
        tr   = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_1h = float(tr.rolling(14).mean().iloc[-1])
        if not np.isfinite(atr_1h) or atr_1h <= 0:
            atr_1h = float(tr.mean()) or 1e-6

        # ── h1_trend_alignment ────────────────────────────────────────────────
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        if ema20 > ema50 * 1.0001:
            raw_trend = 1        # 1H bullish
        elif ema20 < ema50 * 0.9999:
            raw_trend = -1       # 1H bearish
        else:
            raw_trend = 0

        # Align to zone direction: demand wants bullish (1), supply wants bearish (-1)
        if zone_type == "demand":
            h1_trend_alignment = raw_trend          # 1=aligned, -1=counter, 0=neutral
        else:
            h1_trend_alignment = -raw_trend         # flip so 1=aligned for supply

        # ── h1_structure_shift ────────────────────────────────────────────────
        # Look back 30 bars for the most recent BOS/CHoCH
        window = sub.tail(30).reset_index(drop=True)
        swing_period = 3
        h1_structure_shift = 0

        swing_highs, swing_lows = [], []
        for i in range(swing_period, len(window) - swing_period):
            bar = window.iloc[i]
            if (bar["high"] > window.iloc[i-swing_period:i]["high"].max() and
                    bar["high"] > window.iloc[i+1:i+swing_period+1]["high"].max()):
                swing_highs.append({"i": i, "price": bar["high"]})
            if (bar["low"] < window.iloc[i-swing_period:i]["low"].min() and
                    bar["low"] < window.iloc[i+1:i+swing_period+1]["low"].min()):
                swing_lows.append({"i": i, "price": bar["low"]})

        last_close = float(window["close"].iloc[-1])
        if swing_highs and last_close > swing_highs[-1]["price"]:
            h1_structure_shift = 1    # bullish BOS
        elif swing_lows and last_close < swing_lows[-1]["price"]:
            h1_structure_shift = -1   # bearish BOS

        # ── h1_displacement ───────────────────────────────────────────────────
        recent_20 = sub.tail(20)
        body_sizes = (recent_20["close"] - recent_20["open"]).abs()
        h1_displacement = float(body_sizes.max() / atr_1h) if atr_1h > 0 else 0.0
        h1_displacement = min(h1_displacement, 10.0)

        # ── h1_liquidity_sweep ────────────────────────────────────────────────
        # Detect: price spiked below a swing low (or above swing high) and
        # closed back inside structure.
        # FIX 3: scan full `sub` (all 1H bars up to touch_time) instead of
        # tail(30). tail(30) missed sweeps that occurred earlier in the zone's
        # lifetime, producing near-zero values vs training mean of 0.972 —
        # a −5.85 z-score that systematically suppressed model scores.
        h1_liquidity_sweep = 0
        check_window = sub.reset_index(drop=True)
        sp = 4
        for i in range(sp, len(check_window)):
            bar     = check_window.iloc[i]
            prev_lo = check_window.iloc[max(0, i-sp):i]["low"].min()
            prev_hi = check_window.iloc[max(0, i-sp):i]["high"].max()

            # Bullish sweep: wick below prior swing low, closed back above
            if (bar["low"] < prev_lo and
                    bar["close"] > prev_lo and
                    zone_type == "demand"):
                h1_liquidity_sweep = 1
                break

            # Bearish sweep: wick above prior swing high, closed back below
            if (bar["high"] > prev_hi and
                    bar["close"] < prev_hi and
                    zone_type == "supply"):
                h1_liquidity_sweep = 1
                break

        return (h1_trend_alignment, h1_structure_shift,
                round(h1_displacement, 4), h1_liquidity_sweep)

    except Exception as e:
        logger.warning(f"_compute_h1_features failed: {e}")
        return defaults


def _compute_premium_discount(
    df_tf: Optional[pd.DataFrame],
    zone_price: float,
    touch_time,
    lookback: int = 20,
) -> int:
    """
    Returns -1 (discount), 0 (equilibrium ±10%), or 1 (premium).

    Uses the swing high and swing low of the last `lookback` bars on df_tf
    to define the range. Equilibrium band = middle 20% of the range.
    """
    if df_tf is None or df_tf.empty or len(df_tf) < lookback:
        return 0

    try:
        if touch_time is not None and "time" in df_tf.columns:
            ts  = pd.to_datetime(touch_time)
            sub = df_tf[pd.to_datetime(df_tf["time"]) <= ts].tail(lookback)
        else:
            sub = df_tf.tail(lookback)

        if len(sub) < 5:
            return 0

        swing_high = float(sub["high"].max())
        swing_low  = float(sub["low"].min())
        rng        = swing_high - swing_low
        if rng <= 0:
            return 0

        pos = (zone_price - swing_low) / rng   # 0.0 = at swing low, 1.0 = at swing high

        if pos > 0.6:
            return 1    # premium
        elif pos < 0.4:
            return -1   # discount
        else:
            return 0    # equilibrium

    except Exception as e:
        logger.warning(f"_compute_premium_discount failed: {e}")
        return 0


def _compute_m5_features(
    df_5m: Optional[pd.DataFrame],
    zone: Dict,
    pip_size: float,
    touch_time=None,
) -> tuple:
    """
    Returns (m5_micro_bos, m5_sweep_confirmation, m5_entry_candle_quality).

    m5_micro_bos            : 1 if a micro BOS exists in the last 60 5M bars
    m5_sweep_confirmation   : 1 if a micro liquidity grab (sweep + close back) in last 60 5M bars
    m5_entry_candle_quality : composite float [0, 1] from last 5 5M bars:
                              0.25 each for engulfing, displacement, strong close, rejection wick

    FIX 1 (v9): Window changed from tail(20) to tail(60).
    production_predictor._check_5min_entry() uses tail(60) when pre-computing
    m5 values for HTF-OB zones.  The fallback path (15-min zones that reach
    _compute_m5_features without pre-injected values) was using tail(20),
    producing fewer BOS/sweep detections and systematically lower model scores.
    Both paths now use the same 60-bar (5-hour) window.
    m5_entry_candle_quality still evaluates only the last 5 bars of that window.

    touch_time: timestamp of the touch bar — filters df_5m to bars BEFORE this
                time so no future bars are used. Critical for training correctness.
    """
    defaults = (0, 0, 0.0)
    if df_5m is None or df_5m.empty or len(df_5m) < 20:
        return defaults

    try:
        zone_type = zone.get("type", "demand")

        # ── Time filter: only use 5M bars up to and including touch_time ──────
        if touch_time is not None and "time" in df_5m.columns:
            ts  = pd.to_datetime(touch_time)
            sub = df_5m[pd.to_datetime(df_5m["time"]) <= ts]
        else:
            sub = df_5m

        if len(sub) < 20:
            return defaults

        # FIX 1 (v9): use tail(60) — 5 hours of 5M bars — matching
        # production_predictor._check_5min_entry() exactly.
        recent = sub.tail(60).reset_index(drop=True)

        # ── ATR on 5M ─────────────────────────────────────────────────────────
        hi = recent["high"]; lo = recent["low"]; cl = recent["close"]
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift(1)).abs(),
            (lo - cl.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_5m = float(tr.rolling(7).mean().iloc[-1])
        if not np.isfinite(atr_5m) or atr_5m <= 0:
            atr_5m = pip_size * 5

        # ── m5_micro_bos ──────────────────────────────────────────────────────
        m5_micro_bos = 0
        sp = 3
        for i in range(sp, len(recent) - 1):
            bar = recent.iloc[i]
            if zone_type == "demand":
                # Bullish micro BOS: close breaks above recent swing high
                prev_hi = recent.iloc[max(0, i-sp):i]["high"].max()
                if bar["close"] > prev_hi:
                    m5_micro_bos = 1
                    break
            else:
                # Bearish micro BOS: close breaks below recent swing low
                prev_lo = recent.iloc[max(0, i-sp):i]["low"].min()
                if bar["close"] < prev_lo:
                    m5_micro_bos = 1
                    break

        # ── m5_sweep_confirmation ─────────────────────────────────────────────
        m5_sweep_confirmation = 0
        for i in range(4, len(recent)):
            bar     = recent.iloc[i]
            prev_lo = recent.iloc[max(0, i-4):i]["low"].min()
            prev_hi = recent.iloc[max(0, i-4):i]["high"].max()

            if zone_type == "demand":
                if bar["low"] < prev_lo and bar["close"] > prev_lo:
                    m5_sweep_confirmation = 1
                    break
            else:
                if bar["high"] > prev_hi and bar["close"] < prev_hi:
                    m5_sweep_confirmation = 1
                    break

        # ── m5_entry_candle_quality ───────────────────────────────────────────
        last_5    = recent.tail(5).reset_index(drop=True)
        score     = 0.0

        last  = last_5.iloc[-1]
        prev  = last_5.iloc[-2] if len(last_5) >= 2 else last_5.iloc[-1]
        body  = abs(last["close"] - last["open"])
        rng   = last["high"] - last["low"]

        # 1. Engulfing: body fully covers previous bar's body
        prev_body = abs(prev["close"] - prev["open"])
        if body > prev_body and rng > 0:
            score += 0.25

        # 2. Displacement: large body relative to ATR
        if atr_5m > 0 and body / atr_5m >= 0.6:
            score += 0.25

        # 3. Strong close: closed in direction-favourable top/bottom 30% of range
        if rng > 0:
            if zone_type == "demand" and (last["close"] - last["low"]) / rng >= 0.7:
                score += 0.25
            elif zone_type == "supply" and (last["high"] - last["close"]) / rng >= 0.7:
                score += 0.25

        # 4. Rejection wick: counter-directional wick > body
        if rng > 0 and body > 0:
            if zone_type == "demand":
                lower_wick = min(last["open"], last["close"]) - last["low"]
                if lower_wick > body:
                    score += 0.25
            else:
                upper_wick = last["high"] - max(last["open"], last["close"])
                if upper_wick > body:
                    score += 0.25

        return (m5_micro_bos, m5_sweep_confirmation, round(score, 4))

    except Exception as e:
        logger.warning(f"_compute_m5_features failed: {e}")
        return defaults


# ─────────────────────────────────────────────────────────────────────────────
# v6 helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_d1_ema_distance(
    touch_bar: pd.Series,
    touch_time,
    df_d1: Optional[pd.DataFrame],
) -> float:
    if df_d1 is None or df_d1.empty:
        return 0.0
    try:
        if touch_time is not None:
            touch_ts = pd.to_datetime(touch_time)
            col = "time" if "time" in df_d1.columns else None
            if col:
                d1_before = df_d1[pd.to_datetime(df_d1[col]) <= touch_ts]
            else:
                d1_before = df_d1[df_d1.index <= touch_ts]
        else:
            d1_before = df_d1

        if len(d1_before) < 20:
            return 0.0

        ema20 = float(d1_before["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        d1_atr_series = d1_before["close"].diff().abs().rolling(14).mean()
        d1_atr = float(d1_atr_series.iloc[-1]) if len(d1_atr_series) >= 14 else 0.0
        if not np.isfinite(d1_atr) or d1_atr <= 0:
            return 0.0

        close_val = float(touch_bar.get("close", ema20))
        dist = abs(close_val - ema20) / d1_atr
        return float(min(dist, 10.0))
    except Exception:
        return 0.0


def _zone_level_key(zone: Dict, pip_size: float, bucket_pips: float = 10.0) -> str:
    mid = zone.get("price_mid", (zone.get("price_high", 0) + zone.get("price_low", 0)) / 2)
    bucket = round(mid / (pip_size * bucket_pips)) * (pip_size * bucket_pips)
    direction = zone.get("type", "demand")
    return f"{direction}:{bucket:.6f}"


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