"""
DUAL-TASK LABEL GENERATOR
Generates BOTH mean-reversion AND continuation labels from the same zones.

Output: Single parquet with TWO label columns:
  - label          → probability zone holds (mean-reversion)
  - label_cont     → probability zone breaks (continuation)

CHANGE LOG
----------
v7 (current):
  - WIRING: generate_labels_from_zones() now builds zone_history as it processes
    zones chronologically. Each entry records the outcome and touch timestamp of
    the most recent prior visit to a price level (keyed by _zone_level_key()).
    Passed to build_raw_feature_dict() so prev_zone_outcome and
    time_since_last_touch (v6 features) are populated with real values instead
    of the 0.5/0.0 defaults that fire when zone_history=None.
  - WIRING: add_context_features() now accepts and threads df_d1 through to
    build_raw_feature_dict() so d1_ema_distance is computed from real D1 bars.
    Previously always fell back to 0.0 (df_d1 was never forwarded).
  - zone dict passed to build_raw_feature_dict() now includes h1_confluence,
    sources, and tf_source so h1_confluence and h4_fvg_inside are populated
    directly from smc_detector output (not defaulting to 0).
  - risk_pips filter changed from a hard 30-pip cap to a dynamic
    MAX_RISK_ATR_MULTIPLE (3.0) × mean-ATR cap — the old flat 30-pip
    threshold killed nearly all USDJPY/XAUUSD rows (see MAX_RISK_ATR_MULTIPLE
    comment below for per-pair ATR reasoning).
v6:
  - CRITICAL FIX: check_reversion_outcome() and check_continuation_outcome()
    now use bar-by-bar simulation that respects the ORDER in which SL/TP/break
    levels are hit. The old code tracked best_tp_progress and best_sl_progress
    independently — meaning it could award a timeout score of 0.75 even if SL
    was hit first. Likewise, continuation used any() semantics that labelled a
    zone as a break win even when the zone had already reversed past the entry.
    Both functions now return immediately on the FIRST decisive bar.
  - CRITICAL FIX: rows with risk_pips > 30 are filtered OUT after label
    generation. A 30+ pip SL on a 15min zone is not a valid retail trade and
    indicates upstream geometry errors (HTF zone leaking into 15min set, or
    ATR-multiple SL on an oversized zone). EDA showed 40% of old rows exceeded
    50 pips. Threshold set at 30 pips matching the ATR fallback cap in
    feature_builder.
  - HIGH FIX: rows with zone_touch_count > 5 are filtered out before saving.
    EDA showed mean touch count of 14.2 — heavily recycled zones train the
    model on exhausted liquidity levels. Cap at 5 keeps the first two standard
    deviations of fresh zone interactions.
v5:
  - risk_pips clamped to [2, 150] before storing in candidate dict.
  - zone_confluence clamped to [1, 20] before storing.
v4:
  - risk_pips batch validation after generation; fallback rate > 10% raises.
v3:
  - calculate_zone_touch_count(), calculate_htf_trend_alignment() added.
v2:
  - find_first_touch_index() uses 1-ATR proximity.
v1:
  - Initial dual-label implementation.

Usage:
  python label_generator.py

Output:
  data/labels/training_labels_dual.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import feature_builder as _fb
from feature_builder import build_raw_feature_dict, _zone_level_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum valid risk in pips for a 15min zone trade.
# This is now ATR-relative, not a hard constant — different pairs have
# completely different pip scales:
#   EURUSD/GBPUSD: ATR ~8–12 pips   → 50 pip SL = ~5x ATR (reasonable cap)
#   USDJPY:        ATR ~80–120 pips  → 50 pip SL = <1x ATR (kills everything)
#   XAUUSD:        ATR ~150–300 pips → 50 pip SL = <1x ATR (kills everything)
# Cap is set at MAX_RISK_ATR_MULTIPLE × mean ATR, computed dynamically per run.
MAX_RISK_ATR_MULTIPLE = 3.0   # drop SLs wider than 3× mean ATR
MAX_RISK_PIPS_FALLBACK = 50.0 # used only if ATR unavailable
# Maximum zone touch count — zones tested more than this are considered exhausted
MAX_TOUCH_COUNT = 20


class DualLabelGenerator:
    """
    Generates BOTH reversion and continuation labels.

    REVERSION:
    - Entry: proximal edge
    - SL: floor-protected (max(zone_size * 1.5, ATR * 0.8))
    - TP: entry + zone_size * RR
    - Label: 1.0 TP hit first | 0.0 SL hit first | step-bucket score on timeout

    CONTINUATION:
    - Break level: zone edge + continuation_break_atr buffer
    - Label: 1.0 if close confirms break BEFORE reversion entry is hit |
             step-bucket proximity score on timeout
    """

    def __init__(self,
                 risk_reward: float = 2.0,
                 horizon_bars: int = 72,
                 sl_atr_multiple: float = 0.5,
                 continuation_break_atr: float = 0.3,
                 pip_size: float = 0.0001):
        self.risk_reward            = risk_reward
        self.horizon_bars           = horizon_bars
        self.sl_atr_multiple        = sl_atr_multiple
        self.continuation_break_atr = continuation_break_atr
        self.pip_size               = pip_size

        logger.info("DualLabelGenerator initialized:")
        logger.info(f"  Reversion: RR=1:{risk_reward}, Horizon={horizon_bars}, SL={sl_atr_multiple}ATR")
        logger.info(f"  Continuation: Break={continuation_break_atr}ATR beyond zone edge")
        logger.info(f"  Post-gen filters: risk_pips ≤ {MAX_RISK_ATR_MULTIPLE}×ATR, touch_count ≤ {MAX_TOUCH_COUNT}")

    # ─────────────────────────────────────────────────────────────────────────
    # ATR helper
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high       = df['high']
        low        = df['low']
        close_prev = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            abs(high - close_prev),
            abs(low - close_prev)
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # Touch index
    # ─────────────────────────────────────────────────────────────────────────

    def find_first_touch_index(self,
                               zone: Dict,
                               df: pd.DataFrame,
                               start_index: int,
                               atr: pd.Series) -> Optional[int]:
        """
        Scan forward and return index of first bar where price comes within
        1 ATR of the proximal zone edge.
        """
        end_index = min(start_index + self.horizon_bars, len(df))

        if zone['type'] == 'demand':
            proximal = zone['price_high']
            for i in range(start_index, end_index):
                bar         = df.iloc[i]
                current_atr = atr.iloc[i] if pd.notna(atr.iloc[i]) else 0.0010
                if bar['low'] <= proximal + current_atr:
                    return i
        else:
            proximal = zone['price_low']
            for i in range(start_index, end_index):
                bar         = df.iloc[i]
                current_atr = atr.iloc[i] if pd.notna(atr.iloc[i]) else 0.0010
                if bar['high'] >= proximal - current_atr:
                    return i

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Zone touch count (LOOKBACK only — no forward leakage)
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_zone_touch_count(self,
                                   zone: Dict,
                                   df: pd.DataFrame,
                                   zone_index: int,
                                   atr: pd.Series) -> int:
        """
        Count distinct price visits to this zone level in the 500-bar window
        BEFORE zone formation. Lookback only — no forward leakage.
        """
        if zone['type'] == 'demand':
            proximal = zone['price_high']
        else:
            proximal = zone['price_low']

        lookback    = 500
        start_index = max(0, zone_index - lookback)
        touch_count = 0
        i           = start_index

        while i < zone_index:
            bar         = df.iloc[i]
            current_atr = atr.iloc[i] if pd.notna(atr.iloc[i]) else 0.0010
            threshold   = current_atr * 0.5

            touched = False
            if zone['type'] == 'demand':
                if bar['low'] <= proximal + threshold:
                    touched = True
            else:
                if bar['high'] >= proximal - threshold:
                    touched = True

            if touched:
                touch_count += 1
                i += 20
            else:
                i += 1

        return touch_count

    # ─────────────────────────────────────────────────────────────────────────
    # HTF trend alignment
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_htf_trend_alignment(self,
                                      zone: Dict,
                                      df_d1: pd.DataFrame,
                                      zone_time) -> int:
        """
        Returns 1 if the D1 EMA20/50 trend aligns with the zone direction.
        Returns 0 if counter-trend or D1 data unavailable.
        """
        if df_d1 is None or df_d1.empty or zone_time is None:
            return 0

        try:
            zone_time = pd.to_datetime(zone_time)
        except Exception:
            return 0

        # ── CLOSED-D1 GATE ────────────────────────────────────────────────────
        # Use only FULLY CLOSED D1 bars — i.e. bars whose open time is strictly
        # BEFORE the calendar date of zone_time. A D1 bar with open=2026-06-26
        # 00:00 is only complete once 2026-06-27 opens; at 05:23 on Jun 26 it
        # still has 18+ hours of price action ahead. Including it (the old <= logic)
        # meant training saw the full day's close — a lookahead that live can't match.
        zone_date = pd.to_datetime(zone_time).normalize()  # floor to midnight UTC
        if 'time' in df_d1.columns:
            d1_before = df_d1[pd.to_datetime(df_d1['time']).dt.normalize() < zone_date]
        else:
            d1_before = df_d1[pd.to_datetime(df_d1.index).normalize() < zone_date]
        # ─────────────────────────────────────────────────────────────────────

        if len(d1_before) < 50:
            return 0

        ema20 = d1_before['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = d1_before['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        d1_uptrend = bool(ema20 > ema50)

        if zone['type'] == 'demand':
            return 1 if d1_uptrend else 0
        else:
            return 1 if not d1_uptrend else 0

    # ─────────────────────────────────────────────────────────────────────────
    # Reversion outcome — bar-by-bar simulation (v6 CRITICAL FIX)
    # ─────────────────────────────────────────────────────────────────────────

    def check_reversion_outcome(self,
                                zone: Dict,
                                df: pd.DataFrame,
                                start_index: int,
                                current_atr: float) -> Tuple[float, str]:
        """
        REVERSION: Check if zone holds (TP hit before SL).

        v6 FIX: Bar-by-bar simulation — checks SL and TP on EVERY bar in order.
        Returns immediately on the first decisive bar so SL-hit-first can never
        be awarded a timeout score. The old code tracked best_tp_progress and
        best_sl_progress across all bars independently, which could assign 0.75
        even when SL was hit before the TP progress was reached.

        Timeout (no decisive hit within horizon): step-bucket scoring based on
        how close price got to TP or SL, giving a partial signal rather than
        a hard 0/1.
        """
        zone_size = abs(zone['price_high'] - zone['price_low'])
        if zone_size <= 0:
            zone_size = current_atr * 0.3

        # SL placed just beyond zone distal edge with small ATR buffer.
        # Risk = zone_size so TP = entry + zone_size * RR (realistic target).
        atr_buffer = current_atr * 0.15

        if zone['type'] == 'demand':
            entry     = zone['price_high']          # proximal edge (top of zone)
            sl        = zone['price_low'] - atr_buffer  # just below distal edge
            risk      = zone_size                   # zone width only
            tp        = entry + (risk * self.risk_reward)
            direction = 'buy'
        else:
            entry     = zone['price_low']           # proximal edge (bottom of zone)
            sl        = zone['price_high'] + atr_buffer  # just above distal edge
            risk      = zone_size                   # zone width only
            tp        = entry - (risk * self.risk_reward)
            direction = 'sell'

        end_index        = min(start_index + self.horizon_bars, len(df))
        entry_to_tp      = abs(tp - entry)
        entry_to_sl      = abs(sl - entry)
        best_tp_progress = 0.0
        best_sl_progress = 0.0

        for i in range(start_index, end_index):
            bar = df.iloc[i]

            if direction == 'buy':
                # Check SL first (conservative — bad news first)
                if bar['low'] <= sl:
                    return 0.0, f'reversion_sl_hit_bar_{i - start_index}'
                if bar['high'] >= tp:
                    return 1.0, f'reversion_tp_hit_bar_{i - start_index}'
                tp_prog = (bar['high'] - entry) / entry_to_tp if entry_to_tp > 0 else 0
                sl_prog = (entry - bar['low'])  / entry_to_sl if entry_to_sl > 0 else 0
            else:
                if bar['high'] >= sl:
                    return 0.0, f'reversion_sl_hit_bar_{i - start_index}'
                if bar['low'] <= tp:
                    return 1.0, f'reversion_tp_hit_bar_{i - start_index}'
                tp_prog = (entry - bar['low'])  / entry_to_tp if entry_to_tp > 0 else 0
                sl_prog = (bar['high'] - entry) / entry_to_sl if entry_to_sl > 0 else 0

            best_tp_progress = max(best_tp_progress, tp_prog)
            best_sl_progress = max(best_sl_progress, sl_prog)

        # Timeout: step-bucket scoring (no decisive hit within horizon)
        if best_tp_progress >= 0.70:
            return 0.75, 'reversion_timeout_near_tp'
        elif best_tp_progress >= 0.30:
            return 0.60, 'reversion_timeout_tp_progress'
        elif best_sl_progress >= 0.70:
            return 0.20, 'reversion_timeout_near_sl'
        elif best_sl_progress >= 0.30:
            return 0.35, 'reversion_timeout_sl_progress'
        else:
            return 0.50, 'reversion_timeout_neutral'

    # ─────────────────────────────────────────────────────────────────────────
    # Continuation outcome — bar-by-bar simulation (v6 CRITICAL FIX)
    # ─────────────────────────────────────────────────────────────────────────

    def check_continuation_outcome(self,
                                   zone: Dict,
                                   df: pd.DataFrame,
                                   start_index: int,
                                   current_atr: float) -> Tuple[float, str]:
        """
        CONTINUATION: Check if zone breaks (close confirms break + ATR buffer)
        BEFORE the reversion entry is hit.

        v6 FIX: Entry-first validation. On each bar we check:
          1. Has the reversion entry been touched? If yes → zone held, NOT a
             continuation win. Return 0.0 immediately.
          2. Has the break level been confirmed? If yes → continuation win 1.0.
        This prevents labelling a zone as a break when price actually bounced
        from the proximal edge first (the old any() check missed this ordering).
        """
        if zone['type'] == 'demand':
            # Break = price closes BELOW the zone floor (distal edge - ATR buffer)
            break_level     = zone['price_low'] - (self.continuation_break_atr * current_atr)
            break_direction = 'bearish'
            # Reversion guard = price closes ABOVE proximal edge (zone held = bounced up)
            # Use close, not low, with a small buffer so wick micro-touches don't count
            reversion_guard = zone['price_high'] + (current_atr * 0.1)
        else:
            # Break = price closes ABOVE zone ceiling (distal edge + ATR buffer)
            break_level     = zone['price_high'] + (self.continuation_break_atr * current_atr)
            break_direction = 'bullish'
            # Reversion guard = price closes BELOW proximal edge (zone held = bounced down)
            reversion_guard = zone['price_low'] - (current_atr * 0.1)

        end_index        = min(start_index + self.horizon_bars, len(df))
        closest_approach = float('inf')

        for i in range(start_index, end_index):
            bar = df.iloc[i]

            if break_direction == 'bearish':
                # Zone held (bounced up) — close confirms reversion BEFORE break
                if bar['close'] >= reversion_guard:
                    return 0.0, f'continuation_reversion_confirmed_bar_{i - start_index}'
                # Zone broke — close below the break level
                if bar['close'] < break_level:
                    return 1.0, f'continuation_break_bar_{i - start_index}'
                closest_approach = min(closest_approach, bar['low'] - break_level)
            else:
                # Zone held (bounced down) — close confirms reversion BEFORE break
                if bar['close'] <= reversion_guard:
                    return 0.0, f'continuation_reversion_confirmed_bar_{i - start_index}'
                # Zone broke — close above the break level
                if bar['close'] > break_level:
                    return 1.0, f'continuation_break_bar_{i - start_index}'
                closest_approach = min(closest_approach, break_level - bar['high'])

        # Timeout proximity scoring
        zone_size   = abs(zone['price_high'] - zone['price_low'])
        buffer_size = self.continuation_break_atr * current_atr
        total_dist  = zone_size / 2 + buffer_size

        if closest_approach < 0:
            proximity_score = 0.8
        elif closest_approach < buffer_size * 0.5:
            proximity_score = 0.6
        elif closest_approach < buffer_size:
            proximity_score = 0.4
        elif closest_approach < total_dist:
            proximity_score = 0.2
        else:
            proximity_score = 0.0

        return proximity_score, f'continuation_timeout_{proximity_score:.1f}'

    # ─────────────────────────────────────────────────────────────────────────
    # Main label generation loop
    # ─────────────────────────────────────────────────────────────────────────

    def generate_labels_from_zones(self,
                                   zones: List[Dict],
                                   df: pd.DataFrame,
                                   df_d1: Optional[pd.DataFrame] = None,
                                   df_1h: Optional[pd.DataFrame] = None,
                                   df_4h: Optional[pd.DataFrame] = None,
                                   df_5m: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate BOTH reversion and continuation labels.

        Post-generation filters applied (v7):
          • risk_pips > MAX_RISK_ATR_MULTIPLE × mean ATR → dropped (invalid 15min zone SL,
            dynamic per-pair threshold; see MAX_RISK_ATR_MULTIPLE above)
          • zone_touch_count > MAX_TOUCH_COUNT (20) → dropped (exhausted zone)
        """
        logger.info(f"Generating DUAL labels for {len(zones)} zones...")
        if df_d1 is not None:
            logger.info(f"  D1 bars loaded: {len(df_d1)} — HTF alignment active")
        else:
            logger.info("  D1 bars not provided — htf_trend_alignment will be 0")
        if df_1h is not None:
            logger.info(f"  1H bars loaded: {len(df_1h)} — 1H alignment features active")
        else:
            logger.info("  1H bars not provided — h1 features will be 0")
        if df_4h is not None:
            logger.info(f"  4H bars loaded: {len(df_4h)} — h4_premium_discount active")
        else:
            logger.info("  4H bars not provided — h4_premium_discount will be 0")
        if df_5m is not None:
            logger.info(f"  5M bars loaded: {len(df_5m)} — m5 execution features active")
        else:
            logger.info("  5M bars not provided — m5 features will be 0")

        atr = self.calculate_atr(df)

        candidates         = []
        skipped            = 0
        reversion_tp       = 0
        reversion_sl       = 0
        reversion_timeout  = 0
        continuation_break = 0
        continuation_hold  = 0
        touch_found        = 0

        # zone_history: level_key → {"outcome": float, "time": Timestamp}
        # Built chronologically so each zone only sees PAST outcomes — no leakage.
        zone_history: Dict = {}

        for zone in zones:
            zone_index = zone.get('index', 0)

            # FIX: skip FVG zones for reversion labelling — unfilled FVGs are
            # selected because price hasn't returned, inflating reversion wins.
            # FVG zones kept for continuation labels only.
            zone_sources = zone.get('sources', [zone.get('source', 'order_block')])
            is_fvg_only  = all(s == 'fvg' for s in zone_sources)
            # FVG zones: skip reversion label (biased) but keep for continuation

            if zone_index + self.horizon_bars >= len(df):
                skipped += 1
                continue

            current_atr = atr.iloc[zone_index]
            if pd.isna(current_atr) or current_atr <= 0:
                skipped += 1
                continue

            zone_size = abs(zone['price_high'] - zone['price_low'])
            if zone_size <= 0:
                zone_size = current_atr * 0.3

            atr_buffer = current_atr * 0.15

            if zone['type'] == 'demand':
                entry     = zone['price_high']
                sl        = zone['price_low'] - atr_buffer
                risk      = zone_size
                tp        = entry + (risk * self.risk_reward)
                direction = 'buy'
            else:
                entry     = zone['price_low']
                sl        = zone['price_high'] + atr_buffer
                risk      = zone_size
                tp        = entry - (risk * self.risk_reward)
                direction = 'sell'

            touch_index = self.find_first_touch_index(zone, df, zone_index, atr)

            # ── LEAKAGE FIX ───────────────────────────────────────────────────
            # Outcome simulation MUST start at touch_index, not zone_index.
            #
            # Why this matters:
            #   zone_index = bar where the zone was detected (past event)
            #   touch_index = bar where price first returns to the zone (future event,
            #                 the moment we would actually enter the trade)
            #
            # Starting at zone_index causes the simulation to scan bars between
            # zone formation and the touch — bars where price is still moving
            # toward the zone. For tight zones (RR < 1), the TP is so close to
            # entry that these pre-touch bars already reach TP, giving a spurious
            # label=1.0. This produced 99.3% win rates on RR<1 rows.
            #
            # If no touch is found within horizon_bars, skip this zone entirely.
            # A zone that price never returns to is not a tradeable signal — it
            # should not contribute a label of any kind.
            if touch_index is None:
                skipped += 1
                continue
            touch_found += 1

            # Use ATR at touch bar for outcome simulation (more accurate than
            # zone-formation ATR, which may be stale by many bars)
            touch_atr = atr.iloc[touch_index]
            if pd.isna(touch_atr) or touch_atr <= 0:
                touch_atr = current_atr   # fallback to zone ATR

            zone_touch_count    = self.calculate_zone_touch_count(zone, df, zone_index, atr)
            htf_trend_alignment = self.calculate_htf_trend_alignment(
                zone, df_d1, zone.get('time')
            )

            # Start both simulations from touch_index (entry bar), not zone_index
            label_reversion, reason_reversion = self.check_reversion_outcome(
                zone, df, touch_index, touch_atr
            )
            label_continuation, reason_continuation = self.check_continuation_outcome(
                zone, df, touch_index, touch_atr
            )

            if label_reversion == 1.0:
                reversion_tp += 1
            elif label_reversion == 0.0:
                reversion_sl += 1
            else:
                reversion_timeout += 1

            if label_continuation >= 0.9:
                continuation_break += 1
            else:
                continuation_hold += 1

            # ── Update zone_history (for prev_zone_outcome feature) ───────────
            # Record the outcome at this price level so subsequent zones at the
            # same level can read it. Keyed by rounded price bucket to survive
            # minor zone migrations. Only decisive outcomes (TP=1.0, SL=0.0) are
            # recorded; timeouts (fractional scores) are left as 0.5 (the feature
            # default) so they don't pollute the history with ambiguous labels.
            level_key = _zone_level_key(zone, self.pip_size)
            touch_time = (df["time"].iloc[touch_index]
                          if "time" in df.columns else None)
            if label_reversion in (0.0, 1.0):
                zone_history[level_key] = {
                    "outcome": label_reversion,
                    "time":    touch_time,
                }

            # ── Clamp noisy raw values ────────────────────────────────────────
            raw_risk_pips     = risk / self.pip_size
            clamped_risk_pips = float(np.clip(raw_risk_pips, 2.0, 150.0))
            if clamped_risk_pips != raw_risk_pips:
                logger.debug(
                    f"risk_pips clamped {raw_risk_pips:.1f} → "
                    f"{clamped_risk_pips:.1f} (zone_index={zone_index})"
                )

            raw_conf     = int(zone.get('confluence', 1))
            clamped_conf = int(np.clip(raw_conf, 1, 20))
            if clamped_conf != raw_conf:
                logger.debug(
                    f"zone_confluence clamped {raw_conf} → {clamped_conf} "
                    f"(zone_index={zone_index})"
                )

            candidate = {
                'zone_type':            zone['type'],
                'direction':            direction,
                'entry':                entry,
                'sl':                   sl,
                'tp':                   tp,
                'risk_pips':            clamped_risk_pips,
                'reward_pips':          (risk * self.risk_reward) / self.pip_size,
                'zone_index':           zone_index,
                'zone_time':            zone.get('time'),
                'zone_strength':        zone.get('strength', 1.0),
                'zone_confluence':      clamped_conf,
                'touch_index':          touch_index,
                'zone_touch_count':     zone_touch_count,
                'htf_trend_alignment':  htf_trend_alignment,
                'label':                label_reversion,
                'label_cont':           label_continuation,
                'outcome_reversion':    reason_reversion,
                'outcome_continuation': reason_continuation,
                # v6 MTF fields — forwarded to add_context_features → build_raw_feature_dict
                'h1_confluence':        bool(zone.get('h1_confluence', False)),
                'zone_sources':         zone.get('sources', [zone.get('source', 'order_block')]),
                'tf_source':            zone.get('tf_source', '4h'),
                'zone_price_high':      zone['price_high'],
                'zone_price_low':       zone['price_low'],
                'zone_price_mid':       zone.get('price_mid',
                                            (zone['price_high'] + zone['price_low']) / 2),
                'fvg_overlap':          bool(zone.get('fvg_overlap', False)),
            }

            candidates.append(candidate)

        logger.info(f"Generated {len(candidates)} dual-labeled candidates (before filters)")
        logger.info(f"  Skipped (insufficient data): {skipped}")
        logger.info(
            f"  Touch index found: {touch_found}/{len(candidates)} "
            f"({touch_found/max(len(candidates),1)*100:.0f}% coverage)"
        )

        logger.info("\n📊 REVERSION Labels (pre-filter):")
        logger.info(f"  TP hits:  {reversion_tp}")
        logger.info(f"  SL hits:  {reversion_sl}")
        logger.info(f"  Timeouts: {reversion_timeout}")

        logger.info("\n📊 CONTINUATION Labels (pre-filter):")
        logger.info(f"  Breaks:   {continuation_break}")
        logger.info(f"  Holds:    {continuation_hold}")

        if not candidates:
            logger.warning("No valid candidates generated!")
            return pd.DataFrame()

        labels_df = pd.DataFrame(candidates)

        # ── Filter 1: drop degenerate risk_pips ≤ 0 ──────────────────────────
        bad_mask = labels_df['risk_pips'] <= 0
        n_bad    = bad_mask.sum()
        if n_bad > 0:
            fallback_rate = n_bad / len(labels_df)
            logger.warning(
                f"⚠️  risk_pips ≤ 0 in {n_bad}/{len(labels_df)} rows "
                f"({fallback_rate:.1%}). Dropping."
            )
            labels_df = labels_df[~bad_mask].reset_index(drop=True)
            if fallback_rate > 0.10:
                raise ValueError(
                    f"risk_pips fallback rate {fallback_rate:.1%} exceeds 10%. "
                    "Fix smc_detector zone geometry before training."
                )

        # ── Filter 2: drop risk_pips > ATR-relative ceiling (v7) ────────────
        # Hard pip limits break non-forex pairs (XAUUSD, USDJPY) because their
        # ATR is 10–100× larger than EURUSD in pip terms.
        # Solution: compute ceiling as MAX_RISK_ATR_MULTIPLE × mean ATR in pips.
        if 'risk_pips' in labels_df.columns and len(labels_df) > 0:
            mean_atr_pips = float(
                df['atr'].dropna().mean() / self.pip_size
                if 'atr' in df.columns
                else labels_df['risk_pips'].median() * 2
            )
            dynamic_max_risk = max(
                MAX_RISK_ATR_MULTIPLE * mean_atr_pips,
                MAX_RISK_PIPS_FALLBACK,   # never go below 50p fallback floor
            )
        else:
            dynamic_max_risk = MAX_RISK_PIPS_FALLBACK

        n_before       = len(labels_df)
        high_risk_mask = labels_df['risk_pips'] > dynamic_max_risk
        n_dropped      = high_risk_mask.sum()
        if n_dropped > 0:
            logger.warning(
                f"⚠️  Dropping {n_dropped}/{n_before} rows with risk_pips > "
                f"{dynamic_max_risk:.1f}p ({n_dropped/n_before:.1%}). "
                f"[ATR-relative cap: {MAX_RISK_ATR_MULTIPLE}× mean ATR "
                f"({mean_atr_pips:.1f}p)]"
            )
            labels_df = labels_df[~high_risk_mask].reset_index(drop=True)
        else:
            logger.info(
                f"  risk_pips filter: all {n_before} rows within "
                f"{dynamic_max_risk:.1f}p cap ✅"
            )

        # ── Filter 3: drop zone_touch_count > MAX_TOUCH_COUNT (v6) ───────────
        n_before      = len(labels_df)
        stale_mask    = labels_df['zone_touch_count'] > MAX_TOUCH_COUNT
        n_stale       = stale_mask.sum()
        if n_stale > 0:
            logger.warning(
                f"⚠️  Dropping {n_stale}/{n_before} rows with zone_touch_count > "
                f"{MAX_TOUCH_COUNT} ({n_stale/n_before:.1%}). "
                "Zones visited 6+ times have likely had their liquidity exhausted."
            )
            labels_df = labels_df[~stale_mask].reset_index(drop=True)

        logger.info(f"\n✅ Final dataset after all filters: {len(labels_df)} rows")

        # ── Log fallbacks from feature_builder ───────────────────────────────
        fb_fallbacks = _fb.RISK_PIPS_FALLBACK_COUNT
        if fb_fallbacks > 0:
            logger.warning(
                f"⚠️  feature_builder risk_pips ATR fallback fired "
                f"{fb_fallbacks} time(s) during add_context_features()."
            )

        # ── Touch count distribution ──────────────────────────────────────────
        tc = labels_df['zone_touch_count']
        logger.info("\n📊 Zone Touch Count Distribution (post-filter):")
        logger.info(f"  0 touches (fresh): {(tc == 0).sum()} zones")
        logger.info(f"  1 touch:           {(tc == 1).sum()} zones")
        logger.info(f"  2 touches:         {(tc == 2).sum()} zones")
        logger.info(f"  3-5 touches:       {((tc >= 3) & (tc <= 5)).sum()} zones")

        htf = labels_df['htf_trend_alignment']
        logger.info("\n📊 HTF Alignment:")
        logger.info(f"  Aligned:   {(htf == 1).sum()} zones")
        logger.info(f"  Unaligned: {(htf == 0).sum()} zones")

        logger.info("\n📈 Label Statistics (post-filter):")
        logger.info(f"  Reversion avg:     {labels_df['label'].mean():.3f}")
        logger.info(f"  Continuation avg:  {labels_df['label_cont'].mean():.3f}")
        logger.info(f"  Correlation:       {labels_df['label'].corr(labels_df['label_cont']):.3f}")

        # Warn if continuation win rate still looks inflated
        cont_win_rate = (labels_df['label_cont'] >= 0.9).mean()
        if cont_win_rate > 0.35:
            logger.warning(
                f"⚠️  Continuation win rate is {cont_win_rate:.1%} — expected ~20-25%. "
                "Verify entry-first validation in check_continuation_outcome()."
            )

        # Stash zone_history and MTF dataframes as DataFrame attrs so
        # add_context_features can pick them up without a public signature change.
        # attrs are NOT written to parquet — purely in-memory transport.
        labels_df.attrs["_zone_history"] = zone_history
        labels_df.attrs["_df_d1"]        = df_d1
        labels_df.attrs["_df_1h"]        = df_1h
        labels_df.attrs["_df_4h"]        = df_4h
        labels_df.attrs["_df_5m"]        = df_5m

        return labels_df

    # ─────────────────────────────────────────────────────────────────────────
    # Feature attachment
    # ─────────────────────────────────────────────────────────────────────────

    def add_context_features(self,
                             labels_df: pd.DataFrame,
                             df: pd.DataFrame) -> pd.DataFrame:
        """
        Add canonical feature columns to the labels DataFrame.
        """
        logger.info("Adding context features via canonical feature_builder...")

        if labels_df.empty:
            return labels_df

        df = df.copy()

        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        delta        = df["close"].diff()
        gain         = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss         = -delta.where(delta < 0, 0).rolling(window=14).mean()
        df["rsi"]    = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df["atr"]    = self.calculate_atr(df)

        _fb.RISK_PIPS_FALLBACK_COUNT = 0

        # Retrieve zone_history and MTF dataframes stashed by generate_labels_from_zones().
        # Fall back gracefully if add_context_features is called standalone.
        zone_history: Dict = labels_df.attrs.get("_zone_history", {})
        df_d1_ctx          = labels_df.attrs.get("_df_d1", None)
        df_1h_ctx          = labels_df.attrs.get("_df_1h", None)
        df_4h_ctx          = labels_df.attrs.get("_df_4h", None)
        df_5m_ctx          = labels_df.attrs.get("_df_5m", None)

        feature_cols = []
        for _, row in labels_df.iterrows():
            zone = {
                "index":         int(row.get("zone_index", 0)),
                "type":          row.get("zone_type", "demand"),
                "strength":      float(row.get("zone_strength", 1.0)),
                "confluence":    int(row.get("zone_confluence", 1)),
                "price_low":     float(row.get("zone_price_low",  row.get("sl",    0.0))),
                "price_high":    float(row.get("zone_price_high", row.get("entry", 0.0))),
                "price_mid":     float(row.get("zone_price_mid",
                                    (row.get("zone_price_high", row.get("entry", 0.0)) +
                                     row.get("zone_price_low",  row.get("sl",    0.0))) / 2)),
                "touch_count":   int(row.get("zone_touch_count", 1)),
                "htf_alignment": int(row.get("htf_trend_alignment", 0)),
                # v6 MTF fields — populated from smc_detector output stored in candidate
                "h1_confluence": bool(row.get("h1_confluence", False)),
                "sources":       row.get("zone_sources", ["order_block"]),
                "tf_source":     row.get("tf_source", "4h"),
                "fvg_overlap":   bool(row.get("fvg_overlap", False)),
            }
            entry = float(row.get("entry", 0.0))
            sl    = float(row.get("sl",    0.0))

            touch_index = row.get("touch_index", None)
            if touch_index is not None and not pd.isna(touch_index):
                touch_index = int(touch_index)
            else:
                touch_index = None

            feat = build_raw_feature_dict(
                zone, df, entry, sl,
                pip_size=self.pip_size,
                touch_index=touch_index,
                df_d1=df_d1_ctx,
                df_1h=df_1h_ctx,
                df_4h=df_4h_ctx,
                df_5m=df_5m_ctx,
                zone_history=zone_history,
            )
            feature_cols.append(feat)

        if _fb.RISK_PIPS_FALLBACK_COUNT > 0:
            logger.warning(
                f"⚠️  {_fb.RISK_PIPS_FALLBACK_COUNT} risk_pips ATR fallback(s) "
                "fired in add_context_features()."
            )

        features_df = pd.DataFrame(feature_cols)

        overlap = [c for c in features_df.columns if c in labels_df.columns]
        if overlap:
            labels_df = labels_df.drop(columns=overlap)

        labels_df = pd.concat([labels_df.reset_index(drop=True), features_df], axis=1)

        logger.info(f"✅ Added {len(features_df.columns)} canonical context features")
        return labels_df

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────

    def save_labels(self, labels_df: pd.DataFrame,
                    output_path: str = "data/labels/training_labels_dual.parquet"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        labels_df.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved {len(labels_df)} dual labels to {output_path}")

        print("\n" + "=" * 60)
        print("DUAL LABEL GENERATION SUMMARY")
        print("=" * 60)
        print(f"Total candidates: {len(labels_df)}")

        print("\n📊 REVERSION (Mean-Revert):")
        print(f"  Average: {labels_df['label'].mean():.3f}")
        print(f"  Win rate (label=1.0): {(labels_df['label'] == 1.0).mean():.1%}")
        print(f"  Signal rate (≥0.5):   {(labels_df['label'] >= 0.5).mean():.1%}")

        print("\n📊 CONTINUATION (Break-Out):")
        print(f"  Average: {labels_df['label_cont'].mean():.3f}")
        print(f"  Break rate (label=1.0): {(labels_df['label_cont'] == 1.0).mean():.1%}")
        print(f"  Signal rate (≥0.5):     {(labels_df['label_cont'] >= 0.5).mean():.1%}")

        print("\n🔗 Label Correlation:")
        corr = labels_df['label'].corr(labels_df['label_cont'])
        print(f"  Pearson r = {corr:.3f}")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────



# ── Symbol config — single source of truth ───────────────────────────────────
SYMBOL_CONFIG = {
    'EURUSD': 0.0001,
    'GBPUSD': 0.0001,
    'USDJPY': 0.01,
    'XAUUSD': 1.0,
    'AUDUSD': 0.0001,
}


def run_one_symbol(sym: str, pip_size: float) -> pd.DataFrame:
    """
    Generate labels for a single symbol. Returns the labels DataFrame
    (empty if anything goes wrong). Saves per-symbol parquet as a side effect.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  GENERATING LABELS: {sym}  (pip_size={pip_size})")
    logger.info(f"{'='*60}")

    # ── 15min bars ────────────────────────────────────────────────────────────
    bars_path = Path(f"data/bars_15min_{sym}.parquet")
    if not bars_path.exists():
        # try without underscore suffix (legacy naming)
        bars_path = Path(f"data/bars_15min.parquet")
    if not bars_path.exists():
        logger.error(f"15min bars not found for {sym}: {bars_path}")
        return pd.DataFrame()

    df = pd.read_parquet(bars_path)
    if 'time' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'time'})
    logger.info(f"Loaded {len(df):,} 15min bars")

    # ── D1 bars ───────────────────────────────────────────────────────────────
    df_d1 = None
    for d1_name in [f"data/bars_1d_{sym}.parquet", f"data/bars_d1_{sym}.parquet"]:
        if Path(d1_name).exists():
            df_d1 = pd.read_parquet(d1_name)
            if 'time' not in df_d1.columns:
                df_d1 = df_d1.reset_index()
                if 'index' in df_d1.columns:
                    df_d1 = df_d1.rename(columns={'index': 'time'})
            logger.info(f"Loaded {len(df_d1):,} D1 bars")
            break
    if df_d1 is None:
        logger.warning(f"D1 bars not found for {sym} — htf_trend_alignment=0")

    # ── 1H bars ───────────────────────────────────────────────────────────────
    df_1h = None
    for h1_name in [f"data/bars_1h_{sym}.parquet", f"data/bars_h1_{sym}.parquet"]:
        if Path(h1_name).exists():
            df_1h = pd.read_parquet(h1_name)
            if 'time' not in df_1h.columns:
                df_1h = df_1h.reset_index()
                if 'index' in df_1h.columns:
                    df_1h = df_1h.rename(columns={'index': 'time'})
            logger.info(f"Loaded {len(df_1h):,} 1H bars")
            break
    if df_1h is None:
        logger.warning(f"1H bars not found for {sym} — h1_trend_alignment/structure/displacement/sweep will be 0")

    # ── 4H bars ───────────────────────────────────────────────────────────────
    df_4h = None
    for h4_name in [f"data/bars_4h_{sym}.parquet", f"data/bars_h4_{sym}.parquet"]:
        if Path(h4_name).exists():
            df_4h = pd.read_parquet(h4_name)
            if 'time' not in df_4h.columns:
                df_4h = df_4h.reset_index()
                if 'index' in df_4h.columns:
                    df_4h = df_4h.rename(columns={'index': 'time'})
            logger.info(f"Loaded {len(df_4h):,} 4H bars")
            break
    if df_4h is None:
        logger.warning(f"4H bars not found for {sym} — h4_premium_discount will be 0")

    # ── 5M bars ───────────────────────────────────────────────────────────────
    df_5m = None
    for m5_name in [f"data/bars_5min_{sym}.parquet", f"data/bars_5m_{sym}.parquet", f"data/bars_m5_{sym}.parquet"]:
        if Path(m5_name).exists():
            df_5m = pd.read_parquet(m5_name)
            if 'time' not in df_5m.columns:
                df_5m = df_5m.reset_index()
                if 'index' in df_5m.columns:
                    df_5m = df_5m.rename(columns={'index': 'time'})
            logger.info(f"Loaded {len(df_5m):,} 5M bars")
            break
    if df_5m is None:
        logger.warning(f"5M bars not found for {sym} — m5 execution features will be 0")

    # ── Merged zones (from smc_detector.py output) ───────────────────────────
    zones_path = Path(f"data/features/merged_zones_{sym}.parquet")
    if not zones_path.exists():
        logger.error(f"Zones not found: {zones_path}  — run smc_detector.py first")
        return pd.DataFrame()

    zones_df = pd.read_parquet(zones_path)
    zones    = zones_df.to_dict('records')
    logger.info(f"Loaded {len(zones):,} zones")

    # ── Generate labels ───────────────────────────────────────────────────────
    generator = DualLabelGenerator(
        risk_reward=1.5,
        horizon_bars=120,
        sl_atr_multiple=0.5,
        continuation_break_atr=0.3,
        pip_size=pip_size,
    )

    labels_df = generator.generate_labels_from_zones(
        zones, df, df_d1=df_d1, df_1h=df_1h, df_4h=df_4h, df_5m=df_5m
    )

    if labels_df.empty:
        logger.error(f"❌ No labels generated for {sym}")
        return pd.DataFrame()

    labels_df = generator.add_context_features(labels_df, df)

    # Tag with symbol so the combined file is queryable per-pair
    labels_df['symbol'] = sym

    # Save per-symbol file (used by generate_outcomes.py)
    per_sym_path = f"data/labels/training_labels_{sym}.parquet"
    generator.save_labels(labels_df, per_sym_path)
    logger.info(f"✅ {sym} → {per_sym_path}  ({len(labels_df):,} rows)")

    return labels_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dual-task label generator")
    parser.add_argument(
        "--symbol", default="ALL",
        help="Symbol to process, e.g. EURUSD. Use ALL (default) to run all 4 symbols."
    )
    parser.add_argument(
        "--pip_size", type=float, default=None,
        help="Override pip size. If omitted, uses the built-in config per symbol."
    )
    args = parser.parse_args()

    sym_arg = args.symbol.upper().strip()

    # ── Decide which symbols to run ───────────────────────────────────────────
    if sym_arg == 'ALL':
        symbols_to_run = list(SYMBOL_CONFIG.items())  # [(sym, pip_size), ...]
        logger.info(f"Running ALL symbols: {[s for s,_ in symbols_to_run]}")
    else:
        if sym_arg not in SYMBOL_CONFIG and args.pip_size is None:
            logger.error(
                f"Unknown symbol '{sym_arg}' and no --pip_size provided. "
                f"Known symbols: {list(SYMBOL_CONFIG.keys())}"
            )
            return
        pip = args.pip_size if args.pip_size is not None else SYMBOL_CONFIG[sym_arg]
        symbols_to_run = [(sym_arg, pip)]

    # Override pip_size if explicitly provided (single-symbol mode only)
    if args.pip_size is not None and sym_arg != 'ALL':
        symbols_to_run = [(sym_arg, args.pip_size)]

    # ── Run each symbol ───────────────────────────────────────────────────────
    Path("data/labels").mkdir(parents=True, exist_ok=True)
    all_dfs = []
    failed  = []

    for sym, pip_size in symbols_to_run:
        try:
            df = run_one_symbol(sym, pip_size)
            if not df.empty:
                all_dfs.append(df)
            else:
                failed.append(sym)
        except Exception as exc:
            logger.error(f"❌ {sym} failed: {exc}", exc_info=True)
            failed.append(sym)

    # ── Combine all symbols into the dual parquet (model_trainer.py needs this) ──
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('zone_time').reset_index(drop=True)

        dual_path = "data/labels/training_labels_dual.parquet"
        combined.to_parquet(dual_path, compression='snappy', index=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"COMBINED LABEL FILE")
        logger.info(f"{'='*60}")
        logger.info(f"  Total rows:   {len(combined):,}")
        logger.info(f"  Symbols:      {combined['symbol'].value_counts().to_dict()}")
        logger.info(f"  Date range:   {pd.to_datetime(combined['zone_time']).min().date()} → "
                    f"{pd.to_datetime(combined['zone_time']).max().date()}")
        logger.info(f"  Reversion WR: {combined['label'].mean():.1%}")
        logger.info(f"  Saved →       {dual_path}")
        logger.info(f"{'='*60}")

        if failed:
            logger.warning(f"⚠️  Failed symbols (excluded from combined): {failed}")

        logger.info("\n🔥 Next steps:")
        logger.info("  python model_trainer.py")
        logger.info("  python generate_outcomes.py")
    else:
        logger.error("❌ No labels generated for any symbol — check smc_detector output")


if __name__ == "__main__":
    main()