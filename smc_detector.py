"""
MULTI-TIMEFRAME SMC DETECTOR  v2
=================================
Institutional SMC hierarchy:

  D1  → trend bias only (EMA20/50 alignment)          [was already done]
  4H  → PRIMARY zone detection (FVGs, OBs)             [NEW — institutional TF]
  1H  → zone alignment / confirmation                  [NEW — refines 4H zones]
  15min → entry precision, features, outcome sim       [unchanged]

Why 4H for zone detection:
  - 15min FVGs / OBs are retail noise (too small, too many)
  - 4H candles represent ~4 hours of institutional order flow
  - A bearish 4H OB that also has a 1H FVG inside it = high-confluence zone
  - These are the levels hedge funds and banks defend

Zone pipeline:
  1. Detect FVGs + OBs on 4H bars                     → raw 4H zones
  2. For each 4H zone, check if a 1H FVG/OB exists
     inside or within 0.5 ATR of that zone             → adds confluence flag
  3. Map each confirmed zone to its 15min bar index    → for label_generator
  4. D1 EMA trend is stored per zone for alignment     → htf_trend_alignment

Output: data/features/merged_zones_{SYMBOL}.parquet
  Same schema as before + new fields:
    tf_source       : '4h' | '1h_confirmed' | '4h_1h_confluence'
    h1_confluence   : bool  — 1H FVG/OB found inside 4H zone
    d1_trend        : 'up' | 'down' | 'neutral'
    h4_strength_atr : reaction size in 4H ATR units (bigger = stronger)

Usage:
  python smc_detector.py --symbol EURUSD --pip_size 0.0001
  python smc_detector.py --symbol XAUUSD --pip_size 1.0

Requires bar files:
  data/bars_4h_{SYMBOL}.parquet     ← primary detection TF
  data/bars_1h_{SYMBOL}.parquet     ← confirmation TF
  data/bars_15min_{SYMBOL}.parquet  ← entry TF (index mapping)
  data/bars_1d_{SYMBOL}.parquet     ← trend TF (optional)

If 4H/1H bars don't exist, they are auto-resampled from 15min bars.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
# 4H detection thresholds (intentionally stricter than 15min — we want
# fewer, higher-quality institutional zones, not thousands of retail ones)
H4_FVG_MIN_GAP_ATR      = 0.3    # gap must be ≥ 30% of 4H ATR
H4_OB_MIN_REACTION_ATR  = 1.0    # reaction must be ≥ 1× 4H ATR
H4_MERGE_ATR_THRESHOLD  = 0.5    # merge zones within 0.5× 4H ATR

# 1H confirmation thresholds (looser — we're checking INSIDE an existing 4H zone)
H1_FVG_MIN_GAP_ATR      = 0.3  # raised from 0.2 to match 4H quality threshold
H1_OB_MIN_REACTION_ATR  = 1.5
H1_CONFIRMATION_WINDOW_ATR = 0.0  # 1H zone must be INSIDE the 4H zone price range (strict)

MAX_CLUSTER_BARS_4H     = 336    # ~6 months on 4H bars (2 bars/day * 5days * 28wks)
FILL_WINDOW_4H          = 18     # 18 × 4H bars = 72 hours to check FVG fill


class SMCDetector:
    """
    Multi-timeframe SMC zone detector.

    Primary detection on 4H, confirmed by 1H, entries on 15min.
    """

    def __init__(self, pip_size: float = 0.0001):
        self.pip_size = pip_size

    # ─────────────────────────────────────────────────────────────────────────
    # ATR
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high       = df['high']
        low        = df['low']
        close_prev = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            abs(high - close_prev),
            abs(low - close_prev),
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # Resample helpers — build 4H/1H from 15min if not already on disk
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def resample_bars(df_15min: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample 15min OHLCV to any higher timeframe.

        Args:
            df_15min : DataFrame with columns [time, open, high, low, close, volume]
            rule     : pandas offset alias — '4h', '1h', '1d' etc.
        """
        df = df_15min.copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').sort_index()

        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        if 'volume' in df.columns:
            agg['volume'] = 'sum'

        resampled = df.resample(rule, label='left', closed='left').agg(agg).dropna()
        resampled = resampled.reset_index().rename(columns={'time': 'time'})
        return resampled

    # ─────────────────────────────────────────────────────────────────────────
    # FVG detection (works on any timeframe)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_fvg(self,
                   df: pd.DataFrame,
                   min_gap_atr: float = 0.25,
                   fill_window: int = 72,
                   tf_label: str = '4h') -> List[Dict]:
        """
        Detect Fair Value Gaps on the given DataFrame.

        A bullish FVG: bar[i-2].high < bar[i].low  (gap between candles 1 and 3)
        A bearish FVG: bar[i-2].low  > bar[i].high

        fill_window: bars to look forward to mark as filled (default=72 for 15min,
                     use 18 for 4H, 72 for 1H).
        """
        logger.info(f"Detecting FVGs on {tf_label} bars (min_gap={min_gap_atr}×ATR)...")

        fvgs = []
        atr  = self.calculate_atr(df)

        for i in range(2, len(df)):
            bar1 = df.iloc[i - 2]
            bar3 = df.iloc[i]

            current_atr = float(atr.iloc[i])
            if not np.isfinite(current_atr) or current_atr <= 0:
                current_atr = self.pip_size * 15

            # Bullish FVG
            if bar1['high'] < bar3['low']:
                gap_size = bar3['low'] - bar1['high']
                if gap_size >= min_gap_atr * current_atr:
                    fvgs.append({
                        'type':       'fvg_bullish',
                        'time':       bar3['time'],
                        'index':      i,
                        'price_high': bar3['low'],
                        'price_low':  bar1['high'],
                        'price_mid':  (bar3['low'] + bar1['high']) / 2,
                        'size_pips':  gap_size / self.pip_size,
                        'size_atr':   gap_size / current_atr,
                        'filled':     False,
                        'tf':         tf_label,
                    })

            # Bearish FVG
            elif bar1['low'] > bar3['high']:
                gap_size = bar1['low'] - bar3['high']
                if gap_size >= min_gap_atr * current_atr:
                    fvgs.append({
                        'type':       'fvg_bearish',
                        'time':       bar3['time'],
                        'index':      i,
                        'price_high': bar1['low'],
                        'price_low':  bar3['high'],
                        'price_mid':  (bar1['low'] + bar3['high']) / 2,
                        'size_pips':  gap_size / self.pip_size,
                        'size_atr':   gap_size / current_atr,
                        'filled':     False,
                        'tf':         tf_label,
                    })

        # Mark fills within the fill window
        for fvg in fvgs:
            future = df.iloc[fvg['index'] + 1 : fvg['index'] + 1 + fill_window]
            if fvg['type'] == 'fvg_bullish':
                fvg['filled'] = bool(any(future['low'] <= fvg['price_mid']))
            else:
                fvg['filled'] = bool(any(future['high'] >= fvg['price_mid']))

        n_unfilled = sum(1 for f in fvgs if not f['filled'])
        logger.info(
            f"  {tf_label} FVGs: {len(fvgs)} total, {n_unfilled} unfilled "
            f"({sum(1 for f in fvgs if 'bullish' in f['type'])} bull / "
            f"{sum(1 for f in fvgs if 'bearish' in f['type'])} bear)"
        )
        return fvgs

    # ─────────────────────────────────────────────────────────────────────────
    # Order Block detection (works on any timeframe)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_order_blocks(self,
                            df: pd.DataFrame,
                            lookback: int = 20,
                            min_reaction_atr: float = 1.0,
                            tf_label: str = '4h') -> List[Dict]:
        """
        Detect Order Blocks on the given DataFrame.

        A bullish OB = last bearish candle before a strong bullish move (≥ min_reaction_atr).
        A bearish OB = last bullish candle before a strong bearish move.

        Using min_reaction_atr=1.0 on 4H means the subsequent move must be
        ≥ 1 full 4H ATR, filtering out weak / noise reactions.
        """
        logger.info(f"Detecting Order Blocks on {tf_label} bars (min_reaction={min_reaction_atr}×ATR)...")

        order_blocks = []
        atr          = self.calculate_atr(df)

        for i in range(lookback, len(df) - 3):
            current_atr = float(atr.iloc[i])
            if not np.isfinite(current_atr) or current_atr <= 0:
                current_atr = self.pip_size * 15

            bar       = df.iloc[i]
            next_bars = df.iloc[i + 1 : i + 4]

            if len(next_bars) == 0:
                continue

            # Bullish OB — last bearish candle before strong up move
            if bar['close'] < bar['open']:
                max_high  = float(next_bars['high'].max())
                move_size = max_high - bar['low']
                if move_size >= min_reaction_atr * current_atr:
                    order_blocks.append({
                        'type':          'order_block_bullish',
                        'time':          bar['time'],
                        'index':         i,
                        'price_high':    bar['high'],
                        'price_low':     bar['low'],
                        'price_mid':     (bar['high'] + bar['low']) / 2,
                        'size_pips':     (bar['high'] - bar['low']) / self.pip_size,
                        'reaction_pips': move_size / self.pip_size,
                        'reaction_atr':  move_size / current_atr,
                        'tf':            tf_label,
                    })

            # Bearish OB — last bullish candle before strong down move
            elif bar['close'] > bar['open']:
                min_low   = float(next_bars['low'].min())
                move_size = bar['high'] - min_low
                if move_size >= min_reaction_atr * current_atr:
                    order_blocks.append({
                        'type':          'order_block_bearish',
                        'time':          bar['time'],
                        'index':         i,
                        'price_high':    bar['high'],
                        'price_low':     bar['low'],
                        'price_mid':     (bar['high'] + bar['low']) / 2,
                        'size_pips':     (bar['high'] - bar['low']) / self.pip_size,
                        'reaction_pips': move_size / self.pip_size,
                        'reaction_atr':  move_size / current_atr,
                        'tf':            tf_label,
                    })

        logger.info(f"  {tf_label} Order Blocks: {len(order_blocks)}")
        return order_blocks

    # ─────────────────────────────────────────────────────────────────────────
    # Liquidity sweeps (15min only — too noisy on 4H)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_liquidity_sweeps(self,
                                df: pd.DataFrame,
                                swing_period: int = 10) -> List[Dict]:
        logger.info("Detecting Liquidity Sweeps on 15min bars...")
        sweeps = []
        atr    = self.calculate_atr(df)

        for i in range(swing_period, len(df) - swing_period):
            bar         = df.iloc[i]
            current_atr = float(atr.iloc[i])
            if not np.isfinite(current_atr) or current_atr <= 0:
                current_atr = self.pip_size * 15

            prev_highs = df.iloc[i - swing_period : i]['high']
            next_highs = df.iloc[i + 1 : i + swing_period + 1]['high']
            prev_lows  = df.iloc[i - swing_period : i]['low']
            next_lows  = df.iloc[i + 1 : i + swing_period + 1]['low']

            is_swing_high = (bar['high'] > prev_highs.max() and
                             bar['high'] > next_highs.max())
            is_swing_low  = (bar['low'] < prev_lows.min() and
                             bar['low'] < next_lows.min())

            next_3 = df.iloc[i + 1 : i + 4]
            if len(next_3) == 0:
                continue

            if is_swing_low:
                if next_3['close'].iloc[-1] > bar['low'] + current_atr:
                    sweeps.append({
                        'type':                'sweep_bullish',
                        'time':                bar['time'],
                        'index':               i,
                        'price':               bar['low'],
                        'reversal_size_pips':  (next_3['close'].iloc[-1] - bar['low']) / self.pip_size,
                    })

            if is_swing_high:
                if next_3['close'].iloc[-1] < bar['high'] - current_atr:
                    sweeps.append({
                        'type':                'sweep_bearish',
                        'time':                bar['time'],
                        'index':               i,
                        'price':               bar['high'],
                        'reversal_size_pips':  (bar['high'] - next_3['close'].iloc[-1]) / self.pip_size,
                    })

        logger.info(f"  Liquidity Sweeps: {len(sweeps)}")
        return sweeps

    # ─────────────────────────────────────────────────────────────────────────
    # BOS / CHoCH (15min only — structure breaks most useful for entries)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_bos_choch(self,
                         df: pd.DataFrame,
                         swing_period: int = 5) -> List[Dict]:
        logger.info("Detecting BOS/CHoCH...")
        structures  = []
        swing_highs = []
        swing_lows  = []

        for i in range(swing_period, len(df) - swing_period):
            bar        = df.iloc[i]
            prev_highs = df.iloc[i - swing_period : i]['high']
            next_highs = df.iloc[i + 1 : i + swing_period + 1]['high']
            prev_lows  = df.iloc[i - swing_period : i]['low']
            next_lows  = df.iloc[i + 1 : i + swing_period + 1]['low']

            if bar['high'] > prev_highs.max() and bar['high'] > next_highs.max():
                swing_highs.append({'index': i, 'price': bar['high'], 'time': bar['time']})
            if bar['low'] < prev_lows.min() and bar['low'] < next_lows.min():
                swing_lows.append({'index': i, 'price': bar['low'],  'time': bar['time']})

        for i in range(len(df)):
            bar = df.iloc[i]
            for swing in swing_highs:
                if swing['index'] < i and bar['close'] > swing['price']:
                    structures.append({'type': 'bos_bullish', 'time': bar['time'],
                                       'index': i, 'break_price': swing['price']})
                    break
            for swing in swing_lows:
                if swing['index'] < i and bar['close'] < swing['price']:
                    structures.append({'type': 'bos_bearish', 'time': bar['time'],
                                       'index': i, 'break_price': swing['price']})
                    break

        logger.info(f"  BOS/CHoCH: {len(structures)}")
        return structures

    # ─────────────────────────────────────────────────────────────────────────
    # D1 trend per zone
    # ─────────────────────────────────────────────────────────────────────────

    def get_d1_trend(self,
                     zone_time: pd.Timestamp,
                     df_d1: Optional[pd.DataFrame]) -> str:
        """
        Returns 'up', 'down', or 'neutral' based on D1 EMA20/50 at zone_time.
        """
        if df_d1 is None or df_d1.empty:
            return 'neutral'
        try:
            zone_time = pd.to_datetime(zone_time)
        except Exception:
            return 'neutral'

        col = 'time' if 'time' in df_d1.columns else df_d1.index.name
        if col and col in df_d1.columns:
            d1_before = df_d1[pd.to_datetime(df_d1[col]) <= zone_time]
        else:
            d1_before = df_d1[df_d1.index <= zone_time]

        if len(d1_before) < 50:
            return 'neutral'

        ema20 = float(d1_before['close'].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(d1_before['close'].ewm(span=50, adjust=False).mean().iloc[-1])

        if ema20 > ema50:
            return 'up'
        elif ema20 < ema50:
            return 'down'
        return 'neutral'

    # ─────────────────────────────────────────────────────────────────────────
    # 1H confirmation — does a 1H FVG or OB exist inside/near a 4H zone?
    # ─────────────────────────────────────────────────────────────────────────

    def check_h1_confirmation(self,
                              zone: Dict,
                              h1_fvgs: List[Dict],
                              h1_obs: List[Dict],
                              h4_atr: float) -> Tuple[bool, str]:
        """
        Returns (confirmed: bool, detail: str).

        A 4H zone is H1-confirmed if a 1H FVG or OB exists:
          - whose price_mid falls inside the 4H zone, OR
          - within H1_CONFIRMATION_WINDOW_ATR × 4H ATR of the 4H zone's price_mid

        This ensures the 1H structure is inside or directly adjacent to
        the 4H institutional level, not just somewhere on the chart.
        """
        z_low     = zone['price_low']
        z_high    = zone['price_high']
        zone_time = pd.to_datetime(zone.get('time'))
        max_time  = zone_time + pd.Timedelta(hours=80)

        for h1z in h1_fvgs:
            if h1z.get('filled', False):
                continue
            h1_time = pd.to_datetime(h1z.get('time'))
            if h1_time < zone_time or h1_time > max_time:
                continue
            if z_low <= h1z['price_mid'] <= z_high:
                return True, f"h1_fvg_{h1z['type']}"

        for h1z in h1_obs:
            h1_time = pd.to_datetime(h1z.get('time'))
            if h1_time < zone_time or h1_time > max_time:
                continue
            if z_low <= h1z['price_mid'] <= z_high:
                return True, f"h1_ob_{h1z['type']}"

        return False, 'none'

    # ─────────────────────────────────────────────────────────────────────────
    # Map 4H zone time → 15min bar index (for label_generator)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def map_to_15min_index(zone_time: pd.Timestamp,
                           df_15min: pd.DataFrame) -> int:
        """
        Find the 15min bar index that corresponds to the 4H zone's timestamp.
        Returns the index of the first 15min bar at or after zone_time.
        """
        times = pd.to_datetime(df_15min['time'])
        mask  = times >= pd.to_datetime(zone_time)
        if mask.any():
            return int(mask.idxmax())
        return len(df_15min) - 1  # fallback: last bar

    # ─────────────────────────────────────────────────────────────────────────
    # Merge 4H zones (same logic as before, tuned for 4H scale)
    # ─────────────────────────────────────────────────────────────────────────

    def merge_zones(self,
                    fvgs: List[Dict],
                    order_blocks: List[Dict],
                    atr_threshold: float = 0.5,
                    df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Merge nearby 4H FVGs and OBs into unified zones.
        Unfilled FVGs and all OBs are included; filled FVGs are dropped.
        """
        logger.info("Merging 4H zones...")

        avg_atr = (float(df['atr'].dropna().mean())
                   if df is not None and 'atr' in df.columns
                   else self.pip_size * 60)   # ~60-pip fallback for 4H

        all_zones = []
        for fvg in fvgs:
            if fvg.get('filled', False):
                continue
            all_zones.append({
                'source':     'fvg',
                'type':       'demand' if 'bullish' in fvg['type'] else 'supply',
                'price_high': fvg['price_high'],
                'price_low':  fvg['price_low'],
                'price_mid':  fvg['price_mid'],
                'time':       fvg['time'],
                'index':      fvg['index'],
                'strength':   fvg.get('size_atr', 1.0),
            })
        for ob in order_blocks:
            all_zones.append({
                'source':     'order_block',
                'type':       'demand' if 'bullish' in ob['type'] else 'supply',
                'price_high': ob['price_high'],
                'price_low':  ob['price_low'],
                'price_mid':  ob['price_mid'],
                'time':       ob['time'],
                'index':      ob['index'],
                'strength':   ob.get('reaction_atr', 1.0),
            })

        if not all_zones:
            logger.info("No 4H zones to merge")
            return []

        proximity = avg_atr * atr_threshold
        all_zones.sort(key=lambda x: x['price_mid'])

        fvg_zones = [z for z in all_zones if z['source'] == 'fvg']
        ob_zones  = [z for z in all_zones if z['source'] == 'order_block']

        def _cluster(zone_list):
            zone_list = sorted(zone_list, key=lambda x: x['price_mid'])
            result, used = [], set()
            for i, z1 in enumerate(zone_list):
                if i in used:
                    continue
                cluster = [z1]
                for j, z2 in enumerate(zone_list[i + 1:], i + 1):
                    if j in used:
                        continue
                    if (z1['type'] == z2['type'] and
                            abs(z1['price_mid'] - z2['price_mid']) < proximity and
                            abs(z1['index'] - z2['index']) <= MAX_CLUSTER_BARS_4H):
                        cluster.append(z2)
                        used.add(j)
                result.append({
                    'type':        z1['type'],
                    'price_high':  max(z['price_high'] for z in cluster),
                    'price_low':   min(z['price_low']  for z in cluster),
                    'price_mid':   float(np.mean([z['price_mid'] for z in cluster])),
                    'time':        max(z['time']  for z in cluster),
                    'index':       max(z['index'] for z in cluster),
                    'strength':    max(z['strength'] for z in cluster),
                    'strength_atr': max(z['strength'] for z in cluster),
                    'confluence':  len(cluster),
                    'sources':     [z['source'] for z in cluster],
                })
            return result

        merged = _cluster(fvg_zones) + _cluster(ob_zones)
        # ── Post-merge quality filters ─────────────────────────────────────
        before = len(merged)
        merged = [z for z in merged
                  if (z['price_high'] - z['price_low']) / self.pip_size >= 2.0]
        merged = [z for z in merged
                  if (z['price_high'] - z['price_low']) / self.pip_size <= 300.0]
        dropped = before - len(merged)
        if dropped:
            logger.info(f"  Dropped {dropped} zones outside [2, 300] pip range")
        logger.info(f"  Merged into {len(merged)} 4H zones "
                    f"(proximity={proximity/self.pip_size:.1f}p)")
        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN: multi-timeframe extraction
    # ─────────────────────────────────────────────────────────────────────────

    def extract_all_features_mtf(self,
                                 df_4h:    pd.DataFrame,
                                 df_1h:    pd.DataFrame,
                                 df_15min: pd.DataFrame,
                                 df_d1:    Optional[pd.DataFrame] = None) -> Dict:
        """
        Full multi-timeframe SMC extraction.

        Steps:
          1. Detect FVGs + OBs on 4H → primary institutional zones
          2. Detect FVGs + OBs on 1H → confirmation layer
          3. Merge 4H zones
          4. For each merged 4H zone:
             a. Check 1H confirmation (FVG/OB within zone bounds)
             b. Get D1 trend bias
             c. Map 4H bar index → 15min bar index (for label_generator)
             d. Compute h4_strength_atr
          5. Detect sweeps + BOS/CHoCH on 15min (entry-level structure)

        Returns dict with same keys as old extract_all_features() + mtf_ extras.
        """
        logger.info("=" * 70)
        logger.info("MULTI-TIMEFRAME SMC EXTRACTION")
        logger.info(f"  4H bars:   {len(df_4h):,}")
        logger.info(f"  1H bars:   {len(df_1h):,}")
        logger.info(f"  15min bars:{len(df_15min):,}")
        logger.info(f"  D1 bars:   {len(df_d1):,}" if df_d1 is not None else "  D1 bars:   not provided")
        logger.info("=" * 70)

        # ── Step 1: 4H detection ──────────────────────────────────────────────
        df_4h = df_4h.copy()
        df_4h['atr'] = self.calculate_atr(df_4h)
        h4_fvgs = self.detect_fvg(df_4h,
                                   min_gap_atr=H4_FVG_MIN_GAP_ATR,
                                   fill_window=FILL_WINDOW_4H,
                                   tf_label='4h')
        h4_obs  = self.detect_order_blocks(df_4h,
                                           min_reaction_atr=H4_OB_MIN_REACTION_ATR,
                                           tf_label='4h')

        # ── Step 2: 1H detection (confirmation layer) ────────────────────────
        df_1h = df_1h.copy()
        df_1h['atr'] = self.calculate_atr(df_1h)
        h1_fvgs = self.detect_fvg(df_1h,
                                   min_gap_atr=H1_FVG_MIN_GAP_ATR,
                                   fill_window=72,
                                   tf_label='1h')
        h1_obs  = self.detect_order_blocks(df_1h,
                                           min_reaction_atr=H1_OB_MIN_REACTION_ATR,
                                           tf_label='1h')

        # ── Step 3: Merge 4H zones ────────────────────────────────────────────
        merged_4h = self.merge_zones(h4_fvgs, h4_obs,
                                     atr_threshold=H4_MERGE_ATR_THRESHOLD,
                                     df=df_4h)

        # ── Step 4: Annotate each 4H zone ─────────────────────────────────────
        h4_avg_atr = float(df_4h['atr'].dropna().mean())
        annotated_zones = []
        n_h1_confirmed  = 0
        n_d1_aligned    = 0
        _fvg_match_ages: List[int] = []  # DIAGNOSTIC: bars between zone and its matched FVG
        _h1_sweep_magnitudes: List[float] = []  # DIAGNOSTIC: sweep excursion size / ATR

        for zone in merged_4h:
            zone_time = pd.to_datetime(zone['time'])

            # 4a. 1H confirmation
            h1_confirmed, h1_detail = self.check_h1_confirmation(
                zone, h1_fvgs, h1_obs, h4_avg_atr
            )
            if h1_confirmed:
                n_h1_confirmed += 1

            # 4b. D1 trend
            d1_trend = self.get_d1_trend(zone_time, df_d1)

            # D1 alignment: demand zone needs uptrend, supply needs downtrend
            if zone['type'] == 'demand':
                d1_aligned = 1 if d1_trend == 'up' else 0
            else:
                d1_aligned = 1 if d1_trend == 'down' else 0
            if d1_aligned:
                n_d1_aligned += 1

            # 4c. Map to 15min index
            idx_15min = self.map_to_15min_index(zone_time, df_15min)

            # 4d. TF source label
            if h1_confirmed:
                tf_source = '4h_1h_confluence'
            else:
                tf_source = '4h'

            # Boost confluence score for multi-TF confirmation
            confluence_boost = 1 if h1_confirmed else 0
            final_confluence = zone.get('confluence', 1) + confluence_boost

            # ── v7: detailed 1H features stored on zone for feature_builder ──
            h1_trend_val, h1_mss_val, h1_disp_val, h1_sweep_val, h1_sweep_mag = \
                self._compute_h1_zone_features(df_1h, zone_time, zone.get('type', 'demand'))
            if h1_sweep_mag is not None:
                _h1_sweep_magnitudes.append(h1_sweep_mag)

            # FVG overlap check (v9 alignment)
            # FIX: h4_fvgs is the FULL-HISTORY list detected once for the whole
            # dataset (see extract_all_features_mtf, computed before this loop
            # even starts) — it contains FVGs from before AND after this zone's
            # own formation time. Checking overlap against the unfiltered list
            # is look-ahead leakage: a zone could be flagged as "FVG-backed"
            # because of a 4H FVG that only formed months/years later, which
            # live trading could never have known about at signal time. This is
            # almost certainly why h4_fvg_inside is true for ~87% of zones and
            # dominates feature importance (45-60%) even after the feature_
            # builder.py circular-reference fix — the leak was one layer
            # upstream of that fix, here. Only FVGs already formed as of this
            # zone's own 4H bar index are eligible.
            fvg_overlap_flag = False
            if 'order_block' in zone.get('sources', []):
                _zone_4h_idx = zone.get('index', -1)
                _best_match_age = None  # bars between this zone and its matched FVG
                for fvg_item in h4_fvgs:
                    if fvg_item.get('index', -1) > _zone_4h_idx:
                        continue  # FVG hadn't formed yet at this zone's time — skip
                    fb = fvg_item.get('price_low', 0)
                    ft = fvg_item.get('price_high', 0)
                    if fb < zone['price_high'] and ft > zone['price_low']:
                        fvg_overlap_flag = True
                        _age = _zone_4h_idx - fvg_item.get('index', _zone_4h_idx)
                        if _best_match_age is None or _age < _best_match_age:
                            _best_match_age = _age
                        # NOTE: not breaking here (temporarily) so we find the
                        # CLOSEST match for the diagnostic below, not just the
                        # first one found in list order.
                if fvg_overlap_flag:
                    _fvg_match_ages.append(_best_match_age)

            annotated_zones.append({
                **zone,
                'index':            idx_15min,
                'index_4h':         zone['index'],
                'tf_source':        tf_source,
                'h1_confluence':    h1_confirmed,   # kept for backwards compat
                'h1_detail':        h1_detail,
                'd1_trend':         d1_trend,
                'htf_alignment':    d1_aligned,
                'confluence':       final_confluence,
                'fvg_overlap':      fvg_overlap_flag,
                # v7: granular 1H fields consumed by feature_builder
                'h1_trend_alignment':    h1_trend_val,
                'h1_structure_shift':    h1_mss_val,
                'h1_displacement':       h1_disp_val,
                'h1_liquidity_sweep':    h1_sweep_val,
            })

        # DIAGNOSTIC: distribution of how far back (in 4H bars) the matched
        # FVG was relative to the zone it confirmed. 1 bar = 4 hours, so
        # 6 bars/day. This tells us whether fvg_overlap is catching genuinely
        # recent/relevant FVGs or matching against arbitrarily old ones from
        # years earlier in the dataset (same price level revisited by chance).
        # DIAGNOSTIC: distribution of H1 liquidity sweep excursion sizes, in
        # ATR units. Bounding the window (tail(60), sp2=10) only reduced the
        # near-constant rate slightly (100% -> 98.7% in one run) — the
        # remaining near-constancy may mean most "sweeps" are real but
        # economically tiny (a fraction of a pip beyond the recent extreme).
        # If most magnitudes below are small, a minimum-magnitude threshold
        # (e.g. requiring the excursion to exceed some fraction of 1H ATR,
        # the same style already used for h1_displacement) would separate
        # genuine liquidity grabs from mechanical noise-crossings.
        if _h1_sweep_magnitudes:
            import numpy as _np
            _mags = _np.array(_h1_sweep_magnitudes)
            logger.info(f"\n📏 H1 sweep magnitude diagnostic (excursion size / ATR):")
            logger.info(f"  n sweeps detected: {len(_mags)}")
            logger.info(f"  min={_mags.min():.3f}  p10={_np.percentile(_mags,10):.3f}  "
                        f"p25={_np.percentile(_mags,25):.3f}  median={_np.median(_mags):.3f}  "
                        f"p75={_np.percentile(_mags,75):.3f}  max={_mags.max():.3f}")
            for _bound in [0.05, 0.10, 0.20, 0.30, 0.50]:
                _pct = (_mags < _bound).mean() * 100
                logger.info(f"  below {_bound:.2f}×ATR: {_pct:5.1f}% of detected sweeps")

        if _fvg_match_ages:
            import numpy as _np
            _ages = _np.array(_fvg_match_ages)
            logger.info(f"\n📏 FVG match-age diagnostic (bars back, 1 bar = 4h):")
            logger.info(f"  n matches: {len(_ages)}")
            logger.info(f"  min={_ages.min()}  p25={_np.percentile(_ages,25):.0f}  "
                        f"median={_np.median(_ages):.0f}  p75={_np.percentile(_ages,75):.0f}  "
                        f"p90={_np.percentile(_ages,90):.0f}  max={_ages.max()}")
            for _bound, _label in [(30, '<5 days'), (180, '<30 days'), (1080, '<6 months'), (2160, '<1 year')]:
                _pct = (_ages <= _bound).mean() * 100
                logger.info(f"  {_label:10s} (≤{_bound:5d} bars): {_pct:5.1f}%")

        logger.info(f"\n4H Zone Annotation Summary:")
        logger.info(f"  Total 4H zones:      {len(merged_4h)}")
        logger.info(f"  1H confirmed:        {n_h1_confirmed} "
                    f"({n_h1_confirmed/max(len(merged_4h),1)*100:.0f}%)")
        logger.info(f"  D1 trend aligned:    {n_d1_aligned} "
                    f"({n_d1_aligned/max(len(merged_4h),1)*100:.0f}%)")
        logger.info(f"  4H+1H confluence:   "
                    f"{sum(1 for z in annotated_zones if z['h1_confluence'])} zones")

        # ── Step 5: 15min entry-level structure ───────────────────────────────
        df_15min = df_15min.copy()
        df_bos   = df_15min.tail(50000).reset_index(drop=True)
        sweeps    = self.detect_liquidity_sweeps(df_15min)
        structures = self.detect_bos_choch(df_bos, swing_period=5)

        return {
            'merged_zones':    annotated_zones,    # primary output for label_generator
            'fvgs':            h4_fvgs,            # raw 4H FVGs
            'order_blocks':    h4_obs,             # raw 4H OBs
            'h1_fvgs':         h1_fvgs,            # 1H confirmation FVGs
            'h1_obs':          h1_obs,             # 1H confirmation OBs
            'liquidity_sweeps': sweeps,            # 15min sweeps
            'bos_choch':       structures,         # 15min structure
        }

    # ─────────────────────────────────────────────────────────────────────────
    # v7: Compute detailed 1H features per zone (used by feature_builder v7)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_h1_zone_features(
        self,
        df_1h: pd.DataFrame,
        zone_time: pd.Timestamp,
        zone_type: str,
    ) -> Tuple[int, int, float, int]:
        """
        Returns (h1_trend_alignment, h1_structure_shift, h1_displacement, h1_liquidity_sweep).
        All computed on 1H bars up to zone_time.

        h1_trend_alignment : -1=bearish, 0=neutral, 1=bullish (aligned to zone_type)
        h1_structure_shift : -1=bearish BOS, 0=none, 1=bullish BOS (last 30 1H bars)
        h1_displacement    : max impulse candle body / ATR in last 20 1H bars
        h1_liquidity_sweep : 1 if 1H swept swing low/high before reversal (last 30 bars)
        """
        defaults = (0, 0, 0.0, 0, None)
        try:
            sub = df_1h[pd.to_datetime(df_1h['time']) <= zone_time].copy()
            if len(sub) < 20:
                return defaults

            hi = sub['high']; lo = sub['low']; cl = sub['close']
            tr = pd.concat([
                hi - lo,
                (hi - cl.shift(1)).abs(),
                (lo - cl.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_1h = float(tr.rolling(14).mean().iloc[-1])
            if not np.isfinite(atr_1h) or atr_1h <= 0:
                atr_1h = float(tr.mean()) or 1e-6

            # trend alignment
            ema20 = float(cl.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(cl.ewm(span=50, adjust=False).mean().iloc[-1])
            if ema20 > ema50 * 1.0001:
                raw_trend = 1
            elif ema20 < ema50 * 0.9999:
                raw_trend = -1
            else:
                raw_trend = 0
            h1_trend = raw_trend if zone_type == 'demand' else -raw_trend

            # structure shift (BOS) in last 30 bars
            w = sub.tail(30).reset_index(drop=True)
            sp = 3
            swing_highs, swing_lows = [], []
            for i in range(sp, len(w) - sp):
                bar = w.iloc[i]
                if (bar['high'] > w.iloc[i-sp:i]['high'].max() and
                        bar['high'] > w.iloc[i+1:i+sp+1]['high'].max()):
                    swing_highs.append(bar['high'])
                if (bar['low'] < w.iloc[i-sp:i]['low'].min() and
                        bar['low'] < w.iloc[i+1:i+sp+1]['low'].min()):
                    swing_lows.append(bar['low'])
            last_cl = float(w['close'].iloc[-1])
            if swing_highs and last_cl > swing_highs[-1]:
                h1_mss = 1
            elif swing_lows and last_cl < swing_lows[-1]:
                h1_mss = -1
            else:
                h1_mss = 0

            # displacement: largest body / ATR in last 20 bars
            r20   = sub.tail(20)
            bodies = (r20['close'] - r20['open']).abs()
            h1_disp = float(min(bodies.max() / atr_1h, 10.0))

            # liquidity sweep — bounded window (v10 fix)
            # ROOT CAUSE (found via train/test consistency audit): this was
            # near-constant regardless of outer window size — tail(30) gave a
            # 97% hit rate (the original Bug 2), and "fixing" that by widening
            # to the FULL unbounded history gave 100% (still near-constant,
            # just differently broken). Neither extreme was the real problem:
            # sp2=4 is a 4-hour inner swing lookback, trivially broken by
            # ordinary noise almost every window regardless of how large the
            # search space is. Same root cause as m5_micro_bos/
            # m5_sweep_confirmation, fixed the same way: bound the outer
            # window to a reasonable recent context AND widen the inner
            # lookback so "swept a swing" requires a genuine move.
            # liquidity sweep — bounded window + minimum magnitude (v11 fix)
            # v10 fixed the window/lookback (see comment above) but the flag
            # was still near-constant (98.7% true) — measured via the
            # diagnostic added after v10: most detected "sweeps" were real
            # but economically trivial. Across all 4 symbols the magnitude
            # distribution was consistent (median 0.176-0.190×ATR, ~52-55%
            # of detected sweeps below 0.20×ATR). MIN_SWEEP_ATR=0.20 is
            # chosen from that measured median, not guessed — it roughly
            # halves the "true" rate, turning this into an actually
            # discriminating feature instead of a near-constant one.
            MIN_SWEEP_ATR = 0.20
            h1_sweep = 0
            h1_sweep_magnitude_atr = None  # DIAGNOSTIC: excursion size / ATR
            cw  = sub.tail(60).reset_index(drop=True)
            sp2 = 10
            for i in range(sp2, len(cw)):
                bar     = cw.iloc[i]
                prev_lo = cw.iloc[max(0, i-sp2):i]['low'].min()
                prev_hi = cw.iloc[max(0, i-sp2):i]['high'].max()
                if zone_type == 'demand' and bar['low'] < prev_lo and bar['close'] > prev_lo:
                    _mag = (prev_lo - bar['low']) / atr_1h
                    if _mag >= MIN_SWEEP_ATR:
                        h1_sweep = 1
                        h1_sweep_magnitude_atr = _mag
                        break
                if zone_type == 'supply' and bar['high'] > prev_hi and bar['close'] < prev_hi:
                    _mag = (bar['high'] - prev_hi) / atr_1h
                    if _mag >= MIN_SWEEP_ATR:
                        h1_sweep = 1
                        h1_sweep_magnitude_atr = _mag
                        break

            return (h1_trend, h1_mss, round(h1_disp, 4), h1_sweep, h1_sweep_magnitude_atr)

        except Exception as e:
            logger.warning(f"_compute_h1_zone_features failed: {e}")
            return defaults

    # ─────────────────────────────────────────────────────────────────────────
    # Backwards-compatible single-TF extraction (15min only, as before)
    # ─────────────────────────────────────────────────────────────────────────

    def extract_all_features(self, df: pd.DataFrame) -> Dict:
        """
        Legacy single-timeframe extraction (15min only).
        Use extract_all_features_mtf() for the full multi-TF pipeline.
        Kept for backwards compatibility with existing scripts.
        """
        logger.warning("Using single-TF extraction (15min only). "
                       "For institutional SMC quality use extract_all_features_mtf().")
        fvgs         = self.detect_fvg(df, min_gap_atr=0.15, tf_label='15min')
        order_blocks = self.detect_order_blocks(df, min_reaction_atr=0.7, tf_label='15min')
        sweeps       = self.detect_liquidity_sweeps(df)
        df_bos       = df.tail(50000).reset_index(drop=True)
        structures   = self.detect_bos_choch(df_bos, swing_period=5)
        merged_zones = self.merge_zones(fvgs, order_blocks, atr_threshold=0.2, df=df)
        return {
            'fvgs':             fvgs,
            'order_blocks':     order_blocks,
            'liquidity_sweeps': sweeps,
            'bos_choch':        structures,
            'merged_zones':     merged_zones,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Multi-timeframe SMC detector")
    _p.add_argument("--symbol",   default="EURUSD")
    _p.add_argument("--pip_size", type=float, default=0.0001)
    _a, _ = _p.parse_known_args()
    SYMBOL   = _a.symbol
    PIP_SIZE = _a.pip_size

    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = SMCDetector(pip_size=PIP_SIZE)

    # ── Load 15min bars (required) ────────────────────────────────────────────
    path_15 = Path(f"data/bars_15min_{SYMBOL}.parquet")
    if not path_15.exists():
        logger.error(f"15min bars not found: {path_15}")
        return
    df_15 = pd.read_parquet(path_15)
    if 'time' not in df_15.columns:
        df_15 = df_15.reset_index().rename(columns={'index': 'time'})
    df_15['time'] = pd.to_datetime(df_15['time'])
    logger.info(f"Loaded {len(df_15):,} 15min bars")

    # ── Load or resample 4H bars ──────────────────────────────────────────────
    path_4h = Path(f"data/bars_4h_{SYMBOL}.parquet")
    if path_4h.exists():
        df_4h = pd.read_parquet(path_4h)
        if 'time' not in df_4h.columns:
            df_4h = df_4h.reset_index().rename(columns={'index': 'time'})
        df_4h['time'] = pd.to_datetime(df_4h['time'])
        logger.info(f"Loaded {len(df_4h):,} 4H bars from {path_4h}")
    else:
        logger.info("4H bars not found — resampling from 15min...")
        df_4h = SMCDetector.resample_bars(df_15, '4h')
        df_4h.to_parquet(path_4h, index=False)
        logger.info(f"✅ Saved {len(df_4h):,} 4H bars → {path_4h}")

    # ── Load or resample 1H bars ──────────────────────────────────────────────
    path_1h = Path(f"data/bars_1h_{SYMBOL}.parquet")
    if path_1h.exists():
        df_1h = pd.read_parquet(path_1h)
        if 'time' not in df_1h.columns:
            df_1h = df_1h.reset_index().rename(columns={'index': 'time'})
        df_1h['time'] = pd.to_datetime(df_1h['time'])
        logger.info(f"Loaded {len(df_1h):,} 1H bars from {path_1h}")
    else:
        logger.info("1H bars not found — resampling from 15min...")
        df_1h = SMCDetector.resample_bars(df_15, '1h')
        df_1h.to_parquet(path_1h, index=False)
        logger.info(f"✅ Saved {len(df_1h):,} 1H bars → {path_1h}")

    # ── Load D1 bars (optional) ───────────────────────────────────────────────
    df_d1    = None
    path_d1  = Path(f"data/bars_1d_{SYMBOL}.parquet")
    path_d1b = Path(f"data/bars_d1_{SYMBOL}.parquet")
    for p in [path_d1, path_d1b]:
        if p.exists():
            df_d1 = pd.read_parquet(p)
            if 'time' not in df_d1.columns:
                df_d1 = df_d1.reset_index().rename(columns={'index': 'time'})
            df_d1['time'] = pd.to_datetime(df_d1['time'])
            logger.info(f"Loaded {len(df_d1):,} D1 bars from {p}")
            break
    if df_d1 is None:
        logger.warning("D1 bars not found — htf_alignment will be 0 for all zones")

    # ── Run multi-timeframe extraction ────────────────────────────────────────
    features = detector.extract_all_features_mtf(
        df_4h=df_4h, df_1h=df_1h, df_15min=df_15, df_d1=df_d1
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    zones = features['merged_zones']
    h1c   = sum(1 for z in zones if z.get('h1_confluence', False))
    d1a   = sum(1 for z in zones if z.get('htf_alignment', 0) == 1)

    print("\n" + "=" * 60)
    print("MULTI-TF SMC EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"4H FVGs detected:         {len(features['fvgs'])}")
    print(f"4H Order Blocks detected: {len(features['order_blocks'])}")
    print(f"4H Merged zones:          {len(zones)}")
    print(f"  ↳ with 1H confirmation: {h1c}  ({h1c/max(len(zones),1)*100:.0f}%)")
    print(f"  ↳ D1 trend aligned:     {d1a}  ({d1a/max(len(zones),1)*100:.0f}%)")
    print(f"  ↳ 4H+1H+D1 (all 3):     "
          f"{sum(1 for z in zones if z.get('h1_confluence') and z.get('htf_alignment'))}")
    print(f"15min Sweeps:             {len(features['liquidity_sweeps'])}")
    print(f"15min BOS/CHoCH:          {len(features['bos_choch'])}")
    print("=" * 60)

    if len(zones) < 50:
        logger.warning(
            f"Only {len(zones)} zones. Consider lowering H4_FVG_MIN_GAP_ATR "
            "or H4_OB_MIN_REACTION_ATR thresholds."
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    for name, data in features.items():
        if data:
            path = output_dir / f"{name}_{SYMBOL}.parquet"
            pd.DataFrame(data).to_parquet(path, compression='snappy', index=False)
            logger.info(f"Saved {len(data):,} {name} → {path}")

    # Primary output for label_generator
    primary = output_dir / f"merged_zones_{SYMBOL}.parquet"
    pd.DataFrame(zones).to_parquet(primary, compression='snappy', index=False)
    logger.info(f"\n✅ Primary zones → {primary}  ({len(zones)} zones)")
    logger.info("Next: python label_generator.py --symbol " + SYMBOL +
                " --pip_size " + str(PIP_SIZE))


if __name__ == "__main__":
    main()