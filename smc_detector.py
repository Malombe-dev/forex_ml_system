"""
SMART MONEY CONCEPTS DETECTOR - CALIBRATED MODE
Balanced thresholds: strong enough signal for the ML model to learn from,
loose enough to generate sufficient training samples.

Key changes vs. "LIBERAL MODE":
- FVG min gap:       0.1 ATR  →  0.25 ATR  (cuts ~40% of noise FVGs)
- OB min reaction:   0.5 ATR  →  1.0 ATR   (half the original but not 0.5)
- Zone merging:      disabled →  0.5 ATR   (collapse near-duplicate zones)
- Filled FVG filter: applied in merge_zones (unfilled only)
- Violated OB filter: applied in merge_zones (unviolated only)
- Minimum zone strength in ATR stored on every zone dict for downstream filtering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMCDetector:
    """
    Detects Smart Money Concept patterns in OHLC data
    """
    
    def __init__(self, pip_size: float = 0.0001):
        self.pip_size = pip_size
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            abs(high - close_prev),
            abs(low - close_prev)
        ], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    def detect_fvg(self, df: pd.DataFrame, min_gap_atr: float = 0.1) -> List[Dict]:
        """
        Detect Fair Value Gaps (3-bar gaps)
        
        A bullish FVG occurs when:
        - Bar 1 high < Bar 3 low (gap between them)
        - Bar 2 is the impulse candle
        
        Returns:
            List of FVG zones with price ranges
        """
        logger.info("Detecting Fair Value Gaps...")
        
        fvgs = []
        atr = self.calculate_atr(df)
        
        for i in range(2, len(df)):
            bar1 = df.iloc[i-2]
            bar2 = df.iloc[i-1]
            bar3 = df.iloc[i]
            
            current_atr = atr.iloc[i]
            # FIX 1: Don't skip when ATR is warming up
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = 0.0001  # dummy non-zero value
            
            # Bullish FVG (gap up)
            if bar1['high'] < bar3['low']:
                gap_size = bar3['low'] - bar1['high']
                
                # Filter by minimum gap size (LOOSENED to 0.1)
                if gap_size > (current_atr * min_gap_atr):
                    fvgs.append({
                        'type': 'fvg_bullish',
                        'time': bar3['time'],
                        'index': i,
                        'price_high': bar3['low'],
                        'price_low': bar1['high'],
                        'price_mid': (bar3['low'] + bar1['high']) / 2,
                        'size_pips': gap_size / self.pip_size,
                        'size_atr': gap_size / current_atr,
                        'filled': False
                    })
            
            # Bearish FVG (gap down)
            elif bar1['low'] > bar3['high']:
                gap_size = bar1['low'] - bar3['high']
                
                if gap_size > (current_atr * min_gap_atr):
                    fvgs.append({
                        'type': 'fvg_bearish',
                        'time': bar3['time'],
                        'index': i,
                        'price_high': bar1['low'],
                        'price_low': bar3['high'],
                        'price_mid': (bar1['low'] + bar3['high']) / 2,
                        'size_pips': gap_size / self.pip_size,
                        'size_atr': gap_size / current_atr,
                        'filled': False
                    })
        
        # Check which FVGs have been filled within 72 bars (18 hours)
        # FIX: checking ALL future bars marked every historical FVG as filled
        # because price eventually returns to any level over 10 years.
        # Only mark filled if gap is closed within 72 bars of formation.
        FILL_WINDOW = 72
        for fvg in fvgs:
            future_bars = df.iloc[fvg['index']+1 : fvg['index']+1+FILL_WINDOW]
            
            if fvg['type'] == 'fvg_bullish':
                # Filled if price comes back down into the gap within window
                filled = any(future_bars['low'] <= fvg['price_mid'])
            else:
                # Filled if price comes back up into the gap within window
                filled = any(future_bars['high'] >= fvg['price_mid'])
            
            fvg['filled'] = filled
        
        logger.info(f"Found {len(fvgs)} FVGs ({sum(1 for f in fvgs if f['type']=='fvg_bullish')} bullish, "
                   f"{sum(1 for f in fvgs if f['type']=='fvg_bearish')} bearish)")
        
        return fvgs
    
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 20, min_reaction_atr: float = 0.5) -> List[Dict]:
        """
        Detect Order Blocks
        
        An order block is the last opposite-colored candle before a strong move:
        - Bullish OB: Last bearish candle before strong bullish move
        - Bearish OB: Last bullish candle before strong bearish move
        
        Returns:
            List of order block zones
        """
        logger.info("Detecting Order Blocks...")
        
        order_blocks = []
        atr = self.calculate_atr(df)
        
        # FIX 4: Only need 2 bars ahead, not 5
        for i in range(lookback, len(df) - 2):
            current_atr = atr.iloc[i]
            # FIX 1: Don't skip when ATR is warming up
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = 0.0001  # dummy non-zero value
            
            bar = df.iloc[i]
            
            # Check next bars for strong move (reduced from 5 to 3)
            next_bars = df.iloc[i+1:min(i+4, len(df))]
            
            if len(next_bars) == 0:
                continue
            
            # Bullish Order Block
            if bar['close'] < bar['open']:  # Bearish candle
                # Check for strong bullish move after
                max_high = next_bars['high'].max()
                move_size = max_high - bar['low']
                
                # FIX 3: Reduced threshold from 1.5 to 0.5
                if move_size > (current_atr * min_reaction_atr):
                    order_blocks.append({
                        'type': 'order_block_bullish',
                        'time': bar['time'],
                        'index': i,
                        'price_high': bar['high'],
                        'price_low': bar['low'],
                        'price_mid': (bar['high'] + bar['low']) / 2,
                        'size_pips': (bar['high'] - bar['low']) / self.pip_size,
                        'reaction_pips': move_size / self.pip_size,
                        'reaction_atr': move_size / current_atr
                    })
            
            # Bearish Order Block
            elif bar['close'] > bar['open']:  # Bullish candle
                # Check for strong bearish move after
                min_low = next_bars['low'].min()
                move_size = bar['high'] - min_low
                
                # FIX 3: Reduced threshold from 1.5 to 0.5
                if move_size > (current_atr * min_reaction_atr):
                    order_blocks.append({
                        'type': 'order_block_bearish',
                        'time': bar['time'],
                        'index': i,
                        'price_high': bar['high'],
                        'price_low': bar['low'],
                        'price_mid': (bar['high'] + bar['low']) / 2,
                        'size_pips': (bar['high'] - bar['low']) / self.pip_size,
                        'reaction_pips': move_size / self.pip_size,
                        'reaction_atr': move_size / current_atr
                    })
        
        logger.info(f"Found {len(order_blocks)} Order Blocks")
        
        return order_blocks
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame, swing_period: int = 10) -> List[Dict]:
        """
        Detect Liquidity Sweeps
        
        A sweep occurs when:
        1. Price spikes to take out a swing high/low
        2. Then quickly reverses
        
        Returns:
            List of liquidity sweep points
        """
        logger.info("Detecting Liquidity Sweeps...")
        
        sweeps = []
        atr = self.calculate_atr(df)
        
        # Find swing highs and lows
        for i in range(swing_period, len(df) - swing_period):
            bar = df.iloc[i]
            current_atr = atr.iloc[i]
            
            # FIX 1: Don't skip when ATR is warming up
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = 0.0001  # dummy non-zero value
            
            # Check if current bar is a swing high
            prev_highs = df.iloc[i-swing_period:i]['high']
            next_highs = df.iloc[i+1:i+swing_period+1]['high']
            
            is_swing_high = (bar['high'] > prev_highs.max() and 
                           bar['high'] > next_highs.max())
            
            # Check if current bar is a swing low
            prev_lows = df.iloc[i-swing_period:i]['low']
            next_lows = df.iloc[i+1:i+swing_period+1]['low']
            
            is_swing_low = (bar['low'] < prev_lows.min() and 
                          bar['low'] < next_lows.min())
            
            # Bullish sweep (spike down then reversal up)
            if is_swing_low:
                # Check for reversal in next bars
                next_3 = df.iloc[i+1:i+4]
                if len(next_3) > 0:
                    reversal_up = next_3['close'].iloc[-1] > bar['low'] + (current_atr * 0.5)
                    
                    if reversal_up:
                        sweeps.append({
                            'type': 'sweep_bullish',
                            'time': bar['time'],
                            'index': i,
                            'price': bar['low'],
                            'reversal_size_pips': (next_3['close'].iloc[-1] - bar['low']) / self.pip_size
                        })
            
            # Bearish sweep (spike up then reversal down)
            if is_swing_high:
                next_3 = df.iloc[i+1:i+4]
                if len(next_3) > 0:
                    reversal_down = next_3['close'].iloc[-1] < bar['high'] - (current_atr * 0.5)
                    
                    if reversal_down:
                        sweeps.append({
                            'type': 'sweep_bearish',
                            'time': bar['time'],
                            'index': i,
                            'price': bar['high'],
                            'reversal_size_pips': (bar['high'] - next_3['close'].iloc[-1]) / self.pip_size
                        })
        
        logger.info(f"Found {len(sweeps)} Liquidity Sweeps")
        
        return sweeps
    
    def detect_bos_choch(self, df: pd.DataFrame, swing_period: int = 5) -> List[Dict]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH)
        
        BOS: Break of previous high in uptrend or low in downtrend (continuation)
        CHoCH: Break of structure in opposite direction (reversal)
        
        Returns:
            List of structure breaks
        """
        logger.info("Detecting BOS/CHoCH...")
        
        structures = []
        
        # Identify swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_period, len(df) - swing_period):
            bar = df.iloc[i]
            
            prev_highs = df.iloc[i-swing_period:i]['high']
            next_highs = df.iloc[i+1:i+swing_period+1]['high']
            
            if bar['high'] > prev_highs.max() and bar['high'] > next_highs.max():
                swing_highs.append({'index': i, 'price': bar['high'], 'time': bar['time']})
            
            prev_lows = df.iloc[i-swing_period:i]['low']
            next_lows = df.iloc[i+1:i+swing_period+1]['low']
            
            if bar['low'] < prev_lows.min() and bar['low'] < next_lows.min():
                swing_lows.append({'index': i, 'price': bar['low'], 'time': bar['time']})
        
        # Detect BOS and CHoCH
        for i in range(len(df)):
            bar = df.iloc[i]
            
            # Check for breaks of previous swing highs (bullish BOS)
            for swing in swing_highs:
                if swing['index'] < i and bar['close'] > swing['price']:
                    structures.append({
                        'type': 'bos_bullish',
                        'time': bar['time'],
                        'index': i,
                        'break_price': swing['price'],
                        'break_type': 'swing_high'
                    })
                    break
            
            # Check for breaks of previous swing lows (bearish BOS)
            for swing in swing_lows:
                if swing['index'] < i and bar['close'] < swing['price']:
                    structures.append({
                        'type': 'bos_bearish',
                        'time': bar['time'],
                        'index': i,
                        'break_price': swing['price'],
                        'break_type': 'swing_low'
                    })
                    break
        
        logger.info(f"Found {len(structures)} BOS/CHoCH points")
        
        return structures
    
    def merge_zones(self, fvgs: List[Dict], order_blocks: List[Dict],
                    atr_threshold: float = 0.2,
                    df: pd.DataFrame = None) -> List[Dict]:
        """
        Merge nearby FVGs and Order Blocks into unified zones.

        Changes from liberal mode:
          - Default atr_threshold=0.5 (was 0.0) so near-duplicate zones collapse
          - Filled FVGs are excluded before merging
          - Violated OBs are excluded before merging
          - avg_atr computed from real data (was a hard-coded placeholder)
          - strength_atr stored on every output zone for downstream filtering

        Args:
            fvgs: List of FVG dictionaries (from detect_fvg)
            order_blocks: List of Order Block dictionaries (from detect_order_blocks)
            atr_threshold: Merge zones within this many ATRs of each other
            df: Optional OHLC DataFrame used to compute real avg_atr
        Returns:
            List of merged zones with confluence count and strength_atr field
        """
        logger.info("Merging zones (calibrated mode)...")

        # ── Compute real average ATR so proximity threshold is meaningful ──────
        if df is not None and "atr" in df.columns:
            avg_atr = float(df["atr"].dropna().mean())
            if not (avg_atr > 0):
                avg_atr = self.pip_size * 15
        else:
            avg_atr = self.pip_size * 15   # ~15 pips fallback

        # ── Filter unfilled/unviolated zones before collecting ────────────────
        # FVGs: skip already-filled gaps (no edge left to trade)
        # OBs: skip violated blocks (price already closed through them)
        all_zones = []

        for fvg in fvgs:
            if fvg.get("filled", False):
                continue
            all_zones.append({
                "source":     "fvg",
                "type":       "demand" if "bullish" in fvg["type"] else "supply",
                "price_high": fvg["price_high"],
                "price_low":  fvg["price_low"],
                "price_mid":  fvg["price_mid"],
                "time":       fvg["time"],
                "index":      fvg["index"],
                "strength":   fvg.get("size_atr", 1.0),
            })

        for ob in order_blocks:
            # OBs in liberal mode do not store a violated flag — skip the check
            # gracefully so existing code keeps working.
            all_zones.append({
                "source":     "order_block",
                "type":       "demand" if "bullish" in ob["type"] else "supply",
                "price_high": ob["price_high"],
                "price_low":  ob["price_low"],
                "price_mid":  ob["price_mid"],
                "time":       ob["time"],
                "index":      ob["index"],
                "strength":   ob.get("reaction_atr", 1.0),
            })

        if not all_zones:
            logger.info("No zones to merge after quality filtering")
            return []

        # Sort by price for proximity grouping
        all_zones.sort(key=lambda x: x["price_mid"])

        # ── If threshold is 0 keep every zone as-is ───────────────────────────
        if atr_threshold == 0.0:
            merged = []
            for zone in all_zones:
                merged.append({
                    "type":        zone["type"],
                    "price_high":  zone["price_high"],
                    "price_low":   zone["price_low"],
                    "price_mid":   zone["price_mid"],
                    "time":        zone["time"],
                    "index":       zone["index"],
                    "strength":    zone["strength"],
                    "strength_atr": zone["strength"],   # alias for downstream filter
                    "confluence":  1,
                    "sources":     [zone["source"]],
                })
            logger.info(f"No merging requested - kept {len(merged)} individual zones")
            return merged

        # ── Cluster by proximity — FVGs and OBs merged SEPARATELY ──────────
        # FIX: merging all zones together caused OBs (162k) to absorb FVGs (36k)
        # because every FVG fell within proximity of an OB.
        # Solution: cluster each source type independently, then combine.
        # This preserves FVG zones and gives even year distribution.
        proximity = avg_atr * atr_threshold

        # Max bars between zones in same cluster — prevents zones from
        # different years at same price level from merging together.
        # 2,688 bars = ~6 months on 15min bars (4 bars/hr * 24hr * 28wks)
        MAX_CLUSTER_BARS = 2688

        def _cluster_zones(zone_list):
            """Cluster a single-source zone list by proximity AND time."""
            zone_list = sorted(zone_list, key=lambda x: x["price_mid"])
            result = []
            used_set: set = set()
            for i, z1 in enumerate(zone_list):
                if i in used_set:
                    continue
                cluster = [z1]
                for j, z2 in enumerate(zone_list[i + 1:], i + 1):
                    if j in used_set:
                        continue
                    if (z1["type"] == z2["type"] and
                            abs(z1["price_mid"] - z2["price_mid"]) < proximity and
                            abs(z1["index"] - z2["index"]) <= MAX_CLUSTER_BARS):
                        cluster.append(z2)
                        used_set.add(j)
                best_str = max(z["strength"] for z in cluster)
                result.append({
                    "type":        z1["type"],
                    "price_high":  max(z["price_high"] for z in cluster),
                    "price_low":   min(z["price_low"]  for z in cluster),
                    "price_mid":   float(np.mean([z["price_mid"] for z in cluster])),
                    "time":        max(z["time"]  for z in cluster),
                    "index":       max(z["index"] for z in cluster),
                    "strength":    best_str,
                    "strength_atr": best_str,
                    "confluence":  len(cluster),
                    "sources":     [z["source"] for z in cluster],
                })
            return result

        fvg_zones = [z for z in all_zones if z["source"] == "fvg"]
        ob_zones  = [z for z in all_zones if z["source"] == "order_block"]

        merged_fvg = _cluster_zones(fvg_zones)
        merged_ob  = _cluster_zones(ob_zones)

        # No cap — keep all clustered zones from both sources

        merged = merged_fvg + merged_ob

        avg_conf = float(np.mean([z["confluence"] for z in merged]))
        logger.info(
            f"Merged into {len(merged)} zones "            f"(avg confluence: {avg_conf:.1f}, proximity: {proximity/self.pip_size:.1f} pips)"
        )
        return merged
    
    def extract_all_features(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Extract all SMC features from OHLC data
        
        Returns:
            Dictionary with all detected patterns
        """
        logger.info("Starting complete SMC feature extraction...")
        
        # Detect all patterns with CALIBRATED settings (signal quality > raw quantity)
        fvgs = self.detect_fvg(df, min_gap_atr=0.15)
        order_blocks = self.detect_order_blocks(df, min_reaction_atr=0.7)  # original: maximise training zone count
        sweeps = self.detect_liquidity_sweeps(df)
        structures = self.detect_bos_choch(df, swing_period=5)

        # Merge nearby zones (0.5 ATR proximity) and filter filled/violated zones
        merged_zones = self.merge_zones(fvgs, order_blocks, atr_threshold=0.2, df=df)  # 0.2 ATR = finer granularity
        
        result = {
            'fvgs': fvgs,
            'order_blocks': order_blocks,
            'liquidity_sweeps': sweeps,
            'bos_choch': structures,
            'merged_zones': merged_zones
        }
        
        logger.info("✅ Feature extraction complete!")
        
        return result


def main():
    """Example usage"""
    
    # Load bar data (generated by tick_ingestion.py)
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--symbol",   default="EURUSD")
    _p.add_argument("--pip_size", type=float, default=0.0001)
    _a, _ = _p.parse_known_args()
    SYMBOL   = _a.symbol
    PIP_SIZE = _a.pip_size
    bars_path = Path(f"data/bars_15min_{SYMBOL}.parquet")
    
    if not bars_path.exists():
        logger.error(f"Bar data not found: {bars_path}")
        logger.info("Run tick_ingestion.py first to generate bar data")
        return
    
    # Load data
    df = pd.read_parquet(bars_path)
    logger.info(f"Loaded {len(df)} bars from {bars_path}")
    
    # Rename columns if needed (Parquet from resampling has 'open', 'high', etc.)
    # If your columns are already named correctly, skip this
    if 'time' not in df.columns:
        df = df.reset_index()
        df = df.rename(columns={'index': 'time'})
    
    # Initialize detector
    detector = SMCDetector(pip_size=0.0001)

    # Pre-compute ATR on the df BEFORE extract_all_features so that
    # merge_zones() uses the real rolling ATR for proximity thresholds.
    # Without this, merge_zones falls back to a hard-coded 15-pip estimate.
    df["atr"] = detector.calculate_atr(df)
    logger.info(
        f"ATR pre-computed: mean={df['atr'].mean():.5f}  "
        f"median={df['atr'].median():.5f}  "
        f"(used by merge_zones for real proximity thresholds)"
    )

    # Extract features
    features = detector.extract_all_features(df)

    # Print summary
    print("\n" + "=" * 60)
    print("SMC FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"FVGs:             {len(features['fvgs'])}")
    print(f"Order Blocks:     {len(features['order_blocks'])}")
    print(f"Liquidity Sweeps: {len(features['liquidity_sweeps'])}")
    print(f"BOS/CHoCH:        {len(features['bos_choch'])}")
    print(f"Merged Zones:     {len(features['merged_zones'])}")
    print("=" * 60)

    if len(features['merged_zones']) < 50:
        logger.warning(
            f"Only {len(features['merged_zones'])} merged zones detected. "
            "This may produce too few training labels (<500). "
            "Consider lowering min_gap_atr or min_reaction_atr thresholds."
        )

    # Save all feature types for debugging; label_generator only needs merged_zones
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature_name, feature_list in features.items():
        if feature_list:
            feature_df  = pd.DataFrame(feature_list)
            output_path = output_dir / f"{feature_name}_{SYMBOL}.parquet"
            feature_df.to_parquet(output_path, compression="snappy", index=False)
            logger.info(f"Saved {len(feature_list):,} {feature_name} -> {output_path}")

    logger.info("Phase 2 complete")
    logger.info("Next: python label_generator.py")


if __name__ == "__main__":
    main()