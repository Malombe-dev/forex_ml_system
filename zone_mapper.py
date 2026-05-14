#!/usr/bin/env python3
"""
Elite Zone Mapper – outputs ONLY the best high-probability SMC zones (LIVE)
Filters 80+ zones down to top 5-10 based on:
  1. Strength (ATR multiples)
  2. Freshness (recent formation)
  3. Distance from current price
  4. HTF confluence (aligns with higher timeframes)
  5. Touch count (fewer touches = fresher)
  
Uses TwelveData API for LIVE market data.

Usage:
  export TWELVE_API_KEY=your_key_here
  python elite_zone_mapper.py > elite_zones.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

API_KEY   = os.getenv("TWELVE_API_KEY")
SYMBOL    = "EUR/USD"
INTERVAL  = "15min"
BARS_BACK = 5000          # TwelveData max = 5000 (free tier)
PIP       = 0.0001

# ---------- FILTER SETTINGS (adjust these) -------------------------
MIN_STRENGTH_ATR = 1.0        # Minimum 1.0 ATR zone size
MAX_DISTANCE_PIPS = 250       # Within 250 pips of current price
MAX_AGE_BARS = 500            # Formed within last 500 bars (~5 days 15min)
MAX_ZONES_OUTPUT = 10         # Output top 10 zones only
MIN_SCORE = 5.0               # Minimum composite score (0-10 scale)

# ---------- helpers -------------------------------------------------
def fetch_bars(symbol, interval, outputsize):
    """Fetch live data from TwelveData API"""
    if not API_KEY:
        sys.exit("❌ Error: TWELVE_API_KEY not set. Run: export TWELVE_API_KEY=your_key")
    
    base_url = "https://api.twelvedata.com"
    endpoint = f"{base_url}/time_series"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON"
    }
    
    try:
        print(f"🔄 Fetching {outputsize} bars of {symbol} ({interval})...", file=sys.stderr)
        response = requests.get(endpoint, params=params, timeout=30)
        data = response.json()
        
        if "values" not in data:
            print(f"❌ API Error: {data.get('message', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)
        
        # Parse API response
        df_data = []
        for candle in data['values']:
            try:
                df_data.append({
                    'time': pd.to_datetime(candle['datetime']),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close'])
                })
            except Exception as e:
                continue
        
        if not df_data:
            print("❌ No valid data received from API", file=sys.stderr)
            sys.exit(1)
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('time').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} bars (from {df['time'].iloc[0]} to {df['time'].iloc[-1]})", file=sys.stderr)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error fetching data: {e}", file=sys.stderr)
        sys.exit(1)

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ---------- SMC detectors with metadata -----------------------------
def fvg_zones(df):
    """Bullish & bearish FVG with formation index"""
    zones = []
    atr_ = atr(df).iloc[-1] or 10*PIP
    for i in range(2, len(df)-1):
        b1, b2, b3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        
        # bullish FVG
        if b1["high"] < b3["low"] and b2["low"] > b1["high"]:
            top, bottom = b1["high"], b3["low"]
            filled = (df["close"].iloc[i+1:].between(bottom, top)).any()
            if not filled:
                zones.append(dict(
                    type="FVG_BULL",
                    top=top,
                    bottom=bottom,
                    mid=(top+bottom)/2,
                    strength=abs(top-bottom)/atr_,
                    status="virgin",
                    formation_idx=i,
                    age_bars=len(df) - i - 1,
                    touches=0  # virgin = 0 touches
                ))
        
        # bearish FVG
        if b1["low"] > b3["high"] and b2["high"] < b1["low"]:
            bottom, top = b1["low"], b3["high"]
            filled = (df["close"].iloc[i+1:].between(bottom, top)).any()
            if not filled:
                zones.append(dict(
                    type="FVG_BEAR",
                    top=top,
                    bottom=bottom,
                    mid=(top+bottom)/2,
                    strength=abs(top-bottom)/atr_,
                    status="virgin",
                    formation_idx=i,
                    age_bars=len(df) - i - 1,
                    touches=0
                ))
    return zones

def ob_zones(df):
    """Order-Blocks with formation index and touch count"""
    zones = []
    atr_ = atr(df).iloc[-1] or 10*PIP
    for i in range(5, len(df)-5):
        bar = df.iloc[i]
        
        # swing high (supply)
        if bar["high"] > df.iloc[i-5:i]["high"].max() and bar["high"] > df.iloc[i+1:i+6]["high"].max():
            drop = bar["high"] - df.iloc[i+1:i+6]["low"].min()
            if drop > atr_ * 0.5:
                top, bottom = bar["high"], bar["high"] - atr_*0.3
                filled = (df["close"].iloc[i+1:].between(bottom, top)).any()
                
                # count touches (how many times price came close)
                touches = ((df["high"].iloc[i+1:] >= bottom) & (df["low"].iloc[i+1:] <= top)).sum()
                
                if not filled:
                    zones.append(dict(
                        type="OB_SUPPLY",
                        top=top,
                        bottom=bottom,
                        mid=(top+bottom)/2,
                        strength=drop/atr_,
                        status="virgin",
                        formation_idx=i,
                        age_bars=len(df) - i - 1,
                        touches=touches
                    ))
        
        # swing low (demand)
        if bar["low"] < df.iloc[i-5:i]["low"].min() and bar["low"] < df.iloc[i+1:i+6]["low"].min():
            rise = df.iloc[i+1:i+6]["high"].max() - bar["low"]
            if rise > atr_ * 0.5:
                bottom, top = bar["low"], bar["low"] + atr_*0.3
                filled = (df["close"].iloc[i+1:].between(bottom, top)).any()
                
                touches = ((df["high"].iloc[i+1:] >= bottom) & (df["low"].iloc[i+1:] <= top)).sum()
                
                if not filled:
                    zones.append(dict(
                        type="OB_DEMAND",
                        top=top,
                        bottom=bottom,
                        mid=(top+bottom)/2,
                        strength=rise/atr_,
                        status="virgin",
                        formation_idx=i,
                        age_bars=len(df) - i - 1,
                        touches=touches
                    ))
    return zones

# ---------- scoring system ------------------------------------------
def score_zones(zones, df, current_price):
    """
    Score each zone on 0-10 scale based on:
      1. Strength (bigger = better)
      2. Freshness (recent = better)
      3. Distance (closer = better, but not too close)
      4. Touch count (fewer = better)
      5. HTF confluence (aligned with trend = better)
    """
    if not zones:
        return []
    
    # Calculate HTF trend (1H equivalent ~= last 4 bars on 15min)
    ema20 = ema(df["close"], 20).iloc[-1]
    ema50 = ema(df["close"], 50).iloc[-1]
    htf_trend = "bullish" if ema20 > ema50 else "bearish"
    
    for zone in zones:
        score = 0.0
        
        # 1. Strength score (0-3 points)
        # 1.0 ATR = 1pt, 2.0 ATR = 2pts, 3.0+ ATR = 3pts
        strength_score = min(zone["strength"], 3.0)
        score += strength_score
        
        # 2. Freshness score (0-2 points)
        # Recent zones better: <100 bars = 2pts, <300 = 1pt, >300 = 0pt
        if zone["age_bars"] < 100:
            freshness_score = 2.0
        elif zone["age_bars"] < 300:
            freshness_score = 1.0
        else:
            freshness_score = 0.0
        score += freshness_score
        
        # 3. Distance score (0-2 points)
        # Sweet spot: 20-150 pips = 2pts, 10-250 = 1pt, else 0pt
        distance_pips = abs(zone["mid"] - current_price) / PIP
        zone["distance_pips"] = distance_pips
        
        if 20 <= distance_pips <= 150:
            distance_score = 2.0
        elif 10 <= distance_pips <= 250:
            distance_score = 1.0
        else:
            distance_score = 0.0
        score += distance_score
        
        # 4. Touch score (0-1 point)
        # Virgin zones preferred: 0 touches = 1pt, 1-2 = 0.5pt, 3+ = 0pt
        if zone["touches"] == 0:
            touch_score = 1.0
        elif zone["touches"] <= 2:
            touch_score = 0.5
        else:
            touch_score = 0.0
        score += touch_score
        
        # 5. HTF confluence score (0-2 points)
        # Zone type aligned with trend = 2pts, against = 0pts
        is_bullish_zone = "BULL" in zone["type"] or "DEMAND" in zone["type"]
        if (is_bullish_zone and htf_trend == "bullish") or \
           (not is_bullish_zone and htf_trend == "bearish"):
            htf_score = 2.0
        else:
            htf_score = 0.0  # counter-trend zones get 0
        score += htf_score
        
        zone["score"] = round(score, 2)
        zone["strength_pts"] = round(strength_score, 1)
        zone["freshness_pts"] = round(freshness_score, 1)
        zone["distance_pts"] = round(distance_score, 1)
        zone["touch_pts"] = round(touch_score, 1)
        zone["htf_pts"] = round(htf_score, 1)
    
    return zones

def filter_elite_zones(zones):
    """Apply hard filters and return top N zones"""
    
    # Hard filters
    filtered = [z for z in zones if
                z["strength"] >= MIN_STRENGTH_ATR and
                z["distance_pips"] <= MAX_DISTANCE_PIPS and
                z["age_bars"] <= MAX_AGE_BARS and
                z["score"] >= MIN_SCORE]
    
    # Sort by score (highest first)
    filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)
    
    # Take top N
    return filtered[:MAX_ZONES_OUTPUT]

# ---------- main ----------------------------------------------------
def main():
    print("🔍 Elite Zone Mapper - Finding best LIVE zones...\n", file=sys.stderr)
    
    if not API_KEY:
        print("❌ Error: TWELVE_API_KEY not set", file=sys.stderr)
        print("   Run: export TWELVE_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)
    
    df = fetch_bars(SYMBOL, INTERVAL, BARS_BACK)
    current_price = df["close"].iloc[-1]
    current_time = df["time"].iloc[-1]
    
    print(f"\n📊 Market Data:", file=sys.stderr)
    print(f"   Symbol:        {SYMBOL}", file=sys.stderr)
    print(f"   Current Price: {current_price:.5f}", file=sys.stderr)
    print(f"   Latest Bar:    {current_time} GMT", file=sys.stderr)
    print(f"   Bars loaded:   {len(df)}", file=sys.stderr)
    
    # Detect all zones
    all_zones = fvg_zones(df) + ob_zones(df)
    print(f"\n🔍 Total zones detected: {len(all_zones)}", file=sys.stderr)
    
    # Score and filter
    scored_zones = score_zones(all_zones, df, current_price)
    elite_zones = filter_elite_zones(scored_zones)
    
    print(f"✅ Elite zones after filtering: {len(elite_zones)}", file=sys.stderr)
    print(f"\n{'='*80}", file=sys.stderr)
    print("FILTER SETTINGS:", file=sys.stderr)
    print(f"  Min Strength:     {MIN_STRENGTH_ATR} ATR", file=sys.stderr)
    print(f"  Max Distance:     {MAX_DISTANCE_PIPS} pips", file=sys.stderr)
    print(f"  Max Age:          {MAX_AGE_BARS} bars", file=sys.stderr)
    print(f"  Min Score:        {MIN_SCORE}/10", file=sys.stderr)
    print(f"  Max Output:       {MAX_ZONES_OUTPUT} zones", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)
    
    # Output CSV
    if not elite_zones:
        print("symbol,type,top,bottom,mid,strength,distance_pips,age_bars,touches,score,status")
        print("\n⚠️  No elite zones found with current filters", file=sys.stderr)
        print("   Try adjusting:", file=sys.stderr)
        print(f"   • MIN_SCORE (currently {MIN_SCORE})", file=sys.stderr)
        print(f"   • MAX_DISTANCE_PIPS (currently {MAX_DISTANCE_PIPS})", file=sys.stderr)
        print(f"   • MIN_STRENGTH_ATR (currently {MIN_STRENGTH_ATR})", file=sys.stderr)
        return
    
    print("symbol,type,top,bottom,mid,strength,distance_pips,age_bars,touches,score,strength_pts,freshness_pts,distance_pts,touch_pts,htf_pts,status")
    for z in elite_zones:
        print(f"{SYMBOL},{z['type']},{z['top']:.5f},{z['bottom']:.5f},{z['mid']:.5f},"
              f"{z['strength']:.2f},{z['distance_pips']:.1f},{z['age_bars']},"
              f"{z['touches']},{z['score']:.1f},{z['strength_pts']:.1f},"
              f"{z['freshness_pts']:.1f},{z['distance_pts']:.1f},{z['touch_pts']:.1f},"
              f"{z['htf_pts']:.1f},{z['status']}")
    
    # Print summary to stderr
    print(f"\n📊 ELITE ZONES SUMMARY ({current_time} GMT):", file=sys.stderr)
    for i, z in enumerate(elite_zones, 1):
        emoji = "🟢" if "BULL" in z["type"] or "DEMAND" in z["type"] else "🔴"
        print(f"{emoji} #{i} {z['type']:12s} @ {z['mid']:.5f} | "
              f"Score: {z['score']:.1f}/10 | Dist: {z['distance_pips']:>6.1f} pips | "
              f"Strength: {z['strength']:.1f} ATR | Age: {z['age_bars']} bars",
              file=sys.stderr)
    
    print(f"\n💡 Feed these {len(elite_zones)} zones to your ML predictor!", file=sys.stderr)
    print(f"🔄 Re-run this script every 15-30 minutes for updated zones", file=sys.stderr)

if __name__ == "__main__":
    main()