"""
SIGNAL COMPARATOR
=================
Pulls live bars from Twelve Data and compares your predicted signals against
what price actually did — so you can audit entry quality and improve zone
detection / prediction logic.

USAGE
-----
    # Keys are read automatically from production_predictor.py
    python signal_comparator.py

    # Or pass a key manually
    python signal_comparator.py --api-key YOUR_KEY

    # Custom paths / options
    python signal_comparator.py --signals path/to/all_signals.json --interval 1h --bars 200
    python signal_comparator.py --out report.csv --save-bars --status pending

HOW SL / TP ARE CHECKED
------------------------
  Both use WICKS (high / low), not closes — matching real broker execution:

  BUY trade:
    SL triggered  → bar LOW  <= sl price   (wick pierced below SL)
    TP triggered  → bar HIGH >= tp price   (wick pierced above TP)

  SELL trade:
    SL triggered  → bar HIGH >= sl price   (wick pierced above SL)
    TP triggered  → bar LOW  <= tp price   (wick pierced below TP)

  When both SL and TP wicks appear on the same bar the earlier index wins.
  If they share the same bar index the trade is marked "sl_hit" (worst case).

OUTPUT
------
  Console  : rich table with per-signal breakdown + improvement hints
  CSV      : full row data (--out, default signal_report.csv)
  Bar CSVs : raw OHLCV bars per symbol (--save-bars)
"""

import argparse
import ast
import json
import re
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# AUTO-IMPORT API KEYS FROM production_predictor.py
# ---------------------------------------------------------------------------
def load_keys_from_predictor(predictor_path: str = "production_predictor.py") -> dict:
    """
    Parse PAIR_API_KEYS and _FALLBACK_KEY out of production_predictor.py
    without importing the module (avoids dependency on feature_builder etc).
    Returns dict like {"EUR/USD": "key...", "__fallback__": "key..."}
    """
    path = Path(predictor_path)
    if not path.exists():
        return {}
    src = path.read_text()

    keys = {}

    # Extract PAIR_API_KEYS dict literal
    m = re.search(r'PAIR_API_KEYS\s*=\s*(\{[^}]+\})', src, re.DOTALL)
    if m:
        try:
            keys.update(ast.literal_eval(m.group(1)))
        except Exception:
            pass

    # Extract _FALLBACK_KEY
    m2 = re.search(r'_FALLBACK_KEY\s*=\s*["\']([a-f0-9]+)["\']', src)
    if m2:
        keys["__fallback__"] = m2.group(1)

    return keys


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_URL   = "https://api.twelvedata.com/time_series"
RATE_LIMIT = 7          # stay under free-tier 8/min
_last_calls: list = []

PIP_SIZES = {
    "EUR/USD": 0.0001, "GBP/USD": 0.0001, "USD/CHF": 0.0001,
    "AUD/USD": 0.0001, "NZD/USD": 0.0001, "USD/CAD": 0.0001,
    "USD/JPY": 0.01,   "EUR/JPY": 0.01,   "GBP/JPY": 0.01,
    "XAU/USD": 1.0,    "XAG/USD": 0.01,
}

def pip_size(symbol: str) -> float:
    return PIP_SIZES.get(symbol.upper(), 0.0001)

def to_pips(price_diff: float, symbol: str) -> float:
    return abs(price_diff) / pip_size(symbol)


# ---------------------------------------------------------------------------
# RATE-LIMITED FETCH
# ---------------------------------------------------------------------------
def fetch_bars(symbol: str, interval: str, outputsize: int, api_key: str) -> Optional[pd.DataFrame]:
    global _last_calls
    now = time.time()
    _last_calls = [t for t in _last_calls if now - t < 60]
    if len(_last_calls) >= RATE_LIMIT:
        wait = 61 - (now - _last_calls[0])
        print(f"  ⏳ Rate limit — waiting {wait:.1f}s ...")
        time.sleep(max(0, wait))

    params = {
        "symbol": symbol, "interval": interval,
        "outputsize": outputsize, "apikey": api_key, "format": "JSON",
    }

    for attempt in range(3):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            _last_calls.append(time.time())

            if "values" not in data:
                print(f"  ❌ {symbol} {interval}: [{data.get('status','?')}] {data.get('message','')}")
                return None

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            return df

        except Exception as e:
            print(f"  ⚠️  Attempt {attempt+1}/3 failed for {symbol} {interval}: {e}")
            time.sleep(2)

    print(f"  ❌ All 3 attempts failed for {symbol} {interval}")
    return None


# ---------------------------------------------------------------------------
# SIGNAL ANALYSIS
# ---------------------------------------------------------------------------
def analyse_signal(sig: dict, df: pd.DataFrame) -> dict:
    symbol    = sig["symbol"]
    direction = sig["direction"].upper()
    entry     = sig["entry"]
    sl        = sig["sl"]
    tp        = sig["tp"]
    zone_top  = sig.get("zone_top",    entry)
    zone_bot  = sig.get("zone_bottom", entry)
    sig_time  = pd.to_datetime(sig["timestamp"], utc=False)
    sig_time  = sig_time.tz_localize(None) if sig_time.tzinfo is not None else sig_time
    ps        = pip_size(symbol)

    future = df[df["datetime"] >= sig_time].copy()

    result = {
        "timestamp":       sig["timestamp"],
        "symbol":          symbol,
        "direction":       direction,
        "timeframe":       sig.get("timeframe", "?"),
        "strategy":        sig.get("strategy", "?"),
        "entry":           entry,
        "sl":              sl,
        "tp":              tp,
        "risk_pips":       sig.get("risk_pips"),
        "reward_pips":     sig.get("reward_pips"),
        "predicted_rr":    sig.get("rr", "?"),
        "probability":     sig.get("probability"),
        "zone_top":        zone_top,
        "zone_bot":        zone_bot,
        "zone_strength":   sig.get("zone_strength"),
        "confirmations":   sig.get("confirmations"),
        "htf_trend":       sig.get("htf_trend"),
        "d1_trend":        sig.get("d1_trend"),
        "entry_hit":       False,
        "bars_to_entry":   None,
        "touch_price":     None,
        "slippage_pips":   None,
        "sl_hit":          False,
        "tp_hit":          False,
        "outcome":         "never_triggered",
        "bars_to_outcome": None,
        "max_adverse_pips":   None,
        "max_favorable_pips": None,
        "rr_achieved":     None,
        "zone_respect":    None,
        "note":            "",
    }

    if future.empty:
        result["note"] = "no bars after signal time"
        return result

    # ---- 1. Did price reach the entry zone? (wick check) -------------------
    if direction == "BUY":
        # demand zone — price wicks down to zone_top or lower
        touched = future[future["low"] <= zone_top]
    else:
        # supply zone — price wicks up to zone_bot or higher
        touched = future[future["high"] >= zone_bot]

    if touched.empty:
        result["note"] = "price never reached entry zone"
        return result

    first_touch_idx = touched.index[0]
    bars_to_entry   = int(first_touch_idx - future.index[0])
    touch_bar       = future.loc[first_touch_idx]
    touch_price     = touch_bar["low"] if direction == "BUY" else touch_bar["high"]

    result["entry_hit"]     = True
    result["bars_to_entry"] = bars_to_entry
    result["touch_price"]   = round(touch_price, 6)
    result["slippage_pips"] = round(to_pips(touch_price - entry, symbol), 1)

    # ---- 2. Did price close inside the zone? (zone respect) ----------------
    post_zone = future.loc[first_touch_idx:]
    closed_inside = post_zone[
        (post_zone["close"] >= zone_bot) & (post_zone["close"] <= zone_top)
    ]
    result["zone_respect"] = not closed_inside.empty

    # ---- 3. SL / TP via WICKS, MAE / MFE ----------------------------------
    post_entry = future.loc[first_touch_idx:].copy()

    if direction == "BUY":
        # SL: wick LOW pierces below SL level
        sl_bars = post_entry[post_entry["low"]  <= sl]
        # TP: wick HIGH pierces above TP level
        tp_bars = post_entry[post_entry["high"] >= tp]
        adverse_extreme   = post_entry["low"].min()    # lowest wick (against BUY)
        favorable_extreme = post_entry["high"].max()   # highest wick (for BUY)
        mae = to_pips(entry - adverse_extreme,   symbol)
        mfe = to_pips(favorable_extreme - entry, symbol)
    else:
        # SL: wick HIGH pierces above SL level
        sl_bars = post_entry[post_entry["high"] >= sl]
        # TP: wick LOW pierces below TP level
        tp_bars = post_entry[post_entry["low"]  <= tp]
        adverse_extreme   = post_entry["high"].max()   # highest wick (against SELL)
        favorable_extreme = post_entry["low"].min()    # lowest wick (for SELL)
        mae = to_pips(adverse_extreme - entry,   symbol)
        mfe = to_pips(entry - favorable_extreme, symbol)

    result["max_adverse_pips"]   = round(max(0, mae), 1)
    result["max_favorable_pips"] = round(max(0, mfe), 1)

    sl_bar_i = sl_bars.index[0] if not sl_bars.empty else None
    tp_bar_i = tp_bars.index[0] if not tp_bars.empty else None

    # Determine outcome — same-bar conflict → SL wins (worst case)
    if tp_bar_i is not None and sl_bar_i is not None:
        outcome = "tp_hit" if tp_bar_i < sl_bar_i else "sl_hit"
        bars_to_outcome = int(min(tp_bar_i, sl_bar_i) - first_touch_idx)
    elif tp_bar_i is not None:
        outcome = "tp_hit"
        bars_to_outcome = int(tp_bar_i - first_touch_idx)
    elif sl_bar_i is not None:
        outcome = "sl_hit"
        bars_to_outcome = int(sl_bar_i - first_touch_idx)
    else:
        outcome = "still_open"
        bars_to_outcome = len(post_entry)

    risk_pips = to_pips(abs(entry - sl), symbol) or 1
    result["tp_hit"]          = (outcome == "tp_hit")
    result["sl_hit"]          = (outcome == "sl_hit")
    result["outcome"]         = outcome
    result["bars_to_outcome"] = bars_to_outcome
    result["rr_achieved"]     = round((-1.0 if outcome == "sl_hit" else mfe / risk_pips), 2)

    return result


# ---------------------------------------------------------------------------
# CONSOLE REPORT
# ---------------------------------------------------------------------------
OUTCOME_ICON = {
    "tp_hit":          "✅ TP HIT",
    "sl_hit":          "❌ SL HIT  (wick triggered)",
    "still_open":      "🟡 STILL OPEN",
    "never_triggered": "⬜ NEVER TRIGGERED",
}

def print_report(results: list):
    sep = "─" * 120
    print(f"\n{'SIGNAL COMPARISON REPORT':^120}")
    print(f"{'(SL and TP checked on bar WICKS — high/low — matching broker execution)':^120}")
    print(sep)

    for r in results:
        icon = OUTCOME_ICON.get(r["outcome"], r["outcome"])
        print(
            f"\n{r['timestamp'][:16]}  {r['symbol']:8s}  {r['direction']:4s}  "
            f"{r['strategy']}  [{r['timeframe']}]  "
            f"prob={r['probability']}%  confs={r['confirmations']}"
        )
        print(
            f"  Entry={r['entry']}  SL={r['sl']}  TP={r['tp']}  "
            f"Pred RR={r['predicted_rr']}  "
            f"Zone=[{r['zone_bot']} – {r['zone_top']}]  "
            f"Strength={r['zone_strength']}  "
            f"HTF={r['htf_trend']}  D1={r['d1_trend']}"
        )
        if r["entry_hit"]:
            slip_arrow = "↓worse" if r["slippage_pips"] and r["slippage_pips"] > 0 else "↑better"
            print(
                f"  📍 Entry touched after {r['bars_to_entry']} bars  "
                f"touch={r['touch_price']}  "
                f"slippage={r['slippage_pips']:+.1f} pips {slip_arrow}  "
                f"zone_close={'yes' if r['zone_respect'] else 'no (blown through)'}"
            )
            print(
                f"  📊 MAE={r['max_adverse_pips']} pips  "
                f"MFE={r['max_favorable_pips']} pips  "
                f"bars_to_resolution={r['bars_to_outcome']}  "
                f"RR_achieved={r['rr_achieved']}"
            )
        else:
            print(f"  ⬜ {r['note']}")
        print(f"  {icon}")

    # ---- summary -----------------------------------------------------------
    print(f"\n{sep}")
    total      = len(results)
    hit        = [r for r in results if r["entry_hit"]]
    tp         = [r for r in results if r["outcome"] == "tp_hit"]
    sl         = [r for r in results if r["outcome"] == "sl_hit"]
    open_      = [r for r in results if r["outcome"] == "still_open"]
    no_trig    = [r for r in results if r["outcome"] == "never_triggered"]
    winrate    = len(tp) / len(hit) * 100 if hit else 0
    avg_mae    = np.nanmean([r["max_adverse_pips"]   for r in hit if r["max_adverse_pips"]   is not None] or [np.nan])
    avg_mfe    = np.nanmean([r["max_favorable_pips"] for r in hit if r["max_favorable_pips"] is not None] or [np.nan])
    avg_slip   = np.nanmean([r["slippage_pips"]      for r in hit if r["slippage_pips"]      is not None] or [np.nan])

    print(f"\nSUMMARY  |  {total} signals")
    print(f"  Triggered : {len(hit)}/{total} ({len(hit)/total*100:.0f}%)")
    print(f"  TP ✅     : {len(tp)}   SL ❌: {len(sl)}   Open 🟡: {len(open_)}   Never ⬜: {len(no_trig)}")
    print(f"  Win-rate (of triggered) : {winrate:.1f}%")
    print(f"  Avg slippage            : {avg_slip:+.1f} pips   (+ = filled worse than planned entry)")
    print(f"  Avg MAE                 : {avg_mae:.1f} pips   (max wick against you from entry)")
    print(f"  Avg MFE                 : {avg_mfe:.1f} pips   (max wick in your favour from entry)")

    # ---- improvement hints -------------------------------------------------
    print(f"\nIMPROVEMENT HINTS")
    print("─" * 60)

    # 1. Entry timing
    if hit:
        avg_bars = np.mean([r["bars_to_entry"] for r in hit])
        print(f"  • Avg bars to first zone touch : {avg_bars:.1f}")
        if avg_bars > 20:
            print("    → Zones forming too early. Consider waiting for a 5M BOS or")
            print("      engulfing close before marking entry rather than the zone top/bottom.")

    # 2. Slippage
    if not np.isnan(avg_slip) and abs(avg_slip) > 5:
        side = "below zone_bottom" if avg_slip < 0 else "above zone_top"
        print(f"  • Avg slippage {avg_slip:+.1f} pips — price consistently fills {side}.")
        adj = "zone_bottom" if avg_slip < 0 else "zone_top"
        print(f"    → Shift entry anchor to zone mid-point or use {adj} as limit.")

    # 3. SL wick analysis
    if sl:
        sl_mae = [r["max_adverse_pips"] for r in sl if r["max_adverse_pips"] is not None]
        sl_risk = [r["risk_pips"] for r in sl if r["risk_pips"] is not None]
        if sl_mae and sl_risk:
            ratio = np.mean(sl_mae) / np.mean(sl_risk)
            print(f"  • SL losses: avg MAE was {np.mean(sl_mae):.1f} pips vs avg SL of {np.mean(sl_risk):.1f} pips ({ratio:.0%} used)")
            if ratio < 0.65:
                print("    → Wicks are stopping you out well inside the SL. SL may be too wide —")
                print("      tighten to just beyond the OB candle wick extreme.")
            elif ratio > 0.95:
                print("    → Price was using almost all of the SL before reversing.")
                print("      Consider a small ATR buffer beyond the OB wick.")

    # 4. Never-triggered
    nt_pct = len(no_trig) / total * 100
    if nt_pct > 40:
        print(f"  • {nt_pct:.0f}% of signals never triggered. Zone distance filter may be too tight,")
        print("    or valid setups are being generated too far from current price.")
        print("    → Review MAX_DIST_PIPS / 20×ATR config in run_dual_strategy().")

    # 5. Zone close respect
    if hit:
        resp  = sum(1 for r in hit if r["zone_respect"])
        pct   = resp / len(hit) * 100
        print(f"  • Zone close-respect : {resp}/{len(hit)} ({pct:.0f}%) — "
              f"price closed inside zone on at least one bar after touch")
        if pct < 50:
            print("    → Price is blowing through zones without a proper close.")
            print("      Increase zone_strength filter or add a close-inside-zone confirmation.")

    # 6. Same-bar SL+TP conflict
    same_bar = sum(
        1 for r in results
        if r["sl_hit"]
        and r["bars_to_outcome"] == 0
    )
    if same_bar > 0:
        print(f"  • {same_bar} trade(s) had SL and TP wick on the SAME bar → marked SL (worst case).")
        print("    → These are ambiguous bars; consider using a tighter TP or checking a lower TF.")

    print(sep)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare predicted signals against Twelve Data live bars")
    parser.add_argument("--api-key",   default=None,                 help="Twelve Data API key (auto-read from predictor if omitted)")
    parser.add_argument("--predictor", default="production_predictor.py", help="Path to production_predictor.py")
    parser.add_argument("--signals",   default="all_signals.json",   help="Path to signals JSON (default: all_signals.json)")
    parser.add_argument("--interval",  default="15min",              help="Bar interval (default: 15min)")
    parser.add_argument("--bars",      default=300, type=int,        help="Bars to fetch per symbol (default: 300)")
    parser.add_argument("--out",       default="signal_report.csv",  help="CSV output path")
    parser.add_argument("--save-bars", action="store_true",          help="Save OHLCV bars to CSV per symbol")
    parser.add_argument("--status",    default="all",                help="Filter: all / pending / hit_tp / hit_sl")
    args = parser.parse_args()

    # ---- resolve API keys --------------------------------------------------
    pair_keys = load_keys_from_predictor(args.predictor)
    if pair_keys:
        print(f"🔑 Loaded API keys from {args.predictor}: {list(pair_keys.keys())}")
    fallback_key = args.api_key or pair_keys.get("__fallback__")
    if not fallback_key and not pair_keys:
        print("❌ No API key found. Pass --api-key or ensure production_predictor.py is in the same folder.")
        sys.exit(1)

    # ---- load signals ------------------------------------------------------
    signals_path = Path(args.signals)
    if not signals_path.exists():
        print(f"❌ Signals file not found: {signals_path}")
        sys.exit(1)

    with open(signals_path) as f:
        signals = json.load(f)

    if args.status != "all":
        signals = [s for s in signals if s.get("status") == args.status]

    print(f"\n📂 Loaded {len(signals)} signals from {signals_path}")

    # ---- fetch bars once per symbol ----------------------------------------
    by_symbol: dict = {}
    for sig in signals:
        by_symbol.setdefault(sig["symbol"], []).append(sig)

    bar_cache: dict = {}
    for symbol in by_symbol:
        key = pair_keys.get(symbol, fallback_key)
        print(f"  📡 Fetching {args.interval} bars for {symbol} (key: {key[:8]}...) ...")
        df = fetch_bars(symbol, args.interval, args.bars, key)
        if df is not None:
            bar_cache[symbol] = df
            print(f"     ✅ {len(df)} bars  [{df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}]")
            if args.save_bars:
                fname = f"{symbol.replace('/', '_')}_{args.interval}_bars.csv"
                df.to_csv(fname, index=False)
                print(f"     💾 Saved to {fname}")
        else:
            print(f"     ❌ Could not fetch bars for {symbol}")

    # ---- analyse -----------------------------------------------------------
    results = []
    for sig in signals:
        sym = sig["symbol"]
        if sym not in bar_cache:
            results.append({
                "timestamp": sig.get("timestamp"), "symbol": sym,
                "direction": sig.get("direction"), "outcome": "bars_unavailable",
                "note": "bars fetch failed",
                **{k: None for k in [
                    "timeframe","strategy","entry","sl","tp","risk_pips","reward_pips",
                    "predicted_rr","probability","zone_top","zone_bot","zone_strength",
                    "confirmations","htf_trend","d1_trend","entry_hit","bars_to_entry",
                    "touch_price","slippage_pips","sl_hit","tp_hit","bars_to_outcome",
                    "max_adverse_pips","max_favorable_pips","rr_achieved","zone_respect",
                ]},
            })
        else:
            results.append(analyse_signal(sig, bar_cache[sym]))

    print_report(results)

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.out, index=False)
    print(f"📊 Full report saved to: {args.out}\n")


if __name__ == "__main__":
    main()