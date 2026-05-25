"""
diagnose_data_feeds.py
======================
Tests every data fetch your production_predictor relies on.
Reports PASS / WARN / FAIL for each feed with no silent fallbacks.

Usage:
    python diagnose_data_feeds.py
    python diagnose_data_feeds.py --symbol GBPUSD
    python diagnose_data_feeds.py --symbol XAUUSD --all-symbols
"""

import sys
import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Colour helpers ─────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
B  = "\033[94m"   # blue
W  = "\033[97m"   # white bold
NC = "\033[0m"    # reset

def ok(msg):   print(f"  {G}✅ PASS{NC}  {msg}")
def warn(msg): print(f"  {Y}⚠️  WARN{NC}  {msg}")
def fail(msg): print(f"  {R}❌ FAIL{NC}  {msg}")
def hdr(msg):  print(f"\n{W}{msg}{NC}")
def sep():     print(f"  {'─'*65}")


# ── Load API key from .env or environment ──────────────────────────────────────
def load_api_key() -> str:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip().lstrip("export").strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in ("TWELVE_API_KEY", "API_KEY"):
                    return v.strip().strip('"').strip("'")
    # Fall back to the hardcoded key in production_predictor.py
    return os.getenv("TWELVE_API_KEY", "")


# ── Single fetch with full diagnostics ────────────────────────────────────────
def fetch_bars(symbol: str, interval: str, outputsize: int, api_key: str) -> dict:
    """
    Returns a result dict:
        ok        : bool
        df        : pd.DataFrame | None
        rows      : int
        latency_ms: int
        error     : str | None
        warning   : str | None
        fallback  : str | None   — what the predictor silently uses on failure
    """
    url    = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     api_key,
        "format":     "JSON",
    }

    t0 = time.time()
    try:
        r    = requests.get(url, params=params, timeout=30)
        ms   = int((time.time() - t0) * 1000)
        data = r.json()
    except Exception as e:
        return dict(ok=False, df=None, rows=0, latency_ms=0,
                    error=str(e), warning=None, fallback=None)

    if "values" not in data:
        return dict(ok=False, df=None, rows=0, latency_ms=ms,
                    error=f"No 'values' in response — status={data.get('status','?')} "
                          f"msg={data.get('message', str(data)[:100])}",
                    warning=None, fallback=None)

    rows = []
    for c in data["values"]:
        try:
            rows.append({
                "time":  pd.to_datetime(c["datetime"]),
                "open":  float(c["open"]),
                "high":  float(c["high"]),
                "low":   float(c["low"]),
                "close": float(c["close"]),
            })
        except Exception:
            continue

    if not rows:
        return dict(ok=False, df=None, rows=0, latency_ms=ms,
                    error="Response had 'values' but no parseable rows",
                    warning=None, fallback=None)

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    warning  = None
    fallback = None

    # ── Check minimum bar counts that the predictor actually needs ────────────
    min_required = {
        "15min": 200,
        "5min":  60,
        "1h":    100,
        "4h":    200,   # outputsize=200 used for structure detection
        "1day":  70,    # outputsize=70 used for D1 gate
    }
    needed = min_required.get(interval, 20)
    if len(df) < needed:
        warning = (f"Only {len(df)} bars returned, need ≥{needed}. "
                   f"Predictor may use fallback values.")

    # ── Check for stale data ──────────────────────────────────────────────────
    latest     = df["time"].max()
    age_hours  = (datetime.utcnow() - latest).total_seconds() / 3600
    if age_hours > 6:
        warning = (warning or "") + f" Latest bar is {age_hours:.1f}h old — possible stale feed."

    return dict(ok=True, df=df, rows=len(df), latency_ms=ms,
                error=None, warning=warning, fallback=fallback)


# ── Compute indicators the predictor needs ─────────────────────────────────────
def check_indicators(df: pd.DataFrame, interval: str) -> list:
    """Returns list of issues found in indicator computation."""
    issues = []
    close = df["close"]

    # ATR
    high, low = df["high"], df["low"]
    close_prev = close.shift(1)
    tr = pd.concat([high - low,
                    (high - close_prev).abs(),
                    (low  - close_prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    if atr.iloc[-1] != atr.iloc[-1]:  # NaN check
        issues.append("ATR is NaN — not enough bars for 14-bar window")

    # EMA 20/50/200
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    if len(df) < 200 and interval in ("4h", "1h"):
        issues.append(f"Only {len(df)} bars — EMA200 will be unreliable (needs 200+)")

    # RSI
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-10)))
    if rsi.iloc[-1] != rsi.iloc[-1]:
        issues.append("RSI is NaN")

    return issues


# ── Check D1 premium/discount gate ────────────────────────────────────────────
def check_d1_gate(df_d1: pd.DataFrame, current_price: float) -> dict:
    tail60  = df_d1.tail(60)
    d1_high = float(np.percentile(tail60["high"], 95))
    d1_low  = float(np.percentile(tail60["low"],   5))
    d1_rng  = d1_high - d1_low
    d1_pos  = (current_price - d1_low) / d1_rng if d1_rng > 0 else 0.5

    if d1_pos > 0.6:
        zone, d1_pd = "🔴 PREMIUM", 1
    elif d1_pos < 0.4:
        zone, d1_pd = "🟢 DISCOUNT", -1
    else:
        zone, d1_pd = "🟡 EQUILIBRIUM", 0

    return dict(
        d1_high=d1_high, d1_low=d1_low, d1_rng=d1_rng,
        d1_pos=d1_pos, zone=zone, d1_pd=d1_pd,
        bars_used=len(tail60)
    )


# ── Check H4 structure detection ──────────────────────────────────────────────
def check_h4_structure(df_4h: pd.DataFrame) -> dict:
    """
    Replicates what detect_market_structure_shift() does so we can
    see what trend the predictor will derive without importing the whole predictor.
    """
    if len(df_4h) < 20:
        return dict(trend="neutral", reason="not enough bars (<20)", swings=0)

    close  = df_4h["close"]
    ema20  = close.ewm(span=20, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()

    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]

    if last_ema20 > last_ema50 * 1.001:
        trend = "bullish"
    elif last_ema20 < last_ema50 * 0.999:
        trend = "bearish"
    else:
        trend = "neutral"

    return dict(
        trend=trend,
        ema20=round(last_ema20, 5),
        ema50=round(last_ema50, 5),
        close=round(close.iloc[-1], 5),
        bars=len(df_4h),
        reason=f"EMA20={last_ema20:.5f} vs EMA50={last_ema50:.5f}"
    )


# ── Model files check ──────────────────────────────────────────────────────────
def check_model_files() -> list:
    issues = []
    required = [
        "models/model_v5_reversion.pkl",
        "models/model_v5_continuation.pkl",
        "models/scaler_v5_reversion.pkl",
        "models/scaler_v5_continuation.pkl",
        "models/encoders_v5_reversion.pkl",
        "models/encoders_v5_continuation.pkl",
        "models/training_stats_v5_reversion.pkl",
        "models/training_stats_v5_continuation.pkl",
        "models/scaler_preprocessing.pkl",
        "models/encoders_preprocessing.pkl",
    ]
    for path in required:
        p = Path(path)
        if not p.exists():
            issues.append(("FAIL", path, "FILE NOT FOUND"))
        else:
            size_kb = p.stat().st_size // 1024
            age_days = (datetime.utcnow().timestamp() - p.stat().st_mtime) / 86400
            if size_kb < 1:
                issues.append(("WARN", path, f"suspiciously small ({size_kb}KB)"))
            else:
                issues.append(("PASS", path, f"{size_kb}KB, modified {age_days:.0f}d ago"))
    return issues


# ── Gate prediction ────────────────────────────────────────────────────────────
def predict_gates(h4_trend: str, d1_pd: int) -> list:
    """Given h4_trend and d1_pd, show exactly which zones pass and which are blocked."""
    results = []
    combos = [
        ("demand", "BUY  demand"),
        ("supply", "SELL supply"),
    ]
    for zone_type, label in combos:
        blocked_by = None

        # 4H bias gate
        if h4_trend == "bullish" and zone_type == "supply":
            blocked_by = "4h_bias_bullish_blocks_supply"
        elif h4_trend == "bearish" and zone_type == "demand":
            blocked_by = "4h_bias_bearish_blocks_demand"

        # D1 gate (only if not already blocked)
        if blocked_by is None:
            if d1_pd == 1 and zone_type == "demand":
                if h4_trend != "bullish":
                    blocked_by = "d1_premium_blocks_demand"
            elif d1_pd == -1 and zone_type == "supply":
                if h4_trend == "bullish":
                    blocked_by = "d1_discount_blocks_supply"
                elif h4_trend == "neutral":
                    blocked_by = "d1_discount_blocks_supply_neutral"

        results.append((label, blocked_by))
    return results


# ── Main diagnostic runner ────────────────────────────────────────────────────
def run_diagnostics(symbol: str, api_key: str):
    print(f"\n{'═'*70}")
    print(f"  DATA FEED DIAGNOSTIC — {symbol}  [{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}]")
    print(f"{'═'*70}")

    results_summary = []   # (label, status, detail)

    # ── 1. API KEY ─────────────────────────────────────────────────────────────
    hdr("1. API KEY")
    if not api_key:
        fail("No API key found. Add TWELVE_API_KEY to .env or environment.")
        results_summary.append(("API Key", "FAIL", "missing"))
        return   # nothing else will work
    else:
        ok(f"API key loaded: {api_key[:8]}…{api_key[-4:]}")
        results_summary.append(("API Key", "PASS", "loaded"))

    # ── 2. DATA FETCHES ────────────────────────────────────────────────────────
    hdr("2. DATA FETCHES (TwelveData API)")
    sep()

    fetches = [
        ("15min", 200, "Primary bars (zone detection)"),
        ("5min",   60, "5min refinement"),
        ("1h",    200, "H1 structure + FVGs"),
        ("4h",    200, "H4 structure + FVGs + bias gate"),
        ("1day",   70, "D1 premium/discount gate"),
    ]

    fetched = {}
    for interval, outputsize, purpose in fetches:
        print(f"\n  [{interval:6s}] {purpose}")
        result = fetch_bars(symbol, interval, outputsize, api_key)
        fetched[interval] = result

        if not result["ok"]:
            fail(f"Fetch FAILED — {result['error']}")
            results_summary.append((f"Fetch {interval}", "FAIL", result["error"]))

            # Show what the predictor silently does on failure
            fallbacks = {
                "4h":   "h4_structure → {'trend': 'neutral'}  ← BLOCKS ALL SUPPLY IN D1 DISCOUNT",
                "1h":   "h1_structure → {'trend': 'neutral'}",
                "1day": "daily_pivot, daily_r1, daily_s1 → None",
            }
            if interval in fallbacks:
                warn(f"Silent fallback: {fallbacks[interval]}")
        else:
            df = result["df"]
            latest = df["time"].max()
            age_h  = (datetime.utcnow() - latest).total_seconds() / 3600
            ok(f"{result['rows']} bars  |  latest: {latest.strftime('%Y-%m-%d %H:%M')} "
               f"({age_h:.1f}h ago)  |  {result['latency_ms']}ms")
            results_summary.append((f"Fetch {interval}", "PASS",
                                     f"{result['rows']} bars, {result['latency_ms']}ms"))

            if result["warning"]:
                warn(result["warning"])
                results_summary.append((f"Fetch {interval} quality", "WARN", result["warning"]))

            # Indicator check
            issues = check_indicators(df, interval)
            for issue in issues:
                warn(f"Indicator: {issue}")

    # ── 3. H4 STRUCTURE DETECTION ─────────────────────────────────────────────
    hdr("3. H4 STRUCTURE (bias gate input)")
    sep()
    r4h = fetched.get("4h", {})
    if r4h.get("ok") and r4h.get("df") is not None:
        h4_info = check_h4_structure(r4h["df"])
        trend   = h4_info["trend"]
        emoji   = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "🟡"
        print(f"\n  {emoji} H4 trend = {trend.upper()}")
        print(f"     {h4_info['reason']}")
        print(f"     Close={h4_info['close']}  Bars={h4_info['bars']}")

        if trend == "neutral":
            warn("H4 is NEUTRAL — this activates the d1_discount_blocks_supply_neutral gate")
            warn("Any supply zone while D1 is in DISCOUNT will be blocked even if setup is valid")
        results_summary.append(("H4 Trend", "PASS" if trend != "neutral" else "WARN",
                                 trend))
    else:
        fail("Cannot determine H4 trend — fetch failed")
        h4_info = {"trend": "neutral"}
        results_summary.append(("H4 Trend", "FAIL", "fetch failed → neutral fallback"))

    # ── 4. D1 PREMIUM/DISCOUNT GATE ───────────────────────────────────────────
    hdr("4. D1 PREMIUM/DISCOUNT GATE")
    sep()
    r1d = fetched.get("1day", {})
    if r1d.get("ok") and r1d.get("df") is not None:
        df_d1 = r1d["df"]
        # Get current price from 15min
        r15 = fetched.get("15min", {})
        current_price = (r15["df"]["close"].iloc[-1]
                         if r15.get("ok") and r15.get("df") is not None
                         else df_d1["close"].iloc[-1])

        gate = check_d1_gate(df_d1, current_price)
        print(f"\n  {gate['zone']}")
        print(f"     P95 high : {gate['d1_high']:.5f}")
        print(f"     P5  low  : {gate['d1_low']:.5f}")
        print(f"     Range    : {gate['d1_rng']:.5f}")
        print(f"     Position : {gate['d1_pos']:.3f}  (0=bottom, 1=top)")
        print(f"     Price    : {current_price:.5f}")
        print(f"     Bars used: {gate['bars_used']} (of 60-bar tail)")

        d1_pd = gate["d1_pd"]
        results_summary.append(("D1 Gate", "PASS", gate["zone"]))
    else:
        fail("Cannot compute D1 gate — D1 fetch failed → _d1_pd defaults to 0 (equilibrium)")
        d1_pd = 0
        gate  = {"zone": "UNKNOWN (fetch failed)", "d1_pd": 0, "d1_pos": 0.5}
        results_summary.append(("D1 Gate", "FAIL", "fetch failed → equilibrium fallback"))

    # ── 5. GATE SIMULATION ────────────────────────────────────────────────────
    hdr("5. GATE SIMULATION — what passes right now")
    sep()
    h4_trend = h4_info.get("trend", "neutral")
    gates    = predict_gates(h4_trend, d1_pd)

    print(f"\n  Current state: H4={h4_trend.upper()}  D1={gate['zone']}")
    print()
    any_pass = False
    for label, blocked_by in gates:
        if blocked_by:
            fail(f"{label}  →  BLOCKED by [{blocked_by}]")
        else:
            ok(f"{label}  →  PASSES gates → reaches ML model")
            any_pass = True

    if not any_pass:
        print()
        warn("ALL zone types are currently blocked by gates.")
        warn("The ML model will NOT be called for any zone right now.")
        warn("This explains why you see 0 live signals.")
        results_summary.append(("Gate Result", "FAIL", "ALL zones blocked"))
    else:
        results_summary.append(("Gate Result", "PASS", "at least one zone type passes"))

    # ── 6. MODEL FILES ────────────────────────────────────────────────────────
    hdr("6. MODEL FILES")
    sep()
    for status, path, detail in check_model_files():
        msg = f"{path:<55} {detail}"
        if status == "PASS":
            ok(msg)
        elif status == "WARN":
            warn(msg)
        else:
            fail(msg)
        results_summary.append((f"File {Path(path).name}", status, detail))

    # ── 7. SIGNALS DIRECTORY ──────────────────────────────────────────────────
    hdr("7. SIGNAL FILES")
    sep()
    for path in ["signals/all_signals.json", "signals/observation_signals.json"]:
        p = Path(path)
        if p.exists():
            try:
                import json
                data    = json.load(open(p))
                signals = data if isinstance(data, list) else []
                age_h   = (datetime.utcnow().timestamp() - p.stat().st_mtime) / 3600
                ok(f"{path}: {len(signals)} signals, last modified {age_h:.1f}h ago")
                if path.endswith("observation_signals.json") and signals:
                    from collections import Counter
                    gates_c = Counter(s.get("gate_blocked_by", "?") for s in signals)
                    trends_c = Counter(s.get("h4_trend", "?") for s in signals)
                    print(f"       Gate breakdown: {dict(gates_c)}")
                    print(f"       H4 at block:    {dict(trends_c)}")
            except Exception as e:
                warn(f"{path}: exists but could not parse — {e}")
        else:
            warn(f"{path}: NOT FOUND (predictor has not run yet or wrote to different path)")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    hdr("SUMMARY")
    print(f"{'═'*70}")
    passes = sum(1 for _, s, _ in results_summary if s == "PASS")
    warns  = sum(1 for _, s, _ in results_summary if s == "WARN")
    fails  = sum(1 for _, s, _ in results_summary if s == "FAIL")
    print(f"  {G}PASS: {passes}{NC}   {Y}WARN: {warns}{NC}   {R}FAIL: {fails}{NC}")
    print()
    if fails > 0:
        print(f"  {R}Critical issues to fix:{NC}")
        for label, status, detail in results_summary:
            if status == "FAIL":
                print(f"    ❌ {label}: {detail}")
    if warns > 0:
        print(f"\n  {Y}Warnings to investigate:{NC}")
        for label, status, detail in results_summary:
            if status == "WARN":
                print(f"    ⚠️  {label}: {detail}")
    print(f"{'═'*70}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose production_predictor data feeds")
    parser.add_argument("--symbol",      default="EUR/USD", help="Symbol to test (default: EUR/USD)")
    parser.add_argument("--all-symbols", action="store_true",
                        help="Run for all 4 symbols: EUR/USD GBP/USD USD/JPY XAU/USD")
    parser.add_argument("--api-key",     default="", help="TwelveData API key (overrides .env)")
    args = parser.parse_args()

    api_key = args.api_key or load_api_key()

    symbols = (["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
               if args.all_symbols else [args.symbol])

    for sym in symbols:
        run_diagnostics(sym, api_key)
        if len(symbols) > 1:
            time.sleep(2)   # be kind to the API rate limit