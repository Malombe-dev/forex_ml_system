"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            FOREX ML SYSTEM — FILTER PIPELINE DIAGNOSTIC v1                 ║
║  Traces every 4H OB through each filter gate and reports exactly WHY       ║
║  zones die before reaching the ML model. Covers:                           ║
║    Gate 1 — Distance (MAX_DIST_ATR)                                        ║
║    Gate 2 — Confirmation candles (confs < 2)                               ║
║    Gate 3 — Direction / side-of-price                                      ║
║    Gate 4 — Risk/Reward (MIN_HTF_RR=1.2 … MAX_HTF_RR=6 Gold/10 Forex)   ║
║    Gate 5 — Zone strength (MIN_HTF_STRENGTH=0.5)                           ║
║    Gate 6 — ML probability (base_prob < 0.55 in htf_trade_detect)          ║
║    Gate 7 — Final prob threshold (prob < self.prob_threshold * 100)        ║
║    Gate 8 — Bias filter (h1_trend vs direction vs price side)              ║
║  Also diagnoses the 15min funnel and D1 gate per pair.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    cd ~/Desktop/forex_ml_system/forex_ml_system
    source venv/bin/activate
    python diagnose_filters.py

    # Single pair:
    python diagnose_filters.py --symbol XAU/USD

    # All pairs, save to file:
    python diagnose_filters.py --all --out diag_report.txt
"""

import sys
import os
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Make sure the project root is on the path ─────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd

# ── Colour helpers (auto-disabled when writing to file) ───────────────────────
_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    CODES = {
        "red":    "\033[91m",
        "green":  "\033[92m",
        "yellow": "\033[93m",
        "cyan":   "\033[96m",
        "bold":   "\033[1m",
        "reset":  "\033[0m",
    }
    return f"{CODES.get(code,'')}{text}{CODES['reset']}"

def _hr(char: str = "─", width: int = 90) -> str:
    return char * width

def _section(title: str) -> None:
    print(f"\n{_hr('═')}")
    print(_c("bold", f"  {title}"))
    print(_hr("═"))


# ══════════════════════════════════════════════════════════════════════════════
#  Import production modules — fail gracefully with actionable messages
# ══════════════════════════════════════════════════════════════════════════════
try:
    from production_predictor import ProductionForexPredictor, HTFInstitutionalAnalyzer as HTFAnalyzer
except ImportError as e:
    print(_c("red", f"\n[FATAL] Cannot import production_predictor: {e}"))
    print("  → Run this script from the project root with the venv active.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Config — mirrors run_dual_strategy() exactly
# ══════════════════════════════════════════════════════════════════════════════
PAIR_API_KEYS = {
    "EUR/USD": "ea3e4fce074e403f8cd4c83dd0035643",
    "GBP/USD": "2522b470eba249f88d2e29575035914e",
    "USD/JPY": "5bb48b4211144b92a4487ce7c63d1b7a",
    "XAU/USD": "63a513bb7eb848748d28d3deb64c1b59",
}

SYMBOL_PARAMS = {
    "EUR/USD": dict(max_distance_pips=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "GBP/USD": dict(max_distance_pips=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "USD/JPY": dict(max_distance_pips=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "XAU/USD": dict(max_distance_pips=80.0, zone_distance_threshold=150.0, min_strength_atr=0.40),
}

MODEL_FILES = {
    "rev": {
        "model":    "models/model_v5_reversion.pkl",
        "scaler":   "models/scaler_v5_reversion.pkl",
        "encoders": "models/encoders_v5_reversion.pkl",
        "stats":    "models/training_stats_v5_reversion.pkl",
    },
    "cont": {
        "model":    "models/model_v5_continuation.pkl",
        "scaler":   "models/scaler_v5_continuation.pkl",
        "encoders": "models/encoders_v5_continuation.pkl",
        "stats":    "models/training_stats_v5_continuation.pkl",
    },
}

# Hard-coded constants matching htf_trade_detect()
MAX_DIST_ATR      = 20.0
MIN_HTF_RR        = 1.2
# MAX_HTF_RR is now per-symbol — set dynamically per pair in the diagnostic functions
MIN_HTF_STRENGTH  = 0.5
ATR_TOL_MULT      = 1.0    # atr_tol = current_atr * ATR_TOL_MULT

def _get_max_rr(symbol: str) -> float:
    """Mirror production: Gold keeps 6.0, all forex pairs use 10.0."""
    return 6.0 if symbol == 'XAU/USD' else 10.0

def _get_ml_gate(symbol: str) -> float:
    """Mirror production: Gold always 0.55; forex 0.45 during Asia (00-07 UTC), else 0.55."""
    if symbol == 'XAU/USD':
        return 0.55
    hour_utc = datetime.utcnow().hour
    return 0.45 if 0 <= hour_utc < 7 else 0.55


# ══════════════════════════════════════════════════════════════════════════════
#  Core diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def _safe_predictor(symbol: str, strategy: str) -> Optional[ProductionForexPredictor]:
    """Instantiate a predictor for `symbol` / `strategy` ('rev' or 'cont')."""
    p   = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS["EUR/USD"])
    key = PAIR_API_KEYS.get(symbol, list(PAIR_API_KEYS.values())[0])
    mf  = MODEL_FILES[strategy]
    is_cont = (strategy == "cont")
    try:
        pred = ProductionForexPredictor(
            api_key=key,
            model_path=mf["model"],
            scaler_path=mf["scaler"],
            encoders_path=mf["encoders"],
            stats_path=mf["stats"],
            symbol=symbol,
            prob_threshold=0.65,
            max_rr=5.0 if is_cont else 3.5,
            min_risk_atr=0.8,
            max_distance_pips=p["max_distance_pips"],
            min_strength_atr=p["min_strength_atr"],
            use_5min_refinement=True,
            refine_max_dist_multiplier=2.0,
            enable_continuation=is_cont,
            use_htf_analysis=True,
            min_confluence_score=1,
            zone_ttl_hours=5.8,
            zone_distance_threshold=p["zone_distance_threshold"],
            htf_fvg_min_gap_atr=0.5,
            htf_ob_min_reaction_atr=2.5,
            is_continuation_strategy=is_cont,
            debug=False,   # we do our own printing
        )
        return pred
    except Exception as e:
        print(_c("red", f"  [ERROR] Could not build predictor for {symbol}/{strategy}: {e}"))
        return None


def _fetch_bars_safe(pred: ProductionForexPredictor, interval: str,
                     outputsize: int) -> Optional[pd.DataFrame]:
    try:
        df = pred.fetch_bars(interval=interval, outputsize=outputsize)
        return df
    except Exception as e:
        print(_c("red", f"    [fetch_bars error] {interval}/{outputsize}: {e}"))
        return None


def diagnose_ob_filter_pipeline(symbol: str) -> None:
    """
    Full gate-by-gate trace of every 4H OB for `symbol`.
    Uses the REV predictor (the strategy that generates HTF signals).
    """
    _section(f"OB FILTER PIPELINE — {symbol}")

    pred = _safe_predictor(symbol, "rev")
    if pred is None:
        return

    # ── Fetch bars ────────────────────────────────────────────────────────────
    print("  Fetching bars …")
    df_4h  = _fetch_bars_safe(pred, "4h",    200)
    df_15m = _fetch_bars_safe(pred, "15min", 200)
    df_1h  = _fetch_bars_safe(pred, "1h",    100)
    df_5m  = _fetch_bars_safe(pred, "5min",   60)
    df_1d  = _fetch_bars_safe(pred, "1day",   90)

    if df_4h is None or len(df_4h) < 50:
        print(_c("red", "  [SKIP] 4H bars unavailable."))
        return
    if df_15m is None or len(df_15m) < 10:
        print(_c("yellow", "  [WARN] 15min bars unavailable — direction filter may be imprecise."))

    # ── Market state ──────────────────────────────────────────────────────────
    current_price = df_15m["close"].iloc[-1] if (df_15m is not None and len(df_15m) > 0) \
                    else df_4h["close"].iloc[-1]
    current_atr   = df_4h["atr"].iloc[-1]
    atr_tol       = current_atr * ATR_TOL_MULT
    max_dist_pips = MAX_DIST_ATR * (current_atr / pred.pip_size)
    MAX_HTF_RR    = _get_max_rr(symbol)
    MIN_PROB_GATE = _get_ml_gate(symbol)

    htf_context = pred.get_htf_context(
        df_1h   = df_1h,
        df_4h_c = df_4h,
        df_4h   = df_4h,
        df_1d   = df_1d,
    )
    h1_trend = htf_context.get("h1_trend", "neutral") if htf_context else "neutral"

    print(f"\n  {'Symbol':<20}: {symbol}")
    print(f"  {'Current price':<20}: {current_price:.5f}")
    print(f"  {'4H ATR':<20}: {current_atr:.5f}  ({current_atr / pred.pip_size:.1f} pips)")
    print(f"  {'ATR tolerance':<20}: {atr_tol:.5f}  ({atr_tol / pred.pip_size:.1f} pips)")
    print(f"  {'Max dist (pips)':<20}: {max_dist_pips:.1f}  ({MAX_DIST_ATR}× ATR)")
    print(f"  {'H1 trend (bias)':<20}: {_c('cyan', h1_trend)}")
    print(f"  {'Pip size':<20}: {pred.pip_size}")
    print(f"  {'Prob threshold':<20}: {pred.prob_threshold * 100:.0f}%")
    print(f"  {'MAX_HTF_RR':<20}: {MAX_HTF_RR}  {'(Gold: 6.0)' if symbol == 'XAU/USD' else '(Forex: 10.0)'}")
    print(f"  {'ML gate':<20}: {MIN_PROB_GATE*100:.0f}%  {'(Gold fixed)' if symbol == 'XAU/USD' else f'(Asia<07UTC=45%, else 55%) — now {MIN_PROB_GATE*100:.0f}%'}")

    # ── Detect OBs ────────────────────────────────────────────────────────────
    obs = pred.htf_analyzer.detect_order_blocks(df_4h, "4h")
    print(f"\n  Total 4H OBs detected : {len(obs)}")
    if not obs:
        print(_c("yellow", "  No 4H OBs found — check ob_min_reaction_atr or bar data quality."))
        return

    # ── Per-OB trace table ────────────────────────────────────────────────────
    headers = ["#", "Type", "Mid", "Dist(p)", "Dir", "reac_sz",
               "risk", "reward", "RR", "str", "G1", "G2", "G3", "G4", "G5", "Fate"]
    col_w   = [3, 8, 11, 8, 5, 9, 9, 9, 6, 5, 4, 4, 4, 4, 4, 18]

    def _row(*vals):
        return "  " + "  ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    print("\n" + _row(*headers))
    print("  " + _hr("-", sum(col_w) + 2 * len(col_w)))

    gate_totals = {"dist": 0, "confs": 0, "direction": 0, "rr": 0,
                   "strength": 0, "prob_ml": 0, "bias": 0, "passed": 0}

    survivors = []   # OBs that pass all pre-ML gates, for ML scoring below

    for idx, ob in enumerate(obs):
        direction = "BUY" if ob["type"] == "bullish" else "SELL"
        mid       = ob["mid"]
        dist_p    = abs(current_price - mid) / pred.pip_size

        # Gate 1: distance
        if dist_p > max_dist_pips:
            gate_totals["dist"] += 1
            print(_row(idx + 1, ob["type"][:4], f"{mid:.4f}", f"{dist_p:.1f}",
                       direction, "—", "—", "—", "—", "—",
                       _c("red", "✗"), "—", "—", "—", "—",
                       _c("red", f"DIST>{max_dist_pips:.0f}p")))
            continue

        # Gate 2: confirmation candles
        future = df_4h.iloc[ob["index"] + 1:]
        if len(future) < 2:
            gate_totals["confs"] += 1
            print(_row(idx + 1, ob["type"][:4], f"{mid:.4f}", f"{dist_p:.1f}",
                       direction, "—", "—", "—", "—", "—",
                       "✓", _c("red", "✗"), "—", "—", "—",
                       _c("red", "NO FUTURE BARS")))
            continue
        if ob["type"] == "bullish":
            confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]["low"] >= ob["bottom"])
        else:
            confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]["high"] <= ob["top"])
        if confs < 2:
            gate_totals["confs"] += 1
            print(_row(idx + 1, ob["type"][:4], f"{mid:.4f}", f"{dist_p:.1f}",
                       direction, "—", "—", "—", "—", "—",
                       "✓", _c("red", f"✗c{confs}"), "—", "—", "—",
                       _c("red", f"CONFS={confs}<2")))
            continue

        # Gate 3: direction / side of price
        side_fail = (direction == "BUY"  and mid > current_price + atr_tol) or \
                    (direction == "SELL" and mid < current_price - atr_tol)
        if side_fail:
            gate_totals["direction"] += 1
            side_info = (f"BUY mid {mid:.4f} > price+tol {current_price + atr_tol:.4f}"
                         if direction == "BUY"
                         else f"SELL mid {mid:.4f} < price-tol {current_price - atr_tol:.4f}")
            print(_row(idx + 1, ob["type"][:4], f"{mid:.4f}", f"{dist_p:.1f}",
                       direction, "—", "—", "—", "—", "—",
                       "✓", f"✓c{confs}", _c("red", "✗"), "—", "—",
                       _c("red", f"DIR: {side_info}")))
            continue

        # Gate 4: RR  — with TP clamping for non-Gold (mirrors production)
        entry    = mid
        _atr_buf = df_4h["atr"].iloc[-1] * 0.2 if "atr" in df_4h.columns else 0.0002
        sl       = ob["bottom"] - _atr_buf if direction == "BUY" else ob["top"] + _atr_buf
        tp_raw   = entry + ob["reaction_size"] * 1.5 if direction == "BUY" \
                   else entry - ob["reaction_size"] * 1.5
        if symbol != "XAU/USD" and abs(entry - sl) > 0:
            _max_rwd = abs(entry - sl) * MAX_HTF_RR * 0.95
            tp       = min(tp_raw, entry + _max_rwd) if direction == "BUY" \
                       else max(tp_raw, entry - _max_rwd)
        else:
            tp       = tp_raw
        risk     = abs(entry - sl)
        reward   = abs(tp - entry)
        rr       = reward / risk if risk > 0 else 0.0
        risk_p   = risk   / pred.pip_size
        reward_p = reward / pred.pip_size
        reac_p   = ob["reaction_size"] / pred.pip_size

        rr_fail = not (MIN_HTF_RR <= rr <= MAX_HTF_RR)
        if rr_fail:
            gate_totals["rr"] += 1
            rr_reason = (f"RR={rr:.2f} < {MIN_HTF_RR}" if rr < MIN_HTF_RR
                         else f"RR={rr:.2f} > {MAX_HTF_RR}")
            print(_row(idx + 1, ob["type"][:4], f"{mid:.4f}", f"{dist_p:.1f}",
                       direction, f"{reac_p:.1f}p", f"{risk_p:.1f}p", f"{reward_p:.1f}p",
                       f"{rr:.2f}", f"{ob.get('strength', 0):.2f}",
                       "✓", f"✓c{confs}", "✓", _c("red", "✗"), "—",
                       _c("red", rr_reason)))
            continue

        # Gate 5: strength
        str_fail = ob.get("strength", 0) < MIN_HTF_STRENGTH
        if str_fail:
            gate_totals["strength"] += 1
            print(_row(idx + 1, ob["type"][:4], f"{mid:.4f}", f"{dist_p:.1f}",
                       direction, f"{reac_p:.1f}p", f"{risk_p:.1f}p", f"{reward_p:.1f}p",
                       f"{rr:.2f}", f"{ob.get('strength', 0):.2f}",
                       "✓", f"✓c{confs}", "✓", "✓", _c("red", "✗"),
                       _c("red", f"STR={ob.get('strength', 0):.2f}<{MIN_HTF_STRENGTH}")))
            continue

        # All pre-ML gates passed
        print(_row(idx + 1, ob["type"][:4], f"{mid:.4f}", f"{dist_p:.1f}",
                   direction, f"{reac_p:.1f}p", f"{risk_p:.1f}p", f"{reward_p:.1f}p",
                   f"{rr:.2f}", f"{ob.get('strength', 0):.2f}",
                   "✓", f"✓c{confs}", "✓", "✓", "✓",
                   _c("green", "→ ML scoring")))
        survivors.append(dict(ob=ob, direction=direction, entry=entry,
                              sl=sl, tp=tp, risk=risk, reward=reward, rr=rr,
                              risk_p=risk_p, reward_p=reward_p,
                              dist_p=dist_p, confs=confs))

    # ── Gate summary ──────────────────────────────────────────────────────────
    print(f"\n  {'Gate summary':}")
    total = len(obs)
    print(f"    {'OBs detected':<35}: {total}")
    print(f"    {'Killed by distance (Gate 1)':<35}: {gate_totals['dist']}")
    print(f"    {'Killed by confs < 2 (Gate 2)':<35}: {gate_totals['confs']}")
    print(f"    {'Killed by direction/side (Gate 3)':<35}: {gate_totals['direction']}")
    print(f"    {'Killed by RR out of range (Gate 4)':<35}: {gate_totals['rr']}")
    print(f"    {'Killed by strength < 0.5 (Gate 5)':<35}: {gate_totals['strength']}")
    print(f"    {'Survivors for ML scoring':<35}: {len(survivors)}")

    if not survivors:
        print(_c("red", f"\n  ⛔ All OBs died before ML scoring. "
                        f"Primary killer: {_top_killer(gate_totals)}"))
        _suggest_fixes(gate_totals, symbol, current_atr, pred.pip_size, h1_trend)
        return

    # ── ML scoring for survivors ──────────────────────────────────────────────
    print(f"\n{_hr()}")
    print(_c("bold", "  ML SCORING — surviving OBs"))
    print(_hr())
    print(f"  {'#':<3}  {'Dir':<5}  {'Entry':>10}  {'base_prob':>10}  "
          f"{'final_prob':>11}  {'threshold':>10}  Result")
    print(f"  {_hr('-', 75)}")

    for i, s in enumerate(survivors):
        ob        = s["ob"]
        direction = s["direction"]

        # Replicate the full_ob_zone dict from htf_trade_detect
        try:
            h1_conf = pred._check_1h_confluence(ob["top"], ob["bottom"], direction, df_1h)
            m5_conf = pred._check_5min_entry(ob["top"], ob["bottom"], direction, df_5m)
            approach = pred._get_15min_approach(ob["top"], ob["bottom"], direction, df_15m)

            real_overlap = 1
            fvg_overlap_flag = False
            if htf_context:
                all_htf_items = (htf_context.get("h1_fvgs", []) +
                                 htf_context.get("h4_fvgs", []) +
                                 htf_context.get("h1_order_blocks", []) +
                                 htf_context.get("h4_order_blocks", []))
                for item in all_htf_items:
                    ib = item.get("bottom", item.get("price_low", 0))
                    it = item.get("top",    item.get("price_high", 0))
                    if ib < ob["top"] and it > ob["bottom"]:
                        real_overlap += 1
                for fvg_item in htf_context.get("h4_fvgs", []):
                    fb = fvg_item.get("bottom", fvg_item.get("price_low", 0))
                    ft = fvg_item.get("top",    fvg_item.get("price_high", 0))
                    if fb < ob["top"] and ft > ob["bottom"]:
                        fvg_overlap_flag = True
                        break

            full_ob_zone = {
                "type":                    "demand" if direction == "BUY" else "supply",
                "price_high":              ob["top"],
                "price_low":               ob["bottom"],
                "price_mid":               ob["mid"],
                "strength":                ob.get("strength", 1.0),
                "confluence":              real_overlap,
                "index":                   ob.get("index", len(df_4h) - 10),
                "htf_alignment":           h1_conf["h1_trend_alignment"],
                "h1_trend_alignment":      h1_conf["h1_trend_alignment"],
                "h1_structure_shift":      h1_conf["h1_structure_shift"],
                "h1_displacement":         h1_conf["h1_displacement"],
                "h1_liquidity_sweep":      h1_conf["h1_liquidity_sweep"],
                "m5_micro_bos":            m5_conf["m5_micro_bos"],
                "m5_sweep_confirmation":   m5_conf["m5_sweep_confirmation"],
                "m5_entry_candle_quality": m5_conf["m5_entry_candle_quality"],
                "approach_distance":       approach["approach_distance"],
                "approach_momentum":       approach["approach_momentum"],
                "candle_body_ratio":       approach["candle_body_ratio"],
                "tf_source":               "4h",
                "touch_count":             max(1, int(((df_4h["low"] <= ob["top"]) &
                                                        (df_4h["high"] >= ob["bottom"])).sum())),
                "fvg_overlap":             fvg_overlap_flag,
            }

            base_prob = pred._ml_score_htf_zone(
                {"direction": direction, "strength": ob["strength"],
                 "index": ob.get("index", len(df_4h) - 1),
                 "mid": ob["mid"], "top": ob["top"], "bottom": ob["bottom"]},
                df_4h, "4h", htf_context=htf_context, zone_override=full_ob_zone)

            confs_count = s["confs"]
            boost = min(confs_count * 0.02, pred.max_boost)
            final_prob = pred.apply_boost_ceiling(base_prob, boost)

            threshold_pct = pred.prob_threshold * 100
            pass_ml  = base_prob >= MIN_PROB_GATE
            pass_thr = final_prob * 100 >= threshold_pct

            result_str = (
                _c("green", "PASS → bias check") if (pass_ml and pass_thr)
                else (_c("red", f"FAIL base<{MIN_PROB_GATE}") if not pass_ml
                      else _c("red", f"FAIL final<{threshold_pct:.0f}%"))
            )

            if pass_ml and pass_thr:
                gate_totals["passed"] += 1
            elif not pass_ml:
                gate_totals["prob_ml"] += 1
            else:
                gate_totals["bias"] += 1   # may also be killed by bias filter later

            print(f"  {i+1:<3}  {direction:<5}  {s['entry']:>10.5f}  "
                  f"{base_prob*100:>9.1f}%  {final_prob*100:>10.1f}%  "
                  f"{threshold_pct:>9.1f}%  {result_str}")

            # Print feature vector for ALL ML-scored zones (failing ones most useful for diagnosis)
            _print_feature_snapshot(full_ob_zone, s, h1_conf, m5_conf,
                                    approach, real_overlap, fvg_overlap_flag)

        except Exception as e:
            print(f"  {i+1:<3}  {direction:<5}  {s['entry']:>10.5f}  "
                  f"{'ERROR':>10}  {'—':>11}  {'—':>10}  "
                  + _c("red", str(e)))
            if "--verbose" in sys.argv:
                traceback.print_exc()

    # ── Bias filter simulation ─────────────────────────────────────────────────
    _section(f"BIAS FILTER SIMULATION — {symbol}")
    print(f"  H1 trend = {_c('cyan', h1_trend)}")
    print(f"  Rule: bullish → keep BUY with entry < {current_price:.5f}")
    print(f"        bearish → keep SELL with entry > {current_price:.5f}")
    print(f"        neutral → keep everything (no filter applied)\n")
    for s in survivors:
        entry     = s["entry"]
        direction = s["direction"]
        if h1_trend == "neutral":
            verdict = _c("green", "PASS (neutral — no bias filter)")
        elif h1_trend == "bullish" and direction == "BUY" and entry < current_price:
            verdict = _c("green", f"PASS  BUY {entry:.5f} < price {current_price:.5f}")
        elif h1_trend == "bearish" and direction == "SELL" and entry > current_price:
            verdict = _c("green", f"PASS  SELL {entry:.5f} > price {current_price:.5f}")
        else:
            reason = (
                f"BUY but trend is {h1_trend}" if direction == "BUY" and h1_trend != "bullish"
                else f"SELL but trend is {h1_trend}" if direction == "SELL" and h1_trend != "bearish"
                else f"entry {entry:.5f} on wrong side of price {current_price:.5f}"
            )
            verdict = _c("red", f"FAIL  {reason}")
        print(f"    {direction} @ {entry:.5f}  → {verdict}")

    _suggest_fixes(gate_totals, symbol, current_atr, pred.pip_size, h1_trend)


def _top_killer(gate_totals: Dict) -> str:
    order = ["dist", "confs", "direction", "rr", "strength", "prob_ml", "bias"]
    top   = max(order, key=lambda k: gate_totals.get(k, 0))
    labels = {
        "dist": "Gate 1 (distance)", "confs": "Gate 2 (confirmation candles)",
        "direction": "Gate 3 (side-of-price)", "rr": "Gate 4 (RR out of range)",
        "strength": "Gate 5 (strength < 0.5)", "prob_ml": "Gate 6 (ML prob < 0.55)",
        "bias": "Gate 8 (bias filter)",
    }
    return labels.get(top, top)


def _print_feature_snapshot(zone, s, h1_conf, m5_conf, approach,
                              real_overlap, fvg_overlap) -> None:
    print(f"\n    {'Feature snapshot (key model inputs)':}")
    features = {
        "h1_trend_alignment":    h1_conf["h1_trend_alignment"],
        "h1_structure_shift":    h1_conf["h1_structure_shift"],
        "h1_displacement":       round(h1_conf["h1_displacement"], 3),
        "h1_liquidity_sweep":    h1_conf["h1_liquidity_sweep"],
        "m5_micro_bos":          m5_conf["m5_micro_bos"],
        "m5_sweep_confirmation": m5_conf["m5_sweep_confirmation"],
        "m5_entry_candle_qual":  round(m5_conf["m5_entry_candle_quality"], 3),
        "approach_distance(p)":  round(approach["approach_distance"], 1),
        "approach_momentum":     round(approach["approach_momentum"], 3),
        "candle_body_ratio":     round(approach["candle_body_ratio"], 3),
        "zone_confluence":       real_overlap,
        "h4_fvg_inside":         int(zone.get("fvg_overlap", False)),
        "zone_strength":         round(zone.get("strength", 0), 3),
        "touch_count":           zone.get("touch_count", 1),
        "risk_pips":             round(s["risk_p"], 1),
    }
    for k, v in features.items():
        flag = ""
        if k == "h1_trend_alignment" and v == -1:
            flag = _c("yellow", "  ⚠ counter-trend (-1 suppresses model)")
        if k == "h1_liquidity_sweep" and v == 0:
            flag = _c("yellow", "  ⚠ no sweep (model trained mean=0.97)")
        if k == "m5_micro_bos" and v == 0:
            flag = _c("yellow", "  ⚠ no 5M BOS")
        print(f"      {k:<26}: {v}{flag}")
    print()


def _suggest_fixes(gate_totals: Dict, symbol: str, atr: float,
                   pip_size: float, h1_trend: str) -> None:
    _MAX_RR   = _get_max_rr(symbol)
    _ml_gate  = _get_ml_gate(symbol)
    print(f"\n{_hr()}")
    print(_c("bold", "  STATUS & REMAINING ISSUES"))
    print(_hr())

    if gate_totals["rr"] > 0:
        atr_p = atr / pip_size
        if symbol == "XAU/USD":
            print(f"  ★ Gate 4 (RR) killed {gate_totals['rr']} OB(s). [Gold: MAX_RR=6.0 unchanged]")
        else:
            print(f"  ✅ Gate 4 fix APPLIED: MAX_HTF_RR raised to {_MAX_RR} for {symbol}.")
            print(f"     TP clamping also active — reward capped at risk × {_MAX_RR} × 0.95.")
            print(f"     {gate_totals['rr']} OB(s) still killed = their raw RR still exceeds {_MAX_RR}.")
            print(f"     These have reaction_size so large that even clamped TP > {_MAX_RR}× risk.\n")

    if gate_totals["direction"] > 0:
        print(f"  ★ Gate 3 (direction) killed {gate_totals['direction']} OB(s).")
        print(f"    atr_tol = {atr * ATR_TOL_MULT:.5f} ({atr * ATR_TOL_MULT / pip_size:.1f}p).")
        print(f"    Zone mid is on wrong side of price ± 1 ATR.\n")

    if gate_totals["confs"] > 0:
        print(f"  ★ Gate 2 (confs) killed {gate_totals['confs']} OB(s).")
        print(f"    Fresh OBs with only 1 post-formation bar always fail confs < 2.\n")

    if gate_totals["prob_ml"] > 0:
        if symbol == "XAU/USD":
            print(f"  ★ Gate 6 (ML) killed {gate_totals['prob_ml']} OB(s). [Gold: gate fixed at 0.55]")
        else:
            print(f"  ✅ Gate 6 fix APPLIED: ML gate = {_ml_gate*100:.0f}% for {symbol} at current UTC hour.")
            print(f"     {gate_totals['prob_ml']} OB(s) still failed — base_prob < {_ml_gate*100:.0f}%.")
            print(f"     Root cause: h1_liquidity_sweep, m5_micro_bos, m5_sweep_confirmation = 0.")
            if datetime.utcnow().hour < 7:
                print(f"     ⏰ Still Asia session — these features are expected to be 0 pre-London.")
                print(f"     Rerun at 08:00 UTC to see if London open activates these features.\n")
            else:
                print(f"     London session active — low scores indicate genuine lack of confirmation.\n")

    if gate_totals["bias"] > 0:
        print(f"  ★ Bias filter killed {gate_totals['bias']} OB(s) post-ML.")
        print(f"    h1_trend = {h1_trend} → only {('BUY' if h1_trend == 'bullish' else 'SELL')} zones below/above price pass.\n")

    if gate_totals.get("passed", 0) > 0:
        print(_c("green", f"  ✅ {gate_totals['passed']} OB(s) passed all gates including ML and bias filter."))


def _get_htf_analyzer_param():
    """Dummy — used only in the fallback message above."""
    return type("_", (), {"ob_min_reaction_atr": 2.5})()


# ══════════════════════════════════════════════════════════════════════════════
#  D1 gate diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_d1_gate(symbol: str) -> None:
    _section(f"D1 PREMIUM/DISCOUNT GATE — {symbol}")
    pred = _safe_predictor(symbol, "rev")
    if pred is None:
        return

    df_1d = _fetch_bars_safe(pred, "1day", 90)
    df_15m = _fetch_bars_safe(pred, "15min", 200)
    if df_1d is None or len(df_1d) < 20:
        print(_c("yellow", "  [SKIP] D1 bars unavailable."))
        return

    current_price = df_15m["close"].iloc[-1] if (df_15m is not None and len(df_15m) > 0) \
                    else float(df_1d["close"].iloc[-1])

    tail = df_1d.tail(60)
    raw_max  = float(tail["high"].max())
    raw_min  = float(tail["low"].min())
    p95_high = float(tail["high"].quantile(0.95))
    p5_low   = float(tail["low"].quantile(0.05))
    rng      = p95_high - p5_low
    position = (current_price - p5_low) / rng if rng > 0 else 0.5

    zone_label = (
        _c("green",  "🟢 DISCOUNT (<40%)")  if position < 0.40 else
        _c("yellow", "🟡 EQUILIBRIUM (40-60%)") if position < 0.60 else
        _c("red",    "🔴 PREMIUM (>60%)")
    )

    print(f"  {'Symbol':<22}: {symbol}")
    print(f"  {'Current price':<22}: {current_price:.5f}")
    print(f"  {'D1 60-bar raw max':<22}: {raw_max:.5f}")
    print(f"  {'D1 60-bar raw min':<22}: {raw_min:.5f}")
    print(f"  {'P95 high (filter)':<22}: {p95_high:.5f}")
    print(f"  {'P5 low (filter)':<22}: {p5_low:.5f}")
    print(f"  {'P5→P95 range':<22}: {rng:.5f}  ({rng / pred.pip_size:.1f} pips)")
    print(f"  {'Price position':<22}: {position:.3f}  ({position * 100:.1f}%)")
    print(f"  {'D1 zone':<22}: {zone_label}")
    print()
    print(f"  Interpretation:")
    print(f"    BUY signals are favoured when price is in DISCOUNT (position < 0.40).")
    print(f"    SELL signals are favoured when price is in PREMIUM (position > 0.60).")
    if position < 0.20:
        print(_c("green", f"    → Deep discount ({position*100:.0f}%). High-confidence BUY bias. (Gold-like)"))
    elif position > 0.80:
        print(_c("red", f"    → Deep premium ({position*100:.0f}%). High-confidence SELL bias."))
    else:
        print(_c("yellow", f"    → Mid-range ({position*100:.0f}%). Model gets less D1 premium/discount signal."))


# ══════════════════════════════════════════════════════════════════════════════
#  RR sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_rr_sensitivity(symbol: str) -> None:
    _section(f"RR SENSITIVITY ANALYSIS — {symbol}")
    pred = _safe_predictor(symbol, "rev")
    if pred is None:
        return

    df_4h = _fetch_bars_safe(pred, "4h", 200)
    df_15m = _fetch_bars_safe(pred, "15min", 200)
    if df_4h is None:
        return

    current_price = df_15m["close"].iloc[-1] if (df_15m is not None and len(df_15m)) else df_4h["close"].iloc[-1]
    current_atr   = df_4h["atr"].iloc[-1]
    atr_p         = current_atr / pred.pip_size

    obs = pred.htf_analyzer.detect_order_blocks(df_4h, "4h")
    # Only look at OBs that passed gates 1-3
    max_dist_pips = MAX_DIST_ATR * atr_p
    atr_tol = current_atr * ATR_TOL_MULT

    rr_candidates = []
    for ob in obs:
        mid   = ob["mid"]
        dist_p = abs(current_price - mid) / pred.pip_size
        if dist_p > max_dist_pips:
            continue
        future = df_4h.iloc[ob["index"] + 1:]
        if len(future) < 2:
            continue
        if ob["type"] == "bullish":
            confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]["low"] >= ob["bottom"])
        else:
            confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]["high"] <= ob["top"])
        if confs < 2:
            continue
        direction = "BUY" if ob["type"] == "bullish" else "SELL"
        side_fail = (direction == "BUY"  and mid > current_price + atr_tol) or \
                    (direction == "SELL" and mid < current_price - atr_tol)
        if side_fail:
            continue
        rr_candidates.append(ob)

    if not rr_candidates:
        print(f"  No OBs survived Gates 1-3 to test RR sensitivity.")
        return

    _MAX_RR = _get_max_rr(symbol)
    print(f"  {'4H ATR':<30}: {current_atr:.5f} = {atr_p:.1f} pips")
    print(f"  {'Current price':<30}: {current_price:.5f}")
    print(f"  {'MAX_HTF_RR (this pair)':<30}: {_MAX_RR}")
    print(f"  {'OBs reaching RR gate':<30}: {len(rr_candidates)}\n")

    # Show RR under different TP formulas including the production TP-clamp
    print(f"  {'OB #':<5}  {'type':<8}  {'reac(p)':>8}  {'risk(p)':>8}  "
          f"{'TP=reac×1.5':>12}  {'TP=clamped':>11}  {'TP=ATR×2':>10}  Result")
    print(f"  {_hr('-', 80)}")

    for i, ob in enumerate(rr_candidates):
        direction = "BUY" if ob["type"] == "bullish" else "SELL"
        entry    = ob["mid"]
        _atr_buf = current_atr * 0.2
        sl       = ob["bottom"] - _atr_buf if direction == "BUY" else ob["top"] + _atr_buf
        risk     = abs(entry - sl)
        risk_p   = risk / pred.pip_size
        reac_p   = ob["reaction_size"] / pred.pip_size

        rr_reac  = (ob["reaction_size"] * 1.5) / risk if risk > 0 else 0

        # Production TP-clamped RR (forex only)
        if symbol != "XAU/USD" and risk > 0:
            _max_rwd   = risk * _MAX_RR * 0.95
            tp_clamped = min(ob["reaction_size"] * 1.5, _max_rwd)
            rr_clamped = tp_clamped / risk
        else:
            rr_clamped = rr_reac   # Gold: no clamp applied

        rr_atr2 = (current_atr * 2.0) / risk if risk > 0 else 0

        def _fmt_rr(rr_val, max_rr=_MAX_RR):
            if MIN_HTF_RR <= rr_val <= max_rr:
                return _c("green", f"{rr_val:.2f} ✓")
            return _c("red", f"{rr_val:.2f} ✗")

        clamp_note = ""
        if symbol != "XAU/USD":
            if rr_reac > _MAX_RR and MIN_HTF_RR <= rr_clamped <= _MAX_RR:
                clamp_note = _c("green", " ← CLAMP SAVES THIS")
            elif rr_reac > _MAX_RR and rr_clamped > _MAX_RR:
                clamp_note = _c("red", " ← still > ceiling after clamp")

        print(f"  {i+1:<5}  {ob['type'][:7]:<8}  {reac_p:>8.1f}  {risk_p:>8.1f}  "
              f"  {_fmt_rr(rr_reac):<20}  {_fmt_rr(rr_clamped):<19}  {_fmt_rr(rr_atr2):<18}"
              f"{clamp_note}")

    print(f"\n  Fix recommendation:")
    print(f"  In htf_trade_detect(), replace:")
    print(f"    tp_target = entry + ob['reaction_size'] * 1.5")
    print(f"  With:")
    print(f"    tp_target = entry + max(ob['reaction_size'], current_atr) * 1.5")
    print(f"  This ensures TP is at least 1.5× ATR even when the initial impulse was small.")


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-pair comparison summary
# ══════════════════════════════════════════════════════════════════════════════

def cross_pair_summary(symbols: List[str]) -> None:
    _section("CROSS-PAIR FILTER COMPARISON SUMMARY")

    rows = []
    for sym in symbols:
        pred = _safe_predictor(sym, "rev")
        if pred is None:
            continue
        df_4h  = _fetch_bars_safe(pred, "4h",    200)
        df_15m = _fetch_bars_safe(pred, "15min", 200)
        df_1d  = _fetch_bars_safe(pred, "1day",   90)

        if df_4h is None:
            continue

        current_price = df_15m["close"].iloc[-1] if (df_15m is not None and len(df_15m)) else df_4h["close"].iloc[-1]
        current_atr   = df_4h["atr"].iloc[-1]
        atr_p         = current_atr / pred.pip_size
        max_dist_p    = MAX_DIST_ATR * atr_p
        atr_tol       = current_atr * ATR_TOL_MULT

        htf_ctx = pred.get_htf_context(df_1h=df_15m, df_4h_c=df_4h, df_4h=df_4h, df_1d=df_1d)
        h1_trend = htf_ctx.get("h1_trend", "?") if htf_ctx else "?"
        _MAX_RR  = _get_max_rr(sym)
        _ml_g    = _get_ml_gate(sym)

        obs = pred.htf_analyzer.detect_order_blocks(df_4h, "4h")
        g1 = g2 = g3 = g4 = g5 = passed = 0
        for ob in obs:
            mid = ob["mid"]
            dist_p = abs(current_price - mid) / pred.pip_size
            if dist_p > max_dist_p:     g1 += 1; continue
            fut = df_4h.iloc[ob["index"] + 1:]
            if len(fut) < 2:             g2 += 1; continue
            if ob["type"] == "bullish":
                confs = sum(1 for i in range(min(5, len(fut))) if fut.iloc[i]["low"] >= ob["bottom"])
            else:
                confs = sum(1 for i in range(min(5, len(fut))) if fut.iloc[i]["high"] <= ob["top"])
            if confs < 2:                g2 += 1; continue
            direction = "BUY" if ob["type"] == "bullish" else "SELL"
            if (direction == "BUY"  and mid > current_price + atr_tol) or \
               (direction == "SELL" and mid < current_price - atr_tol):
                                         g3 += 1; continue
            entry = mid
            sl    = ob["bottom"] - current_atr * 0.2 if direction == "BUY" else ob["top"] + current_atr * 0.2
            risk  = abs(entry - sl)
            tp_raw = ob["reaction_size"] * 1.5
            if sym != "XAU/USD" and risk > 0:
                _max_rwd = risk * _MAX_RR * 0.95
                reward   = min(tp_raw, _max_rwd)
            else:
                reward   = tp_raw
            rr    = reward / risk if risk > 0 else 0
            if not (MIN_HTF_RR <= rr <= _MAX_RR):
                                         g4 += 1; continue
            if ob.get("strength", 0) < MIN_HTF_STRENGTH:
                                         g5 += 1; continue
            passed += 1

        # D1 position
        d1_pos = "?"
        if df_1d is not None and len(df_1d) >= 20:
            tail = df_1d.tail(60)
            p95  = float(tail["high"].quantile(0.95))
            p5   = float(tail["low"].quantile(0.05))
            rng  = p95 - p5
            if rng > 0:
                pos = (current_price - p5) / rng
                d1_pos = f"{pos*100:.0f}% {'DIS' if pos < 0.4 else ('PREM' if pos > 0.6 else 'EQ')}"

        rows.append({
            "Symbol": sym, "Price": f"{current_price:.4f}",
            "ATR(p)": f"{atr_p:.1f}", "H1 Trend": h1_trend,
            "MaxRR": f"{_MAX_RR:.0f}",
            "MLgate": f"{_ml_g*100:.0f}%",
            "OBs": len(obs), "G1 dist": g1, "G2 confs": g2,
            "G3 dir": g3, "G4 rr": g4, "G5 str": g5,
            "ML cands": passed, "D1 pos": d1_pos,
        })

    if not rows:
        print("  No data available.")
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}

    header = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  " + _hr("-", len(header) - 2))
    for r in rows:
        line_parts = []
        for c in cols:
            v = str(r[c])
            if c == "ML cands":
                v = _c("green", v) if int(r[c]) > 0 else _c("red", v)
            elif c in ("G1 dist", "G2 confs", "G3 dir", "G4 rr", "G5 str") and int(r[c]) > 0:
                v = _c("red", v)
            elif c == "H1 Trend":
                v = _c("cyan", v)
            line_parts.append(v.ljust(widths[c]))
        print("  " + "  ".join(line_parts))

    print(f"\n  G1=dist  G2=confs  G3=direction  G4=RR  G5=strength  ML cands=passed all gates")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Forex ML System — filter pipeline diagnostic")
    parser.add_argument("--symbol",  default=None,
                        help="Single symbol to diagnose (e.g. EUR/USD)")
    parser.add_argument("--all",     action="store_true",
                        help="Run all four pairs")
    parser.add_argument("--summary", action="store_true",
                        help="Cross-pair comparison table only")
    parser.add_argument("--rr",      action="store_true",
                        help="Include RR sensitivity analysis")
    parser.add_argument("--d1",      action="store_true",
                        help="Include D1 gate diagnostic")
    parser.add_argument("--out",     default=None,
                        help="Write output to file instead of stdout")
    parser.add_argument("--verbose", action="store_true",
                        help="Print tracebacks for ML scoring errors")
    args = parser.parse_args()

    # Redirect to file if requested
    if args.out:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

    symbols_to_run = (
        ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"] if args.all or args.summary
        else ([args.symbol] if args.symbol else ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"])
    )

    print(f"\n{'═'*90}")
    print(_c("bold", f"  FOREX ML FILTER DIAGNOSTIC   —   {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"))
    print(f"{'═'*90}")

    if args.summary:
        cross_pair_summary(symbols_to_run)
    else:
        for sym in symbols_to_run:
            diagnose_ob_filter_pipeline(sym)
            if args.d1:
                diagnose_d1_gate(sym)
            if args.rr:
                diagnose_rr_sensitivity(sym)

        if len(symbols_to_run) > 1:
            cross_pair_summary(symbols_to_run)

    print(f"\n{'═'*90}\n")

    if args.out:
        content = buf.getvalue()
        sys.stdout = old_stdout
        # Strip ANSI codes for file output
        import re
        content = re.sub(r'\033\[[0-9;]+m', '', content)
        with open(args.out, "w") as f:
            f.write(content)
        print(f"  Report saved → {args.out}")


if __name__ == "__main__":
    main()