#!/usr/bin/env python3
"""
reevaluate_live_signals.py
==========================
Re-evaluates every live signal in signals/all_signals.json from its original
proposal state using the verified Two-Stage Execution Engine, Unified Live Book
Advisor Gate (ADVISOR_GATE=1), and proper confirmation-anchored outcome tracking.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from zone_arrival_engine import ZoneState, ZoneArrivalConfig, evaluate_zone_step
from signal_advisor import advise as advisor_advise
from production_predictor import ProductionForexPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Reevaluator")

API_KEYS = {
    "EUR/USD": os.getenv("TWELVE_KEY_EURUSD", "ea3e4fce074e403f8cd4c83dd0035643"),
    "GBP/USD": os.getenv("TWELVE_KEY_GBPUSD", "2522b470eba249f88d2e29575035914e"),
    "USD/JPY": os.getenv("TWELVE_KEY_USDJPY", "5bb48b4211144b92a4487ce7c63d1b7a"),
    "XAU/USD": os.getenv("TWELVE_KEY_XAUUSD", "63a513bb7eb848748d28d3deb64c1b59"),
}

PIP_SIZES = {
    "EUR/USD": 0.0001,
    "GBP/USD": 0.0001,
    "USD/JPY": 0.01,
    "XAU/USD": 1.0,
}

def reevaluate_all():
    signals_file = Path("signals/all_signals.json")
    if not signals_file.exists():
        print("signals/all_signals.json not found!")
        return

    with open(signals_file, "r") as f:
        signals = json.load(f)

    print(f"Loaded {len(signals)} signals from {signals_file}\n")

    # Initialize predictors for data fetching
    predictors = {}
    for sym, key in API_KEYS.items():
        try:
            predictors[sym] = ProductionForexPredictor(
                api_key=key,
                symbol=sym,
                model_path="models/model_v6_reversion.pkl",
                scaler_path="models/scaler_v6_reversion.pkl",
                encoders_path="models/encoders_v6_reversion.pkl",
                stats_path="models/training_stats_v6_reversion.pkl",
            )
        except Exception as e:
            logger.error(f"Failed to initialize predictor for {sym}: {e}")

    # Pre-fetch bars
    bar_cache = {}
    for sym, pred in predictors.items():
        print(f"Fetching bars for {sym}...")
        df15 = pred.fetch_bars("15min", outputsize=500)
        df5  = pred.fetch_bars("5min",  outputsize=500)
        df4h = pred.fetch_bars("4h",    outputsize=150)
        bar_cache[sym] = {"15min": df15, "5min": df5, "4h": df4h}

    cfg = ZoneArrivalConfig(
        min_rejection_wick_atr=0.40,
        max_bars_in_zone=16,
    )

    results = []

    for i, sig in enumerate(signals):
        sym = sig.get("symbol")
        direction = sig.get("direction")
        pip_size = PIP_SIZES.get(sym, 0.0001)

        # Restore original proposal levels
        orig_entry = sig.get("original_entry", sig.get("entry"))
        orig_sl    = sig.get("original_sl", sig.get("sl"))
        orig_tp    = sig.get("original_tp", sig.get("tp"))
        zone_top   = float(sig.get("zone_top", orig_entry + pip_size))
        zone_bot   = float(sig.get("zone_bottom", orig_entry - pip_size))
        zone_mid   = float(sig.get("zone_mid", (zone_top + zone_bot) / 2.0))
        prop_time  = pd.to_datetime(sig.get("timestamp")).tz_localize(None) if pd.to_datetime(sig.get("timestamp")).tzinfo else pd.to_datetime(sig.get("timestamp"))

        old_status = sig.get("status")
        old_stage  = sig.get("stage")
        old_adv_q  = sig.get("advisor_qualified")

        bars = bar_cache.get(sym)
        if not bars or bars["15min"] is None or bars["15min"].empty:
            print(f"[{i}] {sym}: No price data available, skipping")
            continue

        df15 = bars["15min"].copy()
        df5  = bars["5min"].copy()
        df4h = bars["4h"].copy()

        # Step through bars from proposal time
        times = df15["time"]
        eval_bars = df15[times >= prop_time]
        if eval_bars.empty:
            eval_bars = df15.iloc[-1:]

        zone = {
            "symbol": sym, "direction": direction,
            "zone_top": zone_top, "zone_bottom": zone_bot,
            "zone_mid": zone_mid, "tp": orig_tp,
            "state": ZoneState.ARMED,
            "bars_in_zone": 0,
        }

        final_action = None
        for _, bar in eval_bars.iterrows():
            pos = int(times.searchsorted(bar['time'], side='right'))
            recent = df15.iloc[max(0, pos - 30):pos]
            res = evaluate_zone_step(zone, bar, recent, pip_size, cfg)
            act = res.get("action")
            if act in ("confirm", "violated", "expired"):
                final_action = (act, res, bar)
                break

        # Build re-evaluated record
        new_sig = dict(sig)
        new_sig["entry"] = orig_entry
        new_sig["sl"]    = orig_sl
        new_sig["tp"]    = orig_tp
        new_sig["original_entry"] = orig_entry
        new_sig["original_sl"]    = orig_sl
        new_sig["original_tp"]    = orig_tp
        new_sig.pop("exit_price", None)
        new_sig.pop("exit_time", None)
        new_sig.pop("outcome_reason", None)
        new_sig.pop("resolved_at", None)
        new_sig.pop("triggered", None)

        if final_action is None:
            new_sig["stage"] = zone.get("state", ZoneState.ARMED)
            new_sig["status"] = "pending"
            new_sig["_zone"] = zone
            new_sig["advisor_qualified"] = None
            new_sig["advisor_advice"] = None
            outcome_desc = f"Still {new_sig['stage']} (watching)"
        else:
            act, res, bar = final_action
            if act == "violated":
                new_sig["stage"] = ZoneState.VIOLATED
                new_sig["status"] = "violated"
                new_sig["advisor_qualified"] = False
                new_sig["advisor_tier"] = None
                new_sig["advisor_blocked"] = True
                new_sig["advisor_advice"] = "Distal violation (0 risk)"
                outcome_desc = "VIOLATED distal edge (0 risk)"
            elif act == "expired":
                new_sig["stage"] = ZoneState.EXPIRED
                new_sig["status"] = "expired_untriggered"
                new_sig["advisor_qualified"] = False
                new_sig["advisor_tier"] = None
                new_sig["advisor_blocked"] = True
                new_sig["advisor_advice"] = "Expired without confirmation (0 risk)"
                outcome_desc = "EXPIRED without confirmation (0 risk)"
            elif act == "confirm":
                ts = res["trade_signal"]
                new_sig["stage"] = ZoneState.CONFIRMED
                new_sig["entry"] = ts["entry"]
                new_sig["sl"]    = ts["sl"]
                new_sig["tp"]    = ts["tp"]
                new_sig["risk_pips"]   = ts.get("risk_pips")
                new_sig["reward_pips"] = ts.get("reward_pips")
                new_sig["risk_reward"] = ts.get("risk_reward")
                new_sig["rr"]          = ts.get("risk_reward")
                new_sig["adjusted"]    = True
                new_sig["confirmed_at"] = str(bar["time"])
                new_sig["sweep_extreme"] = ts.get("sweep_extreme")
                new_sig["_ts_wick"] = ts.get("rejection_wick_atr")
                new_sig["_ts_touch_time"] = ts.get("touch_time") or str(bar["time"])

                # Run Advisor
                adv = advisor_advise(
                    direction, df15, df5, df4h,
                    pd.to_datetime(new_sig["_ts_touch_time"]),
                    new_sig["_ts_wick"],
                    risk_reward=new_sig.get("risk_reward"),
                    consecutive_bars=(ts.get("features") or {}).get("consecutive_directional_bars")
                )
                # If API bar history is too short for old signals, preserve the original real-time score
                if adv.get("missing_inputs") and sig.get("advisor") and not sig.get("advisor", {}).get("missing_inputs"):
                    adv = sig["advisor"]
                new_sig["advisor"] = adv
                new_sig["advisor_qualified"] = adv.get("qualified", False)
                new_sig["advisor_tier"] = adv.get("tier", "skip")
                new_sig["advisor_advice"] = adv.get("advice", "")

                # Check ADVISOR_GATE
                is_blocked = not new_sig["advisor_qualified"]
                new_sig["advisor_blocked"] = is_blocked

                # Replay outcome from confirmation bar forward for all confirmed trades
                conf_time = bar["time"]
                hold_bars = df15[times > conf_time]
                trade_status = "pending"
                resolved_bar_time = None
                c_sl = new_sig["sl"]
                c_tp = new_sig["tp"]

                for _, h_bar in hold_bars.iterrows():
                    hit_sl = (h_bar["low"] <= c_sl) if direction == "BUY" else (h_bar["high"] >= c_sl)
                    hit_tp = (h_bar["high"] >= c_tp) if direction == "BUY" else (h_bar["low"] <= c_tp)
                    if hit_sl and hit_tp:
                        trade_status = "hit_sl"
                        resolved_bar_time = h_bar["time"]
                        break
                    elif hit_sl:
                        trade_status = "hit_sl"
                        resolved_bar_time = h_bar["time"]
                        break
                    elif hit_tp:
                        trade_status = "hit_tp"
                        resolved_bar_time = h_bar["time"]
                        break

                new_sig["status"] = trade_status
                if trade_status in ("hit_tp", "hit_sl"):
                    new_sig["resolved_at"] = str(resolved_bar_time)
                    new_sig["triggered"] = True
                    outcome_desc = f"CONFIRMED [{new_sig['advisor_tier'].upper()}] -> {trade_status.upper()} @ {resolved_bar_time}"
                else:
                    if not hold_bars.empty:
                        new_sig["status"] = "triggered"
                        new_sig["triggered"] = True
                        outcome_desc = f"CONFIRMED [{new_sig['advisor_tier'].upper()}] (TRIGGERED/Active)"
                    else:
                        new_sig["status"] = "pending"
                        outcome_desc = f"CONFIRMED [{new_sig['advisor_tier'].upper()}] (Active Pending)"

        results.append((sig, new_sig, outcome_desc))

    print("\n" + "="*110)
    print(f"{'#':<3} {'Symbol':<8} {'Dir':<5} {'Proposed':<16} {'Old Status / Stage':<26} {'New Status / Stage':<26} {'Action / Outcome'}")
    print("="*110)

    updated_signals = []
    for i, (orig, updated, desc) in enumerate(results):
        old_label = f"{orig.get('status')} [{orig.get('stage')}]"
        new_label = f"{updated.get('status')} [{updated.get('stage')}]"
        sym_dir = f"{updated.get('symbol')} {updated.get('direction')}"
        ts_str = str(orig.get('timestamp'))[:16]
        print(f"{i:<3} {updated.get('symbol'):<8} {updated.get('direction'):<5} {ts_str:<16} {old_label:<26} {new_label:<26} {desc}")
        updated_signals.append(updated)
    print("="*110)

    # Save to all_signals.json
    with open(signals_file, "w") as f:
        json.dump(updated_signals, f, indent=2, default=str)
    print(f"\n💾 Successfully saved {len(updated_signals)} re-evaluated signals to {signals_file}")

if __name__ == "__main__":
    reevaluate_all()
