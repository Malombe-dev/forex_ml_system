"""
zone_arrival_engine.py
======================
Two-Stage Execution Engine for Order Blocks and Fair Value Gaps.

Stage 1: Zone Watchlist (ARMED)
  HTF (4H/1H) zones are registered in an ARMED state. Blind limit orders
  are NOT placed while price is far away.

Stage 2: Arrival, Invalidation & Closed-Bar Recomputation (IN_ZONE -> CONFIRMED / VIOLATED)
  - Distal Invalidation: If a candle closes through the distal edge, the zone
    is flagged VIOLATED and discarded with 0 risk.
  - In-Zone Tracking: While price is inside the zone, track the sweep extreme
    (lowest low for demand BUY, highest high for supply SELL).
  - Confirmation Candle (Rule A): When a closed candle confirms rejection
    (closes directionally with a rejection wick >= min_rejection_wick_atr),
    trigger CONFIRMED entry.
  - Geometry Recomputation:
      entry = bar.close (or equilibrium limit)
      sl    = sweep_extreme +/- dynamic_sl_cushion (anchored behind actual wick)
      tp    = opposing liquidity maintaining >= 1.5 R:R floor
  - Equilibrium Limit (Rule B): Limit order strictly at the 50% zone midpoint
    if enabled, preventing shallow boundary fills.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ZoneState:
    ARMED = "ARMED"          # POI identified, watching for arrival
    IN_ZONE = "IN_ZONE"      # Price penetrated proximal boundary, awaiting confirmation
    CONFIRMED = "CONFIRMED"  # Rejection candle confirmed -> trade ready for execution
    VIOLATED = "VIOLATED"    # Bar closed through distal boundary -> destroyed, 0 risk
    EXPIRED = "EXPIRED"      # TTL elapsed or max bars in zone reached without confirmation


@dataclass
class ZoneArrivalConfig:
    min_rejection_wick_atr: float = 0.40     # minimum rejection wick / 15m ATR
    min_rr_floor: float = 1.5                # minimum risk:reward floor
    max_bars_in_zone: int = 16               # max 15m bars inside zone without rejection (4 hours)
    violation_buffer_atr: float = 0.05       # buffer beyond distal edge to confirm breakout
    sl_pad_base_atr: float = 0.40            # baseline SL buffer (ATR units)
    sl_pad_wick_atr: float = 0.60            # dynamic buffer scaling with observed rejection wick
    entry_mode: str = "confirmation_candle"  # 'confirmation_candle' (Rule A) or 'equilibrium_limit' (Rule B)
    exhaustion_mid_threshold: float = 0.70
    exhaustion_high_threshold: float = 0.85

    # Arrival-time re-scoring (see ProductionForexPredictor.rescore_at_arrival).
    # The proposal-time probability is computed hours before price reaches the zone, from
    # approach/5m/1H features frozen at that moment. On CONFIRMED the model is re-run on
    # fresh closed bars. Log-only by default: rescore_min_prob=None never rejects anything.
    # Set e.g. 0.60 (fraction, same unit as prob_threshold) to reject confirmed zones whose
    # ARRIVAL probability falls below it -- only after the A/B shows arrival scores have edge.
    rescore_enabled: bool = True
    rescore_min_prob: Optional[float] = None

    # Baseline pip pads per symbol (matches production_predictor)
    baseline_pad_pips: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 3.0,
        "GBP/USD": 3.0,
        "USD/JPY": 4.0,
        "XAU/USD": 3.5,  # points
    })

    def get_baseline_pad(self, symbol: str, atr_15m: float, pip_size: float) -> float:
        sym = (symbol or "").upper()
        if "XAU" in sym or "GOLD" in sym:
            return 0.35 * atr_15m if atr_15m and atr_15m > 0 else 3.5
        elif "JPY" in sym:
            return self.baseline_pad_pips.get("USD/JPY", 4.0) * pip_size
        elif "GBP" in sym:
            return self.baseline_pad_pips.get("GBP/USD", 3.0) * pip_size
        else:
            return self.baseline_pad_pips.get(sym, 3.0) * pip_size


def compute_arrival_features(df: pd.DataFrame, direction: str, pip_size: float) -> Dict[str, Any]:
    """
    Computes pre-entry and touch dynamics on 15m bars up to the current closed bar.
    Identical feature set to backtest_replay and PriceZoneMonitor.
    """
    out = {}
    if df is None or len(df) < 2:
        return {"error": "insufficient_bars"}

    try:
        hi, lo, cl = df['high'], df['low'], df['close']
        tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
        atr_15m = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        if not np.isfinite(atr_15m) or atr_15m <= 0:
            atr_15m = pip_size * 10 if pip_size else 0.0010

        b_touch = df.iloc[-1]
        b_1ago  = df.iloc[-2] if len(df) >= 2 else b_touch
        b_2ago  = df.iloc[-3] if len(df) >= 3 else b_1ago

        if direction == 'BUY':
            approach_vel_atr = (b_2ago['open'] - b_touch['low']) / atr_15m
            touch_wick_atr   = (b_touch['close'] - b_touch['low']) / atr_15m
            is_directional   = bool(b_touch['close'] > b_touch['open'])
        else:
            approach_vel_atr = (b_touch['high'] - b_2ago['open']) / atr_15m
            touch_wick_atr   = (b_touch['high'] - b_touch['close']) / atr_15m
            is_directional   = bool(b_touch['close'] < b_touch['open'])

        pre_bar_range    = b_1ago['high'] - b_1ago['low']
        pre_bar_size_atr = pre_bar_range / atr_15m
        pre_body_ratio   = abs(b_1ago['close'] - b_1ago['open']) / (pre_bar_range + 1e-9)

        touch_bar_range   = b_touch['high'] - b_touch['low']
        touch_bar_size_atr = touch_bar_range / atr_15m
        touch_body_ratio   = abs(b_touch['close'] - b_touch['open']) / (touch_bar_range + 1e-9)

        consec = 0
        for k in range(1, min(7, len(df))):
            prev_b = df.iloc[-1 - k]
            is_dir = (prev_b['close'] < prev_b['open']) if direction == 'BUY' else (prev_b['close'] > prev_b['open'])
            if is_dir:
                consec += 1
            else:
                break

        # Exhaustion score calculation
        weights = {
            "touch_rejection_wick_atr":      0.45,
            "consecutive_directional_bars":  0.15,
            "touch_bar_size_atr":            0.20,
            "pre_touch_bar_size_atr":        0.10,
            "approach_velocity_30m_atr":     0.10,
        }
        caps = {
            "touch_rejection_wick_atr":      1.4,
            "consecutive_directional_bars":  3.0,
            "touch_bar_size_atr":            3.0,
            "pre_touch_bar_size_atr":        2.0,
            "approach_velocity_30m_atr":     3.5,
        }
        metrics = {
            "touch_rejection_wick_atr":      touch_wick_atr,
            "consecutive_directional_bars":  float(consec),
            "touch_bar_size_atr":            touch_bar_size_atr,
            "pre_touch_bar_size_atr":        pre_bar_size_atr,
            "approach_velocity_30m_atr":     approach_vel_atr,
        }
        score = sum(weights[k] * min(max(metrics[k], 0.0) / caps[k], 1.5) for k in weights)

        out = {
            "atr_15m":                     round(float(atr_15m), 6),
            "approach_velocity_30m_atr":   round(float(approach_vel_atr), 3),
            "pre_touch_bar_size_atr":      round(float(pre_bar_size_atr), 3),
            "pre_touch_body_ratio":        round(float(pre_body_ratio), 3),
            "touch_bar_size_atr":          round(float(touch_bar_size_atr), 3),
            "touch_body_ratio":            round(float(touch_body_ratio), 3),
            "touch_rejection_wick_atr":    round(float(touch_wick_atr), 3),
            "consecutive_directional_bars": int(consec),
            "is_directional_close":        is_directional,
            "exhaustion_score":            round(float(score), 4),
        }
    except Exception as e:
        out = {"error": str(e), "atr_15m": pip_size * 10 if pip_size else 0.0010}
    return out


def recompute_trade_geometry(zone: Dict[str, Any], entry_price: float,
                             sweep_extreme: float, features: Dict[str, Any],
                             pip_size: float, cfg: ZoneArrivalConfig) -> Dict[str, Any]:
    """
    Recomputes Entry, SL, and TP dynamically at zone confirmation:
    - SL is anchored behind sweep_extreme + dynamic buffer (absorbing momentum wicks)
    - TP holds the structural target if RR >= min_rr_floor, else extends to maintain min_rr_floor.
    """
    direction = zone.get("direction", "BUY")
    symbol    = zone.get("symbol", "")
    atr_15m   = features.get("atr_15m", pip_size * 10 if pip_size else 0.0010)
    wick_atr  = features.get("touch_rejection_wick_atr", 0.0) or 0.0

    zone_top  = float(zone.get("zone_top", entry_price))
    zone_bot  = float(zone.get("zone_bottom", entry_price))

    baseline_pad = cfg.get_baseline_pad(symbol, atr_15m, pip_size)
    dyn_sl_pad   = (cfg.sl_pad_base_atr + cfg.sl_pad_wick_atr * min(wick_atr, 1.5)) * atr_15m
    sl_cushion   = max(baseline_pad, dyn_sl_pad)

    if direction == "BUY":
        # Anchor SL below the lowest point reached (either zone bottom or sweep wick)
        sl_anchor = min(sweep_extreme, zone_bot)
        new_sl = round(sl_anchor - sl_cushion, 5)
        new_entry = round(entry_price, 5)
        risk = new_entry - new_sl
    else:
        # Anchor SL above the highest point reached (either zone top or sweep wick)
        sl_anchor = max(sweep_extreme, zone_top)
        new_sl = round(sl_anchor + sl_cushion, 5)
        new_entry = round(entry_price, 5)
        risk = new_sl - new_entry

    if risk <= 0:
        risk = baseline_pad
        new_sl = round(new_entry - (risk if direction == "BUY" else -risk), 5)

    # Check structural TP from original zone setup
    orig_tp = zone.get("tp")
    if orig_tp is not None:
        orig_tp = float(orig_tp)
        struct_reward = (orig_tp - new_entry) if direction == "BUY" else (new_entry - orig_tp)
        struct_rr = struct_reward / risk if risk > 0 else 0.0
        if struct_rr >= cfg.min_rr_floor:
            new_tp = round(orig_tp, 5)
            reward = struct_reward
            rr = round(struct_rr, 2)
        else:
            reward = risk * cfg.min_rr_floor
            new_tp = round(new_entry + (reward if direction == "BUY" else -reward), 5)
            rr = round(cfg.min_rr_floor, 2)
    else:
        reward = risk * cfg.min_rr_floor
        new_tp = round(new_entry + (reward if direction == "BUY" else -reward), 5)
        rr = round(cfg.min_rr_floor, 2)

    risk_pips   = round(risk / pip_size, 1) if pip_size else round(risk * 10000, 1)
    reward_pips = round(reward / pip_size, 1) if pip_size else round(reward * 10000, 1)

    return {
        "entry":       new_entry,
        "sl":          new_sl,
        "tp":          new_tp,
        "risk":        round(risk, 5),
        "reward":      round(reward, 5),
        "risk_pips":   risk_pips,
        "reward_pips": reward_pips,
        "rr":          f"1:{rr:.1f}",
        "risk_reward": f"1:{rr:.1f}",
        "sl_cushion_pips": round(sl_cushion / pip_size, 1) if pip_size else 0.0,
    }


def evaluate_zone_step(zone: Dict[str, Any], current_bar: pd.Series,
                       recent_bars: pd.DataFrame, pip_size: float,
                       cfg: Optional[ZoneArrivalConfig] = None) -> Dict[str, Any]:
    """
    Evaluates a single closed candle against an ARMED or IN_ZONE zone.
    
    State transitions:
      - ARMED -> VIOLATED (if bar closes through distal boundary)
      - ARMED -> IN_ZONE  (if bar penetrates proximal boundary without violating)
      - IN_ZONE -> VIOLATED (if bar closes through distal boundary)
      - IN_ZONE -> CONFIRMED (if confirmation candle rule triggers or equilibrium fill)
      - IN_ZONE -> EXPIRED (if max_bars_in_zone reached without confirmation)
      - ARMED -> EXPIRED (if valid_until elapsed)
    
    Returns:
      {
        "state": str,
        "action": "none" | "confirm" | "violated" | "expired",
        "reason": str,
        "features": dict,
        "trade_signal": dict | None,
      }
    """
    cfg = cfg or ZoneArrivalConfig()
    current_state = zone.get("state", ZoneState.ARMED)
    if current_state in (ZoneState.VIOLATED, ZoneState.CONFIRMED, ZoneState.EXPIRED):
        return {"state": current_state, "action": "none", "reason": f"terminal_state_{current_state}", "features": {}, "trade_signal": None}

    direction   = zone.get("direction", "BUY")
    zone_top    = float(zone.get("zone_top", zone.get("entry", 0.0)))
    zone_bottom = float(zone.get("zone_bottom", zone.get("entry", 0.0)))
    zone_mid    = float(zone.get("zone_mid", (zone_top + zone_bottom) / 2.0))

    bar_time  = current_bar.get("time", current_bar.get("datetime"))
    bar_open  = float(current_bar["open"])
    bar_high  = float(current_bar["high"])
    bar_low   = float(current_bar["low"])
    bar_close = float(current_bar["close"])

    # Compute arrival features up to this closed bar
    features = compute_arrival_features(recent_bars, direction, pip_size)
    atr_15m  = features.get("atr_15m", pip_size * 10 if pip_size else 0.0010)

    # 1. TTL Check
    valid_until = zone.get("valid_until")
    if valid_until and bar_time:
        try:
            bar_ts = pd.to_datetime(bar_time).tz_localize(None) if pd.to_datetime(bar_time).tzinfo else pd.to_datetime(bar_time)
            vu_ts  = pd.to_datetime(valid_until).tz_localize(None) if pd.to_datetime(valid_until).tzinfo else pd.to_datetime(valid_until)
            if bar_ts > vu_ts:
                zone["state"] = ZoneState.EXPIRED
                return {"state": ZoneState.EXPIRED, "action": "expired", "reason": "TTL elapsed", "features": features, "trade_signal": None}
        except Exception:
            pass

    # 2. Distal Violation Check (Did a closed bar blow through the far side?)
    distal_buffer = cfg.violation_buffer_atr * atr_15m
    if direction == "BUY":
        violated = bar_close < (zone_bottom - distal_buffer)
    else:
        violated = bar_close > (zone_top + distal_buffer)

    if violated:
        zone["state"] = ZoneState.VIOLATED
        zone["violated_at"] = str(bar_time)
        return {
            "state": ZoneState.VIOLATED,
            "action": "violated",
            "reason": f"Closed beyond distal edge (close={bar_close:.5f} vs distal={zone_bottom if direction=='BUY' else zone_top:.5f})",
            "features": features,
            "trade_signal": None,
        }

    # 3. Handle ARMED State
    if current_state == ZoneState.ARMED:
        # Check if price has touched/entered the zone
        if direction == "BUY":
            entered = bar_low <= zone_top
        else:
            entered = bar_high >= zone_bottom

        if not entered:
            # Still approaching
            dist_to_edge = abs(bar_close - (zone_top if direction == "BUY" else zone_bottom))
            return {
                "state": ZoneState.ARMED,
                "action": "none",
                "reason": f"Approaching zone ({dist_to_edge / pip_size:.1f} pips away)",
                "features": features,
                "trade_signal": None,
            }

        # Entered zone! Transition to IN_ZONE
        zone["state"] = ZoneState.IN_ZONE
        zone["touch_time"] = str(bar_time)
        zone["touch_price"] = bar_low if direction == "BUY" else bar_high
        zone["sweep_extreme"] = bar_low if direction == "BUY" else bar_high
        # 0, not 1: the IN_ZONE block below increments once for this same (first) bar, so
        # starting at 1 counted the entry bar twice and expired zones one bar early.
        zone["bars_in_zone"] = 0
        current_state = ZoneState.IN_ZONE

    # 4. Handle IN_ZONE State
    if current_state == ZoneState.IN_ZONE:
        # Update sweep extreme
        if direction == "BUY":
            zone["sweep_extreme"] = min(zone.get("sweep_extreme", bar_low), bar_low)
        else:
            zone["sweep_extreme"] = max(zone.get("sweep_extreme", bar_high), bar_high)

        bars_in_zone = zone.get("bars_in_zone", 1)

        # Check confirmation:
        # Rule A: Confirmation Candle (Rejection Wick + Directional Close)
        if cfg.entry_mode == "confirmation_candle":
            wick_atr = features.get("touch_rejection_wick_atr", 0.0)
            is_directional = features.get("is_directional_close", False)
            has_rejection = wick_atr >= cfg.min_rejection_wick_atr

            if is_directional and has_rejection:
                # Check exhaustion gate
                exhaustion = features.get("exhaustion_score", 0.0)
                if exhaustion >= cfg.exhaustion_high_threshold:
                    # Too high momentum knife, skip or wait for subsequent stabilizing bar
                    return {
                        "state": ZoneState.IN_ZONE,
                        "action": "none",
                        "reason": f"High exhaustion score ({exhaustion:.2f} >= {cfg.exhaustion_high_threshold}), waiting for stabilization",
                        "features": features,
                        "trade_signal": None,
                    }

                # Confirmed! Recompute geometry
                geo = recompute_trade_geometry(
                    zone, entry_price=bar_close, sweep_extreme=zone["sweep_extreme"],
                    features=features, pip_size=pip_size, cfg=cfg
                )
                zone["state"] = ZoneState.CONFIRMED
                zone["confirmed_at"] = str(bar_time)

                trade_signal = dict(zone)
                trade_signal.update({
                    "entry":             geo["entry"],
                    "sl":                geo["sl"],
                    "tp":                geo["tp"],
                    "risk_pips":         geo["risk_pips"],
                    "reward_pips":       geo["reward_pips"],
                    "risk_reward":       geo["risk_reward"],
                    "rr":                geo["rr"],
                    "stage":             ZoneState.CONFIRMED,
                    "status":            "pending",
                    "confirmation_rule": "Rule A (Rejection Candle)",
                    "rejection_wick_atr": wick_atr,
                    "sweep_extreme":     zone["sweep_extreme"],
                    "confirmed_at":      str(bar_time),
                    "features":          features,
                })
                return {
                    "state": ZoneState.CONFIRMED,
                    "action": "confirm",
                    "reason": f"Confirmation candle closed directionally with {wick_atr:.2f}x ATR wick",
                    "features": features,
                    "trade_signal": trade_signal,
                }

        # Rule B: Equilibrium Limit (Passive 50% Midpoint Fill)
        elif cfg.entry_mode == "equilibrium_limit":
            mid_reached = (bar_low <= zone_mid) if direction == "BUY" else (bar_high >= zone_mid)
            if mid_reached:
                geo = recompute_trade_geometry(
                    zone, entry_price=zone_mid, sweep_extreme=zone["sweep_extreme"],
                    features=features, pip_size=pip_size, cfg=cfg
                )
                zone["state"] = ZoneState.CONFIRMED
                zone["confirmed_at"] = str(bar_time)

                trade_signal = dict(zone)
                trade_signal.update({
                    "entry":             geo["entry"],
                    "sl":                geo["sl"],
                    "tp":                geo["tp"],
                    "risk_pips":         geo["risk_pips"],
                    "reward_pips":       geo["reward_pips"],
                    "risk_reward":       geo["risk_reward"],
                    "rr":                geo["rr"],
                    "stage":             ZoneState.CONFIRMED,
                    "status":            "pending",
                    "confirmation_rule": "Rule B (Equilibrium Midpoint)",
                    "sweep_extreme":     zone["sweep_extreme"],
                    "confirmed_at":      str(bar_time),
                    "features":          features,
                })
                return {
                    "state": ZoneState.CONFIRMED,
                    "action": "confirm",
                    "reason": f"Equilibrium 50% midpoint reached at {zone_mid:.5f}",
                    "features": features,
                    "trade_signal": trade_signal,
                }

        # Stagnation / Max Bars in Zone Check
        if bars_in_zone >= cfg.max_bars_in_zone:
            zone["state"] = ZoneState.EXPIRED
            return {
                "state": ZoneState.EXPIRED,
                "action": "expired",
                "reason": f"Stagnation: {bars_in_zone} bars inside zone without confirmation",
                "features": features,
                "trade_signal": None,
            }

        zone["bars_in_zone"] = bars_in_zone + 1
        return {
            "state": ZoneState.IN_ZONE,
            "action": "none",
            "reason": f"Inside zone (bar {bars_in_zone}), awaiting confirmation candle",
            "features": features,
            "trade_signal": None,
        }

    return {"state": current_state, "action": "none", "reason": "unchanged", "features": features, "trade_signal": None}
