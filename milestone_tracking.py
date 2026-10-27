"""
milestone_tracking.py
======================
Stage-1 -> Stage-2 milestone TP tracking ("Method 1"), shared by
backtest_replay.py (historical) and outcome_tracker.py (live).

Stage 1: trade targets the baseline TP1 (whatever your existing dynamic
         TP logic already computed -- this module does NOT touch entry/
         SL/TP1 construction).
Stage 2: the instant TP1 is touched, the touch bar's own momentum is
         evaluated (body ratio, consecutive directional bars, rejection
         wick vs ATR). If it confirms continuation, SL is locked into
         profit and the target is extended to TP2. If not, the trade is
         treated as banked at TP1.

This is a pure post-hoc enrichment: call `evaluate_milestone(...)` at
the exact point your existing code already confirms "TP1 was hit",
passing the touch bar and the bars after it. It never re-derives
entry/SL/TP1 itself and never changes the win/loss classification your
existing SL-priority logic already produced up to that point.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MilestoneConfig:
    min_body_ratio: float = 0.60           # |close-open| / (high-low) on the TP1 touch bar
    min_consecutive_bars: int = 2          # directional 15m bars in a row, including touch bar
    max_rejection_wick_atr: float = 0.30   # opposing wick / ATR on the touch bar
    sl_lock_fraction: float = 0.80         # fraction of (TP1-entry) distance locked in as new SL
    tp2_extension_mult: float = 1.50       # TP2 = entry + (TP1-entry) * this, in trade direction
    max_hold_bars: int = 300               # safety cap (~3 days of 15m bars) for the Stage-2 walk


def _pip_size_for(symbol: str, default_pip: float) -> float:
    s = (symbol or "").upper()
    if "JPY" in s:
        return 0.01
    if "XAU" in s or "XAG" in s:
        return 1.0
    return default_pip


def _body_ratio(bar) -> float:
    rng = bar["high"] - bar["low"]
    return abs(bar["close"] - bar["open"]) / rng if rng > 0 else 0.0


def _is_directional(bar, direction: str) -> bool:
    return (bar["close"] > bar["open"]) if direction == "BUY" else (bar["close"] < bar["open"])


def _rejection_wick(bar, direction: str) -> float:
    if direction == "BUY":
        return bar["high"] - max(bar["close"], bar["open"])
    return min(bar["close"], bar["open"]) - bar["low"]


def _get_atr(bar, atr_col: str, fallback: float) -> float:
    if atr_col in bar and pd.notna(bar[atr_col]):
        return float(bar[atr_col])
    return fallback


def evaluate_milestone(touch_bar, bars_after: pd.DataFrame, direction: str, entry: float,
                        tp1: float, sl: float, symbol: str = "", pip_size: float = 0.0001,
                        atr_col: str = "atr", cfg: Optional[MilestoneConfig] = None,
                        prior_bars: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Called exactly once, at the moment your existing code confirms TP1
    was hit. `touch_bar` is that bar (dict-like/Series with open/high/
    low/close/time[/atr]). `bars_after` is every subsequent 15m bar,
    strictly after touch_bar's time, in chronological order.

    Returns:
      tp1, tp1_achieved (always True -- caller only invokes this after
                          confirming the hit; kept for schema consistency)
      tp1_hit_time
      momentum_checks: dict of the raw metrics + pass/fail on the touch bar
      stage_reached: 1 (banked at TP1) or 2 (extended)
      extension_triggered: bool
      locked_sl, tp2: None unless extension_triggered
      tp2_achieved: bool
      tp2_hit_time
      final_outcome: "banked_tp1" | "extended_tp2_hit" |
                      "extended_then_sl_hit" | "extended_still_open"
      events: list of per-stage decision records, ready to append to a
              JSONL log via log_milestone_event()
    """
    cfg = cfg or MilestoneConfig()
    pip = _pip_size_for(symbol, pip_size)

    tp1_time = touch_bar["time"] if "time" in touch_bar else touch_bar.get("time")
    result: Dict[str, Any] = {
        "tp1": tp1, "tp1_achieved": True, "tp1_hit_time": tp1_time,
        "stage_reached": 1, "extension_triggered": False,
        "locked_sl": None, "tp2": None, "tp2_achieved": False, "tp2_hit_time": None,
        "final_outcome": "banked_tp1", "momentum_checks": None, "events": [],
    }

    atr = _get_atr(touch_bar, atr_col, fallback=pip * 15)
    body_ratio = _body_ratio(touch_bar)
    directional = _is_directional(touch_bar, direction)

    # Consecutive directional bars INTO the touch, counting backward from
    # (and including) the touch bar itself. Without this lookback, the
    # touch bar in isolation can never satisfy min_consecutive_bars >= 2,
    # which would make the momentum check fail almost every time.
    consecutive = 1 if directional else 0
    if directional and prior_bars is not None and len(prior_bars) > 0:
        for i in range(len(prior_bars) - 1, -1, -1):
            b = prior_bars.iloc[i]
            if _is_directional(b, direction):
                consecutive += 1
            else:
                break

    wick = _rejection_wick(touch_bar, direction)
    wick_atr = wick / atr if atr > 0 else 0.0

    checks = {
        "body_ratio": round(float(body_ratio), 3),
        "body_ratio_pass": bool(body_ratio > cfg.min_body_ratio),
        "consecutive_directional_bars": consecutive,
        "consecutive_pass": bool(consecutive >= cfg.min_consecutive_bars),
        "rejection_wick_atr": round(float(wick_atr), 3),
        "wick_pass": bool(wick_atr < cfg.max_rejection_wick_atr),
    }
    checks["confirmed"] = checks["body_ratio_pass"] and checks["consecutive_pass"] and checks["wick_pass"]
    result["momentum_checks"] = checks

    result["events"].append({
        "time": tp1_time, "event": "TP1_HIT", "price": tp1, "checks": checks,
    })

    if not checks["confirmed"]:
        result["events"].append({"time": tp1_time, "event": "HELD_AT_TP1"})
        return result

    # ── Momentum confirmed: lock SL, extend to TP2 ──────────────────────
    tp1_dist = (tp1 - entry) if direction == "BUY" else (entry - tp1)
    locked_sl = (entry + cfg.sl_lock_fraction * tp1_dist if direction == "BUY"
                 else entry - cfg.sl_lock_fraction * tp1_dist)
    tp2 = (entry + tp1_dist * cfg.tp2_extension_mult if direction == "BUY"
           else entry - tp1_dist * cfg.tp2_extension_mult)

    result.update({"stage_reached": 2, "extension_triggered": True,
                    "locked_sl": locked_sl, "tp2": tp2})
    result["events"].append({
        "time": tp1_time, "event": "TP_EXTENDED", "new_sl": locked_sl, "new_tp": tp2,
    })

    # Same-bar edge case: a spike candle could touch TP1 and blow straight
    # through the extended SL or TP2 within the same 15m bar.
    if direction == "BUY":
        sl_hit_same_bar = touch_bar["low"] <= locked_sl
        tp2_hit_same_bar = touch_bar["high"] >= tp2
    else:
        sl_hit_same_bar = touch_bar["high"] >= locked_sl
        tp2_hit_same_bar = touch_bar["low"] <= tp2

    if sl_hit_same_bar:
        result["final_outcome"] = "extended_then_sl_hit"
        result["events"].append({"time": tp1_time, "event": "SL_HIT_POST_EXTENSION", "price": locked_sl})
        return result
    if tp2_hit_same_bar:
        result["tp2_achieved"] = True
        result["tp2_hit_time"] = tp1_time
        result["final_outcome"] = "extended_tp2_hit"
        result["events"].append({"time": tp1_time, "event": "TP2_HIT", "price": tp2})
        return result

    # ── Walk forward bar-by-bar for Stage-2 resolution ──────────────────
    window = bars_after[bars_after["time"] > tp1_time].head(cfg.max_hold_bars) \
        if "time" in bars_after.columns else bars_after.head(cfg.max_hold_bars)

    for _, bar in window.iterrows():
        if direction == "BUY":
            sl_hit = bar["low"] <= locked_sl
            tp_hit = bar["high"] >= tp2
        else:
            sl_hit = bar["high"] >= locked_sl
            tp_hit = bar["low"] <= tp2

        if sl_hit:
            result["final_outcome"] = "extended_then_sl_hit"
            result["events"].append({"time": bar["time"], "event": "SL_HIT_POST_EXTENSION", "price": locked_sl})
            return result
        if tp_hit:
            result["tp2_achieved"] = True
            result["tp2_hit_time"] = bar["time"]
            result["final_outcome"] = "extended_tp2_hit"
            result["events"].append({"time": bar["time"], "event": "TP2_HIT", "price": tp2})
            return result

    result["final_outcome"] = "extended_still_open"
    return result


def log_milestone_event(jsonl_path, record: dict):
    """Append one milestone record as a JSON line. Kept dependency-light
    (no import of backtest_replay) so this module can be used standalone
    from outcome_tracker.py's live path too."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def summarize_milestones(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of evaluate_milestone() results (or signal dicts
    that embed one under 'milestone') into pair-level / overall stats."""
    def _m(r):
        return r.get("milestone", r)

    total = len(records)
    banked = sum(1 for r in records if _m(r).get("final_outcome") == "banked_tp1")
    ext_hit = sum(1 for r in records if _m(r).get("final_outcome") == "extended_tp2_hit")
    ext_sl = sum(1 for r in records if _m(r).get("final_outcome") == "extended_then_sl_hit")
    ext_open = sum(1 for r in records if _m(r).get("final_outcome") == "extended_still_open")
    return {
        "total_tp1_hits": total,
        "banked_at_tp1": banked,
        "extended_and_hit_tp2": ext_hit,
        "extended_then_gave_back_to_sl": ext_sl,
        "extended_still_open": ext_open,
        "extension_rate_pct": round((ext_hit + ext_sl + ext_open) / total * 100, 1) if total else 0.0,
        "tp2_success_rate_pct": round(ext_hit / (ext_hit + ext_sl) * 100, 1) if (ext_hit + ext_sl) else None,
    }


def evaluate_tp_progress(current_bar, direction: str, entry: float, sl: float, tp: float,
                         pip_size: float = 0.0001, symbol: str = "",
                         atr: Optional[float] = None,
                         prior_bars: Optional[pd.DataFrame] = None,
                         hours_near_peak: float = 0.0) -> Dict[str, Any]:
    """
    Evaluates in-flight trade progress toward TP (starting at 30% TP).
    Detects momentum stagnation, opposing rejection wicks, or structural rollover
    to advise taking profit or pulling TP to lock in gains.

    Returns dict with:
      pct_tp: float (% distance to TP reached by this bar's extreme)
      current_r: float (current R-multiple at this bar's close)
      milestone: str ("<30%", "30%-50%", "50%-70%", ">=70%")
      action: "HOLD" | "ADVISE_EXIT" | "ADVISE_TIGHTEN_TP"
      reason: str explaining the decision
      suggested_tp: float | None
    """
    target_dist = abs(tp - entry)
    risk_dist = abs(entry - sl)
    if target_dist <= 0 or risk_dist <= 0:
        return {"action": "HOLD", "pct_tp": 0.0, "current_r": 0.0, "milestone": "<30%", "suggested_tp": None, "reason": "invalid_bracket"}

    pip = _pip_size_for(symbol, pip_size)
    atr = atr if (atr is not None and atr > 0) else pip * 15

    if direction == "BUY":
        fav_extreme = float(current_bar["high"]) - entry
        cur_dist = float(current_bar["close"]) - entry
        opp_wick = float(current_bar["high"]) - max(float(current_bar["open"]), float(current_bar["close"]))
    else:
        fav_extreme = entry - float(current_bar["low"])
        cur_dist = entry - float(current_bar["close"])
        opp_wick = min(float(current_bar["open"]), float(current_bar["close"])) - float(current_bar["low"])

    pct_tp = fav_extreme / target_dist
    cur_r = cur_dist / risk_dist
    planned_rr = target_dist / risk_dist          # NEW: original R:R this trade was built for
    opp_wick_atr = opp_wick / atr if atr > 0 else 0.0
    retreat_atr = (fav_extreme - cur_dist) / atr if atr > 0 else 0.0

    # Milestone classification
    if pct_tp >= 0.70:
        milestone = ">=70%"
    elif pct_tp >= 0.50:
        milestone = "50%-70%"
    elif pct_tp >= 0.30:
        milestone = "30%-50%"
    else:
        milestone = "<30%"

    # Decision logic
    #
    # FIX: the old gate used a flat `cur_r >= 1.0` floor and fired on a SINGLE
    # ordinary wick/retreat as soon as price reached 70% of TP. For trades
    # planned at 1.5-2.5R (the norm for this system's structural TPs), 70% of
    # target is already >=1.0R, so almost any minor pullback there clipped the
    # trade down near +1.0-1.3R -- discarding most of the planned edge.
    # Two changes:
    #   1. Require BOTH stagnation AND rejection (not OR) below 90% of TP --
    #      one normal wick shouldn't end a trade that's not yet near its target.
    #   2. Scale the minimum secured R to the trade's own planned R:R
    #      (min_capture_frac of planned_rr) instead of a flat 1.0R floor, so
    #      higher-RR setups aren't clipped disproportionately harder than
    #      lower-RR ones.
    # Once truly close to the target (>=90%) there's little edge left on the
    # table, so the original single-signal OR logic is kept there.
    min_capture_frac = 0.85  # tune: fraction of planned R:R that must be secured before exiting early
    min_secured_r = min_capture_frac * planned_rr

    if pct_tp >= 0.90:
        is_stagnated = hours_near_peak >= 6.0
        has_rejection = opp_wick_atr >= 0.50 or retreat_atr >= 0.70
        if (is_stagnated or has_rejection) and cur_r >= min_secured_r:
            return {
                "action": "ADVISE_EXIT",
                "pct_tp": round(pct_tp * 100, 1),
                "current_r": round(cur_r, 2),
                "milestone": milestone,
                "suggested_tp": round(float(current_bar["close"]), 5),
                "reason": (f"Momentum exhaustion at {pct_tp*100:.1f}% TP (wick={opp_wick_atr:.2f}x ATR, "
                           f"retreat={retreat_atr:.2f}x ATR, stalled={hours_near_peak:.1f}h). Secured: +{cur_r:.2f}R."),
            }
    elif pct_tp >= 0.70:
        is_stagnated = hours_near_peak >= 6.0
        has_rejection = opp_wick_atr >= 0.65 or retreat_atr >= 0.90
        if is_stagnated and has_rejection and cur_r >= min_secured_r:
            return {
                "action": "ADVISE_EXIT",
                "pct_tp": round(pct_tp * 100, 1),
                "current_r": round(cur_r, 2),
                "milestone": milestone,
                "suggested_tp": round(float(current_bar["close"]), 5),
                "reason": (f"Momentum exhaustion at {pct_tp*100:.1f}% TP (wick={opp_wick_atr:.2f}x ATR, "
                           f"retreat={retreat_atr:.2f}x ATR, stalled={hours_near_peak:.1f}h). Secured: +{cur_r:.2f}R."),
            }

    elif pct_tp >= 0.50:
        if (opp_wick_atr >= 0.70 or retreat_atr >= 1.0 or hours_near_peak >= 12.0) and cur_r >= 0.8:
            return {
                "action": "ADVISE_TIGHTEN_TP",
                "pct_tp": round(pct_tp * 100, 1),
                "current_r": round(cur_r, 2),
                "milestone": milestone,
                "suggested_tp": round(float(current_bar["close"]), 5),
                "reason": (f"Halfway mark ({pct_tp*100:.1f}% TP) encountered opposing pressure (wick={opp_wick_atr:.2f}x ATR). "
                           f"Secured: +{cur_r:.2f}R."),
            }

    return {
        "action": "HOLD",
        "pct_tp": round(pct_tp * 100, 1),
        "current_r": round(cur_r, 2),
        "milestone": milestone,
        "suggested_tp": None,
        "reason": f"Progressing smoothly ({pct_tp*100:.1f}% of TP, +{cur_r:.2f}R).",
    }