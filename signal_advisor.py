"""
signal_advisor.py -- entry advisor (Unified Live Book: Primary OR-71 + Promoted Quality SKIPs)
=============================================================================================
One implementation shared by live (PriceZoneMonitor) and the replay, so the two can never drift.

RULES (Unified Live Book: 96 trades / 66.7% WR / +99.38 R):
  1. PRIMARY SETUP (OR-71, any one qualifies -> TAKE Primary):
       • h4_slope           >= 5.0   4H EMA20 slope, signed with trade, in 15m-ATR units
       • m5_consec_against  >= 4     consecutive 5m bars pushing into zone before touch
       • wick               >= 1.5   rejection wick of confirming candle / 15m ATR

  2. SKIP PROMOTION FILTER (if Primary fails, all three required -> TAKE Promoted):
       • risk_reward                  <= 2.2   planned Risk:Reward ratio (capped target)
       • h4_slope                     >= 1.0   moderate 4H trend alignment
       • consecutive_directional_bars >= 2     2+ completed 15m approach bars into zone

SHADOW / LIVE ADVICE:
  This module computes and labels signals. Live predictors log and alert the exact tier
  ("primary", "promoted", or "skip") and reasons.

Time rules (no look-ahead): a bar is used only if it was COMPLETE at `asof`, i.e. bar_open + interval <= asof.
"""
from __future__ import annotations
import json
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd

PRIMARY_RULES = {
    "h4_slope":          {"op": ">=", "value": 5.0, "stage": "touch"},
    "m5_consec_against": {"op": ">=", "value": 4,   "stage": "touch"},
    "wick":              {"op": ">=", "value": 1.5, "stage": "confirmation"},
}

PROMOTION_RULES = {
    "risk_reward":                  {"op": "<=", "value": 2.2, "stage": "setup"},
    "h4_slope":                     {"op": ">=", "value": 1.0, "stage": "touch"},
    "consecutive_directional_bars": {"op": ">=", "value": 2,   "stage": "touch"},
}

DEFAULT_RULES = PRIMARY_RULES

_OPS = {
    ">=": lambda a, b: a >= b,
    ">":  lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<":  lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def load_rules(path: Optional[str] = None) -> dict:
    if not path:
        return {"primary": PRIMARY_RULES, "promotion": PROMOTION_RULES}
    with open(path) as f:
        cfg = json.load(f)
    return cfg.get("rules", {"primary": PRIMARY_RULES, "promotion": PROMOTION_RULES})


def parse_risk_reward(val: Any) -> Optional[float]:
    """Parse '1:2.0', 2.0, or numeric strings into float."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) if np.isfinite(val) else None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("1:"):
            s = s[2:]
        try:
            f = float(s)
            return f if np.isfinite(f) else None
        except ValueError:
            return None
    return None


def _t(x):
    t = pd.to_datetime(x)
    return t.tz_localize(None) if getattr(t, "tzinfo", None) else t


def _complete(df: pd.DataFrame, asof, minutes: int) -> pd.DataFrame:
    """bars fully closed at `asof`"""
    asof_dt = _t(asof)
    cutoff = np.datetime64(asof_dt - pd.Timedelta(minutes=minutes))
    times = df["time"]
    t_vals = times.values
    if len(t_vals) == 0:
        return df.iloc[:0]
    if not isinstance(t_vals[0], np.datetime64):
        t_vals = pd.to_datetime(times).values
    idx = int(np.searchsorted(t_vals, cutoff, side="right"))
    return df.iloc[:idx]


def _atr14(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def h4_slope(df15: pd.DataFrame, df4h: pd.DataFrame, direction: str, asof) -> Optional[float]:
    """s * (EMA20_4h[now] - EMA20_4h[now-6]) / ATR14_15m ; None if not enough history."""
    asof = _t(asof); s = 1 if direction == "BUY" else -1
    m = _complete(df15, asof, 15); q = _complete(df4h, asof, 240)
    if len(m) < 20 or len(q) < 20:
        return None
    a = float(_atr14(m).iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return None
    e20 = q["close"].ewm(span=20, adjust=False).mean()
    return float(s * (e20.iloc[-1] - e20.iloc[-7]) / a)


def m5_consec_against(df5: pd.DataFrame, direction: str, asof, max_back: int = 30) -> Optional[int]:
    """number of consecutive completed 5m bars (counting back from the last one) closing AGAINST the trade."""
    asof = _t(asof); s = 1 if direction == "BUY" else -1
    b = _complete(df5, asof, 5)
    if len(b) < 5:
        return None
    n = 0
    for c, o in zip(b["close"].values[::-1][:max_back], b["open"].values[::-1][:max_back]):
        if s * (c - o) < 0:
            n += 1
        else:
            break
    return n


def consecutive_directional_bars(df15: pd.DataFrame, direction: str, asof, max_back: int = 6) -> Optional[int]:
    """
    Number of consecutive completed 15m bars closing into the zone before the touch bar.
    BUY (price dropping into demand zone): count bars where close < open.
    SELL (price rising into supply zone): count bars where close > open.
    """
    asof = _t(asof)
    b = _complete(df15, asof, 15)
    if len(b) < 1:
        return None
    consec = 0
    for c, o in zip(b["close"].values[::-1][:max_back], b["open"].values[::-1][:max_back]):
        is_dir = (c < o) if direction == "BUY" else (c > o)
        if is_dir:
            consec += 1
        else:
            break
    return consec


def advise(direction: str, df15: pd.DataFrame, df5: pd.DataFrame, df4h: pd.DataFrame,
           touch_time, wick_atr: Optional[float] = None,
           risk_reward: Optional[Any] = None,
           consecutive_bars: Optional[int] = None,
           rules: Optional[dict] = None) -> Dict[str, Any]:
    """
    Evaluates entry qualification for the Unified Live Book.
    
    Parameters:
      direction        : 'BUY' or 'SELL'
      df15, df5, df4h  : Historical OHLC DataFrames
      touch_time       : when price first entered the zone
      wick_atr         : rejection wick of the confirming candle / 15m ATR
      risk_reward      : planned R:R (e.g. 2.0 or '1:2.0')
      consecutive_bars : 15m approach bars into the zone (computed if None)
      rules            : optional override dict with 'primary' and/or 'promotion' keys
    """
    p_rules = (rules.get("primary") if isinstance(rules, dict) and "primary" in rules else rules) or PRIMARY_RULES
    pr_rules = (rules.get("promotion") if isinstance(rules, dict) and "promotion" in rules else None) or PROMOTION_RULES

    # 1. Feature measurements
    h4 = h4_slope(df15, df4h, direction, touch_time)
    m5 = m5_consec_against(df5, direction, touch_time)
    wick = None if wick_atr is None else float(wick_atr)
    rr = parse_risk_reward(risk_reward)
    if consecutive_bars is not None and not pd.isna(consecutive_bars):
        consec = int(consecutive_bars)
    else:
        consec = consecutive_directional_bars(df15, direction, touch_time)

    vals = {
        "h4_slope": h4,
        "m5_consec_against": m5,
        "wick": wick,
        "risk_reward": rr,
        "consecutive_directional_bars": consec,
    }

    # 2. Evaluate Primary Setup (OR-71: any condition qualifies)
    primary_met, primary_missing = [], []
    for name, r in p_rules.items():
        v = vals.get(name)
        if v is None:
            primary_missing.append(name)
            continue
        if _OPS[r["op"]](v, r["value"]):
            primary_met.append(f"{name} {r['op']} {r['value']} ({v:.2f})")

    if len(primary_met) > 0:
        return {
            "rule_set": "COMBINED LIVE BOOK v1 (Primary OR-71 + Promoted SKIP)",
            "tier": "primary",
            "qualified": True,
            "advice": "TAKE (Primary)",
            "conditions_met": primary_met,
            "primary_conditions_met": primary_met,
            "promotion_conditions_met": [],
            "missing_inputs": primary_missing,
            "values": {k: (None if v is None else round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in vals.items()},
        }

    # 3. Evaluate SKIP Promotion Filter (all conditions required)
    promo_met, promo_missing = [], []
    promo_all_met = True
    for name, r in pr_rules.items():
        v = vals.get(name)
        if v is None:
            promo_missing.append(name)
            promo_all_met = False
            continue
        if _OPS[r["op"]](v, r["value"]):
            if name == "risk_reward":
                promo_met.append(f"risk_reward {r['op']} {r['value']} (1:{v:.1f})")
            elif name == "consecutive_directional_bars":
                promo_met.append(f"consec_bars {r['op']} {r['value']} ({v})")
            else:
                promo_met.append(f"{name} {r['op']} {r['value']} ({v:.2f})")
        else:
            promo_all_met = False

    if promo_all_met and len(promo_met) == len(pr_rules):
        return {
            "rule_set": "COMBINED LIVE BOOK v1 (Primary OR-71 + Promoted SKIP)",
            "tier": "promoted",
            "qualified": True,
            "advice": "TAKE (Promoted)",
            "conditions_met": promo_met,
            "primary_conditions_met": [],
            "promotion_conditions_met": promo_met,
            "missing_inputs": promo_missing,
            "values": {k: (None if v is None else round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in vals.items()},
        }

    # 4. Neither qualified -> SKIP
    return {
        "rule_set": "COMBINED LIVE BOOK v1 (Primary OR-71 + Promoted SKIP)",
        "tier": "skip",
        "qualified": False,
        "advice": "SKIP (shadow only - trade not blocked)",
        "conditions_met": [],
        "primary_conditions_met": [],
        "promotion_conditions_met": [],
        "missing_inputs": list(set(primary_missing + promo_missing)),
        "values": {k: (None if v is None else round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in vals.items()},
    }
