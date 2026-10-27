"""
reentry_logic.py
=================
Post-SL re-entry ("Method 2"), the companion to milestone_tracking.py's
TP-progress tracking. Where evaluate_tp_progress operates BEFORE a stop
is hit (bank profit before a live trade gives it back), evaluate_reentry
operates AFTER a stop is hit (catch a genuine liquidity sweep that
reverses back in the original direction post stop-out).

Backtested unbiased across all 51 hit_sl trades in the 2026-09-01 replay
(not just known sweeps -- the live system can't see that label in
advance): 29/51 triggered a re-entry attempt, 19 won. Total R across the
51 trades went from -51.0 (as originally traded) to -13.01 with this
layered on top -- concentrated almost entirely in XAU/USD, which flipped
from -20.0R to +5.22R. See conversation log 2026-09-10 for the full
per-pair breakdown and caveats (in-sample, GBP/JPY win counts are small
samples).

This is a pure post-hoc enrichment: call `evaluate_reentry(...)` at the
exact point your existing code already confirms "SL was hit". It never
re-derives the original entry/SL/TP and never changes the original
trade's win/loss classification -- it only decides whether a SEPARATE,
new position should be opened afterward.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReentryConfig:
    window_hours: float = 40.0             # how long to watch for a reclaim after SL is hit
    window_hours_overrides: Optional[Dict[str, float]] = None  # e.g. {"XAU/USD": 72.0}
    atr_pad_mult: float = 0.30             # new SL = post-SL swing extreme +/- this * ATR
    atr_lookback: int = 25                 # bars of history for the rolling ATR calc used for the pad
    atr_rolling_window: int = 14

    def window_for(self, symbol: str) -> float:
        overrides = self.window_hours_overrides or {"XAU/USD": 72.0}
        return overrides.get(symbol, self.window_hours)


def _rolling_atr(df15: pd.DataFrame, t_idx: int, cfg: ReentryConfig) -> Optional[float]:
    start = max(0, t_idx - cfg.atr_lookback)
    sub = df15.iloc[start:t_idx + 1]
    if len(sub) < 2:
        return None
    hi, lo, cl = sub["high"], sub["low"], sub["close"]
    tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(cfg.atr_rolling_window, min_periods=1).mean().iloc[-1])
    return atr if np.isfinite(atr) and atr > 0 else None


def evaluate_reentry(symbol: str, direction: str, orig_entry: float, orig_sl: float, orig_tp: float,
                      sl_hit_time, bars_after_sl: pd.DataFrame, cfg: Optional[ReentryConfig] = None
                      ) -> Dict[str, Any]:
    """
    Called exactly once, at the moment your existing code confirms the
    original SL was hit. `bars_after_sl` is every subsequent 15m bar for
    this symbol, in chronological order, WITH enough history before
    sl_hit_time included for the ATR lookback to work (i.e. don't slice
    starting exactly at sl_hit_time -- include cfg.atr_lookback bars
    before it too).

    Trigger condition: price CLOSES back beyond the original entry level
    (a reclaim) on a bar that is itself directional in the trade's favor
    (confirms momentum, not just a wick poking back across the level).

    Returns:
      reentered: bool
      reentry_time, new_entry, new_sl, tp: set only if reentered
      reason: str, set when reentered is False
    """
    cfg = cfg or ReentryConfig()
    window_h = cfg.window_for(symbol)
    window_end = pd.Timestamp(sl_hit_time) + pd.Timedelta(hours=window_h)

    df = bars_after_sl.reset_index(drop=True)
    after_sl_idx = df.index[df["time"] > sl_hit_time]
    if len(after_sl_idx) == 0:
        return {"reentered": False, "reason": "no_data_after_sl"}

    swing_extreme = orig_sl
    for idx in after_sl_idx:
        bar = df.iloc[idx]
        if bar["time"] > window_end:
            break

        swing_extreme = (min(swing_extreme, bar["low"]) if direction == "BUY"
                          else max(swing_extreme, bar["high"]))

        reclaimed = (bar["close"] > orig_entry) if direction == "BUY" else (bar["close"] < orig_entry)
        directional = (bar["close"] > bar["open"]) if direction == "BUY" else (bar["close"] < bar["open"])

        if reclaimed and directional:
            atr = _rolling_atr(df, idx, cfg) or (orig_entry * 0.001)
            pad = cfg.atr_pad_mult * atr
            new_sl = swing_extreme - pad if direction == "BUY" else swing_extreme + pad
            new_entry = float(bar["close"])
            new_risk = abs(new_entry - new_sl)
            if new_risk <= 0:
                continue
            return {
                "reentered": True,
                "reentry_time": bar["time"],
                "new_entry": round(new_entry, 5),
                "new_sl": round(new_sl, 5),
                "tp": round(orig_tp, 5),
                "reentry_rr": round(abs(orig_tp - new_entry) / new_risk, 2),
            }

    return {"reentered": False, "reason": "no_reclaim_in_window"}