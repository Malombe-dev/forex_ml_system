"""
outcome_tracker.py
=====================
Drop-in replacement for SignalOutcomeTracker in production_predictor.py.

KEY FIXES vs the original:
  FIX-1  Entry must be touched FIRST before TP/SL are evaluated.
         A signal is only valid if price actually enters the trade.
  FIX-2  Bar-by-bar simulation (same logic as backtest_engine.py) instead
         of shortcut .any() checks that ignore ordering.
  FIX-3  Three-state lifecycle: pending → triggered → resolved.
  FIX-4  Expired signals use valid_until from the signal record, not raw age.
         EXPIRED    = TTL (valid_until) elapsed, price never reached the zone.
         ENTRY_MISSED = reserved for explicit invalidation (structure break /
                        zone bypass confirmed by bar data). Never auto-assigned
                        on expiry — the two statuses are semantically distinct.
  FIX-5  Resolved timestamp is set to the actual bar time, not utcnow().
  FIX-6  BUY entry trigger tightened: bar_open must be >= entry (not entry - tol).
         Prevents false fills when price is already below entry when signal fires.
  FIX-7  Same-bar SL+TP conflict: SL always takes priority.
         A trade is only a WIN if TP is hit WITHOUT SL being hit first.
         The open-proximity coin-flip logic has been removed entirely.

USAGE — replace in production_predictor.py:
    # Old
    from production_predictor import SignalOutcomeTracker
    tracker = SignalOutcomeTracker()

    # New
    from outcome_tracker import SignalOutcomeTrackerV2
    tracker = SignalOutcomeTrackerV2()
    tracker.check_outcomes(fetch_fn)  # same call signature
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

from milestone_tracking import evaluate_milestone, log_milestone_event, summarize_milestones, evaluate_tp_progress
from reentry_logic import evaluate_reentry, ReentryConfig

logger = logging.getLogger(__name__)


# ── Status constants ───────────────────────────────────────────────────────────
class Status:
    PENDING       = "pending"        # signal fired, entry not yet touched
    TRIGGERED     = "triggered"      # entry touched, trade is live
    HIT_TP        = "hit_tp"         # TP reached after entry, SL never hit → WIN
    HIT_SL        = "hit_sl"         # SL reached after entry → LOSS
    ADVISE_EXIT   = "advise_exit"    # Momentum exhaustion / stagnation at >=70% TP → PROFIT SECURED
    ENTRY_MISSED  = "entry_missed"   # price never reached entry zone → NOT counted
    EXPIRED       = "expired"        # older than TTL without entry → NOT counted
    VIOLATED      = "violated"       # zone broken through on closed bar → NOT counted


# ── Core bar-by-bar simulator ──────────────────────────────────────────────────
def simulate_signal_bar_by_bar(
    bars_after_signal: pd.DataFrame,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    entry_tolerance_pips: float = 0.5,
    pip_size: float = 0.0001,
    closed_cutoff=None,
    symbol: str = "",
    enable_tp_advisory: bool = True,
    already_triggered: bool = False,
) -> Dict:
    """
    Walk bars in chronological order and simulate exactly what would happen
    in a live trade. Mirrors the logic in backtest_engine.py.

    SUCCESS RULE: A trade is only HIT_TP if:
      1. Entry is touched first (price enters the zone)
      2. TP is reached on a subsequent bar
      3. SL has NOT been hit on any bar up to that point

    If SL and TP are hit on the same bar, SL takes priority (conservative/realistic).
    If price reaches >=70% TP and exhibits momentum stagnation or strong opposing rejection,
    ADVISE_EXIT is triggered to secure accrued profit.

    Returns
    -------
    dict with keys:
        status        : one of Status.*
        triggered     : bool
        resolved_bar  : pd.Series | None  (the bar that resolved the trade)
        bars_to_entry : int | None
        bars_to_exit  : int | None
    """
    tol         = entry_tolerance_pips * pip_size
    triggered   = bool(already_triggered)
    entry_bar_i = 0 if already_triggered else None
    peak_fav    = 0.0
    peak_time   = None
    prev_close  = None

    # ── Real rolling ATR for the TP-advisory (previously not passed at all,
    # so evaluate_tp_progress() silently fell back to a flat pip_size*15
    # constant -- badly wrong for symbols like XAU/USD where real volatility
    # varies a lot day to day). Computed once, up front, over the whole
    # window; early bars use whatever history is available (min_periods=1).
    _hi, _lo, _cl = bars_after_signal["high"], bars_after_signal["low"], bars_after_signal["close"]
    _tr = pd.concat([_hi - _lo, (_hi - _cl.shift(1)).abs(), (_lo - _cl.shift(1)).abs()], axis=1).max(axis=1)
    _atr_series = _tr.rolling(14, min_periods=1).mean().reset_index(drop=True)

    for i, (idx, bar) in enumerate(bars_after_signal.iterrows()):
        bar_high = bar["high"]
        bar_low  = bar["low"]
        bar_open = bar.get("open", entry) if hasattr(bar, "get") else getattr(bar, "open", entry)
        _bar_time = bar.get("time") if hasattr(bar, "get") else getattr(bar, "time", None)
        bar_is_closed = (closed_cutoff is None or _bar_time is None
                          or _bar_time <= closed_cutoff)
        if not triggered:
            if direction == "BUY":
                entry_touched = (bar_low <= entry + tol) and (bar_open >= entry)
            else:
                entry_touched = (bar_high >= entry - tol) and (bar_open <= entry)

            if entry_touched:
                triggered   = True
                entry_bar_i = i
                peak_time   = _bar_time
                cur_fav     = max(0.0, (bar_high - entry) if direction == "BUY" else (entry - bar_low))
                peak_fav    = cur_fav

                # On the same bar that triggers entry, also check for
                # immediate SL/TP hit (gap open / spike candle scenarios).
                if direction == "BUY":
                    sl_hit = bar_low  <= sl
                    tp_hit = bar_high >= tp
                else:
                    sl_hit = bar_high >= sl
                    tp_hit = bar_low  <= tp

                # ── FIX-7: SL always wins on same-bar conflict ────────────
                # Removed open-proximity coin-flip. If both SL and TP are
                # hit on the entry bar, we cannot confirm TP was hit first
                # without tick data → treat as SL (conservative, realistic).
                if bar_is_closed:
                    if sl_hit and tp_hit:
                        return {
                            "status":        Status.HIT_SL,
                            "triggered":     True,
                            "resolved_bar":  bar,
                            "bars_to_entry": i,
                            "bars_to_exit":  0,
                        }
                    elif sl_hit:
                        return {
                            "status":        Status.HIT_SL,
                            "triggered":     True,
                            "resolved_bar":  bar,
                            "bars_to_entry": i,
                            "bars_to_exit":  0,
                        }
                    elif tp_hit:
                        return {
                            "status":        Status.HIT_TP,
                            "triggered":     True,
                            "resolved_bar":  bar,
                            "bars_to_entry": i,
                            "bars_to_exit":  0,
                        }
                # Entry touched, neither SL nor TP resolved yet — continue to next bar

        else:
            # ── Trade is live — check SL first, then TP ───────────────────
            # SL is evaluated before TP on every bar.
            # A trade is only a WIN if TP is hit on a bar where SL is NOT hit.
            fav = (bar_high - entry) if direction == "BUY" else (entry - bar_low)
            if fav > peak_fav:
                peak_fav  = fav
                peak_time = _bar_time

            if direction == "BUY":
                sl_hit = bar_low  <= sl
                tp_hit = bar_high >= tp
            else:
                sl_hit = bar_high >= sl
                tp_hit = bar_low  <= tp

            # ── FIX-7: SL always wins on same-bar conflict ────────────────
            if bar_is_closed:
                if sl_hit and tp_hit:
                    return {
                        "status":        Status.HIT_SL,
                        "triggered":     True,
                        "resolved_bar":  bar,
                        "bars_to_entry": entry_bar_i,
                        "bars_to_exit":  i - entry_bar_i,
                    }
                elif sl_hit:
                    return {
                        "status":        Status.HIT_SL,
                        "triggered":     True,
                        "resolved_bar":  bar,
                        "bars_to_entry": entry_bar_i,
                        "bars_to_exit":  i - entry_bar_i,
                    }
                elif tp_hit:
                    return {
                        "status":        Status.HIT_TP,
                        "triggered":     True,
                        "resolved_bar":  bar,
                        "bars_to_entry": entry_bar_i,
                        "bars_to_exit":  i - entry_bar_i,
                    }

                # ── In-Flight Dynamic TP & Momentum Exhaustion Advisory ──────
                if enable_tp_advisory:
                    hours_near_peak = 0.0
                    if _bar_time is not None and peak_time is not None:
                        try:
                            hours_near_peak = max(0.0, (pd.to_datetime(_bar_time) - pd.to_datetime(peak_time)).total_seconds() / 3600.0)
                        except Exception:
                            hours_near_peak = 0.0

                    eval_res = evaluate_tp_progress(
                        bar, direction, entry, sl, tp,
                        pip_size=pip_size, symbol=symbol,
                        hours_near_peak=hours_near_peak,
                        atr=float(_atr_series.iloc[i]) if i < len(_atr_series) else None,
                    )
                    if eval_res.get("action") == "ADVISE_EXIT":
                        return {
                            "status":        Status.ADVISE_EXIT,
                            "triggered":     True,
                            "resolved_bar":  bar,
                            "bars_to_entry": entry_bar_i,
                            "bars_to_exit":  i - entry_bar_i,
                            "exit_price":    float(bar["close"]),
                            "secured_r":     eval_res.get("current_r", 0.0),
                            "advisory":      eval_res,
                        }

        prev_close = bar.get("close", bar_open) if hasattr(bar, "get") else getattr(bar, "close", bar_open)

    # ── Ran out of bars ────────────────────────────────────────────────────────
    if triggered:
        # Entry was hit but trade is still open (not enough bars yet)
        latest_eval = None
        if len(bars_after_signal) > 0:
            latest_bar = bars_after_signal.iloc[-1]
            _lt = latest_bar.get("time") if hasattr(latest_bar, "get") else getattr(latest_bar, "time", None)
            h_near_peak = 0.0
            if _lt is not None and peak_time is not None:
                try:
                    h_near_peak = max(0.0, (pd.to_datetime(_lt) - pd.to_datetime(peak_time)).total_seconds() / 3600.0)
                except Exception:
                    h_near_peak = 0.0
            latest_eval = evaluate_tp_progress(
                latest_bar, direction, entry, sl, tp,
                pip_size=pip_size, symbol=symbol,
                hours_near_peak=h_near_peak,
                atr=float(_atr_series.iloc[-1]) if len(_atr_series) else None,
            )
        return {
            "status":        Status.TRIGGERED,
            "triggered":     True,
            "resolved_bar":  None,
            "bars_to_entry": entry_bar_i,
            "bars_to_exit":  None,
            "tp_progress":   latest_eval,
        }
    else:
        return {
            "status":        Status.PENDING,
            "triggered":     False,
            "resolved_bar":  None,
            "bars_to_entry": None,
            "bars_to_exit":  None,
        }


# ── Drop-in replacement ────────────────────────────────────────────────────────
class SignalOutcomeTrackerV2:
    """
    Replaces SignalOutcomeTracker in production_predictor.py.

    Changes
    -------
    • Uses bar-by-bar simulation (simulate_signal_bar_by_bar) instead of .any()
    • Adds pending / triggered / entry_missed states
    • resolved_at is the actual bar timestamp, not utcnow()
    • Emits structured per-signal result for easy logging / retraining
    • HIT_TP only when: entry hit → TP reached → SL never touched
    """

    def __init__(
        self,
        signals_file: str           = "signals/all_signals.json",
        pip_size: float             = 0.0001,
        entry_tolerance_pips: float = 0.5,
        expire_after_hours: float   = 48.0,
        milestone_log_path: str     = None,
    ):
        self.signals_file          = Path(signals_file)
        self.pip_size              = pip_size
        self.entry_tolerance_pips  = entry_tolerance_pips
        self.expire_after_hours    = expire_after_hours
        # Every TP1 hit's momentum evaluation (banked-at-TP1 or
        # extended-to-TP2, and whether TP2 was actually achieved) is
        # appended here as JSONL, one line per event. Defaults to a
        # sibling file next to signals_file.
        self.milestone_log_path = (
            Path(milestone_log_path) if milestone_log_path
            else self.signals_file.parent / f"{self.signals_file.stem}_milestones.jsonl"
        )

    # ── Public API (same signature as original) ───────────────────────────────
    def check_outcomes(self, fetch_fn: Callable) -> Dict:
        """
        fetch_fn(interval: str, outputsize: int) → pd.DataFrame | None
        Returns summary dict identical to the original tracker.
        """
        if not self.signals_file.exists():
            return {}

        try:
            with open(self.signals_file) as f:
                signals: List[Dict] = json.load(f)
        except Exception as e:
            logger.warning(f"OutcomeTrackerV2: load failed — {e}")
            return {}

        # Cache fetched bar data by timeframe
        df_cache: Dict[str, Optional[pd.DataFrame]] = {}

        counts = {
            "updated": 0, "hit_tp": 0, "hit_sl": 0,
            "expired": 0, "entry_missed": 0, "still_pending": 0,
        }

        for sig in signals:
            current_status = sig.get("status")
            
            # For EXPIRED signals — check if price has returned to the zone.
            # If it has, reactivate to PENDING so the bar simulator can run.
            if current_status == Status.EXPIRED:
                zone_top = float(sig.get("zone_top",  sig.get("entry", 0)))
                zone_bot = float(sig.get("zone_bottom", sig.get("entry", 0)))
                
                # Get latest price from cached bars (or fetch)
                tf = sig.get("timeframe", "15min")
                if tf not in df_cache:
                    df_cache[tf] = fetch_fn(tf, 500)
                df = df_cache.get(tf)
                
                if df is not None and not df.empty:
                    time_col = "datetime" if "datetime" in df.columns else "time"
                    latest_bar = df.iloc[-1]
                    bar_high = float(latest_bar.get("high", 0))
                    bar_low  = float(latest_bar.get("low",  0))
                    
                    # Price is touching the zone again
                    price_in_zone = (bar_low <= zone_top and bar_high >= zone_bot)
                    
                    if price_in_zone:
                        sig["status"]      = Status.PENDING
                        sig["triggered"]   = False
                        sig["bars_to_entry"] = None
                        sig["bars_to_exit"]  = None
                        sig.pop("resolved_at", None)
                        # Reset timestamp to now so TTL restarts
                        sig["timestamp"]   = datetime.utcnow().isoformat()
                        logger.info(
                            f"Reactivated EXPIRED signal: {sig.get('symbol')} "
                            f"{sig.get('direction')} @ {sig.get('entry')} — price returned to zone"
                        )
                        current_status = Status.PENDING  # fall through to normal processing
                    else:
                        continue  # still expired, price hasn't returned
                else:
                    continue

            # Only process signals that are not yet resolved (and exclude zones still in ARMED/IN_ZONE stage)
            if current_status in (Status.HIT_TP, Status.HIT_SL, Status.ADVISE_EXIT, Status.EXPIRED, Status.ENTRY_MISSED, Status.VIOLATED) or sig.get("stage") in ("ARMED", "IN_ZONE", "VIOLATED"):
                continue

            # Always simulate trade outcomes on 15min bars for high resolution, matching replay
            tf = "15min"
            if tf not in df_cache:
                df_cache[tf] = fetch_fn(tf, 500)   # fetch more bars for accuracy

            df = df_cache[tf]
            if df is None or df.empty:
                continue

            try:
                ref_ts = sig.get("confirmed_at") or sig.get("timestamp")
                ts        = pd.to_datetime(ref_ts, utc=False)
                # Strip timezone so comparison with datetime64[ns] (tz-naive) works
                ts        = ts.tz_localize(None) if ts.tzinfo is not None else ts
                entry     = float(sig["entry"])
                sl        = float(sig["sl"])
                tp        = float(sig["tp"])
                direction = sig.get("direction", "BUY")
                age_h     = (datetime.utcnow() - ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600

                # Bars strictly after signal generation time
                # Twelve Data fetch_bars() uses "datetime" column; normalise here
                time_col   = "datetime" if "datetime" in df.columns else "time"
                # Ensure df time column is also tz-naive before comparison
                if df[time_col].dtype == "datetime64[ns, UTC]" or hasattr(df[time_col].dtype, "tz"):
                    df[time_col] = df[time_col].dt.tz_localize(None)
                bars_after = df[df[time_col] > ts].copy()
                if "datetime" in bars_after.columns:
                    bars_after = bars_after.rename(columns={"datetime": "time"})
                if bars_after.empty:
                    continue

                # Use correct pip size per symbol (JPY pairs = 0.01, Gold = 1.0)
                _sym = sig.get("symbol", "").upper()
                _pip = (0.01 if "JPY" in _sym
                        else 1.0 if "XAU" in _sym or "XAG" in _sym
                        else self.pip_size)

                is_two_stage_confirmed = (sig.get("stage") == "CONFIRMED" or sig.get("confirmation_rule") is not None)
                result = simulate_signal_bar_by_bar(
                    bars_after, direction, entry, sl, tp,
                    entry_tolerance_pips=self.entry_tolerance_pips,
                    pip_size=_pip,
                    symbol=sig.get("symbol", ""),
                    enable_tp_advisory=True,
                    already_triggered=is_two_stage_confirmed or sig.get("triggered", False),
                )

                new_status = result["status"]

                # ── Expiry classification ─────────────────────────────────────────
                # Use the signal's own valid_until if present; fall back to raw age.
                # ENTRY_MISSED  = price reached the zone level but never filled
                #                 (gapped past, or the simulator found no touch).
                #                 Only set this when the bar simulator returns PENDING
                #                 AND we can confirm price actually traded through the
                #                 zone — we cannot confirm that here, so this branch
                #                 is reserved for explicit invalidation paths.
                # EXPIRED       = TTL ran out while signal was still PENDING (price
                #                 never came close enough to fill).
                # The two must never be conflated — ENTRY_MISSED inflates "near-miss"
                # stats; EXPIRED is genuinely "zone never tested".
                if new_status == Status.PENDING:
                    valid_until_str = sig.get("valid_until")
                    if valid_until_str:
                        try:
                            vu = datetime.fromisoformat(valid_until_str)
                            timed_out = datetime.utcnow() > vu
                        except ValueError:
                            timed_out = age_h > self.expire_after_hours
                    else:
                        timed_out = age_h > self.expire_after_hours

                    # Distance-based early expiry: if price has moved so far past the
                    # TP level that re-entry is implausible within the remaining TTL,
                    # expire the signal early.
                    #
                    # MULTIPLIER IS TIMEFRAME-AWARE:
                    #   15min / 1h  → 3× risk  (tight, fast-moving zones)
                    #   4h          → 15× risk  (HTF OBs sit far from price legitimately)
                    #   1d / weekly → no distance expiry at all (let TTL decide)
                    #
                    # This prevents 4H OB signals at 1.17 being instantly expired
                    # just because price is currently at 1.15 — those zones are valid
                    # until price either reaches them or the plan window closes.
                    if not timed_out:
                        _tf = sig.get("timeframe", "15min")
                        if _tf in ("1d", "1day", "daily", "1w", "weekly"):
                            _dist_multiplier = None
                        elif _tf in ("4h", "4hour"):
                            _dist_multiplier = 15
                        elif _tf in ("1h", "1hour"):
                            _dist_multiplier = 8
                        else:
                            _dist_multiplier = 3   # 15min default

                        if _dist_multiplier is not None:
                            current_price = float(bars_after.iloc[-1].get("close", entry))
                            risk_pips     = float(sig.get("risk_pips", 0)) or 0
                            if risk_pips > 0:
                                threshold = risk_pips * _dist_multiplier * _pip
                                if direction == "SELL" and current_price < entry - threshold:
                                    timed_out = True   # price blew through TP, zone invalidated
                                elif direction == "BUY" and current_price > entry + threshold:
                                    timed_out = True   # price blew through TP, zone invalidated

                    if timed_out:
                        new_status = Status.EXPIRED   # TTL elapsed, price never reached zone

                # A triggered trade that has been open far too long (data gap or feed
                # outage) — mark expired so it doesn't stay pending forever.
                if new_status == Status.TRIGGERED and age_h > self.expire_after_hours * 2:
                    new_status = Status.EXPIRED

                # Write back only if status changed
                if new_status != sig.get("status"):
                    sig["status"]        = new_status
                    sig["triggered"]     = result["triggered"]
                    sig["bars_to_entry"] = result["bars_to_entry"]
                    sig["bars_to_exit"]  = result["bars_to_exit"]
                    _rb = result["resolved_bar"]
                    _ts = ((_rb.get("time") or _rb.get("datetime")) if _rb is not None else None)
                    sig["resolved_at"] = (
                        _ts.isoformat() if _ts is not None and hasattr(_ts, "isoformat")
                        else datetime.utcnow().isoformat()
                    )

                    # SL overflow: how far past SL the resolving bar's worst-case
                    # wick reached (mirrors backtest_replay.py's approach).
                    if new_status == Status.HIT_SL and _rb is not None:
                        _fill = _rb.get("low") if direction == "BUY" else _rb.get("high")
                        if _fill is not None:
                            sig["sl_overflow_pips"] = round(abs(float(_fill) - sl) / _pip, 2)

                        # ── POST-SL RE-ENTRY (Method 2) ───────────────────────
                        # Uses the full fetched `df` (not just bars_after) so
                        # evaluate_reentry has history before the SL-hit bar
                        # available for its ATR lookback. This is a pure
                        # post-hoc check -- it never touches sig["status"],
                        # which stays HIT_SL either way; the re-entry verdict
                        # is attached separately under sig["reentry"].
                        try:
                            _sl_hit_time = _ts
                            _reentry_bars = (df if time_col == "time"
                                              else df.rename(columns={time_col: "time"}))
                            reentry_result = evaluate_reentry(
                                symbol=sig.get("symbol", ""), direction=direction,
                                orig_entry=entry, orig_sl=sl, orig_tp=tp,
                                sl_hit_time=_sl_hit_time, bars_after_sl=_reentry_bars,
                            )
                            if reentry_result.get("reentry_time") is not None:
                                reentry_result["reentry_time"] = str(reentry_result["reentry_time"])
                            sig["reentry"] = reentry_result
                            if reentry_result.get("reentered"):
                                logger.info(
                                    f"🔁 [RE-ENTRY] {sig.get('symbol')} {direction}: "
                                    f"reclaim @ {reentry_result.get('new_entry')} "
                                    f"(new SL {reentry_result.get('new_sl')}, RR {reentry_result.get('reentry_rr')})"
                                )
                        except Exception as e:
                            logger.warning(f"OutcomeTrackerV2: reentry evaluation failed — {e}")

                    # Entry slippage: ideal entry vs. the entry bar's touch price.
                    _be = result.get("bars_to_entry")
                    if _be is not None and _be < len(bars_after):
                        _entry_bar = bars_after.iloc[_be]
                        _touch = _entry_bar.get("high") if direction == "SELL" else _entry_bar.get("low")
                        if _touch is not None:
                            sig["entry_slippage_pips"] = round(abs(float(_touch) - entry) / _pip, 2)

                    # ── MILESTONE TP1/TP2 TRACKING (Method 1) ─────────────────
                    # tp here is TP1 (the baseline target). Evaluate the touch
                    # bar's momentum to decide whether the trade would have
                    # locked profit and extended to TP2, log the decision, and
                    # persist tp1/tp2 achievement onto the signal record.
                    if new_status == Status.HIT_TP and _rb is not None:
                        try:
                            touch_time = _rb.get("time") or _rb.get("datetime")
                            bars_after_tp1 = bars_after[bars_after["time"] > touch_time] \
                                if "time" in bars_after.columns else bars_after.iloc[0:0]
                            prior_bars_tp1 = df[df[time_col] < touch_time].tail(10)
                            milestone = evaluate_milestone(
                                touch_bar=_rb, bars_after=bars_after_tp1, direction=direction,
                                entry=entry, tp1=tp, sl=sl, symbol=sig.get("symbol", ""), pip_size=_pip,
                                prior_bars=prior_bars_tp1,
                            )
                            sig["milestone"] = milestone
                            log_milestone_event(self.milestone_log_path, {
                                "signal_id": sig.get("id", sig.get("signal_id")),
                                "symbol": sig.get("symbol"), "direction": direction,
                                "timestamp": sig.get("timestamp"),
                                "entry": entry, "sl": sl, "tp1": tp, **milestone,
                            })
                        except Exception as e:
                            logger.warning(f"OutcomeTrackerV2: milestone evaluation failed — {e}")

                    if new_status == Status.ADVISE_EXIT:
                        sig["exit_price"] = result.get("exit_price")
                        sig["secured_r"]  = result.get("secured_r")
                        sig["advisory"]   = result.get("advisory")
                        logger.info(f"🔔 [TP ADVISORY EXIT] {sig.get('symbol')} {direction}: {result.get('advisory', {}).get('reason')}")

                    counts["updated"] += 1

                    if new_status == Status.HIT_TP:
                        counts["hit_tp"] += 1
                    elif new_status == Status.ADVISE_EXIT:
                        counts["advise_exit"] = counts.get("advise_exit", 0) + 1
                    elif new_status == Status.HIT_SL:
                        counts["hit_sl"] += 1
                    elif new_status == Status.ENTRY_MISSED:
                        counts["entry_missed"] += 1
                    elif new_status == Status.EXPIRED:
                        counts["expired"] += 1

                if new_status in (Status.PENDING, Status.TRIGGERED):
                    counts["still_pending"] += 1
                    if new_status == Status.TRIGGERED and result.get("tp_progress"):
                        sig["tp_progress"] = result["tp_progress"]
                        if result["tp_progress"].get("action") in ("ADVISE_EXIT", "ADVISE_TIGHTEN_TP"):
                            logger.info(f"🔔 [TP IN-FLIGHT ALERT] {sig.get('symbol')} {direction}: {result['tp_progress'].get('reason')}")

            except Exception as e:
                logger.warning(f"OutcomeTrackerV2: error on signal — {e}")
                continue

        # Save if anything changed
        if counts["updated"]:
            try:
                with open(self.signals_file, "w") as f:
                    json.dump(signals, f, indent=2)
                logger.info(f"OutcomeTrackerV2: saved {counts['updated']} updates")
            except Exception as e:
                logger.warning(f"OutcomeTrackerV2: save failed — {e}")

        # Win rate computed only over resolved triggered trades (not entry_missed)
        resolved = counts["hit_tp"] + counts["hit_sl"] + counts.get("advise_exit", 0)
        wins = counts["hit_tp"] + counts.get("advise_exit", 0)
        win_rate = round(wins / resolved * 100, 1) if resolved else 0.0
        counts["win_rate_pct"] = win_rate

        logger.info(
            f"OutcomeTrackerV2: TP={counts['hit_tp']} AdviseExit={counts.get('advise_exit', 0)} SL={counts['hit_sl']} "
            f"EntryMissed={counts['entry_missed']} WinRate={win_rate}% "
            f"Pending={counts['still_pending']}"
        )
        return counts


    def check_outcomes_by_symbol(self, sym_predictor: dict) -> Dict:
        """
        Symbol-aware version of check_outcomes.

        BUG-FIX: the original check_outcomes(fetch_fn) accepted a single fetch
        function and used it for every signal regardless of symbol.  This meant
        all GBP/USD, USD/JPY and XAU/USD signals were being simulated against
        EUR/USD bars — the bars for whichever predictor happened to be first in
        the list.  Wrong bars → wrong entry/SL/TP hit detection → outcomes never
        update (or update incorrectly).

        sym_predictor: dict mapping symbol string (e.g. "EUR/USD") to a
                       ProductionForexPredictor instance.  The predictor's
                       fetch_bars(interval, outputsize) method is called to
                       obtain the correct bars for each signal's symbol.
        """
        if not self.signals_file.exists():
            return {}

        try:
            with open(self.signals_file) as f:
                signals: List[Dict] = json.load(f)
        except Exception as e:
            logger.warning(f"OutcomeTrackerV2: load failed — {e}")
            return {}

        # Cache per (symbol, timeframe) so we fetch each pair's bars only once
        df_cache: Dict[str, Optional[pd.DataFrame]] = {}

        counts = {
            "updated": 0, "hit_tp": 0, "hit_sl": 0,
            "expired": 0, "entry_missed": 0, "still_pending": 0,
        }

        for sig in signals:
            # Exclude resolved signals and zones still awaiting two-stage confirmation
            if sig.get("status") in (Status.HIT_TP, Status.HIT_SL, Status.ADVISE_EXIT, Status.EXPIRED, Status.ENTRY_MISSED, Status.VIOLATED) or sig.get("stage") in ("ARMED", "IN_ZONE", "VIOLATED"):
                continue

            sym = sig.get("symbol", "")
            # Always simulate trade outcomes on 15min bars for high resolution, matching replay
            tf  = "15min"
            cache_key = f"{sym}_{tf}"

            if cache_key not in df_cache:
                pred = sym_predictor.get(sym)
                if pred is None:
                    # Unknown symbol — skip rather than use wrong bars
                    logger.warning(f"OutcomeTrackerV2: no predictor for symbol '{sym}', skipping")
                    df_cache[cache_key] = None
                else:
                    df_cache[cache_key] = pred.fetch_bars(tf, 500)

            df = df_cache[cache_key]
            if df is None or df.empty:
                continue

            try:
                # In two-stage execution, track exposure from confirmation time, not original proposal
                ref_ts = sig.get("confirmed_at") or sig.get("timestamp")
                ts        = pd.to_datetime(ref_ts, utc=False)
                ts        = ts.tz_localize(None) if ts.tzinfo is not None else ts
                if sig.get("entry") is None or sig.get("sl") is None or sig.get("tp") is None:
                    continue
                entry     = float(sig["entry"])
                sl        = float(sig["sl"])
                tp        = float(sig["tp"])
                direction = sig.get("direction", "BUY")
                age_h     = (datetime.utcnow() - ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600

                time_col   = "datetime" if "datetime" in df.columns else "time"
                if df[time_col].dtype == "datetime64[ns, UTC]" or hasattr(df[time_col].dtype, "tz"):
                    df[time_col] = df[time_col].dt.tz_localize(None)
                bars_after = df[df[time_col] > ts].copy()
                if "datetime" in bars_after.columns:
                    bars_after = bars_after.rename(columns={"datetime": "time"})

                if bars_after.empty:
                    continue

                _closed_cutoff = datetime.utcnow() - pd.Timedelta(hours=0.25)

                _sym = sym.upper()
                _pip = (0.01 if "JPY" in _sym
                        else 1.0 if "XAU" in _sym or "XAG" in _sym
                        else self.pip_size)

                is_two_stage_confirmed = (sig.get("stage") == "CONFIRMED" or sig.get("confirmation_rule") is not None)
                result = simulate_signal_bar_by_bar(
                    bars_after, direction, entry, sl, tp,
                    entry_tolerance_pips=self.entry_tolerance_pips,
                    pip_size=_pip,
                    closed_cutoff=_closed_cutoff,
                    symbol=sym,
                    enable_tp_advisory=True,
                    already_triggered=is_two_stage_confirmed or sig.get("triggered", False),
                )

                new_status = result["status"]

                # ── Expiry classification ─────────────────────────────────────────
                # Use the signal's own valid_until if present; fall back to raw age.
                # ENTRY_MISSED  = price reached the zone level but never filled
                #                 (gapped past, or the simulator found no touch).
                #                 Only set this when the bar simulator returns PENDING
                #                 AND we can confirm price actually traded through the
                #                 zone — we cannot confirm that here, so this branch
                #                 is reserved for explicit invalidation paths.
                # EXPIRED       = TTL ran out while signal was still PENDING (price
                #                 never came close enough to fill).
                # The two must never be conflated — ENTRY_MISSED inflates "near-miss"
                # stats; EXPIRED is genuinely "zone never tested".
                if new_status == Status.PENDING:
                    valid_until_str = sig.get("valid_until")
                    if valid_until_str:
                        try:
                            vu = datetime.fromisoformat(valid_until_str)
                            timed_out = datetime.utcnow() > vu
                        except ValueError:
                            timed_out = age_h > self.expire_after_hours
                    else:
                        timed_out = age_h > self.expire_after_hours

                    # Distance-based early expiry: if price has moved so far past the
                    # TP level that re-entry is implausible within the remaining TTL,
                    # expire the signal early.
                    #
                    # MULTIPLIER IS TIMEFRAME-AWARE:
                    #   15min / 1h  → 3× risk  (tight, fast-moving zones)
                    #   4h          → 15× risk  (HTF OBs sit far from price legitimately)
                    #   1d / weekly → no distance expiry at all (let TTL decide)
                    #
                    # This prevents 4H OB signals at 1.17 being instantly expired
                    # just because price is currently at 1.15 — those zones are valid
                    # until price either reaches them or the plan window closes.
                    if not timed_out:
                        _tf = sig.get("timeframe", "15min")
                        if _tf in ("1d", "1day", "daily", "1w", "weekly"):
                            # Daily/weekly zones — never distance-expire, let TTL decide
                            _dist_multiplier = None
                        elif _tf in ("4h", "4hour"):
                            _dist_multiplier = 15
                        elif _tf in ("1h", "1hour"):
                            _dist_multiplier = 8
                        else:
                            _dist_multiplier = 3   # 15min default

                        if _dist_multiplier is not None:
                            current_price = float(bars_after.iloc[-1].get("close", entry))
                            risk_pips     = float(sig.get("risk_pips", 0)) or 0
                            if risk_pips > 0:
                                threshold = risk_pips * _dist_multiplier * _pip
                                if direction == "SELL" and current_price < entry - threshold:
                                    timed_out = True   # price blew through TP, zone invalidated
                                elif direction == "BUY" and current_price > entry + threshold:
                                    timed_out = True   # price blew through TP, zone invalidated

                    if timed_out:
                        new_status = Status.EXPIRED   # TTL elapsed, price never reached zone

                

                if new_status != sig.get("status"):
                    sig["status"]        = new_status
                    sig["triggered"]     = result["triggered"]
                    sig["bars_to_entry"] = result["bars_to_entry"]
                    sig["bars_to_exit"]  = result["bars_to_exit"]
                    _rb = result["resolved_bar"]
                    _ts = ((_rb.get("time") or _rb.get("datetime")) if _rb is not None else None)
                    sig["resolved_at"] = (
                        _ts.isoformat() if _ts is not None and hasattr(_ts, "isoformat")
                        else datetime.utcnow().isoformat()
                    )

                    if new_status == Status.HIT_TP and _rb is not None:
                        try:
                            touch_time = _rb.get("time") or _rb.get("datetime")
                            bars_after_tp1 = bars_after[bars_after["time"] > touch_time] \
                                 if "time" in bars_after.columns else bars_after.iloc[0:0]
                            prior_bars_tp1 = df[df[time_col] < touch_time].tail(10)
                            milestone = evaluate_milestone(
                                touch_bar=_rb, bars_after=bars_after_tp1, direction=direction,
                                entry=entry, tp1=tp, sl=sl, symbol=sig.get("symbol", ""), pip_size=_pip,
                                prior_bars=prior_bars_tp1,
                            )
                            sig["milestone"] = milestone
                            log_milestone_event(self.milestone_log_path, {
                                "signal_id": sig.get("id", sig.get("signal_id")),
                                "symbol": sig.get("symbol"), "direction": direction,
                                "timestamp": sig.get("timestamp"),
                                "entry": entry, "sl": sl, "tp1": tp, **milestone,
                            })
                        except Exception as e:
                            logger.warning(f"OutcomeTrackerV2: milestone evaluation failed — {e}")

                    if new_status == Status.ADVISE_EXIT:
                        sig["exit_price"] = result.get("exit_price")
                        sig["secured_r"]  = result.get("secured_r")
                        sig["advisory"]   = result.get("advisory")
                        logger.info(f"🔔 [TP ADVISORY EXIT] {sym} {direction}: {result.get('advisory', {}).get('reason')}")

                    if new_status == Status.HIT_SL and _rb is not None:
                        _fill = _rb.get("low") if direction == "BUY" else _rb.get("high")
                        if _fill is not None:
                            sig["sl_overflow_pips"] = round(abs(float(_fill) - sl) / _pip, 2)

                        # ── POST-SL RE-ENTRY (Method 2) ───────────────────────
                        try:
                            _sl_hit_time = _ts
                            _reentry_bars = (df if time_col == "time"
                                              else df.rename(columns={time_col: "time"}))
                            reentry_result = evaluate_reentry(
                                symbol=sym, direction=direction,
                                orig_entry=entry, orig_sl=sl, orig_tp=tp,
                                sl_hit_time=_sl_hit_time, bars_after_sl=_reentry_bars,
                            )
                            if reentry_result.get("reentry_time") is not None:
                                reentry_result["reentry_time"] = str(reentry_result["reentry_time"])
                            sig["reentry"] = reentry_result
                            if reentry_result.get("reentered"):
                                logger.info(
                                    f"🔁 [RE-ENTRY] {sym} {direction}: "
                                    f"reclaim @ {reentry_result.get('new_entry')} "
                                    f"(new SL {reentry_result.get('new_sl')}, RR {reentry_result.get('reentry_rr')})"
                                )
                        except Exception as e:
                            logger.warning(f"OutcomeTrackerV2: reentry evaluation failed — {e}")

                    counts["updated"] += 1

                    if new_status == Status.HIT_TP:
                        counts["hit_tp"] += 1
                    elif new_status == Status.ADVISE_EXIT:
                        counts["advise_exit"] = counts.get("advise_exit", 0) + 1
                    elif new_status == Status.HIT_SL:
                        counts["hit_sl"] += 1
                    elif new_status == Status.ENTRY_MISSED:
                        counts["entry_missed"] += 1
                    elif new_status == Status.EXPIRED:
                        counts["expired"] += 1

                if new_status in (Status.PENDING, Status.TRIGGERED):
                    counts["still_pending"] += 1
                    if new_status == Status.TRIGGERED and result.get("tp_progress"):
                        sig["tp_progress"] = result["tp_progress"]
                        if result["tp_progress"].get("action") in ("ADVISE_EXIT", "ADVISE_TIGHTEN_TP"):
                            logger.info(f"🔔 [TP IN-FLIGHT ALERT] {sym} {direction}: {result['tp_progress'].get('reason')}")

            except Exception as e:
                logger.warning(f"OutcomeTrackerV2: error on signal — {e}")
                continue

        if counts["updated"]:
            try:
                with open(self.signals_file, "w") as f:
                    json.dump(signals, f, indent=2)
                logger.info(f"OutcomeTrackerV2: saved {counts['updated']} updates")
            except Exception as e:
                logger.warning(f"OutcomeTrackerV2: save failed — {e}")

        resolved = counts["hit_tp"] + counts["hit_sl"] + counts.get("advise_exit", 0)
        wins = counts["hit_tp"] + counts.get("advise_exit", 0)
        win_rate = round(wins / resolved * 100, 1) if resolved else 0.0
        counts["win_rate_pct"] = win_rate

        logger.info(
            f"OutcomeTrackerV2: TP={counts['hit_tp']} AdviseExit={counts.get('advise_exit', 0)} SL={counts['hit_sl']} "
            f"EntryMissed={counts['entry_missed']} WinRate={win_rate}% "
            f"Pending={counts['still_pending']}"
        )
        return counts



# ── Standalone backfill utility ────────────────────────────────────────────────
def backfill_outcomes_from_bars(
    signals_file: str,
    bars_df: pd.DataFrame,
    pip_size: float = 0.0001,
    entry_tolerance_pips: float = 0.5,
) -> pd.DataFrame:
    """
    Offline utility: re-evaluate ALL signals in signals_file against a
    preloaded bar DataFrame (e.g. from your backtest parquet files).
    Returns a DataFrame of results — useful for retraining data quality checks.

    Usage
    -----
        import pandas as pd
        bars = pd.read_parquet("data/bars_15min.parquet").reset_index()
        bars.rename(columns={"index": "time"}, inplace=True)
        results = backfill_outcomes_from_bars("signals/all_signals.json", bars)
        print(results[["timestamp","direction","entry","status","triggered","bars_to_entry","bars_to_exit"]])
    """
    with open(signals_file) as f:
        signals = json.load(f)

    rows = []
    for sig in signals:
        try:
            ts         = pd.to_datetime(sig.get("timestamp"), utc=False)
            ts         = ts.tz_localize(None) if ts.tzinfo is not None else ts
            entry      = float(sig["entry"])
            sl         = float(sig["sl"])
            tp         = float(sig["tp"])
            direction  = sig.get("direction", "BUY")
            _tc        = "datetime" if "datetime" in bars_df.columns else "time"
            if bars_df[_tc].dtype == "datetime64[ns, UTC]" or hasattr(bars_df[_tc].dtype, "tz"):
                bars_df[_tc] = bars_df[_tc].dt.tz_localize(None)
            bars_after = bars_df[bars_df[_tc] > ts].copy()
            if "datetime" in bars_after.columns:
                bars_after = bars_after.rename(columns={"datetime": "time"})
            if bars_after.empty:
                result = {
                    "status": "no_bars", "triggered": False,
                    "bars_to_entry": None, "bars_to_exit": None,
                }
            else:
                result = simulate_signal_bar_by_bar(
                    bars_after, direction, entry, sl, tp,
                    entry_tolerance_pips=entry_tolerance_pips,
                    pip_size=pip_size,
                )
            rows.append({
                **{k: sig.get(k) for k in (
                    "timestamp", "symbol", "timeframe", "strategy", "direction",
                    "entry", "sl", "tp", "rr", "probability", "zone_strength"
                )},
                "sim_status":      result["status"],
                "sim_triggered":   result["triggered"],
                "bars_to_entry":   result["bars_to_entry"],
                "bars_to_exit":    result["bars_to_exit"],
                "original_status": sig.get("status"),
            })
        except Exception as e:
            logger.warning(f"backfill: error — {e}")
            continue

    return pd.DataFrame(rows)


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Test 1: BUY — entry touched, TP hit, SL never hit → HIT_TP ──────────
    bars = pd.DataFrame({
        "time":  pd.date_range("2026-04-24 08:00", periods=10, freq="15min"),
        "open":  [1.1680] * 10,
        "high":  [1.1685, 1.1685, 1.1683, 1.1681, 1.1679, 1.1678, 1.1676, 1.1695, 1.1700, 1.1710],
        "low":   [1.1675, 1.1676, 1.1674, 1.1674, 1.1677, 1.1676, 1.1674, 1.1672, 1.1680, 1.1685],
        "close": [1.1682] * 10,
    })
    result = simulate_signal_bar_by_bar(bars, "BUY", 1.16750, 1.16700, 1.16950)
    print(f"Test 1 BUY HIT_TP:   {result['status']} | triggered={result['triggered']} | "
          f"bars_to_entry={result['bars_to_entry']} | bars_to_exit={result['bars_to_exit']}")
    assert result["status"] == Status.HIT_TP, f"Expected HIT_TP, got {result['status']}"

    # ── Test 2: BUY — entry never touched → PENDING ──────────────────────────
    result2 = simulate_signal_bar_by_bar(bars, "BUY", 1.16600, 1.16550, 1.16950)
    print(f"Test 2 BUY PENDING:  {result2['status']}")
    assert result2["status"] == Status.PENDING

    # ── Test 3: BUY — entry touched, SL hit before TP → HIT_SL ───────────────
    bars_sl = pd.DataFrame({
        "time":  pd.date_range("2026-04-24 08:00", periods=5, freq="15min"),
        "open":  [1.1680, 1.1678, 1.1676, 1.1674, 1.1672],
        "high":  [1.1682, 1.1680, 1.1678, 1.1676, 1.1674],
        "low":   [1.1675, 1.1673, 1.1671, 1.1669, 1.1665],
        "close": [1.1677, 1.1675, 1.1673, 1.1671, 1.1668],
    })
    result3 = simulate_signal_bar_by_bar(bars_sl, "BUY", 1.16750, 1.16700, 1.16950)
    print(f"Test 3 BUY HIT_SL:   {result3['status']} | triggered={result3['triggered']}")
    assert result3["status"] == Status.HIT_SL, f"Expected HIT_SL, got {result3['status']}"

    # ── Test 4: BUY — SL and TP hit on same bar → HIT_SL (SL priority) ───────
    bars_same = pd.DataFrame({
        "time":  pd.date_range("2026-04-24 08:00", periods=3, freq="15min"),
        "open":  [1.1680, 1.1675, 1.1675],
        "high":  [1.1682, 1.1700, 1.1710],   # bar 1 high hits TP
        "low":   [1.1675, 1.1665, 1.1670],   # bar 1 low hits SL
        "close": [1.1678, 1.1690, 1.1695],
    })
    result4 = simulate_signal_bar_by_bar(bars_same, "BUY", 1.16750, 1.16700, 1.16950)
    print(f"Test 4 same-bar SL+TP: {result4['status']} (expect HIT_SL)")
    assert result4["status"] == Status.HIT_SL, f"Expected HIT_SL, got {result4['status']}"

    # ── Test 5: BUY — bar opens BELOW entry → no false trigger ───────────────
    bars_false = pd.DataFrame({
        "time":  pd.date_range("2026-04-24 08:00", periods=3, freq="15min"),
        "open":  [1.1670, 1.1670, 1.1670],   # opens below entry 1.16750
        "high":  [1.1680, 1.1680, 1.1680],
        "low":   [1.1665, 1.1665, 1.1665],
        "close": [1.1675, 1.1675, 1.1675],
    })
    result5 = simulate_signal_bar_by_bar(bars_false, "BUY", 1.16750, 1.16700, 1.16950)
    print(f"Test 5 false trigger:  {result5['status']} (expect PENDING)")
    assert result5["status"] == Status.PENDING, f"Expected PENDING, got {result5['status']}"

    # ── Test 6: BUY — trade reaches >=90% TP and stagnates >6h → ADVISE_EXIT ──
    times6 = pd.date_range("2026-04-24 08:00", periods=35, freq="15min")
    opens6 = [1.1680] + [1.1675] + [1.1697] * 33
    highs6 = [1.1682] + [1.1699] + [1.1699] * 33
    lows6  = [1.1674] + [1.1673] + [1.1696] * 33
    closes6= [1.1676] + [1.1698] + [1.1698] * 33
    bars_adv = pd.DataFrame({"time": times6, "open": opens6, "high": highs6, "low": lows6, "close": closes6})
    result6 = simulate_signal_bar_by_bar(bars_adv, "BUY", 1.16750, 1.16650, 1.17000, pip_size=0.0001, symbol="EUR/USD")
    print(f"Test 6 ADVISE_EXIT:   {result6['status']} | secured_r={result6.get('secured_r')} (expect ADVISE_EXIT)")
    assert result6["status"] == Status.ADVISE_EXIT, f"Expected ADVISE_EXIT, got {result6['status']}"

    print("\n✅ All 6 tests passed")