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

logger = logging.getLogger(__name__)


# ── Status constants ───────────────────────────────────────────────────────────
class Status:
    PENDING       = "pending"        # signal fired, entry not yet touched
    TRIGGERED     = "triggered"      # entry touched, trade is live
    HIT_TP        = "hit_tp"         # TP reached after entry, SL never hit → WIN
    HIT_SL        = "hit_sl"         # SL reached after entry → LOSS
    ENTRY_MISSED  = "entry_missed"   # price never reached entry zone → NOT counted
    EXPIRED       = "expired"        # older than TTL without entry → NOT counted


# ── Core bar-by-bar simulator ──────────────────────────────────────────────────
def simulate_signal_bar_by_bar(
    bars_after_signal: pd.DataFrame,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    entry_tolerance_pips: float = 0.5,
    pip_size: float = 0.0001,
) -> Dict:
    """
    Walk bars in chronological order and simulate exactly what would happen
    in a live trade. Mirrors the logic in backtest_engine.py.

    SUCCESS RULE: A trade is only HIT_TP if:
      1. Entry is touched first (price enters the zone)
      2. TP is reached on a subsequent bar
      3. SL has NOT been hit on any bar up to that point

    If SL and TP are hit on the same bar, SL takes priority (conservative/realistic).

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
    triggered   = False
    entry_bar_i = None

    for i, (idx, bar) in enumerate(bars_after_signal.iterrows()):
        bar_high = bar["high"]
        bar_low  = bar["low"]
        bar_open = bar.get("open", entry) if hasattr(bar, "get") else getattr(bar, "open", entry)

        if not triggered:
            # ── FIX-6: Tightened entry trigger ───────────────────────────
            # BUY:  price must dip INTO the demand zone from ABOVE.
            #       bar_low  <= entry + tol  (low touches entry level)
            #       bar_open >= entry        (was above entry → approaching from above)
            #       Using >= entry (not entry - tol) prevents false fills when
            #       price is already below the entry when the bar opens.
            #
            # SELL: price must rally INTO the supply zone from BELOW.
            #       bar_high >= entry - tol  (high touches entry level)
            #       bar_open <= entry        (was below entry → approaching from below)
            if direction == "BUY":
                entry_touched = (bar_low <= entry + tol) and (bar_open >= entry)
            else:
                entry_touched = (bar_high >= entry - tol) and (bar_open <= entry)

            if entry_touched:
                triggered   = True
                entry_bar_i = i

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
                    # TP hit on entry bar with no SL → valid WIN
                    return {
                        "status":        Status.HIT_TP,
                        "triggered":     True,
                        "resolved_bar":  bar,
                        "bars_to_entry": i,
                        "bars_to_exit":  0,
                    }
                # Entry touched, neither SL nor TP yet — continue to next bar

        else:
            # ── Trade is live — check SL first, then TP ───────────────────
            # SL is evaluated before TP on every bar.
            # A trade is only a WIN if TP is hit on a bar where SL is NOT hit.
            if direction == "BUY":
                sl_hit = bar_low  <= sl
                tp_hit = bar_high >= tp
            else:
                sl_hit = bar_high >= sl
                tp_hit = bar_low  <= tp

            # ── FIX-7: SL always wins on same-bar conflict ────────────────
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
                # TP reached, SL was never hit → genuine WIN
                return {
                    "status":        Status.HIT_TP,
                    "triggered":     True,
                    "resolved_bar":  bar,
                    "bars_to_entry": entry_bar_i,
                    "bars_to_exit":  i - entry_bar_i,
                }

    # ── Ran out of bars ────────────────────────────────────────────────────────
    if triggered:
        # Entry was hit but trade is still open (not enough bars yet)
        return {
            "status":        Status.TRIGGERED,
            "triggered":     True,
            "resolved_bar":  None,
            "bars_to_entry": entry_bar_i,
            "bars_to_exit":  None,
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
    ):
        self.signals_file          = Path(signals_file)
        self.pip_size              = pip_size
        self.entry_tolerance_pips  = entry_tolerance_pips
        self.expire_after_hours    = expire_after_hours

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
            # Only process signals that are not yet resolved
            if sig.get("status") not in (None, Status.PENDING, Status.TRIGGERED, "active"):
                continue

            tf = sig.get("timeframe", "15min")
            if tf not in df_cache:
                df_cache[tf] = fetch_fn(tf, 500)   # fetch more bars for accuracy

            df = df_cache[tf]
            if df is None or df.empty:
                continue

            try:
                ts        = pd.to_datetime(sig.get("timestamp"), utc=False)
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

                result = simulate_signal_bar_by_bar(
                    bars_after, direction, entry, sl, tp,
                    entry_tolerance_pips=self.entry_tolerance_pips,
                    pip_size=_pip,
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
                    counts["updated"] += 1

                    if new_status == Status.HIT_TP:
                        counts["hit_tp"] += 1
                    elif new_status == Status.HIT_SL:
                        counts["hit_sl"] += 1
                    elif new_status == Status.ENTRY_MISSED:
                        counts["entry_missed"] += 1
                    elif new_status == Status.EXPIRED:
                        counts["expired"] += 1

                if new_status in (Status.PENDING, Status.TRIGGERED):
                    counts["still_pending"] += 1

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
        resolved = counts["hit_tp"] + counts["hit_sl"]
        win_rate = round(counts["hit_tp"] / resolved * 100, 1) if resolved else 0.0
        counts["win_rate_pct"] = win_rate

        logger.info(
            f"OutcomeTrackerV2: TP={counts['hit_tp']} SL={counts['hit_sl']} "
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
            if sig.get("status") not in (None, Status.PENDING, Status.TRIGGERED, "active"):
                continue

            sym = sig.get("symbol", "")
            tf  = sig.get("timeframe", "15min")
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
                ts        = pd.to_datetime(sig.get("timestamp"), utc=False)
                ts        = ts.tz_localize(None) if ts.tzinfo is not None else ts
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

                _sym = sym.upper()
                _pip = (0.01 if "JPY" in _sym
                        else 1.0 if "XAU" in _sym or "XAG" in _sym
                        else self.pip_size)

                result = simulate_signal_bar_by_bar(
                    bars_after, direction, entry, sl, tp,
                    entry_tolerance_pips=self.entry_tolerance_pips,
                    pip_size=_pip,
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
                    counts["updated"] += 1

                    if new_status == Status.HIT_TP:
                        counts["hit_tp"] += 1
                    elif new_status == Status.HIT_SL:
                        counts["hit_sl"] += 1
                    elif new_status == Status.ENTRY_MISSED:
                        counts["entry_missed"] += 1
                    elif new_status == Status.EXPIRED:
                        counts["expired"] += 1

                if new_status in (Status.PENDING, Status.TRIGGERED):
                    counts["still_pending"] += 1

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

        resolved = counts["hit_tp"] + counts["hit_sl"]
        win_rate = round(counts["hit_tp"] / resolved * 100, 1) if resolved else 0.0
        counts["win_rate_pct"] = win_rate

        logger.info(
            f"OutcomeTrackerV2: TP={counts['hit_tp']} SL={counts['hit_sl']} "
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

    print("\n✅ All 5 tests passed")