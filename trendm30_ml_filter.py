#!/usr/bin/env python3
"""
TrendM30 Production ML Filter & Live Scanner
Loads trained calibrated model (models/trendm30_ml_classifier.pkl)
Scans live Twelve Data feeds for major and minor FX pairs + Gold.
Strictly evaluates on CLOSED M30 candles to eliminate repainting.
Gates execution using Calibrated ML Win Probability (Threshold >= 70% / 75% Apex).
Dispatches signals to Telegram Bot and Rich HTML Email.
Supports continuous daemon execution synchronized to M30 candle closes (:00 & :30 UTC).
"""

import os
import sys
import time
import json
import argparse
import joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from backtest_trendm30_twelvedata import fetch_twelve_data, calc_atr, calc_rsi, calc_macd, calc_adx, get_env
from email_signal_pusher import send_signal_email
from signal_ledger import load_ledger, save_ledger, update_ledger, ledger_payload, LEDGER_FILE
try:
    from signal_pusher import TelegramSignalPusher
except Exception:
    TelegramSignalPusher = None

MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "trendm30_ml_classifier.pkl"
META_FILE = MODELS_DIR / "trendm30_ml_metadata.json"

DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
EXTENDED_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD",
    "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
    "EUR/JPY", "GBP/JPY"
]

ACTIVE_TRADES_FILE = Path("signals/trendm30_active_trades.json")
TRADE_HISTORY_FILE = Path("signals/trendm30_trade_history.json")
ALERTED_CANDLES_FILE = Path("signals/trendm30_alerted_candles.json")

def load_alerted_candles() -> Dict[str, str]:
    """Load alerted candle timestamps from disk to persist deduplication across restarts."""
    alerted = {}
    if ALERTED_CANDLES_FILE.exists():
        try:
            with open(ALERTED_CANDLES_FILE) as f:
                alerted = json.load(f)
        except Exception:
            alerted = {}
    elif ACTIVE_TRADES_FILE.exists():
        try:
            with open(ACTIVE_TRADES_FILE) as f:
                for t in json.load(f):
                    if "symbol" in t and "candle_time" in t:
                        alerted[t["symbol"]] = t["candle_time"]
        except Exception:
            pass
    return alerted

def save_alerted_candles(alerted: Dict[str, str]):
    """Persist alerted candles map to disk."""
    try:
        ALERTED_CANDLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERTED_CANDLES_FILE, "w") as f:
            json.dump(alerted, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save alerted candles: {e}")

# Global tracking of alerted candle timestamps per symbol (persisted on disk)
_ALERTED_CANDLES: Dict[str, str] = load_alerted_candles()

def track_trade_outcomes(scan_results: List[dict]):
    """Track active trades and resolve them as TP1, TP2, or SL when price hits them"""
    if not ACTIVE_TRADES_FILE.exists():
        return [], []

    try:
        with open(ACTIVE_TRADES_FILE) as f:
            active_trades = json.load(f)
    except Exception:
        active_trades = []

    if not active_trades:
        return [], []

    # Map current prices and swings by symbol from scan_results
    price_map = {}
    for r in scan_results:
        sym = r.get("symbol")
        if sym and "entry" in r:
            price_map[sym] = r

    remaining_trades = []
    history = []
    if TRADE_HISTORY_FILE.exists():
        try:
            with open(TRADE_HISTORY_FILE) as f:
                history = json.load(f)
        except Exception:
            history = []

    for t in active_trades:
        sym = t["symbol"]
        curr = price_map.get(sym)
        if not curr:
            remaining_trades.append(t)
            continue

        direction = t["direction"]
        entry = t["entry"]
        sl = t["sl"]
        tp1 = t["tp1"]
        c = curr.get("close", curr.get("entry")) # current close price
        hi = curr.get("high", c)
        lo = curr.get("low", c)

        resolved = False
        outcome = None
        exit_price = c

        if direction == "BUY":
            if hi >= tp1:
                outcome = "TP1 HIT (+1.5R)"
                resolved = True
                exit_price = tp1
            elif lo <= sl:
                outcome = "SL HIT (-1.0R)"
                resolved = True
                exit_price = sl
        else: # SELL
            if lo <= tp1:
                outcome = "TP1 HIT (+1.5R)"
                resolved = True
                exit_price = tp1
            elif hi >= sl:
                outcome = "SL HIT (-1.0R)"
                resolved = True
                exit_price = sl

        if resolved:
            t["resolved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            t["status"] = outcome
            t["exit_price"] = exit_price
            history.append(t)

            emoji = "🎯" if "TP" in outcome else "🛑"
            msg = (
                f"{emoji} *TRADE RESOLUTION UPDATE* {emoji}\n\n"
                f"Pair: `{sym}` ({direction})\n"
                f"Outcome: *{outcome}*\n"
                f"Entry: `{entry}` | Exit: `{c}`\n"
                f"R:R Target: 1.5R\n"
                f"Resolved: `{t['resolved_at']}`"
            )
            print(f"\n{emoji} [{sym}] Trade Resolved: {outcome} at price {c}")
            if TelegramSignalPusher:
                env = get_env()
                token = env.get("TELEGRAM_TOKEN")
                chat_id = env.get("TELEGRAM_CHAT_ID")
                if token and chat_id:
                    try:
                        TelegramSignalPusher(token, chat_id).send_message(msg, parse_mode="Markdown")
                    except Exception:
                        pass
        else:
            remaining_trades.append(t)

    try:
        with open(ACTIVE_TRADES_FILE, "w") as f:
            json.dump(remaining_trades, f, indent=2)

        with open(TRADE_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not update active trades: {e}")

    return remaining_trades, history

def send_telegram_signal(sig: dict) -> bool:
    """Send formatted signal card to Telegram"""
    if not TelegramSignalPusher:
        return False
    env = get_env()
    token = env.get("TELEGRAM_TOKEN")
    chat_id = env.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    emoji = "🟢" if sig["direction"] == "BUY" else "🔴"
    tier_badge = "🔥 *SNIPER APEX (88%-92% WR)*" if "SNIPER" in sig["tier"] else "⭐ *HIGH CONFIDENCE (78% WR)*"

    text = (
        f"{emoji} *TRENDM30 PRO ML SIGNAL* {emoji}\n\n"
        f"{tier_badge}\n"
        f"Pair: `{sig['symbol']}`\n"
        f"Action: *{sig['direction']}*\n"
        f"ML Win Probability: *{sig['ml_probability']}%*\n"
        f"Candle Closed: `{sig.get('candle_time', sig.get('datetime'))}`\n\n"
        f"📍 *Entry Price:* `{sig['entry']}`\n"
        f"🛑 *Stop Loss:* `{sig['sl']}` ({sig['risk_pips']} pips)\n"
        f"🎯 *Take Profit 1 (1.5R):* `{sig['tp1']}`\n"
        f"🎯 *Take Profit 2 (2.5R Runner):* `{sig['tp2']}`\n\n"
        f"📊 *Confluences:*\n"
        f"• MA60 Distance: `{sig['dist_ma60_atr']} ATR`\n"
        f"• 4H Slope Velocity: `{sig['4h_slope']}`\n"
        f"• ADX Strength: `{sig['adx']}`\n"
        f"• Retest Status: `{sig['status']}`"
    )
    try:
        pusher = TelegramSignalPusher(token, chat_id)
        res = pusher.send_message(text, parse_mode="Markdown")
        return bool(res)
    except Exception as e:
        print(f"⚠️ Telegram dispatch failed: {e}")
        return False


class TrendM30MLFilter:
    def __init__(self, min_prob_threshold: float = 0.70):
        self.threshold = min_prob_threshold
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Run trendm30_ml_trainer.py first.")

        self.model = joblib.load(MODEL_FILE)
        with open(META_FILE) as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["features"]
        print(f"✅ [TrendM30 ML Filter] Loaded Calibrated Model (Features: {len(self.feature_names)} | Threshold: {self.threshold*100:.0f}%)")

    def prepare(self, df30: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators once so any closed bar can be evaluated (used for live scan and backfill)"""
        df = df30.copy().sort_values("datetime").reset_index(drop=True)

        # Moving Averages
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["white"] = df["close"].rolling(60).mean()
        df["yellow"] = df["close"].ewm(span=21, adjust=False).mean()

        df["atr14"] = calc_atr(df, 14)
        df["atr_sma50"] = df["atr14"].rolling(50).mean()
        df["adx"] = calc_adx(df, 14)
        df["rsi"] = calc_rsi(df["close"], 14)
        _, _, df["macd_hist"] = calc_macd(df["close"], 12, 26, 9)

        # 4H EMA50 Slope
        df4h_calc = df4h.copy().sort_values("datetime").reset_index(drop=True)
        df4h_calc["ema50"] = df4h_calc["close"].ewm(span=50, adjust=False).mean()
        df4h_calc["ema50_slope"] = (df4h_calc["ema50"] - df4h_calc["ema50"].shift(3))
        df4h_calc["datetime"] = df4h_calc["datetime"] + pd.Timedelta(hours=4)

        merged = pd.merge_asof(
            df.sort_values("datetime"),
            df4h_calc[["datetime", "close", "ema50", "ema50_slope"]],
            on="datetime",
            suffixes=("", "_htf")
        )
        df["htf_close"] = merged["close_htf"]
        df["htf_ema"] = merged["ema50"]
        df["htf_slope"] = merged["ema50_slope"].fillna(0.0) / (df["atr14"] + 1e-10)

        # Swings
        df["sw10_lo"] = df["low"].rolling(10).min()
        df["sw10_hi"] = df["high"].rolling(10).max()
        return df

    def evaluate_live_signal(self, symbol: str, df30: pd.DataFrame, df4h: pd.DataFrame,
                             bar_idx: Optional[int] = None, prepared: Optional[pd.DataFrame] = None) -> dict:
        """Evaluate a CLOSED M30 bar (default: the newest closed bar) for a high-probability pullback signal"""
        n = len(df30)
        if n < 65:
            return {"symbol": symbol, "status": "ERROR", "message": f"Insufficient 30m bars ({n} < 65)"}

        mintick = 0.01 if ("JPY" in symbol or "XAU" in symbol) else 0.0001
        df = prepared if prepared is not None else self.prepare(df30, df4h)
        n = len(df)


        # Exact Closed Bar Selection:
        # Search backwards to find the newest bar whose 30m period has completely closed (bar_open + 30m <= now_utc)
        now_utc = datetime.now(timezone.utc)
        idx = -1
        if bar_idx is not None:
            idx = bar_idx
        for i in (range(n - 1, -1, -1) if bar_idx is None else []):
            b_dt = df["datetime"].iloc[i]
            if b_dt.tzinfo is None:
                b_dt = b_dt.tz_localize(timezone.utc)
            if now_utc >= (b_dt + timedelta(minutes=30)):
                idx = i
                break

        if idx < 60:
            return {"symbol": symbol, "status": "ERROR", "message": "No fully closed M30 candle found with sufficient history"}

        c = df["close"].iloc[idx]
        y_val = df["yellow"].iloc[idx]
        w_val = df["white"].iloc[idx]
        atr_val = df["atr14"].iloc[idx]

        bull_align = (df["ma5"].iloc[idx] > df["ma10"].iloc[idx] > y_val > w_val)
        bear_align = (df["ma5"].iloc[idx] < df["ma10"].iloc[idx] < y_val < w_val)
        white_up = (w_val > df["white"].iloc[idx-3])
        white_dn = (w_val < df["white"].iloc[idx-3])

        htf_bull = (df["htf_close"].iloc[idx] > df["htf_ema"].iloc[idx])
        htf_bear = (df["htf_close"].iloc[idx] < df["htf_ema"].iloc[idx])

        # Direction detection
        direction = None
        if bull_align and white_up and htf_bull:
            direction = "BUY"
        elif bear_align and white_dn and htf_bear:
            direction = "SELL"
        else:
            candle_str = df["datetime"].iloc[idx].strftime("%Y-%m-%d %H:%M UTC")
            return {
                "symbol": symbol,
                "candle_time": candle_str,
                "status": "NO_SETUP",
                "message": "Moving averages not in alignment or 4H trend counter-aligned."
            }

        # Check pullback proximity to EMA21 (within 0.4 ATR)
        dist_yellow = abs(c - y_val) / atr_val
        is_at_retest = (dist_yellow <= 0.4) or (df["low"].iloc[idx] <= y_val + 0.3*atr_val if direction=="BUY" else df["high"].iloc[idx] >= y_val - 0.3*atr_val)

        # Build ML Feature Row
        dist_ma60 = abs(c - w_val) / atr_val
        atr_ratio = atr_val / (df["atr_sma50"].iloc[idx] + 1e-10)
        dt_val = df["datetime"].iloc[idx]
        candle_str = dt_val.strftime("%Y-%m-%d %H:%M UTC")

        feature_dict = {
            "dist_ma60": dist_ma60,
            "htf_slope": df["htf_slope"].iloc[idx] if direction == "BUY" else -df["htf_slope"].iloc[idx],
            "atr_ratio": atr_ratio,
            "rsi": df["rsi"].iloc[idx],
            "adx": df["adx"].iloc[idx],
            "in_ob_zone": 1 if abs(c - y_val) <= 0.3 * atr_val else 0,
            "yellow_white_near": 1 if dist_ma60 <= 0.5 else 0,
            "is_tue_thu": 1 if dt_val.day_name() in ["Tuesday", "Thursday"] else 0,
            "is_friday": 1 if dt_val.day_name() == "Friday" else 0,
            "hour": dt_val.hour
        }

        X_input = pd.DataFrame([feature_dict])[self.feature_names].values
        prob_win = float(self.model.predict_proba(X_input)[0, 1])

        # Decision & Dynamic R:R based on ML Confidence
        if prob_win >= 0.83:
            rr_tp1 = 2.1
            rr_tp2 = 3.0
            tier = "SNIPER_APEX (Prob >= 83% | 2.1R Target)"
            decision = "TAKE"
        elif prob_win >= 0.75:
            rr_tp1 = 1.7
            rr_tp2 = 2.7
            tier = "SNIPER_APEX (Prob >= 75%)"
            decision = "TAKE"
        elif prob_win >= self.threshold:
            rr_tp1 = 1.5
            rr_tp2 = 2.5
            tier = "HIGH_CONFIDENCE (78% WR)"
            decision = "TAKE"
        elif prob_win >= 0.60:
            rr_tp1 = 1.2
            rr_tp2 = 2.0
            tier = "QUALIFIED (65% WR)"
            decision = "MONITOR"
        else:
            rr_tp1 = 1.0
            rr_tp2 = 1.5
            tier = "LOW_PROBABILITY"
            decision = "SKIP"

        # Trade Levels
        entry_price = round(c, 5)
        sw_low = df["sw10_lo"].iloc[idx]
        sw_high = df["sw10_hi"].iloc[idx]

        if direction == "BUY":
            sl_price = round(min(sw_low - 0.5 * atr_val, entry_price - 5 * mintick), 5)
            risk = abs(entry_price - sl_price)
            tp1_price = round(entry_price + risk * rr_tp1, 5)
            tp2_price = round(entry_price + risk * rr_tp2, 5)
        else:
            sl_price = round(max(sw_high + 0.5 * atr_val, entry_price + 5 * mintick), 5)
            risk = abs(entry_price - sl_price)
            tp1_price = round(entry_price - risk * rr_tp1, 5)
            tp2_price = round(entry_price - risk * rr_tp2, 5)

        pip_size = 0.01 if ("JPY" in symbol or "XAU" in symbol) else 0.0001
        risk_pips = round(risk / pip_size, 1)

        return {
            "symbol": symbol,
            "datetime": dt_val.isoformat(),
            "candle_time": candle_str,
            "direction": direction,
            "status": "QUALIFIED" if is_at_retest else "WAITING_RETEST",
            "decision": decision,
            "ml_probability": round(prob_win * 100, 1),
            "tier": tier,
            "entry": entry_price,
            "close": round(c, 5),
            "high": round(float(df["high"].iloc[idx]), 5),
            "low": round(float(df["low"].iloc[idx]), 5),
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "risk_pips": risk_pips,
            "dist_ma60_atr": round(dist_ma60, 2),
            "4h_slope": round(feature_dict["htf_slope"], 2),
            "adx": round(feature_dict["adx"], 1)
        }


def scan_all_symbols(symbols: List[str] = None, threshold: float = 0.70, force_refresh: bool = True,
                     rebuild_ledger: bool = False) -> List[dict]:
    """Scan all specified symbols on Twelve Data live feeds"""
    if symbols is None:
        symbols = DEFAULT_PAIRS

    print("=" * 85)
    print(f"  TRENDM30 ML LIVE SCANNER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Scanning {len(symbols)} Pairs: {', '.join(symbols)} | Min Threshold: {threshold*100:.0f}%")
    print("=" * 85)

    flt = TrendM30MLFilter(min_prob_threshold=threshold)
    qualified_signals = []
    waiting_retest_signals = []
    all_results = []
    if rebuild_ledger:
        # Start the signal ledger from scratch and replay every closed bar we have history for
        if LEDGER_FILE.exists():
            LEDGER_FILE.replace(LEDGER_FILE.with_suffix(".bak.json"))
            print(f"  🗂️ Old ledger backed up to {LEDGER_FILE.with_suffix('.bak.json')}")
        ledger = {"signals": [], "cursor": {}}
    else:
        ledger = load_ledger()

    for sym in symbols:
        try:
            df30 = fetch_twelve_data(sym, "30min", outputsize=500, force_refresh=force_refresh)
            # Smart 4H caching: 4H bars only close every 4 hours, no need to re-fetch every 30m
            df4h = fetch_twelve_data(sym, "4h", outputsize=60, force_refresh=False)
            res = flt.evaluate_live_signal(sym, df30, df4h)
            all_results.append(res)

            # Record / advance every signal lifecycle (waiting -> triggered -> TP/SL/expired), incl. missed bars
            try:
                for ev in update_ledger(flt, sym, df30, df4h, ledger, lookback=None if rebuild_ledger else 96):
                    print(f"  📒 {ev}")
            except Exception as e:
                print(f"  ⚠️ Ledger update failed for {sym}: {e}")

            c_time = res.get("candle_time", "N/A")

            if res.get("status") == "NO_SETUP":
                print(f"⚪ {sym:<8} [{c_time}]: {res['message']}")
            elif res.get("status") == "ERROR":
                print(f"⚠️ {sym:<8}: {res['message']}")
            else:
                emoji = "🟢" if res["direction"] == "BUY" else "🔴"
                if res["status"] == "QUALIFIED" and res["decision"] == "TAKE":
                    dec_tag = "🚀 [TAKE]"
                    qualified_signals.append(res)
                elif res["status"] == "WAITING_RETEST":
                    dec_tag = "⏳ [WAITING RETEST]"
                    if res["decision"] == "TAKE":
                        waiting_retest_signals.append(res)
                else:
                    dec_tag = f"⚪ [{res['decision']}]"

                print(f"{emoji} {sym:<8} [{c_time}] {res['direction']:<4} | Prob: {res['ml_probability']}% | Tier: {res['tier']} | {dec_tag} | Entry: {res['entry']} | SL: {res['sl']} | TP1: {res['tp1']} ({res['risk_pips']} pips)")
            
            # Short throttle to prevent rate-limit bursts
            time.sleep(1.2)
        except Exception as e:
            print(f"❌ Error scanning {sym}: {e}")

    save_ledger(ledger)

    # Dispatch notifications for qualified signals (with deduplication)
    if qualified_signals:
        dispatched_count = 0
        for sig in qualified_signals:
            sym = sig["symbol"]
            c_time = sig.get("candle_time", sig.get("datetime"))
            last_sent = _ALERTED_CANDLES.get(sym)

            if last_sent == c_time:
                print(f"  ℹ️ [{sym}] Already alerted for closed candle {c_time}. Skipping duplicate alert.")
                continue

            print(f"\n📡 Dispatched high-confidence signal for {sym} via Telegram & Email!")
            # 1. Email Card
            send_signal_email({
                "symbol": sig["symbol"],
                "action": sig["direction"],
                "entry": sig["entry"],
                "sl": sig["sl"],
                "tp": sig["tp1"],
                "tp2": sig["tp2"],
                "rr": "1.5",
                "score": f"{sig['ml_probability']}% ML"
            })
            # 2. Telegram Card
            send_telegram_signal(sig)
            _ALERTED_CANDLES[sym] = c_time
            save_alerted_candles(_ALERTED_CANDLES)
            dispatched_count += 1

        if dispatched_count == 0:
            print("\n✅ All active setups were previously notified (no duplicates sent).")
    else:
        if waiting_retest_signals:
            pairs_str = ", ".join([s['symbol'] for s in waiting_retest_signals])
            print(f"\n⏳ {len(waiting_retest_signals)} pair(s) [{pairs_str}] meet ML quality (>= 70%) but currently waiting for pullback to Yellow EMA21 line before entry.")
        else:
            print("\n💤 No setups currently meeting the >= 70% ML threshold on live bars (strict quality filter active).")

    # 1. Register newly qualified setups into active trades
    if qualified_signals:
        new_trades = []
        if ACTIVE_TRADES_FILE.exists():
            try:
                with open(ACTIVE_TRADES_FILE) as f:
                    new_trades = json.load(f)
            except Exception:
                new_trades = []

        existing_keys = {(t["symbol"], t["entry"]) for t in new_trades}
        for sig in qualified_signals:
            key = (sig["symbol"], sig["entry"])
            if key not in existing_keys:
                new_trades.append({
                    "symbol": sig["symbol"],
                    "direction": sig["direction"],
                    "entry": sig["entry"],
                    "sl": sig["sl"],
                    "tp1": sig["tp1"],
                    "tp2": sig["tp2"],
                    "risk_pips": sig["risk_pips"],
                    "ml_probability": sig["ml_probability"],
                    "tier": sig["tier"],
                    "candle_time": sig["candle_time"],
                    "status": "OPEN",
                    "opened_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                })
        try:
            with open(ACTIVE_TRADES_FILE, "w") as f:
                json.dump(new_trades, f, indent=2)
        except Exception:
            pass

    # 2. Track & Resolve outcomes of active trades
    active_trades, trade_history = track_trade_outcomes(all_results)

    # 3. Persist live scan results & outcomes for Browser Dashboard
    try:
        signals_dir = Path("signals")
        signals_dir.mkdir(parents=True, exist_ok=True)
        dump_payload = {
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "threshold": threshold,
            "count": len(all_results),
            "results": all_results,
            "qualified": qualified_signals,
            "waiting_retest": waiting_retest_signals,
            "active_trades": active_trades,
            "trade_history": trade_history,
            **ledger_payload(ledger)
        }
        with open(signals_dir / "trendm30_signals.json", "w") as f:
            json.dump(dump_payload, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save trendm30_signals.json: {e}")

    return qualified_signals


def seconds_until_next_m30(grace_secs: int = 25) -> Tuple[int, datetime]:
    """Calculate seconds until next 30-minute boundary (:00 or :30) + grace period for Twelve Data settlement"""
    now = datetime.now(timezone.utc)
    if now.minute < 30:
        target_minute = 30
        target = now.replace(minute=target_minute, second=0, microsecond=0)
    else:
        target = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    target_with_grace = target + timedelta(seconds=grace_secs)
    wait_secs = max(5, int((target_with_grace - now).total_seconds()))
    return wait_secs, target_with_grace


def run_continuous_daemon(symbols: List[str], threshold: float = 0.70, grace_secs: int = 25, rebuild_ledger: bool = False):
    """Continuous live runner refreshing automatically after the close of every M30 candle"""
    print("=" * 85)
    print("  🚀 TRENDM30 ML LIVE MONITOR DAEMON — AUTO-REFRESH ON M30 CLOSE")
    print(f"  Pairs Monitored ({len(symbols)}): {', '.join(symbols)}")
    print(f"  Execution Schedule: Synchronized to M30 Closes (:00 & :30 UTC + {grace_secs}s settlement)")
    print(f"  Signal Threshold: >= {threshold*100:.0f}% ML Win Probability")
    print(f"  Alert Channels: Telegram Bot + Rich HTML Email")
    print("=" * 85)

    # 1. Run initial scan on startup
    print("\n🔍 Running initial startup scan on current closed bars...")
    scan_all_symbols(symbols=symbols, threshold=threshold, force_refresh=True, rebuild_ledger=rebuild_ledger)

    # 2. Continuous loop synced to candle closes
    while True:
        wait_secs, next_target = seconds_until_next_m30(grace_secs=grace_secs)
        m, s = divmod(wait_secs, 60)
        print(f"\n⏳ Next M30 close scan scheduled at {next_target.strftime('%H:%M:%S UTC')} (in {m}m {s}s)...")
        time.sleep(wait_secs)

        print("\n" + "=" * 85)
        print(f"  🔔 M30 CANDLE CLOSED — REFRESHING SCANNER ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})")
        print("=" * 85)
        scan_all_symbols(symbols=symbols, threshold=threshold, force_refresh=True)


def parse_args():
    parser = argparse.ArgumentParser(description="TrendM30 ML Live Scanner & Continuous Daemon")
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        help="Run continuously in background, auto-refreshing after the close of every 30-min candle"
    )
    parser.add_argument(
        "--pairs", "-p",
        type=str,
        default="",
        help="Comma-separated symbols to scan (e.g. 'EUR/USD,GBP/USD,USD/JPY,AUD/USD')"
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Scan all 10 major and minor FX pairs + Gold"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.70,
        help="Minimum ML win probability threshold to take trade (default: 0.70)"
    )
    parser.add_argument(
        "--grace", "-g",
        type=int,
        default=25,
        help="Seconds to wait after M30 boundary (:00 / :30) for data provider candle finalization (default: 25)"
    )
    parser.add_argument(
        "--rebuild-ledger",
        action="store_true",
        help="Replay all available closed bars (about 10 days) into a fresh signal ledger; the old ledger is backed up"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine symbols
    if args.pairs:
        symbols = [s.strip() for s in args.pairs.split(",") if s.strip()]
    elif args.all_pairs:
        symbols = EXTENDED_PAIRS
    else:
        symbols = DEFAULT_PAIRS

    if args.loop:
        try:
            run_continuous_daemon(symbols=symbols, threshold=args.threshold, grace_secs=args.grace, rebuild_ledger=args.rebuild_ledger)
        except KeyboardInterrupt:
            print("\n🛑 TrendM30 ML Daemon stopped by user.")
    else:
        scan_all_symbols(symbols=symbols, threshold=args.threshold, force_refresh=True, rebuild_ledger=args.rebuild_ledger)
