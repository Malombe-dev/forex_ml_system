#!/usr/bin/env python3
"""
Backtest engine for TrendM30 Pro Pine Script using live Twelve Data feed.
Implements the exact Pine Script v6 indicator logic:
  - Moving Averages: MA5, MA10, MA60, EMA21, EMA3, EMA8
  - Alignment: bullAlign, bearAlign, MA60 slope
  - Confluence Checks (Score out of 7):
      1. HTF Trend (4H EMA50)
      2. ADX > 22
      3. Breakout (20-bar High/Low within 6 bars)
      4. Momentum (MACD Hist & RSI 50)
      5. Volatility (ATR / SMA(ATR, 50) >= 1.0)
      6. Room to EMA21 (< 2.5 * ATR)
      7. SMC (Premium/Discount + Liquidity Sweeps + Supply/Demand)
  - Trade Execution: Swing + ATR Buffer Stop Loss, 2.0 R:R Take Profit
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

# ── Load environment keys ───────────────────────────────────────────────────
def get_env():
    env = {}
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    env[k] = v
    return env

ENV = get_env()
CACHE_DIR = Path("data/twelvedata_live")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Dynamic Key Pool & Rate Limiter ──────────────────────────────────────────
class TwelveDataKeyManager:
    """
    Thread-safe multi-key pool with per-key sliding window rate limiting
    and automatic failover / cooldown management.
    """
    def __init__(self, max_calls_per_min: int = 7, window_secs: float = 62.0):
        self.max_calls = max_calls_per_min
        self.window = window_secs
        self.lock = threading.Lock()
        self.keys: list = []
        self.pair_primary: dict = {}
        self.call_history: dict = {}
        self.blocked_until: dict = {}
        self._reload_keys()

    def _reload_keys(self):
        env = get_env()
        all_keys = []
        for k, v in env.items():
            if k.startswith("TWELVE_KEY_") and v and v.strip():
                val = v.strip()
                if val not in all_keys:
                    all_keys.append(val)

        self.keys = all_keys
        for k in self.keys:
            if k not in self.call_history:
                self.call_history[k] = []
            if k not in self.blocked_until:
                self.blocked_until[k] = 0.0

        self.pair_primary = {
            "EUR/USD": env.get("TWELVE_KEY_EURUSD"),
            "GBP/USD": env.get("TWELVE_KEY_GBPUSD"),
            "USD/JPY": env.get("TWELVE_KEY_USDJPY"),
            "XAU/USD": env.get("TWELVE_KEY_XAUUSD"),
        }

    def _clean_history(self, key: str, now: float) -> list:
        cutoff = now - self.window
        valid = [t for t in self.call_history.get(key, []) if t > cutoff]
        self.call_history[key] = valid
        return valid

    def acquire_key(self, symbol: str) -> str:
        """
        Select an available API key that has capacity under the rate limit.
        If all keys are full, safely sleeps until the earliest key slot frees up.
        """
        while True:
            with self.lock:
                now = time.time()
                self._reload_keys()
                if not self.keys:
                    raise ValueError("No Twelve Data API keys found in .env (TWELVE_KEY_*)")

                # 1. Try preferred key for this symbol first
                preferred = self.pair_primary.get(symbol) or ENV.get(f"TWELVE_KEY_{symbol.replace('/', '')}")
                if preferred and preferred in self.keys and now >= self.blocked_until.get(preferred, 0):
                    calls = self._clean_history(preferred, now)
                    if len(calls) < self.max_calls:
                        self.call_history[preferred].append(now)
                        return preferred

                # 2. Try any available key with capacity
                available_keys = []
                for k in self.keys:
                    if now < self.blocked_until.get(k, 0):
                        continue
                    calls = self._clean_history(k, now)
                    if len(calls) < self.max_calls:
                        available_keys.append((len(calls), k))

                if available_keys:
                    # Pick key with the least recent usage (load balancing)
                    available_keys.sort(key=lambda x: x[0])
                    chosen_key = available_keys[0][1]
                    self.call_history[chosen_key].append(now)
                    return chosen_key

                # 3. All keys are currently busy: find which key frees up first
                wait_times = []
                for k in self.keys:
                    blocked_wait = max(0.0, self.blocked_until.get(k, 0) - now)
                    calls = self._clean_history(k, now)
                    if calls:
                        rate_wait = max(0.0, (calls[0] + self.window) - now)
                    else:
                        rate_wait = 0.0
                    wait_times.append(max(blocked_wait, rate_wait))

                shortest_wait = min(wait_times) if wait_times else 5.0
                sleep_secs = max(0.5, shortest_wait + 0.2)

            print(f"  ⏳ Twelve Data API Pool full across all {len(self.keys)} keys. Throttling for {sleep_secs:.1f}s...")
            time.sleep(sleep_secs)

    def mark_rate_limited(self, key: str, cooldown_secs: float = 65.0):
        """Mark a specific key as temporarily exhausted due to 429"""
        with self.lock:
            now = time.time()
            self.blocked_until[key] = now + cooldown_secs
            print(f"  ⚠️ Key ...{key[-6:]} returned 429. Marked in cooldown for {cooldown_secs:.0f}s. Rotating to next key.")

KEY_MANAGER = TwelveDataKeyManager(max_calls_per_min=7, window_secs=62.0)

# ── Fetch Data from Twelve Data ─────────────────────────────────────────────
def fetch_twelve_data(symbol: str, interval: str, outputsize: int = 5000, force_refresh: bool = False) -> pd.DataFrame:
    safe_sym = symbol.replace("/", "_")
    cache_file = CACHE_DIR / f"{safe_sym}_{interval}_{outputsize}.parquet"
    now = datetime.now()

    # Smart Caching:
    # 4H candles take 4 hours to close! In live scanning, reuse 4H cache if it's less than 75 minutes old.
    cache_ttl = 4500 if interval == "4h" else 60
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = (now - mtime).total_seconds()
        if (not force_refresh and age < 7200) or (interval == "4h" and age < cache_ttl):
            print(f"  📂 Loaded {symbol} {interval} from cache ({cache_file.name}, age: {age/60:.1f}m)")
            return pd.read_parquet(cache_file)

    max_retries = 3
    last_err = None

    for attempt in range(max_retries):
        api_key = KEY_MANAGER.acquire_key(symbol)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": api_key,
            "format": "JSON",
            "timezone": "UTC"
        }

        try:
            print(f"  📡 Fetching {symbol} {interval} ({outputsize} bars) via key ...{api_key[-6:]}...")
            resp = requests.get(url, params=params, timeout=35)

            if resp.status_code == 429:
                KEY_MANAGER.mark_rate_limited(api_key, cooldown_secs=65.0)
                continue

            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "error" or data.get("code") == 429:
                msg = data.get("message", "")
                if "API call rate limit" in msg or data.get("code") == 429:
                    KEY_MANAGER.mark_rate_limited(api_key, cooldown_secs=65.0)
                    continue
                raise RuntimeError(f"Twelve Data error for {symbol} {interval}: {msg}")

            if "values" not in data:
                raise RuntimeError(f"Twelve Data error for {symbol} {interval}: {data.get('message', data)}")

            rows = []
            for candle in data["values"]:
                rows.append({
                    "datetime": pd.to_datetime(candle["datetime"]),
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                })

            df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
            df.to_parquet(cache_file)
            print(f"  ✅ Fetched & cached {len(df)} bars ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")
            return df

        except requests.exceptions.RequestException as e:
            last_err = e
            if "429" in str(e):
                KEY_MANAGER.mark_rate_limited(api_key, cooldown_secs=65.0)
            else:
                time.sleep(1.5)

    raise RuntimeError(f"Failed to fetch {symbol} {interval} after {max_retries} attempts: {last_err}")

# ── Technical Indicators Helpers ─────────────────────────────────────────────
def calc_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA (matches Pine Script ta.rma)."""
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

def calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """True Range smoothed by RMA (matches Pine Script ta.atr)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return calc_rma(tr, length)

def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI (matches Pine Script ta.rsi)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = calc_rma(up, length)
    ma_down = calc_rma(down, length)
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal, and histogram (matches Pine Script ta.macd)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's ADX (matches Pine Script ta.dmi)."""
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - df["close"].shift(1)).abs(),
        (low - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)

    tr_rma = calc_rma(tr, length)
    plus_di = 100 * calc_rma(pd.Series(plus_dm, index=df.index), length) / (tr_rma + 1e-10)
    minus_di = 100 * calc_rma(pd.Series(minus_dm, index=df.index), length) / (tr_rma + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = calc_rma(dx, length)
    return adx

# ── Full Strategy Backtester ────────────────────────────────────────────────
class TrendM30Backtester:
    def __init__(self,
                 min_score: int = 4,
                 rr_target: float = 2.0,
                 sl_mode: str = "swing_atr",  # "swing_atr" or "atr"
                 sl_mult: float = 1.5,
                 swing_len: int = 10,
                 swing_buf: float = 0.25,
                 only_major: bool = True,
                 require_htf: bool = False,
                 entry_mode: str = "breakout"): # "breakout" or "pullback"
        self.min_score = min_score
        self.rr_target = rr_target
        self.sl_mode = sl_mode
        self.sl_mult = sl_mult
        self.swing_len = swing_len
        self.swing_buf = swing_buf
        self.only_major = only_major
        self.require_htf = require_htf
        self.entry_mode = entry_mode

    def run_pair(self, symbol: str, df30: pd.DataFrame, df4h: pd.DataFrame) -> dict:
        """Run backtest on a single currency pair."""
        df = df30.copy()
        n = len(df)
        mintick = 0.01 if "JPY" in symbol or "XAU" in symbol else 0.0001

        # ── 1. Base Moving Averages on M30 ──────────────────────────────────
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["white"] = df["close"].rolling(60).mean()
        df["yellow"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema3"] = df["close"].ewm(span=3, adjust=False).mean()
        df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()

        # MA60 Slope (3 bars)
        df["white_up"] = df["white"] > df["white"].shift(3)
        df["white_down"] = df["white"] < df["white"].shift(3)

        # MA Alignment
        df["bull_align"] = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["yellow"]) & (df["yellow"] > df["white"])
        df["bear_align"] = (df["ma5"] < df["ma10"]) & (df["ma10"] < df["yellow"]) & (df["yellow"] < df["white"])

        # ── 2. HTF (4H EMA50) Alignment ─────────────────────────────────────
        df4h = df4h.copy().sort_values("datetime").reset_index(drop=True)
        df4h["ema50"] = df4h["close"].ewm(span=50, adjust=False).mean()
        
        # Merge HTF completed bars (no lookahead: shift by 4h so only closed candles match)
        df4h_closed = df4h.copy()
        df4h_closed["datetime"] = df4h_closed["datetime"] + pd.Timedelta(hours=4)
        merged = pd.merge_asof(
            df.sort_values("datetime"),
            df4h_closed[["datetime", "close", "ema50"]],
            on="datetime",
            suffixes=("", "_htf")
        )
        df["htf_close"] = merged["close_htf"]
        df["htf_ema"] = merged["ema50"]
        check1_bull = df["htf_close"] > df["htf_ema"]
        check1_bear = df["htf_close"] < df["htf_ema"]

        # ── 3. Check 2: ADX > 22 ─────────────────────────────────────────────
        df["adx"] = calc_adx(df, 14)
        check2 = df["adx"] > 22.0

        # ── 4. Check 3: Breakout of 20-bar High/Low within 6 bars ────────────
        df["hi_brk"] = df["high"].rolling(20).max().shift(1)
        df["lo_brk"] = df["low"].rolling(20).min().shift(1)
        brk_up = df["close"] > df["hi_brk"]
        brk_dn = df["close"] < df["lo_brk"]

        # Barssince breakout <= 6 bars
        check3_bull = brk_up.rolling(7, min_periods=1).max().astype(bool)
        check3_bear = brk_dn.rolling(7, min_periods=1).max().astype(bool)

        # ── 5. Check 4: MACD & RSI ───────────────────────────────────────────
        _, _, df["macd_hist"] = calc_macd(df["close"], 12, 26, 9)
        df["rsi"] = calc_rsi(df["close"], 14)
        check4_bull = (df["macd_hist"] > 0) & (df["rsi"] > 50)
        check4_bear = (df["macd_hist"] < 0) & (df["rsi"] < 50)

        # ── 6. Check 5: Volatility ATR / SMA(ATR, 50) >= 1.0 ─────────────────
        df["atr14"] = calc_atr(df, 14)
        df["atr_sma50"] = df["atr14"].rolling(50).mean()
        check5 = (df["atr14"] / (df["atr_sma50"] + 1e-10)) >= 1.0

        # ── 7. Check 6: Room to EMA21 (< 2.5 * ATR) ──────────────────────────
        check6 = (df["close"] - df["yellow"]).abs() < (df["atr14"] * 2.5)

        # ── 8. Check 7: SMC Confluence (Premium/Discount + Sweeps + S/D) ─────
        # Premium / Discount:
        df["rng_hi"] = df["high"].rolling(100).max()
        df["rng_lo"] = df["low"].rolling(100).min()
        df["eq_p"] = (df["rng_hi"] + df["rng_lo"]) / 2.0
        in_disc = df["close"] < df["eq_p"]
        in_prem = df["close"] > df["eq_p"]

        # Liquidity sweeps (pivots 5, 5)
        # In Pine Script: sweepHigh = high > lastPH and close < lastPH
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atrs = df["atr14"].values
        opens = df["open"].values

        sw_hi_recent = np.zeros(n, dtype=bool)
        sw_lo_recent = np.zeros(n, dtype=bool)
        in_dem = np.zeros(n, dtype=bool)
        in_sup = np.zeros(n, dtype=bool)

        last_ph = np.nan
        last_pl = np.nan
        last_sweep_hi = -9999
        last_sweep_lo = -9999

        # Supply / Demand boxes tracking
        dem_boxes = []  # list of (top, bottom)
        sup_boxes = []

        for i in range(10, n):
            # Check pivot high confirmed at i-5
            piv_idx = i - 5
            if piv_idx >= 5:
                # Is piv_idx highest among [piv_idx-5 .. piv_idx+5]?
                if highs[piv_idx] == np.max(highs[piv_idx-5 : piv_idx+6]):
                    last_ph = highs[piv_idx]
                if lows[piv_idx] == np.min(lows[piv_idx-5 : piv_idx+6]):
                    last_pl = lows[piv_idx]

            # Sweep checks on current bar
            if not np.isnan(last_ph):
                if highs[i] > last_ph and closes[i] < last_ph:
                    last_sweep_hi = i
                if highs[i] > last_ph:
                    last_ph = np.nan

            if not np.isnan(last_pl):
                if lows[i] < last_pl and closes[i] > last_pl:
                    last_sweep_lo = i
                if lows[i] < last_pl:
                    last_pl = np.nan

            if (i - last_sweep_hi) <= 10:
                sw_hi_recent[i] = True
            if (i - last_sweep_lo) <= 10:
                sw_lo_recent[i] = True

            # S/D impulse candle check
            c, o, a = closes[i], opens[i], atrs[i]
            imp_up = (c > o) and ((c - o) > (a * 1.5))
            imp_dn = (c < o) and ((o - c) > (a * 1.5))

            if imp_up:
                base_dn = 1 if closes[i-1] < opens[i-1] else (2 if closes[i-2] < opens[i-2] else 0)
                if base_dn > 0:
                    dem_boxes.append((highs[i - base_dn], lows[i - base_dn]))
                    if len(dem_boxes) > 4:
                        dem_boxes.pop(0)

            if imp_dn:
                base_up = 1 if closes[i-1] > opens[i-1] else (2 if closes[i-2] > opens[i-2] else 0)
                if base_up > 0:
                    sup_boxes.append((highs[i - base_up], lows[i - base_up]))
                    if len(sup_boxes) > 4:
                        sup_boxes.pop(0)

            # Check if active boxes broken or occupied
            # Filter demand boxes
            new_dem = []
            for top, bot in dem_boxes:
                if c < bot:
                    continue  # broken zone deleted
                new_dem.append((top, bot))
                if lows[i] <= top and highs[i] >= bot:
                    in_dem[i] = True
            dem_boxes = new_dem

            new_sup = []
            for top, bot in sup_boxes:
                if c > top:
                    continue  # broken zone deleted
                new_sup.append((top, bot))
                if highs[i] >= bot and lows[i] <= top:
                    in_sup[i] = True
            sup_boxes = new_sup

        smc_bull = in_disc & (sw_lo_recent | in_dem)
        smc_bear = in_prem & (sw_hi_recent | in_sup)

        # ── 9. Total Confluence Score Calculation ───────────────────────────
        df["score_bull"] = (
            check1_bull.astype(int) +
            check2.astype(int) +
            check3_bull.astype(int) +
            check4_bull.astype(int) +
            check5.astype(int) +
            check6.astype(int) +
            smc_bull.astype(int)
        )
        df["score_bear"] = (
            check1_bear.astype(int) +
            check2.astype(int) +
            check3_bear.astype(int) +
            check4_bear.astype(int) +
            check5.astype(int) +
            check6.astype(int) +
            smc_bear.astype(int)
        )

        # Major Swing States
        cond_bull = df["bull_align"] & df["white_up"] & (df["score_bull"] >= self.min_score)
        cond_bear = df["bear_align"] & df["white_down"] & (df["score_bear"] >= self.min_score)

        if self.require_htf:
            cond_bull = cond_bull & check1_bull
            cond_bear = cond_bear & check1_bear

        df["state_maj_bull"] = cond_bull
        df["state_maj_bear"] = cond_bear

        # Trigger on first bar entering state
        df["sig_bull"] = df["state_maj_bull"] & (~df["state_maj_bull"].shift(1).fillna(False))
        df["sig_bear"] = df["state_maj_bear"] & (~df["state_maj_bear"].shift(1).fillna(False))

        # ── 10. Trade Simulation (Forward Walk) ─────────────────────────────
        trades = []
        sw_low_series = df["low"].rolling(self.swing_len).min()
        sw_high_series = df["high"].rolling(self.swing_len).max()

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        atrs = df["atr14"].values
        yellows = df["yellow"].values
        sw_los = sw_low_series.values
        sw_his = sw_high_series.values
        dts = df["datetime"].values
        sig_bull_arr = df["sig_bull"].values
        sig_bear_arr = df["sig_bear"].values
        score_bull_arr = df["score_bull"].values
        score_bear_arr = df["score_bear"].values

        i = 100
        while i < n - 1:
            is_long = sig_bull_arr[i]
            is_short = sig_bear_arr[i]

            if not is_long and not is_short:
                i += 1
                continue

            direction = "LONG" if is_long else "SHORT"
            entry_bar = i
            entry = closes[i]
            atr_cur = atrs[i]

            if self.entry_mode == "pullback":
                # Wait up to 8 bars for a retest to EMA21
                entered = False
                for pb in range(i, min(i + 8, n)):
                    if is_long:
                        if lows[pb] <= yellows[pb] + 0.3 * atrs[pb]:
                            entry_bar = pb
                            entry = yellows[pb]
                            atr_cur = atrs[pb]
                            entered = True
                            break
                    else:
                        if highs[pb] >= yellows[pb] - 0.3 * atrs[pb]:
                            entry_bar = pb
                            entry = yellows[pb]
                            atr_cur = atrs[pb]
                            entered = True
                            break
                if not entered:
                    i += 1
                    continue

            if self.sl_mode == "swing_atr":
                sl_raw = (sw_los[entry_bar] - atr_cur * self.swing_buf) if is_long else (sw_his[entry_bar] + atr_cur * self.swing_buf)
            else:
                sl_raw = (entry - atr_cur * self.sl_mult) if is_long else (entry + atr_cur * self.sl_mult)

            # Safety clamp: minimum 5 ticks risk
            min_risk = 5.0 * mintick
            sl = min(sl_raw, entry - min_risk) if is_long else max(sl_raw, entry + min_risk)
            risk = abs(entry - sl)
            tp = (entry + risk * self.rr_target) if is_long else (entry - risk * self.rr_target)

            # Walk forward from bar entry_bar+1 to track outcome
            outcome = "OPEN"
            exit_bar = entry_bar
            exit_price = entry

            for j in range(entry_bar + 1, min(entry_bar + 300, n)):
                bar_h = highs[j]
                bar_l = lows[j]

                hit_tp = (bar_h >= tp) if is_long else (bar_l <= tp)
                hit_sl = (bar_l <= sl) if is_long else (bar_h >= sl)

                if hit_tp and hit_sl:
                    outcome = "LOSS"
                    exit_bar = j
                    exit_price = sl
                    break
                elif hit_tp:
                    outcome = "WIN"
                    exit_bar = j
                    exit_price = tp
                    break
                elif hit_sl:
                    outcome = "LOSS"
                    exit_bar = j
                    exit_price = sl
                    break

            r_mult = self.rr_target if outcome == "WIN" else (-1.0 if outcome == "LOSS" else 0.0)
            trades.append({
                "symbol": symbol,
                "entry_time": dts[entry_bar],
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "risk": risk,
                "score": score_bull_arr[i] if is_long else score_bear_arr[i],
                "outcome": outcome,
                "exit_time": dts[exit_bar],
                "bars_held": exit_bar - entry_bar,
                "r_mult": r_mult,
            })

            i = exit_bar + 1

        return {
            "symbol": symbol,
            "bars_tested": n,
            "date_start": df["datetime"].iloc[0],
            "date_end": df["datetime"].iloc[-1],
            "trades": trades
        }

# ── Main Run ────────────────────────────────────────────────────────────────
def main():
    print("=" * 85)
    print("  TRENDM30 PRO PINE SCRIPT — LIVE TWELVE DATA COMPARATIVE BACKTEST")
    print("=" * 85)

    symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
    dfs = {}
    for sym in symbols:
        df30 = fetch_twelve_data(sym, "30min", outputsize=5000)
        df4h = fetch_twelve_data(sym, "4h", outputsize=1000)
        dfs[sym] = (df30, df4h)

    # 1. Baseline Run
    print("\n" + "=" * 85)
    print("  RUNNING 1: ORIGINAL PINE SCRIPT BASELINE (Min Score 4, RR 2.0, Breakout Entry)")
    print("=" * 85)
    tester_base = TrendM30Backtester(
        min_score=4,
        rr_target=2.0,
        sl_mode="swing_atr",
        swing_len=10,
        swing_buf=0.25,
        only_major=True,
        require_htf=False,
        entry_mode="breakout"
    )
    base_trades = []
    for sym in symbols:
        res = tester_base.run_pair(sym, dfs[sym][0], dfs[sym][1])
        base_trades.extend(res["trades"])

    # 2. Optimized High Win-Rate Run
    print("\n" + "=" * 85)
    print("  RUNNING 2: OPTIMIZED HIGH WIN-RATE SYSTEM (Score 5, RR 1.0, HTF=True, EMA21 Retest)")
    print("=" * 85)
    tester_opt = TrendM30Backtester(
        min_score=5,
        rr_target=1.0,
        sl_mode="swing_atr",
        swing_len=10,
        swing_buf=0.5,
        only_major=True,
        require_htf=True,
        entry_mode="pullback"
    )
    opt_trades = []
    for sym in symbols:
        res = tester_opt.run_pair(sym, dfs[sym][0], dfs[sym][1])
        opt_trades.extend(res["trades"])

    def calc_metrics(trades):
        resolved = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
        w = sum(1 for t in resolved if t["outcome"] == "WIN")
        l = sum(1 for t in resolved if t["outcome"] == "LOSS")
        wr = (w / len(resolved) * 100) if resolved else 0.0
        tot_r = sum(t["r_mult"] for t in resolved)
        gw = sum(t["r_mult"] for t in resolved if t["r_mult"] > 0)
        gl = abs(sum(t["r_mult"] for t in resolved if t["r_mult"] < 0))
        pf = (gw / gl) if gl > 0 else 0.0
        return len(resolved), w, l, wr, tot_r, pf

    b_tot, b_w, b_l, b_wr, b_r, b_pf = calc_metrics(base_trades)
    o_tot, o_w, o_l, o_wr, o_r, o_pf = calc_metrics(opt_trades)

    print("\n" + "=" * 85)
    print("  SIDE-BY-SIDE PORTFOLIO COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Original Baseline':<25} | {'Optimized High-WR':<25}")
    print("-" * 85)
    print(f"{'Total Trades':<25} | {b_tot:<25} | {o_tot:<25}")
    print(f"{'Wins':<25} | {b_w:<25} | {o_w:<25}")
    print(f"{'Losses':<25} | {b_l:<25} | {o_l:<25}")
    print(f"{'Win Rate':<25} | {f'{b_wr:.2f}%':<25} | {f'{o_wr:.2f}%':<25}  (▲ +{o_wr - b_wr:.1f}%)")
    print(f"{'Net Return':<25} | {f'{b_r:+.2f} R':<25} | {f'{o_r:+.2f} R':<25}  (▲ +{o_r - b_r:.1f} R)")
    print(f"{'Profit Factor':<25} | {f'{b_pf:.2f}':<25} | {f'{o_pf:.2f}':<25}")

    # Breakdown by Symbol
    df_opt = pd.DataFrame([t for t in opt_trades if t["outcome"] in ("WIN", "LOSS")])
    df_base = pd.DataFrame([t for t in base_trades if t["outcome"] in ("WIN", "LOSS")])
    print("\n" + "=" * 85)
    print("  PER-SYMBOL PERFORMANCE COMPARISON")
    print("=" * 85)
    for sym in symbols:
        so = df_base[df_base["symbol"] == sym]
        si = df_opt[df_opt["symbol"] == sym]
        w_b = (so["outcome"] == "WIN").sum()
        l_b = (so["outcome"] == "LOSS").sum()
        wr_b = (w_b / len(so) * 100) if len(so) else 0.0
        r_b = so["r_mult"].sum()

        w_o = (si["outcome"] == "WIN").sum()
        l_o = (si["outcome"] == "LOSS").sum()
        wr_o = (w_o / len(si) * 100) if len(si) else 0.0
        r_o = si["r_mult"].sum()

        print(f"{sym:<8} Baseline: {len(so):<3} trades | WR: {wr_b:>5.1f}% | Net R: {r_b:>+6.1f} R")
        print(f"         Optimized: {len(si):<3} trades | WR: {wr_o:>5.1f}% | Net R: {r_o:>+6.1f} R  (▲ WR +{wr_o - wr_b:.1f}%)")
        print("-" * 85)

    # Save to CSV
    out_csv = Path("backtest_results/trendm30_twelvedata_report.csv")
    df_opt.to_csv(out_csv, index=False)
    print(f"\nSaved detailed optimized trade records to {out_csv}")

if __name__ == "__main__":
    main()
