"""
backtest_replay.py — Walk-forward replay of the live Reversion predictor
over historical data, so you get statistically useful feedback in one run
instead of waiting days for live signals to accumulate.

═══════════════════════════════════════════════════════════════════════════
WHAT THIS DOES
═══════════════════════════════════════════════════════════════════════════
For each pair, it re-runs your EXACT live zone-detection + ensemble scoring
pipeline (get_htf_context / htf_trade_detect from production_predictor.py)
at every historical 4H candle close over the replay window, using only data
that would have existed at that point in time (no lookahead). Any signal it
generates is then walked forward through subsequent historical price data
to see whether it would have hit TP or SL first.

This is NOT a separate/simplified model — it calls the same methods your
live system calls, so what fires here is what would have fired live.

═══════════════════════════════════════════════════════════════════════════
HONEST LIMITATIONS — read before trusting the output
═══════════════════════════════════════════════════════════════════════════
1. TRIGGER LOGIC IS A SIMPLIFIED REPLICA, not a byte-for-byte clone of
   PriceZoneMonitor.check(). It uses the same zone_top/zone_bottom touch
   criteria, but doesn't replicate the full live JSON state machine
   (save_signals_to_single_file dedup, check_signal_outcomes, etc.) since
   that machinery exists for incremental cron-cycle updates, not a
   single in-memory replay pass.

2. INTRABAR TIE-BREAK: if a single 15-min bar's range touches both SL and
   TP, this assumes SL was hit first (conservative/pessimistic). Real
   intrabar order is unknown without tick data — so reported win rate is
   a slightly pessimistic estimate, not exact.

3. NEWS FILTERING (historical) is best-effort via Finnhub's economic
   calendar endpoint. It only implements the PRE-GENERATION gate (skip
   proposing a new signal within 30 min of high-impact news) — it does
   NOT replicate the live post-release directional close-flag logic,
   since that requires tracking an actively open position, which adds
   real complexity for a first pass. If your Finnhub plan doesn't expose
   historical economic calendar data, this step is skipped with a clear
   warning and the run continues without news filtering.

4. Only ONE open trade per symbol at a time is modeled (mirrors the live
   OPEN-TRADE GUARD) -- but "open" now means an actually-triggered,
   capital-deployed trade (hit_tp/hit_sl/timeout), not merely a proposed
   zone that's still being watched. A zone that expires untriggered or
   gets adjusted-but-never-filled uses no capital, so it no longer blocks
   the next 4H bar from proposing a fresh signal on the same pair --
   including another look at the same still-live structural zone (the
   exact-match zone dedup a few sections down is what prevents that
   specific case from double-counting as two signals). If a proposal's
   eventual trigger lands inside a window another already-triggered trade
   on that pair is still open, it's kept in the results but flagged
   `would_be_blocked_by_open_trade=True` rather than dropped, so you can
   filter to "what live would actually have taken" after the fact without
   losing signal volume for stats.

5. This only replays the REVERSION strategy (per your request) — the
   Continuation model and the 15-min signal pipeline are not exercised.

Run this, then sanity-check a handful of individual trades against the
chart by hand before trusting the aggregate win rate.

═══════════════════════════════════════════════════════════════════════════
SETUP
═══════════════════════════════════════════════════════════════════════════
Run from your forex_ml_system/ directory (same place production_predictor.py
lives), with the same venv active:

    python3 backtest_replay.py --days 60

Optional:
    --pairs EUR/USD,XAU/USD     (default: all 4)
    --days 60                   (how far back to replay)
    --finnhub-key YOUR_KEY      (or set FINNHUB_API_KEY env var)
    --no-news                   (skip news filtering entirely)
    --out-dir backtest_results  (where to save the report)
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from milestone_tracking import evaluate_milestone, log_milestone_event, summarize_milestones


# ═══════════════════════════════════════════════════════════════════════
# 0. DETAILED LOGGING — every print() below is duplicated into a
#    timestamped log file, so a run's full output survives after your
#    terminal scrollback is gone. Also writes a machine-readable
#    per-signal JSONL log (one line per signal, full field dump) that's
#    easier to grep/pandas-load than the human-readable console log.
# ═══════════════════════════════════════════════════════════════════════

class _Tee:
    """Duplicates writes to the real stream and a log file simultaneously."""
    def __init__(self, real_stream, log_file):
        self._real = real_stream
        self._log  = log_file

    def write(self, data):
        self._real.write(data)
        self._log.write(data)
        self._log.flush()  # flush immediately so the log is useful if the run is killed/crashes

    def flush(self):
        self._real.flush()
        self._log.flush()


def setup_detailed_logging(out_dir: Path, run_tag: str):
    """
    Redirects stdout/stderr through a Tee so the full console output
    (all the ▶/📌/✅/❌ progress lines) is also written verbatim to
    out_dir/logs/replay_<run_tag>.log. Returns (log_path, jsonl_path,
    close_fn) — call close_fn() when the run is finished.
    """
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path   = log_dir / f"replay_{run_tag}.log"
    jsonl_path = log_dir / f"replay_{run_tag}_signals.jsonl"

    log_fh = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)

    def _close():
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not log_fh.closed:
            log_fh.close()

    import atexit
    atexit.register(_close)  # safety net if the run crashes/is Ctrl-C'd before the normal close

    return log_path, jsonl_path, _close


def log_signal_jsonl(jsonl_path: Path, record: dict):
    """Append one full signal/outcome record as a JSON line — every field
    that produced the trade (zone, HTF context, confluence, probability,
    outcome, timings) so you can later slice reliability by any dimension
    without re-parsing the human-readable log."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

# ── Load .env if present (plain parser, no extra dependency required) ──
def _load_dotenv(path: str = ".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

_load_dotenv()

# Import the live predictor code directly so we reuse its exact logic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from production_predictor import ProductionForexPredictor  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 1. HISTORICAL BAR FETCHING (Twelve Data, chunked for intraday ranges)
# ═══════════════════════════════════════════════════════════════════════

# Max days per request per interval, kept comfortably under Twelve Data's
# 5000-point-per-request cap.
_CHUNK_DAYS = {
    "5min":  15,
    "15min": 45,
    "1h":    200,
    "4h":    800,
    "1day":  2000,
}

PAIR_API_KEYS = {
    "EUR/USD": os.getenv("TWELVE_KEY_EURUSD", "ea3e4fce074e403f8cd4c83dd0035643"),
    "GBP/USD": os.getenv("TWELVE_KEY_GBPUSD", "2522b470eba249f88d2e29575035914e"),
    "USD/JPY": os.getenv("TWELVE_KEY_USDJPY", "5bb48b4211144b92a4487ce7c63d1b7a"),
    "XAU/USD": os.getenv("TWELVE_KEY_XAUUSD", "63a513bb7eb848748d28d3deb64c1b59"),
}

# ── Twelve Data free-tier pacing (2026-08-18 fix) ───────────────────────
# Free tier: 8 requests/minute, 800/day, 5000 data points/request. The old
# code did a flat time.sleep(1) between chunks — fine for a 60-day replay
# (few chunks needed) but at ~1 req/sec that's ~60 req/min, well over the
# 8/min cap. A 4-month replay needs enough extra chunks that this WILL hit
# the limit. Each symbol uses its own API key (see PAIR_API_KEYS above),
# so the 8/min budget is tracked separately per key — throttling is scoped
# to whichever key is making the request, not global across all 4 pairs.
MIN_SECONDS_BETWEEN_CALLS = 8.0   # 60s / 8 req = 7.5s min; 8.0s for headroom
_last_call_time: dict = {}        # api_key -> last request timestamp

def _throttle(api_key: str):
    """Block just long enough to stay under the free-tier per-minute cap
    for this specific key. Cheap no-op once enough time has passed."""
    last = _last_call_time.get(api_key)
    if last is not None:
        elapsed = time.time() - last
        wait = MIN_SECONDS_BETWEEN_CALLS - elapsed
        if wait > 0:
            time.sleep(wait)
    _last_call_time[api_key] = time.time()


def _is_rate_limit_response(data: dict) -> bool:
    """Twelve Data returns HTTP 200 with an error CODE in the JSON body on
    rate-limit/credit exhaustion, not an HTTP 429 — so resp.raise_for_status()
    never catches it. Previously this fell straight into the generic
    '"values" not in data -> treat as no data for this period' branch and
    silently skipped the chunk, leaving quiet gaps in longer replays. This
    checks for that specific case so it can be retried instead of dropped.
    """
    if not isinstance(data, dict):
        return False
    code = data.get("code")
    msg  = str(data.get("message", "")).lower()
    return code in (429, 414) or "run out of api credits" in msg or "limit" in msg and "credit" in msg


def _find_reusable_cache(symbol: str, interval: str, start: datetime, end: datetime,
                          cache_dir: Path) -> Optional[pd.DataFrame]:
    """Look for any existing cache file for this (symbol, interval) whose
    date range fully covers [start, end], and slice it instead of refetching.
    Cache files are named {symbol}_{interval}_{start}_{end}.csv, so this
    parses filenames rather than requiring an exact-range match — lets a
    4-month run reuse data already fetched by an earlier 60-day run over
    an overlapping window instead of re-downloading it from scratch."""
    safe_symbol = symbol.replace("/", "")
    if not cache_dir.exists():
        return None
    for f in cache_dir.glob(f"{safe_symbol}_{interval}_*.csv"):
        parts = f.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            cached_start = datetime.strptime(parts[-2], "%Y-%m-%d")
            cached_end   = datetime.strptime(parts[-1], "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            continue
        if cached_start <= start and cached_end >= end:
            try:
                df = pd.read_csv(f, parse_dates=["time"])
            except Exception:
                continue
            sliced = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
            if len(sliced) > 0:
                print(f"  ♻️  Reusing cached {symbol} {interval} from {f.name} "
                      f"({len(sliced)} bars, no API call needed)")
                return sliced
    return None


def fetch_historical_bars(symbol: str, interval: str, start: datetime, end: datetime,
                           api_key: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch full OHLC history for [start, end], chunked + cached to disk.
    Reuses any overlapping prior cache, throttles to the free-tier rate
    limit, and retries (rather than silently skipping) on rate-limit hits.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "")
    cache_file = cache_dir / f"{safe_symbol}_{interval}_{start.date()}_{end.date()}.csv"

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    reused = _find_reusable_cache(symbol, interval, start, end, cache_dir)
    if reused is not None:
        reused.to_csv(cache_file, index=False)  # save under this exact range too
        return reused

    # Collect existing local sources (parquet files + existing cache CSVs)
    local_dfs = []
    p_parquet = Path(f"data/bars_{interval}_{safe_symbol}.parquet")
    if p_parquet.exists():
        try:
            df_p = pd.read_parquet(p_parquet)
            df_p["time"] = pd.to_datetime(df_p["time"])
            df_p = df_p[(df_p["time"] >= start) & (df_p["time"] <= end)]
            if len(df_p) > 0:
                local_dfs.append(df_p)
        except Exception:
            pass

    for f in cache_dir.glob(f"{safe_symbol}_{interval}_*.csv"):
        try:
            df_c = pd.read_csv(f, parse_dates=["time"])
            df_c = df_c[(df_c["time"] >= start) & (df_c["time"] <= end)]
            if len(df_c) > 0:
                local_dfs.append(df_c)
        except Exception:
            pass

    existing_df = pd.concat(local_dfs, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True) if local_dfs else pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    # Determine missing time gaps that need to be fetched via API
    max_gap_days = 3.5  # weekend is ~2.5 days max
    gaps = []
    if len(existing_df) == 0:
        gaps.append((start, end))
    else:
        times = existing_df["time"].tolist()
        if times[0] > start + timedelta(days=max_gap_days):
            gaps.append((start, times[0]))
        for i in range(len(times) - 1):
            if (times[i+1] - times[i]) > timedelta(days=max_gap_days):
                gaps.append((times[i], times[i+1]))
        if times[-1] < end - timedelta(days=max_gap_days):
            gaps.append((times[-1], end))

    chunk_days = _CHUNK_DAYS.get(interval, 200)
    all_rows = []

    for g_start, g_end in gaps:
        chunk_end = g_end
        while chunk_end > g_start:
            chunk_start = max(g_start, chunk_end - timedelta(days=chunk_days))
            params = {
                "symbol": symbol, "interval": interval,
                "start_date": chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date":   chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                "apikey": api_key, "format": "JSON",
            }
            print(f"  📡 Fetching missing {symbol} {interval} chunk: {chunk_start.date()} → {chunk_end.date()}")

            max_retries = 5
            data = None
            for attempt in range(max_retries):
                _throttle(api_key)
                try:
                    resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    print(f"  ❌ Fetch failed for {symbol} {interval} chunk: {e}")
                    data = None
                    break

                if _is_rate_limit_response(data):
                    backoff = MIN_SECONDS_BETWEEN_CALLS * (attempt + 2)
                    print(f"  ⏳ Rate-limited on {symbol} {interval} "
                          f"(attempt {attempt+1}/{max_retries}) — waiting {backoff:.0f}s and retrying...")
                    time.sleep(backoff)
                    continue
                break

            if data is None or "values" not in data:
                reason = (data or {}).get("message", (data or {}).get("status", "unknown error")) if data else "request error"
                print(f"  ⚠️  No data for {symbol} {interval} chunk ({reason})")
                chunk_end = chunk_start - timedelta(seconds=1)
                continue

            for candle in data["values"]:
                try:
                    all_rows.append({
                        "time":  pd.to_datetime(candle["datetime"]),
                        "open":  float(candle["open"]),
                        "high":  float(candle["high"]),
                        "low":   float(candle["low"]),
                        "close": float(candle["close"]),
                    })
                except Exception:
                    continue

            chunk_end = chunk_start - timedelta(seconds=1)

    if all_rows:
        new_df = pd.DataFrame(all_rows)
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)

    if len(existing_df) == 0:
        print(f"  ⚠️  No historical data retrieved for {symbol} {interval}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = existing_df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. REPLAY CLOCK + FETCH_BARS MONKEYPATCH
# ═══════════════════════════════════════════════════════════════════════

class ReplayClock:
    """Mutable 'current simulated time' shared across all patched predictors."""
    def __init__(self):
        self.now: datetime = datetime.utcnow()


def make_replay_fetch_bars(symbol: str, historical: dict, clock: ReplayClock, pred: ProductionForexPredictor):
    """
    Returns a drop-in replacement for pred.fetch_bars() that serves only
    data up to clock.now — the key piece that prevents lookahead bias.
    Reuses the predictor's own _calculate_indicators() so EMA/RSI/ATR are
    computed identically to live, just on the time-sliced window.
    """
    def _replay_fetch_bars(interval: str = "15min", outputsize: int = 200):
        full_df = historical.get((symbol, interval))
        if full_df is None or len(full_df) == 0:
            return None
        sliced = full_df[full_df["time"] <= clock.now]
        if len(sliced) == 0:
            return None
        out = sliced.tail(outputsize).reset_index(drop=True).copy()
        out["avg_spread"] = 0.00005
        try:
            pred._calculate_indicators(out)
        except Exception as e:
            print(f"  ⚠️  indicator calc failed for {symbol} {interval}: {e}")
            return None
        return out
    return _replay_fetch_bars


# ═══════════════════════════════════════════════════════════════════════
# 3. HISTORICAL NEWS (best-effort — Finnhub economic calendar)
# ═══════════════════════════════════════════════════════════════════════

def fetch_historical_high_impact_events(start: datetime, end: datetime, finnhub_key: str) -> list:
    """
    Best-effort historical high-impact news fetch via Finnhub's economic
    calendar endpoint. Returns [] and prints a warning if unavailable —
    the replay still runs, just without historical news filtering.
    NOTE: verify your Finnhub plan actually includes historical range
    access to /calendar/economic before relying on this.
    """
    if not finnhub_key:
        print("  ⚠️  No Finnhub API key provided — skipping historical news filtering.")
        return []
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/calendar/economic",
            params={"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"), "token": finnhub_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        events = data.get("economicCalendar", data.get("data", []))
        if not events:
            print("  ⚠️  Finnhub returned no events — check plan access to historical economic calendar. "
                  "Continuing without news filtering.")
            return []
        high_impact = []
        for e in events:
            impact = str(e.get("impact", "")).lower()
            if impact not in ("high", "3"):  # Finnhub sometimes uses numeric impact levels
                continue
            try:
                ev_time = pd.to_datetime(e.get("time") or e.get("date"))
                if ev_time.tzinfo is not None:
                    ev_time = ev_time.tz_convert("UTC").tz_localize(None)
                high_impact.append({
                    "time": ev_time.to_pydatetime(),
                    "country": str(e.get("country", "")).upper(),
                    "event": e.get("event", "unknown"),
                })
            except Exception:
                continue
        print(f"  ✅ Loaded {len(high_impact)} historical high-impact events from Finnhub.")
        return high_impact
    except Exception as e:
        print(f"  ⚠️  Historical news fetch failed ({e}) — continuing without news filtering.")
        return []


def is_news_active_at(t: datetime, currencies: list, events: list, window_minutes: int = 30):
    for ev in events:
        if ev["country"] not in currencies:
            continue
        diff_min = abs((t - ev["time"]).total_seconds()) / 60.0
        if diff_min <= window_minutes:
            return True, ev
    return False, None


# ═══════════════════════════════════════════════════════════════════════
# 4. ATR CAP FUNCTION — Asset-Specific Entry Push Limits
# ═══════════════════════════════════════════════════════════════════════

def get_forex_atr_cap(symbol: str, raw_push: float, atr_15m: float, pip_size: float, scale: float = 1.0) -> float:
    """
    Cap the ATR-based entry push for Forex pairs to prevent missed fills.
    Gold/commodities keep a tight cap (max 0.5x ATR or $2.50) to catch shallow pullbacks.
    
    Args:
        symbol: Trading symbol (EUR/USD, XAU/USD, etc.)
        raw_push: Raw calculated push (ATR * velocity * scale * wick_damping)
        atr_15m: 15-minute ATR value
        pip_size: Pip size for this symbol
        scale: Tier scale (1.0 for mid, 1.75 for high)
    
    Returns:
        Capped entry push in price units (not pips)
    """
    if 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
        # Gold: dynamic ATR-scaled push cap (0.50x ATR * scale)
        max_push = atr_15m * 0.50 * scale if atr_15m and atr_15m > 0 else 5.0 * scale
        capped = min(raw_push, max_push)
        
    elif 'JPY' in symbol.upper():
        # USD/JPY: cap scaled by tier
        max_push_pips = 6.0 * scale
        max_push = max_push_pips * pip_size
        capped = min(raw_push, max_push)
        
    elif 'GBP' in symbol.upper():
        # GBP/USD: cap scaled by tier
        max_push_pips = 6.0 * scale
        max_push = max_push_pips * pip_size
        capped = min(raw_push, max_push)

    else:
        # EUR/USD, etc: cap scaled by tier
        max_push_pips = 4.0 * scale
        max_push = max_push_pips * pip_size
        capped = min(raw_push, max_push)
    
    return capped


# ═══════════════════════════════════════════════════════════════════════
# 5. SIGNAL TRIGGER + OUTCOME SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def trace_post_sl_price_action(df15: pd.DataFrame, exit_time: datetime, direction: str,
                                entry: float, sl: float, tp: float, pip_size: float,
                                trace_days: float = 3.0) -> dict:
    """
    After a hit_sl exit, keep walking forward through 15-min bars for
    `trace_days` to see what price actually did next. This is the
    sweep-vs-genuine-reversal check: a stop hunt / liquidity sweep looks
    like price tagging the SL, then reclaiming the original entry level
    and continuing on toward (or through) the original TP. A genuine
    directional move against the trade just keeps going the SL direction
    and never comes back.

    Returns:
      post_sl_reached_tp        — did price later touch the original TP level?
      post_sl_time_to_tp_hours  — hours from SL exit to that TP touch (None if never)
      post_sl_reclaimed_entry   — did price close back beyond the original entry
                                   price (i.e. fully round-tripped the loss)?
      post_sl_max_favorable_pips — best price reached in the TP direction during
                                   the trace window, measured in pips past the SL
                                   level (0 if price never recovered at all)
      post_sl_bars_checked      — how many 15-min bars were available to check
                                   (small numbers near the end of the dataset mean
                                   this read is truncated / not yet conclusive)
    """
    cutoff = exit_time + timedelta(days=trace_days)
    window = df15[(df15["time"] > exit_time) & (df15["time"] <= cutoff)]

    result = {
        "post_sl_reached_tp": False,
        "post_sl_time_to_tp_hours": None,
        "post_sl_reclaimed_entry": False,
        "post_sl_max_favorable_pips": 0.0,
        "post_sl_bars_checked": len(window),
    }
    if len(window) == 0 or not pip_size:
        return result

    if direction == "BUY":
        max_favorable = max(0.0, (window["high"].max() - sl) / pip_size)
        tp_hits = window[window["high"] >= tp]
        reclaim = window[window["close"] >= entry]
    else:
        max_favorable = max(0.0, (sl - window["low"].min()) / pip_size)
        tp_hits = window[window["low"] <= tp]
        reclaim = window[window["close"] <= entry]

    result["post_sl_max_favorable_pips"] = max_favorable
    result["post_sl_reclaimed_entry"] = len(reclaim) > 0
    if len(tp_hits) > 0:
        first = tp_hits.iloc[0]
        result["post_sl_reached_tp"] = True
        result["post_sl_time_to_tp_hours"] = (first["time"] - exit_time).total_seconds() / 3600.0

    return result


def trace_post_tp_price_action(df15: pd.DataFrame, exit_time: datetime, direction: str,
                                tp: float, pip_size: float, trace_days: float = 2.0) -> dict:
    """
    After hitting TP, keep walking forward to see how much further price moved
    in the trade direction. This reveals if you're exiting too early.
    
    Returns:
      post_tp_max_extension_pips — how many pips price continued after TP
      post_tp_bars_checked       — number of 15min bars checked
      post_tp_continued          — True if price moved >3 pips past TP
    """
    cutoff = exit_time + timedelta(days=trace_days)
    window = df15[(df15["time"] > exit_time) & (df15["time"] <= cutoff)]
    
    result = {
        "post_tp_max_extension_pips": 0.0,
        "post_tp_bars_checked": len(window),
        "post_tp_continued": False,
    }
    if len(window) == 0 or not pip_size:
        return result
    
    if direction == "BUY":
        max_price = window["high"].max()
        extension = (max_price - tp) / pip_size
    else:
        min_price = window["low"].min()
        extension = (tp - min_price) / pip_size
    
    # pandas' .max()/.min() return numpy.float64, so the comparison below
    # produces numpy.bool_ -- which the stdlib json module doesn't recognize
    # as a boolean. log_signal_jsonl's json.dumps(..., default=str) then
    # silently stringifies it to "True"/"False" instead of a real JSON
    # true/false. Any downstream code reading the jsonl and doing
    # `if row["post_tp_continued"]:` would treat BOTH "True" and "False" as
    # truthy, since any non-empty string is truthy -- meaning this field is
    # unusable for filtering as loaded, not just cosmetically wrong. Cast
    # explicitly so it's always a native Python bool and always serializes
    # as a real JSON boolean.
    result["post_tp_max_extension_pips"] = float(max(0.0, extension))
    result["post_tp_continued"] = bool(extension > pip_size * 3)  # >3 pips = meaningful extension
    return result


# ─────────────────────────────────────────────────────────────────────────
# Exhaustion scoring — replaces the old binary "consec>=2 -> skip" gate.
#
# Calibrated off replay_2026-08-20_130934 (16 triggered signals, 7 win / 9
# loss). Win-avg vs loss-avg on the touch-bar metrics:
#
#     metric                        win_avg   loss_avg   ratio
#     touch_rejection_wick_atr       0.23       1.35      5.9x   (strongest)
#     consecutive_directional_bars   0.43       1.44      3.4x
#     touch_bar_size_atr             1.50       3.01      2.0x
#     pre_touch_bar_size_atr         1.15       1.97      1.7x
#     approach_velocity_30m_atr      2.03       3.28      1.6x
#
# Weights below are roughly proportional to that separation; caps are
# roughly the loss_avg for each metric so a "typical bad touch" scores near
# 1.0 on that component alone. n=16 is thin -- retune once more replay data
# accumulates, same caveat that applied to the old skip_consec_threshold=2.
# Reweighted 2026-08-24 after checking the 3 "adjusted_never_filled" signals
# from the 083902 replay against real EUR/USD & USD/JPY bars. consecutive_
# directional_bars was maxing out at the minimum qualifying streak (2 bars,
# cap=2.0), contributing the same 0.25 to every case regardless of severity --
# zero discriminative value, 25% of total weight. touch_rejection_wick_atr was
# the feature that actually separated the one case that should NOT have been
# adjusted (wick=0.81, unadjusted trade hit TP on real bars) from the two that
# correctly were (wick=1.85-1.98, unadjusted trades hit SL on real bars).
# Moved weight off consec onto wick, widened the consec cap so it stops
# saturating on a bare 2-bar streak, and raised the mid threshold to match.
# Validated against only 3 signals -- rerun and check the win-rate-by-tier
# breakdown in the Stage-3 summary before trusting this beyond that sample.
EXHAUSTION_WEIGHTS = {
    "touch_rejection_wick_atr":      0.35,
    "consecutive_directional_bars":  0.25,
    "approach_velocity_30m_atr":     0.20,
    "touch_bar_size_atr":            0.10,
    "pre_touch_bar_size_atr":        0.10,
}
EXHAUSTION_CAPS = {
    "touch_rejection_wick_atr":      1.4,
    "consecutive_directional_bars":  3.0,
    "touch_bar_size_atr":            3.0,
    "pre_touch_bar_size_atr":        2.0,
    "approach_velocity_30m_atr":     3.5,
}
EXHAUSTION_MID_THRESHOLD  = 0.58
EXHAUSTION_HIGH_THRESHOLD = 0.75
ACCEL_SCORE_BONUS  = 0.30   # 5min knife-bar bump, folds the old accel check into the score
SL_PAD_BASE_ATR    = 0.40   # flat cushion added regardless of wick size
SL_PAD_WICK_ATR    = 0.60   # extra cushion proportional to the observed rejection wick
ENTRY_PUSH_VEL_ATR = 0.35   # entry push proportional to approach velocity
TIER_SCALE = {"mid": 1.0, "high": 1.75}   # how much harder "high" pushes than "mid"
MIN_RR_FLOOR = 1.5          # same floor concept as MIN_RR in the live TP-construction code
ADJUST_FILL_WINDOW_HOURS = 6.0   # how long to wait for price to reach an adjusted (pushed) entry
                                  # before giving up on it -- touch dynamics are a 15-30min-scale
                                  # phenomenon, so a sweep that hasn't resolved in 6h probably isn't
                                  # the same setup anymore. Retune with real data.

# Per-symbol override of the above. get_forex_atr_cap() already caps HOW FAR
# the entry gets pushed per symbol (gold's push is much larger in absolute
# terms than FX), but it doesn't change how LONG price has to retrace that
# distance. The 2026-08-24 replay showed EUR/USD expiring 2/2 adjusted
# signals and USD/JPY 1/4 as "adjusted_never_filled" within the flat 6h
# window, while XAU/USD and GBP/USD filled every adjusted signal (2/2, 1/1)
# -- gold's absolute pip moves retrace into its (larger) pushed limit fast
# enough that 6h is generous, but the tighter FX caps still need more
# wall-clock time to travel the same relative distance. Mirrors the
# zone_ttl_overrides pattern above; falls back to ADJUST_FILL_WINDOW_HOURS
# for any symbol not listed.
ADJUST_FILL_WINDOW_OVERRIDES = {
    "EUR/USD": 13.5,
    "GBP/USD": 13.5,
    "USD/JPY": 13.0,
    "XAU/USD": 12.0,   # increased for Gold to allow adequate retracement window
}


def get_fill_window_hours(symbol: str) -> float:
    return ADJUST_FILL_WINDOW_OVERRIDES.get(symbol, ADJUST_FILL_WINDOW_HOURS)


def exhaustion_score(features: dict, weights: dict = None, caps: dict = None) -> float:
    """Weighted composite of touch-bar exhaustion metrics, ~0 (clean touch)
    to ~1.5+ (severe). Replaces the old hard consec>=2 skip gate -- consec is
    now just one of five inputs instead of an automatic disqualifier."""
    weights = weights or EXHAUSTION_WEIGHTS
    caps = caps or EXHAUSTION_CAPS
    score = 0.0
    for key, weight in weights.items():
        val = features.get(key, 0.0) or 0.0
        cap = caps[key]
        score += weight * min(val / cap, 1.5)
    return round(score, 4)


def compute_trigger_decision(sig: dict, direction: str, approach_metrics: dict,
                              zone_top: float, zone_bot: float, pip_size: float,
                              symbol: str,
                              df5: Optional[pd.DataFrame],
                              skip_consec_threshold: int = 2,
                              accel_bar_size_atr_min: float = 2.0,
                              accel_body_ratio_min: float = 0.70,
                              sl_pad_base_atr: float = SL_PAD_BASE_ATR,
                              sl_pad_wick_atr: float = SL_PAD_WICK_ATR,
                              exhaustion_mid_threshold: float = EXHAUSTION_MID_THRESHOLD,
                              exhaustion_high_threshold: float = EXHAUSTION_HIGH_THRESHOLD,
                              accel_score_bonus: float = ACCEL_SCORE_BONUS) -> dict:
    """Replay-side port of PriceZoneMonitor._recompute_final_signal (live
    production_predictor.py).

    Decision, in order:
      1. Compute exhaustion_score() from the touch-bar metrics.
      2. score < EXHAUSTION_MID_THRESHOLD -> "unchanged" (clean touch).
      3. score >= EXHAUSTION_MID_THRESHOLD -> "adjust_mid" / "adjust_high":
         - Uses Wick Damping to compress entry push if rejection wick is already formed.
         - Capped tightly per pair by get_forex_atr_cap().
         - Minimal baseline SL padding only (does not inflate raw risk).
    """
    score = exhaustion_score(approach_metrics)

    accel_detected = False
    trigger_time = approach_metrics.get("_trigger_time_internal")
    try:
        if df5 is not None and len(df5) > 0 and trigger_time is not None:
            df5s = df5[df5["time"] <= trigger_time].tail(20)
            if len(df5s) >= 15:
                hi5, lo5, cl5 = df5s["high"], df5s["low"], df5s["close"]
                tr5 = pd.concat([hi5 - lo5, (hi5 - cl5.shift(1)).abs(), (lo5 - cl5.shift(1)).abs()], axis=1).max(axis=1)
                atr_5m = float(tr5.rolling(14).mean().iloc[-1])
                if not np.isfinite(atr_5m) or atr_5m <= 0:
                    atr_5m = pip_size * 5

                b = df5s.iloc[-1]
                bar_range = b["high"] - b["low"]
                if bar_range > 0:
                    bar_size_atr = bar_range / atr_5m
                    body_ratio   = abs(b["close"] - b["open"]) / bar_range
                    if bar_size_atr >= accel_bar_size_atr_min and body_ratio >= accel_body_ratio_min:
                        accel_detected = True
    except Exception:
        pass

    if accel_detected:
        score = round(score + accel_score_bonus, 4)

    orig_entry = float(sig["entry"])
    orig_sl    = float(sig["sl"])
    orig_tp    = float(sig["tp"])
    atr_15m    = approach_metrics.get("atr_15m") or (pip_size * 10)

    sign = 1.0 if direction == "SELL" else -1.0
    orig_risk   = abs(orig_entry - orig_sl)
    orig_reward = abs(orig_tp - orig_entry)
    rr = max(orig_reward / orig_risk, MIN_RR_FLOOR) if orig_risk > 0 else MIN_RR_FLOOR
    orig_risk_pips = orig_risk / pip_size if pip_size else 0.0

    is_gold = ('XAU' in symbol.upper() or 'GOLD' in symbol.upper())
    is_jpy  = ('JPY' in symbol.upper())

    # Minimal baseline buffer to prevent minor sweep wick-outs without inflating risk
    if is_gold:
        baseline_sl_pad = 0.35 * atr_15m if atr_15m and atr_15m > 0 else 3.50
    elif is_jpy:
        baseline_sl_pad = 4.0 * pip_size
    else:
        baseline_sl_pad = 3.0 * pip_size

    if score < exhaustion_mid_threshold:
        return {"action": "unchanged", "reason": f"score={score:.2f} -- clean touch",
                "score": score, "final_signal": None}

    tier = "mid" if score < exhaustion_high_threshold else "high"
    scale = TIER_SCALE[tier]

    # ── Wick Damping: if touch candle already formed a strong rejection wick, reduce entry push ──
    touch_wick_atr = approach_metrics.get("touch_rejection_wick_atr", 0.0) or 0.0
    wick_damping = max(0.10, 1.0 - 0.60 * min(touch_wick_atr, 1.5))

    # ── Calculate raw push and apply asset-specific cap ────────────────
    raw_entry_push = ENTRY_PUSH_VEL_ATR * approach_metrics.get("approach_velocity_30m_atr", 0.0) * atr_15m * scale * wick_damping
    entry_push = get_forex_atr_cap(symbol, raw_entry_push, atr_15m, pip_size, scale)
    
    # ── Dynamic SL padding: combine base ATR cushion with observed rejection wick ──
    dyn_sl_pad = (sl_pad_base_atr + sl_pad_wick_atr * touch_wick_atr) * atr_15m
    # Prevent risk compression: SL pad must at least cover entry_push + baseline buffer,
    # and absorb incoming momentum wicks (dyn_sl_pad).
    sl_pad = max(baseline_sl_pad, dyn_sl_pad, entry_push + baseline_sl_pad)

    new_entry = orig_entry + sign * entry_push
    new_sl    = orig_sl + sign * sl_pad
    new_risk  = abs(new_entry - new_sl)
    if new_risk <= 0:
        return {"action": "unchanged", "reason": "degenerate risk after recompute -- kept original",
                "score": score, "final_signal": None}

    # Preserve structural target if it provides adequate R:R (>= MIN_RR_FLOOR),
    # otherwise recompute off new risk to maintain the floor.
    struct_reward = abs(orig_tp - new_entry)
    struct_rr = struct_reward / new_risk if new_risk > 0 else 0.0
    if struct_rr >= MIN_RR_FLOOR:
        new_tp = orig_tp
        new_reward = struct_reward
        final_rr = struct_rr
    else:
        new_reward = new_risk * MIN_RR_FLOOR
        new_tp = new_entry - sign * new_reward
        final_rr = MIN_RR_FLOOR

    final_signal = dict(sig)
    final_signal.update({
        "entry":       round(new_entry, 5),
        "sl":          round(new_sl, 5),
        "tp":          round(new_tp, 5),
        "risk_pips":   round(new_risk / pip_size, 1),
        "reward_pips": round(new_reward / pip_size, 1),
        "risk_reward": f"1:{final_rr:.1f}",
        "adjusted":    True,
        "tier":        tier,
        "adjustment_reason": f"score={score:.2f} ({tier}) -- entry pushed {entry_push / pip_size:.1f}p (wick_damped), "
                              f"SL buffered {sl_pad / pip_size:.1f}p, TP preserved/held R:R {final_rr:.1f}",
        "original_entry":  orig_entry,
        "original_sl":     orig_sl,
        "original_tp":     orig_tp,
        "original_risk_pips": sig.get("risk_pips"),
    })
    return {
        "action": f"adjust_{tier}",
        "reason": final_signal["adjustment_reason"],
        "score": score,
        "final_signal": final_signal,
    }


def simulate_trigger_and_outcome(sig: dict, symbol: str, proposed_at: datetime,
                                  historical: dict, zone_ttl_hours: float, pip_size: float,
                                  max_hold_days: int = 15, sl_trace_days: float = 3.0,
                                  apply_trigger_decision: bool = True,
                                  final_log_path: Optional[Path] = None,
                                  skip_consec_threshold: int = 2,
                                  accel_bar_size_atr_min: float = 2.0,
                                  accel_body_ratio_min: float = 0.70,
                                  sl_pad_base_atr: float = SL_PAD_BASE_ATR,
                                  sl_pad_wick_atr: float = SL_PAD_WICK_ATR,
                                  exhaustion_mid_threshold: float = EXHAUSTION_MID_THRESHOLD,
                                  exhaustion_high_threshold: float = EXHAUSTION_HIGH_THRESHOLD,
                                  accel_score_bonus: float = ACCEL_SCORE_BONUS):
    """
    Walks forward through historical 15-min bars to determine:
    1. Whether/when the zone got touched (trigger)
    2. Whether TP or SL was hit first once triggered

    Mirrors PriceZoneMonitor's zone_top/zone_bottom touch criteria for
    triggering. Assumes SL wins on same-bar ambiguity (see module docstring).

    Also reports, for diagnosing whether losses are an entry/SL/TP placement
    problem rather than a "wrong direction" problem:
      - entry_slippage_pips: how far the actual trigger bar's touch price
        was from the originally proposed entry (a large slippage means the
        zone was touched well past where the model thought entry would be).
      - sl_overflow_pips (losses only): how far PAST the SL level the exit
        bar's extreme (low for BUY / high for SELL) actually reached. A
        small overflow (a few pips) suggests a marginal wick take-out —
        SL may simply be too tight for the pair's volatility. A large
        overflow suggests a genuine directional move against the trade,
        not a placement problem.
      - same_bar_tie (bool): whether the exit bar's range ALSO touched TP,
        i.e. this loss is one of the ambiguous intrabar cases the module
        docstring flags — SL was assumed to win per the pessimistic
        tie-break rule, but on tick data it could have gone either way.
    """
    df15 = historical.get((symbol, "15min"))
    if df15 is None or len(df15) == 0:
        return {"outcome": "no_data"}

    direction = sig.get("direction")
    zone_top  = sig.get("zone_top", sig.get("entry"))
    zone_bot  = sig.get("zone_bottom", sig.get("entry"))
    entry     = sig.get("entry")
    sl        = sig.get("sl")
    tp        = sig.get("tp")
    if sl is None or tp is None or entry is None:
        return {"outcome": "invalid_setup"}

    ttl_cutoff = proposed_at + timedelta(hours=zone_ttl_hours)
    future = df15[(df15["time"] > proposed_at) & (df15["time"] <= ttl_cutoff)]

    trigger_time = None
    trigger_price = None
    for _, bar in future.iterrows():
        if direction == "BUY":
            in_zone = bar["low"] <= zone_top and bar["close"] >= zone_bot - (2 * pip_size)
        else:
            in_zone = bar["high"] >= zone_bot and bar["close"] <= zone_top + (2 * pip_size)
        if in_zone:
            trigger_time  = bar["time"]
            # Actual touch price: the extreme that pulled price into the zone.
            trigger_price = bar["low"] if direction == "BUY" else bar["high"]
            break

    if trigger_time is None:
        # Was the zone genuinely watched for its full TTL with no touch, or
        # did the replay just run out of price history before the TTL
        # clock finished? df15's last bar is the true end of what we know
        # for this symbol -- if the TTL window extends past that, this
        # signal hasn't actually been resolved one way or the other yet.
        # Mislabeling these as "expired_untriggered" understates the live
        # signal count for anything proposed near the end of the replay
        # window (most visibly on the pair/timeframe with the sparsest
        # data), and inflates the untriggered bucket with signals that may
        # still fill once more price data exists.
        last_available_time = df15["time"].max()
        if ttl_cutoff > last_available_time:
            return {
                "outcome": "still_pending",
                "zone_ttl_hours": zone_ttl_hours,
                "ttl_cutoff": ttl_cutoff,
                "last_available_time": last_available_time,
                "hours_of_ttl_remaining": (ttl_cutoff - last_available_time).total_seconds() / 3600,
            }
        return {"outcome": "expired_untriggered"}

    entry_slippage_pips = abs(trigger_price - entry) / pip_size if pip_size else None
    entry_decay_frac = None  # set below only when an adjusted entry fills via decay, not at full push

    # ── Pre-Entry & Approach Dynamics (30-min window before touch) ─────────
    approach_metrics = {}
    try:
        idx_match = df15.index[df15["time"] == trigger_time]
        if len(idx_match) > 0:
            t_idx = idx_match[0]
            start_atr_idx = max(0, t_idx - 25)
            sub_df = df15.iloc[start_atr_idx : t_idx + 1].copy()
            hi, lo, cl = sub_df["high"], sub_df["low"], sub_df["close"]
            tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
            atr_15m = float(tr.rolling(14).mean().iloc[-1])
            if not np.isfinite(atr_15m) or atr_15m <= 0:
                atr_15m = pip_size * 10 if pip_size else 0.0010

            b_touch = df15.iloc[t_idx]
            b_1ago = df15.iloc[t_idx - 1] if t_idx >= 1 else b_touch
            b_2ago = df15.iloc[t_idx - 2] if t_idx >= 2 else b_1ago

            # 1. 30m net approach velocity in ATR
            if direction == "BUY":
                approach_vel_atr = (b_2ago["open"] - b_touch["low"]) / atr_15m
                touch_wick_atr = (b_touch["close"] - b_touch["low"]) / atr_15m
            else:
                approach_vel_atr = (b_touch["high"] - b_2ago["open"]) / atr_15m
                touch_wick_atr = (b_touch["high"] - b_touch["close"]) / atr_15m

            # 2. Pre-touch candle expansion size in ATR & Body Ratio
            pre_bar_range = b_1ago["high"] - b_1ago["low"]
            pre_bar_size_atr = pre_bar_range / atr_15m
            pre_body_ratio = abs(b_1ago["close"] - b_1ago["open"]) / (pre_bar_range + 1e-9)

            # 3. Touch bar size in ATR & Body Ratio
            touch_bar_range = b_touch["high"] - b_touch["low"]
            touch_bar_size_atr = touch_bar_range / atr_15m
            touch_body_ratio = abs(b_touch["close"] - b_touch["open"]) / (touch_bar_range + 1e-9)

            # 4. Consecutive directional bars into zone
            consec = 0
            for k in range(1, min(7, t_idx + 1)):
                prev_b = df15.iloc[t_idx - k]
                is_dir = (prev_b["close"] < prev_b["open"]) if direction == "BUY" else (prev_b["close"] > prev_b["open"])
                if is_dir:
                    consec += 1
                else:
                    break

            # 5. Time from proposal to trigger (hours)
            time_to_trigger_hours = (trigger_time - proposed_at).total_seconds() / 3600.0

            approach_metrics = {
                "approach_velocity_30m_atr": round(float(approach_vel_atr), 3),
                "pre_touch_bar_size_atr":    round(float(pre_bar_size_atr), 3),
                "pre_touch_body_ratio":       round(float(pre_body_ratio), 3),
                "touch_bar_size_atr":        round(float(touch_bar_size_atr), 3),
                "touch_body_ratio":           round(float(touch_body_ratio), 3),
                "touch_rejection_wick_atr":   round(float(touch_wick_atr), 3),
                "consecutive_directional_bars": int(consec),
                "time_to_trigger_hours":      round(float(time_to_trigger_hours), 2),
                "atr_15m":                    round(float(atr_15m), 6),
            }
    except Exception as e:
        approach_metrics = {"approach_metrics_error": str(e)}

    # -- Stage-3 trigger-time decision: recompute the final signal ---------
    # Ported from PriceZoneMonitor._recompute_final_signal (live path).
    # This must run BEFORE the SL/TP forward walk below, since an
    # adjust_mid/adjust_high action means the walk needs to use the
    # recomputed entry/sl/tp (and, since the new entry is a limit price not
    # yet reached, wait for that fill) instead of the originally proposed
    # entry/sl/tp.
    original_sig = dict(sig)
    action, decision_reason, final_signal = "unchanged", None, None
    hold_start_time = trigger_time  # overridden below if we have to wait for an adjusted-entry fill
    if apply_trigger_decision:
        approach_metrics["_trigger_time_internal"] = trigger_time  # internal use only, not logged
        decision = compute_trigger_decision(
            sig, direction, approach_metrics, zone_top, zone_bot, pip_size, symbol,
            historical.get((symbol, "5min")),
            skip_consec_threshold=skip_consec_threshold,
            accel_bar_size_atr_min=accel_bar_size_atr_min,
            accel_body_ratio_min=accel_body_ratio_min,
            sl_pad_base_atr=sl_pad_base_atr,
            sl_pad_wick_atr=sl_pad_wick_atr,
            exhaustion_mid_threshold=exhaustion_mid_threshold,
            exhaustion_high_threshold=exhaustion_high_threshold,
            accel_score_bonus=accel_score_bonus,
        )
        action, decision_reason, final_signal = decision["action"], decision["reason"], decision["final_signal"]
        approach_metrics.pop("_trigger_time_internal", None)

        if final_log_path is not None:
            log_signal_jsonl(final_log_path, {
                "proposed_at": str(proposed_at), "symbol": symbol,
                "trigger_time": str(trigger_time), "action": action, "reason": decision_reason,
                "score": decision.get("score"),
                "original_signal": {k: original_sig.get(k) for k in
                                     ("direction", "entry", "sl", "tp", "risk_pips",
                                      "reward_pips", "risk_reward", "probability")},
                "final_signal": final_signal,
                "trigger_features": {k: v for k, v in approach_metrics.items()
                                      if k != "_trigger_time_internal"},
            })

        if action in ("adjust_mid", "adjust_high") and final_signal is not None:
            # New entry is a LIMIT order at a price not yet reached (pushed
            # further in the direction price was already running when it
            # touched the zone) -- have to wait for price to actually get
            # there before the SL/TP hold loop can start.
            #
            # Originally this held the pushed price fixed for the whole
            # fill window: either price fully retraced to the aggressive
            # level, or the signal expired with zero risk taken. Checked
            # against real EUR/USD & USD/JPY bars (2026-08-25): in the
            # persistent no-fill cases, price retraced PART way back --
            # sometimes most of the way -- but never the full pushed
            # distance, even over a 24h lookahead. That's a real tradeable
            # retracement being thrown away as a non-signal.
            #
            # First attempt was a linear time-decay (relax the required
            # price from pushed toward raw as the window elapses) -- but
            # checked against real bars, this DIDN'T work: the actual peak
            # retracement on the EUR/USD case happened ~1.25h after
            # trigger, while the decay schedule was still too strict at
            # that point; by the time decay loosened enough (~4.7h in),
            # price had already reversed and moved on. Time-decay assumes
            # the best opportunity comes late; in practice it came early.
            #
            # Fixed version: treat it as a resting limit order across the
            # whole [raw_entry, pushed_entry] range, filled the FIRST time
            # price reaches ANY point in that range, at whatever price is
            # actually offered on that bar (capped at the ideal pushed
            # price -- you never pay worse than what triggered the
            # adjustment, and never demand better than what actually
            # happened). Verified against real bars: this fills the
            # EUR/USD 07-02 case in 15 min at 1.14557 (real, decent price).
            # It does NOT eliminate genuine losses -- if price never comes
            # back to the raw entry either (confirmed still true for the
            # USD/JPY 08-08 case even with this fix), it still correctly
            # expires with zero risk taken.
            entry = final_signal["entry"]
            sl    = final_signal["sl"]
            tp    = final_signal["tp"]
            raw_entry = original_sig.get("entry", entry)

            window_hours = get_fill_window_hours(symbol)
            fill_cutoff = trigger_time + timedelta(hours=window_hours)
            fill_window = df15[(df15["time"] > trigger_time) & (df15["time"] <= fill_cutoff)]
            fill_time = None
            filled_entry = None
            for _, fbar in fill_window.iterrows():
                if direction == "SELL":
                    reached = fbar["high"] >= raw_entry
                    price_offered = min(fbar["high"], entry)   # entry here is still the pushed price
                else:
                    reached = fbar["low"] <= raw_entry
                    price_offered = max(fbar["low"], entry)
                if reached:
                    fill_time = fbar["time"]
                    filled_entry = price_offered
                    break

            if fill_time is None:
                last_available_time = df15["time"].max()
                if fill_cutoff > last_available_time:
                    return {
                        "outcome": "still_pending", "trigger_time": trigger_time,
                        "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips,
                        "action": action, "adjusted": True, "tier": final_signal.get("tier"),
                        "adjustment_reason": decision_reason,
                        "fill_window_hours": window_hours,
                        "fill_cutoff": fill_cutoff,
                        "last_available_time": last_available_time,
                        "hours_of_fill_window_remaining": (fill_cutoff - last_available_time).total_seconds() / 3600,
                    }
                return {
                    "outcome": "adjusted_never_filled", "trigger_time": trigger_time,
                    "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips,
                    "action": action, "adjusted": True, "tier": final_signal.get("tier"),
                    "adjustment_reason": decision_reason,
                    "entry": entry, "sl": sl, "tp": tp,
                    **approach_metrics,
                }
            # Re-center SL/TP on the price we actually got filled at, keeping
            # the same risk/reward distances the adjustment intended (same
            # SL cushion, same target R:R) rather than reusing the bracket
            # computed for the fully-pushed entry that never happened.
            entry_decay_frac = (abs(entry - filled_entry) / abs(entry - raw_entry)
                                 if entry != raw_entry else 0.0)  # 0 = filled at full push, 1 = filled at raw entry
            sl_shift = sl - entry
            tp_shift = tp - entry
            entry = filled_entry
            sl = entry + sl_shift
            tp = entry + tp_shift
            hold_start_time = fill_time
        elif final_signal is not None:
            # Baseline buffer or non-pushed adjustment: update entry, sl, tp directly
            entry = final_signal["entry"]
            sl    = final_signal["sl"]
            tp    = final_signal["tp"]
            hold_start_time = trigger_time

    max_cutoff = hold_start_time + timedelta(days=max_hold_days)
    hold = df15[(df15["time"] > hold_start_time) & (df15["time"] <= max_cutoff)]

    for _, bar in hold.iterrows():
        hit_sl = (bar["low"] <= sl) if direction == "BUY" else (bar["high"] >= sl)
        hit_tp = (bar["high"] >= tp) if direction == "BUY" else (bar["low"] <= tp)
        if hit_sl:  # conservative: SL wins ties within the same bar
            sl_overflow = (sl - bar["low"]) if direction == "BUY" else (bar["high"] - sl)
            sl_overflow_pips = sl_overflow / pip_size if pip_size else None
            post_sl = trace_post_sl_price_action(
                df15, bar["time"], direction, entry, sl, tp, pip_size, trace_days=sl_trace_days
            )
            return {
                "outcome": "hit_sl", "trigger_time": trigger_time, "exit_time": bar["time"],
                "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips,
                "sl_overflow_pips": sl_overflow_pips, "same_bar_tie": bool(hit_tp),
                "action": action, "adjusted": bool(final_signal is not None and final_signal.get("adjusted", False)),
                "tier": (final_signal or {}).get("tier"), "entry_decay_frac": entry_decay_frac,
                "entry": entry, "sl": sl, "tp": tp,
                **approach_metrics,
                **post_sl,
            }
        if hit_tp:
            # ── POST-TP TRACE: how far did price continue after hitting TP? ──
            post_tp = trace_post_tp_price_action(df15, bar["time"], direction, tp, pip_size)

            # ── MILESTONE TP1/TP2 TRACKING (Method 1) ────────────────────
            # tp here is TP1 (the baseline target already computed above).
            # Evaluate the touch bar's momentum to decide whether this run
            # would have locked profit and extended to TP2, then log
            # whether TP1 / TP2 were each actually achieved.
            bars_after_tp1 = df15[df15["time"] > bar["time"]]
            prior_bars_tp1 = df15[df15["time"] < bar["time"]].tail(10)
            milestone = evaluate_milestone(
                touch_bar=bar, bars_after=bars_after_tp1, direction=direction,
                entry=entry, tp1=tp, sl=sl, symbol=symbol, pip_size=pip_size,
                prior_bars=prior_bars_tp1,
            )
            if final_log_path is not None:
                milestone_log_path = final_log_path.parent / f"{final_log_path.stem}_milestones.jsonl"
                log_milestone_event(milestone_log_path, {
                    "symbol": symbol, "direction": direction, "proposed_at": str(proposed_at),
                    "entry": entry, "sl": sl, "tp1": tp, **milestone,
                })

            return {
                "outcome": "hit_tp", "trigger_time": trigger_time, "exit_time": bar["time"],
                "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips,
                "action": action, "adjusted": bool(final_signal is not None and final_signal.get("adjusted", False)),
                "tier": (final_signal or {}).get("tier"), "entry_decay_frac": entry_decay_frac,
                "entry": entry, "sl": sl, "tp": tp,
                **approach_metrics,
                **post_tp,
                "milestone": milestone,
            }

    return {"outcome": "timeout", "trigger_time": trigger_time,
            "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips,
            "action": action, "adjusted": bool(final_signal is not None and final_signal.get("adjusted", False)),
            "tier": (final_signal or {}).get("tier"), "entry_decay_frac": entry_decay_frac,
            "entry": entry, "sl": sl, "tp": tp,
            **approach_metrics}


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN REPLAY LOOP
# ═══════════════════════════════════════════════════════════════════════

SYMBOL_PARAMS = {
    "EUR/USD": dict(max_distance_pips_rev=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "GBP/USD": dict(max_distance_pips_rev=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "USD/JPY": dict(max_distance_pips_rev=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "XAU/USD": dict(max_distance_pips_rev=80.0, zone_distance_threshold=150.0, min_strength_atr=0.40),
}


def build_reversion_predictor(symbol: str, api_key: str, debug: bool = True,
                               zone_ttl_hours: float = 48.0,
                               prob_threshold: float = 0.60) -> ProductionForexPredictor:
    """Same construction params as run_dual_strategy()'s `rev` predictor, live.

    zone_ttl_hours default (48.0) matches the 4H institutional FVG/OB zone
    horizon — per the 2026-08-17/18 replay audit, 4H zones typically take
    24-72h to get revisited by price.
    """
    p = SYMBOL_PARAMS[symbol]
    return ProductionForexPredictor(
        api_key=api_key,
        model_path="models/model_v6_reversion.pkl",
        scaler_path="models/scaler_v6_reversion.pkl",
        encoders_path="models/encoders_v6_reversion.pkl",
        stats_path="models/training_stats_v6_reversion.pkl",
        symbol=symbol,
        prob_threshold=prob_threshold,   # matches live run_dual_strategy() REV predictor
        max_rr=8.0, min_risk_atr=2.5, max_distance_pips=p["max_distance_pips_rev"],
        min_strength_atr=0.50, use_5min_refinement=True,
        refine_max_dist_multiplier=2.0, enable_continuation=False,
        use_htf_analysis=True, min_confluence_score=1,
        zone_ttl_hours=zone_ttl_hours, zone_distance_threshold=p["zone_distance_threshold"],
        htf_fvg_min_gap_atr=0.5, htf_ob_min_reaction_atr=2.5,
        is_continuation_strategy=False, require_zone_close=True, debug=debug,
    )


def plot_signal_chart(result: dict, df15: pd.DataFrame, out_path: Path,
                       lookback_bars: int = 40, lookforward_bars: int = 40):
    """Render one signal as a candlestick PNG: entry/SL/TP lines, the zone
    box, and trigger/exit markers, so signals can be reviewed one-by-one
    against a real chart instead of just a CSV row. Pure matplotlib (no
    extra dependency like mplfinance) so it runs anywhere pandas/matplotlib
    already do.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless — just writes files, no display needed
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    proposed_at = pd.to_datetime(result["proposed_at"])
    trigger_time = pd.to_datetime(result.get("trigger_time")) if result.get("trigger_time") else None
    exit_time    = pd.to_datetime(result.get("exit_time")) if result.get("exit_time") else None

    anchor_end = exit_time or (proposed_at + timedelta(hours=lookforward_bars * 0.25))

    df15 = df15.sort_values("time").reset_index(drop=True)
    idx = (df15["time"] - proposed_at).abs().idxmin()
    end_idx = df15["time"].searchsorted(anchor_end, side="right")
    lo = max(0, idx - lookback_bars)
    hi = min(len(df15), max(end_idx, idx) + lookforward_bars)
    window = df15.iloc[lo:hi].reset_index(drop=True)
    if len(window) == 0:
        return  # nothing to plot (shouldn't normally happen)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = mdates.date2num(window["time"])
    width = (x[1] - x[0]) * 0.6 if len(x) > 1 else 0.005

    up = window["close"] >= window["open"]
    for i in range(len(window)):
        color = "#1baf7a" if up.iloc[i] else "#e34948"
        ax.plot([x[i], x[i]], [window["low"].iloc[i], window["high"].iloc[i]],
                color=color, linewidth=1, zorder=2)
        ax.add_patch(plt.Rectangle(
            (x[i] - width / 2, min(window["open"].iloc[i], window["close"].iloc[i])),
            width, max(abs(window["close"].iloc[i] - window["open"].iloc[i]), 1e-9),
            facecolor=color, edgecolor=color, zorder=3))

    entry = result.get("entry")
    sl = result.get("sl")
    tp = result.get("tp")
    zone_top = result.get("zone_top")
    zone_bottom = result.get("zone_bottom")

    if zone_top is not None and zone_bottom is not None:
        ax.axhspan(min(zone_top, zone_bottom), max(zone_top, zone_bottom),
                   color="#8a8dfa", alpha=0.15, zorder=1, label="Zone")
    if entry is not None:
        ax.axhline(entry, color="#2a78d6", linestyle="--", linewidth=1.3, label=f"Entry {entry}")
    if sl is not None:
        ax.axhline(sl, color="#e34948", linestyle="--", linewidth=1.3, label=f"SL {sl}")
    if tp is not None:
        ax.axhline(tp, color="#1baf7a", linestyle="--", linewidth=1.3, label=f"TP {tp}")

    for label, ts, color in [("Proposed", proposed_at, "#52514e"),
                              ("Triggered", trigger_time, "#2a78d6"),
                              ("Exit", exit_time, "#e34948" if result.get("outcome") == "hit_sl" else "#1baf7a")]:
        if ts is not None and window["time"].min() <= ts <= window["time"].max():
            xv = mdates.date2num(ts)
            ax.axvline(xv, color=color, linestyle=":", linewidth=1, alpha=0.8)
            ax.text(xv, ax.get_ylim()[1], label, rotation=90, fontsize=8,
                    va="top", ha="right", color=color)

    outcome = result.get("outcome", "?")
    outcome_label = {"hit_tp": "WIN (hit TP)", "hit_sl": "LOSS (hit SL)",
                      "expired_untriggered": "EXPIRED — never triggered",
                      "still_pending": "PENDING — replay ended before TTL/fill window closed",
                      "timeout": "TIMEOUT"}.get(outcome, outcome)
    r_mult = result.get("r_multiple")
    r_note = f"  R={r_mult:.2f}" if r_mult is not None else ""
    ax.set_title(f"{result.get('symbol')}  {result.get('direction')}  "
                 f"proposed {proposed_at}  —  {outcome_label}{r_note}", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_ylabel("Price")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def generate_signal_charts(symbol: str, all_results: list, df15: pd.DataFrame, charts_dir: Path):
    """Plot every signal for one symbol as a numbered PNG (chronological
    order) so they can be flipped through one-by-one against a real chart."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    sym_results = [r for r in all_results if r["symbol"] == symbol]
    sym_results.sort(key=lambda r: r["proposed_at"])
    safe_symbol = symbol.replace("/", "")
    outcome_tag = {"hit_tp": "TP", "hit_sl": "SL", "expired_untriggered": "EXP",
                   "still_pending": "PEND", "timeout": "TO"}
    for n, r in enumerate(sym_results, 1):
        tag = outcome_tag.get(r["outcome"], r["outcome"])
        date_str = str(pd.to_datetime(r["proposed_at"]).date())
        fname = f"{n:02d}_{safe_symbol}_{date_str}_{tag}.png"
        try:
            plot_signal_chart(r, df15, charts_dir / fname)
        except Exception as e:
            print(f"  ⚠️  Chart failed for {symbol} {r['proposed_at']}: {e}")
    print(f"  🖼️  Saved {len(sym_results)} signal charts for {symbol} → {charts_dir}/")


def run_replay(pairs: list, days_back: int, finnhub_key: str, use_news: bool, out_dir: Path,
                debug: bool = True, sl_trace_days: float = 3.0, zone_ttl_hours: float = 48.0,
                zone_dedup_days: float = 3.0, make_charts: bool = True,
                zone_ttl_overrides: dict = None, apply_trigger_decision: bool = True,
                skip_consec_threshold: int = 2, accel_bar_size_atr_min: float = 2.0,
                accel_body_ratio_min: float = 0.70, prob_threshold: float = 0.60,
                timeframe: str = "4h",
                sl_pad_base_atr: float = SL_PAD_BASE_ATR,
                sl_pad_wick_atr: float = SL_PAD_WICK_ATR,
                exhaustion_mid_threshold: float = EXHAUSTION_MID_THRESHOLD,
                exhaustion_high_threshold: float = EXHAUSTION_HIGH_THRESHOLD,
                accel_score_bonus: float = ACCEL_SCORE_BONUS,
                end_date: Optional[str] = None):
    zone_ttl_overrides = zone_ttl_overrides or {}
    zone_ttl_overrides.setdefault("XAU/USD", 72.0)

    # Parse timeframes (e.g. "4h" or "4h,1h")
    if isinstance(timeframe, str):
        tf_list = [tf.strip() for tf in timeframe.split(",") if tf.strip()]
    else:
        tf_list = list(timeframe)
    if not tf_list:
        tf_list = ["4h"]

    if end_date:
        if isinstance(end_date, str):
            end = pd.to_datetime(end_date).to_pydatetime()
        else:
            end = end_date
    else:
        end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    warmup_start = start - timedelta(days=45)  # buffer for EMA200/ATR warmup on 4H

    cache_dir = out_dir / "bar_cache"

    run_tag = end.strftime("%Y-%m-%d_%H%M%S")
    log_path, jsonl_path, close_logging = setup_detailed_logging(out_dir, run_tag)
    final_log_path = (out_dir / "logs" / f"final_signals_{run_tag}.jsonl") if apply_trigger_decision else None

    print(f"\n{'═'*70}\nREPLAY WINDOW: {start.date()} → {end.date()}  ({days_back} days)\n{'═'*70}")
    print(f"📝 Detailed log:   {log_path}")
    print(f"📝 Per-signal log: {jsonl_path}  (JSON-lines, one record per proposed signal)")
    if apply_trigger_decision:
        print(f"📝 Stage-3 dual-signal log: {final_log_path}  "
              f"(original vs. final signal, per trigger — exhaustion-score adjust decisions)")
        print(f"⚙️  Stage-3 trigger decision ENABLED: exhaustion_mid_threshold={exhaustion_mid_threshold}, "
              f"exhaustion_high_threshold={exhaustion_high_threshold}, "
              f"accel_bar_size_atr_min={accel_bar_size_atr_min}, accel_body_ratio_min={accel_body_ratio_min}, "
              f"accel_score_bonus={accel_score_bonus}, "
              f"sl_pad_base_atr={sl_pad_base_atr}, sl_pad_wick_atr={sl_pad_wick_atr}, "
              f"adjust_fill_window_hours(default)={ADJUST_FILL_WINDOW_HOURS}, "
              f"adjust_fill_window_overrides={ADJUST_FILL_WINDOW_OVERRIDES}")
    else:
        print(f"⚙️  Stage-3 trigger decision DISABLED (--no-trigger-decision) — raw signals only, "
              f"same behavior as before this feature existed.")
    print(f"Run params: pairs={pairs} timeframes={tf_list} days_back={days_back} use_news={use_news} debug={debug} "
          f"prob_threshold={prob_threshold} "
          f"zone_ttl_hours={zone_ttl_hours} zone_ttl_overrides={zone_ttl_overrides} "
          f"zone_dedup_days={zone_dedup_days} sl_trace_days={sl_trace_days}")

    news_events = []
    if use_news:
        print("\n📰 Fetching historical high-impact news...")
        news_events = fetch_historical_high_impact_events(start, end, finnhub_key)


    all_results = []

    for symbol in pairs:
        print(f"\n{'─'*70}\n{symbol}\n{'─'*70}")
        api_key = PAIR_API_KEYS.get(symbol, list(PAIR_API_KEYS.values())[0])
        symbol_ttl_hours = zone_ttl_overrides.get(symbol, zone_ttl_hours)
        if symbol in zone_ttl_overrides:
            print(f"  ⏱️  [{symbol}] using overridden zone_ttl_hours={symbol_ttl_hours} "
                  f"(default for this run is {zone_ttl_hours})")

        historical = {}
        for interval in ["4h", "1h", "15min", "5min", "1day"]:
            historical[(symbol, interval)] = fetch_historical_bars(
                symbol, interval, warmup_start, end, api_key, cache_dir
            )

        pred = build_reversion_predictor(symbol, api_key, debug=debug, zone_ttl_hours=symbol_ttl_hours,
                                         prob_threshold=prob_threshold)
        clock = ReplayClock()
        pred.fetch_bars = make_replay_fetch_bars(symbol, historical, clock, pred)
        pip_size = pred.pip_size
        print(f"  📏 [{symbol}] pip_size resolved to {pip_size} "
              f"(0.0001=standard FX, 0.01=JPY pair, 1.0=XAU/gold — confirms this pair "
              f"isn't silently using the wrong scale)")

        walk_tf = "1h" if "1h" in tf_list else "4h"
        walk_bars = historical[(symbol, walk_tf)]
        walk_points = walk_bars[(walk_bars["time"] >= start) & (walk_bars["time"] <= end)]["time"].tolist()
        four_h_times = set(historical[(symbol, "4h")]["time"].tolist())
        n_points = len(walk_points)
        print(f"  🚶 Walking {n_points} historical {walk_tf.upper()} candle closes for {symbol} (Scanning: {tf_list})...")
        loop_start_ts = time.time()

        # ── Open-trade guard (real capital only) ────────────────────────
        # Only blocks NEW proposals while a signal has ACTUALLY triggered
        # (touched its zone and started running as a live trade) — mirrors
        # live's one-trade-per-symbol guard. A zone that's merely proposed
        # and still being watched (untriggered) no longer blocks detection:
        # multiple pending/watched zones can coexist and each is scored
        # independently. Tracked as a list of (trigger_time, exit_time)
        # windows for trades that actually got filled; a walk-point t is
        # guarded only if it falls inside one of these.
        open_trade_windows = []  # list[(trigger_time, exit_time)]
        currencies = [c.upper() for c in symbol.replace("/", "_").split("_")]

        last_zone_key  = None
        last_zone_time = None

        for i, t in enumerate(walk_points, 1):
            if i == 1 or i % 20 == 0 or i == n_points:
                elapsed = time.time() - loop_start_ts
                rate = elapsed / i if i > 0 else 0
                eta = rate * (n_points - i)
                print(f"     [{i}/{n_points}] {t}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                      f"{len(all_results)} signals so far)")

            clock.now = t

            # Only skip detection while a REAL open trade (already triggered)
            # is running for this symbol — watched-but-untriggered zones no
            # longer block anything.
            if any(w0 <= t < w1 for (w0, w1) in open_trade_windows):
                continue

            if use_news and news_events:
                active, ev = is_news_active_at(t, currencies, news_events)
                if active:
                    continue

            # Scan timeframes (e.g. 4h, 1h)
            for scan_tf in tf_list:
                if scan_tf == "4h" and "1h" in tf_list and t not in four_h_times:
                    continue

                try:
                    htf_context = pred.get_htf_context()
                    raw = pred.htf_trade_detect(scan_tf, htf_context=htf_context)
                except Exception as e:
                    print(f"  ⚠️  {symbol} [{scan_tf}] @ {t}: detection error — {e}")
                    continue

                signals = raw.get("htf_signals", raw) if isinstance(raw, dict) else raw
                if not signals:
                    continue

                sig = signals[0]  # highest-probability signal, mirrors live's top pick
                effective_ttl_hours = getattr(pred.zone_tracker, 'ttl_seconds',
                                               symbol_ttl_hours * 3600) / 3600.0

                # ── Dedup check: same (direction, exact zone bounds) as last proposal? ──
                _entry = sig.get('entry')
                _zt = sig.get('zone_top')
                _zb = sig.get('zone_bottom')
                if pip_size:
                    quantize = pip_size * 0.1  # sub-0.1-pip differences = float noise, not a new zone
                    _q = lambda v: round(v / quantize) if v is not None else None
                    zone_key = (sig.get('direction'),
                                _q(_zt) if _zt is not None else _q(_entry),
                                _q(_zb) if _zb is not None else _q(_entry))
                else:
                    zone_key = (sig.get('direction'), _entry, _entry)
                if (last_zone_key is not None and zone_key == last_zone_key
                        and last_zone_time is not None
                        and (t - last_zone_time) <= timedelta(days=zone_dedup_days)):
                    if debug:
                        hrs_since = (t - last_zone_time).total_seconds() / 3600.0
                        print(f"  🔁 [{symbol}] Exact same zone re-detected {hrs_since:.1f}h after last "
                              f"proposal ({sig.get('direction')} {_zb}-{_zt}) — treating as a "
                              f"continuation, NOT logging as a new signal.")
                    break

                last_zone_key, last_zone_time = zone_key, t

                print(f"  📌 [{symbol}|{scan_tf.upper()}] Signal proposed @ {t}: {sig.get('direction')} "
                      f"entry={sig.get('entry')} sl={sig.get('sl')} tp={sig.get('tp')} "
                      f"prob={sig.get('probability')}")
                print(f"     ↳ full signal fields: {json.dumps(sig, default=str)}")

                result = simulate_trigger_and_outcome(sig, symbol, t, historical,
                                                       zone_ttl_hours=effective_ttl_hours, pip_size=pip_size,
                                                       sl_trace_days=sl_trace_days,
                                                       apply_trigger_decision=apply_trigger_decision,
                                                       final_log_path=final_log_path,
                                                       skip_consec_threshold=skip_consec_threshold,
                                                       accel_bar_size_atr_min=accel_bar_size_atr_min,
                                                       accel_body_ratio_min=accel_body_ratio_min,
                                                       sl_pad_base_atr=sl_pad_base_atr,
                                                       sl_pad_wick_atr=sl_pad_wick_atr,
                                                       exhaustion_mid_threshold=exhaustion_mid_threshold,
                                                       exhaustion_high_threshold=exhaustion_high_threshold,
                                                       accel_score_bonus=accel_score_bonus)

                entry = result.get("entry", sig.get("entry"))
                sl    = result.get("sl", sig.get("sl"))
                tp    = result.get("tp", sig.get("tp"))
                risk_pips = reward_pips = r_multiple = None
                if entry is not None and sl is not None and tp is not None and pip_size:
                    risk_pips   = abs(entry - sl) / pip_size
                    reward_pips = abs(tp - entry) / pip_size
                    if result["outcome"] == "hit_tp":
                        r_multiple = reward_pips / risk_pips if risk_pips else None
                    elif result["outcome"] == "hit_sl":
                        r_multiple = -1.0

                result.update({
                    "symbol": symbol, "proposed_at": t, "direction": sig.get("direction"),
                    "entry": entry, "sl": sl, "tp": tp,
                    "probability": sig.get("probability"),
                    "risk_reward": f"1:{(reward_pips / risk_pips):.1f}" if risk_pips else sig.get("risk_reward"),
                    "confluence_score": sig.get("confluence_score", sig.get("confluence")),
                    "risk_pips": risk_pips, "reward_pips": reward_pips, "r_multiple": r_multiple,
                    "zone_top": sig.get("zone_top"), "zone_bottom": sig.get("zone_bottom"),
                    "timeframe": sig.get("timeframe", scan_tf), "pip_size": pip_size,
                })
                all_results.append(result)
                log_signal_jsonl(jsonl_path, {**result, "raw_signal": sig})

                outcome = result["outcome"]

                # ── Open-trade collision check ──────────────────────────
                # Only "hit_tp"/"hit_sl"/"timeout" represent a REAL trigger
                # (real capital risked). If this trigger_time lands inside a
                # window another already-proposed signal for this symbol is
                # still open in, live would have blocked it — flag it
                # rather than dropping it, so you keep every signal for
                # validation but can still filter down to "what live would
                # actually have taken" after the fact. Non-colliding real
                # trades register their own window so later proposals can
                # be checked against them too.
                would_be_blocked = False
                if outcome in ("hit_tp", "hit_sl", "timeout"):
                    trig_t = result.get("trigger_time")
                    exit_t = result.get("exit_time", trig_t + timedelta(days=15) if trig_t else None)
                    if trig_t is not None:
                        would_be_blocked = any(w0 <= trig_t < w1 for (w0, w1) in open_trade_windows)
                        if not would_be_blocked:
                            open_trade_windows.append((trig_t, exit_t))
                result["would_be_blocked_by_open_trade"] = would_be_blocked
                if would_be_blocked:
                    print(f"     🔒 NOTE: this trigger overlaps another open trade on {symbol} — "
                          f"live would have blocked it (flagged, not dropped, in this replay).")

                if outcome == "expired_untriggered":
                    print(f"     ⌛ Never triggered — zone expired after {effective_ttl_hours:.1f}h watch window.")
                    # No lock: an untriggered zone never risked capital.
                elif outcome == "still_pending":
                    remaining = result.get("hours_of_ttl_remaining", result.get("hours_of_fill_window_remaining"))
                    print(f"     ⏳ STILL PENDING — replay ended before this signal's watch/fill "
                          f"window closed (~{remaining:.1f}h of window left unobserved). "
                          f"Excluded from win-rate/R stats, not counted as expired.")
                elif outcome == "adjusted_never_filled":
                    print(f"     🚫 ADJUSTED but never filled — {result.get('adjustment_reason')}")
                    print(f"        (pushed entry {result.get('entry')} never reached within "
                          f"{get_fill_window_hours(symbol):.0f}h; zero risk taken, not counted as win/loss)")
                    # Same as above — never filled means no capital at risk.
                elif outcome == "timeout":
                    print(f"     ⏱️  Triggered {result.get('trigger_time')} → ran past max hold "
                          f"without hitting TP/SL (timeout).")
                elif outcome in ("hit_tp", "hit_sl"):
                    emoji = "✅" if outcome == "hit_tp" else "❌"
                    print(f"     {emoji} Triggered {result.get('trigger_time')} → {outcome.upper()} "
                          f"at {result.get('exit_time')}  (R={r_multiple})")
                    
                    if outcome == "hit_tp":
                        ext_pips = result.get("post_tp_max_extension_pips", 0)
                        continued = result.get("post_tp_continued", False)
                        if ext_pips > 0:
                            print(f"        ↳ POST-TP EXTENSION: price continued {ext_pips:.1f} pips past TP "
                                  f"({continued and '⬆️ meaningful' or '⬇️ minimal'})")
                            if ext_pips > 10:
                                print(f"           ⚠️  You're exiting TOO EARLY — consider wider TP or trailing stop!")
                    
                    if result.get("adjusted"):
                        print(f"        ↳ 🔧 ADJUSTED at trigger ({result.get('tier')}): "
                              f"entry {sig.get('entry')} → {entry}, sl {sig.get('sl')} → {sl}, "
                              f"tp {sig.get('tp')} → {tp} ({result.get('action')})")
                    if "approach_velocity_30m_atr" in result:
                        print(f"        ↳ APPROACH DYNAMICS (30m pre-entry): vel_30m={result.get('approach_velocity_30m_atr')}x ATR | "
                              f"pre_bar={result.get('pre_touch_bar_size_atr')}x ATR (body={result.get('pre_touch_body_ratio')}) | "
                              f"touch_wick={result.get('touch_rejection_wick_atr')}x ATR | "
                              f"consec={result.get('consecutive_directional_bars')} bars | "
                              f"time_to_touch={result.get('time_to_trigger_hours')}h")
                    if outcome == "hit_sl":
                        overflow = result.get("sl_overflow_pips")
                        tie      = result.get("same_bar_tie")
                        slip     = result.get("entry_slippage_pips")
                        tie_note = "⚠️ SAME-BAR TIE (also touched TP — ambiguous, SL assumed by pessimistic tie-break)" if tie else "clean SL (TP not touched same bar)"
                        print(f"        ↳ LOSS DIAGNOSTIC: sl_overflow={overflow:.1f} pips past SL | "
                              f"entry_slippage={slip:.1f} pips from proposed entry | {tie_note}")

                        reached_tp   = result.get("post_sl_reached_tp")
                        time_to_tp   = result.get("post_sl_time_to_tp_hours")
                        reclaimed    = result.get("post_sl_reclaimed_entry")
                        max_fav      = result.get("post_sl_max_favorable_pips")
                        bars_checked = result.get("post_sl_bars_checked")
                        if bars_checked == 0:
                            print(f"        ↳ POST-SL TRACE: no bars available after exit — truncated, can't tell.")
                        elif reached_tp:
                            sweep_note = "reclaimed entry first — looks like a LIQUIDITY SWEEP, not a real reversal" if reclaimed else "did NOT reclaim entry before running to TP — check chart, unusual"
                            print(f"        ↳ POST-SL TRACE: price reached the original TP {time_to_tp:.1f}h later ({sweep_note})")
                        elif reclaimed:
                            print(f"        ↳ POST-SL TRACE: price reclaimed entry but never reached TP within "
                                  f"{sl_trace_days:.0f}d (best recovery: {max_fav:.1f} pips past SL) — partial bounce, not a clean sweep")
                        else:
                            print(f"        ↳ POST-SL TRACE: price never recovered ({max_fav:.1f} pips max bounce, "
                                  f"never touched original entry or TP) — CLEAN LOSS, genuine move against trade.")

                break
            else:
                pass

        if make_charts:
            print(f"  🖼️  Generating signal charts for {symbol}...")
            generate_signal_charts(symbol, all_results, historical[(symbol, "15min")],
                                    out_dir / "charts")

    # ── Report ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"replay_report_{end.date()}.json"
    with open(report_file, "w") as f:
        json.dump(all_results, f, default=str, indent=2)

    csv_file = out_dir / f"replay_report_{end.date()}.csv"
    try:
        pd.DataFrame(all_results).to_csv(csv_file, index=False)
    except Exception as e:
        print(f"  ⚠️  Could not write CSV report: {e}")
        csv_file = None

    def _bucket_stats(rows, label_fn):
        """Group resolved trades by label_fn(row) and print win-rate + expectancy per group."""
        buckets = {}
        for r in rows:
            buckets.setdefault(label_fn(r), []).append(r)
        for label in sorted(buckets, key=lambda x: (x is None, x)):
            grp = buckets[label]
            grp_wins = [r for r in grp if r["outcome"] == "hit_tp"]
            wr = len(grp_wins) / len(grp) * 100 if grp else 0.0
            r_vals = [r["r_multiple"] for r in grp if r.get("r_multiple") is not None]
            expectancy = sum(r_vals) / len(r_vals) if r_vals else 0.0
            print(f"    {str(label):<12} n={len(grp):<4} win_rate={wr:>5.1f}%  avg_R={expectancy:>+.2f}")

    print(f"\n{'═'*70}\nREPLAY SUMMARY\n{'═'*70}")
    resolved = [r for r in all_results if r["outcome"] in ("hit_tp", "hit_sl")]
    wins     = [r for r in resolved if r["outcome"] == "hit_tp"]
    untriggered = [r for r in all_results if r["outcome"] == "expired_untriggered"]
    still_pending = [r for r in all_results if r["outcome"] == "still_pending"]
    timed_out   = [r for r in all_results if r["outcome"] == "timeout"]

    print(f"Total proposed signals:  {len(all_results)}")
    print(f"Resolved (TP or SL):     {len(resolved)}")
    print(f"Never triggered (genuinely expired past TTL): {len(untriggered)}")
    print(f"Still pending (replay ended before TTL/fill window closed — excluded "
          f"from win-rate stats, not a real expiry): {len(still_pending)}")
    if still_pending:
        for r in sorted(still_pending, key=lambda r: r["proposed_at"]):
            remaining = r.get("hours_of_ttl_remaining", r.get("hours_of_fill_window_remaining"))
            print(f"    ⏳ {r['symbol']:<8} proposed {r['proposed_at']}  "
                  f"(~{remaining:.1f}h of window still unobserved — could still resolve on newer data)")
    print(f"Triggered but timed out: {len(timed_out)}")
    if resolved:
        r_vals = [r["r_multiple"] for r in resolved if r.get("r_multiple") is not None]
        overall_expectancy = sum(r_vals) / len(r_vals) if r_vals else 0.0
        print(f"Win rate:                 {len(wins) / len(resolved) * 100:.1f}%")
        print(f"Expectancy (avg R):       {overall_expectancy:+.2f} per resolved trade")

    print(f"\nBy symbol:")
    for symbol in pairs:
        sym_resolved = [r for r in resolved if r["symbol"] == symbol]
        sym_wins     = [r for r in sym_resolved if r["outcome"] == "hit_tp"]
        wr = (len(sym_wins) / len(sym_resolved) * 100) if sym_resolved else 0.0
        print(f"  {symbol:<10} signals={len([r for r in all_results if r['symbol']==symbol]):<4} "
              f"resolved={len(sym_resolved):<4} win_rate={wr:.1f}%")

    if resolved:
        print(f"\nBy direction:")
        _bucket_stats(resolved, lambda r: r.get("direction"))

        print(f"\nBy confluence score (higher should mean more reliable, if the model is well-calibrated):")
        _bucket_stats(resolved, lambda r: r.get("confluence_score"))

        print(f"\nBy predicted probability bucket (checks calibration — 70-80% bucket should win ~70-80% of the time):")
        def _prob_bucket(r):
            p = r.get("probability")
            if p is None:
                return None
            p = float(p)
            lo = int(p // 10) * 10
            return f"{lo}-{lo+10}%"
        _bucket_stats(resolved, _prob_bucket)

    losses = [r for r in resolved if r["outcome"] == "hit_sl"]
    if losses:
        print(f"\nLOSS DIAGNOSTICS ({len(losses)} losing trades) — is this an entry/SL/TP")
        print("placement problem, or clean directional moves against the trade?")
        overflows = [r["sl_overflow_pips"] for r in losses if r.get("sl_overflow_pips") is not None]
        slips     = [r["entry_slippage_pips"] for r in losses if r.get("entry_slippage_pips") is not None]
        ties      = [r for r in losses if r.get("same_bar_tie")]
        if overflows:
            print(f"  SL overflow (pips past SL when hit):  avg={sum(overflows)/len(overflows):.1f}  "
                  f"min={min(overflows):.1f}  max={max(overflows):.1f}")
            small_overflow = [o for o in overflows if o <= 3.0]
            print(f"    → {len(small_overflow)}/{len(overflows)} losses overflowed SL by ≤3 pips — "
                  f"a marginal wick, suggests SL may be sitting too tight for this pair's volatility "
                  f"rather than a genuine reversal against the trade.")
        if slips:
            print(f"  Entry slippage (pips from proposed entry to actual trigger):  "
                  f"avg={sum(slips)/len(slips):.1f}  max={max(slips):.1f}")
        print(f"  Same-bar TP/SL ties (ambiguous — SL assumed by pessimistic rule): "
              f"{len(ties)}/{len(losses)} losses ({len(ties)/len(losses)*100:.0f}%) — "
              f"these could actually have been wins on live tick data.")

        traced = [r for r in losses if r.get("post_sl_bars_checked", 0) > 0]
        swept  = [r for r in traced if r.get("post_sl_reached_tp") and r.get("post_sl_reclaimed_entry")]
        reclaimed_only = [r for r in traced if r.get("post_sl_reclaimed_entry") and not r.get("post_sl_reached_tp")]
        genuine = [r for r in traced if not r.get("post_sl_reclaimed_entry")]
        if traced:
            print(f"  Post-SL trace ({len(traced)}/{len(losses)} losses had enough forward data to check):")
            print(f"    → Likely liquidity sweeps (reclaimed entry, then ran to original TP): "
                  f"{len(swept)}/{len(traced)} ({len(swept)/len(traced)*100:.0f}%)")
            print(f"    → Reclaimed entry but never reached TP (partial bounce):              "
                  f"{len(reclaimed_only)}/{len(traced)} ({len(reclaimed_only)/len(traced)*100:.0f}%)")
            print(f"    → Never reclaimed entry (genuine move against the trade):              "
                  f"{len(genuine)}/{len(traced)} ({len(genuine)/len(traced)*100:.0f}%)")
        print(f"\n  By symbol:")
        for symbol in pairs:
            sym_losses = [r for r in losses if r["symbol"] == symbol]
            if not sym_losses:
                continue
            sym_overflows = [r["sl_overflow_pips"] for r in sym_losses if r.get("sl_overflow_pips") is not None]
            sym_ties = [r for r in sym_losses if r.get("same_bar_tie")]
            avg_of = sum(sym_overflows) / len(sym_overflows) if sym_overflows else 0.0
            print(f"    {symbol:<10} losses={len(sym_losses):<3} avg_sl_overflow={avg_of:>5.1f}p  "
                  f"same_bar_ties={len(sym_ties)}/{len(sym_losses)}  pip_size={sym_losses[0].get('pip_size')}")

    # ── Approach & Entry Dynamics Summary ──────────────────────────────
    if resolved:
        print(f"\n{'─'*70}\nAPPROACH & ENTRY DYNAMICS (30-min window before entry touch)\n{'─'*70}")
        wins_app = [r for r in resolved if r["outcome"] == "hit_tp" and "approach_velocity_30m_atr" in r]
        losses_app = [r for r in resolved if r["outcome"] == "hit_sl" and "approach_velocity_30m_atr" in r]
        sweeps_app = [r for r in losses_app if r.get("post_sl_reached_tp") and r.get("post_sl_reclaimed_entry")]
        genuine_losses_app = [r for r in losses_app if not r.get("post_sl_reclaimed_entry")]

        def _avg(lst, key):
            vals = [x[key] for x in lst if x.get(key) is not None]
            return sum(vals)/len(vals) if vals else 0.0

        print(f"  🟢 Winning Trades (TP, n={len(wins_app)}):")
        print(f"     ↳ Avg 30m Velocity:     {_avg(wins_app, 'approach_velocity_30m_atr'):>5.2f}x ATR  (entry slippage: {_avg(wins_app, 'entry_slippage_pips'):>5.1f}p)")
        print(f"     ↳ Avg Pre-Bar Size:     {_avg(wins_app, 'pre_touch_bar_size_atr'):>5.2f}x ATR  (body ratio: {_avg(wins_app, 'pre_touch_body_ratio'):>5.2f})")
        print(f"     ↳ Avg Rejection Wick:   {_avg(wins_app, 'touch_rejection_wick_atr'):>5.2f}x ATR  (time to touch: {_avg(wins_app, 'time_to_trigger_hours'):>5.1f}h)")

        if sweeps_app:
            print(f"  🟡 Post-SL Liquidity Sweeps (Hit SL then reached TP, n={len(sweeps_app)}):")
            print(f"     ↳ Avg 30m Velocity:     {_avg(sweeps_app, 'approach_velocity_30m_atr'):>5.2f}x ATR  (entry slippage: {_avg(sweeps_app, 'entry_slippage_pips'):>5.1f}p)")
            print(f"     ↳ Avg Pre-Bar Size:     {_avg(sweeps_app, 'pre_touch_bar_size_atr'):>5.2f}x ATR  (body ratio: {_avg(sweeps_app, 'pre_touch_body_ratio'):>5.2f})")
            print(f"     ↳ Avg Rejection Wick:   {_avg(sweeps_app, 'touch_rejection_wick_atr'):>5.2f}x ATR  (sl overflow: {_avg(sweeps_app, 'sl_overflow_pips'):>5.1f}p)")

        if genuine_losses_app:
            print(f"  🔴 Genuine Trend Break Losses (n={len(genuine_losses_app)}):")
            print(f"     ↳ Avg 30m Velocity:     {_avg(genuine_losses_app, 'approach_velocity_30m_atr'):>5.2f}x ATR  (entry slippage: {_avg(genuine_losses_app, 'entry_slippage_pips'):>5.1f}p)")
            print(f"     ↳ Avg Pre-Bar Size:     {_avg(genuine_losses_app, 'pre_touch_bar_size_atr'):>5.2f}x ATR  (body ratio: {_avg(genuine_losses_app, 'pre_touch_body_ratio'):>5.2f})")
            print(f"     ↳ Avg Rejection Wick:   {_avg(genuine_losses_app, 'touch_rejection_wick_atr'):>5.2f}x ATR  (sl overflow: {_avg(genuine_losses_app, 'sl_overflow_pips'):>5.1f}p)")

    # ── POST-TP EXTENSION SUMMARY ──────────────────────────────────────
    tp_trades = [r for r in resolved if r["outcome"] == "hit_tp" and "post_tp_max_extension_pips" in r]
    if tp_trades:
        print(f"\n{'─'*70}\nPOST-TP EXTENSION ANALYSIS (did price keep going after TP?)\n{'─'*70}")
        ext_vals = [r["post_tp_max_extension_pips"] for r in tp_trades if r.get("post_tp_max_extension_pips") is not None]
        continued = [r for r in tp_trades if r.get("post_tp_continued")]
        if ext_vals:
            print(f"  📊 Average extension past TP: {sum(ext_vals)/len(ext_vals):.1f} pips  (max: {max(ext_vals):.1f} pips)")
            print(f"  🏷️  Trades that continued >3 pips past TP: {len(continued)}/{len(tp_trades)} "
                  f"({len(continued)/len(tp_trades)*100:.1f}%)")
            if len(continued) / len(tp_trades) > 0.5:
                print(f"     ⚠️  OVER 50% of wins continued significantly past TP — you're exiting TOO EARLY!")
                print(f"     → Consider increasing TP target or implementing a trailing stop.")
            elif len(continued) / len(tp_trades) > 0.3:
                print(f"     ⚠️  ~{len(continued)/len(tp_trades)*100:.0f}% of wins continued past TP — "
                      f"consider modest TP increases.")
            else:
                print(f"     ✅ Most wins exited near the peak — TP placement looks reasonable.")
        
        # By symbol
        print(f"\n  By symbol:")
        for symbol in pairs:
            sym_tp = [r for r in tp_trades if r["symbol"] == symbol]
            if not sym_tp:
                continue
            sym_ext = [r["post_tp_max_extension_pips"] for r in sym_tp if r.get("post_tp_max_extension_pips") is not None]
            sym_cont = [r for r in sym_tp if r.get("post_tp_continued")]
            if sym_ext:
                print(f"    {symbol:<10} avg_extension={sum(sym_ext)/len(sym_ext):>5.1f}p  "
                      f"continued={len(sym_cont)}/{len(sym_tp)} ({len(sym_cont)/len(sym_tp)*100:.0f}%)")

    # ── Milestone TP1/TP2 tracking summary (Method 1) ──────────────────
    milestone_trades = [r for r in tp_trades if r.get("milestone")]
    if milestone_trades:
        print(f"\n{'─'*70}\nMILESTONE TP1/TP2 TRACKING (Method 1)\n{'─'*70}")
        m_summary = summarize_milestones(milestone_trades)
        print(f"  TP1 hits evaluated:        {m_summary['total_tp1_hits']}")
        print(f"  Banked at TP1 (no ext.):   {m_summary['banked_at_tp1']}")
        print(f"  Extended -> hit TP2:       {m_summary['extended_and_hit_tp2']}")
        print(f"  Extended -> gave back to SL: {m_summary['extended_then_gave_back_to_sl']}")
        print(f"  Extended -> still open:    {m_summary['extended_still_open']}")
        print(f"  Extension trigger rate:    {m_summary['extension_rate_pct']}%")
        if m_summary['tp2_success_rate_pct'] is not None:
            print(f"  TP2 success rate (of extended trades): {m_summary['tp2_success_rate_pct']}%")

        print(f"\n  By symbol:")
        for symbol in pairs:
            sym_m = [r["milestone"] for r in milestone_trades if r["symbol"] == symbol]
            if not sym_m:
                continue
            s = summarize_milestones(sym_m)
            print(f"    {symbol:<10} tp1={s['total_tp1_hits']:<3} banked={s['banked_at_tp1']:<3} "
                  f"tp2_hit={s['extended_and_hit_tp2']:<3} gave_back={s['extended_then_gave_back_to_sl']:<3} "
                  f"tp2_rate={s['tp2_success_rate_pct']}%")

        print(f"\n  Per-trade detail:")
        for r in milestone_trades:
            m = r["milestone"]
            print(f"    {r['symbol']:<8} {r['direction']:<4} proposed={r.get('proposed_at')} "
                  f"tp1={m['tp1']:.5f} tp1_achieved={m['tp1_achieved']} "
                  f"stage_reached={m['stage_reached']} "
                  f"tp2={m['tp2'] if m['tp2'] else '-'} tp2_achieved={m['tp2_achieved']} "
                  f"outcome={m['final_outcome']}")

    # ── Stage-3 trigger decision summary — did the exhaustion-score recompute actually help? ──
    never_filled = [r for r in all_results if r["outcome"] == "adjusted_never_filled"]
    adj_mid   = [r for r in resolved if r.get("action") == "adjust_mid"]
    adj_high  = [r for r in resolved if r.get("action") == "adjust_high"]
    unchanged = [r for r in resolved if not r.get("adjusted")]
    if never_filled or adj_mid or adj_high:
        print(f"\n{'─'*70}\nSTAGE-3 TRIGGER DECISION (exhaustion-score recompute)\n{'─'*70}")
        if never_filled:
            nf_syms = {}
            for r in never_filled:
                nf_syms[r["symbol"]] = nf_syms.get(r["symbol"], 0) + 1
            # Fill window is per-symbol now (ADJUST_FILL_WINDOW_OVERRIDES), so a
            # single flat hours figure here would be wrong/misleading whenever
            # never_filled spans more than one symbol -- show each symbol's
            # actual window instead of the ADJUST_FILL_WINDOW_HOURS default.
            print(f"  Adjusted but never filled: {len(never_filled)} signals, zero risk taken")
            nf_detail = {sym: f"{count} (window={get_fill_window_hours(sym):.1f}h)"
                         for sym, count in nf_syms.items()}
            print(f"    By symbol: {nf_detail}")

        def _tier_stats(label, rows):
            if not rows:
                return
            wins = [r for r in rows if r["outcome"] == "hit_tp"]
            wr = len(wins) / len(rows) * 100
            tot_r = sum(r.get("r_multiple") or 0 for r in rows)
            print(f"  {label}: {len(rows)} signals — win_rate={wr:.1f}%  total_R={tot_r:+.2f}")

        _tier_stats("Adjust-mid  (score in [EXHAUSTION_MID_THRESHOLD, EXHAUSTION_HIGH_THRESHOLD))", adj_mid)
        _tier_stats("Adjust-high (score >= EXHAUSTION_HIGH_THRESHOLD)", adj_high)
        if unchanged:
            un_wins = [r for r in unchanged if r["outcome"] == "hit_tp"]
            un_wr = len(un_wins) / len(unchanged) * 100
            un_r  = sum(r.get("r_multiple") or 0 for r in unchanged)
            print(f"  Unchanged (score below mid threshold): {len(unchanged)} signals — "
                  f"win_rate={un_wr:.1f}%  total_R={un_r:+.2f}")
        print(f"  → Compare tiers vs unchanged win_rate/total_R to see whether the recompute is "
              f"actually earning its keep, and whether adjust_high needs different constants than "
              f"adjust_mid. Still thin data — retune EXHAUSTION_* / SL_PAD_* / ENTRY_PUSH_VEL_ATR "
              f"constants at the top of compute_trigger_decision as more runs accumulate.")

    print(f"\nFull JSON report:   {report_file}")
    if csv_file:
        print(f"Full CSV report:    {csv_file}")
    print(f"Detailed run log:   {log_path}")
    print(f"Per-signal JSONL:   {jsonl_path}")
    print("\n⚠️  Remember: trigger/outcome logic is a simplified replica (see module")
    print("   docstring). Spot-check a handful of trades against the real chart")
    print("   before trusting the aggregate numbers.")

    close_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward replay of the live Reversion predictor.")
    parser.add_argument("--pairs", type=str, default="EUR/USD,GBP/USD,USD/JPY,XAU/USD")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--finnhub-key", type=str, default=os.getenv("FINNHUB_API_KEY", ""))
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--out-dir", type=str, default="backtest_results")
    parser.add_argument("--sl-trace-days", type=float, default=3.0,
                         help="After a hit_sl, how many days forward to keep watching price "
                              "for a return to the original TP (sweep-vs-genuine-reversal check). "
                              "Default 3.")
    parser.add_argument("--zone-ttl-hours", type=float, default=48.0,
                         help="How long to watch a proposed zone for entry before marking it "
                              "expired_untriggered. Default 48.0 hours (matches 4H institutional "
                              "FVG/OB structure horizon). Applies to every pair unless overridden "
                              "per-symbol with --zone-ttl-overrides.")
    parser.add_argument("--zone-ttl-overrides", type=str, default="",
                         help="Per-symbol TTL overrides on top of --zone-ttl-hours, as "
                              "'SYMBOL=HOURS,SYMBOL=HOURS'. Example: "
                              "--zone-ttl-overrides 'XAU/USD=72' gives Gold a 72h window while "
                              "every other pair still uses --zone-ttl-hours.")
    parser.add_argument("--zone-dedup-days", type=float, default=3.0,
                         help="If the same (direction, price) zone gets re-detected within this "
                              "many days of its last proposal, treat it as a continuation instead "
                              "of logging it as a brand-new signal. Fixes the 31-signals/16-unique-"
                              "zones duplicate-churn issue. Set to 0 to disable dedup entirely.")
    parser.add_argument("--timeframe", "--tf", type=str, default="4h",
                         help="Timeframe(s) to scan: '4h', '1h', or '4h,1h'. Default '4h'.")
    parser.add_argument("--quiet", action="store_true",
                         help="Suppress per-zone detection/scoring logs (faster, less output). "
                              "Default is verbose, matching your live run's log style.")
    parser.add_argument("--no-charts", action="store_true",
                         help="Skip generating per-signal PNG charts (entry/SL/TP/zone overlaid on "
                              "candles). Charts are on by default, saved to <out-dir>/charts/, "
                              "numbered chronologically per symbol so you can review them one by one.")
    parser.add_argument("--no-trigger-decision", action="store_true",
                         help="Disable the stage-3 trigger-time decision (exhaustion-score "
                              "entry/SL/TP recompute + 5min acceleration bonus). On by default. "
                              "Use this to get the old raw-signal-only behavior for an "
                              "apples-to-apples comparison.")
    parser.add_argument("--skip-consec-threshold", type=int, default=2,
                         help="DEPRECATED / no longer used to hard-skip — kept only for CLI "
                              "compatibility. consecutive_directional_bars now feeds the composite "
                              "exhaustion score (see EXHAUSTION_* constants near "
                              "compute_trigger_decision) instead of triggering an outright skip.")
    parser.add_argument("--accel-bar-size-atr-min", type=float, default=2.0,
                         help="5-min 'knife' detection: bar range / 5min ATR must be >= this to "
                              "count as an acceleration bar (adds ACCEL_SCORE_BONUS to the "
                              "exhaustion score). Default 2.0.")
    parser.add_argument("--accel-body-ratio-min", type=float, default=0.70,
                         help="5-min 'knife' detection: bar body / bar range must be >= this "
                              "(solid, low-wick candle) to count as an acceleration bar. Default 0.70.")
    parser.add_argument("--prob-threshold", type=float, default=0.55,
                         help="Minimum ML probability threshold for proposing zones (default 0.55, was "
                              "0.60, before that 0.70). 2026-08-25 replay of the 60% gate showed no "
                              "monotonic relationship between probability bucket and win rate "
                              "(85-100% bucket won at the same rate as 0-65%), so this is a volume "
                              "lever, not yet a validated quality lever -- track the 55-60% cohort's "
                              "win rate separately once you have enough of them to judge.")
    parser.add_argument("--sl-pad-base-atr", type=float, default=SL_PAD_BASE_ATR,
                         help=f"Flat SL-padding coefficient (x ATR) added on top of the observed "
                              f"rejection wick for adjust_mid/adjust_high trades. Default {SL_PAD_BASE_ATR}. "
                              f"Raise this if adjusted trades are consistently overflowing their SL by "
                              f"a small margin (check sl_overflow_pips on hit_sl outcomes with "
                              f"adjusted=True) — this is the lever to test, NOT the per-asset "
                              f"max_sl_pad_pips cap, which rarely binds in practice.")
    parser.add_argument("--sl-pad-wick-atr", type=float, default=SL_PAD_WICK_ATR,
                         help=f"SL-padding coefficient (x ATR) scaled by the touch_rejection_wick_atr "
                              f"of the trigger bar — bigger rejection wick means more padding. "
                              f"Default {SL_PAD_WICK_ATR}.")
    parser.add_argument("--exhaustion-mid-threshold", type=float, default=EXHAUSTION_MID_THRESHOLD,
                         help=f"Exhaustion score (0=clean touch, 1.5+=severe) at or above which a "
                              f"signal gets adjust_mid'd instead of taken unchanged. Default "
                              f"{EXHAUSTION_MID_THRESHOLD}. RAISE THIS to save more signals as "
                              f"'unchanged' -- the 2026-08-25 replay showed unchanged trades winning "
                              f"64.3%% (+13.03R over 14 trades) vs. adjust_mid winning only 25%% "
                              f"(-1.27R over 4 trades), so fewer signals crossing into mid is the "
                              f"single biggest lever for 'saving' signals from a worse-performing tier.")
    parser.add_argument("--exhaustion-high-threshold", type=float, default=EXHAUSTION_HIGH_THRESHOLD,
                         help=f"Exhaustion score at or above which a signal escalates from adjust_mid "
                              f"to adjust_high (harder push/pad). Default {EXHAUSTION_HIGH_THRESHOLD}. "
                              f"Note from the 2026-08-25 replay: adjust_high (33.3%% win, -0.67R over "
                              f"6 trades) actually outperformed adjust_mid per-trade (25%% win, -1.27R "
                              f"over 4) -- so raising this threshold mostly just relabels signals from "
                              f"'high' to the WORSE-performing 'mid' bucket rather than saving them. "
                              f"--exhaustion-mid-threshold is the more effective knob; treat this one "
                              f"as secondary and re-check the tier breakdown after any change.")
    parser.add_argument("--end-date", type=str, default=None,
                         help="End date for replay (e.g. '2026-08-31'). If not specified, defaults to current UTC time.")
    args = parser.parse_args()

    zone_ttl_overrides = {}
    if args.zone_ttl_overrides.strip():
        for pair_str in args.zone_ttl_overrides.split(","):
            pair_str = pair_str.strip()
            if not pair_str:
                continue
            if "=" not in pair_str:
                print(f"⚠️  Ignoring malformed --zone-ttl-overrides entry (expected SYMBOL=HOURS): "
                      f"'{pair_str}'")
                continue
            sym, hours_str = pair_str.split("=", 1)
            try:
                zone_ttl_overrides[sym.strip()] = float(hours_str.strip())
            except ValueError:
                print(f"⚠️  Ignoring --zone-ttl-overrides entry with non-numeric hours: '{pair_str}'")

    run_replay(
        pairs=args.pairs.split(","),
        days_back=args.days,
        finnhub_key=args.finnhub_key,
        use_news=not args.no_news,
        out_dir=Path(args.out_dir),
        debug=not args.quiet,
        sl_trace_days=args.sl_trace_days,
        zone_ttl_hours=args.zone_ttl_hours,
        zone_ttl_overrides=zone_ttl_overrides,
        zone_dedup_days=args.zone_dedup_days,
        make_charts=not args.no_charts,
        apply_trigger_decision=not args.no_trigger_decision,
        skip_consec_threshold=args.skip_consec_threshold,
        accel_bar_size_atr_min=args.accel_bar_size_atr_min,
        accel_body_ratio_min=args.accel_body_ratio_min,
        prob_threshold=args.prob_threshold,
        timeframe=args.timeframe,
        sl_pad_base_atr=args.sl_pad_base_atr,
        sl_pad_wick_atr=args.sl_pad_wick_atr,
        end_date=args.end_date,
    )