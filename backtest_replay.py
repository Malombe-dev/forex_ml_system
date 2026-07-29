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
   OPEN-TRADE GUARD), so overlapping signals on the same pair are skipped
   until the prior one resolves — same behavior as live.

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
from pathlib import Path
from datetime import datetime, timedelta


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


def fetch_historical_bars(symbol: str, interval: str, start: datetime, end: datetime,
                           api_key: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch full OHLC history for [start, end], chunked + cached to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "")
    cache_file = cache_dir / f"{safe_symbol}_{interval}_{start.date()}_{end.date()}.csv"

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    chunk_days = _CHUNK_DAYS.get(interval, 200)
    all_rows = []
    chunk_end = end
    while chunk_end > start:
        chunk_start = max(start, chunk_end - timedelta(days=chunk_days))
        params = {
            "symbol": symbol, "interval": interval,
            "start_date": chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date":   chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            "apikey": api_key, "format": "JSON",
        }
        print(f"  📡 Fetching {symbol} {interval}: {chunk_start.date()} → {chunk_end.date()}")
        try:
            resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ❌ Fetch failed for {symbol} {interval} chunk: {e}")
            chunk_end = chunk_start - timedelta(seconds=1)
            continue

        if "values" not in data:
            print(f"  ⚠️  No data for {symbol} {interval} chunk "
                  f"({data.get('message', data.get('status', 'unknown error'))})")
            chunk_end = chunk_start - timedelta(seconds=1)
            time.sleep(1)
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
        time.sleep(1)  # be polite to the rate limit across chunks

    if not all_rows:
        print(f"  ⚠️  No historical data retrieved for {symbol} {interval}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.DataFrame(all_rows).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
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
# 4. SIGNAL TRIGGER + OUTCOME SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def simulate_trigger_and_outcome(sig: dict, symbol: str, proposed_at: datetime,
                                  historical: dict, zone_ttl_hours: float, pip_size: float,
                                  max_hold_days: int = 15):
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
        return {"outcome": "expired_untriggered"}

    entry_slippage_pips = abs(trigger_price - entry) / pip_size if pip_size else None

    max_cutoff = trigger_time + timedelta(days=max_hold_days)
    hold = df15[(df15["time"] > trigger_time) & (df15["time"] <= max_cutoff)]

    for _, bar in hold.iterrows():
        hit_sl = (bar["low"] <= sl) if direction == "BUY" else (bar["high"] >= sl)
        hit_tp = (bar["high"] >= tp) if direction == "BUY" else (bar["low"] <= tp)
        if hit_sl:  # conservative: SL wins ties within the same bar
            sl_overflow = (sl - bar["low"]) if direction == "BUY" else (bar["high"] - sl)
            sl_overflow_pips = sl_overflow / pip_size if pip_size else None
            return {
                "outcome": "hit_sl", "trigger_time": trigger_time, "exit_time": bar["time"],
                "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips,
                "sl_overflow_pips": sl_overflow_pips, "same_bar_tie": bool(hit_tp),
            }
        if hit_tp:
            return {
                "outcome": "hit_tp", "trigger_time": trigger_time, "exit_time": bar["time"],
                "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips,
            }

    return {"outcome": "timeout", "trigger_time": trigger_time,
            "trigger_price": trigger_price, "entry_slippage_pips": entry_slippage_pips}


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN REPLAY LOOP
# ═══════════════════════════════════════════════════════════════════════

SYMBOL_PARAMS = {
    "EUR/USD": dict(max_distance_pips_rev=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "GBP/USD": dict(max_distance_pips_rev=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "USD/JPY": dict(max_distance_pips_rev=25.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
    "XAU/USD": dict(max_distance_pips_rev=80.0, zone_distance_threshold=150.0, min_strength_atr=0.40),
}


def build_reversion_predictor(symbol: str, api_key: str, debug: bool = True) -> ProductionForexPredictor:
    """Same construction params as run_dual_strategy()'s `rev` predictor, live."""
    p = SYMBOL_PARAMS[symbol]
    return ProductionForexPredictor(
        api_key=api_key,
        model_path="models/model_v5_reversion.pkl",
        scaler_path="models/scaler_v5_reversion.pkl",
        encoders_path="models/encoders_v5_reversion.pkl",
        stats_path="models/training_stats_v5_reversion.pkl",
        symbol=symbol,
        prob_threshold=0.62,
        max_rr=8.0, min_risk_atr=2.5, max_distance_pips=p["max_distance_pips_rev"],
        min_strength_atr=0.50, use_5min_refinement=True,
        refine_max_dist_multiplier=2.0, enable_continuation=False,
        use_htf_analysis=True, min_confluence_score=1,
        zone_ttl_hours=5.8, zone_distance_threshold=p["zone_distance_threshold"],
        htf_fvg_min_gap_atr=0.5, htf_ob_min_reaction_atr=2.5,
        is_continuation_strategy=False, require_zone_close=True, debug=debug,
    )


def run_replay(pairs: list, days_back: int, finnhub_key: str, use_news: bool, out_dir: Path, debug: bool = True):
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    warmup_start = start - timedelta(days=45)  # buffer for EMA200/ATR warmup on 4H

    cache_dir = out_dir / "bar_cache"

    run_tag = end.strftime("%Y-%m-%d_%H%M%S")
    log_path, jsonl_path, close_logging = setup_detailed_logging(out_dir, run_tag)

    print(f"\n{'═'*70}\nREPLAY WINDOW: {start.date()} → {end.date()}  ({days_back} days)\n{'═'*70}")
    print(f"📝 Detailed log:   {log_path}")
    print(f"📝 Per-signal log: {jsonl_path}  (JSON-lines, one record per proposed signal)")
    print(f"Run params: pairs={pairs} days_back={days_back} use_news={use_news} debug={debug}")

    news_events = []
    if use_news:
        print("\n📰 Fetching historical high-impact news...")
        news_events = fetch_historical_high_impact_events(start, end, finnhub_key)

    all_results = []

    for symbol in pairs:
        print(f"\n{'─'*70}\n{symbol}\n{'─'*70}")
        api_key = PAIR_API_KEYS.get(symbol, list(PAIR_API_KEYS.values())[0])

        historical = {}
        for interval in ["4h", "1h", "15min", "5min", "1day"]:
            historical[(symbol, interval)] = fetch_historical_bars(
                symbol, interval, warmup_start, end, api_key, cache_dir
            )

        pred = build_reversion_predictor(symbol, api_key, debug=debug)
        clock = ReplayClock()
        pred.fetch_bars = make_replay_fetch_bars(symbol, historical, clock, pred)
        pip_size = pred.pip_size
        print(f"  📏 [{symbol}] pip_size resolved to {pip_size} "
              f"(0.0001=standard FX, 0.01=JPY pair, 1.0=XAU/gold — confirms this pair "
              f"isn't silently using the wrong scale)")

        four_h = historical[(symbol, "4h")]
        walk_points = four_h[(four_h["time"] >= start) & (four_h["time"] <= end)]["time"].tolist()
        n_points = len(walk_points)
        print(f"  🚶 Walking {n_points} historical 4H candle closes for {symbol}...")
        loop_start_ts = time.time()

        # Locks out re-proposing while a zone is either still being watched
        # for entry (untriggered) or an open trade is running — mirrors
        # live's one-trade-per-symbol OPEN-TRADE GUARD, and also fixes the
        # bug where the same persisting unfilled zone got re-counted as a
        # brand-new signal on every 4H candle it continued to exist.
        symbol_locked_until = None
        currencies = [c.upper() for c in symbol.replace("/", "_").split("_")]

        for i, t in enumerate(walk_points, 1):
            if i == 1 or i % 20 == 0 or i == n_points:
                elapsed = time.time() - loop_start_ts
                rate = elapsed / i if i > 0 else 0
                eta = rate * (n_points - i)
                print(f"     [{i}/{n_points}] {t}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                      f"{len(all_results)} signals so far)")

            clock.now = t

            if symbol_locked_until and t < symbol_locked_until:
                continue

            if use_news and news_events:
                active, ev = is_news_active_at(t, currencies, news_events)
                if active:
                    continue

            try:
                htf_context = pred.get_htf_context()
                raw = pred.htf_trade_detect("4h", htf_context=htf_context)
            except Exception as e:
                print(f"  ⚠️  {symbol} @ {t}: detection error — {e}")
                continue

            signals = raw.get("htf_signals", raw) if isinstance(raw, dict) else raw
            if not signals:
                continue

            sig = signals[0]  # highest-probability signal, mirrors live's top pick
            zone_ttl_hours = getattr(pred.zone_tracker, 'ttl_seconds', 5.8 * 3600) / 3600.0
            print(f"  📌 [{symbol}] Signal proposed @ {t}: {sig.get('direction')} "
                  f"entry={sig.get('entry')} sl={sig.get('sl')} tp={sig.get('tp')} "
                  f"prob={sig.get('probability')}")
            # Full field dump of what actually produced this signal (zone bounds,
            # confluence breakdown, HTF confluence tags, etc.) — console stays
            # readable, this goes to the log file for later inspection.
            print(f"     ↳ full signal fields: {json.dumps(sig, default=str)}")

            result = simulate_trigger_and_outcome(sig, symbol, t, historical,
                                                   zone_ttl_hours=zone_ttl_hours, pip_size=pip_size)

            entry, sl, tp = sig.get("entry"), sig.get("sl"), sig.get("tp")
            # Pip-normalized risk/reward — divide by THIS symbol's pip_size so
            # EUR/USD, USD/JPY, and XAU/USD numbers are directly comparable
            # instead of mixing raw price-difference scales.
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
                "probability": sig.get("probability"), "risk_reward": sig.get("risk_reward"),
                "confluence_score": sig.get("confluence_score", sig.get("confluence")),
                "risk_pips": risk_pips, "reward_pips": reward_pips, "r_multiple": r_multiple,
                "zone_top": sig.get("zone_top"), "zone_bottom": sig.get("zone_bottom"),
                "timeframe": sig.get("timeframe"), "pip_size": pip_size,
            })
            all_results.append(result)
            log_signal_jsonl(jsonl_path, {**result, "raw_signal": sig})

            outcome = result["outcome"]
            if outcome == "expired_untriggered":
                print(f"     ⌛ Never triggered — zone expired after {zone_ttl_hours:.1f}h watch window.")
                symbol_locked_until = t + timedelta(hours=zone_ttl_hours)
            elif outcome in ("hit_tp", "hit_sl"):
                emoji = "✅" if outcome == "hit_tp" else "❌"
                print(f"     {emoji} Triggered {result.get('trigger_time')} → {outcome.upper()} "
                      f"at {result.get('exit_time')}  (R={r_multiple})")
                if outcome == "hit_sl":
                    # Loss diagnostic: is this an entry/SL/TP placement problem
                    # (small overflow / tie / big slippage) or a clean directional
                    # move against the trade (large overflow, no tie)?
                    overflow = result.get("sl_overflow_pips")
                    tie      = result.get("same_bar_tie")
                    slip     = result.get("entry_slippage_pips")
                    tie_note = "⚠️ SAME-BAR TIE (also touched TP — ambiguous, SL assumed by pessimistic tie-break)" if tie else "clean SL (TP not touched same bar)"
                    print(f"        ↳ LOSS DIAGNOSTIC: sl_overflow={overflow:.1f} pips past SL | "
                          f"entry_slippage={slip:.1f} pips from proposed entry | {tie_note}")
                symbol_locked_until = result.get("exit_time", t + timedelta(days=15))
            elif outcome == "timeout":
                print(f"     ⏱️  Triggered {result.get('trigger_time')} → ran past max hold "
                      f"without hitting TP/SL (timeout).")
                symbol_locked_until = t + timedelta(days=15)
            else:
                print(f"     ⚠️  outcome={outcome}")
                symbol_locked_until = t + timedelta(hours=zone_ttl_hours)

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
    timed_out   = [r for r in all_results if r["outcome"] == "timeout"]

    print(f"Total proposed signals:  {len(all_results)}")
    print(f"Resolved (TP or SL):     {len(resolved)}")
    print(f"Never triggered:         {len(untriggered)}")
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
    parser.add_argument("--quiet", action="store_true",
                         help="Suppress per-zone detection/scoring logs (faster, less output). "
                              "Default is verbose, matching your live run's log style.")
    args = parser.parse_args()

    run_replay(
        pairs=args.pairs.split(","),
        days_back=args.days,
        finnhub_key=args.finnhub_key,
        use_news=not args.no_news,
        out_dir=Path(args.out_dir),
        debug=not args.quiet,
    )