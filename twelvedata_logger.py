"""
TWELVE DATA API QUALITY LOGGER
================================
Runs alongside production_predictor.py to monitor data quality issues:
  - Stale bars (last bar timestamp too old)
  - Frozen OHLC (identical bar repeated across cycles)
  - Missing bars (gaps in time series)
  - API errors / credit exhaustion
  - Anomalous price spikes (bar range > N × ATR)

Writes to: logs/twelvedata_quality.log  (rotating, 7-day retention)
Run:  python3 twelvedata_logger.py   (standalone, every 15 min)
Or:   import and call check_all_pairs() from your main cycle.
"""

import requests
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler

# ── Config ────────────────────────────────────────────────────────────────────
KEYS = {
    'EUR/USD': 'ea3e4fce074e403f8cd4c83dd0035643',
    'GBP/USD': '2522b470eba249f88d2e29575035914e',
    'USD/JPY': '5bb48b4211144b92a4487ce7c63d1b7a',
    'XAU/USD': '63a513bb7eb848748d28d3deb64c1b59',
}
BASE_URL    = 'https://api.twelvedata.com'
LOG_DIR     = Path('logs')
LOG_FILE    = LOG_DIR / 'twelvedata_quality.log'
STATE_FILE  = LOG_DIR / 'twelvedata_last_bars.json'  # tracks previous bar per symbol/interval

# How many minutes before a bar is considered stale
STALE_THRESHOLDS = {
    '15min': 20,   # 15min bar should arrive within 20 min
    '1h':    75,
    '4h':    255,
    '1day':  1500,
}

# Spike detection: bar range > SPIKE_ATR_MULT × rolling ATR is suspicious
SPIKE_ATR_MULT = 5.0

INTERVALS = ['15min', '1h', '4h', '1day']

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=7)
_handler.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s'))
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s'))

log = logging.getLogger('twelvedata_quality')
log.setLevel(logging.DEBUG)
log.addHandler(_handler)
log.addHandler(_console)


# ── State helpers ─────────────────────────────────────────────────────────────
def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── API fetch ─────────────────────────────────────────────────────────────────
def _fetch(symbol: str, interval: str, outputsize: int = 10) -> dict | None:
    try:
        r = requests.get(
            f'{BASE_URL}/time_series',
            params={
                'symbol':     symbol,
                'interval':   interval,
                'outputsize': outputsize,
                'apikey':     KEYS[symbol],
                'format':     'JSON',
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if 'values' not in data:
            log.error(f"[{symbol}|{interval}] API error: {data.get('message', data)}")
            return None
        return data
    except Exception as e:
        log.error(f"[{symbol}|{interval}] Request failed: {e}")
        return None


# ── Individual checks ─────────────────────────────────────────────────────────
def check_staleness(symbol: str, interval: str, bars: list) -> bool:
    """Returns True if OK, False if stale."""
    if not bars:
        log.warning(f"[{symbol}|{interval}] No bars returned")
        return False

    last_bar_time_str = bars[0]['datetime']  # newest first
    try:
        last_bar_time = datetime.strptime(last_bar_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            last_bar_time = datetime.strptime(last_bar_time_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            log.error(f"[{symbol}|{interval}] Cannot parse datetime: {last_bar_time_str}")
            return False

    now_utc   = datetime.now(timezone.utc)
    age_mins  = (now_utc - last_bar_time).total_seconds() / 60
    threshold = STALE_THRESHOLDS.get(interval, 30)

    if age_mins > threshold:
        log.warning(
            f"[{symbol}|{interval}] ⚠️  STALE — last bar {last_bar_time_str} "
            f"is {age_mins:.0f}min old (threshold={threshold}min)"
        )
        return False

    log.debug(f"[{symbol}|{interval}] ✅ Fresh — last bar {last_bar_time_str} ({age_mins:.0f}min ago)")
    return True


def check_frozen(symbol: str, interval: str, bars: list, state: dict) -> bool:
    """Returns True if OK, False if frozen (same bar as last cycle)."""
    if not bars:
        return True  # already flagged by staleness

    key     = f"{symbol}|{interval}"
    bar_sig = f"{bars[0]['datetime']}|{bars[0]['close']}"

    if key in state and state[key] == bar_sig:
        log.warning(
            f"[{symbol}|{interval}] ❄️  FROZEN — bar {bars[0]['datetime']} "
            f"close={bars[0]['close']} identical to last check"
        )
        return False

    state[key] = bar_sig
    return True


def check_gaps(symbol: str, interval: str, bars: list) -> bool:
    """Detect unexpected time gaps between consecutive bars."""
    if len(bars) < 3:
        return True

    interval_mins = {'15min': 15, '1h': 60, '4h': 240, '1day': 1440}.get(interval, 15)
    ok = True

    for i in range(len(bars) - 1):
        try:
            t1 = datetime.strptime(bars[i]['datetime'],   '%Y-%m-%d %H:%M:%S')
            t2 = datetime.strptime(bars[i+1]['datetime'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue  # daily bars use date-only format — skip gap check
        gap_mins = (t1 - t2).total_seconds() / 60
        # Allow up to 3× the expected interval (weekends, thin sessions)
        if gap_mins > interval_mins * 3:
            log.warning(
                f"[{symbol}|{interval}] 🕳️  GAP — {gap_mins:.0f}min between "
                f"{bars[i+1]['datetime']} and {bars[i]['datetime']} "
                f"(expected ~{interval_mins}min)"
            )
            ok = False

    return ok


def check_spikes(symbol: str, interval: str, bars: list) -> bool:
    """Flag bars with anomalously large range vs recent ATR."""
    if len(bars) < 6:
        return True

    # Approximate ATR from last 5 bars
    ranges = [abs(float(b['high']) - float(b['low'])) for b in bars[1:6]]
    avg_range = sum(ranges) / len(ranges) if ranges else 0
    if avg_range == 0:
        return True

    latest      = bars[0]
    latest_range = abs(float(latest['high']) - float(latest['low']))
    ok = True

    if latest_range > avg_range * SPIKE_ATR_MULT:
        log.warning(
            f"[{symbol}|{interval}] ⚡ SPIKE — latest bar range "
            f"{latest_range:.5f} is {latest_range/avg_range:.1f}× avg range {avg_range:.5f} "
            f"[{latest['datetime']}  H={latest['high']}  L={latest['low']}]"
        )
        ok = False

    return ok


def check_ohlc_integrity(symbol: str, interval: str, bars: list) -> bool:
    """Check that H >= max(O,C) and L <= min(O,C) for each bar."""
    ok = True
    for b in bars[:5]:  # check latest 5
        o, h, l, c = float(b['open']), float(b['high']), float(b['low']), float(b['close'])
        if h < max(o, c) or l > min(o, c):
            log.error(
                f"[{symbol}|{interval}] 🚨 OHLC INTEGRITY FAIL @ {b['datetime']} "
                f"O={o} H={h} L={l} C={c} — H < max(O,C) or L > min(O,C)"
            )
            ok = False
    return ok


# ── Main check ────────────────────────────────────────────────────────────────
def check_pair(symbol: str, state: dict) -> dict:
    results = {}
    for interval in INTERVALS:
        data = _fetch(symbol, interval, outputsize=10)
        if data is None:
            results[interval] = 'API_ERROR'
            continue

        bars = data['values']
        ok   = True
        ok  &= check_staleness(symbol, interval, bars)
        ok  &= check_frozen(symbol, interval, bars, state)
        ok  &= check_gaps(symbol, interval, bars)
        ok  &= check_spikes(symbol, interval, bars)
        ok  &= check_ohlc_integrity(symbol, interval, bars)
        results[interval] = 'OK' if ok else 'WARN'

    return results


def check_all_pairs() -> dict:
    log.info("=" * 60)
    log.info(f"  TWELVE DATA QUALITY CHECK — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 60)

    state   = _load_state()
    summary = {}

    for symbol in KEYS:
        log.info(f"  Checking {symbol}...")
        summary[symbol] = check_pair(symbol, state)

    _save_state(state)

    # Print summary table
    log.info("\n  SUMMARY")
    log.info(f"  {'Symbol':<10}  {'15min':<8}  {'1h':<8}  {'4h':<8}  {'1day':<8}")
    log.info(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for sym, res in summary.items():
        row = "  ".join(f"{res.get(iv, '?'):<8}" for iv in INTERVALS)
        log.info(f"  {sym:<10}  {row}")

    issues = sum(
        1 for sym_res in summary.values()
        for v in sym_res.values()
        if v != 'OK'
    )
    if issues == 0:
        log.info("\n  ✅ All pairs clean — no data quality issues detected")
    else:
        log.warning(f"\n  ⚠️  {issues} issue(s) detected — see warnings above")

    log.info("=" * 60 + "\n")
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    import argparse

    parser = argparse.ArgumentParser(description='Twelve Data quality monitor')
    parser.add_argument('--once',     action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=15, help='Check interval in minutes (default 15)')
    args = parser.parse_args()

    if args.once:
        check_all_pairs()
    else:
        log.info(f"Starting continuous monitor — checking every {args.interval} min")
        while True:
            check_all_pairs()
            log.info(f"Next check in {args.interval} min...")
            time.sleep(args.interval * 60)