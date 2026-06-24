import pandas as pd
import numpy as np
import requests
import pickle
import logging
import os
import threading
import schedule
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from feature_builder import build_feature_row
from features import get_session as _canonical_get_session
from outcome_tracker import SignalOutcomeTrackerV2

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# SHARED BAR CACHE — 60-second TTL, keyed by (symbol, interval, outputsize).
# TTL is intentionally short (60 s) so the cache never goes stale across
# cycles.  Every new cycle the REV predictor always re-fetches fresh bars;
# the cache just prevents the duplicate within the same cycle.
_SHARED_BAR_CACHE: Dict[tuple, Dict] = {}
_SHARED_BAR_CACHE_LOCK = threading.Lock()
_SHARED_BAR_CACHE_TTL  = 60.0   # seconds — long enough for REV→CONT within one cycle

class _RateLimiter:
    def __init__(self, max_calls: int = 7, window_seconds: float = 62.0):
        self._max      = max_calls
        self._window   = window_seconds
        self._calls    = []
        self._lock     = threading.Lock()

    def acquire(self):
        """Block until a call slot is available within the rolling window."""
        while True:
            with self._lock:
                now     = time.time()
                cutoff  = now - self._window
                self._calls = [t for t in self._calls if t > cutoff]
                if len(self._calls) < self._max:
                    self._calls.append(now)
                    return
                wait_until = self._calls[0] + self._window
            sleep_secs = max(0.1, wait_until - time.time())
            print(f"  ⏳ Rate limit: waiting {sleep_secs:.1f}s for next API slot "
                  f"({len(self._calls)}/{self._max} calls in window)...")
            time.sleep(sleep_secs)

    def try_acquire(self) -> bool:
        """Non-blocking: take a slot now if one is free, else return False
        immediately without sleeping. Used for fallback-key probing."""
        with self._lock:
            now    = time.time()
            cutoff = now - self._window
            self._calls = [t for t in self._calls if t > cutoff]
            if len(self._calls) < self._max:
                self._calls.append(now)
                return True
            return False

    def next_free_in(self) -> float:
        """Seconds until this limiter will next have a free slot (0 if free now)."""
        with self._lock:
            now    = time.time()
            cutoff = now - self._window
            self._calls = [t for t in self._calls if t > cutoff]
            if len(self._calls) < self._max:
                return 0.0
            return max(0.0, (self._calls[0] + self._window) - now)

# Per-API-key rate limiters — each key gets its own 7-call/min budget.
# If you use two keys (one for REV, one for CONT), each gets independent
# throttling so combined throughput is 14 real calls/min instead of 7.
_API_RATE_LIMITERS: Dict[str, "_RateLimiter"] = {}
_API_RATE_LIMITERS_LOCK = threading.Lock()

def _get_rate_limiter(api_key: str) -> "_RateLimiter":
    """Return (or create) the rate limiter for a given API key."""
    with _API_RATE_LIMITERS_LOCK:
        if api_key not in _API_RATE_LIMITERS:
            _API_RATE_LIMITERS[api_key] = _RateLimiter(max_calls=7, window_seconds=62.0)
        return _API_RATE_LIMITERS[api_key]

# ---------------------------------------------------------------------------
# DYNAMIC FALLBACK POOL
# ---------------------------------------------------------------------------
# A shared pool of "spare" API keys any symbol can borrow from when its own
# primary key's 7-call/min budget is full. Populated at startup in
# run_dual_strategy() from TWELVE_KEY_SPARE_1, TWELVE_KEY_SPARE_2, etc.
_SPARE_KEY_POOL: List[str] = []

def _register_spare_keys(keys: List[str]):
    """Call once at startup with the list of unused/spare API keys."""
    for k in keys:
        if k and k not in _SPARE_KEY_POOL:
            _SPARE_KEY_POOL.append(k)

def acquire_any(primary_key: str) -> str:
    """
    Acquire a rate-limit slot for a call, trying the primary key first and
    falling back to any free spare key if the primary is currently full.
    Returns the API key whose slot was actually taken — callers MUST use
    this returned key for the actual HTTP request (not the original
    primary_key), since a different account's key may have been used.
    """
    primary_limiter = _get_rate_limiter(primary_key)

    # Fast path — primary key has room right now.
    if primary_limiter.try_acquire():
        return primary_key

    # Primary is full — try every registered spare key, in order.
    for spare_key in _SPARE_KEY_POOL:
        if spare_key == primary_key:
            continue
        spare_limiter = _get_rate_limiter(spare_key)
        if spare_limiter.try_acquire():
            print(f"  🔁 Rate limit: primary key full — borrowed spare key "
                  f"...{spare_key[-6:]} instead")
            return spare_key

    # Everyone is full — block on whichever (primary or spare) frees up soonest.
    candidates = [primary_key] + [k for k in _SPARE_KEY_POOL if k != primary_key]
    waits = [(_get_rate_limiter(k).next_free_in(), k) for k in candidates]
    waits.sort(key=lambda pair: pair[0])
    soonest_wait, soonest_key = waits[0]
    print(f"  ⏳ Rate limit: all keys full — waiting {soonest_wait:.1f}s "
          f"on key ...{soonest_key[-6:]}...")
    _get_rate_limiter(soonest_key).acquire()
    return soonest_key



class HTFInstitutionalAnalyzer:
    """Analyzes higher timeframes for institutional footprints"""

    def __init__(self, pip_size: float = 0.0001,
                 fvg_min_gap_atr: float = 0.4,
                 ob_min_reaction_atr: float = 1.8):
        self.pip_size = pip_size
        self.fvg_min_gap_atr = fvg_min_gap_atr
        self.ob_min_reaction_atr = ob_min_reaction_atr

    def _recent_extreme(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        recent = df.tail(lookback)
        return {'swing_high': recent['high'].max(),
                'swing_low':  recent['low'].min()}

    def detect_fvg(self, df: pd.DataFrame, timeframe: str = '4h') -> List[Dict]:
        fvgs = []
        current_price = df['close'].iloc[-1]
        min_gap_atr = self.fvg_min_gap_atr

        for i in range(2, len(df) - 3):
            candle  = df.iloc[i]
            prev_c  = df.iloc[i - 1]
            next_c  = df.iloc[i + 1]
            atr     = df['atr'].iloc[i]
            if pd.isna(atr) or atr <= 0:
                continue

            # Bullish FVG: prev_high < next_low
            if prev_c['high'] < next_c['low']:
                gap_size = next_c['low'] - prev_c['high']
                if gap_size < atr * min_gap_atr:
                    continue
                future_bars = df.iloc[i + 1:]
                if (future_bars['low'] <= prev_c['high']).any():
                    continue
                if candle['close'] <= candle['open']:
                    continue
                top    = next_c['low']
                bottom = prev_c['high']
                mid    = (top + bottom) / 2
                fvgs.append({
                    'type': 'bullish', 'index': i, 'time': candle['time'],
                    'top': top, 'bottom': bottom, 'mid': mid,
                    'size': gap_size, 'size_atr': gap_size / atr,
                    'distance_pips': abs(current_price - mid) / self.pip_size,
                    'mitigated': False, 'strength': gap_size / atr
                })

            # Bearish FVG: prev_low > next_high
            elif prev_c['low'] > next_c['high']:
                gap_size = prev_c['low'] - next_c['high']
                if gap_size < atr * min_gap_atr:
                    continue
                future_bars = df.iloc[i + 1:]
                if (future_bars['high'] >= prev_c['low']).any():
                    continue
                if candle['close'] >= candle['open']:
                    continue
                top    = prev_c['low']
                bottom = next_c['high']
                mid    = (top + bottom) / 2
                fvgs.append({
                    'type': 'bearish', 'index': i, 'time': candle['time'],
                    'top': top, 'bottom': bottom, 'mid': mid,
                    'size': gap_size, 'size_atr': gap_size / atr,
                    'distance_pips': abs(current_price - mid) / self.pip_size,
                    'mitigated': False, 'strength': gap_size / atr
                })
        return fvgs

    def detect_order_blocks(self, df: pd.DataFrame, timeframe: str = '4h') -> List[Dict]:
        obs = []
        current_price = df['close'].iloc[-1]
        min_reaction_atr = self.ob_min_reaction_atr

        for i in range(5, len(df) - 6):
            candle    = df.iloc[i]
            atr       = df['atr'].iloc[i]
            if pd.isna(atr) or atr <= 0:
                continue
            next_5    = df.iloc[i + 1:i + 6]

            # Bullish OB
            if candle['close'] < candle['open']:
                impulse = next_5['high'].max() - candle['low']
                if impulse >= atr * min_reaction_atr:
                    future = df.iloc[i + 1:]
                    if not (future['close'] < candle['low']).any():
                        obs.append({
                            'type': 'bullish', 'index': i, 'time': candle['time'],
                            'top': candle['high'], 'bottom': candle['low'],
                            'mid': (candle['high'] + candle['low']) / 2,
                            'reaction_size': impulse, 'reaction_atr': impulse / atr,
                            'distance_pips': abs(current_price - ((candle['high'] + candle['low']) / 2)) / self.pip_size,
                            'violated': False, 'strength': impulse / atr
                        })

            # Bearish OB
            elif candle['close'] > candle['open']:
                impulse = candle['high'] - next_5['low'].min()
                if impulse >= atr * min_reaction_atr:
                    future = df.iloc[i + 1:]
                    if not (future['close'] > candle['high']).any():
                        obs.append({
                            'type': 'bearish', 'index': i, 'time': candle['time'],
                            'top': candle['high'], 'bottom': candle['low'],
                            'mid': (candle['high'] + candle['low']) / 2,
                            'reaction_size': impulse, 'reaction_atr': impulse / atr,
                            'distance_pips': abs(((candle['high'] + candle['low']) / 2) - current_price) / self.pip_size,
                            'violated': False, 'strength': impulse / atr
                        })
        return obs

    def detect_market_structure_shift(self, df: pd.DataFrame) -> Dict:
        swing_highs, swing_lows = [], []
        for i in range(5, len(df) - 5):
            bar = df.iloc[i]
            if bar['high'] > df.iloc[i-5:i]['high'].max() and bar['high'] > df.iloc[i+1:i+6]['high'].max():
                swing_highs.append({'index': i, 'price': bar['high'], 'time': bar['time']})
            if bar['low'] < df.iloc[i-5:i]['low'].min() and bar['low'] < df.iloc[i+1:i+6]['low'].min():
                swing_lows.append({'index': i, 'price': bar['low'], 'time': bar['time']})

        current_price = df['close'].iloc[-1]
        recent_highs  = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows   = swing_lows[-3:]  if len(swing_lows)  >= 3 else swing_lows

        trend, bos_detected, mss_detected = 'neutral', False, False
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[-1]['price'] > recent_highs[-2]['price']
            hl = recent_lows[-1]['price']  > recent_lows[-2]['price']
            lh = recent_highs[-1]['price'] < recent_highs[-2]['price']
            ll = recent_lows[-1]['price']  < recent_lows[-2]['price']
            if hh and hl:
                trend = 'bullish'
                if current_price > recent_highs[-1]['price']:
                    bos_detected = True
            if lh and ll:
                trend = 'bearish'
                if current_price < recent_lows[-1]['price']:
                    bos_detected = True
            if trend == 'bullish' and current_price < recent_lows[-2]['price']:
                mss_detected = True
            elif trend == 'bearish' and current_price > recent_highs[-2]['price']:
                mss_detected = True

        return {
            'trend': trend, 'bos_detected': bos_detected, 'mss_detected': mss_detected,
            'swing_highs': swing_highs[-3:], 'swing_lows': swing_lows[-3:],
            'last_swing_high': swing_highs[-1] if swing_highs else None,
            'last_swing_low':  swing_lows[-1]  if swing_lows  else None
        }

    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        recent     = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low  = recent['low'].min()
        diff       = swing_high - swing_low
        return {
            '0.0': swing_high, '0.236': swing_high - diff * 0.236,
            '0.382': swing_high - diff * 0.382, '0.5': swing_high - diff * 0.5,
            '0.618': swing_high - diff * 0.618, '0.786': swing_high - diff * 0.786,
            '1.0': swing_low, 'swing_high': swing_high, 'swing_low': swing_low
        }

    def check_confluence(self, zone: Dict, fvgs: List[Dict], order_blocks: List[Dict],
                         fib_levels: Dict, current_price: float) -> Dict:
        score, factors = 0, []
        zone_top    = zone.get('price_high', zone.get('price_mid', current_price))
        zone_bottom = zone.get('price_low',  zone.get('price_mid', current_price))
        zone_mid    = zone.get('price_mid',  current_price)

        for fvg in fvgs:
            if self._ranges_overlap(zone_bottom, zone_top, fvg['bottom'], fvg['top']):
                score += 3; factors.append(f"FVG_{fvg['type']}"); break

        for ob in order_blocks:
            if self._ranges_overlap(zone_bottom, zone_top, ob['bottom'], ob['top']):
                score += 2; factors.append(f"OB_{ob['type']}"); break

        fib_hit = False
        # tolerance = 10 * 0.0001
        tolerance = 10 * self.pip_size
        for level_name, level_price in fib_levels.items():
            if level_name in ['swing_high', 'swing_low']:
                continue
            if abs(zone_mid - level_price) <= tolerance:
                score += 2; factors.append(f"Fib_{level_name}"); fib_hit = True; break

        if score == 0:
            score = 1; factors.append('zone_structure')

        session      = self._get_session(datetime.utcnow().hour)
        distance_pips = abs(zone.get('distance_pips', 999))
        if session in ('london', 'ny') and distance_pips < 25:
            score += 1; factors.append('session_lq')

        return {
            'score': score, 'factors': factors, 'max_score': 10,
            'percentage': (score / 10) * 100,
            'has_fvg': any('FVG' in f for f in factors),
            'has_ob':  any('OB'  in f for f in factors),
            'has_fib': fib_hit
        }

    def _ranges_overlap(self, a_low, a_high, b_low, b_high) -> bool:
        return not (a_high < b_low or b_high < a_low)

    def _get_session(self, hour: int) -> str:
        return _canonical_get_session(hour)



class ActiveZoneTracker:

    def _get_broker_server_offset(self) -> int:
        return 0

    def __init__(self, ttl_hours: float = 5.8,
                 distance_threshold_pips: float = 50,
                 broker_offset: int = 0,
                 state_file: str = "signals/zones_state.json"):
        self.active_zones             = {}
        self.ttl_seconds              = ttl_hours * 3600
        self.distance_threshold_pips  = distance_threshold_pips
        self.pip_size                 = 0.0001  # updated per symbol via set_pip_size()
        self.broker_offset            = self._get_broker_server_offset()
        self.state_file               = Path(state_file)
        self._load_state()

    def _load_state(self):
        """Load persisted zones from disk on startup."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.active_zones = json.load(f)
            except Exception as e:
                logger.warning(f"Zone tracker: could not load state — {e}. Starting fresh.")
                self.active_zones = {}

    def _save_state(self):
        """Persist current zones to disk."""
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.active_zones, f, indent=2)
        except Exception as e:
            logger.warning(f"Zone tracker: could not save state — {e}")

    def get_canonical_signature(self, zone: Dict, symbol: str, precision: int = 5) -> str:
        direction = zone['type'].upper()
        if zone['type'] == 'demand':
            proximal, distal = zone['price_high'], zone['price_low']
        else:
            proximal, distal = zone['price_low'],  zone['price_high']
        rf = 10 ** precision
        return (f"{symbol}|{direction}|"
                f"{round(proximal * rf) / rf:.5f}|"
                f"{round(distal  * rf) / rf:.5f}")

    def _should_expire(self, zone_meta: Dict, current_time: datetime,
                       current_price: float) -> bool:
        try:
            created = datetime.fromisoformat(zone_meta['created_at'])
        except Exception:
            return True
        if (current_time - created).total_seconds() > self.ttl_seconds:
            return True
        zone_mid = zone_meta.get('price_mid', 0)
        if zone_mid > 0:
            dist_pips = abs(current_price - zone_mid) / self.pip_size
            if dist_pips > self.distance_threshold_pips:
                return True
        return False

    def is_duplicate(self, signature: str, current_time: datetime,
                      current_price: float) -> Tuple[bool, Optional[str]]:
        self._load_state()  # Reload disk state before query
        if signature not in self.active_zones:
            return False, None
        meta = self.active_zones[signature]
        if self._should_expire(meta, current_time, current_price):
            del self.active_zones[signature]
            self._save_state()
            return False, None
        
        # Zone still active — increment touch count
        touch_count = meta.get('touch_count', 1)
        meta['touch_count'] = min(touch_count + 1, 15)
        meta['last_touch'] = current_time.isoformat()
        self._save_state()

        created = datetime.fromisoformat(meta['created_at'])
        age_m   = int((current_time - created).total_seconds() / 60)
        touches = meta.get('touch_count', 1)
        return True, f"Active zone (age: {age_m}m, touches: {touches}, status: {meta.get('status','active')})"

    def add_zone(self, signature: str, zone: Dict, status: str = 'active'):
        self._load_state()  # Reload disk state before addition/update
        if signature in self.active_zones:
            # Zone already tracked — increment touch count
            self.active_zones[signature]['touch_count'] = min(
                self.active_zones[signature].get('touch_count', 1) + 1,
                15  # Cap at 15 to match training ceiling
            )
            self.active_zones[signature]['last_touch'] = datetime.utcnow().isoformat()
        else:
            # New zone
            self.active_zones[signature] = {
                'signature':  signature,
                'status':     status,
                'created_at': datetime.utcnow().isoformat(),
                'last_touch': datetime.utcnow().isoformat(),
                'price_mid':  zone.get('price_mid'),
                'zone_type':  zone.get('type'),
                'probability': zone.get('probability', 0),
                'touch_count': 1
            }
        self._save_state()

    def get_active_count(self) -> int:
        self._load_state()
        return len(self.active_zones)

    def cleanup_expired(self, current_time: datetime, current_price: float) -> int:
        self._load_state()
        expired = [s for s, m in self.active_zones.items()
                   if self._should_expire(m, current_time, current_price)]
        for s in expired:
            del self.active_zones[s]
        if expired:
            self._save_state()
        return len(expired)




def save_signals_to_single_file(signals: List[Dict],
                                filepath: str = "signals/all_signals.json"):
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)

    existing_signals: List[Dict] = []
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    existing_signals = data if isinstance(data, list) else [data]
        except Exception as e:
            logger.warning(f"save_signals: could not read existing file — {e}")
            existing_signals = []

    # Resolved statuses that must never be overwritten
    RESOLVED_STATUSES = {'hit_tp', 'hit_sl', 'expired', 'entry_missed'}

    def get_signal_id(signal):
        sym   = str(signal.get('symbol', ''))
        entry = float(signal.get('entry', 0))
        if 'XAU' in sym or 'GOLD' in sym.upper():
            entry_bucketed = round(entry / 5) * 5
        elif 'JPY' in sym:
            entry_bucketed = round(entry * 2) / 2
        else:
            entry_bucketed = round(entry, 3)
        return (f"{sym}_{signal.get('direction')}_"
                f"{entry_bucketed}_{signal.get('timeframe', '')}")

    existing_ids = {get_signal_id(s): i for i, s in enumerate(existing_signals)}
    new_added = updated = skipped_resolved = 0

    for signal in signals:
        sig_id = get_signal_id(signal)

        if sig_id in existing_ids:
            idx = existing_ids[sig_id]
            ex  = existing_signals[idx]

            # Never touch resolved outcomes — they are permanent historical records
            if ex.get('status') in RESOLVED_STATUSES:
                skipped_resolved += 1
                continue

            # Only update signals still awaiting entry
            if ex.get('status') == 'pending':

                # Rule 1 — always refresh execution levels (model has fresher data)
                ex['entry']        = signal.get('entry',       ex.get('entry'))
                ex['sl']           = signal.get('sl',          ex.get('sl'))
                ex['tp']           = signal.get('tp',          ex.get('tp'))
                ex['risk_pips']    = signal.get('risk_pips',   ex.get('risk_pips'))
                ex['reward_pips']  = signal.get('reward_pips', ex.get('reward_pips'))
                ex['risk_reward']  = signal.get('risk_reward', ex.get('risk_reward'))
                ex['rr']           = signal.get('rr',          ex.get('rr'))

                # Rule 2 — only upgrade probability, never degrade
                new_prob = float(signal.get('probability', 0))
                old_prob = float(ex.get('probability', 0))
                if new_prob > old_prob:
                    ex['probability']      = new_prob
                    ex['base_probability'] = signal.get(
                        'base_probability',
                        signal.get('probability', ex.get('base_probability', 0))
                    )

                # Rule 4 — refresh metadata with latest values where available
                ex['confluence']    = signal.get('confluence',    ex.get('confluence'))
                ex['distance_pips'] = signal.get('distance_pips', ex.get('distance_pips'))
                # Never overwrite it on a dedup cycle-update — the expiry clock must not
                # reset just because the same zone is seen again.  Only set it if the
                # existing record has no valid_until yet (should never happen, but safe).
                if not ex.get('valid_until'):
                    ex['valid_until'] = signal.get('valid_until')
                ex['last_seen']     = signal.get('last_seen', signal.get('timestamp', ex.get('last_seen')))

                updated += 1
            continue

        # Brand-new signal — add with pending status
        signal['status']    = 'pending'
        signal['last_seen'] = signal.get('timestamp')
        existing_signals.append(signal)
        new_added += 1

    if skipped_resolved:
        logger.debug(f"save_signals: {skipped_resolved} resolved signal(s) left unchanged")

    existing_signals = _convert_numpy(existing_signals)
    with open(filepath, 'w') as f:
        json.dump(existing_signals, f, indent=2)
    return new_added, updated


def _convert_numpy(obj):
    """Recursively convert NumPy/Pandas types to Python native types for JSON."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas Timestamp → ISO string
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except ImportError:
        pass
    # Any other datetime-like → string
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(i) for i in obj]
    return obj



class ProductionForexPredictor:
    """V7 Production Forex Predictor — all improvements applied"""

    def __init__(self,
                 api_key: str,
                 model_path: str = "models/model_v5_reversion.pkl",
                 scaler_path: str = "models/scaler_v5_reversion.pkl",
                 encoders_path: str = "models/encoders_v5_reversion.pkl",
                 stats_path: str = "models/training_stats_v5_reversion.pkl",
                 symbol: str = "EUR/USD",
                 prob_threshold: float = None,
                 max_rr: float = 3.5,
                 min_risk_atr: float = 0.8,
                 max_distance_pips: float = 25.0,
                 min_strength_atr: float = 0.35,
                 use_5min_refinement: bool = True,
                 refine_max_dist_multiplier: float = 1.0,
                 enable_continuation: bool = False,
                 continuation_prob_boost: float = 0.05,
                 use_htf_analysis: bool = True,
                 min_confluence_score: int = 4,
                 zone_ttl_hours: float = 5.8,
                 zone_distance_threshold: float = 50.0,
                 htf_fvg_min_gap_atr: float = 0.5,
                 htf_ob_min_reaction_atr: float = 2.0,
                 is_continuation_strategy: bool = False,
                 require_zone_close: bool = False,
                 debug: bool = False):

        self.api_key     = api_key
        self.symbol      = symbol
        self.base_url    = "https://api.twelvedata.com"
        self.pip_size    = 0.0001
        self.broker_offset = 0
        self.debug       = debug

        key = "cont" if is_continuation_strategy else "rev"
        validated_stats = {
            "rev":  {"base_prob": 0.65, "max_boost": 0.07},
            "cont": {"base_prob": 0.65, "max_boost": 0.06}
        }[key]

        self.prob_threshold  = validated_stats["base_prob"] if prob_threshold is None else prob_threshold
        self.max_boost       = validated_stats["max_boost"]
        self.max_rr          = max_rr
        self.min_risk_atr    = min_risk_atr
        self.max_distance_pips   = max_distance_pips
        self.min_strength_atr    = min_strength_atr
        self.use_5min_refinement = use_5min_refinement
        self.refine_max_dist_multiplier = refine_max_dist_multiplier
        self.enable_continuation     = enable_continuation
        self.continuation_prob_boost = continuation_prob_boost
        self.use_htf_analysis        = use_htf_analysis
        self.is_continuation_strategy = is_continuation_strategy
        self.require_zone_close       = require_zone_close  # IMPROVE: filter blown-through zones

        if not self.is_continuation_strategy:
            # TASK 1: REVERSION STRATEGY
            # NOTE (was hardcoded to 0.60 here, silently overriding the constructor
            # arg passed by run_dual_strategy, e.g. 0.62). That override is removed —
            # prob_threshold now respects whatever the caller passes in.
            # Prior rationale for 0.60 (kept for reference, not applied):
            #   SL hits clustered in the 65-71% probability band (6 of 6 losses were
            #   below 0.73). The 95.76% precision stat was measured at 0.65 on training
            #   data; live data showed more noise in that band.
            pass
        else:
            # TASK 2: CONTINUATION STRATEGY
            # NOTE: previously hardcoded to 0.60 here too, overriding the constructor
            # arg (e.g. 0.65). Removed for the same reason as above.
            # Prior rationale (kept for reference, not applied):
            #   Precision peaked stably at 80.0% at 0.60; pushing to 0.65 caused a
            #   severe sample-scarcity collapse.
            pass
        self.min_confluence_score     = min_confluence_score

        # Symbol-specific pip size (JPY pairs, Gold)
        sym_upper = symbol.upper()
        if 'JPY' in sym_upper:
            self.pip_size = 0.01
        elif 'XAU' in sym_upper or 'GOLD' in sym_upper:
            self.pip_size = 1.0   # $1 = 1 pip (MT4/MT5 standard, matches training)

        self.htf_analyzer = HTFInstitutionalAnalyzer(
            pip_size=self.pip_size,
            fvg_min_gap_atr=htf_fvg_min_gap_atr,
            ob_min_reaction_atr=htf_ob_min_reaction_atr
        )

        safe_sym = symbol.replace('/', '_')
        self.zone_tracker = ActiveZoneTracker(
            ttl_hours=zone_ttl_hours,
            distance_threshold_pips=zone_distance_threshold,
            broker_offset=self.broker_offset,
            state_file=f"signals/zones_{safe_sym}.json"
        )

        self.load_model(model_path, scaler_path, encoders_path, stats_path)

        self.htf_cache             = {}  # kept for attribute compatibility; not used for caching
        self.tracked_zones_15min   = []
        self.last_15min_bar_time   = None
        self.zone_history          = {}   # FIX: was never initialised — caused AttributeError at runtime

        self.stats = {
            'total_iterations': 0, 'total_candidates': 0,
            'total_duplicates': 0, 'total_new_zones': 0,
            'total_rejected': 0,   'htf_boosted': 0,
            'refine_rejections': 0
        }

        if not self.debug:
            print(f"✓ {key.upper()} [{symbol}] loaded | Base: {self.prob_threshold*100:.0f}% | MaxBoost: {self.max_boost*100:.0f}%")

    # Cache removed — pre_warm_cache is a no-op kept for call-site compatibility.
    def pre_warm_cache(self) -> None:
        pass


    def load_model(self, model_path, scaler_path, encoders_path, stats_path):
        # Ensemble weights — must match model_trainer.py exactly
        self._ensemble_weights = {'xgb': 0.50, 'rf': 0.35, 'lr': 0.15}

        try:
            with open(model_path,    'rb') as f: self.model          = pickle.load(f)  # XGBoost
            with open(scaler_path,   'rb') as f: self.scaler         = pickle.load(f)
            with open(encoders_path, 'rb') as f: self.encoders       = pickle.load(f)
            with open(stats_path,    'rb') as f: self.training_stats = pickle.load(f)

            # Load RF and LR — saved alongside XGBoost by model_trainer.save_models()
            rf_path = model_path.replace("model_", "rf_")
            lr_path = model_path.replace("model_", "lr_")
            self.rf_model = None
            self.lr_model = None
            try:
                with open(rf_path, 'rb') as f: self.rf_model = pickle.load(f)
            except FileNotFoundError:
                logger.warning(f"RF model not found at {rf_path} — ensemble will use XGBoost only")
            try:
                with open(lr_path, 'rb') as f: self.lr_model = pickle.load(f)
            except FileNotFoundError:
                logger.warning(f"LR model not found at {lr_path} — ensemble will use XGBoost only")

            cal_path = model_path.replace("model_", "calibrator_")
            self._calibrator = None
            try:
                with open(cal_path, 'rb') as f:
                    self._calibrator = pickle.load(f)
            except FileNotFoundError:
                pass

            strat    = "continuation" if self.is_continuation_strategy else "reversion"
            cal_st   = "with calibrator" if self._calibrator else "no calibrator"
            rf_st    = "RF✓" if self.rf_model else "RF✗"
            lr_st    = "LR✓" if self.lr_model else "LR✗"
            print(
                f"  ✅ Model v5 ({'continuation' if self.is_continuation_strategy else 'reversion'}) loaded — "
                f"XGB✓ RF✓ LR✓ [PROBABILITY CALIBRATED]"
            )

        except FileNotFoundError:
            logger.warning("Model files not found, falling back to reversion model")
            with open("models/model_v5_reversion.pkl",          'rb') as f: self.model          = pickle.load(f)
            with open("models/scaler_v5_reversion.pkl",         'rb') as f: self.scaler         = pickle.load(f)
            with open("models/encoders_v5_reversion.pkl",       'rb') as f: self.encoders       = pickle.load(f)
            with open("models/training_stats_v5_reversion.pkl", 'rb') as f: self.training_stats = pickle.load(f)
            self.rf_model    = None
            self.lr_model    = None
            self._calibrator = None


    def _get_session(self, hour: int) -> str:
        return _canonical_get_session(hour)


    def _calculate_indicators(self, df: pd.DataFrame):
        high       = df['high']
        low        = df['low']
        close_prev = df['close'].shift(1)
        tr         = pd.concat([high - low,
                                 (high - close_prev).abs(),
                                 (low  - close_prev).abs()], axis=1).max(axis=1)

        # Wilder's smoothing: seed with 14-bar SMA, then exponential
        atr_values = np.full(len(df), np.nan)
        seed_end   = 14
        if len(df) >= seed_end:
            atr_values[seed_end - 1] = tr.iloc[:seed_end].mean()
            for i in range(seed_end, len(df)):
                atr_values[i] = (atr_values[i - 1] * 13 + tr.iloc[i]) / 14
        df['atr'] = atr_values

        df['ema_20']  = df['close'].ewm(span=20,  adjust=False).mean()
        df['ema_50']  = df['close'].ewm(span=50,  adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        delta = df['close'].diff()
        # Wilder's EMA smoothing (alpha = 1/14), seeded with a 14-bar SMA.
        # Rolling SMA gives different values on trending data and misclassifies
        # overbought/oversold zones — Wilder's method matches every RSI indicator.
        gain  = delta.where(delta > 0, 0.0)
        loss  = (-delta.where(delta < 0, 0.0))
        gain  = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss  = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs    = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        for col in ['atr', 'ema_20', 'ema_50', 'ema_200', 'rsi']:
            df[col] = df[col].bfill().ffill()
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())


    def fetch_bars(self, interval: str = "15min", outputsize: int = 200) -> Optional[pd.DataFrame]:
        cache_key = (self.symbol, interval, outputsize)
        with _SHARED_BAR_CACHE_LOCK:
            entry = _SHARED_BAR_CACHE.get(cache_key)
            if entry is not None:
                age = time.time() - entry['ts']
                if age < _SHARED_BAR_CACHE_TTL:
                    return entry['df']
                # Stale entry — evict and re-fetch
                del _SHARED_BAR_CACHE[cache_key]

        endpoint = f"{self.base_url.rstrip()}/time_series"
        params   = {
            "symbol": self.symbol, "interval": interval,
            "outputsize": outputsize, "apikey": self.api_key, "format": "JSON"
        }

        # Acquire a rate-limit slot — tries this predictor's own key first,
        # then falls back to any free spare key if the primary is full.
        call_key = acquire_any(self.api_key)
        params["apikey"] = call_key

        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data       = response.json()
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep((attempt + 1) * 5)

        if last_error is not None:
            print(f"  ❌ fetch_bars({interval}): 3 attempts failed — {last_error}")
            logger.warning(f"fetch_bars({interval}): 3 attempts failed — {last_error}")
            return None
        if "values" not in data:
            api_status = data.get("status", "?")
            api_msg    = data.get("message", str(data)[:120])
            print(f"  ❌ fetch_bars({interval}): no 'values' in response — status={api_status} msg={api_msg}")
            return None

        df_data = []
        for candle in data["values"]:
            try:
                df_data.append({
                    "time":  pd.to_datetime(candle["datetime"]) + timedelta(hours=self.broker_offset),
                    "open":  float(candle["open"]),
                    "high":  float(candle["high"]),
                    "low":   float(candle["low"]),
                    "close": float(candle["close"])
                })
            except Exception as e:
                logger.warning(f"fetch_bars: bad candle row — {e}")
                continue

        if not df_data:
            return None

        df = pd.DataFrame(df_data)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["avg_spread"] = 0.00005
        self._calculate_indicators(df)

        # Store in the shared cache so the sibling predictor (REV↔CONT) can
        # reuse these bars within the same 15-min cycle without a duplicate fetch.
        with _SHARED_BAR_CACHE_LOCK:
            _SHARED_BAR_CACHE[cache_key] = {'df': df, 'ts': time.time()}

        return df


    def get_htf_context(self,
                         df_1h:   Optional[pd.DataFrame] = None,
                         df_4h_c: Optional[pd.DataFrame] = None,
                         df_4h:   Optional[pd.DataFrame] = None,
                         df_1d:   Optional[pd.DataFrame] = None) -> Dict:
        context = {}

        # Resolve bars — use pre-fetched if supplied, otherwise fetch live.
        _df_1h   = df_1h   if df_1h   is not None else self.fetch_bars(interval="1h",   outputsize=100)
        _df_4h_c = df_4h_c if df_4h_c is not None else self.fetch_bars(interval="4h",   outputsize=100)
        _df_4h   = df_4h   if df_4h   is not None else self.fetch_bars(interval="4h",   outputsize=200)
        _df_1d   = df_1d   if df_1d   is not None else self.fetch_bars(interval="1day", outputsize=90)

        if _df_1h is not None and len(_df_1h) > 50:
            context['h1_trend']  = 'bullish' if _df_1h['ema_20'].iloc[-1] > _df_1h['ema_50'].iloc[-1] else 'bearish'
            context['h1_close']  = _df_1h['close'].iloc[-1]
            context['trend_source'] = '1H'
        else:
            if _df_4h_c is not None and len(_df_4h_c) > 50:
                context['h1_trend']  = 'bullish' if _df_4h_c['ema_20'].iloc[-1] > _df_4h_c['ema_50'].iloc[-1] else 'bearish'
                context['h1_close']  = _df_4h_c['close'].iloc[-1]
                context['trend_source'] = '4H'
            else:
                context['h1_trend']     = 'neutral'
                context['h1_close']     = None
                context['trend_source'] = 'FALLBACK'

        context['h4_atr'] = _df_4h_c['atr'].iloc[-1] if _df_4h_c is not None and len(_df_4h_c) > 14 else None

        # outputsize=90: the D1 premium/discount gate uses tail(60) so we need
        # at least 60 bars here.
        if _df_1d is not None and len(_df_1d) > 1:
            y = _df_1d.iloc[-2]
            pivot = (y['high'] + y['low'] + y['close']) / 3
            context.update({
                'daily_pivot': pivot, 'daily_r1': 2 * pivot - y['low'],
                'daily_s1': 2 * pivot - y['high'],
                'yesterday_high': y['high'], 'yesterday_low': y['low']
            })
        else:
            context.update({'daily_pivot': None, 'daily_r1': None, 'daily_s1': None})

        if self.use_htf_analysis:
            if _df_4h is not None and len(_df_4h) > 20:
                context['h4_fvgs']         = self.htf_analyzer.detect_fvg(_df_4h, '4h')
                context['h4_order_blocks'] = self.htf_analyzer.detect_order_blocks(_df_4h, '4h')
                context['h4_structure']    = self.htf_analyzer.detect_market_structure_shift(_df_4h)
                context['h4_fib_levels']   = self.htf_analyzer.calculate_fibonacci_levels(_df_4h, 100)
                context['h4_atr']          = _df_4h['atr'].iloc[-1]
                if self.debug:
                    print(f"\n  🔍 4H [{self.symbol}]: {len(context['h4_fvgs'])} FVGs, {len(context['h4_order_blocks'])} OBs")
            else:
                context.update({'h4_fvgs': [], 'h4_order_blocks': [], 'h4_structure': {'trend': 'neutral'}, 'h4_fib_levels': {}})

            # Reuse _df_1h (outputsize=100) for 1H FVG/OB detection.
            # 100 bars covers ~4 days of 1H history — sufficient for zone detection.
            if _df_1h is not None and len(_df_1h) > 20:
                context['h1_fvgs']         = self.htf_analyzer.detect_fvg(_df_1h, '1h')
                context['h1_order_blocks'] = self.htf_analyzer.detect_order_blocks(_df_1h, '1h')
                context['h1_structure']    = self.htf_analyzer.detect_market_structure_shift(_df_1h)
                context['h1_fib_levels']   = self.htf_analyzer.calculate_fibonacci_levels(_df_1h, 50)
                if self.debug:
                    print(f"  🔍 1H [{self.symbol}]: {len(context['h1_fvgs'])} FVGs, {len(context['h1_order_blocks'])} OBs")
            else:
                context.update({'h1_fvgs': [], 'h1_order_blocks': [], 'h1_structure': {'trend': 'neutral'}, 'h1_fib_levels': {}})

            if _df_1d is not None and len(_df_1d) > 20:
                context['daily_fvgs']         = self.htf_analyzer.detect_fvg(_df_1d, 'daily')
                context['daily_order_blocks']  = self.htf_analyzer.detect_order_blocks(_df_1d, 'daily')
                context['daily_structure']     = self.htf_analyzer.detect_market_structure_shift(_df_1d)
            else:
                context.update({'daily_fvgs': [], 'daily_order_blocks': [], 'daily_structure': {'trend': 'neutral'}})

        return context


    def _detect_zones_with_validation(self, df: pd.DataFrame, htf_context: Dict) -> List[Dict]:
        zones        = []
        current_price = df['close'].iloc[-1]
        current_atr   = df['atr'].iloc[-1]
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = self.pip_size * 10

        lookback = min(384, len(df) - 10)
        recent   = df.tail(lookback)

        for i in range(10, len(recent) - 5):
            idx_in_df = len(df) - lookback + i
            bar       = recent.iloc[i]

            # ── SUPPLY ZONE ──────────────────────────────────────────────────
            prev_highs = recent.iloc[i - 5:i]['high']
            next_highs = recent.iloc[i + 1:i + 6]['high']

            if bar['high'] > prev_highs.max() and bar['high'] > next_highs.max():
                next_lows = recent.iloc[i + 1:i + 6]['low']
                drop = bar['high'] - next_lows.min()

                if drop > current_atr * self.min_strength_atr and bar['high'] > current_price:
                    strength     = drop / current_atr
                    # scanning to end-of-dataset — a Monday zone was being checked
                    # against all of Friday's future price action.
                    future_bars  = recent.iloc[i:i + 50]
                    # that normal retest wicks don't discard valid supply zones.
                    violated     = (future_bars['close'] > bar['high'] + current_atr * 1.5).any()

                    if not violated:
                        if htf_context.get('h1_trend') == 'bullish':
                            strength *= 0.7

                        body_top    = max(bar['open'], bar['close'])
                        # When body_top == bar['high'] the zone has no width,
                        # risk collapses to zero and SL falls entirely on ATR floor.
                        if (bar['high'] - body_top) < current_atr * 0.1:
                            continue
                        # Confluence = count of HTF structures overlapping this zone
                        # Matches training schema where confluence is an integer count
                        z_top = bar['high']; z_bot = body_top
                        conf_count = 1
                        for htf_item in (htf_context.get('h4_fvgs', []) +
                                         htf_context.get('h1_fvgs', []) +
                                         htf_context.get('h4_order_blocks', []) +
                                         htf_context.get('h1_order_blocks', [])):
                            ib = htf_item.get('bottom', htf_item.get('price_low', 0))
                            it = htf_item.get('top',    htf_item.get('price_high', 0))
                            if ib < z_top and it > z_bot:
                                conf_count += 1
                        zones.append({
                            'type':         'supply',
                            'price_mid':    (bar['high'] + body_top) / 2,
                            'price_high':   bar['high'],
                            'price_low':    body_top,
                            'strength':     strength,
                            'confluence':   conf_count,
                            'index':        idx_in_df,
                            'distance_pips': (bar['high'] - current_price) / self.pip_size,
                            'validated':    True,
                            'bar_time':     bar['time']
                        })

            # ── DEMAND ZONE ──────────────────────────────────────────────────
            prev_lows = recent.iloc[i - 5:i]['low']
            next_lows = recent.iloc[i + 1:i + 6]['low']

            if bar['low'] < prev_lows.min() and bar['low'] < next_lows.min():
                next_highs = recent.iloc[i + 1:i + 6]['high']
                rise = next_highs.max() - bar['low']

                if rise > current_atr * self.min_strength_atr and bar['low'] < current_price:
                    strength    = rise / current_atr
                    future_bars = recent.iloc[i:i + 50]
                    violated    = (future_bars['close'] < bar['low'] - current_atr * 1.5).any()

                    if not violated:
                        if htf_context.get('h1_trend') == 'bearish':
                            strength *= 0.7

                        body_bottom = min(bar['open'], bar['close'])
                        # When body_bottom == bar['low'] the zone has no width,
                        # risk collapses to zero and SL falls entirely on ATR floor.
                        if (body_bottom - bar['low']) < current_atr * 0.1:
                            continue
                        z_top = body_bottom; z_bot = bar['low']
                        conf_count = 1
                        for htf_item in (htf_context.get('h4_fvgs', []) +
                                         htf_context.get('h1_fvgs', []) +
                                         htf_context.get('h4_order_blocks', []) +
                                         htf_context.get('h1_order_blocks', [])):
                            ib = htf_item.get('bottom', htf_item.get('price_low', 0))
                            it = htf_item.get('top',    htf_item.get('price_high', 0))
                            if ib < z_top and it > z_bot:
                                conf_count += 1
                        zones.append({
                            'type':         'demand',
                            'price_mid':    (body_bottom + bar['low']) / 2,
                            'price_high':   body_bottom,
                            'price_low':    bar['low'],
                            'strength':     strength,
                            'confluence':   conf_count,
                            'index':        idx_in_df,
                            'distance_pips': (current_price - bar['low']) / self.pip_size,
                            'validated':    True,
                            'bar_time':     bar['time']
                        })

        # Calculate touch count for each zone: how many times has price
        # entered this zone's range in the recent lookback window?
        for zone in zones:
            z_high = zone['price_high']
            z_low  = zone['price_low']
            touches = int(((recent['low'] <= z_high) & (recent['high'] >= z_low)).sum())
            zone['zone_touch_count'] = max(1, touches)

        adaptive_max = max(35.0, current_atr / self.pip_size * 4.0)
        return [z for z in zones if abs(z['distance_pips']) < adaptive_max]


    def _htf_direction_filter(self, zones: List[Dict], htf_context: Dict,
                               current_price: float) -> List[Dict]:
        trend = htf_context.get('h1_trend', 'neutral')
        if trend == 'neutral':
            return zones
        allowed_type  = 'supply'  if trend == 'bearish' else 'demand'
        pullback_type = 'demand'  if trend == 'bearish' else 'supply'
        out = []
        for z in zones:
            if z['type'] == allowed_type:
                out.append(z)
            elif z['type'] == pullback_type and abs(z['distance_pips']) <= 25:
                z['is_pullback'] = True
                out.append(z)
        return out


    def detect_zone_flip(self, zone: Dict, df_5min: pd.DataFrame,
                         current_atr_15min: float) -> Optional[Dict]:
        if df_5min is None or len(df_5min) < 20:
            return None
        current_price   = df_5min['close'].iloc[-1]
        current_5min_atr = df_5min['atr'].iloc[-1]
        if pd.isna(current_5min_atr) or current_5min_atr <= 0:
            current_5min_atr = self.pip_size * 5

        zone_type       = zone['type']
        break_threshold = 5 * self.pip_size

        if zone_type == 'demand':
            flip_level  = zone['price_high']
            break_level = zone['price_low'] - break_threshold
            if not (df_5min.tail(20)['low'] < break_level).any():
                return None
            dist_pips = (flip_level - current_price) / self.pip_size
            if not (0 <= dist_pips <= 20):
                return None
            if df_5min['ema_20'].iloc[-1] >= df_5min['ema_50'].iloc[-1]:
                return None
            break_bars_ago = next(
                (len(df_5min) - 1 - i for i in range(len(df_5min) - 1, max(0, len(df_5min) - 21), -1)
                 if df_5min.iloc[i]['low'] < break_level), None)
            if break_bars_ago is None or break_bars_ago > 15:
                return None
            return {
                'flipped': True, 'original_type': 'demand', 'new_type': 'resistance',
                'flip_direction': 'SELL', 'flip_level': flip_level,
                'break_bars_ago': break_bars_ago, 'pullback_pips': dist_pips,
                'momentum_aligned': True
            }
        else:
            flip_level  = zone['price_low']
            break_level = zone['price_high'] + break_threshold
            if not (df_5min.tail(20)['high'] > break_level).any():
                return None
            dist_pips = (current_price - flip_level) / self.pip_size
            if not (0 <= dist_pips <= 20):
                return None
            if df_5min['ema_20'].iloc[-1] <= df_5min['ema_50'].iloc[-1]:
                return None
            break_bars_ago = next(
                (len(df_5min) - 1 - i for i in range(len(df_5min) - 1, max(0, len(df_5min) - 21), -1)
                 if df_5min.iloc[i]['high'] > break_level), None)
            if break_bars_ago is None or break_bars_ago > 15:
                return None
            return {
                'flipped': True, 'original_type': 'supply', 'new_type': 'support',
                'flip_direction': 'BUY', 'flip_level': flip_level,
                'break_bars_ago': break_bars_ago, 'pullback_pips': dist_pips,
                'momentum_aligned': True
            }


    def refine_entry_with_5min(self, zone: Dict, current_price: float,
                                current_atr_15min: float,
                                df_5min: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        _df = df_5min if df_5min is not None else self.fetch_bars(interval="5min", outputsize=100)
        if _df is None or len(_df) < 20:
            return None
        current_5min_price = _df['close'].iloc[-1]
        atr_5min           = _df['atr'].iloc[-1]
        if pd.isna(atr_5min) or atr_5min <= 0:
            atr_5min = self.pip_size * 5

        atr_15min_pips    = current_atr_15min / self.pip_size
        dynamic_threshold = atr_15min_pips * self.refine_max_dist_multiplier
        zone_mid          = zone['price_mid']
        dist_to_zone      = abs(current_5min_price - zone_mid) / self.pip_size

        if dist_to_zone > dynamic_threshold:
            if self.debug:
                print(f"    ⚠️  5-min far zone: {dist_to_zone:.1f}p > {dynamic_threshold:.1f}p "
                      f"(passing to ML — no hard reject)")

        recent_5min    = _df.tail(20)
        rejection_score = 0
        for i in range(-5, 0):
            bar        = recent_5min.iloc[i]
            body       = abs(bar['close'] - bar['open'])
            total_range = bar['high'] - bar['low']
            if total_range > 0:
                if zone['type'] == 'demand':
                    if (min(bar['open'], bar['close']) - bar['low']) > body * 1.5:
                        rejection_score += 1
                else:
                    if (bar['high'] - max(bar['open'], bar['close'])) > body * 1.5:
                        rejection_score += 1

        micro_zone_found = False
        refined_entry    = zone_mid
        max_micro_dist   = max(15, atr_5min / self.pip_size * 1.5)

        if zone['type'] == 'demand':
            for i in range(5, len(recent_5min) - 2):
                bar = recent_5min.iloc[i]
                if (bar['low'] < recent_5min.iloc[i-3:i]['low'].min() and
                        bar['low'] < recent_5min.iloc[i+1:i+3]['low'].min() and
                        abs(bar['low'] - zone_mid) / self.pip_size < max_micro_dist):
                    refined_entry = bar['low']; micro_zone_found = True; break
        else:
            for i in range(5, len(recent_5min) - 2):
                bar = recent_5min.iloc[i]
                if (bar['high'] > recent_5min.iloc[i-3:i]['high'].max() and
                        bar['high'] > recent_5min.iloc[i+1:i+3]['high'].max() and
                        abs(bar['high'] - zone_mid) / self.pip_size < max_micro_dist):
                    refined_entry = bar['high']; micro_zone_found = True; break

        momentum_aligned = (
            (zone['type'] == 'demand' and recent_5min['ema_20'].iloc[-1] > recent_5min['ema_50'].iloc[-1]) or
            (zone['type'] == 'supply' and recent_5min['ema_20'].iloc[-1] < recent_5min['ema_50'].iloc[-1])
        )
        total_score = rejection_score + (2 if micro_zone_found else 0) + (1 if momentum_aligned else 0)

        if total_score < 2:
            if self.debug:
                print(f"    ⚠️  5-min score low: {total_score}/8 < 2 "
                      f"(no refinement — zone continues to ML with base geometry)")
            return None

        sl_mult    = 2.2 if micro_zone_found else 2.8
        # tight. Use current_atr_15min (the 15min ATR passed in) so the SL is
        # always anchored to the correct timeframe's noise level.
        # sl_mult still controls how many ATRs of buffer beyond the entry.
        refined_sl = (refined_entry - current_atr_15min * sl_mult if zone['type'] == 'demand'
                      else refined_entry + current_atr_15min * sl_mult)

        atr_5min_pips  = atr_5min / self.pip_size
        # filled worse than planned; 0.15×ATR was systematically under-estimating fill slip)
        slippage_pips  = max(1.0, atr_5min_pips * 0.30)
        if zone['type'] == 'demand':
            refined_entry += slippage_pips * self.pip_size
            refined_sl    += slippage_pips * self.pip_size
        else:
            refined_entry -= slippage_pips * self.pip_size
            refined_sl    -= slippage_pips * self.pip_size

        return {
            'refined_entry': refined_entry, 'refined_sl': refined_sl,
            'confidence_boost': min(total_score * 0.02, 0.08),
            'confirmation_score': total_score, 'rejection_candles': rejection_score,
            'micro_zone_found': micro_zone_found, 'momentum_aligned': momentum_aligned,
            'distance_used': dist_to_zone, 'threshold_used': dynamic_threshold
        }


    def calculate_realistic_levels(self, zone: Dict, current_atr: float,
                                    current_price: float, refinement: Optional[Dict],
                                    htf_context: Dict) -> Tuple[float, float, float, float]:
        MIN_RR = 1.5
        is_buy = zone['type'] == 'demand'

        # ── Entry and SL ─────────────────────────────────────────────────────
        if refinement:
            entry = refinement['refined_entry']
            sl    = refinement['refined_sl']
            # Re-enforce the 15min ATR floor here so the SL is never tighter
            # than 1.5 × 15min ATR regardless of which path set it.
            _min_sl_dist = current_atr * max(self.min_risk_atr, 2.5)
            if abs(entry - sl) < _min_sl_dist:
                sl = entry - _min_sl_dist if is_buy else entry + _min_sl_dist
        else:
            if zone.get('is_continuation') and zone.get('flip_info'):
                entry = zone['flip_info']['flip_level']
            else:
                # limit order absorbs the normal wick overshoot before filling.
                # Without this, price routinely overshoots the body by 5-25 pips
                # and the fill is already at or past the SL.
                _body_proximal = (zone.get('price_high', zone['price_mid']) if is_buy
                                  else zone.get('price_low', zone['price_mid']))
                # slippage of +7.7 pips — price consistently blows through the body
                _buf = current_atr * 0.55
                entry = _body_proximal - _buf if is_buy else _body_proximal + _buf

            # not just the body edge. price_low (demand) / price_high (supply)
            # already holds the wick extreme from _detect_zones_with_validation.
            # Use 1.0 ATR as minimum so normal 15min noise never stops us out.
            if is_buy:
                wick_risk = entry - zone['price_low']          # entry to wick low
                risk      = max(wick_risk, current_atr * max(self.min_risk_atr, 2.5))
                sl        = entry - risk
            else:
                wick_risk = zone['price_high'] - entry         # wick high to entry
                risk      = max(wick_risk, current_atr * max(self.min_risk_atr, 2.5))
                sl        = entry + risk

        risk = abs(entry - sl)
        if risk <= 0:
            risk = current_atr * max(self.min_risk_atr, 2.5)
            sl   = entry - risk if is_buy else entry + risk

        # ── Spread on SL (entry slip, once) ──────────────────────────────────
        session_spread = {'sydney': 1.5, 'tokyo': 1.2, 'london': 0.7, 'ny': 0.8}
        spread_pips    = session_spread.get(self._get_session(datetime.utcnow().hour), 1.0)
        spread_price   = spread_pips * self.pip_size
        sl   = sl - spread_price * 0.5 if is_buy else sl + spread_price * 0.5
        risk = abs(entry - sl)

        # ── Collect opposing zones as TP candidates ───────────────────────────
        # BUY  → TP at SUPPLY zones (bearish FVGs + bearish OBs) above entry
        # SELL → TP at DEMAND zones (bullish FVGs + bullish OBs) below entry
        opposing_fvg_type = 'bearish' if is_buy else 'bullish'
        opposing_ob_type  = 'bearish' if is_buy else 'bullish'

        all_fvgs = (htf_context.get('h1_fvgs', []) +
                    htf_context.get('h4_fvgs', []) +
                    htf_context.get('daily_fvgs', []))
        all_obs  = (htf_context.get('h1_order_blocks', []) +
                    htf_context.get('h4_order_blocks', []) +
                    htf_context.get('daily_order_blocks', []))

        tp_candidates = []  # list of (distance_from_entry, source_label, raw_level)

        # FVGs — target the near edge (bottom of bearish FVG for BUY, top of bullish FVG for SELL)
        for fvg in all_fvgs:
            if fvg.get('type') != opposing_fvg_type:
                continue
            level = fvg['bottom'] if is_buy else fvg['top']
            if (is_buy and level > entry) or (not is_buy and level < entry):
                tp_candidates.append((abs(level - entry), 'fvg', level))

        # OBs — target the proximal edge of the opposing OB
        for ob in all_obs:
            if ob.get('type') != opposing_ob_type:
                continue
            level = ob['bottom'] if is_buy else ob['top']
            if (is_buy and level > entry) or (not is_buy and level < entry):
                tp_candidates.append((abs(level - entry), 'ob', level))

        # Swing structure — nearest swing high (BUY) or swing low (SELL)
        for struct in [htf_context.get('h4_structure', {}), htf_context.get('h1_structure', {})]:
            if is_buy:
                for sh in struct.get('swing_highs', []):
                    lvl = sh.get('price', 0)
                    if lvl > entry:
                        tp_candidates.append((abs(lvl - entry), 'swing_high', lvl))
            else:
                for sl_s in struct.get('swing_lows', []):
                    lvl = sl_s.get('price', 0)
                    if lvl < entry:
                        tp_candidates.append((abs(lvl - entry), 'swing_low', lvl))

        # ── Filter by RR validity and pick best candidate ─────────────────────
        TP_MAX_RR   = max(self.max_rr, 12.0)  # winners ran to RR 7-11, raise ceiling
        TARGET_RR   = 6.0                      # aim for this RR (was 4.0, winners averaged 9.7)

        def effective_rr(raw_level: float) -> float:
            tp_adj = raw_level - spread_price * 0.5 if is_buy else raw_level + spread_price * 0.5
            reward = abs(tp_adj - entry)
            return reward / risk if risk > 0 else 0.0

        tp_candidates.sort(key=lambda x: x[0])  # nearest first
        tp, actual_rr, tp_source = None, None, 'fallback'

        valid_candidates = []
        for dist, source, raw_level in tp_candidates:
            rr = effective_rr(raw_level)
            if MIN_RR <= rr <= TP_MAX_RR:
                valid_candidates.append((rr, dist, source, raw_level))

        if valid_candidates:
            # Pick candidate whose RR is closest to TARGET_RR
            best = min(valid_candidates, key=lambda x: abs(x[0] - TARGET_RR))
            _, _, tp_source, raw_level = best
            tp_adj    = raw_level - spread_price * 0.5 if is_buy else raw_level + spread_price * 0.5
            tp        = tp_adj
            actual_rr = effective_rr(raw_level)

        # ── Fallback: fixed RR if no opposing zone found ──────────────────────
        if tp is None:
            tp        = entry + risk * self.max_rr if is_buy else entry - risk * self.max_rr
            tp        = tp - spread_price * 0.5 if is_buy else tp + spread_price * 0.5
            actual_rr = abs(tp - entry) / risk if risk > 0 else self.max_rr
            tp_source = 'fixed_rr'

        # Hard floor
        if actual_rr < MIN_RR:
            tp        = entry + risk * MIN_RR if is_buy else entry - risk * MIN_RR
            actual_rr = MIN_RR
            tp_source = 'min_rr_floor'

        if self.debug:
            print(f"    📐 TP [{tp_source}]  Entry={entry:.5f}  SL={sl:.5f}  "
                  f"TP={tp:.5f}  RR=1:{actual_rr:.2f}  "
                  f"({len(tp_candidates)} candidates checked)")

        return entry, sl, tp, actual_rr

    def _predict(self, zone: Dict, df: pd.DataFrame,
                 entry: float, sl: float,
                 df_1h: Optional[pd.DataFrame] = None,
                 df_4h: Optional[pd.DataFrame] = None,
                 df_5m: Optional[pd.DataFrame] = None,
                 df_d1: Optional[pd.DataFrame] = None) -> float:
        try:
            # ── Per-zone touch_index ───────────────────────────────────────────
            # zones on the same pair shared the same approach bar → identical
            # candle_body_ratio, approach_momentum, approach_distance across every
            # OB regardless of price level. This collapsed model discrimination.
            _zone_top = zone.get('price_high', zone.get('top',    df['close'].iloc[-1]))
            _zone_bot = zone.get('price_low',  zone.get('bottom', df['close'].iloc[-1]))
            _cur_atr  = (df['atr'].iloc[-1]
                         if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1])
                         else self.pip_size * 15)

            # Last bar where price was within 1 ATR of the zone boundaries
            _near = (df['low']  <= _zone_top + _cur_atr) & \
                    (df['high'] >= _zone_bot - _cur_atr)
            if _near.any():
                _touch_index = int(_near.values.nonzero()[0][-1])
            else:
                _touch_index = len(df) - 1   # fallback: current bar

            row_df = build_feature_row(
                zone=zone, df=df, entry=entry, sl=sl,
                encoders=self.encoders, pip_size=self.pip_size,
                touch_index=_touch_index,
                df_1h=df_1h, df_4h=df_4h, df_5m=df_5m, df_d1=df_d1,
            )
        except ValueError as exc:
            print(f"  ❌ build_feature_row FAILED: {exc}")
            logger.error(f"Feature extraction failed for {zone.get('id','unknown')} ({self.symbol}): {exc}", exc_info=True)
            return 0.5
        except Exception as exc:
            print(f"  ❌ build_feature_row ERROR: {exc}")
            logger.error(f"Unexpected error building features for {zone.get('id','unknown')} ({self.symbol}): {exc}", exc_info=True)
            return 0.5

        _d = row_df.iloc[0].to_dict()

        if self.debug:
            print(
                f"  🔍 features: h4_fvg_inside={_d.get('h4_fvg_inside')}"
                f"  rsi={_d.get('rsi', 0):.1f}"
                f"  h1_liq_sweep={_d.get('h1_liquidity_sweep')}"
                f"  m5_bos={_d.get('m5_micro_bos')}"
                f"  m5_sweep={_d.get('m5_sweep_confirmation')}"
                f"  touch={_d.get('zone_touch_count')}"
                f"  zone_age={_d.get('zone_age_bars')}"
                f"  htf_align={_d.get('htf_trend_alignment')}"
                f"  h4_prem={_d.get('h4_premium_discount')}"
                f"  d1_prem={_d.get('d1_premium_discount')}"
            )

        # ── Feature audit log — ONE entry per unique zone per day ─────────────
        # Intentionally OUTSIDE if self.debug so it runs in production too.
        # Dedup guard prevents re-logging when rev + cont predictors both score
        # the same zone (previously caused 4× duplicates across 4 symbol passes).
        try:
            _log_dir  = Path('signals/feature_logs')
            _log_dir.mkdir(parents=True, exist_ok=True)
            _log_file = _log_dir / f"features_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            _fingerprint = (
                self.symbol,
                round(_d.get('rsi', 0), 1),
                round(_d.get('distance_from_ema20', 0), 4),
                _d.get('zone_age_bars'),
                _d.get('zone_touch_count'),
            )
            _seen: set = set()
            if _log_file.exists():
                with open(_log_file) as _rf:
                    for _line in _rf:
                        try:
                            _e = json.loads(_line)
                            _f = _e.get('features', {})
                            _seen.add((
                                _e.get('symbol'),
                                round(_f.get('rsi', 0), 1),
                                round(_f.get('distance_from_ema20', 0), 4),
                                _f.get('zone_age_bars'),
                                _f.get('zone_touch_count'),
                            ))
                        except Exception:
                            pass
            if _fingerprint not in _seen:
                _log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'symbol':    self.symbol,
                    'zone_id':   (f"{self.symbol}_age{_d.get('zone_age_bars')}"
                                  f"_rsi{round(_d.get('rsi', 0), 1)}"
                                  f"_touch{_d.get('zone_touch_count')}"),
                    'features':  {k: round(float(v), 6) for k, v in _d.items()},
                }
                with open(_log_file, 'a') as _lf:
                    _lf.write(json.dumps(_log_entry) + '\n')

                # ── DISTRIBUTION TRACKING for zone_touch_count & h1_structure_shift ──
                zone_touch = _d.get('zone_touch_count', 0)
                h1_structure = _d.get('h1_structure_shift', 0)

                # Log distribution every 100 signals (roughly hourly)
                if hasattr(self, '_signal_count'):
                    self._signal_count += 1
                    if self._signal_count % 100 == 0:
                        logger.info(f"Distribution snapshot ({self.symbol}, {self._signal_count} signals): "
                                  f"zone_touch_count={zone_touch}, h1_structure_shift={h1_structure} "
                                  f"(zero={1 if h1_structure==0 else 0})")
                else:
                    self._signal_count = 1
        except Exception:
            pass

        X = row_df.values
        if np.isnan(X).any() or np.isinf(X).any():
            bad_features = {row_df.columns[i]: str(X[0][i])
                          for i in range(len(row_df.columns))
                          if np.isnan(X[0][i]) or np.isinf(X[0][i])}
            logger.warning(f"NaN/Inf detected in features for {zone.get('id','unknown')} ({self.symbol}): {bad_features}")
            return 0.5
        try:
            X_scaled = self.scaler.transform(X)

            # ── Full weighted ensemble: XGB×0.50 + RF×0.35 + LR×0.15 ──────────
            # Matches model_trainer.py evaluate_ensemble() exactly.
            w = self._ensemble_weights
            p_xgb = float(self.model.predict_proba(X_scaled)[0][1])

            if self.rf_model is not None and self.lr_model is not None:
                p_rf  = float(self.rf_model.predict_proba(X_scaled)[0][1])
                p_lr  = float(self.lr_model.predict_proba(X_scaled)[0][1])
                raw_prob = (p_xgb * w['xgb']) + (p_rf * w['rf']) + (p_lr * w['lr'])
            elif self.rf_model is not None:
                # LR missing — redistribute its weight to XGBoost
                raw_prob = p_xgb * (w['xgb'] + w['lr']) + \
                           float(self.rf_model.predict_proba(X_scaled)[0][1]) * w['rf']
            else:
                # Only XGBoost available — use it alone
                raw_prob = p_xgb

            raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

            if getattr(self, '_calibrator', None) is not None:
                try:
                    raw_prob = float(self._calibrator.transform(
                        np.clip([raw_prob], 1e-6, 1 - 1e-6))[0])
                except Exception:
                    pass
            return float(np.clip(raw_prob, 0.0, 1.0))
        except Exception as exc:
            print(f"  ❌ model.predict FAILED: {exc}")
            logger.error(f"Model prediction failed for {zone.get('id','unknown')} ({self.symbol}, task={task}): {exc}", exc_info=True)
            return 0.5
            
    
    def predict_with_proper_features(self, zone: Dict, df: pd.DataFrame,
                                      htf_context: Dict,
                                      df_1h: Optional[pd.DataFrame] = None,
                                      df_4h: Optional[pd.DataFrame] = None,
                                      df_5m: Optional[pd.DataFrame] = None,
                                      df_d1: Optional[pd.DataFrame] = None) -> float:
        current_atr = df['atr'].iloc[-1]
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = self.pip_size * 10

        if zone['type'] == 'demand':
            entry = zone.get('price_high', zone['price_mid'])
            risk  = max(entry - zone['price_low'], current_atr * self.min_risk_atr)
            sl    = entry - risk
        else:
            entry = zone.get('price_low', zone['price_mid'])
            risk  = max(zone['price_high'] - entry, current_atr * self.min_risk_atr)
            sl    = entry + risk

        # The old code set 0 for counter-trend zones instead of -1, silencing
        # the model's counter-trend penalty for the entire 15-min pipeline.
        # Now: aligned=1, counter-trend=-1, neutral/unknown=0 for both directions.
        if 'htf_alignment' not in zone:
            d1_trend = htf_context.get('daily_structure', {}).get('trend', 'neutral')
            if d1_trend == 'neutral':
                d1_trend = htf_context.get('h4_structure', {}).get('trend', 'neutral')
            if zone['type'] == 'demand':
                # demand: bullish=aligned(1), bearish=counter(-1), neutral=0
                zone['htf_alignment'] = (1 if d1_trend == 'bullish'
                                         else -1 if d1_trend == 'bearish'
                                         else 0)
            else:
                # supply: bearish=aligned(1), bullish=counter(-1), neutral=0
                zone['htf_alignment'] = (1 if d1_trend == 'bearish'
                                         else -1 if d1_trend == 'bullish'
                                         else 0)

        # Use pre-fetched bars if supplied; fall back to live fetch only when
        # called standalone (e.g. outside the main per-cycle batch path).
        _df_1h = df_1h if df_1h is not None else self.fetch_bars(interval="1h",   outputsize=100)
        _df_4h = df_4h if df_4h is not None else self.fetch_bars(interval="4h",   outputsize=100)
        _df_5m = df_5m if df_5m is not None else self.fetch_bars(interval="5min", outputsize=60)
        _df_d1 = df_d1 if df_d1 is not None else self.fetch_bars(interval="1day", outputsize=90)

        return self._predict(zone, df, entry, sl,
                             df_1h=_df_1h, df_4h=_df_4h, df_5m=_df_5m, df_d1=_df_d1)

    def _ml_score_htf_zone(self, zone: Dict, df: pd.DataFrame, tf: str,
                            htf_context: Optional[Dict] = None,
                            zone_override: Optional[Dict] = None,
                            df_1h: Optional[pd.DataFrame] = None,
                            df_4h: Optional[pd.DataFrame] = None,
                            df_5m: Optional[pd.DataFrame] = None,
                            df_d1: Optional[pd.DataFrame] = None) -> float:
        # Fast path: pre-built zone with all MTF features already injected
        if zone_override is not None:
            current_atr = df['atr'].iloc[-1]
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = self.pip_size * 15
            zone_type = zone_override.get('type', 'demand')
            entry = (zone_override['price_high'] if zone_type == 'demand'
                     else zone_override['price_low'])
            risk  = max(abs(entry - (zone_override['price_low'] if zone_type == 'demand'
                                     else zone_override['price_high'])),
                        current_atr * self.min_risk_atr)
            sl = entry - risk if zone_type == 'demand' else entry + risk
            _df_1h = df_1h if df_1h is not None else self.fetch_bars(interval="1h",   outputsize=100)
            _df_4h = df_4h if df_4h is not None else self.fetch_bars(interval="4h",   outputsize=100)
            _df_5m = df_5m if df_5m is not None else self.fetch_bars(interval="5min", outputsize=60)
            _df_d1 = df_d1 if df_d1 is not None else self.fetch_bars(interval="1day", outputsize=30)
            return self.apply_boost_ceiling(
                self._predict(zone_override, df, entry, sl,
                              df_1h=_df_1h, df_4h=_df_4h, df_5m=_df_5m, df_d1=_df_d1), 0)
        current_atr = df['atr'].iloc[-1]
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = self.pip_size * 15

        direction  = zone.get('direction', 'BUY')
        zone_type  = 'demand' if direction == 'BUY' else 'supply'
        zone_mid   = zone.get('mid', df['close'].iloc[-1])
        zone_top   = zone.get('top',    zone_mid + current_atr * 0.5)
        zone_bot   = zone.get('bottom', zone_mid - current_atr * 0.5)
        actual_idx = zone.get('index', max(0, len(df) - int(len(df) * 0.2)))

        d1_trend  = htf_context.get('daily_structure', {}).get('trend', 'neutral') if htf_context else 'neutral'
        htf_align = 1 if (
            (zone_type == 'demand' and d1_trend == 'bullish') or
            (zone_type == 'supply' and d1_trend == 'bearish')
        ) else 0

        # so feature_builder gets a training-compatible value (not always 1).
        overlap_count = 1
        if htf_context:
            all_htf = (htf_context.get('h1_fvgs', []) + htf_context.get('h4_fvgs', []) +
                       htf_context.get('h1_order_blocks', []) + htf_context.get('h4_order_blocks', []))
            for item in all_htf:
                item_bot = item.get('bottom', item.get('price_low', 0))
                item_top = item.get('top',    item.get('price_high', 0))
                if item_bot < zone_top and item_top > zone_bot:
                    overlap_count += 1

        full_zone = {
            'type': zone_type, 'price_mid': zone_mid, 'price_high': zone_top,
            'price_low': zone_bot, 'strength': zone.get('strength', 1.0),
            'confluence': overlap_count,  # integer, training-compatible
            'index': actual_idx, 'htf_alignment': htf_align
        }
        if zone_type == 'demand':
            entry = zone_top; risk = max(entry - zone_bot, current_atr * self.min_risk_atr); sl = entry - risk
        else:
            entry = zone_bot; risk = max(zone_top - entry, current_atr * self.min_risk_atr); sl = entry + risk

        return self.apply_boost_ceiling(self._predict(full_zone, df, entry, sl,
                                                       df_1h=df_1h, df_4h=df_4h,
                                                       df_5m=df_5m, df_d1=df_d1), 0)

    def apply_boost_ceiling(self, base: float, boost: float) -> float:
        return min(base + min(boost, self.max_boost), 0.95)

    # ARCHITECTURE (confirmed from training_labels_clean.parquet):
    #   tf_source:    4h_1h_confluence=5190 (90%),  4h=561 (10%),  15min=0
    #   zone_sources: order_block=96%,  fvg=4%
    # The old 15min demand/supply zone pipeline is SECONDARY — the model was
    # never trained on raw 15min zones and scores them unreliably.

    def _check_d1_alignment(self, direction: str, df_d1) -> int:
        """
        Returns 1 if the D1 EMA20/50 trend aligns with the zone direction,
        else 0 (counter-trend OR D1 data unavailable).

        Mirrors smc_detector.SMCDetector.get_d1_trend() / label_generator's
        calculate_htf_trend_alignment() exactly, so the live htf_alignment
        feature matches what the model was trained on:
          - binary {0, 1}, NOT {-1, 0, 1}
          - computed from D1 close EMA20 vs EMA50, not from H1 data
          - requires >= 50 D1 bars, else neutral (0)
        """
        if df_d1 is None or len(df_d1) < 50:
            return 0
        try:
            ema20 = float(df_d1['close'].ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(df_d1['close'].ewm(span=50, adjust=False).mean().iloc[-1])
        except Exception:
            return 0

        if ema20 > ema50:
            d1_trend = 'up'
        elif ema20 < ema50:
            d1_trend = 'down'
        else:
            d1_trend = 'neutral'

        if direction == 'BUY':
            return 1 if d1_trend == 'up' else 0
        else:
            return 1 if d1_trend == 'down' else 0

    def _check_1h_confluence(self, zone_top: float, zone_bot: float,
                              direction: str, df_1h) -> Dict:
        result = {
            'h1_trend_alignment': 0,
            'h1_structure_shift': 0,
            'h1_displacement':    0.0,
            'h1_liquidity_sweep': 0,
            'confirmed':          False,
        }
        if df_1h is None or len(df_1h) < 20:
            return result

        try:
            sub = df_1h.copy()

            # ── 1H ATR (Wilder 14-period on true range) ───────────────────────
            high  = sub['high']; low = sub['low']; close = sub['close']
            tr    = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_1h = float(tr.rolling(14).mean().iloc[-1])
            if not np.isfinite(atr_1h) or atr_1h <= 0:
                atr_1h = float(tr.mean()) or 1e-6

            # ── h1_trend_alignment : {-1, 0, 1} ──────────────────────────────
            # Training threshold: 0.0001 band around EMA crossover (1.0001 / 0.9999)
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            if ema20 > ema50 * 1.0001:
                raw_trend = 1       # 1H bullish
            elif ema20 < ema50 * 0.9999:
                raw_trend = -1      # 1H bearish
            else:
                raw_trend = 0       # flat / neutral

            # Align to zone direction so 1=aligned, -1=counter, 0=neutral for both
            if direction == 'BUY':
                result['h1_trend_alignment'] = raw_trend           # demand: bullish=1
            else:
                result['h1_trend_alignment'] = -raw_trend          # supply: bearish=1

            window       = sub.tail(30).reset_index(drop=True)
            swing_period = 3
            swing_highs, swing_lows = [], []
            for i in range(swing_period, len(window) - swing_period):
                bar = window.iloc[i]
                if (bar['high'] > window.iloc[i - swing_period:i]['high'].max() and
                        bar['high'] > window.iloc[i + 1:i + swing_period + 1]['high'].max()):
                    swing_highs.append({'i': i, 'price': bar['high']})
                if (bar['low'] < window.iloc[i - swing_period:i]['low'].min() and
                        bar['low'] < window.iloc[i + 1:i + swing_period + 1]['low'].min()):
                    swing_lows.append({'i': i, 'price': bar['low']})

            last_close = float(window['close'].iloc[-1])
            if swing_highs and last_close > swing_highs[-1]['price']:
                result['h1_structure_shift'] = 1    # bullish BOS
            elif swing_lows and last_close < swing_lows[-1]['price']:
                result['h1_structure_shift'] = -1   # bearish BOS

            # ── h1_displacement : ATR-normalised max body, last 20 bars ───────
            # Training: max(|close-open|) / atr_1h  capped at 10.0
            # Dimensionless ratio, ~0.5–3.0 for normal bars.
            recent_20  = sub.tail(20)
            body_sizes = (recent_20['close'] - recent_20['open']).abs()
            h1_disp    = float(body_sizes.max() / atr_1h) if atr_1h > 0 else 0.0
            result['h1_displacement'] = round(min(h1_disp, 10.0), 4)

            # ── h1_liquidity_sweep : zone-boundary wick + reclaim ─────────────
            # sp=4 rolling window is preserved to match feature_builder exactly.
            check_window = sub.reset_index(drop=True)
            sp = 4
            for i in range(sp, len(check_window)):
                bar     = check_window.iloc[i]
                prev_lo = check_window.iloc[max(0, i - sp):i]['low'].min()
                prev_hi = check_window.iloc[max(0, i - sp):i]['high'].max()
                if direction == 'BUY' and bar['low'] < prev_lo and bar['close'] > prev_lo:
                    result['h1_liquidity_sweep'] = 1
                    break
                if direction == 'SELL' and bar['high'] > prev_hi and bar['close'] < prev_hi:
                    result['h1_liquidity_sweep'] = 1
                    break

            result['confirmed'] = bool(
                result['h1_trend_alignment'] != 0 or result['h1_structure_shift'] != 0)
            return result

        except Exception as e:
            logger.warning(f"_check_1h_confluence failed: {e}")
            return result

    def _check_5min_entry(self, zone_top: float, zone_bot: float,
                           direction: str, df_5m) -> Dict:
        result = {
            'm5_micro_bos':            0,
            'm5_sweep_confirmation':   0,
            'm5_entry_candle_quality': 0.0,
        }
        if df_5m is None or len(df_5m) < 20:
            return result

        try:
            import numpy as np
            import pandas as pd

            zone_type = 'demand' if direction == 'BUY' else 'supply'
            # using df_5m.tail(20) at the TOUCH MOMENT — meaning the BOS/sweep had
            # already happened (it was what caused the 15min signal). Live, we score
            # inside the zone. Both features default to 0 → z=-2.5 each.
            recent = df_5m.tail(60).reset_index(drop=True)

            # ── ATR on 5M ────────────────────────────────────────────────────
            hi = recent['high']; lo = recent['low']; cl = recent['close']
            tr = pd.concat([
                hi - lo,
                (hi - cl.shift(1)).abs(),
                (lo - cl.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_5m = float(tr.rolling(7).mean().iloc[-1])
            if not np.isfinite(atr_5m) or atr_5m <= 0:
                atr_5m = self.pip_size * 5

            # ── m5_micro_bos — swing-high/low BOS (sp=3) ────────────────────
            sp = 3
            for i in range(sp, len(recent) - 1):
                bar = recent.iloc[i]
                if zone_type == 'demand':
                    prev_hi = recent.iloc[max(0, i - sp):i]['high'].max()
                    if bar['close'] > prev_hi:
                        result['m5_micro_bos'] = 1
                        break
                else:
                    prev_lo = recent.iloc[max(0, i - sp):i]['low'].min()
                    if bar['close'] < prev_lo:
                        result['m5_micro_bos'] = 1
                        break

            # ── m5_sweep_confirmation — wick-beyond + close-back (lookback=4) ─
            for i in range(4, len(recent)):
                bar     = recent.iloc[i]
                prev_lo = recent.iloc[max(0, i - 4):i]['low'].min()
                prev_hi = recent.iloc[max(0, i - 4):i]['high'].max()
                if zone_type == 'demand':
                    if bar['low'] < prev_lo and bar['close'] > prev_lo:
                        result['m5_sweep_confirmation'] = 1
                        break
                else:
                    if bar['high'] > prev_hi and bar['close'] < prev_hi:
                        result['m5_sweep_confirmation'] = 1
                        break

            # ── m5_entry_candle_quality — composite score on last 5 bars ─────
            last_5 = recent.tail(5).reset_index(drop=True)
            score  = 0.0
            last   = last_5.iloc[-1]
            prev   = last_5.iloc[-2] if len(last_5) >= 2 else last_5.iloc[-1]
            body   = abs(last['close'] - last['open'])
            rng    = last['high'] - last['low']

            # 1. Engulfing
            prev_body = abs(prev['close'] - prev['open'])
            if body > prev_body and rng > 0:
                score += 0.25

            # 2. Displacement (body ≥ 60% of ATR)
            if atr_5m > 0 and body / atr_5m >= 0.6:
                score += 0.25

            # 3. Strong close (top/bottom 30% of range in zone direction)
            if rng > 0:
                if zone_type == 'demand' and (last['close'] - last['low']) / rng >= 0.7:
                    score += 0.25
                elif zone_type == 'supply' and (last['high'] - last['close']) / rng >= 0.7:
                    score += 0.25

            # 4. Rejection wick (counter-directional wick > body)
            if rng > 0 and body > 0:
                if zone_type == 'demand':
                    lower_wick = min(last['open'], last['close']) - last['low']
                    if lower_wick > body:
                        score += 0.25
                else:
                    upper_wick = last['high'] - max(last['open'], last['close'])
                    if upper_wick > body:
                        score += 0.25

            result['m5_entry_candle_quality'] = round(score, 4)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"_check_5min_entry failed: {e}")

        return result

    def _get_15min_approach(self, zone_top: float, zone_bot: float,
                             direction: str, df_15m) -> Dict:
        result = {
            'approach_distance':  0.0,
            'approach_momentum':  0.0,
            'candle_body_ratio':  0.0,
        }
        if df_15m is None or len(df_15m) < 5:
            return result

        current_price = df_15m['close'].iloc[-1]
        # where zone_edge = price_high for demand (proximal top), price_low for supply.
        # The old code used zone_mid which gives a different value that the scaler
        # maps to a different position in feature space.
        if direction == 'BUY':
            zone_edge = zone_top   # proximal edge for demand = top of zone
        else:
            zone_edge = zone_bot   # proximal edge for supply = bottom of zone
        result['approach_distance'] = min(abs(current_price - zone_edge) / self.pip_size, 50.0)

        if 'atr' in df_15m.columns and len(df_15m) >= 5:
            atr = df_15m['atr'].iloc[-1]
            if atr and atr > 0:
                momentum = ((df_15m['close'].iloc[-1] - df_15m['close'].iloc[-5])
                            / atr)
                result['approach_momentum'] = round(float(momentum), 4)

        last = df_15m.iloc[-1]
        body = abs(last['close'] - last['open'])
        wick = last['high'] - last['low']
        result['candle_body_ratio'] = round(body / wick, 4) if wick > 0 else 0.0
        return result

    def _calculate_touch_count(self, zone: Dict, df_15m: Optional[pd.DataFrame]) -> int:
        """
        Mirror of label_generator.calculate_zone_touch_count().
        Counts distinct price visits to the proximal edge within ATR/2 threshold
        in the 500-bar window BEFORE the zone index, on 15min bars.
        Uses 20-bar skip after each touch to avoid double-counting.
        This matches exactly what the model was trained on.
        """
        if df_15m is None or len(df_15m) < 10:
            return 1

        zone_type = zone.get('type', 'demand')
        proximal  = zone.get('price_high') if zone_type == 'demand' else zone.get('price_low')
        if proximal is None:
            return 1

        # Use zone index if available, else last bar
        zone_index = min(int(zone.get('index', len(df_15m) - 1)), len(df_15m) - 1)
        lookback   = 500
        start      = max(0, zone_index - lookback)
        touch_count = 0
        i = start

        atr_col = 'atr' if 'atr' in df_15m.columns else None

        while i < zone_index:
            bar = df_15m.iloc[i]
            if atr_col:
                atr_val = float(bar[atr_col]) if pd.notna(bar[atr_col]) else self.pip_size * 10
            else:
                atr_val = self.pip_size * 10
            threshold = atr_val * 0.5

            if zone_type == 'demand':
                touched = float(bar['low']) <= proximal + threshold
            else:
                touched = float(bar['high']) >= proximal - threshold

            if touched:
                touch_count += 1
                i += 20   # skip to avoid double-counting same visit
            else:
                i += 1

        return max(1, touch_count)

    def htf_trade_detect(self, tf: str = "4h", htf_context: Optional[Dict] = None,
                          df_4h:  Optional[pd.DataFrame] = None,
                          df_15m: Optional[pd.DataFrame] = None,
                          df_5m:  Optional[pd.DataFrame] = None,
                          df_1h:  Optional[pd.DataFrame] = None,
                          df_d1:  Optional[pd.DataFrame] = None) -> List[Dict]:
        df = df_4h if df_4h is not None else self.fetch_bars(interval=tf, outputsize=200)
        if df is None or len(df) < 50:
            return []

        # The 4H candle close is the price at the END of a 4-hour bar — often
        # hours old. Using it caused the bias gate and dist filter to run on a
        # frozen price. 15min bars have a 15-minute close which is always live.
        _df_15m = df_15m if df_15m is not None else self.fetch_bars(interval="15min", outputsize=700)
        _df_5m  = df_5m  if df_5m  is not None else self.fetch_bars(interval="5min",  outputsize=60)
        _df_1h  = df_1h  if df_1h  is not None else self.fetch_bars(interval="1h",    outputsize=100)
        _df_d1  = df_d1  if df_d1  is not None else self.fetch_bars(interval="1day",  outputsize=90)

        if _df_15m is not None and len(_df_15m) > 0:
            current_price = _df_15m['close'].iloc[-1]
        else:
            current_price = df['close'].iloc[-1]  # fallback only

        current_atr   = df['atr'].iloc[-1]
        fvgs          = self.htf_analyzer.detect_fvg(df, tf)
        obs           = self.htf_analyzer.detect_order_blocks(df, tf)
        fib_levels    = self.htf_analyzer.calculate_fibonacci_levels(df, 100)

        # Derive h4_trend for the hard bias gate.
        # Prefer a fresh detect_market_structure_shift on the passed df;
        # fall back to whatever htf_context already holds.
        _h4_struct = self.htf_analyzer.detect_market_structure_shift(df)
        h4_trend   = _h4_struct.get('trend', 'neutral')
        if h4_trend == 'neutral' and htf_context:
            h4_trend = htf_context.get('h4_structure', {}).get('trend', 'neutral')

        # D1 Premium / Discount gate calculation
        _d1_pd = 0
        if _df_d1 is not None and len(_df_d1) >= 20:
            _tail60  = _df_d1.tail(60)
            _d1_high = float(np.percentile(_tail60["high"], 95))
            _d1_low  = float(np.percentile(_tail60["low"],   5))
            _d1_rng  = _d1_high - _d1_low
            _d1_pos  = (current_price - _d1_low) / _d1_rng if _d1_rng > 0 else 0.5
            if _d1_rng > 0:
                if _d1_pos > 0.6:
                    _d1_pd = 1   # premium
                elif _d1_pos < 0.4:
                    _d1_pd = -1  # discount

        htf_setups       = []
        MAX_DIST_ATR     = 20.0
        MAX_DIST_PIPS    = MAX_DIST_ATR * (current_atr / self.pip_size)
        MIN_HTF_RR   = 1.2
        MAX_HTF_RR   = 6.0 if self.symbol == 'XAU/USD' else 10.0
        MIN_HTF_STRENGTH = 0.5  # lowered from 0.8 — 0.8 kills valid zones in low-vol sessions

        if self.debug:
            print(f"\n  🔍 HTF [{self.symbol}|{tf}] current_price={current_price:.5f} "
                  f"ATR={current_atr:.5f}  FVGs={len(fvgs)}  OBs={len(obs)}")
        htf_fvg_rejected = {'dist': 0, 'confs': 0, 'direction': 0, 'rr': 0, 'strength': 0, 'prob': 0, 'd1_pd': 0}
        htf_ob_rejected  = {'dist': 0, 'confs': 0, 'direction': 0, 'rr': 0, 'strength': 0, 'prob': 0, 'd1_pd': 0}

        # ATR tolerance for direction filter: zones whose mid is within 1 ATR of
        # current price are still valid candidates (price is approaching them).
        atr_tol = current_atr * 1.0

        def _build_setup(zone_obj, zone_type_str, direction, entry, sl, tp_target,
                         risk, reward, rr, base_prob, conf_str, confirmations):
            final_prob = self.apply_boost_ceiling(
                base_prob, min(confirmations * 0.02, self.max_boost))
            return {
                'timestamp':    datetime.utcnow().isoformat(),
                'symbol':       self.symbol,
                'timeframe':    tf,
                'type':         zone_type_str,
                'strategy':     'CONTINUATION' if self.is_continuation_strategy else 'REVERSION',
                'direction':    direction,
                'entry':        round(entry, 5),
                'sl':           round(sl, 5),
                'tp':           round(tp_target, 5),
                'risk_pips':    round(risk / self.pip_size, 1),
                'reward_pips':  round(reward / self.pip_size, 1),
                'risk_reward':  f"1:{rr:.1f}",
                'rr':           f"1:{rr:.1f}",
                'probability':  round(final_prob * 100, 1),
                'base_probability': round(base_prob * 100, 1),
                'confluence':   conf_str,
                'confirmations': confirmations,
                'distance_pips': round(abs(current_price - entry) / self.pip_size, 1),
                'd1_trend':     htf_context.get('daily_structure', {}).get('trend', 'neutral') if htf_context else 'unknown',
                'zone_strength': round(zone_obj.get('strength', 0), 2),
                'htf_trend':    htf_context.get('h1_trend', 'unknown') if htf_context else 'unknown',
                # Zone body boundaries — required by PriceZoneMonitor for Step 3 body check
                'zone_top':     round(zone_obj.get('top', zone_obj.get('price_high', entry)), 5),
                'zone_bottom':  round(zone_obj.get('bottom', zone_obj.get('price_low', sl)), 5),
            }

        # FVGs
        for fvg in fvgs:
            if abs(current_price - fvg['mid']) / self.pip_size > MAX_DIST_PIPS:
                htf_fvg_rejected['dist'] += 1; continue
            future = df.iloc[fvg['index'] + 1:]
            if len(future) < 2:
                continue
            if fvg['type'] == 'bullish':
                confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]['low'] >= fvg['bottom'])
            else:
                confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]['high'] <= fvg['top'])
            if confs < 2:
                htf_fvg_rejected['confs'] += 1; continue

            direction = 'BUY' if fvg['type'] == 'bullish' else 'SELL'
            entry     = fvg['mid']
            sl        = fvg['bottom'] if direction == 'BUY' else fvg['top']

            # ── 4H HARD BIAS BLOCK (HTF FVG) ─────────────────────────────────
            # Bullish 4H trend → block SELL signals. Bearish → block BUY signals.
            if h4_trend == 'bullish' and direction == 'SELL':
                htf_fvg_rejected['direction'] += 1
                if self.debug:
                    print(f"     ⛔ HTF FVG HARD BLOCK: SELL blocked — 4H trend is bullish")
                continue
            if h4_trend == 'bearish' and direction == 'BUY':
                htf_fvg_rejected['direction'] += 1
                if self.debug:
                    print(f"     ⛔ HTF FVG HARD BLOCK: BUY blocked — 4H trend is bearish")
                continue

            # ── D1 Premium / Discount Gate (HTF FVG) ───────────────────────────
            if _d1_pd == 1 and direction == 'BUY':
                if h4_trend != 'bullish':
                    htf_fvg_rejected['d1_pd'] += 1
                    if self.debug:
                        print(f"     ⛔ HTF FVG D1 PREMIUM BLOCK: BUY blocked — price in D1 premium and 4H={h4_trend}")
                    continue
            if _d1_pd == -1 and direction == 'SELL':
                if h4_trend == 'bullish':
                    htf_fvg_rejected['d1_pd'] += 1
                    if self.debug:
                        print(f"     ⛔ HTF FVG D1 DISCOUNT BLOCK: SELL blocked — price in D1 discount and 4H is bullish")
                    continue

            # Bullish FVG: mid must be below current price (support below price).
            # Bearish FVG: mid must be above current price (resistance above price).
            # Same fix as OB direction filter — use mid + ATR tolerance.
            if direction == 'BUY'  and fvg['mid'] > current_price + atr_tol: htf_fvg_rejected['direction'] += 1; continue
            if direction == 'SELL' and fvg['mid'] < current_price - atr_tol: htf_fvg_rejected['direction'] += 1; continue

            tp_target = (max(fib_levels.get('0.5', entry + fvg['size'] * 2), entry + fvg['size'] * 2)
                         if direction == 'BUY' else
                         min(fib_levels.get('0.5', entry - fvg['size'] * 2), entry - fvg['size'] * 2))

            risk   = abs(entry - sl)
            reward = abs(tp_target - entry)
            rr     = reward / risk if risk > 0 else 2.0
            if not (MIN_HTF_RR <= rr <= MAX_HTF_RR):
                htf_fvg_rejected['rr'] += 1; continue
            if fvg.get('strength', 0) < MIN_HTF_STRENGTH:
                htf_fvg_rejected['strength'] += 1; continue

            # The old dict had no 'sources'/'tf_source' so h4_fvg_inside was always
            # 0.0 — this single bug was suppressing 40% of the model's decision weight.
            fvg_zone_for_model = {
                'direction':   direction,
                'type':        'demand' if direction == 'BUY' else 'supply',
                'price_high':  fvg.get('top',    fvg.get('price_high', entry)),
                'price_low':   fvg.get('bottom', fvg.get('price_low',  sl)),
                'price_mid':   fvg.get('mid',    fvg.get('price_mid',  entry)),
                'strength':    fvg['strength'],
                'index':       fvg.get('index', len(df) - 1),
                'h4_fvg_zone': True,   # ← triggers h4_fvg_inside=1 in feature_builder
            }
            base_prob = self._ml_score_htf_zone(
                fvg_zone_for_model, df, tf, htf_context=htf_context,
                df_4h=df)
            # Gold keeps strict 0.55 gate unchanged
            # Forex gets a softer gate during Asia/pre-London (00:00–07:00 UTC)
            if self.symbol == 'XAU/USD':
                _ml_gate = 0.55
            else:
                _hour_utc = datetime.utcnow().hour
                _ml_gate  = 0.45 if 0 <= _hour_utc < 7 else 0.55

            if base_prob < _ml_gate:
                if self.debug:
                    print(f"    ❌ FVG prob={base_prob:.3f} < gate={_ml_gate:.2f}"
                          f"  dir={direction}"
                          f"  strength={fvg.get('strength', 0):.2f}"
                          f"  age_bars={fvg.get('index', '?')}")
                htf_fvg_rejected['prob'] += 1; continue

            htf_setups.append(_build_setup(fvg, 'FVG', direction, entry, sl, tp_target,
                                           risk, reward, rr, base_prob,
                                           f"FVG-{fvg['strength']:.2f}", confs))

        # Order Blocks
        for ob in obs:
            if abs(current_price - ob['mid']) / self.pip_size > MAX_DIST_PIPS:
                htf_ob_rejected['dist'] += 1; continue
            future = df.iloc[ob['index'] + 1:]
            if len(future) < 2:
                continue
            if ob['type'] == 'bullish':
                confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]['low'] >= ob['bottom'])
            else:
                confs = sum(1 for i in range(min(5, len(future))) if future.iloc[i]['high'] <= ob['top'])
            if confs < 2:
                htf_ob_rejected['confs'] += 1; continue

            direction = 'BUY' if ob['type'] == 'bullish' else 'SELL'

            # ── 4H HARD BIAS BLOCK (HTF OB) ──────────────────────────────────
            # Bullish 4H trend → block SELL signals. Bearish → block BUY signals.
            if h4_trend == 'bullish' and direction == 'SELL':
                htf_ob_rejected['direction'] += 1
                if self.debug:
                    print(f"     ⛔ HTF OB HARD BLOCK: SELL blocked — 4H trend is bullish")
                continue
            if h4_trend == 'bearish' and direction == 'BUY':
                htf_ob_rejected['direction'] += 1
                if self.debug:
                    print(f"     ⛔ HTF OB HARD BLOCK: BUY blocked — 4H trend is bearish")
                continue

            # ── D1 Premium / Discount Gate (HTF OB) ────────────────────────────
            if _d1_pd == 1 and direction == 'BUY':
                if h4_trend != 'bullish':
                    htf_ob_rejected['d1_pd'] += 1
                    if self.debug:
                        print(f"     ⛔ HTF OB D1 PREMIUM BLOCK: BUY blocked — price in D1 premium and 4H={h4_trend}")
                    continue
            if _d1_pd == -1 and direction == 'SELL':
                if h4_trend == 'bullish':
                    htf_ob_rejected['d1_pd'] += 1
                    if self.debug:
                        print(f"     ⛔ HTF OB D1 DISCOUNT BLOCK: SELL blocked — price in D1 discount and 4H is bullish")
                    continue

            _atr_buf = df['atr'].iloc[-1] * 0.2 if 'atr' in df.columns else 0.0002
            if direction == 'BUY':
                entry = ob['top']               # proximal body edge for demand
                sl    = ob['bottom'] - _atr_buf # distal wick + ATR buffer
            else:
                entry = ob['bottom']            # proximal body edge for supply
                sl    = ob['top'] + _atr_buf    # distal wick + ATR buffer

            # Direction filter uses ob['mid'] — correctly identifies zone side
            # relative to price without being confused by the candle wick
            if direction == 'BUY'  and ob['mid'] > current_price + atr_tol: htf_ob_rejected['direction'] += 1; continue
            if direction == 'SELL' and ob['mid'] < current_price - atr_tol: htf_ob_rejected['direction'] += 1; continue

            tp_raw = (entry + ob['reaction_size'] * 1.5 if direction == 'BUY'
                    else entry - ob['reaction_size'] * 1.5)

            # For non-Gold pairs, clamp TP so RR never exceeds MAX_HTF_RR ceiling
            # Gold is untouched — its reaction_size already produces valid RR naturally
            if self.symbol != 'XAU/USD' and abs(entry - sl) > 0:
                max_reward = abs(entry - sl) * MAX_HTF_RR * 0.95   # 5% buffer under ceiling
                if direction == 'BUY':
                    tp_target = min(tp_raw, entry + max_reward)
                else:
                    tp_target = max(tp_raw, entry - max_reward)
            else:
                tp_target = tp_raw

            risk   = abs(entry - sl)
            reward = abs(tp_target - entry)
            rr     = reward / risk if risk > 0 else 1.5

            if not (MIN_HTF_RR <= rr <= MAX_HTF_RR):
                htf_ob_rejected['rr'] += 1; continue
            if ob.get('strength', 0) < MIN_HTF_STRENGTH:
                htf_ob_rejected['strength'] += 1; continue

            # ── Step 2: 1H confluence (90% of training required 4H+1H) ───────
            h1_conf = self._check_1h_confluence(ob['top'], ob['bottom'], direction, _df_1h)
            # Only hard-reject if 1H trend is CONFIRMED OPPOSITE (not merely neutral/flat).
            # trend_alignment=0 + structure_shift=0 in a quiet session = neutral, not opposing.
            # We still pass the h1 features to the model — it will discount them appropriately.
            h1_trend_str = htf_context.get('h1_trend', 'neutral') if htf_context else 'neutral'
            if self.debug and not h1_conf['confirmed']:
                print(f"     ℹ️  OB passing with neutral 1H (trend={h1_conf['h1_trend_alignment']} "
                      f"struct={h1_conf['h1_structure_shift']}) — letting model decide")

            # ── Step 3: 15min approach features ──────────────────────────────
            approach = self._get_15min_approach(ob['top'], ob['bottom'], direction, _df_15m)

            m5_conf      = self._check_5min_entry(ob['top'], ob['bottom'], direction, _df_5m)
            # 5M hard gate REMOVED — training labeled every zone touch regardless
            # of 5M state. m5_micro_bos / m5_sweep_confirmation flow to the model
            # as soft features; the 0.65 probability threshold does the filtering.

            # ── Step 5: Build full zone dict matching training schema exactly ─
            # Injects pre-computed MTF features so build_feature_row receives
            # training-compatible values (not recomputed from wrong TF bars).
            real_overlap = 1
            fvg_overlap_flag = False   # FIX A: track whether this OB overlaps any 4H FVG
            if htf_context:
                all_htf_items = (htf_context.get('h1_fvgs', []) +
                                 htf_context.get('h4_fvgs', []) +
                                 htf_context.get('h1_order_blocks', []) +
                                 htf_context.get('h4_order_blocks', []))
                h4_fvgs_only  = htf_context.get('h4_fvgs', [])
                for item in all_htf_items:
                    item_bot = item.get('bottom', item.get('price_low', 0))
                    item_top = item.get('top',    item.get('price_high', 0))
                    if item_bot < ob['top'] and item_top > ob['bottom']:
                        real_overlap += 1
                # Check specifically against 4H FVGs for fvg_overlap
                for fvg_item in h4_fvgs_only:
                    fb = fvg_item.get('bottom', fvg_item.get('price_low', 0))
                    ft = fvg_item.get('top',    fvg_item.get('price_high', 0))
                    if fb < ob['top'] and ft > ob['bottom']:
                        fvg_overlap_flag = True
                        break
            # Convert 4H OB index → 15min bar index via timestamp matching.
            # ob['index'] is a position in the 4H df. build_feature_row uses
            # zone['index'] to look up ema_trend/rsi/distance_from_ema20 from
            # the 15min df. Without conversion, 4H index=150 maps to 15min bar
            # 150 = only ~37 hours into data — completely wrong formation bar.
            _ob_15m_index = len(_df_15m) - 1 if _df_15m is not None else len(df) - 1
            if _df_15m is not None and 'time' in df.columns and 'time' in _df_15m.columns:
                try:
                    _ob_4h_time = df['time'].iloc[ob.get('index', len(df) - 1)]
                    _matched = (_df_15m['time'] <= _ob_4h_time)
                    if _matched.any():
                        _ob_15m_index = int(_matched.values.nonzero()[0][-1])
                except Exception:
                    pass
            full_ob_zone = {
                'type':                    'demand' if direction == 'BUY' else 'supply',
                'price_high':              ob['top'],
                'price_low':               ob['bottom'],
                'price_mid':               ob['mid'],
                'strength':                ob.get('strength', 1.0),
                'confluence':              real_overlap,
                'index':                   _ob_15m_index,  # converted: 4H index → 15min index
                'htf_alignment':           self._check_d1_alignment(direction, _df_d1),
                # 1H features — zone-aligned values from _check_1h_confluence()
                # (h1_trend_alignment is already direction-flipped; do not flip again)
                'h1_trend_alignment':      h1_conf['h1_trend_alignment'],
                'h1_structure_shift':      h1_conf['h1_structure_shift'],
                'h1_displacement':         h1_conf['h1_displacement'],
                'h1_liquidity_sweep':      h1_conf['h1_liquidity_sweep'],
                # 5min entry trigger features
                'm5_micro_bos':            m5_conf['m5_micro_bos'],
                'm5_sweep_confirmation':   m5_conf['m5_sweep_confirmation'],
                'm5_entry_candle_quality': m5_conf['m5_entry_candle_quality'],
                # 15min approach features
                'approach_distance':       approach['approach_distance'],
                'approach_momentum':       approach['approach_momentum'],
                'candle_body_ratio':       approach['candle_body_ratio'],
                'tf_source':               tf,
                # low <= z_high AND high >= z_low (price entered zone range at all).
                # The old criterion used close strictly inside the OB box, which gives
                # 0–3 for a fresh OB. Training mean=11.24, so value=1 → z=-1.80.
                # Also use tail(200) (all available 4H bars) not tail(50) to match the
                # historical depth that produced mean=11.24.
                'touch_count':             self._calculate_touch_count(
                                               {'type': 'demand' if direction == 'BUY' else 'supply',
                                                'price_high': ob['top'], 'price_low': ob['bottom'],
                                                'index': _ob_15m_index},
                                               _df_15m),
                # when this OB's price range intersects a 4H FVG — matching training
                # where any zone inside a 4H FVG was labeled with h4_fvg_inside=1.
                'fvg_overlap':             fvg_overlap_flag,
            }

            base_prob = self._ml_score_htf_zone(
                {'direction': direction, 'strength': ob['strength'],
                 'index': ob.get('index', len(df) - 1),
                 'mid': ob['mid'], 'top': ob['top'], 'bottom': ob['bottom']},
                df, tf, htf_context=htf_context, zone_override=full_ob_zone,
                df_4h=df)
            # Gold keeps strict 0.55 gate unchanged
            # Forex gets a softer gate during Asia/pre-London (00:00–07:00 UTC)
            if self.symbol == 'XAU/USD':
                _ml_gate = 0.55
            else:
                _hour_utc = datetime.utcnow().hour
                _ml_gate  = 0.45 if 0 <= _hour_utc < 7 else 0.55

            if base_prob < _ml_gate:
                if self.debug:
                    print(f"    ❌ OB  prob={base_prob:.3f} < gate={_ml_gate:.2f}"
                          f"  dir={direction}"
                          f"  strength={ob.get('strength', 0):.2f}"
                          f"  age_bars={ob.get('index', '?')}"
                          f"  mid={ob.get('mid', 0):.5f}")
                htf_ob_rejected['prob'] += 1; continue

            conf_label = (f"OB-{ob['strength']:.2f} "
                          f"1H={'✅' if h1_conf['confirmed'] else '❌'} "
                          f"5m={'✅' if m5_conf['m5_micro_bos'] else '⏳'}")
            htf_setups.append(_build_setup(ob, 'OB', direction, entry, sl, tp_target,
                                           risk, reward, rr, base_prob,
                                           conf_label, confs))

        if self.debug:
            print(f"         FVG rejects → {htf_fvg_rejected}")
            print(f"         OB  rejects → {htf_ob_rejected}")
            print(f"         HTF passing before bias filter: {len(htf_setups)}")

        if htf_context:
            h1 = htf_context.get('h1_trend', 'neutral')
            if h1 == 'bullish':
                # Only keep BUY setups where the zone entry is below current price
                htf_setups = [s for s in htf_setups
                              if s['direction'] == 'BUY' and s['entry'] < current_price]
            elif h1 == 'bearish':
                # Only keep SELL setups where the zone entry is above current price
                htf_setups = [s for s in htf_setups
                              if s['direction'] == 'SELL' and s['entry'] > current_price]

        if self.debug:
            print(f"         HTF after bias filter: {len(htf_setups)}")

        return htf_setups


    def generate_signals(self) -> Dict:
        self.stats['total_iterations'] += 1

        _bars: Dict[str, Optional[pd.DataFrame]] = {
            '15min/200': self.fetch_bars(interval="15min", outputsize=700),
            '1h/100':    self.fetch_bars(interval="1h",    outputsize=100),
            '4h/200':    self.fetch_bars(interval="4h",    outputsize=200),
            '4h/100':    self.fetch_bars(interval="4h",    outputsize=100),
            '5min/60':   self.fetch_bars(interval="5min",  outputsize=60),
            '1day/90':   self.fetch_bars(interval="1day",  outputsize=90),
        }

        df = _bars['15min/200']
        if df is None or len(df) < 50:
            strat = "CONT" if self.is_continuation_strategy else "REV "
            bars  = len(df) if df is not None else 0
            print(f"  ❌ [{self.symbol}|{strat}] 15min fetch returned {bars} bars (need ≥50) — skipping cycle")
            return {'15min_signals': [], 'htf_signals': [], 'stats': {}}

        current_price = df['close'].iloc[-1]
        current_atr   = df['atr'].iloc[-1]
        current_time  = datetime.utcnow()

        expired_count = self.zone_tracker.cleanup_expired(current_time, current_price)
        htf_context   = self.get_htf_context(
            df_1h   = _bars['1h/100'],
            df_4h_c = _bars['4h/100'],
            df_4h   = _bars['4h/200'],
            df_1d   = _bars['1day/90'],
        )

        htf_raw = self.htf_trade_detect("4h", htf_context=htf_context,
                                          df_4h  = _bars['4h/200'],
                                          df_15m = _bars['15min/200'],
                                          df_5m  = _bars['5min/60'],
                                          df_1h  = _bars['1h/100'],
                                          df_d1  = _bars['1day/90'])
        if self.debug:
            print(f"     [HTF] 4H OB signals before prob filter: {len(htf_raw)}")
        # here with zero output — causing the "1 before filter, 0 after" mystery in USD/JPY.
        htf_prob_filtered = []
        for _s in htf_raw:
            _p = float(_s.get('probability', 0))
            if _p >= self.prob_threshold * 100:
                htf_prob_filtered.append(_s)
            elif self.debug:
                print(f"     ⛔ HTF prob filter dropped: {_s.get('direction')} "
                      f"{_s.get('type')} @ {_s.get('entry')} "
                      f"prob={_p:.1f}% < threshold={self.prob_threshold * 100:.1f}%")

        # Deduplicate HTF
        htf_signals = []
        for s in htf_prob_filtered:
            sig_key  = (f"{s.get('timeframe')}|{s.get('direction')}|"
                        f"{s.get('entry'):.5f}|{s.get('sl'):.5f}")
            is_dup, _ = self.zone_tracker.is_duplicate(sig_key, current_time, current_price)
            if not is_dup:
                self.zone_tracker.add_zone(sig_key, {
                    'price_mid': s.get('entry'),
                    'type': 'demand' if s.get('direction') == 'BUY' else 'supply'
                })
                htf_signals.append(s)
            else:
                self.stats['total_duplicates'] += 1

        # Conflict suppressor
        def _suppress_conflicts(signals, cur_atr, h1_trend, pip_size):
            min_sep = cur_atr / pip_size
            buys    = [s for s in signals if s['direction'] == 'BUY']
            sells   = [s for s in signals if s['direction'] == 'SELL']
            remove  = set()
            for b in buys:
                for s in sells:
                    if abs(b['entry'] - s['entry']) / pip_size < min_sep:
                        if h1_trend == 'bullish':   remove.add(id(s))
                        elif h1_trend == 'bearish': remove.add(id(b))
                        else:                       remove.add(id(b)); remove.add(id(s))
            return [s for s in signals if id(s) not in remove]

        htf_signals = _suppress_conflicts(
            htf_signals, current_atr, htf_context.get('h1_trend', 'neutral'), self.pip_size)

        zones_raw = self._detect_zones_with_validation(df, htf_context)
        zones     = self._htf_direction_filter(zones_raw, htf_context, current_price)

        # ── DIAGNOSTIC ──────────────────────────────────────────────────────
        strat_tag = "CONT" if self.is_continuation_strategy else "REV "
        
        current_bar_time = df['time'].iloc[-1]
        if self.last_15min_bar_time != current_bar_time:
            self.tracked_zones_15min   = zones.copy()
            self.last_15min_bar_time   = current_bar_time

        continuation_zones = []
        if self.enable_continuation and self.tracked_zones_15min:
            df_5min = _bars['5min/60']
            if df_5min is not None and len(df_5min) >= 20:
                for zone in self.tracked_zones_15min:
                    flip = self.detect_zone_flip(zone, df_5min, current_atr)
                    if flip:
                        cz = zone.copy()
                        cz['flip_info']       = flip
                        cz['is_continuation'] = True
                        cz['type']            = 'demand' if flip['flip_direction'] == 'BUY' else 'supply'
                        cz['price_mid']       = flip['flip_level']
                        cz['distance_pips']   = abs(current_price - flip['flip_level']) / self.pip_size
                        continuation_zones.append(cz)

        if continuation_zones:
            continuation_zones = self._htf_direction_filter(continuation_zones, htf_context, current_price)

        all_zones = zones + continuation_zones

        # ── ARCHITECTURE GATE: 15min fresh zones are never emitted as signals ──
        # training_labels_clean.parquet: tf_source = 4h_1h_confluence=90%, 4h=10%, 15min=0%
        # Model was never trained on raw 15min zones — live WR confirmed ~7%.
        # Only HTF_SMC signals (4H OBs/FVGs confirmed by 1H) are valid outputs.

        if not htf_signals:
            print(f"     ⚠️  No HTF signals this cycle — returning empty")
            return {'15min_signals': [], 'htf_signals': [], 'stats': {
                'duplicates': 0, 'rejected': 0,
                'expired': expired_count, 'active_zones': self.zone_tracker.get_active_count()
            }}
        # Return only HTF signals — skip all 15min zone scoring below
        return {'15min_signals': [], 'htf_signals': htf_signals, 'stats': {
            'duplicates': self.stats['total_duplicates'],
            'rejected':   self.stats['total_rejected'],
            'expired':    expired_count,
            'active_zones': self.zone_tracker.get_active_count()
        }}


        if self.use_htf_analysis and all_zones:
            all_fvgs = (htf_context.get('h1_fvgs', []) +
                        htf_context.get('h4_fvgs', []) +
                        htf_context.get('daily_fvgs', []))
            all_obs  = (htf_context.get('h1_order_blocks', []) +
                        htf_context.get('h4_order_blocks', []) +
                        htf_context.get('daily_order_blocks', []))
            fib_lvls = {**htf_context.get('h1_fib_levels', {}),
                        **htf_context.get('h4_fib_levels', {})}
            all_htf_items = all_fvgs + all_obs   # used for real overlap count below
            for zone in all_zones:
                conf_dict = self.htf_analyzer.check_confluence(
                    zone, all_fvgs, all_obs, fib_lvls, current_price)
                zone['htf_confluence'] = conf_dict          # display / gate use only

                # this zone's price range — identical to the HTF-OB path at
                # htf_trade_detect() lines ~2247–2259.
                z_top = zone.get('price_high', zone.get('price_mid', current_price))
                z_bot = zone.get('price_low',  zone.get('price_mid', current_price))
                real_overlap = 1  # minimum 1 (the zone itself)
                for item in all_htf_items:
                    item_bot = item.get('bottom', item.get('price_low', 0))
                    item_top = item.get('top',    item.get('price_high', 0))
                    if item_bot < z_top and item_top > z_bot:
                        real_overlap += 1
                zone['confluence'] = real_overlap  # int ≥1, training-compatible
                assert isinstance(zone['confluence'], int), \
                    f"confluence must be int, got {type(zone['confluence'])}"

        before_conf = len(all_zones)
        if self.use_htf_analysis and all_zones:
            all_zones = [z for z in all_zones
                         if z.get('htf_confluence', {}).get('score', 0) >= self.min_confluence_score]

        # ── DIAGNOSTIC ───────────────────────────────────────────────────────
        conf_scores = [z.get('htf_confluence', {}).get('score', 0) for z in (zones + continuation_zones)]
        print(f"     [4] After confluence gate (≥{self.min_confluence_score}): {len(all_zones)} / {before_conf}")
        if conf_scores:
            print(f"         Confluence scores: min={min(conf_scores)} max={max(conf_scores)} avg={sum(conf_scores)/len(conf_scores):.1f}")
        # ── END DIAGNOSTIC ───────────────────────────────────────────────────

        # Dedup 15-min zones
        new_zones = []
        dup_count = 0
        for zone in all_zones:
            sig = self.zone_tracker.get_canonical_signature(zone, self.symbol)
            is_dup, _ = self.zone_tracker.is_duplicate(sig, current_time, current_price)
            if is_dup:
                dup_count += 1
            else:
                new_zones.append(zone)
        self.stats['total_candidates'] += len(all_zones)
        self.stats['total_duplicates'] += dup_count
        all_zones = new_zones

        # ── DIAGNOSTIC ───────────────────────────────────────────────────────
        print(f"     [5] After dedup:               {len(all_zones)}  (dups={dup_count})")
        # ── END DIAGNOSTIC ───────────────────────────────────────────────────

        signals             = []
        observation_signals = []   # zones blocked by gates — saved for study
        rejected_cnt = 0
        reject_reasons = {'5min_refine': 0, 'prob_low': 0, 'risk_range': 0,
                          'h4_bias': 0, 'd1_premium_discount': 0, 'zone_close': 0}

        def _make_obs_signal(zone, gate_reason):
            """Build a lightweight observation signal dict for a gate-blocked zone."""
            z_entry = zone.get('price_high') if zone['type'] == 'demand' else zone.get('price_low', zone.get('price_mid'))
            z_sl    = zone.get('price_low')  if zone['type'] == 'demand' else zone.get('price_high', zone.get('price_mid'))
            z_atr   = current_atr if current_atr else self.pip_size * 10
            z_risk  = max(abs((z_entry or current_price) - (z_sl or current_price)), z_atr * self.min_risk_atr)
            z_tp    = ((z_entry or current_price) + z_risk * 2.0
                       if zone['type'] == 'demand'
                       else (z_entry or current_price) - z_risk * 2.0)
            return {
                'timestamp':        current_time.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                'date':             current_time.strftime('%Y-%m-%d'),
                'time':             current_time.strftime('%H:%M:%S'),
                'symbol':           self.symbol,
                'timeframe':        '15min',
                'model_version':    'v7',
                'strategy':         'CONTINUATION' if self.is_continuation_strategy else 'REVERSION',
                'signal_type':      'observation',
                'direction':        'BUY' if zone['type'] == 'demand' else 'SELL',
                'entry':            round(z_entry or current_price, 5),
                'sl':               round(z_sl    or current_price, 5),
                'tp':               round(z_tp, 5),
                'risk_pips':        round(z_risk / self.pip_size, 1),
                'zone_strength':    round(zone.get('strength', 0), 2),
                'distance_pips':    round(zone.get('distance_pips', 0), 1),
                'confluence_score': zone.get('htf_confluence', {}).get('score', 0),
                'zone_type':        zone['type'],
                'gate_blocked_by':  gate_reason,
                'h4_trend':         h4_trend,
                'd1_zone':          ('premium' if _d1_pd == 1 else 'discount' if _d1_pd == -1 else 'equilibrium'),
                'd1_position':      round(_d1_pos, 3),
                'status':           'observation',
            }

        # ── v7 HARD GATE 1: 4H Bias ───────────────────────────────────────────
        # Recompute h4_structure fresh from live 4H bars every cycle.
        # the trend label could be hours old and wrong side of a structure flip.
        _df_4h_bias = _bars['4h/200']
        if _df_4h_bias is not None and len(_df_4h_bias) > 20:
            _fresh_h4_struct = self.htf_analyzer.detect_market_structure_shift(_df_4h_bias)
            h4_trend = _fresh_h4_struct.get('trend', 'neutral')
        else:
            h4_trend = htf_context.get('h4_structure', {}).get('trend', 'neutral')

        # ── v7 HARD GATE 2: D1 Premium / Discount ─────────────────────────────
        _d1_pd = 0  # default: equilibrium (no gate)
        _df_d1_gate = _bars['1day/90']
        if _df_d1_gate is not None and len(_df_d1_gate) >= 20:
            _tail60  = _df_d1_gate.tail(60)
            _d1_high = float(np.percentile(_tail60["high"], 95))
            _d1_low  = float(np.percentile(_tail60["low"],   5))
            _d1_rng  = _d1_high - _d1_low
            _d1_pos  = (current_price - _d1_low) / _d1_rng if _d1_rng > 0 else 0.5

            # ── D1 GATE DIAGNOSTIC ────────────────────────────────────────────
            print(f"\n  📊 D1 GATE DIAGNOSTIC ({len(_tail60)} bars, 60-day window):")
            print(f"     Bars in tail-60:")
            for _, row in _tail60[['time', 'high', 'low', 'close']].iterrows():
                marker = " ◀ MAX" if row['high'] == _tail60['high'].max() else \
                         " ◀ MIN" if row['low']  == _tail60['low'].min()  else ""
                print(f"       {row['time']}  H={row['high']:.5f}  "
                      f"L={row['low']:.5f}  C={row['close']:.5f}{marker}")
            print(f"     → Raw  max={_tail60['high'].max():.5f}  "
                  f"min={_tail60['low'].min():.5f}")
            print(f"     → P95  high={_d1_high:.5f}  P5 low={_d1_low:.5f}  "
                  f"range={_d1_rng:.5f}")
            print(f"     → Price={current_price:.5f}  "
                  f"Position={_d1_pos:.3f}  "
                  f"Zone={'🔴 PREMIUM' if _d1_pos > 0.6 else '🟢 DISCOUNT' if _d1_pos < 0.4 else '🟡 EQUILIBRIUM'}")
            # ── END DIAGNOSTIC ────────────────────────────────────────────────

            if _d1_rng > 0:
                if _d1_pos > 0.6:
                    _d1_pd = 1   # premium
                elif _d1_pos < 0.4:
                    _d1_pd = -1  # discount

        for zone in all_zones:
            is_cont  = zone.get('is_continuation', False)
            flip_info = zone.get('flip_info')

            # ── Apply 4H bias gate — HARD BLOCK ──────────────────────────────
            # If 4H trend is bullish → block all SELL (supply) signals.
            # If 4H trend is bearish → block all BUY (demand) signals.
            if h4_trend == 'bullish' and zone['type'] == 'supply':
                rejected_cnt += 1
                reject_reasons['d1_premium_discount'] += 1
                observation_signals.append(_make_obs_signal(zone, '4h_bullish_blocks_sell'))
                if self.debug:
                    print(f"     ⛔ 4H BIAS HARD BLOCK: supply (SELL) blocked — 4H trend is bullish")
                continue
            elif h4_trend == 'bearish' and zone['type'] == 'demand':
                rejected_cnt += 1
                reject_reasons['d1_premium_discount'] += 1
                observation_signals.append(_make_obs_signal(zone, '4h_bearish_blocks_buy'))
                if self.debug:
                    print(f"     ⛔ 4H BIAS HARD BLOCK: demand (BUY) blocked — 4H trend is bearish")
                continue
            _h4_counter_trend_penalty = 0.0  # no penalty needed — counter-trend already blocked

            if _d1_pd == 1 and zone['type'] == 'demand':
                # Price in D1 premium — block demand unless 4H is also bullish
                # (if 4H bearish, demand here is doubly counter-trend → block)
                if h4_trend != 'bullish':
                    rejected_cnt += 1
                    reject_reasons['d1_premium_discount'] += 1
                    observation_signals.append(_make_obs_signal(zone, 'd1_premium_blocks_demand'))
                    if self.debug:
                        print(f"     ⛔ D1 PREMIUM GATE: demand zone blocked — price in D1 premium and 4H={h4_trend}")
                    continue
            if _d1_pd == -1 and zone['type'] == 'supply':
                # Price in D1 discount — only block supply when 4H is also bullish
                # (counter-trend sell). If 4H is bearish, supply in D1 discount
                # is a valid continuation sell — do NOT block it.
                if h4_trend == 'bullish':
                    rejected_cnt += 1
                    reject_reasons['d1_premium_discount'] += 1
                    observation_signals.append(_make_obs_signal(zone, 'd1_discount_blocks_supply'))
                    if self.debug:
                        print(f"     ⛔ D1 DISCOUNT GATE: supply zone blocked — price in D1 discount but 4H is bullish (counter-trend)")
                    continue
                elif h4_trend == 'neutral':
                    pass  # neutral 4H = no strong bias, let ML model decide
                # h4_trend == 'bearish': fall through — with-trend continuation sell, allow it

            if 'm5_micro_bos' not in zone:
                _m5_dir = 'BUY' if zone['type'] == 'demand' else 'SELL'
                _m5_conf = self._check_5min_entry(
                    zone.get('price_high', zone.get('price_mid', 0)),
                    zone.get('price_low',  zone.get('price_mid', 0)),
                    _m5_dir,
                    _bars['5min/60'],
                )
                zone['m5_micro_bos']            = _m5_conf['m5_micro_bos']
                zone['m5_sweep_confirmation']   = _m5_conf['m5_sweep_confirmation']
                zone['m5_entry_candle_quality'] = _m5_conf['m5_entry_candle_quality']

            if 'fvg_overlap' not in zone and 'h4_fvg_zone' not in zone:
                _h4_fvgs = htf_context.get('h4_fvgs', [])
                _z_top   = zone.get('price_high', zone.get('price_mid', 0))
                _z_bot   = zone.get('price_low',  zone.get('price_mid', 0))
                zone['fvg_overlap'] = any(
                    not (fvg.get('top', fvg.get('price_high', 0)) < _z_bot or
                         fvg.get('bottom', fvg.get('price_low', 0)) > _z_top)
                    for fvg in _h4_fvgs
                )

            if 'htf_alignment' not in zone:
                if zone['type'] == 'demand':
                    zone['htf_alignment'] = (1  if h4_trend == 'bullish'
                                             else -1 if h4_trend == 'bearish'
                                             else 0)
                else:
                    zone['htf_alignment'] = (1  if h4_trend == 'bearish'
                                             else -1 if h4_trend == 'bullish'
                                             else 0)

            base_probability = self.predict_with_proper_features(
                zone, df, htf_context,
                df_1h = _bars['1h/100'],
                df_4h = _bars['4h/100'],
                df_5m = _bars['5min/60'],
                df_d1 = _bars['1day/90'],
            )
            probability      = base_probability

            zone_key = f"{self.symbol}_{zone.get('type','')}_{round(zone.get('price_mid', 0), 4)}"
            history  = self.zone_history.get(zone_key, {})
            if 'last_outcome' in history:
                zone['prev_zone_outcome']    = history['last_outcome']
                zone['time_since_last_touch'] = min(
                    (datetime.utcnow() - history['last_touch']).total_seconds() / 3600,
                    2000)
            # staleness_penalty REMOVED — no training equivalent.
            # zone_age_bars is already a model feature; the model learned its effect.
            # Post-hoc subtraction double-penalises old zones and breaks calibration.
            zone_age_bars = len(df) - zone.get('index', len(df) - 1)

            # NOTE: confluence boost removed — was never validated in backtesting.
            # Raw base_probability is used directly. The model already learned
            # confluence via the zone_confluence feature during training.

            # NOTE: multi-touch boost/penalty removed — unvalidated post-hoc adjustment.
            # touch_count effect is already captured by zone_touch_count feature in model.

            if is_cont and flip_info:
                probability = self.apply_boost_ceiling(probability, self.continuation_prob_boost)

            # zone geometry instead of hard-rejecting the zone.
            # This was killing every zone in low-volatility / early-session periods.
            refinement = None
            if self.use_5min_refinement and not is_cont:
                refinement = self.refine_entry_with_5min(
                    zone, current_price, current_atr, df_5min=_bars['5min/60'])

                # GUARD: discard refinement if price already inside the zone.
                # A micro swing found behind price is noise, not confirmation.
                # Fall back to zone geometry which uses the proximal body edge.
                if refinement is not None:
                    already_inside = (
                        (zone['type'] == 'demand' and current_price <= zone['price_high']) or
                        (zone['type'] == 'supply' and current_price >= zone['price_low'])
                    )
                    if already_inside:
                        refinement = None
                        if self.debug:
                            print(f"    ⚠️  Refinement discarded — price already inside zone")

                if refinement is None:
                    pass
                else:
                    probability = self.apply_boost_ceiling(probability, refinement['confidence_boost'])

            # (4H counter-trend penalty removed — counter-trend zones are hard-blocked above)

            d1_trend = htf_context.get('daily_structure', {}).get('trend', 'neutral')
            counter_trend = (
                (zone['type'] == 'demand' and d1_trend == 'bearish') or
                (zone['type'] == 'supply' and d1_trend == 'bullish')
            ) if d1_trend != 'neutral' else False

            _hour = datetime.utcnow().hour
            _base_threshold = min(self.prob_threshold, 0.62) if not is_cont else self.prob_threshold
            effective_threshold = _base_threshold + (0.03 if counter_trend else 0.0)
            if base_probability < 0.55 or probability < effective_threshold:
                rejected_cnt += 1
                reject_reasons['prob_low'] += 1
                if self.debug:
                    print(f"     ⛔ Zone rejected: base_prob={base_probability:.3f} "
                          f"prob={probability:.3f} threshold={effective_threshold:.3f} "
                          f"type={zone['type']} dist={zone['distance_pips']:.1f}p")
                continue

            entry, sl, tp, actual_rr = self.calculate_realistic_levels(
                zone, current_atr, current_price, refinement, htf_context)

            risk_pips   = abs(entry - sl) / self.pip_size
            reward_pips = abs(tp   - entry) / self.pip_size
            # Symbol-adaptive risk limits: gold has wider natural ranges
            _is_gold    = 'XAU' in self.symbol.upper() or 'GOLD' in self.symbol.upper()
            _max_risk   = 300.0 if _is_gold else 50.0
            _min_risk   = 5.0   if _is_gold else 2.0
            if risk_pips < _min_risk or risk_pips > _max_risk or reward_pips > 2000:
                rejected_cnt += 1
                reject_reasons['risk_range'] += 1
                if self.debug:
                    print(f"     ⛔ Zone rejected risk range: risk={risk_pips:.1f}p reward={reward_pips:.1f}p")
                continue

            sig = self.zone_tracker.get_canonical_signature(zone, self.symbol)
            zone['probability'] = probability
            self.zone_tracker.add_zone(sig, zone)

            signal = {
                'timestamp':       current_time.strftime('%Y-%m-%d %H:%M:%S') + ' UTC',
                'date':            current_time.strftime('%Y-%m-%d'),
                'time':            current_time.strftime('%H:%M:%S'),
                'symbol':          self.symbol,
                'timeframe':       '15min',
                'model_version':   'v7',
                'signal_type':     'real',
                'strategy':        'CONTINUATION' if self.is_continuation_strategy else 'REVERSION',
                'direction':       'BUY' if zone['type'] == 'demand' else 'SELL',
                'trade_type':      'continuation' if is_cont else 'fresh_zone',
                'entry':           round(entry, 5),
                'sl':              round(sl, 5),
                'tp':              round(tp, 5),
                'risk_pips':       round(risk_pips, 1),
                'reward_pips':     round(reward_pips, 1),
                'risk_reward':     f"1:{actual_rr:.1f}",
                'rr':              f"1:{actual_rr:.1f}",
                'zone_top':        round(zone.get('price_high', entry), 5),
                'zone_bottom':     round(zone.get('price_low',  sl),    5),
                'probability':     round(probability * 100, 1),
                'base_probability': round(base_probability * 100, 1),
                'zone_strength':   round(zone['strength'], 2),
                'distance_pips':   round(zone['distance_pips'], 1),
                'stale':           zone['distance_pips'] > 15,
                'htf_trend':       htf_context.get('h1_trend', 'unknown'),
                'validated':       zone.get('validated', False),
                '5min_refined':    refinement is not None,
                'confluence_score': zone.get('htf_confluence', {}).get('score', 0),
                'counter_trend':   counter_trend,
                'd1_trend':        htf_context.get('daily_structure', {}).get('trend', 'neutral'),
                'h4_trend':        htf_context.get('h4_structure', {}).get('trend', 'neutral'),
                'zone_age_bars':   zone_age_bars,
                'touch_count':     touch_count,
                'session':         _canonical_get_session(datetime.utcnow().hour),
            }
            if is_cont and flip_info:
                signal['continuation_details'] = {
                    'original_type': flip_info['original_type'],
                    'flipped_to':    flip_info['new_type'],
                    'flip_level':    round(flip_info['flip_level'], 5)
                }

            if self.require_zone_close and not self.is_continuation_strategy:
                z_top      = signal['zone_top']
                z_bot      = signal['zone_bottom']
                zone_width = abs(z_top - z_bot)
                _atr       = signal.get('atr', current_atr)
                recent5    = df.tail(5)   # df = _bars['15min/200']

                # Skip for tight zones — close inside is physically improbable
                if zone_width < _atr * 0.5:
                    pass  # zone too tight — skip close requirement
                else:
                    # Accept either: close inside zone, OR wick penetration into zone
                    if zone['type'] == 'supply':
                        touched = any(
                            (z_bot <= c <= z_top) or (h >= z_bot)
                            for c, h in zip(recent5['close'], recent5['high'])
                        )
                    else:  # demand
                        touched = any(
                            (z_bot <= c <= z_top) or (l <= z_top)
                            for c, l in zip(recent5['close'], recent5['low'])
                        )
                    if not touched:
                        if self.debug:
                            print(f"     ⛔ REVERSION zone rejected (require_zone_close): "
                                  f"no close or wick touch of [{z_bot:.5f} – {z_top:.5f}] "
                                  f"in last 5 bars (width={zone_width/self.pip_size:.1f}p, "
                                  f"ATR={_atr/self.pip_size:.1f}p)")
                        rejected_cnt += 1
                        reject_reasons['zone_close'] += 1
                        continue

            signals.append(signal)

        self.stats['total_new_zones'] += len(signals)
        self.stats['total_rejected']  += rejected_cnt
        signals = sorted(signals, key=lambda x: x['probability'], reverse=True)[:5]

        # Save observation signals (gate-blocked) to their own file
        if observation_signals:
            save_signals_to_single_file(observation_signals, "signals/observation_signals.json")

        # ── Conflict filter: remove opposing signals within 1 ATR of each other
        # Keeps highest probability signal when BUY and SELL are at same price level
        h1_trend = htf_context.get('h1_trend', 'neutral')
        signals  = _suppress_conflicts(signals, current_atr, h1_trend, self.pip_size)

        # ── DIAGNOSTIC SUMMARY ──────────────────────────────────────────────
        print(f"     [6] After prob/risk filters:   {len(signals)}")
        print(f"         Reject reasons → 5min_refine={reject_reasons['5min_refine']} "
              f"prob_low={reject_reasons['prob_low']} risk_range={reject_reasons['risk_range']} "
              f"zone_close={reject_reasons['zone_close']}")
        print(f"     ✅ Final signals: {len(signals)} (15min) + {len(htf_signals)} (HTF)")
        # ── END DIAGNOSTIC ───────────────────────────────────────────────────

        return {
            '15min_signals':      signals,
            'htf_signals':        htf_signals,
            'observation_signals': observation_signals,
            'stats': {
                'duplicates':        dup_count,
                'rejected':          rejected_cnt,
                'expired':           expired_count,
                'active_zones':      self.zone_tracker.get_active_count(),
                'observation_count': len(observation_signals),
            }
        }



class WeeklyPlanScheduler:

    def __init__(self, predictors: List['ProductionForexPredictor'],
                 output_file: str = "signals/weekly_plan.json"):
        self.predictors  = predictors
        self.output_file = Path(output_file)

    def generate(self):
        print(f"\n{'═'*80}")
        print(f"📆 WEEKLY PLAN — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'═'*80}")

        weekly_context = {}
        for pred in self.predictors:
            try:
                ctx = pred.get_htf_context()
                # Weekly high/low from 1D
                df_1d = pred.fetch_bars(interval="1day", outputsize=7)
                if df_1d is not None and len(df_1d) >= 5:
                    ctx['weekly_high'] = df_1d.tail(5)['high'].max()
                    ctx['weekly_low']  = df_1d.tail(5)['low'].min()
                weekly_context[pred.symbol] = {
                    'h1_trend':      ctx.get('h1_trend', 'neutral'),
                    'daily_structure': ctx.get('daily_structure', {}),
                    'weekly_high':   ctx.get('weekly_high'),
                    'weekly_low':    ctx.get('weekly_low'),
                    'daily_pivot':   ctx.get('daily_pivot'),
                    'h4_fvg_count':  len(ctx.get('h4_fvgs', [])),
                    'h4_ob_count':   len(ctx.get('h4_order_blocks', [])),
                    'generated_at':  datetime.utcnow().isoformat(),
                    'valid_until':   (datetime.utcnow() + timedelta(days=5)).isoformat()
                }
                print(f"  {pred.symbol:<10} trend={ctx.get('h1_trend','?'):<10} "
                      f"D1={ctx.get('daily_structure',{}).get('trend','?'):<10} "
                      f"4H FVGs={len(ctx.get('h4_fvgs',[])):<4} "
                      f"4H OBs={len(ctx.get('h4_order_blocks',[]))}")
            except Exception as e:
                logger.warning(f"WeeklyPlan: error for {pred.symbol} — {e}")

        self.output_file.parent.mkdir(exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(_convert_numpy(weekly_context), f, indent=2)

        print(f"\n✅ Weekly plan saved → {self.output_file}")
        return weekly_context


class PriceZoneMonitor:

    def __init__(self, predictors: List['ProductionForexPredictor'],
                 signals_file: str = "signals/all_signals.json",
                 alert_callback=None):
        self.predictors       = predictors
        self.signals_file     = Path(signals_file)
        self.alert_callback   = alert_callback or self._default_alert
        self.alerted_signals  = set()

    def _default_alert(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        print(f"\n🚨 ZONE HIT → {emoji} {signal['symbol']} {signal['direction']} "
              f"| Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']} "
              f"| Prob: {signal['probability']}%")

    def check(self):
        if not self.signals_file.exists():
            return

        try:
            with open(self.signals_file, 'r') as f:
                all_sigs = json.load(f)
        except Exception as e:
            logger.warning(f"PriceMonitor: could not load signals — {e}"); return

        # Only watch zones still awaiting entry — resolved/expired signals are skipped.
        plan = [s for s in all_sigs if s.get('status') == 'pending']

        # Group predictors by symbol for current-price lookup
        pred_map = {p.symbol: p for p in self.predictors}
        now      = datetime.utcnow()

        for sig in plan:
            sig_id = f"{sig.get('symbol')}_{sig.get('entry')}_{sig.get('sl')}"
            if sig_id in self.alerted_signals:
                continue

            # Check validity window
            try:
                valid_until = datetime.fromisoformat(sig.get('valid_until', ''))
                if now > valid_until:
                    continue
            except Exception:
                pass

            symbol = sig.get('symbol')
            pred   = pred_map.get(symbol)
            if pred is None:
                continue

            df = pred.fetch_bars(interval="15min", outputsize=5)
            if df is None or len(df) == 0:
                continue

            current_price = df['close'].iloc[-1]
            bar_low       = df['low'].iloc[-1]
            bar_high      = df['high'].iloc[-1]
            direction     = sig.get('direction', 'BUY')

            # ── Step 3: Price must be INSIDE the zone body ───────────────────
            # Use zone_top/zone_bottom stored in the signal (added in _build_setup).
            # Fall back to entry ± 1 pip if zone boundaries are missing (legacy signals).
            zone_top = float(sig.get('zone_top',  sig.get('entry', 0) + pred.pip_size))
            zone_bot = float(sig.get('zone_bottom', sig.get('entry', 0) - pred.pip_size))

            if direction == 'BUY':
                in_zone = (bar_low  <= zone_top and
                           current_price >= zone_bot - pred.pip_size * 2)
            else:
                in_zone = (bar_high >= zone_bot and
                           current_price <= zone_top + pred.pip_size * 2)

            if in_zone:
                self.alerted_signals.add(sig_id)
                self.alert_callback(sig)



def check_signal_outcomes(predictors: List['ProductionForexPredictor'],
                           signals_file: str = "signals/all_signals.json"):
    sym_predictor = {}
    for p in predictors:
        if p.symbol not in sym_predictor:
            sym_predictor[p.symbol] = p

    tracker = SignalOutcomeTrackerV2(signals_file)
    result  = tracker.check_outcomes_by_symbol(sym_predictor)
    if result.get('updated', 0) > 0:
        print(f"  📊 Outcomes updated: TP={result['hit_tp']} SL={result['hit_sl']} "
              f"Expired={result['expired']} | Win rate: {result['win_rate_pct']}%")
    return result



def run_dual_strategy(symbols: Optional[List[str]] = None):
    # One API key per pair — each key handles ~480 calls/day (well under 800 free limit).
    # REV and CONT for the same pair share a key because the shared bar cache means
    # they never double-fetch; the per-key rate limiter still enforces 7 calls/min.
    PAIR_API_KEYS = {
        "EUR/USD": os.getenv("TWELVE_KEY_EURUSD", "ea3e4fce074e403f8cd4c83dd0035643"),
        "GBP/USD": os.getenv("TWELVE_KEY_GBPUSD", "2522b470eba249f88d2e29575035914e"),
        "USD/JPY": os.getenv("TWELVE_KEY_USDJPY", "5bb48b4211144b92a4487ce7c63d1b7a"),
        "XAU/USD": os.getenv("TWELVE_KEY_XAUUSD", "63a513bb7eb848748d28d3deb64c1b59"),
    }
    # Fallback key if a symbol not in the dict above is passed
    _FALLBACK_KEY = os.getenv("TWELVE_KEY_EURUSD", "ea3e4fce074e403f8cd4c83dd0035643")

    # Register spare key(s) any symbol can dynamically borrow from when its
    # own primary key's 7-call/min budget is full. Add more TWELVE_KEY_SPARE_N
    # entries here later if you get more keys.
    _register_spare_keys([
        os.getenv("TWELVE_KEY_SPARE_1", "152db33cd9e64bd4976cbbc0264bc753"),
    ])

    if symbols is None:
        symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]

    SYMBOL_PARAMS = {
        "EUR/USD": dict(max_distance_pips_rev=25.0, max_distance_pips_cont=15.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
        "GBP/USD": dict(max_distance_pips_rev=25.0, max_distance_pips_cont=15.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
        "USD/JPY": dict(max_distance_pips_rev=25.0, max_distance_pips_cont=15.0, zone_distance_threshold=50.0,  min_strength_atr=0.35),
        "XAU/USD": dict(max_distance_pips_rev=80.0, max_distance_pips_cont=50.0, zone_distance_threshold=150.0, min_strength_atr=0.40),
    }
    # Fallback for any symbol not listed above
    DEFAULT_PARAMS = dict(max_distance_pips_rev=25.0, max_distance_pips_cont=15.0, zone_distance_threshold=50.0, min_strength_atr=0.35)

    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║   DUAL STRATEGY MONITOR V7 — SCHEDULED + MULTI-PAIR          ║")
    print(f"║   Pairs: {', '.join(symbols):<52}║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    all_predictors: List[ProductionForexPredictor] = []

    for sym in symbols:
        p   = SYMBOL_PARAMS.get(sym, DEFAULT_PARAMS)
        key = PAIR_API_KEYS.get(sym, _FALLBACK_KEY)
        rev  = ProductionForexPredictor(
            api_key=key,
            model_path="models/model_v5_reversion.pkl",
            scaler_path="models/scaler_v5_reversion.pkl",
            encoders_path="models/encoders_v5_reversion.pkl",
            stats_path="models/training_stats_v5_reversion.pkl",
            symbol=sym,
            prob_threshold=0.62,   # FIX-THRESHOLD: lowered from 0.72 → 0.62
            max_rr=8.0, min_risk_atr=2.5, max_distance_pips=p['max_distance_pips_rev'],  # min_risk_atr: 1.5→2.5
            min_strength_atr=0.50, use_5min_refinement=True,
            refine_max_dist_multiplier=2.0, enable_continuation=False,
            use_htf_analysis=True, min_confluence_score=1,
            zone_ttl_hours=5.8, zone_distance_threshold=p['zone_distance_threshold'],
            htf_fvg_min_gap_atr=0.5, htf_ob_min_reaction_atr=2.5,
            is_continuation_strategy=False, require_zone_close=True, debug=True
        )
        cont = ProductionForexPredictor(
            api_key=key,
            model_path="models/model_v5_continuation.pkl",
            scaler_path="models/scaler_v5_continuation.pkl",
            encoders_path="models/encoders_v5_continuation.pkl",
            stats_path="models/training_stats_v5_continuation.pkl",
            symbol=sym,
            prob_threshold=0.65,
            max_rr=10.0, min_risk_atr=2.5, max_distance_pips=p['max_distance_pips_cont'],  # min_risk_atr: 1.5→2.5
            min_strength_atr=0.50, use_5min_refinement=True,
            refine_max_dist_multiplier=2.0, enable_continuation=True,
            use_htf_analysis=True, min_confluence_score=1,
            zone_ttl_hours=5.8, zone_distance_threshold=p['zone_distance_threshold'],
            htf_fvg_min_gap_atr=0.5, htf_ob_min_reaction_atr=2.5,
            is_continuation_strategy=True, debug=True
        )
        all_predictors.extend([rev, cont])

    weekly_scheduler = WeeklyPlanScheduler(all_predictors)
    price_monitor    = PriceZoneMonitor(all_predictors)

    # Schedule weekly plan on Sunday 20:00 UTC
    schedule.every().sunday.at("20:00").do(weekly_scheduler.generate)

    # Run weekly plan on startup if it's Sunday, otherwise load any existing one
    if datetime.utcnow().weekday() == 6:
        weekly_scheduler.generate()

    print(f"\n🕐 15-min price monitor active | Ctrl-C to stop\n")

    iteration = 0
    rev_predictor  = all_predictors[0]   # first symbol reversion — used for outcome tracking
    cont_predictor = all_predictors[1] if len(all_predictors) > 1 else rev_predictor

    try:
        while True:
            iteration   += 1
            cycle_time   = datetime.utcnow()

            print(f"\n{'═'*100}")
            print(f"CYCLE #{iteration} │ {cycle_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'═'*100}")

            # Run pending scheduled jobs (weekly plan generation)
            schedule.run_pending()

            # Price monitor: check if any pending zone (from all_signals.json) has been touched
            price_monitor.check()

            # Still run full signal generation every cycle for live signals
            # (the sole source of truth — saves to all_signals.json, which
            # PriceZoneMonitor also reads to watch pending zones)
            for i in range(0, len(all_predictors), 2):
                rev_p  = all_predictors[i]
                cont_p = all_predictors[i + 1] if i + 1 < len(all_predictors) else None

                # Pre-warm shared cache for this symbol before both predictors run.
                # REV and CONT will hit cache instead of making duplicate API calls.
                try:
                    rev_p.pre_warm_cache()
                except Exception as e:
                    logger.warning(f"Cycle pre_warm_cache failed for {rev_p.symbol}: {e}")

                rev_result  = rev_p.generate_signals()
                cont_result = cont_p.generate_signals() if cont_p else {'15min_signals': [], 'htf_signals': [], 'stats': {}}

                rev_15min  = rev_result.get('15min_signals', [])
                rev_htf    = rev_result.get('htf_signals', [])
                rev_stats  = rev_result.get('stats', {})
                cont_15min = cont_result.get('15min_signals', [])
                cont_htf   = cont_result.get('htf_signals', [])
                cont_stats = cont_result.get('stats', {})

                sym = rev_p.symbol
                print(f"\n[{sym}] Rev(15m:{len(rev_15min)} HTF:{len(rev_htf)}) | "
                      f"Cont(15m:{len(cont_15min)} HTF:{len(cont_htf)}) | "
                      f"Dup:{rev_stats.get('duplicates',0)+cont_stats.get('duplicates',0)} | "
                      f"Active:{rev_stats.get('active_zones',0)+cont_stats.get('active_zones',0)}")

                all_15min = []
                for sig in rev_15min:  sig['strategy'] = 'REVERSION';    all_15min.append(sig)
                for sig in cont_15min: sig['strategy'] = 'CONTINUATION'; all_15min.append(sig)

                if all_15min:
                    print(f"\n┌─ 15-MIN SIGNALS [{sym}] ({len(all_15min)}) " + "─"*60 + "┐")
                    print("│ Time  │ Strat        │ Type    │ Dir  │ Entry    │ SL       │ TP       │ Risk │ R:R   │ Prob │ Conf │")
                    print("├" + "─"*107 + "┤")
                    for sig in sorted(all_15min, key=lambda x: x['probability'], reverse=True):
                        emoji    = "🟢" if sig['direction'] == 'BUY' else "🔴"
                        t_type   = "CONT" if sig.get('trade_type') == 'continuation' else "FRESH"
                        refined  = "✓"   if sig.get('5min_refined') else ""
                        ct_warn  = "⚠️CT" if sig.get('counter_trend') else "   "
                        print(f"│ {sig['timestamp']} │ {sig['strategy']:<12} │ {t_type:<7} │ "
                              f"{emoji} {sig['direction']:<2} │ {sig['entry']:<8.5f} │ {sig['sl']:<8.5f} │ "
                              f"{sig['tp']:<8.5f} │ {sig['risk_pips']:>4.1f}p │ {sig['rr']:<5} │ "
                              f"{sig['probability']:>4.1f}% │ {sig.get('confluence_score',0)}/10 {refined} {ct_warn} │")
                    print("└" + "─"*107 + "┘")

                all_htf = list({
                    (s['timeframe'], s['direction'], s['entry'], s['sl']): s
                    for s in rev_htf + cont_htf
                }.values())

                if all_htf:
                    print(f"\n┌─ HTF SMC SIGNALS [{sym}] ({len(all_htf)}) " + "─"*65 + "┐")
                    print("│ TF │ Type │ Dir  │ Entry    │ SL       │ TP       │ Risk  │ R:R    │ Prob │ Conf       │")
                    print("├" + "─"*102 + "┤")
                    for sig in sorted(all_htf, key=lambda x: x['probability'], reverse=True)[:10]:
                        emoji = "🟢" if sig['direction'] == 'BUY' else "🔴"
                        print(f"│ {sig['timeframe']:<2} │ {sig['type']:<4} │ "
                              f"{emoji} {sig['direction']:<2} │ {sig['entry']:<8.5f} │ {sig['sl']:<8.5f} │ "
                              f"{sig['tp']:<8.5f} │ {sig['risk_pips']:>5.1f}p │ {sig['rr']:<6} │ "
                              f"{sig['probability']:>4.1f}% │ {sig.get('confluence',''):<10} │")
                    print("└" + "─"*102 + "┘")

                all_sigs = all_15min + all_htf
                if all_sigs:
                    for sig in all_sigs:
                        if not sig.get('timestamp'):
                            sig['timestamp'] = cycle_time.isoformat()
                        sig['last_seen'] = cycle_time.isoformat()
                        if 'valid_until' not in sig or not sig.get('valid_until'):
                            tf = sig.get('timeframe', '15min')
                            if tf in ('1d', '1day', 'daily'):
                                hours = 240   # 10 days
                            elif tf in ('4h', '4hour'):
                                hours = 120   # 5 days
                            elif tf in ('1h', '1hour'):
                                hours = 48    # 2 days
                            else:
                                hours = 8     # 15min/other — stale fast
                            sig['valid_until'] = (cycle_time + timedelta(hours=hours)).isoformat()
                    save_signals_to_single_file(all_sigs, "signals/all_signals.json")

            # written this cycle are visible to the tracker immediately.
            check_signal_outcomes(all_predictors)

            # Per-cycle cap for any downstream Telegram push
            MAX_SIGNALS_PER_CYCLE = 3
            # (Telegram push integration point — add your send function here)

            print(f"\n{'─'*100}")
            print(f"⏰ Next cycle: {(cycle_time + timedelta(minutes=15)).strftime('%H:%M:%S UTC')}")
            print(f"{'─'*100}\n")
            time.sleep(900)

    except KeyboardInterrupt:
        print("\n\n╔════════════════════════════════════════════════════════════════╗")
        print("║                    MONITORING STOPPED                          ║")
        print("╚════════════════════════════════════════════════════════════════╝\n")
        print(f"Total Cycles: {iteration}\n")

        for p in all_predictors:
            strat = "CONTINUATION" if p.is_continuation_strategy else "REVERSION"
            print(f"{strat} [{p.symbol}]  new_zones={p.stats['total_new_zones']}  "
                  f"duplicates={p.stats['total_duplicates']}  "
                  f"rejected={p.stats['total_rejected']}  "
                  f"active={p.zone_tracker.get_active_count()}")

        print(f"\nWalk-forward validated thresholds (2022-Q3 → 2025-Q1):")
        print(f"  • Zone strength: 0.35 ATR  |  Wilder's ATR smoothing (IMPROVE-1)")
        print(f"  • HTF FVG: 0.5 ATR | HTF OB: 2.5 ATR")
        print(f"  • Max boost: Rev 7% | Cont 6%")
        print(f"  • Zone persistence: zones_state.json (IMPROVE-3)")
        print(f"{'─'*80}\n")


if __name__ == "__main__":
    # Runs all four pairs by default.
    # To restrict to a subset, pass an explicit list, e.g.:
    #   run_dual_strategy(symbols=["EUR/USD", "GBP/USD"])
    run_dual_strategy()