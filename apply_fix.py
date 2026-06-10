"""
Run this script from your project folder to patch both files in-place:
  python apply_fix.py
"""
import re

# ── Patch production_predictor.py ────────────────────────────────
with open('production_predictor.py', 'r') as f:
    prod = f.read()

old_get_id = re.search(
    r'    def get_signal_id\(signal\):.*?return \(f"\{.*?\}"\)',
    prod, re.DOTALL
)
if old_get_id:
    new_get_id = '''    def get_signal_id(signal):
        """Zone-region identity — symbol-aware bucketed entry, no SL/TP/date.

        Bucket sizes are wider than max ATR-buffer drift between cycles:
          XAU/USD : $5 bucket  → round(entry/5)*5    (ATR drift up to $3)
          JPY     : 0.5 bucket → round(entry*2)/2    (ATR drift ~0.02 JPY)
          FX      : 0.001 bucket → round(entry, 3)   (ATR drift ~0.0002)
        """
        sym   = str(signal.get('symbol', ''))
        entry = float(signal.get('entry', 0))
        if 'XAU' in sym or 'GOLD' in sym.upper():
            entry_bucketed = round(entry / 5) * 5
        elif 'JPY' in sym:
            entry_bucketed = round(entry * 2) / 2
        else:
            entry_bucketed = round(entry, 3)
        return (f"{sym}_{signal.get('direction')}_"
                f"{entry_bucketed}_{signal.get('timeframe', '')}")'''
    prod = prod[:old_get_id.start()] + new_get_id + prod[old_get_id.end():]
    with open('production_predictor.py', 'w') as f:
        f.write(prod)
    print("✅ production_predictor.py patched")
else:
    print("⚠️  get_signal_id not found — may already be patched")

# ── Patch test_predictor.py ───────────────────────────────────────
with open('test_predictor.py', 'r') as f:
    test = f.read()

new_helper = '''    def _get_id(self, s):
        """Mirror the symbol-aware bucketing in save_signals_to_single_file."""
        sym   = str(s.get('symbol', ''))
        entry = float(s.get('entry', 0))
        if 'XAU' in sym or 'GOLD' in sym.upper():
            entry_bucketed = round(entry / 5) * 5
        elif 'JPY' in sym:
            entry_bucketed = round(entry * 2) / 2
        else:
            entry_bucketed = round(entry, 3)
        return (f"{sym}_{s.get('direction')}_"
                f"{entry_bucketed}_{s.get('timeframe', '')}")'''

new_gold_test = '''    def test_gold_same_zone_minor_drift(self):
        """XAU/USD — $5 bucket: $2.1 drift stays in same bucket"""
        s1 = {'symbol': 'XAU/USD', 'direction': 'SELL',
               'entry': 4520.3, 'timeframe': '4h'}
        s2 = {'symbol': 'XAU/USD', 'direction': 'SELL',
               'entry': 4522.4, 'timeframe': '4h'}
        assert self._get_id(s1) == self._get_id(s2)'''

new_jpy_test = '''    def test_jpy_same_zone_minor_drift(self):
        """USD/JPY — 0.5 bucket: 0.06 JPY drift stays in same bucket"""
        s1 = {'symbol': 'USD/JPY', 'direction': 'BUY',
               'entry': 159.41, 'timeframe': '4h'}
        s2 = {'symbol': 'USD/JPY', 'direction': 'BUY',
               'entry': 159.47, 'timeframe': '4h'}
        assert self._get_id(s1) == self._get_id(s2)'''

# Replace _get_id helper
test = re.sub(
    r'    def _get_id\(self.*?return \(f"\{.*?\}"\)',
    new_helper, test, flags=re.DOTALL)

# Replace gold test
test = re.sub(
    r'    def test_gold_same_zone_minor_drift\(self\):.*?assert self\._get_id\(s1\) == self\._get_id\(s2\)',
    new_gold_test, test, flags=re.DOTALL)

# Replace jpy test
test = re.sub(
    r'    def test_jpy_same_zone_minor_drift\(self\):.*?assert self\._get_id\(s1\) == self\._get_id\(s2\)',
    new_jpy_test, test, flags=re.DOTALL)

with open('test_predictor.py', 'w') as f:
    f.write(test)
print("✅ test_predictor.py patched")
print("\nNow run:  pytest test_predictor.py -v")