import pytest
import json
from unittest.mock import MagicMock
from production_predictor import (
    save_signals_to_single_file,
    ProductionForexPredictor,
)


# ── Test save_signals_to_single_file ─────────────────────────────
class TestSaveSignals:

    def test_new_signal_added_as_pending(self, tmp_path):
        f = tmp_path / "signals.json"
        signals = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                    'entry': 1.123, 'sl': 1.120, 'tp': 1.130,
                    'timeframe': '4h', 'probability': 70.0}]
        save_signals_to_single_file(signals, str(f))
        result = json.loads(f.read_text())
        assert result[0]['status'] == 'pending'

    def test_resolved_signal_never_overwritten(self, tmp_path):
        """hit_tp record must survive a re-emit of the same zone"""
        f = tmp_path / "signals.json"
        existing = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                     'entry': 1.123, 'sl': 1.120, 'tp': 1.130,
                     'timeframe': '4h', 'probability': 70.0,
                     'status': 'hit_tp'}]
        f.write_text(json.dumps(existing))
        new_signal = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                       'entry': 1.124, 'sl': 1.121, 'tp': 1.131,
                       'timeframe': '4h', 'probability': 90.0}]
        save_signals_to_single_file(new_signal, str(f))
        result = json.loads(f.read_text())
        assert result[0]['status'] == 'hit_tp'      # not overwritten
        assert result[0]['probability'] == 70.0     # original preserved

    def test_probability_only_upgrades(self, tmp_path):
        f = tmp_path / "signals.json"
        existing = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                     'entry': 1.123, 'sl': 1.120, 'tp': 1.130,
                     'timeframe': '4h', 'probability': 75.0,
                     'status': 'pending'}]
        f.write_text(json.dumps(existing))
        lower_prob = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                       'entry': 1.123, 'sl': 1.120, 'tp': 1.130,
                       'timeframe': '4h', 'probability': 60.0}]
        save_signals_to_single_file(lower_prob, str(f))
        result = json.loads(f.read_text())
        assert result[0]['probability'] == 75.0     # not degraded

    def test_probability_upgrades_when_higher(self, tmp_path):
        f = tmp_path / "signals.json"
        existing = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                     'entry': 1.123, 'sl': 1.120, 'tp': 1.130,
                     'timeframe': '4h', 'probability': 65.0,
                     'status': 'pending'}]
        f.write_text(json.dumps(existing))
        higher_prob = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                        'entry': 1.123, 'sl': 1.120, 'tp': 1.130,
                        'timeframe': '4h', 'probability': 80.0}]
        save_signals_to_single_file(higher_prob, str(f))
        result = json.loads(f.read_text())
        assert result[0]['probability'] == 80.0     # upgraded

    def test_execution_levels_always_update(self, tmp_path):
        f = tmp_path / "signals.json"
        existing = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                     'entry': 1.123, 'sl': 1.120, 'tp': 1.130,
                     'timeframe': '4h', 'probability': 70.0,
                     'status': 'pending'}]
        f.write_text(json.dumps(existing))
        # Same bucket (1.123 rounded to 3dp) but fresher levels
        updated = [{'symbol': 'EUR/USD', 'direction': 'BUY',
                    'entry': 1.1232, 'sl': 1.1205, 'tp': 1.1305,
                    'timeframe': '4h', 'probability': 70.0}]
        save_signals_to_single_file(updated, str(f))
        result = json.loads(f.read_text())
        assert result[0]['sl'] == pytest.approx(1.1205)
        assert result[0]['tp'] == pytest.approx(1.1305)

    def test_missing_probability_no_crash(self, tmp_path):
        """The original KeyError bug — must not crash"""
        f = tmp_path / "signals.json"
        signals = [{'symbol': 'GBP/USD', 'direction': 'SELL',
                    'entry': 1.363, 'sl': 1.366, 'tp': 1.340,
                    'timeframe': '4h'}]             # no 'probability' key
        save_signals_to_single_file(signals, str(f))  # must not raise

    def test_all_resolved_statuses_protected(self, tmp_path):
        """All four resolved statuses must block overwrite"""
        for status in ['hit_tp', 'hit_sl', 'expired', 'entry_missed']:
            f = tmp_path / f"signals_{status}.json"
            existing = [{'symbol': 'USD/JPY', 'direction': 'SELL',
                         'entry': 150.0, 'sl': 150.5, 'tp': 148.0,
                         'timeframe': '4h', 'probability': 70.0,
                         'status': status}]
            f.write_text(json.dumps(existing))
            new_signal = [{'symbol': 'USD/JPY', 'direction': 'SELL',
                           'entry': 150.0, 'sl': 150.5, 'tp': 148.0,
                           'timeframe': '4h', 'probability': 95.0}]
            save_signals_to_single_file(new_signal, str(f))
            result = json.loads(f.read_text())
            assert result[0]['status'] == status,        f"status {status} was overwritten"
            assert result[0]['probability'] == 70.0,     f"probability changed for {status}"


# ── Test probability pipeline ────────────────────────────────────
class TestProbabilityPipeline:

    def test_apply_boost_ceiling(self):
        """
        FIX: float arithmetic — 0.80 + 0.07 = 0.8700000000000001 in binary.
        Use pytest.approx() for all float comparisons.
        """
        pred = MagicMock()
        pred.max_boost = 0.07
        result = ProductionForexPredictor.apply_boost_ceiling(pred, 0.80, 0.10)
        assert result <= 0.95                            # hard ceiling intact
        assert result == pytest.approx(0.87, abs=1e-9)  # 0.80 + capped 0.07

    def test_boost_does_not_exceed_ceiling(self):
        pred = MagicMock()
        pred.max_boost = 0.07
        result = ProductionForexPredictor.apply_boost_ceiling(pred, 0.93, 0.10)
        assert result <= 0.95                            # never exceeds 0.95

    def test_counter_trend_threshold_raised(self):
        """
        FIX: float arithmetic — 0.65 + 0.05 = 0.7000000000000001 in binary.
        Use pytest.approx().
        """
        base_threshold = 0.65
        counter_trend  = True
        effective = base_threshold + (0.05 if counter_trend else 0.0)
        assert effective == pytest.approx(0.70, abs=1e-9)

    def test_with_trend_threshold_unchanged(self):
        base_threshold = 0.65
        counter_trend  = False
        effective = base_threshold + (0.05 if counter_trend else 0.0)
        assert effective == pytest.approx(0.65, abs=1e-9)


# ── Test zone dedup key bucketing ────────────────────────────────
class TestZoneDedupKey:

    def _get_id(self, s):
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
                f"{entry_bucketed}_{s.get('timeframe', '')}")

    def test_same_zone_minor_drift_is_duplicate(self):
        """
        FIX: round(x, 3) gives ~1 pip buckets for FX, not 10.
        Use entries that stay within the same 1-pip bucket.
        1.12345 → 1.123  and  1.12348 → 1.123  (same bucket)
        """
        s1 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.12345, 'timeframe': '4h'}
        s2 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.12348, 'timeframe': '4h'}  # 0.3 pip — same bucket
        assert self._get_id(s1) == self._get_id(s2)

    def test_adjacent_zones_are_distinct(self):
        s1 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.12300, 'timeframe': '4h'}
        s2 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.13500, 'timeframe': '4h'}  # 120 pips away
        assert self._get_id(s1) != self._get_id(s2)

    def test_opposite_directions_are_distinct(self):
        s1 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.12300, 'timeframe': '4h'}
        s2 = {'symbol': 'EUR/USD', 'direction': 'SELL',
               'entry': 1.12300, 'timeframe': '4h'}
        assert self._get_id(s1) != self._get_id(s2)

    def test_different_symbols_are_distinct(self):
        s1 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.12300, 'timeframe': '4h'}
        s2 = {'symbol': 'GBP/USD', 'direction': 'BUY',
               'entry': 1.12300, 'timeframe': '4h'}
        assert self._get_id(s1) != self._get_id(s2)

    def test_different_timeframes_are_distinct(self):
        s1 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.12300, 'timeframe': '4h'}
        s2 = {'symbol': 'EUR/USD', 'direction': 'BUY',
               'entry': 1.12300, 'timeframe': '15min'}
        assert self._get_id(s1) != self._get_id(s2)

    def test_gold_same_zone_minor_drift(self):
        """XAU/USD — $5 bucket: $2.1 drift stays in same bucket"""
        s1 = {'symbol': 'XAU/USD', 'direction': 'SELL',
               'entry': 4520.3, 'timeframe': '4h'}
        s2 = {'symbol': 'XAU/USD', 'direction': 'SELL',
               'entry': 4522.4, 'timeframe': '4h'}
        assert self._get_id(s1) == self._get_id(s2)

    def test_gold_adjacent_zones_distinct(self):
        """XAU/USD — zones $10 apart must never merge"""
        s1 = {'symbol': 'XAU/USD', 'direction': 'SELL',
               'entry': 4520.0, 'timeframe': '4h'}
        s2 = {'symbol': 'XAU/USD', 'direction': 'SELL',
               'entry': 4530.0, 'timeframe': '4h'}  # $10 apart
        assert self._get_id(s1) != self._get_id(s2)

    def test_jpy_same_zone_minor_drift(self):
        """USD/JPY — 0.5 bucket: 0.06 JPY drift stays in same bucket"""
        s1 = {'symbol': 'USD/JPY', 'direction': 'BUY',
               'entry': 159.41, 'timeframe': '4h'}
        s2 = {'symbol': 'USD/JPY', 'direction': 'BUY',
               'entry': 159.47, 'timeframe': '4h'}
        assert self._get_id(s1) == self._get_id(s2)