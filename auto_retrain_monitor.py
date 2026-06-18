#!/usr/bin/env python3
"""
AUTO-RETRAIN MONITOR
====================
Monitors live win rate and triggers retrain when needed.

Usage:
    python auto_retrain_monitor.py --check       # Check now
    python auto_retrain_monitor.py --schedule    # Run on schedule (every Monday 2am)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import argparse
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SIGNALS_FILE = Path("signals/all_signals.json")
MIN_SIGNAL_COUNT = 20


def calculate_win_rate():
    """Calculate win rate from recent signals."""
    if not SIGNALS_FILE.exists():
        logger.warning(f"Signals file not found: {SIGNALS_FILE}")
        return None

    try:
        with open(SIGNALS_FILE) as f:
            signals = json.load(f)

        if not isinstance(signals, list):
            signals = [signals]

        # Last 20 signals only
        recent = signals[-MIN_SIGNAL_COUNT:] if len(signals) >= MIN_SIGNAL_COUNT else signals

        resolved = [s for s in recent if s.get('status') in ['hit_tp', 'hit_sl']]
        if not resolved:
            logger.warning(f"Not enough resolved signals yet ({len(resolved)}/{MIN_SIGNAL_COUNT})")
            return None

        wins = len([s for s in resolved if s.get('status') == 'hit_tp'])
        win_rate = wins / len(resolved)

        return {
            'win_rate': win_rate,
            'wins': wins,
            'losses': len(resolved) - wins,
            'total': len(resolved),
            'last_signal_time': recent[-1].get('timestamp', 'unknown')
        }

    except Exception as e:
        logger.error(f"Error calculating win rate: {e}")
        return None


def should_retrain(win_rate_data):
    """Determine if retraining is needed."""
    if not win_rate_data:
        return False, "Not enough data"

    wr = win_rate_data['win_rate']

    if wr >= 0.55:
        return False, f"✅ Win rate {wr:.1%} — no retrain needed"
    elif wr >= 0.50:
        return False, f"⚠️ Win rate {wr:.1%} — monitor next 10 signals"
    elif wr >= 0.45:
        return True, f"⚠️ Win rate {wr:.1%} — retrain recommended"
    else:
        return True, f"❌ Win rate {wr:.1%} — RETRAIN REQUIRED"


def run_retrain():
    """Execute retraining."""
    logger.info("Starting retraining...")
    try:
        result = subprocess.run(
            ["python", "retrain_all.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )

        if result.returncode == 0:
            logger.info("✅ Retraining completed successfully")
            return True
        else:
            logger.error(f"❌ Retraining failed:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ Retraining timed out (>1 hour)")
        return False
    except Exception as e:
        logger.error(f"❌ Error running retrain: {e}")
        return False


def check_and_report():
    """Check win rate and report."""
    logger.info("=" * 70)
    logger.info("RETRAIN MONITOR CHECK")
    logger.info("=" * 70)

    wr_data = calculate_win_rate()

    if not wr_data:
        print("\n⏳ Waiting for more signals (need 20+ resolved trades)")
        return False

    print(f"\n📊 Recent Performance (last {wr_data['total']} signals):")
    print(f"   Wins:     {wr_data['wins']}")
    print(f"   Losses:   {wr_data['losses']}")
    print(f"   Win rate: {wr_data['win_rate']:.1%}")
    print(f"   Last signal: {wr_data['last_signal_time']}")

    should_retrain_flag, reason = should_retrain(wr_data)

    print(f"\n{reason}")

    if should_retrain_flag:
        print("\n🔄 Triggering retrain...")
        success = run_retrain()
        return success

    return False


def schedule_weekly():
    """Schedule retrain check every Monday at 2am."""
    try:
        import schedule
        import time

        def job():
            logger.info("Running scheduled retrain check...")
            check_and_report()

        schedule.every().monday.at("02:00").do(job)

        logger.info("✅ Scheduled: Every Monday 02:00 UTC")
        logger.info("Press Ctrl+C to stop")

        while True:
            schedule.run_pending()
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Schedule stopped")
    except ImportError:
        logger.error("schedule package not found. Run: pip install schedule")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Check now')
    parser.add_argument('--schedule', action='store_true', help='Run on schedule')
    args = parser.parse_args()

    if args.schedule:
        schedule_weekly()
    else:
        check_and_report()


if __name__ == "__main__":
    main()
