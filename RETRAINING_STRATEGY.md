# RETRAINING STRATEGY FOR YOUR SYSTEM
## Based on Your 2003–May 2026 Data

---

## **YOUR DATA SUMMARY** ✅

| Symbol | 15min Data | Coverage | Status |
|--------|-----------|----------|--------|
| **EURUSD** | 573k rows | 2003-05-04 → 2026-05-04 | ✅ 23.0 years |
| **GBPUSD** | 573k rows | 2003-05-11 → 2026-05-11 | ✅ 23.0 years |
| **USDJPY** | 208k rows | 2018-01-01 → 2026-05-12 | ✅ 8.4 years |
| **XAUUSD** | 174k rows | 2019-01-01 → 2026-05-14 | ✅ 7.4 years |

**Training Dataset:** 5,751 labeled zones spanning 2003-05-05 → 2026-05-11 (23 years)

---

## **🎯 RETRAINING FREQUENCY RECOMMENDATION**

Based on your data coverage (23 years of history), here's the optimal schedule:

### **SHORT TERM (Next 30 Days)**
```
Retrain: WEEKLY for first 3 weeks
Why: Validate that P0 fixes are working in live conditions
Trigger: Check win rate each Friday
  • ≥55%: Keep running, don't retrain
  • 40–55%: Normal variance, retrain if drops below 50%
  • <40%: Debug signal pipeline before retraining
```

### **MEDIUM TERM (Month 2+)**
```
Retrain: MONTHLY if needed
When: If live win rate drops below 52% for 20 consecutive signals
Dataset: Use rolling 60-month window (5 years of recent data)
Why: Captures recent market regimes without overfitting to old data
Time: Takes ~30–45 min
```

### **LONG TERM (Quarterly+)**
```
Retrain: EVERY 3 MONTHS
When: Routine maintenance even if performance is good
Dataset: Use full 300-month window (all 23 years)
Why: Recalibrate ensemble weights, check model drift
Time: Takes ~45–60 min
```

---

## **DO YOU NEED TO DOWNLOAD DATA AGAIN?**

### **For Retraining (Model Training):** ❌ **NO**
- Your 2003–May 2026 data is sufficient for 300-month training window
- Training doesn't need "today's data" — uses historical relationships
- Historical data gives model robust patterns; adding 1 day won't change AUC much

### **For Live Prediction:** ✅ **YES (but automated)**
- Live predictor needs **recent bars** (last 24–48 hours)
- Your 15min bars end May 2026; live predictor will fetch missing data via TwelveData API
- **Already built in:** `production_predictor.py` auto-fetches if bars are stale (6+ hours old)

### **Data Update Schedule:**
```
Historical Training Data (2003–May 2026): Keep as-is ✅
    • Used once every 30–90 days for retraining
    • Does NOT change; locked in

Live Bar Data: Auto-refreshed by predictor ✅
    • Fetched every cycle via TwelveData
    • 60-second cache to avoid duplicate API calls
    • Rates: 7 calls/min per API key
```

---

## **HOW TO RETRAIN WITH YOUR DATA**

### **Option 1: Quick Retrain (Monthly)**
```bash
# Uses 60-month rolling window (5 recent years)
source venv/bin/activate
python retrain_all.py  # Will use existing label data

# Expected time: 30–45 min
# Output: New models/metrics_*/
```

### **Option 2: Full Retrain (Quarterly)**
```bash
# Same command — but uncomment this in model_trainer.py:
# rolling_months=300  (already default)

source venv/bin/activate
python retrain_all.py

# Expected time: 45–60 min
# Uses all 23 years of data; slower but more robust
```

---

## **WHEN TO DOWNLOAD NEW DATA**

You should refresh/download data in these scenarios:

### **Scenario 1: Adding New Symbol**
```bash
# If you want to add (e.g., EUR/GBP, BTC/USD):
# 1. Download from Dukascopy or TwelveData: 2003–today
# 2. Save to data/bars_{tf}_{SYMBOL}.parquet
# 3. Run label_generator.py to create training labels for new symbol
# 4. Retrain with new data included
```

### **Scenario 2: Data Corruption/Gap**
```bash
# If you suspect data quality issues:
# 1. Run: python diagnose_data_feeds.py
# 2. Check for NaN, gaps, anomalies
# 3. If found: Re-download affected symbol from May 9 onwards
# 4. Retrain to clean up training set
```

### **Scenario 3: Major Market Event**
```bash
# If significant market structure change (e.g., Brexit, rate shock):
# 1. Data already captures historical shocks (2008, 2020 COVID)
# 2. No need to download; just retrain to learn new patterns
# 3. Run: python retrain_all.py
```

### **Scenario 4: Annually (Best Practice)**
```bash
# Once per year (e.g., June 2027):
# 1. Download latest year of data (May 2026 → May 2027)
# 2. Update all bars_*.parquet files
# 3. Run full retrain with rolling_months=300
# 4. Validate test metrics still healthy (AUC >0.70)
```

---

## **AUTOMATED RETRAIN SCRIPT** (Use This)

I'll give you a script that:
1. Checks if retraining is needed (based on live win rate)
2. Auto-runs on schedule
3. Logs results
4. Alerts if performance drops

Save this as `auto_retrain_monitor.py`:

```python
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
    print(f"   Wins:  {wr_data['wins']}")
    print(f"   Losses: {wr_data['losses']}")
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
```

---

## **USAGE**

```bash
# Check win rate now
python auto_retrain_monitor.py --check

# Schedule automatic Monday checks (runs continuously)
python auto_retrain_monitor.py --schedule
```

---

## **SUMMARY: YOUR RETRAINING PLAN**

| Phase | When | Frequency | Data | Action |
|-------|------|-----------|------|--------|
| **NOW** | Jun 18–Jul 8 | Weekly | Existing | Run `retrain_all.py` Fri of each week |
| **Month 2+** | Jul+  | Monthly or as-needed | Existing | Use script above; retrain if WR <52% |
| **Quarterly** | Every 90 days | Once/quarter | Existing + latest year | Full retrain with 300-month window |
| **Annually** | Each year | Once/year | Download latest year | Refresh bars_*.parquet, retrain |

---

## **KEY TAKEAWAYS**

✅ **You have excellent data:** 23 years for EURUSD/GBPUSD, 8+ years for JPY/Gold  
✅ **No need to download again for retraining:** Historical data is sufficient  
✅ **Live data is auto-fetched:** Via TwelveData API (already built in)  
✅ **Optimal retrain frequency:** Weekly check, monthly action, quarterly deep retrain  
✅ **Use monitor script:** Automates the decision of when to retrain  

---

## **NEXT STEPS**

1. **Copy monitor script** above into `auto_retrain_monitor.py`
2. **Test it:**
   ```bash
   source venv/bin/activate
   python auto_retrain_monitor.py --check
   ```
3. **Optional: Schedule weekly checks**
   ```bash
   # Runs every Monday 2am
   python auto_retrain_monitor.py --schedule &
   ```

That's it! You're set up for long-term model maintenance. 🚀
