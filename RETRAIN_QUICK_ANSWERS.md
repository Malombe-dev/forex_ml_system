# YOUR DATA & RETRAINING QUICK REFERENCE

## **YOUR DATA STATUS** ✅

```
EURUSD:   573k rows | 2003-05-04 → 2026-05-04 | ✅ 23.0 years
GBPUSD:   573k rows | 2003-05-11 → 2026-05-11 | ✅ 23.0 years
USDJPY:   208k rows | 2018-01-01 → 2026-05-12 | ✅ 8.4 years
XAUUSD:   174k rows | 2019-01-01 → 2026-05-14 | ✅ 7.4 years

TOTAL: ~1.5M bars | Training: 5,751 labeled zones over 23 years
```

---

## **RETRAINING FREQUENCY ANSWER**

| Period | Frequency | Trigger | Data Download |
|--------|-----------|---------|----------------|
| **Weeks 1-3** | Weekly | Validate P0 fixes | ❌ NO |
| **Month 2+** | Monthly if needed | Win rate <52% | ❌ NO |
| **Every 3 months** | Quarterly | Routine maintenance | ❌ NO |
| **Once per year** | Annually | Best practice | ✅ YES (latest year) |

---

## **DO YOU NEED TO DOWNLOAD DATA AGAIN?**

### **Short Answer: NO** ❌ (for retraining)

- Your 2003–May 2026 data is sufficient forever
- Training uses historical relationships, not today's data
- Adding 1 day won't improve model much

### **Except:** ✅ (for live prediction)

- Live predictor auto-fetches recent bars via TwelveData API
- Fetches are automatic (no manual download needed)
- Your data files stay locked in place

### **Annual Refresh:** ✅ (best practice)

- Once per year (e.g., June 2027), download latest year
- Update: `data/bars_15min_EURUSD.parquet` etc.
- Retrain with updated data
- Ensures long-term model health

---

## **HOW TO RETRAIN** (3 Options)

### **Option 1: Manual Check & Retrain**
```bash
# Check if retrain is needed
source venv/bin/activate
python auto_retrain_monitor.py --check

# If it says "retrain recommended":
python retrain_all.py  # Takes 30–45 min
```

### **Option 2: Automated Weekly Check**
```bash
# Runs every Monday 2am (auto-retrains if needed)
python auto_retrain_monitor.py --schedule &
```

### **Option 3: Scheduled Quarterly**
```bash
# Add to crontab (Linux/Mac)
0 2 1 1 * cd /path/to/forex_ml_system && python retrain_all.py

# Every 1st of month, 2am UTC
```

---

## **WHAT RETRAIN USES**

```
Your existing data files:
  • data/labels/training_labels_clean.parquet (5,751 rows)
  • All bars_*.parquet files (for feature engineering if needed)

Does NOT download anything new — uses what you have
```

---

## **WHEN TRAINING IMPROVES (Upgrade Your Data)**

Only download new data if:

1. **Adding new trading pair** (e.g., EUR/GBP)
   - Download 2003–today for that symbol
   
2. **Data corruption detected** (rare)
   - Regenerate affected symbol from original source
   
3. **Significant market event** (2008 crash, COVID, etc.)
   - Data already in your files → just retrain
   
4. **Annual maintenance** (best practice)
   - Download May 2026 → May 2027
   - Update one batch of files
   - Retrain once per year

---

## **FILES YOU NOW HAVE**

| File | Purpose | Use When |
|------|---------|----------|
| `retrain_all.py` | Full retraining | Every 30–90 days |
| `auto_retrain_monitor.py` | Auto-check win rate | Set & forget |
| `RETRAINING_STRATEGY.md` | Detailed guide | Need details |
| `cleanup_stale_models.sh` | Remove old files | After retrain |
| `FIX_zone_touch_count.py` | Enhancement (optional) | Want +1-2% gains |

---

## **RETRAIN TIMELINE**

```
NOW (Jun 18)
  ↓ Week 1: Run retrain_all.py (validate fixes)
  ↓ Week 2-3: Run weekly checks
  ↓
30 DAYS (Mid-July)
  ↓ Start using auto_retrain_monitor.py --check weekly
  ↓ Retrain only if win_rate < 52%
  ↓
90 DAYS (Mid-September)
  ↓ Quarterly full retrain with 300-month window
  ↓
1 YEAR (Jun 2027)
  ↓ Download May 2026 → May 2027 data
  ↓ Update bars_*.parquet files
  ↓ Run full retrain
```

---

## **QUICK COMMANDS**

```bash
# Check right now
source venv/bin/activate && python auto_retrain_monitor.py --check

# Full retrain (manual)
source venv/bin/activate && python retrain_all.py

# Setup automated weekly checks
source venv/bin/activate
nohup python auto_retrain_monitor.py --schedule > retrain.log 2>&1 &

# View last check
tail -50 retrain.log
```

---

## **SUMMARY**

✅ **Keep your existing data** — it's excellent (23 years for major pairs)  
✅ **No manual downloads needed** for retraining  
✅ **Live data auto-fetched** by predictor (no action needed)  
✅ **Retrain weekly first month**, then monthly if needed, quarterly maintenance  
✅ **Use auto_retrain_monitor.py** to automate the decision  
✅ **Download new data annually** (June 2027) for year-2 refresh

**Your setup is sustainable for years with minimal maintenance.** 🚀
