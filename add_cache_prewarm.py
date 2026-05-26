"""
fix_cache_ttl.py
----------------
Sets interval-appropriate cache TTLs in fetch_bars() so that bars are only
refetched when a new candle could realistically have closed.

Before (all intervals shared 300s TTL):
  Every 15-min cycle expired all cached bars → 8 API calls per symbol per
  cycle → rate limit hit on cycle 2+.

After:
  5min  →   120s  (fetch every cycle — new candle every 5min)
  15min →   840s  (fetch every cycle — slightly under 900s so a new candle
                   is always picked up without fetching twice per candle)
  1h    →  3540s  (fetch once per hour)
  4h    → 14340s  (fetch once per 4 hours)
  1day  → 86400s  (fetch once per day)

This means in a normal 15-min cycle only 15min and 5min bars are ever
refetched. 1H, 4H and 1D are served from cache until their candle closes,
keeping steady-state API usage at 2 real fetches per symbol per cycle
(× 4 pairs = 8 calls/cycle, within the 7-call/min budget with pre-warming).

Usage:
    python fix_cache_ttl.py [path/to/production_predictor.py]

Default target: production_predictor.py in the current directory.
A timestamped .bak backup is written before any edit.
"""

import sys
import shutil
import datetime
from pathlib import Path

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("production_predictor.py")
if not target.exists():
    sys.exit(f"❌  File not found: {target}")

stamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = target.with_suffix(f".{stamp}.bak")
shutil.copy2(target, backup)
print(f"✅  Backup created: {backup}")

src = target.read_text(encoding="utf-8")

OLD = "        cache_duration = 120 if interval == \"5min\" else 300"

NEW = """\
        # Interval-appropriate TTLs — only refetch when a new candle could have closed.
        # 15min/5min are fetched every cycle; 1H/4H/1D are cached for their full duration.
        cache_duration = {
            "5min":  120,     # new candle every 5min  — always fresh
            "15min": 840,     # new candle every 15min — 840s keeps one cycle under TTL
            "1h":    3540,    # new candle every 1hr   — refetch once per hour
            "4h":    14340,   # new candle every 4hrs  — refetch once per 4 hours
            "1day":  86400,   # new candle every 24hrs — refetch once per day
        }.get(interval, 300)"""

if OLD not in src:
    sys.exit("❌  Anchor not found — cache_duration line differs from expected.")

src = src.replace(OLD, NEW, 1)
print("✅  cache_duration updated with interval-appropriate TTLs")

target.write_text(src, encoding="utf-8")
print(f"\n🎉  Done — {target} updated successfully.")
print(f"    Restore at any time with: cp {backup} {target}")
print()
print("Steady-state API calls per 15-min cycle:")
print("  15min/200  — fetched every cycle       (TTL=840s  < 900s cycle)")
print("  5min/100   — fetched every cycle       (TTL=120s  < 900s cycle)")
print("  1h bars    — cached for 1 hour         (TTL=3540s, refetch 1×/hr)")
print("  4h bars    — cached for 4 hours        (TTL=14340s, refetch 1×/4hr)")
print("  1day bars  — cached for 24 hours       (TTL=86400s, refetch 1×/day)")
print()
print("  Normal cycle: ~2 real fetches/symbol × 4 pairs = 8 calls")
print("  Well within the 7-call/min budget with pre-warming spreading them out.")