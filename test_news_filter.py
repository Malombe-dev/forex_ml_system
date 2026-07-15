"""
Quick manual test for news_filter.py.
Run from the same directory as news_filter.py:

    python3 test_news_filter.py

What to look at:
1. Does the download succeed? (check for the 💾 success log)
2. Print out the next few upcoming events — do their times look right
   compared to forexfactory.com/calendar for the same event?
3. Try is_news_active() for a pair and see if it matches what you'd
   expect given what's currently on the calendar.
"""

import logging
from datetime import datetime, timezone
from news_filter import ForexFactoryNewsFilter

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Delete signals/ff_calendar.xml first if you want to force a fresh download.
filt = ForexFactoryNewsFilter(
    cache_file="signals/ff_calendar.xml",
    cache_expiry_hours=6,
    source_utc_offset_hours=0,  # <-- adjust this after you check step 2 below
)

print("\n=== Loading events ===")
events = filt.load_events()
print(f"Loaded {len(events)} timed events. last_load_reliable = {filt.last_load_reliable}\n")

print("=== Next 10 upcoming events (any impact) ===")
now = datetime.now(timezone.utc)
upcoming = sorted([e for e in events if e['time_utc'] >= now], key=lambda e: e['time_utc'])
for e in upcoming[:10]:
    print(f"{e['time_utc']}  {e['country']:4s}  {e['impact']:6s}  {e['title']}")

print("\n=== High-impact events only ===")
high = [e for e in events if e['impact'] == 'high']
for e in sorted(high, key=lambda e: e['time_utc'])[:10]:
    print(f"{e['time_utc']}  {e['country']:4s}  {e['title']}")

print("\n=== is_news_active() checks ===")
for pair in ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]:
    active, event = filt.is_news_active(pair, window_minutes=30)
    print(f"{pair}: active={active}  event={event}")