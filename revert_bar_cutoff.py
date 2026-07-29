path = "outcome_tracker.py"

old = """                bars_after = df[df[time_col] > ts].copy()
                if "datetime" in bars_after.columns:
                    bars_after = bars_after.rename(columns={"datetime": "time"})

                _tf_hours = {"4h": 4, "4hour": 4, "1h": 1, "1hour": 1,
                             "15min": 0.25, "1d": 24, "daily": 24}.get(tf, 4)
                _cutoff = datetime.utcnow() - pd.Timedelta(hours=_tf_hours)
                bars_after = bars_after[bars_after[time_col] <= _cutoff]

                if bars_after.empty:
                    continue"""

new = """                bars_after = df[df[time_col] > ts].copy()
                if "datetime" in bars_after.columns:
                    bars_after = bars_after.rename(columns={"datetime": "time"})
                if bars_after.empty:
                    continue

                _tf_hours   = {"4h": 4, "4hour": 4, "1h": 1, "1hour": 1,
                               "15min": 0.25, "1d": 24, "daily": 24}.get(tf, 4)
                _closed_cutoff = datetime.utcnow() - pd.Timedelta(hours=_tf_hours)"""

with open(path) as f:
    content = f.read()

if content.count(old) != 1:
    raise SystemExit(f"ANCHOR found {content.count(old)} times, expected 1 - check manually")

content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)

print("Step 1 done: cutoff no longer hides the forming candle from entry detection")
