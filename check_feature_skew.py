#!/usr/bin/env python3
"""
Compare the inputs your models receive LIVE with the distribution they were TRAINED on.

Usage (from your project folder):
    python check_feature_skew.py models/scaler_v6_continuation.pkl
    python check_feature_skew.py models/scaler_v6_reversion.pkl "signals/feature_logs/features_*.jsonl"

It reads the live feature rows that production_predictor.py already logs, then for each of the
26 features reports where the live values sit relative to the training mean/std (z-score) and
whether the live spread is much narrower or wider than in training.

How to read it
  FLAG "shifted"    live mean is more than 1.0 training-std away from the training mean
  FLAG "compressed" live values vary far less than they did in training
  FLAG "stretched"  live values vary far more than they did in training
A feature that is flagged AND important to the model (approach_distance, in your case) means
the model is being fed numbers it never learned from.
"""
import glob
import json
import pickle
import sys

import numpy as np

if len(sys.argv) < 2:
    sys.exit(__doc__)

scaler = pickle.load(open(sys.argv[1], "rb"))
pattern = sys.argv[2] if len(sys.argv) > 2 else "signals/feature_logs/features_*.jsonl"
names = list(scaler.feature_names_in_)

rows = []
for path in sorted(glob.glob(pattern)):
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line)["features"])
            except Exception:
                continue
if not rows:
    sys.exit(f"No feature rows found for pattern: {pattern}")

print(f"{len(rows)} live feature rows from {pattern}\n")
print(f"{'feature':28s} {'train mean':>10s} {'train std':>10s} {'live mean':>10s} {'live std':>9s} {'live z':>7s}  flag")
report = []
for i, name in enumerate(names):
    vals = np.array([r.get(name, np.nan) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        continue
    t_mean, t_std = float(scaler.mean_[i]), float(scaler.scale_[i])
    l_mean, l_std = float(vals.mean()), float(vals.std())
    z = (l_mean - t_mean) / t_std if t_std else 0.0
    ratio = l_std / t_std if t_std else 1.0
    flags = []
    if abs(z) > 1.0:
        flags.append("shifted")
    if len(vals) >= 10 and ratio < 0.25:
        flags.append("compressed")
    if ratio > 3:
        flags.append("stretched")
    report.append((abs(z), name, t_mean, t_std, l_mean, l_std, z, ",".join(flags)))

for _, name, tm, ts, lm, ls, z, flag in sorted(report, reverse=True):
    print(f"{name:28s} {tm:10.3f} {ts:10.3f} {lm:10.3f} {ls:9.3f} {z:7.2f}  {flag}")

bad = [r[1] for r in report if r[7]]
print(f"\n{len(bad)} flagged feature(s): {', '.join(bad) if bad else 'none'}")
