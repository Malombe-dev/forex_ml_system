"""
generate_outcomes.py  — LEAKAGE-FREE VERSION
=============================================
Produces outcomes.parquet using ONLY the held-out test set:
  - Loads all label parquets and sorts by date
  - Splits by the SAME calendar dates used in model_trainer.py
  - Runs ensemble prediction ONLY on test rows
  - Never touches train or val rows

ROOT CAUSE OF OLD LEAKAGE (fixed here):
  The old script loaded ALL labels, ran prediction on all of them,
  then took the last 30% by row index as the "test set". Because the
  model was trained on 70% of those same rows, it had memorised them,
  producing 95–98% win rates. This script eliminates that entirely.

Usage:
  python generate_outcomes.py

Outputs:
  data/outcomes/outcomes_full.parquet  — all rows (for analysis only)
  data/outcomes/outcomes.parquet       — TEST SET ONLY (for backtesting)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS
from model_trainer import TRAIN_END_DATE, VAL_END_DATE   # shared date boundaries

# ── Config ────────────────────────────────────────────────────────────────────
PAIRS     = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
PIP_SIZES = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01, 'XAUUSD': 1.0}
WEIGHTS   = {'xgb': 0.5, 'rf': 0.3, 'ridge': 0.2}

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading v5 reversion ensemble...")
xgb_model = pickle.load(open('models/model_v5_reversion.pkl',   'rb'))
rf_model  = pickle.load(open('models/rf_v5_reversion.pkl',      'rb'))
lr_model  = pickle.load(open('models/lr_v5_reversion.pkl',      'rb'))
scaler    = pickle.load(open('models/scaler_v5_reversion.pkl',  'rb'))
encoders  = pickle.load(open('models/encoders_v5_reversion.pkl','rb'))
print(f"  Encoders loaded for: {list(encoders.keys())}")

# ── Load and sort ALL labels ──────────────────────────────────────────────────
dfs = []
for pair in PAIRS:
    df = pd.read_parquet(f'data/labels/training_labels_{pair}.parquet')
    df['symbol'] = pair
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined['zone_time'] = pd.to_datetime(combined['zone_time'])
combined = combined.sort_values('zone_time').reset_index(drop=True)
print(f"Loaded {len(combined):,} rows across {len(PAIRS)} pairs")
print(f"  Full date range: {combined['zone_time'].min().date()} → {combined['zone_time'].max().date()}")

# ── Apply the SAME date split used in model_trainer.py ───────────────────────
# CRITICAL: the model was trained on rows where zone_time < TRAIN_END_DATE.
# Only rows where zone_time >= VAL_END_DATE are truly out-of-sample test rows.
# Using any earlier rows here would reintroduce leakage.
test_mask = combined['zone_time'] >= VAL_END_DATE
test_df   = combined[test_mask].copy()

train_count = (combined['zone_time'] < TRAIN_END_DATE).sum()
val_count   = ((combined['zone_time'] >= TRAIN_END_DATE) & (combined['zone_time'] < VAL_END_DATE)).sum()
test_count  = test_mask.sum()

print(f"\nDate-based split (same boundaries as model_trainer.py):")
print(f"  Train : < {TRAIN_END_DATE.date()}  ({train_count:,} rows)  ← model WAS trained on these")
print(f"  Val   : {TRAIN_END_DATE.date()} – {VAL_END_DATE.date()}  ({val_count:,} rows)  ← used for early stopping")
print(f"  Test  : >= {VAL_END_DATE.date()}  ({test_count:,} rows)  ← ONLY these scored below")
print(f"\n⚠️  Scoring {test_count:,} test rows only (model never saw these during training)")

if test_count == 0:
    raise ValueError(
        f"Test set is empty! VAL_END_DATE={VAL_END_DATE.date()} is after all data. "
        f"Adjust VAL_END_DATE in model_trainer.py."
    )

# ── Encode categoricals on test set only ─────────────────────────────────────
for col in CATEGORICAL_FEATURES:
    enc_col = f"{col}_encoded"
    le      = encoders[col]
    vals    = test_df[col].astype(str)
    encoded = []
    for v in vals:
        if v in le.classes_:
            encoded.append(int(le.transform([v])[0]))
        else:
            encoded.append(0)
    test_df[enc_col] = encoded

# ── Run ensemble prediction on test rows only ─────────────────────────────────
feature_cols = list(FEATURE_COLUMNS)
missing = [c for c in feature_cols if c not in test_df.columns]
if missing:
    raise ValueError(f"Test set missing feature columns: {missing}")

X        = test_df[feature_cols].values.astype(float)
X_scaled = scaler.transform(X)      # transform only — scaler was fit on train

p_xgb = xgb_model.predict_proba(X_scaled)[:, 1]
p_rf  = rf_model.predict_proba(X_scaled)[:, 1]
p_lr  = lr_model.predict_proba(X_scaled)[:, 1]
p_ens = (p_xgb * WEIGHTS['xgb'] +
         p_rf  * WEIGHTS['rf']  +
         p_lr  * WEIGHTS['ridge'])
p_ens = np.clip(p_ens, 1e-7, 1 - 1e-7)

test_df['predicted_prob'] = p_ens
print(f"\nProbability range on test set: {p_ens.min():.3f} – {p_ens.max():.3f}  mean={p_ens.mean():.3f}")

# ── Build outcomes DataFrame ──────────────────────────────────────────────────
rows = []
geometry_errors = 0

for _, r in test_df.iterrows():
    sym      = r['symbol']
    pip_size = PIP_SIZES.get(sym, 0.0001)
    is_win   = r['label'] == 1.0
    entry    = float(r['entry'])
    sl       = float(r['sl'])
    tp       = float(r['tp'])

    direction = 'BUY' if r['zone_type'] == 'demand' else 'SELL'

    if direction == 'BUY' and not (tp > entry > sl):
        geometry_errors += 1
        continue
    if direction == 'SELL' and not (sl > entry > tp):
        geometry_errors += 1
        continue

    rows.append({
        'signal_timestamp': r['zone_time'],
        'resolved_at':      r['zone_time'],
        'symbol':           sym,
        'direction':        direction,
        'entry':            entry,
        'sl':               sl,
        'tp':               tp,
        'exit_price':       tp if is_win else sl,
        'predicted_prob':   float(r['predicted_prob']),
        'outcome':          'win' if is_win else 'loss',
        'strategy':         'REVERSION',
        'pip_size':         pip_size,
        'bars_held':        0,
    })

if geometry_errors:
    print(f"⚠️  Skipped {geometry_errors} rows with invalid geometry")

out = pd.DataFrame(rows)
out = out.sort_values('signal_timestamp').reset_index(drop=True)

# ── EV calibration preview ────────────────────────────────────────────────────
print(f"\nTest set outcomes: {len(out):,}")
print(f"Overall win rate: {(out['outcome']=='win').mean():.1%}")
print(f"  (Expect 50–60% for a real model — NOT 95%+)")

def get_pip_val(r, col):
    pip = PIP_SIZES.get(r['symbol'], 0.0001)
    if r['direction'] == 'BUY':
        return (r['tp'] - r['entry']) / pip if col == 'tp' else (r['entry'] - r['sl']) / pip
    else:
        return (r['entry'] - r['tp']) / pip if col == 'tp' else (r['sl'] - r['entry']) / pip

print(f"\n  {'Bucket':<10} {'N':>6} {'WR':>7} {'AvgTP':>7} {'AvgSL':>7} {'EV':>8}")
print("  " + "─" * 50)
for lo, hi, label in [
    (0.0,  0.50, "<50%"),
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 0.75, "70-75%"),
    (0.75, 1.00, "75%+"),
]:
    mask     = (out['predicted_prob'] >= lo) & (out['predicted_prob'] < hi)
    bucket   = out[mask]
    n        = len(bucket)
    if n == 0:
        continue
    wins_b   = bucket[bucket['outcome'] == 'win']
    losses_b = bucket[bucket['outcome'] == 'loss']
    wr       = len(wins_b) / n
    avg_tp   = wins_b.apply(lambda r: get_pip_val(r,'tp'), axis=1).mean() if len(wins_b)   else 0
    avg_sl   = losses_b.apply(lambda r: get_pip_val(r,'sl'), axis=1).mean() if len(losses_b) else 0
    ev       = (wr * avg_tp) - ((1 - wr) * avg_sl)
    flag     = '✅' if ev > 0 else '❌'
    print(f"  {flag} {label:<9} {n:>6,}  {wr:>6.1%}  {avg_tp:>7.1f}  {avg_sl:>7.1f}  {ev:>+8.1f}p")

# ── Save ──────────────────────────────────────────────────────────────────────
Path('data/outcomes').mkdir(exist_ok=True)

# Save full combined dataset (all rows, with predictions only on test set)
# Useful for analysis — but DO NOT use for backtesting (train rows have no predictions)
out.to_parquet('data/outcomes/outcomes_full.parquet', index=False)
print(f"\n✅ Test outcomes → data/outcomes/outcomes_full.parquet  ({len(out):,} rows)")

# outcomes.parquet IS the test set (no further slicing needed)
out.to_parquet('data/outcomes/outcomes.parquet', index=False)
print(f"✅ Test outcomes → data/outcomes/outcomes.parquet  ({len(out):,} rows)")
print(f"   Date range: {pd.to_datetime(out['signal_timestamp']).min().date()} → "
      f"{pd.to_datetime(out['signal_timestamp']).max().date()}")

print("\n" + "─" * 60)
print("LEAKAGE CHECK SUMMARY")
print("─" * 60)
print(f"  Model trained on:  zone_time < {TRAIN_END_DATE.date()}")
print(f"  Outcomes scored:   zone_time >= {VAL_END_DATE.date()}")
print(f"  Overlap possible:  NO — {(VAL_END_DATE - TRAIN_END_DATE).days} day gap between train end and test start")
print("─" * 60)
print("\nNext: python backtest_engine.py --source outcomes --balance 10000 --risk 1")
