"""
generate_outcomes.py  — LEAKAGE-FREE VERSION  (v3: REVERSION + CONTINUATION)
==============================================================================
Produces outcomes.parquet using ONLY the held-out test set.

STRATEGIES
----------
  REVERSION    : demand zone → BUY, supply zone → SELL
                 Label column : 'label'
                 Models       : model_v6_reversion.pkl

  CONTINUATION : trades in direction of HTF trend
                 Label column : 'label_cont'
                 Models       : model_v6_continuation.pkl

Both strategies use the SAME label files and SAME feature columns.
The only difference is which label column is used for scoring.

Usage:
  python generate_outcomes.py                       # both strategies
  python generate_outcomes.py --strategy reversion
  python generate_outcomes.py --strategy continuation

Outputs:
  data/outcomes/outcomes_reversion.parquet
  data/outcomes/outcomes_continuation.parquet
  data/outcomes/outcomes.parquet          ← combined (use for backtest)
  data/outcomes/outcomes_full.parquet     ← combined (for analysis)
"""

import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS
from model_trainer import TRAIN_END_DATE, VAL_END_DATE

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    '--strategy',
    choices=['reversion', 'continuation', 'both'],
    default='both',
    help="Which strategy to score (default: both)"
)
args = parser.parse_args()

RUN_REVERSION    = args.strategy in ('reversion',    'both')
RUN_CONTINUATION = args.strategy in ('continuation', 'both')

# ── Config ────────────────────────────────────────────────────────────────────
PAIRS     = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
PIP_SIZES = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01, 'XAUUSD': 1.0}
WEIGHTS   = {'xgb': 0.50, 'rf': 0.35, 'ridge': 0.15}

MAX_ENTRY_BARS = 96
MAX_HOLD_BARS  = 96

Path('data/outcomes').mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_labels() -> pd.DataFrame:
    """
    Load combined label file. Tries clean → dual → per-symbol fallback.
    Returns a single sorted DataFrame with both 'label' and 'label_cont'.
    """
    for path in [
        Path('data/labels/training_labels_clean.parquet'),
        Path('data/labels/training_labels_dual.parquet'),
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"Loaded labels from {path}  ({len(df):,} rows)")
            df['zone_time'] = pd.to_datetime(df['zone_time'])
            df = df.sort_values('zone_time').reset_index(drop=True)
            return df

    # Per-symbol fallback
    print("Combined label file not found — loading per-symbol files...")
    dfs = []
    for pair in PAIRS:
        p = Path(f'data/labels/training_labels_{pair}.parquet')
        if p.exists():
            tmp = pd.read_parquet(p)
            if 'symbol' not in tmp.columns:
                tmp['symbol'] = pair
            dfs.append(tmp)
            print(f"  {pair}: {len(tmp):,} rows")
        else:
            print(f"  ⚠️  {pair}: not found")
    if not dfs:
        raise FileNotFoundError("No label files found. Run label_generator.py first.")
    combined = pd.concat(dfs, ignore_index=True)
    combined['zone_time'] = pd.to_datetime(combined['zone_time'])
    combined = combined.sort_values('zone_time').reset_index(drop=True)
    print(f"Combined: {len(combined):,} rows")
    return combined


def load_bars() -> dict:
    print("\nLoading 15min bars...")
    bars_by_sym = {}
    for sym in PAIRS:
        p = Path(f'data/bars_15min_{sym}.parquet')
        if p.exists():
            df = pd.read_parquet(p)
            if 'time' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'time'})
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            bars_by_sym[sym] = df
            print(f"  {sym}: {len(df):,} bars  "
                  f"({df['time'].min().date()} → {df['time'].max().date()})")
        else:
            print(f"  ⚠️  {sym}: not found — will use label fallback")
    return bars_by_sym


def simulate_trade(sym, direction, signal_time, entry, sl, tp, bars_by_sym):
    if sym not in bars_by_sym:
        return None, None, None, None
    df        = bars_by_sym[sym]
    sig_ts    = pd.Timestamp(signal_time)
    start_idx = df['time'].searchsorted(sig_ts, side='right')
    if start_idx >= len(df):
        return 'expired', None, None, None

    # Phase 1 — entry fill
    fill_idx = None
    for i in range(start_idx, min(start_idx + MAX_ENTRY_BARS, len(df))):
        bar = df.iloc[i]
        if direction == 'BUY'  and bar['low']  <= entry: fill_idx = i; break
        if direction == 'SELL' and bar['high'] >= entry: fill_idx = i; break
    if fill_idx is None:
        return 'expired', None, 0, None

    bars_to_fill = fill_idx - start_idx

    # Phase 2 — TP / SL
    scan_start = fill_idx + 1
    for j in range(scan_start, min(scan_start + MAX_HOLD_BARS, len(df))):
        bar = df.iloc[j]
        if direction == 'BUY':
            if bar['high'] >= tp: return 'win',  tp, bars_to_fill, j - fill_idx
            if bar['low']  <= sl: return 'loss', sl, bars_to_fill, j - fill_idx
        else:
            if bar['low']  <= tp: return 'win',  tp, bars_to_fill, j - fill_idx
            if bar['high'] >= sl: return 'loss', sl, bars_to_fill, j - fill_idx

    last      = df.iloc[min(scan_start + MAX_HOLD_BARS - 1, len(df) - 1)]
    return 'timeout', float(last['close']), bars_to_fill, MAX_HOLD_BARS


def encode_categoricals(test_df, encoders):
    for col in CATEGORICAL_FEATURES:
        le      = encoders[col]
        encoded = []
        for v in test_df[col].astype(str):
            encoded.append(int(le.transform([v])[0]) if v in le.classes_ else 0)
        test_df[f'{col}_encoded'] = encoded
    return test_df


def score_ensemble(test_df, models, scaler):
    feature_cols = list(FEATURE_COLUMNS)
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X        = test_df[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)
    p = (models['xgb'].predict_proba(X_scaled)[:, 1] * WEIGHTS['xgb'] +
         models['rf'].predict_proba(X_scaled)[:, 1]  * WEIGHTS['rf']  +
         models['ridge'].predict_proba(X_scaled)[:, 1] * WEIGHTS['ridge'])
    return np.clip(p, 1e-7, 1 - 1e-7)


def simulate_all(test_df, strategy_name, label_col, bars_by_sym):
    """Run bar-by-bar simulation. label_col used only for no-bars fallback."""
    rows            = []
    geometry_errors = 0
    sim_stats       = {'win': 0, 'loss': 0, 'expired': 0, 'timeout': 0, 'no_bars': 0}

    print(f"\nSimulating {len(test_df):,} {strategy_name} trades...")

    for idx, (_, r) in enumerate(test_df.iterrows()):
        if idx % 200 == 0:
            print(f"  {idx}/{len(test_df)}...")

        sym       = r['symbol']
        pip_size  = PIP_SIZES.get(sym, 0.0001)
        entry     = float(r['entry'])
        sl        = float(r['sl'])
        tp        = float(r['tp'])

        # Direction:
        # REVERSION    — demand→BUY, supply→SELL (counter-trend)
        # CONTINUATION — demand→SELL, supply→BUY (with trend)
        zone = str(r.get('zone_type', '')).lower()
        if strategy_name == 'REVERSION':
            direction = 'BUY' if 'demand' in zone else 'SELL'
        else:
            # CONTINUATION trades WITH the trend:
            # price returning to a demand zone in a downtrend → SELL continuation
            # price returning to a supply zone in an uptrend  → BUY  continuation
            direction = 'SELL' if 'demand' in zone else 'BUY'
            # Labels were built with REVERSION geometry (demand→BUY, supply→SELL).
            # For CONTINUATION the direction is flipped, so tp/sl must be swapped
            # to restore valid geometry (BUY: tp>entry>sl, SELL: sl>entry>tp).
            tp, sl = sl, tp

        # Geometry guard
        if direction == 'BUY'  and not (tp > entry > sl): geometry_errors += 1; continue
        if direction == 'SELL' and not (sl > entry > tp): geometry_errors += 1; continue

        outcome, exit_price, bars_to_fill, bars_held = simulate_trade(
            sym, direction, r['zone_time'], entry, sl, tp, bars_by_sym
        )

        if outcome is None:
            sim_stats['no_bars'] += 1
            outcome      = 'win' if r.get(label_col, 0) == 1.0 else 'loss'
            exit_price   = tp if outcome == 'win' else sl
            bars_to_fill = 0
            bars_held    = 0
        else:
            sim_stats[outcome] += 1

        rows.append({
            'signal_timestamp': r['zone_time'],
            'resolved_at':      r['zone_time'],
            'symbol':           sym,
            'direction':        direction,
            'entry':            entry,
            'sl':               sl,
            'tp':               tp,
            'exit_price':       exit_price if exit_price is not None
                                else (tp if r.get(label_col, 0) == 1.0 else sl),
            'predicted_prob':   float(r['predicted_prob']),
            'outcome':          outcome,
            'strategy':         strategy_name,
            'pip_size':         pip_size,
            'bars_held':        bars_held    if bars_held    is not None else 0,
            'bars_to_fill':     bars_to_fill if bars_to_fill is not None else 0,
        })

    if geometry_errors:
        print(f"  ⚠️  Skipped {geometry_errors} geometry errors")
    print(f"  Results: " + "  ".join(f"{k}={v:,}" for k, v in sim_stats.items()))
    return rows


def print_ev_table(out, strategy_name):
    filled = out[out['outcome'].isin(['win', 'loss'])]
    print(f"\n{'─'*60}")
    print(f"  {strategy_name} — CALIBRATION & EV")
    print(f"{'─'*60}")
    print(f"  Total signals  : {len(out):,}")
    print(f"  Filled trades  : {len(filled):,}")
    if len(filled) == 0:
        print("  No filled trades."); return
    print(f"  Overall WR     : {(filled['outcome']=='win').mean():.1%}")

    def pip_val(r, col):
        pip = PIP_SIZES.get(r['symbol'], 0.0001)
        if r['direction'] == 'BUY':
            return (r['tp']-r['entry'])/pip if col=='tp' else (r['entry']-r['sl'])/pip
        return (r['entry']-r['tp'])/pip if col=='tp' else (r['sl']-r['entry'])/pip

    print(f"\n  {'Bucket':<10} {'N':>6} {'WR':>7} {'AvgTP':>7} {'AvgSL':>7} {'EV':>8}")
    print("  " + "─"*50)
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
        if n == 0: continue
        wins_b   = bucket[bucket['outcome'] == 'win']
        losses_b = bucket[bucket['outcome'] == 'loss']
        wr       = len(wins_b) / n
        avg_tp   = wins_b.apply(lambda r: pip_val(r,'tp'),  axis=1).mean() if len(wins_b)   else 0
        avg_sl   = losses_b.apply(lambda r: pip_val(r,'sl'),axis=1).mean() if len(losses_b) else 0
        ev       = (wr * avg_tp) - ((1-wr) * avg_sl)
        print(f"  {'✅' if ev > 0 else '❌'} {label:<9} {n:>6,}  {wr:>6.1%}  "
              f"{avg_tp:>7.1f}  {avg_sl:>7.1f}  {ev:>+8.1f}p")


# ─────────────────────────────────────────────────────────────────────────────
# Load shared data
# ─────────────────────────────────────────────────────────────────────────────

all_labels  = load_labels()
bars_by_sym = load_bars()

# Date split — test set only (same boundaries as model_trainer.py)
test_mask  = all_labels['zone_time'] >= VAL_END_DATE
test_df    = all_labels[test_mask].copy()
train_count = (all_labels['zone_time'] < TRAIN_END_DATE).sum()
val_count   = ((all_labels['zone_time'] >= TRAIN_END_DATE) &
               (all_labels['zone_time'] <  VAL_END_DATE)).sum()

print(f"\nDate split (matches model_trainer.py):")
print(f"  Train : < {TRAIN_END_DATE.date()}  ({train_count:,} rows)  ← model trained on these")
print(f"  Val   : {TRAIN_END_DATE.date()} – {VAL_END_DATE.date()}  ({val_count:,} rows)")
print(f"  Test  : >= {VAL_END_DATE.date()}  ({len(test_df):,} rows)  ← scoring these only")

if len(test_df) == 0:
    raise ValueError(f"Test set is empty. Check VAL_END_DATE={VAL_END_DATE.date()} in model_trainer.py")

print(f"\n⚠️  Scoring {len(test_df):,} out-of-sample test rows only")


# ─────────────────────────────────────────────────────────────────────────────
# REVERSION
# ─────────────────────────────────────────────────────────────────────────────

reversion_out = pd.DataFrame()

if RUN_REVERSION:
    print("\n" + "="*60)
    print("  STRATEGY: REVERSION  (label col: 'label')")
    print("="*60)

    if 'label' not in test_df.columns:
        print("  ⚠️  'label' column not found — skipping REVERSION")
        RUN_REVERSION = False
    else:
        print("Loading REVERSION models...")
        rev_models = {
            'xgb':   pickle.load(open('models/model_v6_reversion.pkl', 'rb')),
            'rf':    pickle.load(open('models/rf_v6_reversion.pkl',    'rb')),
            'ridge': pickle.load(open('models/lr_v6_reversion.pkl',    'rb')),
        }
        rev_scaler   = pickle.load(open('models/scaler_v6_reversion.pkl',   'rb'))
        rev_encoders = pickle.load(open('models/encoders_v6_reversion.pkl', 'rb'))

        test_rev = test_df.copy()
        test_rev = encode_categoricals(test_rev, rev_encoders)
        p_rev    = score_ensemble(test_rev, rev_models, rev_scaler)
        test_rev['predicted_prob'] = p_rev
        print(f"Probability range: {p_rev.min():.3f} – {p_rev.max():.3f}  mean={p_rev.mean():.3f}")

        rev_rows      = simulate_all(test_rev, 'REVERSION', 'label', bars_by_sym)
        reversion_out = (pd.DataFrame(rev_rows)
                         .sort_values('signal_timestamp')
                         .reset_index(drop=True))

        print_ev_table(reversion_out, 'REVERSION')
        reversion_out.to_parquet('data/outcomes/outcomes_reversion.parquet', index=False)
        print(f"\n✅ data/outcomes/outcomes_reversion.parquet  ({len(reversion_out):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CONTINUATION
# ─────────────────────────────────────────────────────────────────────────────

continuation_out = pd.DataFrame()

if RUN_CONTINUATION:
    print("\n" + "="*60)
    print("  STRATEGY: CONTINUATION  (label col: 'label_cont')")
    print("="*60)

    if 'label_cont' not in test_df.columns:
        print("  ⚠️  'label_cont' column not found — skipping CONTINUATION")
        print("  Make sure your label file is training_labels_dual.parquet or")
        print("  training_labels_clean.parquet (not per-symbol files)")
        RUN_CONTINUATION = False
    else:
        # Check model files
        cont_files = [
            'models/model_v6_continuation.pkl',
            'models/rf_v6_continuation.pkl',
            'models/lr_v6_continuation.pkl',
            'models/scaler_v6_continuation.pkl',
            'models/encoders_v6_continuation.pkl',
        ]
        missing = [f for f in cont_files if not Path(f).exists()]
        if missing:
            print(f"  ⚠️  Missing model files: {missing}")
            print("  Run: python model_trainer.py  to train both tasks first")
            RUN_CONTINUATION = False
        else:
            print("Loading CONTINUATION models...")
            cont_models = {
                'xgb':   pickle.load(open('models/model_v6_continuation.pkl', 'rb')),
                'rf':    pickle.load(open('models/rf_v6_continuation.pkl',    'rb')),
                'ridge': pickle.load(open('models/lr_v6_continuation.pkl',   'rb')),
            }
            cont_scaler   = pickle.load(open('models/scaler_v6_continuation.pkl',   'rb'))
            cont_encoders = pickle.load(open('models/encoders_v6_continuation.pkl', 'rb'))

            test_cont = test_df.copy()
            test_cont = encode_categoricals(test_cont, cont_encoders)
            p_cont    = score_ensemble(test_cont, cont_models, cont_scaler)
            test_cont['predicted_prob'] = p_cont
            print(f"Probability range: {p_cont.min():.3f} – {p_cont.max():.3f}  mean={p_cont.mean():.3f}")

            cont_rows        = simulate_all(test_cont, 'CONTINUATION', 'label_cont', bars_by_sym)
            if not cont_rows:
                print("  ⚠️  No CONTINUATION trades survived simulation — skipping output")
                RUN_CONTINUATION = False
            else:
                continuation_out = (pd.DataFrame(cont_rows)
                                    .sort_values('signal_timestamp')
                                    .reset_index(drop=True))
                print_ev_table(continuation_out, 'CONTINUATION')
                continuation_out.to_parquet('data/outcomes/outcomes_continuation.parquet', index=False)
                print(f"\n✅ data/outcomes/outcomes_continuation.parquet  ({len(continuation_out):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Combine and save
# ─────────────────────────────────────────────────────────────────────────────

parts = [df for df in [reversion_out, continuation_out] if not df.empty]

if not parts:
    print("\n⚠️  No outcomes produced.")
else:
    combined = (pd.concat(parts, ignore_index=True)
                .sort_values('signal_timestamp')
                .reset_index(drop=True))

    combined.to_parquet('data/outcomes/outcomes.parquet',      index=False)
    combined.to_parquet('data/outcomes/outcomes_full.parquet', index=False)

    print(f"\n{'='*60}")
    print("  COMBINED SUMMARY")
    print(f"{'='*60}")
    for strat, grp in combined.groupby('strategy'):
        filled = grp[grp['outcome'].isin(['win','loss'])]
        wr     = (filled['outcome'] == 'win').mean() if len(filled) > 0 else 0
        above65 = grp[grp['predicted_prob'] >= 0.65]
        filled65 = above65[above65['outcome'].isin(['win','loss'])]
        wr65 = (filled65['outcome'] == 'win').mean() if len(filled65) > 0 else 0
        print(f"\n  {strat}")
        print(f"    Total signals : {len(grp):,}")
        print(f"    Filled trades : {len(filled):,}")
        print(f"    Overall WR    : {wr:.1%}")
        print(f"    Above 65% WR  : {wr65:.1%}  ({len(filled65)} filled signals)")

    print(f"\n✅ data/outcomes/outcomes.parquet       ({len(combined):,} rows)")
    print(f"✅ data/outcomes/outcomes_full.parquet  ({len(combined):,} rows)")
    print(f"   Date range: {pd.to_datetime(combined['signal_timestamp']).min().date()} → "
          f"{pd.to_datetime(combined['signal_timestamp']).max().date()}")

    print("\n" + "─"*60)
    print("LEAKAGE CHECK")
    print("─"*60)
    print(f"  Trained on  : zone_time < {TRAIN_END_DATE.date()}")
    print(f"  Scored on   : zone_time >= {VAL_END_DATE.date()}")
    print(f"  Gap         : {(VAL_END_DATE - TRAIN_END_DATE).days} days — NO overlap")
    print("─"*60)
    print("\nNext:")
    print("  python backtest_engine.py --source outcomes --balance 10000 --risk 1")
    print("  python backtest_engine.py --source outcomes --balance 10000 --risk 1 --min-prob 0.65")