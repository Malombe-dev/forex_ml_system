
"""
SMC DATA QUALITY AUDIT
======================
Checks the quality of detected SMC structures before running label_generator.

Checks:
  1. FVG Quality     — size distribution, fill rate, bull/bear balance
  2. OB Quality      — reaction size, zone width, distribution over time
  3. Zone Quality    — price range, confluence scores, D1 alignment rate
  4. 1H Confirmation — breakdown by FVG vs OB vs neither, per symbol
  5. Cross-symbol    — consistency check across all 4 pairs
  6. Temporal        — zones distributed evenly over time (not clustered)
  7. Data gaps       — missing bars in 4H/1H data

Usage:
  python audit_smc.py                    # all symbols
  python audit_smc.py --symbol EURUSD    # one symbol
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

SYMBOL_CONFIG = {
    'EURUSD': 0.0001,
    'GBPUSD': 0.0001,
    'USDJPY': 0.01,
    'XAUUSD': 1.0,
}

FEATURES_DIR = Path("data/features")
BARS_DIR     = Path("data")


def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def subsection(title):
    print(f"\n  ── {title} {'─'*(55-len(title))}")


def load(path, name):
    if not path.exists():
        print(f"  ❌ NOT FOUND: {path}  — run smc_detector.py first")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    return df


def check_bars(sym, pip_size):
    subsection(f"Bar data gaps — {sym}")
    for tf, expected_gap_h in [('4h', 4), ('1h', 1)]:
        path = BARS_DIR / f"bars_{tf}_{sym}.parquet"
        if not path.exists():
            print(f"    ❌ bars_{tf}_{sym}.parquet not found")
            continue
        df = pd.read_parquet(path)
        if 'time' not in df.columns:
            df = df.reset_index()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        gaps = df['time'].diff().dropna()
        expected = pd.Timedelta(hours=expected_gap_h)
        big_gaps = gaps[gaps > expected * 3]
        print(f"    {tf}: {len(df):,} bars  "
              f"range={df['time'].min().date()}→{df['time'].max().date()}  "
              f"gaps>{expected_gap_h*3}h: {len(big_gaps)}", end="")
        if len(big_gaps) > 0:
            worst = big_gaps.nlargest(1)
            print(f"  ⚠️  worst gap: {worst.values[0]}")
        else:
            print("  ✅")


def check_fvgs(sym, pip_size):
    subsection(f"FVG quality — {sym}")
    path = FEATURES_DIR / f"fvgs_{sym}.parquet"
    df = load(path, f"fvgs_{sym}")
    if df.empty:
        return {}

    n_total  = len(df)
    n_bull   = (df['type'] == 'fvg_bullish').sum()
    n_bear   = (df['type'] == 'fvg_bearish').sum()
    n_filled = df['filled'].sum() if 'filled' in df.columns else 0
    fill_pct = n_filled / n_total if n_total > 0 else 0

    size = df['size_pips'] if 'size_pips' in df.columns else \
           (df['price_high'] - df['price_low']) / pip_size
    size_atr = df['size_atr'] if 'size_atr' in df.columns else pd.Series([np.nan]*n_total)

    print(f"    Total FVGs:  {n_total:,}  (bull={n_bull:,} / bear={n_bear:,})")
    fill_flag = '✅ normal' if 0.50 <= fill_pct <= 0.85 else '⚠️  unusual'
    print(f"    Fill rate:   {fill_pct:.1%}  ({fill_flag})")
    print(f"    Size (pips): min={size.min():.1f}  mean={size.mean():.1f}  "
          f"p95={size.quantile(0.95):.1f}  max={size.max():.1f}")
    if size_atr.notna().any():
        print(f"    Size (ATR):  mean={size_atr.mean():.2f}  p95={size_atr.quantile(0.95):.2f}")

    balance  = n_bull / n_total if n_total > 0 else 0.5
    bal_flag = "✅" if 0.40 <= balance <= 0.60 else "⚠️  imbalanced"
    print(f"    Bull/bear:   {balance:.0%} / {1-balance:.0%}  {bal_flag}")

    if 'time' in df.columns:
        by_year = df.groupby(df['time'].dt.year).size()
        cv = by_year.std() / by_year.mean() if by_year.mean() > 0 else 0
        flag = "✅" if cv < 1.0 else "⚠️  clustered"
        print(f"    Temporal CV: {cv:.2f}  {flag}  "
              f"(zones/yr: min={by_year.min()} mean={by_year.mean():.0f} max={by_year.max()})")

    return {'n_fvgs': n_total, 'fill_pct': fill_pct, 'fvg_size_mean': size.mean()}


def check_obs(sym, pip_size):
    subsection(f"Order Block quality — {sym}")
    path = FEATURES_DIR / f"order_blocks_{sym}.parquet"
    df = load(path, f"order_blocks_{sym}")
    if df.empty:
        return {}

    n_total = len(df)
    n_bull  = df['type'].str.contains('bullish').sum()
    n_bear  = df['type'].str.contains('bearish').sum()

    reaction = df['reaction_atr'] if 'reaction_atr' in df.columns else pd.Series([np.nan]*n_total)
    size_p   = df['size_pips']    if 'size_pips'    in df.columns else \
               (df['price_high'] - df['price_low']) / pip_size

    print(f"    Total OBs:   {n_total:,}  (bull={n_bull:,} / bear={n_bear:,})")
    print(f"    Reaction(ATR): min={reaction.min():.2f}  mean={reaction.mean():.2f}  "
          f"p95={reaction.quantile(0.95):.2f}")
    print(f"    Size (pips): min={size_p.min():.1f}  mean={size_p.mean():.1f}  "
          f"max={size_p.max():.1f}")

    weak = (reaction < 1.0).sum()
    if weak > n_total * 0.1:
        print(f"    ⚠️  {weak} OBs ({weak/n_total:.0%}) reaction < 1.0 ATR")
    else:
        print(f"    ✅ Reaction strength OK")

    return {'n_obs': n_total, 'reaction_mean': float(reaction.mean())}


def check_zones(sym, pip_size):
    subsection(f"Merged zone quality — {sym}")
    path = FEATURES_DIR / f"merged_zones_{sym}.parquet"
    df = load(path, f"merged_zones_{sym}")
    if df.empty:
        return {}

    n_total  = len(df)
    n_demand = (df['type'] == 'demand').sum()
    n_supply = (df['type'] == 'supply').sum()
    size_p   = (df['price_high'] - df['price_low']) / pip_size
    strength = df['strength']   if 'strength'   in df.columns else pd.Series([np.nan]*n_total)
    conf     = df['confluence'] if 'confluence' in df.columns else pd.Series([1]*n_total)

    print(f"    Total zones: {n_total:,}  (demand={n_demand:,} / supply={n_supply:,})")

    bal      = n_demand / n_total if n_total > 0 else 0.5
    bal_flag = '✅' if 0.40 <= bal <= 0.60 else '⚠️  imbalanced'
    print(f"    D/S balance: {bal:.0%}/{1-bal:.0%}  {bal_flag}")

    print(f"    Size (pips): min={size_p.min():.1f}  mean={size_p.mean():.1f}  "
          f"p95={size_p.quantile(0.95):.1f}  max={size_p.max():.1f}")

    tiny = (size_p < 3.0).sum()
    huge = (size_p > 200.0).sum()
    if tiny > 0:
        print(f"    ⚠️  {tiny} zones < 3 pips — check pip_size")
    if huge > 0:
        print(f"    ⚠️  {huge} zones > 200 pips — very wide, low precision")
    if tiny == 0 and huge == 0:
        print(f"    ✅ Zone sizes look healthy")

    if strength.notna().any():
        print(f"    Strength:    min={strength.min():.2f}  mean={strength.mean():.2f}  "
              f"max={strength.max():.2f}")

    conf_dist = conf.value_counts().sort_index().to_dict()
    print(f"    Confluence dist: {conf_dist}")

    if 'h1_confluence' in df.columns:
        n_conf = df['h1_confluence'].sum()
        pct    = n_conf / n_total
        flag   = "✅" if pct < 0.80 else "⚠️  very high — check time filter"
        print(f"    1H confirmed:{n_conf:,} ({pct:.0%})  {flag}")

    if 'htf_alignment' in df.columns:
        n_align = df['htf_alignment'].sum()
        print(f"    D1 aligned:  {n_align:,} ({n_align/n_total:.0%})")

    if 'time' in df.columns:
        by_year = df.groupby(df['time'].dt.year).size()
        print(f"    Zones/year:  min={by_year.min()}  mean={by_year.mean():.0f}  "
              f"max={by_year.max()}")

    return {
        'n_zones':    n_total,
        'size_mean':  float(size_p.mean()),
        'size_max':   float(size_p.max()),
        'tiny_zones': int(tiny),
        'huge_zones': int(huge),
    }


def check_h1_breakdown(sym, pip_size):
    subsection(f"1H confirmation breakdown — {sym}")

    paths = {
        'fvgs':  FEATURES_DIR / f"h1_fvgs_{sym}.parquet",
        'obs':   FEATURES_DIR / f"h1_obs_{sym}.parquet",
        'zones': FEATURES_DIR / f"merged_zones_{sym}.parquet",
    }
    if not all(p.exists() for p in paths.values()):
        print(f"    ⚠️  Missing files — skipping")
        return

    h1_fvgs = pd.read_parquet(paths['fvgs'])
    h1_obs  = pd.read_parquet(paths['obs'])
    zones   = pd.read_parquet(paths['zones'])
    for d in [h1_fvgs, h1_obs, zones]:
        d['time'] = pd.to_datetime(d['time'])

    print(f"    1H FVGs available: {len(h1_fvgs):,}")
    print(f"    1H OBs available:  {len(h1_obs):,}")

    sample   = zones.sample(min(500, len(zones)), random_state=42)
    fvg_only = ob_only = both = neither = 0
    counts   = []

    for _, zone in sample.iterrows():
        zt       = zone['time']
        max_t    = zt + pd.Timedelta(hours=80)
        lo, hi   = zone['price_low'], zone['price_high']

        fvg_hit = h1_fvgs[
            (h1_fvgs['time'] >= zt) & (h1_fvgs['time'] <= max_t) &
            (h1_fvgs['price_mid'] >= lo) & (h1_fvgs['price_mid'] <= hi)
        ]
        ob_hit = h1_obs[
            (h1_obs['time'] >= zt) & (h1_obs['time'] <= max_t) &
            (h1_obs['price_mid'] >= lo) & (h1_obs['price_mid'] <= hi)
        ]
        nf, no = len(fvg_hit), len(ob_hit)
        counts.append(nf + no)
        if   nf > 0 and no > 0: both     += 1
        elif nf > 0:             fvg_only += 1
        elif no > 0:             ob_only  += 1
        else:                    neither  += 1

    n         = len(sample)
    confirmed = fvg_only + ob_only + both
    cs        = pd.Series(counts)

    print(f"    Sample: {n} zones")
    print(f"    FVG only:  {fvg_only:>4} ({fvg_only/n:.0%})")
    print(f"    OB only:   {ob_only:>4} ({ob_only/n:.0%})")
    print(f"    Both:      {both:>4} ({both/n:.0%})")
    print(f"    Neither:   {neither:>4} ({neither/n:.0%})")
    conf_flag = "✅" if confirmed/n < 0.80 else "⚠️  high — 4H zones may be too wide"
    print(f"    Confirmed: {confirmed:>4} ({confirmed/n:.0%})  {conf_flag}")
    if cs[cs > 0].any():
        print(f"    Avg h1 structures per confirmed zone: {cs[cs>0].mean():.1f}")


def cross_symbol_summary(results):
    section("CROSS-SYMBOL SUMMARY")
    if not results:
        return
    print(f"\n  {'Symbol':<8} {'FVGs':>7} {'OBs':>7} {'Zones':>7} "
          f"{'AvgSize':>8} {'Tiny':>5} {'Huge':>5}")
    print(f"  {'-'*50}")
    for sym, r in results.items():
        flag = " ⚠️" if r.get('tiny_zones',0) > 50 or r.get('huge_zones',0) > 50 else " ✅"
        print(f"  {sym:<8} {r.get('n_fvgs',0):>7,} {r.get('n_obs',0):>7,} "
              f"{r.get('n_zones',0):>7,} {r.get('size_mean',0):>8.1f}p "
              f"{r.get('tiny_zones',0):>5} {r.get('huge_zones',0):>5}{flag}")


def audit(symbols):
    print("\n" + "█"*65)
    print("█" + " "*18 + "SMC DATA QUALITY AUDIT" + " "*23 + "█")
    print("█"*65)

    all_results = {}
    for sym in symbols:
        pip_size = SYMBOL_CONFIG.get(sym, 0.0001)
        section(f"SYMBOL: {sym}  (pip_size={pip_size})")
        check_bars(sym, pip_size)
        fvg_r  = check_fvgs(sym, pip_size)
        ob_r   = check_obs(sym, pip_size)
        zone_r = check_zones(sym, pip_size)
        check_h1_breakdown(sym, pip_size)
        all_results[sym] = {**fvg_r, **ob_r, **zone_r}

    cross_symbol_summary(all_results)

    section("VERDICT")
    issues = []
    for sym, r in all_results.items():
        if r.get('tiny_zones', 0) > 100:
            issues.append(f"{sym}: {r['tiny_zones']} tiny zones — check pip_size")
        if r.get('huge_zones', 0) > 200:
            issues.append(f"{sym}: {r['huge_zones']} huge zones — merge threshold too wide")
        if r.get('fill_pct', 0.5) > 0.90:
            issues.append(f"{sym}: FVG fill rate {r.get('fill_pct',0):.0%} — fill window too large")

    if issues:
        print("\n  ⚠️  Issues to fix before label_generator.py:")
        for i in issues:
            print(f"     • {i}")
    else:
        print("\n  ✅ SMC data quality looks good — safe to run label_generator.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="SMC data quality audit")
    parser.add_argument("--symbol", default="ALL")
    args = parser.parse_args()
    sym     = args.symbol.upper().strip()
    symbols = list(SYMBOL_CONFIG.keys()) if sym == "ALL" else [sym]
    missing = [s for s in symbols if s not in SYMBOL_CONFIG]
    if missing:
        print(f"Unknown symbols: {missing}")
        sys.exit(1)
    audit(symbols)


if __name__ == "__main__":
    main()