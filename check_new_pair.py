"""
Sanity-check a newly ingested pair (e.g. AUDUSD) against the existing pairs
before it flows into label_generator.py / model_trainer.py.

Usage:
    python check_new_pair.py AUDUSD
    python check_new_pair.py AUDUSD --against EURUSD GBPUSD USDJPY XAUUSD
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

TIMEFRAMES = {
    "15min": ["bars_15min_{sym}.parquet"],
    "1h":    ["bars_1h_{sym}.parquet", "bars_h1_{sym}.parquet"],
    "4h":    ["bars_4h_{sym}.parquet", "bars_h4_{sym}.parquet"],
    "1d":    ["bars_1d_{sym}.parquet", "bars_d1_{sym}.parquet"],
    "5min":  ["bars_5min_{sym}.parquet", "bars_5m_{sym}.parquet", "bars_m5_{sym}.parquet"],
}
ZONES_PATTERN = "data/features/merged_zones_{sym}.parquet"


def find_file(patterns, sym):
    for pat in patterns:
        p = Path("data") / pat.format(sym=sym)
        if p.exists():
            return p
    return None


def load_bars(sym):
    out = {}
    for tf, patterns in TIMEFRAMES.items():
        p = find_file(patterns, sym)
        out[tf] = pd.read_parquet(p) if p else None
    return out


def summarize(sym, bars, zones):
    print(f"\n{'='*70}\n  {sym}\n{'='*70}")
    for tf, df in bars.items():
        if df is None:
            print(f"  [{tf:5s}] ❌ MISSING")
            continue
        dtypes = dict(df.dtypes.astype(str))
        time_col = next((c for c in df.columns if c.lower() in ("time", "datetime", "date")), df.columns[0])
        tmin, tmax = df[time_col].min(), df[time_col].max()
        n_nulls = df.isnull().sum().sum()
        print(f"  [{tf:5s}] rows={len(df):>8,}  cols={list(df.columns)}")
        print(f"           range: {tmin} → {tmax}   nulls={n_nulls}")
    if zones is not None:
        print(f"  [zones] rows={len(zones):>8,}  cols={list(zones.columns)}")
    else:
        print(f"  [zones] ❌ MISSING — run smc_detector.py --symbol {sym}")
    return {tf: (df.dtypes.astype(str).to_dict() if df is not None else None) for tf, df in bars.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", help="New symbol to check, e.g. AUDUSD")
    ap.add_argument("--against", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
                     help="Existing symbols to compare schema against")
    args = ap.parse_args()

    all_schemas = {}
    all_syms = [args.symbol] + [s for s in args.against if s != args.symbol]

    for sym in all_syms:
        bars = load_bars(sym)
        zones_path = Path(ZONES_PATTERN.format(sym=sym))
        zones = pd.read_parquet(zones_path) if zones_path.exists() else None
        all_schemas[sym] = summarize(sym, bars, zones)

    # ── Schema comparison against the first reference symbol that has data ──
    ref_sym = next((s for s in args.against if all_schemas.get(s, {}).get("15min")), None)
    print(f"\n{'='*70}\n  SCHEMA DIFF: {args.symbol} vs {ref_sym}\n{'='*70}")
    if ref_sym is None:
        print("  No reference symbol had readable 15min bars — can't compare.")
        return

    mismatches = 0
    for tf in TIMEFRAMES:
        new_schema = all_schemas[args.symbol].get(tf)
        ref_schema = all_schemas[ref_sym].get(tf)
        if new_schema is None or ref_schema is None:
            continue
        if new_schema != ref_schema:
            mismatches += 1
            print(f"  [{tf}] ⚠️  MISMATCH")
            new_cols, ref_cols = set(new_schema), set(ref_schema)
            if new_cols - ref_cols:
                print(f"           extra columns in {args.symbol}: {new_cols - ref_cols}")
            if ref_cols - new_cols:
                print(f"           missing columns in {args.symbol}: {ref_cols - new_cols}")
            for c in new_cols & ref_cols:
                if new_schema[c] != ref_schema[c]:
                    print(f"           dtype differs on '{c}': {args.symbol}={new_schema[c]}  {ref_sym}={ref_schema[c]}")
        else:
            print(f"  [{tf}] ✅ matches")

    print(f"\n{'='*70}")
    if mismatches == 0:
        print(f"  ✅ {args.symbol} schema matches {ref_sym} across all available timeframes.")
    else:
        print(f"  ⚠️  {mismatches} timeframe(s) differ — fix before running label_generator.py,")
        print(f"     since feature_builder.py / label_generator.py assume a consistent schema.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()