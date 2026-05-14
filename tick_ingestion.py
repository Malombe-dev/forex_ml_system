"""
TICK DATA INGESTION — STREAMING VERSION
========================================
Handles 13GB CSV on 7.6GB RAM by:
1. Reading CSV in 500k-row chunks
2. Writing each chunk's bars DIRECTLY to parquet (append mode)
3. Final consolidation reads only the small bar files — not the ticks

Memory peak: ~800MB at any point.
"""

import gc
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500_000          # smaller chunks = less RAM
TIMEFRAMES = ["1min", "5min", "15min", "1h", "4h", "1d"]

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--csv",    default="data/EURUSD.csv")
_parser.add_argument("--symbol", default="EURUSD")
_parser.add_argument("--pip_size", type=float, default=0.0001)
_args, _ = _parser.parse_known_args()

CSV_PATH   = _args.csv
SYMBOL     = _args.symbol
PIP_SIZE   = _args.pip_size
OUTPUT_DIR = "data"


# ── Column detection ───────────────────────────────────────────────────────────

def detect_columns(csv_path: str):
    header = pd.read_csv(csv_path, nrows=0)
    cols   = header.columns.tolist()
    cols_l = {c.lower(): c for c in cols}
    logger.info(f"CSV columns: {cols}")

    time_col = None
    for pat in ["time (eet)", "time (utc)", "time", "timestamp", "date", "datetime"]:
        matches = [c for c in cols_l if pat in c]
        if matches:
            time_col = cols_l[matches[0]]
            break

    bid_col = None
    for pat in ["bid", "bidprice", "bid_price"]:
        matches = [c for c in cols_l if pat in c]
        if matches:
            bid_col = cols_l[matches[0]]
            break

    ask_col = None
    for pat in ["ask", "askprice", "ask_price"]:
        matches = [c for c in cols_l if pat in c]
        if matches:
            ask_col = cols_l[matches[0]]
            break

    if not all([time_col, bid_col, ask_col]):
        raise ValueError(f"Cannot detect columns. Available: {cols}")

    is_eet = "eet" in time_col.lower()
    logger.info(f"time='{time_col}', bid='{bid_col}', ask='{ask_col}', EET={is_eet}")
    return time_col, bid_col, ask_col, is_eet


# ── Process one chunk → return cleaned mid/spread df ──────────────────────────

def clean_chunk(chunk, time_col, bid_col, ask_col, is_eet):
    df = chunk[[time_col, bid_col, ask_col]].copy()
    df.columns = ["time", "bid", "ask"]
    
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["time"])
    if df.empty:
        return None
    if is_eet:
        df["time"] = df["time"] - pd.Timedelta(hours=2)
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["bid", "ask"])
    df["mid"]    = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    return df[["time", "mid", "spread"]]


# ── Resample one chunk to one timeframe ───────────────────────────────────────

def resample_chunk(df, timeframe):
    idx  = df.set_index("time").sort_index()
    bars = idx["mid"].resample(timeframe).ohlc()
    bars["avg_spread"] = idx["spread"].resample(timeframe).mean()
    bars["tick_count"] = idx["mid"].resample(timeframe).count()
    return bars.dropna(subset=["open"])


# ── Append bars to a temp parquet file (one per timeframe) ────────────────────

def append_bars(bars: pd.DataFrame, tmp_path: Path, chunk_num: int):
    """Save this chunk's bars as a separate small parquet file."""
    chunk_file = tmp_path.parent / f"{tmp_path.stem}_chunk{chunk_num:05d}.parquet"
    bars.reset_index().to_parquet(chunk_file, compression="snappy", index=False)


# ── Consolidate all chunk parquets into one final bar file ────────────────────

def consolidate(tmp_dir: Path, timeframe: str, out_path: Path):
    """
    Read all chunk parquet files for this timeframe, merge properly,
    and write the final consolidated bar file.
    Does this in sorted batches to avoid loading everything at once.
    """
    pattern = f"bars_{timeframe}_{SYMBOL}_chunk*.parquet"
    chunk_files = sorted(tmp_dir.glob(pattern))

    if not chunk_files:
        logger.warning(f"  No chunk files found for {timeframe}")
        return 0

    logger.info(f"  Consolidating {len(chunk_files)} chunks for {timeframe}...")

    # Read all chunk files — they are tiny (bars not ticks)
    # Each chunk file for 15min is ~50KB. 400 chunks = ~20MB total. Safe.
    dfs = []
    for cf in chunk_files:
        try:
            dfs.append(pd.read_parquet(cf))
        except Exception:
            pass

    if not dfs:
        return 0

    combined = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # Ensure time column
    if "time" not in combined.columns and combined.index.name == "time":
        combined = combined.reset_index()

    combined["time"] = pd.to_datetime(combined["time"])
    combined = combined.set_index("time").sort_index()

    # Re-aggregate overlapping periods at chunk boundaries
    final = combined.groupby(combined.index).agg(
        open       =("open",       "first"),
        high       =("high",       "max"),
        low        =("low",        "min"),
        close      =("close",      "last"),
        avg_spread =("avg_spread", "mean"),
        tick_count =("tick_count", "sum"),
    ).reset_index()

    final.columns = ["time"] + [c for c in final.columns if c != "time"]
    final.to_parquet(out_path, compression="snappy", index=False)

    del combined, final
    gc.collect()

    # Clean up chunk files
    for cf in chunk_files:
        try:
            cf.unlink()
        except Exception:
            pass

    return len(pd.read_parquet(out_path))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import time as time_mod
    start    = time_mod.time()
    out      = Path(OUTPUT_DIR)
    tmp_dir  = out / "tmp_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    time_col, bid_col, ask_col, is_eet = detect_columns(CSV_PATH)

    chunk_num   = 0
    total_ticks = 0
    first_ts    = None
    last_ts     = None

    logger.info(f"Streaming {CSV_PATH} in {CHUNK_SIZE:,}-row chunks")
    logger.info("Bars written per chunk — no large accumulation in RAM")
    logger.info("-" * 60)

    reader = pd.read_csv(
        CSV_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        usecols=[time_col, bid_col, ask_col],
    )

    for raw in reader:
        chunk_num  += 1
        cleaned     = clean_chunk(raw, time_col, bid_col, ask_col, is_eet)
        del raw
        gc.collect()

        if cleaned is None or cleaned.empty:
            continue

        total_ticks += len(cleaned)
        if first_ts is None:
            first_ts = cleaned["time"].min()
        last_ts = cleaned["time"].max()

        # Resample and write each timeframe separately
        for tf in TIMEFRAMES:
            try:
                bars = resample_chunk(cleaned, tf)
                if not bars.empty:
                    append_bars(bars, tmp_dir / f"bars_{tf}_{SYMBOL}.parquet", chunk_num)
            except Exception as exc:
                logger.warning(f"chunk {chunk_num} {tf}: {exc}")

        del cleaned
        gc.collect()

        if chunk_num % 20 == 0:
            logger.info(
                f"  Chunk {chunk_num:>4} | {total_ticks/1_000_000:.1f}M ticks "
                f"| up to {last_ts.date() if last_ts else '?'}"
            )

    logger.info("-" * 60)
    logger.info(f"Pass 1 done: {chunk_num} chunks | {total_ticks:,} ticks")
    logger.info(f"Range: {first_ts} → {last_ts}")
    logger.info("Consolidating bar files (reading bar chunks only, not ticks)...")

    results = {}
    for tf in TIMEFRAMES:
        out_path = out / f"bars_{tf}_{SYMBOL}.parquet"
        count    = consolidate(tmp_dir, tf, out_path)
        if count:
            results[tf] = count
            logger.info(f"  ✅ {tf}: {count:,} bars")

    # Remove tmp dir if empty
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    elapsed = time_mod.time() - start
    print(f"\n{'='*55}")
    print("INGESTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Time : {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"  Ticks: {total_ticks:,}")
    print(f"  Range: {first_ts} → {last_ts}")
    print()
    for tf, count in results.items():
        print(f"  {tf:<8} {count:>12,} bars")
    print(f"{'='*55}")
    print("\nNext: python smc_detector.py")


if __name__ == "__main__":
    main()