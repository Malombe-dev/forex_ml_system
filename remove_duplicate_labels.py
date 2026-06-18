#!/usr/bin/env python3
"""
Remove 19 duplicate rows from training_labels_clean.parquet
Prevents overfitting from seeing the same zone twice during training.

Usage:
    python remove_duplicate_labels.py
"""

import pandas as pd
from pathlib import Path

LABELS_FILE = Path('data/labels/training_labels_clean.parquet')

def main():
    if not LABELS_FILE.exists():
        print(f"❌ Error: {LABELS_FILE} not found")
        return

    # Load original
    df_original = pd.read_parquet(LABELS_FILE)
    original_count = len(df_original)
    print(f"📊 Original dataset: {original_count} rows")

    # Identify duplicates: same entry, sl, tp, symbol, zone_time
    # These uniquely identify a trading zone+entry
    duplicates = df_original.duplicated(
        subset=['entry', 'sl', 'tp', 'zone_time'],
        keep='first'
    )
    dup_count = duplicates.sum()
    print(f"🔍 Found {dup_count} duplicate rows")

    if dup_count == 0:
        print("✅ No duplicates found — dataset is clean")
        return

    # Remove duplicates
    df_clean = df_original[~duplicates].reset_index(drop=True)
    new_count = len(df_clean)
    removed_count = original_count - new_count

    # Backup original
    backup_path = LABELS_FILE.with_stem(f"{LABELS_FILE.stem}_backup_{original_count}")
    df_original.to_parquet(backup_path)
    print(f"💾 Backup saved: {backup_path}")

    # Save cleaned
    df_clean.to_parquet(LABELS_FILE)
    print(f"✅ Cleaned dataset saved: {LABELS_FILE}")
    print(f"   Removed: {removed_count} rows")
    print(f"   New size: {new_count} rows")

    # Summary
    print(f"\n📈 Summary:")
    print(f"   Before: {original_count} rows")
    print(f"   After:  {new_count} rows")
    print(f"   Reduction: {removed_count} ({100*removed_count/original_count:.1f}%)")
    print(f"\n⚠️  Remember to retrain models with: python retrain_all.py")

if __name__ == "__main__":
    main()
