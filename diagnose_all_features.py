"""
DIAGNOSE ALL FEATURES: train/test consistency audit
======================================================
Standalone, read-only diagnostic. Does NOT touch the pipeline.
Generalizes the h4_fvg_inside check (which found an inverted-but-real
relationship) to every feature in features.py, so we catch anything
else with the same shape of surprise BEFORE it's found by accident.

For each feature:
  - DISCRETE (<=6 unique values, e.g. h4_fvg_inside, zone_type,
    h1_liquidity_sweep): win rate per value, in train and test
    periods independently (same approach as Test A for h4_fvg_inside).
  - CONTINUOUS (e.g. zone_strength, rsi, risk_pips): Pearson
    correlation with the label, in train and test periods independently.

Flags:
  ⚠️  SIGN FLIP     — relationship points opposite directions in train vs test
  ⚠️  MAGNITUDE DROP — strong in train, collapses in test (classic overfit/leak shape)
  ⚠️  NEAR-CONSTANT  — one value covers >=95% of rows (near-zero information,
                       same shape as the h1_liquidity_sweep finding)
  ✅  CONSISTENT     — same direction, comparable magnitude in both periods

This does NOT prove a feature is a bug — h4_fvg_inside itself flagged as
"inconsistent-looking" under a naive read (large magnitude difference)
but turned out to be real and consistent in DIRECTION, which is what
actually matters. Read the printed win-rates/correlations yourself for
anything flagged; this script narrows down where to look, it doesn't
replace judgment.

Usage:
    python diagnose_all_features.py
    python diagnose_all_features.py --input data/labels/training_labels_dual.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

try:
    from model_trainer import TRAIN_END_DATE
except ImportError:
    TRAIN_END_DATE = pd.Timestamp("2022-08-04")

TEST_START_DATE = pd.Timestamp("2024-02-01")  # matches model_trainer.py's date split
DISCRETE_MAX_UNIQUE = 6                        # <= this many unique values -> discrete
NEAR_CONSTANT_THRESHOLD = 0.95                 # dominant value covers >= this fraction
MAGNITUDE_DROP_RATIO = 0.4                     # test/train ratio below this -> flag


def load(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df["zone_time"] = pd.to_datetime(df["zone_time"])
    df = df.sort_values("zone_time").reset_index(drop=True)
    return df


def audit_discrete(name: str, train: pd.Series, test: pd.Series,
                    train_label: pd.Series, test_label: pd.Series) -> dict:
    result = {"name": name, "kind": "discrete", "flags": []}

    # Near-constant check (on the full combined series' dominant value share)
    combined = pd.concat([train, test])
    top_share = combined.value_counts(normalize=True, dropna=False).iloc[0]
    if top_share >= NEAR_CONSTANT_THRESHOLD:
        result["flags"].append(f"NEAR-CONSTANT ({top_share:.1%} one value)")

    tr_wr = train_label.groupby(train).mean() if len(train) else pd.Series(dtype=float)
    te_wr = test_label.groupby(test).mean() if len(test) else pd.Series(dtype=float)
    tr_n  = train.groupby(train).size() if len(train) else pd.Series(dtype=int)
    te_n  = test.groupby(test).size() if len(test) else pd.Series(dtype=int)
    result["train_winrates"] = tr_wr.to_dict()
    result["test_winrates"]  = te_wr.to_dict()
    result["train_n"]        = tr_n.to_dict()
    result["test_n"]         = te_n.to_dict()

    # Sign-flip check: compare win rate spread between the two most common values
    common_vals = [v for v in tr_wr.index if v in te_wr.index]
    if len(common_vals) >= 2:
        # use min/max value present in both for spread direction
        lo, hi = sorted(common_vals)[0], sorted(common_vals)[-1]
        tr_spread = tr_wr[hi] - tr_wr[lo]
        te_spread = te_wr[hi] - te_wr[lo]
        result["train_spread"] = tr_spread
        result["test_spread"]  = te_spread
        if np.sign(tr_spread) != np.sign(te_spread) and abs(tr_spread) > 0.03 and abs(te_spread) > 0.03:
            result["flags"].append(
                f"SIGN FLIP (train spread {tr_spread:+.1%}, test spread {te_spread:+.1%})"
            )
        elif abs(tr_spread) > 0.05 and abs(te_spread) < abs(tr_spread) * MAGNITUDE_DROP_RATIO:
            result["flags"].append(
                f"MAGNITUDE DROP (train spread {tr_spread:+.1%}, test spread {te_spread:+.1%})"
            )

    return result


def audit_continuous(name: str, train_x: pd.Series, test_x: pd.Series,
                      train_label: pd.Series, test_label: pd.Series) -> dict:
    result = {"name": name, "kind": "continuous", "flags": []}

    if train_x.nunique() <= 1 or test_x.nunique() <= 1:
        result["flags"].append("NEAR-CONSTANT (no variance in a split)")
        result["train_corr"] = float("nan")
        result["test_corr"]  = float("nan")
        return result

    tr_corr = train_x.corr(train_label)
    te_corr = test_x.corr(test_label)
    result["train_corr"] = tr_corr
    result["test_corr"]  = te_corr

    if pd.notna(tr_corr) and pd.notna(te_corr):
        if np.sign(tr_corr) != np.sign(te_corr) and abs(tr_corr) > 0.05 and abs(te_corr) > 0.05:
            result["flags"].append(f"SIGN FLIP (train r={tr_corr:+.3f}, test r={te_corr:+.3f})")
        elif abs(tr_corr) > 0.10 and abs(te_corr) < abs(tr_corr) * MAGNITUDE_DROP_RATIO:
            result["flags"].append(f"MAGNITUDE DROP (train r={tr_corr:+.3f}, test r={te_corr:+.3f})")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labels/training_labels_dual.parquet")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"❌ Not found: {src}")
        return

    df = load(str(src))
    train = df[df["zone_time"] < TRAIN_END_DATE]
    test  = df[df["zone_time"] >= TEST_START_DATE]
    print(f"Loaded {len(df):,} rows. Train={len(train):,}  Test={len(test):,}\n")

    all_features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    results = []

    for feat in all_features:
        if feat not in df.columns:
            print(f"  (skip) {feat}: not found in parquet")
            continue

        n_unique = df[feat].nunique(dropna=False)
        if n_unique <= DISCRETE_MAX_UNIQUE:
            res = audit_discrete(feat, train[feat], test[feat],
                                  train["label"], test["label"])
        else:
            res = audit_continuous(feat, train[feat], test[feat],
                                    train["label"], test["label"])
        results.append(res)

    # ── Print flagged features first ────────────────────────────────────────
    flagged = [r for r in results if r["flags"]]
    clean   = [r for r in results if not r["flags"]]

    print("=" * 78)
    print(f"FLAGGED FEATURES ({len(flagged)} of {len(results)}) — look at these first")
    print("=" * 78)
    for r in flagged:
        print(f"\n  {r['name']}  [{r['kind']}]")
        for f in r["flags"]:
            print(f"    ⚠️  {f}")
        if r["kind"] == "discrete":
            tr_disp = {k: f"{v:.1%} (n={r['train_n'].get(k, 0)})" for k, v in r["train_winrates"].items()}
            te_disp = {k: f"{v:.1%} (n={r['test_n'].get(k, 0)})" for k, v in r["test_winrates"].items()}
            print(f"    train win-rates: {tr_disp}")
            print(f"    test  win-rates: {te_disp}")
        else:
            print(f"    train corr={r['train_corr']:+.3f}   test corr={r['test_corr']:+.3f}")

    print("\n" + "=" * 78)
    print(f"CONSISTENT FEATURES ({len(clean)} of {len(results)})")
    print("=" * 78)
    for r in clean:
        if r["kind"] == "discrete":
            spread_tr = r.get("train_spread", float("nan"))
            spread_te = r.get("test_spread", float("nan"))
            print(f"  {r['name']:28s} [discrete]    train spread={spread_tr:+.1%}  test spread={spread_te:+.1%}")
        else:
            print(f"  {r['name']:28s} [continuous]  train r={r['train_corr']:+.3f}  test r={r['test_corr']:+.3f}")

    print(f"""
{"=" * 78}
HOW TO READ THIS
{"=" * 78}
  - SIGN FLIP is the most serious flag: the feature points one direction
    in train and the opposite in test. Worth understanding before trusting
    it (h4_fvg_inside looked surprising but did NOT sign-flip -- same
    direction, different magnitude -- which is why it turned out to be real).
  - MAGNITUDE DROP (strong in train, weak in test) is the classic shape of
    overfitting or a subtle leak that only shows up on repeated/nearby data.
  - NEAR-CONSTANT means the feature rarely varies -- even if "correct," it
    contributes little to the model (same issue found with
    h1_liquidity_sweep separately).
  - A flag here is a prompt to go look at the feature's actual code path
    (like we did for h4_fvg_inside), not proof of a bug by itself.
""")


if __name__ == "__main__":
    main()