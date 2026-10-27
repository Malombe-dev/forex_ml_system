#!/usr/bin/env python3
"""
analyze_rescore.py -- does the ARRIVAL-time probability predict outcomes better than the PROPOSAL-time one?

Reads the per-signal JSONL from a two-stage backtest run (<out_dir>/logs/replay_*_signals.jsonl),
keeps trades that actually filled and resolved (hit_tp / hit_sl) and were re-scored successfully,
then compares the two scores head to head.

    python3 analyze_rescore.py bt_twostage
    python3 analyze_rescore.py bt_twostage/logs/replay_XXXX_signals.jsonl --exclude-blocked

What it reports
  * AUC of each score for "trade won" (0.5 = no information) + a paired bootstrap CI on the difference
  * Spearman correlation of each score with the trade's R
  * Win rate / avg R by score quartile
  * Threshold sweep: what you'd have kept and earned if you gated on each score (this is the
    --rescore-min-prob decision, made on data instead of by guess)

Small samples lie. Under ~60 resolved trades every number here is anecdotal; the script says so.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd


def load_rows(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "replay_*_signals.jsonl"), recursive=True)
            files += glob.glob(os.path.join(p, "replay_*_signals.jsonl"))
        else:
            files.append(p)
    rows = []
    for f in sorted(set(files)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "outcome" in r:
                    r["_file"] = os.path.basename(f)
                    rows.append(r)
    return rows, sorted(set(files))


def auc(y, s):
    """Rank-based AUC (Mann-Whitney). 0.5 = no information."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc_diff(y, a, b, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, bool); a = np.asarray(a, float); b = np.asarray(b, float)
    m = len(y); diffs = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        da, db = auc(y[i], a[i]), auc(y[i], b[i])
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    d = np.array(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def quartiles(df, col):
    try:
        q = pd.qcut(df[col], 4, duplicates="drop")
    except ValueError:
        return None
    g = df.groupby(q, observed=True).agg(n=("win", "size"), win_pct=("win", lambda x: 100 * x.mean()),
                                         avg_R=("R", "mean"), total_R=("R", "sum"))
    return g.round(3)


def sweep(df, col, thresholds):
    out = []
    for t in thresholds:
        k = df[df[col] >= t]
        out.append({"gate": f">= {t:g}%", "kept": len(k), "kept_pct": round(100 * len(k) / max(len(df), 1), 1),
                    "win_pct": round(100 * k["win"].mean(), 1) if len(k) else np.nan,
                    "total_R": round(k["R"].sum(), 2), "R_per_trade": round(k["R"].mean(), 3) if len(k) else np.nan})
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="backtest out dir(s) or replay_*_signals.jsonl file(s)")
    ap.add_argument("--exclude-blocked", action="store_true",
                    help="drop trades live's one-trade-per-symbol guard would have blocked")
    ap.add_argument("--thresholds", type=str, default="40,45,50,55,60,65,70,75,80")
    args = ap.parse_args(argv)

    rows, files = load_rows(args.paths)
    if not rows:
        print("No signal rows found. Pass the backtest out dir or the replay_*_signals.jsonl file."); return 1
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} signal rows from {len(files)} file(s).")

    for c in ("arrival_prob", "prob_at_proposal", "r_multiple"):
        if c not in df.columns:
            df[c] = np.nan
    if "arrival_ok" not in df.columns:
        print("No arrival_prob column: this run was not two-stage with re-scoring "
              "(need --two-stage and NOT --no-rescore)."); return 1

    filled = df[df["outcome"].isin(["hit_tp", "hit_sl"])]
    print(f"  filled & resolved (hit_tp/hit_sl): {len(filled)}")
    if "two_stage_end" in df.columns:
        print("  zones that never filled, by reason:",
              df["two_stage_end"].value_counts(dropna=True).to_dict() or "none")
    fail = filled[filled["arrival_ok"] != True]                                    # noqa: E712
    if len(fail):
        print(f"  re-score FAILED on {len(fail)} filled trades (excluded below): "
              f"{fail['rescore_reason'].value_counts().to_dict()}")
    d = filled[(filled["arrival_ok"] == True) & filled["arrival_prob"].notna()      # noqa: E712
               & filled["prob_at_proposal"].notna() & filled["r_multiple"].notna()].copy()
    if args.exclude_blocked and "would_be_blocked_by_open_trade" in d.columns:
        nb = int(d["would_be_blocked_by_open_trade"].fillna(False).sum())
        d = d[~d["would_be_blocked_by_open_trade"].fillna(False)]
        print(f"  --exclude-blocked dropped {nb} trades")
    if "rescore_suspect_0p5" in d.columns:
        sus = int((d["rescore_suspect_0p5"] == True).sum())                         # noqa: E712
        if sus:
            print(f"  ⚠️  {sus} trades scored exactly 0.500 (the model's feature-failure default) — inspect/exclude them")

    d["win"] = (d["outcome"] == "hit_tp").astype(int)
    d["R"] = d["r_multiple"].astype(float)
    n = len(d)
    print(f"\nUsable trades: {n}   wins: {int(d['win'].sum())} ({100 * d['win'].mean():.1f}%)   total R: {d['R'].sum():+.2f}   "
          f"R/trade: {d['R'].mean():+.3f}" if n else "\nNo usable trades.")
    if n < 10:
        print("Too few trades to say anything."); return 0
    if n < 60:
        print(f"⚠️  n={n} < 60: treat every number below as anecdotal, not evidence.")

    a_auc, p_auc = auc(d["win"], d["arrival_prob"]), auc(d["win"], d["prob_at_proposal"])
    lo, hi, p_better = boot_auc_diff(d["win"], d["arrival_prob"], d["prob_at_proposal"])
    print("\n── Discrimination (does the score rank winners above losers?) ──")
    print(f"  AUC  arrival-time : {a_auc:.3f}")
    print(f"  AUC  proposal-time: {p_auc:.3f}      (0.500 = coin flip)")
    print(f"  AUC difference (arrival - proposal): {a_auc - p_auc:+.3f}   95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]   "
          f"P(arrival better)={p_better:.2f}")
    print(f"  Spearman(score, R)  arrival: {d['arrival_prob'].corr(d['R'], method='spearman'):+.3f}   "
          f"proposal: {d['prob_at_proposal'].corr(d['R'], method='spearman'):+.3f}")
    print(f"  mean |arrival - proposal| = {(d['arrival_prob'] - d['prob_at_proposal']).abs().mean():.1f} pts "
          f"(if ~0 the re-score barely changed anything)")

    for col, name in (("arrival_prob", "ARRIVAL"), ("prob_at_proposal", "PROPOSAL")):
        q = quartiles(d, col)
        if q is not None:
            print(f"\n── {name}-time score by quartile ──"); print(q.to_string())

    ths = [float(x) for x in args.thresholds.split(",")]
    for col, name in (("arrival_prob", "ARRIVAL"), ("prob_at_proposal", "PROPOSAL")):
        print(f"\n── Gate on {name}-time score (what you'd keep / earn) ──"); print(sweep(d, col, ths).to_string(index=False))

    print("\n── Reading this ──")
    if n >= 60 and np.isfinite(lo) and lo > 0:
        print("  Arrival score ranks trades better than the proposal score, beyond bootstrap noise. "
              "Reasonable to try --rescore-min-prob on a fresh window (don't tune the gate on this same data).")
    elif n >= 60 and np.isfinite(hi) and hi < 0:
        print("  Arrival score is WORSE than the proposal score here. Do not gate on it; look at which refreshed features moved.")
    else:
        print("  No reliable difference. Keep re-scoring log-only; more data or a longer window is needed before gating.")
    print("  Caveat: a gate picked from the sweep above and evaluated on the same trades will look better than it is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
