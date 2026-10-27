#!/usr/bin/env python3
"""
Comprehensive Comparison: Original Pine Script vs Improved High-Winrate TrendM30 Pro.
Backtested on 5,000 live M30 bars + 1,000 4H bars from Twelve Data
across EUR/USD, GBP/USD, USD/JPY, and XAU/USD.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from advanced_trendm30_research import load_data, precompute_features, simulate_signals, SYMBOLS

def run_comparison():
    print("=" * 85)
    print("  TWELVE DATA BACKTEST & OPTIMIZATION REPORT: TRENDM30 PRO")
    print("=" * 85)

    data = load_data()
    DATA = {}
    for s in SYMBOLS:
        df30, df4h = data[s]
        DATA[s] = precompute_features(df30, df4h, s)

    # 1. ORIGINAL BASELINE (min_score=4, rr=2.0, sl_mode='swing', swing_buf=0.25, market breakout)
    orig_trades = []
    for s in SYMBOLS:
        tr = simulate_signals(
            DATA[s], s,
            min_score=4,
            require_htf=False,
            sl_mode="swing",
            swing_buf=0.25,
            rr_target=2.0,
            use_be=False,
            pullback_entry=False
        )
        orig_trades.extend(tr)

    # 2. IMPROVED STRATEGY A (High Winrate 1:1 R, min_score=6, require_htf=True, swing_buf=0.5, pullback=True)
    imp_a_trades = []
    for s in SYMBOLS:
        tr = simulate_signals(
            DATA[s], s,
            min_score=6,
            require_htf=True,
            sl_mode="swing",
            swing_buf=0.5,
            rr_target=1.0,
            use_be=False,
            pullback_entry=True
        )
        imp_a_trades.extend(tr)

    # 3. IMPROVED STRATEGY B (Balanced 1.2:1 R, min_score=6, require_htf=True, swing_buf=0.5, pullback=True)
    imp_b_trades = []
    for s in SYMBOLS:
        tr = simulate_signals(
            DATA[s], s,
            min_score=6,
            require_htf=True,
            sl_mode="swing",
            swing_buf=0.5,
            rr_target=1.2,
            use_be=False,
            pullback_entry=True
        )
        imp_b_trades.extend(tr)

    # 4. IMPROVED STRATEGY C (ATR Stop 1.5, RR=1.0, min_score=6, require_htf=True, pullback=True)
    imp_c_trades = []
    for s in SYMBOLS:
        tr = simulate_signals(
            DATA[s], s,
            min_score=6,
            require_htf=True,
            sl_mode="atr",
            sl_mult=1.5,
            rr_target=1.0,
            use_be=False,
            pullback_entry=True
        )
        imp_c_trades.extend(tr)

    def summarize(trades, name):
        res = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
        w = sum(1 for t in res if t["outcome"] == "WIN")
        l = sum(1 for t in res if t["outcome"] == "LOSS")
        wr = (w / len(res) * 100) if res else 0.0
        tot_r = sum(t["r_mult"] for t in res)
        gw = sum(t["r_mult"] for t in res if t["r_mult"] > 0)
        gl = abs(sum(t["r_mult"] for t in res if t["r_mult"] < 0))
        pf = (gw / gl) if gl > 0 else 0.0
        
        # Max Drawdown in R
        cum_r = np.cumsum([t["r_mult"] for t in res])
        peak = np.maximum.accumulate(cum_r)
        dd = peak - cum_r
        max_dd = np.max(dd) if len(dd) > 0 else 0.0

        return {
            "Strategy": name,
            "Total Trades": len(res),
            "Wins": w,
            "Losses": l,
            "Win Rate (%)": f"{wr:.1f}%",
            "Net Return (R)": f"{tot_r:+.1f} R",
            "Profit Factor": f"{pf:.2f}",
            "Max Drawdown (R)": f"{max_dd:.1f} R"
        }

    summary_rows = [
        summarize(orig_trades, "1. Original Pine Script (Baseline)"),
        summarize(imp_a_trades, "2. Improved A (Pullback, Score>=6, RR=1.0, Swing0.5)"),
        summarize(imp_b_trades, "3. Improved B (Pullback, Score>=6, RR=1.2, Swing0.5)"),
        summarize(imp_c_trades, "4. Improved C (Pullback, Score>=6, RR=1.0, ATR 1.5)"),
    ]

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False))

    # Per-Symbol Comparison Table
    print("\n" + "=" * 85)
    print("  PER-SYMBOL BREAKDOWN: ORIGINAL BASELINE vs IMPROVED STRATEGY A")
    print("=" * 85)

    df_orig = pd.DataFrame([t for t in orig_trades if t["outcome"] in ("WIN", "LOSS")])
    df_impa = pd.DataFrame([t for t in imp_a_trades if t["outcome"] in ("WIN", "LOSS")])

    sym_rows = []
    for s in SYMBOLS:
        sub_o = df_orig[df_orig["symbol"] == s]
        sub_i = df_impa[df_impa["symbol"] == s]

        wo = (sub_o["outcome"] == "WIN").sum()
        lo = (sub_o["outcome"] == "LOSS").sum()
        wro = wo / len(sub_o) * 100 if len(sub_o) else 0.0
        ro = sub_o["r_mult"].sum()

        wi = (sub_i["outcome"] == "WIN").sum()
        li = (sub_i["outcome"] == "LOSS").sum()
        wri = wi / len(sub_i) * 100 if len(sub_i) else 0.0
        ri = sub_i["r_mult"].sum()

        sym_rows.append({
            "Symbol": s,
            "Base Trades": len(sub_o),
            "Base WR": f"{wro:.1f}%",
            "Base Return": f"{ro:+.1f} R",
            "Imp Trades": len(sub_i),
            "Imp Wins": wi,
            "Imp Losses": li,
            "Imp WR": f"{wri:.1f}%",
            "Imp Return": f"{ri:+.1f} R",
            "WR Gain": f"+{wri - wro:.1f}%"
        })

    df_sym = pd.DataFrame(sym_rows)
    print(df_sym.to_string(index=False))

    # Save detailed CSV for improved A
    Path("backtest_results").mkdir(exist_ok=True)
    df_impa.to_csv("backtest_results/trendm30_improved_trades.csv", index=False)
    print("\nSaved improved trade log to backtest_results/trendm30_improved_trades.csv")

if __name__ == "__main__":
    run_comparison()
