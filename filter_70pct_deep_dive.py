#!/usr/bin/env python3
"""
Deep Multi-Factor Filter Discovery for 70%+ Win Rate Trades.
Tests:
  1. 4H Slope (HTF Trend Velocity)
  2. Order Block / Demand Zone presence at retest
  3. ATR Volatility Expansion Ratio
  4. MA60 Distance & M30 EMA21 Slope
  5. RSI Pullback Sweet Spot (healthy retest vs exhaustion)
  6. Interaction / Combinations of above
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from advanced_trendm30_research import load_data, precompute_features, SYMBOLS

def run_deep_filter_analysis():
    data = load_data()
    all_trades = []

    for sym in SYMBOLS:
        df30, df4h = data[sym]
        df = precompute_features(df30, df4h, sym)
        n = len(df)
        mintick = 0.01 if ("JPY" in sym or "XAU" in sym) else 0.0001

        # 4H EMA50 Slope
        # We can calculate 4H slope on the 4H dataframe and merge it
        df4h_calc = df4h.copy().sort_values("datetime").reset_index(drop=True)
        df4h_calc["ema50"] = df4h_calc["close"].ewm(span=50, adjust=False).mean()
        df4h_calc["ema50_slope"] = (df4h_calc["ema50"] - df4h_calc["ema50"].shift(3)) # 3-bar slope
        df4h_calc["datetime"] = df4h_calc["datetime"] + pd.Timedelta(hours=4)

        merged = pd.merge_asof(
            df.sort_values("datetime"),
            df4h_calc[["datetime", "ema50_slope"]],
            on="datetime",
            suffixes=("", "_htf")
        )
        df["htf_slope"] = merged["ema50_slope"].fillna(0.0)
        df["htf_slope_norm"] = df["htf_slope"] / (df["atr14"] + 1e-10) # normalized by ATR

        # Order block presence: in_dem / in_sup computed in precompute_features (c7 components)
        # Distance from MA60
        df["dist_ma60"] = (df["close"] - df["white"]).abs() / (df["atr14"] + 1e-10)
        # ATR ratio
        df["atr_ratio"] = df["atr14"] / (df["atr_sma50"] + 1e-10)

        # Baseline High-WR setup
        cond_bull = df["bull_align"] & df["white_up"] & (df["score_bull"] >= 5) & df["c1_bull"] & df["c4_bull"]
        cond_bear = df["bear_align"] & df["white_down"] & (df["score_bear"] >= 5) & df["c1_bear"] & df["c4_bear"]

        sig_bull = cond_bull & (~cond_bull.shift(1).fillna(False))
        sig_bear = cond_bear & (~cond_bear.shift(1).fillna(False))

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        yellows = df["yellow"].values
        atrs = df["atr14"].values
        sw10_los = df["sw10_lo"].values
        sw10_his = df["sw10_hi"].values
        sig_b_arr = sig_bull.values
        sig_s_arr = sig_bear.values

        i = 100
        while i < n - 1:
            is_l = sig_b_arr[i]
            is_s = sig_s_arr[i]
            if not is_l and not is_s:
                i += 1
                continue
            is_long = is_l

            entered = False
            for pb in range(i, min(i + 8, n)):
                if is_long:
                    if lows[pb] <= yellows[pb] + 0.3 * atrs[pb]:
                        entry_p = yellows[pb]
                        e_bar = pb
                        entered = True
                        break
                else:
                    if highs[pb] >= yellows[pb] - 0.3 * atrs[pb]:
                        entry_p = yellows[pb]
                        e_bar = pb
                        entered = True
                        break
            if not entered:
                i += 1
                continue

            atr_cur = atrs[e_bar]
            sl = (sw10_los[e_bar] - 0.5 * atr_cur) if is_long else (sw10_his[e_bar] + 0.5 * atr_cur)
            risk = abs(entry_p - sl)
            tp1 = (entry_p + risk * 1.0) if is_long else (entry_p - risk * 1.0)
            tp2 = (entry_p + risk * 2.0) if is_long else (entry_p - risk * 2.0)

            # Outcomes
            out_single = "LOSS"
            # Dual target tracking
            tp1_hit = False
            cur_sl = sl
            r_dual = 0.0

            for j in range(e_bar + 1, min(e_bar + 300, n)):
                h, l = highs[j], lows[j]
                if not tp1_hit:
                    hit_tp1 = (h >= tp1) if is_long else (l <= tp1)
                    hit_sl = (l <= cur_sl) if is_long else (h >= cur_sl)
                    if hit_sl:
                        r_dual = -1.0
                        out_single = "LOSS"
                        break
                    elif hit_tp1:
                        tp1_hit = True
                        out_single = "WIN"
                        cur_sl = entry_p # move to BE
                if tp1_hit:
                    hit_tp2 = (h >= tp2) if is_long else (l <= tp2)
                    hit_be = (l <= cur_sl) if is_long else (h >= cur_sl)
                    if hit_tp2:
                        r_dual = +1.5 # 0.5*1.0 + 0.5*2.0 = +1.5R
                        break
                    elif hit_be:
                        r_dual = +0.5 # 0.5*1.0 + 0.5*0.0 = +0.5R
                        break

            all_trades.append({
                "symbol": sym,
                "direction": "LONG" if is_long else "SHORT",
                "out_single": out_single,
                "is_win_single": 1 if out_single == "WIN" else 0,
                "is_win_dual": 1 if r_dual > 0 else 0,
                "r_dual": r_dual,
                # Candidate filter metrics at entry
                "htf_slope": df["htf_slope_norm"].iloc[e_bar] if is_long else -df["htf_slope_norm"].iloc[e_bar],
                "dist_ma60": df["dist_ma60"].iloc[e_bar],
                "atr_ratio": df["atr_ratio"].iloc[e_bar],
                "rsi": df["rsi"].iloc[e_bar],
                "adx": df["adx"].iloc[e_bar],
                "in_ob_zone": int(df["c7_bull"].iloc[e_bar] if is_long else df["c7_bear"].iloc[e_bar]),
                "day": df["datetime"].dt.day_name().iloc[e_bar],
                "hour": df["datetime"].dt.hour.iloc[e_bar]
            })
            i = j + 1

    return pd.DataFrame(all_trades)

def main():
    df = run_deep_filter_analysis()
    print("=" * 85)
    print(f"  FACTOR ANALYSIS: ISOLATING THE 70%+ TO 80% WIN-RATE TRADES")
    print(f"  Sample: {len(df)} Trades | Dual Profitable: {(df['is_win_dual']==1).sum()} ({(df['is_win_dual']==1).mean()*100:.1f}%)")
    print("=" * 85)

    # 1. Test 4H EMA50 Slope
    print("\n1. 4-HOUR EMA50 SLOPE (HTF Trend Strength):")
    print("-" * 85)
    for q in [0.0, 0.05, 0.10, 0.20]:
        sub = df[df["htf_slope"] >= q]
        wr_dual = sub["is_win_dual"].mean() * 100
        wr_s = sub["is_win_single"].mean() * 100
        net_r = sub["r_dual"].sum()
        print(f"  4H Slope >= {q:>4.2f} ATR -> Trades: {len(sub):<3} | Dual WR: {wr_dual:>5.1f}% | Single WR: {wr_s:>5.1f}% | Net Dual R: {net_r:>+6.1f} R")

    # 2. Distance from MA60 (White Line Separation)
    print("\n2. DISTANCE FROM MA60 (White Line Trend Maturity):")
    print("-" * 85)
    for d in [0.5, 1.0, 1.5, 2.0]:
        sub = df[df["dist_ma60"] >= d]
        wr_dual = sub["is_win_dual"].mean() * 100
        wr_s = sub["is_win_single"].mean() * 100
        net_r = sub["r_dual"].sum()
        print(f"  Dist from MA60 >= {d:>3.1f} ATR -> Trades: {len(sub):<3} | Dual WR: {wr_dual:>5.1f}% | Single WR: {wr_s:>5.1f}% | Net Dual R: {net_r:>+6.1f} R")

    # 3. ATR Volatility Expansion Ratio
    print("\n3. ATR VOLATILITY EXPANSION (ATR14 / SMA(ATR, 50)):")
    print("-" * 85)
    for a in [0.9, 1.0, 1.1, 1.2]:
        sub = df[df["atr_ratio"] >= a]
        wr_dual = sub["is_win_dual"].mean() * 100
        wr_s = sub["is_win_single"].mean() * 100
        net_r = sub["r_dual"].sum()
        print(f"  ATR Ratio >= {a:>3.1f} -> Trades: {len(sub):<3} | Dual WR: {wr_dual:>5.1f}% | Single WR: {wr_s:>5.1f}% | Net Dual R: {net_r:>+6.1f} R")

    # 4. RSI Pullback Range
    print("\n4. RSI PULLBACK LEVEL (Sweet Spot at Retest):")
    print("-" * 85)
    sub_rsi1 = df[(df["rsi"] >= 40) & (df["rsi"] <= 60)]
    sub_rsi2 = df[(df["rsi"] >= 45) & (df["rsi"] <= 55)]
    print(f"  RSI between 40 - 60 -> Trades: {len(sub_rsi1):<3} | Dual WR: {sub_rsi1['is_win_dual'].mean()*100:>5.1f}% | Net R: {sub_rsi1['r_dual'].sum():>+6.1f} R")
    print(f"  RSI between 45 - 55 -> Trades: {len(sub_rsi2):<3} | Dual WR: {sub_rsi2['is_win_dual'].mean()*100:>5.1f}% | Net R: {sub_rsi2['r_dual'].sum():>+6.1f} R")

    # 5. Combined Multi-Factor High-Probability Filter
    print("\n" + "=" * 85)
    print("  5. COMBINED MULTI-FACTOR FILTERS (TARGETING 75% - 80%+ WIN RATE)")
    print("=" * 85)

    filters = [
        ("Base Pullback (HTF Aligned + Momentum)", df),
        ("Rule 1: Base + Dist MA60 >= 1.0 ATR", df[df["dist_ma60"] >= 1.0]),
        ("Rule 2: Base + Dist MA60 >= 1.0 + 4H Slope > 0", df[(df["dist_ma60"] >= 1.0) & (df["htf_slope"] > 0)]),
        ("Rule 3: Base + Dist MA60 >= 1.2 + 4H Slope > 0.05", df[(df["dist_ma60"] >= 1.2) & (df["htf_slope"] >= 0.05)]),
        ("Rule 4: Rule 3 + ADX >= 25", df[(df["dist_ma60"] >= 1.2) & (df["htf_slope"] >= 0.05) & (df["adx"] >= 25)]),
        ("Rule 5: Rule 4 + Exclude Friday", df[(df["dist_ma60"] >= 1.2) & (df["htf_slope"] >= 0.05) & (df["adx"] >= 25) & (df["day"] != "Friday")]),
    ]

    for name, sub in filters:
        n_tr = len(sub)
        wr_dual = sub["is_win_dual"].mean() * 100 if n_tr else 0
        wr_s = sub["is_win_single"].mean() * 100 if n_tr else 0
        net_r = sub["r_dual"].sum()
        pf = (sub[sub["r_dual"] > 0]["r_dual"].sum() / abs(sub[sub["r_dual"] < 0]["r_dual"].sum())) if len(sub[sub["r_dual"] < 0]) else 0
        print(f"{name:<50} | Trades: {n_tr:<3} | Single WR: {wr_s:>5.1f}% | Dual WR: {wr_dual:>5.1f}% | Net R: {net_r:>+6.1f} R | PF: {pf:.2f}")

if __name__ == "__main__":
    main()
