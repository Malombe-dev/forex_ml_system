#!/usr/bin/env python3
"""
Test Combining ALL Factors Together on 24,000 Twelve Data bars.
Evaluates progressive layers from Baseline up to The Ultimate Combined Filter.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from advanced_trendm30_research import load_data, precompute_features, SYMBOLS

def run_all_factors_test():
    data = load_data()
    all_trade_records = []

    for sym in SYMBOLS:
        df30, df4h = data[sym]
        df = precompute_features(df30, df4h, sym)
        n = len(df)
        mintick = 0.01 if ("JPY" in sym or "XAU" in sym) else 0.0001

        # 4H EMA50 Slope
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
        df["htf_slope_norm"] = df["htf_slope"] / (df["atr14"] + 1e-10)

        # Distance from MA60
        df["dist_ma60"] = (df["close"] - df["white"]).abs() / (df["atr14"] + 1e-10)
        # ATR ratio
        df["atr_ratio"] = df["atr14"] / (df["atr_sma50"] + 1e-10)
        # Day of week
        df["day_name"] = df["datetime"].dt.day_name()
        df["hour_val"] = df["datetime"].dt.hour

        # Base signal trigger
        cond_bull = df["bull_align"] & df["white_up"] & (df["score_bull"] >= 5) & df["c1_bull"]
        cond_bear = df["bear_align"] & df["white_down"] & (df["score_bear"] >= 5) & df["c1_bear"]

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

            # Single target 1.0R & 1.5R tracking
            tp15 = (entry_p + risk * 1.5) if is_long else (entry_p - risk * 1.5)
            out_10 = "LOSS"
            out_15 = "LOSS"
            
            # Dual target tracking
            tp1_hit = False
            r_dual = 0.0
            cur_sl = sl

            for j in range(e_bar + 1, min(e_bar + 300, n)):
                h, l = highs[j], lows[j]
                
                # Single 1.0R
                if out_10 == "LOSS":
                    hit_tp10 = (h >= tp1) if is_long else (l <= tp1)
                    hit_sl10 = (l <= sl) if is_long else (h >= sl)
                    if hit_tp10: out_10 = "WIN"
                    
                # Single 1.5R
                if out_15 == "LOSS":
                    hit_tp15 = (h >= tp15) if is_long else (l <= tp15)
                    hit_sl15 = (l <= sl) if is_long else (h >= sl)
                    if hit_tp15: out_15 = "WIN"

                # Dual target
                if not tp1_hit:
                    hit_tp1 = (h >= tp1) if is_long else (l <= tp1)
                    hit_sl1 = (l <= cur_sl) if is_long else (h >= cur_sl)
                    if hit_sl1:
                        r_dual = -1.0
                        break
                    elif hit_tp1:
                        tp1_hit = True
                        cur_sl = entry_p # move to BE
                else:
                    hit_tp2 = (h >= tp2) if is_long else (l <= tp2)
                    hit_be = (l <= cur_sl) if is_long else (h >= cur_sl)
                    if hit_tp2:
                        r_dual = +1.5
                        break
                    elif hit_be:
                        r_dual = +0.5
                        break

            all_trade_records.append({
                "symbol": sym,
                "direction": "LONG" if is_long else "SHORT",
                "out_10": out_10,
                "out_15": out_15,
                "r_dual": r_dual,
                "is_win_10": 1 if out_10 == "WIN" else 0,
                "is_win_15": 1 if out_15 == "WIN" else 0,
                "is_win_dual": 1 if r_dual > 0 else 0,
                # All candidate factors
                "dist_ma60": df["dist_ma60"].iloc[e_bar],
                "htf_slope": df["htf_slope_norm"].iloc[e_bar] if is_long else -df["htf_slope_norm"].iloc[e_bar],
                "c4_momentum": int(df["c4_bull"].iloc[e_bar] if is_long else df["c4_bear"].iloc[e_bar]),
                "c3_breakout": int(df["c3_bull"].iloc[e_bar] if is_long else df["c3_bear"].iloc[e_bar]),
                "atr_ratio": df["atr_ratio"].iloc[e_bar],
                "adx": df["adx"].iloc[e_bar],
                "day": df["day_name"].iloc[e_bar],
                "hour": df["hour_val"].iloc[e_bar],
                "score": df["score_bull"].iloc[i] if is_long else df["score_bear"].iloc[i]
            })
            i = j + 1

    return pd.DataFrame(all_trade_records)

def evaluate_subset(sub, label):
    n = len(sub)
    if n == 0:
        print(f"{label:<55} | No trades")
        return
    wr_10 = sub["is_win_10"].mean() * 100
    wr_15 = sub["is_win_15"].mean() * 100
    wr_dual = sub["is_win_dual"].mean() * 100
    net_r_dual = sub["r_dual"].sum()
    gw = sub[sub["r_dual"] > 0]["r_dual"].sum()
    gl = abs(sub[sub["r_dual"] < 0]["r_dual"].sum())
    pf = (gw / gl) if gl > 0 else 0.0

    print(f"{label:<55} | Trades: {n:<3} | 1.0R WR: {wr_10:>5.1f}% | 1.5R WR: {wr_15:>5.1f}% | Dual WR: {wr_dual:>5.1f}% | Net Dual R: {net_r_dual:>+6.1f} R | PF: {pf:.2f}")

def main():
    df = run_all_factors_test()
    print("=" * 115)
    print("  WHAT IF WE COMBINE ALL FACTORS? — PROGRESSIVE CONFLUENCE STUDY")
    print(f"  Total Trades Tested: {len(df)} on EUR/USD, GBP/USD, USD/JPY, XAU/USD (July-Oct 2026)")
    print("=" * 115)

    evaluate_subset(df, "Level 0: Pullback Base (HTF + EMA21 Retest)")
    
    # Layer 1: + Momentum
    sub1 = df[df["c4_momentum"] == 1]
    evaluate_subset(sub1, "Level 1: Level 0 + MACD & RSI Momentum")

    # Layer 2: + MA60 Separation (No chop on white line)
    sub2 = sub1[sub1["dist_ma60"] >= 1.0]
    evaluate_subset(sub2, "Level 2: Level 1 + MA60 Separation >= 1.0 ATR")

    # Layer 3: + 4H Active Slope
    sub3 = sub2[sub2["htf_slope"] > 0.02]
    evaluate_subset(sub3, "Level 3: Level 2 + Active 4H Slope > 0.02")

    # Layer 4: + Volatility Expansion (ATR Ratio >= 1.0)
    sub4 = sub3[sub3["atr_ratio"] >= 1.0]
    evaluate_subset(sub4, "Level 4: Level 3 + Volatility Expansion (ATR >= Avg)")

    # Layer 5: + ADX Trend Velocity (ADX >= 25)
    sub5 = sub4[sub4["adx"] >= 25]
    evaluate_subset(sub5, "Level 5: Level 4 + ADX Trend Strength >= 25")

    # Layer 6: + Structural Breakout (Check 3 confirmed)
    sub6 = sub5[sub5["c3_breakout"] == 1]
    evaluate_subset(sub6, "Level 6: Level 5 + Structural Breakout Confirmed")

    # Layer 7: + Exclude Friday Late Session (Weekend risk)
    sub7 = sub6[~((sub6["day"] == "Friday") & (sub6["hour"] >= 14))]
    evaluate_subset(sub7, "Level 7 (THE APEX): All Factors Combined (Excl Fri)")

    # Layer 8: Ultra-Conservative (Dist MA60 >= 1.5 ATR + All above)
    sub8 = sub7[sub7["dist_ma60"] >= 1.5]
    evaluate_subset(sub8, "Level 8 (HOLY GRAIL): Apex + MA60 Separation >= 1.5")

if __name__ == "__main__":
    main()
