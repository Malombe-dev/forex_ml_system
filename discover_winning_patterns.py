#!/usr/bin/env python3
"""
Deep Data Science & Feature Discovery:
What do WINNING trades have in common compared to LOSING trades?
Analyzes ~20,000 bars from Twelve Data across EUR/USD, GBP/USD, USD/JPY, XAU/USD.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from advanced_trendm30_research import load_data, precompute_features, SYMBOLS

def extract_detailed_trade_features():
    data = load_data()
    all_trade_records = []

    for sym in SYMBOLS:
        df30, df4h = data[sym]
        df = precompute_features(df30, df4h, sym)
        n = len(df)
        mintick = 0.01 if ("JPY" in sym or "XAU" in sym) else 0.0001

        # Additional feature calculations
        df["ma60_slope_5"] = (df["white"] - df["white"].shift(5)) / (df["atr14"] + 1e-10)
        df["ema21_slope_5"] = (df["yellow"] - df["yellow"].shift(5)) / (df["atr14"] + 1e-10)
        df["dist_ma60_atr"] = (df["close"] - df["white"]).abs() / (df["atr14"] + 1e-10)
        df["dist_ema21_atr"] = (df["close"] - df["yellow"]).abs() / (df["atr14"] + 1e-10)
        df["day_of_week"] = df["datetime"].dt.day_name()
        df["session"] = np.where(
            (df["hour"] >= 7) & (df["hour"] < 13), "London",
            np.where((df["hour"] >= 13) & (df["hour"] < 18), "New York",
            np.where((df["hour"] >= 18) & (df["hour"] < 21), "London/NY Close", "Asian"))
        )

        # 50-bar swing range position (0.0 = at low, 1.0 = at high)
        hi50 = df["high"].rolling(50).max()
        lo50 = df["low"].rolling(50).min()
        df["swing50_pos"] = (df["close"] - lo50) / (hi50 - lo50 + 1e-10)

        # Signal trigger (Score >= 5, HTF aligned)
        cond_bull = df["bull_align"] & df["white_up"] & (df["score_bull"] >= 5) & df["c1_bull"]
        cond_bear = df["bear_align"] & df["white_down"] & (df["score_bear"] >= 5) & df["c1_bear"]
        sig_bull = cond_bull & (~cond_bull.shift(1).fillna(False))
        sig_bear = cond_bear & (~cond_bear.shift(1).fillna(False))

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atrs = df["atr14"].values
        yellows = df["yellow"].values
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
            
            # Pullback to EMA21 within 8 bars
            entered = False
            entry_bar = i
            entry_price = closes[i]
            atr_cur = atrs[i]

            for pb in range(i, min(i + 8, n)):
                if is_long:
                    if lows[pb] <= yellows[pb] + 0.3 * atrs[pb]:
                        entry_bar = pb
                        entry_price = yellows[pb]
                        atr_cur = atrs[pb]
                        entered = True
                        break
                else:
                    if highs[pb] >= yellows[pb] - 0.3 * atrs[pb]:
                        entry_bar = pb
                        entry_price = yellows[pb]
                        atr_cur = atrs[pb]
                        entered = True
                        break
            if not entered:
                i += 1
                continue

            sl = (sw10_los[entry_bar] - 0.5 * atr_cur) if is_long else (sw10_his[entry_bar] + 0.5 * atr_cur)
            min_risk = 5.0 * mintick
            sl = min(sl, entry_price - min_risk) if is_long else max(sl, entry_price + min_risk)
            risk = abs(entry_price - sl)
            tp = (entry_price + risk * 1.0) if is_long else (entry_price - risk * 1.0)

            outcome = "OPEN"
            exit_bar = entry_bar
            for j in range(entry_bar + 1, min(entry_bar + 300, n)):
                hit_tp = (highs[j] >= tp) if is_long else (lows[j] <= tp)
                hit_sl = (lows[j] <= sl) if is_long else (highs[j] >= sl)
                if hit_tp and hit_sl:
                    outcome = "LOSS"
                    exit_bar = j
                    break
                elif hit_tp:
                    outcome = "WIN"
                    exit_bar = j
                    break
                elif hit_sl:
                    outcome = "LOSS"
                    exit_bar = j
                    break

            if outcome in ("WIN", "LOSS"):
                rec = {
                    "symbol": sym,
                    "direction": "LONG" if is_long else "SHORT",
                    "outcome": outcome,
                    "is_win": 1 if outcome == "WIN" else 0,
                    "score": df["score_bull"].iloc[i] if is_long else df["score_bear"].iloc[i],
                    "c1_htf": 1,
                    "c2_adx": int(df["c2"].iloc[i]),
                    "c3_breakout": int(df["c3_bull"].iloc[i] if is_long else df["c3_bear"].iloc[i]),
                    "c4_momentum": int(df["c4_bull"].iloc[i] if is_long else df["c4_bear"].iloc[i]),
                    "c5_volatility": int(df["c5"].iloc[i]),
                    "c6_room": int(df["c6"].iloc[i]),
                    "c7_smc": int(df["c7_bull"].iloc[i] if is_long else df["c7_bear"].iloc[i]),
                    "adx_val": df["adx"].iloc[entry_bar],
                    "rsi_val": df["rsi"].iloc[entry_bar],
                    "dist_ma60_atr": df["dist_ma60_atr"].iloc[entry_bar],
                    "dist_ema21_atr": df["dist_ema21_atr"].iloc[entry_bar],
                    "ma60_slope": df["ma60_slope_5"].iloc[entry_bar] if is_long else -df["ma60_slope_5"].iloc[entry_bar],
                    "swing50_pos": df["swing50_pos"].iloc[entry_bar] if is_long else (1.0 - df["swing50_pos"].iloc[entry_bar]),
                    "session": df["session"].iloc[entry_bar],
                    "day_of_week": df["day_of_week"].iloc[entry_bar],
                    "risk_pips": risk / (mintick * 10),
                    "bars_to_fill": entry_bar - i
                }
                all_trade_records.append(rec)

            i = exit_bar + 1

    return pd.DataFrame(all_trade_records)

def main():
    df = extract_detailed_trade_features()
    print("=" * 85)
    print(f"  QUANTITATIVE DISCOVERY: WHAT DO WINNING TRADES HAVE IN COMMON?")
    print(f"  Sample: {len(df)} Trades | Wins: {(df['outcome']=='WIN').sum()} | Losses: {(df['outcome']=='LOSS').sum()}")
    print("=" * 85)

    wins = df[df["outcome"] == "WIN"]
    losses = df[df["outcome"] == "LOSS"]

    print("\n1. NUMERICAL FEATURE COMPARISON (Average Win vs Average Loss):")
    print("-" * 85)
    num_cols = ["adx_val", "rsi_val", "dist_ma60_atr", "ma60_slope", "swing50_pos", "bars_to_fill", "risk_pips"]
    stats = []
    for col in num_cols:
        mw = wins[col].mean()
        ml = losses[col].mean()
        diff = mw - ml
        stats.append({
            "Feature": col,
            "Wins Avg": round(mw, 3),
            "Losses Avg": round(ml, 3),
            "Difference": round(diff, 3),
            "Correlation with Win": round(df[col].corr(df["is_win"]), 3)
        })
    print(pd.DataFrame(stats).to_string(index=False))

    print("\n2. CONFLUENCE CHECKS IMPACT ON WIN RATE:")
    print("-" * 85)
    for c in ["c2_adx", "c3_breakout", "c4_momentum", "c5_volatility", "c6_room", "c7_smc"]:
        wr_true = df[df[c] == 1]["is_win"].mean() * 100
        n_true = (df[c] == 1).sum()
        wr_false = df[df[c] == 0]["is_win"].mean() * 100
        n_false = (df[c] == 0).sum()
        print(f"  {c:<15}: When TRUE -> WR = {wr_true:>5.1f}% (n={n_true:<3}) | When FALSE -> WR = {wr_false:>5.1f}% (n={n_false:<3}) | Edge: {wr_true - wr_false:>+5.1f}%")

    print("\n3. SWING CYCLE POSITION (Are Wins at Major Swings?):")
    print("-" * 85)
    df["swing_bucket"] = pd.qcut(df["swing50_pos"], q=4, labels=["Early Swing (0-25%)", "Mid-Early (25-50%)", "Mid-Late (50-75%)", "Late Swing (75-100%)"])
    print(df.groupby("swing_bucket")["is_win"].agg(trades="count", wins="sum", win_rate=lambda x: round(x.mean() * 100, 1)))

    print("\n4. MA60 DISTANCE (Room to white line):")
    print("-" * 85)
    df["ma60_bucket"] = pd.qcut(df["dist_ma60_atr"], q=3, labels=["Close (< 1.5 ATR)", "Moderate (1.5 - 3.0 ATR)", "Far (> 3.0 ATR)"])
    print(df.groupby("ma60_bucket")["is_win"].agg(trades="count", wins="sum", win_rate=lambda x: round(x.mean() * 100, 1)))

    print("\n5. TRADING SESSION IMPACT:")
    print("-" * 85)
    print(df.groupby("session")["is_win"].agg(trades="count", wins="sum", win_rate=lambda x: round(x.mean() * 100, 1)))

    print("\n6. DAY OF THE WEEK IMPACT:")
    print("-" * 85)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=days_order, ordered=True)
    print(df.groupby("day_of_week")["is_win"].agg(trades="count", wins="sum", win_rate=lambda x: round(x.mean() * 100, 1)))

    print("\n7. ADX TREND STRENGTH SWEET SPOT:")
    print("-" * 85)
    df["adx_bucket"] = pd.cut(df["adx_val"], bins=[0, 20, 28, 40, 100], labels=["Chop (<20)", "Developing (20-28)", "Strong (28-40)", "Parabolic (>40)"])
    print(df.groupby("adx_bucket")["is_win"].agg(trades="count", wins="sum", win_rate=lambda x: round(x.mean() * 100, 1)))

if __name__ == "__main__":
    main()
