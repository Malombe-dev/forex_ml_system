#!/usr/bin/env python3
"""
Deep Research and Optimization for TrendM30 Pro Pine Script Strategy.
Loads cached Twelve Data bars (EUR/USD, GBP/USD, USD/JPY, XAU/USD).
Systematically tests:
  1. Score thresholds & Individual Confluences (HTF, ADX, Momentum, Breakout, SMC, Room)
  2. Stop Loss Models (Swing, ATR, minimum ATR floors)
  3. Risk:Reward Ratios (1.0, 1.2, 1.5, 2.0, 2.5)
  4. Breakeven stops (locking in 0R once 1R reached)
  5. Session filtering (London & NY overlap 07:00-18:00 UTC vs all hours)
  6. Pullback entry vs Breakout market entry
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_trendm30_twelvedata import calc_atr, calc_rsi, calc_macd, calc_adx

CACHE_DIR = Path("data/twelvedata_live")
SYMBOLS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]

def load_data():
    data = {}
    for sym in SYMBOLS:
        safe = sym.replace("/", "_")
        df30 = pd.read_parquet(CACHE_DIR / f"{safe}_30min_5000.parquet").sort_values("datetime").reset_index(drop=True)
        df4h = pd.read_parquet(CACHE_DIR / f"{safe}_4h_1000.parquet").sort_values("datetime").reset_index(drop=True)
        data[sym] = (df30, df4h)
    return data

def precompute_features(df30: pd.DataFrame, df4h: pd.DataFrame, sym: str) -> pd.DataFrame:
    df = df30.copy()
    n = len(df)
    
    # Base MAs
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["white"] = df["close"].rolling(60).mean()
    df["yellow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema3"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()

    # MA60 Slope (3 bars)
    df["white_up"] = df["white"] > df["white"].shift(3)
    df["white_down"] = df["white"] < df["white"].shift(3)

    # MA Alignment
    df["bull_align"] = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["yellow"]) & (df["yellow"] > df["white"])
    df["bear_align"] = (df["ma5"] < df["ma10"]) & (df["ma10"] < df["yellow"]) & (df["yellow"] < df["white"])

    # HTF 4H EMA50 (No lookahead)
    df4h_calc = df4h.copy().sort_values("datetime").reset_index(drop=True)
    df4h_calc["ema50"] = df4h_calc["close"].ewm(span=50, adjust=False).mean()
    df4h_closed = df4h_calc.copy()
    df4h_closed["datetime"] = df4h_closed["datetime"] + pd.Timedelta(hours=4)
    merged = pd.merge_asof(
        df.sort_values("datetime"),
        df4h_closed[["datetime", "close", "ema50"]],
        on="datetime",
        suffixes=("", "_htf")
    )
    df["htf_close"] = merged["close_htf"]
    df["htf_ema"] = merged["ema50"]
    
    # Check 1: HTF Trend
    df["c1_bull"] = (df["htf_close"] > df["htf_ema"]).fillna(False)
    df["c1_bear"] = (df["htf_close"] < df["htf_ema"]).fillna(False)

    # Check 2: ADX > 22
    df["adx"] = calc_adx(df, 14)
    df["c2"] = (df["adx"] > 22.0).fillna(False)

    # Check 3: Breakout 20-bar High/Low within 6 bars
    df["hi_brk"] = df["high"].rolling(20).max().shift(1)
    df["lo_brk"] = df["low"].rolling(20).min().shift(1)
    df["brk_up"] = df["close"] > df["hi_brk"]
    df["brk_dn"] = df["close"] < df["lo_brk"]
    df["c3_bull"] = df["brk_up"].rolling(7, min_periods=1).max().astype(bool)
    df["c3_bear"] = df["brk_dn"].rolling(7, min_periods=1).max().astype(bool)

    # Check 4: Momentum MACD Hist & RSI 50
    _, _, df["macd_hist"] = calc_macd(df["close"], 12, 26, 9)
    df["rsi"] = calc_rsi(df["close"], 14)
    df["c4_bull"] = ((df["macd_hist"] > 0) & (df["rsi"] > 50)).fillna(False)
    df["c4_bear"] = ((df["macd_hist"] < 0) & (df["rsi"] < 50)).fillna(False)

    # Check 5: Volatility
    df["atr14"] = calc_atr(df, 14)
    df["atr_sma50"] = df["atr14"].rolling(50).mean()
    df["c5"] = ((df["atr14"] / (df["atr_sma50"] + 1e-10)) >= 1.0).fillna(False)

    # Check 6: Room to EMA21 (< 2.5 * ATR)
    df["dist_yellow"] = (df["close"] - df["yellow"]).abs()
    df["c6"] = (df["dist_yellow"] < (df["atr14"] * 2.5)).fillna(False)

    # Check 7: SMC (Premium/Discount + Sweeps + S/D)
    df["rng_hi"] = df["high"].rolling(100).max()
    df["rng_lo"] = df["low"].rolling(100).min()
    df["eq_p"] = (df["rng_hi"] + df["rng_lo"]) / 2.0
    in_disc = df["close"] < df["eq_p"]
    in_prem = df["close"] > df["eq_p"]

    highs, lows, closes, atrs, opens = df["high"].values, df["low"].values, df["close"].values, df["atr14"].values, df["open"].values
    sw_hi_recent = np.zeros(n, dtype=bool)
    sw_lo_recent = np.zeros(n, dtype=bool)
    in_dem = np.zeros(n, dtype=bool)
    in_sup = np.zeros(n, dtype=bool)
    last_ph = np.nan
    last_pl = np.nan
    last_sweep_hi = -9999
    last_sweep_lo = -9999
    dem_boxes, sup_boxes = [], []

    for i in range(10, n):
        piv_idx = i - 5
        if piv_idx >= 5:
            if highs[piv_idx] == np.max(highs[piv_idx-5 : piv_idx+6]):
                last_ph = highs[piv_idx]
            if lows[piv_idx] == np.min(lows[piv_idx-5 : piv_idx+6]):
                last_pl = lows[piv_idx]

        if not np.isnan(last_ph):
            if highs[i] > last_ph and closes[i] < last_ph:
                last_sweep_hi = i
            if highs[i] > last_ph:
                last_ph = np.nan

        if not np.isnan(last_pl):
            if lows[i] < last_pl and closes[i] > last_pl:
                last_sweep_lo = i
            if lows[i] < last_pl:
                last_pl = np.nan

        if (i - last_sweep_hi) <= 10:
            sw_hi_recent[i] = True
        if (i - last_sweep_lo) <= 10:
            sw_lo_recent[i] = True

        c, o, a = closes[i], opens[i], atrs[i]
        imp_up = (c > o) and ((c - o) > (a * 1.5))
        imp_dn = (c < o) and ((o - c) > (a * 1.5))
        if imp_up:
            base_dn = 1 if closes[i-1] < opens[i-1] else (2 if closes[i-2] < opens[i-2] else 0)
            if base_dn > 0:
                dem_boxes.append((highs[i - base_dn], lows[i - base_dn]))
                if len(dem_boxes) > 4:
                    dem_boxes.pop(0)
        if imp_dn:
            base_up = 1 if closes[i-1] > opens[i-1] else (2 if closes[i-2] > opens[i-2] else 0)
            if base_up > 0:
                sup_boxes.append((highs[i - base_up], lows[i - base_up]))
                if len(sup_boxes) > 4:
                    sup_boxes.pop(0)

        new_dem = []
        for top, bot in dem_boxes:
            if c < bot:
                continue
            new_dem.append((top, bot))
            if lows[i] <= top and highs[i] >= bot:
                in_dem[i] = True
        dem_boxes = new_dem

        new_sup = []
        for top, bot in sup_boxes:
            if c > top:
                continue
            new_sup.append((top, bot))
            if highs[i] >= bot and lows[i] <= top:
                in_sup[i] = True
        sup_boxes = new_sup

    df["c7_bull"] = (in_disc & (sw_lo_recent | in_dem)).fillna(False)
    df["c7_bear"] = (in_prem & (sw_hi_recent | in_sup)).fillna(False)

    # Score calculation
    df["score_bull"] = (df["c1_bull"].astype(int) + df["c2"].astype(int) + df["c3_bull"].astype(int) +
                        df["c4_bull"].astype(int) + df["c5"].astype(int) + df["c6"].astype(int) + df["c7_bull"].astype(int))
    df["score_bear"] = (df["c1_bear"].astype(int) + df["c2"].astype(int) + df["c3_bear"].astype(int) +
                        df["c4_bear"].astype(int) + df["c5"].astype(int) + df["c6"].astype(int) + df["c7_bear"].astype(int))

    # Swings
    df["sw10_lo"] = df["low"].rolling(10).min()
    df["sw10_hi"] = df["high"].rolling(10).max()
    df["sw20_lo"] = df["low"].rolling(20).min()
    df["sw20_hi"] = df["high"].rolling(20).max()

    # Session Hour (UTC)
    df["hour"] = df["datetime"].dt.hour
    df["is_london_ny"] = (df["hour"] >= 7) & (df["hour"] <= 18)

    return df

# ── Simulation Runner ────────────────────────────────────────────────────────
def simulate_signals(df: pd.DataFrame,
                     symbol: str,
                     min_score: int = 4,
                     require_htf: bool = False,
                     require_c4: bool = False,
                     session_only: bool = False,
                     sl_mode: str = "swing", # "swing", "atr", "ema21"
                     sl_mult: float = 1.5,
                     swing_buf: float = 0.25,
                     min_sl_atr: float = 0.0,
                     rr_target: float = 2.0,
                     use_be: bool = False,
                     pullback_entry: bool = False):
    mintick = 0.01 if ("JPY" in symbol or "XAU" in symbol) else 0.0001
    n = len(df)
    
    # Conditions
    cond_bull = df["bull_align"] & df["white_up"] & (df["score_bull"] >= min_score)
    cond_bear = df["bear_align"] & df["white_down"] & (df["score_bear"] >= min_score)

    if require_htf:
        cond_bull = cond_bull & df["c1_bull"]
        cond_bear = cond_bear & df["c1_bear"]

    if require_c4:
        cond_bull = cond_bull & df["c4_bull"]
        cond_bear = cond_bear & df["c4_bear"]

    if session_only:
        cond_bull = cond_bull & df["is_london_ny"]
        cond_bear = cond_bear & df["is_london_ny"]

    sig_bull = cond_bull & (~cond_bull.shift(1).fillna(False))
    sig_bear = cond_bear & (~cond_bear.shift(1).fillna(False))

    sig_bull_arr = sig_bull.values
    sig_bear_arr = sig_bear.values

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    atrs = df["atr14"].values
    yellows = df["yellow"].values
    sw10_los = df["sw10_lo"].values
    sw10_his = df["sw10_hi"].values
    dts = df["datetime"].values

    trades = []
    i = 100
    while i < n - 1:
        is_long = sig_bull_arr[i]
        is_short = sig_bear_arr[i]

        if not is_long and not is_short:
            i += 1
            continue

        direction = "LONG" if is_long else "SHORT"
        trigger_entry = closes[i]
        atr_cur = atrs[i]

        entry_bar = i
        entry_price = trigger_entry

        if pullback_entry:
            # Wait up to 5 bars for a pullback towards EMA21 / within 0.5 ATR of EMA21
            entered = False
            for pb_bar in range(i + 1, min(i + 6, n)):
                # Check if invalidated first (e.g. alignment broken or cross white)
                if is_long:
                    # long pullback: low touches yellow or close approaches yellow
                    if lows[pb_bar] <= yellows[pb_bar] + 0.5 * atrs[pb_bar]:
                        entry_bar = pb_bar
                        entry_price = max(lows[pb_bar], yellows[pb_bar]) # filled near EMA21
                        atr_cur = atrs[pb_bar]
                        entered = True
                        break
                else:
                    if highs[pb_bar] >= yellows[pb_bar] - 0.5 * atrs[pb_bar]:
                        entry_bar = pb_bar
                        entry_price = min(highs[pb_bar], yellows[pb_bar])
                        atr_cur = atrs[pb_bar]
                        entered = True
                        break
            if not entered:
                i += 1
                continue

        # Stop loss calculation
        if sl_mode == "swing":
            sl_raw = (sw10_los[entry_bar] - atr_cur * swing_buf) if is_long else (sw10_his[entry_bar] + atr_cur * swing_buf)
        elif sl_mode == "ema21":
            sl_raw = (yellows[entry_bar] - atr_cur * swing_buf) if is_long else (yellows[entry_bar] + atr_cur * swing_buf)
        else: # "atr"
            sl_raw = (entry_price - atr_cur * sl_mult) if is_long else (entry_price + atr_cur * sl_mult)

        # Minimum ATR distance floor
        if min_sl_atr > 0.0:
            if is_long:
                sl_raw = min(sl_raw, entry_price - min_sl_atr * atr_cur)
            else:
                sl_raw = max(sl_raw, entry_price + min_sl_atr * atr_cur)

        # Safety clamp: min 5 ticks
        min_risk = 5.0 * mintick
        sl = min(sl_raw, entry_price - min_risk) if is_long else max(sl_raw, entry_price + min_risk)
        risk = abs(entry_price - sl)
        tp = (entry_price + risk * rr_target) if is_long else (entry_price - risk * rr_target)

        # Trade tracking walk forward
        outcome = "OPEN"
        exit_bar = entry_bar
        exit_price = entry_price
        r_mult = 0.0
        be_active = False
        cur_sl = sl

        for j in range(entry_bar + 1, min(entry_bar + 300, n)):
            bar_h = highs[j]
            bar_l = lows[j]

            # Breakeven check
            if use_be and not be_active:
                be_thresh = (entry_price + risk * 1.0) if is_long else (entry_price - risk * 1.0)
                if (is_long and bar_h >= be_thresh) or (not is_long and bar_l <= be_thresh):
                    cur_sl = entry_price # SL moved to BE!
                    be_active = True

            hit_tp = (bar_h >= tp) if is_long else (bar_l <= tp)
            hit_sl = (bar_l <= cur_sl) if is_long else (bar_h >= cur_sl)

            if hit_tp and hit_sl:
                outcome = "LOSS" if not be_active else "BE"
                exit_bar = j
                exit_price = cur_sl
                r_mult = -1.0 if not be_active else 0.0
                break
            elif hit_tp:
                outcome = "WIN"
                exit_bar = j
                exit_price = tp
                r_mult = rr_target
                break
            elif hit_sl:
                outcome = "LOSS" if not be_active else "BE"
                exit_bar = j
                exit_price = cur_sl
                r_mult = -1.0 if not be_active else 0.0
                break

        trades.append({
            "symbol": symbol,
            "entry_time": dts[entry_bar],
            "direction": direction,
            "entry": entry_price,
            "sl": cur_sl,
            "tp": tp,
            "risk": risk,
            "outcome": outcome,
            "r_mult": r_mult,
            "bars_held": exit_bar - entry_bar
        })

        i = exit_bar + 1

    return trades

def eval_config(data_dict, **kwargs):
    all_trades = []
    for sym in SYMBOLS:
        tr = simulate_signals(data_dict[sym], sym, **kwargs)
        all_trades.extend(tr)
    
    resolved = [t for t in all_trades if t["outcome"] in ("WIN", "LOSS", "BE")]
    if not resolved:
        return {"trades": 0, "wins": 0, "losses": 0, "be": 0, "win_rate": 0.0, "net_r": 0.0, "profit_factor": 0.0}

    wins = sum(1 for t in resolved if t["outcome"] == "WIN")
    losses = sum(1 for t in resolved if t["outcome"] == "LOSS")
    bes = sum(1 for t in resolved if t["outcome"] == "BE")
    tot = len(resolved)
    wr = (wins / tot * 100)
    net_r = sum(t["r_mult"] for t in resolved)
    gross_win = sum(t["r_mult"] for t in resolved if t["r_mult"] > 0)
    gross_loss = abs(sum(t["r_mult"] for t in resolved if t["r_mult"] < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

    return {
        "trades": tot,
        "wins": wins,
        "losses": losses,
        "be": bes,
        "win_rate": wr,
        "net_r": round(net_r, 2),
        "profit_factor": round(pf, 2)
    }

def main():
    print("=" * 80)
    print("  TRENDM30 PRO SYSTEM DEEP OPTIMIZATION & WIN-RATE RESEARCH")
    print("=" * 80)
    data = load_data()
    DATA = {}
    for s in SYMBOLS:
        df30, df4h = data[s]
        DATA[s] = precompute_features(df30, df4h, s)

    # 1. Baseline Benchmark
    base = eval_config(DATA, min_score=4, rr_target=2.0, sl_mode="swing", swing_buf=0.25, require_htf=False)
    print(f"\n[BASELINE Pine Script Default]:")
    print(f"  Trades: {base['trades']} | WR: {base['win_rate']:.2f}% | Net R: {base['net_r']:+.2f} R | Profit Factor: {base['profit_factor']:.2f}")

    # 2. Test Confluence Thresholds (min_score 4, 5, 6) with HTF filter
    print("\n" + "-" * 80)
    print("  STEP 1: Confluence Score & Mandatory HTF Trend Check (4H EMA50)")
    print("-" * 80)
    for ms in [4, 5, 6]:
        for req_htf in [False, True]:
            res = eval_config(DATA, min_score=ms, require_htf=req_htf, rr_target=2.0, sl_mode="swing", swing_buf=0.25)
            print(f"  Score >= {ms} | Require HTF={str(req_htf):<5} -> Trades: {res['trades']:<3} | WR: {res['win_rate']:>5.1f}% | Net R: {res['net_r']:>+6.1f} R | PF: {res['profit_factor']:.2f}")

    # 3. Stop Loss Methods & Buffers (Preventing premature wicks)
    print("\n" + "-" * 80)
    print("  STEP 2: Stop Loss Placement & Noise Buffers")
    print("-" * 80)
    for sl_mode in ["swing", "atr"]:
        for buf in ([0.25, 0.5, 1.0, 1.5] if sl_mode == "swing" else [1.0, 1.5, 2.0, 2.5]):
            for min_sl in [0.0, 1.0]:
                res = eval_config(DATA, min_score=5, require_htf=True, rr_target=1.5,
                                  sl_mode=sl_mode, sl_mult=buf if sl_mode=="atr" else 1.5,
                                  swing_buf=buf if sl_mode=="swing" else 0.25,
                                  min_sl_atr=min_sl)
                print(f"  Mode={sl_mode:<5} | Buf={buf:>4.2f} | MinSL={min_sl:>3.1f} ATR -> Trades: {res['trades']:<3} | WR: {res['win_rate']:>5.1f}% | Net R: {res['net_r']:>+6.1f} R | PF: {res['profit_factor']:.2f}")

    # 4. Target R:R (1.0, 1.2, 1.5, 2.0) + Breakeven
    print("\n" + "-" * 80)
    print("  STEP 3: Risk:Reward Targets & Breakeven Protection")
    print("-" * 80)
    for rr in [1.0, 1.2, 1.5, 1.8, 2.0]:
        for be in [False, True]:
            res = eval_config(DATA, min_score=5, require_htf=True, rr_target=rr, use_be=be,
                              sl_mode="atr", sl_mult=1.5)
            print(f"  Target RR={rr:>3.1f}R | BE Active={str(be):<5} -> Trades: {res['trades']:<3} | WR: {res['win_rate']:>5.1f}% (Wins={res['wins']}, BE={res['be']}) | Net R: {res['net_r']:>+6.1f} R | PF: {res['profit_factor']:.2f}")

    # 5. Session Filter (London / NY overlap)
    print("\n" + "-" * 80)
    print("  STEP 4: Trading Session Filter (07:00 - 18:00 UTC London & NY)")
    print("-" * 80)
    for sess in [False, True]:
        res = eval_config(DATA, min_score=5, require_htf=True, session_only=sess, rr_target=1.5,
                          sl_mode="atr", sl_mult=1.5, use_be=True)
        print(f"  Session 07-18 UTC Only={str(sess):<5} -> Trades: {res['trades']:<3} | WR: {res['win_rate']:>5.1f}% | Net R: {res['net_r']:>+6.1f} R | PF: {res['profit_factor']:.2f}")

    # 6. Pullback to EMA21 Entry vs Market Breakout
    print("\n" + "-" * 80)
    print("  STEP 5: Pullback Entry (Buying EMA21 Value vs Chasing Breakout)")
    print("-" * 80)
    for pb in [False, True]:
        res = eval_config(DATA, min_score=5, require_htf=True, pullback_entry=pb, rr_target=1.5,
                          sl_mode="atr", sl_mult=1.5, use_be=True)
        print(f"  Pullback Entry={str(pb):<5} -> Trades: {res['trades']:<3} | WR: {res['win_rate']:>5.1f}% | Net R: {res['net_r']:>+6.1f} R | PF: {res['profit_factor']:.2f}")

    # 7. Comprehensive Grid Search across Top Variables to find the Optimal High-Winrate Configuration
    print("\n" + "=" * 80)
    print("  TOP OPTIMIZED CONFIGURATIONS RANKED BY WIN RATE & PROFIT FACTOR")
    print("=" * 80)
    records = []
    for ms in [5, 6]:
        for req_htf in [True, False]:
            for sl_mode in ["atr", "swing"]:
                bufs = [1.5, 2.0] if sl_mode == "atr" else [0.5, 1.0]
                for buf in bufs:
                    for rr in [1.0, 1.2, 1.5, 2.0]:
                        for be in [True, False]:
                            for pb in [True, False]:
                                r = eval_config(DATA, min_score=ms, require_htf=req_htf,
                                                sl_mode=sl_mode, sl_mult=buf if sl_mode=="atr" else 1.5,
                                                swing_buf=buf if sl_mode=="swing" else 0.25,
                                                rr_target=rr, use_be=be, pullback_entry=pb, session_only=False)
                                if r["trades"] >= 20: # statistically meaningful sample
                                    records.append({
                                        "min_score": ms,
                                        "require_htf": req_htf,
                                        "sl_mode": sl_mode,
                                        "buf": buf,
                                        "rr": rr,
                                        "use_be": be,
                                        "pullback": pb,
                                        **r
                                    })
    df_all = pd.DataFrame(records)
    out_all_csv = Path("backtest_results/optimization_grid_results.csv")
    df_all.to_csv(out_all_csv, index=False)
    print(f"Saved {len(df_all)} tested configs to {out_all_csv}")
    df_sorted = df_all.sort_values(by=["win_rate", "profit_factor"], ascending=[False, False])
    print(df_sorted.head(15).to_string(index=False))

    print("\nTop 5 Ranked by Net R:")
    df_r = df_all.sort_values(by=["net_r", "win_rate"], ascending=[False, False])
    print(df_r.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
