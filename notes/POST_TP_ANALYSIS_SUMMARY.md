# Post-TP Analysis & Single-TP Optimization Report
**Date:** 2026-08-29  
**Status:** Saved for continuation  

---

## 1. Executive Summary & Core Discovery
An in-depth investigation into post-Take Profit (Post-TP) price action was performed across all backtest replay runs and 40 unique triggered trades.

### The Problem:
* **100% (24/24) of winning trades** in replay test history continued moving significantly beyond the Take Profit target by $>3$ pips.
* On average, trades ran an extra **+167.2% beyond target** (+94.1 pips), leaving **+$2.93R** on the table per winning trade.
* **Root Cause:** In `production_predictor.py:L1972` (`_clamp_tp_to_structure`), the TP is clamped to the *nearest* 1H/4H opposing zone or 30-bar swing point with an extra `0.15 * ATR` buffer, prematurely capping trades before the macro 4H/Daily impulse leg completes.

---

## 2. Currency-by-Currency Post-TP Extension Dynamics

| Currency Pair | Wins ($n$) | Avg Captured TP | Avg Post-TP Extension | Max Extension | Avg Ext % of TP | Captured $R$ | Extra $R$ Left | Total Potential $R$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **USD/JPY** | 4 | 56.1p | **176.4p** | **574.9p** | **+262.7%** | $1.68R$ | **$+3.99R$** | **$5.67R$** |
| **XAU/USD** | 8 | 50.1p | **59.5p** | **172.7p** | **+156.0%** | $1.68R$ | **$+2.70R$** | **$4.37R$** |
| **EUR/USD** | 6 | 56.2p | **62.2p** | **107.7p** | **+135.1%** | $2.08R$ | **$+2.38R$** | **$4.46R$** |
| **GBP/USD** | 6 | 67.3p | **78.4p** | **157.4p** | **+115.2%** | $2.27R$ | **$+2.65R$** | **$4.92R$** |
| **ALL PAIRS** | **24** | **57.4p** | **94.1p** | **574.9p** | **+167.2%** | **$1.93R$** | **$+2.93R$** | **$4.86R$** |

---

## 3. Backtest Simulation of Single-TP Methods (40 Historical Trades)

We evaluated 4 distinct methods for achieving an optimal single-TP target using 15-minute bar-by-bar price action:

| Method / Configuration | Win Rate | Wins / Losses | Total Net Profit ($R$) | Avg $R$ / Trade | Profit Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Baseline (Current System TP)** | **52.5%** | 21 / 14 | **$+25.84R$** | $+0.65R$ | Baseline |
| **🏆 Method 3B (1.5x Target Scaling)** | **40.0%** | 16 / 16 | **$+36.94R$** | **$+0.92R$** | **$+43.0\%$** |
| **🥈 Method 3A (1.4x Target Scaling)** | **42.5%** | 17 / 16 | **$+33.88R$** | **$+0.85R$** | **$+31.1\%$** |
| **Method 1 (HTF-Only Clamping, 1.6x)** | **32.5%** | 13 / 19 | **$+29.18R$** | $+0.73R$ | $+12.9\%$ |
| **Method 4A (Fixed 2.5R Target)** | **42.5%** | 17 / 17 | **$+27.21R$** | $+0.68R$ | $+5.3\%$ |
| **Method 2 (Major 4H Swing High/Low)** | **22.5%** | 9 / 19 | **$+12.13R$** | $+0.30R$ | $-53.1\%$ |

---

## 4. Verification on the Latest Replay Run (8 Trades: 7 Wins, 1 Loss)

On the exact 8 resolved trades from `replay_2026-08-28_145055_signals.jsonl`:

| Trade # | Symbol / Dir | Trigger Time | Risk | Base TP | Post-TP Ext. | Base ($R$) | 1.4x ($R$) | 1.5x ($R$) | Fixed 3.0R |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **EUR/USD** BUY | 2026-07-31 | 23.1p | 34.6p | $+47.5\text{p}$ | $+1.50R$ | $+2.10R$ | $+2.25R$ | $+3.00R$ |
| **2** | **EUR/USD** BUY | 2026-08-10 | 35.8p | 97.8p | $+68.2\text{p}$ | $+0.68R$ | $+0.68R$ | $+0.68R$ | $+0.68R$ |
| **3** | **GBP/USD** BUY | 2026-07-30 | 36.4p | 81.1p | $+78.6\text{p}$ | $+2.23R$ | $+3.12R$ | $+3.34R$ | $+3.00R$ |
| **4** | **GBP/USD** BUY | 2026-08-10 | 22.1p | 62.4p | $+22.3\text{p}$ | $+2.82R$ | $+3.95R$ | $+4.24R$ | $+3.00R$ |
| **5** | **GBP/USD** BUY | 2026-08-20 | 28.9p | 57.6p | $+29.5\text{p}$ | $+1.99R$ | $+2.79R$ | $+2.99R$ | $+3.00R$ |
| **6** | **USD/JPY** SELL | 2026-07-30 | 58.8p | 81.0p | $+574.9\text{p}$ | $+1.38R$ | $+1.93R$ | $+2.07R$ | $+3.00R$ |
| **7** | **USD/JPY** SELL | 2026-08-20 | 46.3p | 92.0p | — | $-1.00R$ | $-1.00R$ | $-1.00R$ | $-1.00R$ |
| **8** | **XAU/USD** SELL | 2026-07-30 | 35.5p | 53.2p | $+11.0\text{p}$ | $+1.50R$ | $+1.36R$ | $+1.36R$ | $+1.36R$ |
| **TOTAL** | — | — | — | — | — | **$+11.10R$** | **$+14.93R$** | **$+15.92R$** | **$+16.04R$** |
| **GAIN** | — | — | — | — | — | **Base** | **+34.5%** | **+43.4%** | **+44.5%** |

---

## 5. Critical Takeaways & Next Action Items for Tomorrow

1. **Top Recommendation:** Implement **1.4x–1.5x Target Scaling** in `production_predictor.py`:
   - Increase FVG target multiplier from `2.0x` $\rightarrow$ **`2.8x – 3.0x`**
   - Increase OB reaction multiplier from `1.5x` $\rightarrow$ **`2.1x – 2.25x`**
   - Retain HTF structural clamping for safety.
2. **Breakeven Rule Finding:** Do NOT move SL to Breakeven at $+1.0R$ (chokes 50%+ of winning trades during normal 15m retests). Keep SL anchored at the zone invalidation level.
3. **Next Steps Tomorrow:**
   - Update `production_predictor.py` target calculation constants.
   - Run full `backtest_replay.py` pipeline to generate updated charts, logs, and verification reports.
