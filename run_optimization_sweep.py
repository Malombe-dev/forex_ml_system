#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path("data/twelvedata_live")
symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'XAU/USD']

# Import calculation functions from backtest_trendm30_twelvedata
from backtest_trendm30_twelvedata import TrendM30Backtester

dfs = {}
for s in symbols:
    safe = s.replace('/', '_')
    df30 = pd.read_parquet(CACHE_DIR / f'{safe}_30min_5000.parquet')
    df4h = pd.read_parquet(CACHE_DIR / f'{safe}_4h_1000.parquet')
    dfs[s] = (df30, df4h)

results = []

for min_score in [4, 5, 6]:
    for rr in [1.2, 1.5, 2.0, 2.5]:
        for sl_mode in ["swing_atr", "atr"]:
            for buf in ([0.25, 0.5, 1.0] if sl_mode == "swing_atr" else [1.5, 2.0]):
                tester = TrendM30Backtester(
                    min_score=min_score,
                    rr_target=rr,
                    sl_mode=sl_mode,
                    sl_mult=buf if sl_mode == "atr" else 1.5,
                    swing_buf=buf if sl_mode == "swing_atr" else 0.25
                )
                tot_trades = 0
                tot_wins = 0
                tot_losses = 0
                tot_r = 0.0

                for s in symbols:
                    res = tester.run_pair(s, dfs[s][0], dfs[s][1])
                    resolved = [t for t in res['trades'] if t['outcome'] in ('WIN', 'LOSS')]
                    w = sum(1 for t in resolved if t['outcome'] == 'WIN')
                    l = sum(1 for t in resolved if t['outcome'] == 'LOSS')
                    r = sum(t['r_mult'] for t in resolved)
                    tot_trades += len(resolved)
                    tot_wins += w
                    tot_losses += l
                    tot_r += r

                wr = (tot_wins / tot_trades * 100) if tot_trades else 0.0
                pf = (tot_wins * rr) / (tot_losses * 1.0) if tot_losses else 0.0
                results.append({
                    "min_score": min_score,
                    "rr": rr,
                    "sl_mode": sl_mode,
                    "buffer": buf,
                    "trades": tot_trades,
                    "wins": tot_wins,
                    "losses": tot_losses,
                    "win_rate": wr,
                    "net_r": tot_r,
                    "profit_factor": pf
                })

df_res = pd.DataFrame(results)
df_res = df_res.sort_values("net_r", ascending=False)
print("=" * 85)
print("  TOP 15 BEST PERFORMING CONFIGURATIONS (LIVE TWELVEDATA DATA)")
print("=" * 85)
print(df_res.head(15).to_string(index=False))

print("\n" + "=" * 85)
print("  WORST 5 PERFORMING CONFIGURATIONS")
print("=" * 85)
print(df_res.tail(5).to_string(index=False))
