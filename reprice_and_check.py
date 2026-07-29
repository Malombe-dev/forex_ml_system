"""
reprice_and_check.py — Take a fixed list of past trades from all_signals.json,
recompute their TP to a flat 1:2.5 RR (SL untouched), and walk real historical
15-min price data to see how many would have hit the new TP before the
original SL.

Run from the same directory as production_predictor.py / backtest_replay.py:

    python3 reprice_and_check.py

Trigger time is approximated as timestamp + bars_to_entry * 4h (4H timeframe),
matching how the live system itself defines bars_to_entry. This is the same
approximation backtest_replay.py's trigger simulation makes.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_replay import fetch_historical_bars, PAIR_API_KEYS  # noqa: E402

RR_MULTIPLIER = 2  # <-- change this one number to whatever ratio you want (e.g. 2 for 1:2)

TRADES = [
    dict(symbol="XAU/USD", direction="SELL", entry=4120.82252, sl=4180.95224,
         timestamp="2026-07-09T09:27:55.032394", bars_to_entry=2),
    dict(symbol="EUR/USD", direction="SELL", entry=1.14478, sl=1.14669,
         timestamp="2026-07-14T13:00:53.745108", bars_to_entry=1),
    dict(symbol="XAU/USD", direction="SELL", entry=4044.81657, sl=4080.83602,
         timestamp="2026-07-15T05:22:24.027887", bars_to_entry=3),
    dict(symbol="USD/JPY", direction="SELL", entry=162.35836, sl=162.54929,
         timestamp="2026-07-16T05:59:39.231133", bars_to_entry=4),
    dict(symbol="GBP/USD", direction="BUY", entry=1.34769, sl=1.34306,
         timestamp="2026-07-16T11:51:29.020376", bars_to_entry=2),
    dict(symbol="USD/JPY", direction="SELL", entry=162.36748, sl=162.54007,
         timestamp="2026-07-20T11:34:35.503623", bars_to_entry=0),
]


def main():
    cache_dir = Path("backtest_results/bar_cache")
    results = []

    for t in TRADES:
        risk = abs(t["entry"] - t["sl"])
        new_tp = t["entry"] + RR_MULTIPLIER * risk if t["direction"] == "BUY" else t["entry"] - RR_MULTIPLIER * risk

        proposed_at = datetime.fromisoformat(t["timestamp"])
        trigger_time = proposed_at + timedelta(hours=4 * t["bars_to_entry"])

        fetch_start = trigger_time - timedelta(days=1)
        fetch_end   = trigger_time + timedelta(days=20)
        api_key = PAIR_API_KEYS.get(t["symbol"], list(PAIR_API_KEYS.values())[0])

        print(f"\n{t['symbol']} {t['direction']} entry={t['entry']} sl={t['sl']} new_tp={new_tp:.5f}")
        df15 = fetch_historical_bars(t["symbol"], "15min", fetch_start, fetch_end, api_key, cache_dir)
        if df15 is None or len(df15) == 0:
            print("  ⚠️  No historical data available — skipping.")
            results.append({**t, "new_tp": new_tp, "outcome": "no_data"})
            continue

        hold = df15[df15["time"] >= trigger_time]
        outcome = "timeout"
        exit_time = None
        for _, bar in hold.iterrows():
            if t["direction"] == "BUY":
                hit_sl = bar["low"] <= t["sl"]
                hit_tp = bar["high"] >= new_tp
            else:
                hit_sl = bar["high"] >= t["sl"]
                hit_tp = bar["low"] <= new_tp
            if hit_sl:  # conservative: SL wins same-bar ties
                outcome, exit_time = "hit_sl", bar["time"]
                break
            if hit_tp:
                outcome, exit_time = "hit_tp", bar["time"]
                break

        emoji = "✅" if outcome == "hit_tp" else ("❌" if outcome == "hit_sl" else "⏱️")
        print(f"  {emoji} {outcome}" + (f" at {exit_time}" if exit_time else ""))
        results.append({**t, "new_tp": new_tp, "outcome": outcome, "exit_time": str(exit_time)})

    print(f"\n{'='*60}\nSUMMARY (1:{RR_MULTIPLIER} RR, SL unchanged)\n{'='*60}")
    wins = sum(1 for r in results if r["outcome"] == "hit_tp")
    losses = sum(1 for r in results if r["outcome"] == "hit_sl")
    other = len(results) - wins - losses
    print(f"Would have hit new TP:  {wins}")
    print(f"Would have hit SL:      {losses}")
    print(f"Other (timeout/no data): {other}")
    if wins + losses > 0:
        print(f"Win rate at 1:{RR_MULTIPLIER}:      {wins / (wins + losses) * 100:.1f}%")


if __name__ == "__main__":
    main()