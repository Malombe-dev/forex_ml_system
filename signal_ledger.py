#!/usr/bin/env python3
"""
TrendM30 Signal Ledger
Records EVERY qualifying signal and follows it through its whole life:

  WAITING   -> setup is valid (>= threshold) but price hasn't pulled back to EMA21 yet
  OPEN      -> entry triggered (price at the retest)
  TP1_HIT   -> TP1 reached, stop moved to break-even, runner still open
  TP2_HIT   -> runner reached TP2                       (closed, win)
  BE_STOP   -> TP1 banked, runner stopped at entry      (closed, small win)
  SL_HIT    -> stop loss hit before TP1                 (closed, loss)
  EXPIRED   -> never triggered (no retest / setup lost) or timed out

Every event is stamped with the time of the M30 bar that caused it, so times stay
correct even when the scanner was down and is catching up (backfill).
Each scan walks every closed bar since the last processed bar, so no signal is skipped.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LEDGER_FILE = Path("signals/trendm30_signal_ledger.json")
BAR_MIN = 30
WAIT_MAX_BARS = 8          # same retest window as the backtests
OPEN_MAX_BARS = 300        # same max trade length as the backtests
FIRST_RUN_LOOKBACK = 96    # bars to backfill the first time a symbol is seen (2 days)
MAX_KEPT = 2000
ACTIVE = {"WAITING", "OPEN", "TP1_HIT"}


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(t) -> str:
    return _utc(t).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_ledger() -> Dict:
    try:
        with open(LEDGER_FILE) as f:
            data = json.load(f)
        data.setdefault("signals", [])
        data.setdefault("cursor", {})
        return data
    except Exception:
        return {"signals": [], "cursor": {}}


def save_ledger(ledger: Dict) -> None:
    try:
        ledger["signals"] = ledger["signals"][-MAX_KEPT:]
        LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEDGER_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(ledger, f, indent=2)
        tmp.replace(LEDGER_FILE)
    except Exception as e:
        print(f"⚠️ Could not save signal ledger: {e}")


def _levels(s: dict, res: dict) -> None:
    s["entry"], s["sl"], s["tp1"], s["tp2"] = res["entry"], res["sl"], res["tp1"], res["tp2"]
    s["risk_pips"] = res["risk_pips"]
    s["ml_probability"], s["tier"] = res["ml_probability"], res["tier"]
    risk = abs(s["entry"] - s["sl"]) or 1e-10
    s["risk"] = risk
    s["rr1"] = round(abs(s["tp1"] - s["entry"]) / risk, 2)
    s["rr2"] = round(abs(s["tp2"] - s["entry"]) / risk, 2)


def _trigger(s: dict, res: dict, bar_open, bar_close) -> None:
    _levels(s, res)
    s["state"] = "OPEN"
    s["triggered_at"] = _iso(bar_close)
    s["trigger_bar"] = _iso(bar_open)
    s["bars_open"] = 0


def _close(s: dict, state: str, bar_close, reason: str, exit_price=None, r=0.0) -> None:
    s["state"] = state
    s["closed_at"] = _iso(bar_close)
    s["close_reason"] = reason
    s["exit_price"] = exit_price
    s["r_result"] = round(r, 2)


def _advance(s: dict, hi: float, lo: float, close: float, bar_close) -> None:
    """Move an OPEN / TP1_HIT trade forward by one closed bar. If SL and a target share a bar, SL wins."""
    s["bars_open"] += 1
    long = s["direction"] == "BUY"
    stop = s["sl"] if s["state"] == "OPEN" else s["entry"]  # break-even after TP1
    hit_stop = (lo <= stop) if long else (hi >= stop)

    if s["state"] == "OPEN":
        hit_tp1 = (hi >= s["tp1"]) if long else (lo <= s["tp1"])
        if hit_stop:
            note = "SL and TP1 both touched in one bar; counted as SL" if hit_tp1 else "Stop loss hit"
            _close(s, "SL_HIT", bar_close, note, s["sl"], -1.0)
        elif hit_tp1:
            s["state"] = "TP1_HIT"
            s["tp1_at"] = _iso(bar_close)
    else:
        hit_tp2 = (hi >= s["tp2"]) if long else (lo <= s["tp2"])
        if hit_tp2:
            _close(s, "TP2_HIT", bar_close, "Runner reached TP2", s["tp2"], 0.5 * s["rr1"] + 0.5 * s["rr2"])
        elif hit_stop:
            _close(s, "BE_STOP", bar_close, "Runner stopped at break-even after TP1", s["entry"], 0.5 * s["rr1"])

    if s["state"] in ("OPEN", "TP1_HIT") and s["bars_open"] >= OPEN_MAX_BARS:
        move = ((close - s["entry"]) if long else (s["entry"] - close)) / s["risk"]
        r = (0.5 * s["rr1"] + 0.5 * move) if s["state"] == "TP1_HIT" else move
        _close(s, "EXPIRED", bar_close, f"Timed out after {OPEN_MAX_BARS} bars", close, r)


def _process_bar(ledger: Dict, sym: str, res: dict, bar, backfilled: bool) -> List[str]:
    bar_open = _utc(bar["datetime"])
    bar_close = bar_open + timedelta(minutes=BAR_MIN)
    hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])
    events: List[str] = []

    has_dir = res.get("direction") in ("BUY", "SELL") and res.get("status") in ("QUALIFIED", "WAITING_RETEST")
    take = has_dir and res.get("decision") == "TAKE"

    for s in ledger["signals"]:
        if s["symbol"] != sym or s["state"] not in ACTIVE:
            continue
        before = s["state"]
        if s["state"] == "WAITING":
            if take and res["direction"] == s["direction"] and res["status"] == "QUALIFIED":
                _trigger(s, res, bar_open, bar_close)
            elif s["bars_waited"] >= WAIT_MAX_BARS:
                _close(s, "EXPIRED", bar_close, f"No retest within {WAIT_MAX_BARS} bars")
            elif not has_dir or res["direction"] != s["direction"]:
                _close(s, "EXPIRED", bar_close, "Setup invalidated (alignment or 4H trend lost)")
            else:
                s["bars_waited"] += 1
        elif bar_open > _utc(s["trigger_bar"]):
            _advance(s, hi, lo, cl, bar_close)
        if s["state"] != before:
            events.append(f"{sym} {s['direction']}: {before} -> {s['state']} @ {_iso(bar_close)}")

    if take and not any(s["symbol"] == sym and s["state"] in ACTIVE for s in ledger["signals"]):
        sid = f"{sym}|{res['direction']}|{_iso(bar_open)}"
        if not any(s["id"] == sid for s in ledger["signals"]):
            s = {
                "id": sid, "symbol": sym, "direction": res["direction"],
                "produced_at": _iso(bar_close), "bar_open": _iso(bar_open),
                "detected_at": _iso(datetime.now(timezone.utc)), "backfilled": backfilled,
                "state": "WAITING", "bars_waited": 0, "bars_open": 0,
                "triggered_at": None, "tp1_at": None, "closed_at": None,
                "close_reason": None, "exit_price": None, "r_result": None,
            }
            _levels(s, res)
            if res["status"] == "QUALIFIED":
                _trigger(s, res, bar_open, bar_close)
            ledger["signals"].append(s)
            events.append(f"{sym} {s['direction']}: new signal {s['ml_probability']}% -> {s['state']} @ {_iso(bar_close)}")
    return events


def update_ledger(flt, sym: str, df30, df4h, ledger: Dict, lookback: Optional[int] = FIRST_RUN_LOOKBACK) -> List[str]:
    """Walk every closed M30 bar since the last one processed for this symbol.
    lookback = bars to replay when the symbol has no cursor yet; None replays everything that has enough history."""
    df = flt.prepare(df30, df4h)
    n = len(df)
    if n < 65:
        return []
    now = pd.Timestamp(datetime.now(timezone.utc))
    times = pd.to_datetime(df["datetime"], utc=True)
    closed = (times + pd.Timedelta(minutes=BAR_MIN)) <= now
    if not closed.any():
        return []
    last = int(closed.values.nonzero()[0][-1])
    if last < 60:
        return []

    cur = ledger["cursor"].get(sym)
    if cur:
        mask = (times > _utc(cur)).values
        start = int(mask.argmax()) if mask.any() else last + 1
    else:
        start = 60 if lookback is None else last - lookback + 1
    start = max(start, 60)

    events: List[str] = []
    for idx in range(start, last + 1):
        res = flt.evaluate_live_signal(sym, df30, df4h, bar_idx=idx, prepared=df)
        events += _process_bar(ledger, sym, res, df.iloc[idx], backfilled=(idx < last))
    ledger["cursor"][sym] = _iso(times.iloc[last])
    return events


def ledger_payload(ledger: Dict, limit: int = 300) -> Dict:
    sigs = sorted(ledger["signals"], key=lambda s: s["produced_at"], reverse=True)
    # "traded" = entered the market and finished (including timed-out trades); never-triggered expiries are counted apart
    closed = [s for s in sigs if s["state"] not in ACTIVE and s.get("triggered_at")]
    wins = [s for s in closed if (s.get("r_result") or 0) > 0]
    losses = [s for s in closed if (s.get("r_result") or 0) < 0]
    never = [s for s in sigs if s["state"] == "EXPIRED" and not s.get("triggered_at")]
    stats = {
        "total": len(sigs),
        "active": sum(1 for s in sigs if s["state"] in ACTIVE),
        "waiting": sum(1 for s in sigs if s["state"] == "WAITING"),
        "traded": len(closed), "wins": len(wins), "losses": len(losses),
        "never_triggered": len(never),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "net_r": round(sum(s.get("r_result") or 0 for s in closed), 2),
    }
    return {"signal_ledger": sigs[:limit], "ledger_stats": stats}
