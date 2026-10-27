#!/usr/bin/env python3
"""
Dashboard API Server — V8
=========================
Adds /api/funnel endpoint so the Funnel tab in index.html gets live data.

Usage:
    cd ~/Desktop/forex_ml_system/forex_ml_system
    source venv/bin/activate
    python dash/server.py
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR    = Path(__file__).resolve().parent
SIGNALS_DIR = Path(os.getenv("SIGNALS_DIR", str(Path(__file__).resolve().parent.parent / "signals")))


# ── helpers ───────────────────────────────────────────────────────────────────
def _read(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _mtime(path: Path):
    if not path.exists():
        return None
    return datetime.utcfromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S UTC")


def _sig(filename: str) -> Path:
    return SIGNALS_DIR / filename


def _normalize_signal(s: dict) -> dict:
    s = dict(s)
    is_htf = s.get("strategy", "").upper().startswith("HTF") or s.get("timeframe", "") in ("4h", "1d", "1h")

    # ── htf_trend ─────────────────────────────────────────────────────────────
    # HTF signals store it as "htf_trend"; 15min signals store it as "h4_trend".
    # Expose both as "h4_trend" so the frontend only needs one key.
    if "h4_trend" not in s or not s["h4_trend"]:
        s["h4_trend"] = s.get("htf_trend") or "—"
    if "htf_trend" not in s or not s["htf_trend"]:
        s["htf_trend"] = s.get("h4_trend") or "—"

    # ── confluence ────────────────────────────────────────────────────────────
    # HTF signals: "confluence" is a string like "OB-8.03 1H=✅ 5m=✅"
    # 15min signals: "confluence_score" is an integer 0–10
    # Normalise to "confluence_display" (human string) and "confluence_score" (int).
    raw_conf = s.get("confluence")
    score    = s.get("confluence_score")

    if raw_conf and isinstance(raw_conf, str):
        s["confluence_display"] = raw_conf
        if score is None:
            import re
            m = re.search(r"[-](\d+(?:\.\d+)?)", raw_conf)
            s["confluence_score"] = round(float(m.group(1))) if m else None
    elif score is not None:
        s["confluence_display"] = f"{score}/10"
    else:
        s["confluence_display"] = "—"
        s["confluence_score"]   = None

    # ── strategy display ──────────────────────────────────────────────────────
    # Normalise "HTF_SMC" → "HTF" for the frontend strategy column
    strat = s.get("strategy", "")
    if strat.upper() == "HTF_SMC":
        s["strategy"] = "HTF"

    return s


def _merge_signals():
    live = _read(_sig("all_signals.json")) or []
    plan = _read(_sig("daily_plan.json"))  or []

    live_keys = {
        (s.get("direction"), round(float(s.get("entry", 0)), 5))
        for s in live
    }

    merged = [_normalize_signal(s) for s in live]
    for p in plan:
        key = (p.get("direction"), round(float(p.get("entry", 0)), 5))
        if key not in live_keys:
            p = dict(p)
            p.setdefault("status", "planned")
            p["source"] = "daily_plan"
            merged.append(_normalize_signal(p))

    return merged


def _observation_signals():
    return _read(_sig("observation_signals.json")) or []


# ── static ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/trendm30")
def trendm30_view():
    return send_from_directory(BASE_DIR, "trendm30.html")


@app.route("/api/trendm30")
def api_trendm30():
    data = _read(_sig("trendm30_signals.json"))
    if data is None:
        return jsonify({"status": "waiting_first_scan", "results": [], "updated_at": None})
    return jsonify(data)


@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(BASE_DIR / "static", fname)


# ── /api/daily_plan ───────────────────────────────────────────────────────────
@app.route("/api/daily_plan")
def api_daily_plan():
    data = _read(_sig("daily_plan.json"))
    if data is None:
        return jsonify({"error": "daily_plan.json not found", "data": []}), 404
    return jsonify({"data": data, "count": len(data), "updated": _mtime(_sig("daily_plan.json"))})


# ── /api/weekly_plan ──────────────────────────────────────────────────────────
@app.route("/api/weekly_plan")
def api_weekly_plan():
    data = _read(_sig("weekly_plan.json"))
    if data is None:
        return jsonify({"data": None, "updated": None}), 200
    return jsonify({"data": data, "updated": _mtime(_sig("weekly_plan.json"))})


# ── /api/observation_signals ──────────────────────────────────────────────────
@app.route("/api/observation_signals")
def api_observation_signals():
    data = _observation_signals()
    return jsonify({
        "data":    data,
        "count":   len(data),
        "updated": _mtime(_sig("observation_signals.json")),
    })


# ── /api/zones ────────────────────────────────────────────────────────────────
@app.route("/api/zones")
def api_zones():
    result = {}
    for path in sorted(SIGNALS_DIR.glob("zones_*.json")):
        data = _read(path)
        if data is not None:
            result[path.stem] = {
                "data":    data,
                "count":   len(data),
                "updated": _mtime(path),
            }
    return jsonify(result)


@app.route("/api/zones/<strategy>/<symbol>")
def api_zones_specific(strategy, symbol):
    filename = f"zones_{symbol}_{strategy}.json"
    path     = _sig(filename)
    data     = _read(path)
    if data is None:
        return jsonify({"error": f"{filename} not found", "data": {}}), 404
    return jsonify({"data": data, "count": len(data), "updated": _mtime(path)})


# ── /api/zones_state ──────────────────────────────────────────────────────────
@app.route("/api/zones_state")
def api_zones_state():
    data = _read(_sig("zones_state.json"))
    if data is None:
        return jsonify({"error": "zones_state.json not found", "data": {}}), 404
    return jsonify({"data": data, "updated": _mtime(_sig("zones_state.json"))})


# ── /api/funnel ───────────────────────────────────────────────────────────────
@app.route("/api/funnel")
def api_funnel():
    """
    Reads funnel_state.json written by production_predictor.py each cycle.

    Expected shape of funnel_state.json:
    {
      "cycle": 1,
      "cycle_time": "2026-05-25 07:07:22 UTC",
      "next_cycle":  "2026-05-25 07:22:22 UTC",
      "pairs": {
        "EUR/USD": {
          "h1_trend": "bullish",
          "price":    "1.16414",
          "atr":      "0.00054",
          "fvg_4h": 2,  "ob_4h": 8,
          "fvg_1h": 0,  "ob_1h": 8,
          "zones_15m":  6,
          "htf_before": 0,
          "htf_after":  0,
          "ob_skips":   1,
          "reject_reasons": { "dist":1, "dir":1, "prob":1, "rr":3, "conf":1 }
        },
        ...
      }
    }

    If the file doesn't exist yet, returns an empty structure so the UI
    degrades gracefully without a 404.
    """
    data = _read(_sig("funnel_state.json"))
    if data is None:
        return jsonify({
            "cycle":      None,
            "cycle_time": None,
            "next_cycle": None,
            "pairs":      {},
            "updated":    None,
            "note": "funnel_state.json not found — add funnel serialisation to production_predictor.py"
        })
    return jsonify({
        **data,
        "updated": _mtime(_sig("funnel_state.json"))
    })


# ── /api/stats ────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    signals = _merge_signals()
    daily   = _read(_sig("daily_plan.json")) or []

    all_zones: dict = {}
    for p in SIGNALS_DIR.glob("zones_*.json"):
        d = _read(p)
        if d:
            all_zones.update(d)

    triggered    = [s for s in signals if s.get("status") in ("hit_tp", "hit_sl")]
    wins         = [s for s in triggered if s.get("status") == "hit_tp"]
    active       = [s for s in signals if s.get("status") in ("active", "triggered")]
    planned      = [s for s in signals if s.get("status") == "planned"]
    entry_missed = [s for s in signals if s.get("status") == "entry_missed"]

    probs   = [s["probability"] for s in signals if "probability" in s]
    rr_vals = []
    for s in signals:
        rr = s.get("rr", "")
        if isinstance(rr, str) and ":" in rr:
            try:
                rr_vals.append(float(rr.split(":")[1]))
            except ValueError:
                pass

    strats: dict = {}
    for s in signals:
        k = s.get("strategy", "unknown")
        strats[k] = strats.get(k, 0) + 1

    symbols = list({s.get("symbol", "EUR/USD").replace("/", "_") for s in signals})

    obs_signals   = _observation_signals()
    d1_positions  = [s.get("d1_position") for s in obs_signals if s.get("d1_position") is not None]
    d1_zone_label = "unknown"
    if d1_positions:
        latest_pos    = d1_positions[-1]
        d1_zone_label = "premium" if latest_pos > 0.6 else "discount" if latest_pos < 0.4 else "equilibrium"

    return jsonify({
        "signals_total":    len(signals),
        "signals_live":     len(_read(_sig("all_signals.json")) or []),
        "signals_planned":  len(planned),
        "signals_active":   len(active),
        "triggered_total":  len(triggered),
        "win_count":        len(wins),
        "loss_count":       len(triggered) - len(wins),
        "win_rate":         round(len(wins) / len(triggered) * 100, 1) if triggered else 0,
        "entry_missed":     len(entry_missed),
        "avg_probability":  round(sum(probs) / len(probs), 1) if probs else 0,
        "avg_rr":           f"1:{round(sum(rr_vals)/len(rr_vals),1)}" if rr_vals else "—",
        "best_rr":          f"1:{max(rr_vals)}" if rr_vals else "—",
        "zones_total":      len(all_zones),
        "daily_plan_count": len(daily),
        "strategy_counts":  strats,
        "symbols_detected": symbols,
        "observation_count":   len(obs_signals),
        "d1_zone":             d1_zone_label,
        "d1_position":         round(d1_positions[-1], 3) if d1_positions else None,
        "updated":             datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


# ── /api/status ───────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    files = {
        "all_signals":   _sig("all_signals.json").exists(),
        "daily_plan":    _sig("daily_plan.json").exists(),
        "weekly_plan":   _sig("weekly_plan.json").exists(),
        "zones_state":   _sig("zones_state.json").exists(),
        "funnel_state":  _sig("funnel_state.json").exists(),
    }
    file_updated = {}
    for k, fname in [("all_signals","all_signals.json"),("daily_plan","daily_plan.json"),("funnel_state","funnel_state.json")]:
        mt = _mtime(_sig(fname))
        if mt:
            file_updated[k] = mt

    zone_files = [p.name for p in SIGNALS_DIR.glob("zones_*.json")]
    symbols    = [p.stem.replace("zones_","").replace("_cont","").replace("_rev","")
                  for p in SIGNALS_DIR.glob("zones_*_cont.json")]

    return jsonify({
        "status":           "ok",
        "files":            files,
        "file_updated":     file_updated,
        "zone_files":       zone_files,
        "symbols_detected": list(set(symbols)),
        "signals_dir":      str(SIGNALS_DIR),
        "server_time":      datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


# ── /api/health ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    ot_v2_path = Path(__file__).parent.parent / "outcome_tracker_v2.py"
    checks = {
        "signals_dir_exists":   SIGNALS_DIR.exists(),
        "all_signals_json":     _sig("all_signals.json").exists(),
        "daily_plan_json":      _sig("daily_plan.json").exists(),
        "funnel_state_json":    _sig("funnel_state.json").exists(),
        "outcome_tracker_v2":   ot_v2_path.exists(),
    }
    if not ot_v2_path.exists():
        checks["outcome_tracker_v2_fix"] = (
            "Copy outcome_tracker_v2.py into forex_ml_system/forex_ml_system/ "
            "then remove the import from production_predictor.py until you do"
        )
    if not _sig("funnel_state.json").exists():
        checks["funnel_state_fix"] = (
            "Add funnel serialisation to production_predictor.py — "
            "save per-pair htf_before/after/ob_skips/reject_reasons to signals/funnel_state.json each cycle"
        )

    zone_files = [p.name for p in SIGNALS_DIR.glob("zones_*.json")]
    return jsonify({
        "status":      "ok" if checks["signals_dir_exists"] else "error",
        "checks":      checks,
        "zone_files":  zone_files,
        "signals_dir": str(SIGNALS_DIR),
        "server_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    })

@app.route("/api/signals")
def api_signals():
    data = _merge_signals()

    # ── filters ──────────────────────────────────────────────────────────────
    from flask import request
    tf  = request.args.get("timeframe")   # ?timeframe=HTF
    typ = request.args.get("type")        # ?type=rev  or  ?type=htf

    if tf:
        data = [s for s in data if s.get("timeframe", "").upper() == tf.upper()]
    if typ:
        t = typ.lower()
        if t == "htf":
            data = [s for s in data if s.get("strategy", "").upper().startswith("HTF")]
        elif t == "rev":
            data = [s for s in data if "rev" in s.get("strategy", "").lower()
                                    or "rev" in s.get("type", "").lower()]

    return jsonify({
        "data":        data,
        "count":       len(data),
        "live":        len(_read(_sig("all_signals.json")) or []),
        "planned":     len(_read(_sig("daily_plan.json"))  or []),
        "observation": len(_observation_signals()),
        "updated":     _mtime(_sig("all_signals.json")) or _mtime(_sig("daily_plan.json")),
    })
# ══════════════════════════════════════════════════════════════════════════════
#  MT5 BRIDGE — signal lifecycle: waiting → ready → triggered | expired
# ══════════════════════════════════════════════════════════════════════════════
#
#  Source: the 30-minute scanner's own files (the 30-minute strategy only).
#
#  waiting   : ledger state WAITING — signal produced, price has not retested the entry yet.
#  ready     : listed in trendm30_signals.json "qualified" (QUALIFIED + TAKE), i.e. exactly what
#              the dashboard's Ready tab shows: enter now. The EA only ever sees signals in this state.
#  triggered : the EA confirmed it placed the order(s) (POST /api/mt5/ack). Terminal.
#  expired   : ledger EXPIRED/closed, backfilled (found after the fact), too old when first
#              seen, left the feed, or the EA rejected it (drift, spread, pair busy...). Terminal.
#
#  The server is the single source of truth. State lives in signals/mt5_state.json
#  so it survives restarts and the EA can never be handed the same signal twice
#  after it has acknowledged it.
#
#  Config (environment variables, all optional):
#    MT5_TOKEN             shared secret; EA sends it as X-Token header
#    MT5_READY_STATUSES    ledger states that mean "enter now"      (default: open)
#    MT5_READY_MAX_MIN     a ready signal expires N min after the retest (default: 35)
#    MT5_WAITING_MAX_HRS   a waiting signal expires after N hours   (default: 24)
#    MT5_MIN_PROB          minimum ml_probability in %, lower is ignored (default: 70)
# ──────────────────────────────────────────────────────────────────────────────
MT5_STATE_FILE      = "mt5_state.json"
MT5_TOKEN           = os.getenv("MT5_TOKEN", "")
MT5_READY_MAX_MIN   = int(os.getenv("MT5_READY_MAX_MIN", "35"))
MT5_WAITING_MAX_HRS = int(os.getenv("MT5_WAITING_MAX_HRS", "24"))
MT5_MIN_PROB        = float(os.getenv("MT5_MIN_PROB", "70"))   # ignore signals below this ml_probability (%)
MT5_LEDGER_FILE     = "trendm30_signal_ledger.json"
MT5_SCANNER_FILE    = "trendm30_signals.json"
MT5_READY_STATUSES  = {x.strip().lower()
                       for x in os.getenv("MT5_READY_STATUSES", "open").split(",")
                       if x.strip()}
MT5_WAITING_STATUSES = {"waiting"}
# any other ledger state (expired, closed, tp1...) is treated as finished and never traded
MT5_KEEP_DAYS       = 14          # terminal records older than this are pruned

_mt5_lock = threading.Lock()


def _num(s: dict, *keys):
    """First key in `keys` that holds a usable number, else None."""
    for k in keys:
        v = s.get(k)
        if v in (None, "", "—"):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _ts(v):
    """'2026-10-27 08:00:00 UTC' -> epoch seconds (None if missing/unparseable)."""
    if not v:
        return None
    text = str(v).replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(text, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def _mt5_source():
    """Signals for the bridge, from the 30-minute scanner's own files.

    READY   = the scanner's `qualified` list (QUALIFIED + decision TAKE). This is exactly what the
              dashboard's Ready tab shows and it covers both immediate entries and retest entries.
              It is NOT taken from the ledger: the ledger keeps one active signal per pair and silently
              drops a new signal while an older trade on that pair is still open.
    WAITING = the ledger's WAITING signals (stable ids, so the lifecycle view stays readable).
    """
    out = []

    ledger = _read(_sig(MT5_LEDGER_FILE)) or {}
    for s in (ledger.get("signals", []) if isinstance(ledger, dict) else ledger):
        s = dict(s)
        if str(s.get("state", "")).upper() in ("OPEN", "TP1_HIT"):
            s["state"] = "RETESTED"          # the entry itself arrives via the qualified list below
        s.setdefault("strategy", "trendm30")
        s.setdefault("timeframe", "30m")
        out.append(s)

    scan = _read(_sig(MT5_SCANNER_FILE)) or {}
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for q in (scan.get("qualified", []) if isinstance(scan, dict) else []):
        if q.get("decision") != "TAKE" or q.get("direction") not in ("BUY", "SELL"):
            continue
        bar_open = _ts(q.get("candle_time"))
        if bar_open is None:
            continue
        out.append({                          # later entries win over earlier ones with the same id
            "id":             f"{q['symbol']}|{q['direction']}|{fmt(bar_open)}",
            "symbol":         q["symbol"],
            "direction":      q["direction"],
            "state":          "OPEN",         # = ready now
            "backfilled":     False,
            "triggered_at":   fmt(bar_open + 1800),     # the moment the M30 bar closed
            "entry":          q.get("entry"),
            "sl":             q.get("sl"),
            "tp1":            q.get("tp1"),
            "tp2":            q.get("tp2"),
            "ml_probability": q.get("ml_probability"),
            "strategy":       "trendm30",
            "timeframe":      "30m",
        })
    return out


def _mt5_build(s: dict):
    """Turn one pipeline signal into the compact shape the EA needs."""
    entry = _num(s, "entry", "entry_price")
    sl    = _num(s, "sl", "stop_loss", "stop")
    tp1   = _num(s, "tp1", "take_profit_1", "tp", "take_profit", "target")
    tp2   = _num(s, "tp2", "take_profit_2") or tp1

    d    = str(s.get("direction", "")).lower()
    side = "buy" if d in ("buy", "long", "bullish") else \
           "sell" if d in ("sell", "short", "bearish") else None
    if entry is None or side is None:
        return None

    raw_symbol = str(s.get("symbol", "EUR/USD"))
    stamp = s.get("id") or s.get("signal_id")
    if not stamp:
        stamp = "|".join(str(x) for x in (
            raw_symbol, side, s.get("strategy"), s.get("timeframe"),
            round(entry, 5),
            s.get("timestamp") or s.get("time") or s.get("created_at") or "",
        ))
    sid = hashlib.sha1(str(stamp).encode()).hexdigest()[:12]

    return {
        "id":          sid,
        "symbol":      raw_symbol.replace("/", "").replace("_", ""),
        "symbol_raw":  raw_symbol,
        "side":        side,
        "entry":       entry,
        "sl":          sl,
        "tp1":         tp1,
        "tp2":         tp2,
        "strategy":    s.get("strategy", ""),
        "timeframe":   s.get("timeframe", ""),
        "probability": _num(s, "ml_probability", "probability"),
        "raw_status":  str(s.get("state") or s.get("status") or "").lower(),
        "backfilled":  bool(s.get("backfilled")),
        "triggered_ts": _ts(s.get("triggered_at")),
        "tp1_hit":     bool(s.get("tp1_at")),
        "closed":      bool(s.get("closed_at")),
    }


def _mt5_load() -> dict:
    return _read(_sig(MT5_STATE_FILE)) or {}


def _mt5_save(state: dict):
    path = _sig(MT5_STATE_FILE)
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)          # atomic: the EA never reads a half-written file


def _mt5_set(rec: dict, new_state: str, reason: str, now: int):
    rec["state"]   = new_state
    rec["reason"]  = reason
    rec["updated"] = now
    rec.setdefault("history", []).append({"state": new_state, "reason": reason, "t": now})


def _mt5_sync(state: dict) -> dict:
    """Advance every signal through the lifecycle based on the live feed and the clock."""
    now  = int(time.time())
    feed = {}
    for s in _mt5_source():
        sig = _mt5_build(s)
        if sig:
            feed[sig["id"]] = sig

    # 1) new + existing signals from the feed
    for sid, sig in feed.items():
        rec = state.get(sid)
        if rec is None:
            rec = {**sig, "state": "waiting", "first_seen": now, "updated": now,
                   "reason": "new", "history": [{"state": "waiting", "reason": "new", "t": now}]}
            state[sid] = rec

        if rec["state"] in ("triggered", "expired"):
            continue                                    # terminal — never reopened

        for k in ("entry", "sl", "tp1", "tp2", "symbol", "side",
                  "strategy", "timeframe", "probability", "raw_status"):
            rec[k] = sig[k]                             # keep levels fresh until traded

        raw  = sig["raw_status"]
        prob = sig["probability"]
        if prob is None or prob < MT5_MIN_PROB:         # below the confidence floor, or unknown
            _mt5_set(rec, "expired", f"low_probability:{prob}", now)
        elif raw in MT5_WAITING_STATUSES:
            if rec["state"] == "ready":                 # ledger pulled it back
                rec.pop("ready_at", None)
                _mt5_set(rec, "waiting", "ledger:waiting", now)
        elif raw in MT5_READY_STATUSES and not sig["tp1_hit"] and not sig["closed"]:
            opened = sig["triggered_ts"] or now         # the retest time, not when we noticed it
            if sig["sl"] is None or sig["tp1"] is None:
                rec["reason"] = "missing_levels"        # never trade without SL/TP
            elif sig["backfilled"]:
                _mt5_set(rec, "expired", "backfilled", now)         # found after the fact
            elif now - opened > MT5_READY_MAX_MIN * 60:
                _mt5_set(rec, "expired", "stale_on_arrival", now)   # retest was too long ago
            elif rec["state"] == "waiting":
                rec["ready_at"] = opened
                _mt5_set(rec, "ready", "retest_triggered", now)
        else:                                           # expired / closed / tp1 / anything unknown
            _mt5_set(rec, "expired", f"ledger:{raw or 'unknown'}", now)

    # 2) clock-based expiry and cleanup
    for sid in list(state):
        rec = state[sid]
        if rec["state"] in ("triggered", "expired"):
            if now - rec.get("updated", now) > MT5_KEEP_DAYS * 86400:
                del state[sid]
            continue
        if sid not in feed:
            _mt5_set(rec, "expired", "removed_from_feed", now)
        elif rec["state"] == "ready" and now - rec.get("ready_at", now) > MT5_READY_MAX_MIN * 60:
            _mt5_set(rec, "expired", "stale", now)
        elif rec["state"] == "waiting" and now - rec.get("first_seen", now) > MT5_WAITING_MAX_HRS * 3600:
            _mt5_set(rec, "expired", "waiting_timeout", now)
    return state


def _mt5_refresh() -> dict:
    """Load → sync → save (only when something changed). Caller must hold the lock."""
    state  = _mt5_load()
    before = json.dumps(state, sort_keys=True)
    _mt5_sync(state)
    if json.dumps(state, sort_keys=True) != before:
        _mt5_save(state)
    return state


def _mt5_authorized() -> bool:
    if not MT5_TOKEN:
        return True
    return (request.headers.get("X-Token") == MT5_TOKEN
            or request.args.get("token") == MT5_TOKEN)


def _mt5_unauthorized():
    return jsonify({"error": "unauthorized"}), 401


# ── /api/mt5/signals  (EA polls this; returns READY signals only) ─────────────
@app.route("/api/mt5/signals")
def api_mt5_signals():
    if not _mt5_authorized():
        return _mt5_unauthorized()
    now = int(time.time())
    with _mt5_lock:
        state = _mt5_refresh()
    ready = [
        {
            "id":       r["id"],
            "symbol":   r["symbol"],
            "side":     r["side"],
            "entry":    r["entry"],
            "sl":       r["sl"],
            "tp1":      r["tp1"],
            "tp2":      r["tp2"],
            "age_sec":  now - r.get("ready_at", now),
        }
        for r in state.values() if r["state"] == "ready"
    ]
    return jsonify({"server_time": now, "count": len(ready), "signals": ready})


# ── /api/mt5/ack  (EA: "I placed the order(s)") ───────────────────────────────
@app.route("/api/mt5/ack", methods=["POST"])
def api_mt5_ack():
    if not _mt5_authorized():
        return _mt5_unauthorized()
    body = request.get_json(silent=True) or {}
    sid  = str(body.get("id", ""))
    now  = int(time.time())
    with _mt5_lock:
        state = _mt5_refresh()
        rec   = state.get(sid)
        if rec is None:
            return jsonify({"error": "unknown id"}), 404
        if rec["state"] != "triggered":
            # even if it went stale while the order was in flight, the trade exists
            rec["tickets"]         = str(body.get("tickets", ""))
            rec["fill_price"]      = body.get("price")
            rec["triggered_at"]    = now
            _mt5_set(rec, "triggered", "ea_ack", now)
            _mt5_save(state)
    return jsonify({"ok": True, "id": sid, "state": "triggered"})


# ── /api/mt5/reject  (EA: "I refused it" — drift, spread, pair busy, ...) ─────
@app.route("/api/mt5/reject", methods=["POST"])
def api_mt5_reject():
    if not _mt5_authorized():
        return _mt5_unauthorized()
    body   = request.get_json(silent=True) or {}
    sid    = str(body.get("id", ""))
    reason = str(body.get("reason", "rejected"))[:80]
    now    = int(time.time())
    with _mt5_lock:
        state = _mt5_refresh()
        rec   = state.get(sid)
        if rec is None:
            return jsonify({"error": "unknown id"}), 404
        if rec["state"] not in ("triggered", "expired"):
            _mt5_set(rec, "expired", f"ea:{reason}", now)
            _mt5_save(state)
    return jsonify({"ok": True, "id": sid, "state": rec["state"]})


# ── /api/mt5/lifecycle  (dashboard: every signal and where it is) ─────────────
@app.route("/api/mt5/lifecycle")
def api_mt5_lifecycle():
    with _mt5_lock:
        state = _mt5_refresh()
    want = request.args.get("state")
    rows = [r for r in state.values() if not want or r["state"] == want]
    rows.sort(key=lambda r: r.get("updated", 0), reverse=True)
    counts = {"waiting": 0, "ready": 0, "triggered": 0, "expired": 0}
    for r in state.values():
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    return jsonify({"counts": counts, "count": len(rows), "data": rows[:200]})


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  ML Signal Dashboard — V8")
    print(f"  Signals dir : {SIGNALS_DIR}")
    print(f"  Exists      : {SIGNALS_DIR.exists()}")
    print(f"  URL         : http://localhost:5050")
    print(f"{'='*60}\n")

    sig_file    = _sig("all_signals.json")
    plan_file   = _sig("daily_plan.json")
    funnel_file = _sig("funnel_state.json")
    zone_files  = list(SIGNALS_DIR.glob("zones_*.json"))

    print(f"  all_signals.json  : {'✅' if sig_file.exists()    else '❌ missing — will use daily_plan as fallback'}")
    print(f"  daily_plan.json   : {'✅' if plan_file.exists()   else '❌ missing'}")
    print(f"  funnel_state.json : {'✅' if funnel_file.exists() else '⚠️  missing — Funnel tab will be empty until you add serialisation'}")
    print(f"  Zone files        : {[f.name for f in zone_files] or '❌ none found'}")

    ot_v2 = Path(__file__).parent.parent / "outcome_tracker_v2.py"
    if not ot_v2.exists():
        print(f"\n  ⚠️  outcome_tracker_v2.py not found")
        print(f"     Either copy it there OR comment out its import in production_predictor.py")

    print()
    app.run(host="0.0.0.0", port=5050, debug=False)
