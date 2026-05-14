#!/usr/bin/env python3
"""
Dashboard API Server — fixed
============================
Key fixes vs previous version:
  FIX-1  /api/signals now merges all_signals.json + daily_plan.json so the
         dashboard always has data even when the live cycle hasn't fired yet.
  FIX-2  /api/stats win rate counts only triggered trades (hit_tp + hit_sl),
         not expired/entry_missed which inflate the denominator.
  FIX-3  /api/zones/<strategy>/<symbol> route order fixed to match app.js calls
         (/api/zones/cont/EUR_USD  not /api/zones/EUR_USD/cont).
  FIX-4  /api/status endpoint added so app.js setStatus() doesn't 404.
  FIX-5  outcome_tracker_v2 import error surfaced clearly in /api/health.

Usage:
    cd ~/Desktop/forex_ml_system/forex_ml_system
    source venv/bin/activate
    python dash/server.py
"""

import json
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR    = Path(__file__).resolve().parent
SIGNALS_DIR = Path(__file__).resolve().parent.parent / "signals"


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


def _merge_signals():
    """
    FIX-1: Return live signals + daily plan entries as one unified list.
    Live signals (all_signals.json) take priority — daily plan entries are
    added only if no matching entry/direction already exists in live signals.
    Each daily plan entry is tagged with source='daily_plan' so the frontend
    can style them differently if needed.
    """
    live    = _read(_sig("all_signals.json")) or []
    plan    = _read(_sig("daily_plan.json"))  or []

    # Build a set of (direction, entry) keys already in live signals
    live_keys = {
        (s.get("direction"), round(float(s.get("entry", 0)), 5))
        for s in live
    }

    merged = list(live)
    for p in plan:
        key = (p.get("direction"), round(float(p.get("entry", 0)), 5))
        if key not in live_keys:
            p = dict(p)
            p.setdefault("status", "planned")   # not yet active
            p["source"] = "daily_plan"
            merged.append(p)

    return merged


# ── static ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(BASE_DIR / "static", fname)


# ── /api/signals ─────────────────────────────────────────────────────────────
@app.route("/api/signals")
def api_signals():
    data = _merge_signals()
    return jsonify({
        "data":    data,
        "count":   len(data),
        "live":    len(_read(_sig("all_signals.json")) or []),
        "planned": len(_read(_sig("daily_plan.json"))  or []),
        "updated": _mtime(_sig("all_signals.json")) or _mtime(_sig("daily_plan.json")),
    })


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


# FIX-3: route order matches what app.js actually calls
# app.js calls: /api/zones/cont/EUR_USD and /api/zones/rev/EUR_USD
@app.route("/api/zones/<strategy>/<symbol>")
def api_zones_specific(strategy, symbol):
    filename = f"zones_{symbol}_{strategy}.json"
    path     = _sig(filename)
    data     = _read(path)
    if data is None:
        return jsonify({"error": f"{filename} not found", "data": {}}), 404
    return jsonify({"data": data, "count": len(data), "updated": _mtime(path)})


# ── /api/zones_state ─────────────────────────────────────────────────────────
@app.route("/api/zones_state")
def api_zones_state():
    data = _read(_sig("zones_state.json"))
    if data is None:
        return jsonify({"error": "zones_state.json not found", "data": {}}), 404
    return jsonify({"data": data, "updated": _mtime(_sig("zones_state.json"))})


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

    # FIX-2: win rate = wins / triggered only (not expired/entry_missed/planned)
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

    # Detect which symbols are present
    symbols = list({s.get("symbol", "EUR/USD").replace("/", "_") for s in signals})

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
        "updated":          datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


# ── /api/status  (FIX-4: was missing, caused 404 in app.js) ──────────────────
@app.route("/api/status")
def api_status():
    files = {
        "all_signals":  _sig("all_signals.json").exists(),
        "daily_plan":   _sig("daily_plan.json").exists(),
        "weekly_plan":  _sig("weekly_plan.json").exists(),
        "zones_state":  _sig("zones_state.json").exists(),
    }
    file_updated = {}
    for k, fname in [("all_signals","all_signals.json"),("daily_plan","daily_plan.json")]:
        mt = _mtime(_sig(fname))
        if mt:
            file_updated[k] = mt

    zone_files = [p.name for p in SIGNALS_DIR.glob("zones_*.json")]
    symbols    = [p.stem.replace("zones_","").replace("_cont","").replace("_rev","")
                  for p in SIGNALS_DIR.glob("zones_*_cont.json")]
    symbols_detected = list(set(symbols))

    return jsonify({
        "status":           "ok",
        "files":            files,
        "file_updated":     file_updated,
        "zone_files":       zone_files,
        "symbols_detected": symbols_detected,
        "signals_dir":      str(SIGNALS_DIR),
        "server_time":      datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


# ── /api/health ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    # FIX-5: check for outcome_tracker_v2 and report clearly
    ot_v2_path = Path(__file__).parent.parent / "outcome_tracker_v2.py"
    checks = {
        "signals_dir_exists":    SIGNALS_DIR.exists(),
        "all_signals_json":      _sig("all_signals.json").exists(),
        "daily_plan_json":       _sig("daily_plan.json").exists(),
        "outcome_tracker_v2":    ot_v2_path.exists(),
    }
    zone_files = [p.name for p in SIGNALS_DIR.glob("zones_*.json")]

    if not ot_v2_path.exists():
        checks["outcome_tracker_v2_fix"] = (
            "Copy outcome_tracker_v2.py into forex_ml_system/forex_ml_system/ "
            "then remove the import from production_predictor.py until you do"
        )

    return jsonify({
        "status":      "ok" if checks["signals_dir_exists"] else "error",
        "checks":      checks,
        "zone_files":  zone_files,
        "signals_dir": str(SIGNALS_DIR),
        "server_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  ML Signal Dashboard")
    print(f"  Signals dir : {SIGNALS_DIR}")
    print(f"  Exists      : {SIGNALS_DIR.exists()}")
    print(f"  URL         : http://localhost:5050")
    print(f"{'='*60}\n")

    zone_files = list(SIGNALS_DIR.glob("zones_*.json"))
    sig_file   = _sig("all_signals.json")
    plan_file  = _sig("daily_plan.json")

    print(f"  all_signals.json : {'✅' if sig_file.exists()  else '❌ missing — will use daily_plan as fallback'}")
    print(f"  daily_plan.json  : {'✅' if plan_file.exists() else '❌ missing'}")
    print(f"  Zone files       : {[f.name for f in zone_files] or '❌ none found'}")

    ot_v2 = Path(__file__).parent.parent / "outcome_tracker_v2.py"
    if not ot_v2.exists():
        print(f"\n  ⚠️  outcome_tracker_v2.py not found in forex_ml_system/forex_ml_system/")
        print(f"     Either copy it there OR comment out its import in production_predictor.py")

    print()
    app.run(host="0.0.0.0", port=5050, debug=False)