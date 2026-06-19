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

import json
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
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