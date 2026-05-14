/* ============================================================
   Forex ML Dashboard — app.js  (patched)
   Key fixes:
   - renderSignals: shows planned/null status by default
   - statusBadge: handles "planned" status
   - renderMetrics: uses correct stat keys from server.py
   - index.html element IDs matched (zoneMap vs contZoneMap/revZoneMap)
   ============================================================ */

const API        = "";
const REFRESH_MS = 60_000;

let STATE  = { signals: [], contZones: {}, revZones: {}, dailyPlan: [], stats: {} };
let CHARTS = {};

/* ── Fetch ──────────────────────────────────────────────────────────────────── */
async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

async function fetchAll() {
  try {
    const [sigs, zones, daily, stats] = await Promise.all([
      get("/api/signals"),
      get("/api/zones"),
      get("/api/daily_plan"),
      get("/api/stats"),
    ]);

    STATE.signals   = sigs.data  || [];
    STATE.dailyPlan = daily.data || [];
    STATE.stats     = stats;

    STATE.contZones = {};
    STATE.revZones  = {};
    for (const [key, val] of Object.entries(zones)) {
      if (key.endsWith("_cont")) STATE.contZones = val.data || {};
      if (key.endsWith("_rev"))  STATE.revZones  = val.data || {};
    }

    setStatus("ok");
    const updEl = document.getElementById("last-updated");
    if (updEl) updEl.textContent = "Updated " + (stats.updated || new Date().toUTCString());

    render();
  } catch (err) {
    console.error("fetchAll error:", err);
    setStatus("error");
  }
}

function setStatus(s) {
  const el = document.getElementById("server-status");
  if (!el) return;
  el.className = "status-dot " + s;
  el.title = s === "ok" ? "Server connected" : "Server unreachable";
}

/* ── Clock ───────────────────────────────────────────────────────────────────── */
function tickClock() {
  const el = document.getElementById("clock");
  if (!el) return;
  const now = new Date();
  const eat = new Date(now.getTime() + 3*60*60*1000);
  el.textContent = eat.toISOString().replace("T"," ").substr(0,16) + " EAT";
}
setInterval(tickClock, 1000);
tickClock();

/* ── Render ──────────────────────────────────────────────────────────────────── */
function render() {
  renderPairTags();
  renderMetrics();
  renderStratBar();
  renderSignals();
  renderProbChart();
  renderRRChart();
  renderZoneMap();
  renderZoneTable("cont-zones", STATE.contZones);
  renderZoneTable("rev-zones",  STATE.revZones);
  renderDailyPlan();
}

/* ── Pair tags ───────────────────────────────────────────────────────────────── */
function renderPairTags() {
  const el = document.getElementById("pair-tags");
  if (!el) return;
  const symbols = [...new Set(STATE.signals.map(s => s.symbol).filter(Boolean))];
  const htf = STATE.signals.find(s => s.htf_trend)?.htf_trend || "neutral";
  el.innerHTML = symbols.map(sym =>
    `<span class="pair-tag ${htf}">${sym} · HTF ${htf.toUpperCase()}</span>`
  ).join("") + `<span class="pair-tag">v7</span>`;
}

/* ── Metrics ─────────────────────────────────────────────────────────────────── */
function renderMetrics() {
  const st  = STATE.stats;
  // signals_closed not in new server — derive it
  const closed = (st.triggered_total ?? 0);
  const cards = [
    { label: "Signals fired",   val: st.signals_total   ?? 0,  sub: `${st.signals_live ?? 0} live · ${st.signals_planned ?? 0} planned`, cls: "" },
    { label: "Win rate",        val: `${st.win_rate ?? 0}%`,   sub: `${st.win_count ?? 0}W / ${st.loss_count ?? 0}L`,  cls: (st.win_rate ?? 0) >= 60 ? "green" : "red" },
    { label: "Active",          val: st.signals_active  ?? 0,  sub: "open positions",           cls: "blue" },
    { label: "Avg probability", val: `${st.avg_probability ?? 0}%`, sub: "across all signals",  cls: (st.avg_probability ?? 0) >= 75 ? "green" : "" },
    { label: "Avg RR",          val: st.avg_rr  ?? "—",        sub: "reward per risk",          cls: "" },
    { label: "Best RR",         val: st.best_rr ?? "—",        sub: "single trade",             cls: "amber" },
    { label: "Active zones",    val: st.zones_total ?? 0,      sub: "cont + rev",               cls: "" },
    { label: "Daily plan",      val: st.daily_plan_count ?? 0, sub: "setups today",             cls: "" },
  ];
  const el = document.getElementById("metrics-row");
  if (!el) return;
  el.innerHTML = cards.map(c => `
    <div class="metric">
      <div class="metric-label">${c.label}</div>
      <div class="metric-val ${c.cls}">${c.val}</div>
      <div class="metric-sub">${c.sub}</div>
    </div>`).join("");
}

/* ── Strategy bar ────────────────────────────────────────────────────────────── */
function renderStratBar() {
  const counts = STATE.stats.strategy_counts || {};
  const colors = { HTF_SMC: "#4f9ef8", REVERSION: "#a78bfa", CONTINUATION: "#20d48a" };
  const el = document.getElementById("strat-bar");
  if (!el) return;
  if (!Object.keys(counts).length) { el.innerHTML = `<span class="empty">No data</span>`; return; }
  el.innerHTML = Object.entries(counts).map(([k, v]) =>
    `<div class="strat-item">
      <span class="dot" style="background:${colors[k] || '#7a82a0'}"></span>
      <span>${k}</span>
      <strong style="color:var(--text)">${v}</strong>
    </div>`
  ).join("");
}

/* ── Signals table ───────────────────────────────────────────────────────────── */
function renderSignals() {
  const stratF = (document.getElementById("f-strategy") || {}).value || "";
  const statF  = (document.getElementById("f-status")   || {}).value || "";
  const dirF   = (document.getElementById("f-dir")      || {}).value || "";
  const tfF    = (document.getElementById("f-tf")       || {}).value || "";

  let data = STATE.signals;
  if (stratF) data = data.filter(s => s.strategy  === stratF);
  if (dirF)   data = data.filter(s => s.direction  === dirF);
  if (tfF)    data = data.filter(s => s.timeframe  === tfF);

  // Status filter — "planned" and null/undefined treated as "pending/planned"
  if (statF) {
    if (statF === "planned") {
      data = data.filter(s => !s.status || s.status === "planned");
    } else {
      data = data.filter(s => s.status === statF);
    }
  }

  const tbody = document.getElementById("signals-tbody");
  if (!tbody) return;

  if (!data.length) {
    tbody.innerHTML = `<tr><td colspan="14" class="empty">No signals match filters</td></tr>`;
    return;
  }

  tbody.innerHTML = data.map((s, i) => {
    const time   = fmtTime(s.timestamp);
    const pColor = probColor(s.probability);
    const confs  = s.confirmations ?? s.confluence_score ?? "—";
    const refined = s["5min_refined"]
      ? `<span style="color:var(--green)">✓</span>`
      : `<span style="color:var(--text3)">—</span>`;
    const ct = s.counter_trend
      ? `<span style="color:var(--amber);font-size:10px">⚠</span>`
      : `<span style="color:var(--text3)">—</span>`;
    // Source badge for plan entries
    const srcBadge = s.source === "daily_plan"
      ? `<span style="font-size:9px;color:var(--amber);margin-left:3px">plan</span>`
      : "";
    return `<tr>
      <td class="mono c-dim">${time}</td>
      <td><span class="badge b-tf">${s.timeframe}</span></td>
      <td class="c-dim" style="font-size:11px">${s.strategy || "—"}${srcBadge}</td>
      <td>${dirBadge(s.direction)}</td>
      <td class="mono">${Number(s.entry).toFixed(5)}</td>
      <td class="mono c-sl">${Number(s.sl).toFixed(5)}</td>
      <td class="mono c-tp">${Number(s.tp).toFixed(5)}</td>
      <td class="c-dim">${s.risk_pips}p</td>
      <td style="font-weight:700">${s.rr}</td>
      <td>
        <div class="prob-wrap">
          <div class="prob-track">
            <div class="prob-fill" style="width:${Math.min(s.probability,100)}%;background:${pColor}"></div>
          </div>
          <span style="color:${pColor};font-size:11px;font-weight:700">${Number(s.probability).toFixed(1)}%</span>
        </div>
      </td>
      <td class="c-dim">${confs}</td>
      <td>${refined}</td>
      <td>${ct}</td>
      <td>${statusBadge(s.status)}</td>
    </tr>`;
  }).join("");
}

/* ── Charts ──────────────────────────────────────────────────────────────────── */
function renderProbChart() {
  const sigs   = STATE.signals;
  if (!sigs.length) return;
  const labels = sigs.map((s, i) => `S${i+1} ${(s.strategy||"").replace("_SMC","")}`);
  const data   = sigs.map(s => s.probability);
  const colors = sigs.map(s => s.direction === "BUY" ? "#20d48a" : "#f05b5b");

  const legEl = document.getElementById("prob-legend");
  if (legEl) legEl.innerHTML = sigs.map((s, i) =>
    `<span><span class="ld" style="background:${colors[i]}"></span>S${i+1} ${s.strategy} ${s.direction}</span>`
  ).join("");

  rebuildChart("probChart", "prob", {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 5, borderSkipped: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${c.parsed.y.toFixed(1)}%` } } },
      scales: {
        x: chartAxis(),
        y: { ...chartAxis(true), min: 50, max: 95, ticks: { ...chartAxis(true).ticks, callback: v => v + "%" } }
      }
    }
  });
}

function renderRRChart() {
  const sigs = STATE.signals;
  if (!sigs.length) return;
  rebuildChart("rrChart", "rr", {
    type: "bar",
    data: {
      labels: sigs.map((_, i) => `S${i+1}`),
      datasets: [
        { label: "Risk pips",   data: sigs.map(s => s.risk_pips),   backgroundColor: "rgba(240,91,91,0.7)",  borderRadius: 4, borderSkipped: false },
        { label: "Reward pips", data: sigs.map(s => s.reward_pips), backgroundColor: "rgba(32,212,138,0.7)", borderRadius: 4, borderSkipped: false },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#7a82a0", font:{size:11}, boxWidth:10 } }, tooltip: { callbacks: { label: c => ` ${c.parsed.y}p` } } },
      scales: { x: chartAxis(), y: { ...chartAxis(true), ticks: { ...chartAxis(true).ticks, callback: v => v+"p" } } }
    }
  });
}

function chartAxis(isY = false) {
  return {
    ticks: { color: "#464e68", font: { size: 10 }, autoSkip: isY ? undefined : false },
    grid:  { color: "rgba(255,255,255,0.04)" }
  };
}

function rebuildChart(canvasId, key, config) {
  if (CHARTS[key]) { CHARTS[key].destroy(); delete CHARTS[key]; }
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  CHARTS[key] = new Chart(canvas, config);
}

/* ── Zone map ────────────────────────────────────────────────────────────────── */
function renderZoneMap() {
  // Supports both single canvas (id=zoneMap) and dual canvas (contZoneMap/revZoneMap)
  const single = document.getElementById("zoneMap");
  if (single) {
    drawZoneCanvas(single, {...STATE.contZones, ...STATE.revZones}, STATE.signals);
  } else {
    const cc = document.getElementById("contZoneMap");
    const rc = document.getElementById("revZoneMap");
    if (cc) drawZoneCanvas(cc, STATE.contZones, STATE.signals);
    if (rc) drawZoneCanvas(rc, STATE.revZones,  STATE.signals);
  }
}

function drawZoneCanvas(canvas, zones, signals) {
  const ctx = canvas.getContext("2d");
  canvas.width  = canvas.offsetWidth  || canvas.parentElement?.offsetWidth  || 600;
  canvas.height = canvas.offsetHeight || 320;
  const W = canvas.width, H = canvas.height;
  if (W < 10 || H < 10) return;
  ctx.clearRect(0, 0, W, H);

  const prices = [];
  Object.values(zones).forEach(z => prices.push(z.price_mid));
  signals.forEach(s => prices.push(s.entry, s.sl, s.tp));
  if (!prices.length) {
    ctx.fillStyle = "#464e68"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("No zone data", W/2, H/2);
    return;
  }

  const pad  = 0.0010;
  const minP = Math.min(...prices) - pad;
  const maxP = Math.max(...prices) + pad;
  const rng  = maxP - minP;
  const P2Y  = p => H * 0.95 - ((p - minP) / rng) * H * 0.90;

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.04)"; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 7; i++) {
    const y = (H / 7) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Zones
  for (const [sig, z] of Object.entries(zones)) {
    const parts = sig.split("|");
    const lo = parseFloat(parts[2]), hi = parseFloat(parts[3]);
    if (isNaN(lo) || isNaN(hi)) continue;
    const isS  = z.zone_type === "supply";
    const y1   = P2Y(Math.max(lo, hi));
    const y2   = P2Y(Math.min(lo, hi));
    const h    = Math.max(y2 - y1, 2);
    const prob = z.probability || 0;
    const alpha = 0.08 + prob * 0.15;
    ctx.fillStyle   = isS ? `rgba(240,91,91,${alpha})`  : `rgba(32,212,138,${alpha})`;
    ctx.strokeStyle = isS ? `rgba(240,91,91,0.6)`       : `rgba(32,212,138,0.6)`;
    ctx.lineWidth = 0.8;
    ctx.fillRect(W*0.04, y1, W*0.82, h);
    ctx.strokeRect(W*0.04, y1, W*0.82, h);
    if (prob > 0) {
      ctx.fillStyle = isS ? "#f05b5b" : "#20d48a";
      ctx.font = "9px monospace"; ctx.textAlign = "right";
      ctx.fillText((prob*100).toFixed(0)+"%", W*0.97, y1+9);
    }
  }

  // Signal entry lines
  ctx.font = "9px sans-serif";
  signals.forEach((s, i) => {
    const y   = P2Y(s.entry);
    const col = s.direction === "BUY" ? "#20d48a" : "#f05b5b";
    ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(W*0.02, y); ctx.lineTo(W*0.96, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.textAlign = "left";
    ctx.fillText(`S${i+1} ${s.direction} ${Number(s.entry).toFixed(5)}`, W*0.05, y-3);
  });

  // Price axis
  ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.font = "8px monospace"; ctx.textAlign = "right";
  for (let i = 0; i <= 7; i++) {
    const p = minP + (rng/7)*i;
    ctx.fillText(p.toFixed(5), W-2, P2Y(p));
  }
}

/* ── Zone tables ─────────────────────────────────────────────────────────────── */
function renderZoneTable(elId, zonesObj) {
  const el = document.getElementById(elId);
  if (!el) return;
  const zones = Object.values(zonesObj)
    .filter(z => z.probability > 0)
    .sort((a, b) => b.probability - a.probability);

  if (!zones.length) { el.innerHTML = `<p class="empty">No scored zones</p>`; return; }

  el.innerHTML = zones.map(z => {
    const parts  = z.signature.split("|");
    const range  = parts.length >= 4
      ? `${parseFloat(parts[2]).toFixed(5)} – ${parseFloat(parts[3]).toFixed(5)}`
      : z.price_mid.toFixed(5);
    const isS    = z.zone_type === "supply";
    const pct    = (z.probability * 100).toFixed(0);
    const barCol = isS ? "#f05b5b" : "#20d48a";
    return `<div class="zone-row">
      <span class="zone-type ${isS ? "supply" : "demand"}">${isS ? "SUPPLY" : "DEMAND"}</span>
      <span class="zone-range">${range}</span>
      <div class="prob-wrap">
        <div class="prob-track" style="width:48px">
          <div class="prob-fill" style="width:${pct}%;background:${barCol}"></div>
        </div>
        <span style="color:${barCol};font-size:11px;font-weight:700;min-width:28px;text-align:right">${pct}%</span>
      </div>
    </div>`;
  }).join("");
}

/* ── Daily plan ──────────────────────────────────────────────────────────────── */
function renderDailyPlan() {
  const el   = document.getElementById("daily-plan");
  if (!el) return;
  const plan = STATE.dailyPlan;
  if (!plan.length) {
    el.innerHTML = `<p class="empty">No daily plan — predictor generates one at session open</p>`;
    return;
  }
  el.innerHTML = plan.map(p => {
    const time = fmtTime(p.timestamp);
    const validUntil = p.valid_until
      ? new Date(p.valid_until).toLocaleDateString("en-GB", { day:"2-digit", month:"short" })
      : "—";
    const pCol = probColor(p.probability);
    return `<div class="plan-row">
      <div class="plan-meta">
        <span class="plan-time">${time} UTC</span>
        <span class="badge b-tf">${p.timeframe}</span>
        <span class="c-dim" style="font-size:11px">${p.strategy}</span>
        ${dirBadge(p.direction)}
        ${p.stale ? `<span class="badge b-stale">stale</span>` : ""}
        ${p.counter_trend ? `<span class="badge" style="background:var(--amber-d);color:var(--amber);font-size:9px">CT</span>` : ""}
      </div>
      <div class="plan-detail">
        <span><span class="lbl">Entry</span> <span class="mono">${Number(p.entry).toFixed(5)}</span></span>
        <span><span class="lbl">SL</span>    <span class="mono c-sl">${Number(p.sl).toFixed(5)}</span></span>
        <span><span class="lbl">TP</span>    <span class="mono c-tp">${Number(p.tp).toFixed(5)}</span></span>
        <span><span class="lbl">RR</span>    <strong style="color:var(--text)">${p.rr}</strong></span>
        <span><span class="lbl">Prob</span>  <strong style="color:${pCol}">${Number(p.probability).toFixed(1)}%</strong></span>
        <span><span class="lbl">Risk</span>  ${p.risk_pips}p</span>
        <span><span class="lbl">Valid</span> ${validUntil}</span>
      </div>
    </div>`;
  }).join("");
}

/* ── Helpers ─────────────────────────────────────────────────────────────────── */
function dirBadge(dir) {
  if (!dir) return "";
  return `<span class="badge ${dir === "BUY" ? "b-buy" : "b-sell"}">${dir}</span>`;
}

function statusBadge(status) {
  if (status === "hit_tp")  return `<span class="badge b-tp">TP ✓</span>`;
  if (status === "hit_sl")  return `<span class="badge b-sl">SL ✗</span>`;
  if (status === "active")  return `<span class="badge b-active">Active</span>`;
  if (status === "planned" || !status) return `<span class="badge" style="background:var(--amber-d);color:var(--amber)">Planned</span>`;
  if (status === "triggered") return `<span class="badge b-active">Triggered</span>`;
  if (status === "entry_missed") return `<span class="badge" style="background:var(--bg3);color:var(--text3)">Missed</span>`;
  return `<span class="c-dim" style="font-size:10px">${status}</span>`;
}

function probColor(p) {
  if (p >= 80) return "#20d48a";
  if (p >= 70) return "#4f9ef8";
  return "#7a82a0";
}

function fmtTime(ts) {
  if (!ts) return "—";
  if (typeof ts === "string") {
    // Plain time string like "16:01:05 UTC" — parse hours and add 3
    if (ts[2] === ":" || ts[1] === ":") {
      const parts = ts.split(":");
      const h = parseInt(parts[0], 10);
      const m = parts[1] ? parts[1].slice(0,2) : "00";
      const eat = (h + 3) % 24;
      return String(eat).padStart(2,"0") + ":" + m;
    }
    // ISO timestamp — force UTC parsing then add 3h for EAT
    if (ts.includes("T")) {
      const utcTs = ts.endsWith("Z") || ts.includes("+") ? ts : ts + "Z";
      const d = new Date(utcTs);
      if (isNaN(d.getTime())) return ts.slice(0,5);
      const eat = new Date(d.getTime() + 3 * 60 * 60 * 1000);
      return eat.toISOString().substr(11, 5);
    }
  }
  return String(ts).slice(0, 5);
}

/* ── Init ────────────────────────────────────────────────────────────────────── */
window.addEventListener("DOMContentLoaded", () => {
  fetchAll();
  setInterval(fetchAll, REFRESH_MS);
  window.addEventListener("resize", () => {
    if (STATE.signals.length || Object.keys(STATE.contZones).length) renderZoneMap();
  });
  // Wire filter dropdowns
  ["f-strategy","f-status","f-dir","f-tf"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderSignals);
  });
});