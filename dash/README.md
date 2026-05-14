# Forex ML Signal Dashboard

Live dashboard for `production_predictor.py` (v7).
Reads directly from the `signals/` folder your predictor writes to.

## Folder placement

Drop the `dashboard/` folder **inside** `forex_ml_system/forex_ml_system/`:

```
forex_ml_system/
└── forex_ml_system/
    ├── production_predictor.py
    ├── signals/                   ← predictor writes here
    │   ├── all_signals.json
    │   ├── daily_plan.json
    │   ├── weekly_plan.json
    │   ├── zones_EUR_USD_cont.json
    │   ├── zones_EUR_USD_rev.json
    │   └── zones_state.json
    └── dashboard/                 ← this folder
        ├── server.py
        ├── index.html
        ├── requirements.txt
        └── static/
            ├── css/style.css
            └── js/app.js
```

## Setup (one time)

```bash
cd forex_ml_system/forex_ml_system
source venv/bin/activate
pip install flask flask-cors
```

## Run

```bash
# Terminal 1 — ML predictor (already running)
python production_predictor.py

# Terminal 2 — Dashboard server
python dashboard/server.py
```

Open **http://localhost:5050** in your browser.

The dashboard auto-refreshes every 60 seconds — matching the predictor's 15-min cycle.

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/signals` | All signals from `all_signals.json` |
| `GET /api/daily_plan` | Daily plan from `daily_plan.json` |
| `GET /api/weekly_plan` | Weekly plan from `weekly_plan.json` |
| `GET /api/zones` | All `zones_*.json` files combined |
| `GET /api/zones/EUR_USD/cont` | Specific zone file |
| `GET /api/zones_state` | Active zone tracker state |
| `GET /api/stats` | Derived summary stats (win rate, avg RR, etc.) |
| `GET /api/health` | Server health + file availability |

## Multi-pair

If your predictor runs multiple symbols (e.g. EUR/USD + GBP/USD), the
`/api/zones` endpoint auto-discovers all `zones_*.json` files. The zone map
currently renders the first pair found. To add multi-pair zone tabs, edit
`static/js/app.js` — the data is already loaded.

## Run as background service (optional)

```bash
nohup python dashboard/server.py > dashboard/server.log 2>&1 &
echo $! > dashboard/server.pid
# Stop: kill $(cat dashboard/server.pid)
```
