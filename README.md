# Production Forex ML Trading System (v7)

A production-ready forex trading system that combines Smart Money Concepts (SMC) with Machine Learning (calibrated ensembles of XGBoost, Random Forest, and Logistic Regression) to deliver high-probability trade signals.

> [!IMPORTANT]
> **System Status: READY FOR LIVE DEPLOYMENT**
> - **Models:** Fully calibrated v5 Ensembles (XGBoost 50%, Random Forest 35%, Logistic Regression 15%).
> - **Performance:** Reversion Model ROC-AUC: **~0.719** | Continuation Model ROC-AUC: **~0.709**.
> - **Pairs Supported:** EUR/USD, GBP/USD, USD/JPY, XAU/USD.
> - **Historical Base:** Retrained on a 23-year dataset (5,751 labeled zones).

---

## 🎯 System Features & Safety Guards

1. **Dual-Strategy Pipeline:** Detects and executes both **Reversion** (prob. threshold: `62%`) and **Continuation** (prob. threshold: `65%`) trade setups.
2. **SMC Zone Detection:** Auto-detects Fair Value Gaps (FVG), Order Blocks (OB), Liquidity Sweeps, and BOS/CHoCH.
3. **Twelve Data API Caching & Rate-Limiter:** Limits calls to 7/min per key with a dynamic fallback pool. Uses a shared bar cache (60s TTL) to prevent duplicate calls.
4. **Open-Trade Guard:** Prevents doubling exposure on the same zone. Blocks re-firing if a triggered trade exists on the same symbol and direction within $2 \times \text{ATR}$.
5. **Calibrated Probability Gate:** Live signals must clear a strict `_ml_gate` of **0.75** (75%) for execution.
6. **Data Quality Monitor:** A background quality logger (`twelvedata_logger.py`) checks for stale, frozen, or missing bars, and anomalous price spikes.

---

## 📊 Realistic Expectations

| Metric | Target | Notes |
| :--- | :--- | :--- |
| **Win Rate** | 55–62% | Professional range at $0.65+$ threshold |
| **Sharpe Ratio** | 0.7–1.2 | Highly competitive risk-adjusted returns |
| **Max Drawdown** | <12% | Acceptable risk parameters |
| **Avg Risk:Reward** | 2.0–2.5 | Risk-to-Reward ratio |

> [!WARNING]
> This is **NOT** a 90%+ accuracy system. Consistent profitability is achieved via strict risk management and taking trades with a validated, calibrated statistical edge.

---

## 📁 Project Structure

```text
forex_ml_system/
├── main_pipeline.py              # Central orchestrator (runs Phases 1 to 6)
├── production_predictor.py        # Live signal engine (Module 6: Reversion + Continuation)
├── run_predictor.sh              # Shell launcher for live signal engine
├── tick_ingestion.py             # Phase 1: dukascopy tick CSV -> Parquet
├── smc_detector.py               # Phase 2: multi-timeframe SMC zone detection
├── label_generator.py            # Phase 3: dual-label generation (TP/SL/Timeouts)
├── validate_pipeline.py          # Phase 3.5: pipeline data consistency check
├── model_trainer.py              # Phase 4: trains XGBoost, Random Forest, LR ensemble
├── backtest_engine.py            # Phase 5: realistic walk-forward backtester
├── outcome_tracker.py            # Phase 6: tracks and updates live trade outcomes
│
├── retrain_all.py                # Periodic model retraining (runs every 30–90 days)
├── auto_retrain_monitor.py       # Automated weekly win-rate monitor & alert checker
├── twelvedata_logger.py          # API live data quality and spike monitor
├── signal_pusher.py              # Telegram/Webhook signal transmitter
├── telegram_signal_monitor.py    # Verification monitor for live Telegram alerts
│
├── dash/                         # Flask-based monitoring dashboard
│   ├── server.py                 # Backend API service
│   ├── index.html                # Signal & Zone maps visualization frontend
│   ├── static/                   # Static styling assets
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── requirements.txt          # Dashboard python requirements
│
├── data/                         # Data directory (ignored in git except config)
│   ├── backtest_results/         # Walk-forward backtest logs and charts
│   ├── features/                 # Generated SMC zones, OBs, and FVGs per pair
│   ├── labels/                   # Compiled training dataset
│   └── outcomes/                 # Outcome tracking records
│
├── models/                       # Stored model weights (v5)
│   ├── model_v5_continuation.pkl # Continuation XGBoost model
│   ├── model_v5_reversion.pkl    # Reversion XGBoost model
│   ├── scaler_v5_*.pkl           # Scalers for feature normalization
│   ├── encoders_v5_*.pkl         # Categorical label encoders
│   ├── lr_v5_*.pkl               # Probability calibration model weights
│   └── metrics_*/                # Confusion matrices & calibration curve plots
│
└── signals/                      # Live signals outputs
    ├── all_signals.json          # Combined signals file (Dashboard source of truth)
    ├── daily_plan.json           # Daily active zones
    └── weekly_plan.json          # Weekly setup planner
```

---

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd forex_ml_system

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Live Signal Predictor Execution

Provide your Twelve Data API keys inside your environment variables:
```bash
export TWELVE_KEY_EURUSD="your_api_key"
export TWELVE_KEY_GBPUSD="your_api_key"
export TWELVE_KEY_USDJPY="your_api_key"
export TWELVE_KEY_XAUUSD="your_api_key"
export TWELVE_KEY_SPARE_1="your_spare_key"  # Optional fallback key
```

Run the live monitoring loop (runs continuously, checking every 15 minutes):
```bash
python production_predictor.py
```
Or run the launcher script:
```bash
./run_predictor.sh
```

---

## ⚙️ Running the ML Pipeline

The `main_pipeline.py` script coordinates the complete historical backtesting and training cycle.

### Full Pipeline Run (Raw Dukascopy CSV -> Training)
```bash
python main_pipeline.py data/EURUSD_ticks.csv
```

### Run Specific Phases
You can resume or isolate phases using `--phase`:
- **Phase 1:** Ingest tick CSV (`tick_ingestion.py`)
- **Phase 2:** Feature Engineering (`smc_detector.py`)
- **Phase 3:** Label Generation (`label_generator.py`)
- **Phase 4:** Model Training (`model_trainer.py`)
- **Phase 5:** Backtesting (`backtest_engine.py`)
- **Phase 6:** Outcome Tracking (`outcome_tracker.py`)

```bash
# Retrain models only (using existing parquet labels)
python main_pipeline.py dummy --phase 4

# Run backtest only
python main_pipeline.py dummy --phase 5
```

---

## 🩺 System Diagnostics & Maintenance

### 1. Automated Model Retraining
Models should be retrained every 30–90 days to adapt to changing market dynamics.
```bash
python retrain_all.py
```

### 2. Live Quality Logger
Run the Twelve Data quality monitoring script in the background to detect data anomalies or API credit depletion:
```bash
python twelvedata_logger.py
```
Outputs are written directly to `logs/twelvedata_quality.log`.

### 3. Automated Weekly Check
Check live performance metrics and verify if retraining is triggered:
```bash
python auto_retrain_monitor.py --check
```

---

## 🖥️ Live Dashboard

The project includes a Flask-based signal dashboard that displays live signals, active zones, daily plans, and overall success metrics.

```bash
# Launch dashboard server
python dash/server.py
```
Open **[http://localhost:5050](http://localhost:5050)** in your web browser.

---

## ⚠️ Disclaimers

1. **Past performance does not guarantee future results.** Financial markets change, and machine learning models degrade over time without proper retraining.
2. **No Holy Grail:** Maintain strict risk parameters. Never risk more than 1–2% of equity per trade.
3. **Consult a professional:** This software is for educational purposes only. Trade live funds at your own risk.

---

## 📄 License
MIT License - Use at your own risk.