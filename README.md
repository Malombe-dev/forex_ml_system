# Production Forex ML Trading System

A complete, production-ready forex trading system that combines Smart Money Concepts (SMC) with Machine Learning for high-probability trade signals.

## 🎯 What This System Does

1. **Ingests tick data** from Dukascopy CSV files
2. **Detects Smart Money patterns** (FVG, Order Blocks, Liquidity Sweeps, BOS/CHoCH)
3. **Trains ML model** to predict which zones will result in profitable trades
4. **Backtests** with realistic slippage and risk management
5. **Generates live signals** with calibrated probabilities

## 📊 Realistic Expectations

| Metric | Target | Notes |
|--------|--------|-------|
| **Win Rate** | 58-62% | Professional range |
| **Sharpe Ratio** | 0.7-1.2 | Profitable |
| **Max Drawdown** | <12% | Acceptable |
| **Avg R:R** | 2.0-2.5 | Risk:Reward |

⚠️ **This is NOT a 95% accuracy system**. We target consistent profitability through proper risk management, not high win rates.

## 🚀 Quick Start

### Installation

```bash
# Clone/download the system
git clone <your-repo>
cd forex-ml-system

# Install dependencies
pip install -r requirements.txt
```

### Get Tick Data

**Option 1: Download from Dukascopy**
1. Go to https://www.dukascopy.com/swiss/english/marketwatch/historical/
2. Download tick data as CSV
3. Save as `your_dukascopy_ticks.csv`

**Option 2: Use dukascopy-python**
```python
import dukascopy_python as dukas
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(days=7)

df = dukas.fetch('EURUSD', 'TICK', 'BID_ASK', start, end)
df.to_csv('your_dukascopy_ticks.csv')
```

### Run Complete Pipeline

```bash
# Run everything at once
python main_pipeline.py your_dukascopy_ticks.csv
```

This will:
1. ✅ Ingest tick data
2. ✅ Extract SMC features
3. ✅ Generate training labels
4. ✅ Train XGBoost model
5. ✅ Generate live signals

### Run Individual Modules

```bash
# 1. Data Ingestion
python tick_ingestion.py

# 2. Feature Engineering
python smc_detector.py

# 3. Label Generation
python label_generator.py

# 4. Model Training
python model_trainer.py

# 5. Backtesting
python backtest_engine.py

# 6. Live Signals
python production_engine.py
```

## 📁 Project Structure

```
forex_ml_system/
├── data/
│   ├── raw_ticks.parquet          # Ingested tick data
│   ├── bars_*.parquet             # Resampled bars (1min, 5min, 15min, etc.)
│   ├── features/
│   │   ├── fvgs.parquet          # Fair Value Gaps
│   │   ├── order_blocks.parquet   # Order Blocks
│   │   ├── merged_zones.parquet   # Final zones
│   │   └── ...
│   └── labels/
│       └── training_labels.parquet # ML training data
│
├── models/
│   ├── model_v1.pkl               # Trained XGBoost model
│   ├── scaler_v1.pkl              # Feature scaler
│   ├── encoders_v1.pkl            # Label encoders
│   └── training_results.json      # Training metrics
│
├── signals/
│   └── live_signals.json          # Generated signals
│
├── tick_ingestion.py              # Module 1: Data loading
├── smc_detector.py                # Module 2: SMC detection
├── label_generator.py             # Module 3: Label creation
├── model_trainer.py               # Module 4: ML training
├── backtest_engine.py             # Module 5: Backtesting
├── production_engine.py           # Module 6: Live signals
├── main_pipeline.py               # Orchestrator
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 How It Works

### 1. Data Ingestion (`tick_ingestion.py`)

```python
from tick_ingestion import TickDataIngestion

ingestion = TickDataIngestion(output_dir="data")
ticks = ingestion.load_dukascopy_csv("your_ticks.csv")
ingestion.save_ticks(ticks)
all_bars = ingestion.generate_all_timeframes(ticks)
```

- Loads Dukascopy tick CSV
- Calculates mid price and spread
- Resamples to 1min, 5min, 15min, 1h, 4h, 1d bars
- Saves as efficient Parquet files

### 2. SMC Feature Detection (`smc_detector.py`)

```python
from smc_detector import SMCDetector

detector = SMCDetector(pip_size=0.0001)
features = detector.extract_all_features(df)
```

Detects:
- **FVG (Fair Value Gaps)**: 3-bar price gaps
- **Order Blocks**: Last opposite candle before strong move
- **Liquidity Sweeps**: Spikes that reverse quickly
- **BOS/CHoCH**: Structure breaks
- **Merged Zones**: Confluence areas

### 3. Label Generation (`label_generator.py`)

```python
from label_generator import LabelGenerator

generator = LabelGenerator(risk_reward=2.5, horizon_bars=30)
labels_df = generator.generate_labels_from_zones(zones, df)
```

For each zone:
- Entry: Zone mid-price
- Stop Loss: Beyond zone edge (1 ATR)
- Take Profit: 2.5:1 risk-reward
- Horizon: 30 bars maximum

Labels:
- **1** if TP hit before SL ✅
- **0** if SL hit before TP ❌
- **Discard** if timeout (no edge)

### 4. ML Model Training (`model_trainer.py`)

```python
from model_trainer import ForexMLTrainer

trainer = ForexMLTrainer()
X_train, y_train = trainer.prepare_features(train_df)
trainer.train_model(X_train, y_train)
trainer.save_model(version="v1")
```

Model:
- **Algorithm**: XGBoost + Probability Calibration
- **Features**: Zone strength, HTF trend, RSI, ATR regime, session, etc.
- **Validation**: Walk-forward time series splits
- **Output**: Calibrated probability (0-1)

### 5. Backtesting (`backtest_engine.py`)

```python
from backtest_engine import BacktestEngine

backtest = BacktestEngine(
    initial_balance=10000,
    risk_per_trade=0.01,  # 1%
    slippage_pips=0.2
)
metrics = backtest.run_backtest(signals, df)
```

Includes:
- Realistic slippage (0.2 pips)
- Position sizing (1% risk per trade)
- Walk-forward validation
- Max concurrent trades limit

Metrics:
- Win rate, profit factor, Sharpe ratio
- Max drawdown, average bars held
- Returns by probability bucket

### 6. Live Signal Generation (`production_engine.py`)

```python
from production_engine import ProductionForexEngine

engine = ProductionForexEngine(threshold=0.65)
signals = engine.generate_signals(df)
engine.display_signals(signals)
```

Generates signals with:
- Entry, SL, TP levels
- ML probability (65%+ threshold)
- Risk:Reward ratio (2.5:1)
- Distance from current price
- Recommendation (STRONG/GOOD/MODERATE)

## 📈 Example Signal Output

```
🟢 SIGNAL #1 - BUY - STRONG
----------------------------------------------------------------------
   Probability:    78.3%
   Entry:          1.15749
   Stop Loss:      1.15706
   Take Profit:    1.15856
   Risk:Reward:    1:2.5
   Distance:       23.4 pips
   Zone Strength:  2.14
   Current Price:  1.15983
```

## ⚙️ Configuration

### Model Parameters

Edit in `model_trainer.py`:
```python
self.model = xgb.XGBClassifier(
    n_estimators=100,      # Number of trees
    max_depth=6,           # Tree depth
    learning_rate=0.05,    # Learning rate
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8   # Feature sampling
)
```

### Risk Management

Edit in `backtest_engine.py` or `production_engine.py`:
```python
risk_per_trade = 0.01      # 1% risk per trade
risk_reward = 2.5          # 2.5:1 target
max_concurrent = 3         # Max open trades
slippage_pips = 0.2        # Realistic slippage
```

### Signal Threshold

Edit in `production_engine.py`:
```python
threshold = 0.65  # Minimum 65% probability
```

## 🧪 Testing

### Backtest on Historical Data

```bash
python backtest_engine.py
```

### Paper Trade

1. Generate signals daily
2. Track in spreadsheet
3. Compare predicted vs actual outcomes
4. Minimum 2 weeks before going live

### Go Live Checklist

- [ ] Backtest shows Sharpe > 0.5
- [ ] Win rate > 55% in paper trading
- [ ] Calibration error < 5%
- [ ] Max drawdown acceptable (<15%)
- [ ] Paper traded for 2+ weeks
- [ ] Risk management tested
- [ ] Broker integration working
- [ ] Start with 0.01 lots only

## 🔍 Troubleshooting

### "Model files not found"
Run the training pipeline:
```bash
python main_pipeline.py your_ticks.csv
```

### "No zones detected"
- Check if tick data has sufficient history (6+ months ideal)
- Verify data quality (no gaps, correct format)
- Adjust detection parameters in `smc_detector.py`

### "Low win rate in backtest"
- Normal! Target is 55-65%, not 95%
- Check if slippage is too high
- Verify risk:reward ratio
- May need more training data

### "No signals generated"
- Current market may not have high-probability zones
- Lower threshold temporarily (min 0.55)
- Check if zones are too far from current price

## 📚 Understanding the Results

### Model Metrics

- **Accuracy**: % of correct predictions (target: 60-65%)
- **Precision**: % of wins when model says "win" (target: 65-70%)
- **Recall**: % of actual wins captured (target: 55-65%)
- **Log Loss**: Calibration quality (target: <0.65)
- **ROC AUC**: Discrimination ability (target: >0.65)

### Backtest Metrics

- **Win Rate**: 58-62% is professional
- **Sharpe Ratio**: >0.7 is good, >1.0 is excellent
- **Profit Factor**: >1.5 is profitable
- **Max Drawdown**: <12% is acceptable

### Calibration

Model should be well-calibrated:
- If it says 70% probability → should win ~70% of the time
- Calibration error should be <5%
- Check calibration plot after training

## 🎓 Learning Resources

### Smart Money Concepts
- Order Blocks: Last opposite candle before institutional move
- FVG: Inefficiency that price often revisits
- Liquidity Sweep: Taking out stops before reversing

### Machine Learning
- XGBoost: Gradient boosting for classification
- Calibration: Making probabilities realistic
- Walk-forward: Time series validation (never train on future!)

### Risk Management
- 1% rule: Never risk more than 1% per trade
- Kelly Criterion: Optimal position sizing
- Drawdown management: Stop trading if DD > 15%

## ⚠️ Disclaimers

1. **Past performance ≠ Future results**
   - Backtest results don't guarantee live performance
   - Markets change, models degrade

2. **No holy grail**
   - 60% win rate is realistic, not 95%
   - Expect losing streaks (normal!)
   - Risk management is critical

3. **You can lose money**
   - Only trade with capital you can afford to lose
   - Start small (0.01 lots)
   - Paper trade first (2+ weeks minimum)

4. **Not financial advice**
   - This is educational software
   - Do your own research
   - Consult financial advisor

## 🤝 Contributing

Improvements welcome! Areas to enhance:
- Additional SMC patterns (Breaker Blocks, Mitigation Blocks)
- Deep learning models (LSTM, Transformers)
- Real-time data feeds
- Broker integration (MetaTrader, Oanda, etc.)
- Advanced risk management (correlation filters, drawdown limits)

## 📄 License

MIT License - Use at your own risk

## 💬 Support

For questions or issues:
1. Check troubleshooting section above
2. Review code comments (heavily documented)
3. Check scikit-learn and XGBoost documentation
4. Test individual modules in isolation

## 🎉 Summary

You now have:
- ✅ Complete tick data pipeline
- ✅ SMC feature extraction
- ✅ ML training with proper validation
- ✅ Realistic backtesting
- ✅ Production signal generation

**Next steps:**
1. Run the pipeline on your tick data
2. Analyze backtest results
3. Paper trade for 2 weeks
4. Go live with small size
5. Monitor and iterate

**Remember:** Professional trading is about consistent execution of an edge, not high win rates. Target 58-62% wins with 2:1 R:R and proper risk management.

Good luck! 🚀