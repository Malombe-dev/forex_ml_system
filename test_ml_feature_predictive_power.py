#!/usr/bin/env python3
"""
ML Feasibility Study:
Trains an ML model (RandomForest & GradientBoosting) on the TrendM30 features
to predict trade outcomes (Win vs Loss) and evaluates model accuracy, AUC, and precision at high thresholds (>= 0.70).
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from filter_70pct_deep_dive import run_deep_filter_analysis

def main():
    print("=" * 80)
    print("  ML SYSTEM INTEGRATION FEASIBILITY STUDY: TRENDM30 FEATURES")
    print("=" * 80)

    # 1. Load dataset of trades generated across all 4 pairs
    df = run_deep_filter_analysis()
    print(f"Total trade records: {len(df)}")
    
    # Target: 1 if win (TP hit), 0 if loss (SL hit)
    y = df["is_win_single"].values

    # Features:
    feature_cols = [
        "htf_slope",       # 4H EMA50 slope
        "dist_ma60",       # Distance from MA60 white line
        "atr_ratio",       # Volatility expansion
        "rsi",             # RSI level
        "adx",             # ADX strength
        "in_ob_zone",      # Order Block presence
    ]

    X = df[feature_cols].copy()
    
    # Add engineered features
    X["yellow_white_near"] = (X["dist_ma60"] <= 0.5).astype(int) # Yellow meets white
    X["is_tue_thu"] = df["day"].isin(["Tuesday", "Thursday"]).astype(int)
    X["is_friday"] = (df["day"] == "Friday").astype(int)

    feature_names = X.columns.tolist()
    X_mat = X.values

    # 2. Time-Series Walk-Forward Cross Validation (No lookahead!)
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_accs, rf_precs, rf_aucs = [], [], []
    gb_accs, gb_precs, gb_aucs = [], [], []
    
    # Track high-confidence predictions (probability >= 0.70)
    high_conf_wins = 0
    high_conf_trades = 0

    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)

    for train_idx, test_idx in tscv.split(X_mat):
        X_train, X_test = X_mat[train_idx], X_mat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        gb.fit(X_train, y_train)
        probs = gb.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        gb_accs.append(accuracy_score(y_test, preds))
        gb_aucs.append(roc_auc_score(y_test, probs))
        
        # High confidence evaluation (probs >= 0.70)
        hc_mask = (probs >= 0.70)
        high_conf_trades += hc_mask.sum()
        high_conf_wins += y_test[hc_mask].sum()

    print("\n--- Model Evaluation (Time-Series Cross-Validation) ---")
    print(f"Mean Out-of-Sample Accuracy: {np.mean(gb_accs)*100:.1f}%")
    print(f"Mean Out-of-Sample ROC AUC:  {np.mean(gb_aucs):.3f}")
    
    if high_conf_trades > 0:
        hc_wr = high_conf_wins / high_conf_trades * 100
        print(f"\n🎯 High-Confidence Trades (ML Probability >= 70%):")
        print(f"   Trades Filtered: {high_conf_trades}")
        print(f"   Winning Trades:  {high_conf_wins}")
        print(f"   ML Win Rate:     {hc_wr:.1f}%")

    # 3. Feature Importance
    gb.fit(X_mat, y)
    importances = gb.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance (%)": importances * 100}).sort_values("Importance (%)", ascending=False)
    print("\n--- ML Feature Importance Ranking ---")
    print(df_imp.to_string(index=False))

if __name__ == "__main__":
    main()
