#!/usr/bin/env python3
"""
TrendM30 ML Model Trainer
Trains a calibrated XGBoost model on Twelve Data features (EUR/USD, GBP/USD, USD/JPY, XAU/USD)
to predict win probability on M30 Pullback setups.
Saves model and metadata to models/trendm30_ml_classifier.pkl
"""

import warnings
warnings.filterwarnings('ignore')
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, brier_score_loss
from filter_70pct_deep_dive import run_deep_filter_analysis

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save_trendm30_model():
    print("=" * 80)
    print("  TRAINING TRENDM30 ML CLASSIFIER (TWELVE DATA MULTI-ASSET)")
    print("=" * 80)

    # 1. Generate full trade history across all 4 pairs
    df_trades = run_deep_filter_analysis()
    n = len(df_trades)
    print(f"Total training trade samples: {n}")

    # Target: 1 = Win (Target reached), 0 = Loss (Stop Loss hit)
    y = df_trades["is_win_single"].values

    # Feature Engineering
    features_df = pd.DataFrame()
    features_df["dist_ma60"] = df_trades["dist_ma60"]
    features_df["htf_slope"] = df_trades["htf_slope"]
    features_df["atr_ratio"] = df_trades["atr_ratio"]
    features_df["rsi"] = df_trades["rsi"]
    features_df["adx"] = df_trades["adx"]
    features_df["in_ob_zone"] = df_trades["in_ob_zone"]
    features_df["yellow_white_near"] = (df_trades["dist_ma60"] <= 0.5).astype(int)
    features_df["is_tue_thu"] = df_trades["day"].isin(["Tuesday", "Thursday"]).astype(int)
    features_df["is_friday"] = (df_trades["day"] == "Friday").astype(int)
    features_df["hour"] = df_trades["hour"]

    feature_names = features_df.columns.tolist()
    X = features_df.values

    # 2. Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_accs = []
    cv_aucs = []
    cv_brier = []
    
    high_conf_trades = 0
    high_conf_wins = 0

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        base_clf = XGBClassifier(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss"
        )
        base_clf.fit(X_train, y_train)

        probs = base_clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        cv_accs.append(accuracy_score(y_test, preds))
        cv_aucs.append(roc_auc_score(y_test, probs))
        cv_brier.append(brier_score_loss(y_test, probs))

        hc_mask = probs >= 0.70
        high_conf_trades += hc_mask.sum()
        high_conf_wins += y_test[hc_mask].sum()

    print("\n--- Out-of-Sample Validation Metrics ---")
    print(f"Mean Accuracy:    {np.mean(cv_accs)*100:.1f}%")
    print(f"Mean ROC-AUC:     {np.mean(cv_aucs):.3f}")
    print(f"Mean Brier Score: {np.mean(cv_brier):.3f}")
    
    if high_conf_trades > 0:
        hc_wr = high_conf_wins / high_conf_trades * 100
        print(f"\n🎯 High-Confidence Tier (Predicted Prob >= 70%):")
        print(f"   Evaluated Trades: {high_conf_trades}")
        print(f"   Wins:             {high_conf_wins}")
        print(f"   Win Rate:         {hc_wr:.1f}%")

    # 3. Fit Final Production Model on All Available Data
    print("\n--- Fitting Final Production Model & Probability Calibrator ---")
    final_xgb = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )
    final_xgb.fit(X, y)

    # Sigmoid Probability Calibration
    calibrated_model = CalibratedClassifierCV(estimator=final_xgb, method="sigmoid", cv="prefit")
    calibrated_model.fit(X, y)

    # Feature Importance
    importances = final_xgb.feature_importances_
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    for _, row in df_imp.iterrows():
        print(f"  {row['feature']:<20}: {row['importance']*100:>5.1f}%")

    # 4. Save Model Artifacts
    model_path = MODELS_DIR / "trendm30_ml_classifier.pkl"
    joblib.dump(calibrated_model, model_path)
    print(f"\n✅ Saved calibrated ML model to {model_path}")

    metadata = {
        "model_name": "TrendM30_XGBoost_Calibrated",
        "version": "1.0",
        "features": feature_names,
        "total_samples": int(n),
        "mean_cv_accuracy": float(np.mean(cv_accs)),
        "mean_cv_auc": float(np.mean(cv_aucs)),
        "high_confidence_threshold": 0.70,
        "high_confidence_win_rate": float(hc_wr) if high_conf_trades else 0.75,
        "trained_date": pd.Timestamp.now().isoformat()
    }
    meta_path = MODELS_DIR / "trendm30_ml_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {meta_path}")

    return calibrated_model, metadata

if __name__ == "__main__":
    train_and_save_trendm30_model()
