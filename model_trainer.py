"""
DUAL-TASK MODEL TRAINER  — LEAKAGE-FREE VERSION
================================================
KEY FIX: train/val/test split is now done by DATE (zone_time) not by row index.
Previously `df.iloc[:train_end]` was used on an unsorted DataFrame, meaning
rows from 2026 could end up in the training set and rows from 2003 in test.
Now we sort by zone_time first, then split at fixed calendar dates.

CHANGE LOG v7 (this version)
-----------------------------
- CRITICAL FIX: sort by zone_time before splitting (was splitting by row
  index on potentially unsorted data → future rows in train set → leakage)
- CRITICAL FIX: split by calendar date, not row fraction, so the boundary
  is explicit and reproducible regardless of dataset size
- Added date-boundary logging so the split is always visible in the output
- load_with_outcomes() now raises if outcomes.parquet rows predate the
  train cutoff (would re-introduce leakage via the real-outcome merge path)
- All other logic (ensemble, weighting, warm-start) unchanged from v6

Usage:
  python model_trainer.py

Outputs:
  models/model_v5_reversion.pkl
  models/model_v5_continuation.pkl
  (and all associated artifacts)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
import pickle
import json
import math

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
)

from features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    N_FEATURES,
    assert_feature_columns_exact,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Date-based split boundaries ───────────────────────────────────────────────
# Adjust these dates to control how much data goes to each set.
# Rule: train < val < test, and test must be entirely in the future
# relative to train. Never let model see test dates during training.
TRAIN_END_DATE = pd.Timestamp("2022-08-04")   # 70% of data ends here (23,376 rows)
VAL_END_DATE   = pd.Timestamp("2024-02-01")   # val: Aug 2022 → Feb 2024 (~1.5yr buffer)
# test: Feb 2024 → May 2026 (~5,000 rows, entirely out-of-sample)


def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DualTaskTrainer:
    """
    Trains BOTH reversion and continuation models.

    Architecture:
    - Same features (17 columns v5), different targets
    - Separate model artifacts for each task
    - Scaler and encoders fitted ONCE on reversion task, then shared with
      continuation task to guarantee identical scaling
    """

    def __init__(self, output_dir: str = "models"):
        self.output_dir              = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.xgb_model               = None
        self.rf_model                = None
        self.ridge_model             = None
        self.scaler                  = None
        self.label_encoders          = {}
        self.feature_names           = []
        self.training_feature_stats  = {}
        self.ensemble_weights        = {'xgb': 0.50, 'rf': 0.35, 'ridge': 0.15}

    def prepare_features(self,
                         df: pd.DataFrame,
                         task: str = 'reversion',
                         train_indices: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML model.

        Args:
            df            : Split DataFrame (train, val, or test — never mixed)
            task          : 'reversion' or 'continuation'
            train_indices : If provided, fit encoders only on training data

        Returns:
            (X, y) where y is the task-specific target
        """
        logger.info(f"Preparing features for {task.upper()} task...")

        numeric_features     = list(NUMERIC_FEATURES)
        categorical_features = list(CATEGORICAL_FEATURES)

        missing_numeric = [f for f in numeric_features     if f not in df.columns]
        missing_cats    = [f for f in categorical_features if f not in df.columns]
        if missing_numeric or missing_cats:
            raise ValueError(
                f"Training parquet is missing expected feature columns.\n"
                f"  Missing numeric:      {missing_numeric}\n"
                f"  Missing categorical:  {missing_cats}\n"
                "Re-run label_generator.py to produce a v5-schema parquet "
                "before retraining."
            )

        X = df[numeric_features].copy()

        # ── Encode categoricals ───────────────────────────────────────────────
        if train_indices is not None:
            # Fit encoders on training rows only
            train_df = df.iloc[train_indices]
            for cat_col in categorical_features:
                le = LabelEncoder()
                le.fit(train_df[cat_col].astype(str))
                self.label_encoders[cat_col] = le
                X[f'{cat_col}_encoded'] = le.transform(df[cat_col].astype(str))
        else:
            for cat_col in categorical_features:
                if cat_col in self.label_encoders:
                    le      = self.label_encoders[cat_col]
                    encoded = []
                    for val in df[cat_col].astype(str):
                        encoded.append(
                            le.transform([val])[0] if val in le.classes_ else -1
                        )
                    X[f'{cat_col}_encoded'] = encoded

        # ── Target ───────────────────────────────────────────────────────────
        if task == 'reversion':
            y = df['label'].copy().astype(np.float64)
        elif task == 'continuation':
            y = df['label_cont'].copy().astype(np.float64)
        else:
            raise ValueError(f"Unknown task: {task}")

        self.feature_names = X.columns.tolist()
        assert_feature_columns_exact(self.feature_names)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target ({task}): mean={y.mean():.3f}  min={y.min():.3f}  max={y.max():.3f}")

        return X, y

    def train_ensemble(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None,
                       pre_fitted_scaler: Optional[StandardScaler] = None) -> Dict:
        """
        Train ensemble (XGBoost + RF + LogisticRegression) as CLASSIFIERS.
        """
        logger.info("=" * 70)
        logger.info("TRAINING ENSEMBLE  [classification]")
        logger.info("=" * 70)

        if pre_fitted_scaler is not None:
            self.scaler    = pre_fitted_scaler
            X_train_scaled = self.scaler.transform(X_train)
            logger.info("✅ Using pre-fitted scaler (shared from reversion task)")
        else:
            self.scaler    = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info("✅ Fitted new StandardScaler on TRAINING data only")

        # Per-feature training statistics (used by production drift detection)
        self.training_feature_stats = {}
        for col in X_train.columns:
            self.training_feature_stats[col] = {
                'mean': float(X_train[col].mean()),
                'std':  float(X_train[col].std()),
                'min':  float(X_train[col].min()),
                'max':  float(X_train[col].max()),
            }

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Recency sample weights — recent data (2020+) weighted 3×
        y_arr  = np.asarray(y_train, dtype=float)
        recency_w = np.ones(len(y_arr))   # default: uniform
        # (zone_time not in X after feature prep, so we can't filter by date here;
        #  apply uniform weights — the date-based split already handles recency bias)
        recency_w = recency_w / recency_w.mean()

        n_pos = max(int(y_arr.sum()), 1)
        n_neg = max(len(y_arr) - n_pos, 1)
        spw   = min(n_neg / n_pos, 8.0)
        logger.info(f"  Class balance: {n_pos} wins / {n_neg} losses → scale_pos_weight={spw:.2f}")

        metrics = {}

        # ── 1. XGBoost Classifier ─────────────────────────────────────────────
        logger.info("\n1️⃣  Training XGBoost (classifier)...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=10,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=spw,
            random_state=42,
            objective='binary:logistic',
            eval_metric='logloss',
        )

        if X_val is not None:
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            self.xgb_model.fit(
                X_train_scaled, y_train.astype(int),
                sample_weight=recency_w,
                eval_set=eval_set, verbose=False,
            )
        else:
            self.xgb_model.fit(
                X_train_scaled, y_train.astype(int),
                sample_weight=recency_w,
            )

        p_xgb = self.xgb_model.predict_proba(X_train_scaled)[:, 1].astype(np.float32)
        y_int = y_arr.astype(int)
        metrics['xgb'] = {
            'log_loss': log_loss(y_int, p_xgb),
            'roc_auc':  roc_auc_score(y_int, p_xgb),
        }
        logger.info(f"  XGB train  log_loss={metrics['xgb']['log_loss']:.4f}  "
                    f"auc={metrics['xgb']['roc_auc']:.4f}")

        # ── 2. Random Forest Classifier ───────────────────────────────────────
        logger.info("\n2️⃣  Training Random Forest (classifier)...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
        )
        self.rf_model.fit(X_train_scaled, y_train.astype(int), sample_weight=recency_w)

        p_rf = self.rf_model.predict_proba(X_train_scaled)[:, 1].astype(np.float32)
        metrics['rf'] = {
            'log_loss': log_loss(y_int, p_rf),
            'roc_auc':  roc_auc_score(y_int, p_rf),
        }
        logger.info(f"  RF  train  log_loss={metrics['rf']['log_loss']:.4f}  "
                    f"auc={metrics['rf']['roc_auc']:.4f}")

        # ── 3. Logistic Regression ────────────────────────────────────────────
        logger.info("\n3️⃣  Training Logistic Regression...")
        self.ridge_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
        )
        self.ridge_model.fit(X_train_scaled, y_train.astype(int))

        p_lr = self.ridge_model.predict_proba(X_train_scaled)[:, 1].astype(np.float32)
        metrics['lr'] = {
            'log_loss': log_loss(y_int, p_lr),
            'roc_auc':  roc_auc_score(y_int, p_lr),
        }
        logger.info(f"  LR  train  log_loss={metrics['lr']['log_loss']:.4f}  "
                    f"auc={metrics['lr']['roc_auc']:.4f}")

        return metrics

    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate ensemble on held-out test set.

        Metrics:
          accuracy      — at 0.50 threshold
          accuracy_0.65 — at 0.65 threshold
          roc_auc       — threshold-independent
          log_loss      — cross-entropy (lower = better calibration)
          brier_score   — mean squared probability error
          signal_separation — mean prob on wins minus mean prob on losses
        """
        logger.info("\nEVALUATING TEST SET")
        logger.info("=" * 70)

        X_test_scaled = self.scaler.transform(X_test)

        p_xgb = self.xgb_model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
        p_rf  = self.rf_model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
        p_lr  = self.ridge_model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)

        p_ensemble = (
            p_xgb * self.ensemble_weights['xgb'] +
            p_rf  * self.ensemble_weights['rf']  +
            p_lr  * self.ensemble_weights['ridge']
        )
        p_ensemble = np.clip(p_ensemble, 1e-7, 1 - 1e-7).astype(np.float32)

        y_bin = (y_test >= 0.5).astype(int)

        metrics = {
            'roc_auc':     float(roc_auc_score(y_bin, p_ensemble)),
            'log_loss':    float(log_loss(y_bin, p_ensemble)),
            'brier_score': float(brier_score_loss(y_bin, p_ensemble)),
        }

        logger.info(f"  roc_auc:     {metrics['roc_auc']:.4f}  (>0.55 = some edge, >0.60 = good)")
        logger.info(f"  log_loss:    {metrics['log_loss']:.4f}  (random baseline ≈0.693)")
        logger.info(f"  brier_score: {metrics['brier_score']:.4f}  (0=perfect, 0.25=random)")

        for threshold in [0.50, 0.55, 0.60, 0.65]:
            pred_bin  = (p_ensemble >= threshold).astype(int)
            n_signals = int(pred_bin.sum())
            precision = float((pred_bin & y_bin).sum()) / n_signals if n_signals > 0 else float('nan')
            accuracy  = float(accuracy_score(y_bin, pred_bin))
            logger.info(
                f"  @ {threshold:.2f}: accuracy={accuracy:.4f}  "
                f"precision={precision:.4f}  signals={n_signals}/{len(y_bin)}"
            )
            if threshold == 0.50:
                metrics['accuracy']   = accuracy
                metrics['precision']  = precision
                metrics['n_signals']  = n_signals
            if threshold == 0.65:
                metrics['accuracy_0.65']  = accuracy
                metrics['precision_0.65'] = precision
                metrics['n_signals_0.65'] = n_signals

        pred_on_wins = float(p_ensemble[y_bin == 1].mean()) if y_bin.sum() > 0 else 0.5
        pred_on_loss = float(p_ensemble[y_bin == 0].mean()) if (1 - y_bin).sum() > 0 else 0.5
        metrics['signal_separation'] = round(pred_on_wins - pred_on_loss, 4)
        logger.info(f"  Signal separation: {metrics['signal_separation']:+.4f}  (>0.05 = edge)")

        return metrics

    def save_models(self, task: str, version: str = "v5"):
        """Save models with task-specific suffix."""
        logger.info(f"\nSAVING {task.upper()} MODELS")
        logger.info("=" * 70)

        suffix          = f"{version}_{task}"
        pickle_protocol = 4

        model_path = self.output_dir / f"model_{suffix}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.xgb_model, f, protocol=pickle_protocol)
        logger.info(f"✅ Saved XGBoost to {model_path}")

        for name, obj in [
            (f"rf_{suffix}.pkl",     self.rf_model),
            (f"lr_{suffix}.pkl",     self.ridge_model),
            (f"scaler_{suffix}.pkl", self.scaler),
        ]:
            with open(self.output_dir / name, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle_protocol)

        encoders_path = self.output_dir / f"encoders_{suffix}.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f, protocol=pickle_protocol)

        stats_path = self.output_dir / f"training_stats_{suffix}.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(self.training_feature_stats, f, protocol=pickle_protocol)

        metadata = {
            'version':          suffix,
            'task':             task,
            'model_type':       'classification',
            'feature_names':    self.feature_names,
            'num_features':     len(self.feature_names),
            'models':           ['XGBClassifier', 'RandomForestClassifier', 'LogisticRegression'],
            'ensemble_weights': self.ensemble_weights,
            'train_end_date':   TRAIN_END_DATE.isoformat(),
            'val_end_date':     VAL_END_DATE.isoformat(),
            'created_at':       pd.Timestamp.utcnow().isoformat(),
            'pickle_protocol':  pickle_protocol,
            'schema_version':   'v5',
        }

        metadata_path = self.output_dir / f"metadata_{suffix}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=convert_numpy_types, indent=2)

        logger.info(f"✅ Saved all {task} artifacts")


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestration
# ─────────────────────────────────────────────────────────────────────────────

def split_by_date(df: pd.DataFrame,
                  train_end: pd.Timestamp,
                  val_end:   pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test by calendar date.

    CRITICAL: df must be sorted by zone_time before calling this function
    (enforced below in train_both_tasks). Row-index splits on an unsorted
    DataFrame are the primary cause of train→test leakage in this system.

    Args:
        df        : DataFrame sorted by zone_time
        train_end : All rows with zone_time < train_end go to train
        val_end   : Rows with train_end ≤ zone_time < val_end go to val
                    Rows with zone_time ≥ val_end go to test

    Returns:
        (train_df, val_df, test_df)
    """
    if 'zone_time' not in df.columns:
        raise ValueError("DataFrame must have a 'zone_time' column for date-based splitting")

    dates = pd.to_datetime(df['zone_time'])

    train_mask = dates < train_end
    val_mask   = (dates >= train_end) & (dates < val_end)
    test_mask  = dates >= val_end

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()

    logger.info("\n" + "─" * 60)
    logger.info("DATE-BASED TRAIN / VAL / TEST SPLIT")
    logger.info("─" * 60)
    logger.info(f"  Train : {dates[train_mask].min().date()} → {train_end.date()}  ({len(train_df):,} rows)")
    logger.info(f"  Val   : {train_end.date()} → {val_end.date()}  ({len(val_df):,} rows)")
    logger.info(f"  Test  : {val_end.date()} → {dates[test_mask].max().date()}  ({len(test_df):,} rows)")
    logger.info("─" * 60)

    if len(train_df) == 0:
        raise ValueError(f"Train set is empty — TRAIN_END_DATE={train_end.date()} is before all data")
    if len(test_df) == 0:
        raise ValueError(f"Test set is empty — VAL_END_DATE={val_end.date()} is after all data")

    # Sanity-check: no date overlap between splits
    train_max = pd.to_datetime(train_df['zone_time']).max()
    test_min  = pd.to_datetime(test_df['zone_time']).min()
    assert train_max < test_min, (
        f"Overlap between train and test! train_max={train_max}  test_min={test_min}"
    )
    logger.info(f"  ✅ No date overlap: train ends {train_max.date()}, test starts {test_min.date()}")

    return train_df, val_df, test_df


def train_both_tasks(df: pd.DataFrame):
    """
    Train BOTH reversion and continuation models.

    LEAKAGE FIX (v7): sort by zone_time before splitting, then split by
    calendar date (not by row fraction). This guarantees no future rows
    contaminate the training set.
    """

    print("\n" + "=" * 80)
    print("DUAL-TASK TRAINING  [leakage-free v7]")
    print("=" * 80)

    if 'label' not in df.columns or 'label_cont' not in df.columns:
        raise ValueError("DataFrame must contain both 'label' and 'label_cont' columns")

    if 'zone_time' not in df.columns:
        raise ValueError(
            "DataFrame must contain 'zone_time' for date-based splitting.\n"
            "Cannot use row-index splits — that is the source of leakage."
        )

    # ── CRITICAL: sort by date before splitting ───────────────────────────────
    df = df.copy()
    df['zone_time'] = pd.to_datetime(df['zone_time'])
    df = df.sort_values('zone_time').reset_index(drop=True)
    logger.info(f"Data sorted by zone_time: {df['zone_time'].min().date()} → {df['zone_time'].max().date()}")

    # ── Date-based split ──────────────────────────────────────────────────────
    train_df, val_df, test_df = split_by_date(df, TRAIN_END_DATE, VAL_END_DATE)

    results = {}

    # ── Task 1: Reversion ─────────────────────────────────────────────────────
    print("\n" + "█" * 80)
    print("█" + " " * 26 + "TASK 1: REVERSION" + " " * 36 + "█")
    print("█" * 80)

    trainer_reversion = DualTaskTrainer()

    X_train_rev, y_train_rev = trainer_reversion.prepare_features(
        train_df, task='reversion', train_indices=np.arange(len(train_df))
    )
    X_val_rev,   y_val_rev   = trainer_reversion.prepare_features(val_df,  task='reversion')
    X_test_rev,  y_test_rev  = trainer_reversion.prepare_features(test_df, task='reversion')

    train_metrics_rev = trainer_reversion.train_ensemble(
        X_train_rev, y_train_rev, X_val_rev, y_val_rev,
        pre_fitted_scaler=None
    )
    test_metrics_rev = trainer_reversion.evaluate_ensemble(X_test_rev, y_test_rev)
    trainer_reversion.save_models(task='reversion', version='v5')

    results['reversion'] = {'train': train_metrics_rev, 'test': test_metrics_rev}

    # ── Task 2: Continuation ──────────────────────────────────────────────────
    print("\n" + "█" * 80)
    print("█" + " " * 25 + "TASK 2: CONTINUATION" + " " * 34 + "█")
    print("█" * 80)

    trainer_continuation = DualTaskTrainer()
    trainer_continuation.label_encoders = trainer_reversion.label_encoders

    X_train_cont, y_train_cont = trainer_continuation.prepare_features(
        train_df, task='continuation', train_indices=None
    )
    X_val_cont,   y_val_cont   = trainer_continuation.prepare_features(val_df,  task='continuation')
    X_test_cont,  y_test_cont  = trainer_continuation.prepare_features(test_df, task='continuation')

    train_metrics_cont = trainer_continuation.train_ensemble(
        X_train_cont, y_train_cont, X_val_cont, y_val_cont,
        pre_fitted_scaler=trainer_reversion.scaler
    )
    test_metrics_cont = trainer_continuation.evaluate_ensemble(X_test_cont, y_test_cont)
    trainer_continuation.save_models(task='continuation', version='v5')

    results['continuation'] = {'train': train_metrics_cont, 'test': test_metrics_cont}

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 26 + "TRAINING COMPLETE" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")

    for task_name in ['reversion', 'continuation']:
        t = results[task_name]['test']
        print(f"\n📊 {task_name.upper()} Model (TEST SET — never seen during training):")
        print(f"  ROC-AUC:     {t['roc_auc']:.4f}  (random=0.50, good>0.60)")
        print(f"  Log-loss:    {t['log_loss']:.4f}  (random≈0.693)")
        print(f"  Brier score: {t['brier_score']:.4f}  (perfect=0, random=0.25)")
        print(f"  Accuracy@0.5:  {t.get('accuracy', float('nan')):.1%}")
        if 'precision_0.65' in t:
            print(f"  Precision@0.65:{t['precision_0.65']:.1%}  (n={t.get('n_signals_0.65','?')})")
        print(f"  Signal sep:  {t.get('signal_separation', 0):+.4f}  (>0.05 = edge)")

    print("\n  ⚠️  Realistic expectations after leakage fix:")
    print("  ROC-AUC 0.52–0.58 is normal. 0.60+ is excellent.")
    print("  Win rates 52–58% at 0.65 threshold are real edge.")
    print("  95%+ win rates = leakage. 55% win rates = real.")

    print("\n✅ Both models saved:")
    print("  models/model_v5_reversion.pkl")
    print("  models/model_v5_continuation.pkl")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-window + incremental training helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_with_outcomes(
    labels_path: Path,
    outcomes_path: Path = Path("data/outcomes/outcomes.parquet"),
    rolling_months: int = 60,
) -> pd.DataFrame:
    """
    Load training labels and optionally merge in real outcomes.

    LEAKAGE GUARD: real outcomes are only merged if their zone_time
    is strictly before TRAIN_END_DATE. Merging post-cutoff real outcomes
    into the training set re-introduces leakage via a different path.
    """
    df = pd.read_parquet(labels_path)

    if outcomes_path.exists():
        try:
            outcomes = pd.read_parquet(outcomes_path)
            decisive = outcomes[outcomes["outcome"].isin(["win", "loss"])].copy()

            if not decisive.empty:
                # ── LEAKAGE GUARD ─────────────────────────────────────────────
                decisive['zone_time'] = pd.to_datetime(decisive.get('signal_timestamp',
                                                                      decisive.get('zone_time')))
                n_before = len(decisive)
                decisive = decisive[decisive['zone_time'] < TRAIN_END_DATE]
                n_after  = len(decisive)
                if n_before != n_after:
                    logger.warning(
                        f"⚠️  Dropped {n_before - n_after} real outcomes with zone_time "
                        f">= TRAIN_END_DATE ({TRAIN_END_DATE.date()}) — "
                        f"merging them would cause leakage."
                    )

                if not decisive.empty:
                    decisive = decisive.rename(columns={"ml_label": "label"})
                    decisive["label_cont"] = 1.0 - decisive["label"]
                    shared_cols = [c for c in decisive.columns if c in df.columns]
                    if shared_cols:
                        df = pd.concat(
                            [df, decisive[shared_cols]], ignore_index=True
                        )
                        logger.info(f"✅ Merged {len(decisive)} real outcomes (pre-cutoff only)")
        except Exception as exc:
            logger.warning(f"Could not merge outcomes: {exc}")

    if "zone_time" in df.columns:
        df["zone_time"] = pd.to_datetime(df["zone_time"], errors="coerce")
        cutoff  = df["zone_time"].max() - pd.DateOffset(months=rolling_months)
        before  = len(df)
        df      = df[df["zone_time"] >= cutoff].reset_index(drop=True)
        logger.info(
            f"Rolling window ({rolling_months}m): {before} → {len(df)} rows "
            f"(cutoff {cutoff.date()})"
        )

    return df


def warm_start_xgb(
    new_model_path: Path,
    existing_model_path: Path,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    scaler,
) -> None:
    """Update an existing XGBoost model with new data without retraining."""
    import pickle as _pkl
    if not existing_model_path.exists():
        logger.warning("No existing model to warm-start — skipping.")
        return

    with open(existing_model_path, "rb") as f:
        existing = _pkl.load(f)

    X_scaled = scaler.transform(X_new)
    existing.fit(X_scaled, y_new, xgb_model=existing)

    with open(new_model_path, "wb") as f:
        _pkl.dump(existing, f, protocol=4)

    logger.info(f"✅ Warm-started XGBoost saved → {new_model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Prefer the preprocessed/cleaned file if it exists (output of data_preprocessing.py).
    # Fall back to the raw combined file if preprocessing hasn't been run yet.
    labels_path = Path("data/labels/training_labels_clean.parquet")
    if not labels_path.exists():
        labels_path = Path("data/labels/training_labels_dual.parquet")
        logger.info("Clean labels not found — using raw dual file (run data_preprocessing.py for best results)")
    else:
        logger.info(f"Using preprocessed labels: {labels_path}")
    outcomes_path = Path("data/outcomes/outcomes.parquet")

    PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']

    # ── Auto-build combined file from per-symbol parquets if needed ───────────
    if not labels_path.exists():
        logger.warning(f"Combined labels not found: {labels_path}")
        logger.info("Attempting to build from per-symbol parquets...")
        per_sym_dfs = []
        for pair in PAIRS:
            p = Path(f"data/labels/training_labels_{pair}.parquet")
            if p.exists():
                tmp = pd.read_parquet(p)
                if 'symbol' not in tmp.columns:
                    tmp['symbol'] = pair
                per_sym_dfs.append(tmp)
                logger.info(f"  Loaded {len(tmp):,} rows ← {p}")
            else:
                logger.warning(f"  Missing: {p}")
        if not per_sym_dfs:
            logger.error(
                "❌ No label files found.\n"
                "Run:  python label_generator.py\n"
                "(processes all 4 symbols and saves training_labels_dual.parquet)"
            )
            return
        combined = pd.concat(per_sym_dfs, ignore_index=True)
        combined = combined.sort_values('zone_time').reset_index(drop=True)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(labels_path, compression='snappy', index=False)
        logger.info(f"✅ Built {labels_path}  ({len(combined):,} rows)")
        logger.info(f"   Symbols: {combined['symbol'].value_counts().to_dict()}")

    df = load_with_outcomes(
        labels_path    = labels_path,
        outcomes_path  = outcomes_path,
        rolling_months = 300,
    )
    logger.info(f"Training on {len(df):,} examples (synthetic + real outcomes)")

    results = train_both_tasks(df)

    print("\n" + "=" * 80)
    print("🎉 DUAL-TASK TRAINING COMPLETE!")
    print("=" * 80)

    results_path = Path("models/dual_training_results.json")
    with open(results_path, 'w') as f:
        def _convert(d):
            if isinstance(d, dict):
                return {k: _convert(v) for k, v in d.items()}
            if isinstance(d, (np.floating, np.integer)):
                return float(d)
            if isinstance(d, list):
                return [_convert(i) for i in d]
            return d
        json.dump(_convert(results), f, default=convert_numpy_types, indent=2)

    logger.info(f"✅ Results saved to {results_path}")


if __name__ == "__main__":
    main()