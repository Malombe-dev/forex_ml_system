"""
DUAL-TASK MODEL TRAINER
Trains BOTH reversion and continuation models in one run.

CHANGE LOG v6 (current)
-----------------------
- Issue 10 FIX: Switched from regression to CLASSIFICATION throughout.
  Labels are binary (0/1), so training regressors with MSE loss caused
  negative R² — the regressor was penalised for not predicting exactly 0 or 1.
  Changes:
    · XGBRegressor → XGBClassifier (objective='binary:logistic')
    · RandomForestRegressor → RandomForestClassifier
    · Ridge → LogisticRegression (replaces the linear regressor)
    · predict() → predict_proba()[:,1] everywhere for probability outputs
    · Removed r2_score / MSE / MAE from training metrics (meaningless for
      classifiers); replaced with log_loss + roc_auc on train set
    · evaluate_ensemble() now reports: accuracy, ROC-AUC, log-loss,
      Brier score, and signal separation — all meaningful for binary tasks
    · scale_pos_weight replaces the manual win_boost / decisiveness weights
      for XGBoost; RF uses class_weight='balanced'; LogReg uses class_weight='balanced'
    · Recency weighting retained (recent years 3× historical) — still valid
    · schema_version corrected to 'v5' (was 'v3', model is 17-feature v5)
    · metadata model_type corrected to 'classification'
    · Output filenames bumped to v5_reversion / v5_continuation

CHANGE LOG v5
-------------
- Issue 7 FIX: prepare_features() now imports NUMERIC_FEATURES and
  CATEGORICAL_FEATURES directly from features.py (single source of truth)
  instead of hardcoding the lists. Any drift between features.py and
  model_trainer.py will now surface as an ImportError at startup.

- Issue 8 FIX: The silent `if f in df.columns` guard has been replaced with
  a hard ValueError that fires immediately if the training parquet is missing
  any expected feature column. Previously a missing column (e.g. from an
  old v2 parquet) would silently train a 12- or 13-feature model, save it
  with schema_version='v3', and corrupt all downstream predictions.

- Issue 9 FIX: assert_feature_columns_exact() is called after encoding to
  verify the final column order matches FEATURE_COLUMNS exactly before
  fitting the scaler. Catches any future column-order drift.

CHANGE LOG v4
-------------
- Issue 5 FIX: Scaler and label encoders are now fitted ONCE on the reversion
  task and shared with the continuation trainer via explicit assignment.
  Previously, each DualTaskTrainer() instance called scaler.fit_transform()
  independently. They happened to produce identical scalers because both tasks
  use the same feature matrix — but this was an accidental property, not
  enforced. If feature sets ever diverge per task the old code would silently
  produce different scalers. The new code makes the sharing explicit and
  documented. train_ensemble() accepts a pre_fitted_scaler kwarg; if supplied
  it calls transform() instead of fit_transform().

- Issue 6 FIX: schema_version in save_models() corrected from 'v2' to 'v3'.
  The models were trained on the 14-feature v3 schema (zone_touch_count and
  htf_trend_alignment added in v3) but the metadata was still stamped 'v2'.
  This would cause any code that gates on schema_version to load wrong.

CHANGE LOG v2
-------------
- Updated feature lists to match features.py v2 schema (12 features).
- ema_trend moved from categorical_features → numeric_features.
- atr_regime removed; replaced by approach_atr_regime.
- All other logic (ensemble, weighting, rolling window, warm-start) unchanged.

Usage:
  python model_trainer.py

Outputs:
  models/model_v4_reversion.pkl
  models/model_v4_continuation.pkl
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


def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DualTaskTrainer:
    """
    Trains BOTH reversion and continuation models.

    Architecture:
    - Same features (14 columns), different targets
    - Separate model artifacts for each task
    - Scaler and encoders fitted ONCE on reversion task, then shared with
      continuation task to guarantee identical scaling (Issue 5 fix)
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

        Feature schema (must match features.py v3 — N_FEATURES = 14):

        Numeric (11) — no encoding:
            zone_strength, zone_confluence, risk_pips,
            distance_from_ema20, rsi, hour, ema_trend,
            approach_rsi, approach_ema_dist,
            zone_touch_count, htf_trend_alignment

        Categorical (3) — LabelEncoder applied:
            zone_type, session, approach_atr_regime

        Args:
            df            : Full DataFrame with BOTH label columns
            task          : 'reversion' or 'continuation'
            train_indices : If provided, fit encoders only on training data

        Returns:
            (X, y) where y is the task-specific target
        """
        logger.info(f"Preparing features for {task.upper()} task...")

        # ── Canonical feature lists — single source of truth is features.py ──
        # Hard assertion: if the parquet is missing any expected column the
        # trainer raises immediately rather than silently training on fewer
        # features and saving a broken model with schema_version='v3'.
        numeric_features     = list(NUMERIC_FEATURES)
        categorical_features = list(CATEGORICAL_FEATURES)

        missing_numeric = [f for f in numeric_features     if f not in df.columns]
        missing_cats    = [f for f in categorical_features if f not in df.columns]
        if missing_numeric or missing_cats:
            raise ValueError(
                f"Training parquet is missing expected feature columns.\n"
                f"  Missing numeric:      {missing_numeric}\n"
                f"  Missing categorical:  {missing_cats}\n"
                "Re-run label_generator.py to produce a v3-schema parquet "
                "before retraining."
            )

        X = df[numeric_features].copy()

        # ── Encode categoricals ───────────────────────────────────────────────
        if train_indices is not None:
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

        # Final sanity-check: column order must match FEATURE_COLUMNS exactly.
        # Catches any future drift between prepare_features() and features.py.
        assert_feature_columns_exact(self.feature_names)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target ({task}) statistics:")
        logger.info(f"  Min: {y.min():.3f}")
        logger.info(f"  Mean: {y.mean():.3f}")
        logger.info(f"  Max: {y.max():.3f}")

        return X, y

    def train_ensemble(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None,
                       pre_fitted_scaler: Optional[StandardScaler] = None) -> Dict:
        """
        Train ensemble (XGBoost + RF + LogisticRegression) as CLASSIFIERS.

        Labels are binary (0/1).  All models output probabilities via
        predict_proba()[:,1].  Ensemble = weighted average of probabilities.

        Parameters
        ----------
        pre_fitted_scaler : If provided, use this already-fitted scaler instead
                            of fitting a new one. Shared from the reversion task
                            to guarantee identical scaling. (Issue 5 fix)
        """
        logger.info("=" * 70)
        logger.info("TRAINING ENSEMBLE  [classification]")
        logger.info("=" * 70)

        # ── Share scaler across tasks (Issue 5 fix) ───────────────────────────
        if pre_fitted_scaler is not None:
            self.scaler    = pre_fitted_scaler
            X_train_scaled = self.scaler.transform(X_train)
            logger.info("✅ Using pre-fitted scaler (shared from reversion task)")
        else:
            self.scaler    = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info("✅ Fitted new StandardScaler on training data")

        # Per-feature training statistics (used by production drift detection)
        self.training_feature_stats = {}
        for col in X_train.columns:
            self.training_feature_stats[col] = {
                'mean': float(X_train[col].mean()),
                'std':  float(X_train[col].std()),
                'min':  float(X_train[col].min()),
                'max':  float(X_train[col].max()),
            }
        logger.info(f"✅ Stored training stats for {len(self.training_feature_stats)} features")

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # ── Recency sample weights (still valid for classifiers) ──────────────
        # Recent data (2020+) weighted 3× over historical — no win/loss boost
        # needed because XGBoost handles class imbalance via scale_pos_weight,
        # RF via class_weight='balanced', LogReg via class_weight='balanced'.
        y_arr  = np.asarray(y_train, dtype=float)
        if hasattr(X_train, 'index') and 'zone_time' in getattr(X_train, 'columns', []):
            times     = pd.to_datetime(X_train['zone_time'])
            cutoff    = pd.Timestamp('2020-01-01')
            recency_w = np.where(times >= cutoff, 3.0, 1.0)
        else:
            recency_w = np.ones(len(y_arr))
        recency_w = recency_w / recency_w.mean()  # normalise so mean=1

        # scale_pos_weight for XGBoost: ratio of negatives to positives
        n_pos = max(int(y_arr.sum()), 1)
        n_neg = max(len(y_arr) - n_pos, 1)
        spw   = min(n_neg / n_pos, 8.0)  # cap at 8× to avoid over-correction
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
            scale_pos_weight=spw,       # handles class imbalance natively
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
            class_weight='balanced',    # handles class imbalance natively
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

        # ── 3. Logistic Regression (replaces Ridge) ───────────────────────────
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

        # ── 4. Ensemble (probability average) ────────────────────────────────
        logger.info("\n4️⃣  Evaluating Ensemble...")
        p_ensemble = (
            p_xgb * self.ensemble_weights['xgb'] +
            p_rf  * self.ensemble_weights['rf']  +
            p_lr  * self.ensemble_weights['ridge']   # key kept as 'ridge' for artifact compat
        )
        p_ensemble = np.clip(p_ensemble, 1e-7, 1 - 1e-7).astype(np.float32)
        metrics['ensemble'] = {
            'log_loss': log_loss(y_int, p_ensemble),
            'roc_auc':  roc_auc_score(y_int, p_ensemble),
        }
        logger.info(f"  ENS train  log_loss={metrics['ensemble']['log_loss']:.4f}  "
                    f"auc={metrics['ensemble']['roc_auc']:.4f}")

        return metrics

    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate ensemble on test set.

        Metrics reported (all meaningful for binary classification):
          accuracy      — at 0.50 probability threshold
          accuracy_0.65 — at 0.65 threshold (higher-confidence signals only)
          roc_auc       — area under ROC curve (threshold-independent)
          log_loss      — cross-entropy loss (lower = better calibration)
          brier_score   — mean squared probability error (0=perfect, 0.25=random)
          signal_separation — mean prob on wins minus mean prob on losses
          n_signals_*   — count of predictions above each threshold
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
            'roc_auc':    float(roc_auc_score(y_bin, p_ensemble)),
            'log_loss':   float(log_loss(y_bin, p_ensemble)),
            'brier_score': float(brier_score_loss(y_bin, p_ensemble)),
        }

        logger.info("Test metrics:")
        logger.info(f"  roc_auc:     {metrics['roc_auc']:.4f}  (>0.55 = some edge, >0.60 = good)")
        logger.info(f"  log_loss:    {metrics['log_loss']:.4f}  (lower = better; random baseline ≈0.693)")
        logger.info(f"  brier_score: {metrics['brier_score']:.4f}  (0=perfect, 0.25=random)")

        # Accuracy + signal count at multiple thresholds
        for threshold in [0.50, 0.55, 0.60, 0.65]:
            pred_bin  = (p_ensemble >= threshold).astype(int)
            n_signals = int(pred_bin.sum())
            # Among signals fired, what fraction were correct?
            if n_signals > 0:
                precision = float((pred_bin & y_bin).sum()) / n_signals
            else:
                precision = float('nan')
            accuracy = float(accuracy_score(y_bin, pred_bin))
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

        # Signal separation
        pred_on_wins = float(p_ensemble[y_bin == 1].mean()) if y_bin.sum() > 0 else 0.5
        pred_on_loss = float(p_ensemble[y_bin == 0].mean()) if (1 - y_bin).sum() > 0 else 0.5
        metrics['signal_separation'] = round(pred_on_wins - pred_on_loss, 4)
        logger.info(
            f"  Signal separation (wins - losses): "
            f"{metrics['signal_separation']:+.4f}  (>0.05 = edge)"
        )

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
            (f"lr_{suffix}.pkl",     self.ridge_model),   # LogisticRegression
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
        logger.info(
            f"✅ Saved training stats ({len(self.training_feature_stats)} features) "
            f"to {stats_path}"
        )

        metadata = {
            'version':          suffix,
            'task':             task,
            'model_type':       'classification',           # Issue 10 FIX: was 'regression'
            'feature_names':    self.feature_names,
            'num_features':     len(self.feature_names),
            'models':           ['XGBClassifier', 'RandomForestClassifier', 'LogisticRegression'],
            'ensemble_weights': self.ensemble_weights,
            'created_at':       pd.Timestamp.utcnow().isoformat(),
            'pickle_protocol':  pickle_protocol,
            'schema_version':   'v5',                       # Issue 10 FIX: was 'v3'; now 17-feature v5
        }

        metadata_path = self.output_dir / f"metadata_{suffix}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=convert_numpy_types, indent=2)

        logger.info(f"✅ Saved all {task} artifacts")


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestration
# ─────────────────────────────────────────────────────────────────────────────

def train_both_tasks(df: pd.DataFrame):
    """
    Train BOTH reversion and continuation models.

    ISSUE 5 FIX: The scaler and label encoders fitted on the reversion task
    are explicitly passed to the continuation trainer via pre_fitted_scaler
    and by copying label_encoders. This guarantees:
      1. Both models are scaled identically — no risk of divergence if
         feature distributions differ slightly between tasks.
      2. The scaler is fitted only once (on the same X_train), not twice.
      3. Future per-task feature changes would surface immediately as an
         error rather than silently producing different scalers.
    """

    print("\n" + "=" * 80)
    print("DUAL-TASK TRAINING")
    print("=" * 80)
    print("Training TWO models from the same data:")
    print("  1️⃣  REVERSION  (fade the zone)")
    print("  2️⃣  CONTINUATION (ride the break-out)")
    print("=" * 80)

    if 'label' not in df.columns or 'label_cont' not in df.columns:
        raise ValueError("DataFrame must contain both 'label' and 'label_cont' columns")

    n         = len(df)
    train_end = math.floor(n * 0.70)
    val_end   = math.floor(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    logger.info(f"\nData split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

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

    # Fit scaler on reversion training data — it will be shared with continuation
    train_metrics_rev = trainer_reversion.train_ensemble(
        X_train_rev, y_train_rev, X_val_rev, y_val_rev,
        pre_fitted_scaler=None   # explicitly None → fits new scaler here
    )
    test_metrics_rev = trainer_reversion.evaluate_ensemble(X_test_rev, y_test_rev)
    trainer_reversion.save_models(task='reversion', version='v5')

    results['reversion'] = {'train': train_metrics_rev, 'test': test_metrics_rev}

    # ── Task 2: Continuation ──────────────────────────────────────────────────
    print("\n" + "█" * 80)
    print("█" + " " * 25 + "TASK 2: CONTINUATION" + " " * 34 + "█")
    print("█" * 80)

    trainer_continuation = DualTaskTrainer()

    # Share encoders from reversion so categoricals encode identically
    trainer_continuation.label_encoders = trainer_reversion.label_encoders

    X_train_cont, y_train_cont = trainer_continuation.prepare_features(
        train_df, task='continuation',
        train_indices=None   # encoders already fitted — don't refit
    )
    X_val_cont,   y_val_cont   = trainer_continuation.prepare_features(val_df,  task='continuation')
    X_test_cont,  y_test_cont  = trainer_continuation.prepare_features(test_df, task='continuation')

    # Pass the fitted reversion scaler — continuation uses transform(), not fit_transform()
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

    print("\n📊 REVERSION Model:")
    print(f"  Test Accuracy (0.5):  {results['reversion']['test']['accuracy']:.1%}")
    if 'accuracy_0.65' in results['reversion']['test']:
        print(f"  Test Accuracy (0.65): {results['reversion']['test']['accuracy_0.65']:.1%}")
    if 'precision_0.65' in results['reversion']['test']:
        print(f"  Precision  (0.65):   {results['reversion']['test']['precision_0.65']:.1%}  "
              f"(n={results['reversion']['test'].get('n_signals_0.65', '?')})")
    print(f"  ROC-AUC:              {results['reversion']['test']['roc_auc']:.4f}")
    print(f"  Log-loss:             {results['reversion']['test']['log_loss']:.4f}")
    print(f"  Brier score:          {results['reversion']['test']['brier_score']:.4f}")
    print(f"  Signal separation:    {results['reversion']['test']['signal_separation']:+.4f}")

    print("\n📊 CONTINUATION Model:")
    print(f"  Test Accuracy (0.5):  {results['continuation']['test']['accuracy']:.1%}")
    if 'accuracy_0.65' in results['continuation']['test']:
        print(f"  Test Accuracy (0.65): {results['continuation']['test']['accuracy_0.65']:.1%}")
    if 'precision_0.65' in results['continuation']['test']:
        print(f"  Precision  (0.65):   {results['continuation']['test']['precision_0.65']:.1%}  "
              f"(n={results['continuation']['test'].get('n_signals_0.65', '?')})")
    print(f"  ROC-AUC:              {results['continuation']['test']['roc_auc']:.4f}")
    print(f"  Log-loss:             {results['continuation']['test']['log_loss']:.4f}")
    print(f"  Brier score:          {results['continuation']['test']['brier_score']:.4f}")
    print(f"  Signal separation:    {results['continuation']['test']['signal_separation']:+.4f}")

    print("\n✅ Both models saved:")
    print("  models/model_v5_reversion.pkl")
    print("  models/model_v5_continuation.pkl")
    print("  (+ all associated artifacts)")

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
    Load training labels and optionally merge in real outcomes from
    outcome_tracker.py.  Applies a rolling time window so the model
    always trains on the most recent `rolling_months` of data.

    Priority: real outcomes override synthetic labels for the same zone.
    """
    df = pd.read_parquet(labels_path)

    if outcomes_path.exists():
        try:
            outcomes = pd.read_parquet(outcomes_path)
            decisive = outcomes[outcomes["outcome"].isin(["win", "loss"])].copy()
            if not decisive.empty:
                decisive = decisive.rename(columns={"ml_label": "label"})
                decisive["label_cont"] = 1.0 - decisive["label"]
                shared_cols = [c for c in decisive.columns if c in df.columns]
                if shared_cols:
                    df = pd.concat(
                        [df, decisive[shared_cols]], ignore_index=True
                    )
                    logger.info(f"✅ Merged {len(decisive)} real outcomes into training set")
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
    """
    Update an existing XGBoost model with new data without retraining from
    scratch (XGBoost warm-start via xgb_model parameter).
    """
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
    labels_path   = Path("data/labels/training_labels_dual.parquet")
    outcomes_path = Path("data/outcomes/outcomes.parquet")

    if not labels_path.exists():
        logger.error(f"❌ Dual labels not found: {labels_path}")
        logger.info("Run label_generator.py first")
        return

    df = load_with_outcomes(
        labels_path    = labels_path,
        outcomes_path  = outcomes_path,
        rolling_months = 300,
    )
    logger.info(f"Training on {len(df)} examples (synthetic + real outcomes)")
    logger.info(f"Label columns found: {[c for c in df.columns if 'label' in c]}")

    results = train_both_tasks(df)

    print("\n" + "=" * 80)
    print("🎉 DUAL-TASK TRAINING COMPLETE!")
    print("=" * 80)
    print("\n🔥 Next: Use production_predictor.py for live trading")
    print("=" * 80 + "\n")

    # Save results JSON
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