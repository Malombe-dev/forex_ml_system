"""
DUAL-TASK MODEL TRAINER  — LEAKAGE-FREE VERSION
================================================
KEY FIX: train/val/test split is now done by DATE (zone_time) not by row index.
Previously `df.iloc[:train_end]` was used on an unsorted DataFrame, meaning
rows from 2026 could end up in the training set and rows from 2003 in test.
Now we sort by zone_time first, then split at fixed calendar dates.

CHANGE LOG v8 (this version)
-----------------------------
- Added full metrics report: precision/recall/F1, confusion matrix,
  calibration curves, feature importances, ROC curves — all saved to
  models/metrics_<task>/ and printed to terminal during training.
- All other logic (ensemble, weighting, warm-start) unchanged from v7
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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not installed — plots will be skipped")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Date-based split boundaries ───────────────────────────────────────────────
TRAIN_END_DATE = pd.Timestamp("2022-08-04")
VAL_END_DATE   = pd.Timestamp("2024-02-01")


def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_dir(task: str) -> Path:
    p = Path(f"models/metrics_{task}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def print_section(title: str):
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70)


def compute_per_model_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list = None,
) -> Dict:
    """Precision / recall / F1 / AUC for one model at multiple thresholds."""
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    results = {
        'model':      model_name,
        'roc_auc':    float(roc_auc_score(y_true, y_prob)),
        'log_loss':   float(log_loss(y_true, y_prob)),
        'brier':      float(brier_score_loss(y_true, y_prob)),
        'thresholds': {},
    }

    print_section(f"{model_name} — per-threshold breakdown")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Signals':>8}  {'Coverage':>9}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")

    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        n    = int(pred.sum())
        if n == 0:
            prec = rec = f1 = float('nan')
        else:
            prec = float(precision_score(y_true, pred, zero_division=0))
            rec  = float(recall_score(y_true, pred,    zero_division=0))
            f1   = float(f1_score(y_true, pred,        zero_division=0))
        cov = n / len(y_true)
        results['thresholds'][str(thr)] = {
            'precision': prec, 'recall': rec, 'f1': f1,
            'n_signals': n, 'coverage': cov,
        }
        p_str   = f"{prec:.1%}"   if not math.isnan(prec) else "  n/a"
        rec_str = f"{rec:.1%}"    if not math.isnan(rec)  else "  n/a"
        f1_str  = f"{f1:.3f}"     if not math.isnan(f1)   else "  n/a"
        print(f"  {thr:>10.2f}  {p_str:>10}  {rec_str:>8}  {f1_str:>8}  {n:>8,}  {cov:>8.1%}")

    print(f"\n  ROC-AUC: {results['roc_auc']:.4f}  |  Log-loss: {results['log_loss']:.4f}  |  Brier: {results['brier']:.4f}")
    return results


def print_confusion_matrix(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.65):
    pred = (y_prob >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
    print_section(f"{model_name} — Confusion Matrix @ threshold={threshold}")
    print(f"  {'':15}  {'Pred LOSS':>10}  {'Pred WIN':>10}")
    labels = ['Actual LOSS', 'Actual WIN']
    for i, row_label in enumerate(labels):
        if i < len(cm):
            print(f"  {row_label:15}  {cm[i][0]:>10,}  {cm[i][1]:>10,}")
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    print(f"\n  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
    if tp + fp > 0:
        print(f"  Precision = {tp/(tp+fp):.1%}   (of flagged signals, how many won)")
    if tp + fn > 0:
        print(f"  Recall    = {tp/(tp+fn):.1%}   (of all wins, how many were caught)")


def print_classification_report(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.65):
    pred = (y_prob >= threshold).astype(int)
    print_section(f"{model_name} — Full Classification Report @ {threshold}")
    print(classification_report(y_true, pred, target_names=['LOSS', 'WIN'], digits=3))


def save_roc_curves(
    task: str,
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],   # {'XGBoost': array, 'RF': array, ...}
    out_dir: Path,
    thresholds: list = None,
):
    if not MATPLOTLIB_AVAILABLE:
        return
    if thresholds is None:
        thresholds = [0.60, 0.65]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"ROC & Precision-Recall Curves — {task.upper()}", fontsize=13, fontweight='bold')

    colors = ['#2b6cb0', '#00a854', '#c05621', '#6b46c1']

    # ROC
    ax = axes[0]
    ax.plot([0,1],[0,1],'--',color='gray',lw=1,label='Random (AUC=0.50)')
    for (name, prob), col in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel('False Positive Rate');  ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve');  ax.legend(fontsize=9);  ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[1]
    baseline = float(y_true.mean())
    ax.axhline(baseline, color='gray', lw=1, ls='--', label=f'Baseline={baseline:.2f}')
    for (name, prob), col in zip(probs.items(), colors):
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, prob)
        ax.plot(rec_arr, prec_arr, color=col, lw=2, label=name)
    # Mark threshold operating points on ensemble
    if 'Ensemble' in probs:
        for thr in thresholds:
            pred = (probs['Ensemble'] >= thr).astype(int)
            p = precision_score(y_true, pred, zero_division=0)
            r = recall_score(y_true, pred, zero_division=0)
            ax.scatter(r, p, zorder=5, s=80)
            ax.annotate(f'{thr}', (r, p), textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax.set_xlabel('Recall');  ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve');  ax.legend(fontsize=9);  ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'roc_pr_curves.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"  📊 Saved ROC/PR curves → {out_path}")


def save_calibration_curves(
    task: str,
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
    out_dir: Path,
):
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],'--',color='gray',lw=1,label='Perfect calibration')
    colors = ['#2b6cb0', '#00a854', '#c05621', '#6b46c1']
    for (name, prob), col in zip(probs.items(), colors):
        try:
            frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy='uniform')
            ax.plot(mean_pred, frac_pos, 'o-', color=col, lw=2, label=name)
        except Exception:
            pass
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (actual win rate)')
    ax.set_title(f'Calibration Curves — {task.upper()}\n(closer to diagonal = better calibrated)')
    ax.legend(fontsize=9);  ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / 'calibration_curves.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"  📊 Saved calibration curves → {out_path}")


def save_feature_importance(
    task: str,
    feature_names: list,
    xgb_model,
    rf_model,
    lr_model,
    out_dir: Path,
    top_n: int = 27,
):
    if not MATPLOTLIB_AVAILABLE:
        return

    importances = {}

    # XGBoost — from base estimators inside CalibratedClassifierCV
    try:
        xgb_imps = []
        for est in xgb_model.calibrated_classifiers_:
            base = est.estimator if hasattr(est, 'estimator') else est.base_estimator
            xgb_imps.append(base.feature_importances_)
        importances['XGBoost'] = np.mean(xgb_imps, axis=0)
    except Exception as e:
        logger.warning(f"Could not extract XGBoost importances: {e}")

    # Random Forest
    try:
        rf_imps = []
        for est in rf_model.calibrated_classifiers_:
            base = est.estimator if hasattr(est, 'estimator') else est.base_estimator
            rf_imps.append(base.feature_importances_)
        importances['Random Forest'] = np.mean(rf_imps, axis=0)
    except Exception as e:
        logger.warning(f"Could not extract RF importances: {e}")

    # Logistic Regression — abs coefficients, averaged over base estimators
    # inside CalibratedClassifierCV (same pattern as XGBoost/RF above).
    try:
        if hasattr(lr_model, 'calibrated_classifiers_'):
            lr_coefs = []
            for est in lr_model.calibrated_classifiers_:
                base = est.estimator if hasattr(est, 'estimator') else est.base_estimator
                lr_coefs.append(np.abs(base.coef_[0]))
            lr_coef = np.mean(lr_coefs, axis=0)
        else:
            # Backward-compat: plain (uncalibrated) LogisticRegression
            lr_coef = np.abs(lr_model.coef_[0])
        lr_coef = lr_coef / lr_coef.sum()
        importances['Log. Regression'] = lr_coef
    except Exception as e:
        logger.warning(f"Could not extract LR importances: {e}")

    if not importances:
        return

    # Print top features to terminal
    print_section(f"Feature Importances — {task.upper()}")
    for model_name, imps in importances.items():
        idx_sorted = np.argsort(imps)[::-1][:15]
        print(f"\n  Top 15 — {model_name}:")
        print(f"  {'Rank':>4}  {'Feature':<35}  {'Importance':>10}")
        print(f"  {'─'*4}  {'─'*35}  {'─'*10}")
        for rank, idx in enumerate(idx_sorted, 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            print(f"  {rank:>4}  {fname:<35}  {imps[idx]:>10.4f}")

    # Plot
    n_models  = len(importances)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(f"Feature Importances — {task.upper()}", fontsize=13, fontweight='bold')
    colors_map = {'XGBoost': '#2b6cb0', 'Random Forest': '#00a854', 'Log. Regression': '#c05621'}

    for ax, (model_name, imps) in zip(axes, importances.items()):
        idx_sorted = np.argsort(imps)[::-1][:top_n]
        names_sorted = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in idx_sorted]
        vals_sorted  = imps[idx_sorted]
        col = colors_map.get(model_name, '#6b46c1')
        bars = ax.barh(range(len(names_sorted))[::-1], vals_sorted, color=col, alpha=0.8)
        ax.set_yticks(range(len(names_sorted))[::-1])
        ax.set_yticklabels(names_sorted, fontsize=8)
        ax.set_title(model_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'feature_importances.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"  📊 Saved feature importances → {out_path}")


def save_confusion_matrix_plot(
    task: str,
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
    out_dir: Path,
    threshold: float = 0.65,
):
    if not MATPLOTLIB_AVAILABLE:
        return

    n = len(probs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Confusion Matrices @ threshold={threshold} — {task.upper()}", fontsize=12, fontweight='bold')

    for ax, (model_name, prob) in zip(axes, probs.items()):
        pred = (prob >= threshold).astype(int)
        cm   = confusion_matrix(y_true, pred)
        im   = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(model_name, fontweight='bold')
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks); ax.set_xticklabels(['LOSS', 'WIN'])
        ax.set_yticks(tick_marks); ax.set_yticklabels(['LOSS', 'WIN'])
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
        thresh_cm = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                        color='white' if cm[i, j] > thresh_cm else 'black', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out_path = out_dir / f'confusion_matrix_{int(threshold*100)}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"  📊 Saved confusion matrices → {out_path}")


def run_full_metrics_report(
    task: str,
    y_true: np.ndarray,
    p_xgb: np.ndarray,
    p_rf: np.ndarray,
    p_lr: np.ndarray,
    p_ensemble: np.ndarray,
    feature_names: list,
    xgb_model,
    rf_model,
    lr_model,
    thresholds: list = None,
    split_label: str = "TEST",
):
    """Run and save all metrics for a task. Called after evaluate_ensemble."""
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    out_dir = _metrics_dir(task)

    probs_dict = {
        'XGBoost':     p_xgb,
        'Random Forest': p_rf,
        'Log. Regression': p_lr,
        'Ensemble':    p_ensemble,
    }

    print("\n" + "█" * 70)
    print(f"█  FULL METRICS REPORT — {task.upper()} — {split_label} SET")
    print("█" * 70)

    # 1. Per-model precision/recall/F1
    all_model_metrics = {}
    for name, prob in probs_dict.items():
        all_model_metrics[name] = compute_per_model_metrics(name, y_true, prob, thresholds)

    # 2. Confusion matrix (terminal) — ensemble at trading threshold
    trading_thr = 0.65 if task == 'reversion' else 0.60
    for name, prob in probs_dict.items():
        print_confusion_matrix(name, y_true, prob, threshold=trading_thr)

    # 3. Full sklearn classification report for ensemble
    print_classification_report('Ensemble', y_true, p_ensemble, threshold=trading_thr)

    # 4. Calibration check (terminal)
    print_section(f"Calibration Check — {task.upper()} (actual win rate per probability bucket)")
    print(f"  {'Prob bucket':>12}  {'Predicted':>10}  {'Actual win %':>12}  {'N samples':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*10}")
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_ensemble >= lo) & (p_ensemble < hi)
        n    = mask.sum()
        if n == 0:
            continue
        actual_wr = y_true[mask].mean()
        pred_mean = p_ensemble[mask].mean()
        flag = "  ✅" if abs(actual_wr - pred_mean) < 0.08 else "  ⚠️ "
        print(f"  {lo:.1f}–{hi:.1f}       {pred_mean:>10.1%}  {actual_wr:>12.1%}  {n:>10,}{flag}")

    # 5. Save plots
    save_roc_curves(task, y_true, probs_dict, out_dir, thresholds=[0.60, 0.65])
    save_calibration_curves(task, y_true, probs_dict, out_dir)
    save_feature_importance(task, feature_names, xgb_model, rf_model, lr_model, out_dir)
    save_confusion_matrix_plot(task, y_true, probs_dict, out_dir, threshold=trading_thr)

    # 6. Save JSON metrics
    metrics_json = {
        'task':        task,
        'split':       split_label,
        'threshold_trading': trading_thr,
        'models':      all_model_metrics,
    }
    json_path = out_dir / f'metrics_{split_label.lower()}.json'
    with open(json_path, 'w') as f:
        def _cvt(d):
            if isinstance(d, dict):  return {k: _cvt(v) for k, v in d.items()}
            if isinstance(d, list):  return [_cvt(i) for i in d]
            if isinstance(d, float) and math.isnan(d): return None
            if isinstance(d, (np.floating, np.integer)): return float(d)
            return d
        json.dump(_cvt(metrics_json), f, indent=2)
    logger.info(f"  📄 Saved metrics JSON → {json_path}")

    print(f"\n  ✅ All {task.upper()} {split_label} metrics saved to {out_dir}/")
    return all_model_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer class
# ─────────────────────────────────────────────────────────────────────────────

class DualTaskTrainer:
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
        self.calibrator               = None  # fitted post-hoc ensemble calibrator (Platt or isotonic)
        self._calibrator_kind         = None  # 'platt' | 'isotonic' | None

    def prepare_features(self,
                         df: pd.DataFrame,
                         task: str = 'reversion',
                         train_indices: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Preparing features for {task.upper()} task...")

        # ── Sentinel & outlier cleanup (applied before column selection) ────────
        df = df.copy()
        # time_since_last_touch is excluded from NUMERIC_FEATURES (v8) but may
        # still exist in older parquet files — neutralise it if present so it
        # can't accidentally leak in via a schema mismatch.
        if 'time_since_last_touch' in df.columns:
            sentinel_mask = df['time_since_last_touch'] >= 2000
            if sentinel_mask.any():
                logger.warning(
                    f"  ⚠️  {sentinel_mask.sum()} rows have time_since_last_touch=2000 sentinel — "
                    f"clipping to 500 and adding is_first_touch flag"
                )
            df['is_first_touch'] = sentinel_mask.astype(float)
            df['time_since_last_touch'] = df['time_since_last_touch'].clip(upper=500)
        # Winsorise zone_strength at 99th percentile to prevent USDJPY outliers
        # (e.g. strength=16.39 during 2024-07 JPY intervention) from dominating
        # Logistic Regression in the ensemble.
        if 'zone_strength' in df.columns:
            cap = df['zone_strength'].quantile(0.99)
            n_clipped = (df['zone_strength'] > cap).sum()
            if n_clipped > 0:
                logger.info(f"  Clipping {n_clipped} zone_strength outliers at 99th pct ({cap:.3f})")
                df['zone_strength'] = df['zone_strength'].clip(upper=cap)
        # ── End sentinel cleanup ────────────────────────────────────────────────

        numeric_features     = list(NUMERIC_FEATURES)
        categorical_features = list(CATEGORICAL_FEATURES)
        missing_numeric = [f for f in numeric_features     if f not in df.columns]
        missing_cats    = [f for f in categorical_features if f not in df.columns]
        if missing_numeric or missing_cats:
            raise ValueError(
                f"Training parquet is missing expected feature columns.\n"
                f"  Missing numeric:      {missing_numeric}\n"
                f"  Missing categorical:  {missing_cats}\n"
                "Re-run label_generator.py to produce a v7-schema parquet "
                "before retraining."
            )
        X = df[numeric_features].copy()
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
                            le.transform([val])[0] if val in le.classes_ else 0
                        )
                    X[f'{cat_col}_encoded'] = encoded
        if task == 'reversion':
            y_raw = df['label'].copy().astype(np.float64)
        elif task == 'continuation':
            y_raw = df['label_cont'].copy().astype(np.float64)
        else:
            raise ValueError(f"Unknown task: {task}")

        # FIX: label_generator.py's check_reversion_outcome / check_continuation_outcome
        # return *soft* scores for timeouts (e.g. 0.75, 0.60, 0.35, 0.20 for reversion;
        # 0.2/0.4/0.6/0.8 proximity for continuation) alongside clean 0.0/1.0 for decisive
        # hits. Downstream this used to be passed straight into `.astype(int)` before
        # fitting the classifiers, which TRUNCATES toward zero -- every non-1.0 timeout
        # score silently became class 0, indistinguishable from a clean loss, regardless
        # of how close the trade actually got to winning. That's label noise injected
        # right at the decision boundary the model needs to learn best.
        #
        # Fix: drop ambiguous (non-decisive) rows from CLASSIFIER training entirely
        # rather than mislabeling them. NaN also gets dropped here, which covers the
        # load_with_outcomes() live-merge case (see that function's fix) where a
        # decisive continuation label genuinely isn't known.
        decisive_mask = np.isclose(y_raw, 0.0) | np.isclose(y_raw, 1.0)
        n_ambiguous = int((~decisive_mask).sum())
        if n_ambiguous:
            logger.warning(
                f"  Dropping {n_ambiguous}/{len(y_raw)} ambiguous/timeout rows "
                f"(soft label strictly between 0 and 1, or NaN) from {task} classifier "
                f"training -- previously these were silently truncated to class 0."
            )
        X = X.loc[decisive_mask].reset_index(drop=True)
        y = y_raw.loc[decisive_mask].astype(int).reset_index(drop=True)

        self.feature_names = X.columns.tolist()
        assert_feature_columns_exact(self.feature_names)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target ({task}): mean={y.mean():.3f}  min={y.min()}  max={y.max()}")
        return X, y

    def train_ensemble(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None,
                       pre_fitted_scaler: Optional[StandardScaler] = None) -> Dict:
        logger.info("=" * 70)
        logger.info("TRAINING CALIBRATED ENSEMBLE  [classification]")
        logger.info("=" * 70)
        from sklearn.calibration import CalibratedClassifierCV
        if pre_fitted_scaler is not None:
            self.scaler    = pre_fitted_scaler
            X_train_scaled = self.scaler.transform(X_train)
            logger.info("✅ Using pre-fitted scaler (shared from reversion task)")
        else:
            self.scaler    = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info("✅ Fitted new StandardScaler on TRAINING data only")
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
        y_arr     = np.asarray(y_train, dtype=float)
        recency_w = np.ones(len(y_arr))
        recency_w = recency_w / recency_w.mean()
        n_pos = max(int(y_arr.sum()), 1)
        n_neg = max(len(y_arr) - n_pos, 1)
        spw   = min(n_neg / n_pos, 8.0)
        logger.info(f"  Class balance: {n_pos} wins / {n_neg} losses → scale_pos_weight={spw:.2f}")
        metrics = {}

        logger.info("\n1️⃣  Training Calibrated XGBoost...")
        raw_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=spw, random_state=42,
            objective='binary:logistic', eval_metric='logloss',
        )
        self.xgb_model = CalibratedClassifierCV(estimator=raw_xgb, method='sigmoid', cv=3)
        self.xgb_model.fit(X_train_scaled, y_train.astype(int), sample_weight=recency_w)
        p_xgb = self.xgb_model.predict_proba(X_train_scaled)[:, 1].astype(np.float32)
        y_int = y_arr.astype(int)
        metrics['xgb'] = {'log_loss': log_loss(y_int, p_xgb), 'roc_auc': roc_auc_score(y_int, p_xgb)}
        logger.info(f"  XGB train  log_loss={metrics['xgb']['log_loss']:.4f}  auc={metrics['xgb']['roc_auc']:.4f}")

        logger.info("\n2️⃣  Training Calibrated Random Forest...")
        raw_rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=20,
            min_samples_leaf=10, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=1,
        )
        self.rf_model = CalibratedClassifierCV(estimator=raw_rf, method='sigmoid', cv=3)
        self.rf_model.fit(X_train_scaled, y_train.astype(int), sample_weight=recency_w)
        p_rf = self.rf_model.predict_proba(X_train_scaled)[:, 1].astype(np.float32)
        metrics['rf'] = {'log_loss': log_loss(y_int, p_rf), 'roc_auc': roc_auc_score(y_int, p_rf)}
        logger.info(f"  RF  train  log_loss={metrics['rf']['log_loss']:.4f}  auc={metrics['rf']['roc_auc']:.4f}")

        logger.info("\n3️⃣  Training Calibrated Logistic Regression...")
        raw_lr = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs',
        )
        # NOTE: plain LogisticRegression is NOT automatically well-calibrated —
        # class_weight='balanced' reweights the loss during fit, which skews
        # predict_proba away from true posterior probabilities (that's the
        # source of the 0.07 ↔ 0.89 extremes). Wrap in CalibratedClassifierCV,
        # same as XGBoost and RF, so all three base models are calibrated
        # before they're blended into the ensemble.
        self.ridge_model = CalibratedClassifierCV(estimator=raw_lr, method='sigmoid', cv=3)
        self.ridge_model.fit(X_train_scaled, y_train.astype(int), sample_weight=recency_w)
        p_lr = self.ridge_model.predict_proba(X_train_scaled)[:, 1].astype(np.float32)
        metrics['lr'] = {'log_loss': log_loss(y_int, p_lr), 'roc_auc': roc_auc_score(y_int, p_lr)}
        logger.info(f"  LR  train  log_loss={metrics['lr']['log_loss']:.4f}  auc={metrics['lr']['roc_auc']:.4f}")

        # ── 4️⃣  Fit ensemble-level calibrator on the VAL set ───────────────────
        # X_val_scaled was previously computed and never used. Per-model
        # CalibratedClassifierCV(cv=3) calibrates each base model using folds
        # carved out of the TRAINING set, which is why the fixed-weight blend
        # of those three "calibrated" outputs can still be systematically
        # overconfident on genuinely unseen data. Here we fit a lightweight
        # Platt (sigmoid) scaler and an isotonic regressor on true out-of-sample
        # val predictions so the final blended probability itself is calibrated.
        self.calibrator = None
        self._calibrator_kind = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            logger.info("\n4️⃣  Fitting ensemble-level calibrator on VAL set...")
            p_xgb_val = self.xgb_model.predict_proba(X_val_scaled)[:, 1]
            p_rf_val  = self.rf_model.predict_proba(X_val_scaled)[:, 1]
            p_lr_val  = self.ridge_model.predict_proba(X_val_scaled)[:, 1]
            p_ens_val = (
                p_xgb_val * self.ensemble_weights['xgb'] +
                p_rf_val  * self.ensemble_weights['rf']  +
                p_lr_val  * self.ensemble_weights['ridge']
            )
            y_val_bin = (np.asarray(y_val, dtype=float) >= 1.0).astype(int)

            from sklearn.linear_model import LogisticRegression as _PlattScaler
            from sklearn.isotonic import IsotonicRegression
            from sklearn.model_selection import KFold

            def _fit_predict(kind, x_tr, y_tr, x_te):
                if kind == 'platt':
                    m = _PlattScaler()
                    m.fit(x_tr.reshape(-1, 1), y_tr)
                    return m.predict_proba(x_te.reshape(-1, 1))[:, 1]
                else:  # isotonic
                    m = IsotonicRegression(out_of_bounds='clip')
                    m.fit(x_tr, y_tr)
                    return m.predict(x_te)

            # ── Proper cross-validated (out-of-fold) evaluation ─────────────────
            # IMPORTANT: isotonic regression can memorize its own fit data almost
            # perfectly (it's a free-knot step function), so evaluating it on the
            # SAME val rows it was fit on always makes it look best — that was the
            # earlier bug. Instead, k-fold CV within the val set: fit on k-1 folds,
            # score on the held-out fold, so every score is genuinely out-of-sample.
            n = len(p_ens_val)
            n_splits = min(5, max(2, n // 40))  # keep folds reasonably sized
            kf = KFold(n_splits=n_splits, shuffle=False)  # preserve time order

            ll_raw = log_loss(y_val_bin, np.clip(p_ens_val, 1e-7, 1 - 1e-7))
            br_raw = brier_score_loss(y_val_bin, np.clip(p_ens_val, 1e-7, 1 - 1e-7))

            oof_scores = {}
            for kind in ('platt', 'isotonic'):
                oof_pred = np.zeros(n)
                for tr_idx, te_idx in kf.split(p_ens_val):
                    if len(np.unique(y_val_bin[tr_idx])) < 2:
                        oof_pred[te_idx] = p_ens_val[te_idx]  # can't fit — fall back to raw for this fold
                        continue
                    oof_pred[te_idx] = _fit_predict(kind, p_ens_val[tr_idx], y_val_bin[tr_idx], p_ens_val[te_idx])
                oof_scores[kind] = {
                    'll': log_loss(y_val_bin, np.clip(oof_pred, 1e-7, 1 - 1e-7)),
                    'br': brier_score_loss(y_val_bin, np.clip(oof_pred, 1e-7, 1 - 1e-7)),
                }
                logger.info(
                    f"     OOF ({n_splits}-fold) log_loss  raw={ll_raw:.4f}  →  {kind}={oof_scores[kind]['ll']:.4f}   "
                    f"(brier raw={br_raw:.4f} → {oof_scores[kind]['br']:.4f})"
                )

            # Guardrail: only keep a candidate if its cross-validated (honest,
            # held-out) score beats raw on BOTH metrics by a real margin.
            MIN_IMPROVEMENT = 0.002
            qualifying = {
                kind: s for kind, s in oof_scores.items()
                if (ll_raw - s['ll'] > MIN_IMPROVEMENT) and (br_raw - s['br'] > MIN_IMPROVEMENT)
            }
            if qualifying:
                best_kind = min(qualifying, key=lambda k: qualifying[k]['ll'])
                # Refit the winning calibrator type on the FULL val set for deployment
                # (CV was only used to pick the type — this is standard practice: use
                # CV to select, then refit champion on all available data).
                if best_kind == 'platt':
                    final_cal = _PlattScaler()
                    final_cal.fit(p_ens_val.reshape(-1, 1), y_val_bin)
                else:
                    final_cal = IsotonicRegression(out_of_bounds='clip')
                    final_cal.fit(p_ens_val, y_val_bin)
                self.calibrator = final_cal
                self._calibrator_kind = best_kind
                logger.info(f"  ✅ Selected '{best_kind}' calibrator (cross-validated) — refit on all {n} val examples")
            else:
                self.calibrator = None
                self._calibrator_kind = None
                logger.warning(
                    f"  ⚠️  Neither Platt nor isotonic meaningfully improved OUT-OF-FOLD log_loss/brier "
                    f"— DISCARDING calibration, will use raw ensemble probabilities instead."
                )
        else:
            logger.warning("  ⚠️  No val set provided — ensemble calibrator NOT fit (predictions will stay raw)")


        return metrics

    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series,
                          task: str = 'unknown', run_full_report: bool = True) -> Dict:
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

        # ── Apply fitted ensemble calibrator (if we have one) ───────────────────
        # Without this, the raw weighted blend can be systematically overconfident
        # in the mid-probability range even though each base model is individually
        # "calibrated" — see train_ensemble() for why.
        if getattr(self, 'calibrator', None) is not None:
            p_ensemble_raw = p_ensemble
            if getattr(self, '_calibrator_kind', None) == 'isotonic':
                p_ensemble = self.calibrator.predict(p_ensemble_raw).astype(np.float32)
            else:
                p_ensemble = self.calibrator.predict_proba(
                    p_ensemble_raw.reshape(-1, 1)
                )[:, 1].astype(np.float32)
            p_ensemble = np.clip(p_ensemble, 1e-7, 1 - 1e-7).astype(np.float32)
            logger.info(f"  ✅ Applied '{getattr(self, '_calibrator_kind', '?')}' calibrator to test-set probabilities")
        else:
            logger.warning("  ⚠️  No ensemble calibrator available — using raw (uncalibrated) probabilities")

        y_bin = (y_test >= 1.0).astype(int)   # FIX: match training cast — int(0.5)=0, int(0.75)=0, only 1.0=win
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
                metrics['accuracy']  = accuracy
                metrics['precision'] = precision
                metrics['n_signals'] = n_signals
            if threshold == 0.65:
                metrics['accuracy_0.65']  = accuracy
                metrics['precision_0.65'] = precision
                metrics['n_signals_0.65'] = n_signals
        pred_on_wins = float(p_ensemble[y_bin == 1].mean()) if y_bin.sum() > 0 else 0.5
        pred_on_loss = float(p_ensemble[y_bin == 0].mean()) if (1 - y_bin).sum() > 0 else 0.5
        metrics['signal_separation'] = round(pred_on_wins - pred_on_loss, 4)
        logger.info(f"  Signal separation: {metrics['signal_separation']:+.4f}  (>0.05 = edge)")

        # ── Full metrics report ───────────────────────────────────────────────
        if run_full_report and task != 'unknown':
            run_full_metrics_report(
                task         = task,
                y_true       = y_bin.values if hasattr(y_bin, 'values') else y_bin,
                p_xgb        = p_xgb,
                p_rf         = p_rf,
                p_lr         = p_lr,
                p_ensemble   = p_ensemble,
                feature_names= self.feature_names,
                xgb_model    = self.xgb_model,
                rf_model     = self.rf_model,
                lr_model     = self.ridge_model,
                split_label  = "TEST",
            )

        return metrics

    def save_models(self, task: str, version: str = "v6"):
        """Save all models and calibration artifacts."""
        logger.info(f"\nSAVING {task.upper()} MODELS")
        logger.info("=" * 70)
        suffix = f"{version}_{task}"
        pickle_protocol = 4
        
        # 1. Save calibrated XGBoost (already CalibratedClassifierCV)
        model_path = self.output_dir / f"model_{suffix}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.xgb_model, f, protocol=pickle_protocol)
        logger.info(f"✅ Saved XGBoost (calibrated) to {model_path}")
        
        # 2. Save calibrated RF
        rf_path = self.output_dir / f"rf_{suffix}.pkl"
        with open(rf_path, 'wb') as f:
            pickle.dump(self.rf_model, f, protocol=pickle_protocol)
        logger.info(f"✅ Saved RF (calibrated) to rf_{suffix}.pkl")
        
        # 3. Save calibrated LR
        lr_path = self.output_dir / f"lr_{suffix}.pkl"
        with open(lr_path, 'wb') as f:
            pickle.dump(self.ridge_model, f, protocol=pickle_protocol)
        logger.info(f"✅ Saved LR (calibrated) to lr_{suffix}.pkl")
        
        # 4. ✅ NEW: Save a dedicated calibrator for the ensemble
        # This saves calibration metadata so production knows the models are calibrated
        try:
            calibrator = {
                'ensemble_weights': self.ensemble_weights,
                'scaler': self.scaler,
                'xgb_calibrated': True,  # Flag that XGBoost is already calibrated
                'rf_calibrated': True,   # Flag that RF is already calibrated
                'lr_calibrated': True,   # Flag that LR is already calibrated
                'calibration_method': 'sigmoid',
                'cv_folds': 3,
            }
            
            # Save the calibration metadata
            cal_path = self.output_dir / f"calibrator_{suffix}.pkl"
            with open(cal_path, 'wb') as f:
                pickle.dump(calibrator, f, protocol=pickle_protocol)
            logger.info(f"✅ Saved calibrator metadata to calibrator_{suffix}.pkl")
            
        except Exception as e:
            logger.warning(f"Could not save calibration artifacts: {e}")
        
        # 5. Save scaler
        scaler_path = self.output_dir / f"scaler_{suffix}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f, protocol=pickle_protocol)
        logger.info(f"✅ Saved scaler to scaler_{suffix}.pkl")
        
        # 6. Save encoders
        encoders_path = self.output_dir / f"encoders_{suffix}.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f, protocol=pickle_protocol)
        logger.info(f"✅ Saved encoders to encoders_{suffix}.pkl")
        
        # 7. Save training stats
        stats_path = self.output_dir / f"training_stats_{suffix}.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(self.training_feature_stats, f, protocol=pickle_protocol)
        logger.info(f"✅ Saved training stats to training_stats_{suffix}.pkl")
        
        # 8. Metadata
        metadata = {
            'version': suffix,
            'task': task,
            'model_type': 'classification',
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'models': ['XGBClassifier(calibrated)', 'RandomForestClassifier(calibrated)', 'LogisticRegression(calibrated)'],
            'ensemble_weights': self.ensemble_weights,
            'train_end_date': TRAIN_END_DATE.isoformat(),
            'val_end_date': VAL_END_DATE.isoformat(),
            'created_at': pd.Timestamp.utcnow().isoformat(),
            'pickle_protocol': pickle_protocol,
            'schema_version': 'v8',
            'calibration': {
                'xgb_method': 'sigmoid',
                'rf_method': 'sigmoid',
                'cv_folds': 3,
            }
        }
        metadata_path = self.output_dir / f"metadata_{suffix}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=convert_numpy_types, indent=2)
        logger.info(f"✅ Saved metadata to metadata_{suffix}.json")
        logger.info(f"✅ All {task} artifacts saved successfully")

    def save_calibration_model(self, task: str, version: str = "v6"):
        """
        Save the full ensemble bundle plus the fitted post-hoc calibrator.

        NOTE: previously this saved a nested closure (`calibration_predict`)
        which pickle cannot serialize ("Can't pickle local object
        'DualTaskTrainer.save_calibration_model.<locals>.calibration_predict'").
        The closure write failed silently into a caught exception, so the
        ensemble bundle saved fine but the actual calibration mapping never
        made it to disk. self.calibrator is a real sklearn LogisticRegression
        instance (fit in train_ensemble on the val set), so it pickles cleanly.
        """
        suffix = f"{version}_{task}"
        try:
            # Save the full ensemble bundle (models, scaler, weights) for convenience
            ensemble_cal = {
                'xgb_model': self.xgb_model,
                'rf_model': self.rf_model,
                'lr_model': self.ridge_model,
                'scaler': self.scaler,
                'weights': self.ensemble_weights,
                'is_calibrated': self.calibrator is not None,
                'calibration_method': 'platt_sigmoid_on_val',
            }
            cal_fn_path = self.output_dir / f"calibration_ensemble_{suffix}.pkl"
            with open(cal_fn_path, 'wb') as f:
                pickle.dump(ensemble_cal, f, protocol=4)
            logger.info(f"✅ Saved calibration ensemble to calibration_ensemble_{suffix}.pkl")

            # Save the actual fitted calibrator — a real object, not a closure.
            # Bundle it with its 'kind' since Platt (LogisticRegression) exposes
            # .predict_proba() while IsotonicRegression exposes .predict() — the
            # two are NOT interchangeable and inference code must branch on this.
            cal_path = self.output_dir / f"calibrator_ensemble_{suffix}.pkl"
            calibrator_bundle = {
                'calibrator': self.calibrator,
                'kind': getattr(self, '_calibrator_kind', None),  # 'platt' | 'isotonic' | None
            }
            with open(cal_path, 'wb') as f:
                pickle.dump(calibrator_bundle, f, protocol=4)
            if self.calibrator is not None:
                logger.info(
                    f"✅ Saved fitted '{calibrator_bundle['kind']}' ensemble calibrator "
                    f"to calibrator_ensemble_{suffix}.pkl"
                )
            else:
                logger.warning(
                    f"⚠️  self.calibrator is None (no val set was provided, or no candidate "
                    f"improved val metrics) — saved a None placeholder to calibrator_ensemble_{suffix}.pkl. "
                    f"Inference code should treat None as 'use raw ensemble probabilities'."
                )

        except Exception as e:
            logger.warning(f"Could not save calibration artifacts: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Training orchestration
# ─────────────────────────────────────────────────────────────────────────────

def split_by_date(df, train_end, val_end):
    if 'zone_time' not in df.columns:
        raise ValueError("DataFrame must have a 'zone_time' column for date-based splitting")
    dates      = pd.to_datetime(df['zone_time'])
    train_mask = dates < train_end
    val_mask   = (dates >= train_end) & (dates < val_end)
    test_mask  = dates >= val_end
    train_df   = df[train_mask].copy()
    val_df     = df[val_mask].copy()
    test_df    = df[test_mask].copy()
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
    train_max = pd.to_datetime(train_df['zone_time']).max()
    test_min  = pd.to_datetime(test_df['zone_time']).min()
    assert train_max < test_min, f"Overlap! train_max={train_max}  test_min={test_min}"
    logger.info(f"  ✅ No date overlap: train ends {train_max.date()}, test starts {test_min.date()}")
    return train_df, val_df, test_df


def train_both_tasks(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("DUAL-TASK TRAINING  [leakage-free v8]")
    print("=" * 80)
    if 'label' not in df.columns or 'label_cont' not in df.columns:
        raise ValueError("DataFrame must contain both 'label' and 'label_cont' columns")
    if 'zone_time' not in df.columns:
        raise ValueError("DataFrame must contain 'zone_time' for date-based splitting.")
    df = df.copy()
    df['zone_time'] = pd.to_datetime(df['zone_time'])
    df = df.sort_values('zone_time').reset_index(drop=True)
    logger.info(f"Data sorted by zone_time: {df['zone_time'].min().date()} → {df['zone_time'].max().date()}")
    train_df, val_df, test_df = split_by_date(df, TRAIN_END_DATE, VAL_END_DATE)
    results = {}

    # ── Task 1: Reversion ──────────────────────────────────────────────────
    print("\n" + "█" * 80)
    print("█" + " " * 26 + "TASK 1: REVERSION" + " " * 36 + "█")
    print("█" * 80)
    trainer_reversion = DualTaskTrainer()
    X_train_rev, y_train_rev = trainer_reversion.prepare_features(train_df, task='reversion', train_indices=np.arange(len(train_df)))
    X_val_rev,   y_val_rev   = trainer_reversion.prepare_features(val_df,   task='reversion')
    X_test_rev,  y_test_rev  = trainer_reversion.prepare_features(test_df,  task='reversion')
    train_metrics_rev = trainer_reversion.train_ensemble(X_train_rev, y_train_rev, X_val_rev, y_val_rev, pre_fitted_scaler=None)
    test_metrics_rev  = trainer_reversion.evaluate_ensemble(X_test_rev, y_test_rev, task='reversion')
    trainer_reversion.save_models(task='reversion', version='v6')
    trainer_reversion.save_calibration_model(task='reversion', version='v6')
    results['reversion'] = {'train': train_metrics_rev, 'test': test_metrics_rev}

    # ── Task 2: Continuation ──────────────────────────────────────────────
    print("\n" + "█" * 80)
    print("█" + " " * 25 + "TASK 2: CONTINUATION" + " " * 34 + "█")
    print("█" * 80)
    trainer_continuation = DualTaskTrainer()
    trainer_continuation.label_encoders = trainer_reversion.label_encoders
    X_train_cont, y_train_cont = trainer_continuation.prepare_features(train_df, task='continuation', train_indices=None)
    X_val_cont,   y_val_cont   = trainer_continuation.prepare_features(val_df,   task='continuation')
    X_test_cont,  y_test_cont  = trainer_continuation.prepare_features(test_df,  task='continuation')
    train_metrics_cont = trainer_continuation.train_ensemble(X_train_cont, y_train_cont, X_val_cont, y_val_cont, pre_fitted_scaler=trainer_reversion.scaler)
    test_metrics_cont  = trainer_continuation.evaluate_ensemble(X_test_cont, y_test_cont, task='continuation')
    trainer_continuation.save_models(task='continuation', version='v6')
    trainer_continuation.save_calibration_model(task='continuation', version='v6')
    results['continuation'] = {'train': train_metrics_cont, 'test': test_metrics_cont}

    # ── Summary ───────────────────────────────────────────────────────────
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

    print("\n  📁 Metrics saved to:")
    print("     models/metrics_reversion/")
    print("     models/metrics_continuation/")
    print("       ├── roc_pr_curves.png")
    print("       ├── calibration_curves.png")
    print("       ├── feature_importances.png")
    print("       ├── confusion_matrix_65.png")
    print("       └── metrics_test.json")

    print("\n✅ Both models saved:")
    print("  models/model_v6_reversion.pkl")
    print("  models/model_v6_continuation.pkl")
    print("  models/calibrator_ensemble_v6_reversion.pkl")
    print("  models/calibrator_ensemble_v6_continuation.pkl")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-window + incremental training helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_with_outcomes(
    labels_path: Path,
    outcomes_path: Path = Path("data/outcomes/outcomes.parquet"),
    rolling_months: int = 60,
) -> pd.DataFrame:
    df = pd.read_parquet(labels_path)
    if outcomes_path.exists():
        try:
            outcomes = pd.read_parquet(outcomes_path)
            decisive = outcomes[outcomes["outcome"].isin(["win", "loss"])].copy()
            if not decisive.empty:
                decisive['zone_time'] = pd.to_datetime(decisive.get('signal_timestamp', decisive.get('zone_time')))
                n_before = len(decisive)
                decisive = decisive[decisive['zone_time'] < TRAIN_END_DATE]
                n_after  = len(decisive)
                if n_before != n_after:
                    logger.warning(
                        f"⚠️  Dropped {n_before - n_after} real outcomes with zone_time "
                        f">= TRAIN_END_DATE ({TRAIN_END_DATE.date()}) — merging them would cause leakage."
                    )
                if not decisive.empty:
                    decisive = decisive.rename(columns={"ml_label": "label"})
                    # FIX: this used to set label_cont = 1.0 - label, i.e. "reversion
                    # lost => call it a confirmed continuation win". That's not the same
                    # event: label_generator.py's real check_continuation_outcome()
                    # requires a CLOSE beyond the break level (with an ATR buffer,
                    # entry-first ordering) before scoring a break. A live SL hit that
                    # never confirmed a real structural break (chopped, timed out,
                    # barely tagged SL) is not evidence the zone broke -- it's just
                    # evidence the reversion trade lost. We don't have the equivalent
                    # of backtest_replay.py's trace_post_sl_price_action() for live
                    # outcomes (outcome_tracker.py doesn't record post-SL price action),
                    # so there's no ground truth to assign here. Leave it NaN; the
                    # decisive_mask filter in prepare_features() will correctly drop
                    # these rows from CONTINUATION training while they still contribute
                    # normally to REVERSION training via the real 'label' column.
                    decisive["label_cont"] = np.nan
                    shared_cols = [c for c in decisive.columns if c in df.columns]
                    if shared_cols:
                        df = pd.concat([df, decisive[shared_cols]], ignore_index=True)
                        logger.info(f"✅ Merged {len(decisive)} real outcomes (pre-cutoff only)")
        except Exception as exc:
            logger.warning(f"Could not merge outcomes: {exc}")
    if "zone_time" in df.columns:
        df["zone_time"] = pd.to_datetime(df["zone_time"], errors="coerce")
        cutoff  = df["zone_time"].max() - pd.DateOffset(months=rolling_months)
        before  = len(df)
        df      = df[df["zone_time"] >= cutoff].reset_index(drop=True)
        logger.info(f"Rolling window ({rolling_months}m): {before} → {len(df)} rows (cutoff {cutoff.date()})")
    return df


def warm_start_xgb(new_model_path, existing_model_path, X_new, y_new, scaler):
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
    labels_path = Path("data/labels/training_labels_clean.parquet")
    if not labels_path.exists():
        labels_path = Path("data/labels/training_labels_dual.parquet")
        logger.info("Clean labels not found — using raw dual file")
    else:
        logger.info(f"Using preprocessed labels: {labels_path}")
    outcomes_path = Path("data/outcomes/outcomes.parquet")
    PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
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

    df = load_with_outcomes(labels_path=labels_path, outcomes_path=outcomes_path, rolling_months=300)
    logger.info(f"Training on {len(df):,} examples (synthetic + real outcomes)")
    results = train_both_tasks(df)

    print("\n" + "=" * 80)
    print("🎉 DUAL-TASK TRAINING COMPLETE!")
    print("=" * 80)

    results_path = Path("models/dual_training_results.json")
    with open(results_path, 'w') as f:
        def _convert(d):
            if isinstance(d, dict):   return {k: _convert(v) for k, v in d.items()}
            if isinstance(d, (np.floating, np.integer)): return float(d)
            if isinstance(d, list):   return [_convert(i) for i in d]
            return d
        json.dump(_convert(results), f, default=convert_numpy_types, indent=2)
    logger.info(f"✅ Results saved to {results_path}")


if __name__ == "__main__":
    main()