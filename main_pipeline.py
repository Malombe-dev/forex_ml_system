# main_pipeline.py
"""
MAIN PIPELINE ORCHESTRATOR
===========================
Runs the complete end-to-end forex ML system in the correct order.

Phases
------
  1. Data ingestion      tick_ingestion.py  → data/bars_*.parquet
  2. Zone detection      smc_detector.py    → data/features/merged_zones.parquet
  3. Label generation    label_generator.py → data/labels/training_labels_dual.parquet
  4. Model training      model_trainer.py   → models/*.pkl
  5. Backtesting         backtest_engine.py → data/backtest_results/
  6. Live signals        production_predictor.py  (run separately / continuously)

Usage
-----
    python main_pipeline.py data/EURUSD.csv           # full run
    python main_pipeline.py data/EURUSD.csv --skip-training
    python main_pipeline.py data/EURUSD.csv --phase 3  # run from phase 3 onwards
"""

import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)
def get_pip_size(sym: str) -> float:
    sym_upper = sym.upper()
    if "JPY" in sym_upper:
        return 0.01
    elif "XAU" in sym_upper or "GOLD" in sym_upper:
        return 1.0
    return 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Individual phase runners
# ─────────────────────────────────────────────────────────────────────────────

def phase1_ingest(csv_path: str) -> bool:
    """Tick ingestion → bar parquets."""
    logger.info("PHASE 1 — DATA INGESTION")
    try:
        from tick_ingestion import TickDataIngestion
        ingestion = TickDataIngestion(output_dir="data")
        ticks     = ingestion.load_dukascopy_csv(csv_path)
        ingestion.save_ticks(ticks)
        ingestion.generate_all_timeframes(ticks)
        logger.info("✅  Phase 1 complete")
        return True
    except Exception as exc:
        logger.error(f"❌  Phase 1 failed: {exc}")
        return False


SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]


def phase2_features(symbols: list = None) -> bool:
    """SMC zone detection (multi-timeframe) → merged_zones_{SYMBOL}.parquet, per symbol."""
    logger.info("PHASE 2 — FEATURE ENGINEERING (SMC detection, MTF, per-symbol)")
    symbols = symbols or SYMBOLS
    try:
        import pandas as pd
        from smc_detector import SMCDetector

        out = Path("data/features")
        out.mkdir(parents=True, exist_ok=True)
        any_ok = False

        for sym in symbols:
            path_15m = Path(f"data/bars_15min_{sym}.parquet")
            if not path_15m.exists():
                logger.warning(f"  {sym}: no 15min bars at {path_15m} — skipping")
                continue

            df_15 = pd.read_parquet(path_15m)
            if "time" not in df_15.columns:
                df_15 = df_15.reset_index().rename(columns={"index": "time"})
            df_15["time"] = pd.to_datetime(df_15["time"])

            # Load or resample 4H/1H; load D1 if present (optional)
            def _load_or_resample(tf_path: Path, tf_label: str):
                if tf_path.exists():
                    d = pd.read_parquet(tf_path)
                    if "time" not in d.columns:
                        d = d.reset_index().rename(columns={"index": "time"})
                    d["time"] = pd.to_datetime(d["time"])
                    return d
                logger.info(f"  {sym}: {tf_label} bars not found — resampling from 15min")
                d = SMCDetector.resample_bars(df_15, tf_label)
                d.to_parquet(tf_path, index=False)
                return d

            df_4h = _load_or_resample(Path(f"data/bars_4h_{sym}.parquet"), "4h")
            df_1h = _load_or_resample(Path(f"data/bars_1h_{sym}.parquet"), "1h")

            df_d1 = None
            for p in [Path(f"data/bars_1d_{sym}.parquet"), Path(f"data/bars_d1_{sym}.parquet")]:
                if p.exists():
                    df_d1 = pd.read_parquet(p)
                    if "time" not in df_d1.columns:
                        df_d1 = df_d1.reset_index().rename(columns={"index": "time"})
                    df_d1["time"] = pd.to_datetime(df_d1["time"])
                    break
            if df_d1 is None:
                logger.warning(f"  {sym}: D1 bars not found — htf_alignment will be 0 for all zones")

            detector = SMCDetector(pip_size=get_pip_size(sym))
            features = detector.extract_all_features_mtf(
                df_4h=df_4h, df_1h=df_1h, df_15min=df_15, df_d1=df_d1
            )

            for name, lst in features.items():
                if lst:
                    pd.DataFrame(lst).to_parquet(out / f"{name}_{sym}.parquet")

            logger.info(f"  ✅ {sym}: {len(features.get('merged_zones', []))} merged zones")
            any_ok = True

        if not any_ok:
            logger.error("❌  Phase 2 failed: no symbol produced output (check data/bars_15min_{SYMBOL}.parquet)")
            return False
        logger.info("✅  Phase 2 complete")
        return True
    except Exception as exc:
        logger.error(f"❌  Phase 2 failed: {exc}")
        return False


def phase3_labels(symbols: list = None) -> bool:
    """Dual label generation → training_labels_{SYMBOL}.parquet, per symbol."""
    logger.info("PHASE 3 — LABEL GENERATION (per-symbol, MTF)")
    symbols = symbols or SYMBOLS
    try:
        import pandas as pd
        from label_generator import DualLabelGenerator

        out = Path("data/labels")
        out.mkdir(parents=True, exist_ok=True)
        all_frames = []

        for sym in symbols:
            path_15m = Path(f"data/bars_15min_{sym}.parquet")
            zones_path = Path(f"data/features/merged_zones_{sym}.parquet")
            if not path_15m.exists() or not zones_path.exists():
                logger.warning(f"  {sym}: missing bars or zones — skipping "
                               f"(need {path_15m} and {zones_path})")
                continue

            df_15 = pd.read_parquet(path_15m)
            if "time" not in df_15.columns:
                df_15 = df_15.reset_index().rename(columns={"index": "time"})
            df_15["time"] = pd.to_datetime(df_15["time"])

            zones_df = pd.read_parquet(zones_path)
            zones = zones_df.to_dict("records")

            def _load(tf_path: Path):
                if not tf_path.exists():
                    return None
                d = pd.read_parquet(tf_path)
                if "time" not in d.columns:
                    d = d.reset_index().rename(columns={"index": "time"})
                d["time"] = pd.to_datetime(d["time"])
                return d

            df_4h = _load(Path(f"data/bars_4h_{sym}.parquet"))
            df_1h = _load(Path(f"data/bars_1h_{sym}.parquet"))
            df_d1 = _load(Path(f"data/bars_1d_{sym}.parquet"))
            if df_d1 is None:
                df_d1 = _load(Path(f"data/bars_d1_{sym}.parquet"))

            generator = DualLabelGenerator(
                risk_reward            = 1.5,
                horizon_bars           = 60,
                sl_atr_multiple        = 1.5,
                continuation_break_atr = 1.0,
                pip_size               = get_pip_size(sym),
            )

            labels_df = generator.generate_labels_from_zones(
                zones, df_15, df_d1=df_d1, df_1h=df_1h, df_4h=df_4h
            )

            if labels_df.empty:
                logger.warning(f"  {sym}: no labels generated — check zones and bar data")
                continue

            labels_df = generator.add_context_features(labels_df, df_15)
            sym_path = out / f"training_labels_{sym}.parquet"
            generator.save_labels(labels_df, str(sym_path))
            logger.info(f"  ✅ {sym}: {len(labels_df)} labels → {sym_path}")
            all_frames.append(labels_df)

        if not all_frames:
            logger.error("❌  Phase 3 failed: no symbol produced labels")
            return False

        # Combined dual-label file for downstream phases that expect it
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet("data/labels/training_labels_dual.parquet")
        logger.info(f"✅  Phase 3 complete — {len(combined)} total labels across "
                    f"{len(all_frames)} symbol(s)")
        return True
    except Exception as exc:
        logger.error(f"❌  Phase 3 failed: {exc}")
        return False


def phase35_validate() -> bool:
    """Data validation — schema, freshness, label distribution, risk_pips."""
    logger.info("PHASE 3.5 — DATA VALIDATION")
    try:
        from data_validation import validate
        validate("data/labels/training_labels_dual.parquet")
        logger.info("✅  Phase 3.5 complete")
        return True
    except Exception as exc:
        logger.error(f"❌  Phase 3.5 failed: {exc}")
        return False


def phase36_preprocess() -> bool:
    """Data preprocessing — clip outliers, filter stale zones, encode, scale."""
    logger.info("PHASE 3.6 — DATA PREPROCESSING")
    try:
        from data_preprocessing import preprocess
        preprocess(
            input_path  = "data/labels/training_labels_dual.parquet",
            output_path = "data/labels/training_labels_clean.parquet",
        )
        logger.info("✅  Phase 3.6 complete")
        return True
    except Exception as exc:
        logger.error(f"❌  Phase 3.6 failed: {exc}")
        return False


def phase4_train(skip_if_exists: bool = False) -> bool:
    """Model training → models/*.pkl."""
    logger.info("PHASE 4 — MODEL TRAINING")

    model_exists = (Path("models/model_v5_reversion.pkl").exists() and
                    Path("models/model_v5_continuation.pkl").exists())

    if skip_if_exists and model_exists:
        logger.info("✅  Phase 4 skipped — models already exist")
        return True

    try:
        import pandas as pd
        from model_trainer import train_both_tasks, load_with_outcomes

        labels_path   = Path("data/labels/training_labels_dual.parquet")
        outcomes_path = Path("data/outcomes/outcomes.parquet")

        # Prefer the clean (preprocessed) parquet; fall back to raw
        clean_path = Path("data/labels/training_labels_clean.parquet")
        if clean_path.exists():
            labels_path = clean_path
            logger.info(f"Using preprocessed parquet: {labels_path}")
        else:
            logger.warning(
                "Clean parquet not found — using raw labels. "
                "Run phase 3.5 and 3.6 for best results."
            )

        df = load_with_outcomes(
            labels_path    = labels_path,
            outcomes_path  = outcomes_path,
            rolling_months = 300,
        )

        logger.info(f"Training on {len(df)} examples")
        train_both_tasks(df)

        logger.info("✅  Phase 4 complete")
        return True
    except Exception as exc:
        logger.error(f"❌  Phase 4 failed: {exc}")
        return False


def phase5_backtest() -> bool:
    """Backtest saved signals against bar data."""
    logger.info("PHASE 5 — BACKTESTING")
    try:
        from backtest_engine import (
            BacktestEngine,
            load_signals_from_json,
            load_bars,
            print_report,
            save_results,
        )

        signals = load_signals_from_json()
        bars    = load_bars()

        engine  = BacktestEngine(initial_balance=10_000, risk_per_trade_pct=1.0)
        metrics = engine.run(signals, bars)

        print_report(metrics, engine)
        save_results(metrics, tag="pipeline")

        logger.info("✅  Phase 5 complete")
        return True
    except FileNotFoundError as exc:
        logger.warning(f"⚠️   Phase 5 skipped — {exc}")
        return True      # not fatal — signals may not exist yet
    except Exception as exc:
        logger.error(f"❌  Phase 5 failed: {exc}")
        return False


def phase6_outcomes() -> bool:
    """Resolve any pending signal outcomes (run after trading session)."""
    logger.info("PHASE 6 — OUTCOME TRACKING")
    try:
        from outcome_tracker import resolve_pending_signals, print_summary
        n = resolve_pending_signals()
        logger.info(f"Resolved {n} new outcome(s)")
        print_summary()
        logger.info("✅  Phase 6 complete")
        return True
    except Exception as exc:
        logger.error(f"❌  Phase 6 failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(csv_path: str,
                 start_phase: int   = 1,
                 skip_training: bool = False,
                 run_backtest: bool  = True,
                 run_outcomes: bool  = False) -> bool:

    logger.info("=" * 70)
    logger.info("FOREX ML TRADING SYSTEM — FULL PIPELINE")
    logger.info("=" * 70)

    phases = {
        1: lambda: phase1_ingest(csv_path),
        2: phase2_features,
        3: phase3_labels,
        35: phase35_validate,
        36: phase36_preprocess,
        4: lambda: phase4_train(skip_if_exists=skip_training),
        5: phase5_backtest if run_backtest else lambda: True,
        6: phase6_outcomes if run_outcomes else lambda: True,
    }

    for num in sorted(phases.keys()):
        if num < start_phase:
            continue
        fn = phases.get(num)
        if fn is None:
            continue
        ok = fn()
        if not ok:
            logger.error(f"Pipeline stopped at phase {num}.")
            return False

    logger.info("=" * 70)
    logger.info("🎉  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  • Run live:     python production_predictor.py")
    logger.info("  • Track Telegram: python telegram_signal_monitor.py")
    logger.info("  • Daily outcomes: python outcome_tracker.py")
    logger.info("  • Weekly retrain: python main_pipeline.py --phase 4")
    logger.info("")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Forex ML Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Full pipeline from raw CSV:
    python main_pipeline.py data/EURUSD.csv

  Re-train only (labels already exist):
    python main_pipeline.py dummy --phase 4

  Labels → train → backtest:
    python main_pipeline.py dummy --phase 3

  Run outcome tracker after session:
    python main_pipeline.py dummy --phase 6
        """,
    )
    parser.add_argument("csv_path",       nargs="?", default="",
                        help="Path to Dukascopy tick CSV (required for phase 1)")
    parser.add_argument("--phase",        type=int, default=1,
                        help="Start from this phase (1-6, default 1)")
    parser.add_argument("--skip-training",action="store_true",
                        help="Skip training if model .pkl files already exist")
    parser.add_argument("--no-backtest",  action="store_true",
                        help="Skip phase 5 backtest")
    parser.add_argument("--with-outcomes",action="store_true",
                        help="Also run outcome tracker (phase 6)")

    args = parser.parse_args()

    if args.phase == 1 and not args.csv_path:
        parser.error("csv_path is required when starting from phase 1")

    if args.phase > 1 and not args.csv_path:
        args.csv_path = "N/A"   # won't be used

    csv = args.csv_path
    if args.phase == 1 and not Path(csv).exists():
        logger.error(f"CSV not found: {csv}")
        sys.exit(1)

    ok = run_pipeline(
        csv_path       = csv,
        start_phase    = args.phase,
        skip_training  = args.skip_training,
        run_backtest   = not args.no_backtest,
        run_outcomes   = args.with_outcomes,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "=" * 70)
        print("  FOREX ML PIPELINE — quick reference")
        print("=" * 70)
        print("  Full run:           python main_pipeline.py path/to/ticks.csv")
        print("  From labels:        python main_pipeline.py dummy --phase 3")
        print("  Retrain only:       python main_pipeline.py dummy --phase 4")
        print("  Backtest only:      python main_pipeline.py dummy --phase 5")
        print("  Outcome tracking:   python main_pipeline.py dummy --phase 6")
        print("=" * 70 + "\n")
    else:
        main()