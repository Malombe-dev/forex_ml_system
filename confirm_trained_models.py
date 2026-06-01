import os
import pickle

def section_header(title):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)

def inspect_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        print(f"❌ File not found: {scaler_path}")
        return
        
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        print(f"✓ Loaded: {scaler_path}")
        print(f"  • Expected Total Features: {getattr(scaler, 'n_features_in_', 'Unknown')}")
        
        if hasattr(scaler, "feature_names_in_"):
            print("  • Trained Features Sequence:")
            for idx, feature in enumerate(scaler.feature_names_in_):
                print(f"    {idx + 1:02d}. {feature}")
        else:
            print("  ⚠️ No 'feature_names_in_' metadata found in this scaler.")
    except Exception as e:
        print(f"❌ Error reading {scaler_path}: {e}")

def inspect_encoders(encoder_path):
    if not os.path.exists(encoder_path):
        print(f"❌ File not found: {encoder_path}")
        return

    try:
        with open(encoder_path, "rb") as f:
            encoders = pickle.load(f)
        
        print(f"✓ Loaded: {encoder_path}")
        if isinstance(encoders, dict):
            for col, encoder in encoders.items():
                classes = list(getattr(encoder, "classes_", []))
                print(f"  • Feature '{col}' -> Trained Classes: {classes}")
        else:
            print(f"  ⚠️ Unexpected structure in encoder file. Expected a dictionary, got {type(encoders)}.")
    except Exception as e:
        print(f"❌ Error reading {encoder_path}: {e}")

def inspect_model_component(model_path, model_label):
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    try:
        with open(model_path, "rb") as f:
            calibrated_wrapper = pickle.load(f)
            
        print(f"✓ Loaded {model_label}: {model_path}")
        
        if hasattr(calibrated_wrapper, "estimator"):
            base_est = calibrated_wrapper.estimator
            method = getattr(calibrated_wrapper, "method", "unknown")
            print(f"  • Calibration Wrapper: CalibratedClassifierCV (Method: '{method}')")
            print(f"  • Base Estimator Core: {type(base_est).__name__}")
        else:
            base_est = calibrated_wrapper
            print(f"  • Core Model Type (No Calibration Wrapper): {type(base_est).__name__}")
            
        est_name = type(base_est).__name__
        if "XGB" in est_name:
            print(f"    - max_depth: {getattr(base_est, 'max_depth', 'N/A')}")
            print(f"    - learning_rate: {getattr(base_est, 'learning_rate', 'N/A')}")
            print(f"    - n_estimators: {getattr(base_est, 'n_estimators', 'N/A')}")
            print(f"    - objective: {getattr(base_est, 'objective', 'N/A')}")
            print(f"    - subsample: {getattr(base_est, 'subsample', 'N/A')}")
        elif "RandomForest" in est_name:
            print(f"    - n_estimators: {getattr(base_est, 'n_estimators', 'N/A')}")
            print(f"    - criterion: {getattr(base_est, 'criterion', 'N/A')}")
            print(f"    - max_depth: {getattr(base_est, 'max_depth', 'N/A')}")
            print(f"    - random_state: {getattr(base_est, 'random_state', 'N/A')}")
        elif "LogisticRegression" in est_name:
            print(f"    - penalty: {getattr(base_est, 'penalty', 'N/A')}")
            print(f"    - solver: {getattr(base_est, 'solver', 'N/A')}")
            print(f"    - class_weight: {getattr(base_est, 'class_weight', 'N/A')}")
            print(f"    - max_iter: {getattr(base_est, 'max_iter', 'N/A')}")
        else:
            print("    - Hyperparameter parsing not customized for this architecture type.")
            
    except Exception as e:
        print(f"❌ Error reading model {model_path}: {e}")

def main():
    print("================================================================================")
    print("                 V5 FOREX PREDICTOR TRAINING VERIFICATION SCRIPT                ")
    print("================================================================================")

    # Base path prefixing models directory
    MPATH = "models"

    # 1. CONTINUATION STRATEGY ARTIFACTS
    section_header("1. V5 Continuation Pipeline Validation")
    inspect_scaler(os.path.join(MPATH, "scaler_v5_continuation.pkl"))
    print("-" * 50)
    inspect_encoders(os.path.join(MPATH, "encoders_v5_continuation.pkl"))
    print("-" * 50)
    inspect_model_component(os.path.join(MPATH, "model_v5_continuation.pkl"), "Ensemble Component 1 (XGBoost)")
    inspect_model_component(os.path.join(MPATH, "rf_v5_continuation.pkl"), "Ensemble Component 2 (Random Forest)")
    inspect_model_component(os.path.join(MPATH, "lr_v5_continuation.pkl"), "Ensemble Component 3 (Logistic Regression)")

    # 2. REVERSION STRATEGY ARTIFACTS
    section_header("2. V5 Reversion Pipeline Validation")
    inspect_scaler(os.path.join(MPATH, "scaler_v5_reversion.pkl"))
    print("-" * 50)
    inspect_encoders(os.path.join(MPATH, "encoders_v5_reversion.pkl"))
    print("-" * 50)
    inspect_model_component(os.path.join(MPATH, "model_v5_reversion.pkl"), "Ensemble Component 1 (XGBoost)")
    inspect_model_component(os.path.join(MPATH, "rf_v5_reversion.pkl"), "Ensemble Component 2 (Random Forest)")
    inspect_model_component(os.path.join(MPATH, "lr_v5_reversion.pkl"), "Ensemble Component 3 (Logistic Regression)")

if __name__ == "__main__":
    main()