"""
Diagnose model files and training stats structure
"""

import pickle
from pathlib import Path

def diagnose_model_files():
    """Check the structure of existing model files"""
    
    print("\n" + "="*80)
    print("MODEL FILES DIAGNOSTIC")
    print("="*80)
    
    models_dir = Path("models")
    
    # Check for model files
    for task in ['reversion', 'continuation']:
        print(f"\n{'='*80}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*80}")
        
        stats_path = models_dir / f"training_stats_v4_{task}.pkl"
        
        if not stats_path.exists():
            print(f"❌ File not found: {stats_path}")
            continue
        
        try:
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            
            print(f"\n✅ Loaded training stats from: {stats_path}")
            print(f"\nType: {type(stats)}")
            
            if isinstance(stats, dict):
                print(f"Number of keys: {len(stats)}")
                print(f"\nFirst 5 keys:")
                for i, key in enumerate(list(stats.keys())[:5]):
                    print(f"  {i+1}. {key}")
                
                # Check structure
                first_key = list(stats.keys())[0]
                first_value = stats[first_key]
                
                print(f"\nFirst entry structure:")
                print(f"  Key: {first_key}")
                print(f"  Value type: {type(first_value)}")
                print(f"  Value: {first_value}")
                
                # Check if it's the OLD structure (mean/std at top level)
                if 'mean' in stats and 'std' in stats:
                    print("\n⚠️  OLD STRUCTURE DETECTED!")
                    print("   This has 'mean' and 'std' as top-level keys")
                    print("   Expected structure: {feature_name: {'mean': val, 'std': val}}")
                    print("\n   FIX: Retrain models using model_trainer_dual_FIXED.py")
                    
                    # Show what it looks like
                    if isinstance(stats['mean'], dict):
                        print(f"\n   Current structure:")
                        print(f"     stats['mean'] = {dict(list(stats['mean'].items())[:3])}...")
                        print(f"     stats['std'] = {dict(list(stats['std'].items())[:3])}...")
                
                # Check if it's the NEW structure (feature_name: {mean, std})
                elif isinstance(first_value, dict) and 'mean' in first_value and 'std' in first_value:
                    print("\n✅ CORRECT STRUCTURE!")
                    print("   Each feature has its own dict with 'mean', 'std', etc.")
                    print(f"\n   Sample entry:")
                    print(f"     '{first_key}': {first_value}")
                else:
                    print("\n❓ UNKNOWN STRUCTURE")
                    print(f"   First value: {first_value}")
            
            else:
                print(f"❌ Expected dict, got {type(stats)}")
        
        except Exception as e:
            print(f"❌ Error loading file: {e}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
If you see "OLD STRUCTURE DETECTED" above:
  1. Run: python model_trainer_dual_FIXED.py
  2. This will retrain both models with correct stats structure
  3. Then your predictor will work without errors

If you see "CORRECT STRUCTURE":
  ✅ Your models are fine! The issue is elsewhere.
    """)
    print("="*80 + "\n")

if __name__ == "__main__":
    diagnose_model_files()