#!/bin/bash
# Cleanup stale preprocessing artifacts (29-feature schema, pre-v5)
# These files are superseded by v5 artifacts and can cause confusion

echo "🧹 Cleaning up stale preprocessing artifacts..."

MODELS_DIR="models"

# Stale preprocessing artifacts (May 22, pre-v5)
STALE_FILES=(
    "$MODELS_DIR/scaler_preprocessing.pkl"
    "$MODELS_DIR/encoders_preprocessing.pkl"
)

for file in "${STALE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Removing stale: $file"
        rm "$file"
    else
        echo "  Already gone: $file"
    fi
done

# List what remains (should all be v5)
echo ""
echo "✅ Remaining model artifacts (v5):"
ls -1 "$MODELS_DIR"/*.pkl | xargs -I {} basename {} | sort

echo ""
echo "✅ Cleanup complete"
