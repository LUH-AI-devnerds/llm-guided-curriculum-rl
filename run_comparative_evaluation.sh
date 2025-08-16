#!/bin/bash
# Run complete comparative evaluation (both curriculum and no-curriculum)

echo "🚀 Starting COMPARATIVE multi-seed evaluation..."
echo "This will run both curriculum and no-curriculum for all deck types"
echo "Phase 1: No-curriculum experiments"
echo "Phase 2: Curriculum experiments"
echo ""

python run_multi_seed_evaluation.py \
    --mode all \
    --seeds 10 \
    --episodes 500000 \
    --eval-episodes 50000 \
    --deck-types 1-deck 4-deck 8-deck infinite

echo ""
echo "✅ COMPARATIVE evaluation completed!"
