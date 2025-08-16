#!/bin/bash
# Run multi-seed evaluation for NO-CURRICULUM experiments only

echo "🚀 Starting NO-CURRICULUM multi-seed evaluation..."
echo "This will run each deck type with 10 seeds (no curriculum learning)"
echo ""

python run_multi_seed_evaluation.py \
    --mode no-curriculum \
    --seeds 10 \
    --episodes 500000 \
    --eval-episodes 50000 \
    --deck-types 1-deck 4-deck 8-deck infinite

echo ""
echo "✅ NO-CURRICULUM evaluation completed!"
