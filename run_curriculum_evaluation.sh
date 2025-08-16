#!/bin/bash
# Run multi-seed evaluation for CURRICULUM experiments only

echo "🚀 Starting CURRICULUM multi-seed evaluation..."
echo "This will run each deck type with 10 seeds (with curriculum learning)"
echo ""

python run_multi_seed_evaluation.py \
    --mode curriculum \
    --seeds 10 \
    --episodes 500000 \
    --eval-episodes 100000 \
    --deck-types 8-deck \
    --max-episodes-per-stage 100000


echo ""
echo "✅ CURRICULUM evaluation completed!"
