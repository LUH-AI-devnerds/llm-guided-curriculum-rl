#!/bin/bash
# Quick test script for multi-seed evaluation

echo "Running quick test with small parameters..."
python run_multi_seed_evaluation.py --test-run --deck-types 8-deck 4-deck 1-deck infinite
