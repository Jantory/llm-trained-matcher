#!/bin/bash
# Submit all 5 cross-dataset (RQ5) inference jobs — one per seed.
# Each job runs all 3 paradigms × 9 sources × 8 targets, loading each base once.
# Usage: bash submit_all.sh  (run from scripts/cross_dataset/)

set -euo pipefail

for SEED in 42 123 456 789 2024; do
    sbatch "seed${SEED}.sh"
done
