#!/bin/bash
# Submit all 7 inference efficiency profiling jobs
# (3 paradigms x 8B, 3 paradigms x {4B,0.6B} combined, 1 baseline).
# Run from scripts/inference_efficiency/.
set -euo pipefail

# sbatch 8b_bi_encoder.sh
sbatch 8b_cross_encoder.sh
# sbatch 8b_generative.sh

# sbatch small_bi_encoder.sh
sbatch small_cross_encoder.sh
# sbatch small_generative.sh

# sbatch baseline_inference.sh
# sbatch oracle_length_compare.sh
