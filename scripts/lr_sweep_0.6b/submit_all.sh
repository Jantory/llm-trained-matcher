#!/bin/bash
# Submit all 9 LR sweep array jobs.
# Usage: bash submit_all.sh  (run from scripts/lr_sweep_0.6b/)

set -euo pipefail

sbatch bi_qwen3-0.6b.sh
sbatch bi_qwen3-0.6b-base.sh
sbatch bi_qwen3-emb-0.6b.sh

sbatch cross_qwen3-0.6b.sh
sbatch cross_qwen3-0.6b-base.sh
sbatch cross_qwen3-emb-0.6b.sh

sbatch gen_qwen3-0.6b.sh
sbatch gen_qwen3-0.6b-base.sh
sbatch gen_qwen3-emb-0.6b.sh
