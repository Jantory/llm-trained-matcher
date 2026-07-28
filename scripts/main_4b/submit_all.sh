#!/bin/bash
# Submit all 4 main experiment jobs.
# Usage: bash submit_all.sh  (run from scripts/main_4b/)

set -euo pipefail

# sbatch bi_qwen3-4b.sh
# sbatch bi_qwen3-4b-base.sh
# sbatch bi_qwen3-emb-4b.sh

# sbatch cross_qwen3-4b.sh
# sbatch cross_qwen3-4b-base.sh
# sbatch cross_qwen3-emb-4b.sh

sbatch gen_qwen3-4b.sh
sbatch gen_qwen3-4b-base.sh
sbatch gen_qwen3-emb-4b.sh
