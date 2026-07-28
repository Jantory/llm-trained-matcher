#!/bin/bash
# Submit all 6 title-collision rebalancing ablation jobs (3 scales x 2 paradigms).
# Each job runs dataset srb x 5 seeds internally. Run from scripts/rebalance_ablation/.
set -euo pipefail

sbatch bi_qwen3-0.6b-base.sh
# sbatch cr_qwen3-0.6b-base.sh
# sbatch gen_qwen3-0.6b-base.sh

sbatch bi_qwen3-4b-base.sh
# sbatch cr_qwen3-4b-base.sh
# sbatch gen_qwen3-4b-base.sh

sbatch bi_qwen3-8b-base.sh
# sbatch cr_qwen3-8b-base.sh
# sbatch gen_qwen3-8b-base.sh

