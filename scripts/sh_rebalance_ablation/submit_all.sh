#!/bin/bash
# Submit all 9 sh dialect-combination rebalancing ablation jobs
# (3 scales x 3 paradigms). Each job runs dataset shb x 5 seeds
# internally. Run from scripts/sh_rebalance_ablation/.
set -euo pipefail

sbatch cr_qwen3-0.6b-base.sh
sbatch gen_qwen3-0.6b-base.sh
sbatch bi_qwen3-0.6b-base.sh

sbatch cr_qwen3-4b-base.sh
sbatch gen_qwen3-4b-base.sh
sbatch bi_qwen3-4b-base.sh

sbatch cr_qwen3-8b-base.sh
sbatch gen_qwen3-8b-base.sh
sbatch bi_qwen3-8b-base.sh

