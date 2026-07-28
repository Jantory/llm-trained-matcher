#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=rome
#SBATCH --job-name=em-openai
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/baselines/logs/slurm-%j-%x.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/baselines/logs/slurm-%j-%x.err

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL="gpt-4.1-mini"
WANDB_PROJECT="em-openai-baseline"

DATASETS=(ab ag ds rt sh sr wa w20 w80)

# ── Environment ────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

# shellcheck source=../../.env
source "${PROJECT_ROOT}/.env"

export WANDB_PROJECT="${WANDB_PROJECT}"

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Model   : ${MODEL}"

cd "${PROJECT_ROOT}"

for DATASET in "${DATASETS[@]}"; do
    MODEL_SLUG="${MODEL//\//-}"
    RESULTS_DIR="${PROJECT_ROOT}/results/baselines/openai/${DATASET}/${MODEL_SLUG}"

    echo "── ${DATASET} ────────────────────────────────────────────────────"

    if [ -f "${RESULTS_DIR}/test_results.json" ]; then
        echo "   Skipping: results already exist at ${RESULTS_DIR}"
        continue
    fi

    if "${PYTHON_BIN}" run_openai_baseline.py \
        --dataset     "${DATASET}" \
        --model       "${MODEL}"   \
        --results_dir "${RESULTS_DIR}"; then
        echo "   Done: ${DATASET}"
    else
        echo "   FAILED: ${DATASET} (exit $?)"
    fi
done
