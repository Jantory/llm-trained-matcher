#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-bift-ba-sh
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/ablation/logs/slurm-%j-%x.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/ablation/logs/slurm-%j-%x.err

set -euo pipefail

# Full FT ablation — RQ2 Hypothesis A vs B
# Variant : base  Dataset: sh  Scale: 0.6B

PARADIGM="bi_encoder"
MODEL="Qwen/Qwen3-0.6B-Base"
MODEL_SHORT="Qwen3-0.6B-Base"
DATASET="sh"
SEEDS=(42 123 456)
LEARNING_RATE=2e-5   # full FT: 1e-5~5e-5; LoRA used 1e-4
BATCH_SIZE=64
EVAL_BATCH_SIZE=256
WANDB_PROJECT="em-bi_encoder-fullft-ablation"

export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch-shared/61600/em-matcher}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"
source "${PROJECT_ROOT}/.env"
export WANDB_PROJECT="${WANDB_PROJECT}"

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Variant : base  Dataset: sh"

cd "${PROJECT_ROOT}"

for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="${CHECKPOINT_ROOT}/ablation_fullft/${PARADIGM}/${DATASET}/${MODEL_SHORT}_seed${SEED}"
    RESULTS_DIR="${PROJECT_ROOT}/results/ablation_fullft/${PARADIGM}/${DATASET}/${MODEL_SHORT}_seed${SEED}"

    echo "── seed${SEED} ────────────────────────────────────────"

    if [ -f "${RESULTS_DIR}/test_results.json" ]; then
        echo "   Skipping: results already exist"
        continue
    fi

    if "${PYTHON_BIN}" train_matcher.py         --dataset         "${DATASET}"                 --model           "${MODEL}"                   --paradigm        "${PARADIGM}"                --seed            "${SEED}"                    --batch_size      "${BATCH_SIZE}"              --eval_batch_size "${EVAL_BATCH_SIZE}"         --learning_rate   "${LEARNING_RATE}"           --no-lora                                      --output_dir      "${OUTPUT_DIR}"              --results_dir     "${RESULTS_DIR}"             --wandb_project   "${WANDB_PROJECT}"; then
        echo "   Done: seed${SEED}"
    else
        echo "   FAILED: seed${SEED} (exit $?)"
    fi
done
