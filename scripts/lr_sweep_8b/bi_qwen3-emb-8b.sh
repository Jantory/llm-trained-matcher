#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-bi-q3-emb
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/lr_sweep_8b/logs/slurm-%j-%x.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/lr_sweep_8b/logs/slurm-%j-%x.err

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET="un"
MODEL="Qwen/Qwen3-Embedding-8B"
MODEL_SHORT="Qwen3-Embedding-8B"
PARADIGM="bi_encoder"
SEED=42
EVAL_BATCH_SIZE=256
MARGIN=0.3
WANDB_PROJECT="em-matcher-lr-sweep-8b"

LRS=(1e-5 3e-5 5e-5 1e-4 2e-4)

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch-shared/61600/em-matcher}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

# shellcheck source=../../.env
source "${PROJECT_ROOT}/.env"

export WANDB_PROJECT="${WANDB_PROJECT}"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Dataset   : ${DATASET}"
echo "Model     : ${MODEL}"
echo "Paradigm  : ${PARADIGM}"

# ── Run ────────────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

for LEARNING_RATE in "${LRS[@]}"; do
    OUTPUT_DIR="${CHECKPOINT_ROOT}/lr_sweep_8b/${PARADIGM}/${DATASET}/${MODEL_SHORT}_lr${LEARNING_RATE}"
    RESULTS_DIR="${PROJECT_ROOT}/results/lr_sweep_8b/${PARADIGM}/${DATASET}/${MODEL_SHORT}_lr${LEARNING_RATE}"

    echo "── LR: ${LEARNING_RATE} ──────────────────────────────────────────────"
    echo "Output    : ${OUTPUT_DIR}"
    echo "Results   : ${RESULTS_DIR}"

    "${PYTHON_BIN}" train_matcher.py \
        --dataset         "${DATASET}"         \
        --model           "${MODEL}"           \
        --paradigm        "${PARADIGM}"        \
        --seed            "${SEED}"            \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --learning_rate   "${LEARNING_RATE}"   \
        --margin          "${MARGIN}"          \
        --output_dir      "${OUTPUT_DIR}"      \
        --results_dir     "${RESULTS_DIR}"     \
        --wandb_project   "${WANDB_PROJECT}"
done
