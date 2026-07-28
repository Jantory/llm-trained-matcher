#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-cr-shb-8b
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/sh_rebalance_ablation/logs/slurm-%j-%x.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/sh_rebalance_ablation/logs/slurm-%j-%x.err

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# sh dialect-combination rebalancing ablation: same 2068 labeled sh pairs,
# re-split so every (A-dialect x B-dialect) combination is spread
# proportionally across train/valid/test (~16.5% previously-unseen-combo
# positives in every split) instead of concentrated in test (82% unseen).
# Serialized tables are UNCHANGED from sh -- isolates the split-composition
# variable. LR / eval_batch / margin copied verbatim from the corresponding
# scripts/main_8b/*.sh.
MODEL="Qwen/Qwen3-8B-Base"
MODEL_SHORT="Qwen3-8B-Base"
PARADIGM="cross_encoder"
LEARNING_RATE=2e-4
EVAL_BATCH_SIZE=512
MARGIN=0.3
WANDB_PROJECT="em-sh-rebalance-ablation"

DATASETS=(shb)
SEEDS=(42 123 456 789 2024)

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch-shared/61600/em-matcher}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

# shellcheck source=../../.env
source "${PROJECT_ROOT}/.env"

export WANDB_PROJECT="${WANDB_PROJECT}"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Model     : ${MODEL}"
echo "Paradigm  : ${PARADIGM}"
echo "LR        : ${LEARNING_RATE}"

# ── Run ────────────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_DIR="${CHECKPOINT_ROOT}/sh_rebalance_ablation/${PARADIGM}/${DATASET}/${MODEL_SHORT}_seed${SEED}"
        RESULTS_DIR="${PROJECT_ROOT}/results/sh_rebalance_ablation/${PARADIGM}/${DATASET}/${MODEL_SHORT}_seed${SEED}"

        echo "── ${DATASET} seed${SEED} ────────────────────────────────────────"

        if [ -f "${RESULTS_DIR}/test_results.json" ]; then
            echo "   Skipping: results already exist at ${RESULTS_DIR}"
            continue
        fi

        if "${PYTHON_BIN}" train_matcher.py \
            --dataset         "${DATASET}"         \
            --model           "${MODEL}"           \
            --paradigm        "${PARADIGM}"        \
            --seed            "${SEED}"            \
            --eval_batch_size "${EVAL_BATCH_SIZE}" \
            --learning_rate   "${LEARNING_RATE}"   \
            --margin          "${MARGIN}"          \
            --output_dir      "${OUTPUT_DIR}"      \
            --results_dir     "${RESULTS_DIR}"     \
            --wandb_project   "${WANDB_PROJECT}"; then
            echo "   Done: ${DATASET} seed${SEED}"
        else
            echo "   FAILED: ${DATASET} seed${SEED} (exit $?) -- will retry on re-submit"
        fi
    done
done
