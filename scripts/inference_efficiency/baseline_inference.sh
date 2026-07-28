#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-baseline-inf
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/baseinf-%j.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/baseinf-%j.err

set -uo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# Baseline inference profiling — same script/methodology as the Qwen runs
# (profile_throughput.py): sbert (stsb-roberta-base-v2, bi deployment) and
# ditto cost proxy (roberta-base, per-pair classification forward).
# Single job: both models are small; all 9 datasets run sequentially.
MODEL_KINDS=(sbert ditto)
DATASETS=(ab ag ds rt sh sr wa w20 w80)

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"

# ── Run ────────────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

for KIND in "${MODEL_KINDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        OUT="${PROJECT_ROOT}/results/inference_efficiency/baselines/${KIND}/${DATASET}.json"

        echo "── ${KIND} ${DATASET} ────────────────────────────────────────"

        if [ -f "${OUT}" ]; then
            echo "   Skipping: ${OUT} exists"
            continue
        fi

        if "${PYTHON_BIN}" profile_throughput.py \
            --paradigm "${KIND}"      \
            --dataset  "${DATASET}"   \
            --output   "${OUT}"; then
            echo "   Done: ${KIND} ${DATASET}"
        else
            echo "   FAILED: ${KIND} ${DATASET} (exit $?) -- will retry on re-submit"
        fi
    done
done
