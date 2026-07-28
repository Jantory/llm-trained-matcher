#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-oracle-len
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/oracle-len-%j.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/oracle-len-%j.err

set -uo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# Oracle length comparison (profile_oracle.py): bi_encoder vs cross_encoder vs
# generative cost as input length grows, across all 3 model scales. Isolates
# pure architecture cost from any dataset's text-length distribution.
MAX_LENS=(64 128 256 512 1024)
PARADIGMS=(bi_encoder cross_encoder generative)
SCALES=(0.6b 4b 8b)

declare -A MODEL_BI=(
    ["0.6b"]="Qwen/Qwen3-Embedding-0.6B"
    ["4b"]="Qwen/Qwen3-Embedding-4B"
    ["8b"]="Qwen/Qwen3-Embedding-8B"
)
declare -A MODEL_BASE=(
    ["0.6b"]="Qwen/Qwen3-0.6B-Base"
    ["4b"]="Qwen/Qwen3-4B-Base"
    ["8b"]="Qwen/Qwen3-8B-Base"
)

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Paradigms : ${PARADIGMS[*]}"
echo "Scales    : ${SCALES[*]}"
echo "Max lens  : ${MAX_LENS[*]}"

cd "${PROJECT_ROOT}"

# ── Run all paradigm x scale x length combos sequentially ────────────────────
for PARADIGM in "${PARADIGMS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
        if [ "${PARADIGM}" == "bi_encoder" ]; then
            MODEL="${MODEL_BI[$SCALE]}"
        else
            MODEL="${MODEL_BASE[$SCALE]}"
        fi

        for MAX_LEN in "${MAX_LENS[@]}"; do
            OUTPUT="${PROJECT_ROOT}/results/inference_efficiency/oracle/${PARADIGM}/${SCALE}_len${MAX_LEN}.json"

            if [ -f "${OUTPUT}" ]; then
                echo "── ${PARADIGM} ${SCALE} len=${MAX_LEN}: skipping (already done)"
                continue
            fi

            mkdir -p "$(dirname "${OUTPUT}")"
            echo "── ${PARADIGM} ${SCALE} len=${MAX_LEN} (model=${MODEL}): starting ──"

            if "${PYTHON_BIN}" profile_oracle.py \
                --paradigm "${PARADIGM}" \
                --model    "${MODEL}"    \
                --max_len  "${MAX_LEN}"  \
                --output   "${OUTPUT}"; then
                echo "── ${PARADIGM} ${SCALE} len=${MAX_LEN}: done"
            else
                echo "── ${PARADIGM} ${SCALE} len=${MAX_LEN}: FAILED (exit $?) -- continuing"
            fi
        done
    done
done

echo "All paradigm/scale/length combos complete"
