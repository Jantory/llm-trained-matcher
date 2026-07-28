#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-inf-gen-8b
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/inf-gen-8b-%j.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/inf-gen-8b-%j.err

set -uo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PARADIGM="generative"
MODEL="Qwen/Qwen3-8B-Base"
SCALE="8b"
DATASETS=(ab ag ds rt sh sr wa w20 w80)
BATCH_INIT=$((1024 * 8))  # starting point; script halves automatically on OOM

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Paradigm   : ${PARADIGM}"
echo "Model      : ${MODEL}"
echo "Scale      : ${SCALE}"
echo "Datasets   : ${DATASETS[*]}"
echo "Batch init : ${BATCH_INIT} (auto-halved on OOM)"

cd "${PROJECT_ROOT}"

# ── Run all 9 datasets sequentially ───────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    OUTPUT="${PROJECT_ROOT}/results/inference_efficiency/${PARADIGM}/${SCALE}/${DATASET}.json"

    if [ -f "${OUTPUT}" ]; then
        echo "── ${DATASET}: skipping (already done)"
        continue
    fi

    mkdir -p "$(dirname "${OUTPUT}")"
    echo "── ${DATASET}: starting ────────────────────────────────────────"

    if "${PYTHON_BIN}" profile_throughput.py \
        --paradigm        "${PARADIGM}"  \
        --model           "${MODEL}"     \
        --dataset         "${DATASET}"   \
        --output          "${OUTPUT}"    \
        --batch_size_init "${BATCH_INIT}"; then
        echo "── ${DATASET}: done"
    else
        echo "── ${DATASET}: FAILED (exit $?) -- continuing to next dataset"
    fi
done

echo "All datasets complete for ${PARADIGM} ${SCALE}"
