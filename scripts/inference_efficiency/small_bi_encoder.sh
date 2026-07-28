#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-inf-bi-sm
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/inf-bi-sm-%j.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/inference_efficiency/logs/inf-bi-sm-%j.err

set -uo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# 4B and 0.6B share a job: both are small/fast relative to the 8B scale, which
# gets its own dedicated job (run_8b_bi_encoder.sh).
PARADIGM="bi_encoder"
SCALES=(4b 0.6b)
declare -A MODEL_FOR_SCALE=(
    ["4b"]="Qwen/Qwen3-Embedding-4B"
    ["0.6b"]="Qwen/Qwen3-Embedding-0.6B"
)
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
echo "Scales     : ${SCALES[*]}"
echo "Datasets   : ${DATASETS[*]}"
echo "Batch init : ${BATCH_INIT} (auto-halved on OOM)"

cd "${PROJECT_ROOT}"

# ── Run both scales, each over all 9 datasets, sequentially ───────────────────
for SCALE in "${SCALES[@]}"; do
    MODEL="${MODEL_FOR_SCALE[$SCALE]}"
    echo "═══ scale=${SCALE} model=${MODEL} ═══════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        OUTPUT="${PROJECT_ROOT}/results/inference_efficiency/${PARADIGM}/${SCALE}/${DATASET}.json"

        if [ -f "${OUTPUT}" ]; then
            echo "── ${SCALE} ${DATASET}: skipping (already done)"
            continue
        fi

        mkdir -p "$(dirname "${OUTPUT}")"
        echo "── ${SCALE} ${DATASET}: starting ───────────────────────────────"

        if "${PYTHON_BIN}" profile_throughput.py \
            --paradigm        "${PARADIGM}"  \
            --model           "${MODEL}"     \
            --dataset         "${DATASET}"   \
            --output          "${OUTPUT}"    \
            --batch_size_init "${BATCH_INIT}"; then
            echo "── ${SCALE} ${DATASET}: done"
        else
            echo "── ${SCALE} ${DATASET}: FAILED (exit $?) -- continuing to next dataset"
        fi
    done
done

echo "All scales/datasets complete for ${PARADIGM}"
