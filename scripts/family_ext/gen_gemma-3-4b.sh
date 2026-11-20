#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1-22:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=fm-g4-gen
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/family_ext/logs/slurm-%j-%x.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/family_ext/logs/slurm-%j-%x.err

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# Training protocol = main experiment: train_matcher.py defaults (--max_train_size 8000,
# --max_steps 400, --eval_steps 20; early stopping with patience 5), 5 seeds.
# Results go to results/family_ext_main/. The earlier runs in results/family_ext/ used
# --max_train_size 2500 --max_steps 150 --eval_steps 15 and are kept unchanged.
# Gemma-3-4B (text decoder extracted by extract_gemma3_text.py), generative.
# One entry per run: "<paradigm> <model> <learning_rate>". Learning rates are borrowed
# from the Qwen3 LR sweep at the nearest size (Qwen3-4B), per paradigm x variant.
RUNS=(
    "generative    /scratch-shared/61600/models/gemma3-text/gemma-3-4b-it 3e-4"
    "generative    /scratch-shared/61600/models/gemma3-text/gemma-3-4b-pt 1e-4"
    "generative    /scratch-shared/61600/models/gemma3-text/NetraEmbed    5e-5"
)
# Every variant renders generative prompts with the instruct checkpoint's chat template.
CHAT_TEMPLATE_FROM="/scratch-shared/61600/models/gemma3-text/gemma-3-4b-it"
# Generative training batch (effective batch 64 in all cases). Gemma's 262k vocabulary does not fit 64 sequences of logits; 32 x 2 accumulation.
GEN_BATCH_ARGS=(--batch_size 32 --gradient_accumulation_steps 2)

DATASETS=(ag wa sh sr w80)
# Seeds as in the main experiment. SEED_LIST overrides them to run a subset, e.g.
#   SEED_LIST="789 2024" sbatch --export=ALL <script>
# Never run two jobs of the same script on overlapping seeds at the same time.
read -r -a SEEDS <<< "${SEED_LIST:-42 123 456 789 2024}"
MARGIN=0.3
EXPERIMENT="family_ext_main"

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
export HF_HUB_OFFLINE=1  # all checkpoints are pre-downloaded into HF_HOME
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch-shared/61600/em-matcher}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

# shellcheck source=../../.env
source "${PROJECT_ROOT}/.env"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Grid      : ${#RUNS[@]} model/paradigm x ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds"
echo "Seeds     : ${SEEDS[*]}"

# ── Run ────────────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for RUN in "${RUNS[@]}"; do
            read -r PARADIGM MODEL LEARNING_RATE <<< "${RUN}"
            MODEL_SHORT="$(basename "${MODEL}")"
            OUTPUT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT}/${PARADIGM}/${DATASET}/${MODEL_SHORT}_seed${SEED}"
            RESULTS_DIR="${PROJECT_ROOT}/results/${EXPERIMENT}/${PARADIGM}/${DATASET}/${MODEL_SHORT}_seed${SEED}"

            # Eval batch sizes as in the main experiment.
            case "${PARADIGM}" in
                bi_encoder)    EXTRA_ARGS=(--eval_batch_size 256) ;;
                cross_encoder) EXTRA_ARGS=(--eval_batch_size 512) ;;
                generative)    EXTRA_ARGS=(--eval_batch_size 64 --chat_template_from "${CHAT_TEMPLATE_FROM}" "${GEN_BATCH_ARGS[@]}") ;;
            esac

            echo "── ${PARADIGM} ${MODEL_SHORT} ${DATASET} seed${SEED} lr${LEARNING_RATE} ──"

            if [ -f "${RESULTS_DIR}/test_results.json" ]; then
                echo "   Skipping: results already exist at ${RESULTS_DIR}"
                continue
            fi

            if "${PYTHON_BIN}" train_matcher.py \
                --dataset         "${DATASET}"         \
                --model           "${MODEL}"           \
                --paradigm        "${PARADIGM}"        \
                --seed            "${SEED}"            \
                --learning_rate   "${LEARNING_RATE}"   \
                --margin          "${MARGIN}"          \
                --output_dir      "${OUTPUT_DIR}"      \
                --results_dir     "${RESULTS_DIR}"     \
                --wandb_project   "em-${EXPERIMENT}-${PARADIGM}" \
                "${EXTRA_ARGS[@]}"; then
                echo "   Done"
            else
                echo "   FAILED (exit $?) -- will retry on re-submit"
            fi
        done
    done
done
