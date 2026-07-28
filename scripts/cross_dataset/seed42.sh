#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-cross-seed42
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/cross_dataset/logs/slurm-%j-%x.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/cross_dataset/logs/slurm-%j-%x.err

set -euo pipefail

# Cross-dataset (RQ5) inference for one seed, all 3 paradigms.
# The 8B base is loaded once per paradigm inside run_cross_dataset.py; source
# adapters are hot-swapped and target datasets tokenized once. Config is
# hard-coded in run_cross_dataset.py — only the seed is passed here.

export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch-shared/61600/em-matcher}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Seed   : 42"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" run_cross_dataset.py --seed 42
