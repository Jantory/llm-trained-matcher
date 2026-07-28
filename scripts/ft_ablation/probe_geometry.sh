#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-probe-geo
#SBATCH --output=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/ft_ablation/logs/slurm-%j-%x.out
#SBATCH --error=/gpfs/home6/zzhang/em-emb/a-train-matcher/scripts/ft_ablation/logs/slurm-%j-%x.err

set -euo pipefail

SCALE="${SCALE:-0.6b}"

export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"
PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"

echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Scale   : ${SCALE}"
echo "HF_HOME : ${HF_HOME}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" probe_geometry.py \
    --scale   "${SCALE}" \
    --output  "results/geometry_probe_${SCALE}.csv"
