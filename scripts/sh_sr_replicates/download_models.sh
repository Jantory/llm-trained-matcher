#!/bin/bash
# Make sure every checkpoint of the sh/sr replicate experiment is in the local cache.
# Run once on the login node before submitting (the job scripts run with HF_HUB_OFFLINE=1).
# Qwen3-0.6B-Base and Qwen3-4B-Base were no longer cached on 2026-11-08; the rest were.
# Re-running only fetches missing files.
set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch-shared/61600/.cache/huggingface}"
HF_BIN="/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/hf"
PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"
G4="/scratch-shared/61600/models/gemma3-text"

# shellcheck source=../../.env
source "${PROJECT_ROOT}/.env"  # HF_TOKEN (Gemma and Llama are gated)

# Trained checkpoints (full weights).
for MODEL in Qwen/Qwen3-0.6B-Base Qwen/Qwen3-4B-Base Qwen/Qwen3-8B-Base \
             google/gemma-3-270m meta-llama/Llama-3.1-8B; do
    echo "── ${MODEL}"
    "${HF_BIN}" download "${MODEL}" --exclude "*.pt" "*.bin" "*.gguf" "original/*" > /dev/null
done

# Chat-template sources for generative prompts (tokenizer files only).
for MODEL in google/gemma-3-270m-it meta-llama/Llama-3.1-8B-Instruct; do
    echo "── ${MODEL} (tokenizer only)"
    "${HF_BIN}" download "${MODEL}" --include "*.json" "*.model" "*.jinja" > /dev/null
done

# Gemma-3-4B text decoders, extracted locally (not downloadable).
for D in gemma-3-4b-pt gemma-3-4b-it; do
    if [ ! -f "${G4}/${D}/EXTRACTED_FROM" ]; then
        echo "MISSING ${G4}/${D}: rerun scripts/family_ext/extract_gemma3_text.py on a GPU node"
        exit 1
    fi
done
echo "All checkpoints are available locally (HF_HOME=${HF_HOME}, ${G4})"
