#!/bin/bash
# Collect main_8b results and report mean ± std F1 over 5 seeds.
# Usage: bash collect_results.sh  (run from scripts/main_8b/)

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"
RESULTS_BASE="${PROJECT_ROOT}/results/main_8b"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"

DATASETS=(ab ag ds rt sh sr wa w20 w80)
SEEDS=(42 123 456 789 2024)

declare -A COMBOS
COMBOS["bi_encoder/Qwen3-8B"]="bi_encoder Qwen3-8B"
COMBOS["bi_encoder/Qwen3-8B-Base"]="bi_encoder Qwen3-8B-Base"
COMBOS["bi_encoder/Qwen3-Embedding-8B"]="bi_encoder Qwen3-Embedding-8B"
COMBOS["cross_encoder/Qwen3-8B"]="cross_encoder Qwen3-8B"
COMBOS["cross_encoder/Qwen3-8B-Base"]="cross_encoder Qwen3-8B-Base"
COMBOS["cross_encoder/Qwen3-Embedding-8B"]="cross_encoder Qwen3-Embedding-8B"
COMBOS["generative/Qwen3-8B"]="generative Qwen3-8B"
COMBOS["generative/Qwen3-8B-Base"]="generative Qwen3-8B-Base"
COMBOS["generative/Qwen3-Embedding-8B"]="generative Qwen3-Embedding-8B"

COMBO_KEYS=(
    "bi_encoder/Qwen3-8B"
    "bi_encoder/Qwen3-8B-Base"
    "bi_encoder/Qwen3-Embedding-8B"
    "cross_encoder/Qwen3-8B"
    "cross_encoder/Qwen3-8B-Base"
    "cross_encoder/Qwen3-Embedding-8B"
    "generative/Qwen3-8B"
    "generative/Qwen3-8B-Base"
    "generative/Qwen3-Embedding-8B"
)

printf "%-6s  %-28s  %14s\n" "Dataset" "Paradigm / Model" "F1 (mean ± std)"
printf '%0.s-' {1..55}; echo

for KEY in "${COMBO_KEYS[@]}"; do
    read -r PARADIGM MODEL_SHORT <<< "${COMBOS[$KEY]}"
    for DATASET in "${DATASETS[@]}"; do
        F1S=()
        for SEED in "${SEEDS[@]}"; do
            json="${RESULTS_BASE}/${PARADIGM}/${DATASET}/${MODEL_SHORT}_seed${SEED}/training_log.json"
            if [[ -f "$json" ]]; then
                f1=$("${PYTHON_BIN}" -c "import json; print(json.load(open('$json'))['test_f1'])")
                F1S+=("$f1")
            fi
        done

        n="${#F1S[@]}"
        if [[ "$n" -eq 0 ]]; then
            printf "%-6s  %-28s  %14s\n" "$DATASET" "${PARADIGM}/${MODEL_SHORT}" "pending"
        else
            F1_CSV=$(IFS=,; echo "${F1S[*]}")
        stats=$("${PYTHON_BIN}" - <<EOF
fs = [${F1_CSV}]
n  = len(fs)
mu = sum(fs) / n
std = (sum((x - mu)**2 for x in fs) / n) ** 0.5
print(f"{mu:.4f} ± {std:.4f}  ({n}/5)")
EOF
)
            printf "%-6s  %-28s  %14s\n" "$DATASET" "${PARADIGM}/${MODEL_SHORT}" "$stats"
        fi
    done
    echo
done
