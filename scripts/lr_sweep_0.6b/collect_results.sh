#!/bin/bash
# Read training_log.json from each LR sweep run and print a summary table.
# Usage: bash scripts/lr_sweep_0.6b/collect_results.sh

PROJECT_ROOT="/gpfs/home6/zzhang/em-emb/a-train-matcher"
RESULTS_BASE="${PROJECT_ROOT}/results/lr_sweep_0.6b"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}"

BI_CROSS_LRS=(1e-5 3e-5 5e-5 1e-4 2e-4)
GEN_LRS=(3e-5 5e-5 1e-4 2e-4 3e-4)

printf "%-15s %-22s %-8s %8s %8s %8s\n" "Paradigm" "Model" "LR" "F1" "P" "R"
printf '%0.s-' {1..70}; echo

for paradigm in bi_encoder cross_encoder generative; do
    if [[ "$paradigm" == "generative" ]]; then
        lrs=("${GEN_LRS[@]}")
    else
        lrs=("${BI_CROSS_LRS[@]}")
    fi

    for model_short in Qwen3-0.6B Qwen3-0.6B-Base Qwen3-Embedding-0.6B; do
        for lr in "${lrs[@]}"; do
            json="${RESULTS_BASE}/${paradigm}/un/${model_short}_lr${lr}/training_log.json"
            if [[ -f "$json" ]]; then
                read -r f1 p r < <(
                    "${PYTHON_BIN}" - <<EOF
import json
d = json.load(open("$json"))
print(f"{d['test_f1']:.4f}", f"{d['test_precision']:.4f}", f"{d['test_recall']:.4f}")
EOF
                )
                printf "%-15s %-22s %-8s %8s %8s %8s\n" \
                    "$paradigm" "$model_short" "$lr" "$f1" "$p" "$r"
            else
                printf "%-15s %-22s %-8s %8s\n" \
                    "$paradigm" "$model_short" "$lr" "pending"
            fi
        done
    done
done
