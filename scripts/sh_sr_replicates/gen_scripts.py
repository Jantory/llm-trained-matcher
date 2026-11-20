"""Generate the Slurm scripts for the sh/sr replicate experiment (8 load-balanced jobs).

Datasets: the two independent replicates of sh and sr built by
data/create_{sh,sr}_replicates.py, each in its imbalanced version (sh1, sh2, sr1,
sr2: same split structure as the original) and its balanced control (sh1b, sh2b,
sr1b, sr2b: same pairs, proportional re-split).

Models: the Base checkpoint of every family and size (Base is the best
cross/gen variant in the main experiment):
  Qwen3-{0.6B,4B,8B}-Base, gemma-3-270m, gemma-3-4b-pt (text decoder extracted by
  scripts/family_ext/extract_gemma3_text.py), Llama-3.1-8B.
Paradigms: cross_encoder and generative. 6 models x 2 paradigms x 8 datasets x
5 seeds = 480 runs.

Protocol = main experiment: train_matcher.py defaults (--max_train_size 8000, a no-op
here since train has 1240/1309 pairs; --max_steps 400; --eval_steps 20; early stopping
with patience 5), seeds 42 123 456 789 2024. Results go to results/sh_sr_replicates_main/.
(An earlier version of these scripts used --max_train_size 2500 --max_steps 150
--eval_steps 15 and wrote to results/sh_sr_replicates/.) Per-model settings are copied
from earlier runs:
  * Qwen3: LR / eval batch from scripts/main_<scale>/{cross,gen}_qwen3-<scale>-base.sh;
    generative uses the tokenizer's own chat template, as in the main runs.
  * Gemma-3 / Llama-3.1: LR, chat template (from the instruct sibling) and generative
    batch (Gemma: 32 x 2 accumulation) from scripts/family_ext/.

Load balancing: the (model, paradigm, dataset) work is packed into 8 jobs of
equal length (~3.3 h per seed), using estimated per-run minutes (COST_MIN). A job
may hold several models; each run carries its family's settings and dataset list.
The script prints each job's estimate and checks that every (model, paradigm,
dataset) is assigned exactly once. Jobs loop seed -> dataset -> run.
DATASET_LIST / SEED_LIST override the defaults to run a subset, e.g.
    DATASET_LIST="sr1 sr1b" SEED_LIST="42 123" sbatch --export=ALL <script>

Run:  /gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python scripts/sh_sr_replicates/gen_scripts.py
"""
import math
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/gpfs/home6/zzhang/em-emb/a-train-matcher"
EXPERIMENT = "sh_sr_replicates_main"
DATASETS = ["sh1", "sh2", "sh1b", "sh2b", "sr1", "sr2", "sr1b", "sr2b"]
SH_SET = ["sh1", "sh2", "sh1b", "sh2b"]
SR_SET = ["sr1", "sr2", "sr1b", "sr2b"]
SEEDS = "42 123 456 789 2024"  # as in the main experiment
N_SEEDS = len(SEEDS.split())
G4 = "/scratch-shared/61600/models/gemma3-text"
TIME_MARGIN = 1.3  # Slurm time limit = estimate x margin, rounded up to whole hours

CR, GEN = "cross_encoder", "generative"
# key -> (checkpoint, family, LR per paradigm)
MODELS = {
    "q06":  ("Qwen/Qwen3-0.6B-Base",        "qwen",     {CR: "2e-4", GEN: "2e-4"}),
    "q4":   ("Qwen/Qwen3-4B-Base",          "qwen",     {CR: "2e-4", GEN: "1e-4"}),
    "q8":   ("Qwen/Qwen3-8B-Base",          "qwen",     {CR: "2e-4", GEN: "3e-4"}),
    "g270": ("google/gemma-3-270m",         "gemma270", {CR: "2e-4", GEN: "2e-4"}),
    "g4":   (f"{G4}/gemma-3-4b-pt",         "gemma4",   {CR: "2e-4", GEN: "1e-4"}),
    "l8":   ("meta-llama/Llama-3.1-8B",     "llama",    {CR: "2e-4", GEN: "3e-4"}),
}
# Estimated minutes per run (sh-size, sr-size) under the main protocol, incl. model loading.
# Qwen3: mean runtime of the same model on sh/sr in results/main_*. Gemma/Llama: per-step
# time measured in results/family_ext x mean steps-to-early-stop of the same-size Qwen3-Base
# on sh/sr in results/main_*, plus validation/test.
COST_MIN = {
    ("q06", CR): (6.1, 8.1),   ("q06", GEN): (4.9, 8.6),
    ("q4", CR): (14.0, 23.8),  ("q4", GEN): (14.5, 23.2),
    ("q8", CR): (17.7, 36.9),  ("q8", GEN): (18.0, 31.6),
    ("g270", CR): (3.3, 4.2),  ("g270", GEN): (3.0, 5.0),
    ("g4", CR): (15.0, 24.9),  ("g4", GEN): (14.7, 23.2),
    ("l8", CR): (16.1, 35.1),  ("l8", GEN): (16.5, 30.1),
}


def all_but(*excluded):
    return [d for d in DATASETS if d not in excluded]


# (script name, Slurm job name, [(model key, paradigm, datasets)])
JOBS = [
    ("job1_qwen3-8b-cross.sh",     "repm1", [("q8", CR, all_but("sh2b"))]),
    ("job2_llama-3.1-8b-cross.sh", "repm2", [("l8", CR, DATASETS)]),
    ("job3_qwen3-8b-gen.sh",       "repm3", [("q8", GEN, DATASETS)]),
    ("job4_llama-3.1-8b-gen.sh",   "repm4", [("l8", GEN, DATASETS), ("q8", CR, ["sh2b"])]),
    ("job5_gemma-3-4b-cross.sh",   "repm5", [("g4", CR, DATASETS), ("g270", GEN, DATASETS)]),
    ("job6_qwen3-4b-cross.sh",     "repm6", [("q4", CR, DATASETS), ("g270", CR, DATASETS), ("q06", CR, SH_SET)]),
    ("job7_gemma-3-4b-gen.sh",     "repm7", [("g4", GEN, DATASETS), ("q06", GEN, SR_SET)]),
    ("job8_qwen3-4b-gen.sh",       "repm8", [("q4", GEN, DATASETS), ("q06", CR, SR_SET), ("q06", GEN, SH_SET)]),
]

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time={time}
#SBATCH --partition=gpu_h100
#SBATCH --job-name={job_name}
#SBATCH --output={here}/logs/slurm-%j-%x.out
#SBATCH --error={here}/logs/slurm-%j-%x.err

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# sh/sr replicate experiment, job {job_name} of {n_jobs} (generated by gen_scripts.py -- edit
# that, not this file). Estimated {est_h:.1f} h for {n_runs} runs.
# Datasets: independent replicates of sh/sr (data/create_{{sh,sr}}_replicates.py);
# sh1/sh2/sr1/sr2 keep the original imbalance, *b are the balanced controls.
# Training protocol = main experiment (train_matcher.py defaults: --max_train_size 8000,
# --max_steps 400, --eval_steps 20; early stopping with patience 5).
# One entry per model/paradigm: "<paradigm> <model> <lr> <family> <datasets, comma-separated>".
RUNS=(
{runs}
)
G4="{g4}"

# DATASET_LIST / SEED_LIST restrict the defaults to a subset, e.g.
#   DATASET_LIST="sr1 sr1b" SEED_LIST="42 123" sbatch --export=ALL <script>
# Never run two jobs of the same script on overlapping dataset/seed pairs at the same time.
read -r -a DATASETS <<< "${{DATASET_LIST:-{datasets}}}"
read -r -a SEEDS <<< "${{SEED_LIST:-{seeds}}}"
MARGIN=0.3
EXPERIMENT="{experiment}"

# Family settings for generative runs: chat template source (empty = the model's own
# template) and training batch (effective batch 64 in all cases).
family_settings() {{
    case "$1" in
        qwen)     CHAT_TEMPLATE_FROM="";                                  GEN_BATCH_ARGS=() ;;
        gemma270) CHAT_TEMPLATE_FROM="google/gemma-3-270m-it";            GEN_BATCH_ARGS=(--batch_size 32 --gradient_accumulation_steps 2) ;;
        gemma4)   CHAT_TEMPLATE_FROM="${{G4}}/gemma-3-4b-it";              GEN_BATCH_ARGS=(--batch_size 32 --gradient_accumulation_steps 2) ;;
        llama)    CHAT_TEMPLATE_FROM="meta-llama/Llama-3.1-8B-Instruct";  GEN_BATCH_ARGS=() ;;
        *)        echo "unknown family $1"; exit 1 ;;
    esac
}}

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${{HF_HOME:-/scratch-shared/61600/.cache/huggingface}}"
export HF_HUB_OFFLINE=1  # checkpoints are pre-downloaded by download_models.sh
export CHECKPOINT_ROOT="${{CHECKPOINT_ROOT:-/scratch-shared/61600/em-matcher}}"
PYTHON_BIN="${{PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="{project_root}"

# shellcheck source=../../.env
source "${{PROJECT_ROOT}}/.env"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID    : ${{SLURM_JOB_ID}}"
echo "Node      : $(hostname)"
echo "Datasets  : ${{DATASETS[*]}}"
echo "Seeds     : ${{SEEDS[*]}}"
printf 'Run       : %s\\n' "${{RUNS[@]}}"

cd "${{PROJECT_ROOT}}"

# ── Preflight: every checkpoint must already be local (else run download_models.sh) ──
CHECK=()
for RUN in "${{RUNS[@]}}"; do
    read -r _ MODEL _ FAMILY _ <<< "${{RUN}}"
    family_settings "${{FAMILY}}"
    CHECK+=("${{MODEL}}")
    if [ -n "${{CHAT_TEMPLATE_FROM}}" ]; then CHECK+=("${{CHAT_TEMPLATE_FROM}}"); fi
done
for M in "${{CHECK[@]}}"; do
    if ! "${{PYTHON_BIN}}" -c "import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])" "${{M}}" > /dev/null 2>&1; then
        echo "MISSING checkpoint ${{M}} in the local cache -- run download_models.sh, then resubmit"
        exit 1
    fi
done

# ── Run ────────────────────────────────────────────────────────────────────────
for SEED in "${{SEEDS[@]}}"; do
    for DATASET in "${{DATASETS[@]}}"; do
        for RUN in "${{RUNS[@]}}"; do
            read -r PARADIGM MODEL LEARNING_RATE FAMILY RUN_DATASETS <<< "${{RUN}}"
            [[ ",${{RUN_DATASETS}}," == *",${{DATASET}},"* ]] || continue
            family_settings "${{FAMILY}}"
            MODEL_SHORT="$(basename "${{MODEL}}")"
            OUTPUT_DIR="${{CHECKPOINT_ROOT}}/${{EXPERIMENT}}/${{PARADIGM}}/${{DATASET}}/${{MODEL_SHORT}}_seed${{SEED}}"
            RESULTS_DIR="${{PROJECT_ROOT}}/results/${{EXPERIMENT}}/${{PARADIGM}}/${{DATASET}}/${{MODEL_SHORT}}_seed${{SEED}}"

            # Eval batch sizes as in the main experiment.
            case "${{PARADIGM}}" in
                cross_encoder) EXTRA_ARGS=(--eval_batch_size 512) ;;
                generative)    EXTRA_ARGS=(--eval_batch_size 64 "${{GEN_BATCH_ARGS[@]}}")
                               if [ -n "${{CHAT_TEMPLATE_FROM}}" ]; then
                                   EXTRA_ARGS+=(--chat_template_from "${{CHAT_TEMPLATE_FROM}}")
                               fi ;;
            esac

            echo "── ${{PARADIGM}} ${{MODEL_SHORT}} ${{DATASET}} seed${{SEED}} lr${{LEARNING_RATE}} ──"

            if [ -f "${{RESULTS_DIR}}/test_results.json" ]; then
                echo "   Skipping: results already exist at ${{RESULTS_DIR}}"
                continue
            fi

            if "${{PYTHON_BIN}}" train_matcher.py \\
                --dataset         "${{DATASET}}"         \\
                --model           "${{MODEL}}"           \\
                --paradigm        "${{PARADIGM}}"        \\
                --seed            "${{SEED}}"            \\
                --learning_rate   "${{LEARNING_RATE}}"   \\
                --margin          "${{MARGIN}}"          \\
                --output_dir      "${{OUTPUT_DIR}}"      \\
                --results_dir     "${{RESULTS_DIR}}"     \\
                --wandb_project   "em-${{EXPERIMENT}}-${{PARADIGM}}" \\
                "${{EXTRA_ARGS[@]}}"; then
                echo "   Done"
            else
                echo "   FAILED (exit $?) -- will retry on re-submit"
            fi
        done
    done
done
"""

SUBMIT_TEMPLATE = """#!/bin/bash
# Submit every job of the sh/sr replicate experiment (generated by gen_scripts.py).
# Run download_models.sh first. Mind the 16-queued-jobs limit.
#   bash submit_all.sh                          # all seeds ({seeds})
#   SEED_LIST="42 123" bash submit_all.sh      # a subset
# The Slurm time limit is set from the number of seeds: estimated minutes per seed x
# seeds x {margin}, rounded up to whole hours. Re-submitting skips runs whose
# test_results.json already exists.
set -euo pipefail
HERE="{here}"

read -r -a SEEDS <<< "${{SEED_LIST:-{seeds}}}"
# "<script> <estimated minutes per seed>"
JOBS=(
{jobs}
)
for JOB in "${{JOBS[@]}}"; do
    read -r SCRIPT MIN_PER_SEED <<< "${{JOB}}"
    HOURS=$(( (MIN_PER_SEED * ${{#SEEDS[@]}} * {margin_pct} / 100 + 59) / 60 ))
    TIME=$(printf "%d-%02d:00:00" $((HOURS / 24)) $((HOURS % 24)))
    echo "${{SCRIPT}}: seeds ${{SEEDS[*]}}, time limit ${{TIME}}"
    SEED_LIST="${{SEEDS[*]}}" sbatch --export=ALL --time="${{TIME}}" "${{HERE}}/${{SCRIPT}}"
done
"""


def item_minutes_per_seed(key, paradigm, datasets):
    sh_min, sr_min = COST_MIN[(key, paradigm)]
    return sum(sr_min if d.startswith("sr") else sh_min for d in datasets)


def check_coverage():
    seen = {}
    for _, _, items in JOBS:
        for key, paradigm, datasets in items:
            for d in datasets:
                seen[(key, paradigm, d)] = seen.get((key, paradigm, d), 0) + 1
    expected = {(k, p, d) for (k, p) in COST_MIN for d in DATASETS}
    missing = expected - set(seen)
    dup = [k for k, n in seen.items() if n > 1]
    if missing or dup or set(seen) - expected:
        raise SystemExit(f"bad job layout: missing={sorted(missing)} duplicated={dup}")


def main():
    check_coverage()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    keep = {name for name, _, _ in JOBS} | {"download_models.sh", "submit_all.sh"}
    for f in os.listdir(HERE):  # drop scripts from an earlier job layout
        if f.endswith(".sh") and f not in keep:
            os.remove(os.path.join(HERE, f))
            print("removed stale", f)
    total = 0.0
    per_seed_min = {}
    for name, job_name, items in JOBS:
        per_seed_min[name] = sum(item_minutes_per_seed(*it) for it in items)
        est_h = N_SEEDS * per_seed_min[name] / 60
        total += est_h
        n_runs = N_SEEDS * sum(len(ds) for _, _, ds in items)
        lines = []
        for key, paradigm, datasets in items:
            model, family, lrs = MODELS[key]
            lines.append((paradigm, model, lrs[paradigm], family, ",".join(datasets)))
        widths = [max(len(line[i]) for line in lines) for i in range(4)]
        run_lines = "\n".join(
            '    "' + " ".join(f"{v:<{w}}" for v, w in zip(line[:4], widths)) + f' {line[4]}"'
            for line in lines)
        hours = math.ceil(est_h * TIME_MARGIN)
        path = os.path.join(HERE, name)
        with open(path, "w") as f:
            f.write(TEMPLATE.format(
                time=f"{hours // 24}-{hours % 24:02d}:00:00", job_name=job_name, n_jobs=len(JOBS), est_h=est_h,
                n_runs=n_runs, here=HERE, runs=run_lines, g4=G4, datasets=" ".join(DATASETS),
                seeds=SEEDS, experiment=EXPERIMENT, project_root=PROJECT_ROOT))
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP)
        print(f"wrote {name:28} {n_runs:3} runs  est {est_h:4.1f} h  limit {hours} h")
    print(f"total {total:.1f} h over {len(JOBS)} jobs")
    with open(os.path.join(HERE, "submit_all.sh"), "w") as f:
        f.write(SUBMIT_TEMPLATE.format(
            here=HERE, seeds=SEEDS, margin=TIME_MARGIN, margin_pct=round(TIME_MARGIN * 100),
            jobs="\n".join(f'    "{name} {per_seed_min[name]:.0f}"' for name, _, _ in JOBS)))
    print("wrote submit_all.sh")

if __name__ == "__main__":
    main()
