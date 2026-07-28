"""Generate the title-collision rebalancing ablation Slurm scripts (cross/gen x 3 scales).

Mirrors scripts/blob_ablation/ exactly, changing only:
  * dataset      -> srb (semi-rel_rebalanced); same pairs as sr, re-split so
                    title-collision negatives are proportionally spread across
                    train/valid/test instead of concentrated in test
                    (data/create_sr_rebalanced.py)
  * output roots -> results/sr_rebalance_ablation, $CHECKPOINT_ROOT/sr_rebalance_ablation
  * wandb project-> em-sr-rebalance-ablation

Hyperparameters (LR, eval batch, margin) are copied from the corresponding
scripts/main_<scale>/{cross,gen}_qwen3-<scale>-base.sh so the ONLY experimental
change vs the sr main run is the train/valid/test partition (the serialized
tables, including the free-text blob, are copied UNCHANGED from sr -- this is
a single-variable ablation independent of scripts/blob_ablation/).

Run:  /gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python scripts/sr_rebalance_ablation/gen_scripts.py
"""
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/gpfs/home6/zzhang/em-emb/a-train-matcher"

PARADIGMS = {
    "cross_encoder": ("cr", 512),
    "generative":    ("gen", 64),
}
SCALES = {
    "0.6b": ("Qwen3-0.6B-Base", "04:00:00", {"cross_encoder": "2e-4", "generative": "2e-4"}),
    "4b":   ("Qwen3-4B-Base",   "08:00:00", {"cross_encoder": "2e-4", "generative": "1e-4"}),
    "8b":   ("Qwen3-8B-Base",   "12:00:00", {"cross_encoder": "2e-4", "generative": "3e-4"}),
}

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time={time}
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-{pshort}-rb-{scale}
#SBATCH --output={here}/logs/slurm-%j-%x.out
#SBATCH --error={here}/logs/slurm-%j-%x.err

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# Title-collision rebalancing ablation: same 2183 labeled sr pairs, re-split so
# "same title, different movie" negatives (78 total) are spread proportionally
# across train/valid/test (~6% each) instead of concentrated in test (28.7% vs
# 0.4-0.5% in train/valid in the original sr split). Serialized tables (incl.
# the free-text blob) are UNCHANGED from sr -- isolates the split-composition
# variable, independent of scripts/blob_ablation/.
# LR / eval_batch / margin copied verbatim from scripts/main_{scale}/{pshort}_qwen3-{scale}-base.sh.
MODEL="Qwen/{model_short}"
MODEL_SHORT="{model_short}"
PARADIGM="{paradigm}"
LEARNING_RATE={lr}
EVAL_BATCH_SIZE={eval_batch}
MARGIN=0.3
WANDB_PROJECT="em-sr-rebalance-ablation"

DATASETS=(srb)
SEEDS=(42 123 456 789 2024)

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME="${{HF_HOME:-/scratch-shared/61600/.cache/huggingface}}"
export CHECKPOINT_ROOT="${{CHECKPOINT_ROOT:-/scratch-shared/61600/em-matcher}}"
PYTHON_BIN="${{PYTHON_BIN:-/gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python}}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="{project_root}"

# shellcheck source=../../.env
source "${{PROJECT_ROOT}}/.env"

export WANDB_PROJECT="${{WANDB_PROJECT}}"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID    : ${{SLURM_JOB_ID}}"
echo "Node      : $(hostname)"
echo "Model     : ${{MODEL}}"
echo "Paradigm  : ${{PARADIGM}}"
echo "LR        : ${{LEARNING_RATE}}"

# ── Run ────────────────────────────────────────────────────────────────────────
cd "${{PROJECT_ROOT}}"

for DATASET in "${{DATASETS[@]}}"; do
    for SEED in "${{SEEDS[@]}}"; do
        OUTPUT_DIR="${{CHECKPOINT_ROOT}}/sr_rebalance_ablation/${{PARADIGM}}/${{DATASET}}/${{MODEL_SHORT}}_seed${{SEED}}"
        RESULTS_DIR="${{PROJECT_ROOT}}/results/sr_rebalance_ablation/${{PARADIGM}}/${{DATASET}}/${{MODEL_SHORT}}_seed${{SEED}}"

        echo "── ${{DATASET}} seed${{SEED}} ────────────────────────────────────────"

        if [ -f "${{RESULTS_DIR}}/test_results.json" ]; then
            echo "   Skipping: results already exist at ${{RESULTS_DIR}}"
            continue
        fi

        if "${{PYTHON_BIN}}" train_matcher.py \\
            --dataset         "${{DATASET}}"         \\
            --model           "${{MODEL}}"           \\
            --paradigm        "${{PARADIGM}}"        \\
            --seed            "${{SEED}}"            \\
            --eval_batch_size "${{EVAL_BATCH_SIZE}}" \\
            --learning_rate   "${{LEARNING_RATE}}"   \\
            --margin          "${{MARGIN}}"          \\
            --output_dir      "${{OUTPUT_DIR}}"      \\
            --results_dir     "${{RESULTS_DIR}}"     \\
            --wandb_project   "${{WANDB_PROJECT}}"; then
            echo "   Done: ${{DATASET}} seed${{SEED}}"
        else
            echo "   FAILED: ${{DATASET}} seed${{SEED}} (exit $?) -- will retry on re-submit"
        fi
    done
done
"""


def main():
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    written = []
    submit = ["#!/bin/bash",
              "# Submit all 6 title-collision rebalancing ablation jobs (3 scales x 2 paradigms).",
              "# Each job runs dataset srb x 5 seeds internally. Run from scripts/sr_rebalance_ablation/.",
              "set -euo pipefail", ""]
    for scale, (model_short, sbtime, lrs) in SCALES.items():
        for paradigm, (pshort, eval_batch) in PARADIGMS.items():
            fname = f"{pshort}_qwen3-{scale}-base.sh"
            body = TEMPLATE.format(
                time=sbtime, pshort=pshort, scale=scale, here=HERE,
                model_short=model_short, paradigm=paradigm,
                lr=lrs[paradigm], eval_batch=eval_batch, project_root=PROJECT_ROOT,
            )
            path = os.path.join(HERE, fname)
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
            written.append(fname)
            submit.append(f"sbatch {fname}")
        submit.append("")
    sp = os.path.join(HERE, "submit_all.sh")
    with open(sp, "w") as f:
        f.write("\n".join(submit) + "\n")
    os.chmod(sp, os.stat(sp).st_mode | stat.S_IEXEC)
    print("Generated:", *written, "submit_all.sh", sep="\n  ")


if __name__ == "__main__":
    main()
