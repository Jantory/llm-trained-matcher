"""Generate the sh dialect-combination rebalancing ablation Slurm scripts
(cross/gen/bi x 3 scales).

Mirrors scripts/rebalance_ablation/ (the sr analog) exactly in structure,
changing only:
  * dataset      -> shb (semi-heter_rebalanced): same pairs as sh, re-split so
                    every (A-dialect x B-dialect) combination lands
                    proportionally in every split (~16.5% previously-unseen-
                    combo positives in train/valid/test) instead of 0%/0%/82%
  * output roots -> results/sh_rebalance_ablation, $CHECKPOINT_ROOT/sh_rebalance_ablation
  * wandb project-> em-sh-rebalance-ablation
  * bi_encoder included (P1's puzzle statement covers all three paradigms,
    unlike sr's rebalancing which was cross/gen only)

Hyperparameters (LR, eval batch, margin) are copied verbatim from the
corresponding scripts/main_<scale>/*_qwen3-<scale>{-base,-emb-<scale>}.sh so
the ONLY experimental change vs the sh main run is the train/valid/test
partition (serialized tables, unchanged, copied by data/create_sh_rebalanced.py).

Run:  /gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python scripts/sh_rebalance_ablation/gen_scripts.py
"""
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/gpfs/home6/zzhang/em-emb/a-train-matcher"

# paradigm -> (short tag, eval_batch, model-name template using {scale_tag})
PARADIGMS = {
    "cross_encoder": ("cr", 512, "Qwen3-{scale_tag}-Base"),
    "generative":    ("gen", 64, "Qwen3-{scale_tag}-Base"),
    "bi_encoder":    ("bi", 256, "Qwen3-Embedding-{scale_tag}"),
}
# scale -> (scale_tag, sbatch_time, {paradigm: lr})
SCALES = {
    "0.6b": ("0.6B", "04:00:00",
             {"cross_encoder": "2e-4", "generative": "2e-4", "bi_encoder": "1e-5"}),
    "4b":   ("4B", "08:00:00",
             {"cross_encoder": "2e-4", "generative": "1e-4", "bi_encoder": "3e-5"}),
    "8b":   ("8B", "12:00:00",
             {"cross_encoder": "2e-4", "generative": "3e-4", "bi_encoder": "1e-4"}),
}

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time={time}
#SBATCH --partition=gpu_h100
#SBATCH --job-name=em-{pshort}-shb-{scale}
#SBATCH --output={here}/logs/slurm-%j-%x.out
#SBATCH --error={here}/logs/slurm-%j-%x.err

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# sh dialect-combination rebalancing ablation: same 2068 labeled sh pairs,
# re-split so every (A-dialect x B-dialect) combination is spread
# proportionally across train/valid/test (~16.5% previously-unseen-combo
# positives in every split) instead of concentrated in test (82% unseen).
# Serialized tables are UNCHANGED from sh -- isolates the split-composition
# variable. LR / eval_batch / margin copied verbatim from the corresponding
# scripts/main_{scale}/*.sh.
MODEL="Qwen/{model_short}"
MODEL_SHORT="{model_short}"
PARADIGM="{paradigm}"
LEARNING_RATE={lr}
EVAL_BATCH_SIZE={eval_batch}
MARGIN=0.3
WANDB_PROJECT="em-sh-rebalance-ablation"

DATASETS=(shb)
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
        OUTPUT_DIR="${{CHECKPOINT_ROOT}}/sh_rebalance_ablation/${{PARADIGM}}/${{DATASET}}/${{MODEL_SHORT}}_seed${{SEED}}"
        RESULTS_DIR="${{PROJECT_ROOT}}/results/sh_rebalance_ablation/${{PARADIGM}}/${{DATASET}}/${{MODEL_SHORT}}_seed${{SEED}}"

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
              "# Submit all 9 sh dialect-combination rebalancing ablation jobs",
              "# (3 scales x 3 paradigms). Each job runs dataset shb x 5 seeds",
              "# internally. Run from scripts/sh_rebalance_ablation/.",
              "set -euo pipefail", ""]
    for scale, (scale_tag, sbtime, lrs) in SCALES.items():
        for paradigm, (pshort, eval_batch, model_tmpl) in PARADIGMS.items():
            model_short = model_tmpl.format(scale_tag=scale_tag)
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
