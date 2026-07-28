# Beyond Scale and Generation: Understanding Language Model-based Entity Matching

This repository contains all code, scripts, and artifacts for the paper
**"Beyond Scale and Generation: Understanding Language Model-based Entity Matching"**,
including data preparation, training/inference entry points, the full set of
experiment launch scripts, raw results for every run, and the visualization
code used to produce all figures in the paper.

📄 Paper: [arXiv:2607.24688](https://arxiv.org/pdf/2607.24688)

## Repository Layout

```
.
├── data/                  # Data preparation scripts + dataset documentation
├── utils/                 # Training utilities (shared library code)
├── train_matcher.py       # Main entry point: trains a Qwen3 matcher (bi-encoder / cross-encoder / generative)
├── train_sbert.py         # Sentence-BERT (RoBERTa) baseline trainer
├── run_cross_dataset.py   # Cross-dataset transfer inference
├── run_openai_baseline.py # OpenAI API generative baseline
├── probe_geometry.py      # Zero-shot embedding-geometry probe (LoRA mechanism analysis)
├── profile_throughput.py  # Dataset-driven inference efficiency profiling
├── profile_oracle.py      # Architecture-only (synthetic input) inference profiling
├── scripts/                # All bash launch scripts for training/inference used in the paper
├── results/                # All run outputs (per-run test predictions + training logs)
└── visualization/           # Figure-reproduction notebook + plotting code
```

### `data/` — Data preparation

- `data_clean.ipynb`, `create_unified_dataset.py`, `create_sh_rebalanced.py`,
  `create_sr_rebalanced.py`: scripts used to clean the raw benchmark exports
  and build the train/valid/test splits used in experiments.
- `raw/`: original, unprocessed benchmark downloads (DeepMatcher, MaChAmp/GEM,
  WDC Products).
- `cleaned/`: the processed datasets actually consumed by training, one folder
  per dataset (e.g. `abt_buy/`, `amazon_google/`, `wdc20/`), each containing
  `tableA.csv` / `tableB.csv` (+ serialized versions) and `train.csv` /
  `valid.csv` / `test.csv`.
- `DATASETS.md`: full description of all nine benchmarks used in the paper —
  domain, schema type, source, and split statistics.

### `utils/` — Training utilities

Shared library code imported by every training/inference entry point:

- `constants.py` — dataset aliases and paths.
- `dataset_builder.py` — builds tokenized `Dataset`/`DatasetDict` objects per paradigm.
- `collators.py` — batch collation for the bi-encoder paradigm.
- `models.py` — model wrapper(s) (pooling, similarity heads, etc.).
- `train.py` — the shared training loop used by `train_matcher.py`.
- `metrics.py` — precision/recall/F1/AUROC/AP computation.

### `train_matcher.py` — Main training entry point

This is the single entry point used for **all** paradigms (bi-encoder,
cross-encoder, generative) reported in the paper:

```bash
python train_matcher.py --dataset ab --model Qwen/Qwen3-8B --paradigm generative
python train_matcher.py --dataset ab --model Qwen/Qwen3-Embedding --paradigm bi_encoder
```

Other root-level scripts support specific experiments:
- `train_sbert.py` — classic Sentence-BERT/RoBERTa baseline.
- `run_openai_baseline.py` — GPT-based generative baseline via the OpenAI API.
- `run_cross_dataset.py` — loads one fine-tuned base once and hot-swaps LoRA
  adapters to evaluate every source→target dataset transfer pair.
- `probe_geometry.py` — zero-shot cosine-similarity/AUROC geometry probe
  (instruct vs. base vs. embedding checkpoints), used for the LoRA mechanism
  analysis in the paper.
- `profile_throughput.py` / `profile_oracle.py` — inference efficiency
  measurements (dataset-driven and architecture-only, respectively).

### `scripts/` — Full set of training/inference launch scripts

Contains **every** SLURM/bash script used to run the paper's experiments,
organized by experiment group:

- `main_0.6b/`, `main_4b/`, `main_8b/` — main results at each model scale, for
  all three paradigms (`bi_*`, `cross_*`, `gen_*`) across all datasets/seeds.
- `lr_sweep_0.6b/`, `lr_sweep_4b/`, `lr_sweep_8b/` — learning-rate sweeps per scale.
- `ft_ablation/` — full fine-tuning vs. LoRA ablations, plus the geometry-probe launcher.
- `cross_dataset/` — cross-dataset transfer runs, one script per seed.
- `sh_rebalance_ablation/`, `sr_rebalance_ablation/` — class-rebalancing ablations for semi-heter / semi-rel.
- `inference_efficiency/` — throughput/latency profiling launchers per paradigm/scale.
- `baselines/` — SBERT/RoBERTa and OpenAI baseline launchers.

Each experiment group has a `submit_all.sh` to launch the full sweep, and most
have a `collect_results.sh` / `results.md` summarizing outcomes. Scripts are
written for a SLURM cluster (H100 partition) — adjust the `#SBATCH` header,
`HF_HOME`, `CHECKPOINT_ROOT`, and `PYTHON_BIN` environment variables for your
own environment before running.

### `results/` — All experiment outputs

Mirrors the `scripts/` grouping (`main_0.6b/`, `lr_sweep_8b/`,
`cross_dataset/`, `inference_efficiency/`, `baselines/`, etc.). For every
individual training run, the corresponding leaf directory contains:

- `test_results.json` — the model's evaluated predictions/metrics on the held-out test set.
- `training_log.json` — the full training log (loss curves, timing, etc.) for that run.

Path convention: `results/<experiment_group>/<paradigm>/<dataset>/<model>_seed<seed>/`.

Additionally, `geometry_probe_0.6b.csv` and `training_efficiency.json` hold
aggregated results for the geometry probe and training-efficiency comparisons.

### `visualization/` — Figures and Tables

- `visualize.ipynb` — fully reproduces every figure in the paper from the CSVs in `data/`.
- `ds_common.py` — shared plotting/data-loading helpers used by the notebook.
- `data/` — the aggregated CSVs (e.g. `rq1_raw_data.csv`, `rq2_raw_data.csv`,
  `cross_dataset_matrix.csv`, `oracle_latency.csv`, `cost_summary.csv`) the
  notebook reads to generate figures.
- `figures/` — the rendered output figures (PDF) as included in the paper.

## Model Checkpoints

All trained checkpoints (every paradigm, scale, dataset, and seed) are
released on Hugging Face:

**https://huggingface.co/buckets/ZeyuZhang/em-matcher-checkpoints**
