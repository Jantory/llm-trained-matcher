"""
Entry point for training Qwen3 entity-matching models.

Usage:
    python train_matcher.py --dataset ab --model Qwen/Qwen3-8B --paradigm generative
    python train_matcher.py --dataset ab --model Qwen/Qwen3-Embedding --paradigm bi_encoder
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, set_seed

from utils.constants import DATASET_ALIASES, DATASET_PATHS
from utils.dataset_builder import build_dataset
from utils.train import train_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_ALIASES),
                   help="Short dataset alias: " + ", ".join(DATASET_ALIASES))
    p.add_argument("--model", required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--paradigm", required=True,
                   choices=["bi_encoder", "cross_encoder", "generative"])
    p.add_argument("--output_dir", default=None,
                   help="Checkpoint directory (defaults to $CHECKPOINT_ROOT/<paradigm>/<dataset>/<model>_seed<seed>)")
    p.add_argument("--results_dir", default=None,
                   help="Directory to save test-set results JSON (auto-generated if omitted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_size", type=int, default=8000)
    p.add_argument("--max_val_size", type=int, default=2000)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="Evaluation batch size (defaults to --batch_size if omitted)")
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--wandb_project", default="em-matcher")
    p.add_argument("--margin", type=float, default=0.3,
                   help="Contrastive loss margin (bi_encoder only)")
    p.add_argument("--no-lora", dest="lora", action="store_false")
    p.set_defaults(lora=True)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def _subsample(dataset, max_size, seed):
    if len(dataset) <= max_size:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_size))


def main():
    args = parse_args()
    set_seed(args.seed)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    data_dir = DATASET_PATHS[args.dataset]
    if not os.path.isdir(data_dir):
        sys.exit(f"Dataset directory not found: {data_dir}")

    model_short = os.path.basename(args.model.rstrip("/"))
    run_name = f"{args.dataset}_{args.paradigm}_{model_short}_lr{args.learning_rate}_seed{args.seed}"
    run_subpath = os.path.join(args.paradigm, args.dataset, f"{model_short}_seed{args.seed}")

    if args.output_dir is None:
        checkpoint_root = os.environ.get("CHECKPOINT_ROOT")
        if not checkpoint_root:
            sys.exit("Error: --output_dir not set and $CHECKPOINT_ROOT env var is missing.")
        args.output_dir = os.path.join(checkpoint_root, run_subpath)

    if args.results_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        args.results_dir = os.path.join(project_root, "results", run_subpath)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ["WANDB_RUN_NAME"] = run_name

    print(f"Dataset     : {args.dataset} → {data_dir}")
    print(f"Model       : {args.model}")
    print(f"Paradigm    : {args.paradigm}")
    print(f"Checkpoints : {args.output_dir}")
    print(f"Results     : {args.results_dir}")
    print(f"Seed        : {args.seed}")
    print(f"Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datasets = build_dataset(
        paradigm=args.paradigm,
        input_folder=data_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    datasets["train"] = _subsample(datasets["train"], args.max_train_size, args.seed)
    datasets["validation"] = _subsample(datasets["validation"], args.max_val_size, args.seed)

    print(f"Train: {len(datasets['train'])}  Val: {len(datasets['validation'])}  Test: {len(datasets['test'])}")

    training_args = {
        "seed": args.seed,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.eval_steps,
        "logging_steps": 1,
        "save_total_limit": 2,
        "run_name": run_name,
    }

    train_config = {
        "eval_batch_size": eval_batch_size,
        "margin": args.margin,
    }

    lora_config = None
    if args.lora:
        lora_config = {
            "enable_lora": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }

    result = train_model(
        paradigm=args.paradigm,
        model_name_or_path=args.model,
        datasets=datasets,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        training_args=training_args,
        lora_config=lora_config,
        train_config=train_config,
        results_dir=args.results_dir,
    )

    print(f"\nTest F1    : {result['test_f1']:.4f}")
    print(f"Test P     : {result['test_precision']:.4f}")
    print(f"Test R     : {result['test_recall']:.4f}")
    print(f"Test AUROC : {result.get('test_auroc', float('nan')):.4f}")
    print(f"Test AUPRC : {result.get('test_auprc', float('nan')):.4f}")
    print(f"Checkpoints: {result['output_dir']}")
    print(f"Results    : {args.results_dir}")


if __name__ == "__main__":
    main()
