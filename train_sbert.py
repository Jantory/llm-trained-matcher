"""
Sentence-BERT (RoBERTa) baseline for entity matching.

Trains a RoBERTa-based bi-encoder with mean pooling using the same
contrastive loss as the main bi-encoder experiments.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils.collators import BiEncoderCollator
from utils.constants import DATASET_ALIASES, DATASET_PATHS
from utils.dataset_builder import build_dataset
from utils.metrics import compute_binary_metrics, compute_ranking_metrics, find_best_threshold
from utils.models import contrastive_loss


class SBertModel(nn.Module):
    """RoBERTa-based bi-encoder with mean pooling and contrastive loss."""

    def __init__(self, model_name_or_path: str, margin: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.margin = margin

    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        z_a = self._pool(input_ids_a, attention_mask_a)
        z_b = self._pool(input_ids_b, attention_mask_b)
        cosine_scores = F.cosine_similarity(z_a, z_b)
        loss = contrastive_loss(z_a, z_b, labels, self.margin) if labels is not None else None
        return {"loss": loss, "logits": cosine_scores}


def _make_compute_metrics():
    def compute_metrics(eval_pred):
        scores, labels = eval_pred
        scores = scores.flatten()
        result = find_best_threshold(scores, labels)
        ranking = compute_ranking_metrics(scores, labels)
        return {
            "f1": result["f1"],
            "precision": result["precision"],
            "recall": result["recall"],
            "threshold": result["threshold"],
            "auroc": ranking["auroc"],
            "auprc": ranking["auprc"],
        }
    return compute_metrics


def _count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_ratio": trainable / total if total > 0 else 0.0,
    }


def _save_results(output_dir: str, predictions: list, metadata: dict | None = None):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump({"metadata": metadata or {}, "predictions": predictions}, f, indent=2)


def _save_log(output_dir: str, log_dict: dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log_dict, f, indent=2, default=str)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_ALIASES))
    p.add_argument("--model", default="sentence-transformers/stsb-roberta-base-v2",
                   help="HuggingFace model name, e.g. sentence-transformers/stsb-roberta-base-v2")
    p.add_argument("--output_dir", default=None,
                   help="Checkpoint directory (defaults to $CHECKPOINT_ROOT/baselines/sbert/...)")
    p.add_argument("--results_dir", default=None,
                   help="Results directory (auto-generated if omitted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_size", type=int, default=8000)
    p.add_argument("--max_val_size", type=int, default=2000)
    p.add_argument("--max_len", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--wandb_project", default="em-matcher-baselines")
    return p.parse_args()


def _subsample(dataset, max_size, seed):
    if len(dataset) <= max_size:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_size))


def main():
    args = parse_args()
    set_seed(args.seed)

    eval_batch_size = args.eval_batch_size or args.batch_size

    data_dir = DATASET_PATHS[args.dataset]
    if not os.path.isdir(data_dir):
        sys.exit(f"Dataset directory not found: {data_dir}")

    model_short = os.path.basename(args.model.rstrip("/"))
    run_name = f"{args.dataset}_sbert_{model_short}_lr{args.learning_rate}_seed{args.seed}"
    run_subpath = os.path.join("sbert", args.dataset, f"{model_short}_seed{args.seed}")

    if args.output_dir is None:
        checkpoint_root = os.environ.get("CHECKPOINT_ROOT")
        if not checkpoint_root:
            sys.exit("Error: --output_dir not set and $CHECKPOINT_ROOT env var is missing.")
        args.output_dir = os.path.join(checkpoint_root, "baselines", run_subpath)

    if args.results_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        args.results_dir = os.path.join(project_root, "results", "baselines", run_subpath)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ["WANDB_RUN_NAME"] = run_name

    print(f"Dataset     : {args.dataset} → {data_dir}")
    print(f"Model       : {args.model}")
    print(f"Checkpoints : {args.output_dir}")
    print(f"Results     : {args.results_dir}")
    print(f"Seed        : {args.seed}")
    print(f"Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datasets = build_dataset(
        paradigm="bi_encoder",
        input_folder=data_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    datasets["train"] = _subsample(datasets["train"], args.max_train_size, args.seed)
    datasets["validation"] = _subsample(datasets["validation"], args.max_val_size, args.seed)

    print(f"Train: {len(datasets['train'])}  Val: {len(datasets['validation'])}  Test: {len(datasets['test'])}")

    if os.path.exists(args.output_dir):
        print(f"[train] Wiping existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    model = SBertModel(args.model, margin=args.margin)
    param_info = _count_parameters(model)

    collator = BiEncoderCollator(pad_token_id=tokenizer.pad_token_id or 0)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=1,
        save_total_limit=2,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        warmup_ratio=0.05,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=_make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    init_eval = trainer.evaluate()
    train_result = trainer.train()

    best_model_path = os.path.join(args.output_dir, "best_model")
    trainer.model.encoder.save_pretrained(best_model_path)
    for entry in os.listdir(args.output_dir):
        if entry.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(args.output_dir, entry))

    t0 = time.time()
    val_pred = trainer.predict(datasets["validation"])
    val_time = time.time() - t0
    val_scores = val_pred.predictions.flatten()
    val_labels = val_pred.label_ids
    best_thresh = find_best_threshold(val_scores, val_labels)
    val_ranking = compute_ranking_metrics(val_scores, val_labels)

    t0 = time.time()
    test_pred = trainer.predict(datasets["test"])
    test_time = time.time() - t0
    test_scores = test_pred.predictions.flatten()
    test_labels = test_pred.label_ids
    test_preds = (test_scores >= best_thresh["threshold"]).astype(int)
    test_metrics = compute_binary_metrics(test_preds, test_labels)
    test_ranking = compute_ranking_metrics(test_scores, test_labels)

    print(f"\nTest F1    : {test_metrics['f1']:.4f}")
    print(f"Test P     : {test_metrics['precision']:.4f}")
    print(f"Test R     : {test_metrics['recall']:.4f}")
    print(f"Test AUROC : {test_ranking['auroc']:.4f}")
    print(f"Test AUPRC : {test_ranking['auprc']:.4f}")
    print(f"Checkpoints: {args.output_dir}")
    print(f"Results    : {args.results_dir}")

    log_data = {
        "paradigm": "sbert",
        "model": args.model,
        "initial_validation_f1": init_eval.get("eval_f1", 0.0),
        "initial_validation_precision": init_eval.get("eval_precision", 0.0),
        "initial_validation_recall": init_eval.get("eval_recall", 0.0),
        "initial_validation_auroc": init_eval.get("eval_auroc", 0.0),
        "initial_validation_auprc": init_eval.get("eval_auprc", 0.0),
        "best_validation_f1": best_thresh["f1"],
        "best_validation_precision": best_thresh["precision"],
        "best_validation_recall": best_thresh["recall"],
        "best_validation_auroc": val_ranking["auroc"],
        "best_validation_auprc": val_ranking["auprc"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_auroc": test_ranking["auroc"],
        "test_auprc": test_ranking["auprc"],
        "threshold": best_thresh["threshold"],
        "efficiency": {
            **param_info,
            "train_runtime": train_result.metrics.get("train_runtime", 0.0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0),
            "validation_runtime": val_time,
            "test_runtime": test_time,
        },
        "output_dir": args.output_dir,
        "trainer_log_history": trainer.state.log_history,
    }
    _save_log(args.results_dir, log_data)
    _save_results(
        args.results_dir,
        [
            {"score": float(s), "pred": int(p), "label": int(l)}
            for s, p, l in zip(test_scores, test_preds, test_labels)
        ],
        metadata={"paradigm": "sbert", "threshold": float(best_thresh["threshold"])},
    )


if __name__ == "__main__":
    main()
