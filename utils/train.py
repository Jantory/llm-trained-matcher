import os
import shutil
import time
import json

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from .collators import BiEncoderCollator, GenerativeTrainCollator
from .metrics import compute_binary_metrics, compute_ranking_metrics, find_best_threshold
from .models import BiEncoderModel


_DEFAULT_TRAIN_CONFIG = {
    "loss_name": "contrastive",
    "margin": 0.3,
    "patience": 5,
    "eval_batch_size": 32,
    "max_new_tokens": 5,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_model(
    paradigm: str,
    model_name_or_path: str,
    datasets,
    tokenizer,
    output_dir: str,
    training_args: dict | TrainingArguments,
    lora_config: dict | None = None,
    train_config: dict | None = None,
    results_dir: str | None = None,
):
    cfg = {**_DEFAULT_TRAIN_CONFIG, **(train_config or {})}
    dispatch = {
        "bi_encoder": _train_bi_encoder,
        "cross_encoder": _train_cross_encoder,
        "generative": _train_generative,
    }
    if paradigm not in dispatch:
        raise ValueError(f"Unknown paradigm: {paradigm!r}")
    return dispatch[paradigm](
        model_name_or_path, datasets, tokenizer, output_dir, training_args, lora_config, cfg,
        results_dir=results_dir or output_dir,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_lora(model, lora_config: dict, task_type=None):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=lora_config.get("lora_rank", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("lora_target_modules", "all-linear"),
        task_type=task_type,
    )
    return get_peft_model(model, config)


class _PeftSavingCallback(TrainerCallback):
    """When LoRA is active, save only adapter weights per Trainer checkpoint."""
    def on_save(self, args, state, control, **kwargs):
        from peft import PeftModel
        model = kwargs.get("model")
        if not isinstance(model, PeftModel):
            return control
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model.save_pretrained(checkpoint_dir)
        for fname in os.listdir(checkpoint_dir):
            if not fname.startswith("adapter_"):
                path = os.path.join(checkpoint_dir, fname)
                if os.path.isfile(path):
                    os.remove(path)
        return control


def _count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_ratio": trainable / total if total > 0 else 0.0,
    }


def _peak_gpu_mb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return None


def _build_training_args(
    training_args: dict | TrainingArguments,
    output_dir: str,
    metric_for_best_model: str = "eval_f1",
    load_best_model_at_end: bool = True,
    eval_strategy: str = "steps",
) -> TrainingArguments:
    if isinstance(training_args, TrainingArguments):
        return training_args
    merged = {
        "report_to": "wandb",
        "per_device_train_batch_size": 64,
        "max_steps": 400,
        "num_train_epochs": 20,
        "gradient_checkpointing": True,
        "warmup_ratio": 0.05,
        **training_args,
        # these are forced after the user dict so they cannot be overridden
        "output_dir": output_dir,
        "eval_strategy": eval_strategy,
        "save_strategy": eval_strategy,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": True,
        "load_best_model_at_end": load_best_model_at_end,
    }
    return TrainingArguments(**merged)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _log_wandb(metrics: dict, step: int | None = None, commit: bool | None = None):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step, commit=commit)
    except ImportError:
        pass


def _save_log(output_dir: str, log_dict: dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log_dict, f, indent=2, default=str)


def _save_results(output_dir: str, predictions: list, metadata: dict | None = None):
    os.makedirs(output_dir, exist_ok=True)
    payload = {"metadata": metadata or {}, "predictions": predictions}
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _efficiency_block(param_info, train_result, n_train, val_runtime, n_val, test_runtime, n_test):
    block = {**param_info}
    if train_result is not None:
        block["train_runtime"] = train_result.metrics.get("train_runtime", 0.0)
        block["train_samples_per_second"] = train_result.metrics.get("train_samples_per_second", 0.0)
    block["validation_runtime"] = val_runtime
    block["test_runtime"] = test_runtime
    gpu = _peak_gpu_mb()
    if gpu is not None:
        block["peak_gpu_memory_mb"] = gpu
    return block


# ---------------------------------------------------------------------------
# Generative evaluation helper
# ---------------------------------------------------------------------------

def _run_generative_eval(
    model, eval_dataset, tokenizer, batch_size: int, max_new_tokens: int = 5,
    return_samples: bool = False,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    all_preds, all_labels, all_answers = [], [], []
    n_invalid = 0
    total_generated = 0
    t0 = time.time()

    n = len(eval_dataset)
    for start in range(0, n, batch_size):
        batch = eval_dataset[start: start + batch_size]
        ids_list = batch["eval_input_ids"]
        mask_list = batch["eval_attention_mask"]
        gold_labels = batch["gold_label"]

        max_len = max(len(x) for x in ids_list)
        padded_ids, padded_mask = [], []
        for ids, mask in zip(ids_list, mask_list):
            pad = max_len - len(ids)
            padded_ids.append([pad_id] * pad + list(ids))
            padded_mask.append([0] * pad + list(mask))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attn_mask = torch.tensor(padded_mask, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,  # prompt-only tokens (no gold answer)
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        new_tokens = outputs[:, input_ids.shape[1]:]
        for tok_ids, gold in zip(new_tokens.tolist(), gold_labels):
            total_generated += len(tok_ids)
            answer = tokenizer.decode(tok_ids, skip_special_tokens=True).strip().lower()
            if answer.startswith("yes"):
                pred = 1
            elif answer.startswith("no"):
                pred = 0
            else:
                n_invalid += 1
                pred = 0
            all_preds.append(pred)
            all_labels.append(int(gold))
            all_answers.append(answer)

    runtime = time.time() - t0
    n_total = len(all_labels)
    metrics = compute_binary_metrics(all_preds, all_labels)
    metrics["invalid_generation_rate"] = n_invalid / n_total if n_total > 0 else 0.0
    metrics["generation_runtime"] = runtime
    metrics["generation_samples_per_second"] = n_total / runtime if runtime > 0 else 0.0
    metrics["max_new_tokens"] = max_new_tokens
    metrics["average_generated_tokens"] = total_generated / n_total if n_total > 0 else 0.0
    if return_samples:
        metrics["samples"] = [
            {"generated": a, "pred": p, "label": l}
            for a, p, l in zip(all_answers, all_preds, all_labels)
        ]
    return metrics


# ---------------------------------------------------------------------------
# Generative early-stop
# ---------------------------------------------------------------------------


class _GenerativeEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, output_dir, patience, eval_steps, eval_batch_size, max_new_tokens):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.best_model_path = os.path.join(output_dir, "best_model")
        self.patience = patience
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.max_new_tokens = max_new_tokens
        self.best_f1 = -1.0
        self.best_result = None
        self.init_result = None
        self.no_improve = 0
        os.makedirs(self.best_model_path, exist_ok=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # WandbCallback.on_train_begin has already fired, so the W&B run is live.
        result = _run_generative_eval(
            model, self.eval_dataset, self.tokenizer,
            self.eval_batch_size, self.max_new_tokens,
        )
        self.init_result = result
        _log_wandb({
            "eval/f1": result["f1"],
            "eval/precision": result["precision"],
            "eval/recall": result["recall"],
            "eval/invalid_generation_rate": result["invalid_generation_rate"],
            "eval/runtime": result["generation_runtime"],
        }, step=0, commit=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control
        result = _run_generative_eval(
            model, self.eval_dataset, self.tokenizer,
            self.eval_batch_size, self.max_new_tokens,
        )
        if result["f1"] > self.best_f1:
            self.best_f1 = result["f1"]
            self.best_result = result
            model.save_pretrained(self.best_model_path)
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                control.should_training_stop = True
        _log_wandb({
            "eval/f1": result["f1"],
            "eval/precision": result["precision"],
            "eval/recall": result["recall"],
            "eval/invalid_generation_rate": result["invalid_generation_rate"],
            "eval/runtime": result["generation_runtime"],
        }, step=state.global_step)
        return control


# ---------------------------------------------------------------------------
# compute_metrics factories for Trainer
# ---------------------------------------------------------------------------

def _make_bi_encoder_compute_metrics():
    def compute_metrics(eval_pred):
        scores, labels = eval_pred
        scores = scores.flatten()
        result = find_best_threshold(scores, labels)
        ranking = compute_ranking_metrics(scores, labels)
        return {
            "f1": result["f1"], "precision": result["precision"],
            "recall": result["recall"], "threshold": result["threshold"],
            "auroc": ranking["auroc"], "auprc": ranking["auprc"],
        }
    return compute_metrics


def _make_cross_encoder_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        scores = _softmax(logits)[:, 1]
        result = find_best_threshold(scores, labels)
        ranking = compute_ranking_metrics(scores, labels)
        return {
            "f1": result["f1"], "precision": result["precision"],
            "recall": result["recall"], "threshold": result["threshold"],
            "auroc": ranking["auroc"], "auprc": ranking["auprc"],
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Bi-encoder training
# ---------------------------------------------------------------------------

def _train_bi_encoder(model_name_or_path, datasets, tokenizer, output_dir, training_args, lora_config, cfg, results_dir=None):
    if os.path.exists(output_dir):
        print(f"[train] Wiping existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    model = BiEncoderModel(model_name_or_path, loss_name=cfg["loss_name"], margin=cfg["margin"])
    if lora_config and lora_config.get("enable_lora"):
        model = _apply_lora(model, lora_config, task_type=None)
    param_info = _count_parameters(model)

    collator = BiEncoderCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    args = _build_training_args(training_args, output_dir, metric_for_best_model="eval_f1")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=_make_bi_encoder_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg["patience"]),
            *([_PeftSavingCallback()] if lora_config and lora_config.get("enable_lora") else []),
        ],
    )

    init_eval = trainer.evaluate()
    _log_wandb({
        "eval/f1": init_eval.get("eval_f1", 0.0),
        "eval/precision": init_eval.get("eval_precision", 0.0),
        "eval/recall": init_eval.get("eval_recall", 0.0),
        "eval/auroc": init_eval.get("eval_auroc", 0.0),
        "eval/auprc": init_eval.get("eval_auprc", 0.0),
        "eval/loss": init_eval.get("eval_loss", 0.0),
    })

    train_result = trainer.train()

    best_model_path = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    if hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(best_model_path)
    else:
        torch.save(trainer.model.state_dict(), os.path.join(best_model_path, "model.pt"))
    for entry in os.listdir(output_dir):
        if entry.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(output_dir, entry))

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

    efficiency = _efficiency_block(
        param_info, train_result,
        len(datasets["train"]), val_time, len(datasets["validation"]),
        test_time, len(datasets["test"]),
    )

    log_data = {
        "paradigm": "bi_encoder",
        "initial_validation_f1": init_eval.get("eval_f1", 0.0),
        "initial_validation_precision": init_eval.get("eval_precision", 0.0),
        "initial_validation_recall": init_eval.get("eval_recall", 0.0),
        "initial_validation_auroc": init_eval.get("eval_auroc", 0.0),
        "initial_validation_auprc": init_eval.get("eval_auprc", 0.0),
        "best_validation_precision": best_thresh["precision"],
        "best_validation_recall": best_thresh["recall"],
        "best_validation_f1": best_thresh["f1"],
        "best_validation_auroc": val_ranking["auroc"],
        "best_validation_auprc": val_ranking["auprc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auroc": test_ranking["auroc"],
        "test_auprc": test_ranking["auprc"],
        "threshold": best_thresh["threshold"],
        "efficiency": efficiency,
        "output_dir": output_dir,
        "trainer_log_history": trainer.state.log_history,
    }
    _log_wandb({
        "test/f1": test_metrics["f1"],
        "test/precision": test_metrics["precision"],
        "test/recall": test_metrics["recall"],
        "test/auroc": test_ranking["auroc"],
        "test/auprc": test_ranking["auprc"],
        "test/eval_time": test_time,
        "test/threshold": best_thresh["threshold"],
    })
    _save_log(results_dir or output_dir, log_data)
    _save_results(
        results_dir or output_dir,
        [
            {"score": float(s), "pred": int(p), "label": int(l)}
            for s, p, l in zip(test_scores, test_preds, test_labels)
        ],
        metadata={"paradigm": "bi_encoder", "threshold": float(best_thresh["threshold"])},
    )
    return log_data


# ---------------------------------------------------------------------------
# Cross-encoder training
# ---------------------------------------------------------------------------

def _train_cross_encoder(model_name_or_path, datasets, tokenizer, output_dir, training_args, lora_config, cfg, results_dir=None):
    if os.path.exists(output_dir):
        print(f"[train] Wiping existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    from peft import TaskType
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenizer.pad_token_id = pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=2, attn_implementation="flash_attention_3",
        pad_token_id=pad_token_id,
    )
    if lora_config and lora_config.get("enable_lora"):
        model = _apply_lora(model, lora_config, task_type=TaskType.SEQ_CLS)
    param_info = _count_parameters(model)

    collator = DataCollatorWithPadding(tokenizer)
    args = _build_training_args(training_args, output_dir, metric_for_best_model="eval_f1")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=_make_cross_encoder_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg["patience"]),
            *([_PeftSavingCallback()] if lora_config and lora_config.get("enable_lora") else []),
        ],
    )

    init_eval = trainer.evaluate()
    _log_wandb({
        "eval/f1": init_eval.get("eval_f1", 0.0),
        "eval/precision": init_eval.get("eval_precision", 0.0),
        "eval/recall": init_eval.get("eval_recall", 0.0),
        "eval/auroc": init_eval.get("eval_auroc", 0.0),
        "eval/auprc": init_eval.get("eval_auprc", 0.0),
        "eval/loss": init_eval.get("eval_loss", 0.0)
    })

    train_result = trainer.train()

    best_model_path = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    if hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(best_model_path)
    else:
        torch.save(trainer.model.state_dict(), os.path.join(best_model_path, "model.pt"))
    for entry in os.listdir(output_dir):
        if entry.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(output_dir, entry))

    t0 = time.time()
    val_pred = trainer.predict(datasets["validation"])
    val_time = time.time() - t0
    val_scores = _softmax(val_pred.predictions)[:, 1]
    val_labels = val_pred.label_ids
    best_thresh = find_best_threshold(val_scores, val_labels)
    val_ranking = compute_ranking_metrics(val_scores, val_labels)

    t0 = time.time()
    test_pred = trainer.predict(datasets["test"])
    test_time = time.time() - t0
    test_scores = _softmax(test_pred.predictions)[:, 1]
    test_labels = test_pred.label_ids
    test_preds = (test_scores >= best_thresh["threshold"]).astype(int)
    test_metrics = compute_binary_metrics(test_preds, test_labels)
    test_ranking = compute_ranking_metrics(test_scores, test_labels)

    efficiency = _efficiency_block(
        param_info, train_result,
        len(datasets["train"]), val_time, len(datasets["validation"]),
        test_time, len(datasets["test"]),
    )

    log_data = {
        "paradigm": "cross_encoder",
        "initial_validation_f1": init_eval.get("eval_f1", 0.0),
        "initial_validation_precision": init_eval.get("eval_precision", 0.0),
        "initial_validation_recall": init_eval.get("eval_recall", 0.0),
        "initial_validation_auroc": init_eval.get("eval_auroc", 0.0),
        "initial_validation_auprc": init_eval.get("eval_auprc", 0.0),
        "best_validation_precision": best_thresh["precision"],
        "best_validation_recall": best_thresh["recall"],
        "best_validation_f1": best_thresh["f1"],
        "best_validation_auroc": val_ranking["auroc"],
        "best_validation_auprc": val_ranking["auprc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auroc": test_ranking["auroc"],
        "test_auprc": test_ranking["auprc"],
        "threshold": best_thresh["threshold"],
        "efficiency": efficiency,
        "output_dir": output_dir,
        "trainer_log_history": trainer.state.log_history,
    }
    _log_wandb({
        "test/f1": test_metrics["f1"],
        "test/precision": test_metrics["precision"],
        "test/recall": test_metrics["recall"],
        "test/auroc": test_ranking["auroc"],
        "test/auprc": test_ranking["auprc"],
        "test/eval_time": test_time,
        "test/threshold": best_thresh["threshold"],
    })
    _save_log(results_dir or output_dir, log_data)
    _save_results(
        results_dir or output_dir,
        [
            {"prob": float(s), "pred": int(p), "label": int(l)}
            for s, p, l in zip(test_scores, test_preds, test_labels)
        ],
        metadata={"paradigm": "cross_encoder", "threshold": float(best_thresh["threshold"])},
    )
    return log_data


# ---------------------------------------------------------------------------
# Generative training
# ---------------------------------------------------------------------------

def _train_generative(model_name_or_path, datasets, tokenizer, output_dir, training_args, lora_config, cfg, results_dir=None):
    if os.path.exists(output_dir):
        print(f"[train] Wiping existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    from peft import TaskType
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, attn_implementation="flash_attention_3"
    )
    if lora_config and lora_config.get("enable_lora"):
        model = _apply_lora(model, lora_config, task_type=TaskType.CAUSAL_LM)
    param_info = _count_parameters(model)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = GenerativeTrainCollator(pad_token_id=pad_id)

    train_cols = ["input_ids", "attention_mask", "labels"]
    extra = [c for c in datasets["train"].column_names if c not in train_cols]
    trainer_train = datasets["train"].remove_columns(extra)

    eval_gen_dataset = datasets["validation"]
    test_gen_dataset = datasets["test"]

    eval_steps = (
        training_args.eval_steps
        if isinstance(training_args, TrainingArguments)
        else training_args.get("eval_steps", 100)
    )
    eval_callback = _GenerativeEvalCallback(
        eval_dataset=eval_gen_dataset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        patience=cfg["patience"],
        eval_steps=eval_steps,
        eval_batch_size=cfg["eval_batch_size"],
        max_new_tokens=cfg["max_new_tokens"],
    )

    args = _build_training_args(
        training_args,
        output_dir,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=False,
        eval_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=trainer_train,
        data_collator=collator,
        callbacks=[eval_callback],
    )

    train_result = trainer.train()

    best_model_path = os.path.join(output_dir, "best_model")
    if os.path.exists(best_model_path):
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model.load_adapter(best_model_path, adapter_name="default")
        else:
            device = next(model.parameters()).device
            model = AutoModelForCausalLM.from_pretrained(
                best_model_path, attn_implementation="flash_attention_3"
            ).to(device)

    init_eval = eval_callback.init_result or {}
    best_val = eval_callback.best_result or {}

    t0 = time.time()
    test_result = _run_generative_eval(
        model, test_gen_dataset, tokenizer, cfg["eval_batch_size"], cfg["max_new_tokens"],
        return_samples=True,
    )
    test_time = time.time() - t0

    efficiency = _efficiency_block(
        param_info, train_result,
        len(trainer_train),
        best_val.get("generation_runtime", 0.0), len(eval_gen_dataset),
        test_time, len(test_gen_dataset),
    )
    efficiency["generation_runtime"] = test_result["generation_runtime"]
    efficiency["generation_samples_per_second"] = test_result["generation_samples_per_second"]
    efficiency["max_new_tokens"] = cfg["max_new_tokens"]
    efficiency["average_generated_tokens"] = test_result.get("average_generated_tokens", 0.0)

    log_data = {
        "paradigm": "generative",
        "initial_validation_f1": init_eval.get("f1", 0.0),
        "initial_validation_precision": init_eval.get("precision", 0.0),
        "initial_validation_recall": init_eval.get("recall", 0.0),
        "initial_invalid_generation_rate": init_eval.get("invalid_generation_rate", 0.0),
        "best_validation_precision": best_val.get("precision", 0.0),
        "best_validation_recall": best_val.get("recall", 0.0),
        "best_validation_f1": best_val.get("f1", 0.0),
        "test_precision": test_result["precision"],
        "test_recall": test_result["recall"],
        "test_f1": test_result["f1"],
        "invalid_generation_rate": test_result["invalid_generation_rate"],
        "efficiency": efficiency,
        "output_dir": output_dir,
        "trainer_log_history": trainer.state.log_history,
    }
    _log_wandb({
        "test/f1": test_result["f1"],
        "test/precision": test_result["precision"],
        "test/recall": test_result["recall"],
        "test/invalid_generation_rate": test_result["invalid_generation_rate"],
        "test/runtime": test_result["generation_runtime"],
    })
    _save_log(results_dir or output_dir, log_data)
    _save_results(
        results_dir or output_dir,
        test_result.get("samples", []),
        metadata={"paradigm": "generative"},
    )
    return log_data
