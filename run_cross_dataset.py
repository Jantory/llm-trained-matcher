"""
Cross-dataset inference (one seed → all paradigms × all source→target pairs).

Self-contained: depends only on utils/. For a fixed (paradigm, seed) the 8B base
model is loaded exactly ONCE; source LoRA adapters are hot-swapped on the shared
base and each target dataset is tokenized once and cached. The previous flow
respawned a process — reloading the full base — for every single pair.

Almost everything is hard-coded; only the seed varies per run:

    python run_cross_dataset.py --seed 42
    python run_cross_dataset.py --seed 42 --paradigm cross_encoder   # one paradigm

Results: results/cross/<paradigm>/<src>_to_<tgt>/seed<seed>.json.
Existing result files are skipped, so re-running resumes where it left off.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

# ── Path / imports ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.constants import DATASET_PATHS
from utils.dataset_builder import EntityMatchingDatasetBuilder
from utils.metrics import find_best_threshold, compute_binary_metrics, compute_ranking_metrics
from utils.models import BiEncoderModel

# ── Hard-coded configuration ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
EVAL_BATCH_SIZE = 32
MAX_NEW_TOKENS = 5

SCALE = "8b"
DATASETS = ["ab", "ag", "ds", "rt", "sh", "sr", "wa", "w20", "w80"]

# In-domain results dir: each source checkpoint's own decision threshold (chosen
# on its *source* validation split during training) lives here. We reuse it for
# the "no validation sweep" metrics — i.e. no re-calibration on the target.
RESULTS_MAIN_ROOT = os.path.join(PROJECT_ROOT, "results", f"main_{SCALE}")

# paradigm → checkpoint variant (model short name) used for RQ5
PARADIGM_MODEL = {
    "bi_encoder":    "Qwen3-Embedding-8B",
    "cross_encoder": "Qwen3-8B-Base",
    "generative":    "Qwen3-8B-Base",
}

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "/scratch-shared/61600/em-matcher")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "cross_dataset")


# ── Model loading ─────────────────────────────────────────────────────────────

def build_base_model(base_model_name: str, paradigm: str):
    """Build the (adapter-free) base model for the given paradigm."""
    attn_impl = "flash_attention_3"

    if paradigm == "bi_encoder":
        return BiEncoderModel(base_model_name)

    if paradigm == "cross_encoder":
        from transformers import AutoModelForSequenceClassification
        tok_tmp = AutoTokenizer.from_pretrained(base_model_name)
        pad_id = tok_tmp.pad_token_id or tok_tmp.eos_token_id
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2,
                attn_implementation=attn_impl, pad_token_id=pad_id,
            )
        except Exception:
            return AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2, pad_token_id=pad_id,
            )

    if paradigm == "generative":
        from transformers import AutoModelForCausalLM
        try:
            return AutoModelForCausalLM.from_pretrained(
                base_model_name, attn_implementation=attn_impl,
            )
        except Exception:
            return AutoModelForCausalLM.from_pretrained(base_model_name)

    raise ValueError(f"Unknown paradigm: {paradigm!r}")


def load_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


# ── Inference ─────────────────────────────────────────────────────────────────

def _pad_batch(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    padded = [[pad_id] * (max_len - len(s)) + list(s) for s in seqs]
    mask   = [[0] * (max_len - len(s)) + [1] * len(s) for s in seqs]
    return (
        torch.tensor(padded, dtype=torch.long, device=DEVICE),
        torch.tensor(mask,   dtype=torch.long, device=DEVICE),
    )


def _bi_encoder_scores(model, dataset, tokenizer):
    pad_id = tokenizer.pad_token_id
    scores, labels = [], []
    for start in range(0, len(dataset), EVAL_BATCH_SIZE):
        batch = dataset[start: start + EVAL_BATCH_SIZE]
        ids_a, mask_a = _pad_batch(batch["input_ids_a"], pad_id)
        ids_b, mask_b = _pad_batch(batch["input_ids_b"], pad_id)
        with torch.no_grad():
            out = model(
                input_ids_a=ids_a, attention_mask_a=mask_a,
                input_ids_b=ids_b, attention_mask_b=mask_b,
            )
        scores.extend(out["logits"].float().cpu().tolist())
        labels.extend(batch["labels"])
    return scores, labels


def _softmax(logits):
    import numpy as np
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _cross_encoder_scores(model, dataset, tokenizer):
    pad_id = tokenizer.pad_token_id
    scores, labels = [], []
    for start in range(0, len(dataset), EVAL_BATCH_SIZE):
        batch = dataset[start: start + EVAL_BATCH_SIZE]
        ids, mask = _pad_batch(batch["input_ids"], pad_id)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
        scores.extend(_softmax(logits)[:, 1].tolist())
        labels.extend(batch["labels"])
    return scores, labels


def _generative_eval(model, dataset, tokenizer):
    from utils.train import _run_generative_eval
    return _run_generative_eval(model, dataset, tokenizer, EVAL_BATCH_SIZE, MAX_NEW_TOKENS)


def compute_result(model, paradigm: str, datasets, tokenizer, no_sweep_threshold=None) -> dict:
    """
    Run inference + metrics for a single target dataset. No I/O.

    no_sweep_threshold: the source checkpoint's own in-domain threshold. When
    given (bi/cross), we also report test metrics at that fixed threshold, i.e.
    without re-sweeping on the target validation split.
    """
    if paradigm in ("bi_encoder", "cross_encoder"):
        score_fn = _bi_encoder_scores if paradigm == "bi_encoder" else _cross_encoder_scores

        val_scores, val_labels = score_fn(model, datasets["validation"], tokenizer)
        thresh_info = find_best_threshold(val_scores, val_labels)
        val_ranking = compute_ranking_metrics(val_scores, val_labels)

        test_scores, test_labels = score_fn(model, datasets["test"], tokenizer)
        test_ranking = compute_ranking_metrics(test_scores, test_labels)

        # Swept threshold: best-F1 threshold picked on the target validation split.
        test_preds = [int(s >= thresh_info["threshold"]) for s in test_scores]
        test_metrics = compute_binary_metrics(test_preds, test_labels)

        result = {
            "val_f1":         thresh_info["f1"],
            "val_auroc":      val_ranking["auroc"],
            "threshold":      thresh_info["threshold"],
            "test_f1":        test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_auroc":     test_ranking["auroc"],
        }

        # No-sweep: reuse the source model's own threshold (no target calibration).
        if no_sweep_threshold is not None:
            fixed_preds = [int(s >= no_sweep_threshold) for s in test_scores]
            fixed_metrics = compute_binary_metrics(fixed_preds, test_labels)
            result.update({
                "threshold_no_sweep":      no_sweep_threshold,
                "test_f1_no_sweep":        fixed_metrics["f1"],
                "test_precision_no_sweep": fixed_metrics["precision"],
                "test_recall_no_sweep":    fixed_metrics["recall"],
            })
        return result

    if paradigm == "generative":
        val_result  = _generative_eval(model, datasets["validation"], tokenizer)
        test_result = _generative_eval(model, datasets["test"],       tokenizer)
        return {
            "val_f1":                  val_result["f1"],
            "threshold":               None,
            "test_f1":                 test_result["f1"],
            "test_precision":          test_result["precision"],
            "test_recall":             test_result["recall"],
            "test_auroc":              None,
            "invalid_generation_rate": test_result["invalid_generation_rate"],
        }

    raise ValueError(f"Unknown paradigm: {paradigm!r}")


# ── Batch driver: one base model, all source→target pairs ─────────────────────

def load_source_threshold(paradigm: str, model_short: str, src: str, seed: int):
    """
    The source checkpoint's own in-domain threshold, saved during training under
    results/main_<scale>/<paradigm>/<src>/<model_short>_seed<seed>/test_results.json.
    Returns None if unavailable (e.g. generative, which has no threshold).
    """
    path = os.path.join(
        RESULTS_MAIN_ROOT, paradigm, src,
        f"{model_short}_seed{seed}", "test_results.json",
    )
    try:
        with open(path) as f:
            return json.load(f)["metadata"].get("threshold")
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def run_paradigm(paradigm: str, seed: int):
    """Evaluate every source→target pair for one (paradigm, seed), base loaded once."""
    from peft import PeftModel

    model_short = PARADIGM_MODEL[paradigm]
    base_model_name = f"Qwen/{model_short}"
    print(f"\n{'=' * 70}\n{paradigm}  ({base_model_name})  seed{seed}\n{'=' * 70}")

    tokenizer = load_tokenizer(base_model_name)

    print(f"[load] base model {base_model_name} (once) ...")
    base = build_base_model(base_model_name, paradigm)

    # Cache target datasets so each is tokenized only once.
    ds_cache: dict[str, dict] = {}

    def target_datasets(tgt: str):
        if tgt not in ds_cache:
            tgt_dir = DATASET_PATHS[tgt]
            print(f"[tok]  target {tgt} ({tgt_dir})")
            ds_cache[tgt] = EntityMatchingDatasetBuilder(
                paradigm, tgt_dir, tokenizer, MAX_LEN
            ).build()
        return ds_cache[tgt]

    model = None                       # PeftModel wrapping `base`; created lazily
    loaded_adapters: set[str] = set()
    n_done = 0

    for src in DATASETS:
        ckpt = os.path.join(
            CHECKPOINT_ROOT, f"main_{SCALE}", paradigm, src,
            f"{model_short}_seed{seed}", "best_model",
        )
        if not os.path.isdir(ckpt):
            print(f"[skip] no checkpoint: {paradigm}/{src} seed{seed}")
            continue

        # Which targets still need running?
        pending = []
        for tgt in DATASETS:
            if tgt == src:
                continue
            out_path = os.path.join(
                OUTPUT_ROOT, paradigm, f"{src}_to_{tgt}", f"seed{seed}.json"
            )
            if os.path.exists(out_path):
                continue
            pending.append((tgt, out_path))
        if not pending:
            print(f"[skip] all pairs done for source {src}")
            continue

        # Attach this source's adapter to the shared base (load once, then swap).
        if model is None:
            model = PeftModel.from_pretrained(base, ckpt, adapter_name=src)
            model = model.to(torch.bfloat16).to(DEVICE).eval()
            loaded_adapters.add(src)
        elif src not in loaded_adapters:
            model.load_adapter(ckpt, adapter_name=src)
            model = model.to(torch.bfloat16).to(DEVICE)   # move freshly-loaded params
            loaded_adapters.add(src)
        model.set_adapter(src)
        model.eval()

        # This source model's own in-domain threshold (for the no-sweep metrics).
        no_sweep_threshold = load_source_threshold(paradigm, model_short, src, seed)

        for tgt, out_path in pending:
            datasets = target_datasets(tgt)
            t0 = time.time()
            result = compute_result(model, paradigm, datasets, tokenizer, no_sweep_threshold)
            result.update({
                "runtime_sec": time.time() - t0,
                "paradigm":    paradigm,
                "seed":        seed,
                "source":      src,
                "target":      tgt,
                "checkpoint":  ckpt,
                "target_data": DATASET_PATHS[tgt],
            })
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            n_done += 1
            print(f"[ok]   {src}→{tgt}  val_f1={result['val_f1']:.4f}  "
                  f"test_f1={result['test_f1']:.4f}  ({result['runtime_sec']:.0f}s)")

    # Release GPU memory before the next paradigm's base is loaded.
    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[done] {paradigm} seed{seed}: {n_done} pair(s) evaluated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed to evaluate")
    parser.add_argument("--paradigm", default=None, choices=list(PARADIGM_MODEL),
                        help="Restrict to one paradigm (default: all three)")
    args = parser.parse_args()

    paradigms = [args.paradigm] if args.paradigm else list(PARADIGM_MODEL)
    for paradigm in paradigms:
        run_paradigm(paradigm, args.seed)


if __name__ == "__main__":
    main()
