"""
Oracle inference profiling.

Measures inference efficiency using the correct deployment strategy per paradigm:
  bi_encoder   : encode all unique entities once → cosine similarity (O(M) not O(N))
  cross_encoder: pairwise forward pass (O(N))
  generative   : pairwise forward pass + generation (O(N))
  sbert        : classic baseline, bi-encoder deployment (mean pooling, no instruction)
  ditto        : classic baseline cost proxy, pairwise sequence-classification forward (O(N))

Does NOT require fine-tuned checkpoints — throughput is determined by model
architecture and input length, not LoRA weights.

All paradigms are timed end-to-end: tokenization happens inside the timed
window alongside the forward pass, on the same H100 partition, bf16, MAX_LEN=512,
batch size auto-halved on OOM from BATCH_INIT, N_REPEATS timing repeats — so
numbers are directly comparable across paradigms/models.

Batch size is determined automatically: starts at --batch_size_init and halves on
OOM until a working size is found. The final batch size is recorded in the output.

Usage:
  python profile_throughput.py \\
      --paradigm bi_encoder \\
      --model Qwen/Qwen3-Embedding-8B \\
      --dataset ab \\
      --output results/inference_efficiency/bi_encoder/8b/ab.json

  python profile_throughput.py \\
      --paradigm sbert \\
      --dataset ab \\
      --output results/inference_efficiency/baselines/sbert/ab.json
"""

import argparse
import json
import os
import sys
import time

# Must be set before torch is imported
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import DATASET_PATHS
from utils.models import last_token_pool

# ── Constants matching training setup ─────────────────────────────────────────
INSTRUCTION = {
    "bi_encoder":    "Represent the entity for entity matching.",
    "cross_encoder": "Determine whether the following two entities refer to the same real-world entity.",
    "generative":    "Determine whether the following two entities refer to the same real-world entity. Answer yes or no.",
}
# Classic baselines: fixed model, no --model needed (override still allowed).
BASELINE_MODELS = {
    "sbert": "sentence-transformers/stsb-roberta-base-v2",
    "ditto": "roberta-base",
}
# Qwen-family paradigms support flash_attention_3; the RoBERTa baselines don't.
FLASH_ATTN_PARADIGMS = {"bi_encoder", "cross_encoder", "generative"}
MAX_LEN = 512
N_REPEATS = 5        # repeat timing measurement and take mean for stability
BATCH_INIT = 1024 * 8      # starting batch size; halved automatically on OOM


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_batch_size(probe_fn, init: int) -> int:
    """Try probe_fn(batch_size); halve on OOM until it succeeds or reaches 1."""
    batch_size = init
    while batch_size >= 1:
        try:
            torch.cuda.empty_cache()
            probe_fn(batch_size)
            print(f"[profile] batch_size={batch_size} OK")
            return batch_size
        except torch.cuda.OutOfMemoryError:
            print(f"[profile] OOM at batch_size={batch_size}, halving ...")
            batch_size //= 2
    raise RuntimeError("OOM even at batch_size=1")

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pooling for the sbert baseline (RoBERTa has no reliable last-token EOS)."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def profile_bi_encoder(model, tokenizer, dataset_path, device, batch_size_init,
                        pooling_fn=last_token_pool, instruction=""):
    df_test = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    df_a = pd.read_csv(os.path.join(dataset_path, "tableA_serialized.csv")).set_index("id")
    df_b = pd.read_csv(os.path.join(dataset_path, "tableB_serialized.csv")).set_index("id")

    n_pairs = len(df_test)
    unique_left_ids  = df_test["ltable_id"].unique().tolist()
    unique_right_ids = df_test["rtable_id"].unique().tolist()
    texts_a = [str(df_a.loc[i, "text"]) for i in unique_left_ids]
    texts_b = [str(df_b.loc[i, "text"]) for i in unique_right_ids]
    texts_all = texts_a + texts_b  # encode all unique entities in one pass

    n_unique_a = len(unique_left_ids)
    n_unique_b = len(unique_right_ids)

    def encode_batch(batch_texts):
        """Tokenize + forward + pool one batch. Left untimed here — the
        per-repeat loop wraps timing around this so tokenize is included."""
        prefixed = [f"{instruction}\n{t}" if instruction else t for t in batch_texts]
        inputs = tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            embeds = pooling_fn(out.last_hidden_state, inputs["attention_mask"])
            embeds = F.normalize(embeds, dim=-1)
        result = embeds.cpu()
        del inputs, out, embeds
        return result

    # Probe with the longest texts (worst-case memory)
    texts_by_len = sorted(texts_all, key=len, reverse=True)
    batch_size = find_batch_size(lambda bs: encode_batch(texts_by_len[:bs]), batch_size_init)

    encode_times = []
    sim_times = []

    id2idx_a = {uid: i for i, uid in enumerate(unique_left_ids)}
    id2idx_b = {uid: i for i, uid in enumerate(unique_right_ids)}
    left_indices  = torch.tensor([id2idx_a[i] for i in df_test["ltable_id"]], dtype=torch.long)
    right_indices = torch.tensor([id2idx_b[i] for i in df_test["rtable_id"]], dtype=torch.long)

    for _ in range(N_REPEATS):
        torch.cuda.empty_cache()  # untimed: allocator hygiene between repeats
        reset_peak_memory()
        cuda_sync()
        t0 = time.perf_counter()
        embeds_all = torch.cat([
            encode_batch(texts_all[start: start + batch_size])
            for start in range(0, len(texts_all), batch_size)
        ], dim=0)
        cuda_sync()
        encode_time = time.perf_counter() - t0
        embeds_a, embeds_b = embeds_all[:n_unique_a], embeds_all[n_unique_a:]

        t0 = time.perf_counter()
        sim_scores = F.cosine_similarity(embeds_a[left_indices], embeds_b[right_indices])
        cuda_sync()
        sim_time = time.perf_counter() - t0

        encode_times.append(encode_time)
        sim_times.append(sim_time)

    peak_mem = peak_memory_mb()
    total_times = [e + s for e, s in zip(encode_times, sim_times)]

    return {
        "n_pairs":             n_pairs,
        "n_unique_a":          n_unique_a,
        "n_unique_b":          n_unique_b,
        "n_unique_total":      n_unique_a + n_unique_b,
        "entity_encode_time_s":          float(np.mean(encode_times)),
        "entity_encode_time_s_repeats":  [float(x) for x in encode_times],
        "similarity_time_s":             float(np.mean(sim_times)),
        "similarity_time_s_repeats":     [float(x) for x in sim_times],
        "total_inference_time_s":        float(np.mean(total_times)),
        "total_inference_time_s_repeats":[float(x) for x in total_times],
        "effective_throughput_pairs_per_s": n_pairs / float(np.mean(total_times)),
        "peak_gpu_memory_mb":  peak_mem,
        "batch_size_used":     batch_size,
    }


def profile_cross_encoder(model, tokenizer, dataset_path, device, batch_size_init,
                           instruction=None, pair_mode=False):
    """pair_mode=True is the ditto baseline: tokenizer(text_a, text_b) sequence-pair
    encoding instead of one concatenated string (matches its classification-head input)."""
    df_test = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    df_a = pd.read_csv(os.path.join(dataset_path, "tableA_serialized.csv")).set_index("id")
    df_b = pd.read_csv(os.path.join(dataset_path, "tableB_serialized.csv")).set_index("id")

    texts = []
    for _, row in df_test.iterrows():
        ta = str(df_a.loc[row["ltable_id"], "text"])
        tb = str(df_b.loc[row["rtable_id"], "text"])
        if pair_mode:
            texts.append((ta, tb))
        else:
            texts.append(f"{instruction}\nEntity A: {ta}\nEntity B: {tb}")

    n_pairs = len(texts)

    def run_batch(batch, bs):
        batch = batch[:bs]
        if pair_mode:
            ta, tb = zip(*batch)
            enc = tokenizer(list(ta), list(tb), padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
        else:
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        del enc, out

    # Probe with the longest texts (worst-case memory)
    sort_key = (lambda p: len(p[0]) + len(p[1])) if pair_mode else len
    texts_by_len = sorted(texts, key=sort_key, reverse=True)
    batch_size = find_batch_size(lambda bs: run_batch(texts_by_len, bs), batch_size_init)

    inf_times = []
    for _ in range(N_REPEATS):
        torch.cuda.empty_cache()  # untimed: allocator hygiene between repeats
        reset_peak_memory()
        cuda_sync()
        t0 = time.perf_counter()
        for start in range(0, n_pairs, batch_size):
            run_batch(texts[start: start + batch_size], batch_size)
        cuda_sync()
        inf_times.append(time.perf_counter() - t0)

    peak_mem = peak_memory_mb()

    return {
        "n_pairs":             n_pairs,
        "total_inference_time_s":         float(np.mean(inf_times)),
        "total_inference_time_s_repeats": [float(x) for x in inf_times],
        "effective_throughput_pairs_per_s": n_pairs / float(np.mean(inf_times)),
        "peak_gpu_memory_mb":  peak_mem,
        "batch_size_used":     batch_size,
    }


def profile_generative(model, tokenizer, dataset_path, device, batch_size_init):
    instruction = INSTRUCTION["generative"]
    df_test = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    df_a = pd.read_csv(os.path.join(dataset_path, "tableA_serialized.csv")).set_index("id")
    df_b = pd.read_csv(os.path.join(dataset_path, "tableB_serialized.csv")).set_index("id")

    texts = []
    for _, row in df_test.iterrows():
        ta = str(df_a.loc[row["ltable_id"], "text"])
        tb = str(df_b.loc[row["rtable_id"], "text"])
        texts.append(f"{instruction}\nEntity A: {ta}\nEntity B: {tb}")

    n_pairs = len(texts)

    def run_gen_batch(batch):
        """Chat-template tokenize + pad + generate one batch, timed as a whole
        by the caller so tokenize is included end-to-end (mirrors run_batch)."""
        ids_list = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False,
            )[:MAX_LEN]
            for t in batch
        ]
        max_len_batch = max(len(x) for x in ids_list)
        padded = [[tokenizer.pad_token_id] * (max_len_batch - len(x)) + x for x in ids_list]
        masks  = [[0] * (max_len_batch - len(x)) + [1] * len(x)           for x in ids_list]
        input_ids      = torch.tensor(padded, dtype=torch.long).to(device)
        attention_mask = torch.tensor(masks,  dtype=torch.long).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        del input_ids, attention_mask, out

    # Probe with the longest prompts (worst-case memory)
    texts_by_len = sorted(texts, key=len, reverse=True)
    batch_size = find_batch_size(lambda bs: run_gen_batch(texts_by_len[:bs]), batch_size_init)

    inf_times = []
    for _ in range(N_REPEATS):
        torch.cuda.empty_cache()  # untimed: allocator hygiene between repeats
        reset_peak_memory()
        cuda_sync()
        t0 = time.perf_counter()
        for start in range(0, n_pairs, batch_size):
            run_gen_batch(texts[start: start + batch_size])
        cuda_sync()
        inf_times.append(time.perf_counter() - t0)

    peak_mem = peak_memory_mb()

    return {
        "n_pairs":             n_pairs,
        "total_inference_time_s":         float(np.mean(inf_times)),
        "total_inference_time_s_repeats": [float(x) for x in inf_times],
        "effective_throughput_pairs_per_s": n_pairs / float(np.mean(inf_times)),
        "peak_gpu_memory_mb":  peak_mem,
        "batch_size_used":     batch_size,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm", required=True,
                   choices=["bi_encoder", "cross_encoder", "generative", "sbert", "ditto"])
    p.add_argument("--model",    default=None,
                   help="HuggingFace model name or local path. Defaults to the fixed "
                        "baseline model for --paradigm sbert/ditto; required otherwise.")
    p.add_argument("--dataset",  required=True, help="Dataset short alias (ab, ag, ...)")
    p.add_argument("--output",   required=True, help="Output JSON path")
    p.add_argument("--batch_size_init", type=int, default=BATCH_INIT,
                   help="Starting batch size; halved automatically on OOM")
    args = p.parse_args()
    if args.model is None:
        args.model = BASELINE_MODELS.get(args.paradigm)
        if args.model is None:
            p.error("--model is required for paradigm " + args.paradigm)
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = DATASET_PATHS[args.dataset]

    print(f"[profile] paradigm={args.paradigm} model={args.model} dataset={args.dataset}")
    print(f"[profile] dataset_path={dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if args.paradigm in FLASH_ATTN_PARADIGMS:
        model_kwargs["attn_implementation"] = "flash_attention_3"

    if args.paradigm == "generative":
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    elif args.paradigm in ("cross_encoder", "ditto"):
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    print(f"[profile] model loaded, running {N_REPEATS} repeats ...")

    if args.paradigm == "bi_encoder":
        result = profile_bi_encoder(model, tokenizer, dataset_path, device, args.batch_size_init,
                                     pooling_fn=last_token_pool, instruction=INSTRUCTION["bi_encoder"])
    elif args.paradigm == "sbert":
        result = profile_bi_encoder(model, tokenizer, dataset_path, device, args.batch_size_init,
                                     pooling_fn=mean_pool, instruction="")
    elif args.paradigm == "cross_encoder":
        result = profile_cross_encoder(model, tokenizer, dataset_path, device, args.batch_size_init,
                                        instruction=INSTRUCTION["cross_encoder"])
    elif args.paradigm == "ditto":
        result = profile_cross_encoder(model, tokenizer, dataset_path, device, args.batch_size_init,
                                        pair_mode=True)
    else:
        result = profile_generative(model, tokenizer, dataset_path, device, args.batch_size_init)

    result.update({
        "paradigm":  args.paradigm,
        "model":     args.model,
        "dataset":   args.dataset,
        "n_repeats": N_REPEATS,
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[profile] throughput = {result['effective_throughput_pairs_per_s']:.1f} pairs/s")
    print(f"[profile] peak GPU   = {result['peak_gpu_memory_mb']:.0f} MB")
    print(f"[profile] saved → {args.output}")


if __name__ == "__main__":
    main()
