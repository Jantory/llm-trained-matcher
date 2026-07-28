"""
Oracle throughput/latency profiling.

Isolates each paradigm's raw architecture cost from any specific dataset's text
length distribution by feeding a synthetic input, padded/truncated to exactly
--max_len tokens, through each paradigm's real deployment head:
  bi_encoder   : representation forward + pool         (AutoModel)
  cross_encoder: classification forward                (AutoModelForSequenceClassification)
  generative   : full model forward--single-token generation            (AutoModelForCausalLM)

Reports two regimes per paradigm, at a fixed input length of --max_len tokens:
  latency_ms_bs1         : single-sample (batch_size=1) end-to-end time
  throughput_texts_per_s : max-batch (OOM-tuned) throughput

--max_len is configurable (default 512) so cross_encoder/generative can be
swept across input lengths to see how cost scales with sequence length.

Same bf16 / flash_attention_3 / H100 setup as profile_throughput.py, so numbers
are on the same footing as the dataset-driven profiling there — this script
just removes the dataset as a confound.

Usage:
  python profile_oracle.py \\
      --paradigm bi_encoder \\
      --model Qwen/Qwen3-Embedding-8B \\
      --output results/inference_efficiency/oracle/bi_encoder/8b.json

  python profile_oracle.py \\
      --paradigm generative \\
      --model Qwen/Qwen3-8B-Base \\
      --max_len 2048 \\
      --output results/inference_efficiency/oracle/generative/8b_len2048.json
"""

import argparse
import json
import os
import sys
import time

# Must be set before torch is imported
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.models import last_token_pool

INSTRUCTION = {
    "bi_encoder":    "Represent the entity for entity matching.",
    "cross_encoder": "Determine whether the following two entities refer to the same real-world entity.",
    "generative":    "Determine whether the following two entities refer to the same real-world entity. Answer yes or no.",
}
# Long filler content — always exceeds --max_len tokens once tokenized (even for
# large sweep values), so every batch is truncated to exactly --max_len real
# (non-pad) tokens. Content itself doesn't matter, only the resulting length.
FILLER_TEXT = ("Acme Corporation industrial widget model X200 stainless steel "
               "housing rated for 240V operation manufactured in Ohio USA. ") * 500
DEFAULT_MAX_LEN = 512
N_REPEATS = 5        # repeat timing measurement and take mean for stability
BATCH_INIT = 1024 * 8     # starting batch size for the max-batch throughput probe


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_batch_size(probe_fn, init: int) -> int:
    """Try probe_fn(batch_size); halve on OOM until it succeeds or reaches 1."""
    batch_size = init
    while batch_size >= 1:
        try:
            torch.cuda.empty_cache()
            probe_fn(batch_size)
            print(f"[oracle] batch_size={batch_size} OK")
            return batch_size
        except torch.cuda.OutOfMemoryError:
            print(f"[oracle] OOM at batch_size={batch_size}, halving ...")
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


def replicate_batch(base_inputs: dict, batch_size: int) -> dict:
    """Build a batch by repeating an already-tokenized single-sample dict along
    dim 0, instead of re-tokenizing `batch_size` copies of the same string.
    FILLER_TEXT is ~8500 words; the fast tokenizer has to run its full
    normalize/pretokenize/BPE pass over the whole string before truncating to
    max_len, so re-tokenizing it per sample per batch made tokenization (not
    the model forward) dominate the timed window."""
    return {k: v.repeat(batch_size, *([1] * (v.dim() - 1))) for k, v in base_inputs.items()}


def time_paradigm(run_batch, batch_size_init, max_len):
    """Shared timing harness: find the max feasible batch size, then measure
    both single-sample latency and max-batch throughput for `run_batch(bs)`."""
    batch_size = find_batch_size(run_batch, batch_size_init)

    # ── Single-sample latency (batch_size=1) ───────────────────────────────
    reset_peak_memory()
    latencies = []
    for _ in range(N_REPEATS*20):
        cuda_sync()
        t0 = time.perf_counter()
        run_batch(1)
        cuda_sync()
        latencies.append(time.perf_counter() - t0)
    latency_peak_mem = peak_memory_mb()

    # ── Max-batch throughput ────────────────────────────────────────────────
    torch.cuda.empty_cache()  # untimed: allocator hygiene across the bs=1 -> bs=batch_size jump
    reset_peak_memory()
    batch_times = []
    for _ in range(N_REPEATS):
        cuda_sync()
        t0 = time.perf_counter()
        run_batch(batch_size)
        cuda_sync()
        batch_times.append(time.perf_counter() - t0)
    throughput_peak_mem = peak_memory_mb()

    mean_latency_s = float(np.mean(latencies))
    mean_batch_time_s = float(np.mean(batch_times))

    return {
        "max_len":                      max_len,
        "batch_size_used":              batch_size,
        "latency_ms_bs1":               mean_latency_s * 1000,
        "latency_ms_bs1_repeats":       [x * 1000 for x in latencies],
        "throughput_texts_per_s":       batch_size / mean_batch_time_s,
        "batch_time_s":                 mean_batch_time_s,
        "batch_time_s_repeats":         batch_times,
        "peak_gpu_memory_mb_bs1":       latency_peak_mem,
        "peak_gpu_memory_mb_maxbatch":  throughput_peak_mem,
    }


# ── Paradigm-specific batch runners ───────────────────────────────────────────

def profile_bi_encoder(model, tokenizer, device, batch_size_init, max_len):
    text = f"{INSTRUCTION['bi_encoder']}\n{FILLER_TEXT}"
    base_inputs = tokenizer(
        text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt",
    )
    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}

    def run_batch(batch_size):
        inputs = replicate_batch(base_inputs, batch_size)
        with torch.no_grad():
            out = model(**inputs)
            embeds = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
            embeds = F.normalize(embeds, dim=-1)
        del inputs, out, embeds

    return time_paradigm(run_batch, batch_size_init, max_len)


def profile_cross_encoder(model, tokenizer, device, batch_size_init, max_len):
    text = f"{INSTRUCTION['cross_encoder']}\n{FILLER_TEXT}"
    base_inputs = tokenizer(
        text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt",
    )
    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}

    def run_batch(batch_size):
        inputs = replicate_batch(base_inputs, batch_size)
        with torch.no_grad():
            out = model(**inputs)
        del inputs, out

    return time_paradigm(run_batch, batch_size_init, max_len)


def profile_generative(model, tokenizer, device, batch_size_init, max_len):
    prompt_text = f"{INSTRUCTION['generative']}\n{FILLER_TEXT}"
    # Chat-template tokenize once: content/length is static across calls, and
    # the resulting prompt is truncated to max_len so prefill length matches
    # the other two paradigms exactly.
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    )[:max_len]
    base_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # [1, L]
    base_inputs = {"input_ids": base_input_ids, "attention_mask": torch.ones_like(base_input_ids)}

    def run_batch(batch_size):
        inputs = replicate_batch(base_inputs, batch_size)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        del inputs, out

    return time_paradigm(run_batch, batch_size_init, max_len)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm", required=True, choices=["bi_encoder", "cross_encoder", "generative"])
    p.add_argument("--model",    required=True, help="HuggingFace model name or local path")
    p.add_argument("--output",   required=True, help="Output JSON path")
    p.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN,
                   help="Fixed input sequence length in tokens (default 512); sweep this to see "
                        "how cross_encoder/generative cost scales with input length")
    p.add_argument("--batch_size_init", type=int, default=BATCH_INIT,
                   help="Starting batch size for the max-batch throughput probe; halved automatically on OOM")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[oracle] paradigm={args.paradigm} model={args.model} max_len={args.max_len}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
                         attn_implementation="flash_attention_3")
    if args.paradigm == "generative":
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    elif args.paradigm == "cross_encoder":
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    print(f"[oracle] model loaded, running {N_REPEATS} repeats ...")

    if args.paradigm == "bi_encoder":
        result = profile_bi_encoder(model, tokenizer, device, args.batch_size_init, args.max_len)
    elif args.paradigm == "cross_encoder":
        result = profile_cross_encoder(model, tokenizer, device, args.batch_size_init, args.max_len)
    else:
        result = profile_generative(model, tokenizer, device, args.batch_size_init, args.max_len)

    result.update({
        "paradigm":  args.paradigm,
        "model":     args.model,
        "n_repeats": N_REPEATS,
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[oracle] latency(bs=1) = {result['latency_ms_bs1']:.2f} ms")
    print(f"[oracle] throughput    = {result['throughput_texts_per_s']:.1f} texts/s @ bs={result['batch_size_used']}")
    print(f"[oracle] peak GPU      = {result['peak_gpu_memory_mb_maxbatch']:.0f} MB")
    print(f"[oracle] saved → {args.output}")


if __name__ == "__main__":
    main()
