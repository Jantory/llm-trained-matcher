"""
Zero-shot geometry probe for the LoRA mechanism hypothesis.

For each checkpoint variant (instruct, base, embedding), we encode all entities
using the same bi-encoder pipeline as training (last-token pooling + instruction
prefix), then compute cosine similarity on the test pairs *before any fine-tuning*.

Metrics reported per dataset × variant:
  - pos_mean / neg_mean : average cosine similarity for positive / negative pairs
  - gap                 : pos_mean - neg_mean  (higher = better geometry)
  - auroc               : AUC-ROC using cosine similarity as score (threshold-free)

Usage:
    python probe_geometry.py --scale 0.6b [--datasets ab ag sh sr]
"""

import argparse
import csv
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "cleaned")

DATASET_DIRS = {
    "ab": "abt_buy",
    "ag": "amazon_google",
    "ds": "dblp_scholar",
    "rt": "rel-text",
    "sh": "semi-heter",
    "sr": "semi-rel",
    "wa": "walmart_amazon",
    "w20": "wdc20",
    "w80": "wdc80",
}

MODELS = {
    "0.6b": {
        "instruct": "Qwen/Qwen3-0.6B",
        "base":     "Qwen/Qwen3-0.6B-Base",
        "emb":      "Qwen/Qwen3-Embedding-0.6B",
    },
    "4b": {
        "instruct": "Qwen/Qwen3-4B",
        "base":     "Qwen/Qwen3-4B-Base",
        "emb":      "Qwen/Qwen3-Embedding-4B",
    },
    "8b": {
        "instruct": "Qwen/Qwen3-8B",
        "base":     "Qwen/Qwen3-8B-Base",
        "emb":      "Qwen/Qwen3-Embedding-8B",
    },
}

INSTRUCTION = "Represent the entity for entity matching."
MAX_LEN = 512
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Helpers ────────────────────────────────────────────────────────────────────

def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, seq_len]


def load_serialized_table(folder: str, side: str) -> dict:
    path = os.path.join(folder, f"table{side}_serialized.csv")
    with open(path) as f:
        return {int(r["id"]): r["text"] for r in csv.DictReader(f)}


def load_test_pairs(folder: str):
    pairs = []
    with open(os.path.join(folder, "test.csv")) as f:
        for r in csv.DictReader(f):
            pairs.append((int(r["ltable_id"]), int(r["rtable_id"]), int(r["label"])))
    return pairs


def encode_texts(texts: list[str], model, tokenizer, has_sentinel: bool) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        prefixed = [INSTRUCTION + "\n" + t for t in batch]
        enc = tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        if has_sentinel:
            sentinel_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            sentinel_col = torch.full((enc["input_ids"].shape[0], 1), sentinel_id)
            mask_col = torch.ones(enc["input_ids"].shape[0], 1, dtype=torch.long)
            enc["input_ids"] = torch.cat([enc["input_ids"], sentinel_col], dim=1)
            enc["attention_mask"] = torch.cat([enc["attention_mask"], mask_col], dim=1)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def auroc(scores: list[float], labels: list[int]) -> float:
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (n_pos * n_neg)


# ── Main ───────────────────────────────────────────────────────────────────────

def probe_dataset(ds_alias: str, model, tokenizer, has_sentinel: bool):
    folder = os.path.join(DATA_ROOT, DATASET_DIRS[ds_alias])
    table_a = load_serialized_table(folder, "A")
    table_b = load_serialized_table(folder, "B")
    pairs    = load_test_pairs(folder)

    # Encode all unique entities
    ids_a = sorted(table_a.keys())
    ids_b = sorted(table_b.keys())
    texts_a = [table_a[i] for i in ids_a]
    texts_b = [table_b[i] for i in ids_b]

    print(f"    encoding tableA ({len(ids_a)} records)...", flush=True)
    embs_a = encode_texts(texts_a, model, tokenizer, has_sentinel)
    print(f"    encoding tableB ({len(ids_b)} records)...", flush=True)
    embs_b = encode_texts(texts_b, model, tokenizer, has_sentinel)

    idx_a = {eid: i for i, eid in enumerate(ids_a)}
    idx_b = {eid: i for i, eid in enumerate(ids_b)}

    scores, labels = [], []
    for lid, rid, label in pairs:
        ea = embs_a[idx_a[lid]]
        eb = embs_b[idx_b[rid]]
        scores.append(F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item())
        labels.append(label)

    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)
    gap = pos_mean - neg_mean
    auc = auroc(scores, labels)

    return {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "gap":      gap,
        "auroc":    auc,
        "n_pos":    len(pos_scores),
        "n_neg":    len(neg_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="0.6b", choices=["0.6b", "4b", "8b"])
    parser.add_argument("--datasets", nargs="+", default=["ab", "ag", "ds", "rt", "sh", "sr", "wa", "w20", "w80"])
    parser.add_argument("--variants", nargs="+", default=["instruct", "base", "emb"])
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME", None)
    results = []

    for variant in args.variants:
        model_name = MODELS[args.scale][variant]
        print(f"\n{'='*60}")
        print(f"Variant: {variant}  |  Model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_home)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=hf_home,
        ).to(DEVICE).eval()

        # Detect sentinel: embedding models auto-append <|endoftext|> with add_special_tokens=True
        probe_with    = tokenizer("x", add_special_tokens=True)["input_ids"]
        probe_without = tokenizer("x", add_special_tokens=False)["input_ids"]
        has_sentinel  = (probe_with[-1] != probe_without[-1])

        for ds in args.datasets:
            print(f"  dataset: {ds}", flush=True)
            stats = probe_dataset(ds, model, tokenizer, has_sentinel)
            results.append({
                "scale": args.scale, "variant": variant, "dataset": ds, **stats
            })
            print(
                f"    pos={stats['pos_mean']:.4f}  neg={stats['neg_mean']:.4f}  "
                f"gap={stats['gap']:+.4f}  auroc={stats['auroc']:.4f}"
            )

        del model
        torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — zero-shot geometry gap (pos_mean - neg_mean)")
    print(f"{'='*60}")
    print(f"{'dataset':>6}  {'instruct':>10}  {'base':>10}  {'emb':>10}  {'winner':>8}")
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], {})[r["variant"]] = r["gap"]
    for ds in args.datasets:
        row = by_ds.get(ds, {})
        vals = {v: row.get(v, float("nan")) for v in args.variants}
        best = max(vals, key=lambda v: vals[v]) if vals else "?"
        val_str = "  ".join(f"{vals.get(v, float('nan')):>10.4f}" for v in args.variants)
        print(f"{ds:>6}  {val_str}  {best:>8}")

    print(f"\nSUMMARY — zero-shot AUROC")
    print(f"{'dataset':>6}  {'instruct':>10}  {'base':>10}  {'emb':>10}  {'winner':>8}")
    by_ds_auc = {}
    for r in results:
        by_ds_auc.setdefault(r["dataset"], {})[r["variant"]] = r["auroc"]
    for ds in args.datasets:
        row = by_ds_auc.get(ds, {})
        vals = {v: row.get(v, float("nan")) for v in args.variants}
        best = max(vals, key=lambda v: vals[v]) if vals else "?"
        val_str = "  ".join(f"{vals.get(v, float('nan')):>10.4f}" for v in args.variants)
        print(f"{ds:>6}  {val_str}  {best:>8}")

    if args.output:
        import csv as csv_mod
        with open(args.output, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
