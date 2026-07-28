from __future__ import annotations
import csv, os, re

# Allow arbitrarily long serialized fields.
csv.field_size_limit(10 ** 7)

DATA_ROOT  = os.path.join("..", "data", "cleaned")
RESULTS_ROOT = os.path.join("..", "results")
OUT_DIR    = os.path.join("data")

# alias -> directory under data/cleaned
DIRMAP = {
    "ab": "abt_buy", "ag": "amazon_google", "ds": "dblp_scholar",
    "rt": "rel-text", "sh": "semi-heter", "sr": "semi-rel",
    "wa": "walmart_amazon", "w20": "wdc20", "w80": "wdc80",
}
ALIASES = list(DIRMAP)  # canonical display order (the 9 main datasets only)

# ablation variants, not part of the canonical 9 (kept out of ALIASES)
DIRMAP.update({
    "srn": "semi-rel_noblob",       # free-text-blob ablation
    "src": "semi-rel_core",         # free-text-blob ablation (dose-response)
    "srb": "semi-rel_rebalanced",   # title-collision rebalancing ablation
    "shb": "semi-heter_rebalanced", # dialect-combination rebalancing ablation
})

PARADIGMS = ["bi_encoder", "cross_encoder", "generative"]
SCALES    = ["0.6b", "4b", "8b"]


def scale_from_path(path: str) -> str | None:
    """Extract scale from a results/main_<scale>/... path."""
    m = re.search(r"main_(0\.6b|4b|8b)", path)
    return m.group(1) if m else None


def variant_from_modeldir(name: str) -> str | None:
    """Map a run directory name to a checkpoint variant.

    Qwen3-8B-Base_seed42      -> 'base'
    Qwen3-8B_seed42           -> 'instruct'
    Qwen3-Embedding-8B_seed42 -> 'emb'
    """
    stem = name.split("_seed")[0]
    if "Embedding" in stem:
        return "emb"
    if stem.endswith("-Base"):
        return "base"
    return "instruct"


def seed_from_modeldir(name: str) -> int | None:
    m = re.search(r"_seed(\d+)", name)
    return int(m.group(1)) if m else None


def load_table(path: str) -> dict[str, str]:
    """id -> serialized text."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = r["text"]
    return out


def load_dataset(alias: str):
    """Return (tableA, tableB, train_rows, valid_rows, test_rows)."""
    d = os.path.join(DATA_ROOT, DIRMAP[alias])
    A = load_table(os.path.join(d, "tableA_serialized.csv"))
    B = load_table(os.path.join(d, "tableB_serialized.csv"))

    def rows(fn):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            return []
        with open(p, newline="") as f:
            return list(csv.DictReader(f))

    return A, B, rows("train.csv"), rows("valid.csv"), rows("test.csv")


def attr_keys(text: str) -> list[str]:
    """Serialized text is '[attr] value [attr] value ...'; return the attr keys."""
    return re.findall(r"\[([^\]]+)\]", text)


def attr_values(text: str) -> list[str]:
    """Return the value strings between successive [attr] markers."""
    parts = re.split(r"\[[^\]]+\]", text)
    return [p.strip() for p in parts[1:]]  # drop leading '' before first marker


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
