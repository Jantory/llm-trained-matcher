"""sr title-collision rebalancing ablation.

Motivation
----------
Analysis (analysis/ notebooks, 2026-07-03) found that sr's precision fragility
is driven almost entirely by "title-collision" negatives -- pairs where the two
records share an EXACT title but refer to different movies (e.g. two different
films both called "Running Scared"). These are the genuinely hard negatives in
sr; sr's aggregate lexical/token-Jaccard hard-negative rate looked near-zero
only because the signal is diluted by the long free-text blob field.

The rate of these negatives is wildly unbalanced across sr's ORIGINAL splits:

    train: 4/764  (0.5%)      valid: 1/256 (0.4%)      test: 73/254 (28.7%)

i.e. the model almost never sees a "same title, different movie" counterexample
during training/validation, then meets them at 57x the rate at test time. This
script does NOT invent any new pairs or labels -- it pools all 2183 existing
labeled pairs across the original train+valid+test and re-splits them with the
title-collision negatives allocated PROPORTIONALLY across the three new splits
(instead of concentrated in test), holding split sizes fixed at the originals
(1309/437/437) for exact comparability. Positives and non-collision negatives
are split the same proportional way, so aside from the collision-negative
distribution, the new splits are statistically like a fresh random re-split of
the same pool -- a single-variable change isolating the collision-distribution
hypothesis, independent of the (separate, already-tested) blob variable: the
serialized tables here are copied UNCHANGED (with the blob), so this ablation
does not overlap with data/create_blob_ablation.py.

Pool composition (verified): 2183 pairs = 909 positive, 1196 negative
non-collision, 78 negative title-collision.

Prediction: if the title-collision distribution shift is the causal driver of
sr's fragility, training on a rebalanced split should reduce the deployed
false-positive rate on collision negatives in the new test split, largely
independent of scale (no more "surprise" concentration of these hard negatives
at test time only).

Output: data/cleaned/semi-rel_rebalanced/{train,valid,test}.csv (same columns
as originals) + tableA_serialized.csv / tableB_serialized.csv copied verbatim.
Alias "srb" registered in utils/constants.py.

Usage: /gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python data/create_sr_rebalanced.py
"""
from __future__ import annotations
import csv
import os
import random
import shutil
import sys

csv.field_size_limit(10 ** 7)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import ds_common as C  # noqa: E402

SEED = 42
SRC = os.path.join(C.DATA_ROOT, "semi-rel")
DST = os.path.join(C.DATA_ROOT, "semi-rel_rebalanced")
SPLIT_SIZES = {"train": 1309, "valid": 437, "test": 437}  # match originals exactly


def get_field(text, keys):
    fields = dict(zip(C.attr_keys(text), C.attr_values(text)))
    for k in keys:
        if k in fields:
            return fields[k].strip().lower()
    return ""


def classify(row, A, B):
    if int(row["label"]) == 1:
        return "pos"
    a = A.get(row["ltable_id"], "")
    b = B.get(row["rtable_id"], "")
    ta = get_field(a, ["title"])
    tb = get_field(b, ["Name", "movie_name", "Title", "title"])
    return "neg_collision" if (ta and ta == tb) else "neg_other"


def proportional_alloc(n, sizes):
    """Largest-remainder allocation of n items across sizes (dict), summing to n."""
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(raw[k]) for k in raw}
    remainder = n - sum(alloc.values())
    order = sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)
    for k in order[:remainder]:
        alloc[k] += 1
    return alloc


def main():
    A, B, train, valid, test = C.load_dataset("sr")
    pool = train + valid + test
    assert len(pool) == sum(SPLIT_SIZES.values())

    buckets = {"pos": [], "neg_collision": [], "neg_other": []}
    for r in pool:
        buckets[classify(r, A, B)].append(r)

    print("pool composition:", {k: len(v) for k, v in buckets.items()})

    rng = random.Random(SEED)
    out = {"train": [], "valid": [], "test": []}
    for cat, rows in buckets.items():
        rows = rows[:]
        rng.shuffle(rows)
        alloc = proportional_alloc(len(rows), SPLIT_SIZES)
        i = 0
        for split in ("train", "valid", "test"):
            n = alloc[split]
            out[split].extend(rows[i:i + n])
            i += n
        print(f"  {cat:14} -> " + ", ".join(f"{s}={alloc[s]}" for s in SPLIT_SIZES))

    os.makedirs(DST, exist_ok=True)
    for split, rows in out.items():
        rng.shuffle(rows)  # interleave categories within the split
        path = os.path.join(DST, f"{split}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ltable_id", "rtable_id", "label"])
            w.writeheader()
            for r in rows:
                w.writerow({"ltable_id": r["ltable_id"], "rtable_id": r["rtable_id"],
                           "label": r["label"]})
        n_coll_neg = sum(1 for r in rows if classify(r, A, B) == "neg_collision"
                        and int(r["label"]) == 0)
        n_neg = sum(1 for r in rows if int(r["label"]) == 0)
        print(f"  wrote {path}: n={len(rows)}  "
              f"collision-negatives={n_coll_neg}/{n_neg} ({n_coll_neg/n_neg:.1%})")

    for table in ("tableA_serialized.csv", "tableB_serialized.csv"):
        shutil.copyfile(os.path.join(SRC, table), os.path.join(DST, table))
    print("Done. Register alias in utils/constants.py (srb -> semi-rel_rebalanced).")


if __name__ == "__main__":
    main()
