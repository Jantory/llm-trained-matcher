"""sh dialect-combination rebalancing ablation.

Motivation
----------
Analysis (`analysis/sh_schema_shift.py`) found sh's paradigm/scale differences
concentrated entirely on (A-dialect x B-dialect) combinations absent from
train: 131/159 (82%) of test positives use a combination seen 0x in train or
valid; validation is 100% one heavily-trained combination, so val F1=1.0 is
structurally uninformative about test.

Unlike sr's title-collision shift (a rare-but-PRESENT subpopulation, fixed
completely by rebalancing -- see data/create_sr_rebalanced.py), sh's shift is
about combinations that are, in the ORIGINAL split, literally ABSENT from
train (not just rare). This script tests the causal question directly: if the
current paradigm gap (cross format-bound, ~flat with scale; generative
recovers with scale) is really about combination novelty, does giving every
model combination-specific training examples close the gap?

Pool composition (2068 labeled pairs = train 1240 + valid 414 + test 414),
by (label, A-dialect, B-dialect):

    positive combos: (amazonA,bn_full)=327 (catalog,bn_full)=154
        (other,other)=100 (amazonA,bn_price)=99 (catalog,other)=77
        (amazonA,amazonA)=18 (other,amazonA)=9 (amazonA,other)=5
        (other,bn_full)=1
    negative combos: 9 combos, 55-406 each (all already reasonably spread)

The combinations currently 100% concentrated in test --
(amazonA,bn_price)=99, (amazonA,amazonA)=18, (other,amazonA)=9,
(amazonA,other)=5 -- account for the entire 131-pair "unseen" gap. This
script pools all 2068 pairs and re-splits them, stratified by (label, combo),
so every combination lands proportionally in every split (matching original
split sizes 1240/414/414) instead of being partitioned by combination. No new
pairs or labels are invented; serialized tables are copied UNCHANGED.

Prediction: if the paradigm gap is purely combination-novelty, cross's recall
on now-represented combinations should rise toward the "seen" baseline
(~0.95) after rebalancing, and the cross-generative gap should shrink. If a
meaningful gap remains even with training examples for every combination,
some of the original "unseen-combo" effect was capacity/architecture, not
pure novelty -- worth distinguishing from sr's complete, permanent fix: even
if rebalancing closes the gap on TODAY's observed combinations, sh's
underlying task (new sources keep appearing) means tomorrow's deployment will
again present un-enumerated combinations -- unlike sr's fixed, enumerable
trap, this fix cannot be "permanent" the way srb's was.

Output: data/cleaned/semi-heter_rebalanced/{train,valid,test}.csv (same
columns as originals) + tableA_serialized.csv / tableB_serialized.csv copied
verbatim. Alias "shb" registered in utils/constants.py.

Usage: /gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python data/create_sh_rebalanced.py
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
SRC = os.path.join(C.DATA_ROOT, "semi-heter")
DST = os.path.join(C.DATA_ROOT, "semi-heter_rebalanced")
SPLIT_SIZES = {"train": 1240, "valid": 414, "test": 414}  # match originals exactly


def dialect(text: str) -> str:
    ks = set(k.lower() for k in C.attr_keys(text))
    if "pubyear" in ks or "pubmonth" in ks:
        return "catalog"
    if "price" in ks or "product dimensions" in ks or "shipping weight" in ks:
        return "amazonA"
    if "publication_date" in ks and "isbn13" in ks:
        return "bn_full"
    if "nook_book" in ks or "paperback" in ks or "sales_rank" in ks:
        return "bn_price"
    if "usedprice" in ks or "newprice" in ks:
        return "used"
    return "other"


def stratum(row, A, B):
    a = A.get(row["ltable_id"], "")
    b = B.get(row["rtable_id"], "")
    return (int(row["label"]), dialect(a), dialect(b))


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
    A, B, train, valid, test = C.load_dataset("sh")
    pool = train + valid + test
    assert len(pool) == sum(SPLIT_SIZES.values())

    buckets = {}
    for r in pool:
        buckets.setdefault(stratum(r, A, B), []).append(r)

    print("pool composition (label, A-dialect, B-dialect) -> n:")
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        print(f"  {k}: {len(buckets[k])}")

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

    os.makedirs(DST, exist_ok=True)
    for split, rows in out.items():
        rng.shuffle(rows)
        path = os.path.join(DST, f"{split}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ltable_id", "rtable_id", "label"])
            w.writeheader()
            for r in rows:
                w.writerow({"ltable_id": r["ltable_id"], "rtable_id": r["rtable_id"],
                           "label": r["label"]})
        # sanity: report the 4 previously-unseen combos' presence in this split
        watch = {("amazonA", "bn_price"), ("amazonA", "amazonA"),
                 ("other", "amazonA"), ("amazonA", "other")}
        pos_watch = sum(1 for r in rows if int(r["label"]) == 1
                       and (dialect(A[r["ltable_id"]]), dialect(B[r["rtable_id"]])) in watch)
        n_pos = sum(1 for r in rows if int(r["label"]) == 1)
        print(f"  wrote {path}: n={len(rows)}  "
              f"previously-unseen-combo positives={pos_watch}/{n_pos} "
              f"({pos_watch/n_pos:.1%})" if n_pos else f"  wrote {path}: n={len(rows)}")

    for table in ("tableA_serialized.csv", "tableB_serialized.csv"):
        shutil.copyfile(os.path.join(SRC, table), os.path.join(DST, table))
    print("Done. Register alias in utils/constants.py (shb -> semi-heter_rebalanced).")


if __name__ == "__main__":
    main()
