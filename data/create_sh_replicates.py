"""sh replicates: two independent re-samplings of sh from its unlabeled entity tables.

Motivation
----------
sh's paradigm/scale differences were traced to (A-format x B-format) source
combinations whose test positives have no counterpart among train positives:
(amazonA, bn_price) is all-negative in train/valid but all-positive in test,
and (amazonA, amazonA), (other, amazonA), (amazonA, other) never occur in train.
That attribution was made on one dataset. To check that the effect
reproduces, this script builds two NEW datasets with sh's exact split
structure from pairs that do not occur in sh. sh labels only 2,068 pairs, but
its entity tables hold ~22k / ~23k book records. New pairs are labeled with
auxiliary fields using rules validated against sh's gold labels (sh's labels
are ISBN-level: the same title in a different printing is a non-match):

    positive : the records share a valid ISBN (any ISBN field, raw tables)
             or (when ISBNs cannot decide, e.g. catalog-format records whose
               ISBNs are float-corrupted) title Jaccard >= 0.6, authors
               overlap, page count equal and publication year equal when
               both records have one
    negative : titles differ and either the author name sets are disjoint
               or both records have ISBNs and none is shared
The script prints the agreement of each rule with sh's gold labels.

Matched re-sampling: every gold pair of sh is replaced, in the same split, by a
new pair with the same label and (A-format, B-format) combination, the same
title-Jaccard bin, and the same "shared ISBN visible in the serialized text"
flag (60% of sh's positives show a shared ISBN to the model). So each
replicate has exactly sh's split sizes, label balance and per-split
combination x label counts, with one exception: some positive strata are
larger than the tables can supply twice over ((catalog, bn_full): ~200
candidates for 2 x 154; (amazonA, bn_full) competes with it for bn_full
records). Positives are therefore drawn test -> valid -> train, so any
shortfall lands in TRAIN only, where a missing (catalog, bn_full) positive is
replaced by a (catalog, other) one and any remaining shortfall of these or of
(amazonA, bn_full) by an (other, other) positive. All of these combinations
are seen in sh's train split with both labels, so the swaps do not touch the
combinations that carry the shift; the printed match quality lists every swap.
The two replicates are drawn alternately, pair by pair, so both get the same
share of scarce strata.
Neither replicate shares any record with sh's
labeled pairs or with the other replicate, and no record is used twice within
a replicate.

Each replicate also gets a balanced control built like data/create_sh_rebalanced.py:
the same pairs re-split stratified by (label, A-format, B-format).

Output: data/cleaned/semi-heter_rep{1,2}/ and data/cleaned/semi-heter_rep{1,2}_balanced/
({train,valid,test}.csv + serialized tables copied verbatim). Aliases sh1, sh2,
sh1b, sh2b in utils/constants.py.

Usage: /gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python data/create_sh_replicates.py
"""
from __future__ import annotations

import collections
import csv
import json
import os
import random
import re
import shutil

csv.field_size_limit(10 ** 8)

SEED = 42
N_REPLICATES = 2
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned")
SRC = os.path.join(DATA_ROOT, "semi-heter")
SPLITS = ("train", "valid", "test")
SPLIT_SIZES = {"train": 1240, "valid": 414, "test": 414}  # sh's original sizes
MAX_NEG_CANDIDATES_PER_A = 60
TOKEN_DF_CAP = 300
TITLE_RULE_MIN_J = 0.6
# Train-only combinations used as substitutes when catalog positives run out.
POSITIVE_SUBSTITUTES = {
    ("catalog", "bn_full"): [("catalog", "other"), ("other", "other")],
    ("catalog", "other"): [("catalog", "bn_full"), ("other", "other")],
    ("amazonA", "bn_full"): [("other", "other")],
}
SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}  # shortfalls (if any) land in train
J_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
AUTHOR_STOP = {"editor", "editors", "eds", "jr", "phd", "and", "the", "by", "foreword",
               "introduction", "author"}
ISBN_RE = re.compile(r"\b(97[89]\d{10}|\d{9}[\dXx])\b")


# ── record features ───────────────────────────────────────────────────────────
def norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())).strip()


def serialized_keys(text: str) -> set:
    return {k.strip().lower() for k in re.findall(r"\[([^\]]+)\]", text)}


def dialect(text: str) -> str:
    """Source format of a record, from its serialized attribute names (as in create_sh_rebalanced.py)."""
    ks = serialized_keys(text)
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


def isbns_in(text: str) -> set:
    """Valid-looking ISBNs (the float-corrupted 978000... values are dropped)."""
    return {m.upper() for m in ISBN_RE.findall(text.replace("-", ""))
            if not re.fullmatch(r"97[89]0+", m)}


def walk(v, key=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from walk(x, k)
    elif isinstance(v, list):
        for x in v:
            yield from walk(x, key)
    elif v is not None:
        yield key, v


def record_isbns(r) -> set:
    return isbns_in(" ".join(str(v) for _, v in walk({k: v for k, v in r.items() if k != "id"})))


def record_title(r) -> str:
    return r.get("title") or r.get("Title") or ""


def record_authors(r) -> set:
    vals = [str(v) for k, v in walk(r) if "author" in k.lower()]
    return set(re.findall(r"[a-z]{3,}", " ".join(vals).lower())) - AUTHOR_STOP


def record_years(r) -> set:
    """Publication years modulo 100 (sh mixes 4-digit years with 2-digit pubyear / m/d/yy)."""
    ys = set()
    for k, v in walk(r):
        kl = k.lower()
        if not any(t in kl for t in ("pubyear", "date", "year")):
            continue
        s = str(v).strip()
        ys |= {int(y) % 100 for y in re.findall(r"\b(19\d\d|20\d\d)\b", s)}
        if kl == "pubyear":
            try:
                ys.add(int(float(s)) % 100)
            except ValueError:
                pass
        m = re.match(r"^\d{1,2}/\d{1,2}/(\d{1,2})$", s)
        if m:
            ys.add(int(m.group(1)) % 100)
    return ys


def record_pages(r) -> set:
    out = set()
    for k, v in walk(r):
        if "page" in k.lower():
            out |= {int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", str(v)) if float(x) > 0}
    return out


def jaccard(x: set, y: set) -> float:
    return len(x & y) / max(len(x | y), 1)


def j_bin(j: float) -> int:
    if j == 0.0:
        return 0
    if j == 1.0:
        return len(J_BINS)
    return next(i for i in range(1, len(J_BINS)) if j <= J_BINS[i])


def load_jsonl(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[str(r["id"])] = r
    return out


class Side:
    def __init__(self, raw_path, ser_path):
        self.raw = load_jsonl(raw_path)
        self.ser = {r["id"]: r["text"] for r in csv.DictReader(open(ser_path))}
        self.fmt = {i: dialect(t) for i, t in self.ser.items()}
        self.title = {i: norm(record_title(r)) for i, r in self.raw.items()}
        self.ttok = {i: set(t.split()) for i, t in self.title.items()}
        self.isbn = {i: record_isbns(r) for i, r in self.raw.items()}
        self.vis_isbn = {i: isbns_in(t) for i, t in self.ser.items()}
        self.authors = {i: record_authors(r) for i, r in self.raw.items()}
        self.years = {i: record_years(r) for i, r in self.raw.items()}
        self.pages = {i: record_pages(r) for i, r in self.raw.items()}


class Records:
    def __init__(self):
        self.A = Side(os.path.join(SRC, "tableA.jsonl"), os.path.join(SRC, "tableA_serialized.csv"))
        self.B = Side(os.path.join(SRC, "tableB.jsonl"), os.path.join(SRC, "tableB_serialized.csv"))

    def combo(self, i, j):
        return self.A.fmt[i], self.B.fmt[j]

    def features(self, i, j):
        A, B = self.A, self.B
        return {"j": jaccard(A.ttok[i], B.ttok[j]),
                "vis": bool(A.vis_isbn[i] & B.vis_isbn[j])}

    def title_rule(self, i, j):
        """Identity without a shared ISBN (catalog records have corrupted ISBNs)."""
        A, B = self.A, self.B
        if A.isbn[i] and B.isbn[j]:
            return False  # both have ISBNs but none shared: a different printing
        if jaccard(A.ttok[i], B.ttok[j]) < TITLE_RULE_MIN_J:
            return False
        if not (A.authors[i] & B.authors[j] and A.pages[i] & B.pages[j]):
            return False
        return not (A.years[i] and B.years[j]) or bool(A.years[i] & B.years[j])

    def rule_label(self, i, j):
        """1 / 0 / None (= no confident label)."""
        A, B = self.A, self.B
        if A.isbn[i] & B.isbn[j]:
            return 1
        if self.title_rule(i, j):
            return 1
        if A.title[i] != B.title[j]:
            if A.authors[i] and B.authors[j] and not (A.authors[i] & B.authors[j]):
                return 0
            if A.isbn[i] and B.isbn[j]:
                return 0
        return None


def match_keys(label, combo, f, split):
    jb = j_bin(f["j"])
    nearby = sorted(range(len(J_BINS) + 1), key=lambda b: abs(b - jb))
    keys = [(label, combo, jb, f["vis"]), (label, combo, jb)]
    keys += [(label, combo, b, f["vis"]) for b in nearby[1:]]
    keys += [(label, combo, b) for b in nearby[1:]]
    if label == 1 and split == "train":
        for sub in POSITIVE_SUBSTITUTES.get(combo, []):
            keys += [(label, sub, b) for b in nearby]
    return keys


def candidate_keys(label, combo, f):
    jb = j_bin(f["j"])
    return [(label, combo, jb, f["vis"]), (label, combo, jb)]


# ── candidate mining ──────────────────────────────────────────────────────────
def mine_candidates(rec: Records, excluded_a: set, excluded_b: set, rng: random.Random):
    A, B = rec.A, rec.B
    a_ids = [i for i in A.raw if i not in excluded_a]
    b_ids = [j for j in B.raw if j not in excluded_b]

    by_isbn = collections.defaultdict(list)
    by_title = collections.defaultdict(list)
    tok_index = collections.defaultdict(list)
    for j in b_ids:
        for x in B.isbn[j]:
            by_isbn[x].append(j)
        if B.title[j]:
            by_title[B.title[j]].append(j)
        for t in B.ttok[j]:
            tok_index[t].append(j)
    tok_index = {t: v for t, v in tok_index.items() if len(v) <= TOKEN_DF_CAP}

    buckets = collections.defaultdict(list)
    n_labeled = collections.Counter()
    seen = set()

    def add(i, j, want):
        if (i, j, want) in seen:
            return
        seen.add((i, j, want))
        label = rec.rule_label(i, j)
        if label is None or label != want:
            return
        n_labeled[label] += 1
        for key in candidate_keys(label, rec.combo(i, j), rec.features(i, j)):
            buckets[key].append((i, j))

    for i in a_ids:
        # positives: shared ISBN, or identical title (strict title/author/year/pages rule)
        for x in A.isbn[i]:
            for j in by_isbn.get(x, ()):
                add(i, j, 1)
        for j in by_title.get(A.title[i], ()):
            add(i, j, 1)
        pool = set()
        for t in A.ttok[i]:
            pool.update(tok_index.get(t, ()))
        if not A.isbn[i]:  # title rule is the only way to find this record's matches
            for j in pool:
                add(i, j, 1)
        # negatives: blocked on rare title tokens
        pool = sorted(pool)
        if len(pool) > MAX_NEG_CANDIDATES_PER_A:
            pool = rng.sample(pool, MAX_NEG_CANDIDATES_PER_A)
        for j in pool:
            add(i, j, 0)
    for v in buckets.values():
        rng.shuffle(v)
    print("rule-labeled candidate pairs:", dict(n_labeled))
    return buckets


# ── matched re-sampling ───────────────────────────────────────────────────────
def build_replicates(gold, buckets, used_a, used_b, rng):
    """Replace every gold pair by a free candidate with the same matching key.

    Replicates are filled alternately, pair by pair, so scarce strata are shared evenly.
    """
    order = list(range(len(gold)))
    rng.shuffle(order)
    # positives first (scarce, and a record taken by a negative is lost to them),
    # then test -> valid -> train, then scarce keys first
    order.sort(key=lambda k: (gold[k][3] == 0, SPLIT_PRIORITY[gold[k][0]],
                              len(buckets.get(candidate_keys(*gold[k][3:])[0], ()))))
    outs = [{s: [] for s in SPLITS} for _ in range(N_REPLICATES)]
    qualities = [collections.Counter() for _ in range(N_REPLICATES)]
    cursor = collections.defaultdict(int)
    for k in order:
        split, _, _, label, combo, f = gold[k]
        for rep in range(N_REPLICATES):
            picked = None
            for level, key in enumerate(match_keys(label, combo, f, split)):
                lst = buckets.get(key, [])
                while cursor[key] < len(lst):
                    i, j = lst[cursor[key]]
                    cursor[key] += 1
                    if i not in used_a and j not in used_b:
                        picked = (i, j)
                        break
                if picked:
                    tag = "exact" if level == 0 else f"relaxed{level}"
                    if key[1] != combo:
                        tag = f"substituted {combo}->{key[1]}"
                    qualities[rep][(label, tag)] += 1
                    break
            if picked is None:
                raise RuntimeError(f"no candidate left for label={label} combo={combo} {f}")
            used_a.add(picked[0])
            used_b.add(picked[1])
            outs[rep][split].append({"ltable_id": picked[0], "rtable_id": picked[1], "label": label})
    for out in outs:
        for s in SPLITS:
            rng.shuffle(out[s])
    return outs, qualities


def proportional_alloc(n, sizes):
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(raw[k]) for k in raw}
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def balanced_split(splits, rec, rng):
    """shb-style control: same pairs, stratified by (label, A-format, B-format)."""
    pool = [r for s in SPLITS for r in splits[s]]
    strata = collections.defaultdict(list)
    for r in pool:
        strata[(r["label"],) + rec.combo(r["ltable_id"], r["rtable_id"])].append(r)
    out = {s: [] for s in SPLITS}
    for key in sorted(strata):
        rows = strata[key][:]
        rng.shuffle(rows)
        alloc = proportional_alloc(len(rows), SPLIT_SIZES)
        start = 0
        for s in SPLITS:
            out[s].extend(rows[start:start + alloc[s]])
            start += alloc[s]
    for s in SPLITS:
        rng.shuffle(out[s])
    return out


def write_dataset(name, splits):
    dst = os.path.join(DATA_ROOT, name)
    os.makedirs(dst, exist_ok=True)
    for s in SPLITS:
        with open(os.path.join(dst, f"{s}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ltable_id", "rtable_id", "label"])
            w.writeheader()
            w.writerows(splits[s])
    for table in ("tableA_serialized.csv", "tableB_serialized.csv"):
        shutil.copyfile(os.path.join(SRC, table), os.path.join(dst, table))
    return dst


def describe(name, splits, rec):
    print(f"  {name}")
    print(f"    {'combo':24} " + " ".join(f"{s:>9}" for s in SPLITS) + "   (neg/pos)")
    cnt = collections.defaultdict(collections.Counter)
    for s in SPLITS:
        for r in splits[s]:
            cnt[rec.combo(r["ltable_id"], r["rtable_id"])][(s, int(r["label"]))] += 1
    for combo in sorted(cnt, key=lambda c: -sum(cnt[c].values())):
        c = cnt[combo]
        print(f"    {str(combo):24} " + " ".join(f"{c[(s, 0)]:4}/{c[(s, 1)]:<4}" for s in SPLITS))


def main():
    rng = random.Random(SEED)
    rec = Records()
    gold = []
    for s in SPLITS:
        for r in csv.DictReader(open(os.path.join(SRC, f"{s}.csv"))):
            i, j, label = r["ltable_id"], r["rtable_id"], int(r["label"])
            gold.append((s, None, (i, j), label, rec.combo(i, j), rec.features(i, j)))
    assert {s: sum(g[0] == s for g in gold) for s in SPLITS} == SPLIT_SIZES

    agree = collections.Counter((g[3], rec.rule_label(*g[2])) for g in gold)
    print("rule label vs sh gold (gold, rule):", dict(sorted(agree.items(), key=str)))

    used_a = {g[2][0] for g in gold}
    used_b = {g[2][1] for g in gold}
    buckets = mine_candidates(rec, used_a, used_b, rng)

    print("original sh:")
    describe("semi-heter", {s: [{"ltable_id": g[2][0], "rtable_id": g[2][1], "label": g[3]}
                                for g in gold if g[0] == s] for s in SPLITS}, rec)
    outs, qualities = build_replicates(gold, buckets, used_a, used_b, rng)
    for rep, (splits, quality) in enumerate(zip(outs, qualities), start=1):
        print(f"replicate {rep}: match quality {dict(sorted(quality.items(), key=str))}")
        describe(write_dataset(f"semi-heter_rep{rep}", splits), splits, rec)
        bal = balanced_split(splits, rec, rng)
        describe(write_dataset(f"semi-heter_rep{rep}_balanced", bal), bal, rec)
    print("Done. Aliases: sh1/sh2 -> semi-heter_rep{1,2}, sh1b/sh2b -> semi-heter_rep{1,2}_balanced.")


if __name__ == "__main__":
    main()
