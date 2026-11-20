"""sr replicates: two independent re-samplings of sr from its unlabeled entity tables.

Motivation
----------
sr's precision fragility was traced to "title-collision" negatives (same exact
title, different movie) that are 0.5% of train negatives but 28.7% of test
negatives. That attribution was made on one dataset. To check that the effect
reproduces, this script builds two NEW datasets with sr's exact split
structure from pairs that do not occur in sr. sr labels only 2,183 pairs, but
its entity tables hold ~29k IMDB and ~33k Rotten-Tomatoes-style records. New
pairs are labeled with auxiliary fields (year, director) using rules that were
validated against sr's gold labels:

    same-title positive   : titles equal, |year diff| <= 1, directors overlap
                            (gold: 804 pos / 1 neg)
    title-collision neg.  : titles equal, |year diff| >= 3, directors disjoint
                            (gold: 55 neg / 1 pos)
    other negative        : title Jaccard < 0.5 and
                            (|year diff| >= 2 or directors disjoint, or
                             directors overlap, |year diff| <= 1, no shared actor)
                            (gold: 1,011 neg / 6 pos)

Matched re-sampling: every gold pair of sr is replaced, in the same split, by a
new pair of the same kind (positive / title-collision negative / other
negative) and the same B-record format, and for other negatives also the same
title-Jaccard bin, director-overlap flag and year-difference bucket. So each
replicate has exactly sr's split sizes, label balance, collision distribution
(4 / 1 / 73 collision negatives in train / valid / test) and B-format mix per
split. Neither replicate shares any record with sr's labeled pairs or with the
other replicate, and no record is used twice within a replicate.

Known deviation: no rule labels different-title positives reliably (best
candidate rule: 83% precision on gold), so sr's 50 different-title positives
(5% of positives) are replaced by same-title positives in the same split and
B format.

Each replicate also gets a balanced control built like data/create_sr_rebalanced.py:
the same pairs re-split with every kind allocated proportionally.

Output: data/cleaned/semi-rel_rep{1,2}/ and data/cleaned/semi-rel_rep{1,2}_balanced/
({train,valid,test}.csv + serialized tables copied verbatim). Aliases sr1, sr2,
sr1b, sr2b in utils/constants.py.

Usage: /gpfs/home6/zzhang/miniconda3/envs/train-embedder/bin/python data/create_sr_replicates.py
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
SRC = os.path.join(DATA_ROOT, "semi-rel")
SPLITS = ("train", "valid", "test")
SPLIT_SIZES = {"train": 1309, "valid": 437, "test": 437}  # sr's original sizes
MAX_NEG_CANDIDATES_PER_A = 60
TOKEN_DF_CAP = 300
J_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# ── record features ───────────────────────────────────────────────────────────
def norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())).strip()


def name_tokens(s) -> set:
    return set(re.findall(r"[a-z]{3,}", str(s or "").lower()))


def b_title(b) -> str:
    for k in ("Name", "movie_name", "Title", "title"):
        if b.get(k):
            return b[k]
    return ""


def b_format(b) -> str:
    """Source format of a B record (they come from several sites with different schemas)."""
    if b.get("Name"):
        return "rt"   # Name / YearRange / ReleaseDate / Director / Description ...
    if b.get("movie_name"):
        return "mn"   # movie_name / directors / movie_rating / genre / duration
    if b.get("Title"):
        return "T"    # Title / Year / Director / Cast / Summary ...
    if b.get("title"):
        return "t"    # title / year / director(s) / writers / actors
    return "?"


def b_years(b) -> set:
    ys = set()
    for k in ("YearRange", "ReleaseDate", "year", "Year"):
        if b.get(k):
            ys |= {int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", str(b[k]))}
    return ys


def b_director(b) -> str:
    for k in ("Director", "director", "directors", "director(s)"):
        if b.get(k):
            return str(b[k])
    return ""


def b_actors(b) -> set:
    out = set()
    for k in ("actors", "Cast", "stars"):
        v = b.get(k)
        if v:
            items = v if isinstance(v, list) else str(v).split(",")
            out |= {x.strip().lower() for x in items if x.strip()}
    return out


def a_year(a):
    m = re.findall(r"\b(19\d\d|20\d\d)\b", a.get("year") or "")
    return int(m[0]) if m else None


def a_actors(a) -> set:
    return {x.strip().lower() for x in (a.get("actors") or "").split(",") if x.strip()}


def jaccard(x: set, y: set) -> float:
    return len(x & y) / max(len(x | y), 1)


def j_bin(j: float) -> int:
    if j == 0.0:
        return 0
    if j == 1.0:
        return len(J_BINS)
    return next(i for i in range(1, len(J_BINS)) if j <= J_BINS[i])


class Records:
    def __init__(self):
        self.A = {r["id"]: r for r in csv.DictReader(open(os.path.join(SRC, "tableA.csv")))}
        self.B = {}
        for line in open(os.path.join(SRC, "tableB.jsonl")):
            r = json.loads(line)
            self.B[str(r["id"])] = r
        self.a_title = {i: norm(r["title"]) for i, r in self.A.items()}
        self.b_title = {i: norm(b_title(r)) for i, r in self.B.items()}
        self.a_ttok = {i: set(t.split()) for i, t in self.a_title.items()}
        self.b_ttok = {i: set(t.split()) for i, t in self.b_title.items()}
        self.a_year = {i: a_year(r) for i, r in self.A.items()}
        self.b_years = {i: b_years(r) for i, r in self.B.items()}
        self.a_dir = {i: name_tokens(r.get("director")) for i, r in self.A.items()}
        self.b_dir = {i: name_tokens(b_director(r)) for i, r in self.B.items()}
        self.a_act = {i: a_actors(r) for i, r in self.A.items()}
        self.b_act = {i: b_actors(r) for i, r in self.B.items()}
        self.b_fmt = {i: b_format(r) for i, r in self.B.items()}

    def pair_features(self, i, j):
        ya, yb = self.a_year[i], self.b_years[j]
        dy = min(abs(ya - y) for y in yb) if ya is not None and yb else None
        da, db = self.a_dir[i], self.b_dir[j]
        dov = bool(da & db) if da and db else None
        aa, ab = self.a_act[i], self.b_act[j]
        aov = len(aa & ab) if aa and ab else None
        same = bool(self.a_title[i]) and self.a_title[i] == self.b_title[j]
        return {"same_title": same, "dy": dy, "dir_overlap": dov, "actor_overlap": aov,
                "j": jaccard(self.a_ttok[i], self.b_ttok[j])}

    def rule_label(self, f):
        """'pos' / 'coll' / 'neg' / None (= no confident label)."""
        dy, dov, aov = f["dy"], f["dir_overlap"], f["actor_overlap"]
        if f["same_title"]:
            if dy is not None and dy <= 1 and dov:
                return "pos"
            if dy is not None and dy >= 3 and dov is False:
                return "coll"
            return None
        if f["j"] < 0.5:
            if (dy is not None and dy >= 2) or dov is False:
                return "neg"
            if dov and dy is not None and dy <= 1 and aov == 0:
                return "neg"
        return None


def dy_bucket(dy):
    return "none" if dy is None else ("<=1" if dy <= 1 else ">=2")


def match_keys(kind, fmt, f):
    """Matching keys from strictest to loosest; the first one with a free candidate wins."""
    if kind in ("pos", "coll"):
        return [(kind, fmt)]
    jb = j_bin(f["j"])
    nearby = sorted(range(len(J_BINS)), key=lambda b: abs(b - jb))
    keys = [(kind, fmt, jb, f["dir_overlap"], dy_bucket(f["dy"])),
            (kind, fmt, jb, f["dir_overlap"]),
            (kind, fmt, jb)]
    keys += [(kind, fmt, b) for b in nearby[1:]]
    return keys


def candidate_keys(kind, fmt, f):
    if kind in ("pos", "coll"):
        return [(kind, fmt)]
    jb = j_bin(f["j"])
    return [(kind, fmt, jb, f["dir_overlap"], dy_bucket(f["dy"])),
            (kind, fmt, jb, f["dir_overlap"]),
            (kind, fmt, jb)]


# ── candidate mining ──────────────────────────────────────────────────────────
def mine_candidates(rec: Records, excluded_a: set, excluded_b: set, rng: random.Random):
    a_ids = [i for i in rec.A if i not in excluded_a]
    b_ids = [j for j in rec.B if j not in excluded_b]

    by_title = collections.defaultdict(list)
    tok_index = collections.defaultdict(list)
    dir_index = collections.defaultdict(list)
    for j in b_ids:
        if rec.b_title[j]:
            by_title[rec.b_title[j]].append(j)
        for t in rec.b_ttok[j]:
            tok_index[t].append(j)
        for t in rec.b_dir[j]:
            dir_index[t].append(j)
    tok_index = {t: v for t, v in tok_index.items() if len(v) <= TOKEN_DF_CAP}
    dir_index = {t: v for t, v in dir_index.items() if len(v) <= TOKEN_DF_CAP}

    buckets = collections.defaultdict(list)
    n_labeled = collections.Counter()
    for i in a_ids:
        # same-title pairs -> positives and title-collision negatives
        for j in by_title.get(rec.a_title[i], ()):
            f = rec.pair_features(i, j)
            kind = rec.rule_label(f)
            if kind in ("pos", "coll"):
                n_labeled[kind] += 1
                for key in candidate_keys(kind, rec.b_fmt[j], f):
                    buckets[key].append((i, j))
        # different-title pairs -> other negatives (blocked on title and director tokens)
        pool = set()
        for t in rec.a_ttok[i]:
            pool.update(tok_index.get(t, ()))
        for t in rec.a_dir[i]:
            pool.update(dir_index.get(t, ()))
        pool = sorted(pool)
        if len(pool) > MAX_NEG_CANDIDATES_PER_A:
            pool = rng.sample(pool, MAX_NEG_CANDIDATES_PER_A)
        for j in pool:
            f = rec.pair_features(i, j)
            if f["same_title"]:
                continue
            if rec.rule_label(f) == "neg":
                n_labeled["neg"] += 1
                for key in candidate_keys("neg", rec.b_fmt[j], f):
                    buckets[key].append((i, j))
    for v in buckets.values():
        rng.shuffle(v)
    print("rule-labeled candidate pairs:", dict(n_labeled))
    return buckets


# ── matched re-sampling ───────────────────────────────────────────────────────
def gold_kind(rec, i, j, label):
    if label == 1:
        return "pos"
    return "coll" if rec.pair_features(i, j)["same_title"] else "neg"


def build_replicate(gold, rec, buckets, used_a, used_b, rng):
    """Replace every gold pair by a free candidate with the same matching key."""
    order = list(range(len(gold)))
    rng.shuffle(order)
    # scarce keys first, so rare strata are not starved by record reuse
    order.sort(key=lambda k: len(buckets.get(match_keys(*gold[k][3:])[0], ())))
    out = {s: [] for s in SPLITS}
    quality = collections.Counter()
    cursor = collections.defaultdict(int)
    for k in order:
        split, label, _, kind, fmt, f = gold[k]
        picked = None
        for level, key in enumerate(match_keys(kind, fmt, f)):
            lst = buckets.get(key, [])
            while cursor[key] < len(lst):
                i, j = lst[cursor[key]]
                cursor[key] += 1
                if i not in used_a and j not in used_b:
                    picked = (i, j)
                    break
            if picked:
                quality[(kind, "exact" if level == 0 else f"relaxed{level}")] += 1
                break
        if picked is None:
            raise RuntimeError(f"no candidate left for {kind} / {fmt} / {f}")
        used_a.add(picked[0])
        used_b.add(picked[1])
        out[split].append({"ltable_id": picked[0], "rtable_id": picked[1], "label": label})
    for s in SPLITS:
        rng.shuffle(out[s])
    return out, quality


def proportional_alloc(n, sizes):
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(raw[k]) for k in raw}
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def balanced_split(splits, rec, rng):
    """srb-style control: same pairs, every kind allocated proportionally to split sizes."""
    pool = [r for s in SPLITS for r in splits[s]]
    by_kind = collections.defaultdict(list)
    for r in pool:
        by_kind[gold_kind(rec, r["ltable_id"], r["rtable_id"], r["label"])].append(r)
    out = {s: [] for s in SPLITS}
    for kind in sorted(by_kind):
        rows = by_kind[kind][:]
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
    for s in SPLITS:
        kinds = collections.Counter(gold_kind(rec, r["ltable_id"], r["rtable_id"], r["label"]) for r in splits[s])
        fmts = collections.Counter((gold_kind(rec, r["ltable_id"], r["rtable_id"], r["label"]),
                                    rec.b_fmt[r["rtable_id"]]) for r in splits[s])
        n_neg = kinds["coll"] + kinds["neg"]
        print(f"    {s:5} n={len(splits[s]):4} pos={kinds['pos']:3} neg={n_neg:3} "
              f"collision={kinds['coll']:2} ({kinds['coll'] / max(n_neg, 1):.1%})  "
              f"by (kind, B-format): {dict(sorted(fmts.items()))}")


def main():
    rng = random.Random(SEED)
    rec = Records()
    gold = []
    for s in SPLITS:
        for r in csv.DictReader(open(os.path.join(SRC, f"{s}.csv"))):
            i, j, label = r["ltable_id"], r["rtable_id"], int(r["label"])
            gold.append((s, label, (i, j), gold_kind(rec, i, j, label), rec.b_fmt[j], rec.pair_features(i, j)))
    assert {s: sum(g[0] == s for g in gold) for s in SPLITS} == SPLIT_SIZES

    used_a = {g[2][0] for g in gold}
    used_b = {g[2][1] for g in gold}
    buckets = mine_candidates(rec, used_a, used_b, rng)

    print("original sr:")
    describe("semi-rel", {s: [{"ltable_id": g[2][0], "rtable_id": g[2][1], "label": g[1]}
                              for g in gold if g[0] == s] for s in SPLITS}, rec)
    for rep in range(1, N_REPLICATES + 1):
        splits, quality = build_replicate(gold, rec, buckets, used_a, used_b, rng)
        print(f"replicate {rep}: match quality {dict(sorted(quality.items()))}")
        describe(write_dataset(f"semi-rel_rep{rep}", splits), splits, rec)
        bal = balanced_split(splits, rec, rng)
        describe(write_dataset(f"semi-rel_rep{rep}_balanced", bal), bal, rec)
    print("Done. Aliases: sr1/sr2 -> semi-rel_rep{1,2}, sr1b/sr2b -> semi-rel_rep{1,2}_balanced.")


if __name__ == "__main__":
    main()
