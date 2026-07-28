# Dataset Descriptions

This document characterizes the nine entity matching benchmarks used in our experiments, covering their origins, schema types, domains, split statistics, and key properties relevant to analysis.

---

## Overview

We use nine datasets drawn from three established benchmarks:

| ID   | Full Name       | Source Benchmark | Domain            | Schema Type         |
|------|-----------------|------------------|-------------------|---------------------|
| `ab` | Abt-Buy         | DeepMatcher      | General products  | Structured / Textual |
| `ag` | Amazon-Google   | DeepMatcher      | Software products | Structured           |
| `ds` | DBLP-Scholar    | DeepMatcher      | Academic papers   | Structured           |
| `wa` | Walmart-Amazon  | DeepMatcher      | Electronics       | Structured           |
| `rt` | Rel-Text        | MaChAmp (GEM)    | Academic papers   | Cross-modal (text ↔ structured) |
| `sh` | Semi-Heter      | MaChAmp (GEM)    | Books             | Semi-structured / Heterogeneous |
| `sr` | Semi-Rel        | MaChAmp (GEM)    | Movies            | Semi-structured ↔ Structured    |
| `w20`| WDC (20% CC)   | WDC Products     | Web products      | Structured           |
| `w80`| WDC (80% CC)   | WDC Products     | Web products      | Structured           |

---

## Dataset Details

### Structured Datasets (DeepMatcher Benchmark)

These four datasets originate from the DeepMatcher benchmark (Mudgal et al., SIGMOD 2018). Both tables share an aligned schema, and records are serialized as key-value attribute sequences.

---

#### `ab` — Abt-Buy

- **Domain**: General retail products
- **Source**: Abt.com (left) vs. Buy.com (right)
- **Attributes (both tables)**: `name`, `price`, `description` — **3 attributes each**
- **Characteristic**: Long, free-text product descriptions; no strong structural alignment between attribute values. High textual variability makes surface-level matching unreliable.
- **Avg. serialized text length**: Left ~333 chars, Right ~126 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 5,743       | 616       | 10.7%         |
| Val   | 1,916       | 206       | 10.8%         |
| Test  | 1,916       | 206       | 10.8%         |
| **All** | **9,575** | **1,028** | **10.7%**     |

| Table  | Size  |
|--------|-------|
| Left   | 1,081 |
| Right  | 1,092 |

---

#### `ag` — Amazon-Google

- **Domain**: Software products
- **Source**: Amazon (left) vs. Google Products (right)
- **Attributes (both tables)**: `title`, `manufacturer`, `price` — **3 attributes each**
- **Characteristic**: Short, sparse records with few attributes. High class imbalance. Title is often the primary discriminating signal.
- **Avg. serialized text length**: Left ~87 chars, Right ~102 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 6,874       | 699       | 10.2%         |
| Val   | 2,293       | 234       | 10.2%         |
| Test  | 2,293       | 234       | 10.2%         |
| **All** | **11,460** | **1,167** | **10.2%**    |

| Table  | Size  |
|--------|-------|
| Left   | 1,363 |
| Right  | 3,226 |

---

#### `ds` — DBLP-Scholar

- **Domain**: Academic publication citations
- **Source**: DBLP (left) vs. Google Scholar (right)
- **Attributes (both tables)**: `title`, `authors`, `venue`, `year` — **4 attributes each**
- **Characteristic**: Moderate-length records; the challenge lies in author name abbreviation and venue abbreviation differences between the two sources. Large right table (64K records) relative to left (2.6K).
- **Avg. serialized text length**: Left ~138 chars, Right ~154 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 17,223      | 3,207     | 18.6%         |
| Val   | 5,742       | 1,070     | 18.6%         |
| Test  | 5,742       | 1,070     | 18.6%         |
| **All** | **28,707** | **5,347** | **18.6%**    |

| Table  | Size   |
|--------|--------|
| Left   | 2,616  |
| Right  | 64,263 |

---

#### `wa` — Walmart-Amazon

- **Domain**: Consumer electronics
- **Source**: Walmart (left) vs. Amazon (right)
- **Attributes (both tables)**: `title`, `category`, `brand`, `modelno`, `price` — **5 attributes each**
- **Characteristic**: Highly imbalanced (lowest positive rate). Model numbers and brand names are key matching signals, but they are inconsistently formatted across sources. High proportion of hard negatives.
- **Avg. serialized text length**: Left ~142 chars, Right ~155 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 6,144       | 576       | 9.4%          |
| Val   | 2,049       | 193       | 9.4%          |
| Test  | 2,049       | 193       | 9.4%          |
| **All** | **10,242** | **962**  | **9.4%**      |

| Table  | Size   |
|--------|--------|
| Left   | 2,554  |
| Right  | 22,074 |

---

### Cross-Modal and Heterogeneous Datasets (MaChAmp / GEM Benchmark)

These three datasets come from the MaChAmp Generalized Entity Matching benchmark (Wang et al., CIKM 2021). They go beyond the homogeneous structured setting: records in the two tables may have different schemas, data formats, or modalities.

---

#### `rt` — Rel-Text

- **Domain**: Academic publications
- **Source**: Left table — paper abstracts (unstructured text); Right table — structured DBLP-style citations (`title`, `authors`, `venue`, `year`)
- **Attributes**: Left: **1** (`abstract`); Right: **4** (`title`, `authors`, `venue`, `year`) in our cleaned version (original MaChAmp right table has 6 attributes from DBLP-ACM)
- **Characteristic**: Cross-modal matching: one side is free-form text, the other is structured. Very long left-side records (abstracts). The model must understand that an abstract corresponds to a structured citation entry — requiring semantic understanding beyond keyword overlap.
- **Avg. serialized text length**: Left ~876 chars (abstracts), Right ~181 chars (structured citation)

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 7,417       | 1,332     | 18.0%         |
| Val   | 2,473       | 444       | 18.0%         |
| Test  | 2,473       | 444       | 18.0%         |
| **All** | **12,363** | **2,220** | **18.0%**    |

| Table  | Size  |
|--------|-------|
| Left   | 2,616 |
| Right  | 2,294 |

---

#### `sh` — Semi-Heter

- **Domain**: Books
- **Source**: Multiple book databases with heterogeneous schemas
- **Attributes**: Both tables are semi-structured with a **union schema** from multiple book databases (MaChAmp official avg: **12.28** attrs/record left, **12.03** attrs/record right). Representative left attributes: `title`, `pubyear`, `publisher`, `authors`; representative right attributes: `title`, `pages`, `authors`, `publication_info`, `series`.
- **Characteristic**: Both tables are semi-structured (JSON-like, many sparse/null attributes). Schemas differ significantly across sources — attribute names, nesting, and coverage vary widely. Small dataset with relatively balanced classes. The highest structural mismatch among all datasets.
- **Avg. serialized text length**: Left ~417 chars, Right ~257 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 1,240       | 474       | 38.2%         |
| Val   | 414         | 157       | 37.9%         |
| Test  | 414         | 159       | 38.4%         |
| **All** | **2,068** | **790**   | **38.2%**     |

| Table  | Size   |
|--------|--------|
| Left   | 22,133 |
| Right  | 23,264 |

---

#### `sr` — Semi-Rel

- **Domain**: Movies
- **Source**: Left table — structured movie records; Right table — semi-structured movie data (JSON) with a union schema from multiple sources
- **Attributes**: Left: **6** attrs per record (`title`, `director`, `actors`, `year`, `rating`, `information`); Right: semi-structured with union schema, MaChAmp official avg **13.81** attrs/record. Representative right attributes: `Name`, `YearRange`, `ReleaseDate`, `Director`, `Creator`, `Duration`, `Genre`, `Description`.
- **Characteristic**: Cross-format matching between a structured relational table and a semi-structured source. Attribute names differ; date/year representations vary. The most balanced class distribution across all datasets. Small total size.
- **Avg. serialized text length**: Left ~410 chars, Right ~524 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 1,309       | 545       | 41.6%         |
| Val   | 437         | 181       | 41.4%         |
| Test  | 437         | 183       | 41.9%         |
| **All** | **2,183** | **909**   | **41.6%**     |

| Table  | Size   |
|--------|--------|
| Left   | 29,179 |
| Right  | 32,823 |

---

### Web-Scale Product Datasets (WDC Products Benchmark)

These two datasets are drawn from the WDC Products benchmark (Peeters et al., 2023), based on product data extracted from 3,259 e-shops using schema.org annotations. The benchmark introduces the concept of **corner cases (CC)**: hard negative pairs involving products with overlapping but non-identical descriptions. The `20` and `80` variants differ in the proportion of corner-case pairs **across the entire dataset** (train, val, and test), resulting in almost entirely disjoint pair sets between the two variants.

---

#### `w20` — WDC Products (20% Corner Cases)

- **Domain**: General web products (multi-category)
- **Attributes (both tables)**: `title`, `brand`, `price`, `currency`, `description` — **5 attributes each**
- **Characteristic**: Lower difficulty variant — 20% of pairs (across all splits) are corner cases. Provides a relatively easier matching setting within the WDC benchmark.
- **Avg. serialized text length**: Left ~507 chars, Right ~504 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 6,000       | 1,500     | 25.0%         |
| Val   | 3,500       | 500       | 14.3%         |
| Test  | 4,500       | 500       | 11.1%         |
| **All** | **14,000** | **2,500** | **17.9%**    |

| Table  | Size  |
|--------|-------|
| Left   | 3,453 |
| Right  | 3,460 |

---

#### `w80` — WDC Products (80% Corner Cases)

- **Domain**: General web products (multi-category)
- **Attributes (both tables)**: `title`, `brand`, `price`, `currency`, `description` — **5 attributes each**
- **Characteristic**: Higher difficulty variant — 80% of pairs (across all splits) are corner cases. Trains and evaluates on a much harder distribution of near-duplicate non-matching products. The pair sets are almost entirely disjoint from `w20`.
- **Avg. serialized text length**: Left ~522 chars, Right ~522 chars

| Split | Total pairs | Positives | Positive rate |
|-------|-------------|-----------|---------------|
| Train | 6,000       | 1,500     | 25.0%         |
| Val   | 3,500       | 500       | 14.3%         |
| Test  | 4,500       | 500       | 11.1%         |
| **All** | **14,000** | **2,500** | **17.9%**    |

| Table  | Size  |
|--------|-------|
| Left   | 3,457 |
| Right  | 3,470 |

---

## Summary Statistics

| ID   | Domain         | Schema type                    | #Attrs (L / R)    | Train pairs | Test pairs | Pos. rate | Avg. text len (L / R) |
|------|----------------|--------------------------------|-------------------|-------------|------------|-----------|------------------------|
| `ab` | Products       | Structured / Textual           | 3 / 3             | 5,743       | 1,916      | 10.7%     | 333 / 126              |
| `ag` | Software       | Structured                     | 3 / 3             | 6,874       | 2,293      | 10.2%     | 87 / 102               |
| `ds` | Academic       | Structured                     | 4 / 4             | 17,223      | 5,742      | 18.6%     | 138 / 154              |
| `wa` | Electronics    | Structured                     | 5 / 5             | 6,144       | 2,049      | 9.4%      | 142 / 155              |
| `rt` | Academic       | Cross-modal (text ↔ structured)| 1 / 4†            | 7,417       | 2,473      | 18.0%     | 876 / 181              |
| `sh` | Books          | Semi-structured, heterogeneous | 12.28 / 12.03 avg | 1,240       | 414        | 38.2%     | 417 / 257              |
| `sr` | Movies         | Semi-structured ↔ Structured   | 8.0 / 13.81 avg   | 1,309       | 437        | 41.6%     | 410 / 524              |
| `w20`| Web products   | Structured                     | 5 / 5             | 6,000       | 4,500      | 17.9%     | 507 / 504              |
| `w80`| Web products   | Structured (high CC)           | 5 / 5             | 6,000       | 4,500      | 17.9%     | 522 / 522              |

**Notes:**
- Positive rate is computed over all pairs (train + val + test) for `ab`/`ag`/`ds`/`wa`/`rt`/`sh`/`sr`; for `w20`/`w80` it reflects the combined set (train has 25%, test has 11.1%).
- Avg. text length is measured on serialized left-table records (`tableA_serialized.csv`).
- † `rt` right table has 4 attributes in our cleaned version; original MaChAmp source (DBLP-ACM) has 6.
- `w20` and `w80` have nearly disjoint pair sets (only 62/6000 training pairs overlap); the 20%/80% corner-case ratio applies across all splits, so both training distribution and evaluation difficulty differ between the two variants.
- `sh` and `sr` have large entity tables but few labeled pairs, reflecting targeted sampling from a large candidate space.
- `ds` has an extremely asymmetric table size (2.6K vs. 64K), reflecting the Scholar crawl covering many more publications than DBLP.

---

## References

- Mudgal et al. *Deep Learning for Entity Matching: A Design Space Exploration.* SIGMOD 2018.
- Wang et al. *Machamp: A Generalized Entity Matching Benchmark.* CIKM 2021.
- Peeters et al. *WDC Products: A Multi-Dimensional Entity Matching Benchmark.* 2023.
