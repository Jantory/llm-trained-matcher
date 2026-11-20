import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")

# Short alias → folder name under DATA_ROOT
DATASET_ALIASES = {
    "ab":  "abt_buy",
    "ag":  "amazon_google",
    "ds":  "dblp_scholar",
    "rt":  "rel-text",
    "sh":  "semi-heter",
    "sr":  "semi-rel",
    "wa":  "walmart_amazon",
    "w20": "wdc20",
    "w80": "wdc80",
    
    "un": "unified",    # for learning rate sweep
    "dss": "dblp_scholar_small",  # for size ablation
    "abs": "abt_buy_small",  # for size ablation
    "w8s": "wdc80_small",  # for size ablation
    "srn": "semi-rel_noblob",  # free-text-blob ablation: drop Description (A: information)
    "src": "semi-rel_core",    # free-text-blob ablation: also drop Cast (relational core only)
    "srb": "semi-rel_rebalanced",  # title-collision rebalancing ablation (same pairs, re-split)
    "shb": "semi-heter_rebalanced",  # dialect-combination rebalancing ablation (same pairs, re-split)
    # Independent replicates of sh / sr built from their unlabeled entity tables
    # (data/create_{sh,sr}_replicates.py): same split structure, new pairs; *b = balanced control.
    "sh1": "semi-heter_rep1",
    "sh2": "semi-heter_rep2",
    "sh1b": "semi-heter_rep1_balanced",
    "sh2b": "semi-heter_rep2_balanced",
    "sr1": "semi-rel_rep1",
    "sr2": "semi-rel_rep2",
    "sr1b": "semi-rel_rep1_balanced",
    "sr2b": "semi-rel_rep2_balanced",
}

DATASET_PATHS = {
    alias: os.path.normpath(os.path.join(DATA_ROOT, folder))
    for alias, folder in DATASET_ALIASES.items()
}
