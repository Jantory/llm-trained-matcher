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
}

DATASET_PATHS = {
    alias: os.path.normpath(os.path.join(DATA_ROOT, folder))
    for alias, folder in DATASET_ALIASES.items()
}
