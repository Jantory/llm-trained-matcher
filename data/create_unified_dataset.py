"""
Create a unified sampled dataset by drawing only from each source
dataset's original train partition and concatenating the results.

Samples per source dataset:
  - 500 rows for the unified train.csv
  - 200 rows for the unified valid.csv
  - 100 rows for the unified test.csv

All three output partitions are sampled from the source dataset's train.csv.
The sampled rows are disjoint within each source dataset.

IDs across datasets are remapped with per-dataset offsets to avoid collisions.

Output:
  - tableA_serialized.csv
  - tableB_serialized.csv
  - train.csv
  - valid.csv
  - test.csv
"""

import argparse
import os

import pandas as pd


DATASETS = [
    "abt_buy",
    "amazon_google",
    "dblp_scholar",
    "rel-text",
    "semi-heter",
    "semi-rel",
    "walmart_amazon",
    "wdc20",
    "wdc80",
]

SPLIT_SIZES = {
    "train": 500,
    "valid": 200,
    "test": 100,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a unified sampled dataset using only each source "
            "dataset's train partition."
        )
    )
    parser.add_argument(
        "--src",
        default=os.path.join(os.path.dirname(__file__), "cleaned"),
        help="Path to the source cleaned data directory.",
    )
    parser.add_argument(
        "--dst",
        default=os.path.join(
            os.path.dirname(__file__),
            "cleaned",
            "unified",
        ),
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Which datasets to include (default: all).",
    )
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    split_parts: dict[str, list[pd.DataFrame]] = {
        split: [] for split in SPLIT_SIZES
    }
    table_a_parts: list[pd.DataFrame] = []
    table_b_parts: list[pd.DataFrame] = []

    left_offset = 0
    right_offset = 0

    total_sample_size = sum(SPLIT_SIZES.values())

    for ds in args.datasets:
        src_dir = os.path.join(args.src, ds)
        print(f"Processing {ds} ...")

        table_a = pd.read_csv(
            os.path.join(src_dir, "tableA_serialized.csv")
        )
        table_b = pd.read_csv(
            os.path.join(src_dir, "tableB_serialized.csv")
        )

        # All output splits are constructed from the original train.csv.
        source_train = pd.read_csv(
            os.path.join(src_dir, "train.csv")
        )

        if len(source_train) < total_sample_size:
            raise ValueError(
                f"{ds}/train.csv has only {len(source_train)} rows, "
                f"but {total_sample_size} disjoint rows are required."
            )

        # Shuffle once, then slice the shuffled rows into disjoint
        # train, valid, and test partitions.
        sampled_rows = source_train.sample(
            n=total_sample_size,
            replace=False,
            random_state=args.seed,
        ).reset_index(drop=True)

        ref_left: set = set()
        ref_right: set = set()

        start = 0

        for split, size in SPLIT_SIZES.items():
            end = start + size
            sampled_split = sampled_rows.iloc[start:end].copy()
            start = end

            ref_left.update(sampled_split["ltable_id"].tolist())
            ref_right.update(sampled_split["rtable_id"].tolist())

            # Remap entity IDs to avoid collisions between datasets.
            sampled_split["ltable_id"] = (
                sampled_split["ltable_id"] + left_offset
            )
            sampled_split["rtable_id"] = (
                sampled_split["rtable_id"] + right_offset
            )

            split_parts[split].append(sampled_split)

        # Keep only entities referenced by the sampled pairs.
        filtered_a = table_a[
            table_a["id"].isin(ref_left)
        ].copy()
        filtered_b = table_b[
            table_b["id"].isin(ref_right)
        ].copy()

        filtered_a["id"] = filtered_a["id"] + left_offset
        filtered_b["id"] = filtered_b["id"] + right_offset

        table_a_parts.append(filtered_a)
        table_b_parts.append(filtered_b)

        # Advance offsets using the complete source entity tables,
        # ensuring that IDs from subsequent datasets cannot collide.
        if not table_a.empty:
            left_offset += int(table_a["id"].max()) + 1

        if not table_b.empty:
            right_offset += int(table_b["id"].max()) + 1

    # Write unified pair partitions.
    for split, parts in split_parts.items():
        if not parts:
            raise ValueError(
                f"No data was collected for output split '{split}'."
            )

        unified_split = pd.concat(parts, ignore_index=True)
        unified_split.to_csv(
            os.path.join(args.dst, f"{split}.csv"),
            index=False,
        )

    # Write unified entity tables.
    if not table_a_parts or not table_b_parts:
        raise ValueError("No datasets were processed.")

    pd.concat(
        table_a_parts,
        ignore_index=True,
    ).to_csv(
        os.path.join(args.dst, "tableA_serialized.csv"),
        index=False,
    )

    pd.concat(
        table_b_parts,
        ignore_index=True,
    ).to_csv(
        os.path.join(args.dst, "tableB_serialized.csv"),
        index=False,
    )

    print(f"Done. Output written to {args.dst}")


if __name__ == "__main__":
    main()