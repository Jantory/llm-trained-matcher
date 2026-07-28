"""
OpenAI generative baseline for entity matching.

Prompt format matches the generative fine-tuning paradigm exactly:
  user: "Determine whether the following two entities refer to the same
         real-world entity. Answer yes or no.
         Entity A: {text_a}
         Entity B: {text_b}"
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

from utils.constants import DATASET_ALIASES, DATASET_PATHS
from utils.metrics import compute_binary_metrics

_INSTRUCTION = (
    "Determine whether the following two entities refer to the same real-world entity."
    " Answer yes or no."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_ALIASES))
    p.add_argument("--model", required=True, help="OpenAI model name, e.g. gpt-4o-mini")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--max_tokens_per_entity", type=int, default=256,
                   help="Token budget per entity (uses tiktoken; falls back to word count)")
    p.add_argument("--max_retries", type=int, default=5)
    return p.parse_args()


def _load_test_data(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    df_a = pd.read_csv(os.path.join(data_dir, "tableA_serialized.csv"))
    df_b = pd.read_csv(os.path.join(data_dir, "tableB_serialized.csv"))
    table_a = dict(zip(df_a["id"], df_a["text"]))
    table_b = dict(zip(df_b["id"], df_b["text"]))
    df["text_a"] = df["ltable_id"].map(table_a)
    df["text_b"] = df["rtable_id"].map(table_b)
    return df


def _truncate(text: str, max_tokens: int) -> str:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        ids = enc.encode(text)
        return enc.decode(ids[:max_tokens]) if len(ids) > max_tokens else text
    except Exception:
        words = text.split()
        return " ".join(words[:max_tokens])


def _query(client, model: str, user_content: str, max_retries: int) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                max_completion_tokens=5,
                temperature=0,
            )
            return resp.choices[0].message.content.strip().lower()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt + 1}): {exc}. Retrying in {wait}s…")
            time.sleep(wait)


def _save_results(results_dir: str, predictions: list, metadata: dict):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump({"metadata": metadata, "predictions": predictions}, f, indent=2)


def _save_log(results_dir: str, log_dict: dict):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "training_log.json"), "w") as f:
        json.dump(log_dict, f, indent=2, default=str)


def main():
    args = parse_args()

    data_dir = DATASET_PATHS[args.dataset]
    if not os.path.isdir(data_dir):
        sys.exit(f"Dataset directory not found: {data_dir}")

    if args.results_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        model_slug = args.model.replace("/", "-").replace(":", "-")
        args.results_dir = os.path.join(
            project_root, "results", "baselines", "openai", args.dataset, model_slug,
        )

    print(f"Dataset  : {args.dataset} → {data_dir}")
    print(f"Model    : {args.model}")
    print(f"Results  : {args.results_dir}")

    df = _load_test_data(data_dir)
    print(f"Test set : {len(df)} examples")

    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env

    preds, labels, answers = [], [], []
    n_invalid = 0
    t0 = time.time()

    for idx, row in enumerate(df.itertuples(index=False), 1):
        text_a = _truncate(str(row.text_a), args.max_tokens_per_entity)
        text_b = _truncate(str(row.text_b), args.max_tokens_per_entity)
        user_content = f"{_INSTRUCTION}\nEntity A: {text_a}\nEntity B: {text_b}"

        answer = _query(client, args.model, user_content, args.max_retries)

        if answer.startswith("yes"):
            pred = 1
        elif answer.startswith("no"):
            pred = 0
        else:
            n_invalid += 1
            pred = 0

        preds.append(pred)
        labels.append(int(row.label))
        answers.append(answer)

        if idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(df)}]  elapsed={elapsed:.0f}s  invalid={n_invalid}")

    runtime = time.time() - t0
    metrics = compute_binary_metrics(preds, labels)
    invalid_rate = n_invalid / len(labels) if labels else 0.0

    print(f"\nTest F1        : {metrics['f1']:.4f}")
    print(f"Test Precision : {metrics['precision']:.4f}")
    print(f"Test Recall    : {metrics['recall']:.4f}")
    print(f"Invalid rate   : {invalid_rate:.4f}")
    print(f"Runtime        : {runtime:.1f}s")

    log_data = {
        "paradigm": "openai_baseline",
        "model": args.model,
        "dataset": args.dataset,
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "invalid_generation_rate": invalid_rate,
        "runtime": runtime,
        "n_examples": len(labels),
    }
    _save_log(args.results_dir, log_data)
    _save_results(
        args.results_dir,
        [{"generated": a, "pred": p, "label": l} for a, p, l in zip(answers, preds, labels)],
        metadata={"paradigm": "openai_baseline", "model": args.model},
    )
    print(f"Results  → {args.results_dir}")


if __name__ == "__main__":
    main()
