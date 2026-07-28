import os
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value

_DEFAULT_INSTRUCTIONS = {
    "bi_encoder": "Represent the entity for entity matching.",
    "cross_encoder": "Determine whether the following two entities refer to the same real-world entity.",
    "generative": "Determine whether the following two entities refer to the same real-world entity. Answer yes or no.",
}

_SPLIT_FILES = {
    "train": "train.csv",
    "validation": "valid.csv",
    "test": "test.csv",
}


class EntityMatchingDatasetBuilder:
    def __init__(self, paradigm, input_folder, tokenizer, max_len, instruction=None):
        self.paradigm = paradigm
        self.input_folder = input_folder
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.instruction = instruction if instruction is not None else _DEFAULT_INSTRUCTIONS[paradigm]
        self.table_a = {}
        self.table_b = {}
        # Detect whether the tokenizer appends a sentinel at the end automatically.
        # Embedding-variant tokenizers do; plain causal-LM tokenizers don't.
        _probe_with = tokenizer("x", add_special_tokens=True)["input_ids"]
        _probe_without = tokenizer("x", add_special_tokens=False)["input_ids"]
        self._sentinel = None if _probe_with[-1] != _probe_without[-1] else \
            tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def build(self) -> DatasetDict:
        self._load_tables()
        return DatasetDict({
            split: self._build_split(split)
            for split in ("train", "validation", "test")
        })

    def _load_tables(self):
        df_a = pd.read_csv(os.path.join(self.input_folder, "tableA_serialized.csv"))
        df_b = pd.read_csv(os.path.join(self.input_folder, "tableB_serialized.csv"))
        self.table_a = dict(zip(df_a["id"], df_a["text"]))
        self.table_b = dict(zip(df_b["id"], df_b["text"]))

    def _load_split(self, split_name) -> pd.DataFrame:
        path = os.path.join(self.input_folder, _SPLIT_FILES[split_name])
        return pd.read_csv(path)

    def _join_split(self, split_df) -> pd.DataFrame:
        df = split_df.copy()
        df["text_a"] = df["ltable_id"].map(self.table_a)
        df["text_b"] = df["rtable_id"].map(self.table_b)
        missing_a = df["ltable_id"][df["text_a"].isna()].tolist()
        missing_b = df["rtable_id"][df["text_b"].isna()].tolist()
        if missing_a or missing_b:
            raise KeyError(
                f"IDs not found in tables — tableA: {missing_a[:5]}, tableB: {missing_b[:5]}"
            )
        return df

    def _truncate_text(self, text, budget) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.tokenizer.decode(ids[:budget])

    def _build_split(self, split_name) -> Dataset:
        df = self._join_split(self._load_split(split_name))

        if self.paradigm == "bi_encoder":
            budget_a = budget_b = self.max_len
        else:
            budget_a = budget_b = self.max_len // 2

        formatter = {
            "bi_encoder": self._format_bi_encoder,
            "cross_encoder": self._format_cross_encoder,
            "generative": self._format_generative,
        }[self.paradigm]

        records = []
        for _, row in df.iterrows():
            text_a = self._truncate_text(str(row["text_a"]), budget_a)
            text_b = self._truncate_text(str(row["text_b"]), budget_b)
            records.append(formatter(text_a, text_b, int(row["label"])))

        if self.paradigm == "generative":
            # explicit schema required because input_ids and eval_input_ids differ in length per row,
            # causing HF to fail type inference without an upfront column definition
            features = Features({
                "input_ids": Sequence(Value("int64")),
                "attention_mask": Sequence(Value("int64")),
                "labels": Sequence(Value("int64")),
                "eval_input_ids": Sequence(Value("int64")),
                "eval_attention_mask": Sequence(Value("int64")),
                "gold_label": Value("int64"),
            })
            return Dataset.from_list(records, features=features)
        return Dataset.from_list(records)

    def _append_sentinel(self, enc: dict) -> dict:
        if self._sentinel is None:
            return enc
        return {
            "input_ids": enc["input_ids"] + [self._sentinel],
            "attention_mask": enc["attention_mask"] + [1],
        }

    def _format_bi_encoder(self, text_a, text_b, label) -> dict:
        enc_a = self._append_sentinel(self.tokenizer(self.instruction + "\n" + text_a, add_special_tokens=True))
        enc_b = self._append_sentinel(self.tokenizer(self.instruction + "\n" + text_b, add_special_tokens=True))
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "labels": label,
        }

    def _format_cross_encoder(self, text_a, text_b, label) -> dict:
        text = f"{self.instruction}\nEntity A: {text_a}\nEntity B: {text_b}"
        enc = self._append_sentinel(self.tokenizer(text, add_special_tokens=True))
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
        }

    def _format_generative(self, text_a, text_b, label) -> dict:
        user_content = f"{self.instruction}\nEntity A: {text_a}\nEntity B: {text_b}"
        assistant_content = "yes" if label == 1 else "no"

        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content},
             {"role": "assistant", "content": assistant_content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "eval_input_ids": prompt_ids,
            "eval_attention_mask": [1] * len(prompt_ids),
            "gold_label": label,
        }


def build_dataset(
    paradigm: str,
    input_folder: str,
    tokenizer,
    max_len: int,
    instruction: str | None = None,
) -> DatasetDict:
    return EntityMatchingDatasetBuilder(
        paradigm, input_folder, tokenizer, max_len, instruction
    ).build()
