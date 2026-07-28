import torch
from dataclasses import dataclass


@dataclass
class BiEncoderCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        def _pad(ids_key: str, mask_key: str):
            ids_list = [x[ids_key] for x in batch]
            mask_list = [x[mask_key] for x in batch]
            max_len = max(len(x) for x in ids_list)
            padded_ids, padded_mask = [], []
            for ids, mask in zip(ids_list, mask_list):
                pad = max_len - len(ids)
                padded_ids.append(list(ids) + [self.pad_token_id] * pad)
                padded_mask.append(list(mask) + [0] * pad)
            return (
                torch.tensor(padded_ids, dtype=torch.long),
                torch.tensor(padded_mask, dtype=torch.long),
            )

        ids_a, mask_a = _pad("input_ids_a", "attention_mask_a")
        ids_b, mask_b = _pad("input_ids_b", "attention_mask_b")
        return {
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }


@dataclass
class GenerativeTrainCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        ids_list = [x["input_ids"] for x in batch]
        mask_list = [x["attention_mask"] for x in batch]
        labs_list = [x["labels"] for x in batch]

        max_len = max(len(x) for x in ids_list)
        padded_ids, padded_mask, padded_labs = [], [], []
        for ids, mask, labs in zip(ids_list, mask_list, labs_list):
            pad = max_len - len(ids)
            padded_ids.append(list(ids) + [self.pad_token_id] * pad)
            padded_mask.append(list(mask) + [0] * pad)
            padded_labs.append(list(labs) + [-100] * pad)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labs, dtype=torch.long),
        }
