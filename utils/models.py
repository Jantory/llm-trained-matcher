import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, seq_len]


def contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    distances = 1.0 - F.cosine_similarity(z_a, z_b)
    pos = labels.float() * distances.pow(2)
    neg = (1.0 - labels.float()) * F.relu(margin - distances).pow(2)
    return (pos + neg).mean()


class BiEncoderModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        loss_name: str = "contrastive",
        margin: float = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path, attn_implementation="flash_attention_3"
        )
        self.loss_name = loss_name
        self.margin = margin

    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return last_token_pool(out.last_hidden_state, attention_mask)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        z_a = self._pool(input_ids_a, attention_mask_a)
        z_b = self._pool(input_ids_b, attention_mask_b)
        cosine_scores = F.cosine_similarity(z_a, z_b)
        loss = None
        if labels is not None:
            loss = contrastive_loss(z_a, z_b, labels, self.margin)
        return {"loss": loss, "logits": cosine_scores}
