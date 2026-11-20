"""
Extract the text-only decoder from the multimodal Gemma-3-4B checkpoints.

google/gemma-3-4b-pt, google/gemma-3-4b-it and Cognitive-Lab/NetraEmbed are all
Gemma3ForConditionalGeneration checkpoints (vision tower + projector + text decoder).
Our pipeline is text-only, so we save each text decoder as a plain Gemma3ForCausalLM
checkpoint (model_type gemma3_text). The standard Auto classes then load it as
Gemma3TextModel / Gemma3TextForSequenceClassification / Gemma3ForCausalLM, exactly like
gemma-3-270m. Without this, LoRA "all-linear" would also attach to the vision tower.

The decoder weights are copied unchanged (only the key prefix changes; lm_head stays tied
to the input embeddings). After saving, the copy is reloaded with the Auto classes used in
training and must reproduce the multimodal model's text-only outputs exactly.

Usage (run on a GPU node; the login node is too slow for this):
    python scripts/family_ext/extract_gemma3_text.py [--out_root DIR] [--skip_existing]
"""

import argparse
import os

import torch
from safetensors.torch import save_file
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    GenerationConfig,
)

SOURCES = ["google/gemma-3-4b-pt", "google/gemma-3-4b-it", "Cognitive-Lab/NetraEmbed"]
DEFAULT_OUT_ROOT = "/scratch-shared/61600/models/gemma3-text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_TEXTS = [
    "Represent the entity for entity matching.\nCOL title VAL sony bravia 55 inch 4k tv COL price VAL 899.99",
    "Determine whether the following two entities refer to the same real-world entity.",
]


def _check_equal(name, ref, got):
    max_diff = (ref.float() - got.float()).abs().max().item()
    print(f"  {name}: max |diff| = {max_diff:.3e}", flush=True)
    if max_diff != 0.0:
        raise RuntimeError(f"{name} differs from the multimodal model (max diff {max_diff})")


@torch.no_grad()
def extract(src, out_dir):
    print(f"== {src} -> {out_dir}  [{DEVICE}]", flush=True)
    mm = Gemma3ForConditionalGeneration.from_pretrained(
        src, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(src)

    # Save the decoder under Gemma3ForCausalLM key names (model.*); lm_head is tied.
    text_config = mm.config.text_config
    if not getattr(text_config, "tie_word_embeddings", True):
        raise RuntimeError("expected tied input/output embeddings")
    text_config.architectures = ["Gemma3ForCausalLM"]
    text_config.dtype = torch.bfloat16
    state = {
        f"model.{k}": v.detach().to("cpu").contiguous()
        for k, v in mm.model.language_model.state_dict().items()
    }
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    text_config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    try:
        GenerationConfig.from_pretrained(src).save_pretrained(out_dir)
    except OSError:
        pass
    n_text = sum(v.numel() for v in state.values())
    n_mm = sum(p.numel() for p in mm.parameters())
    print(f"  params: multimodal {n_mm / 1e9:.3f}B -> text-only {n_text / 1e9:.3f}B", flush=True)
    del state

    enc = tokenizer(PROBE_TEXTS, return_tensors="pt", padding=True).to(DEVICE)
    ref_logits = mm(**enc).logits
    ref_hidden = mm.model.language_model(**enc).last_hidden_state
    del mm
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Reload exactly as the training code does.
    causal = AutoModelForCausalLM.from_pretrained(
        out_dir, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE).eval()
    print(f"  reloaded as {type(causal).__name__}", flush=True)
    assert causal.lm_head.weight.data_ptr() == causal.model.embed_tokens.weight.data_ptr()
    _check_equal("logits (reloaded causal LM)", ref_logits, causal(**enc).logits)
    del causal

    encoder = AutoModel.from_pretrained(
        out_dir, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE).eval()
    print(f"  reloaded as {type(encoder).__name__}", flush=True)
    _check_equal("last_hidden_state (reloaded encoder)", ref_hidden, encoder(**enc).last_hidden_state)
    del encoder

    reloaded_tok = AutoTokenizer.from_pretrained(out_dir)
    assert reloaded_tok(PROBE_TEXTS)["input_ids"] == tokenizer(PROBE_TEXTS)["input_ids"]
    assert reloaded_tok.chat_template == tokenizer.chat_template
    with open(os.path.join(out_dir, "EXTRACTED_FROM"), "w") as f:
        f.write(src + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip sources whose verified output (EXTRACTED_FROM marker) already exists")
    args = p.parse_args()
    for src in SOURCES:
        out_dir = os.path.join(args.out_root, os.path.basename(src))
        if args.skip_existing and os.path.exists(os.path.join(out_dir, "EXTRACTED_FROM")):
            print(f"== {src}: already extracted at {out_dir}, skipping", flush=True)
            continue
        extract(src, out_dir)
    print("All extractions verified.", flush=True)


if __name__ == "__main__":
    main()
