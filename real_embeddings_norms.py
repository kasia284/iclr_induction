"""Plot the distributions of embedding (W_E) and unembedding (W_U) vector
norms across Llama-3.1-8B's full vocabulary.

Reads model.embed_tokens.weight and lm_head.weight directly from the cached
safetensors shards (same trick as real_embeddings_nearest_neighbors.py),
bypassing TransformerLens/HookedTransformer -- this only needs the two
[d_vocab, d_model] matrices (~2GB combined in fp32), so it runs fine on
CPU-only / memory-constrained machines without loading all 32 layers.
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from utils import MODEL_NAME, setup_plotting, save_figure

PLOTS_DIR = "results/embedding-norms"
N_BINS = 100

# Byte-fallback tokens (e.g. "<0x0A>") and other non-printable / reserved
# tokens pollute a norm distribution over "real" vocabulary tokens, so
# they're excluded up front (mirrors real_embeddings_nearest_neighbors.py).
_BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def load_embed_and_unembed():
    """Return (W_E, W_U), both [d_vocab, d_model], read straight out of the
    model's local safetensors shards."""
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_root, f"models--{MODEL_NAME.replace('/', '--')}")
    snapshot_dirs = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for {MODEL_NAME} under {model_dir}")
    snapshot_dir = snapshot_dirs[0]

    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    def _load(key):
        shard_path = os.path.join(snapshot_dir, weight_map[key])
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            return sf.get_tensor(key).float()

    W_E = _load("model.embed_tokens.weight")
    W_U = _load("lm_head.weight")
    return W_E, W_U


def valid_token_mask(tokenizer, d_vocab):
    """Mask out special tokens, byte-fallback tokens, and reserved slots."""
    special_ids = set(tokenizer.all_special_ids)
    mask = torch.ones(d_vocab, dtype=torch.bool)
    for tid in range(d_vocab):
        if tid in special_ids:
            mask[tid] = False
            continue
        s = tokenizer.decode([tid])
        if not s.strip() or _BYTE_FALLBACK_RE.match(s.strip()) or "reserved_special_token" in s:
            mask[tid] = False
    return mask


def plot_norm_distributions(embed_norms, unembed_norms, plots_dir, filename):
    fig, (ax_e, ax_u) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, norms, color, label in [
        (ax_e, embed_norms, "#377eb8", "Embedding"),
        (ax_u, unembed_norms, "#e41a1c", "Unembedding"),
    ]:
        ax.hist(norms, bins=N_BINS, color=color, alpha=0.85)
        ax.axvline(norms.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean = {norms.mean():.2f}")
        ax.set_title(f"{label} norms ({'W_E' if label == 'Embedding' else 'W_U'})")
        ax.set_xlabel(r"$\|w_i\|_2$")
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle(f"Vocabulary embedding/unembedding norm distributions ({MODEL_NAME})")
    plt.tight_layout()
    save_figure(fig, plots_dir, filename)
    print(f"Saved {os.path.join(plots_dir, filename)}")


def main():
    setup_plotting()
    plt.rcParams["text.usetex"] = False  # Bypassing missing LaTeX binaries

    print(f"Loading embedding/unembedding matrices for {MODEL_NAME}...")
    W_E, W_U = load_embed_and_unembed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    mask = valid_token_mask(tokenizer, W_E.shape[0])
    print(f"Keeping {mask.sum().item()} / {mask.shape[0]} vocab tokens after filtering "
          "special/byte-fallback/reserved tokens.")

    embed_norms = W_E[mask].norm(dim=-1).numpy()
    unembed_norms = W_U[mask].norm(dim=-1).numpy()

    plot_norm_distributions(embed_norms, unembed_norms, PLOTS_DIR, "embedding_unembedding_norms.png")


if __name__ == "__main__":
    main()
