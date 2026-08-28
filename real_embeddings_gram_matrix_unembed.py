"""Analogous to real_embeddings_gram_matrix.py, but using the unembedding
matrix (W_U / lm_head.weight) instead of the embedding matrix (W_E), for the
same word lists.

Reads W_U directly from the cached safetensors shard (same lightweight
loader as real_embeddings_nearest_neighbors.py's load_embedding_only_model)
rather than through utils.load_model()'s full HookedTransformer -- loading
all 32 layers of the 8B-parameter model OOMs on CPU-only/sandboxed sessions.
lm_head.weight is stored as an nn.Linear weight, [d_vocab, d_model], so no
transpose is needed to index it per-token here (unlike HookedTransformer's
internal W_U convention of [d_model, d_vocab]).
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from utils import MODEL_NAME, setup_plotting, compute_and_plot_similarity_panels
from word_lists import WORD_LISTS

DATA_DIR = "results/gram-matrix-unembed"
PLOTS_DIR = "results/gram-matrix-unembed"

RUN_WORD_LIST_KEYS = [
    "morphology", "morphology_permuted",
]


def load_unembedding_matrix():
    """Read lm_head.weight ([d_vocab, d_model]) directly out of the cached
    safetensors shard, without loading the rest of the model."""
    hf_home = os.environ.get("HF_HOME")
    cache_root = os.path.join(hf_home, "hub") if hf_home else os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_root, f"models--{MODEL_NAME.replace('/', '--')}")
    snapshot_dirs = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for {MODEL_NAME} under {model_dir}")
    snapshot_dir = snapshot_dirs[0]

    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shard_path = os.path.join(snapshot_dir, weight_map["lm_head.weight"])

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight").float()


def to_single_token(tokenizer, word):
    """Mirror model.to_single_token: try with a leading space first (most
    LLM tokenizers expect it for standalone words), then without. Raises
    ValueError if the word isn't a single token either way."""
    for candidate in (f" {word}", word):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"The word '{word}' cannot be represented as a single token in this model's vocabulary!")


# ── Main ───────────────────────────────────────────────────────────────────────

def process_word_list(W_U, tokenizer, word_list_key, words):
    try:
        print(f"Mapping words directly to unique token IDs for '{word_list_key}'...")
        token_ids = [to_single_token(tokenizer, word) for word in words]
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Pull the weights directly from the unembedding matrix (W_U).
        # Shape: (len(words), d_model)
        static_unembeddings = W_U[token_ids_tensor].numpy()
        print(f"Retrieved unembedding matrix directly from W_U. Shape: {static_unembeddings.shape}")

        # Compute pairwise cosine similarity, Gram, and L2 distance matrices
        # and plot the heatmaps
        compute_and_plot_similarity_panels(
            static_unembeddings, words, PLOTS_DIR,
            f"unembeddings_similarity_{word_list_key}.png",
            title=f"Static Unembeddings ({word_list_key})",
        )

    except ValueError as val_err:
        print(f"\n[Vocabulary Error] ({word_list_key}): {val_err}")
    except Exception as e:
        print(f"\nAn unexpected error occurred ({word_list_key}): {e}")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries

    print(f"Loading unembedding matrix (lm_head.weight) for {MODEL_NAME}...")
    W_U = load_unembedding_matrix()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for word_list_key in RUN_WORD_LIST_KEYS:
        process_word_list(W_U, tokenizer, word_list_key, WORD_LISTS[word_list_key])


if __name__ == "__main__":
    main()
