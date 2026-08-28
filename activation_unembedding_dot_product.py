"""Heatmap of dot products between each token's mean final-layer activation
and each token's unembedding vector, to investigate why next-token accuracy
is *higher* for a permuted word list (e.g. text_numbers_permuted) than for
the real one, despite the permuted list showing no grid structure in its
PCA plots.

Row i, col j = dot(mean_activation(token_i), W_U[token_j]) -- a restricted,
approximate "logit lens" read: it's what token j's logit contribution from
token i's mean representation would be, ignoring ln_final's normalization,
any bias, and the other ~128k vocab entries. Still directly diagnostic here
because next-token accuracy is exactly a sum of softmax'd versions of these
same dot products (restricted to grid-neighbor columns).

Mean activations are read from the already-cached
results/reproduce/data/{key}/pca_all_layers_Grid_{key}.npz (the *last*
layer's hook_resid_pre class means, aggregated over N_SEQUENCES=16 random
walks exactly as in 01_reproduce.py) -- no GPU/model forward pass needed.
W_U (lm_head.weight) is read directly from the cached safetensors shard
(same lightweight loader as real_embeddings_gram_matrix_unembed.py), so
this whole script runs on CPU only.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from utils import MODEL_NAME, Grid, setup_plotting, save_figure, _label_heatmap_ax
from word_lists import WORD_LISTS

DATA_DIR = "results/reproduce/data"
PLOTS_DIR = "results/activation-unembedding-dot-product"


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
    """Mirror model.to_single_token: try with a leading space first, then
    without. Raises ValueError if the word isn't a single token either way."""
    for candidate in (f" {word}", word):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"The word '{word}' cannot be represented as a single token in this model's vocabulary!")


def load_final_layer_class_means(word_list_key):
    """Read the cached last-layer class means ([n_words, d_model], one row
    per word in WORD_LISTS[word_list_key] order) written by 01_reproduce.py."""
    path = os.path.join(DATA_DIR, word_list_key, f"pca_all_layers_Grid_{word_list_key}.npz")
    data = np.load(path)
    final_layer = max(int(k) for k in data.files)
    return data[str(final_layer)], final_layer


def plot_dot_product_heatmap(activations, unembeddings, words, adjacency, word_list_key, final_layer):
    """activations, unembeddings: [n_words, d_model], same word order. Rows
    = source token's mean activation, cols = target token's unembedding.
    Grid-adjacent (row, col) pairs (per this word list's own Grid layout,
    i.e. the actual accuracy targets) get a black box overlay."""
    dots = activations @ unembeddings.T  # [n_words, n_words]

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.abs(dots).max()
    im = ax.imshow(dots, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dot(mean activation, unembedding)")
    _label_heatmap_ax(ax, words)
    ax.set_xlabel("Unembedding vector of token j")
    ax.set_ylabel("Mean activation of token i")

    for i in range(len(words)):
        for j in range(len(words)):
            if adjacency[i, j]:
                ax.add_patch(patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="black", linewidth=1.5,
                ))

    # Mark each row's argmax column -- i.e. the token the model would
    # actually place the most restricted-vocab logit mass on, given token
    # i's representation -- distinguishing whether that top pick is a
    # valid grid neighbor or not.
    row_argmax = dots.argmax(axis=1)
    for i, j in enumerate(row_argmax):
        if adjacency[i, j]:
            ax.scatter(j, i, s=110, facecolors="none", edgecolors="limegreen",
                       linewidths=2.2, marker="o", zorder=5)
        else:
            ax.scatter(j, i, s=90, color="red", marker="x", linewidths=2.2, zorder=5)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="limegreen", markeredgewidth=2, markersize=9,
               label="Row argmax = valid neighbor"),
        Line2D([0], [0], marker="x", color="red", markeredgewidth=2, markersize=9,
               label="Row argmax = not a neighbor", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              frameon=True, framealpha=0.9, edgecolor="gray")

    ax.set_title(
        f"Activation . Unembedding dot products, layer {final_layer} ({word_list_key})\n"
        "black boxes = grid-adjacent (i, j) pairs; markers = each row's argmax",
        fontsize=10,
    )
    save_figure(fig, PLOTS_DIR, f"dot_product_heatmap_{word_list_key}.pdf")
    fig.savefig(os.path.join(PLOTS_DIR, f"dot_product_heatmap_{word_list_key}.png"), dpi=150)
    print(f"Saved {PLOTS_DIR}/dot_product_heatmap_{word_list_key}.png")

    # How much dot-product mass falls on grid-adjacent vs. non-adjacent
    # cells, on average -- a quick numeric readout alongside the heatmap.
    adj_mask = adjacency.astype(bool)
    print(
        f"  mean dot product on grid-adjacent cells:     {dots[adj_mask].mean():.3f}\n"
        f"  mean dot product on non-adjacent cells:      {dots[~adj_mask & ~np.eye(len(words), dtype=bool)].mean():.3f}\n"
        f"  mean dot product on diagonal (self) cells:   {dots[np.eye(len(words), dtype=bool)].mean():.3f}"
    )


def process_word_list(W_U, tokenizer, word_list_key):
    words = WORD_LISTS[word_list_key]
    side = round(len(words) ** 0.5)
    grid = Grid(words=words, rows=side, cols=side)
    adjacency = grid.build_adjacency_matrix()

    activations, final_layer = load_final_layer_class_means(word_list_key)

    token_ids = [to_single_token(tokenizer, w) for w in words]
    unembeddings = W_U[torch.tensor(token_ids, dtype=torch.long)].numpy()

    plot_dot_product_heatmap(activations, unembeddings, words, adjacency, word_list_key, final_layer)


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    print(f"Loading unembedding matrix (lm_head.weight) for {MODEL_NAME}...")
    W_U = load_unembedding_matrix()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for word_list_key in ["text_numbers", "text_numbers_permuted"]:
        print(f"\n=== {word_list_key} ===")
        process_word_list(W_U, tokenizer, word_list_key)


if __name__ == "__main__":
    main()
