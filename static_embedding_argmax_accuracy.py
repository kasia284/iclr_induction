"""A zero-context, no-forward-pass baseline for the grid task: for each
word list, take each word's STATIC embedding (W_E, straight out of the
embedding layer) and predict the token whose UNEMBEDDING vector (W_U /
lm_head.weight) has the largest dot product with it -- i.e. what the
model would guess as "next word" if the residual stream went straight
from embedding to unembedding with nothing in between (no attention, no
MLPs, no context at all).

"Correct" = that argmax word is one of the current word's grid neighbors
(same definition of correctness as the real task's accuracy metric, just
hard argmax instead of softmax probability mass, and restricted to the
n candidate words in the list rather than the full ~128k vocab -- mirrors
the row-argmax check already drawn in activation_unembedding_dot_product.py's heatmaps, just turned into a single scalar per word list
and computed from W_E instead of final-layer mean activations).

static_argmax_accuracy = fraction of words whose top pick is a valid
neighbor, i.e. the "sequence length -> 0" limit of the real accuracy
curve, without ever running the model.

Plots this static baseline against each word list's REAL full-context
accuracy (mean over the last position of the cached accuracy curves from
01_reproduce.py, i.e. sequence length 1400) to see whether words whose
raw pretrained embeddings already "happen" to point at their grid
neighbors' unembeddings predict which word lists the model ends up
solving well after 1400 tokens of context.

W_E and W_U are both read directly from the cached safetensors shards (via
real_embeddings_nearest_neighbors.py's load_embedding_only_model and
activation_unembedding_dot_product.py's load_unembedding_matrix), and
real accuracies come from 01_reproduce.py's cache -- so this whole script
is CPU-only, no GPU / model forward pass required.
"""
import importlib.util
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

from utils import Grid, setup_plotting, save_figure, save_plotly
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nn_mod = _load_module("real_embeddings_nearest_neighbors", "real_embeddings_nearest_neighbors.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation_unembedding_dot_product.py")

PLOTS_DIR = "results/reproduce/plots/static_argmax_accuracy"


def discover_sweep_keys():
    """Every WORD_LISTS key with a cached real accuracy curve."""
    keys = []
    for key in WORD_LISTS:
        path = f"results/reproduce/data/{key}/accuracies_Grid_{key}.npz"
        if os.path.exists(path):
            keys.append(key)
    return keys


SWEEP_KEYS = discover_sweep_keys()


def make_grid(word_list_key):
    words = WORD_LISTS[word_list_key]
    side = math.isqrt(len(words))
    if side * side != len(words):
        raise ValueError(f"{word_list_key!r} has {len(words)} words, not a perfect square.")
    return Grid(words=words, rows=side, cols=side)


def compute_static_argmax_accuracy(W_E, W_U, tokenizer, word_list_key):
    words = WORD_LISTS[word_list_key]
    grid = make_grid(word_list_key)
    adjacency = grid.build_adjacency_matrix()

    token_ids = torch.tensor(
        [dot_product_mod.to_single_token(tokenizer, w) for w in words], dtype=torch.long
    )
    embeddings = W_E[token_ids]      # [n, d_model], static
    unembeddings = W_U[token_ids]    # [n, d_model], restricted to this list's own words

    dots = (embeddings @ unembeddings.T).numpy()  # [n, n]
    row_argmax = dots.argmax(axis=1)
    correct = np.array([adjacency[i, j] for i, j in enumerate(row_argmax)], dtype=bool)
    return float(correct.mean())


def load_real_full_context_accuracy(word_list_key):
    """Mean accuracy at the last position (seq len 1400) of the cached
    accuracy curve, averaged over the 16 random-walk sequences."""
    path = f"results/reproduce/data/{word_list_key}/accuracies_Grid_{word_list_key}.npz"
    all_accs = np.load(path)["all_accs"]
    return float(all_accs[:, -1].mean())


def plot_scatter(static_acc, real_acc, keys, out_dir,
                  filename="static_argmax_accuracy_vs_real_accuracy"):
    xs = np.array([static_acc[k] for k in keys])
    ys = np.array([real_acc[k] for k in keys])
    is_permuted = np.array(["_permuted" in k for k in keys])
    r = float(np.corrcoef(xs, ys)[0, 1])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xs[~is_permuted], ys[~is_permuted], c="#377eb8", label="unpermuted", s=35, zorder=3)
    ax.scatter(xs[is_permuted], ys[is_permuted], c="#e41a1c", label="permuted", s=35, zorder=3)
    for x, y, k in zip(xs, ys, keys):
        ax.annotate(k, (x, y), fontsize=5.5, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    lo, hi = 0.0, max(xs.max(), ys.max())
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, zorder=1, label="y = x")
    ax.set_xlabel("Static argmax accuracy (W_E · W_U, no context)")
    ax.set_ylabel("Real accuracy (full context, seq len 1400)")
    ax.set_title(f"Static embedding→unembedding baseline vs. real accuracy (n={len(keys)}, r={r:.3f})", fontsize=10)
    ax.legend(loc="best", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=8)

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {filename} (Pearson r = {r:.4f})")

    # ── Plotly interactive (hover shows word_list_key) ──────────────────────
    pfig = go.Figure()
    for mask, color, label in [(~is_permuted, "#377eb8", "unpermuted"), (is_permuted, "#e41a1c", "permuted")]:
        pfig.add_trace(go.Scatter(
            x=xs[mask].tolist(), y=ys[mask].tolist(), mode="markers",
            marker=dict(color=color, size=9),
            text=[k for k, m in zip(keys, mask) if m],
            name=label,
            hovertemplate="%{text}<br>static argmax acc: %{x:.3f}<br>real acc: %{y:.3f}<extra></extra>",
        ))
    pfig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash", width=1),
        name="y = x", hoverinfo="skip",
    ))
    pfig.update_layout(
        title=f"Static embedding→unembedding baseline vs. real accuracy (n={len(keys)}, r={r:.3f})",
        xaxis_title="Static argmax accuracy (W_E . W_U, no context)",
        yaxis_title="Real accuracy (full context, seq len 1400)",
        template="plotly_white",
    )
    save_plotly(pfig, out_dir, f"{filename}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    if not SWEEP_KEYS:
        raise RuntimeError(
            "No word lists with cached accuracies_Grid_{key}.npz found -- "
            "run 01_reproduce.py for at least one word list first."
        )
    print(f"Found {len(SWEEP_KEYS)} word lists with cached real accuracy curves.")

    print("Loading static embedding (W_E) and unembedding (W_U) matrices...")
    embed_model = nn_mod.load_embedding_only_model()
    W_E, tokenizer = embed_model.W_E, embed_model.tokenizer
    W_U = dot_product_mod.load_unembedding_matrix()

    static_acc, real_acc = {}, {}
    for key in SWEEP_KEYS:
        static_acc[key] = compute_static_argmax_accuracy(W_E, W_U, tokenizer, key)
        real_acc[key] = load_real_full_context_accuracy(key)
        print(f"{key}: static argmax accuracy = {static_acc[key]:.4f}, real full-context accuracy = {real_acc[key]:.4f}")

    plot_scatter(static_acc, real_acc, SWEEP_KEYS, PLOTS_DIR)


if __name__ == "__main__":
    main()
