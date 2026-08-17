"""Scatter plot testing whether "gridness" of the STATIC embeddings (W_E,
before any forward pass) predicts "gridness" of the FINAL-LAYER
contextualized mean activations (after the grid random-walk task), across
every word list with a cached 01_reproduce.py run.

x axis: distance correlation between each word's raw W_E embedding and its
(row, col) grid position (utils.compute_distance_correlation) -- same
quantity as morphology-distance-correlation-sweep.py, but computed for
every available word list, not just the morphology permutations.

y axis: distance correlation between each word's cached final-layer class
mean activation and its grid position -- same quantity as
morphology-final-dirichlet-energy-sweep.py's companion metric (that script
uses Dirichlet energy; this one uses distance correlation instead), read
via load_final_layer_class_means from activation-unembedding-dot-
product.py.

One point per word list. SWEEP_KEYS is auto-populated with every
WORD_LISTS key that has a cached results/reproduce/data/{key}/
pca_all_layers_Grid_{key}.npz (i.e. every key 01_reproduce.py has already
been run for) -- no new model forward passes needed for the y axis.
Static embeddings are read directly from the safetensors checkpoint (via
real-embeddings-nearest-neighbors.py's load_embedding_only_model), which is
CPU-only and doesn't require TransformerLens or a GPU either. So this
whole script is CPU-only.
"""
import glob
import importlib.util
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

from utils import Grid, setup_plotting, save_figure, save_plotly, compute_distance_correlation
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
nn_mod = _load_module("real_embeddings_nearest_neighbors", "real-embeddings-nearest-neighbors.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")

PLOTS_DIR = "results/reproduce/plots/gridness_static_vs_final"


def discover_sweep_keys():
    """Every WORD_LISTS key with a cached final-layer class-means file."""
    keys = []
    for key in WORD_LISTS:
        path = f"results/reproduce/data/{key}/pca_all_layers_Grid_{key}.npz"
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


def compute_static_dc(W_E, tokenizer, word_list_key):
    words = WORD_LISTS[word_list_key]
    grid = make_grid(word_list_key)
    grid_coords = reproduce_mod.get_grid_coords(grid, words)

    token_ids = [dot_product_mod.to_single_token(tokenizer, w) for w in words]
    embeddings = W_E[torch.tensor(token_ids, dtype=torch.long)].numpy()
    return compute_distance_correlation(embeddings, grid_coords)


def compute_final_dc(word_list_key):
    words = WORD_LISTS[word_list_key]
    grid = make_grid(word_list_key)
    grid_coords = reproduce_mod.get_grid_coords(grid, words)

    class_means, final_layer = dot_product_mod.load_final_layer_class_means(word_list_key)
    return compute_distance_correlation(class_means, grid_coords), final_layer


def plot_scatter(static_dc, final_dc, keys, out_dir, filename="static_vs_final_distance_correlation"):
    xs = np.array([static_dc[k] for k in keys])
    ys = np.array([final_dc[k] for k in keys])
    is_permuted = np.array(["_permuted" in k for k in keys])
    r = float(np.corrcoef(xs, ys)[0, 1])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xs[~is_permuted], ys[~is_permuted], c="#377eb8", label="unpermuted", s=35, zorder=3)
    ax.scatter(xs[is_permuted], ys[is_permuted], c="#e41a1c", label="permuted", s=35, zorder=3)
    for x, y, k in zip(xs, ys, keys):
        ax.annotate(k, (x, y), fontsize=5.5, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    lo, hi = min(xs.min(), ys.min()), max(xs.max(), ys.max())
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, zorder=1, label="y = x")
    ax.set_xlabel("Distance correlation (static W_E vs. grid)")
    ax.set_ylabel("Distance correlation (final-layer activations vs. grid)")
    ax.set_title(f"Static vs. final-layer gridness (n={len(keys)}, r={r:.3f})", fontsize=10)
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
            hovertemplate="%{text}<br>static DC: %{x:.3f}<br>final DC: %{y:.3f}<extra></extra>",
        ))
    pfig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash", width=1),
        name="y = x", hoverinfo="skip",
    ))
    pfig.update_layout(
        title=f"Static vs. final-layer gridness (n={len(keys)}, r={r:.3f})",
        xaxis_title="Distance correlation (static W_E vs. grid)",
        yaxis_title="Distance correlation (final-layer activations vs. grid)",
        template="plotly_white",
    )
    save_plotly(pfig, out_dir, f"{filename}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    if not SWEEP_KEYS:
        raise RuntimeError(
            "No word lists with cached pca_all_layers_Grid_{key}.npz found -- "
            "run 01_reproduce.py for at least one word list first."
        )
    print(f"Found {len(SWEEP_KEYS)} word lists with cached final-layer activations.")

    print("Loading static embedding matrix (W_E)...")
    model = nn_mod.load_embedding_only_model()
    W_E, tokenizer = model.W_E, model.tokenizer

    static_dc, final_dc = {}, {}
    for key in SWEEP_KEYS:
        static_dc[key] = compute_static_dc(W_E, tokenizer, key)
        final_dc[key], final_layer = compute_final_dc(key)
        print(f"{key}: static DC = {static_dc[key]:.4f}, final-layer ({final_layer}) DC = {final_dc[key]:.4f}")

    plot_scatter(static_dc, final_dc, SWEEP_KEYS, PLOTS_DIR)


if __name__ == "__main__":
    main()
