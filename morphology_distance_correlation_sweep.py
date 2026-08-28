"""For each grid-position permutation in the morphology sweep (see
01_reproduce.py's MORPHOLOGY_SWEEP_KEYS), computes the distance correlation
between the permutation's grid layout and its STATIC embeddings -- the
token embeddings straight out of the embedding layer (W_E), before any
forward pass / context -- via utils.compute_distance_correlation.

High distance correlation = representation-space (W_E) distances already
mirror grid (row, col) Manhattan distances, before the model has done
anything with context. Like morphology_dirichlet_energy_sweep.py's
Dirichlet energy, this is a purely static (no forward pass, no GPU) proxy
for "does grid adjacency align with the model's pretrained prior" -- just a
different geometric summary (global rank-order-style correlation over all
pairs, vs. Dirichlet energy's local adjacent-vs-all-pairs energy ratio).

Produces a plot analogous to accuracy_curve_sweep.png (mean accuracy vs.
sequence length, one line per permutation), except lines are colored on a
continuous scale by their permutation's distance correlation instead of by
distinct qualitative colors.

Reads W_E via the same lightweight safetensors-only loader as
real_embeddings_nearest_neighbors.py (imported via importlib, consistent
with how sibling scripts in this repo share helpers), and token ids via
to_single_token from activation_unembedding_dot_product.py. Accuracy
curves are read from 01_reproduce.py's cache
(results/reproduce/data/{key}/accuracies_Grid_{key}.npz), so
01_reproduce.py must be run first for each key. CPU-only.
"""
import importlib.util
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import plotly.graph_objects as go
import torch

from utils import Grid, setup_plotting, save_figure, save_plotly, plotly_line_layout, compute_distance_correlation
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nn_mod = _load_module("real_embeddings_nearest_neighbors", "real_embeddings_nearest_neighbors.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation_unembedding_dot_product.py")

PLOTS_DIR = "results/reproduce/plots/morphology_sweep_comparison"

# Mirrors 01_reproduce.py's MORPHOLOGY_SWEEP_KEYS.
MORPHOLOGY_SWEEP_KEYS = [
    "morphology_corners", "morphology", "morphology_rand3",
    "morphology_rand2", "morphology_rand1", "morphology_permuted",
]


def load_cached_accuracies(word_list_key, graph_type="Grid"):
    """Mirrors 01_reproduce.py's function of the same name."""
    path = os.path.join(
        f"results/reproduce/data/{word_list_key}",
        f"accuracies_{graph_type}_{word_list_key}.npz",
    )
    return np.load(path)["all_accs"]


def get_grid_coords(grid, words):
    """Mirrors 01_reproduce.py's function of the same name: [n_words, 2]
    (row, col) per word, in `words` order."""
    return np.array([[grid.word_to_row[w], grid.word_to_col[w]] for w in words])


def compute_static_distance_correlation(W_E, tokenizer, word_list_key):
    """Distance correlation between word_list_key's grid layout and each
    word's raw W_E embedding (before any forward pass)."""
    words = WORD_LISTS[word_list_key]
    grid = Grid(words=words, rows=4, cols=4)
    grid_coords = get_grid_coords(grid, words)

    token_ids = [dot_product_mod.to_single_token(tokenizer, w) for w in words]
    embeddings = W_E[torch.tensor(token_ids, dtype=torch.long)].numpy()

    return compute_distance_correlation(embeddings, grid_coords)


def plot_accuracy_by_distance_correlation(accs_by_key, dc_by_key, out_dir,
                                           filename="accuracy_curve_sweep_by_distance_correlation"):
    """Like 01_reproduce.py's plot_accuracy_comparison, but each permutation's
    line is colored on a continuous scale by its static-embedding distance
    correlation (dc_by_key), with a colorbar, instead of a fixed qualitative
    palette."""
    dc_values = np.array([dc_by_key[k] for k in accs_by_key])
    norm = Normalize(vmin=dc_values.min(), vmax=dc_values.max())
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(5.2, 3))
    for key, all_accs in accs_by_key.items():
        color = cmap(norm(dc_by_key[key]))
        mean = all_accs.mean(axis=0)
        std = all_accs.std(axis=0)
        ax.plot(mean, color=color, linewidth=1.2, label=f"{key} (DC={dc_by_key[key]:.3f})")
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        alpha=0.15, color=color, edgecolor="none")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title("Accuracy vs sequence length, colored by static-embedding\nDistance correlation (morphology permutations)", fontsize=9)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=7)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="Distance correlation (static W_E)")

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {filename}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure()
    for key, all_accs in accs_by_key.items():
        rgba_tuple = cmap(norm(dc_by_key[key]))
        color = "rgba({},{},{},1.0)".format(*(int(c * 255) for c in rgba_tuple[:3]))
        fill_color = "rgba({},{},{},0.15)".format(*(int(c * 255) for c in rgba_tuple[:3]))
        mean = all_accs.mean(axis=0)
        std = all_accs.std(axis=0)
        x = list(range(len(mean)))
        pfig.add_trace(go.Scatter(
            x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        pfig.add_trace(go.Scatter(
            x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill_color,
            showlegend=False, hoverinfo="skip",
        ))
        pfig.add_trace(go.Scatter(
            x=x, y=mean.tolist(), mode="lines",
            line=dict(color=color, width=2),
            name=f"{key} (DC={dc_by_key[key]:.3f})",
            hovertemplate=f"{key}<br>DC={dc_by_key[key]:.3f}<br>pos: " + "%{x}<br>accuracy: %{y:.3f}<extra></extra>",
        ))
    pfig.update_layout(**plotly_line_layout(
        "Accuracy vs sequence length, colored by static-embedding distance correlation",
        "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, out_dir, f"{filename}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    print("Loading static embedding matrix (W_E)...")
    model = nn_mod.load_embedding_only_model()
    W_E, tokenizer = model.W_E, model.tokenizer

    dc_by_key = {}
    for key in MORPHOLOGY_SWEEP_KEYS:
        dc_by_key[key] = compute_static_distance_correlation(W_E, tokenizer, key)
        print(f"{key}: Distance correlation = {dc_by_key[key]:.4f}")

    accs_by_key = {k: load_cached_accuracies(k) for k in MORPHOLOGY_SWEEP_KEYS}
    plot_accuracy_by_distance_correlation(accs_by_key, dc_by_key, PLOTS_DIR)


if __name__ == "__main__":
    main()
