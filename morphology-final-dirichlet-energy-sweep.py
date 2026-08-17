"""Like morphology-dirichlet-energy-sweep.py, but uses each permutation's
FINAL-LAYER activations (the contextualized, post-forward-pass
representations from 01_reproduce.py's grid task) instead of the static
embedding-layer vectors (W_E) as node features for Dirichlet energy.

Low Dirichlet energy here = grid-adjacent words end up close together in
representation space AFTER the model has processed the random-walk
context -- i.e. the actual "does the grid geometry emerge" question from
Park et al., rather than the purely static W_E proxy.

Produces a plot analogous to accuracy_curve_sweep.png (mean accuracy vs.
sequence length, one line per permutation), except lines are colored on a
continuous scale by their permutation's *final-layer* Dirichlet energy
instead of by distinct qualitative colors.

Final-layer activations and accuracy curves are both read from
01_reproduce.py's cache (results/reproduce/data/{key}/pca_all_layers_Grid_
{key}.npz and accuracies_Grid_{key}.npz respectively, the latter via
load_final_layer_class_means imported from activation-unembedding-dot-
product.py), so 01_reproduce.py must be run first for each key. No model
load needed at all -- pure cache reads, CPU-only.
"""
import importlib.util
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import plotly.graph_objects as go

from utils import Grid, setup_plotting, save_figure, save_plotly, plotly_line_layout, compute_dirichlet_energy
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")

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


def compute_final_layer_dirichlet_energy(word_list_key):
    """Dirichlet energy of word_list_key's grid graph, using each word's
    cached final-layer (contextualized) mean activation as its node
    feature."""
    words = WORD_LISTS[word_list_key]
    grid = Grid(words=words, rows=4, cols=4)
    adjacency = grid.build_adjacency_matrix()

    activations, final_layer = dot_product_mod.load_final_layer_class_means(word_list_key)
    return compute_dirichlet_energy(activations, adjacency), final_layer


def plot_accuracy_by_final_dirichlet_energy(accs_by_key, de_by_key, out_dir,
                                             filename="accuracy_curve_sweep_by_final_dirichlet_energy"):
    """Like 01_reproduce.py's plot_accuracy_comparison, but each
    permutation's line is colored on a continuous scale by its final-layer
    Dirichlet energy (de_by_key), with a colorbar, instead of a fixed
    qualitative palette."""
    de_values = np.array([de_by_key[k] for k in accs_by_key])
    norm = Normalize(vmin=de_values.min(), vmax=de_values.max())
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(5.2, 3))
    for key, all_accs in accs_by_key.items():
        color = cmap(norm(de_by_key[key]))
        mean = all_accs.mean(axis=0)
        std = all_accs.std(axis=0)
        ax.plot(mean, color=color, linewidth=1.2, label=f"{key} (DE={de_by_key[key]:.3f})")
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        alpha=0.15, color=color, edgecolor="none")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title("Accuracy vs sequence length, colored by final-layer\nDirichlet energy (morphology permutations)", fontsize=9)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=7)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="Dirichlet energy (final-layer activations)")

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {filename}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure()
    for key, all_accs in accs_by_key.items():
        rgba_tuple = cmap(norm(de_by_key[key]))
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
            name=f"{key} (DE={de_by_key[key]:.3f})",
            hovertemplate=f"{key}<br>DE={de_by_key[key]:.3f}<br>pos: " + "%{x}<br>accuracy: %{y:.3f}<extra></extra>",
        ))
    pfig.update_layout(**plotly_line_layout(
        "Accuracy vs sequence length, colored by final-layer Dirichlet energy",
        "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, out_dir, f"{filename}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    de_by_key = {}
    for key in MORPHOLOGY_SWEEP_KEYS:
        de, final_layer = compute_final_layer_dirichlet_energy(key)
        de_by_key[key] = de
        print(f"{key}: final-layer ({final_layer}) Dirichlet energy = {de:.4f}")

    accs_by_key = {k: load_cached_accuracies(k) for k in MORPHOLOGY_SWEEP_KEYS}
    plot_accuracy_by_final_dirichlet_energy(accs_by_key, de_by_key, PLOTS_DIR)


if __name__ == "__main__":
    main()
