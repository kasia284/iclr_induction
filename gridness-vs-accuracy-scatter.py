"""Scatter plot testing whether "gridness" of the FINAL-LAYER
representations predicts how well the model actually solves the grid
task, across every word list with both a cached final-layer activation
file and a cached accuracy curve.

x axis: distance correlation between each word's cached final-layer class
mean activation and its (row, col) grid position (utils.
compute_distance_correlation) -- same quantity as gridness-static-vs-
final-scatter.py's y axis.

y axis: real accuracy at full context (mean over the last position, seq
len 1400, averaged over the 16 random-walk sequences) -- same quantity as
static-embedding-argmax-accuracy.py's y axis.

One point per word list. SWEEP_KEYS is auto-populated with every
WORD_LISTS key that has BOTH a cached results/reproduce/data/{key}/
pca_all_layers_Grid_{key}.npz AND a cached accuracies_Grid_{key}.npz
(i.e. every key 01_reproduce.py has already been run for) -- pure cache
reads, no model / GPU needed, CPU-only.
"""
import importlib.util
import math
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go

from utils import Grid, setup_plotting, save_figure, save_plotly, compute_distance_correlation
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = "results/reproduce/plots/gridness_vs_accuracy"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")


def discover_sweep_keys():
    """Every WORD_LISTS key with BOTH a cached final-layer class-means
    file and a cached accuracy curve."""
    keys = []
    for key in WORD_LISTS:
        pca_path = f"results/reproduce/data/{key}/pca_all_layers_Grid_{key}.npz"
        acc_path = f"results/reproduce/data/{key}/accuracies_Grid_{key}.npz"
        if os.path.exists(pca_path) and os.path.exists(acc_path):
            keys.append(key)
    return keys


SWEEP_KEYS = discover_sweep_keys()


def make_grid(word_list_key):
    words = WORD_LISTS[word_list_key]
    side = math.isqrt(len(words))
    if side * side != len(words):
        raise ValueError(f"{word_list_key!r} has {len(words)} words, not a perfect square.")
    return Grid(words=words, rows=side, cols=side)


def compute_final_dc(word_list_key):
    words = WORD_LISTS[word_list_key]
    grid = make_grid(word_list_key)
    grid_coords = reproduce_mod.get_grid_coords(grid, words)

    class_means, final_layer = dot_product_mod.load_final_layer_class_means(word_list_key)
    return compute_distance_correlation(class_means, grid_coords), final_layer


def load_real_full_context_accuracy(word_list_key):
    """Mean accuracy at the last position (seq len 1400) of the cached
    accuracy curve, averaged over the 16 random-walk sequences."""
    path = f"results/reproduce/data/{word_list_key}/accuracies_Grid_{word_list_key}.npz"
    all_accs = np.load(path)["all_accs"]
    return float(all_accs[:, -1].mean())


def assign_families(keys):
    """Group keys by their underlying SET of words (frozenset(WORD_LISTS[key])),
    so permuted/unpermuted/other grid-position variants of the same word set
    (e.g. morphology, morphology_permuted, morphology_rand1/2/3,
    morphology_corners) share one family/color, while lists that merely
    share a name prefix but use different words (e.g. two_digit_numbers_4to7,
    two_digit_numbers_2468) do NOT. Returns {key: (family_label, family_index)}."""
    family_of = {}
    label_of_family = {}
    for key in keys:
        fam = frozenset(WORD_LISTS[key])
        family_of.setdefault(fam, []).append(key)
    # Label each family by its shortest member key (usually the unpermuted base).
    for fam, members in family_of.items():
        label_of_family[fam] = min(members, key=len)
    families_sorted = sorted(family_of.keys(), key=lambda f: label_of_family[f])
    family_index = {fam: i for i, fam in enumerate(families_sorted)}
    return {
        key: (label_of_family[frozenset(WORD_LISTS[key])], family_index[frozenset(WORD_LISTS[key])])
        for key in keys
    }, len(families_sorted)


def plot_scatter(dc_by_key, acc_by_key, keys, out_dir,
                  filename="gridness_vs_accuracy"):
    xs = np.array([dc_by_key[k] for k in keys])
    ys = np.array([acc_by_key[k] for k in keys])
    r = float(np.corrcoef(xs, ys)[0, 1])

    key_to_family, n_families = assign_families(keys)
    cmap = cm.get_cmap("tab20", max(n_families, 1))
    colors = [cmap(key_to_family[k][1]) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xs, ys, c=colors, s=35, zorder=3)
    for x, y, k in zip(xs, ys, keys):
        ax.annotate(k, (x, y), fontsize=5.5, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Distance correlation (final-layer activations vs. grid)")
    ax.set_ylabel("Real accuracy (full context, seq len 1400)")
    ax.set_title(f"Final-layer gridness vs. real accuracy (n={len(keys)}, r={r:.3f})", fontsize=10)

    # One legend entry per family (word-set), not per point.
    from matplotlib.lines import Line2D
    family_labels_seen = sorted({key_to_family[k] for k in keys}, key=lambda t: t[1])
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(idx),
               markeredgecolor="none", markersize=7, label=label)
        for label, idx in family_labels_seen
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=True, framealpha=1.0, edgecolor="gray", fontsize=7, ncol=1,
              title="Word-set family", title_fontsize=8)

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {filename} (Pearson r = {r:.4f})")

    # ── Plotly interactive (hover shows word_list_key) ──────────────────────
    pfig = go.Figure()
    for label, idx in family_labels_seen:
        mask = np.array([key_to_family[k][0] == label for k in keys])
        rgba = cmap(idx)
        color = "rgba({},{},{},1.0)".format(*(int(c * 255) for c in rgba[:3]))
        pfig.add_trace(go.Scatter(
            x=xs[mask].tolist(), y=ys[mask].tolist(), mode="markers",
            marker=dict(color=color, size=9),
            text=[k for k, m in zip(keys, mask) if m],
            name=label,
            hovertemplate="%{text}<br>final DC: %{x:.3f}<br>real accuracy: %{y:.3f}<extra></extra>",
        ))
    pfig.update_layout(
        title=f"Final-layer gridness vs. real accuracy (n={len(keys)}, r={r:.3f})",
        xaxis_title="Distance correlation (final-layer activations vs. grid)",
        yaxis_title="Real accuracy (full context, seq len 1400)",
        template="plotly_white",
        legend_title_text="Word-set family",
    )
    save_plotly(pfig, out_dir, f"{filename}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    if not SWEEP_KEYS:
        raise RuntimeError(
            "No word lists with both cached final-layer activations and cached "
            "accuracy curves found -- run 01_reproduce.py for at least one word list first."
        )
    print(f"Found {len(SWEEP_KEYS)} word lists with cached final-layer activations + accuracy curves.")

    dc_by_key, acc_by_key = {}, {}
    for key in SWEEP_KEYS:
        dc_by_key[key], final_layer = compute_final_dc(key)
        acc_by_key[key] = load_real_full_context_accuracy(key)
        print(f"{key}: final-layer ({final_layer}) DC = {dc_by_key[key]:.4f}, real accuracy = {acc_by_key[key]:.4f}")

    plot_scatter(dc_by_key, acc_by_key, SWEEP_KEYS, PLOTS_DIR)


if __name__ == "__main__":
    main()
