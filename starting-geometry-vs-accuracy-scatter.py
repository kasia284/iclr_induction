"""Scatter plots testing whether "conflict" between a word list's STARTING
geometry -- its raw W_E embeddings, straight out of the embedding layer,
before any forward pass / context -- and its task GRAPH predicts how well
the model actually solves the task, across every word list with a cached
accuracy curve.

Two geometry metrics, both computed from each word's static W_E embedding
against its own grid, using the SAME machinery every other static-vs-grid
script in this repo uses (morphology-distance-correlation-sweep.py,
gridness-static-vs-final-scatter.py, permutation-graph-similarity.py):

  - Distance correlation (utils.compute_distance_correlation_graph_embedding):
    Pearson correlation between W_E Euclidean distances and the grid's
    shortest-path (hop-count) distances -- graph topology, not (row, col)
    coordinates, so it's unaffected by Grid's periodic-boundary wraparound
    (raw Manhattan distance on (row, col) doesn't see that row 0 and row 3
    are adjacent on a 4x4 torus; hop-count does). HIGH = starting geometry
    already mirrors the grid -- LOW conflict.
  - Dirichlet energy (utils.compute_dirichlet_energy): sum of squared W_E
    distances over grid-adjacent pairs, normalized by the sum over all
    pairs. LOW = grid-adjacent words already sit close together in W_E --
    LOW conflict. HIGH = grid adjacency carries no signal in the starting
    geometry -- HIGH conflict.

So on the distance-correlation scatter, "more conflicting starting
geometry" is to the LEFT (low x); on the Dirichlet-energy scatter, it's to
the RIGHT (high x). Both share the same y axis: real accuracy at full
context (mean over the last position, seq len 1400, averaged over the 16
random-walk sequences) -- same quantity as gridness-vs-accuracy-scatter.py's
y axis, just plotted here against the STARTING geometry's conflict instead
of the FINAL-LAYER geometry's alignment.

Usage:
    python starting-geometry-vs-accuracy-scatter.py

    # Same, but "correct" also counts probability mass on the CURRENT
    # token itself, not just on its grid neighbors ("staying put" counts
    # too). Reads accuracies_Grid_{key}_with_self.npz instead of
    # accuracies_Grid_{key}.npz -- run compute-accuracy-with-self.py
    # first (needs a GPU) to produce that cache. Output filenames get a
    # "_with_self" suffix so they never collide with the regular run's.
    python starting-geometry-vs-accuracy-scatter.py --with-self

One point per word list. Restricted to the "morphology" / "text_numbers" /
"two_digit_numbers" families (FAMILY_BASE_KEYS below, same restriction as
distance-correlation-accuracy-phase-plane.py) -- i.e. each base key plus
all of its grid-position permutations/rand variants (same underlying word
SET, different grid layout). SWEEP_KEYS is those family members that also
have a cached accuracies_Grid_{key}.npz (i.e. 01_reproduce.py has already
been run for them) -- no final-layer activation cache needed, since both
metrics here only touch W_E. Reads W_E via the
same lightweight safetensors-only loader as real-embeddings-nearest-
neighbors.py (imported via importlib since its filename isn't a valid
module identifier), and token ids via to_single_token from
activation-unembedding-dot-product.py. Family coloring (assign_families)
is reused from gridness-vs-accuracy-scatter.py. CPU-only, no GPU / new
model forward passes.
"""
import importlib.util
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch

from utils import Grid, setup_plotting, save_figure, save_plotly, compute_distance_correlation_graph_embedding, compute_dirichlet_energy
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = "results/reproduce/plots/gridness_vs_accuracy"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nn_mod = _load_module("real_embeddings_nearest_neighbors", "real-embeddings-nearest-neighbors.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")
scatter_mod = _load_module("gridness_vs_accuracy_scatter", "gridness-vs-accuracy-scatter.py")


FAMILY_BASE_KEYS = ["morphology", "text_numbers", "two_digit_numbers"]


def family_keys():
    """Every WORD_LISTS key sharing a word SET with one of FAMILY_BASE_KEYS
    (i.e. each base key plus all its grid-position permutations/rand
    variants)."""
    family_word_sets = {frozenset(WORD_LISTS[base]) for base in FAMILY_BASE_KEYS}
    return [k for k in WORD_LISTS if frozenset(WORD_LISTS[k]) in family_word_sets]


def accuracy_cache_path(word_list_key, with_self=False):
    suffix = "_with_self" if with_self else ""
    return f"results/reproduce/data/{word_list_key}/accuracies_Grid_{word_list_key}{suffix}.npz"


def load_real_full_context_accuracy(word_list_key, with_self=False):
    """Mean accuracy at the last position (seq len 1400), averaged over the
    cached random-walk sequences. Mirrors gridness-vs-accuracy-scatter.py's
    function of the same name, but can read the include_self=True variant
    (accuracies_Grid_{key}_with_self.npz, from compute-accuracy-with-self.py)
    instead of the regular "neighbors only" cache."""
    all_accs = np.load(accuracy_cache_path(word_list_key, with_self=with_self))["all_accs"]
    return float(all_accs[:, -1].mean())


def discover_sweep_keys(with_self=False):
    """family_keys() members that also have a cached accuracy curve."""
    keys = []
    for key in family_keys():
        acc_path = accuracy_cache_path(key, with_self=with_self)
        if os.path.exists(acc_path):
            keys.append(key)
        else:
            print(f"Skipping {key}: missing {os.path.basename(acc_path)} "
                  f"(run {'compute-accuracy-with-self.py' if with_self else '01_reproduce.py'} "
                  f"with this key first).")
    return keys


def make_grid(word_list_key):
    words = WORD_LISTS[word_list_key]
    side = math.isqrt(len(words))
    if side * side != len(words):
        raise ValueError(f"{word_list_key!r} has {len(words)} words, not a perfect square.")
    return Grid(words=words, rows=side, cols=side)


def compute_starting_geometry(W_E, tokenizer, word_list_key):
    """(distance correlation, Dirichlet energy) between word_list_key's own
    grid and its words' raw W_E embeddings."""
    words = WORD_LISTS[word_list_key]
    grid = make_grid(word_list_key)
    adjacency = grid.build_adjacency_matrix()  # already in `words` order

    token_ids = [dot_product_mod.to_single_token(tokenizer, w) for w in words]
    embeddings = W_E[torch.tensor(token_ids, dtype=torch.long)].numpy()

    dc = compute_distance_correlation_graph_embedding(embeddings, adjacency)
    de = compute_dirichlet_energy(embeddings, adjacency)
    return dc, de


def plot_scatter(x_by_key, acc_by_key, keys, out_dir, x_label, title_prefix, filename):
    """Mirrors gridness-vs-accuracy-scatter.py's plot_scatter, just with a
    caller-supplied x axis (static distance correlation or Dirichlet
    energy) instead of always final-layer distance correlation."""
    xs = np.array([x_by_key[k] for k in keys])
    ys = np.array([acc_by_key[k] for k in keys])
    r = float(np.corrcoef(xs, ys)[0, 1])

    key_to_family, n_families = scatter_mod.assign_families(keys)
    cmap = cm.get_cmap("tab20", max(n_families, 1))
    colors = [cmap(key_to_family[k][1]) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xs, ys, c=colors, s=35, zorder=3)
    for x, y, k in zip(xs, ys, keys):
        ax.annotate(k, (x, y), fontsize=5.5, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Real accuracy (full context, seq len 1400)")
    ax.set_title(f"{title_prefix} (n={len(keys)}, r={r:.3f})", fontsize=10)

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
            hovertemplate="%{text}<br>x: %{x:.3f}<br>real accuracy: %{y:.3f}<extra></extra>",
        ))
    pfig.update_layout(
        title=f"{title_prefix} (n={len(keys)}, r={r:.3f})",
        xaxis_title=x_label,
        yaxis_title="Real accuracy (full context, seq len 1400)",
        template="plotly_white",
        legend_title_text="Word-set family",
    )
    save_plotly(pfig, out_dir, f"{filename}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    with_self = "--with-self" in sys.argv[1:]
    sweep_keys = discover_sweep_keys(with_self=with_self)

    if not sweep_keys:
        cache_name = "accuracies_Grid_{key}_with_self.npz" if with_self else "accuracies_Grid_{key}.npz"
        producer = "compute-accuracy-with-self.py" if with_self else "01_reproduce.py"
        raise RuntimeError(f"No word lists with a cached {cache_name} found -- run {producer} first.")
    print(f"Found {len(sweep_keys)} word lists with cached accuracy curves"
          f"{' (with-self)' if with_self else ''}.")

    print("Loading static embedding matrix (W_E)...")
    model = nn_mod.load_embedding_only_model()
    W_E, tokenizer = model.W_E, model.tokenizer

    dc_by_key, de_by_key, acc_by_key = {}, {}, {}
    for key in sweep_keys:
        dc_by_key[key], de_by_key[key] = compute_starting_geometry(W_E, tokenizer, key)
        acc_by_key[key] = load_real_full_context_accuracy(key, with_self=with_self)
        print(f"{key}: starting DC = {dc_by_key[key]:.4f}, starting DE = {de_by_key[key]:.4f}, "
              f"real accuracy = {acc_by_key[key]:.4f}")

    acc_label_suffix = " (neighbors + self)" if with_self else ""
    filename_suffix = "_with_self" if with_self else ""
    plot_scatter(
        dc_by_key, acc_by_key, sweep_keys, PLOTS_DIR,
        x_label="Distance correlation (static W_E vs. grid, hop-count)",
        title_prefix=f"Starting-geometry distance correlation vs. real accuracy{acc_label_suffix}",
        filename=f"starting_geometry_distance_correlation_vs_accuracy{filename_suffix}",
    )
    plot_scatter(
        de_by_key, acc_by_key, sweep_keys, PLOTS_DIR,
        x_label="Dirichlet energy (static W_E vs. grid)",
        title_prefix=f"Starting-geometry Dirichlet energy vs. real accuracy{acc_label_suffix}",
        filename=f"starting_geometry_dirichlet_energy_vs_accuracy{filename_suffix}",
    )


if __name__ == "__main__":
    main()
