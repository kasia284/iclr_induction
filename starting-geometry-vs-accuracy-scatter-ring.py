"""Ring-task analogue of starting-geometry-vs-accuracy-scatter.py: scatter
plots testing whether "conflict" between a word list's STARTING geometry
-- its raw W_E embeddings, straight out of the embedding layer, before any
forward pass / context -- and its task RING predicts how well the model
actually solves the task, across the "days_of_week" / "months_of_year"
families (01_reproduce_ring.py's RING_SWEEP_KEYS: each base key's natural
cyclic order plus its permuted/rand variants).

Two geometry metrics, both computed from each word's static W_E embedding
against its own ring's TOPOLOGY (adjacency-derived shortest-path hop-count
distance), not a 2D coordinate layout:

  - Distance correlation (utils.compute_distance_correlation_graph_embedding):
    Pearson correlation between W_E Euclidean distances and the ring's
    shortest-path (hop-count) distances -- graph distance on the ring side,
    never Euclidean/Manhattan on embedded coordinates, so it depends only
    on ring topology (which already handles wraparound correctly, e.g.
    Sunday and Monday are 1 hop apart, not maximally far). HIGH = starting
    geometry already mirrors the ring -- LOW conflict.
  - Dirichlet energy (utils.compute_dirichlet_energy): sum of squared W_E
    distances over ring-adjacent pairs, normalized by the sum over all
    pairs. LOW = ring-adjacent words already sit close together in W_E --
    LOW conflict. HIGH = ring adjacency carries no signal in the starting
    geometry -- HIGH conflict.

y axis (both): real accuracy at full context (mean over the last position,
seq len 1400, averaged over the cached random-walk sequences) -- read from
01_reproduce_ring.py's cache (results/reproduce_ring/data/{key}/
accuracies_Ring_{key}.npz), same "graph_accs"-or-"all_accs" fallback as
gridness-vs-accuracy-scatter-ring.py's load_real_full_context_accuracy.

One point per word list. SWEEP_KEYS is RING_SWEEP_KEYS filtered to keys
with a cached accuracies_Ring_{key}.npz (i.e. 01_reproduce_ring.py has
already been run for them) -- no PCA/class-means cache needed, since both
metrics here only touch W_E. Reads W_E via the same lightweight
safetensors-only loader as real-embeddings-nearest-neighbors.py, and token
ids via to_single_token from activation-unembedding-dot-product.py. Family
coloring (assign_families) is reused from gridness-vs-accuracy-scatter.py
(topology-agnostic). CPU-only, no GPU / new model forward passes.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch

from utils import Ring, setup_plotting, save_figure, save_plotly, compute_distance_correlation_graph_embedding, compute_dirichlet_energy
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "results/reproduce_ring/data"
PLOTS_DIR = "results/reproduce_ring/plots/gridness_vs_accuracy"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ring_mod = _load_module("reproduce_ring", "01_reproduce_ring.py")
nn_mod = _load_module("real_embeddings_nearest_neighbors", "real-embeddings-nearest-neighbors.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")
scatter_mod = _load_module("gridness_vs_accuracy_scatter", "gridness-vs-accuracy-scatter.py")

RING_SWEEP_KEYS = ring_mod.RING_SWEEP_KEYS


def discover_sweep_keys():
    """RING_SWEEP_KEYS members that also have a cached accuracy curve."""
    keys = []
    for key in RING_SWEEP_KEYS:
        acc_path = os.path.join(DATA_DIR, key, f"accuracies_Ring_{key}.npz")
        if os.path.exists(acc_path):
            keys.append(key)
        else:
            print(f"Skipping {key}: missing accuracies_Ring_{key}.npz "
                  f"(run 01_reproduce_ring.py with this key first).")
    return keys


SWEEP_KEYS = discover_sweep_keys()


def load_real_full_context_accuracy(word_list_key):
    """Mirrors gridness-vs-accuracy-scatter-ring.py's function of the same
    name."""
    path = os.path.join(DATA_DIR, word_list_key, f"accuracies_Ring_{word_list_key}.npz")
    acc_data = np.load(path)
    acc_key = "graph_accs" if "graph_accs" in acc_data else "all_accs"
    all_accs = acc_data[acc_key]
    return float(all_accs[:, -1].mean())


def compute_starting_geometry(W_E, tokenizer, word_list_key):
    """(distance correlation, Dirichlet energy) between word_list_key's own
    ring and its words' raw W_E embeddings."""
    words = WORD_LISTS[word_list_key]
    ring = Ring(words)
    adjacency = ring.build_adjacency_matrix()  # already in `words` order

    token_ids = [dot_product_mod.to_single_token(tokenizer, w) for w in words]
    embeddings = W_E[torch.tensor(token_ids, dtype=torch.long)].numpy()

    dc = compute_distance_correlation_graph_embedding(embeddings, adjacency)
    de = compute_dirichlet_energy(embeddings, adjacency)
    return dc, de


def plot_scatter(x_by_key, acc_by_key, keys, out_dir, x_label, title_prefix, filename):
    """Mirrors starting-geometry-vs-accuracy-scatter.py's plot_scatter."""
    xs = np.array([x_by_key[k] for k in keys])
    ys = np.array([acc_by_key[k] for k in keys])
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(keys) > 1 else float("nan")

    key_to_family, n_families = scatter_mod.assign_families(keys)
    cmap = cm.get_cmap("tab10", max(n_families, 1))
    colors = [cmap(key_to_family[k][1]) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(xs, ys, c=colors, s=45, zorder=3)
    for x, y, k in zip(xs, ys, keys):
        ax.annotate(k, (x, y), fontsize=6.5, alpha=0.8,
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
    ax.legend(handles=legend_elements, loc="best", frameon=True, framealpha=1.0,
              edgecolor="gray", fontsize=7, title="Word-set family", title_fontsize=8)

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
            marker=dict(color=color, size=10),
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

    if not SWEEP_KEYS:
        raise RuntimeError(
            "No ring word lists with a cached accuracies_Ring_{key}.npz found -- "
            "run 01_reproduce_ring.py for at least one word list first."
        )
    print(f"Found {len(SWEEP_KEYS)} ring word lists with cached accuracy curves.")

    print("Loading static embedding matrix (W_E)...")
    model = nn_mod.load_embedding_only_model()
    W_E, tokenizer = model.W_E, model.tokenizer

    dc_by_key, de_by_key, acc_by_key = {}, {}, {}
    for key in SWEEP_KEYS:
        dc_by_key[key], de_by_key[key] = compute_starting_geometry(W_E, tokenizer, key)
        acc_by_key[key] = load_real_full_context_accuracy(key)
        print(f"{key}: starting DC = {dc_by_key[key]:.4f}, starting DE = {de_by_key[key]:.4f}, "
              f"real accuracy = {acc_by_key[key]:.4f}")

    plot_scatter(
        dc_by_key, acc_by_key, SWEEP_KEYS, PLOTS_DIR,
        x_label="Distance correlation (static W_E vs. ring, hop-count)",
        title_prefix="Starting-geometry distance correlation vs. real accuracy (ring)",
        filename="starting_geometry_distance_correlation_vs_accuracy_ring",
    )
    plot_scatter(
        de_by_key, acc_by_key, SWEEP_KEYS, PLOTS_DIR,
        x_label="Dirichlet energy (static W_E vs. ring)",
        title_prefix="Starting-geometry Dirichlet energy vs. real accuracy (ring)",
        filename="starting_geometry_dirichlet_energy_vs_accuracy_ring",
    )


if __name__ == "__main__":
    main()
