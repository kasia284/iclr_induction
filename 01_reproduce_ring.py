"""Reproduces the core accuracy-curve, class-mean PCA, and bigram PCA
analyses from 01_reproduce.py (Figs 2 and 6), but for the RING graph
topology (utils.Ring) instead of the square Grid -- see utils.Ring's
docstring for why a ring needs its own random-walk / adjacency logic.

Covers two families: "days_of_week" (7-node ring) and "months_of_year"
(12-node ring), each the natural cyclic order plus its systematic and
random grid-position (here: ring-position) permutations, added to
word_lists.py alongside utils.Ring.

For each word_list_key, the accuracy curve reports two metrics over the
same random-walk sequences: "graph" accuracy (ground truth = adjacency on
the ring as constructed for that key -- the in-context structure) and
"semantic" accuracy (ground truth = adjacency under the word list's
natural/default order, e.g. real weekday or calendar order). These
coincide for the base keys ("days_of_week", "months_of_year") and diverge
for the "_permuted"/"_rand*" keys, where a gap indicates the model is
relying on prior semantic knowledge rather than the in-context graph.

Structurally this mirrors reproduce_alternative_graphs.py (a generic,
self-contained pipeline: accuracy curve + class-mean PCA + bigram PCA for
any object exposing generate_batch/generate_sequence/get_valid_next_words/
build_adjacency_matrix), but parameterized per WORD_LISTS key -- like
01_reproduce.py's configure_for_word_list -- so it can sweep every
permutation of both families in one run instead of using one fixed word
list.

Requires a CUDA GPU with >= 48 GB memory (Llama-3.1-8B via TransformerLens).
Results cached to results/reproduce_ring/data/{key}/, plots to
results/reproduce_ring/plots/{key}/. Safe to re-run: skips the model pass
entirely for any key that's already fully cached.
"""
import os
import re
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import plotly.graph_objects as go
import tqdm

from utils import (
    Ring, build_word_to_color,
    LAYER, set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
    set_square_limits,
)
from word_lists import WORD_LISTS

RING_SWEEP_KEYS = [
    "days_of_week", "days_of_week_permuted",
    "days_of_week_rand1", "days_of_week_rand2", "days_of_week_rand3",
    "months_of_year", "months_of_year_permuted",
    "months_of_year_rand1", "months_of_year_rand2", "months_of_year_rand3",
]

N_LOOKBACK = 50
SEQ_LEN = 1400  # matches 01_reproduce.py's context length
DATA_DIR_ROOT = "results/reproduce_ring/data"
PLOTS_DIR_ROOT = "results/reproduce_ring/plots"


def get_base_word_list_key(word_list_key):
    """Strip a ring word list's permutation suffix to recover the natural
    cyclic-order list it's a reordering of, e.g. "days_of_week_permuted" ->
    "days_of_week", "months_of_year_rand2" -> "months_of_year". For a
    base key itself this is a no-op."""
    m = re.match(r"^(.*?)(?:_permuted|_rand\d+)$", word_list_key)
    return m.group(1) if m else word_list_key


# ── Fig 2 left analog: accuracy curve ───────────────────────────────────────

def plot_accuracy_curve(graph_accs, semantic_accs, word_list_key, plots_dir):
    """Average accuracy across len(words) sequences with uniform starting
    positions (one per ring word, same convention as 01_reproduce.py).

    Plots two curves against the same random-walk sequences:
    - "graph" accuracy: ground truth = adjacency on the ring as actually
      constructed for this word_list_key (the in-context structure).
    - "semantic" accuracy: ground truth = adjacency under the word list's
      natural/default order (e.g. real weekday or calendar order), which
      only differs from graph accuracy for the "_permuted"/"_rand*" keys.
      A gap between the two indicates the model is falling back on prior
      semantic knowledge rather than the in-context graph structure.
    """
    x_vals = np.arange(1, graph_accs.shape[1] + 1)
    title = f"Accuracy vs sequence length ({word_list_key}, ring)"

    curves = [
        ("Graph accuracy", graph_accs, "black", "gray"),
        ("Semantic accuracy", semantic_accs, "tab:orange", "moccasin"),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    for label, accs, line_color, band_color in curves:
        mean = accs.mean(axis=0)
        std = accs.std(axis=0)
        ax.plot(x_vals, mean, color=line_color, linewidth=1.0, label=label)
        ax.fill_between(x_vals, mean - std, mean + std,
                         alpha=0.15, color=band_color, edgecolor="none")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=8)
    save_figure(fig, plots_dir, "accuracy_curve.pdf")

    pfig = go.Figure()
    x = x_vals.tolist()
    plotly_colors = {
        "Graph accuracy": ("black", "rgba(128,128,128,0.2)"),
        "Semantic accuracy": ("orange", "rgba(255,165,0,0.2)"),
    }
    for label, accs, _, _ in curves:
        mean = accs.mean(axis=0)
        std = accs.std(axis=0)
        line_color, fill_color = plotly_colors[label]
        pfig.add_trace(go.Scatter(
            x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip", legendgroup=label,
        ))
        pfig.add_trace(go.Scatter(
            x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill_color,
            showlegend=False, hoverinfo="skip", legendgroup=label,
        ))
        pfig.add_trace(go.Scatter(
            x=x, y=mean.tolist(), mode="lines",
            line=dict(color=line_color, width=2), name=label, legendgroup=label,
            hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>",
        ))
    pfig.update_layout(**plotly_line_layout(title, "Sequence length", "Accuracy"))
    save_plotly(pfig, plots_dir, "accuracy_curve.html")
    print(f"Saved {word_list_key}/accuracy_curve")


# ── Fig 2 right analog: class-mean PCA ──────────────────────────────────────

def plot_class_mean_pca(ring, class_means, pca_dirs, words, word_to_color, word_list_key, plots_dir,
                         explained_variance=None):
    """Scatter of per-word centroids with ring edges (2D only -- a ring
    has no notion of a 3D "face" the way a grid cell does)."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T

    fig, ax = plt.subplots(figsize=(5, 5))
    A = ring.build_adjacency_matrix()
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )
    for i, word in enumerate(words):
        ax.scatter(
            projected[i, 0].item(), projected[i, 1].item(),
            color=word_to_color[word], s=120, marker="*",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax.annotate(
            word, (projected[i, 0].item(), projected[i, 1].item()),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    if explained_variance is not None:
        ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    set_square_limits(ax, projected[:, 0], projected[:, 1])
    ax.set_aspect("equal")
    ax.set_title(f"PCA of per-node mean activations ({word_list_key}, ring)", fontsize=10)
    save_figure(fig, plots_dir, "pca_class_means.pdf")

    pfig = go.Figure(data=plotly_pca_traces(projected, ring, words=words, word_to_color=word_to_color))
    pfig.update_layout(**plotly_pca_layout(f"PCA of per-node mean activations ({word_list_key}, ring)"))
    save_plotly(pfig, plots_dir, "pca_class_means.html")
    print(f"Saved {word_list_key}/pca_class_means")


# ── Fig 6 analog: bigram PCA ─────────────────────────────────────────────────

def plot_bigram_pca(ring, sequence, activations, class_means, pca_dirs, words, word_to_color,
                     word_list_key, plots_dir):
    """Individual activations projected onto the class-mean PCA directions.
    Fill = current token, border = previous token -- same rendering as
    01_reproduce.py's Fig 6 (_draw_bigram_scatter / plot_bigram_pca)."""
    tail = sequence[-N_LOOKBACK:]
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T
    projected_means = (class_means - class_means_mean) @ pca_dirs.T

    fig, ax = plt.subplots(figsize=(8, 8))
    A = ring.build_adjacency_matrix()
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if A[i, j]:
                ax.plot(
                    [projected_means[i, 0].item(), projected_means[j, 0].item()],
                    [projected_means[i, 1].item(), projected_means[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )
    for idx in range(1, len(tail)):
        cur_word, prev_word = tail[idx], tail[idx - 1]
        ax.scatter(
            projected_all[idx, 0].item(), projected_all[idx, 1].item(),
            c=word_to_color[cur_word], edgecolors=word_to_color[prev_word],
            linewidths=1.0, s=25, alpha=1.0, zorder=3,
        )
    for i, word in enumerate(words):
        ax.scatter(
            projected_means[i, 0].item(), projected_means[i, 1].item(),
            color=word_to_color[word], s=120, marker="*",
            edgecolors="black", linewidths=1.0, zorder=5,
        )
        ax.annotate(
            word, (projected_means[i, 0].item(), projected_means[i, 1].item()),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", markeredgewidth=1.5,
               label="Fill = current token"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markersize=8, markeredgecolor="gray", markeredgewidth=1.5,
               label="Border = previous token"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markersize=12, markeredgecolor="black", markeredgewidth=0.8,
               label="Token centroid"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    set_square_limits(ax, projected_all[:, 0], projected_all[:, 1])
    ax.set_aspect("equal")
    ax.set_title(f"PCA of individual activations by bigram ({word_list_key}, ring)", fontsize=10)
    save_figure(fig, plots_dir, "bigram_pca.pdf")

    pfig = go.Figure(data=plotly_pca_traces(projected_means, ring, words=words, word_to_color=word_to_color))
    for word in words:
        idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
        if not idxs:
            continue
        pfig.add_trace(go.Scatter(
            x=[projected_all[idx, 0].item() for idx in idxs],
            y=[projected_all[idx, 1].item() for idx in idxs],
            mode="markers",
            marker=dict(
                size=8, color=word_to_color[word],
                line=dict(width=2, color=[word_to_color[tail[idx - 1]] for idx in idxs]),
            ),
            customdata=[[tail[idx - 1]] for idx in idxs],
            hovertemplate="current: <b>" + word + "</b><br>previous: <b>%{customdata[0]}</b><extra></extra>",
            hoverlabel=dict(bgcolor=word_to_color[word], font=dict(color="black")),
            name=word, showlegend=False,
        ))
    pfig.update_layout(**plotly_pca_layout(f"PCA of individual activations by bigram ({word_list_key}, ring)"))
    pfig.update_layout(width=1000, height=1000)
    pfig.add_annotation(
        text="● Fill = current token<br>● Border = previous token<br>★ Token centroid",
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, font=dict(size=12),
        align="left", bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, borderpad=6,
    )
    save_plotly(pfig, plots_dir, "bigram_pca.html")
    print(f"Saved {word_list_key}/bigram_pca")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_experiment_pipeline(word_list_key, model):
    words = WORD_LISTS[word_list_key]
    word_to_color = build_word_to_color(words)
    ring = Ring(words)
    n_sequences = len(words)

    base_key = get_base_word_list_key(word_list_key)
    canonical_ring = Ring(WORD_LISTS[base_key])

    data_dir = os.path.join(DATA_DIR_ROOT, word_list_key)
    plots_dir = os.path.join(PLOTS_DIR_ROOT, word_list_key)
    acc_path = os.path.join(data_dir, f"accuracies_Ring_{word_list_key}.npz")
    pca_path = os.path.join(data_dir, f"pca_Ring_{word_list_key}.npz")
    seq_path = os.path.join(data_dir, f"sequence_Ring_{word_list_key}.json")

    os.makedirs(data_dir, exist_ok=True)

    # ── accuracies (graph vs. semantic ground truth) ────────────────────────
    acc_cache = np.load(acc_path) if os.path.exists(acc_path) else None
    if acc_cache is not None and "graph_accs" in acc_cache and "semantic_accs" in acc_cache:
        print(f"\n=== {word_list_key}: accuracies (cached) ===")
        graph_accs = acc_cache["graph_accs"]
        semantic_accs = acc_cache["semantic_accs"]
    else:
        print(f"\n=== {word_list_key}: accuracies ===")
        set_seed(42)
        sequences = ring.generate_batch(SEQ_LEN, n_sequences)
        graph_accs, semantic_accs = [], []
        for seq in tqdm.tqdm(sequences, desc=f"{word_list_key}: accuracies"):
            graph_accs.append(get_model_accuracies(model, ring, seq))
            semantic_accs.append(get_model_accuracies(model, canonical_ring, seq))
        graph_accs = np.array(graph_accs)
        semantic_accs = np.array(semantic_accs)
        np.savez(acc_path, graph_accs=graph_accs, semantic_accs=semantic_accs)

    # ── activations / PCA ────────────────────────────────────────────────────
    if os.path.exists(pca_path) and os.path.exists(seq_path):
        print(f"=== {word_list_key}: PCA (cached) ===")
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        explained_variance = pca_data["explained_variance"]
        with open(seq_path) as f:
            sequence = json.load(f)
    else:
        print(f"=== {word_list_key}: PCA ===")
        set_seed(42)
        sequence = ring.generate_sequence(SEQ_LEN)
        activations_t = get_activations(model, sequence, LAYER, N_LOOKBACK)
        class_means_t = compute_class_means(activations_t, sequence, words, N_LOOKBACK)
        pca_dirs_t, explained_variance_t = compute_pca_directions(class_means_t, top_n=2)

        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()
        explained_variance = explained_variance_t.cpu().numpy() if hasattr(explained_variance_t, "cpu") \
            else np.asarray(explained_variance_t)

        np.savez(pca_path, activations=activations, class_means=class_means,
                 pca_dirs=pca_dirs, explained_variance=explained_variance)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)

    plot_accuracy_curve(graph_accs, semantic_accs, word_list_key, plots_dir)
    plot_class_mean_pca(ring, class_means, pca_dirs, words, word_to_color, word_list_key, plots_dir,
                         explained_variance=explained_variance)
    plot_bigram_pca(ring, sequence, activations, class_means, pca_dirs, words, word_to_color,
                     word_list_key, plots_dir)


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    model = load_model()
    for word_list_key in RING_SWEEP_KEYS:
        run_experiment_pipeline(word_list_key, model)


if __name__ == "__main__":
    main()
