import os
import json
import math
import functools
import gc

import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

from utils import (
    LAYER, build_word_to_color,
    Grid, set_seed, load_model,
    compute_class_means_batch, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
    set_square_limits, compute_and_plot_similarity_panels,
    compute_dirichlet_energy, compute_distance_correlation,
    draw_class_mean_pca_on_ax, add_pca_grid_faces_3d,
    draw_bigram_scatter_on_ax, make_bigram_legend,
)
from utils import plot_attention_heatmaps
from word_lists import WORD_LISTS

WORD_LIST_KEY = "synonyms_happy_permuted"
WORDS = WORD_LISTS[WORD_LIST_KEY]
WORD_TO_COLOR = build_word_to_color(WORDS)
GRID_ROWS, GRID_COLS = 4, 4

DATA_DIR = f"results/reproduce/data/{WORD_LIST_KEY}"
PLOTS_DIR = f"results/reproduce/plots/{WORD_LIST_KEY}"


def configure_for_word_list(word_list_key):
    """Reassign the word-list-dependent globals (WORDS, WORD_TO_COLOR,
    GRID_ROWS, GRID_COLS, N_SEQUENCES, DATA_DIR, PLOTS_DIR) so main() can be
    run for a different WORD_LISTS vocabulary without editing the
    module-level constants by hand. Grid dims are derived as sqrt(len(words))
    since every current word list is a perfect square (16, 36, or 64 words)."""
    global WORD_LIST_KEY, WORDS, WORD_TO_COLOR, GRID_ROWS, GRID_COLS, N_SEQUENCES, DATA_DIR, PLOTS_DIR
    WORD_LIST_KEY = word_list_key
    WORDS = WORD_LISTS[WORD_LIST_KEY]
    WORD_TO_COLOR = build_word_to_color(WORDS)
    side = math.isqrt(len(WORDS))
    if side * side != len(WORDS):
        raise ValueError(
            f"WORD_LISTS[{word_list_key!r}] has {len(WORDS)} words, "
            f"which is not a perfect square, so it can't form a square grid."
        )
    GRID_ROWS, GRID_COLS = side, side
    N_SEQUENCES = len(WORDS)
    DATA_DIR = f"results/reproduce/data/{WORD_LIST_KEY}"
    PLOTS_DIR = f"results/reproduce/plots/{WORD_LIST_KEY}"


N_LOOKBACK = 50
SEQ_LEN = 1400  # long random walk, matching the original paper's context length
# LAYERS = list(range(0, 32, 2))  # every 2nd layer across Llama-3.1-8B's 32 layers
LAYERS = ['0', '6', '13', '20', '26', '31']
N_SEQUENCES = len(WORDS)  # one sequence per starting word, for full grid coverage
N_ITERATIONS = 1  # number of iterative forward passes (see run_iterative_forward_passes)


# ── Fig 2 Left: Accuracy curve ────────────────────────────────────────────────

def plot_accuracy_curve(all_accs, suffix=""):
    """Average accuracy across 16 sequences with uniform starting positions."""
    # Smoothing disabled: it shifted the visual rise ~window/2 steps left of
    # where it actually occurs (see np.convolve 'valid'-mode centering), which
    # distorted the log-scale x-axis exactly where induction kicks in.
    # mean = smooth(all_accs.mean(axis=0))
    # std = smooth(all_accs.std(axis=0))
    mean = all_accs.mean(axis=0)
    std = all_accs.std(axis=0)
    # Context length is 1-indexed: mean[i] is the accuracy after i+1 words of context.
    x_vals = np.arange(1, len(mean) + 1)
    title = f"Accuracy vs sequence length ({WORD_LIST_KEY})"

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(x_vals, mean, color="black", linewidth=1.0, label="Mean accuracy")
    ax.fill_between(x_vals, mean - std, mean + std,
                    alpha=0.15, color="gray", edgecolor="none", label="$\\pm$1 std")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray")

    save_figure(fig, PLOTS_DIR, f"accuracy_curve{suffix}.pdf")
    print(f"Saved accuracy_curve{suffix}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure()
    x = x_vals.tolist()
    pfig.add_trace(go.Scatter(
        x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(128,128,128,0.2)",
        showlegend=False, hoverinfo="skip",
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=mean.tolist(), mode="lines",
        line=dict(color="black", width=2), name="Mean accuracy",
        hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>",
    ))
    pfig.update_layout(**plotly_line_layout(
        title, "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, PLOTS_DIR, f"accuracy_curve{suffix}.html")


def load_cached_accuracies(word_list_key, graph_type="Grid"):
    """Load a previously-cached accuracies_*.npz for one word list (written
    by main()'s "Accuracy data" step)."""
    path = os.path.join(
        f"results/reproduce/data/{word_list_key}",
        f"accuracies_{graph_type}_{word_list_key}.npz",
    )
    return np.load(path)["all_accs"]


def plot_accuracy_comparison(accs_by_key, out_dir, filename="accuracy_curve_comparison"):
    """Overlay mean accuracy curves for several word lists on one plot, so
    e.g. a word list and its permuted variant can be compared directly."""
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    for (key, all_accs), color in zip(accs_by_key.items(), colors):
        mean = all_accs.mean(axis=0)
        std = all_accs.std(axis=0)
        ax.plot(mean, color=color, linewidth=1.0, label=key)
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        alpha=0.15, color=color, edgecolor="none")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title("Accuracy vs sequence length", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=8)

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {filename}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure()
    for (key, all_accs), color in zip(accs_by_key.items(), colors):
        mean = all_accs.mean(axis=0)
        std = all_accs.std(axis=0)
        x = list(range(len(mean)))
        rgba = "rgba({},{},{},0.15)".format(
            int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        pfig.add_trace(go.Scatter(
            x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        pfig.add_trace(go.Scatter(
            x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=rgba,
            showlegend=False, hoverinfo="skip",
        ))
        pfig.add_trace(go.Scatter(
            x=x, y=mean.tolist(), mode="lines",
            line=dict(color=color, width=2), name=key,
            hovertemplate=f"{key}<br>pos: " + "%{x}<br>accuracy: %{y:.3f}<extra></extra>",
        ))
    pfig.update_layout(**plotly_line_layout(
        "Accuracy vs sequence length", "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, out_dir, f"{filename}.html")



def plot_class_mean_pca(grid, class_means, pca_dirs, suffix="", title=None, ax=None, explained_variance=None):
    """Scatter of 16 class-mean centroids with grid edges (Supports 2D and 3D).

    ax: matplotlib Axes, optional. Draw onto this existing axes instead of
    creating (and saving) a new standalone figure. Used to compose several
    layers into one row (see plot_pca_across_layers).
    explained_variance: array-like, optional. Fraction of variance explained
    by each principal component (e.g. from compute_pca_directions), shown on
    the axis labels as a percentage.
    """
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected = (class_means - class_means_mean) @ pca_dirs.T  # [16, num_components]
    num_dims = projected.shape[1]
    is_3d = (num_dims == 3)

    # Setup the figure canvas based on dimensionality, unless drawing onto
    # a caller-provided axes.
    standalone = ax is None
    if standalone:
        if is_3d:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(5, 5))

    # Grey grid-cell faces (3D only) first, so they stay behind the grid
    # edges (gray dashed) + star centroid markers + word labels drawn next.
    if is_3d:
        add_pca_grid_faces_3d(ax, grid, WORDS, projected)
    draw_class_mean_pca_on_ax(ax, grid, projected, words=WORDS, word_to_color=WORD_TO_COLOR,
                               marker_size=120, label_fontsize=8, label_offset=(5, 5), is_3d=is_3d)

    # Axis Labels & Aspect Ratio
    if explained_variance is not None:
        lbl_pc1 = f"PC1 ({explained_variance[0]*100:.1f}%)"
        lbl_pc2 = f"PC2 ({explained_variance[1]*100:.1f}%)"
        lbl_pc3 = f"PC3 ({explained_variance[2]*100:.1f}%)" if is_3d else ""
    else:
        lbl_pc1, lbl_pc2, lbl_pc3 = "PC1", "PC2", "PC3"

    ax.set_xlabel(lbl_pc1)
    ax.set_ylabel(lbl_pc2)

    if is_3d:
        ax.set_zlabel(lbl_pc3)
        ax.set_box_aspect((1, 1, 1))  # Sets equal aspect ratio for 3D bounding box
    else:
        set_square_limits(ax, projected[:, 0], projected[:, 1])
        ax.set_aspect("equal")

    ax.set_title(title or f"{'3D ' if is_3d else ''}PCA of per-node mean activations", fontsize=10)

    if not standalone:
        return

    # Save files with distinct names depending on dimensions and graph type
    dim_suffix = "3d" if is_3d else "2d"
    save_figure(fig, PLOTS_DIR, f"pca_class_means{suffix}_{dim_suffix}.pdf")
    print(f"Saved pca_class_means{suffix}_{dim_suffix}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    # Note: Ensure your `plotly_pca_traces` helper is capable of reading
    # the second dimension of `projected` to return `go.Scatter3d` traces.
    pfig = go.Figure(data=plotly_pca_traces(projected, grid, words=WORDS, word_to_color=WORD_TO_COLOR))
    pfig.update_layout(**plotly_pca_layout(f"{'3D ' if is_3d else ''}PCA of per-node mean activations"))
    save_plotly(pfig, PLOTS_DIR, f"pca_class_means{suffix}_{dim_suffix}.html")


def save_pca_layer_plotly_3d(grid, class_means, pca_dirs, layer, suffix="", explained_variance=None):
    """Interactive, rotatable 3D PCA scatter for a single layer."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T  # [16, 3]

    A = grid.build_adjacency_matrix()
    edge_x, edge_y, edge_z = [], [], []
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                edge_x += [projected[i, 0].item(), projected[j, 0].item(), None]
                edge_y += [projected[i, 1].item(), projected[j, 1].item(), None]
                edge_z += [projected[i, 2].item(), projected[j, 2].item(), None]

    # Grey face for each grid cell (two triangles per quad), same cells as
    # the matplotlib Poly3DCollection version.
    word_to_idx = {w: i for i, w in enumerate(WORDS)}
    is_torus = type(grid).__name__ == "Torus"
    r_range = range(grid.rows) if is_torus else range(grid.rows - 1)
    c_range = range(grid.cols) if is_torus else range(grid.cols - 1)
    tri_i, tri_j, tri_k = [], [], []
    for r in r_range:
        for c in c_range:
            r2, c2 = (r + 1) % grid.rows, (c + 1) % grid.cols
            corners = [grid.grid[r][c], grid.grid[r][c2], grid.grid[r2][c2], grid.grid[r2][c]]
            idxs = [word_to_idx[w] for w in corners]
            tri_i += [idxs[0], idxs[0]]
            tri_j += [idxs[1], idxs[2]]
            tri_k += [idxs[2], idxs[3]]

    traces = [go.Mesh3d(
        x=projected[:, 0].tolist(), y=projected[:, 1].tolist(), z=projected[:, 2].tolist(),
        i=tri_i, j=tri_j, k=tri_k,
        color="grey", opacity=0.4, flatshading=True,
        hoverinfo="skip", showlegend=False,
    )]

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(color="dimgray", width=2.5), opacity=0.8,
        hoverinfo="skip", showlegend=False,
    ))

    for i, word in enumerate(WORDS):
        traces.append(go.Scatter3d(
            x=[projected[i, 0].item()], y=[projected[i, 1].item()], z=[projected[i, 2].item()],
            mode="markers+text",
            marker=dict(size=8, symbol="diamond", color=WORD_TO_COLOR[word],
                        line=dict(width=1, color="black")),
            text=word, textposition="top center", textfont=dict(size=10),
            hovertemplate=f"<b>{word}</b><extra></extra>",
            showlegend=False,
        ))

    if explained_variance is not None:
        xaxis_title = f"PC1 ({explained_variance[0]*100:.1f}%)"
        yaxis_title = f"PC2 ({explained_variance[1]*100:.1f}%)"
        zaxis_title = f"PC3 ({explained_variance[2]*100:.1f}%)"
    else:
        xaxis_title, yaxis_title, zaxis_title = "PC1", "PC2", "PC3"

    pfig = go.Figure(data=traces)
    pfig.update_layout(
        title=f"Layer {layer} -- PCA of per-node mean activations ({WORD_LIST_KEY})",
        scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=zaxis_title, aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40),
        width=700, height=700,
    )

    out_name = f"pca_layer_{layer}{suffix}_3d"
    save_plotly(pfig, PLOTS_DIR, f"{out_name}.html")


def plot_pca_across_layers(grid, layer_class_means, layers, suffix="", top_n=2):
    """One figure with a class-mean PCA panel per layer, showing how the
    grid geometry emerges across the model's depth. Supports 2D and 3D.

    For the 3D case, also saves each layer as its own interactive, rotatable
    Plotly HTML (the static matplotlib panels can't be rotated)."""
    is_3d = (top_n == 3)
    subplot_kw = {"projection": "3d"} if is_3d else {}
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), subplot_kw=subplot_kw)
    for ax, layer in zip(axes, layers):
        class_means = layer_class_means[layer]
        pca_dirs_t, var = compute_pca_directions(torch.tensor(class_means), top_n=top_n)
        pca_dirs = pca_dirs_t.numpy()
        plot_class_mean_pca(grid, class_means, pca_dirs, title=f"Layer {layer}", ax=ax, explained_variance=var)
        if is_3d:
            save_pca_layer_plotly_3d(grid, class_means, pca_dirs, layer, suffix=suffix, explained_variance=var)

    fig.suptitle(f"{'3D ' if is_3d else ''}PCA of per-node mean activations across layers ({WORD_LIST_KEY})")
    dim_suffix = "3d" if is_3d else "2d"
    out_name = f"pca_across_layers{suffix}_{dim_suffix}"
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")


def plot_pca_across_layers_grid(grid, layer_class_means, layers, suffix="",
                                 n_rows=4, n_cols=8, suptitle=None):
    """Grid of per-layer class-mean PCA panels (2D only), one panel per
    layer, arranged in n_rows x n_cols. Defaults to 4x8=32 panels, matching
    Llama-3.1-8B's 32 layers, so the whole depth of the model can be seen at
    once instead of the handful of layers in LAYERS/plot_pca_across_layers."""
    if len(layers) > n_rows * n_cols:
        raise ValueError(
            f"{len(layers)} layers don't fit in a {n_rows}x{n_cols} grid "
            f"({n_rows * n_cols} panels)"
        )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes_flat = axes.flatten()
    for ax, layer in zip(axes_flat, layers):
        class_means = layer_class_means[layer]
        pca_dirs_t, var = compute_pca_directions(torch.tensor(class_means), top_n=2)
        pca_dirs = pca_dirs_t.numpy()
        plot_class_mean_pca(grid, class_means, pca_dirs, title=f"Layer {layer}",
                             ax=ax, explained_variance=var)
    for ax in axes_flat[len(layers):]:
        ax.axis("off")

    fig.suptitle(suptitle or "PCA of per-node mean activations across all layers")
    out_name = f"pca_all_layers_grid{suffix}_2d"
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")


def get_grid_coords(grid, words):
    """[n_words, 2] (row, col) per word, in `words` order -- the "state
    space" coordinates G_i used by Dirichlet energy / distance correlation."""
    return np.array([[grid.word_to_row[w], grid.word_to_col[w]] for w in words])


def plot_de_dc_across_layers(layers_by_series, adjacency, grid_coords, suffix="", title=None):
    """Normalized Dirichlet energy and distance correlation (Eqs. 1-3) vs.
    layer, one line per series in `layers_by_series`
    ({series_label: {layer (int): class_means [n_words, d_model]}}) -- e.g.
    one line per iteration of run_iterative_forward_passes."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for label, layer_class_means in layers_by_series.items():
        layers_sorted = sorted(layer_class_means)
        de_values = [compute_dirichlet_energy(layer_class_means[l], adjacency) for l in layers_sorted]
        dc_values = [compute_distance_correlation(layer_class_means[l], grid_coords) for l in layers_sorted]
        axes[0].plot(layers_sorted, de_values, marker="o", markersize=3, linewidth=1, label=label)
        axes[1].plot(layers_sorted, dc_values, marker="o", markersize=3, linewidth=1, label=label)

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Normalized Dirichlet energy")
    axes[0].set_title("Dirichlet energy vs. layer", fontsize=10)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Distance correlation")
    axes[1].set_title("Distance correlation vs. layer", fontsize=10)
    for ax in axes:
        ax.legend(fontsize=7, frameon=True, framealpha=1.0, edgecolor="gray")

    fig.suptitle(title or f"Dirichlet energy & distance correlation ({WORD_LIST_KEY})")
    out_name = f"de_dc_across_layers{suffix}"
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")


# ── Fig 6: Bigram PCA ─────────────────────────────────────────────────────────

def plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs, suffix="", explained_variance=None):
    """Individual activations colored by (current token, previous token)."""
    tail = sequence[-N_LOOKBACK:]
    # Center on the class-means' mean, since that's the mean pca_dirs was
    # actually derived from (see compute_pca_directions) -- individual
    # bigram activations must be projected relative to that same origin.
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T    # [N_LOOKBACK, 2]
    projected_means = (class_means - class_means_mean) @ pca_dirs.T  # [16, 2]

    # ── Main bigram plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_bigram_scatter_on_ax(ax, projected_all, projected_means, tail, grid,
                               words=WORDS, word_to_color=WORD_TO_COLOR,
                               marker_size=120, label_fontsize=7,
                               individual_size=25, individual_linewidth=1.0, individual_alpha=1.0)
    make_bigram_legend(ax, loc="upper left", fontsize=8)
    if explained_variance is not None:
        ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ax.set_title("PCA of individual activations, labeled by bigram", fontsize=10)
    set_square_limits(ax, projected_all[:, 0], projected_all[:, 1])
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, f"bigram_pca{suffix}.pdf")
    print(f"Saved bigram_pca{suffix}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected_means, grid, words=WORDS, word_to_color=WORD_TO_COLOR))

    # Individual bigram points — group by current word for legend toggle
    for word in WORDS:
        idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
        if not idxs:
            continue
        pfig.add_trace(go.Scatter(
            x=[projected_all[idx, 0].item() for idx in idxs],
            y=[projected_all[idx, 1].item() for idx in idxs],
            mode="markers",
            marker=dict(
                size=8, color=WORD_TO_COLOR[word],
                line=dict(
                    width=2,
                    color=[WORD_TO_COLOR[tail[idx - 1]] for idx in idxs],
                ),
            ),
            customdata=[[tail[idx - 1]] for idx in idxs],
            hovertemplate=(
                "current: <b>" + word + "</b><br>"
                "previous: <b>%{customdata[0]}</b>"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")),
            name=word, showlegend=False,
        ))

    pfig.update_layout(**plotly_pca_layout(
        "PCA of individual activations, labeled by bigram"))
    pfig.update_layout(width=1000, height=1000)

    # Legend annotation (plotly has no custom legend handles like matplotlib)
    pfig.add_annotation(
        text=(
            "● Fill = current token<br>"
            "● Border = previous token<br>"
            "★ Token centroid"
        ),
        xref="paper", yref="paper", x=0.02, y=0.98,
        showarrow=False, font=dict(size=12),
        align="left", bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray", borderwidth=1, borderpad=6,
    )

    save_plotly(pfig, PLOTS_DIR, f"bigram_pca{suffix}.html")


# ── Single-token-safe tokenization ─────────────────────────────────────────────
#
# utils.tokenize_sequence joins words into a string ("apple bird ...") and
# re-tokenizes it, silently assuming every word collapses to exactly one
# token once space-prefixed. That's true for ordinary words (" apple" is one
# token) but false for e.g. two-digit numbers: " 11" is *two* tokens (a
# leading-space token + "11"), since Llama's vocab has no merged " 11" entry.
# That desyncs token positions from word positions throughout the accuracy/
# activation pipeline. The functions below sidestep this by looking up each
# word's single-token id directly (trying " word" first, falling back to the
# bare word) and building token sequences from that lookup table, guaranteeing
# exactly one token per word regardless of which WORD_LISTS vocabulary is active.

def build_word_token_ids(model, words):
    """Look up each word's single Llama token id, trying a leading space
    first (how it appears mid-sequence, e.g. " apple") and falling back to
    the bare word (e.g. "11", which is only single-token without a space).
    Raises a clear error if neither form is a single token."""
    word_to_id = {}
    for w in words:
        try:
            word_to_id[w] = model.to_single_token(f" {w}")
        except AssertionError:
            try:
                word_to_id[w] = model.to_single_token(w)
            except AssertionError:
                raise ValueError(
                    f"'{w}' is not a single token (space-prefixed or bare) "
                    f"in this model's vocabulary."
                )
    return word_to_id


def tokenize_sequence_by_id(model, sequence, word_to_id):
    """Tokenize via direct per-word token-id lookup (BOS + one id per word)."""
    ids = [model.tokenizer.bos_token_id] + [word_to_id[w] for w in sequence]
    return torch.tensor([ids], dtype=torch.long)


def get_model_accuracies_by_id(model, grid, sequence, word_to_id, fwd_hooks=[], include_self=False):
    """Like utils.get_model_accuracies, but tokenizes via word_to_id so every
    word is guaranteed exactly one token. include_self=True also counts
    probability mass on the current token itself as correct, not just on
    its grid neighbors."""
    tokens = tokenize_sequence_by_id(model, sequence, word_to_id)
    logits = model.run_with_hooks(tokens.to(model.cfg.device), fwd_hooks=fwd_hooks)
    probs = torch.softmax(logits, dim=-1)
    probs = probs[0, 1:, :]  # remove BOS position

    accuracies = []
    for i in range(len(sequence)):
        target_words = grid.get_valid_next_words(sequence[i])
        if include_self:
            target_words = target_words + [sequence[i]]
        token_ids = torch.tensor([word_to_id[w] for w in target_words])
        accuracies.append(probs[i, token_ids].sum().item())
    return accuracies


def get_activations_by_id(model, sequence, word_to_id, layer, n_lookback, fwd_hooks=[]):
    """Like utils.get_activations, but tokenizes via word_to_id so every word
    is guaranteed exactly one token."""
    tokens = tokenize_sequence_by_id(model, sequence, word_to_id)
    names_filter = [f"blocks.{layer}.hook_resid_pre"]
    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=names_filter)
    acts = cache[f"blocks.{layer}.hook_resid_pre"]  # [1, seq+1, d_model]
    acts = acts[0, 1:, :]  # remove BOS
    return acts[-n_lookback:, :]


def tokenize_sequences_by_id_batch(model, sequences, word_to_id):
    """Batched version of tokenize_sequence_by_id: `sequences` is a list of
    equal-length word sequences (e.g. one per starting grid word). Returns
    [batch, seq+1] (BOS + one id per word, per sequence)."""
    ids = [
        [model.tokenizer.bos_token_id] + [word_to_id[w] for w in seq]
        for seq in sequences
    ]
    return torch.tensor(ids, dtype=torch.long)


def get_activations_by_id_batch(model, sequences, word_to_id, layer, n_lookback, fwd_hooks=[]):
    """Batched version of get_activations_by_id, over a batch of equal-
    length sequences. Returns [batch, n_lookback, d_model]."""
    tokens = tokenize_sequences_by_id_batch(model, sequences, word_to_id)
    names_filter = [f"blocks.{layer}.hook_resid_pre"]
    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=names_filter)
    acts = cache[f"blocks.{layer}.hook_resid_pre"]  # [batch, seq+1, d_model]
    acts = acts[:, 1:, :]  # remove BOS
    return acts[:, -n_lookback:, :]


# ── Iterative forward passes ───────────────────────────────────────────────────
#
# Idea: instead of only looking at activations within a single forward pass,
# take the residual stream that comes out after the model's last layer and
# feed it back in as the input to the first layer, running the whole 32-layer
# stack again on its own output. Each such pass is one "iteration"; iteration
# 0 is the ordinary forward pass on the real tokens.

def get_resid_pre_all_layers(model, sequences, word_to_id, n_lookback, fwd_hooks=[]):
    """Batched forward pass over `sequences` (equal-length, e.g. one per
    starting grid word), caching hook_resid_pre at every layer plus the
    final layer's hook_resid_post (the residual stream that would normally
    be handed to ln_final/unembed).

    Returns:
      layer_acts: dict {layer (int): [batch, n_lookback, d_model]} tail
        activations (BOS removed).
      final_resid: [batch, seq+1, d_model] resid_post of the last layer,
        including the BOS position -- same shape as blocks.0.hook_resid_pre,
        so it can be fed straight back in via get_resid_pre_all_layers_with_override.
    """
    n_layers = model.cfg.n_layers
    tokens = tokenize_sequences_by_id_batch(model, sequences, word_to_id)
    names_filter = [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)]
    names_filter.append(f"blocks.{n_layers - 1}.hook_resid_post")
    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=names_filter)
    layer_acts = {
        l: cache[f"blocks.{l}.hook_resid_pre"][:, 1:, :][:, -n_lookback:, :]
        for l in range(n_layers)
    }
    final_resid = cache[f"blocks.{n_layers - 1}.hook_resid_post"].clone()
    return layer_acts, final_resid


def _override_resid_hook(resid_pre, hook, new_resid):
    """Replace hook_resid_pre of layer 0 wholesale with a previous
    iteration's final-layer output, so the stack runs on self-generated
    input instead of the token embeddings."""
    return new_resid


def get_resid_pre_all_layers_with_override(model, sequences, word_to_id, n_lookback,
                                            override_resid, fwd_hooks=[]):
    """Like get_resid_pre_all_layers, but overrides blocks.0.hook_resid_pre
    with `override_resid` before running the stack. The real tokens are
    still used to drive tokenization/shapes/causal masking, but their
    embeddings are discarded the moment the override hook fires."""
    override_hook = (
        "blocks.0.hook_resid_pre",
        functools.partial(_override_resid_hook, new_resid=override_resid),
    )
    return get_resid_pre_all_layers(
        model, sequences, word_to_id, n_lookback,
        fwd_hooks=list(fwd_hooks) + [override_hook],
    )


def run_iterative_forward_passes(model, sequences, word_to_id, n_lookback, n_iterations,
                                  fwd_hooks=[]):
    """Run n_iterations forward passes over a batch of sequences (one per
    starting grid word). Iteration 0 is the ordinary forward pass on the
    real tokens. For each subsequent iteration, the previous iteration's
    final-layer residual stream (hook_resid_post of the last block) is
    injected as hook_resid_pre of the first block, so the model processes
    its own last-layer output as if it were a fresh input.

    Returns a list of length n_iterations, each element a dict
    {layer (int): class_means array [n_words, d_model]} -- one entry per
    layer per iteration, suitable for plot_pca_across_layers_grid. Class
    means are averaged over the whole batch (compute_class_means_batch), so
    a word missing from any single sequence's tail doesn't produce a
    degenerate/zero class mean.
    """
    iterations = []
    override_resid = None
    embedding_norm = None
    for _ in tqdm.trange(n_iterations, desc="Iterative forward passes"):
        if override_resid is None:
            layer_acts, final_resid = get_resid_pre_all_layers(
                model, sequences, word_to_id, n_lookback, fwd_hooks=fwd_hooks)
            # Reference scale: the real token embeddings' average norm at
            # blocks.0.hook_resid_pre, over the same (last-n_lookback,
            # BOS-excluded) positions used below for final_norm_mean, so the
            # two are a like-for-like quotient. The residual stream only
            # grows via skip connections (nothing renormalizes it directly),
            # so by the last layer its norm is ~100-300x the embedding-layer
            # norm -- feeding that back in unscaled means every iteration
            # after the first is operating in a completely different
            # magnitude regime than the model was ever trained on at this
            # hook point.
            embedding_norm = layer_acts[0].norm(dim=-1).mean().item()
        else:
            layer_acts, final_resid = get_resid_pre_all_layers_with_override(
                model, sequences, word_to_id, n_lookback, override_resid, fwd_hooks=fwd_hooks)
        layer_class_means = {
            l: compute_class_means_batch(acts, sequences, WORDS, n_lookback).cpu().numpy()
            for l, acts in layer_acts.items()
        }
        iterations.append(layer_class_means)
        del layer_acts  # views into the 33-hook-point cache -- free before the next iteration allocates its own
        # Rescale by a single fixed scalar -- mean embedding norm divided by
        # this iteration's mean final-layer norm -- rather than forcing every
        # position to the exact same norm. This keeps the *average* magnitude
        # matched to the real token embeddings (still preventing runaway
        # growth across iterations), while preserving each position's norm
        # *relative* to the others (a position that came out longer than
        # average stays proportionally longer, instead of every position
        # being flattened onto one common-radius hypersphere).
        # Same window as embedding_norm above (last n_lookback positions,
        # BOS excluded) -- final_resid itself spans the *entire* sequence
        # including BOS, so averaging over all of it here would quietly mix
        # in the (much larger/anomalous) BOS activation and the far-larger
        # population of early, less-contextualized positions, biasing the
        # quotient away from the tail region the PCA/class-means actually
        # look at.
        final_norm_mean = final_resid[:, 1:, :][:, -n_lookback:, :].norm(dim=-1).mean().item()
        scale = embedding_norm / (final_norm_mean + 1e-6)
        override_resid = final_resid * scale
        del final_resid
        gc.collect()
        torch.cuda.empty_cache()
    return iterations


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    grid = Grid(words=WORDS, rows=GRID_ROWS, cols=GRID_COLS)
    # grid = Torus(words=WORDS, rows=GRID_ROWS, cols=GRID_COLS)
    graph_type = type(grid).__name__  # "Grid" or "Torus"

    # Namespaced by graph_type and word list so a Torus run (or a different
    # vocabulary) doesn't silently load/overwrite mismatched cached data.
    tag = f"{graph_type}_{WORD_LIST_KEY}"
    acc_path = os.path.join(DATA_DIR, f"accuracies_{tag}.npz")
    pca_path = os.path.join(DATA_DIR, f"pca_{tag}.npz")
    seq_path = os.path.join(DATA_DIR, f"sequence_{tag}.json")
    layers_path = os.path.join(DATA_DIR, f"pca_layers_{tag}.npz")
    all_layers_path = os.path.join(DATA_DIR, f"pca_all_layers_{tag}.npz")
    iterative_path = os.path.join(DATA_DIR, f"pca_iterative_{tag}.npz")

    # Loaded lazily below, only if a computation actually needs the model
    # (i.e. one of the cache files it depends on is missing).
    model = None
    word_to_id = None

    # N_SEQUENCES (= len(WORDS)) random-walk sequences, one per starting
    # grid word, shared by every computation below (accuracy curves, PCA
    # class means at every granularity, iterative forward passes). A single
    # sequence's tail can easily miss a word entirely (see
    # compute_class_means's zero-vector fallback), which badly distorts its
    # class mean and the PCA built from it; averaging over the whole batch
    # (compute_class_means_batch) fixes that. Deterministic given
    # set_seed(42) and no other RNG use beforehand, so this is safe/cheap to
    # regenerate unconditionally, even on a full cache hit below.
    set_seed(42)
    sequences = grid.generate_batch(SEQ_LEN, N_SEQUENCES)
    sequence = sequences[0]  # single representative walk, for the bigram plot

    if all(os.path.exists(p) for p in (acc_path, pca_path, seq_path, layers_path)):
        print("Loading cached data (delete data/ to recompute)...")
        all_accs = np.load(acc_path)["all_accs"]
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        explained_variance = pca_data["explained_variance"]
        print(f"{pca_dirs.shape=}")
        layers_data = np.load(layers_path)
        layer_class_means = {layer: layers_data[str(layer)] for layer in LAYERS}
    else:
        model = load_model()
        os.makedirs(DATA_DIR, exist_ok=True)

        # Verify every word is a single token, and get its id, up front --
        # fails fast with a clear message instead of desyncing token/word
        # positions or crashing deep in a shape mismatch later.
        word_to_id = build_word_token_ids(model, WORDS)

        # Accuracy data
        all_accs = []
        for seq in tqdm.tqdm(sequences, desc="Accuracy curves"):
            all_accs.append(get_model_accuracies_by_id(model, grid, seq, word_to_id))
        all_accs = np.array(all_accs)
        np.savez(acc_path, all_accs=all_accs)
        print(f"Cached {acc_path}")

        # PCA data: class means batched over all N_SEQUENCES walks, plus the
        # single representative `sequence`'s individual per-position
        # activations (for the bigram plot, which needs one trajectory).
        activations_t = get_activations_by_id(model, sequence, word_to_id, LAYER, N_LOOKBACK)
        batch_activations_t = get_activations_by_id_batch(model, sequences, word_to_id, LAYER, N_LOOKBACK)
        class_means_t = compute_class_means_batch(batch_activations_t, sequences, WORDS, N_LOOKBACK)
        pca_dirs_t, explained_variance = compute_pca_directions(class_means_t, top_n=3)

        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()

        np.savez(pca_path, activations=activations, class_means=class_means,
                  pca_dirs=pca_dirs, explained_variance=explained_variance)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)
        print(f"Cached {pca_path}")

        # Layer sweep: class means at several depths, batched over the same
        # N_SEQUENCES random-walk sequences.
        layer_class_means = {}
        for layer in tqdm.tqdm(LAYERS, desc="Layer sweep"):
            layer_acts_t = get_activations_by_id_batch(model, sequences, word_to_id, layer, N_LOOKBACK)
            layer_class_means[layer] = compute_class_means_batch(
                layer_acts_t, sequences, WORDS, N_LOOKBACK
            ).cpu().numpy()
        np.savez(layers_path, **{str(l): layer_class_means[l] for l in LAYERS})
        print(f"Cached {layers_path}")

    # All-layer class means (every layer, not just the sparse LAYERS list),
    # batched over the same sequences, for the 4x8 grid covering the
    # model's full depth in one figure.
    if os.path.exists(all_layers_path):
        all_layers_data = np.load(all_layers_path)
        all_layer_class_means = {int(k): all_layers_data[k] for k in all_layers_data.files}
    else:
        if model is None:
            model = load_model()
            word_to_id = build_word_token_ids(model, WORDS)
        layer_acts, _ = get_resid_pre_all_layers(model, sequences, word_to_id, N_LOOKBACK)
        all_layer_class_means = {
            l: compute_class_means_batch(acts, sequences, WORDS, N_LOOKBACK).cpu().numpy()
            for l, acts in layer_acts.items()
        }
        np.savez(all_layers_path, **{str(l): v for l, v in all_layer_class_means.items()})
        print(f"Cached {all_layers_path}")

        # layer_acts holds VIEWS into the 33-hook-point cache (32 layers x
        # [n_sequences, seq_len, d_model], ~12GB for a 16-sequence batch), so
        # the underlying storage stays fully resident in GPU memory for as
        # long as layer_acts does -- and since it's a local of main() (not a
        # nested function), that's the rest of main()'s execution, not just
        # this if/else block. Free it now, before the iterative section
        # below needs to allocate a comparably-sized cache of its own.
        del layer_acts
        gc.collect()
        torch.cuda.empty_cache()

    # Iterative forward passes: feed the final layer's output back in as the
    # first layer's input and run the stack again, N_ITERATIONS times,
    # batched over the same sequences.
    if os.path.exists(iterative_path):
        iterative_data = np.load(iterative_path)
        n_cached_iterations = 1 + max(
            int(k.split("_")[0][len("iter"):]) for k in iterative_data.files
        )
        iterations = [
            {l: iterative_data[f"iter{it}_{l}"] for l in all_layer_class_means}
            for it in range(n_cached_iterations)
        ]
    else:
        if model is None:
            model = load_model()
            word_to_id = build_word_token_ids(model, WORDS)
        iterations = run_iterative_forward_passes(
            model, sequences, word_to_id, N_LOOKBACK, N_ITERATIONS,
        )
        np.savez(iterative_path, **{
            f"iter{it}_{l}": means
            for it, layer_means in enumerate(iterations)
            for l, means in layer_means.items()
        })
        print(f"Cached {iterative_path}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    plot_accuracy_curve(all_accs, suffix=f"_{tag}")
    plot_class_mean_pca(grid, class_means, pca_dirs, suffix=f"_{tag}", explained_variance=explained_variance)

    plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs, suffix=f"_{tag}", explained_variance=explained_variance)

    # Pairwise similarity of the final layer's contextualized per-word mean
    # activations -- a non-lossy complement to the PCA scatter above (which
    # only shows the top 2-3 principal components of the same class means).
    final_layer = max(all_layer_class_means)
    compute_and_plot_similarity_panels(
        all_layer_class_means[final_layer], WORDS, PLOTS_DIR,
        f"final_layer_similarity_{tag}.png",
        title=f"{WORD_LIST_KEY}, final layer {final_layer}",
    )

    plot_pca_across_layers(grid, layer_class_means, LAYERS, suffix=f"_{tag}", top_n=2)
    plot_pca_across_layers(grid, layer_class_means, LAYERS, suffix=f"_{tag}", top_n=3)

    plot_pca_across_layers_grid(
        grid, all_layer_class_means, sorted(all_layer_class_means), suffix=f"_{tag}",
        suptitle=f"PCA across all layers ({WORD_LIST_KEY})",
    )
    for it, layer_class_means_it in enumerate(iterations):
        plot_pca_across_layers_grid(
            grid, layer_class_means_it, sorted(layer_class_means_it), suffix=f"_{tag}_iter{it}",
            suptitle=f"PCA across all layers, iterative pass {it} ({WORD_LIST_KEY})",
        )

    # Dirichlet energy & distance correlation vs. layer, one line per
    # iteration (iterations[0] is the ordinary forward pass, same
    # computation as all_layer_class_means).
    adjacency = grid.build_adjacency_matrix()
    grid_coords = get_grid_coords(grid, WORDS)
    plot_de_dc_across_layers(
        {f"iter{it}": layer_class_means_it for it, layer_class_means_it in enumerate(iterations)},
        adjacency, grid_coords, suffix=f"_{tag}",
        title=f"Dirichlet energy & distance correlation vs. layer ({WORD_LIST_KEY})",
    )

    print(f"{len(sequence)=}")
    print(f"{sequence=}")


# Sweep of 6 grid-position permutations of the same 16 morphology words,
# spanning same-lemma grid-edge fraction from 0.67 (morphology_corners, most
# aligned) down to 0.00 (morphology_permuted, fully decorrelated) -- see
# word_lists.py.
MORPHOLOGY_SWEEP_KEYS = [
    "morphology_corners", "morphology", "morphology_rand3",
    "morphology_rand2", "morphology_rand1", "morphology_permuted",
]
RUN_WORD_LIST_KEYS = MORPHOLOGY_SWEEP_KEYS

# word -> lemma, for the 16 morphology words (see word_lists.py).
MORPHOLOGY_LEMMAS = {
    "play": "play", "plays": "play", "played": "play", "playing": "play",
    "walk": "walk", "walks": "walk", "walked": "walk", "walking": "walk",
    "teach": "teach", "teaches": "teach", "taught": "teach", "teaching": "teach",
    "drive": "drive", "drives": "drive", "drove": "drive", "driving": "drive",
}


def same_lemma_edge_fraction(grid, lemma_map):
    """Fraction of the grid's adjacency edges that connect two words with
    the same lemma, using the grid's own adjacency matrix (the exact
    structure the accuracy task's "valid next word" targets are drawn
    from) rather than recomputing edges by hand."""
    A = grid.build_adjacency_matrix()
    words = grid.words
    n = len(words)
    same = sum(
        A[i, j] for i in range(n) for j in range(i + 1, n)
        if A[i, j] and lemma_map[words[i]] == lemma_map[words[j]]
    )
    total = sum(A[i, j] for i in range(n) for j in range(i + 1, n) if A[i, j])
    return same / total


if __name__ == "__main__":
    for word_list_key in RUN_WORD_LIST_KEYS:
        configure_for_word_list(word_list_key)
        print(f"\n=== Running for WORD_LIST_KEY={WORD_LIST_KEY!r} ===")
        main()

    # Overlay all 5 morphology permutations' accuracy curves on one plot,
    # labeled with each permutation's same-lemma grid-edge fraction.
    accs_by_key = {k: load_cached_accuracies(k) for k in MORPHOLOGY_SWEEP_KEYS}
    accs_by_label = {}
    for key, accs in accs_by_key.items():
        grid = Grid(words=WORD_LISTS[key], rows=4, cols=4)
        frac = same_lemma_edge_fraction(grid, MORPHOLOGY_LEMMAS)
        accs_by_label[f"{key} ({frac*100:.0f}% same-lemma)"] = accs
    plot_accuracy_comparison(
        accs_by_label, out_dir="results/reproduce/plots/morphology_sweep_comparison",
        filename="accuracy_curve_sweep",
    )