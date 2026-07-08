"""Reproduce Figures 2 and 6: accuracy curve, class-mean PCA, and bigram PCA."""

import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import tqdm

from utils import (
    LAYER, build_word_to_color,
    Grid, Torus, set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)
from utils import plot_attention_heatmaps
from word_lists import WORD_LISTS

WORD_LIST_KEY = "grid_36"
WORDS = WORD_LISTS[WORD_LIST_KEY]
WORD_TO_COLOR = build_word_to_color(WORDS)
GRID_ROWS, GRID_COLS = 6, 6

DATA_DIR = "results/reproduce/data"
PLOTS_DIR = "results/reproduce/plots"
N_LOOKBACK = 200
SEQ_LEN = 1400  # long random walk, matching the original paper's context length
LAYERS = [0, 6, 13, 20, 26, 31]  # even spread across Llama-3.1-8B's 32 layers
N_SEQUENCES = len(WORDS)  # one sequence per starting word, for full grid coverage


# ── Fig 2 Left: Accuracy curve ────────────────────────────────────────────────

def plot_accuracy_curve(all_accs):
    """Average accuracy across 16 sequences with uniform starting positions."""
    mean = smooth(all_accs.mean(axis=0))
    std = smooth(all_accs.std(axis=0))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(mean, color="black", linewidth=1.0, label="Mean accuracy")
    ax.fill_between(range(len(mean)), mean - std, mean + std,
                    alpha=0.15, color="gray", edgecolor="none", label="$\\pm$1 std")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title("Accuracy vs sequence length", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray")

    save_figure(fig, PLOTS_DIR, "accuracy_curve.pdf")
    print("Saved accuracy_curve")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure()
    x = list(range(len(mean)))
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
        "Accuracy vs sequence length", "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, PLOTS_DIR, "accuracy_curve.html")


# ── Fig 2 Right: Class-mean PCA ───────────────────────────────────────────────

# def plot_class_mean_pca(grid, class_means, pca_dirs):
#     """Scatter of 16 class-mean centroids with grid edges."""
#     projected = class_means @ pca_dirs.T  # [16, 2]

#     fig, ax = plt.subplots(figsize=(5, 5))

#     # Grid edges (gray dashed)
#     A = grid.build_adjacency_matrix()
#     for i in range(len(WORDS)):
#         for j in range(i + 1, len(WORDS)):
#             if A[i, j]:
#                 ax.plot(
#                     [projected[i, 0].item(), projected[j, 0].item()],
#                     [projected[i, 1].item(), projected[j, 1].item()],
#                     color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
#                 )

#     # Scatter + labels
#     for i, word in enumerate(WORDS):
#         ax.scatter(
#             projected[i, 0].item(), projected[i, 1].item(),
#             color=WORD_TO_COLOR[word], s=120, marker="*",
#             edgecolors="black", linewidths=0.5, zorder=5,
#         )
#         ax.annotate(
#             word, (projected[i, 0].item(), projected[i, 1].item()),
#             xytext=(5, 5), textcoords="offset points", fontsize=8,
#             bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
#         )

#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     ax.set_title("PCA of per-node mean activations", fontsize=10)
#     ax.set_aspect("equal")
#     save_figure(fig, PLOTS_DIR, "pca_class_means.pdf")
#     print("Saved pca_class_means")

#     # ── Plotly interactive ───────────────────────────────────────────────────
#     pfig = go.Figure(data=plotly_pca_traces(projected, grid))
#     pfig.update_layout(**plotly_pca_layout("PCA of per-node mean activations"))
#     save_plotly(pfig, PLOTS_DIR, "pca_class_means.html")

def plot_class_mean_pca(grid, class_means, pca_dirs, suffix="", title=None, ax=None):
    """Scatter of 16 class-mean centroids with grid edges (Supports 2D and 3D).

    ax: matplotlib Axes, optional. Draw onto this existing axes instead of
    creating (and saving) a new standalone figure. Used to compose several
    layers into one row (see plot_pca_across_layers).
    """
    projected = class_means @ pca_dirs.T  # [16, num_components]
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

    # Grid edges (gray dashed)
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                if is_3d:
                    ax.plot(
                        [projected[i, 0].item(), projected[j, 0].item()],
                        [projected[i, 1].item(), projected[j, 1].item()],
                        [projected[i, 2].item(), projected[j, 2].item()],
                        color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                    )
                else:
                    ax.plot(
                        [projected[i, 0].item(), projected[j, 0].item()],
                        [projected[i, 1].item(), projected[j, 1].item()],
                        color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                    )

    # Grey face for each grid cell of the plotted shape (3D only).
    if is_3d:
        word_to_idx = {w: i for i, w in enumerate(WORDS)}
        is_torus = type(grid).__name__ == "Torus"
        r_range = range(grid.rows) if is_torus else range(grid.rows - 1)
        c_range = range(grid.cols) if is_torus else range(grid.cols - 1)
        faces = []
        for r in r_range:
            for c in c_range:
                r2, c2 = (r + 1) % grid.rows, (c + 1) % grid.cols
                corners = [grid.grid[r][c], grid.grid[r][c2], grid.grid[r2][c2], grid.grid[r2][c]]
                idxs = [word_to_idx[w] for w in corners]
                faces.append([projected[i, :3].tolist() for i in idxs])
        if faces:
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor=(0.6, 0.6, 0.6, 0.4),
                edgecolor=(0.4, 0.4, 0.4, 0.6), linewidths=0.5,
            ))

    # Scatter + labels
    for i, word in enumerate(WORDS):
        if is_3d:
            ax.scatter(
                projected[i, 0].item(), projected[i, 1].item(), projected[i, 2].item(),
                color=WORD_TO_COLOR[word], s=120, marker="*",
                edgecolors="black", linewidths=0.5, zorder=5,
            )
            # ax.text is used for 3D positioning instead of ax.annotate
            ax.text(
                projected[i, 0].item(), projected[i, 1].item(), projected[i, 2].item(),
                word, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )
        else:
            ax.scatter(
                projected[i, 0].item(), projected[i, 1].item(),
                color=WORD_TO_COLOR[word], s=120, marker="*",
                edgecolors="black", linewidths=0.5, zorder=5,
            )
            ax.annotate(
                word, (projected[i, 0].item(), projected[i, 1].item()),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    # Axis Labels & Aspect Ratio
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    
    if is_3d:
        ax.set_zlabel("PC3")
        ax.set_box_aspect((1, 1, 1))  # Sets equal aspect ratio for 3D bounding box
    else:
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
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout(f"{'3D ' if is_3d else ''}PCA of per-node mean activations"))
    save_plotly(pfig, PLOTS_DIR, f"pca_class_means{suffix}_{dim_suffix}.html")


def save_pca_layer_plotly_3d(grid, class_means, pca_dirs, layer, suffix=""):
    """Interactive, rotatable 3D PCA scatter for a single layer."""
    projected = class_means @ pca_dirs.T  # [16, 3]

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
        line=dict(color="gray", width=2), opacity=0.4,
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

    pfig = go.Figure(data=traces)
    pfig.update_layout(
        title=f"Layer {layer} -- PCA of per-node mean activations",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", aspectmode="cube"),
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
        pca_dirs_t, _ = compute_pca_directions(torch.tensor(class_means), top_n=top_n)
        pca_dirs = pca_dirs_t.numpy()
        plot_class_mean_pca(grid, class_means, pca_dirs, title=f"Layer {layer}", ax=ax)
        if is_3d:
            save_pca_layer_plotly_3d(grid, class_means, pca_dirs, layer, suffix=suffix)

    fig.suptitle(f"{'3D ' if is_3d else ''}PCA of per-node mean activations across layers")
    dim_suffix = "3d" if is_3d else "2d"
    out_name = f"pca_across_layers{suffix}_{dim_suffix}"
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")


import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_class_mean_pca_3d(grid, class_means, pca_dirs, title=None):
    """Scatter of 16 class-mean centroids with grid edges in 3D."""
    # pca_dirs must now contain the top 3 principal components -> [3, d]
    projected = class_means @ pca_dirs.T  # Shape: [16, 3]

    # ── Matplotlib 3D Plot ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Grid edges (gray dashed)
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    [projected[i, 2].item(), projected[j, 2].item()],
                    color="gray",
                    alpha=0.3,
                    linestyle="--",
                    linewidth=0.5,
                )

    # Scatter + labels
    for i, word in enumerate(WORDS):
        x, y, z = (
            projected[i, 0].item(),
            projected[i, 1].item(),
            projected[i, 2].item(),
        )

        ax.scatter(
            x,
            y,
            z,
            color=WORD_TO_COLOR[word],
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        # ax.text is used instead of ax.annotate for 3D coordinate support
        ax.text(
            x,
            y,
            z,
            word,
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(
        title or "3D PCA of per-node mean activations", fontsize=10
    )

    # Note: 'equal' aspect ratio isn't natively supported the same way in 3D across all MPL versions,
    # but you can use ax.set_box_aspect((1,1,1)) to keep the bounding box square.
    ax.set_box_aspect((1, 1, 1))

    save_figure(fig, PLOTS_DIR, "pca_class_means_3d.pdf")
    print("Saved pca_class_means_3d")

    # ── Plotly Interactive 3D Plot ───────────────────────────────────────────
    plotly_traces = []

    # 1. Build 3D lines for edges
    edge_x, edge_y, edge_z = [], [], []
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                edge_x.extend(
                    [projected[i, 0].item(), projected[j, 0].item(), None]
                )
                edge_y.extend(
                    [projected[i, 1].item(), projected[j, 1].item(), None]
                )
                edge_z.extend(
                    [projected[i, 2].item(), projected[j, 2].item(), None]
                )

    plotly_traces.append(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="gray", width=1.5),
            opacity=0.3,
            hoverinfo="skip",
        )
    )

    # 2. Build 3D scatter points
    xs = [projected[i, 0].item() for i in range(len(WORDS))]
    ys = [projected[i, 1].item() for i in range(len(WORDS))]
    zs = [projected[i, 2].item() for i in range(len(WORDS))]
    colors = [WORD_TO_COLOR[word] for word in WORDS]

    plotly_traces.append(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            marker=dict(
                size=10,
                color=colors,
                symbol="star",
                line=dict(color="black", width=1),
            ),
            text=WORDS,
            textposition="top center",
            hoverinfo="text",
        )
    )

    # 3. Create figure and apply 3D layout scene
    pfig = go.Figure(data=plotly_traces)
    pfig.update_layout(
        title=title or "3D PCA of per-node mean activations",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    save_plotly(pfig, PLOTS_DIR, f"pca_class_means_3d_{title}.html")

# ── Fig 6: Bigram PCA ─────────────────────────────────────────────────────────

def _draw_bigram_scatter(ax, projected_all, projected_means, tail, grid, label=True):
    """Draw bigram scatter on a given axes. Shared by main plot and inset."""
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected_means[i, 0].item(), projected_means[j, 0].item()],
                    [projected_means[i, 1].item(), projected_means[j, 1].item()],
                    color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                )

    for idx in range(1, len(tail)):
        cur_word = tail[idx]
        prev_word = tail[idx - 1]
        ax.scatter(
            projected_all[idx, 0].item(), projected_all[idx, 1].item(),
            c=WORD_TO_COLOR[cur_word],
            edgecolors=WORD_TO_COLOR[prev_word],
            linewidths=1.0, s=25, alpha=1.0, zorder=3,
        )

    for i, word in enumerate(WORDS):
        ax.scatter(
            projected_means[i, 0].item(), projected_means[i, 1].item(),
            color=WORD_TO_COLOR[word], s=120, marker="*",
            edgecolors="black", linewidths=1.0, zorder=5,
        )
        if label:
            ax.annotate(
                word, (projected_means[i, 0].item(), projected_means[i, 1].item()),
                xytext=(5, 5), textcoords="offset points", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )


def _make_bigram_legend(ax):
    """Add legend for bigram PCA plots."""
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
    ax.legend(handles=legend_elements, loc="upper left", frameon=True,
              framealpha=1.0, edgecolor="gray", fontsize=8)


def plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs, suffix=""):
    """Individual activations colored by (current token, previous token)."""
    tail = sequence[-N_LOOKBACK:]
    projected_all = activations @ pca_dirs.T        # [N_LOOKBACK, 2]
    projected_means = class_means @ pca_dirs.T      # [16, 2]

    # ── Main bigram plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_bigram_scatter(ax, projected_all, projected_means, tail, grid)
    _make_bigram_legend(ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of individual activations, labeled by bigram", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, f"bigram_pca{suffix}.pdf")
    print(f"Saved bigram_pca{suffix}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected_means, grid))

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    # grid = Grid(words=WORDS, rows=GRID_ROWS, cols=GRID_COLS)
    grid = Torus(words=WORDS, rows=GRID_ROWS, cols=GRID_COLS)
    graph_type = type(grid).__name__  # "Grid" or "Torus"

    # Namespaced by graph_type and word list so a Torus run (or a different
    # vocabulary) doesn't silently load/overwrite mismatched cached data.
    tag = f"{graph_type}_{WORD_LIST_KEY}"
    acc_path = os.path.join(DATA_DIR, f"accuracies_{tag}.npz")
    pca_path = os.path.join(DATA_DIR, f"pca_{tag}.npz")
    seq_path = os.path.join(DATA_DIR, f"sequence_{tag}.json")
    layers_path = os.path.join(DATA_DIR, f"pca_layers_{tag}.npz")

    if all(os.path.exists(p) for p in (acc_path, pca_path, seq_path, layers_path)):
        print("Loading cached data (delete data/ to recompute)...")
        all_accs = np.load(acc_path)["all_accs"]
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        print(f"{pca_dirs.shape=}")
        with open(seq_path) as f:
            sequence = json.load(f)
        layers_data = np.load(layers_path)
        layer_class_means = {layer: layers_data[str(layer)] for layer in LAYERS}
    else:
        model = load_model()
        os.makedirs(DATA_DIR, exist_ok=True)

        # Accuracy data
        set_seed(42)
        sequences = grid.generate_batch(SEQ_LEN, N_SEQUENCES)
        all_accs = []
        for seq in tqdm.tqdm(sequences, desc="Accuracy curves"):
            all_accs.append(get_model_accuracies(model, grid, seq))
        all_accs = np.array(all_accs)
        np.savez(acc_path, all_accs=all_accs)
        print(f"Cached {acc_path}")

        # PCA data
        set_seed(42)
        sequence = grid.generate_sequence(SEQ_LEN)
        activations_t = get_activations(model, sequence, LAYER, N_LOOKBACK)
        class_means_t = compute_class_means(activations_t, sequence, WORDS, N_LOOKBACK)
        pca_dirs_t, _ = compute_pca_directions(class_means_t, top_n=3)

        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()

        np.savez(pca_path, activations=activations, class_means=class_means, pca_dirs=pca_dirs)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)
        print(f"Cached {pca_path}")

        # Layer sweep: class means at several depths, to see the grid
        # geometry emerge across the model's layers on the same sequence.
        layer_class_means = {}
        for layer in tqdm.tqdm(LAYERS, desc="Layer sweep"):
            layer_acts_t = get_activations(model, sequence, layer, N_LOOKBACK)
            layer_class_means[layer] = compute_class_means(
                layer_acts_t, sequence, WORDS, N_LOOKBACK
            ).cpu().numpy()
        np.savez(layers_path, **{str(l): layer_class_means[l] for l in LAYERS})
        print(f"Cached {layers_path}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    # plot_accuracy_curve(all_accs)
    plot_class_mean_pca(grid, class_means, pca_dirs, suffix=f"_{tag}")
    # plot_class_mean_pca_3d(grid, class_means, pca_dirs)

    plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs, suffix=f"_{tag}")
    plot_pca_across_layers(grid, layer_class_means, LAYERS, suffix=f"_{tag}", top_n=2)
    plot_pca_across_layers(grid, layer_class_means, LAYERS, suffix=f"_{tag}", top_n=3)

    print(f"{len(sequence)=}")
    print(f"{sequence=}")


if __name__ == "__main__":
    main()