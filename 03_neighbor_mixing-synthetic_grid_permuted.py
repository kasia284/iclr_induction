"""Synthetic 4x4 grid embeddings, mixed on a permuted graph.

We place 16 points exactly on a perfect grid lattice within a random 2D
subspace of a high-dimensional space, so their true geometry is a perfect
4x4 grid by construction (PCA before any mixing recovers it exactly). We then
run explicit neighbor mixing using the *standard* fixed-position grid graph,
but with the points reassigned to graph nodes via the same diagonal-Latin-
square shuffle used to build every `*_permuted` list in word_lists.py. So
graph-adjacent nodes are (almost) never true grid neighbors, and mixing
repeatedly averages together geometrically unrelated points.
"""

import os
import math

import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import einops

from utils import (
    Grid, set_seed, setup_plotting, save_figure, set_square_limits,
    build_word_to_color,
    plotly_pca_layout, plotly_pca_traces, save_plotly,
)

GRID_ROWS, GRID_COLS = 4, 4
N_POINTS = GRID_ROWS * GRID_COLS
D_EMBED = 4096
ROUNDS = [0, 1, 2, 3, 10]

PLOTS_DIR = "results/neighbor-mixing-synthetic-grid/"

# Every point is labeled by its TRUE (row, col) coordinate in the 2D
# subspace, regardless of which graph node it ends up permuted onto -- so the
# color/label always tracks true identity, not graph position.
TRUE_LABELS = [f"({r},{c})" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
LABEL_TO_COLOR = build_word_to_color(TRUE_LABELS)


def diagonal_permute(items):
    """Diagonal-Latin-square shuffle: reads an n x n grid (row-major) along
    rotating diagonals. Same algorithm used to build every `*_permuted` list
    in word_lists.py (verified there to reproduce `two_digit_numbers_permuted`
    exactly) -- consecutive outputs are never same-row or same-column in the
    input, so grid-adjacent entries in the output were (almost) never
    grid-adjacent in the input.
    """
    n = int(round(math.isqrt(len(items))))
    assert n * n == len(items), f"not a perfect square: {len(items)}"
    out = []
    for d in range(n):
        for k in range(n):
            r = (d + k) % n
            c = (r + d) % n
            out.append(items[r * n + c])
    return out


def make_grid_embeddings(rows, cols, d_embed, seed=42):
    """Points lying exactly on a perfect (rows x cols) grid lattice within a
    random 2D subspace of R^d_embed: embedding(r, c) = (r - r_mid) * u +
    (c - c_mid) * v, for an orthonormal basis {u, v}. Every embedding is an
    exact linear combination of the same two directions, so PCA on these
    (before any mixing) recovers the perfect grid exactly."""
    set_seed(seed)
    raw = torch.randn(d_embed, 2)
    basis, _ = torch.linalg.qr(raw)  # [d_embed, 2], orthonormal columns
    u, v = basis[:, 0], basis[:, 1]

    r_mid, c_mid = (rows - 1) / 2, (cols - 1) / 2
    embeddings = torch.zeros(rows * cols, d_embed)
    for r in range(rows):
        for c in range(cols):
            embeddings[r * cols + c] = (r - r_mid) * u + (c - c_mid) * v
    return embeddings


def pca_2d(embeddings):
    """Mean-center and SVD -> project onto top-2 PCs."""
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, V = torch.svd(centered)
    directions = einops.rearrange(V, "d n -> n d")[:2, :]
    return centered @ directions.T


def run_mixing_rounds(embeddings, adjacency, rounds):
    """embeddings after 0..max(rounds) rounds of e[i] <- e[i] + mean(e[neighbors
    of i]) on the given graph, keeping only the requested rounds."""
    degree = adjacency.sum(dim=1, keepdim=True)
    embs_by_round = {0: embeddings}
    curr = embeddings
    for r in range(1, max(rounds) + 1):
        neighbor_sum = adjacency @ curr
        curr = curr + neighbor_sum / degree
        if r in rounds:
            embs_by_round[r] = curr
    return embs_by_round


def plot_round_on_ax(ax, projected, grid, labels, title):
    A = grid.build_adjacency_matrix()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )
    for i, label in enumerate(labels):
        ax.scatter(
            projected[i, 0].item(), projected[i, 1].item(),
            color=LABEL_TO_COLOR[label], s=100, marker="*",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax.annotate(
            label, (projected[i, 0].item(), projected[i, 1].item()),
            xytext=(4, 4), textcoords="offset points", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=9)
    set_square_limits(ax, projected[:, 0], projected[:, 1])
    ax.set_aspect("equal")


def save_round_plotly(projected, grid, labels, round_num):
    traces = plotly_pca_traces(projected, grid, words=labels, word_to_color=LABEL_TO_COLOR)
    title = (
        "Synthetic grid embeddings -- 0 rounds mixing" if round_num == 0
        else f"Synthetic grid embeddings -- after {round_num} round(s) of mixing on permuted graph"
    )
    pfig = go.Figure(data=traces)
    pfig.update_layout(**plotly_pca_layout(title))
    save_plotly(pfig, PLOTS_DIR, f"pca_round_{round_num}_permuted_graph.html")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries

    # ── 1. Perfect grid embeddings in a 2D subspace of a high-dim space ──────
    embeddings_natural = make_grid_embeddings(GRID_ROWS, GRID_COLS, D_EMBED)

    # ── 2. Permuted assignment: shuffle WHICH true grid point sits at each
    #      graph node (diagonal-Latin-square shuffle, same as word_lists.py's
    #      *_permuted lists). The mixing graph itself is the standard,
    #      fixed-position 4x4 grid adjacency -- only the node labeling moves.
    permuted_order = diagonal_permute(list(range(N_POINTS)))  # true idx at each node
    embeddings_permuted = embeddings_natural[torch.tensor(permuted_order)]
    permuted_labels = [TRUE_LABELS[i] for i in permuted_order]

    node_ids = [str(i) for i in range(N_POINTS)]  # dummy identities, only used for adjacency
    grid = Grid(words=node_ids, rows=GRID_ROWS, cols=GRID_COLS)
    adjacency = torch.tensor(grid.build_adjacency_matrix(), dtype=torch.float32)

    # ── 3. Explicit neighbor mixing on the permuted-graph embeddings ─────────
    embs_by_round = run_mixing_rounds(embeddings_permuted, adjacency, ROUNDS)

    # ── 4. PCA at each requested round ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(ROUNDS), figsize=(4 * len(ROUNDS), 4.2))
    for ax, r in zip(axes, ROUNDS):
        projected = pca_2d(embs_by_round[r]).numpy()
        title = "0 rounds mixing" if r == 0 else f"after {r} round{'s' if r != 1 else ''}\n(permuted graph)"
        plot_round_on_ax(ax, projected, grid, permuted_labels, title)
        save_round_plotly(projected, grid, permuted_labels, r)

    fig.suptitle(
        "PCA of synthetic perfect-grid embeddings across neighbor-mixing rounds\n"
        "(graph node labeling permuted relative to true grid coordinates)"
    )
    save_figure(fig, PLOTS_DIR, "synthetic_grid_permuted_mixing_rounds.png")
    print("Saved synthetic_grid_permuted_mixing_rounds")


if __name__ == "__main__":
    main()
