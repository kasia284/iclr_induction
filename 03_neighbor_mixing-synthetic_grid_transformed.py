"""Synthetic 4x4 grid embeddings, explicitly transformed, then mixed on a
permuted graph.

Generalizes 03_neighbor_mixing-synthetic_grid_permuted.py: instead of lifting
the 16 grid points into R^d_model via a single fixed random 2D subspace, the
2D grid coordinates first go through an explicit affine transform (rotate,
anisotropically stretch, translate) *before* being lifted, so the effect of
each transform on the post-mixing geometry can be inspected directly.

The graph used for neighbor mixing is still the standard fixed-position 4x4
grid adjacency (each node mixes with its 4 physical grid neighbors) -- the
"different connectivity" comes from reassigning *which* transformed point
sits at each graph node via the same diagonal-Latin-square shuffle used to
build every `*_permuted` list in word_lists.py, so graph-adjacent nodes are
(almost) never true grid neighbors and mixing repeatedly averages together
geometrically unrelated points.

Note on which transform parameters are actually visible in the plots below:
- STRETCH (anisotropic scale, sx != sy) is the only one that changes the
  post-PCA geometry: it changes the grid's true aspect ratio, which survives
  mean-centering and PCA re-projection.
- ROTATE is invisible here: rotating the 2D points before lifting them
  through a *random* orthonormal basis is indistinguishable from lifting the
  unrotated points through a differently-random basis, and PCA re-discovers
  the same intrinsic shape (up to which axis it calls PC1 vs PC2) regardless.
- TRANSLATE is also invisible: neighbor mixing (e[i] <- e[i] + mean(e[j] for
  j in neighbors(i))) and PCA's mean-centering both cancel a uniform offset
  added to every point.
They're still implemented as explicit, independent steps (not simplified
away) so this stays a general "build a synthetic grid under any affine map"
tool, and so this asymmetry is demonstrated rather than assumed.
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
ROUNDS = [0, 1, 2, 3, 10, 20, 30, 50, 100]

# Explicit affine transform applied to the 2D grid before lifting into
# R^D_EMBED. See the module docstring for which of these actually show up
# in the post-mixing PCA plots.
ROTATE_DEGREES = 30.0
STRETCH = (2.5, 1.0)   # (sx, sy): anisotropic scale along the grid's row/col axes
TRANSLATE = (5.0, -3.0)  # (tx, ty)

PLOTS_DIR = "results/neighbor-mixing-synthetic-grid-transformed/"

TRUE_LABELS = [f"({r},{c})" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
LABEL_TO_COLOR = build_word_to_color(TRUE_LABELS)


def make_grid_points_2d(rows, cols):
    """16 points at exact integer (row, col) grid coordinates, centered at
    the origin. Returns [rows*cols, 2]."""
    r_mid, c_mid = (rows - 1) / 2, (cols - 1) / 2
    points = torch.zeros(rows * cols, 2)
    for r in range(rows):
        for c in range(cols):
            points[r * cols + c] = torch.tensor([r - r_mid, c - c_mid], dtype=torch.float32)
    return points


def transform_2d(points, rotate_degrees=0.0, stretch=(1.0, 1.0), translate=(0.0, 0.0)):
    """Applies, in order: anisotropic scale -> rotation -> translation.
    points: [N, 2]. Returns [N, 2]."""
    sx, sy = stretch
    scaled = points * torch.tensor([sx, sy], dtype=torch.float32)

    theta = math.radians(rotate_degrees)
    rot = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ], dtype=torch.float32)
    rotated = scaled @ rot.T

    translated = rotated + torch.tensor(translate, dtype=torch.float32)
    return translated


def lift_to_dmodel(points_2d, d_model, seed=42):
    """Embeds 2D points into R^d_model via a random orthonormal 2D subspace
    {u, v}: embedding = x * u + y * v. points_2d: [N, 2]. Returns [N, d_model]."""
    set_seed(seed)
    raw = torch.randn(d_model, 2)
    basis, _ = torch.linalg.qr(raw)  # [d_model, 2], orthonormal columns
    return points_2d @ basis.T  # [N, 2] @ [2, d_model] -> [N, d_model]


def diagonal_permute(items):
    """Diagonal-Latin-square shuffle: reads an n x n grid (row-major) along
    rotating diagonals -- same algorithm used to build every `*_permuted`
    list in word_lists.py. Consecutive outputs are never same-row or
    same-column in the input, so grid-adjacent entries in the output were
    (almost) never grid-adjacent in the input."""
    n = int(round(math.isqrt(len(items))))
    assert n * n == len(items), f"not a perfect square: {len(items)}"
    out = []
    for d in range(n):
        for k in range(n):
            r = (d + k) % n
            c = (r + d) % n
            out.append(items[r * n + c])
    return out


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
        "Transformed synthetic grid embeddings -- 0 rounds mixing" if round_num == 0
        else f"Transformed synthetic grid embeddings -- after {round_num} round(s) of mixing on permuted graph"
    )
    pfig = go.Figure(data=traces)
    pfig.update_layout(**plotly_pca_layout(title))
    save_plotly(pfig, PLOTS_DIR, f"pca_round_{round_num}_permuted_graph.html")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries

    # ── 1. Perfect 4x4 grid in 2D, then an explicit rotate/stretch/translate ──
    points_natural = make_grid_points_2d(GRID_ROWS, GRID_COLS)
    points_transformed = transform_2d(
        points_natural, rotate_degrees=ROTATE_DEGREES, stretch=STRETCH, translate=TRANSLATE,
    )

    # ── 2. Lift into R^D_EMBED via a random orthonormal 2D subspace ──────────
    embeddings_natural = lift_to_dmodel(points_transformed, D_EMBED, seed=42)

    # ── 3. Different connectivity: same fixed-position grid graph, but a
    #      diagonal-Latin-square shuffle decides which transformed point sits
    #      at each graph node. Only the node labeling moves; the adjacency
    #      (which node indices are edges) is the standard grid's. ───────────
    permuted_order = diagonal_permute(list(range(N_POINTS)))  # true idx at each node
    embeddings_permuted = embeddings_natural[torch.tensor(permuted_order)]
    permuted_labels = [TRUE_LABELS[i] for i in permuted_order]

    node_ids = [str(i) for i in range(N_POINTS)]  # dummy identities, only used for adjacency
    grid = Grid(words=node_ids, rows=GRID_ROWS, cols=GRID_COLS)
    adjacency = torch.tensor(grid.build_adjacency_matrix(), dtype=torch.float32)

    # ── 4. Explicit neighbor mixing on the permuted-graph embeddings ─────────
    embs_by_round = run_mixing_rounds(embeddings_permuted, adjacency, ROUNDS)

    # ── 5. PCA at each requested round ────────────────────────────────────────
    n_rows = 2
    n_cols = math.ceil(len(ROUNDS) / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.2 * n_rows))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, ROUNDS):
        projected = pca_2d(embs_by_round[r]).numpy()
        title = "0 rounds mixing" if r == 0 else f"after {r} round{'s' if r != 1 else ''}\n(permuted graph)"
        plot_round_on_ax(ax, projected, grid, permuted_labels, title)
        save_round_plotly(projected, grid, permuted_labels, r)
    for ax in axes_flat[len(ROUNDS):]:
        ax.axis("off")

    fig.suptitle(
        "PCA of transformed synthetic grid embeddings across neighbor-mixing rounds\n"
        f"(rotate={ROTATE_DEGREES}°, stretch={STRETCH}, translate={TRANSLATE}; "
        "graph node labeling permuted relative to true grid coordinates)"
    )
    save_figure(fig, PLOTS_DIR, "synthetic_grid_transformed_permuted_mixing_rounds.png")
    print("Saved synthetic_grid_transformed_permuted_mixing_rounds")


if __name__ == "__main__":
    main()
