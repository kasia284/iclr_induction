"""Figure 5: toy model of previous-token (neighbor) mixing."""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import einops

from utils import (
    WORDS, WORD_TO_COLOR,
    Grid, set_seed, setup_plotting, save_figure,
    plotly_pca_layout, plotly_pca_traces, save_plotly,
)

DATA_DIR = "results/neighbor_mixing/data"
PLOTS_DIR = "results/neighbor_mixing/plots"
D_EMBED = 4096


def pca_2d(embeddings):
    """Mean-center and SVD → project onto top-2 PCs. Returns [16, 2]."""
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, V = torch.svd(centered)
    directions = einops.rearrange(V, "d n -> n d")[:2, :]  # [2, d]
    return centered @ directions.T  # [16, 2]


def plot_pca_scatter(projected, grid, title, filename):
    """Scatter of 16 word embeddings projected onto PC1/PC2 with grid edges."""
    fig, ax = plt.subplots(figsize=(5, 5))

    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                )

    for i, word in enumerate(WORDS):
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

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, filename)
    print(f"Saved {filename}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout(title))
    html_stem = os.path.splitext(filename)[0]
    save_plotly(pfig, PLOTS_DIR, f"{html_stem}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    grid = Grid()

    data_path = os.path.join(DATA_DIR, "mixing.npz")


    A = grid.build_adjacency_matrix()
    A_torch = torch.tensor(A, dtype=torch.float32)

    # Random Gaussian embeddings
    set_seed(42)
    embeddings = torch.randn(16, D_EMBED)

    # Before mixing (0 rounds)
    proj_before = pca_2d(embeddings).numpy()

    # Degree vector used for averaging neighbors
    degree = A_torch.sum(dim=1, keepdim=True)  # [16, 1]

    # Round 1 of neighbor mixing: e_mixed[i] = e[i] + mean(e[neighbors of i])
    neighbor_sum_1 = A_torch @ embeddings  # [16, D_EMBED]
    mixed_1 = embeddings + neighbor_sum_1 / degree
    proj_after = pca_2d(mixed_1).numpy()

    # Round 2 of neighbor mixing
    neighbor_sum_2 = A_torch @ mixed_1
    mixed_2 = mixed_1 + neighbor_sum_2 / degree
    proj_after_2 = pca_2d(mixed_2).numpy()

    # Round 3 of neighbor mixing
    neighbor_sum_3 = A_torch @ mixed_2
    mixed_3 = mixed_2 + neighbor_sum_3 / degree
    proj_after_3 = pca_2d(mixed_3).numpy()

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(
        data_path, 
        proj_before=proj_before, 
        proj_after=proj_after,
        proj_after_2=proj_after_2,
        proj_after_3=proj_after_3
    )
    print(f"Cached updated data to {data_path}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    plot_pca_scatter(
        proj_before, grid,
        "Random embeddings\n(no neighbor mixing)", "before_mixing.pdf"
    )
    plot_pca_scatter(
        proj_after, grid,
        "Random embeddings\n(after one round of neighbor mixing)", "after_mixing.pdf"
    )
    plot_pca_scatter(
        proj_after_2, grid,
        "Random embeddings\n(after two rounds of neighbor mixing)", "after_2_mixing.pdf"
    )
    plot_pca_scatter(
        proj_after_3, grid,
        "Random embeddings\n(after three rounds of neighbor mixing)", "after_3_mixing.pdf"
    )


if __name__ == "__main__":
    main()
    