"""Figure 5: toy model of previous-token (neighbor) mixing."""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils import (
    Grid, set_seed, setup_plotting, save_figure,
    plotly_pca_layout, plotly_pca_traces, save_plotly,
    pca_2d, draw_class_mean_pca_on_ax,
)

DATA_DIR = "results/neighbor_mixing/data"
PLOTS_DIR = "results/neighbor_mixing/plots"
D_EMBED = 4096


def plot_pca_scatter(projected, var_explained, grid, title, filename):
    """Scatter of 16 word embeddings projected onto PC1/PC2 with grid edges."""
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_class_mean_pca_on_ax(ax, grid, projected, is_3d=False)

    # Add variance explained to axes labels
    pc1_label = f"PC1 ({var_explained[0]*100:.1f}%)"
    pc2_label = f"PC2 ({var_explained[1]*100:.1f}%)"
    
    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.set_title(title)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, filename)
    print(f"Saved {filename}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout(title))
    pfig.update_xaxes(title_text=pc1_label)
    pfig.update_yaxes(title_text=pc2_label)
    html_stem = os.path.splitext(filename)[0]
    save_plotly(pfig, PLOTS_DIR, f"{html_stem}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    grid = Grid()

    data_path = os.path.join(DATA_DIR, "mixing_extended_fve.npz")

    A = grid.build_adjacency_matrix()
    A_torch = torch.tensor(A, dtype=torch.float32)

    # Random Gaussian embeddings
    set_seed(42)
    embeddings = torch.randn(16, D_EMBED)

    # Before mixing (0 rounds)
    proj_before, var_before = pca_2d(embeddings)
    proj_before = proj_before.numpy()

    # Degree vector used for averaging neighbors
    degree = A_torch.sum(dim=1, keepdim=True)  # [16, 1]

    # Round 1 of neighbor mixing: e_mixed[i] = e[i] + mean(e[neighbors of i])
    neighbor_sum_1 = A_torch @ embeddings  # [16, D_EMBED]
    mixed_1 = embeddings + neighbor_sum_1 / degree
    proj_after, var_after = pca_2d(mixed_1)
    proj_after = proj_after.numpy()

    # Round 2 of neighbor mixing
    neighbor_sum_2 = A_torch @ mixed_1
    mixed_2 = mixed_1 + neighbor_sum_2 / degree
    proj_after_2, var_after_2 = pca_2d(mixed_2)
    proj_after_2 = proj_after_2.numpy()

    # Round 3 of neighbor mixing
    neighbor_sum_3 = A_torch @ mixed_2
    mixed_3 = mixed_2 + neighbor_sum_3 / degree
    proj_after_3, var_after_3 = pca_2d(mixed_3)
    proj_after_3 = proj_after_3.numpy()

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(
        data_path, 
        proj_before=proj_before, var_before=var_before,
        proj_after=proj_after, var_after=var_after,
        proj_after_2=proj_after_2, var_after_2=var_after_2,
        proj_after_3=proj_after_3, var_after_3=var_after_3
    )
    print(f"Cached updated data to {data_path}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    plot_pca_scatter(
        proj_before, var_before, grid,
        "Random embeddings\n(no neighbor mixing)", "before_mixing_extended_fve.pdf"
    )
    plot_pca_scatter(
        proj_after, var_after, grid,
        "Random embeddings\n(after one round of neighbor mixing)", "after_mixing_extended_fve.pdf"
    )
    plot_pca_scatter(
        proj_after_2, var_after_2, grid,
        "Random embeddings\n(after two rounds of neighbor mixing)", "after_2_mixing_extended_fve.pdf"
    )
    plot_pca_scatter(
        proj_after_3, var_after_3, grid,
        "Random embeddings\n(after three rounds of neighbor mixing)", "after_3_mixing_extended_fve.pdf"
    )


if __name__ == "__main__":
    main()