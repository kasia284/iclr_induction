import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tqdm

from utils import (
    WORDS, LAYER, WORD_TO_COLOR,
    Grid, set_seed, load_toy_model, get_activations_toy_batch, 
    compute_class_means_batch, compute_pca_directions, setup_plotting, save_figure
)

LAYER = 0
PLOTS_DIR = "results/random-transformer/plots"
N_LOOKBACK = 50

# ── Helper Plotting Functions for Matplotlib Axes ────────────────────────────

def draw_class_mean_on_ax(ax, grid, class_means, pca_dirs, title=""):
    """Scatter of class-mean centroids with grid edges on a specific ax."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T

    # Grid edges (gray dashed)
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )

    # Scatter + labels
    for i, word in enumerate(WORDS):
        ax.scatter(
            projected[i, 0].item(), projected[i, 1].item(),
            color=WORD_TO_COLOR[word], s=80, marker="*",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax.annotate(
            word, (projected[i, 0].item(), projected[i, 1].item()),
            xytext=(3, 3), textcoords="offset points", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def draw_bigram_on_ax(ax, grid, sequences, activations, class_means, pca_dirs, title=""):
    """Individual activations colored by bigram on a specific ax."""
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T
    projected_means = (class_means - class_means_mean) @ pca_dirs.T
    
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected_means[i, 0].item(), projected_means[j, 0].item()],
                    [projected_means[i, 1].item(), projected_means[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )

    # Individual sequence tokens
    for b in range(len(sequences)):
        tail = sequences[b][-N_LOOKBACK:]
        for idx in range(1, len(tail)):
            cur_word = tail[idx]
            prev_word = tail[idx - 1]
            ax.scatter(
                projected_all[b, idx, 0].item(), projected_all[b, idx, 1].item(),
                c=WORD_TO_COLOR[cur_word],
                edgecolors=WORD_TO_COLOR[prev_word],
                linewidths=0.8, s=15, alpha=0.9, zorder=3,
            )

    # Class means overlay
    for i, word in enumerate(WORDS):
        ax.scatter(
            projected_means[i, 0].item(), projected_means[i, 1].item(),
            color=WORD_TO_COLOR[word], s=80, marker="*",
            edgecolors="black", linewidths=1.0, zorder=5,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def add_bigram_legend(fig):
    """Adds a single global legend to the bigram figure."""
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
    fig.legend(handles=legend_elements, loc="upper right", frameon=True,
              framealpha=1.0, edgecolor="gray", fontsize=10)

# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    grid = Grid()
    model = load_toy_model() 

    # Fixed hyperparameters
    batch_sizes = [1, 4, 16, 64, 256]
    seq_len = 64
    seed = 42
    
    # Create 1D grids (1 row, N columns). 
    # Adjusted figsize so the row isn't overly tall.
    fig_class, axes_class = plt.subplots(1, len(batch_sizes), figsize=(20, 5))
    fig_class.suptitle(f"PCA of Per-Node Mean Activations (Seq Len: {seq_len})", fontsize=16)

    fig_bigram, axes_bigram = plt.subplots(1, len(batch_sizes), figsize=(20, 5))
    fig_bigram.suptitle(f"PCA of Individual Activations (Labeled by Bigram) (Seq Len: {seq_len})", fontsize=16)

    model.W_pos.data.zero_()

    print("Iterating through batch sizes...")
    for j, batch_size in enumerate(batch_sizes):
        print(f"Processing Seq: {seq_len}, Batch: {batch_size}")
        
        # Setup data
        set_seed(seed)
        sequences = grid.generate_batch(seq_len, batch_size)
        
        # Run Model
        activations_t = get_activations_toy_batch(model, sequences, LAYER, N_LOOKBACK)
        class_means_t = compute_class_means_batch(activations_t, sequences, WORDS, N_LOOKBACK)
        pca_dirs_t, _ = compute_pca_directions(class_means_t, top_n=2)

        # Move to CPU/NumPy
        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()

        # Target Axes (1D indexing now)
        ax_c = axes_class[j]
        ax_b = axes_bigram[j]
        
        hp_title = f"Batch: {batch_size}"

        # Draw onto the grid
        draw_class_mean_on_ax(ax_c, grid, class_means, pca_dirs, title=hp_title)
        draw_bigram_on_ax(ax_b, grid, sequences, activations, class_means, pca_dirs, title=hp_title)

    # Clean up layout and save Class Means Grid
    fig_class.tight_layout(rect=[0, 0, 1, 0.90]) # Adjusted rect for 1-row title spacing
    save_figure(fig_class, PLOTS_DIR, "pca_class_means_grid.pdf")
    print("Saved pca_class_means_grid.pdf")

    # Clean up layout, add legend, and save Bigram Grid
    fig_bigram.tight_layout(rect=[0, 0, 0.85, 0.90]) 
    add_bigram_legend(fig_bigram)
    save_figure(fig_bigram, PLOTS_DIR, "bigram_pca_grid.pdf")
    print("Saved bigram_pca_grid.pdf")

if __name__ == "__main__":
    main()