import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import umap

from utils import (
    WORDS, LAYER, WORD_TO_COLOR,
    Grid, set_seed, load_toy_model, get_activations_toy_batch,
    compute_class_means_batch, setup_plotting, save_figure,
    set_square_limits,
)

LAYER = 0
PLOTS_DIR = "results/random_transformer/plots"
N_LOOKBACK = 50

METHODS = ["tsne", "umap"]
METHOD_TITLES = {"tsne": "t-SNE", "umap": "UMAP"}


def project(combined, method, seed=42):
    """Fit a 2D non-linear embedding jointly on individual activations +
    class means, so both point sets end up in the same shared space --
    mirroring how the PCA version reuses one set of directions for both."""
    if method == "tsne":
        n_samples = combined.shape[0]
        perplexity = min(5, n_samples - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    elif method == "umap":
        n_neighbors = min(5, combined.shape[0] - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3, random_state=seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(combined)


# ── Helper Plotting Functions for Matplotlib Axes ────────────────────────────

def draw_class_mean_on_ax(ax, grid, means_2d, title=""):
    """Scatter of class-mean centroids with grid edges on a specific ax."""
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [means_2d[i, 0], means_2d[j, 0]],
                    [means_2d[i, 1], means_2d[j, 1]],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )

    for i, word in enumerate(WORDS):
        ax.scatter(
            means_2d[i, 0], means_2d[i, 1],
            color=WORD_TO_COLOR[word], s=80, marker="*",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax.annotate(
            word, (means_2d[i, 0], means_2d[i, 1]),
            xytext=(3, 3), textcoords="offset points", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    set_square_limits(ax, means_2d[:, 0], means_2d[:, 1])
    ax.set_aspect("equal")


def draw_bigram_on_ax(ax, grid, sequences, individual_2d, means_2d, title=""):
    """Individual activations colored by bigram on a specific ax.

    individual_2d: [batch_size, N_LOOKBACK, 2] -- already projected.
    means_2d: [16, 2] -- already projected.
    """
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [means_2d[i, 0], means_2d[j, 0]],
                    [means_2d[i, 1], means_2d[j, 1]],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )

    # Individual sequence tokens
    for b in range(len(sequences)):
        tail = sequences[b][-N_LOOKBACK:]
        for idx in range(1, len(tail)):
            cur_word = tail[idx]
            prev_word = tail[idx - 1]
            ax.scatter(
                individual_2d[b, idx, 0], individual_2d[b, idx, 1],
                c=WORD_TO_COLOR[cur_word],
                edgecolors=WORD_TO_COLOR[prev_word],
                linewidths=0.8, s=15, alpha=0.9, zorder=3,
            )

    # Class means overlay
    for i, word in enumerate(WORDS):
        ax.scatter(
            means_2d[i, 0], means_2d[i, 1],
            color=WORD_TO_COLOR[word], s=80, marker="*",
            edgecolors="black", linewidths=1.0, zorder=5,
        )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    all_x = list(means_2d[:, 0]) + [
        individual_2d[b, idx, 0]
        for b in range(len(sequences)) for idx in range(1, len(sequences[b][-N_LOOKBACK:]))
    ]
    all_y = list(means_2d[:, 1]) + [
        individual_2d[b, idx, 1]
        for b in range(len(sequences)) for idx in range(1, len(sequences[b][-N_LOOKBACK:]))
    ]
    set_square_limits(ax, all_x, all_y)
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

    batch_sizes = [1, 4, 16, 64]
    seq_lens = [64, 256]
    seed = 42
    NORMALIZATION_TYPE = None  # None or "LNPre" or "RMSPre"

    model = load_toy_model(seed=seed, n_ctx=max(seq_lens), normalization_type=NORMALIZATION_TYPE)
    model.W_pos.data.zero_()

    norm_str = str(NORMALIZATION_TYPE) if NORMALIZATION_TYPE is not None else "None"
    n_rows, n_cols = len(seq_lens), len(batch_sizes)

    figs = {}
    for method in METHODS:
        fig_class, axes_class = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        fig_class.suptitle(
            f"{METHOD_TITLES[method]} of Per-Node Mean Activations | Normalization: {norm_str}", fontsize=16)

        fig_bigram, axes_bigram = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        fig_bigram.suptitle(
            f"{METHOD_TITLES[method]} of Individual Activations (Labeled by Bigram) | Normalization: {norm_str}",
            fontsize=16)

        figs[method] = {
            "class": fig_class, "class_axes": axes_class,
            "bigram": fig_bigram, "bigram_axes": axes_bigram,
        }

    print("Iterating through seq_len x batch_size grid...")
    for i, seq_len in enumerate(seq_lens):
        for j, batch_size in enumerate(batch_sizes):
            print(f"Processing Seq: {seq_len}, Batch: {batch_size}")

            set_seed(seed)
            sequences = grid.generate_batch(seq_len, batch_size)

            activations_t = get_activations_toy_batch(model, sequences, LAYER, N_LOOKBACK)
            class_means_t = compute_class_means_batch(activations_t, sequences, WORDS, N_LOOKBACK)

            activations = activations_t.cpu().numpy()  # [batch, N_LOOKBACK, d_model]
            class_means = class_means_t.cpu().numpy()  # [16, d_model]

            flat_activations = activations.reshape(-1, activations.shape[-1])
            combined = np.concatenate([flat_activations, class_means], axis=0)

            title = f"Seq: {seq_len} | Batch: {batch_size}"

            for method in METHODS:
                print(f"  Fitting {METHOD_TITLES[method]}...")
                projected = project(combined, method, seed=seed)

                individual_2d = projected[:flat_activations.shape[0]].reshape(batch_size, N_LOOKBACK, 2)
                means_2d = projected[flat_activations.shape[0]:]

                ax_c = figs[method]["class_axes"][i, j]
                ax_b = figs[method]["bigram_axes"][i, j]

                draw_class_mean_on_ax(ax_c, grid, means_2d, title=title)
                draw_bigram_on_ax(ax_b, grid, sequences, individual_2d, means_2d, title=title)

    for method in METHODS:
        fig_class = figs[method]["class"]
        fig_bigram = figs[method]["bigram"]

        fig_class.tight_layout(rect=[0, 0, 1, 0.94])
        class_filename = f"{method}_class_means_sweep_grid_Norm{norm_str}.pdf"
        save_figure(fig_class, PLOTS_DIR, class_filename)
        print(f"Saved {class_filename}")

        fig_bigram.tight_layout(rect=[0, 0, 0.88, 0.94])
        add_bigram_legend(fig_bigram)
        bigram_filename = f"bigram_{method}_sweep_grid_Norm{norm_str}.pdf"
        save_figure(fig_bigram, PLOTS_DIR, bigram_filename)
        print(f"Saved {bigram_filename}")


if __name__ == "__main__":
    main()
