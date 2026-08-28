import math
import functools

import matplotlib.pyplot as plt

from utils import (
    WORDS, LAYER, WORD_TO_COLOR,
    Grid, set_seed, load_toy_model, get_activations_toy_batch,
    compute_class_means_batch, compute_pca_directions, setup_plotting, save_figure,
    set_square_limits, draw_class_mean_pca_on_ax, draw_bigram_scatter_on_ax, make_bigram_legend,
)

LAYER = 0
PLOTS_DIR = "results/random_transformer/plots"
N_LOOKBACK = 50


def layer_norm_normalized_hook(activation, hook, eps=1e-6):
    """Rescale LayerNorm's output (norm sqrt(d)) down to the unit
    hypersphere (norm 1), so attention is computed on unit-norm vectors."""
    norm = activation.norm(dim=-1, keepdim=True)
    return activation / (norm + eps)


def scale_embed_hook(activation, hook, scale):
    """No LayerNorm at all: just scale the raw embeddings by a constant
    scalar (sqrt(d_model)), so their magnitude matches what LayerNorm's
    output would have, without any per-token centering/normalizing."""
    return activation * scale


# ── Helper Plotting Functions for Matplotlib Axes ────────────────────────────

def draw_class_mean_on_ax(ax, grid, class_means, pca_dirs, explained_var, title=""):
    """Scatter of class-mean centroids with grid edges on a specific ax."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T

    draw_class_mean_pca_on_ax(ax, grid, projected, marker_size=80, label_fontsize=6,
                               label_offset=(3, 3), is_3d=False)

    ax.set_title(
        f"{title}\nPC1 {explained_var[0]*100:.1f}% | PC2 {explained_var[1]*100:.1f}%",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    set_square_limits(ax, projected[:, 0], projected[:, 1])
    ax.set_aspect("equal")


def draw_bigram_on_ax(ax, grid, sequences, activations, class_means, pca_dirs, explained_var, title=""):
    """Individual activations colored by bigram on a specific ax."""
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T
    projected_means = (class_means - class_means_mean) @ pca_dirs.T

    # Grid edges + individual sequence tokens (fill=current, border=previous)
    # + class-mean centroid overlay, no labels. Edges/centroids get redrawn
    # once per sequence (harmless -- same lines/points land in the same
    # place each time) since the shared primitive draws all three together.
    for b in range(len(sequences)):
        tail = sequences[b][-N_LOOKBACK:]
        draw_bigram_scatter_on_ax(ax, projected_all[b], projected_means, tail, grid,
                                   label=False, marker_size=80,
                                   individual_size=15, individual_linewidth=0.8, individual_alpha=0.9)

    ax.set_title(
        f"{title}\nPC1 {explained_var[0]*100:.1f}% | PC2 {explained_var[1]*100:.1f}%",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    all_x = list(projected_means[:, 0]) + [
        projected_all[b, idx, 0].item()
        for b in range(len(sequences)) for idx in range(1, len(sequences[b][-N_LOOKBACK:]))
    ]
    all_y = list(projected_means[:, 1]) + [
        projected_all[b, idx, 1].item()
        for b in range(len(sequences)) for idx in range(1, len(sequences[b][-N_LOOKBACK:]))
    ]
    set_square_limits(ax, all_x, all_y)
    ax.set_aspect("equal")


def add_bigram_legend(fig):
    """Adds a single global legend to the bigram figure."""
    make_bigram_legend(fig, loc="upper right", fontsize=10)


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    grid = Grid()

    batch_sizes = [1, 4, 16, 64, 256]
    seq_lens = [4, 16, 64, 256]
    seed = 42
    # None or "LNPre" or "RMSPre" or "layer_norm_normalized" or "scaled_embed_sqrt_d"
    NORMALIZATION_TYPE = "scaled_embed_sqrt_d"
    N_LAYERS = 1

    # layer_norm_normalized and scaled_embed_sqrt_d aren't real
    # TransformerLens normalization_types: they're implemented as fwd hooks
    # on top of a model loaded with a real (or no) normalization_type.
    # - layer_norm_normalized: plain LN ("LN", so ln1 keeps real learnable
    #   gain/bias) plus a hook that renormalizes ln1's output from norm
    #   sqrt(d) down to norm 1 (unit hypersphere) before it reaches attention.
    # - scaled_embed_sqrt_d: no LayerNorm at all (normalization_type=None),
    #   but the raw embeddings are scaled by the constant sqrt(d_model), so
    #   they land at the same magnitude LN's output would have without any
    #   centering/normalizing.
    if NORMALIZATION_TYPE == "layer_norm_normalized":
        model_normalization_type = "LN"
    elif NORMALIZATION_TYPE == "scaled_embed_sqrt_d":
        model_normalization_type = None
    else:
        model_normalization_type = NORMALIZATION_TYPE
    model = load_toy_model(seed=seed, n_ctx=max(seq_lens), normalization_type=model_normalization_type, n_layers=N_LAYERS)
    model.W_pos.data.zero_()
    analysis_layer = N_LAYERS - 1  # final layer, so both layers' effects (e.g. an induction circuit) show up

    fwd_hooks = []
    if NORMALIZATION_TYPE == "layer_norm_normalized":
        fwd_hooks = [
            (f"blocks.{l}.ln1.hook_normalized", layer_norm_normalized_hook)
            for l in range(N_LAYERS)
        ]
    elif NORMALIZATION_TYPE == "scaled_embed_sqrt_d":
        scale = math.sqrt(model.cfg.d_model)
        fwd_hooks = [("hook_embed", functools.partial(scale_embed_hook, scale=scale))]

    norm_str = str(NORMALIZATION_TYPE) if NORMALIZATION_TYPE is not None else "None"
    n_rows, n_cols = len(seq_lens), len(batch_sizes)

    fig_class, axes_class = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig_class.suptitle(
        f"PCA of Per-Node Mean Activations | Normalization: {norm_str} | Layers: {N_LAYERS}", fontsize=16)

    fig_bigram, axes_bigram = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig_bigram.suptitle(
        f"PCA of Individual Activations (Labeled by Bigram) | Normalization: {norm_str} | Layers: {N_LAYERS}",
        fontsize=16)

    print("Iterating through seq_len x batch_size grid...")
    for i, seq_len in enumerate(seq_lens):
        for j, batch_size in enumerate(batch_sizes):
            print(f"Processing Seq: {seq_len}, Batch: {batch_size}")

            set_seed(seed)
            sequences = grid.generate_batch(seq_len, batch_size)

            activations_t = get_activations_toy_batch(model, sequences, analysis_layer, N_LOOKBACK, fwd_hooks=fwd_hooks)
            class_means_t = compute_class_means_batch(activations_t, sequences, WORDS, N_LOOKBACK)
            pca_dirs_t, exp_var = compute_pca_directions(class_means_t, top_n=2)

            activations = activations_t.cpu().numpy()
            class_means = class_means_t.cpu().numpy()
            pca_dirs = pca_dirs_t.cpu().numpy()

            ax_c = axes_class[i, j]
            ax_b = axes_bigram[i, j]

            title = f"Seq: {seq_len} | Batch: {batch_size}"

            draw_class_mean_on_ax(ax_c, grid, class_means, pca_dirs, exp_var, title=title)
            draw_bigram_on_ax(ax_b, grid, sequences, activations, class_means, pca_dirs, exp_var, title=title)

    fig_class.tight_layout(rect=[0, 0, 1, 0.94])
    class_filename = f"pca_class_means_sweep_grid_Norm{norm_str}_Layers{N_LAYERS}.pdf"
    save_figure(fig_class, PLOTS_DIR, class_filename)
    print(f"Saved {class_filename}")

    # fig_bigram.tight_layout(rect=[0, 0, 0.88, 0.94])
    # add_bigram_legend(fig_bigram)
    # bigram_filename = f"bigram_pca_sweep_grid_Norm{norm_str}_Layers{N_LAYERS}.pdf"
    # save_figure(fig_bigram, PLOTS_DIR, bigram_filename)
    # print(f"Saved {bigram_filename}")


if __name__ == "__main__":
    main()
