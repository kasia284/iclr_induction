import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import (
    WORDS, LAYER, WORD_TO_COLOR,
    Grid, set_seed, load_toy_model, setup_plotting, save_figure,
    compute_pca_directions, set_square_limits,
)

PLOTS_DIR = "results/normalization_cosine_sim/plots"

# (display label, normalization_type passed to load_toy_model)
NORMALIZATION_CASES = [("None", None), ("LNPre", "LNPre"), ("RMSPre", "RMSPre")]


def get_post_norm_embeddings(model, words, layer=0):
    """
    Run each word through the model as a lone (position-0) token and grab the
    residual-stream activation right after the pre-attention normalization
    (blocks.{layer}.ln1.hook_normalized). When normalization_type is None,
    ln1 is nn.Identity and has no hook, so we fall back to hook_resid_pre --
    which is exactly what "after normalization" reduces to in that case.

    Returns: [n_words, d_model] tensor.
    """
    word_to_id = {word: i for i, word in enumerate(WORDS)}
    tokens = torch.tensor(
        [[word_to_id[w]] for w in words], dtype=torch.long, device=model.cfg.device
    )  # [n_words, 1]

    normalized_name = f"blocks.{layer}.ln1.hook_normalized"
    resid_pre_name = f"blocks.{layer}.hook_resid_pre"

    _, cache = model.run_with_cache(tokens, names_filter=[normalized_name, resid_pre_name])

    acts = cache[normalized_name] if normalized_name in cache else cache[resid_pre_name]
    return acts[:, 0, :]  # [n_words, d_model]


def plot_gram_matrix_on_ax(ax, fig, gram_matrix, title):
    """Raw inner-product (Gram) matrix -- unlike cosine similarity, this keeps
    the effect of normalization on vector magnitude, not just direction."""
    vmax = np.abs(gram_matrix).max()
    im = ax.imshow(gram_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(WORDS)))
    ax.set_yticks(range(len(WORDS)))
    ax.set_xticklabels(WORDS, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(WORDS, fontsize=7)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Inner Product")


def plot_pca_on_ax(ax, grid, projected, explained_var, title):
    """Scatter of post-normalization word embeddings with grid adjacency edges."""
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0], projected[j, 0]],
                    [projected[i, 1], projected[j, 1]],
                    color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                )

    for i, word in enumerate(WORDS):
        ax.scatter(
            projected[i, 0], projected[i, 1],
            color=WORD_TO_COLOR[word], s=100, marker="*",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax.annotate(
            word, (projected[i, 0], projected[i, 1]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_title(
        f"{title}\nPC1 {explained_var[0]*100:.1f}% | PC2 {explained_var[1]*100:.1f}%",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    set_square_limits(ax, projected[:, 0], projected[:, 1])
    ax.set_aspect("equal")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    seed = 42
    layer = LAYER
    grid = Grid()

    fig_gram, axes_gram = plt.subplots(1, len(NORMALIZATION_CASES), figsize=(6 * len(NORMALIZATION_CASES), 5.5))
    fig_gram.suptitle("Gram Matrix of Word Embeddings After Normalization", fontsize=16)

    fig_pca, axes_pca = plt.subplots(1, len(NORMALIZATION_CASES), figsize=(6 * len(NORMALIZATION_CASES), 6))
    fig_pca.suptitle("PCA of Word Embeddings After Normalization", fontsize=16)

    for ax_gram, ax_pca, (label, norm_type) in zip(axes_gram, axes_pca, NORMALIZATION_CASES):
        print(f"Processing normalization: {label}")

        set_seed(seed)
        model = load_toy_model(normalization_type=norm_type, seed=seed, n_ctx=1)
        model.W_pos.data.zero_()

        embeddings_t = get_post_norm_embeddings(model, WORDS, layer=layer)
        embeddings = embeddings_t.detach().cpu().numpy()

        gram_matrix = embeddings @ embeddings.T
        plot_gram_matrix_on_ax(ax_gram, fig_gram, gram_matrix, title=f"Normalization: {label}")

        pca_dirs_t, explained_var = compute_pca_directions(embeddings_t, top_n=2)
        projected = (embeddings_t @ pca_dirs_t.T).detach().cpu().numpy()
        plot_pca_on_ax(ax_pca, grid, projected, explained_var, title=f"Normalization: {label}")

    save_figure(fig_gram, PLOTS_DIR, "embeddings_gram_matrix_normalization_comparison.png")
    print("Saved embeddings_gram_matrix_normalization_comparison.png")

    save_figure(fig_pca, PLOTS_DIR, "embeddings_pca_normalization_comparison.png")
    print("Saved embeddings_pca_normalization_comparison.png")


if __name__ == "__main__":
    main()
