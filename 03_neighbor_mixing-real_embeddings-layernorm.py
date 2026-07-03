import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR, MODEL_NAME,
    Grid, set_seed, load_model, get_activations, load_toy_model,
    compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

PLOTS_DIR = "results/neighbor-mixing/random-embs/"
N_LOOKBACK = 50


def plot_class_mean_pca(
    grid,
    class_means,
    pca_dirs,
    explained_variance=None,
    title="PCA of per-node mean activations",
    filename_stem="pca_class_means",
    ax=None,
):
    """
    Scatter of 16 class-mean centroids with grid edges (Supports 2D and 3D).

    Parameters:
    -----------
    explained_variance : array-like, optional
        The fraction of variance explained by each principal component
        (e.g., pca.explained_variance_ratio_). Expects shape [num_components].
    ax : matplotlib Axes, optional
        Draw onto this existing axes instead of creating (and saving) a new
        standalone figure. Used to compose several rounds into one row.
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

    # Scatter + labels
    for i, word in enumerate(WORDS):
        if is_3d:
            ax.scatter(
                projected[i, 0].item(), projected[i, 1].item(), projected[i, 2].item(),
                color=WORD_TO_COLOR[word], s=120, marker="*",
                edgecolors="black", linewidths=0.5, zorder=5,
            )
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

    # ── Compute Axis Labels with Variance Explained ─────────────────────────
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
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect("equal")
        
    ax.set_title(f"{'3D ' if is_3d else ''}{title}", fontsize=10)

    if not standalone:
        return

    dim_suffix = "3d" if is_3d else "2d"
    out_name = f"{filename_stem}_{dim_suffix}"

    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    # pfig = go.Figure(data=plotly_pca_traces(projected, grid))

    # # Extract base layout configurations
    # layout_kwargs = plotly_pca_layout(f"{'3D ' if is_3d else ''}{title}")

    # # Dynamically injection of custom axis titles into Plotly layout
    # if is_3d:
    #     layout_kwargs.setdefault('scene', {}).update({
    #         'xaxis': {'title': lbl_pc1},
    #         'yaxis': {'title': lbl_pc2},
    #         'zaxis': {'title': lbl_pc3}
    #     })
    # else:
    #     layout_kwargs.update({
    #         'xaxis': {'title': lbl_pc1},
    #         'yaxis': {'title': lbl_pc2}
    #     })

    # pfig.update_layout(**layout_kwargs)
    # save_plotly(pfig, PLOTS_DIR, f"{out_name}.html")


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


def plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs):
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
    save_figure(fig, PLOTS_DIR, "bigram_pca.pdf")
    print("Saved bigram_pca")

    # ── Plotly interactive ───────────────────────────────────────────────────
    # pfig = go.Figure(data=plotly_pca_traces(projected_means, grid))

    # for word in WORDS:
    #     idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
    #     if not idxs:
    #         continue
    #     pfig.add_trace(go.Scatter(
    #         x=[projected_all[idx, 0].item() for idx in idxs],
    #         y=[projected_all[idx, 1].item() for idx in idxs],
    #         mode="markers",
    #         marker=dict(
    #             size=8, color=WORD_TO_COLOR[word],
    #             line=dict(
    #                 width=2,
    #                 color=[WORD_TO_COLOR[tail[idx - 1]] for idx in idxs],
    #             ),
    #         ),
    #         customdata=[[tail[idx - 1]] for idx in idxs],
    #         hovertemplate=(
    #             "current: <b>" + word + "</b><br>"
    #             "previous: <b>%{customdata[0]}</b>"
    #             "<extra></extra>"
    #         ),
    #         hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")),
    #         name=word, showlegend=False,
    #     ))

    # pfig.update_layout(**plotly_pca_layout(
    #     "PCA of individual activations, labeled by bigram"))
    # pfig.update_layout(width=1000, height=1000)

    # pfig.add_annotation(
    #     text=(
    #         "● Fill = current token<br>"
    #         "● Border = previous token<br>"
    #         "★ Token centroid"
    #     ),
    #     xref="paper", yref="paper", x=0.02, y=0.98,
    #     showarrow=False, font=dict(size=12),
    #     align="left", bgcolor="rgba(255,255,255,0.9)",
    #     bordercolor="gray", borderwidth=1, borderpad=6,
    # )

    # save_plotly(pfig, PLOTS_DIR, "bigram_pca.html")


def rms_norm(tensor, eps=1e-6):
    rms = torch.sqrt(torch.mean(tensor ** 2, dim=-1, keepdim=True) + eps)
    return tensor / rms


def layer_norm(tensor, eps=1e-6):
    mean = torch.mean(tensor, dim=-1, keepdim=True)
    std = torch.sqrt(torch.mean((tensor - mean) ** 2, dim=-1, keepdim=True) + eps)
    return (tensor - mean) / std


def layer_norm_normalized(tensor, eps=1e-6):
    """LayerNorm followed by an extra L2 normalization, so each activation
    lands on the unit hypersphere (radius 1) instead of radius sqrt(d).
    Equivalent to centering then L2-normalizing, since layer_norm(x) always
    has norm sqrt(d)."""
    centered = tensor - torch.mean(tensor, dim=-1, keepdim=True)
    norm = torch.norm(centered, dim=-1, keepdim=True)
    return centered / (norm + eps)


def plot_mixing_rounds_pca(grid, embs_by_round, rounds, norm_label, embs_type, top_n):
    """One figure with a class-mean PCA panel per mixing round, projected
    onto the top `top_n` (2 or 3) principal components."""
    is_3d = (top_n == 3)
    subplot_kw = {"projection": "3d"} if is_3d else {}
    fig, axes = plt.subplots(1, len(rounds), figsize=(4 * len(rounds), 4), subplot_kw=subplot_kw)

    for ax, r in zip(axes, rounds):
        embs_r = embs_by_round[r].numpy()
        pca_dirs_r, var_r = compute_pca_directions(embs_by_round[r], top_n=top_n)
        round_desc = (
            "0 rounds mixing" if r == 0
            else f"after {r} round{'s' if r != 1 else ''} of {norm_label} mixing"
        )
        plot_class_mean_pca(
            grid, embs_r, pca_dirs_r.numpy(), explained_variance=var_r,
            title=round_desc, ax=ax,
        )

    dim_suffix = "3d" if is_3d else "2d"
    fig.suptitle(f"{'3D ' if is_3d else ''}PCA of Learned Embeddings across neighbor-mixing rounds ({norm_label})")
    out_name = f"{embs_type}_embs_mixing_rounds_{norm_label}_{dim_suffix}"
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries
    
    set_seed(42)

    normalization_type = None  # None or "PreLN" or "PostLN" or "RMSNorm" or "PreLNNormalized"
    embs_type = "random"  # "random" or "learned"
    d_embed = 128


    grid = Grid()
    adjacency_matrix = torch.tensor(grid.build_adjacency_matrix(), dtype=torch.float32)
    degree_matrix = adjacency_matrix.sum(dim=1, keepdim=True)
    
    # model = load_model()
    # normalization_type is our own label consumed by apply_norm() below, not
    # a valid HookedTransformerConfig setting, so the model itself is loaded
    # without normalization (its internal LN is never exercised: we only use
    # it to fetch W_E and to_single_token, and do the norm math by hand).
    model = load_toy_model(normalization_type=None, seed=42, n_ctx=SEQ_LEN, n_layers=1, zero_out_pos_emb=True)

    if embs_type == "learned":
        # raw token embeddings (just after the embedding lookup, before any norm).
        # The toy model has no tokenizer; its vocab is WORDS in list order.
        token_ids = [WORDS.index(word) for word in WORDS]
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=model.cfg.device)

        embeddings = model.W_E[token_ids_tensor].detach().float().cpu()
    elif embs_type == "random":     
        embeddings = torch.randn(16, d_embed)

    def apply_norm(x):
        """Dispatch to the configured normalization, so every case (None,
        PreLN, RMSNorm) is handled by the same code path."""
        if normalization_type == "PreLN":
            return layer_norm(x)
        elif normalization_type == "RMSNorm":
            return rms_norm(x)
        elif normalization_type == "PreLNNormalized":
            return layer_norm_normalized(x)
        return x

    # Round 0 = embeddings after the normalization layer (identity if None).
    embs_round_0_t = apply_norm(embeddings)
    embs_by_round = {0: embs_round_0_t}
    embs_curr_t = embs_round_0_t

    # ── Embedding Neighbor Mixing Simulation ──────────────────────────────────
    for r in range(1, 11):
        normalized_embs = apply_norm(embs_curr_t)

        # Apply mixing to normalized states and add to the unnormalized residual stream
        embs_curr_t = embs_curr_t + (adjacency_matrix @ normalized_embs) / degree_matrix

        if r in [1, 2, 3, 4, 10]:
            embs_by_round[r] = embs_curr_t

    norm_label = normalization_type.lower() if normalization_type else "none"
    rounds = [0, 1, 2, 3, 4, 10]

    plot_mixing_rounds_pca(grid, embs_by_round, rounds, norm_label, embs_type, top_n=2)
    # plot_mixing_rounds_pca(grid, embs_by_round, rounds, norm_label, embs_type, top_n=3)



if __name__ == "__main__":
    main()
    