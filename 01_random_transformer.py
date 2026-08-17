import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go

from utils import (
    LAYER, SEQ_LEN,
    Grid, Torus, set_seed, load_toy_model, get_activations_toy_batch,
    compute_class_means_batch, compute_pca_directions, setup_plotting, save_figure,
    plotly_pca_layout, plotly_pca_traces, save_plotly, plot_attention_heatmaps,
    set_square_limits,
)

from word_lists import WORD_LISTS

LAYER = 0
SEQ_LEN = 64
PLOTS_DIR = "results/random_transformer/"
N_LOOKBACK = 50
BATCH_SIZE = 256

print(f"{SEQ_LEN=}, {N_LOOKBACK=}")

WORDS = WORD_LISTS["text_numbers"]

# ── Fig 2 Right: Class-mean PCA ───────────────────────────────────────────────

def plot_class_mean_pca(graph, class_means, pca_dirs, explained_var, hp_str="", hp_title=""):
    """Scatter of class-mean centroids with grid edges and explained variance.
    Supports 2D and 3D, depending on how many PCA directions are passed in."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T
    is_3d = (projected.shape[1] == 3)

    if is_3d:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Grid edges (gray dashed)
    A = graph.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                if is_3d:
                    ax.plot(
                        [projected[i, 0].item(), projected[j, 0].item()],
                        [projected[i, 1].item(), projected[j, 1].item()],
                        [projected[i, 2].item(), projected[j, 2].item()],
                        color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                    )
                else:
                    ax.plot(
                        [projected[i, 0].item(), projected[j, 0].item()],
                        [projected[i, 1].item(), projected[j, 1].item()],
                        color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
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

    # Update labels to include explained variance
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")

    # Display the hyperparams on a new line in the title
    ax.set_title(f"{'3D ' if is_3d else ''}PCA of per-node mean activations\n{hp_title}", fontsize=10)
    if is_3d:
        ax.set_zlabel(f"PC3 ({explained_var[2]*100:.1f}%)")
        ax.set_box_aspect((1, 1, 1))
    else:
        set_square_limits(ax, projected[:, 0], projected[:, 1])
        ax.set_aspect("equal")

    dim_suffix = "3d" if is_3d else "2d"
    save_figure(fig, PLOTS_DIR, f"pca_{hp_str}_{dim_suffix}.pdf")
    print(f"Saved pca_{hp_str}_{dim_suffix}.pdf")

    # ── Plotly interactive ───────────────────────────────────────────────────
    # pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    # pfig.update_layout(**plotly_pca_layout(f"PCA of per-node mean activations<br><span style='font-size:12px'>{hp_title}</span>"))
    # pfig.update_xaxes(title_text=f"PC1 ({explained_var[0]*100:.1f}%)")
    # pfig.update_yaxes(title_text=f"PC2 ({explained_var[1]*100:.1f}%)")
    # save_plotly(pfig, PLOTS_DIR, f"pca_class_means_{hp_str}.html")


# ── Fig 6: Bigram PCA ─────────────────────────────────────────────────────────

def _draw_bigram_scatter(ax, projected_all, projected_means, tail, graph, label=True):
    """Draw bigram scatter on a given axes. Shared by main plot and inset.
    Supports 2D and 3D, depending on the projected dimensionality."""
    is_3d = (projected_means.shape[1] == 3)

    A = graph.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                if is_3d:
                    ax.plot(
                        [projected_means[i, 0].item(), projected_means[j, 0].item()],
                        [projected_means[i, 1].item(), projected_means[j, 1].item()],
                        [projected_means[i, 2].item(), projected_means[j, 2].item()],
                        color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                    )
                else:
                    ax.plot(
                        [projected_means[i, 0].item(), projected_means[j, 0].item()],
                        [projected_means[i, 1].item(), projected_means[j, 1].item()],
                        color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                    )

    for idx in range(1, len(tail)):
        cur_word = tail[idx]
        prev_word = tail[idx - 1]
        if is_3d:
            ax.scatter(
                projected_all[idx, 0].item(), projected_all[idx, 1].item(), projected_all[idx, 2].item(),
                c=WORD_TO_COLOR[cur_word],
                edgecolors=WORD_TO_COLOR[prev_word],
                linewidths=1.0, s=25, alpha=1.0, zorder=3,
            )
        else:
            ax.scatter(
                projected_all[idx, 0].item(), projected_all[idx, 1].item(),
                c=WORD_TO_COLOR[cur_word],
                edgecolors=WORD_TO_COLOR[prev_word],
                linewidths=1.0, s=25, alpha=1.0, zorder=3,
            )

    for i, word in enumerate(WORDS):
        if is_3d:
            ax.scatter(
                projected_means[i, 0].item(), projected_means[i, 1].item(), projected_means[i, 2].item(),
                color=WORD_TO_COLOR[word], s=120, marker="*",
                edgecolors="black", linewidths=1.0, zorder=5,
            )
            if label:
                ax.text(
                    projected_means[i, 0].item(), projected_means[i, 1].item(), projected_means[i, 2].item(),
                    word, fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )
        else:
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
    

def plot_bigram_pca(graph, sequences, activations, class_means, pca_dirs, explained_var, hp_str="", hp_title=""):
    """Individual activations colored by (current token, previous token) for a
    batch of sequences. Supports 2D and 3D, depending on how many PCA
    directions are passed in."""
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T
    projected_means = (class_means - class_means_mean) @ pca_dirs.T
    is_3d = (projected_means.shape[1] == 3)

    # ── Main bigram plot (Matplotlib) ────────────────────────────────────────
    if is_3d:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    for b in range(len(sequences)):
        tail = sequences[b][-N_LOOKBACK:]
        _draw_bigram_scatter(ax, projected_all[b], projected_means, tail, graph)

    _make_bigram_legend(ax)

    # Update labels to include explained variance
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")

    # Display the hyperparams on a new line in the title
    ax.set_title(f"{'3D ' if is_3d else ''}PCA of individual activations, labeled by bigram\n{hp_title}", fontsize=10)

    if is_3d:
        ax.set_zlabel(f"PC3 ({explained_var[2]*100:.1f}%)")
        ax.set_box_aspect((1, 1, 1))
    else:
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

    dim_suffix = "3d" if is_3d else "2d"
    save_figure(fig, PLOTS_DIR, f"bigram_pca_{hp_str}_{dim_suffix}.pdf")
    print(f"Saved bigram_pca_{hp_str}_{dim_suffix}.pdf")

    #── Plotly interactive ───────────────────────────────────────────────────
    # pfig = go.Figure(data=plotly_pca_traces(projected_means, grid))

    # for word in WORDS:
    #     x_coords = []
    #     y_coords = []
    #     prev_words = []
        
    #     for b in range(len(sequences)):
    #         tail = sequences[b][-N_LOOKBACK:]
    #         idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
    #         for idx in idxs:
    #             x_coords.append(projected_all[b, idx, 0].item())
    #             y_coords.append(projected_all[b, idx, 1].item())
    #             prev_words.append(tail[idx - 1])
                
    #     if not x_coords:
    #         continue
            
    #     pfig.add_trace(go.Scatter(
    #         x=x_coords,
    #         y=y_coords,
    #         mode="markers",
    #         marker=dict(
    #             size=8, color=WORD_TO_COLOR[word],
    #             line=dict(width=2, color=[WORD_TO_COLOR[pw] for pw in prev_words]),
    #         ),
    #         customdata=[[pw] for pw in prev_words],
    #         hovertemplate=(
    #             "current: <b>" + word + "</b><br>"
    #             "previous: <b>%{customdata[0]}</b>"
    #             "<extra></extra>"
    #         ),
    #         hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")),
    #         name=word, showlegend=False,
    #     ))

    # pfig.update_layout(**plotly_pca_layout(
    #     f"PCA of individual activations, labeled by bigram<br><span style='font-size:12px'>{hp_title}</span>"))
    # pfig.update_layout(width=1000, height=1000)
    
    # # Add explained variance to Plotly axes
    # pfig.update_xaxes(title_text=f"PC1 ({explained_var[0]*100:.1f}%)")
    # pfig.update_yaxes(title_text=f"PC2 ({explained_var[1]*100:.1f}%)")

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

    # save_plotly(pfig, PLOTS_DIR, f"bigram_pca_{hp_str}.html")


def compute_pca_then_mean_pipeline(activations_t, sequences, words, n_lookback, top_n=2):
    """
    Computes PCA on the global distribution of individual activations, 
    then computes class means in the original space.
    """
    hidden_dim = activations_t.shape[-1]
    
    # 1. Flatten to (Total Tokens, Hidden Dim) and compute global PCA
    flat_activations_t = activations_t.reshape(-1, hidden_dim)
    pca_dirs_t, explained_var = compute_pca_directions(flat_activations_t, top_n=top_n)
    
    # 2. Compute the means in the original high-dimensional space
    class_means_t = compute_class_means_batch(activations_t, sequences, words, n_lookback)
    
    return class_means_t, pca_dirs_t, explained_var

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    graph = Grid()
    #graph = Torus()
    seed = 42
    set_seed(seed)

    NORMALIZATION_TYPE = None # None or "RMS" or "LNPre"

    # model
    model = load_toy_model(normalization_type=NORMALIZATION_TYPE, seed=seed, n_ctx=SEQ_LEN, n_layers=1, zero_out_pos_emb=True) 

    # generate data
    sequences = graph.generate_batch(SEQ_LEN, BATCH_SIZE)

    print(f"{sequences=}")
    # 1. Compute base activations and class means (Shared for both methods)
    activations_t = get_activations_toy_batch(model, sequences, LAYER, N_LOOKBACK)
    class_means_t = compute_class_means_batch(activations_t, sequences, WORDS, N_LOOKBACK)
    
    # 2. Compute Mean-then-PCA directions (Original method)
    pca_dirs_mean_then_t, exp_var_mean_then = compute_pca_directions(class_means_t, top_n=2)
    pca_dirs_mean_then_3d_t, exp_var_mean_then_3d = compute_pca_directions(class_means_t, top_n=3)

    # 3. Compute PCA-then-Mean directions (Global PCA method)
    flat_activations_t = activations_t.reshape(-1, activations_t.shape[-1])
    pca_dirs_pca_then_t, exp_var_pca_then = compute_pca_directions(flat_activations_t, top_n=2)

    # Convert all to numpy
    activations = activations_t.cpu().numpy()
    class_means = class_means_t.cpu().numpy()
    pca_dirs_mean_then = pca_dirs_mean_then_t.cpu().numpy()
    pca_dirs_mean_then_3d = pca_dirs_mean_then_3d_t.cpu().numpy()
    pca_dirs_pca_then = pca_dirs_pca_then_t.cpu().numpy()

    # ── Base Hyperparameter Strings ──────────────────────────────────────────
    # Handle the None case cleanly for strings
    norm_str = str(NORMALIZATION_TYPE) if NORMALIZATION_TYPE is not None else "None"
    graph_type = type(graph).__name__  # "Grid" or "Torus"

    base_hp_str = f"{graph_type}_Seq{SEQ_LEN}_Look{N_LOOKBACK}_Batch{BATCH_SIZE}_Seed{seed}_Norm{norm_str}"
    base_hp_title = f"Graph: {graph_type} | Seq: {SEQ_LEN} | Lookback: {N_LOOKBACK} | Batch: {BATCH_SIZE} | Seed: {seed} | Norm: {norm_str}"

    # ── Plot 1: Mean-then-PCA ────────────────────────────────────────────────
    hp_str_1 = f"{base_hp_str}_MeanThenPCA"
    hp_title_1 = f"{base_hp_title} | Mean-then-PCA"
    
    plot_class_mean_pca(graph, class_means, pca_dirs_mean_then, exp_var_mean_then,
                        hp_str=hp_str_1, hp_title=hp_title_1)
    # plot_bigram_pca(graph, sequences, activations, class_means, pca_dirs_mean_then, exp_var_mean_then,
    #                 hp_str=hp_str_1, hp_title=hp_title_1)

    # ── Plot 1 (3D): Mean-then-PCA ───────────────────────────────────────────
    plot_class_mean_pca(graph, class_means, pca_dirs_mean_then_3d, exp_var_mean_then_3d,
                        hp_str=hp_str_1, hp_title=hp_title_1)
    # plot_bigram_pca(graph, sequences, activations, class_means, pca_dirs_mean_then_3d, exp_var_mean_then_3d,
    #                 hp_str=hp_str_1, hp_title=hp_title_1)

    # # ── Plot 2: PCA-then-Mean ────────────────────────────────────────────────
    # hp_str_2 = f"{base_hp_str}_PCAThenMean"
    # hp_title_2 = f"{base_hp_title} | PCA-then-Mean"
    
    # plot_class_mean_pca(graph, class_means, pca_dirs_pca_then, exp_var_pca_then, 
    #                     hp_str=hp_str_2, hp_title=hp_title_2)
    # plot_bigram_pca(graph, sequences, activations, class_means, pca_dirs_pca_then, exp_var_pca_then, 
    #                 hp_str=hp_str_2, hp_title=hp_title_2)

if __name__ == "__main__":
    main()


