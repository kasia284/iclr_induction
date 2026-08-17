"""Reproduce Figures 2 and 6: accuracy curve, class-mean PCA, and bigram PCA."""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import tqdm
import torch

from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

DATA_DIR = "results/reproduce/data"
PLOTS_DIR = "results/reproduce/plots"
N_LOOKBACK = 200


def plot_class_mean_projection(grid, projected, title="Projection", filename_stem="proj"):
    """Scatter of 16 class-mean centroids with grid edges for direct projections (t-SNE/UMAP)."""
    num_dims = projected.shape[1]
    is_3d = (num_dims == 3)

    # Setup the figure canvas based on dimensionality
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

    # Axis Labels & Aspect Ratio
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    
    if is_3d:
        ax.set_zlabel("Dim 3")
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect("equal")
        
    ax.set_title(f"{'3D ' if is_3d else ''}{title}", fontsize=10)
    
    dim_suffix = "3d" if is_3d else "2d"
    out_name = f"{filename_stem}_{dim_suffix}"
    
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout(f"{'3D ' if is_3d else ''}{title}"))
    save_plotly(pfig, PLOTS_DIR, f"{out_name}.html")


# ── Fig 2 Left: Accuracy curve ────────────────────────────────────────────────

def plot_accuracy_curve(all_accs):
    """Average accuracy across 16 sequences with uniform starting positions."""
    mean = smooth(all_accs.mean(axis=0))
    std = smooth(all_accs.std(axis=0))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(mean, color="black", linewidth=1.0, label="Mean accuracy")
    ax.fill_between(range(len(mean)), mean - std, mean + std,
                    alpha=0.15, color="gray", edgecolor="none", label="$\\pm$1 std")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title("Accuracy vs sequence length", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray")

    save_figure(fig, PLOTS_DIR, "accuracy_curve.pdf")
    print("Saved accuracy_curve")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure()
    x = list(range(len(mean)))
    pfig.add_trace(go.Scatter(
        x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(128,128,128,0.2)",
        showlegend=False, hoverinfo="skip",
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=mean.tolist(), mode="lines",
        line=dict(color="black", width=2), name="Mean accuracy",
        hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>",
    ))
    pfig.update_layout(**plotly_line_layout(
        "Accuracy vs sequence length", "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, PLOTS_DIR, "accuracy_curve.html")


# ── Fig 2 Right & Neighbor Mixing: Class-mean PCA ─────────────────────────────

def plot_class_mean_pca(grid, class_means, pca_dirs, title="PCA of per-node mean activations", filename_stem="pca_class_means"):
    """Scatter of 16 class-mean centroids with grid edges (Supports 2D and 3D)."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T  # [16, num_components]
    num_dims = projected.shape[1]
    is_3d = (num_dims == 3)

    # Setup the figure canvas based on dimensionality
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

    # Axis Labels & Aspect Ratio
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    
    if is_3d:
        ax.set_zlabel("PC3")
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect("equal")
        
    ax.set_title(f"{'3D ' if is_3d else ''}{title}", fontsize=10)
    
    dim_suffix = "3d" if is_3d else "2d"
    out_name = f"{filename_stem}_{dim_suffix}"
    
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout(f"{'3D ' if is_3d else ''}{title}"))
    save_plotly(pfig, PLOTS_DIR, f"{out_name}.html")

    
# ── Fig 6: Bigram PCA ─────────────────────────────────────────────────────────

def _draw_bigram_scatter(ax, projected_all, projected_means, tail, grid, label=True):
    """Draw bigram scatter on a given axes. Shared by main plot and inset."""
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected_means[i, 0].item(), projected_means[j, 0].item()],
                    [projected_means[i, 1].item(), projected_means[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
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
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T    # [N_LOOKBACK, 2]
    projected_means = (class_means - class_means_mean) @ pca_dirs.T  # [16, 2]

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
    pfig = go.Figure(data=plotly_pca_traces(projected_means, grid))

    for word in WORDS:
        idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
        if not idxs:
            continue
        pfig.add_trace(go.Scatter(
            x=[projected_all[idx, 0].item() for idx in idxs],
            y=[projected_all[idx, 1].item() for idx in idxs],
            mode="markers",
            marker=dict(
                size=8, color=WORD_TO_COLOR[word],
                line=dict(
                    width=2,
                    color=[WORD_TO_COLOR[tail[idx - 1]] for idx in idxs],
                ),
            ),
            customdata=[[tail[idx - 1]] for idx in idxs],
            hovertemplate=(
                "current: <b>" + word + "</b><br>"
                "previous: <b>%{customdata[0]}</b>"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")),
            name=word, showlegend=False,
        ))

    pfig.update_layout(**plotly_pca_layout(
        "PCA of individual activations, labeled by bigram"))
    pfig.update_layout(width=1000, height=1000)

    pfig.add_annotation(
        text=(
            "● Fill = current token<br>"
            "● Border = previous token<br>"
            "★ Token centroid"
        ),
        xref="paper", yref="paper", x=0.02, y=0.98,
        showarrow=False, font=dict(size=12),
        align="left", bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray", borderwidth=1, borderpad=6,
    )

    save_plotly(pfig, PLOTS_DIR, "bigram_pca.html")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries
    grid = Grid()

    # acc_path = os.path.join(DATA_DIR, "accuracies.npz")
    # pca_path = os.path.join(DATA_DIR, "pca.npz")
    # seq_path = os.path.join(DATA_DIR, "sequence.json")

    # We defer loading the model unless we strictly need it. 
    # If caches are missing OR we need to extract raw embeddings, we will load it.
    model = None

    # if os.path.exists(acc_path) and os.path.exists(pca_path) and os.path.exists(seq_path):
    #     print("Loading cached data (delete data/ to recompute)...")
    #     all_accs = np.load(acc_path)["all_accs"]
    #     pca_data = np.load(pca_path)
    #     activations = pca_data["activations"]
    #     class_means = pca_data["class_means"]
    #     pca_dirs = pca_data["pca_dirs"]
    #     with open(seq_path) as f:
    #         sequence = json.load(f)
    # else:
    #     model = load_model()
    #     os.makedirs(DATA_DIR, exist_ok=True)

    #     # PCA data
    #     set_seed(42)
    #     sequence = grid.generate_sequence(SEQ_LEN)
    #     activations_t = get_activations(model, sequence, LAYER, N_LOOKBACK)
    #     class_means_t = compute_class_means(activations_t, sequence, WORDS, N_LOOKBACK)
    #     pca_dirs_t, fve = compute_pca_directions(class_means_t, top_n=3)

    #     activations = activations_t.cpu().numpy()
    #     class_means = class_means_t.cpu().numpy()
    #     pca_dirs = pca_dirs_t.cpu().numpy()

    # if model is None:
    #     print("Loading model to extract base embeddings...")
    #     model = load_model()
        
    model = load_model()

    from sklearn.manifold import TSNE
    import umap
    
    set_seed(42)
    sequence = grid.generate_sequence(SEQ_LEN)

    # Get activations immediately following the embedding layer (layer 0)
    embed_acts_t = get_activations(model, sequence, layer=0, n_lookback=N_LOOKBACK)
    
    # Calculate centroids of the 16 tokens
    embs_round_0 = compute_class_means(embed_acts_t, sequence, WORDS, N_LOOKBACK).cpu().numpy()
    
    embs_round_0_t = torch.tensor(embs_round_0)


    # ── 2. t-SNE ─────────────────────────────────────────────────────────
    # Perplexity must be less than n_samples (16)
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca')
    projected_tsne = tsne.fit_transform(embs_round_0)
    
    plot_class_mean_projection(grid, projected_tsne,
                                title="t-SNE of Learned Embeddings",
                                filename_stem="learned_embs_tsne")

    # ── 3. UMAP ──────────────────────────────────────────────────────────
    # n_neighbors must be less than n_samples (16)
    reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3, random_state=42)
    projected_umap = reducer.fit_transform(embs_round_0)
    
    plot_class_mean_projection(grid, projected_umap,
                                title="UMAP of Learned Embeddings",
                                filename_stem="learned_embs_umap")


if __name__ == "__main__":
    main()